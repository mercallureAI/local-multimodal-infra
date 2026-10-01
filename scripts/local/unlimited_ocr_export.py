"""Export baidu/Unlimited-OCR from its PyTorch checkpoint to the LOCAL CUDA package.

The package read by ``crates/adapters/ocr/unlimited-ocr`` has two graphs:

* ``unlimited_ocr_vision.onnx``: ``pixel_values [N,3,1024,1024]`` (base mode,
  normalized with mean = std = 0.5) to ``image_features [N,273,1280]``. Each
  page is 16x16 projected SAM+CLIP features, one ``image_newline`` per row and
  a trailing ``view_seperator``: the 273 image-token embeddings of the prompt.
* ``unlimited_ocr_prefill.onnx`` / ``unlimited_ocr_decode.onnx``: DeepSeek-V2
  steps over the prompt (any length, image embeddings spliced in) and over one
  generated token (static shapes). Every layer reads
  ``past_key_values.{l}.{key,value}`` and writes ``present.{l}.{key,value}`` of
  the same fixed ``[1,10,capacity,128]`` shape: the new tokens' K/V are
  scattered into the slots named by ``write_index``, so the runtime binds past
  and present of both graphs to one device buffer. Attention covers the whole
  buffer plus ``attention_bias`` (0 or -inf per slot), which lets the caller
  express the causal prefill and upstream's R-SWA decode (the prompt KV stays,
  generated tokens share a ring of ``sliding_window_size`` slots). ``logits``
  are for the last input position only. Both graphs reference one weight file
  (``unlimited_ocr_llm.data``), listed in the manifest so the runtime uploads
  it once and hands it to both sessions.

Precision: LLM linear weights are FP16 (FP16 GEMMs); the residual stream,
RMSNorm, router, attention and KV cache stay FP32. Routed experts (~5.2 GB of
the 5.6 GB LLM in FP16) are by default int8 (symmetric per output channel,
CUTLASS-prepacked by ONNX Runtime's own quantizer) and run through
``com.microsoft.QMoE`` in both graphs; ``--expert-precision fp16`` keeps them
FP16. Routed experts run
as upstream's ``greedy`` gate with ``norm_topk_prob=False`` (softmax top-6, not
renormalized): ONNX Runtime's fused ``com.microsoft.MoE`` in prefill, and in
decode a gather of the six selected experts, which is ~10x faster for a single
token than the fused kernel.

The source is the ``baidu/Unlimited-OCR`` checkpoint, verified at revision
``07dea832e22aefee32ad281d4b80551282e1c168``.

``parity`` compares the package with the upstream PyTorch model on an image:
vision features, prefill logits and a greedy decode long enough to cross the
R-SWA ring.

Environment (upstream pins): Python 3.11, ``torch==2.10.0`` (CPU build is
enough), ``torchvision==0.25.0``, ``transformers==4.57.1``, plus ``onnx
onnxruntime-gpu==1.30.0 einops addict easydict safetensors pillow``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ADAPTER = "unlimited_ocr"
MANIFEST_SCHEMA = "local.unlimited_ocr.package.v1"
VISION_GRAPH = "unlimited_ocr_vision.onnx"
PREFILL_GRAPH = "unlimited_ocr_prefill.onnx"
DECODE_GRAPH = "unlimited_ocr_decode.onnx"
LLM_DATA = "unlimited_ocr_llm.data"
IMAGE_SIZE = 1024
GRID = 16  # 1024 / 16 (SAM patch) / 4 (downsample)
IMAGE_TOKENS = (GRID + 1) * GRID + 1
IMAGE_TOKEN_ID = 128815
BOS_TOKEN_ID = 0
EOS_TOKEN_ID = 1
PROMPT = "<image>document parsing."
NO_REPEAT_NGRAM_SIZE = 35
NGRAM_WINDOW = 128
OPSET = 18
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json")


def _lazy_imports():
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    functional = importlib.import_module("torch.nn.functional")
    return torch, nn, functional


# ---------------------------------------------------------------------------
# Upstream model
# ---------------------------------------------------------------------------


def load_upstream(source: Path):
    torch, _, _ = _lazy_imports()
    transformers = importlib.import_module("transformers")
    model = transformers.AutoModel.from_pretrained(
        str(source),
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    # Not in the checkpoint ("newly initialized"): the export bakes it, so it
    # must be the arange the CLIP embeddings expect.
    position_ids = model.model.vision_model.embeddings.position_ids
    if not torch.equal(position_ids.flatten(), torch.arange(position_ids.numel())):
        raise RuntimeError("CLIP position_ids buffer is not arange; the checkpoint loaded incorrectly")
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(source), trust_remote_code=True)
    return model, tokenizer


def prompt_ids(tokenizer, pages: int = 1) -> tuple[list[int], list[bool]]:
    """Upstream ``infer`` (base mode) / ``infer_multi`` prompt layout."""
    before, after = PROMPT.split("<image>")
    ids: list[int] = []
    mask: list[bool] = []
    head = tokenizer.encode(before, add_special_tokens=False)
    ids += head
    mask += [False] * len(head)
    ids += [IMAGE_TOKEN_ID] * (IMAGE_TOKENS * pages)
    mask += [True] * (IMAGE_TOKENS * pages)
    tail = tokenizer.encode(after, add_special_tokens=False)
    ids += tail
    mask += [False] * len(tail)
    return [BOS_TOKEN_ID] + ids, [False] + mask


def preprocess(image_path: Path):
    """Upstream base mode: pad to 1024 with the mean colour, normalize to [-1, 1]."""
    torch, _, _ = _lazy_imports()
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    padded = ImageOps.pad(image, (IMAGE_SIZE, IMAGE_SIZE), color=(127, 127, 127))
    import numpy as np

    array = np.asarray(padded, dtype=np.float32) / 255.0
    array = (array - 0.5) / 0.5
    return torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0)


# ---------------------------------------------------------------------------
# Export modules
# ---------------------------------------------------------------------------


def keep_norms_fp32(module) -> None:
    """Runs every normalization layer of an FP16 module in FP32.

    SAM's ``LayerNorm2d`` computes mean and variance with plain tensor ops, so
    in FP16 they lose precision (and can overflow); ``nn.LayerNorm`` likewise
    exports as an FP16 ``LayerNormalization``. Convolutions and matmuls stay
    FP16.
    """
    torch, nn, _ = _lazy_imports()
    for child in module.modules():
        if isinstance(child, nn.LayerNorm) or type(child).__name__ == "LayerNorm2d":
            child.float()
            inner_forward = child.forward

            def forward(x, _inner=inner_forward):
                return _inner(x.float()).half()

            child.forward = forward


def check_routing(config) -> None:
    """The graphs implement upstream's ``greedy`` softmax gate without top-k
    renormalization or scaling; refuse checkpoints configured otherwise."""
    expected = {
        "scoring_func": "softmax",
        "topk_method": "greedy",
        "norm_topk_prob": False,
        "routed_scaling_factor": 1.0,
    }
    actual = {key: getattr(config, key, value) for key, value in expected.items()}
    if actual != expected:
        raise RuntimeError(f"unsupported MoE routing {actual}; the export implements {expected}")


def build_modules(model, vision_fp16: bool, expert_bits: int = 16, build_llm: bool = True):
    torch, nn, F = _lazy_imports()
    config = model.config
    inner = model.model
    if build_llm:
        check_routing(config)

    class Vision(nn.Module):
        def __init__(self):
            super().__init__()
            self.sam = inner.sam_model
            self.clip = inner.vision_model
            self.projector = inner.projector
            self.register_buffer("newline", inner.image_newline.detach().float().clone())
            self.register_buffer("separator", inner.view_seperator.detach().float().clone())

        def forward(self, pixel_values):
            if vision_fp16:
                pixel_values = pixel_values.half()
            sam = self.sam(pixel_values)
            clip = self.clip(pixel_values, sam)
            features = torch.cat((clip[:, 1:], sam.flatten(2).permute(0, 2, 1)), dim=-1)
            features = self.projector(features).float()
            pages = features.shape[0]
            dim = features.shape[-1]
            grid = features.view(pages, GRID, GRID, dim)
            newline = self.newline.view(1, 1, 1, dim).expand(pages, GRID, 1, dim)
            rows = torch.cat([grid, newline], dim=2).reshape(pages, GRID * (GRID + 1), dim)
            separator = self.separator.view(1, 1, dim).expand(pages, 1, dim)
            return torch.cat([rows, separator], dim=1)

    class RoutedExperts(torch.autograd.Function):
        """``com.microsoft.MoE`` in the graph; an FP32 reference when traced."""

        @staticmethod
        def forward(ctx, hidden, router_logits, fc1, fc2, fc3, top_k):
            scores = router_logits.float().softmax(dim=-1)
            weights, experts = torch.topk(scores, k=top_k, dim=-1, sorted=False)
            x = hidden.float()
            out = torch.zeros_like(x)
            for slot in range(top_k):
                for row in range(x.shape[0]):
                    e = int(experts[row, slot])
                    gate = x[row] @ fc1[e].float().t()
                    up = x[row] @ fc3[e].float().t()
                    act = F.silu(gate) * up
                    out[row] += weights[row, slot] * (act @ fc2[e].float().t())
            return out.to(hidden.dtype)

        @staticmethod
        def symbolic(g, hidden, router_logits, fc1, fc2, fc3, top_k):
            # Bias inputs are placeholders, dropped to "" after export.
            return g.op(
                "com.microsoft::MoE",
                hidden,
                router_logits,
                fc1,
                g.op("Constant", value_t=torch.tensor([], dtype=torch.float16)),
                fc2,
                g.op("Constant", value_t=torch.tensor([], dtype=torch.float16)),
                fc3,
                k_i=top_k,
                activation_type_s="silu",
                normalize_routing_weights_i=0,
            )

    class QuantRoutedExperts(torch.autograd.Function):
        """``com.microsoft.QMoE`` (int8, gate and up fused in fc1). Tracing only
        needs the output shape: the graph is checked by ``parity``."""

        @staticmethod
        def forward(ctx, hidden, router_logits, fc1, scales1, fc2, scales2, top_k):
            return torch.zeros_like(hidden)

        @staticmethod
        def symbolic(g, hidden, router_logits, fc1, scales1, fc2, scales2, top_k):
            empty = lambda: g.op("Constant", value_t=torch.tensor([], dtype=torch.float16))
            return g.op(
                "com.microsoft::QMoE",
                hidden,
                router_logits,
                fc1,
                scales1,
                empty(),
                fc2,
                scales2,
                empty(),
                k_i=top_k,
                activation_type_s="swiglu",
                swiglu_fusion_i=2,
                normalize_routing_weights_i=0,
                expert_weight_bits_i=8,
            )

    def half(weight):
        return nn.Parameter(weight.detach().to(torch.float16).clone(), requires_grad=False)

    def quantize_experts(weights):
        """``[E, N, K]`` to CUTLASS-prepacked int8 ``[E, K, N]`` and FP16 scales ``[E, N]``."""
        from onnxruntime.quantization.cuda_quantizer import CudaQuantizer

        packed, scales = [], []
        for expert in weights:
            q, scale = CudaQuantizer.qmoe_per_channel_quantize(expert.detach().float().contiguous(), 8, True)
            packed.append(q)
            scales.append(scale)
        return torch.stack(packed).contiguous(), torch.stack(scales).to(torch.float16).contiguous()

    def half_t(weight):
        """``[out, in]`` Linear weight stored ``[in, out]``: ``x @ w`` exports a
        MatMul on the parameter itself, so its name is the same in every graph."""
        return half(weight.t().contiguous())

    def full(weight):
        return nn.Parameter(weight.detach().float().clone(), requires_grad=False)

    def rms(x, weight, eps):
        variance = x.pow(2).mean(-1, keepdim=True)
        return weight * (x * torch.rsqrt(variance + eps))

    def linear16(x16, weight_t):
        return torch.matmul(x16, weight_t)

    class Mlp(nn.Module):
        """SwiGLU FFN with FP16 GEMMs; the gate product is formed in FP32."""

        def __init__(self, mlp):
            super().__init__()
            self.gate = half_t(mlp.gate_proj.weight)
            self.up = half_t(mlp.up_proj.weight)
            self.down = half_t(mlp.down_proj.weight)

        def forward(self, x16):
            gate = linear16(x16, self.gate).float()
            up = linear16(x16, self.up).float()
            return linear16((F.silu(gate) * up).half(), self.down).float()

    class Moe(nn.Module):
        def __init__(self, moe):
            super().__init__()
            self.top_k = moe.num_experts_per_tok
            self.router = full(moe.gate.weight.t().contiguous())
            self.int8 = expert_bits == 8
            gate = torch.stack([e.gate_proj.weight for e in moe.experts])
            up = torch.stack([e.up_proj.weight for e in moe.experts])
            down = torch.stack([e.down_proj.weight for e in moe.experts])
            if self.int8:
                fc1, scales1 = quantize_experts(torch.cat([gate, up], dim=1))
                fc2, scales2 = quantize_experts(down)
                self.register_buffer("fc1", fc1)
                self.register_buffer("fc1_scales", scales1)
                self.register_buffer("fc2", fc2)
                self.register_buffer("fc2_scales", scales2)
            else:
                self.fc1 = half(gate)
                self.fc2 = half(down)
                self.fc3 = half(up)
            self.shared = Mlp(moe.shared_experts)

        def forward(self, x, decode):
            flat = x.view(-1, x.shape[-1])
            logits = torch.matmul(flat, self.router)
            flat16 = flat.half()
            if self.int8:
                routed = QuantRoutedExperts.apply(
                    flat16, logits.half(), self.fc1, self.fc1_scales, self.fc2, self.fc2_scales, self.top_k
                ).float()
            elif decode:
                # One token: run only the selected experts (torch weight layout,
                # so the gathered [k, out, in] blocks multiply a column vector).
                weights, experts = torch.topk(logits.softmax(dim=-1), k=self.top_k, dim=-1, sorted=False)
                ids = experts.view(-1)
                column = flat16.view(-1, 1)
                gate = torch.matmul(self.fc1.index_select(0, ids), column).float()
                up = torch.matmul(self.fc3.index_select(0, ids), column).float()
                act = (F.silu(gate) * up).half()
                out = torch.matmul(self.fc2.index_select(0, ids), act).float()
                routed = (out * weights.view(-1, 1, 1)).sum(dim=0).view(1, -1)
            else:
                routed = RoutedExperts.apply(flat16, logits.half(), self.fc1, self.fc2, self.fc3, self.top_k).float()
            return (routed + self.shared(flat16)).view(x.shape)

    class DenseMlp(nn.Module):
        def __init__(self, mlp):
            super().__init__()
            self.mlp = Mlp(mlp)

        def forward(self, x, decode):
            return self.mlp(x.half())

    class Layer(nn.Module):
        def __init__(self, layer):
            super().__init__()
            attn = layer.self_attn
            self.eps = layer.input_layernorm.variance_epsilon
            self.input_norm = full(layer.input_layernorm.weight)
            self.post_norm = full(layer.post_attention_layernorm.weight)
            self.q = half_t(attn.q_proj.weight)
            self.k = half_t(attn.k_proj.weight)
            self.v = half_t(attn.v_proj.weight)
            self.o = half_t(attn.o_proj.weight)
            self.heads = config.num_attention_heads
            self.kv_heads = config.num_key_value_heads
            self.head_dim = attn.head_dim
            assert self.heads == self.kv_heads, "the export assumes MHA (no KV-head grouping)"
            self.mlp = Moe(layer.mlp) if hasattr(layer.mlp, "experts") else DenseMlp(layer.mlp)

        def forward(self, hidden, cos, sin, index, bias, past_key, past_value, decode):
            seq = 1 if decode else hidden.shape[1]
            x16 = rms(hidden, self.input_norm, self.eps).half()

            def heads(weight):
                return linear16(x16, weight).float().view(1, seq, self.heads, self.head_dim).transpose(1, 2)

            q, k, v = heads(self.q), heads(self.k), heads(self.v)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            key = past_key.scatter(2, index, k)
            value = past_value.scatter(2, index, v)
            scores = torch.matmul(q, key.transpose(2, 3)) / math.sqrt(self.head_dim) + bias
            probs = scores.softmax(dim=-1)
            out = torch.matmul(probs, value).transpose(1, 2).reshape(1, seq, self.heads * self.head_dim)
            hidden = hidden + linear16(out.half(), self.o).float()
            hidden = hidden + self.mlp(rms(hidden, self.post_norm, self.eps), decode)
            return hidden, key, value

    def rotate_half(x):
        half_dim = x.shape[-1] // 2
        return torch.cat((-x[..., half_dim:], x[..., :half_dim]), dim=-1)

    class Core(nn.Module):
        """Weights and layers shared by the prefill and decode graphs."""

        def __init__(self):
            super().__init__()
            rotary = inner.layers[0].self_attn.rotary_emb
            self.register_buffer("inv_freq", rotary.inv_freq.detach().float().clone())
            self.rope_scaling = float(getattr(rotary, "attention_scaling", 1.0))
            self.embed = half(inner.embed_tokens.weight)
            self.layers = nn.ModuleList([Layer(layer) for layer in inner.layers])
            self.norm = full(inner.norm.weight)
            self.eps = inner.norm.variance_epsilon
            self.lm_head = half_t(model.lm_head.weight)
            self.heads = config.num_attention_heads
            self.head_dim = self.layers[0].head_dim

        def run(self, hidden, position_ids, write_index, attention_bias, past, decode):
            seq = 1 if decode else hidden.shape[1]
            freqs = position_ids.unsqueeze(-1).float() * self.inv_freq
            angles = torch.cat((freqs, freqs), dim=-1).unsqueeze(1)
            cos = angles.cos() * self.rope_scaling
            sin = angles.sin() * self.rope_scaling
            index = write_index.view(1, 1, seq, 1).expand(1, self.heads, seq, self.head_dim)
            presents = []
            for i, layer in enumerate(self.layers):
                hidden, key, value = layer(hidden, cos, sin, index, attention_bias, past[2 * i], past[2 * i + 1], decode)
                presents += [key, value]
            last = rms(hidden[:, -1, :], self.norm, self.eps)
            logits = linear16(last.half(), self.lm_head).float()
            return (logits, *presents)

    class Prefill(nn.Module):
        def __init__(self, core):
            super().__init__()
            self.core = core

        def forward(self, input_ids, images_seq_mask, image_features, position_ids, write_index, attention_bias, *past):
            text = F.embedding(input_ids, self.core.embed).float()
            slot = (torch.cumsum(images_seq_mask.to(torch.int64), dim=-1) - 1).clamp(min=0)
            image = image_features.index_select(0, slot[0]).unsqueeze(0)
            hidden = torch.where(images_seq_mask.unsqueeze(-1), image, text)
            return self.core.run(hidden, position_ids, write_index, attention_bias, past, decode=False)

    class Decode(nn.Module):
        def __init__(self, core):
            super().__init__()
            self.core = core

        def forward(self, input_ids, position_ids, write_index, attention_bias, *past):
            hidden = F.embedding(input_ids, self.core.embed).float()
            return self.core.run(hidden, position_ids, write_index, attention_bias, past, decode=True)

    vision = Vision().eval()
    if vision_fp16:
        vision.half()
        vision.newline.data = vision.newline.data.float()
        vision.separator.data = vision.separator.data.float()
        keep_norms_fp32(vision)
    if not build_llm:
        return vision, None, None
    core = Core().eval()
    return vision, Prefill(core).eval(), Decode(core).eval()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_graph(module, args, path: Path, input_names, output_names, dynamic_axes):
    """Exports ``module`` to ``path`` with its weights in ``<path>.data``."""
    import onnx

    model = trace_graph(module, args, path, input_names, output_names, dynamic_axes)
    data = path.name + ".data"
    for stale in (path, path.with_name(data)):
        stale.unlink(missing_ok=True)
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data,
        size_threshold=1024,
    )
    return path


def trace_graph(module, args, path: Path, input_names, output_names, dynamic_axes):
    """TorchScript export of ``module``; returns the model with its weights loaded."""
    torch, _, _ = _lazy_imports()
    import onnx

    with tempfile.TemporaryDirectory(dir=path.parent) as scratch:
        raw = Path(scratch) / path.name
        with torch.no_grad():
            torch.onnx.export(
                module,
                args,
                str(raw),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=OPSET,
                do_constant_folding=True,
                custom_opsets={"com.microsoft": 1},
                dynamo=False,
            )
        model = onnx.load(str(raw), load_external_data=True)
    drop_moe_bias_placeholders(model)
    return model


ELEMENTS = {1: "f32", 10: "f16", 7: "i64", 9: "bool", 6: "i32", 2: "u8", 3: "i8"}


def save_with_shared_weights(models: dict[str, Any], out: Path, data_name: str, inline_below: int = 1024):
    """Saves graphs whose weights live in one file, each tensor stored once.

    Initializers of the same name must hold the same bytes in every graph.
    Returns the ranges of the tensors used by all graphs, for the runtime to
    upload once and share between the sessions.
    """
    import numpy as np
    from onnx import TensorProto, numpy_helper

    blobs: dict[str, tuple[bytes, Any]] = {}
    users: dict[str, set[str]] = {}
    for graph_name, model in models.items():
        for init in model.graph.initializer:
            raw = numpy_helper.to_array(init).tobytes()
            if len(raw) < inline_below:
                continue
            known = blobs.get(init.name)
            if known is not None and known[0] != raw:
                raise RuntimeError(f"initializer `{init.name}` differs between graphs")
            blobs[init.name] = (raw, init)
            users.setdefault(init.name, set()).add(graph_name)

    data_path = out / data_name
    offsets: dict[str, tuple[int, int]] = {}
    with data_path.open("wb") as handle:
        for name in sorted(blobs):
            raw = blobs[name][0]
            pad = (-handle.tell()) % 64
            handle.write(b"\0" * pad)
            offsets[name] = (handle.tell(), len(raw))
            handle.write(raw)

    for graph_name, model in models.items():
        for init in model.graph.initializer:
            if init.name not in offsets:
                continue
            offset, length = offsets[init.name]
            init.ClearField("raw_data")
            for field in ("float_data", "int32_data", "int64_data", "double_data", "uint64_data", "string_data"):
                init.ClearField(field)
            del init.external_data[:]
            init.data_location = TensorProto.EXTERNAL
            for key, value in (("location", data_name), ("offset", str(offset)), ("length", str(length))):
                entry = init.external_data.add()
                entry.key = key
                entry.value = value
        target = out / graph_name
        target.unlink(missing_ok=True)
        import onnx

        onnx.save_model(model, str(target))

    shared = []
    for name in sorted(offsets):
        if users[name] != set(models):
            continue
        init = blobs[name][1]
        offset, length = offsets[name]
        shared.append({
            "name": name,
            "element": ELEMENTS[init.data_type],
            "shape": [int(d) for d in init.dims],
            "offset": offset,
            "length": length,
        })
    return shared


def drop_moe_bias_placeholders(model) -> None:
    """MoE bias slots are optional: replace the export placeholders by ""."""
    graph = model.graph
    empties = set()
    for node in graph.node:
        if node.op_type == "Constant" and node.attribute and node.attribute[0].t.dims == [0]:
            empties.add(node.output[0])
    empty_inits = {init.name for init in graph.initializer if list(init.dims) == [0]}
    empties |= empty_inits
    for node in graph.node:
        slots = {"MoE": (3, 5), "QMoE": (4, 7)}.get(node.op_type) if node.domain == "com.microsoft" else None
        if slots:
            for slot in slots:
                if slot < len(node.input) and node.input[slot] in empties:
                    node.input[slot] = ""
    used = {name for node in graph.node for name in node.input}
    keep = [node for node in graph.node if not (node.op_type == "Constant" and node.output[0] in empties and node.output[0] not in used)]
    del graph.node[:]
    graph.node.extend(keep)
    keep_inits = [init for init in graph.initializer if not (init.name in empty_inits and init.name not in used)]
    del graph.initializer[:]
    graph.initializer.extend(keep_inits)


def cmd_export(args: argparse.Namespace) -> int:
    torch, _, _ = _lazy_imports()
    source = Path(args.source).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model, _ = load_upstream(source)
    config = model.config
    head_dim = model.model.layers[0].self_attn.head_dim
    vision, prefill, decode = build_modules(
        model,
        args.vision_precision == "fp16",
        8 if args.expert_precision == "int8" else 16,
        build_llm="llm" in args.graphs,
    )
    del model

    layers = config.num_hidden_layers
    heads = config.num_key_value_heads
    shared = None
    if "vision" in args.graphs:
        print("exporting vision graph", flush=True)
        export_graph(
            vision,
            (torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE),),
            out / VISION_GRAPH,
            ["pixel_values"],
            ["image_features"],
            {"pixel_values": {0: "pages"}, "image_features": {0: "pages"}},
        )

    if "llm" in args.graphs:
        capacity = 8
        past_names, present_names, past = [], [], []
        for layer in range(layers):
            for kind in ("key", "value"):
                past_names.append(f"past_key_values.{layer}.{kind}")
                present_names.append(f"present.{layer}.{kind}")
                past.append(torch.zeros(1, heads, capacity, head_dim))
        cache_axes = {name: {2: "capacity"} for name in past_names + present_names}

        print("exporting prefill graph", flush=True)
        seq = 4
        prefill_model = trace_graph(
            prefill,
            (
                torch.tensor([[BOS_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 16]], dtype=torch.int64),
                torch.tensor([[False, True, True, False]]),
                torch.zeros(2, config.hidden_size),
                torch.arange(seq, dtype=torch.int64).unsqueeze(0),
                torch.arange(seq, dtype=torch.int64),
                torch.zeros(1, 1, seq, capacity),
                *past,
            ),
            out / PREFILL_GRAPH,
            ["input_ids", "images_seq_mask", "image_features", "position_ids", "write_index", "attention_bias", *past_names],
            ["logits", *present_names],
            {
                "input_ids": {1: "sequence"},
                "images_seq_mask": {1: "sequence"},
                "image_features": {0: "image_tokens"},
                "position_ids": {1: "sequence"},
                "write_index": {0: "sequence"},
                "attention_bias": {2: "sequence", 3: "capacity"},
                **cache_axes,
            },
        )
        print("exporting decode graph", flush=True)
        decode_model = trace_graph(
            decode,
            (
                torch.tensor([[16]], dtype=torch.int64),
                torch.tensor([[seq]], dtype=torch.int64),
                torch.tensor([seq], dtype=torch.int64),
                torch.zeros(1, 1, 1, capacity),
                *past,
            ),
            out / DECODE_GRAPH,
            ["input_ids", "position_ids", "write_index", "attention_bias", *past_names],
            ["logits", *present_names],
            {"attention_bias": {3: "capacity"}, **cache_axes},
        )
        print("writing shared weights", flush=True)
        shared = save_with_shared_weights({PREFILL_GRAPH: prefill_model, DECODE_GRAPH: decode_model}, out, LLM_DATA)
        for stale in ("unlimited_ocr_llm.onnx", "unlimited_ocr_llm.onnx.data"):
            (out / stale).unlink(missing_ok=True)

    previous = out / "manifest.json"
    if previous.is_file():
        old = json.loads(previous.read_text(encoding="utf-8"))
        if shared is None:
            # Only the vision graph was re-exported: keep the LLM's weight table.
            shared = old.get("device_shared_initializers", {}).get("tensors")
        if "vision" not in args.graphs:
            args.vision_precision = old.get("precision", {}).get("vision", args.vision_precision)
        if "llm" not in args.graphs:
            args.expert_precision = old.get("precision", {}).get("routed_experts")
    elif "llm" not in args.graphs:
        args.expert_precision = None  # no LLM graphs in this directory
    for name in TOKENIZER_FILES:
        shutil.copy2(source / name, out / name)
    write_manifest(out, config, args, head_dim, shared, time.time() - started)
    print(f"package written to {out}", flush=True)
    return 0


def write_manifest(out: Path, config, args, head_dim: int, shared, seconds: float) -> None:
    torch, _, _ = _lazy_imports()
    transformers = importlib.import_module("transformers")
    files = sorted(p.name for p in out.iterdir() if p.is_file() and p.name != "manifest.json")
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "adapter": ADAPTER,
        "precision": {
            "llm": "fp16-weights-fp32-activations",
            "routed_experts": getattr(args, "expert_precision", "fp16"),
            "vision": args.vision_precision,
        },
        "graphs": {"vision": VISION_GRAPH, "prefill": PREFILL_GRAPH, "decode": DECODE_GRAPH},
        "runtime": {
            "image_size": IMAGE_SIZE,
            "image_tokens_per_page": IMAGE_TOKENS,
            "image_token_id": IMAGE_TOKEN_ID,
            "bos_token_id": BOS_TOKEN_ID,
            "eos_token_id": EOS_TOKEN_ID,
            "prompt": PROMPT,
            "num_hidden_layers": config.num_hidden_layers,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": head_dim,
            "hidden_size": config.hidden_size,
            "vocab_size": config.vocab_size,
            "sliding_window": int(getattr(config, "sliding_window_size", None) or config.sliding_window),
            "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
            "ngram_window": NGRAM_WINDOW,
        },
        "device_shared_initializers": {
            "graphs": [PREFILL_GRAPH, DECODE_GRAPH],
            "data_file": LLM_DATA,
            "tensors": shared or [],
        },
        "files": {name: {"bytes": (out / name).stat().st_size, "sha256": sha256(out / name)} for name in files},
        "provenance": {
            "script": "scripts/local/unlimited_ocr_export.py",
            "source": "baidu/Unlimited-OCR",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "opset": OPSET,
            "export_seconds": round(seconds, 1),
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


def banned_ngrams(history: list[int], size: int, window: int) -> set[int]:
    """Upstream ``SlidingWindowNoRepeatNgramProcessor`` (no whitelist)."""
    if len(history) < size:
        return set()
    start = max(0, len(history) - window)
    end = len(history) - size + 1
    prefix = tuple(history[-(size - 1):]) if size > 1 else tuple()
    return {history[i + size - 1] for i in range(start, end) if tuple(history[i:i + size - 1]) == prefix}


def pick(logits, history: list[int]) -> int:
    import numpy as np

    scores = np.array(logits, dtype=np.float32, copy=True)
    for token in banned_ngrams(history, NO_REPEAT_NGRAM_SIZE, NGRAM_WINDOW):
        scores[token] = -np.inf
    return int(scores.argmax())


class OnnxPackage:
    """Python mirror of the Rust runtime contract (used for parity only)."""

    def __init__(self, package: Path, provider: str, tf32: bool = True, vision_graph: Path | None = None):
        import onnxruntime as ort

        manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
        self.runtime = manifest["runtime"]
        cuda = ("CUDAExecutionProvider", {"use_tf32": "1" if tf32 else "0"})
        providers = [cuda, "CPUExecutionProvider"] if provider == "cuda" else ["CPUExecutionProvider"]
        self.device = "cuda" if provider == "cuda" else "cpu"
        # Sessions are opened one at a time and released after use: Python
        # cannot share the LLM weights between the prefill and decode sessions
        # as the Rust runtime does, and two copies need ~12 GB of VRAM.
        self.paths = {
            "vision": vision_graph or package / manifest["graphs"]["vision"],
            "prefill": package / manifest["graphs"]["prefill"],
            "decode": package / manifest["graphs"]["decode"],
        }
        self.providers = providers

    def session(self, name: str):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.log_severity_level = 3
        session = ort.InferenceSession(str(self.paths[name]), options, providers=self.providers)
        print(f"{name} provider:", session.get_providers()[0], flush=True)
        return session

    def image_features(self, pixel_values):
        vision = self.session("vision")
        features = vision.run(None, {"pixel_values": pixel_values})[0]
        del vision
        return features

    def generate(self, ids: list[int], mask: list[bool], features, max_new: int, on_step=None):
        import numpy as np
        import onnxruntime as ort

        layers = self.runtime["num_hidden_layers"]
        heads = self.runtime["num_key_value_heads"]
        head_dim = self.runtime["head_dim"]
        window = self.runtime["sliding_window"]
        prompt = len(ids)
        capacity = prompt + min(window, max_new)
        caches = [
            ort.OrtValue.ortvalue_from_numpy(np.zeros((1, heads, capacity, head_dim), np.float32), self.device, 0)
            for _ in range(2 * layers)
        ]
        current = {"name": None, "session": None}

        def step(tokens, image_mask, image_rows, positions, slots, bias):
            name = "prefill" if image_rows is not None else "decode"
            if current["name"] != name:
                current["session"] = None  # release the previous graph first
                current["session"] = self.session(name)
                current["name"] = name
            session = current["session"]
            binding = session.io_binding()
            binding.bind_cpu_input("input_ids", np.array([tokens], np.int64))
            if image_rows is not None:
                binding.bind_cpu_input("images_seq_mask", np.array([image_mask], bool))
                binding.bind_cpu_input("image_features", image_rows)
            binding.bind_cpu_input("position_ids", np.array([positions], np.int64))
            binding.bind_cpu_input("write_index", np.array(slots, np.int64))
            binding.bind_cpu_input("attention_bias", bias)
            binding.bind_output("logits", "cpu")
            for layer in range(layers):
                for j, kind in enumerate(("key", "value")):
                    cache = caches[2 * layer + j]
                    binding.bind_ortvalue_input(f"past_key_values.{layer}.{kind}", cache)
                    binding.bind_ortvalue_output(f"present.{layer}.{kind}", cache)
            session.run_with_iobinding(binding)
            return binding.copy_outputs_to_cpu()[0][0]

        causal = np.full((1, 1, prompt, capacity), -np.inf, np.float32)
        for row in range(prompt):
            causal[0, 0, row, : row + 1] = 0.0
        flat = features.reshape(-1, features.shape[-1]).astype(np.float32)
        logits = step(ids, mask, flat, list(range(prompt)), list(range(prompt)), causal)
        history = list(ids)
        generated: list[int] = []
        all_logits = [logits]
        for t in range(max_new):
            token = pick(logits, history)
            if on_step:
                on_step(t, token)
            if token == EOS_TOKEN_ID:
                break
            generated.append(token)
            history.append(token)
            if t + 1 == max_new:
                break
            slot = prompt + (t % window)
            bias = np.full((1, 1, 1, capacity), -np.inf, np.float32)
            bias[..., : prompt + min(t + 1, window)] = 0.0
            logits = step([token], None, None, [prompt + t], [slot], bias)
            all_logits.append(logits)
        return generated, all_logits


def upstream_generate(model, ids, mask, pixel_values, max_new: int, window: int):
    torch, _, _ = _lazy_imports()
    transformers = importlib.import_module("transformers")
    inner = model.model
    pixel_values = pixel_values.to(model.dtype)
    base = type(inner).__mro__[1]  # DeepseekV2Model: skips the CUDA-only image branch
    with torch.no_grad():
        sam = inner.sam_model(pixel_values)
        clip = inner.vision_model(pixel_values, sam)
        features = inner.projector(torch.cat((clip[:, 1:], sam.flatten(2).permute(0, 2, 1)), dim=-1))
        dim = features.shape[-1]
        grid = features.view(GRID, GRID, dim)
        grid = torch.cat([grid, inner.image_newline[None, None, :].expand(GRID, 1, dim)], dim=1).view(-1, dim)
        features = torch.cat([grid, inner.view_seperator[None, :]], dim=0)

        embeds = inner.embed_tokens(torch.tensor([ids]))
        embeds[0][torch.tensor(mask)] = features
        model.config._ring_window = window
        model.config.sliding_window = None
        cache = transformers.DynamicCache()
        out = base.forward(inner, inputs_embeds=embeds, past_key_values=cache, use_cache=True,
                           position_ids=torch.arange(len(ids)).unsqueeze(0))
        logits = model.lm_head(out.last_hidden_state[:, -1, :])[0].float()
        history = list(ids)
        generated: list[int] = []
        all_logits = [logits.numpy()]
        for t in range(max_new):
            token = pick(logits.numpy(), history)
            if token == EOS_TOKEN_ID:
                break
            generated.append(token)
            history.append(token)
            if t + 1 == max_new:
                break
            out = base.forward(inner, input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True,
                               position_ids=torch.tensor([[len(ids) + t]]))
            logits = model.lm_head(out.last_hidden_state[:, -1, :])[0].float()
            all_logits.append(logits.numpy())
    return features.float().numpy(), generated, all_logits


def cmd_parity(args: argparse.Namespace) -> int:
    import numpy as np

    source = Path(args.source).resolve()
    package = Path(args.package).resolve()
    pixel_values = preprocess(Path(args.image))
    onnx_pkg = OnnxPackage(package, args.provider, tf32=not args.no_tf32,
                           vision_graph=Path(args.vision_graph).resolve() if args.vision_graph else None)
    t0 = time.time()
    features = onnx_pkg.image_features(pixel_values.numpy())
    t_vision = time.time() - t0

    model, tokenizer = load_upstream(source)
    if args.reference_dtype == "bf16":
        torch, _, _ = _lazy_imports()
        model = model.to(torch.bfloat16)
    ids, mask = prompt_ids(tokenizer)
    t0 = time.time()
    generated, logits = onnx_pkg.generate(ids, mask, features, args.max_new_tokens)
    t_llm = time.time() - t0
    print(f"onnx: vision {t_vision:.2f}s, generate {len(generated)} tokens in {t_llm:.2f}s", flush=True)
    print("onnx text:", tokenizer.decode(generated)[:600], flush=True)

    window = onnx_pkg.runtime["sliding_window"]
    ref_features, ref_generated, ref_logits = upstream_generate(model, ids, mask, pixel_values, args.max_new_tokens, window)
    a, b = features[0].astype(np.float64).ravel(), ref_features.astype(np.float64).ravel()
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    print("vision max|diff|:", float(np.abs(a - b).max()), "mean|diff|:", float(np.abs(a - b).mean()),
          "cosine:", round(cosine, 7), "ref max:", float(np.abs(b).max()))
    print("prefill logits max|diff|:", float(np.abs(logits[0] - ref_logits[0]).max()),
          "argmax", int(np.argmax(logits[0])), int(np.argmax(ref_logits[0])))
    same = 0
    for a, b in zip(generated, ref_generated):
        if a != b:
            break
        same += 1
    print(f"greedy tokens: onnx {len(generated)}, upstream {len(ref_generated)}, identical prefix {same}")
    if same < min(len(generated), len(ref_generated)):
        def top(values):
            order = np.argsort(-values)[:5]
            return [(tokenizer.decode([int(i)]), round(float(values[i]), 3)) for i in order]
        print(f"step {same} top-5 onnx:", top(logits[same]))
        print(f"step {same} top-5 upstream:", top(ref_logits[same]))
    swapped, _ = onnx_pkg.generate(ids, mask, ref_features[None].astype(np.float32), args.max_new_tokens)
    swapped_same = next((i for i, (a, b) in enumerate(zip(swapped, ref_generated)) if a != b), min(len(swapped), len(ref_generated)))
    print(f"onnx llm on upstream vision features: identical prefix {swapped_same}")
    steps = min(len(logits), len(ref_logits), same + 1)
    diffs = [float(np.abs(logits[i] - ref_logits[i]).max()) for i in range(steps)]
    if diffs:
        print("per-step logits max|diff|: max", max(diffs), "mean", sum(diffs) / len(diffs))
    print("upstream text:", tokenizer.decode(ref_generated)[:600])
    return 0 if same == min(len(generated), len(ref_generated)) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="export the vision and LLM graphs from the PyTorch checkpoint")
    export.add_argument("--source", required=True, help="upstream baidu/Unlimited-OCR checkpoint directory")
    export.add_argument("--out", required=True, help="package directory to write")
    export.add_argument("--vision-precision", choices=("fp32", "fp16"), default="fp16",
                        help="fp16 keeps normalization layers in fp32; OCR text matches fp32 (boxes may move by 1/1000)")
    export.add_argument("--graphs", nargs="+", choices=("vision", "llm"), default=["vision", "llm"])
    export.add_argument("--expert-precision", choices=("int8", "fp16"), default="int8",
                        help="routed MoE experts: int8 halves the LLM weights (QMoE in both graphs)")
    export.set_defaults(func=cmd_export)

    parity = sub.add_parser("parity", help="compare a package with the upstream PyTorch model")
    parity.add_argument("--source", required=True)
    parity.add_argument("--package", required=True)
    parity.add_argument("--image", required=True)
    parity.add_argument("--max-new-tokens", type=int, default=200)
    parity.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parity.add_argument("--no-tf32", action="store_true", help="disable TF32 in the CUDA provider")
    parity.add_argument("--vision-graph", help="vision graph to use instead of the package's")
    parity.add_argument("--reference-dtype", choices=("fp32", "bf16"), default="fp32")
    parity.set_defaults(func=cmd_parity)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
