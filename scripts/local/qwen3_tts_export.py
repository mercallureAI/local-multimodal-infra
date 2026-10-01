"""Export Qwen3-TTS-12Hz-0.6B-Base to ONNX graphs for CUDA + I/O binding.

The graphs are shaped for a Rust decode loop that keeps every tensor on the
GPU and replays one CUDA graph per audio frame:

* ``talker.onnx``  one *frame*: embeds the position's text token, codec codes
  and raw embeddings, runs the 28-layer talker (onnxruntime-genai builder:
  GroupQueryAttention over a past/present-shared FP16 KV cache), samples the
  first codebook in-graph (repetition penalty, suppressed control tokens,
  temperature, top-k, Gumbel-max with host noise) and runs the 5-layer code
  predictor for codebooks 1-15 unrolled (static shapes, in-graph sampling).
  Prefill (S prompt positions) and decode (S = 1) use the same graph; decode
  shapes are static so the CUDA EP can capture and replay it.
* ``vocoder.onnx`` the 12 Hz codec decoder for streaming: codes of a left
  context plus new frames in, only the new frames' 24 kHz audio out (the
  expensive upsampling stack runs on the new frames plus a short receptive
  field margin only).
* ``speaker_encoder.onnx`` / ``codec_encoder.onnx`` reference voice: 24 kHz
  audio to the 1024-d x-vector and to 16-codebook reference codes (run once
  per voice; the adapter caches them).

Run in a Python 3.12 environment with torch (CUDA), the official ``qwen-tts``
package, ``onnx``, ``onnxruntime-gpu`` and ``onnxruntime-genai`` (builder).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ADAPTER = "qwen3_tts"
MANIFEST_SCHEMA = "local.qwen3_tts.package.v1"
NUM_GROUPS = 16
TOP_K = 50
# Talker vocabulary: 2048 codec codes, then 1024 control tokens of which only
# codec EOS may be sampled.
CODEC_SIZE = 2048
NEG = -1.0e9
TALKER_WEIGHTS = ("int8", "fp16", "int4-int8")
# Code predictor layers whose MLP down projection overflows FP16: the
# calibration maxima (FP32 reference, Chinese/English, x-vector and ICL) are
# 1.8e5 at its input and 6.1e4 at its output for layer 2, and the residual
# stream after it holds 6.1e4. The residual stream stays FP32 and these
# layers feed down_proj with input/scale (output*scale back in FP32); FP16
# relative precision does not depend on scale.
CP_DOWN_SCALE = {2: 64.0}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-model-dir", required=True, type=Path, help="Qwen/Qwen3-TTS-12Hz-0.6B-Base snapshot")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--work-dir", type=Path, help="intermediate files (default: a temp dir)")
    parser.add_argument(
        "--talker-weights",
        choices=TALKER_WEIGHTS,
        default="int8",
        help="talker.onnx matmul weights: int8 (weight-only, the default: no measurable quality loss, "
        "27%% faster decode, 40%% smaller), fp16, or int4 for the 28 talker layers with an int8 code predictor",
    )
    parser.add_argument(
        "--builder-python",
        type=Path,
        default=Path(sys.executable),
        help="interpreter with onnxruntime-genai and a transformers recent enough for its builder",
    )
    parser.add_argument(
        "--parts",
        nargs="*",
        default=["talker", "vocoder", "speaker", "codec_encoder", "package"],
        choices=["talker", "vocoder", "speaker", "codec_encoder", "package"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.source_model_dir.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = args.work_dir.expanduser().resolve() if args.work_dir else Path(tempfile.mkdtemp(prefix="qwen3tts-"))
    work.mkdir(parents=True, exist_ok=True)
    model = None

    def load():
        nonlocal model
        if model is None:
            model = load_model(source)
        return model

    if "talker" in args.parts:
        export_talker(load(), work, out, args.builder_python, args.talker_weights)
    if "vocoder" in args.parts:
        export_vocoder(load(), work, out)
    if "speaker" in args.parts:
        export_speaker_encoder(load(), out)
    if "codec_encoder" in args.parts:
        export_codec_encoder(load(), out)
    if "package" in args.parts:
        write_package(source, out, args.talker_weights)
    return 0


def load_model(source: Path):
    import torch
    from qwen_tts import Qwen3TTSModel

    wrapper = Qwen3TTSModel.from_pretrained(
        str(source), device_map="cuda:0", dtype=torch.float32, attn_implementation="eager"
    )
    return wrapper.model.eval()


# --------------------------------------------------------------------------
# Talker frame graph
# --------------------------------------------------------------------------


def export_talker(model, work: Path, out: Path, builder_python: Path, weights: str = "int8") -> None:
    """Builder talker body, wrapped by a torch front (embeddings) and back
    (sampling + code predictor), merged into one graph; matmul weights
    quantized per `weights`."""
    import onnx

    body = build_talker_body(model, work, builder_python)
    merged = quantize_talker(assemble_frame_graph(body, model), weights)
    target = out / "talker.onnx"
    for stale in (target, out / "talker.onnx.data"):
        stale.unlink(missing_ok=True)
    onnx.save_model(
        merged,
        str(target),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="talker.onnx.data",
        size_threshold=1024,
    )
    print(f"[talker] wrote {target}")


def quantize_talker(model, weights: str):
    """Weight-only MatMulNBits (symmetric RTN) for the matmuls with constant
    weights: the 28 talker layers and the code predictor with its heads
    (`/back/`). Embedding tables, norms and sampling stay as they are.

    A/B (scripts/local/qwen3_tts_eval.py, 14 sentences x 2 seeds x 2 modes):
    INT8 everywhere matches FP16 in CER, speaker similarity and DNSMOS; an INT4
    code predictor changes greedy codes from the first frame and lowers
    similarity slightly, so the code predictor stays INT8."""
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        DefaultWeightOnlyQuantConfig,
        MatMulNBitsQuantizer,
    )

    if weights == "fp16":
        return model
    inits = {init.name for init in model.graph.initializer}
    matmuls = [n.name for n in model.graph.node if n.op_type == "MatMul" and n.input[1] in inits]
    back = [name for name in matmuls if name.startswith("/back/")]
    body = [name for name in matmuls if not name.startswith(("/back/", "/front/"))]
    plan = [(back, 8, 128), (body, 4 if weights == "int4-int8" else 8, 32 if weights == "int4-int8" else 128)]
    for names, bits, block in plan:
        chosen = set(names)
        # `nodes_to_include` is OR-ed with the op type and does not restrict:
        # exclude every other MatMul instead.
        others = [n.name for n in model.graph.node if n.op_type == "MatMul" and n.name not in chosen]
        config = DefaultWeightOnlyQuantConfig(block_size=block, is_symmetric=True, bits=bits)
        quantizer = MatMulNBitsQuantizer(
            model, bits=bits, block_size=block, is_symmetric=True, nodes_to_exclude=others, algo_config=config
        )
        quantizer.process()
        model = quantizer.model.model
        print(f"[talker] {len(names)} matmuls -> int{bits} (block {block})")
    return model


def build_talker_body(model, work: Path, builder_python: Path):
    """The 28 decoder layers as a Qwen3 causal LM for the genai builder:
    inputs_embeds in, final-norm hidden_states out, GQA with shared KV."""
    import onnx
    import torch
    from safetensors.torch import save_file

    talker = model.talker
    cfg = talker.config
    body_dir = work / "talker_body"
    if (body_dir / "model.onnx").is_file() and (body_dir / "model.onnx.data").is_file():
        print(f"[talker] reusing builder output in {body_dir}")
        return onnx.load(str(body_dir / "model.onnx"), load_external_data=True)
    hf_dir = work / "talker_hf"
    hf_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "rms_norm_eps": cfg.rms_norm_eps,
        "rope_theta": cfg.rope_theta,
        "max_position_embeddings": cfg.max_position_embeddings,
        "vocab_size": cfg.vocab_size,
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "use_sliding_window": False,
        "sliding_window": None,
        "bos_token_id": cfg.codec_bos_id,
        "eos_token_id": cfg.codec_eos_token_id,
        "torch_dtype": "float32",
    }
    (hf_dir / "config.json").write_text(json.dumps(config, indent=2))
    state = {}
    for name, tensor in talker.model.state_dict().items():
        if name.startswith("layers.") or name == "norm.weight":
            state[f"model.{name}"] = tensor
    state["model.embed_tokens.weight"] = talker.model.codec_embedding.weight
    state["lm_head.weight"] = talker.codec_head.weight
    save_file({k: v.detach().float().cpu().contiguous() for k, v in state.items()}, str(hf_dir / "model.safetensors"))

    if body_dir.exists():
        shutil.rmtree(body_dir)
    subprocess.run(
        [
            str(builder_python), "-m", "onnxruntime_genai.models.builder",
            "-i", str(hf_dir), "-o", str(body_dir), "-p", "fp16", "-e", "cuda",
            "-c", str(work / "builder_cache"),
            "--extra_options", "exclude_embeds=true", "exclude_lm_head=true", "fuse_mlp_gate_up=true",
        ],
        check=True,
    )
    return onnx.load(str(body_dir / "model.onnx"), load_external_data=True)


class GraphBuilder:
    """Appends nodes and initializers to an ONNX graph under a name scope."""

    def __init__(self, graph, scope: str):
        self.graph = graph
        self.scope = scope
        self.count = 0
        self.consts: dict[tuple, str] = {}

    def _name(self, hint: str) -> str:
        self.count += 1
        return f"/{self.scope}/{hint}_{self.count}"

    def init(self, name: str, array: np.ndarray) -> str:
        from onnx import numpy_helper

        full = f"{self.scope}.{name}"
        self.graph.initializer.append(numpy_helper.from_array(np.ascontiguousarray(array), full))
        return full

    def const(self, value, dtype) -> str:
        array = np.array(value, dtype=dtype)
        key = (array.dtype.str, array.shape, array.tobytes())
        if key not in self.consts:
            self.consts[key] = self.init(f"const_{len(self.consts)}", array)
        return self.consts[key]

    def op(self, op_type: str, inputs: list[str], hint: str | None = None, outputs=1, domain: str = "", **attrs):
        from onnx import helper

        if isinstance(outputs, int):
            names = [self._name(f"{hint or op_type}/out{i}") for i in range(outputs)]
        else:
            names = outputs
        self.graph.node.append(
            helper.make_node(op_type, inputs, names, name=self._name(hint or op_type), domain=domain, **attrs)
        )
        return names[0] if len(names) == 1 else names


def _np(t, dtype=np.float16) -> np.ndarray:
    return t.detach().float().cpu().numpy().astype(dtype)


def _slice_last(g: GraphBuilder, x: str) -> str:
    return g.op("Slice", [x, g.const([-1], np.int64), g.const([2**62], np.int64), g.const([1], np.int64)])


def add_front(graph, model):
    """text_ids (1,S), codec_ids (1,S,16), extra_embeds (1,S,H) -> inputs_embeds.

    Returns the codec table (shared with the back half) and its group offsets."""
    import torch
    from onnx import TensorProto, helper

    talker = model.talker
    hidden = talker.config.hidden_size
    g = GraphBuilder(graph, "front")
    with torch.no_grad():
        # text_projection is per token: precompute it for the whole vocabulary.
        emb = talker.model.text_embedding.weight
        text_table = torch.cat(
            [talker.text_projection(emb[i : i + 8192].float()) for i in range(0, emb.shape[0], 8192)]
        )
    codec_tables = [talker.model.codec_embedding.weight] + [
        e.weight for e in talker.code_predictor.model.codec_embedding
    ]
    offsets = np.cumsum([0] + [t.shape[0] for t in codec_tables[:-1]]).astype(np.int64)
    text_w = g.init("text_table", _np(text_table))
    codec_w = g.init("codec_table", np.concatenate([_np(t) for t in codec_tables]))
    graph.input.extend([
        helper.make_tensor_value_info("text_ids", TensorProto.INT64, [1, "sequence_length"]),
        helper.make_tensor_value_info("codec_ids", TensorProto.INT64, [1, "sequence_length", NUM_GROUPS]),
        helper.make_tensor_value_info("extra_embeds", TensorProto.FLOAT16, [1, "sequence_length", hidden]),
    ])
    zero = g.const(0, np.int64)
    axis_last = g.const([-1], np.int64)

    def masked_gather(ids, table, offset=None):
        valid = g.op("Cast", [g.op("GreaterOrEqual", [ids, zero])], to=TensorProto.FLOAT16)
        idx = g.op("Max", [ids, zero])
        if offset is not None:
            idx = g.op("Add", [idx, offset])
        rows = g.op("Gather", [table, idx], axis=0)
        return g.op("Mul", [rows, g.op("Unsqueeze", [valid, axis_last])])

    text = masked_gather("text_ids", text_w)
    codec = masked_gather("codec_ids", codec_w, g.const(offsets, np.int64))
    codec = g.op("ReduceSum", [codec, g.const([2], np.int64)], keepdims=0)
    g.op("Add", [g.op("Add", [text, codec]), "extra_embeds"], outputs=["inputs_embeds"])
    return codec_w, offsets


def add_back(graph, model, codec_table: str, offsets: np.ndarray) -> None:
    """hidden_states (1,S,H) -> codes (1,1,16): samples codebook 0 from the
    codec head at the last position, then runs the code predictor for
    codebooks 1-15, unrolled, with GroupQueryAttention (fused q/k norm and
    rotary, KV concatenated in-op) over at most 17 positions.

    The code predictor's residual stream is FP32 (its layer 2 holds 6e4); the
    GEMMs are FP16."""
    import torch
    from onnx import TensorProto, helper

    talker = model.talker
    tcfg = talker.config
    cp = talker.code_predictor
    ccfg = cp.config
    if not isinstance(cp.small_to_mtp_projection, torch.nn.Identity):
        raise SystemExit("small_to_mtp_projection is not the identity (not a 0.6B checkpoint)")
    hidden = ccfg.hidden_size
    n_heads, n_kv, head_dim = ccfg.num_attention_heads, ccfg.num_key_value_heads, ccfg.head_dim
    inter = ccfg.intermediate_size
    eps = float(ccfg.rms_norm_eps)
    vocab = tcfg.vocab_size
    steps = NUM_GROUPS - 1
    g = GraphBuilder(graph, "back")
    graph.input.extend([
        helper.make_tensor_value_info("seen_tokens", TensorProto.FLOAT, [1, vocab]),
        helper.make_tensor_value_info("noise", TensorProto.FLOAT, [NUM_GROUPS, TOP_K]),
        helper.make_tensor_value_info("sampling", TensorProto.FLOAT, [4]),
    ])
    graph.output.insert(0, helper.make_tensor_value_info("codes", TensorProto.INT64, [1, 1, NUM_GROUPS]))

    # Rotary tables (GQA layout: half width), positions 0..31.
    inv_freq = 1.0 / (ccfg.rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    freqs = np.outer(np.arange(32, dtype=np.float64), inv_freq)
    cos_w = g.init("cos_cache", np.cos(freqs).astype(np.float16))
    sin_w = g.init("sin_cache", np.sin(freqs).astype(np.float16))

    def scalar(i):
        return g.op("Gather", ["sampling", g.const(i, np.int64)], axis=0)

    # sampling = [1/temperature, 1/subtalker_temperature, repetition_penalty, eos_allowed]
    inv_temp, inv_temp_cp, penalty, eos_allowed = (scalar(i) for i in range(4))
    noise_rows = g.op("Split", ["noise", g.const([1] * NUM_GROUPS, np.int64)], axis=0, outputs=NUM_GROUPS)

    def pick(logits32, noise_row):
        # Top-k then Gumbel-max: argmax(log p + g) samples softmax(logits).
        vals, idx = g.op("TopK", [logits32, g.const([TOP_K], np.int64)], outputs=2, axis=-1)
        choice = g.op("ArgMax", [g.op("Add", [vals, noise_row])], axis=-1, keepdims=1)
        return g.op("GatherElements", [idx, choice], axis=-1)  # (1,1) int64

    # Codebook 0 from the codec head at the last position.
    last = _slice_last(g, "hidden_states")  # (1,1,H)
    head_w = g.init("codec_head", _np(talker.codec_head.weight.t()))
    h_last = g.op("Reshape", [last, g.const([1, hidden], np.int64)])
    logits = g.op("Cast", [g.op("MatMul", [h_last, head_w])], to=TensorProto.FLOAT)
    zero_f = g.const(0.0, np.float32)
    penalized = g.op("Where", [g.op("Greater", [logits, zero_f]),
                               g.op("Div", [logits, penalty]), g.op("Mul", [logits, penalty])])
    logits = g.op("Where", [g.op("Greater", ["seen_tokens", zero_f]), penalized, logits])
    suppress = np.zeros((1, vocab), np.float32)
    suppress[0, CODEC_SIZE:] = NEG
    suppress[0, tcfg.codec_eos_token_id] = 0.0
    eos_ban = np.zeros((1, vocab), np.float32)
    eos_ban[0, tcfg.codec_eos_token_id] = NEG
    logits = g.op("Add", [logits, g.init("suppress", suppress)])
    ban = g.op("Mul", [g.init("eos_ban", eos_ban), g.op("Sub", [g.const(1.0, np.float32), eos_allowed])])
    logits = g.op("Mul", [g.op("Add", [logits, ban]), inv_temp])
    code = pick(logits, noise_rows[0])
    codes = [code]

    layers = []
    for i, layer in enumerate(cp.model.layers):
        a, m = layer.self_attn, layer.mlp
        layers.append({
            "ln1": g.init(f"cp.{i}.ln1", _np(layer.input_layernorm.weight, np.float32)),
            "ln2": g.init(f"cp.{i}.ln2", _np(layer.post_attention_layernorm.weight, np.float32)),
            "qkv": g.init(f"cp.{i}.qkv", _np(torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]).t())),
            "qn": g.init(f"cp.{i}.q_norm", _np(a.q_norm.weight)),
            "kn": g.init(f"cp.{i}.k_norm", _np(a.k_norm.weight)),
            "o": g.init(f"cp.{i}.o", _np(a.o_proj.weight.t())),
            "gu": g.init(f"cp.{i}.gate_up", _np(torch.cat([m.gate_proj.weight, m.up_proj.weight]).t())),
            "down": g.init(f"cp.{i}.down", _np(m.down_proj.weight.t())),
            "scale": CP_DOWN_SCALE.get(i, 1.0),
        })
    final_norm = g.init("cp.norm", _np(cp.model.norm.weight, np.float32))
    heads = [g.init(f"cp.head.{i}", _np(h.weight.t())) for i, h in enumerate(cp.lm_head)]
    split_sizes = g.const([n_heads * head_dim, n_kv * head_dim, n_kv * head_dim], np.int64)
    gu_split = g.const([inter, inter], np.int64)

    def embed_code(code_value, group):
        idx = code_value if group == 0 else g.op("Add", [code_value, g.const(int(offsets[group]), np.int64)])
        return g.op("Gather", [codec_table, idx], axis=0)  # (1,1,H) fp16

    # Step 0 sees [talker hidden, embedding of codebook 0]; each later step one
    # new position (the embedding of the code just sampled).
    x = g.op("Cast", [g.op("Concat", [last, embed_code(code, 0)], axis=1)], to=TensorProto.FLOAT)
    past = [("", "")] * len(layers)
    pos = 0
    for step in range(steps):
        s_len = 2 if step == 0 else 1
        total = pos + s_len
        seqlens = g.const([total - 1], np.int32)
        total_len = g.const(total, np.int32)
        residual = x
        mlp_out = None
        for i, w in enumerate(layers):
            if mlp_out is None:
                normed = g.op("SimplifiedLayerNormalization", [residual, w["ln1"]], axis=-1, epsilon=eps, stash_type=1)
            else:
                normed, _, _, residual = g.op(
                    "SkipSimplifiedLayerNormalization", [mlp_out, residual, w["ln1"]],
                    outputs=4, domain="com.microsoft", epsilon=eps)
            h16 = g.op("Cast", [normed], to=TensorProto.FLOAT16)
            q, k, v = g.op("Split", [g.op("MatMul", [h16, w["qkv"]]), split_sizes], axis=-1, outputs=3)
            attn, pk, pv = g.op(
                "GroupQueryAttention",
                [q, k, v, past[i][0], past[i][1], seqlens, total_len, cos_w, sin_w, "", "", "", "", "", w["qn"], w["kn"]],
                outputs=3, domain="com.microsoft", num_heads=n_heads, kv_num_heads=n_kv,
                scale=float(head_dim**-0.5), local_window_size=-1, softcap=0.0, do_rotary=1,
                rotary_interleaved=0, qk_norm_epsilon=eps)
            past[i] = (pk, pv)
            attn32 = g.op("Cast", [g.op("MatMul", [attn, w["o"]])], to=TensorProto.FLOAT)
            normed, _, _, residual = g.op(
                "SkipSimplifiedLayerNormalization", [attn32, residual, w["ln2"]],
                outputs=4, domain="com.microsoft", epsilon=eps)
            gu = g.op("MatMul", [g.op("Cast", [normed], to=TensorProto.FLOAT16), w["gu"]])
            if w["scale"] != 1.0:
                gu = g.op("Cast", [gu], to=TensorProto.FLOAT)
            gate, up = g.op("Split", [gu, gu_split], axis=-1, outputs=2)
            act = g.op("Mul", [g.op("Mul", [gate, g.op("Sigmoid", [gate])]), up])
            if w["scale"] != 1.0:
                act = g.op("Cast", [g.op("Mul", [act, g.const(1.0 / w["scale"], np.float32)])], to=TensorProto.FLOAT16)
                down = g.op("Cast", [g.op("MatMul", [act, w["down"]])], to=TensorProto.FLOAT)
                mlp_out = g.op("Mul", [down, g.const(w["scale"], np.float32)])
            else:
                mlp_out = g.op("Cast", [g.op("MatMul", [act, w["down"]])], to=TensorProto.FLOAT)
        normed, _, _, _ = g.op("SkipSimplifiedLayerNormalization", [mlp_out, residual, final_norm],
                               outputs=4, domain="com.microsoft", epsilon=eps)
        if s_len > 1:
            normed = _slice_last(g, normed)
        h16 = g.op("Reshape", [g.op("Cast", [normed], to=TensorProto.FLOAT16), g.const([1, hidden], np.int64)])
        logits = g.op("Cast", [g.op("MatMul", [h16, heads[step]])], to=TensorProto.FLOAT)
        code = pick(g.op("Mul", [logits, inv_temp_cp]), noise_rows[step + 1])
        codes.append(code)
        pos = total
        if step + 1 < steps:
            x = g.op("Cast", [embed_code(code, step + 1)], to=TensorProto.FLOAT)
    g.op("Reshape", [g.op("Concat", codes, axis=1), g.const([1, 1, NUM_GROUPS], np.int64)], outputs=["codes"])


def assemble_frame_graph(body, model):
    """Wraps the builder body with the front and back halves in place."""
    graph = body.graph
    keep = [o for o in graph.output if o.name != "hidden_states"]
    inputs = [i for i in graph.input if i.name != "inputs_embeds"]
    del graph.output[:]
    graph.output.extend(keep)
    del graph.input[:]
    graph.input.extend(inputs)
    codec_table, offsets = add_front(graph, model)
    add_back(graph, model, codec_table, offsets)
    graph.name = "qwen3_tts_talker_frame"
    return body


# --------------------------------------------------------------------------
# Streaming vocoder
# --------------------------------------------------------------------------

# Frames of pre-transformer output the upsampling stack sees before the new
# frames (its receptive field: 4 matches 25 to within 0.1 dB).
VOCODER_CONV_CONTEXT = 4
# Positions in the vocoder's rotary table: the longest stream (reference
# priming plus generated frames) it decodes.
VOCODER_CAPACITY = 4096


# Set while tracing the vocoder's back half: convs zero the left-context
# frames of their input when the context is not real (a stream's start), which
# is exactly the zero padding they would apply with no context at all.
_CONTEXT_MASK: dict = {}


def _mask_context(x):
    import torch

    if not _CONTEXT_MASK:
        return x
    length = x.shape[-1]
    context = (length // _CONTEXT_MASK["frames"]) * VOCODER_CONV_CONTEXT
    keep = (torch.arange(length, device=x.device) >= context).to(torch.float32)
    mask = torch.maximum(keep, _CONTEXT_MASK["valid"]).to(x.dtype)
    return x * mask


def _shape_free_codec_modules() -> None:
    """Tracing-friendly forwards for the codec decoder's modules.

    The causal convs compute their padding/trim from the input length in
    Python, which tracing bakes in; for stride 1 the extra right padding is
    always 0, so plain left padding and a fixed right trim are equivalent.
    SnakeBeta's exp(alpha) and 1/(exp(beta)+eps) are folded to constants
    (ORT cannot constant-fold FP16 Exp)."""
    import torch
    import torch.nn.functional as F
    from qwen_tts.core.tokenizer_12hz import modeling_qwen3_tts_tokenizer_v2 as codec

    def conv_forward(self, x):
        assert self.stride == 1
        return self.conv(F.pad(_mask_context(x), (self.padding, 0)))

    def trans_forward(self, x):
        x = self.conv(_mask_context(x))
        return x[..., : -self.right_pad] if self.right_pad > 0 else x

    def snake_forward(self, x):
        if not hasattr(self, "folded_alpha"):
            self.folded_alpha = torch.exp(self.alpha.float()).to(self.alpha.dtype)[None, :, None]
            self.folded_inv_beta = (1.0 / (torch.exp(self.beta.float()) + self.no_div_by_zero)).to(self.beta.dtype)[None, :, None]
        s = torch.sin(x * self.folded_alpha)
        return x + self.folded_inv_beta * (s * s)

    codec.Qwen3TTSTokenizerV2CausalConvNet.forward = conv_forward
    codec.Qwen3TTSTokenizerV2CausalTransConvNet.forward = trans_forward
    codec.SnakeBeta.forward = snake_forward


def build_vocoder_halves(model, dtype):
    """The codec decoder around its pre-transformer, as two torch modules:

    * front: codes (1,16,N) + ``pre_ctx`` (1,512,2), the quantized latents of
      the 2 frames before (zeros at a stream's start) -> the transformer
      input (1,N,512) and the next ``pre_ctx``;
    * back: transformer output (1,N,1024) + ``conv_ctx`` (1,1024,4), the
      transformer outputs of the 4 frames before, + ``context_valid`` (1,),
      0 at a stream's start (the context is then masked out) -> the N frames'
      24 kHz audio and the next ``conv_ctx`` (N >= 4).

    Shapes stay fixed for a fixed N: ORT re-plans every cuDNN convolution
    (about 40 ms in all) whenever an input shape changes.
    """
    import torch

    _shape_free_codec_modules()
    dec = model.speech_tokenizer.model.decoder
    cfg = dec.config
    upsample = int(np.prod(cfg.upsample_rates + cfg.upsampling_ratios))

    def codebook(vq):
        cb = vq._codebook
        return (cb.embedding_sum / cb.cluster_usage.clamp(min=cb.epsilon)[:, None]).detach()

    rvq = [dec.quantizer.rvq_first, dec.quantizer.rvq_rest]
    tables = torch.stack([codebook(layer) for r in rvq for layer in r.vq.layers])  # (16, 2048, 256)
    flat = tables.reshape(-1, tables.shape[-1])
    offsets = torch.arange(NUM_GROUPS)[:, None] * tables.shape[1]

    class Front(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("table", flat.to(dtype))
            self.register_buffer("offsets", offsets)
            self.out_first = rvq[0].output_proj
            self.out_rest = rvq[1].output_proj
            self.pre_conv = dec.pre_conv.conv
            self.input_proj = dec.pre_transformer.input_proj

        def forward(self, codes, pre_ctx):
            emb = self.table[codes[0] + self.offsets]  # (16, N, 256)
            q = self.out_first(emb[0].t()[None]) + self.out_rest(emb[1:].sum(0).t()[None])  # (1, 512, N)
            q_all = torch.cat([pre_ctx, q], dim=-1)
            h = self.pre_conv(q_all).transpose(1, 2)  # causal k=3 over the 2 frames before
            return self.input_proj(h), q_all[..., -2:]

    class Back(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.upsample = dec.upsample
            self.decoder = dec.decoder

        def forward(self, h, conv_ctx, context_valid):
            n = h.shape[1]
            x = torch.cat([conv_ctx, h.transpose(1, 2)], dim=-1)
            next_conv = x[..., -VOCODER_CONV_CONTEXT:]
            _CONTEXT_MASK.update(frames=n + VOCODER_CONV_CONTEXT, valid=context_valid.float())
            try:
                for blocks in self.upsample:
                    for block in blocks:
                        x = block(x)
                for block in self.decoder:
                    x = block(x)
            finally:
                _CONTEXT_MASK.clear()
            return x.clamp(-1, 1)[0, :, -n * upsample:].float(), next_conv

    front = Front().to(dtype).cuda().eval()
    back = Back().to(dtype).cuda().eval()
    for module in back.modules():
        for name in ("folded_alpha", "folded_inv_beta"):
            if hasattr(module, name):
                delattr(module, name)
    return front, back, cfg


def _prefix_graph(model, prefix: str, keep: set[str]) -> None:
    """Prefixes every node, value and initializer name except `keep`."""
    g = model.graph

    def rename(name: str) -> str:
        return name if not name or name in keep else prefix + name

    for node in g.node:
        node.name = prefix + node.name
        for i, name in enumerate(node.input):
            node.input[i] = rename(name)
        for i, name in enumerate(node.output):
            node.output[i] = rename(name)
    for init in g.initializer:
        init.name = rename(init.name)
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        vi.name = rename(vi.name)


def add_vocoder_transformer(graph, model, capacity: int) -> None:
    """The codec decoder's 8-layer pre-transformer, ``transformer_in``
    (1,N,512) -> ``transformer_out`` (1,N,1024), with GroupQueryAttention
    over a past/present-shared KV cache (sliding window via
    local_window_size, rotary in-op) so streaming is exact."""
    from onnx import TensorProto, helper

    dec = model.speech_tokenizer.model.decoder
    cfg = dec.config
    tr = dec.pre_transformer
    n_heads, n_kv, head_dim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    eps = float(cfg.rms_norm_eps)
    inter = cfg.intermediate_size
    g = GraphBuilder(graph, "transformer")
    graph.input.extend([
        helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, [1]),
        helper.make_tensor_value_info("total_sequence_length", TensorProto.INT32, []),
    ])
    inv_freq = tr.rotary_emb.inv_freq.detach().double().cpu().numpy()
    freqs = np.outer(np.arange(capacity, dtype=np.float64), inv_freq)
    cos_w = g.init("cos_cache", np.cos(freqs).astype(np.float16))
    sin_w = g.init("sin_cache", np.sin(freqs).astype(np.float16))
    import torch

    x = "transformer_in"
    mlp_out = None
    residual = x
    for i, layer in enumerate(tr.layers):
        a, m = layer.self_attn, layer.mlp
        graph.input.extend([
            helper.make_tensor_value_info(f"past_key.{i}", TensorProto.FLOAT16, [1, n_kv, "capacity", head_dim]),
            helper.make_tensor_value_info(f"past_value.{i}", TensorProto.FLOAT16, [1, n_kv, "capacity", head_dim]),
        ])
        graph.output.extend([
            helper.make_tensor_value_info(f"present_key.{i}", TensorProto.FLOAT16, [1, n_kv, "capacity", head_dim]),
            helper.make_tensor_value_info(f"present_value.{i}", TensorProto.FLOAT16, [1, n_kv, "capacity", head_dim]),
        ])
        ln1 = g.init(f"{i}.ln1", _np(layer.input_layernorm.weight))
        if mlp_out is None:
            normed = g.op("SimplifiedLayerNormalization", [residual, ln1], axis=-1, epsilon=eps, stash_type=1)
        else:
            normed, _, _, residual = g.op("SkipSimplifiedLayerNormalization", [mlp_out, residual, ln1],
                                          outputs=4, domain="com.microsoft", epsilon=eps)
        qkv_w = g.init(f"{i}.qkv", _np(torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]).t()))
        q, k, v = g.op("Split", [g.op("MatMul", [normed, qkv_w]),
                                 g.const([n_heads * head_dim, n_kv * head_dim, n_kv * head_dim], np.int64)],
                       axis=-1, outputs=3)
        attn, _, _ = g.op(
            "GroupQueryAttention",
            [q, k, v, f"past_key.{i}", f"past_value.{i}", "seqlens_k", "total_sequence_length", cos_w, sin_w],
            outputs=[g._name("gqa/out"), f"present_key.{i}", f"present_value.{i}"],
            domain="com.microsoft", num_heads=n_heads, kv_num_heads=n_kv, scale=float(head_dim**-0.5),
            # HF's sliding window of 72 attends to the query and the 71 before.
            local_window_size=cfg.sliding_window - 1, do_rotary=1, rotary_interleaved=0)
        attn = g.op("MatMul", [attn, g.init(f"{i}.o", _np(a.o_proj.weight.t()))])
        attn = g.op("Mul", [attn, g.init(f"{i}.attn_scale", _np(layer.self_attn_layer_scale.scale))])
        normed, _, _, residual = g.op(
            "SkipSimplifiedLayerNormalization", [attn, residual, g.init(f"{i}.ln2", _np(layer.post_attention_layernorm.weight))],
            outputs=4, domain="com.microsoft", epsilon=eps)
        gu = g.op("MatMul", [normed, g.init(f"{i}.gate_up", _np(torch.cat([m.gate_proj.weight, m.up_proj.weight]).t()))])
        gate, up = g.op("Split", [gu, g.const([inter, inter], np.int64)], axis=-1, outputs=2)
        act = g.op("Mul", [g.op("Mul", [gate, g.op("Sigmoid", [gate])]), up])
        mlp_out = g.op("Mul", [g.op("MatMul", [act, g.init(f"{i}.down", _np(m.down_proj.weight.t()))]),
                               g.init(f"{i}.mlp_scale", _np(layer.mlp_layer_scale.scale))])
    normed, _, _, _ = g.op("SkipSimplifiedLayerNormalization", [mlp_out, residual, g.init("norm", _np(tr.norm.weight))],
                           outputs=4, domain="com.microsoft", epsilon=eps)
    out = g.op("MatMul", [normed, g.init("output_proj", _np(tr.output_proj.weight.t()))])
    g.op("Add", [out, g.init("output_proj.bias", _np(tr.output_proj.bias))], outputs=["transformer_out"])


def export_vocoder(model, work: Path, out: Path, capacity: int = VOCODER_CAPACITY) -> None:
    """vocoder.onnx: new frames' codes + stream state -> their audio + state.

    Inputs: ``codes`` (1,16,N>=4), ``pre_ctx`` (1,512,2), ``conv_ctx``
    (1,1024,4), ``context_valid`` (1,) (0 at a stream's start), ``past_key.i``/``past_value.i`` (1,16,capacity,64) shared
    with ``present_*`` (bind the same buffer), ``seqlens_k`` (frames so far
    including the new ones, minus 1) and ``total_sequence_length`` (the KV
    capacity, or frames so far for an unshared cache). Outputs: ``audio``
    (1,N*1920) FP32, ``next_pre_ctx``, ``next_conv_ctx``, ``present_*``."""
    import onnx
    import torch

    dtype = torch.float16
    front, back, cfg = build_vocoder_halves(model, dtype)
    n = 7
    with torch.no_grad():
        torch.onnx.export(
            front,
            (torch.randint(0, cfg.codebook_size, (1, NUM_GROUPS, n), device="cuda"),
             torch.zeros(1, cfg.codebook_dim, 2, dtype=dtype, device="cuda")),
            str(work / "vocoder_front.onnx"),
            input_names=["codes", "pre_ctx"],
            output_names=["transformer_in", "next_pre_ctx"],
            dynamic_axes={"codes": {2: "frames"}, "transformer_in": {1: "frames"}},
            opset_version=20,
            do_constant_folding=False,
            dynamo=False,
        )
        torch.onnx.export(
            back,
            (torch.zeros(1, n, cfg.latent_dim, dtype=dtype, device="cuda"),
             torch.zeros(1, cfg.latent_dim, VOCODER_CONV_CONTEXT, dtype=dtype, device="cuda"),
             torch.zeros(1, dtype=torch.float32, device="cuda")),
            str(work / "vocoder_back.onnx"),
            input_names=["transformer_out", "conv_ctx", "context_valid"],
            output_names=["audio", "next_conv_ctx"],
            dynamic_axes={"transformer_out": {1: "frames"}, "audio": {1: "samples"}},
            opset_version=20,
            do_constant_folding=False,
            dynamo=False,
        )
    f = onnx.load(str(work / "vocoder_front.onnx"))
    b = onnx.load(str(work / "vocoder_back.onnx"))
    _prefix_graph(f, "front/", {"codes", "pre_ctx", "transformer_in", "next_pre_ctx"})
    _prefix_graph(b, "back/", {"transformer_out", "conv_ctx", "context_valid", "audio", "next_conv_ctx"})
    graph = onnx.helper.make_graph(
        nodes=list(f.graph.node),
        name="qwen3_tts_vocoder_stream",
        inputs=list(f.graph.input) + [i for i in b.graph.input if i.name != "transformer_out"],
        outputs=list(b.graph.output) + [o for o in f.graph.output if o.name != "transformer_in"],
        initializer=list(f.graph.initializer),
    )
    add_vocoder_transformer(graph, model, capacity)
    graph.node.extend(b.graph.node)
    graph.initializer.extend(b.graph.initializer)
    merged = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 20), onnx.helper.make_opsetid("com.microsoft", 1)]
    )
    merged.ir_version = 10
    target = out / "vocoder.onnx"
    for stale in (target, out / "vocoder.onnx.data"):
        stale.unlink(missing_ok=True)
    onnx.save_model(merged, str(target), save_as_external_data=True, all_tensors_to_one_file=True,
                    location="vocoder.onnx.data", size_threshold=1024)
    print(f"[vocoder] wrote {target}")


def _rotate_half(x):
    import torch

    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


# --------------------------------------------------------------------------
# Reference voice encoders (run once per voice; FP32)
# --------------------------------------------------------------------------

# The codec encoder takes audio zero-padded to whole frames (a multiple of
# 1920 samples at 24 kHz): exactly what its causal convs' own right padding
# does, so tracing needs no length-dependent padding.
CODEC_FRAME_SAMPLES = 1920


def _trace_friendly_masks() -> None:
    """transformers builds causal masks with torch.vmap, which the tracer
    cannot follow; equivalent plain-tensor masks."""
    import torch
    import transformers.masking_utils as mu

    def mask(config, input_embeds, attention_mask, cache_position, past_key_values, position_ids=None,
             sliding=False, **kwargs):
        dtype, device = input_embeds.dtype, input_embeds.device
        batch, q_len = input_embeds.shape[:2]
        kv_len = q_len + (past_key_values.get_seq_length() if past_key_values is not None else 0)
        rows = cache_position.view(-1, 1)
        cols = torch.arange(kv_len, device=device).view(1, -1)
        attend = cols <= rows
        window = getattr(config, "sliding_window", None)
        if sliding and window is not None:
            attend = attend & (cols > rows - window)
        m = torch.where(attend, torch.tensor(0.0, dtype=dtype, device=device),
                        torch.tensor(torch.finfo(dtype).min, dtype=dtype, device=device))
        return m[None, None].expand(batch, 1, -1, -1)

    def causal(*args, **kwargs):
        return mask(*args, sliding=False, **kwargs)

    def sliding(*args, **kwargs):
        return mask(*args, sliding=True, **kwargs)

    import transformers.models.mimi.modeling_mimi as mimi
    from qwen_tts.core.models import modeling_qwen3_tts as tts

    for module in (mu, mimi, tts):
        if hasattr(module, "create_causal_mask"):
            module.create_causal_mask = causal
        if hasattr(module, "create_sliding_window_causal_mask"):
            module.create_sliding_window_causal_mask = sliding


def build_speaker_module(model):
    """audio (1,L) 24 kHz FP32 -> x-vector (1,1024): the official log-mel
    front end (reflect pad, Hann STFT as a convolution, slaney mel) and the
    ECAPA-TDNN, with mask-free attentive pooling (one unpadded clip)."""
    import torch
    import torch.nn.functional as F
    from librosa.filters import mel as librosa_mel_fn
    from qwen_tts.core.models import modeling_qwen3_tts as tts

    def asp_forward(self, x):
        mean = x.mean(dim=2, keepdim=True)
        std = torch.sqrt((x - mean).pow(2).mean(dim=2, keepdim=True).clamp(self.eps))
        attention = torch.cat([x, mean.expand_as(x), std.expand_as(x)], dim=1)
        attention = F.softmax(self.conv(self.tanh(self.tdnn(attention))), dim=2)
        mean = (attention * x).sum(2)
        std = torch.sqrt((attention * (x - mean.unsqueeze(2)).pow(2)).sum(2).clamp(self.eps))
        return torch.cat((mean, std), dim=1).unsqueeze(2)

    tts.AttentiveStatisticsPooling.forward = asp_forward
    n_fft, hop, win, n_mels, sr = 1024, 256, 1024, 128, 24000
    mel = torch.from_numpy(librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0, fmax=12000)).float()
    window = torch.hann_window(win).double()
    k = torch.arange(n_fft // 2 + 1, dtype=torch.float64)[:, None]
    t = torch.arange(n_fft, dtype=torch.float64)[None]
    ang = 2 * np.pi * k * t / n_fft
    real = (torch.cos(ang) * window).float()[:, None]
    imag = (-torch.sin(ang) * window).float()[:, None]

    class Speaker(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mel", mel)
            self.register_buffer("real", real)
            self.register_buffer("imag", imag)
            self.encoder = model.speaker_encoder.float()

        def forward(self, audio):
            pad = (n_fft - hop) // 2
            y = F.pad(audio[:, None], (pad, pad), mode="reflect")
            re = F.conv1d(y, self.real, stride=hop)
            im = F.conv1d(y, self.imag, stride=hop)
            spec = torch.sqrt(re * re + im * im + 1e-9)
            mels = torch.log(torch.clamp(self.mel @ spec, min=1e-5))  # (1, 128, frames)
            return self.encoder(mels.transpose(1, 2))

    return Speaker().cuda().eval()


def export_speaker_encoder(model, out: Path) -> None:
    import torch

    module = build_speaker_module(model)
    target = out / "speaker_encoder.onnx"
    with torch.no_grad():
        torch.onnx.export(
            module, (torch.randn(1, 24000 * 3, device="cuda") * 0.1,), str(target),
            input_names=["audio"], output_names=["speaker"], dynamic_axes={"audio": {1: "samples"}},
            opset_version=20, dynamo=False,
        )
    print(f"[speaker] wrote {target}")


def export_codec_encoder(model, out: Path) -> None:
    """audio (1, F*1920) 24 kHz FP32, zero-padded to whole frames -> codes (1,16,F)."""
    import torch
    import torch.nn.functional as F
    import transformers.models.mimi.modeling_mimi as mimi

    _trace_friendly_masks()

    def conv_forward(self, x, padding_cache=None):
        # Whole-frame input: every strided conv's extra right padding is 0.
        assert self.causal and padding_cache is None and self.pad_mode in ("constant", "replicate")
        return self.conv(F.pad(x, (int(self.padding_total), 0), mode=self.pad_mode))

    mimi.MimiConv1d.forward = conv_forward
    tok = model.speech_tokenizer.model
    valid = int(tok.encoder_valid_num_quantizers)
    assert int(tok.encode_downsample_rate) == CODEC_FRAME_SAMPLES

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = tok.encoder.float()

        def forward(self, audio):
            return self.encoder.encode(input_values=audio[:, None], return_dict=True).audio_codes[:, :valid]

    module = Encoder().cuda().eval()
    target = out / "codec_encoder.onnx"
    with torch.no_grad():
        torch.onnx.export(
            module, (torch.randn(1, 75 * CODEC_FRAME_SAMPLES, device="cuda") * 0.1,), str(target),
            input_names=["audio"], output_names=["codes"],
            dynamic_axes={"audio": {1: "samples"}, "codes": {2: "frames"}},
            opset_version=20, dynamo=False,
        )
    print(f"[codec_encoder] wrote {target}")


def write_package(source: Path, out: Path, talker_weights: str = "int8") -> None:
    """config.json (dimensions, special tokens, languages, sampling defaults)
    and a fast-tokenizer tokenizer.json for the text."""
    from transformers import AutoTokenizer

    cfg = json.loads((source / "config.json").read_text(encoding="utf-8"))
    gen = json.loads((source / "generation_config.json").read_text(encoding="utf-8"))
    tok_cfg = json.loads((source / "speech_tokenizer" / "config.json").read_text(encoding="utf-8"))
    t = cfg["talker_config"]
    c = t["code_predictor_config"]
    d = tok_cfg["decoder_config"]
    package = {
        "schema": MANIFEST_SCHEMA,
        "adapter": ADAPTER,
        "source": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "hidden_size": t["hidden_size"],
        "talker_layers": t["num_hidden_layers"],
        "talker_kv_heads": t["num_key_value_heads"],
        "head_dim": t["head_dim"],
        "talker_vocab_size": t["vocab_size"],
        "max_position_embeddings": t["max_position_embeddings"],
        "num_code_groups": t["num_code_groups"],
        "codebook_size": c["vocab_size"],
        "top_k": TOP_K,
        "sample_rate": tok_cfg["output_sample_rate"],
        "frame_rate": tok_cfg["output_sample_rate"] / tok_cfg["decode_upsample_rate"],
        "samples_per_frame": tok_cfg["decode_upsample_rate"],
        "tokens": {
            "tts_bos": cfg["tts_bos_token_id"],
            "tts_eos": cfg["tts_eos_token_id"],
            "tts_pad": cfg["tts_pad_token_id"],
            "codec_bos": t["codec_bos_id"],
            "codec_eos": t["codec_eos_token_id"],
            "codec_pad": t["codec_pad_id"],
            "codec_think": t["codec_think_id"],
            "codec_nothink": t["codec_nothink_id"],
            "codec_think_bos": t["codec_think_bos_id"],
            "codec_think_eos": t["codec_think_eos_id"],
        },
        "languages": t["codec_language_id"],
        "graphs": {
            "talker": "talker.onnx",
            "vocoder": "vocoder.onnx",
            "speaker_encoder": "speaker_encoder.onnx",
            "codec_encoder": "codec_encoder.onnx",
        },
        "precision": "fp16",
        "talker_weights": talker_weights,
        "vocoder": {
            "layers": d["num_hidden_layers"],
            "kv_heads": d["num_key_value_heads"],
            "head_dim": d["head_dim"],
            "kv_window": d["sliding_window"],
            "pre_ctx_channels": d["codebook_dim"],
            "pre_ctx_frames": 2,
            "conv_ctx_channels": d["latent_dim"],
            "conv_ctx_frames": VOCODER_CONV_CONTEXT,
            "max_frames": VOCODER_CAPACITY,
        },
        "codec_frame_samples": CODEC_FRAME_SAMPLES,
        "generation": {
            "do_sample": gen.get("do_sample", True),
            "temperature": gen.get("temperature", 0.9),
            "top_k": gen.get("top_k", TOP_K),
            "repetition_penalty": gen.get("repetition_penalty", 1.05),
            "subtalker_temperature": gen.get("subtalker_temperature", 0.9),
            "subtalker_top_k": gen.get("subtalker_top_k", TOP_K),
            "min_new_tokens": 2,
        },
    }
    if package["generation"]["top_k"] != TOP_K or package["generation"]["subtalker_top_k"] != TOP_K:
        raise SystemExit(f"top_k is baked into talker.onnx as {TOP_K}")
    (out / "config.json").write_text(json.dumps(package, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(str(source), use_fast=True)
    tokenizer.backend_tokenizer.save(str(out / "tokenizer.json"))
    print(f"[package] wrote {out / 'config.json'} and tokenizer.json")


if __name__ == "__main__":
    sys.exit(main())
