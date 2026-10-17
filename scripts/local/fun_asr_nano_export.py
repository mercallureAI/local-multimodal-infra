"""Export Fun-ASR-Nano-2512 to ONNX graphs for CUDA (weight-only int8).

Fun-ASR-Nano is SenseVoiceSmall's encoder (SAN-M, 50 + 20 layers, the same
80-bin FBANK + LFR 7/6 frontend, no CMVN) feeding two heads: a 2-layer
transformer adaptor whose output replaces placeholder tokens in a Qwen3-0.6B
prompt (the LLM writes the transcript), and a 5-layer transformer CTC decoder
over a 60515-token vocabulary (the multilingual tiktoken of SenseVoice). The
package read by ``crates/adapters/asr/fun-asr-nano`` has two graphs:

* ``asr/audio_encoder.onnx``: ``speech`` [1, T, 560] (LFR features) and
  ``speech_lengths`` [1] to ``audio_embeds`` [1, T, 1024] (adaptor output),
  ``ctc_ids`` [1, T] (greedy CTC token per frame, argmax taken on the GPU)
  and ``encoder_out_lens``. Only the first ``fake_token_len`` (about T / 8)
  adaptor frames enter the prompt: the adaptor attends over every frame and
  was trained to put the utterance there (upstream ``data_load_speech``).
  Linear weights int8 (``MatMulNBits``, blocks of 32), activations float32:
  the adaptor output reaches |1e3| and the encoder outgrows float16 on loud
  speech (see ``sensevoice_cuda_export``). The CTC head (512 x 60515) stays
  float32: for more than one row the CUDA ``MatMulNBits`` kernel dequantizes
  its whole weight into a float32 scratch buffer (2 x 118 MB here), which
  every fixed-length session's arena would keep. Weights in ``.onnx.data`` at
  256-byte aligned offsets with ``audio_encoder.shared.json``, so the runtime
  uploads them once for its fixed-length CUDA graph sessions.
* ``asr/llm.onnx``: one Qwen3 step over ``embeds`` [1, S, 1024]
  (float32: token embeddings, looked up by the runtime in
  ``asr/embed_tokens.f16``, with the adaptor output at the audio
  positions): the 28 layers from the onnxruntime-genai builder (FP16,
  GroupQueryAttention over a past/present-shared KV cache), the tied LM
  head at the last position and its argmax: ``next_id`` [1, 1]. Prefill (S
  prompt positions) and decode (S = 1) use the same graph; decode shapes are
  static, so the CUDA EP can capture and replay it. Layer and head matmuls
  int8 (``MatMulNBits``, blocks of 128). The embedding table (151936 x 1024
  FP16, 311 MB) stays in host memory: a decode step needs one row of it.
  The residual stream runs scaled by ``RESIDUAL_SCALE``: layer 2's MLP
  writes ~1.25e5 into it at the first token (Qwen3's massive activation),
  past FP16's 65504, which turns every later layer into NaN. The stream is
  only read through RMSNorm, so scaling the embeddings and the o_proj and
  down_proj weights by a power of two changes nothing else.

Plus ``asr/embed_tokens.f16`` (the token embeddings, row-major little-endian
FP16), ``asr/config.yaml`` (upstream: the frontend), ``asr/tokenizer.json``
(Qwen3), ``asr/ctc_tokens.json`` (the CTC vocabulary, base64 bytes per id)
and ``asr/manifest.json`` (prompt token ids, dimensions, special ids). The
VAD and speaker graphs come from an existing sensevoice-small-onnx directory
(``--base-dir``: vad/, speaker/).

Source: the ModelScope snapshot of ``FunAudioLLM/Fun-ASR-Nano-2512`` with the
trained CTC tensors (``model.pt`` SHA-256 ``81fec861...``; the Hugging Face
copy lacks them). Upstream decodes greedily with the prompt::

    <|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n
    <|im_start|>user\\n语音转写：[audio]<|im_end|>\\n<|im_start|>assistant\\n

``parity`` runs the package (onnxruntime, CPU or CUDA) and the PyTorch model
on audio files and compares the adaptor output, the CTC tokens and the
transcripts token by token.

Environment: Python 3.12 with ``torch>=2.9`` and ``torchaudio``,
``funasr>=1.3.26`` (it has ``FunASRNano``), ``transformers>=4.51``,
``onnx``, ``onnxruntime>=1.30`` (CPU is enough to export; ``parity`` takes
``onnxruntime-gpu`` for CUDA), ``safetensors``; the LLM body needs an
interpreter with ``onnxruntime-genai`` (``--builder-python``)::

    python -m scripts.local.fun_asr_nano_export export \\
      --source-model-dir workdir/models/Fun-ASR-Nano-2512 \\
      --base-dir workdir/models/sensevoice-small-onnx \\
      --output-dir workdir/models/fun-asr-nano-onnx \\
      --builder-python <python with onnxruntime-genai>
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from scripts.local.sensevoice_cuda_export import save_package, sort_topologically

MODEL_SHA256 = "81fec8616083c69377f3ceef36aba3655660ee0ca69a5d4a1e9810cd340ca499"
MODELSCOPE_REPO = "FunAudioLLM/Fun-ASR-Nano-2512"
MODELSCOPE_REVISION = "db823224e5c2510001b4a3d3ed97f430bd52ac88"
AUDIO_FILE = "audio_encoder.onnx"
LLM_FILE = "llm.onnx"
EMBED_FILE = "embed_tokens.f16"
MANIFEST_SCHEMA = "local.fun_asr_nano.package.v1"
SYSTEM_PROMPT = "You are a helpful assistant."
USER_PROMPT = "语音转写："
AUDIO_BLOCK = 32
LLM_BLOCK = 128
OPSET = 17
# FP32 maxima on ASR prompts: residual stream 1.26e5 (layers 2-27), layer 2
# down_proj output 1.25e5 (its input 1.5e4); scaled, 1.6e4.
RESIDUAL_SCALE = 0.125


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="export the package from the PyTorch checkpoint")
    export.add_argument("--source-model-dir", type=Path, required=True,
                        help=f"ModelScope snapshot of {MODELSCOPE_REPO}")
    export.add_argument("--base-dir", type=Path, required=True,
                        help="a sensevoice-small-onnx directory (vad/, speaker/)")
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--work-dir", type=Path, help="intermediate files (default: a temp dir)")
    export.add_argument("--builder-python", type=Path, default=Path(sys.executable),
                        help="interpreter with onnxruntime-genai and a transformers recent enough for its builder")
    export.add_argument("--parts", nargs="*", default=["audio", "llm", "package"],
                        choices=["audio", "llm", "package"])
    export.add_argument(
        "--check-wav", type=Path,
        default=Path(__file__).resolve().parents[1] / "assets" / "tts-input-mon3tr.wav")
    export.set_defaults(run=cmd_export)
    parity = commands.add_parser("parity", help="compare a package with the PyTorch model")
    parity.add_argument("--source-model-dir", type=Path, required=True)
    parity.add_argument("--package-dir", type=Path, required=True)
    parity.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    parity.add_argument("audio", type=Path, nargs="+")
    parity.set_defaults(run=cmd_parity)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.run(args)
    return 0


def load_model(source: Path):
    """The FunASRNano model of the snapshot, float32 on the CPU."""
    import torch
    from funasr import AutoModel

    digest = sha256(source / "model.pt")
    if digest != MODEL_SHA256:
        print(f"[fun-asr-nano] warning: model.pt SHA-256 {digest} is not the checked {MODEL_SHA256}")
    wrapper = AutoModel(model=str(source), device="cpu", disable_update=True)
    model = wrapper.model.float().eval()
    if model.ctc_decoder is None:
        raise SystemExit("the checkpoint has no trained CTC decoder (use the ModelScope snapshot)")
    torch.set_grad_enabled(False)
    return wrapper, model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def fake_token_len(frames: int) -> int:
    """Audio placeholders in the prompt for ``frames`` LFR frames (upstream
    ``use_low_frame_rate``: three halvings of a 3-wide, stride-2 window)."""
    length = 1 + (frames - 3 + 2) // 2
    length = 1 + (length - 3 + 2) // 2
    return (length - 1) // 2 + 1


# --------------------------------------------------------------------------
# Audio encoder graph
# --------------------------------------------------------------------------


def audio_module(model):
    """Encoder + adaptor + CTC decoder as one traceable module. The adaptors
    are run here with a mask built from the input width (upstream's
    ``make_pad_mask`` would trace the batch's longest length as a constant)."""
    import torch

    class AudioGraph(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = model.audio_encoder
            self.adaptor = model.audio_adaptor
            self.ctc_decoder = model.ctc_decoder
            self.ctc_lo = model.ctc.ctc_lo

        @staticmethod
        def adapt(block, x, mask):
            if block.k != 1:
                raise SystemExit(f"adaptor downsample_rate {block.k} is not 1")
            x = block.linear2(block.relu(block.linear1(x)))
            for layer in block.blocks:
                x, mask = layer(x, mask)
            return x

        def forward(self, speech, speech_lengths):
            encoded, lengths = self.encoder(speech, speech_lengths)
            frames = torch.arange(encoded.shape[1], device=encoded.device)
            mask = (frames[None, :] < lengths[:, None])[:, None, :]
            embeds = self.adapt(self.adaptor, encoded, mask)
            ctc = self.adapt(self.ctc_decoder, encoded, mask)
            ids = self.ctc_lo(ctc).argmax(-1)
            return embeds, ids, lengths

    return AudioGraph().eval()


def export_audio(model, work: Path):
    """The float32 graph (traced), then the weight-only int8 one."""
    import onnx
    import torch

    fp32 = work / "audio_fp32.onnx"
    if not fp32.is_file():
        speech = torch.randn(2, 30, 560)
        lengths = torch.tensor([6, 30], dtype=torch.int32)
        torch.onnx.export(
            audio_module(model), (speech, lengths), str(fp32),
            input_names=["speech", "speech_lengths"],
            output_names=["audio_embeds", "ctc_ids", "encoder_out_lens"],
            dynamic_axes={
                "speech": {0: "batch_size", 1: "feats_length"},
                "speech_lengths": {0: "batch_size"},
                "audio_embeds": {0: "batch_size", 1: "feats_length"},
                "ctc_ids": {0: "batch_size", 1: "feats_length"},
                "encoder_out_lens": {0: "batch_size"},
            },
            opset_version=OPSET, dynamo=False)
        print(f"[audio] traced {fp32}")
    quantized = quantize(onnx.load(str(fp32)), AUDIO_BLOCK, exclude_prefixes=("/ctc_lo/",))
    model_proto = sort_topologically(quantized)
    onnx.checker.check_model(model_proto)
    return fp32, model_proto


def quantize(model, block: int, exclude_prefixes: tuple[str, ...] = ()):
    """Weight-only int8 MatMulNBits (symmetric) for every MatMul with a
    constant weight, but those whose name starts with one of
    ``exclude_prefixes``."""
    import collections

    from onnxruntime.quantization.matmul_nbits_quantizer import (
        DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer)

    inits = {init.name for init in model.graph.initializer}
    others = [n.name for n in model.graph.node if n.op_type == "MatMul"
              and (n.input[1] not in inits or n.name.startswith(exclude_prefixes))]
    config = DefaultWeightOnlyQuantConfig(block_size=block, is_symmetric=True, accuracy_level=None,
                                          bits=8)
    quantizer = MatMulNBitsQuantizer(model, bits=8, block_size=block, is_symmetric=True,
                                     nodes_to_exclude=others, algo_config=config)
    quantizer.process()
    quantized = quantizer.model.model
    ops = collections.Counter(node.op_type for node in quantized.graph.node)
    print(f"[quantize] {ops['MatMulNBits']} matmuls -> int8 (block {block}), {ops['MatMul']} left")
    return quantized


# --------------------------------------------------------------------------
# LLM graph
# --------------------------------------------------------------------------


def build_llm_body(model, source: Path, work: Path, builder_python: Path):
    """The 28 Qwen3 layers from the onnxruntime-genai builder: inputs_embeds
    in, final-norm hidden_states out, GQA with shared KV."""
    import onnx
    from safetensors.torch import save_file

    body_dir = work / "llm_body"
    if (body_dir / "model.onnx").is_file() and (body_dir / "model.onnx.data").is_file():
        print(f"[llm] reusing builder output in {body_dir}")
        return onnx.load(str(body_dir / "model.onnx"), load_external_data=True)
    hf_dir = work / "llm_hf"
    hf_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "Qwen3-0.6B" / "config.json").read_text(encoding="utf-8"))
    config["torch_dtype"] = "float32"
    (hf_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    state = {}
    for name, tensor in model.llm.state_dict().items():
        if name == "lm_head.weight":
            continue
        tensor = tensor.detach().float().cpu()
        if name.endswith(("self_attn.o_proj.weight", "mlp.down_proj.weight")):
            tensor = tensor * RESIDUAL_SCALE
        state[name] = tensor.contiguous()
    save_file(state, str(hf_dir / "model.safetensors"))
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


def assemble_llm(body, model):
    """The builder body with the embedding front and the greedy head."""
    import torch
    from onnx import TensorProto, helper, numpy_helper

    graph = body.graph
    keep = [o for o in graph.output if o.name != "hidden_states"]
    inputs = [i for i in graph.input if i.name != "inputs_embeds"]
    del graph.output[:]
    graph.output.extend(keep)
    del graph.input[:]
    graph.input.extend(inputs)
    hidden = model.llm.config.hidden_size
    table = embed_table(model)
    head = model.llm.lm_head.weight.detach().float().cpu().numpy()
    if not np.array_equal(table, head):
        raise SystemExit("the LM head is not tied to the embeddings")
    nodes = []

    def init(name, array):
        graph.initializer.append(numpy_helper.from_array(np.ascontiguousarray(array), name))
        return name

    def node(op, inputs, output, **attrs):
        nodes.append(helper.make_node(op, inputs, [output], name=f"/{output}", **attrs))
        return output

    graph.input.append(helper.make_tensor_value_info(
        "embeds", TensorProto.FLOAT, ["batch_size", "sequence_length", hidden]))
    # Scaled into the residual stream's range (see RESIDUAL_SCALE).
    scaled = node("Mul", ["embeds", init("front.scale", np.array(RESIDUAL_SCALE, np.float32))],
                  "front/scaled")
    node("Cast", [scaled], "inputs_embeds", to=TensorProto.FLOAT16)
    # Last position -> tied LM head -> argmax.
    last = node("Slice", ["hidden_states", init("back.start", np.array([-1], np.int64)),
                          init("back.end", np.array([2**62], np.int64)),
                          init("back.axis", np.array([1], np.int64))], "back/last")
    flat = node("Reshape", [last, init("back.shape", np.array([-1, hidden], np.int64))], "back/flat")
    logits = node("MatMul", [flat, init("back.lm_head", head.T.astype(np.float16))], "back/logits")
    ids = node("ArgMax", [logits], "back/argmax", axis=-1, keepdims=1)
    node("Reshape", [ids, init("back.out_shape", np.array([-1, 1], np.int64))], "next_id")
    graph.output.insert(0, helper.make_tensor_value_info("next_id", TensorProto.INT64, ["batch_size", 1]))
    graph.node.extend(nodes)
    graph.name = "fun_asr_nano_llm_step"
    del torch
    return body


def embed_table(model):
    return model.llm.get_input_embeddings().weight.detach().float().cpu().numpy()


def export_llm(model, source: Path, work: Path, out: Path, builder_python: Path) -> None:
    import onnx

    body = build_llm_body(model, source, work, builder_python)
    merged = quantize(sort_topologically(assemble_llm(body, model)), LLM_BLOCK, exclude_prefixes=("/front",))
    target = out / LLM_FILE
    for stale in (target, out / (LLM_FILE + ".data")):
        stale.unlink(missing_ok=True)
    onnx.save_model(merged, str(target), save_as_external_data=True, all_tensors_to_one_file=True,
                    location=LLM_FILE + ".data", size_threshold=1024)
    embed_table(model).astype("<f2").tofile(out / EMBED_FILE)
    print(f"[llm] wrote {target} and {EMBED_FILE}")


# --------------------------------------------------------------------------
# Package
# --------------------------------------------------------------------------


def prompt_ids(tokenizer) -> tuple[list[int], list[int]]:
    """Token ids before and after the audio placeholders."""
    prefix = (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
              f"<|im_start|>user\n{USER_PROMPT}")
    suffix = "<|im_end|>\n<|im_start|>assistant\n"
    return tokenizer.encode(prefix), tokenizer.encode(suffix)


def write_package(wrapper, model, source: Path, out: Path) -> None:
    llm_config = model.llm.config
    prefix, suffix = prompt_ids(wrapper.kwargs["tokenizer"])
    ctc = model.ctc_tokenizer.encoding
    specials = {token_id: name for name, token_id in ctc._special_tokens.items()}
    tokens = []
    for token_id in range(ctc.n_vocab):
        raw = specials[token_id].encode() if token_id in specials else ctc.decode_single_token_bytes(token_id)
        tokens.append(base64.b64encode(raw).decode("ascii"))
    (out / "ctc_tokens.json").write_text(json.dumps({
        "blank_id": int(model.blank_id),
        "special_start": min(specials),
        "tokens": tokens,
    }), encoding="utf-8")
    shutil.copy2(source / "Qwen3-0.6B" / "tokenizer.json", out / "tokenizer.json")
    shutil.copy2(source / "config.yaml", out / "config.yaml")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "source": {"modelscope": MODELSCOPE_REPO, "revision": MODELSCOPE_REVISION,
                   "model_pt_sha256": MODEL_SHA256},
        "audio_encoder": AUDIO_FILE,
        "llm": LLM_FILE,
        "embed_tokens": EMBED_FILE,
        "feature_dim": 560,
        "hidden_size": llm_config.hidden_size,
        "num_layers": llm_config.num_hidden_layers,
        "num_kv_heads": llm_config.num_key_value_heads,
        "head_dim": llm_config.head_dim,
        "vocab_size": llm_config.vocab_size,
        "eos_token_ids": [151645, 151643],
        "prompt": {
            "system": SYSTEM_PROMPT,
            "user": USER_PROMPT,
            "prefix_ids": prefix,
            "suffix_ids": suffix,
        },
        "ctc_blank_id": int(model.blank_id),
        "ctc_frame_ms": 60,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[package] wrote {out / 'manifest.json'}")


def cmd_export(args: argparse.Namespace) -> None:
    import onnxruntime as ort

    source = args.source_model_dir.resolve()
    work = args.work_dir.resolve() if args.work_dir else Path(tempfile.mkdtemp(prefix="fun-asr-nano-"))
    work.mkdir(parents=True, exist_ok=True)
    out = args.output_dir.resolve()
    asr = out / "asr"
    asr.mkdir(parents=True, exist_ok=True)
    wrapper, model = load_model(source)
    if "audio" in args.parts:
        fp32, quantized = export_audio(model, work)
        staging = work / "audio_package"
        staging.mkdir(exist_ok=True)
        save_package(quantized, staging, AUDIO_FILE)
        feats = features(wrapper, args.check_wav)
        expected = run_audio(ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"]), feats)
        actual = run_audio(ort.InferenceSession(str(staging / AUDIO_FILE),
                                                providers=["CPUExecutionProvider"]), feats)
        reference = torch_audio(model, feats)
        report_audio("fp32 graph vs torch", expected, reference)
        report_audio("int8 graph vs torch", actual, reference)
        for name in (AUDIO_FILE, AUDIO_FILE + ".data", Path(AUDIO_FILE).stem + ".shared.json"):
            shutil.move(str(staging / name), str(asr / name))
    if "llm" in args.parts:
        export_llm(model, source, work, asr, args.builder_python)
    if "package" in args.parts:
        write_package(wrapper, model, source, asr)
        for folder in ("vad", "speaker"):
            shutil.copytree(args.base_dir / folder, out / folder, dirs_exist_ok=True)
    print(f"[fun-asr-nano] package in {out}")


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def features(wrapper, audio: Path) -> dict:
    """FunASR's own frontend for ``audio``: LFR features [1, T, 560]."""
    import torch
    from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video

    frontend = wrapper.kwargs["frontend"]
    data = load_audio_text_image_video(str(audio), fs=frontend.fs)
    speech, lengths = extract_fbank(data, data_type="sound", frontend=frontend, is_final=True)
    return {"speech": speech.float().numpy(), "speech_lengths": lengths.int().numpy().reshape(-1)}


def run_audio(session, feats: dict) -> dict:
    embeds, ids, lengths = session.run(None, feats)
    return {"embeds": embeds, "ids": ids, "lengths": lengths}


def torch_audio(model, feats: dict) -> dict:
    import torch

    speech = torch.from_numpy(feats["speech"])
    lengths = torch.from_numpy(feats["speech_lengths"])
    encoded, olens = model.audio_encoder(speech.clone(), lengths)
    embeds, _ = model.audio_adaptor(encoded, olens)
    ctc, _ = model.ctc_decoder(encoded, olens)
    ids = model.ctc.log_softmax(ctc).argmax(-1)
    return {"embeds": embeds.numpy(), "ids": ids.numpy(), "lengths": olens.numpy()}


def ctc_collapse(ids, blank: int) -> list[int]:
    out, previous = [], None
    for token in ids:
        if token != blank and token != previous:
            out.append(int(token))
        previous = token
    return out


def report_audio(label: str, actual: dict, expected: dict, blank: int = 60514) -> None:
    frames = int(expected["lengths"][0])
    used = fake_token_len(frames)
    a = actual["embeds"][0, :used].astype(np.float64)
    e = expected["embeds"][0, :used].astype(np.float64)
    cosine = float((a * e).sum() / (np.linalg.norm(a) * np.linalg.norm(e)))
    max_abs = float(np.abs(a - e).max())
    ta = ctc_collapse(actual["ids"][0, :frames], blank)
    te = ctc_collapse(expected["ids"][0, :frames], blank)
    print(f"[check] {label}: audio embeds ({used} of {frames} frames) cosine {cosine:.6f}, "
          f"max |diff| {max_abs:.4f} (max |ref| {np.abs(e).max():.1f}); CTC tokens "
          f"{sum(x == y for x, y in zip(ta, te))}/{len(te)} equal{'' if ta == te else ' (DIFFER)'}")


def onnx_transcribe(audio_session, llm_session, manifest: dict, asr_dir: Path, feats: dict,
                    max_new_tokens: int = 200) -> tuple[list[int], dict]:
    """The package's greedy decode in Python (KV grows per step)."""
    audio = run_audio(audio_session, feats)
    frames = int(audio["lengths"][0])
    used = fake_token_len(frames)
    prompt = manifest["prompt"]
    ids = prompt["prefix_ids"] + [-1] * used + prompt["suffix_ids"]
    hidden = manifest["hidden_size"]
    table = np.fromfile(asr_dir / manifest["embed_tokens"], dtype="<f2").reshape(-1, hidden)
    embeds = table[np.maximum(ids, 0)].astype(np.float32)[None]
    start = len(prompt["prefix_ids"])
    embeds[0, start:start + used] = audio["embeds"][0, :used]
    layers, kv_heads, head_dim = manifest["num_layers"], manifest["num_kv_heads"], manifest["head_dim"]
    past = {f"past_key_values.{i}.{p}": np.zeros((1, kv_heads, 0, head_dim), np.float16)
            for i in range(layers) for p in ("key", "value")}
    names = [o.name for o in llm_session.get_outputs()]
    out_tokens: list[int] = []
    step_embeds, total = embeds, len(ids)
    eos = set(manifest["eos_token_ids"])
    for _ in range(max_new_tokens):
        feed = {"embeds": step_embeds,
                "attention_mask": np.ones((1, total), np.int64), **past}
        result = dict(zip(names, llm_session.run(None, feed)))
        token = int(result["next_id"][0, 0])
        if token in eos:
            break
        out_tokens.append(token)
        past = {name: result[name.replace("past_key_values", "present")] for name in past}
        step_embeds = table[[token]].astype(np.float32)[None]
        total += 1
    return out_tokens, audio


def cmd_parity(args: argparse.Namespace) -> None:
    import onnxruntime as ort
    import torch

    wrapper, model = load_model(args.source_model_dir.resolve())
    asr = args.package_dir / "asr"
    manifest = json.loads((asr / "manifest.json").read_text(encoding="utf-8"))
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if args.provider == "cuda"
                 else ["CPUExecutionProvider"])
    audio_session = ort.InferenceSession(str(asr / manifest["audio_encoder"]), providers=providers)
    llm_session = ort.InferenceSession(str(asr / manifest["llm"]), providers=providers)
    tokenizer = wrapper.kwargs["tokenizer"]
    same_texts = 0
    for path in args.audio:
        feats = features(wrapper, path)
        tokens, audio = onnx_transcribe(audio_session, llm_session, manifest, asr, feats)
        report_audio(path.name, audio, torch_audio(model, feats))
        reference = model.inference([str(path)], key=[path.stem], tokenizer=tokenizer,
                                    frontend=wrapper.kwargs["frontend"], device="cpu")[0][0]
        text = tokenizer.decode(tokens)
        same = text == reference["text"]
        same_texts += same
        print(f"[parity] {path.name}: {'same' if same else 'DIFFERENT'}\n  torch: {reference['text']}\n  onnx : {text}")
    print(f"[parity] {same_texts}/{len(args.audio)} transcripts identical")
    del torch


if __name__ == "__main__":
    raise SystemExit(main())
