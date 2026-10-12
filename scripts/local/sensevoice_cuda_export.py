"""Export SenseVoiceSmall to a weight-only int8 ONNX graph for CUDA (low latency).

The int8 graph sensevoice-small-onnx uses (``model_quant.onnx``) is
``onnxruntime.quantization.quantize_dynamic``'s (FunASR ``quantize=True``):
uint8 per-channel weights and 281 ``DynamicQuantizeLinear`` +
``MatMulInteger`` pairs, which have no CUDA kernels for uint8 activations,
so under the CUDA provider each layer goes back and forth to the CPU:
~200-350 ms per utterance even on an RTX 4090.

This graph quantizes the same 281 linear layers' weights to int8
(``MatMulNBits``, 8 bits, blocks of 32, symmetric) and keeps the activations
in float32: everything runs on the GPU, as fast as the float32 graph, closer
to it than the dynamic int8 graph (99.8% against 99.0% of a 30-minute
podcast's characters), in a third of its GPU memory.

A run launches ~2400 small kernels (~23 ms whatever the length), so the
runtime replays it as CUDA graphs, one fixed-length session per bucket
(metadata ``asr_cuda_graph_frames``): ~9 ms for 1.5 s, ~16 ms for 21 s. For
that the package is ``asr/model_gpu_int8.onnx`` with an extra ``ctc_ids``
output (the argmax over the vocabulary, taken on the GPU), its weights in
``model_gpu_int8.onnx.data`` at 256-byte aligned offsets (uploaded to the
GPU in one copy and shared by every session) and their byte ranges in
``model_gpu_int8.shared.json``.

Not float16: SenseVoice's residual stream and its attention and feed-forward
outputs outgrow float16 (65504) on loud speech, and the converter also turns
FunASR's ``.float()`` LayerNorm casts into float16 ones. The CUDA kernels
then produce inf/NaN logits, i.e. empty text for the whole segment: 36 of
165 segments of that podcast, a quarter of its text (CPU kernels accumulate
in float32 and never showed it). Keeping only the LayerNorms or the residual
adds in float32 does not help.

``export``: FunAudioLLM/SenseVoiceSmall (pinned revision) exported by FunASR
with the TorchScript exporter (torch < 2.9: 2.9's default dynamo exporter
writes a graph ONNX Runtime rejects), then quantized; the CTC tokens of the
float32 and int8 graphs are compared on a WAV. Inputs ``speech`` [B, T, 560],
``speech_lengths``, ``language``, ``textnorm``; outputs ``ctc_logits``,
``encoder_out_lens`` (the int8 graph's interface) and ``ctc_ids``. The other
files come from an existing sensevoice-small-onnx directory (``--base-dir``:
asr/am.mvn, config.yaml, tokens.json, the FSMN-VAD and CAM++ graphs). Run
with an isolated dependency environment, for example::

    uv run --python 3.11 --with "torch==2.5.1" --with "torchaudio==2.5.1" \\
      --with funasr --with "transformers>=4.40,<4.50" --with "numpy<2" \\
      --with onnx --with "onnxruntime==1.30.0" --with huggingface_hub \\
      python -m scripts.local.sensevoice_cuda_export export \\
        --base-dir workdir/models/sensevoice-small-onnx \\
        --output-dir workdir/models/sensevoice-small-int8-gpu-onnx

``from-fp16``: an earlier float16 export (``asr/model_fp16.onnx``) widened
back to float32 (its weights keep their float16 rounding, which that graph
was checked against) and quantized, in place; needs only ``onnx`` and
``onnxruntime``::

    python -m scripts.local.sensevoice_cuda_export from-fp16 \\
      --model-dir workdir/models/sensevoice-small-fp16-onnx
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

REPO_ID = "FunAudioLLM/SenseVoiceSmall"
REVISION = "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7"
SOURCE_FILES = ("am.mvn", "chn_jpn_yue_eng_ko_spectok.bpe.model", "config.yaml",
                "configuration.json", "model.pt")
MODEL_FILE = "model_gpu_int8.onnx"
DATA_FILE = MODEL_FILE + ".data"
SHARED_FILE = "model_gpu_int8.shared.json"
PACKAGE_FILES = (MODEL_FILE, DATA_FILE, SHARED_FILE)
IDS_OUTPUT = "ctc_ids"
# Initializers from this size on go to DATA_FILE, at offsets aligned to
# ALIGNMENT (local-backend-ort's PACKED_ALIGNMENT).
EXTERNAL_THRESHOLD = 1024
ALIGNMENT = 256
FLOAT16_MODEL_FILE = "model_fp16.onnx"
BLOCK_SIZE = 32


def features(wav: Path, base: Path):
    """The graph input for ``wav`` (16-bit PCM, resampled to 16 kHz):
    FunASR's own frontend (80 fbank, LFR 7/6, CMVN)."""
    import wave

    import numpy as np
    import torch
    import torchaudio
    from funasr.frontends.wav_frontend import WavFrontend

    with wave.open(str(wav)) as reader:
        if reader.getsampwidth() != 2:
            raise SystemExit(f"{wav}: 16-bit PCM wanted")
        rate = reader.getframerate()
        pcm = np.frombuffer(reader.readframes(reader.getnframes()), dtype=np.int16)
        audio = pcm.reshape(-1, reader.getnchannels()).mean(axis=1) / 32768.0
    frontend = WavFrontend(cmvn_file=str(base / "asr" / "am.mvn"), fs=16000, window="hamming",
                           n_mels=80, frame_length=25, frame_shift=10, lfr_m=7, lfr_n=6)
    samples = torch.tensor(audio, dtype=torch.float32)[None]
    if rate != 16000:
        samples = torchaudio.functional.resample(samples, rate, 16000)
    feats, lengths = frontend(samples, torch.tensor([samples.shape[1]]))
    return {
        "speech": feats.numpy().astype(np.float32),
        "speech_lengths": lengths.numpy().astype(np.int32),
        "language": np.array([0], dtype=np.int32),
        "textnorm": np.array([14], dtype=np.int32),
    }


def ctc_tokens(session, inputs):
    """The graph's greedy CTC tokens (repeats and blanks dropped)."""
    import numpy as np

    feed = {}
    for spec in session.get_inputs():
        value = inputs[spec.name]
        feed[spec.name] = value.astype(np.int64) if spec.type == "tensor(int64)" else value
    logits = session.run(["ctc_logits"], feed)[0][0]
    ids = logits.argmax(-1)
    return [int(t) for i, t in enumerate(ids) if t != 0 and (i == 0 or t != ids[i - 1])]


def quantize(model):
    """The weight-only int8 graph of a float32 SenseVoice graph."""
    import collections

    from onnxruntime.quantization.matmul_nbits_quantizer import (
        DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer)

    config = DefaultWeightOnlyQuantConfig(block_size=BLOCK_SIZE, is_symmetric=True,
                                          accuracy_level=None, bits=8)
    quantizer = MatMulNBitsQuantizer(model, algo_config=config)
    quantizer.process()
    quantized = quantizer.model.model
    ops = collections.Counter(node.op_type for node in quantized.graph.node)
    # The 281 linear layers; the 140 attention products (activation by
    # activation) stay MatMul.
    if ops["MatMulNBits"] != 281:
        raise SystemExit(f"expected 281 quantized linear layers, got {ops['MatMulNBits']}")
    return quantized


def upcast(model):
    """A float16 graph widened back to float32: initializers, constants,
    casts and value types."""
    import numpy as np
    import onnx
    from onnx import TensorProto, numpy_helper

    graph = model.graph
    for index, initializer in enumerate(graph.initializer):
        if initializer.data_type == TensorProto.FLOAT16:
            widened = numpy_helper.to_array(initializer).astype(np.float32)
            graph.initializer[index].CopyFrom(numpy_helper.from_array(widened, initializer.name))
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.name == "to" and attribute.i == TensorProto.FLOAT16:
                attribute.i = TensorProto.FLOAT
            if (attribute.type == onnx.AttributeProto.TENSOR
                    and attribute.t.data_type == TensorProto.FLOAT16):
                widened = numpy_helper.to_array(attribute.t).astype(np.float32)
                attribute.t.CopyFrom(numpy_helper.from_array(widened, attribute.t.name))
    for info in list(graph.value_info) + list(graph.input) + list(graph.output):
        if info.type.tensor_type.elem_type == TensorProto.FLOAT16:
            info.type.tensor_type.elem_type = TensorProto.FLOAT
    return model


def sort_topologically(model):
    """The graph's nodes in an order where each follows its inputs'
    producers (the float16 converter left input casts after their users)."""
    graph = model.graph
    available = {i.name for i in graph.input} | {i.name for i in graph.initializer} | {""}
    pending, ordered = list(graph.node), []
    while pending:
        ready = [n for n in pending if all(i in available for i in n.input)]
        if not ready:
            raise SystemExit(f"graph has a cycle or a missing input near {pending[0].name}")
        for node in ready:
            ordered.append(node)
            available.update(node.output)
        ready_ids = {id(n) for n in ready}
        pending = [n for n in pending if id(n) not in ready_ids]
    del graph.node[:]
    graph.node.extend(ordered)
    return model


def export(args: argparse.Namespace) -> None:
    import onnx
    import onnxruntime as ort

    source = args.source_model_dir
    if source is None:
        from huggingface_hub import hf_hub_download

        for name in SOURCE_FILES:
            path = Path(hf_hub_download(REPO_ID, name, revision=REVISION))
        source = path.parent
    from funasr import AutoModel

    work = Path(tempfile.mkdtemp(prefix="sensevoice-export-"))
    try:
        for name in SOURCE_FILES:
            shutil.copy2(Path(source) / name, work / name)
        AutoModel(model=str(work), device="cpu", disable_update=True).export(
            type="onnx", quantize=False)
        fp32 = work / "model.onnx"
        package = work / "package"
        package.mkdir()
        save_package(finish(quantize(onnx.load(str(fp32)))), package)
        checked = package / MODEL_FILE

        inputs = features(args.check_wav, args.base_dir)
        expected = ctc_tokens(
            ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"]), inputs)
        actual = ctc_tokens(
            ort.InferenceSession(str(checked), providers=["CPUExecutionProvider"]), inputs)
        same = sum(a == b for a, b in zip(expected, actual))
        print(f"[sensevoice-cuda] {args.check_wav.name}: {len(expected)} float32 tokens, "
              f"{len(actual)} int8, {same} the same")
        if actual != expected:
            raise SystemExit("the int8 graph's tokens differ from the float32 graph's")

        # Only a checked graph reaches the model directory: all of the base
        # directory but its dynamic int8 SenseVoice graph (vad/ keeps its own
        # model_quant.onnx), then the weight-only int8 one.
        shutil.copytree(args.base_dir, args.output_dir, dirs_exist_ok=True,
                        ignore=lambda folder, names: ["model_quant.onnx"]
                        if Path(folder).name == "asr" else [])
        for name in PACKAGE_FILES:
            shutil.move(str(package / name), str(args.output_dir / "asr" / name))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    report(args.output_dir / "asr")


def from_fp16(args: argparse.Namespace) -> None:
    import onnx

    asr = args.model_dir / "asr"
    source = asr / FLOAT16_MODEL_FILE
    save_package(finish(quantize(upcast(onnx.load(str(source))))), asr)
    report(asr, source.name)


def finish(model):
    """The quantized graph sorted, checked, with a ``ctc_ids`` output: the
    greedy CTC token per output frame (int64 [B, T]), taken on the GPU so a
    run returns T ids instead of T x 25055 logits."""
    import onnx
    from onnx import TensorProto, helper

    model = sort_topologically(model)
    graph = model.graph
    graph.node.append(helper.make_node(
        "ArgMax", ["ctc_logits"], [IDS_OUTPUT], axis=-1, keepdims=0, name="/ctc_ids/ArgMax"))
    graph.output.append(helper.make_tensor_value_info(
        IDS_OUTPUT, TensorProto.INT64, ["batch_size", "logits_length"]))
    onnx.checker.check_model(model)
    return model


def save_package(model, folder: Path) -> None:
    """``model`` as MODEL_FILE with its large initializers in DATA_FILE, and
    SHARED_FILE listing their byte ranges there: the runtime uploads them to
    the GPU once for all its fixed-length (CUDA graph) sessions."""
    import json

    import onnx

    ONNX_ELEMENTS = {1: "f32", 2: "u8", 3: "i8", 5: "i16", 6: "i32", 7: "i64", 9: "bool",
                     10: "f16"}
    from onnx import external_data_helper, numpy_helper

    for name in PACKAGE_FILES:
        (folder / name).unlink(missing_ok=True)
    # Written here rather than by onnx.save: every tensor starts at a
    # multiple of ALIGNMENT, so the runtime uploads the file in one copy and
    # views each tensor inside it (aligned for the CUDA kernels' loads).
    tensors, offset = [], 0
    with open(folder / DATA_FILE, "wb") as data:
        for tensor in model.graph.initializer:
            raw = tensor.raw_data or numpy_helper.to_array(tensor).tobytes()
            if len(raw) < EXTERNAL_THRESHOLD:
                continue
            padding = -offset % ALIGNMENT
            data.write(b"\0" * padding)
            offset += padding
            data.write(raw)
            for field in ("float_data", "int32_data", "int64_data", "double_data", "uint64_data"):
                tensor.ClearField(field)
            tensor.raw_data = raw
            external_data_helper.set_external_data(tensor, DATA_FILE, offset, len(raw))
            tensor.ClearField("raw_data")
            tensor.data_location = onnx.TensorProto.EXTERNAL
            tensors.append({
                "name": tensor.name,
                "element": ONNX_ELEMENTS[tensor.data_type],
                "shape": [int(dim) for dim in tensor.dims],
                "offset": offset,
                "length": len(raw),
            })
            offset += len(raw)
    onnx.save(model, str(folder / MODEL_FILE))
    (folder / SHARED_FILE).write_text(
        json.dumps({"data_file": DATA_FILE, "tensors": tensors}, indent=1), encoding="utf-8")


def report(folder: Path, source: str | None = None) -> None:
    size = sum((folder / name).stat().st_size for name in PACKAGE_FILES) / 1e6
    origin = f" from {source}" if source else ""
    print(f"[sensevoice-cuda] wrote {folder / MODEL_FILE} and its weights ({size:.1f} MB){origin}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    from_torch = commands.add_parser("export", help="export from the PyTorch checkpoint")
    from_torch.add_argument("--base-dir", type=Path, required=True,
                            help="a sensevoice-small-onnx directory (asr/, vad/, speaker/)")
    from_torch.add_argument("--output-dir", type=Path, required=True)
    from_torch.add_argument("--source-model-dir", type=Path,
                            help="a local copy of the HF repo (default: download it)")
    from_torch.add_argument(
        "--check-wav", type=Path,
        default=Path(__file__).resolve().parents[1] / "assets" / "tts-input-mon3tr.wav")
    from_torch.set_defaults(run=export)
    convert = commands.add_parser(
        "from-fp16", help=f"asr/{FLOAT16_MODEL_FILE} -> asr/{MODEL_FILE}, in place")
    convert.add_argument("--model-dir", type=Path, required=True,
                         help="a directory with an earlier float16 export")
    convert.set_defaults(run=from_fp16)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
