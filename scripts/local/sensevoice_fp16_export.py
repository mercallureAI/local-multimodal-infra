"""Export SenseVoiceSmall to a float16 ONNX graph (low latency on CUDA).

The int8 graph sensevoice-small-onnx uses (``model_quant.onnx``) runs its 281
``DynamicQuantizeLinear`` nodes on the CPU under the CUDA provider, so each
layer goes back and forth: ~200-350 ms per utterance even on an RTX 4090. The
float graph runs on the GPU throughout: ~40-90 ms, the same transcripts.

FunAudioLLM/SenseVoiceSmall (pinned revision) is exported by FunASR with the
TorchScript exporter (torch < 2.9: 2.9's default dynamo exporter writes a
graph ONNX Runtime rejects), then converted to float16 by ONNX Runtime's
converter, inputs and outputs kept float32 (``speech`` [B, T, 560],
``speech_lengths``, ``language``, ``textnorm``; ``ctc_logits``,
``encoder_out_lens``: the int8 graph's interface). The CTC tokens of both
graphs are compared on a WAV. The other files come from an existing
sensevoice-small-onnx directory (``--base-dir``: asr/am.mvn, config.yaml,
tokens.json, the FSMN-VAD and CAM++ graphs).

Run with an isolated dependency environment, for example::

    uv run --python 3.11 --with "torch==2.5.1" --with "torchaudio==2.5.1" \\
      --with funasr --with "transformers>=4.40,<4.50" --with "numpy<2" \\
      --with onnx --with onnxruntime --with huggingface_hub \\
      python -m scripts.local.sensevoice_fp16_export \\
        --base-dir workdir/models/sensevoice-small-onnx \\
        --output-dir workdir/models/sensevoice-small-fp16-onnx
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
MODEL_FILE = "model_fp16.onnx"


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-dir", type=Path, required=True,
                        help="a sensevoice-small-onnx directory (asr/, vad/, speaker/)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-model-dir", type=Path,
                        help="a local copy of the HF repo (default: download it)")
    parser.add_argument("--check-wav", type=Path,
                        default=Path(__file__).resolve().parents[1] / "assets" / "tts-input-mon3tr.wav")
    args = parser.parse_args()

    import onnx
    import onnxruntime as ort
    from onnxruntime.transformers.float16 import convert_float_to_float16

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

        model = onnx.shape_inference.infer_shapes(onnx.load(str(fp32)))
        model16 = convert_float_to_float16(model, keep_io_types=True,
                                           force_fp16_initializers=True)
        checked = work / MODEL_FILE
        onnx.save(model16, str(checked))

        inputs = features(args.check_wav, args.base_dir)
        expected = ctc_tokens(
            ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"]), inputs)
        actual = ctc_tokens(
            ort.InferenceSession(str(checked), providers=["CPUExecutionProvider"]), inputs)
        same = sum(a == b for a, b in zip(expected, actual))
        print(f"[sensevoice-fp16] {args.check_wav.name}: {len(expected)} float32 tokens, "
              f"{len(actual)} float16, {same} the same")
        if actual != expected:
            raise SystemExit("the float16 graph's tokens differ from the float32 graph's")

        # Only a checked graph reaches the model directory: all of the base
        # directory but its int8 SenseVoice graph (vad/ keeps its own
        # model_quant.onnx), then the float16 one.
        shutil.copytree(args.base_dir, args.output_dir, dirs_exist_ok=True,
                        ignore=lambda folder, names: ["model_quant.onnx"]
                        if Path(folder).name == "asr" else [])
        out = args.output_dir / "asr" / MODEL_FILE
        shutil.move(str(checked), str(out))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"[sensevoice-fp16] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")

if __name__ == "__main__":
    main()
