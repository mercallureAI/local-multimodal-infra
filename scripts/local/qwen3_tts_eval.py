"""A/B quality and speed of Qwen3-TTS ONNX packages (e.g. FP16 vs INT8).

For every package, mode (x-vector / in-context) and seed, synthesizes a fixed
set of Chinese and English sentences with the streaming pipeline of
``qwen3_tts_onnx_run`` and scores the audio:

* CER of a SenseVoice transcription (the Rust ``transcribe`` example);
* speaker similarity: cosine of ECAPA x-vectors, output vs reference;
* DNSMOS (``speechmos``): predicted overall / signal quality, 1-5;
* greedy decoding: frames identical to the first package's codes;
* talker milliseconds per frame (CUDA graph) and real-time factor.

    python -m scripts.local.qwen3_tts_eval --package fp16=<dir> --package int8=<dir> \\
        --ref-audio ref.wav --ref-text "..." --transcribe <transcribe.exe> \\
        --sensevoice <sensevoice model dir> --out <dir>
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from scripts.local.qwen3_tts_onnx_run import Talker, Vocoder, VoiceEncoder, build_prompt, generate_codes

SENTENCES = [
    ("chinese", "你好，我是你的语音助手，今天有什么可以帮你的吗？"),
    ("chinese", "其实我真的有发现，我是一个特别善于观察别人情绪的人。"),
    ("chinese", "明天上午有小雨，出门记得带伞，气温大概在十五度左右。"),
    ("chinese", "好的，我已经帮你把会议改到下午三点，并通知了所有参会的人。"),
    ("chinese", "这个问题有点复杂，我们可以先从最简单的情况开始考虑。"),
    ("chinese", "放心吧，交给我就行，保证不会出任何差错。"),
    ("chinese", "抱歉，我刚才没有听清楚，你能再说一遍吗？"),
    ("chinese", "春天来了，公园里的花都开了，到处都是踏青的游客。"),
    ("chinese", "根据你的描述，我建议先重启一下路由器，再检查网线有没有松动。"),
    ("chinese", "哈哈，这个笑话真有意思，我差点笑出声来。"),
    ("english", "Hello! The weather today is lovely, isn't it? Let's go for a walk."),
    ("english", "I have moved your meeting to three in the afternoon and told everyone."),
    ("english", "Sorry, I didn't catch that. Could you say it again, please?"),
    ("english", "Thanks for waiting. Your order is on its way and should arrive tomorrow."),
]


def normalize(text: str, language: str) -> str:
    text = text.lower()
    if language == "english":
        return re.sub(r"[^a-z]", "", text)
    return re.sub(r"[^一-鿿]", "", text)


def cer(ref: str, hyp: str) -> float:
    import jiwer

    if not ref:
        return 0.0
    return jiwer.cer(ref, hyp) if hyp else 1.0


def synthesize(talker, vocoder, cfg, tok, text, language, spk, ref_text, ref_codes, seed, greedy):
    prompt = build_prompt(cfg, tok, text, language, spk, ref_text, ref_codes)
    if ref_codes is not None and ref_text:
        vocoder.prime(ref_codes)
    else:
        vocoder.reset()
    pending, audio = [], []
    started = time.perf_counter()

    def on_frame(codes):
        pending.append(codes)
        if len(pending) == vocoder.chunk:
            audio.append(vocoder.run(np.array(pending)))
            pending.clear()

    codes = generate_codes(talker, prompt, cfg, greedy, seed=seed, on_frame=on_frame)
    if pending:
        audio.append(vocoder.run(np.array(pending)))
    elapsed = time.perf_counter() - started
    wav = np.concatenate(audio) if audio else np.zeros(0, np.float32)
    return wav, codes, elapsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", action="append", required=True, help="name=dir; the first is the baseline")
    ap.add_argument("--ref-audio", type=Path, required=True)
    ap.add_argument("--ref-text", required=True)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--transcribe", type=Path, required=True, help="the SenseVoice transcribe example binary")
    ap.add_argument("--sensevoice", type=Path, required=True, help="sensevoice-small-onnx model dir")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import librosa
    import onnxruntime as ort
    import soundfile as sf
    from speechmos import dnsmos
    from tokenizers import Tokenizer

    ort.preload_dlls()
    args.out.mkdir(parents=True, exist_ok=True)
    packages = [tuple(item.split("=", 1)) for item in args.package]
    ref24, _ = librosa.load(str(args.ref_audio), sr=24000, mono=True)
    rows = []  # one per synthesized file
    greedy_codes: dict[str, list[np.ndarray]] = {}
    for name, directory in packages:
        package = Path(directory)
        cfg = json.loads((package / "config.json").read_text())
        tok = Tokenizer.from_file(str(package / "tokenizer.json"))
        encoder = VoiceEncoder(package)
        spk, ref_codes = encoder.encode(ref24)
        spk16 = spk.astype(np.float16)
        talker = Talker(package / "talker.onnx", cfg, cuda_graph=True)
        vocoder = Vocoder(package)
        # Warm-up (capture, cuDNN plans).
        synthesize(talker, vocoder, cfg, tok, "你好。", "chinese", spk16, None, None, 0, False)
        greedy_codes[name] = []
        for index, (language, text) in enumerate(SENTENCES):
            _, codes, _ = synthesize(talker, vocoder, cfg, tok, text, language, spk16, None, None, 0, True)
            greedy_codes[name].append(codes)
            for mode in ("xvec", "icl"):
                for seed in range(args.seeds):
                    ref_text = args.ref_text if mode == "icl" else None
                    wav, codes, elapsed = synthesize(
                        talker, vocoder, cfg, tok, text, language, spk16, ref_text,
                        ref_codes if mode == "icl" else None, seed, False)
                    path = args.out / f"{name}_{mode}_{index:02d}_{seed}.wav"
                    sf.write(str(path), wav, 24000)
                    rows.append({"package": name, "mode": mode, "index": index, "seed": seed, "language": language,
                                 "text": text, "path": str(path), "frames": len(codes),
                                 "seconds": len(wav) / 24000, "elapsed": elapsed})
            print(f"[{name}] {index + 1}/{len(SENTENCES)}", flush=True)
        del talker, vocoder, encoder
        gc.collect()

    # Transcribe everything in one run.
    paths = [row["path"] for row in rows]
    result = subprocess.run([str(args.transcribe), str(args.sensevoice), *paths], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", check=True)
    heard = {}
    for line in result.stdout.splitlines():
        if "\t" in line:
            file, text = line.split("\t", 1)
            heard[str(Path(file))] = text
    speaker = ort.InferenceSession(str(Path(packages[0][1]) / "speaker_encoder.onnx"),
                                   providers=[("CUDAExecutionProvider", {"use_tf32": "0"})])

    def embed(audio):
        vector = speaker.run(None, {"audio": audio[None].astype(np.float32)})[0][0]
        return vector / np.linalg.norm(vector)

    ref_vector = embed(ref24)
    for row in rows:
        audio, _ = librosa.load(row["path"], sr=24000, mono=True)
        row["heard"] = heard.get(str(Path(row["path"])), "")
        row["cer"] = cer(normalize(row["text"], row["language"]), normalize(row["heard"], row["language"]))
        row["sim"] = float(embed(audio) @ ref_vector) if len(audio) > 2400 else 0.0
        mos = dnsmos.run(librosa.resample(audio, orig_sr=24000, target_sr=16000), sr=16000)
        row["ovrl"], row["sig"] = float(mos["ovrl_mos"]), float(mos["sig_mos"])

    base = packages[0][0]
    print(f"\n{'package':10s} {'mode':5s} {'CER%':>6s} {'SIM':>6s} {'OVRL':>5s} {'SIG':>5s} {'ms/fr':>6s} {'RTF':>6s} {'len':>5s} {'greedy=':>8s}")
    summary = []
    for name, _ in packages:
        matches = []
        for mine, ref in zip(greedy_codes[name], greedy_codes[base]):
            n = min(len(mine), len(ref))
            eq = (mine[:n] == ref[:n]).all(axis=1) if n else np.array([])
            first = int(np.argmin(eq)) if n and not eq.all() else n
            matches.append(first / max(len(ref), 1))
        for mode in ("xvec", "icl"):
            sel = [r for r in rows if r["package"] == name and r["mode"] == mode]
            frames = sum(r["frames"] for r in sel)
            line = {
                "package": name, "mode": mode,
                "cer": 100 * np.mean([r["cer"] for r in sel]),
                "sim": np.mean([r["sim"] for r in sel]),
                "ovrl": np.mean([r["ovrl"] for r in sel]),
                "sig": np.mean([r["sig"] for r in sel]),
                "ms_per_frame": 1000 * sum(r["elapsed"] for r in sel) / max(frames, 1),
                "rtf": sum(r["elapsed"] for r in sel) / max(sum(r["seconds"] for r in sel), 1e-6),
                "seconds": sum(r["seconds"] for r in sel),
                "greedy_prefix": float(np.mean(matches)),
            }
            summary.append(line)
            print(f"{name:10s} {mode:5s} {line['cer']:6.2f} {line['sim']:6.3f} {line['ovrl']:5.2f} {line['sig']:5.2f} "
                  f"{line['ms_per_frame']:6.2f} {line['rtf']:6.3f} {line['seconds']:5.0f} {line['greedy_prefix']:8.2f}")
    (args.out / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    bad = [r for r in rows if r["cer"] > 0.2]
    for r in bad:
        print(f"high CER {r['cer']:.2f} {Path(r['path']).name}: {r['text']} -> {r['heard']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
