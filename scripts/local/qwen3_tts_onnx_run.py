"""Reference runner for the Qwen3-TTS ONNX package (CUDA, I/O binding).

Mirrors what the Rust adapter does: builds the prompt rows, prefills the
talker frame graph, then decodes one frame per run with the KV cache bound in
place (past and present share one buffer) and only the 16 codes coming back
to the host. Used for parity against the official PyTorch model and for
latency numbers.

    python -m scripts.local.qwen3_tts_onnx_run --package <dir> --text "..." \\
        [--ref-audio ref.wav --ref-text "..."] [--greedy] [--cuda-graph]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

NUM_GROUPS = 16


@dataclass
class Prompt:
    text_ids: np.ndarray  # (S,) int64, -1 = none
    codec_ids: np.ndarray  # (S, 16) int64, -1 = none
    extra: np.ndarray  # (S, H) float16
    trailing: list[int] = field(default_factory=list)
    ref_codes: np.ndarray | None = None  # (T, 16) for the vocoder's left context


def build_prompt(cfg: dict, tokenizer, text: str, language: str, spk: np.ndarray | None,
                 ref_text: str | None, ref_codes: np.ndarray | None) -> Prompt:
    t = cfg["tokens"]
    hidden = cfg["hidden_size"]
    ids = tokenizer.encode(f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n").ids
    role, body = ids[:3], ids[3:-5]
    rows: list[tuple[int, list[int], np.ndarray | None]] = []
    none = [-1] * NUM_GROUPS

    def codec(g0: int) -> list[int]:
        return [g0] + [-1] * (NUM_GROUPS - 1)

    for r in role:
        rows.append((r, none, None))
    lang = cfg["languages"].get(language.lower()) if language and language.lower() != "auto" else None
    if lang is None:
        prefix = [t["codec_nothink"], t["codec_think_bos"], t["codec_think_eos"]]
    else:
        prefix = [t["codec_think"], t["codec_think_bos"], lang, t["codec_think_eos"]]
    codec_input: list = [codec(c) for c in prefix]
    if spk is not None:
        codec_input.append("spk")
    codec_input += [codec(t["codec_pad"]), codec(t["codec_bos"])]
    for j, c in enumerate(codec_input[:-1]):
        text_id = t["tts_bos"] if j == len(codec_input) - 2 else t["tts_pad"]
        if isinstance(c, str):
            rows.append((text_id, none, spk))
        else:
            rows.append((text_id, c, None))
    trailing: list[int]
    if ref_codes is not None and ref_text:
        ref_ids = tokenizer.encode(f"<|im_start|>assistant\n{ref_text}<|im_end|>\n").ids[3:-2]
        text_seq = list(ref_ids) + list(body) + [t["tts_eos"]]
        codec_seq = [codec(t["codec_bos"])] + [list(map(int, row)) for row in ref_codes]
        if len(text_seq) > len(codec_seq):
            trailing = text_seq[len(codec_seq):]
            text_seq = text_seq[: len(codec_seq)]
        else:
            text_seq = text_seq + [t["tts_pad"]] * (len(codec_seq) - len(text_seq))
            trailing = []
        for tid, c in zip(text_seq, codec_seq):
            rows.append((tid, c, None))
    else:
        rows.append((body[0], codec(t["codec_bos"]), None))
        trailing = list(body[1:]) + [t["tts_eos"]]
    s = len(rows)
    extra = np.zeros((s, hidden), np.float16)
    for i, (_, _, e) in enumerate(rows):
        if e is not None:
            extra[i] = e
    return Prompt(
        text_ids=np.array([r[0] for r in rows], np.int64),
        codec_ids=np.array([r[1] for r in rows], np.int64),
        extra=extra,
        trailing=trailing,
        ref_codes=ref_codes,
    )


class Talker:
    """The talker frame graph with a shared, fixed-capacity KV cache."""

    def __init__(self, path: Path, cfg: dict, capacity: int = 2048, cuda_graph: bool = False):
        import onnxruntime as ort

        self.ort = ort
        self.cfg = cfg
        self.capacity = capacity
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        provider = ("CUDAExecutionProvider", {"device_id": 0, "enable_cuda_graph": "1" if cuda_graph else "0"})
        self.sess = ort.InferenceSession(str(path), opts, providers=[provider])
        assert self.sess.get_providers()[0] == "CUDAExecutionProvider", self.sess.get_providers()
        self.cuda_graph = cuda_graph
        layers, kv_heads, head = cfg["talker_layers"], cfg["talker_kv_heads"], cfg["head_dim"]
        shape = (1, kv_heads, capacity, head)
        self.kv = []
        for i in range(layers):
            k = ort.OrtValue.ortvalue_from_shape_and_type(shape, np.float16, "cuda", 0)
            v = ort.OrtValue.ortvalue_from_shape_and_type(shape, np.float16, "cuda", 0)
            self.kv.append((k, v))
        vocab = cfg["talker_vocab_size"]
        # Static decode buffers (bound once: what a CUDA graph replays).
        dev = lambda a: ort.OrtValue.ortvalue_from_numpy(a, "cuda", 0)  # noqa: E731
        self.d_text = dev(np.zeros((1, 1), np.int64))
        self.d_codec = dev(np.zeros((1, 1, NUM_GROUPS), np.int64))
        self.d_extra = dev(np.zeros((1, 1, cfg["hidden_size"]), np.float16))
        self.d_mask = dev(np.zeros((1, capacity), np.int64))
        self.d_seen = dev(np.zeros((1, vocab), np.float32))
        self.d_noise = dev(np.zeros((NUM_GROUPS, cfg["top_k"]), np.float32))
        self.d_sampling = dev(np.zeros(4, np.float32))
        self.d_codes = dev(np.zeros((1, 1, NUM_GROUPS), np.int64))
        self.decode_binding = None
        self.length = 0

    def _bind_kv(self, binding):
        for i, (k, v) in enumerate(self.kv):
            for name, val in ((f"past_key_values.{i}.key", k), (f"past_key_values.{i}.value", v)):
                binding.bind_ortvalue_input(name, val)
            binding.bind_ortvalue_output(f"present.{i}.key", k)
            binding.bind_ortvalue_output(f"present.{i}.value", v)

    def prefill(self, prompt: Prompt, seen, noise, sampling) -> np.ndarray:
        s = len(prompt.text_ids)
        b = self.sess.io_binding()
        b.bind_cpu_input("text_ids", prompt.text_ids[None])
        b.bind_cpu_input("codec_ids", prompt.codec_ids[None])
        b.bind_cpu_input("extra_embeds", prompt.extra[None])
        b.bind_cpu_input("attention_mask", np.ones((1, s), np.int64))
        b.bind_cpu_input("seen_tokens", seen)
        b.bind_cpu_input("noise", noise)
        b.bind_cpu_input("sampling", sampling)
        self._bind_kv(b)
        b.bind_ortvalue_output("codes", self.d_codes)
        ro = self.ort.RunOptions()
        if self.cuda_graph:
            ro.add_run_config_entry("gpu_graph_id", "-1")
        self.sess.run_with_iobinding(b, ro)
        self.length = s
        return self.d_codes.numpy()[0, 0]

    def step(self, text_id: int, codes: np.ndarray, seen, noise, sampling) -> np.ndarray:
        if self.length + 1 > self.capacity:
            raise RuntimeError("KV cache full")
        mask = np.zeros((1, self.capacity), np.int64)
        mask[0, : self.length + 1] = 1
        if self.cuda_graph:
            mask_arg = mask
        else:
            mask_arg = mask[:, : self.length + 1]
        if self.decode_binding is None or not self.cuda_graph:
            b = self.sess.io_binding()
            b.bind_ortvalue_input("text_ids", self.d_text)
            b.bind_ortvalue_input("codec_ids", self.d_codec)
            b.bind_ortvalue_input("extra_embeds", self.d_extra)
            b.bind_ortvalue_input("seen_tokens", self.d_seen)
            b.bind_ortvalue_input("noise", self.d_noise)
            b.bind_ortvalue_input("sampling", self.d_sampling)
            self._bind_kv(b)
            b.bind_ortvalue_output("codes", self.d_codes)
            if self.cuda_graph:
                b.bind_ortvalue_input("attention_mask", self.d_mask)
            self.decode_binding = b
        b = self.decode_binding
        self.d_text.update_inplace(np.array([[text_id]], np.int64))
        self.d_codec.update_inplace(codes.reshape(1, 1, NUM_GROUPS).astype(np.int64))
        self.d_seen.update_inplace(seen)
        self.d_noise.update_inplace(noise)
        self.d_sampling.update_inplace(sampling)
        if self.cuda_graph:
            self.d_mask.update_inplace(mask_arg)
        else:
            b.bind_cpu_input("attention_mask", mask_arg)
        ro = self.ort.RunOptions()
        if self.cuda_graph:
            ro.add_run_config_entry("gpu_graph_id", "1")
        self.sess.run_with_iobinding(b, ro)
        self.length += 1
        return self.d_codes.numpy()[0, 0]


def generate_codes(talker: Talker, prompt: Prompt, cfg: dict, greedy: bool, seed: int = 0,
                   max_frames: int = 1500, params: dict | None = None, on_frame=None) -> np.ndarray:
    p = {"temperature": 0.9, "subtalker_temperature": 0.9, "repetition_penalty": 1.05, "min_new_tokens": 2}
    p.update(params or {})
    rng = np.random.default_rng(seed)
    vocab = cfg["talker_vocab_size"]
    eos = cfg["tokens"]["codec_eos"]
    seen = np.zeros((1, vocab), np.float32)
    frames = []

    def noise():
        if greedy:
            return np.zeros((NUM_GROUPS, cfg["top_k"]), np.float32)
        u = rng.random((NUM_GROUPS, cfg["top_k"]), dtype=np.float32).clip(1e-10, 1 - 1e-7)
        return (-np.log(-np.log(u))).astype(np.float32)

    def sampling(frame):
        return np.array([1 / p["temperature"], 1 / p["subtalker_temperature"], p["repetition_penalty"],
                         0.0 if frame < p["min_new_tokens"] else 1.0], np.float32)

    codes = talker.prefill(prompt, seen, noise(), sampling(0))
    pad = cfg["tokens"]["tts_pad"]
    for frame in range(max_frames):
        if codes[0] == eos:
            break
        frames.append(codes.copy())
        if on_frame:
            on_frame(codes)
        seen[0, codes[0]] = 1.0
        text_id = prompt.trailing[frame] if frame < len(prompt.trailing) else pad
        codes = talker.step(text_id, codes, seen, noise(), sampling(frame + 1))
    return np.array(frames, np.int64).reshape(-1, NUM_GROUPS)


class VoiceEncoder:
    """Reference audio (24 kHz) -> x-vector and codec codes."""

    def __init__(self, package: Path):
        import onnxruntime as ort

        cuda = ("CUDAExecutionProvider", {"device_id": 0, "use_tf32": "0"})
        self.speaker = ort.InferenceSession(str(package / "speaker_encoder.onnx"), providers=[cuda])
        self.codec = ort.InferenceSession(str(package / "codec_encoder.onnx"), providers=[cuda])

    def encode(self, audio24: np.ndarray, max_seconds: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
        spk = self.speaker.run(None, {"audio": audio24[None].astype(np.float32)})[0][0]
        clip = audio24[: int(max_seconds * 24000)]
        frames = -(-len(clip) // 1920)
        buf = np.zeros((1, frames * 1920), np.float32)
        buf[0, : len(clip)] = clip
        codes = self.codec.run(None, {"audio": buf})[0][0].T
        return spk, codes


class Vocoder:
    """Streaming codec decoder: state on the device, shared KV."""

    def __init__(self, package: Path, capacity: int = 4096):
        import onnxruntime as ort

        self.ort = ort
        self.sess = ort.InferenceSession(str(package / "vocoder.onnx"), providers=[("CUDAExecutionProvider", {"device_id": 0})])
        self.capacity = capacity
        self.layers = sum(1 for i in self.sess.get_inputs() if i.name.startswith("past_key."))
        shape = (1, 16, capacity, 64)
        self.kv = [(ort.OrtValue.ortvalue_from_shape_and_type(shape, np.float16, "cuda", 0),
                    ort.OrtValue.ortvalue_from_shape_and_type(shape, np.float16, "cuda", 0)) for _ in range(self.layers)]
        self.reset()

    def reset(self):
        self.pre = np.zeros((1, 512, 2), np.float16)
        self.conv = np.zeros((1, 1024, 0), np.float16)
        self.length = 0

    def run(self, codes: np.ndarray) -> np.ndarray:
        n = codes.shape[0]
        if self.length + n > self.capacity:
            raise RuntimeError("vocoder KV cache full")
        b = self.sess.io_binding()
        b.bind_cpu_input("codes", np.ascontiguousarray(codes.T[None].astype(np.int64)))
        b.bind_cpu_input("pre_ctx", self.pre)
        b.bind_cpu_input("conv_ctx", self.conv)
        b.bind_cpu_input("seqlens_k", np.array([self.length + n - 1], np.int32))
        b.bind_cpu_input("total_sequence_length", np.array(self.capacity, np.int32))
        for i, (k, v) in enumerate(self.kv):
            b.bind_ortvalue_input(f"past_key.{i}", k)
            b.bind_ortvalue_input(f"past_value.{i}", v)
            b.bind_ortvalue_output(f"present_key.{i}", k)
            b.bind_ortvalue_output(f"present_value.{i}", v)
        for name in ("audio", "next_pre_ctx", "next_conv_ctx"):
            b.bind_output(name, "cpu")
        self.sess.run_with_iobinding(b)
        audio, self.pre, self.conv = (o.numpy() for o in b.get_outputs()[-3:])
        self.length += n
        return audio[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--text", action="append", required=True)
    ap.add_argument("--language", default="chinese")
    ap.add_argument("--ref-audio", type=Path)
    ap.add_argument("--ref-text", help="transcript of --ref-audio: ICL mode (else x-vector only)")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--no-cuda-graph", action="store_true")
    ap.add_argument("--first-chunk", type=int, default=3, help="frames in the first vocoder chunk")
    ap.add_argument("--chunk", type=int, default=6, help="frames in later vocoder chunks")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()
    import onnxruntime as ort
    import soundfile as sf
    from tokenizers import Tokenizer

    ort.preload_dlls()
    cfg = json.loads((args.package / "config.json").read_text())
    tok = Tokenizer.from_file(str(args.package / "tokenizer.json"))
    spk = ref_codes = None
    if args.ref_audio:
        import librosa

        y, sr = librosa.load(str(args.ref_audio), sr=24000, mono=True)
        t0 = time.perf_counter()
        spk, ref_codes = VoiceEncoder(args.package).encode(y)
        print(f"voice: {len(y) / 24000:.2f}s ref -> {len(ref_codes)} frames ({(time.perf_counter() - t0) * 1000:.0f}ms incl. load)")
    talker = Talker(args.package / "talker.onnx", cfg, cuda_graph=not args.no_cuda_graph)
    vocoder = Vocoder(args.package)
    icl = ref_codes is not None and bool(args.ref_text)
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    for ti, text in enumerate(args.text):
        prompt = build_prompt(cfg, tok, text, args.language, None if spk is None else spk.astype(np.float16),
                              args.ref_text, ref_codes if icl else None)
        for rep in range(args.repeat):
            vocoder.reset()
            t0 = time.perf_counter()
            if icl:
                vocoder.run(ref_codes)  # left context: the reference voice
            t_prime = time.perf_counter() - t0
            pending, audio, marks = [], [], {}
            voc_time = [0.0]

            def flush():
                ts = time.perf_counter()
                audio.append(vocoder.run(np.array(pending)))
                voc_time[0] += time.perf_counter() - ts
                pending.clear()
                if "first_audio" not in marks:
                    marks["first_audio"] = time.perf_counter() - t0

            def on_frame(codes):
                pending.append(codes)
                want = args.first_chunk if not audio else args.chunk
                if len(pending) >= want:
                    flush()

            frames = generate_codes(talker, prompt, cfg, args.greedy, seed=rep, on_frame=on_frame)
            if pending:
                flush()
            total = time.perf_counter() - t0
            wav = np.concatenate(audio) if audio else np.zeros(0, np.float32)
            dur = len(wav) / 24000
            print(f"[{ti}.{rep}] {'icl' if icl else 'xvec'} prompt={len(prompt.text_ids)} frames={len(frames)} audio={dur:.2f}s "
                  f"first_audio={marks.get('first_audio', 0) * 1000:.0f}ms (prime {t_prime * 1000:.0f}ms) "
                  f"total={total * 1000:.0f}ms vocoder={voc_time[0] * 1000:.0f}ms RTF={total / max(dur, 1e-6):.3f}")
            if args.out_dir and rep == 0:
                sf.write(str(args.out_dir / f"onnx_{'icl' if icl else 'xvec'}_{ti}.wav"), wav, 24000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
