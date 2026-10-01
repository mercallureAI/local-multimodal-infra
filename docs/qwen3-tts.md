# Qwen3-TTS (`qwen3-tts-0.6b-onnx`)

[Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
(Apache-2.0, 10 languages, 3 s voice cloning) as a streaming TTS for the
realtime voice: a clause starts playing about 50 ms after the cascade asks
for it. Adapter `qwen3_tts` (`crates/adapters/tts/qwen3-tts`), spec
`configs/providers/tts/qwen3-tts.yaml`, export
`scripts/local/qwen3_tts_export.py`.

## Why our own export

| | [sivasub987/…-ONNX-INT8](https://huggingface.co/sivasub987/Qwen3-TTS-0.6B-ONNX-INT8) | [wavekat/…-Base-ONNX](https://huggingface.co/wavekat/Qwen3-TTS-0.6B-Base-ONNX) | this export |
| --- | --- | --- | --- |
| precision | dynamic INT8 (`MatMulInteger`/`ConvInteger`) | FP32 / INT4 RTN | FP16 (FP32 where needed) |
| attention / KV | per-step KV concat | per-step KV concat, copied every step | `GroupQueryAttention`, KV shared in place |
| code predictor | one run per codebook | 15 runs per frame | 15 steps unrolled in the frame graph |
| sampling | host | host | in graph (Gumbel-max) |
| vocoder | whole utterance | whole utterance (reference prepended) | exact streaming, 4-frame chunks |
| reported speed | none | T4 INT4 RTF 0.8–1.06 | 4090: RTF 0.12, first audio 50 ms |

Neither streams audio, and both spend most of a frame on per-run overhead.

## Graphs

* `talker.onnx`: one 80 ms frame. Text ids, codec codes and raw embeddings
  (x-vector) are embedded in-graph (`text_projection` precomputed per token);
  the 28-layer talker is the onnxruntime-genai builder's (FP16, GQA, q/k norm
  and RoPE fused, past/present-shared KV); codebook 0 is sampled in-graph
  (repetition penalty, control tokens suppressed, EOS banned for the first 2
  frames, temperature, top-k 50, Gumbel-max with host noise), then the
  5-layer code predictor runs 15 steps unrolled (GQA, KV concatenated in-op)
  with the same sampler. Out: the frame's 16 codes. Prefill and decode use
  the same graph; decode has static shapes and replays as a CUDA graph.
  The code predictor's layer 2 reaches 1.8e5 inside its MLP and 6.1e4 in the
  residual stream (FP32 calibration), so its residual stream is FP32 and that
  down projection is scaled by 1/64; the talker is plain FP16.
* `vocoder.onnx`: the 12 Hz codec decoder, streaming exactly: the pre-conv's
  2-frame context, the 8-layer sliding-window (72) pre-transformer as GQA with
  a shared KV cache (`local_window_size` 71), and 4 frames of left context for
  the upsampling stack, zeroed before every conv at a stream's start
  (`context_valid` 0), which is what the convs' own zero padding does. Every
  run takes the same 4 frames: ORT re-plans every cuDNN convolution (about
  40 ms) whenever a shape changes. In ICL mode the stream is primed with the
  reference codes, as upstream decodes reference and generated codes together;
  the primed KV is kept per voice.
* `speaker_encoder.onnx` (log-mel + ECAPA, FP32) and `codec_encoder.onnx`
  (Mimi encoder, FP32, input padded to whole 1920-sample frames): the
  reference voice, once per voice (TF32 off: it flips residual-VQ codes).

Checked against the official package (FP32 PyTorch): greedy codes identical
for the first 18 (x-vector) / 37 (ICL) frames, then FP16 drift; streamed
vocoder 47–50 dB SNR against a whole-utterance FP32 decode (the FP16 floor is
52 dB); speaker encoder cosine 1.0; codec encoder 99% of codes; SenseVoice
transcribes every output exactly; x-vector similarity to the reference 0.99
(PyTorch bf16: 0.985–0.988).

## Runtime

The adapter owns one thread: ORT's CUDA EP keeps CUDA graphs per thread, and
the runtime's blocking pool would re-capture the decode frame per request.
Load warms both graphs (capture, cuDNN plans). A capture runs in CUDA's
global mode, which fails other threads' stream syncs, so `backend-ort` holds
a process-wide gate shared around device calls and exclusive for a graph's
first runs (`gpu_gate.rs`). `StaticIoBinding` binds the KV cache in place and
fixed device inputs/outputs; per frame the host writes text id, previous
codes, attention mask, seen-token mask and noise (about 0.2 ms) and reads 16
codes.

Request `params`: `reference_text` (ICL; else x-vector only), `language`
(`chinese`, `english`, … or `auto`), `do_sample`, `temperature`,
`subtalker_temperature`, `repetition_penalty`, `max_frames`,
`max_reference_seconds` (10), `seed`. Spec `metadata`: `max_context` (2048
talker positions), `vocoder_chunk_frames` (4), `cuda_graph` (true).
Streaming: `InferenceEvent::AudioChunk` per vocoder chunk
(`RuntimeManager::infer_streaming`); the WAV file is written as well.

## Numbers (RTX 4090, Windows, warm)

| | x-vector | ICL (6.8 s reference, 96-position prompt) |
| --- | --- | --- |
| talker per frame | 7–10 ms | 7–10 ms |
| first audio | 50–65 ms | 47–65 ms |
| RTF | 0.12–0.16 | 0.12–0.16 |

Realtime cascade (`scripts.local.realtime_e2e`, everything on one GPU):
first audio 0.93–1.05 s after the speaker's audio ends, of which 0.6 s is the
VAD's end-of-speech silence; server side 0.3–0.5 s (ASR ~160 ms, first clause
20–150 ms, TTS 50–150 ms). PyTorch (`qwen-tts`, bf16, HF `generate`): RTF
2.2–3.1 on the same GPU.

RTX 3060 12 GB (not measured): a decode frame reads the talker's 0.9 GB and
the code predictor's 157 MB 15 times, 3.3 GB per frame, so about 9 ms of
360 GB/s bandwidth plus launch overhead: expect 13–16 ms per frame (RTF
about 0.2) and first audio around 100 ms. VRAM: about 2 GB weights plus KV.

## Export and check

```bash
hf download Qwen/Qwen3-TTS-12Hz-0.6B-Base --revision 5d83992436eae1d760afd27aff78a71d676296fc --local-dir <src>
# Python 3.12: torch (CUDA), qwen-tts, onnx, onnxruntime-gpu, librosa, soundfile;
# the builder needs a newer transformers than qwen-tts pins: a second env with
# onnxruntime-genai + transformers + torch (CPU).
python -m scripts.local.qwen3_tts_export --source-model-dir <src> \
    --output-dir workdir/models/qwen3-tts-0.6b-onnx --builder-python <builder env python>
# Python reference pipeline (timings, WAVs):
python -m scripts.local.qwen3_tts_onnx_run --package workdir/models/qwen3-tts-0.6b-onnx \
    --ref-audio scripts/assets/tts-input-mon3tr.wav --ref-text "<transcript>" --text "你好。" --repeat 3
# Rust:
cargo run --release -p local-adapter-qwen3-tts --features cuda --example synthesize -- \
    workdir/models/qwen3-tts-0.6b-onnx scripts/assets/tts-input-mon3tr.wav --ref-text "<transcript>" --repeat 3 "你好。"
LOCAL_QWEN3_TTS_MODEL_DIR=workdir/models/qwen3-tts-0.6b-onnx LOCAL_QWEN3_TTS_REFERENCE=scripts/assets/tts-input-mon3tr.wav \
    cargo test --release -p local-adapter-qwen3-tts --features cuda real_model_smoke_if_env_set -- --nocapture
python -m scripts.local.realtime_e2e --audio-dir <r00.wav ...> --model-dir workdir/models --ref-text "<transcript>"
```

## Open

* Measure on the RTX 3060 server (Linux: lower launch overhead than WDDM).
* The code predictor dominates bandwidth (2.4 GB per frame): INT8 weight-only
  `MatMulNBits` for it would halve that; needs an A/B on quality.
* The vocoder graph keeps shape arithmetic on the CPU, so it cannot replay as
  a CUDA graph; a static-shape export (fixed 4 frames) would fold it away.
* Publish the package (e.g. a `ModaLeap/qwen3-tts-0.6b-onnx` repo) and pin it
  in the spec like the IndexTTS packages.
* Long texts are one prompt (up to `max_context`); the cascade sends clauses.
