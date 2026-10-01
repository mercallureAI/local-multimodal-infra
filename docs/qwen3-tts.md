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
| precision | dynamic INT8 (`MatMulInteger`/`ConvInteger`) | FP32 / INT4 RTN | INT8 weights, FP16 activations (FP32 where needed) |
| attention / KV | per-step KV concat | per-step KV concat, copied every step | `GroupQueryAttention`, KV shared in place |
| code predictor | one run per codebook | 15 runs per frame | 15 steps unrolled in the frame graph |
| sampling | host | host | in graph (Gumbel-max) |
| vocoder | whole utterance | whole utterance (reference prepended) | exact streaming, 4-frame chunks |
| text input | whole | whole | whole or streamed (spoken as it is written) |
| reported speed | none | T4 INT4 RTF 0.8–1.06 | 4090: RTF 0.09, first audio 42 ms |

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

## INT8 weights

`--talker-weights int8` (the default) quantizes the talker's and the code
predictor's matmuls to INT8 weight-only `MatMulNBits` (symmetric RTN, block
128). `scripts/local/qwen3_tts_eval.py` (14 Chinese/English sentences x 2
seeds x x-vector/ICL; SenseVoice CER, ECAPA similarity, DNSMOS):

| talker weights | CER x-vec / ICL | similarity | DNSMOS ovrl / sig | ms/frame (Rust, 4090) | talker.onnx |
| --- | --- | --- | --- | --- | --- |
| FP16 | 0.9% / 2.1% | 0.988 / 0.987 | 3.23 / 3.48 | 7.7 | 1.50 GB |
| INT8 code predictor only | 1.4% / 1.1% | 0.987 / 0.989 | 3.29 / 3.53 | (Python 8.1 vs 9.9) | 1.32 GB |
| **INT8 (default)** | 0.9% / 2.5% | 0.988 / 0.988 | 3.29 / 3.53 | **5.6** | **0.95 GB** |
| INT4 talker + INT8 code predictor | 1.4% / 2.2% | 0.986 / 0.988 | 3.25 / 3.52 | 5.4 | 0.72 GB |
| INT8 talker + INT4 code predictor | 2.8% / 1.5% | 0.986 / 0.988 | 3.28 / 3.54 | (Python 7.8) | 0.86 GB |
| INT4 everywhere | 0.9% / 1.6% | 0.985 / 0.986 | 3.24 / 3.52 | (Python 7.0) | 0.67 GB |

INT8 is within the noise of FP16 (28 samples per cell). An INT4 code predictor
changes greedy codes from the first frame, lowers similarity a little and
produced the only misread ("哈哈" -> "哼哼"): the code predictor stays INT8.
On an RTX 3060 the decode is bandwidth-bound, so INT8 should gain more than
on the 4090 (1.6 instead of 3.3 GB read per frame).

## Streamed text

`synthesize_text_stream` / `RuntimeManager::infer_streaming_text` take the
text as it is written (`TextPiece`s). Speech starts once the first tokens are
there; the rest is fed one token per 80 ms frame, which is how the official
streaming mode feeds text, and a frame whose token has not come yet waits
(the frames before do not depend on it), so the result is the whole-text
result (checked: identical up to the GPU's run-to-run noise, 5e-4). The last
2 tokens of unfinished text are held back: BPE may still merge them with what
comes (with 1 held back, 13 of 19 test sentences would have fed a token the
final tokenization does not have; with 2, 3 of 283 tokens differ).
In-context cloning puts the text under the reference codes, so it starts only
once the text covers them (about 70 tokens for a 7 s reference): no gain there.

Units the model misreads after a number in Chinese text (35ms as 毫米, 5L as
毫升, 20kHz and 1Gbps as noise, 256MB as MG) are written out before the text
is tokenized (`src/units.rs`: 毫秒, 升, 千赫兹, G比特每秒, 兆字节...); it decides by
the text before the number, and of a stream only text up to a word that may
still grow is read, so streamed and whole text read the same. Units it reads
well (km, ℃, %, V) are left, and a bare A or B (3A games, 7B models).

The cascade speaks a reply's first clause while the chat model writes it
(audio mode: a response's first clause while the client streams it;
`tts_stream_text`, on by default, a session may turn it off; only with a TTS
model that takes streamed text, `RuntimeManager::streams_text`); later clauses are ready before the first
has played. The first audio chunk needs about 4 text tokens (plus 2 held
back), so the gain is for first clauses longer than that: on the 4090
(Qwen3-4B INT4 writes ~8 ms per token) a 9-token first clause starts speaking
30–50 ms sooner; on a 3060 (~25 ms per token) it should be 100–150 ms. A
first clause that is a bare "好的，" was already spoken at once.

## Runtime

The adapter owns one thread: ORT's CUDA EP keeps CUDA graphs per thread, and
the runtime's blocking pool would re-capture the decode frame per request.
Load warms both graphs (capture, cuDNN plans). A capture runs in CUDA's
global mode, which fails other threads' stream syncs, so `backend-ort` holds
a process-wide gate (`gpu_gate.rs`) exclusive for a graph's first runs and
shared around every other call that allocates, copies, frees or runs on the
device, in every adapter: binding a host input (ORT copies it to the device
then), creating a binding, a session's load and drop, and the release of
bindings and device tensors (their `Drop` takes it). `StaticIoBinding`
binds the KV cache in place and fixed device inputs/outputs; per frame the
host writes text id, previous codes, attention mask, seen-token mask and
noise (about 0.2 ms) and reads 16 codes.

Request `params`: `reference_text` (ICL; else x-vector only), `language`
(`chinese`, `english`, … or `auto`), `do_sample`, `temperature`,
`subtalker_temperature`, `repetition_penalty`, `max_frames`,
`max_reference_seconds` (15: in-context cloning takes a reference of at most
that long, as its transcript covers all of it; longer ones fail), `seed`. Spec `metadata`: `max_context` (2048
talker positions), `vocoder_chunk_frames` (4), `cuda_graph` (true).
Streaming: `InferenceEvent::AudioChunk` per vocoder chunk
(`RuntimeManager::infer_streaming`); the WAV file is written as well.

## Numbers (RTX 4090, Windows, warm)

| | x-vector | ICL (6.8 s reference, 96-position prompt) |
| --- | --- | --- |
| talker per frame (FP16 / INT8) | 7.7 / 5.6 ms | 7.7 / 5.6 ms |
| first audio | 40–50 ms | 41–50 ms |
| RTF (FP16 / INT8) | 0.12 / 0.09 | 0.12 / 0.09 |

Realtime cascade (`scripts.local.realtime_e2e`, everything on one GPU):
first audio 0.89–0.99 s after the speaker's audio ends (audio mode, where
the test client answers: 1.11–1.16 s), of which 0.6 s is the VAD's
end-of-speech silence; server side 0.3–0.5 s (ASR ~160 ms, first clause
20–150 ms, TTS 50–150 ms). PyTorch (`qwen-tts`, bf16, HF `generate`): RTF
2.2–3.1 on the same GPU.

RTX 3060 12 GB (not measured): an FP16 decode frame reads the talker's 0.9 GB
and the code predictor's 157 MB 15 times, 3.3 GB per frame (INT8: 1.6 GB), so
about 9 (INT8: 4.5) ms of 360 GB/s bandwidth plus launch overhead: expect
roughly 9–12 ms per frame with INT8 (RTF about 0.15) and first audio around
80–100 ms. VRAM: about 1.4 GB weights (INT8) plus KV.

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
# Tests run from the crate directory: absolute paths; ORT 1.30 via ORT_DYLIB_PATH (AGENTS.md).
LOCAL_QWEN3_TTS_MODEL_DIR=<abs>/workdir/models/qwen3-tts-0.6b-onnx LOCAL_QWEN3_TTS_REFERENCE=<abs>/scripts/assets/tts-input-mon3tr.wav \
    ORT_DYLIB_PATH=<onnxruntime 1.30> cargo test --release -p local-adapter-qwen3-tts --features cuda real_model_smoke_if_env_set -- --nocapture
python -m scripts.local.realtime_e2e --audio-dir <r00.wav ...> --model-dir workdir/models --ref-text "<transcript>"
```

## Open

* Measure on the RTX 3060 server (Linux: lower launch overhead than WDDM).
* The cascade with IndexTTS as `tts_model` (whole clauses; no `language` or
  `reference_text`, which only Qwen3-TTS gets) has not been run end to end
  since Qwen3-TTS became the default.
* The vocoder graph keeps shape arithmetic on the CPU, so it cannot replay as
  a CUDA graph; a static-shape export (fixed 4 frames) would fold it away.
* Publish the package (e.g. a `ModaLeap/qwen3-tts-0.6b-onnx` repo) and pin it
  in the spec like the IndexTTS packages.
* Long texts are one prompt (up to `max_context`); the cascade sends clauses.
* Audio mode (client-written replies) still speaks whole clauses: its
  `response.done` accounting needs each clause's text when it is queued.
* The chat model and Qwen3-TTS share a tokenizer: feeding the chat model's
  tokens directly would need no held-back tokens.
