# Speech synthesis

| Model | Get it | Runs on | For |
| --- | --- | --- | --- |
| `indextts-1.5-onnx` | download | CPU or CUDA | voice cloning from a reference WAV |
| `indextts-2.5-onnx` | download (~2.8 GB) | CUDA recommended | cloning plus emotion control |
| `qwen3-tts-0.6b-onnx` | local export | CUDA recommended | streaming, lowest latency; the realtime voice's TTS |

All three clone the voice of a reference recording (`reference_audio`).

## Calling it

- The task `tts.synthesize` (`params.text`, optional `reference_audio`
  file; MCP and JSON-RPC), or JSON-RPC `tts_synthesize` (`text`,
  `reference_audio` file ref): a WAV with a signed download URL.
- `POST /v1/audio/speech`: JSON `{"model", "input" | "text", "reference_path"}`;
  every other field is passed to the model as a parameter (below). Returns
  `{"audio": <file ref>}`.

## `indextts-1.5-onnx`

[IndexTTS](https://github.com/index-tts/index-tts) 1.5 as six FP32 graphs (A–F)
from [ModaLeap/indextts-1.5-onnx](https://huggingface.co/ModaLeap/indextts-1.5-onnx)
at `3f1a422cd97a0b7dbb9b6ad4698dc0fde66796d1`
([spec](../../configs/providers/tts/index-tts.yaml)). Long text is split at
punctuation into segments of 120 model tokens, joined with 200 ms of silence.

On CPU, `LOCAL_INDEXTTS_ORT_INTRA_THREADS` (default `min(CPUs, 8)`) and
`LOCAL_INDEXTTS_ORT_INTER_THREADS` (default 1) set the ORT thread pools.

## `indextts-2.5-onnx`

IndexTTS 2.5, FP16, from
[ModaLeap/indextts-2.5-onnx](https://huggingface.co/ModaLeap/indextts-2.5-onnx)
at `fd246cb6c2cf046113cd3400565edf681ac1b68b`, 22.05 kHz
([spec](../../configs/providers/tts/index-tts2.yaml)). Needs ~6 GB of GPU
memory; CPU is a slow fallback.

Parameters:

| Key | Meaning |
| --- | --- |
| `emotion_vector` | 8 weights in the order happy, angry, sad, afraid, disgusted, melancholic, surprised, calm, or an object keyed by those names |
| `emotion_alpha` | emotion strength (default 1.0) |
| `language` | `zh`, `en`, `ja`, ...; guessed from the text when absent |
| `speed` / `duration_factor` | OpenAI-style speed (2.0 = twice as fast) or its inverse |
| `temperature`, `top_k`, `top_p`, `seed` | sampling |

## Mandarin frontend (both IndexTTS models)

Both specs also download [ModaLeap/zh-tts-frontend](https://huggingface.co/ModaLeap/zh-tts-frontend)
(`ba6b85aeb17ebc58d2d3d73121096f9495ee710e`, ~177 MB, licenses in its
`NOTICE`) into `zh-tts-frontend/` of the model directory: WeText text
normalization and g2pW polyphone readings, so numbers, dates and characters
with several readings are spoken right. Fix a word's reading in
`<frontend dir>/user_phrases.tsv` (`word<TAB>pin1 yin1`). Turn it off with
`LOCAL_ZH_TTS_FRONTEND=off`; `LOCAL_ZH_TTS_FRONTEND_DIR` points to another
copy. How it works: [implementation notes](../implementation-notes.md#mandarin-text-frontend-zh-tts-frontend).

## `qwen3-tts-0.6b-onnx`

[Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
(Apache-2.0, 10 languages) as our own export: INT8 weights, the decode frame
replayed as a CUDA graph, and a vocoder that streams in 80 ms frames. On an
RTX 4090 a frame takes ~5.6 ms and a sentence's first audio starts ~50 ms
after it is asked for; it can also speak text while it is still being written
([spec](../../configs/providers/tts/qwen3-tts.yaml)).

Export (two Python environments: one with torch + `qwen-tts`, one with
`onnxruntime-genai`; details in [qwen3-tts.md](../qwen3-tts.md#export-and-check)):

```bash
hf download Qwen/Qwen3-TTS-12Hz-0.6B-Base --revision 5d83992436eae1d760afd27aff78a71d676296fc --local-dir <src>
python -m scripts.local.qwen3_tts_export --source-model-dir <src> \
  --output-dir workdir/models/qwen3-tts-0.6b-onnx --builder-python <python with onnxruntime-genai>
```

Parameters: `language` (`chinese`, `english`, ...; auto when absent),
`reference_text` (what the reference audio says: clones the voice more
closely), `temperature`, `repetition_penalty`, `seed`, `max_frames`,
`max_reference_seconds` (default 15). Spec metadata: `max_context` (KV
positions, 2048), `vocoder_chunk_frames` (4), `cuda_graph` (`true`).

Design, INT8 quality and measurements: [qwen3-tts.md](../qwen3-tts.md).
