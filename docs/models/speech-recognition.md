# Speech recognition

| Model | Get it | Runs on |
| --- | --- | --- |
| `sensevoice-small-onnx` | download | CPU or CUDA |
| `sensevoice-small-int8-gpu-onnx` | download ([ModaLeap/sensevoice-small-onnx](https://huggingface.co/ModaLeap/sensevoice-small-onnx)) | CUDA (CPU works, slowly) |
| `fun-asr-nano-onnx` | download ([ModaLeap/fun-asr-nano-onnx](https://huggingface.co/ModaLeap/fun-asr-nano-onnx)) | CUDA only |

All are the same pipeline: FSMN-VAD cuts speech segments (at most 20 s,
cut at the weakest speech), a recognizer transcribes them, and CAM++
speaker embeddings label who spoke. SenseVoiceSmall adds language and
emotion tags; Fun-ASR-Nano is more accurate, slower, and takes hotwords.
Audio: WAV, FLAC, Ogg Vorbis, MP3, AAC (MP4/M4A, fragmented `.m4s`) and ALAC;
long recordings are streamed, not loaded whole.

## `sensevoice-small-onnx`

SenseVoiceSmall as a dynamic int8 graph, with FSMN-VAD and CAM++. The default
ASR model ([spec](../../configs/providers/asr/sensevoice-asr.yaml)).

```bash
# admin download (see models/README.md)
{"method":"download_model","params":{"id":"sensevoice-small-onnx"}}
```

| Part | Source | Revision | License |
| --- | --- | --- | --- |
| SenseVoiceSmall ONNX | [haixuantao/SenseVoiceSmall-onnx](https://huggingface.co/haixuantao/SenseVoiceSmall-onnx) | `c4c8747214bed7ebbf2557e0412c19efa540023c` | Apache-2.0 |
| FSMN-VAD ONNX | [funasr/fsmn-vad-onnx](https://huggingface.co/funasr/fsmn-vad-onnx) | `f6e9fbb4cefa7397216c763f21307993f147f585` | |
| FSMN-VAD config | [MoYoYoTech/Translator](https://huggingface.co/MoYoYoTech/Translator) | `58fbad4088820ed1253955c8faf1444cd0b2dc69` | |
| CAM++ speaker | [welcomyou/campplus-3dspeaker-200k-onnx](https://huggingface.co/welcomyou/campplus-3dspeaker-200k-onnx) | `6265ff7af2a104d745b4389026ed9815c6c1c6ff` | |

On CUDA its 281 `DynamicQuantizeLinear` nodes still run on the CPU, so each
layer goes back and forth: about 200–350 ms for one sentence. Use the GPU int8
model below for low latency.

## `sensevoice-small-int8-gpu-onnx`

The same SenseVoiceSmall with only the weights of its 281 linear layers in
int8 (`MatMulNBits`, float32 activations), so it stays on the GPU; transcripts
agree with the float32 graph 99.8%. One run is ~2400 small kernels (~23 ms
whatever the length), so each input-length bucket
(`asr_cuda_graph_frames`, default 64/128/256/352 LFR frames = 3.8/7.7/15.4/21.1 s)
gets a fixed-length session replayed as a CUDA graph, all sharing one copy of
the weights (~0.6 GB for the four). 1.5 s of audio: ~9 ms of SenseVoice, ~16 ms
in all; 6 s: ~40 ms. The realtime voice cascade's default ASR
([spec](../../configs/providers/asr/sensevoice-asr-int8-gpu.yaml)).

Download it with `download_model` (`sensevoice-small-int8-gpu-onnx`): the
package [ModaLeap/sensevoice-small-onnx](https://huggingface.co/ModaLeap/sensevoice-small-onnx)
has the whole directory (FunASR model license for SenseVoice; FSMN-VAD and
CAM++ Apache-2.0). Applications that link the adapter can pass the spoken
language (`recognize_samples_with_language`: zh, en, yue, ja, ko) as
SenseVoice's language prompt instead of detecting it.

It was built from `FunAudioLLM/SenseVoiceSmall` at
`3847d57b6bdf2dd8875cb1508d2af43d80a16bf7` (the rest of the directory is
`sensevoice-small-onnx`'s):

```bash
python -m scripts.local.sensevoice_cuda_export export \
  --base-dir workdir/models/sensevoice-small-onnx \
  --output-dir workdir/models/sensevoice-small-int8-gpu-onnx
# or from an earlier float16 export (needs only onnx + onnxruntime):
python -m scripts.local.sensevoice_cuda_export from-fp16 \
  --model-dir workdir/models/sensevoice-small-fp16-onnx
```

It writes `asr/model_gpu_int8.onnx`, its weights (`.onnx.data`, ~270 MB) and
their byte ranges (`model_gpu_int8.shared.json`). Requests for the retired
`sensevoice-small-fp16-onnx` (its float16 kernels overflowed into NaN on loud
speech) get this model. `auto_select: false`: used only where named.

## `fun-asr-nano-onnx`

[FunAudioLLM/Fun-ASR-Nano-2512](https://modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512)
(Apache-2.0, 0.8B): SenseVoiceSmall's encoder (same FBANK/LFR frontend, no
CMVN) feeding a transformer adaptor whose output takes the place of
placeholder tokens in a Qwen3-0.6B prompt, which writes the transcript
(greedy, upstream's prompt `语音转写：`). A CTC decoder on the same encoder
gives a second, plain transcript ([spec](../../configs/providers/asr/fun-asr-nano.yaml)).

FLEURS dev, with each model's full pipeline on an RTX 4090 (SenseVoice is
`sensevoice-small-int8-gpu-onnx`; punctuation and case ignored):

| | Fun-ASR-Nano | SenseVoiceSmall |
| --- | --- | --- |
| Mandarin CER, 328 utterances without numbers | **4.29%** | 5.27% |
| Mandarin CER, all 409 | 6.33% | 6.26% |
| English WER, all 394 | **7.18%** | 9.76% |
| English WER, 321 without numbers | **5.52%** | 7.80% |
| 1.5 s / 3 s / 6 s of speech | ~30-45 / ~55 / ~85 ms | ~16 / ~22 / ~35 ms |
| ~10 s utterances (mean) | ~105 ms | ~48 ms |
| GPU memory, process (CUDA context included) | ~2.5 GB | ~1.7 GB |

Fun-ASR-Nano often writes numbers in Chinese numerals (二十一比二十,
一七六七年) where FLEURS has digits, hence the "all" Mandarin row. On 20
Mandarin sentences with English terms (帮我打开 YouTube…, synthesized by
Qwen3-TTS) Fun-ASR-Nano kept 39 of 41 English terms, SenseVoice 33 (it drops
or misspells them: issue, Zoom → zom, Claude → cloud, Lakers → lars); neither
switched a sentence to English. `hotwords: ["Tesla"]` turned Fun-ASR-Nano's
特斯拉 into Tesla. A 1.5 s
segment costs ~8 ms of audio graph (CUDA graph), ~10 ms of LLM prefill and
~2 ms per generated token (the decode step replays as a CUDA graph).

Download it with `download_model` (`fun-asr-nano-onnx`): the package
[ModaLeap/fun-asr-nano-onnx](https://huggingface.co/ModaLeap/fun-asr-nano-onnx)
has the whole directory (Apache-2.0). It was built from the ModelScope
snapshot (its `model.pt`, SHA-256 `81fec861...`, has the trained CTC decoder
the Hugging Face copy lacks; `vad/` and `speaker/` are
`sensevoice-small-onnx`'s):

```bash
python -m scripts.local.fun_asr_nano_export export \
  --source-model-dir workdir/models/Fun-ASR-Nano-2512 \
  --base-dir workdir/models/sensevoice-small-onnx \
  --output-dir workdir/models/fun-asr-nano-onnx \
  --builder-python <python with onnxruntime-genai>
```

It writes `asr/audio_encoder.onnx` (encoder, adaptor and CTC decoder: int8
weights, float32 activations, weights shared by the CUDA graph buckets as
for SenseVoice), `asr/llm.onnx` (Qwen3's 28 layers from the
onnxruntime-genai builder: FP16 with int8 weights, GroupQueryAttention over a
fixed KV cache; the residual stream runs scaled by 1/8, since Qwen3's
layer 2 writes ~1.25e5 into it, past FP16's range), `asr/embed_tokens.f16`
(the token embeddings, kept in host memory), the tokenizers and a manifest.
`parity` compares a package with the PyTorch model token by token
(`--provider cuda`: the builder's attention kernel fuses Qwen3's q/k norms,
which only CUDA implements, so the LLM graph does not run on the CPU).

The CTC branch guards the LLM (`ctc_guard`, default true): a segment in
which it hears no speech is left out (no text, no timeline entry, no
speaker time; an LLM makes up text for noise), and an LLM output that runs
into the token limit (16 + 12 per second of speech: a repetition loop) is
replaced by the CTC text. No token timestamps or language/emotion tags.

Per-request parameters besides those below: `hotwords` (a list, or a
comma-separated string: names and terms to favour, upstream's context
prompt; no `<|`/`|>`), `language` (中文, 英文, 日文, or zh, en, ja; any when
absent) and `itn: false` (no inverse text normalization). A request whose
hotwords leave too little of the LLM context (`max_context`) for the
longest speech segment and its output is refused before any audio is
decoded.

## Options

Spec `metadata` (defaults) and per-request parameters of the same name:

| Key | Default | Meaning |
| --- | --- | --- |
| `timestamps` (`timestamps_default`) | `true` | `timestamped_text` and `segments` with times |
| `timestamp_granularity_sec` | `10` | length of the timeline's segments |
| `token_timestamps` (`token_timestamps_default`) | `false` | per-token times |
| `speaker_diarization` (`speaker_diarization_default`) | `true` | `segments[].speaker` and `speakers[]` |
| `vad_max_segment_seconds` | `20` | longest speech segment |
| `asr_model_file` | `model_quant.onnx` | SenseVoice: the graph in `asr/` |
| `asr_cuda_graph_frames` | (unset) | CUDA graph buckets, in LFR frames |
| `max_context` | `1024` | Fun-ASR-Nano: LLM positions of the KV cache (112 KB each) |
| `ctc_guard` | `true` | Fun-ASR-Nano: the CTC branch drops silent segments and replaces runaway output |

## Calling it

- The task `asr.transcribe` with an `audio` file (MCP and JSON-RPC), or
  JSON-RPC `asr_transcribe` (`audio` file ref): full output (`text`,
  `timestamped_text`, `segments`, `speakers`).
- `POST /v1/audio/transcriptions` (JSON `{"model", "file": {"path"}}`, a path
  the worker can read): `{"text"}` only.

Long-audio timing and memory:
`cargo run --release -p local-adapter-sensevoice-asr --features cuda --example long_audio -- <model dir> <audio>`.
Fun-ASR-Nano next to SenseVoice, file by file (text and time; `@list` reads
paths from a file):
`FUN_ASR_NANO_COMPARE_SENSEVOICE=<sensevoice-small-int8-gpu-onnx dir> cargo run --release -p local-adapter-fun-asr-nano --features cuda --example fun_asr_nano_transcribe -- <fun-asr-nano-onnx dir> <audio or @list>...`.
