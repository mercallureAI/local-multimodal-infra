# Speech recognition

| Model | Get it | Runs on |
| --- | --- | --- |
| `sensevoice-small-onnx` | download | CPU or CUDA |
| `sensevoice-small-int8-gpu-onnx` | local export | CUDA (CPU works, slowly) |

Both are the same pipeline: FSMN-VAD cuts speech segments (at most 20 s,
cut at the weakest speech), SenseVoiceSmall transcribes them (language and
emotion tags included), and CAM++ speaker embeddings label who spoke.
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

Export (from `FunAudioLLM/SenseVoiceSmall` at
`3847d57b6bdf2dd8875cb1508d2af43d80a16bf7`, FunASR model license; download
`sensevoice-small-onnx` first, the rest of the directory comes from it):

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

## Options

Spec `metadata` (defaults) and per-request parameters of the same name:

| Key | Default | Meaning |
| --- | --- | --- |
| `timestamps` (`timestamps_default`) | `true` | `timestamped_text` and `segments` with times |
| `timestamp_granularity_sec` | `10` | length of the timeline's segments |
| `token_timestamps` (`token_timestamps_default`) | `false` | per-token times |
| `speaker_diarization` (`speaker_diarization_default`) | `true` | `segments[].speaker` and `speakers[]` |
| `vad_max_segment_seconds` | `20` | longest speech segment |
| `asr_model_file` | `model_quant.onnx` | the graph in `asr/` |
| `asr_cuda_graph_frames` | (unset) | CUDA graph buckets, in LFR frames |

## Calling it

- MCP / JSON-RPC `asr_transcribe` (`audio` file ref), or the generic task
  `asr.transcribe` with an `audio` upload: full output (`text`,
  `timestamped_text`, `segments`, `speakers`).
- `POST /v1/audio/transcriptions` (JSON `{"model", "file": {"path"}}`, a path
  the worker can read): `{"text"}` only.

Long-audio timing and memory:
`cargo run --release -p local-adapter-sensevoice-asr --features cuda --example long_audio -- <model dir> <audio>`.
