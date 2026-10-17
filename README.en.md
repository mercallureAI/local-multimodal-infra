# local-multimodal-infra

> Packages the CPU / NVIDIA GPU compute of local and LAN machines as multimodal inference services for agents.

[中文](README.md)

Speech recognition, speech synthesis, chat, embedding and reranking, vision,
OCR, content moderation and realtime voice, all run locally on ONNX Runtime and
served to agents and applications over standard MCP and an OpenAI-compatible
API.

- **One service, several interfaces**: standard MCP (separate inference and admin catalogs), an OpenAI-compatible API (with a realtime voice WebSocket and content moderation), legacy JSON-RPC.
- **Controller / worker**: the controller keeps the model catalog, files and tasks and serves every interface; workers load and run the models, on one machine or several.
- **Models on demand**: lazy loading, concurrency limits, idle unloading; CUDA first, the CPU when CUDA is unavailable or fails to load (except for GPU-only models).
- **Reproducible models**: Hugging Face files pinned to a commit, URL files by SHA-256 and size; export scripts for models with no published package.
- **Rust services**: the controller and worker are Rust with no Python at run time (the Python scripts only export models and test); ONNX Runtime is loaded at run time.

## Supported models

Get it: **download** = one admin call; **export** = no published ONNX
package, export it locally from the source model as its doc shows.

### Speech recognition · [docs](docs/models/speech-recognition.md)

| Model | What | Precision | Get it | Hardware |
| --- | --- | --- | --- | --- |
| `sensevoice-small-onnx` | SenseVoiceSmall + FSMN-VAD + CAM++: text, timeline, language, emotion, speakers | INT8 | download | CPU / GPU |
| `sensevoice-small-int8-gpu-onnx` | the same, all on the GPU with CUDA graphs, ~16 ms for 1.5 s of speech; the realtime voice's default ASR | INT8 weights | export | GPU |

### Speech synthesis · [docs](docs/models/speech-synthesis.md)

| Model | What | Precision | Get it | Hardware |
| --- | --- | --- | --- | --- |
| `indextts-1.5-onnx` | IndexTTS 1.5, voice cloning from a reference | FP32 | download | CPU / GPU |
| `indextts-2.5-onnx` | IndexTTS 2.5, voice cloning plus emotion control | FP16 | download | GPU |
| `qwen3-tts-0.6b-onnx` | Qwen3-TTS 0.6B, streaming, first audio in ~50 ms; the realtime voice's default TTS | INT8 weights + FP16 | export | GPU |

### Chat · [docs](docs/models/chat.md)

| Model | What | Precision | Get it | Hardware |
| --- | --- | --- | --- | --- |
| `qwen3-4b-instruct-2507-int4-onnx` | Qwen3-4B, streaming, tool calls, KV prefix reuse | INT4 | export | GPU |

### Embedding and reranking · [docs](docs/models/embedding-rerank.md)

| Model | What | Precision | Get it | Hardware |
| --- | --- | --- | --- | --- |
| `multilingual-e5-small-onnx` | multilingual text embedding, 384 dimensions | INT8 (CPU) / FP16 (GPU) | download | CPU / GPU |
| `mmarco-minilm-l12-onnx` | multilingual reranking (cross-encoder) | INT8 (CPU) / FP16 (GPU) | download | CPU / GPU |

### Vision · [docs](docs/models/vision.md)

| Model | What | Precision | Get it | Hardware |
| --- | --- | --- | --- | --- |
| `yolo11n.onnx` | YOLO11n object detection, 80 COCO classes | FP32 | download | CPU / GPU |
| `depth-anything-v2-metric-indoor-small-onnx` | Depth Anything V2 monocular metric depth (indoor) | FP32 | export | CPU / GPU |

### OCR · [docs](docs/models/ocr.md)

| Model | What | Precision | Get it | Hardware |
| --- | --- | --- | --- | --- |
| `ppocrv5-mobile-onnx` | PP-OCRv5 text lines: scene text, UIs, name tags; also reads the text in images under moderation | FP32 | download | CPU / GPU |
| `unlimited-ocr-onnx` | Unlimited-OCR whole pages: layout boxes, HTML tables | FP16 + INT8 experts | export | GPU only |

### Content moderation · [docs](docs/models/moderation.md)

| Model | What | Precision | Get it | Hardware |
| --- | --- | --- | --- | --- |
| `qwen3guard-gen-0.6b-int4-onnx` | Qwen3Guard text moderation, nine categories (FP16 `qwen3guard-gen-0.6b-onnx` optional) | INT4 | export | GPU |
| `freepik-nsfw-image-detector-onnx` | image NSFW levels (Falconsai `falconsai-nsfw-image-detection-onnx` optional) | FP16 | export | CPU / GPU |

### Realtime voice · [docs](docs/models/realtime-voice.md)

| Model | What | Precision | Get it | Hardware |
| --- | --- | --- | --- | --- |
| `voice-cascade` | WebSocket voice conversation: Silero VAD → ASR → chat → TTS, wake words | FP32 (VAD) | download (uses the models above) | GPU |

"GPU" means NVIDIA CUDA; models marked GPU are built and measured for it, the
CPU is only a (slow) fallback, and "GPU only" models refuse the CPU. "INT8
weights" means weight-only INT8, computed in floating point. Model specs live
in [`configs/providers/<category>/`](configs/providers); spec fields, provider
fallback and overriding a model's config: [models](docs/models/README.md).

## Quick start

```bash
cp .env.example .env    # replace the placeholder tokens

docker compose up -d --build                                                    # CPU
ORT_CUDA_VERSION=12 docker compose -f docker-compose-nvidia.yml up -d --build   # or NVIDIA GPU

curl --fail http://127.0.0.1:17890/health
```

The NVIDIA stack needs Linux x86_64, a driver for CUDA 12, the NVIDIA
Container Toolkit and Compose 2.30+; only the worker gets the GPU (a CUDA 12.8
+ cuDNN image), the controller is the CPU image. Models are not baked into the
images; they live in `./workdir/models`.

Download a model:

```bash
curl --fail-with-body http://127.0.0.1:17890/rpc/admin \
  -H 'content-type: application/json' -H "x-local-admin-token: $LOCAL_ADMIN_TOKEN" \
  --data '{"jsonrpc":"2.0","id":1,"method":"download_model","params":{"id":"sensevoice-small-onnx"}}'
```

Port exposure, building from source and checking a deployment:
[getting started](docs/getting-started.md).

## Connecting

| Interface | Address | Auth |
| --- | --- | --- |
| MCP, inference (recommended for agents) | `http://127.0.0.1:17892/mcp/infer` | inference token, if configured |
| MCP, admin | `http://127.0.0.1:17892/mcp/admin` | admin token |
| OpenAI-compatible API | `http://127.0.0.1:17890/v1` | inference token, if configured (not for `GET /v1/models`) |
| JSON-RPC (legacy) | `http://127.0.0.1:17890/rpc/infer`, `/rpc/admin` | inference / admin token |

**Tokens.** Admin calls always need `LOCAL_ADMIN_TOKEN`;
`LOCAL_MCP_INFER_TOKENS` (comma-separated) guards every inference route when
it is not empty, and inference is unauthenticated when it is. Send either as
`Authorization: Bearer <token>` (JSON-RPC and the OpenAI routes also take
`x-local-infer-token`, admin routes `x-local-admin-token`). Compose publishes
17890 and 17891 on all interfaces and the MCP port on loopback only;
`/v1/models` and `/health` are outside the inference token, so protect the
controller port with network access control too.

### OpenAI-compatible and HTTP endpoints

Only these are implemented; it is not the full OpenAI API:

| Endpoint | Use |
| --- | --- |
| `GET /v1/models` | model list; besides the OpenAI fields each model has `name`, `task_kinds` (the tasks it runs), `enabled` and `downloaded` (its files are present) |
| `POST /v1/chat/completions` | chat, streaming (SSE), tool calls |
| `POST /v1/embeddings` | text embedding |
| `POST /rerank`, `/v1/rerank`, `/v2/rerank` | reranking (vLLM / Jina / Cohere style) |
| `POST /v1/audio/transcriptions` | speech recognition (JSON with a path the worker can read, returns `{"text"}`) |
| `POST /v1/audio/speech` | speech synthesis (JSON, returns a WAV file ref) |
| `POST /v1/moderations` | text and image moderation (OpenAI format; images as base64 `data:` URLs, NSFW-graded and their text checked; an input of several parts also gets `parts`, each part's own verdict) |
| `GET /v1/realtime` | realtime voice WebSocket, below |
| `POST /v1/ocr/lines` | PP-OCRv5 text lines; image as the body |
| `POST /v1/detect/objects` | YOLO object detection; image as the body |
| `POST /v1/depth` | metric depth grid; image as the body |
| `GET /v1/inferences` | every worker's recent inferences and their stage timings (a `voice.turn` per realtime voice reply included) |

**Realtime voice** `/v1/realtime`: the client streams 16 kHz 16-bit mono PCM
continuously; the server sends back 24 kHz audio and JSON events. Two modes:

- **Cascade** (default): the server runs the whole conversation, VAD →
  SenseVoice → Qwen3 → TTS, with barge-in, wake words in groups and speaker
  labels from the client; tasks for the client come as `tool.call`
  (`backend_task`), answered with `tool.result` / `note`.
- **Audio** (`"mode": "audio"`): the server only listens and speaks; the
  client runs the conversation with a model of its own (Codex's local voice
  uses this mode). The client gets `input.transcript` with `id` / `replaces` /
  `respond` and `state`, makes the server speak by streaming `response.delta`
  / `response.end` / `response.cancel`, and gets `response.done` when each
  response is over (`spoken`: what was heard). `moderation` can check what is
  heard before it reaches the client.

Every event and option: [realtime voice protocol](docs/realtime-voice.md).

### MCP

The inference catalog `/mcp/infer` has only the generic task tools:
`list_models` (the installed models, the task kinds each runs, and each task
kind's files, params and output), `run_task`, `create_task`, `start_task`,
`wait_task`, `get_task`, `sign_assets`. The admin catalog `/mcp/admin`
downloads and enables models and shows the cluster. Give an agent the
inference catalog only, never with the admin token:

```json
{
  "mcpServers": {
    "local-multimodal": {
      "type": "streamable-http",
      "url": "http://127.0.0.1:17892/mcp/infer",
      "headers": { "Authorization": "Bearer <inference token>" }
    }
  }
}
```

Drop `headers` when `LOCAL_MCP_INFER_TOKENS` is empty. Legacy JSON-RPC
`/rpc/infer` also keeps direct methods such as `asr_transcribe` and
`tts_synthesize`. Every endpoint, the task kinds, and URL results for large
outputs: [API](docs/api.md).

## GPU and ONNX Runtime

- The official ONNX Runtime 1.31.0, loaded at run time (ort `load-dynamic`),
  so building needs none. The Docker images ship it: the CPU build, and the
  CUDA 12 build in the GPU image (CUDA 12.8 + cuDNN).
- From source (Rust 1.92+): a GPU build needs `--features cuda`, without it
  every model runs on the CPU; `fetch_onnxruntime` then puts the
  SHA-256-checked ONNX Runtime beside the binaries:

  ```bash
  cargo build --release --bins --features cuda   # CPU: drop --features cuda
  python -m scripts.local.fetch_onnxruntime      # CPU: add --cpu
  ```

- The worker loads `ORT_DYLIB_PATH` if set, else the ONNX Runtime beside it.
  On Windows put a CUDA 12.8+ runtime (cudart, cuBLAS) and cuDNN 9 on PATH,
  and never let it pick up `C:\Windows\System32\onnxruntime.dll`.
- `/health` says the services are up; the providers a model really uses are
  in the worker's `lazy loading model` log line.

## Used by AstrBot

AstrBot connects to this service in its WebUI under Providers → Local service
(the `local_infra` setting): given the server address (without `/v1`) and an
inference token, it picks a model for each of speech-to-text, text-to-speech,
embeddings, rerank, content moderation and realtime voice from the task kinds
and enabled / downloaded state in `GET /v1/models`, and hands the same server
to plugins through `Context.get_local_infra()`.

## Documentation

- [Getting started](docs/getting-started.md)
- [API](docs/api.md)
- [Models](docs/models/README.md): getting, exporting, options and calls per category
- In depth: [realtime voice protocol](docs/realtime-voice.md) · [content moderation](docs/moderation.md) · [Qwen3-TTS export](docs/qwen3-tts.md) · [IndexTTS latency benchmark](docs/indextts-latency-benchmark.md) · [implementation notes](docs/implementation-notes.md)
- Development: [AGENTS.md](AGENTS.md) (constraints, test commands)

## License

The code is [MIT](LICENSE). Each model keeps its upstream license, named in
its doc.

## Acknowledgements

Model sources, pinned revisions and licenses are listed in each
[model doc](docs/models/README.md); the exact files, revisions and SHA-256 sums are
in [`configs/providers`](configs/providers) (the `local-registry` tests
check that every Hugging Face artifact is pinned to a commit and every URL
artifact has a SHA-256 and size).

Upstream code this project ports or follows:

- [modelcontextprotocol/rust-sdk](https://github.com/modelcontextprotocol/rust-sdk): the standard MCP Rust SDK.
- [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) and [pykeio/ort](https://github.com/pykeio/ort): the CPU / CUDA runtime and its Rust bindings.
- [microsoft/onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai): the Qwen3 and Qwen3Guard INT4 exports (`models.builder`).
- [modelscope/FunASR](https://github.com/modelscope/FunASR), [FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice): SenseVoice preprocessing, inference and the FSMN-VAD pipeline.
- [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics): YOLO preprocessing, output decoding and the COCO labels.
- [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR): PP-OCRv5, DB detection post-processing and CTC decoding.
- [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR): preprocessing, prompt, R-SWA and the no-repeat sampling.
- [index-tts/index-tts](https://github.com/index-tts/index-tts), [DakeQQ/Text-to-Speech-TTS-ONNX](https://github.com/DakeQQ/Text-to-Speech-TTS-ONNX): IndexTTS and its ONNX export and inference.
- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS): Qwen3-TTS (`qwen-tts`), the reference for the export and its checks.
- [PaddlePaddle/PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech), [wetext-rs](https://github.com/SpenserCai/wetext-rs) ([wetext](https://github.com/pengzhendong/wetext)), [jieba-rs](https://github.com/messense/jieba-rs): the Mandarin TTS frontend (see [implementation notes](docs/implementation-notes.md#mandarin-text-frontend-zh-tts-frontend)).
- [snakers4/silero-vad](https://github.com/snakers4/silero-vad): the realtime voice VAD.
- [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), [k2-fsa/icefall](https://github.com/k2-fsa/icefall): the wake word model, its keyword search (streaming zipformer2, boosted beam search, ContextGraph) and training recipe.
- [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2): metric depth.
