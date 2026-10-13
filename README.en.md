# local-multimodal-infra

> Packages the CPU / NVIDIA GPU compute of local and LAN machines as multimodal inference services for agents.

[中文](README.md)

Speech recognition, speech synthesis, chat, embedding and reranking, vision,
OCR, content moderation and realtime voice, all run locally on ONNX Runtime and
served to agents and applications over standard MCP and an OpenAI-compatible
API.

- **One service, several interfaces**: standard MCP (separate inference and admin catalogs), OpenAI-compatible API, JSON-RPC.
- **Controller / worker**: the controller manages models, files and tasks; workers load and run the models, on one machine or several.
- **Models on demand**: lazy loading, concurrency limits, idle unloading; CUDA first, CPU when CUDA is unavailable.
- **Reproducible models**: downloads pinned to commits and checked by SHA-256; export scripts for models with no published package.
- **Pure Rust**: no Python at run time.

## Supported models

Get it: **download** = one admin call; **export** = no published ONNX
package, export it locally from the source model as its doc shows.

### Speech recognition · [docs](docs/models/speech-recognition.md)

| Model | What | Precision | Get it | Hardware |
| --- | --- | --- | --- | --- |
| `sensevoice-small-onnx` | SenseVoiceSmall + FSMN-VAD + CAM++: text, timeline, language, emotion, speakers | INT8 | download | CPU / GPU |
| `sensevoice-small-int8-gpu-onnx` | the same, all on the GPU with CUDA graphs, ~16 ms for 1.5 s of speech | INT8 weights | export | GPU |

### Speech synthesis · [docs](docs/models/speech-synthesis.md)

| Model | What | Precision | Get it | Hardware |
| --- | --- | --- | --- | --- |
| `indextts-1.5-onnx` | IndexTTS 1.5, voice cloning from a reference | FP32 | download | CPU / GPU |
| `indextts-2.5-onnx` | IndexTTS 2.5, emotion control | FP16 | download | GPU |
| `qwen3-tts-0.6b-onnx` | Qwen3-TTS 0.6B, streaming, first audio in ~50 ms | INT8 weights + FP16 | export | GPU |

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
| `ppocrv5-mobile-onnx` | PP-OCRv5 text lines: scene text, UIs, name tags | FP32 | download | CPU / GPU |
| `unlimited-ocr-onnx` | Unlimited-OCR whole pages: layout boxes, HTML tables | FP16 + INT8 experts | export | GPU only |

### Content moderation · [docs](docs/models/moderation.md)

| Model | What | Precision | Get it | Hardware |
| --- | --- | --- | --- | --- |
| `qwen3guard-gen-0.6b-int4-onnx` | Qwen3Guard text moderation (FP16 optional) | INT4 | export | GPU |
| `freepik-nsfw-image-detector-onnx` | image NSFW levels (Falconsai optional) | FP16 | export | CPU / GPU |

### Realtime voice · [docs](docs/models/realtime-voice.md)

| Model | What | Precision | Get it | Hardware |
| --- | --- | --- | --- | --- |
| `voice-cascade` | WebSocket voice conversation: VAD → ASR → chat → TTS, wake words | FP32 (VAD) | download (uses the models above) | GPU |

"GPU" means NVIDIA CUDA; "INT8 weights" means weight-only INT8, computed in floating point. Spec fields, provider fallback and overriding a
model's config: [models](docs/models/README.md).

## Quick start

```bash
cp .env.example .env    # replace the placeholder tokens

docker compose up -d --build                                                    # CPU
ORT_CUDA_VERSION=12 docker compose -f docker-compose-nvidia.yml up -d --build   # or NVIDIA GPU

curl --fail http://127.0.0.1:17890/health
```

Download a model:

```bash
curl --fail-with-body http://127.0.0.1:17890/rpc/admin \
  -H 'content-type: application/json' -H "x-local-admin-token: $LOCAL_ADMIN_TOKEN" \
  --data '{"jsonrpc":"2.0","id":1,"method":"download_model","params":{"id":"sensevoice-small-onnx"}}'
```

Port exposure, building from source and checking a deployment:
[getting started](docs/getting-started.md).

## Connecting

| Interface | Address |
| --- | --- |
| MCP (recommended for agents) | `http://127.0.0.1:17892/mcp/infer` (admin: `/mcp/admin`) |
| OpenAI-compatible API | `http://127.0.0.1:17890/v1` |
| JSON-RPC | `http://127.0.0.1:17890/rpc/infer`, `/rpc/admin` |

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

Every endpoint, authentication, and URL results for large outputs:
[API](docs/api.md).

## Documentation

- [Getting started](docs/getting-started.md)
- [API](docs/api.md)
- [Models](docs/models/README.md): getting, exporting, options and calls per category
- In depth: [realtime voice protocol](docs/realtime-voice.md) · [content moderation](docs/moderation.md) · [Qwen3-TTS export](docs/qwen3-tts.md) · [implementation notes](docs/implementation-notes.md)
- Development: [AGENTS.md](AGENTS.md) (constraints, test commands)

## License

The code is [MIT](LICENSE). Each model keeps its upstream license, named in
its doc.

## Acknowledgements

Model sources, pinned revisions and licenses are listed in each
[model doc](docs/models/README.md); the exact files, revisions and SHA-256 sums are
in [`configs/providers`](configs/providers) (the `local-registry` tests
check that every Hugging Face artifact is pinned to a commit and every URL
artifact has a SHA-256).

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
