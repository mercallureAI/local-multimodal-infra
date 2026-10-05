# local-multimodal-infra

> Turn the CPU / NVIDIA GPU of a local or private-network machine into multimodal inference services that agents can call.

[中文 README](README.md)

## Overview

`local-multimodal-infra` is **local multimodal infrastructure**. It manages models, files, tasks and runtimes in one place and offers them to agents and applications through standard MCP, legacy JSON-RPC and a subset of the OpenAI-compatible API.

The project uses a controller / worker architecture:

| Component | Role | Default address |
| --- | --- | --- |
| Controller | Model and task management, file uploads, APIs, task scheduling | `http://127.0.0.1:17890` |
| Standard MCP Server | Separate admin and inference tool catalogs | `http://127.0.0.1:17892/mcp/admin`, `http://127.0.0.1:17892/mcp/infer` |
| Worker | Loads ONNX models and runs inference | `http://127.0.0.1:17891` |

All runtime data lives in `workdir`:

- `workdir/models`: model artifacts;
- `workdir/data`: SQLite, uploads, generated results, logs and temporary files.

Models and input files are not baked into the images. Local configs bind to loopback by default; the current Docker Compose publishes controller `17890` and worker `17891` on all host interfaces, while MCP `17892` is published on loopback only. Before deploying, adjust the port bindings to your use and configure authentication and network access control as well.

## Features

### Inference

| Capability | Default model | Status | Main output |
| --- | --- | --- | --- |
| Object detection | `yolo11n.onnx` | Enabled by default | Classes, confidences, bounding boxes |
| Monocular metric depth | `depth-anything-v2-metric-indoor-small-onnx` | Enabled after a local export | A grid of mean depths (metres) |
| Document OCR | `unlimited-ocr-onnx` | Enabled after a local export (int8 experts, NVIDIA GPU required) | Page text (Markdown/HTML tables) with layout categories and boxes |
| Text lines OCR | `ppocrv5-mobile-onnx` | Enabled by default | Each line's text, confidence and pixel box (scene text, UI, name tags) |
| Speech recognition | `sensevoice-small-onnx` | Enabled by default | Text, timeline, language, emotion, speaker |
| Speech recognition (low latency) | `sensevoice-small-fp16-onnx` | Enabled after a local export | The same; a float16 graph, ~40-90 ms per utterance on CUDA |
| Speech synthesis | `indextts-1.5-onnx` | Enabled by default | WAV audio |
| Speech synthesis | `indextts-2.5-onnx` | Enabled by default (FP16, NVIDIA GPU recommended) | WAV audio with emotion control |
| Speech synthesis | `qwen3-tts-0.6b-onnx` | Enabled after a local export (INT8 weights, NVIDIA GPU recommended) | Streamed 24 kHz audio, voice cloned from a 3 s reference, text accepted while it is written |
| Text embedding | `multilingual-e5-small-onnx` | Enabled by default | 384-dimensional normalized vectors |
| Reranking | `mmarco-minilm-l12-onnx` | Enabled by default | Document relevance order and scores |
| Chat completion | `qwen3-4b-instruct-2507-int4-onnx` | Enabled after a local export | Streaming text and tool calls (Qwen3 template, KV prefix reuse) |
| Realtime voice | `voice-cascade` | Enabled by default; needs the ASR, chat and TTS models | `/v1/realtime` WebSocket voice conversation (Silero VAD + SenseVoice + Qwen3 + Qwen3-TTS, or IndexTTS for TTS, see `docs/realtime-voice.md`) |
| Wake words | Built into realtime voice (sherpa-onnx KWS zipformer zh-en 3M) | Enabled once `scripts.local.fetch_kws_model` has put the model in place | Spots the bot's name and other wake words in group sessions (`input.wake`); only what calls it wants a reply, the rest is context |

All models run on ONNX Runtime (the official ONNX Runtime 1.30, loaded at run time; see `docs/implementation-notes.md`). Model configs ask for CUDA first with CPU fallback; the provider actually used still depends on the build, the environment and each model's operator support.

Document OCR uses [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) (DeepEncoder + DeepSeek-V2 MoE with R-SWA attention) and recognizes one page image per request (base mode, padded to 1024×1024). The result is the page text with a `<|det|>category [x1, y1, x2, y2]<|/det|>` tag before each layout block, coordinates normalized to 0–999, tables as HTML. Call it through the MCP / legacy RPC `ocr_recognize` (an `image` FileRef or `image_path`) or the generic `ocr.recognize` task (upload `image`).

Text lines in pictures (scene text, UI labels, name tags) use PaddleOCR's PP-OCRv5 mobile detection and recognition models (Chinese, English, Japanese and more in one model; the two ONNX files are about 21 MB), returning each line with its pixel box, top to bottom then left to right. For frame-by-frame use, `POST /v1/ocr/lines[?model=ppocrv5-mobile-onnx]` takes the image (PNG/JPEG/BMP) as the body and answers at once with `{"lines": [{"text", "confidence", "bbox"}]}`, behind the inference token; the generic `ocr.lines` task (upload `image`) works too. Object detection has the same frame-by-frame form, `POST /v1/detect/objects[?model=yolo11n.onnx]`, answering `{"objects": [{"label", "confidence", "bbox"}]}`.

Monocular metric depth uses Depth Anything V2 Metric Indoor Small (ViT-S fine-tuned on indoor Hypersim, up to 20 m, Apache-2.0), exported from a pinned revision to a fixed-input-size ONNX (~99 MB) by `python -m scripts.local.depth_anything_export --size 308x546 --output-dir <models>/depth-anything-v2-metric-indoor-small-onnx`. Images are stretched to that size, so the export suits 16:9 frames (1280x720 game frames); other aspect ratios come out distorted, their depth biased: export for theirs (`--size`). The frame-by-frame `POST /v1/depth[?model=&cols=&rows=]` (default 64x36) answers `{"cols", "rows", "max_depth", "depth": [metres, row by row from the top]}`, each cell the mean depth of its area; the generic `depth.estimate` task (upload `image`, `params.cols/rows`) works too. A 1280x720 frame takes ~30 ms on an RTX 4090 (decoding included).

Low-latency speech recognition `sensevoice-small-fp16-onnx` is the same SenseVoiceSmall as `sensevoice-small-onnx`: the latter's int8 graph has 281 `DynamicQuantizeLinear` nodes that fall back to the CPU under the CUDA provider, each layer going back and forth, ~200-350 ms per utterance; the float16 graph runs on the GPU throughout, ~40-90 ms, the same transcripts. `python -m scripts.local.sensevoice_fp16_export --base-dir <models>/sensevoice-small-onnx --output-dir <models>/sensevoice-small-fp16-onnx` exports it from a pinned revision (~470 MB), the other files taken from `sensevoice-small-onnx`; the metadata `asr_model_file` names the graph in `asr/`. The realtime voice cascade picks it in `voice-cascade.yaml`'s `asr_model` (the default; where it is not exported, a session falls back to `asr_fallback_model`, `sensevoice-small-onnx`, when it starts).

Group sessions of realtime voice answer only what calls the bot by a wake word: k2-fsa's open-vocabulary zipformer transducer KWS model for Chinese and English (`sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20`, Apache-2.0, 3.3M parameters, about 13 MB as the fp32 chunk-16 export), run on the CPU by `local-adapter-kws-zipformer`, a port of sherpa-onnx's keyword spotter (Kaldi fbank, the streaming encoder in 320 ms chunks, a beam search boosted along the wake words): about 2 % of a core per stream, a word spotted 0.2-0.6 s after it is said. Wake words are written as text (the bot's name, aliases, `wake_words`): Chinese read in pinyin, English with the model's dictionary, letters and digits each their own way. `python -m scripts.local.fetch_kws_model` downloads and checks the release archive and lays it out in `<models>/voice-cascade/kws`. See Wake words in `docs/realtime-voice.md`.

SenseVoice ASR includes FSMN-VAD and CAM++ speaker identification. By default it returns plain text, `timestamped_text` at about 10-second granularity, `segments[].speaker` and `speakers[]`. Use `timestamps`, `timestamp_granularity_sec`, `token_timestamps` and `speaker_diarization` to adjust or turn off these results.

### Interfaces

| Interface | Use | Authentication |
| --- | --- | --- |
| `POST /rpc/admin` | Legacy JSON-RPC model, node and asset management | Requires `LOCAL_ADMIN_TOKEN` |
| `POST /rpc/infer` | Legacy JSON-RPC inference and generic tasks | Enforced once `LOCAL_MCP_INFER_TOKENS` is set |
| `/mcp/admin` | Standard MCP admin tools | Requires `LOCAL_ADMIN_TOKEN` |
| `/mcp/infer` | Standard MCP inference tools | Enforced once `LOCAL_MCP_INFER_TOKENS` is set |
| `/v1/models` | OpenAI-compatible model list | None |
| `/v1/audio/transcriptions` | OpenAI-compatible ASR | `LOCAL_MCP_INFER_TOKENS` |
| `/v1/audio/speech` | OpenAI-compatible TTS | `LOCAL_MCP_INFER_TOKENS` |
| `/v1/embeddings` | OpenAI-compatible embeddings | `LOCAL_MCP_INFER_TOKENS` |
| `/v1/chat/completions` | OpenAI-compatible chat (`stream: true` uses SSE) | `LOCAL_MCP_INFER_TOKENS` |
| `/v1/realtime` | Realtime voice WebSocket (VAD → ASR → chat → TTS, see `docs/realtime-voice.md`) | `LOCAL_MCP_INFER_TOKENS` |
| `/rerank`, `/v1/rerank`, `/v2/rerank` | vLLM / Jina / Cohere style reranking | `LOCAL_MCP_INFER_TOKENS` |

Admin and all MCP, RPC and OpenAI-compatible inference interfaces accept `Authorization: Bearer <token>`; legacy JSON-RPC and OpenAI-compatible inference also accept `x-local-infer-token`, and the admin interfaces accept `x-local-admin-token`.

### Infrastructure

- Model configs, asynchronous downloads, SHA-256 checks, download status and deduplication of concurrent downloads;
- Enabling and disabling models, lazy loading, concurrency limits and idle unloading;
- Controller / worker scheduling, with CPU / CUDA provider selection and fallback;
- Signed upload URLs, task inputs, generated artifacts and local asset management;
- Standard MCP direct tools and the generic "create task → upload files → start → wait for result" flow;
- Smoke harness for release, RPC, MCP and real-model call chains.

## Quick deployment

### 1. Prepare the configuration

You need Docker and Docker Compose. Models are not shipped with the images; download them into `workdir/models` after the first start.

```bash
cp .env.example .env
```

Edit `.env` and replace at least these placeholders:

```dotenv
LOCAL_WORKER_REGISTRATION_TOKEN=replace-with-a-long-random-worker-registration-token
LOCAL_UPLOAD_SIGNING_SECRET=replace-with-a-long-random-upload-signing-secret
LOCAL_ADMIN_TOKEN=replace-with-a-long-random-admin-token
LOCAL_MCP_INFER_TOKENS=
LOCAL_PUBLIC_BASE_URL=http://127.0.0.1:17890
```

With `LOCAL_MCP_INFER_TOKENS` empty, the inference interfaces require no authentication; set it to a comma-separated list of tokens and the MCP, JSON-RPC and OpenAI-compatible inference interfaces all require one of them.

If the service is only for this machine, change `17890:17890` and `17891:17891` in the Compose file to `127.0.0.1:17890:17890` and `127.0.0.1:17891:17891`. `/v1/models`, the health check and some asset routes are outside inference authentication, so still protect the controller port with network access control.

### 2. Start the CPU services

```bash
docker compose up -d --build
docker compose ps
curl --fail http://127.0.0.1:17890/health
```

### 3. Start the NVIDIA CUDA services

A CUDA deployment needs the NVIDIA driver, the NVIDIA Container Toolkit and a working `nvidia-smi`. The image uses the CUDA 12 build of ONNX Runtime and supports Linux x86_64 containers:

```bash
nvidia-smi
ORT_CUDA_VERSION=12 docker compose -f docker-compose-nvidia.yml up -d --build
docker compose -f docker-compose-nvidia.yml exec worker nvidia-smi
```

CUDA Compose gives the GPU to the worker only; the controller keeps running the CPU image. `/health` only means the services are up; it does not prove that a model has run inference on CUDA.

### 4. Download models

First list the configured models and their download status:

```bash
curl --fail-with-body http://127.0.0.1:17890/rpc/admin \
  -H 'content-type: application/json' \
  -H 'x-local-admin-token: replace-with-your-admin-token' \
  --data '{"jsonrpc":"2.0","id":"models","method":"list_models","params":{}}'
```

Submit an asynchronous download and query the per-file status:

```bash
curl --fail-with-body http://127.0.0.1:17890/rpc/admin \
  -H 'content-type: application/json' \
  -H 'x-local-admin-token: replace-with-your-admin-token' \
  --data '{"jsonrpc":"2.0","id":"download","method":"download_model","params":{"id":"sensevoice-small-onnx"}}'

curl --fail-with-body http://127.0.0.1:17890/rpc/admin \
  -H 'content-type: application/json' \
  -H 'x-local-admin-token: replace-with-your-admin-token' \
  --data '{"jsonrpc":"2.0","id":"status","method":"get_model_download_status","params":{"id":"sensevoice-small-onnx"}}'
```

Other default model IDs:

- `yolo11n.onnx`
- `ppocrv5-mobile-onnx`
- `multilingual-e5-small-onnx`
- `mmarco-minilm-l12-onnx`
- `indextts-1.5-onnx`
- `indextts-2.5-onnx` (FP16, about 2.8 GB)
- `voice-cascade` (downloads only Silero VAD; the ASR, chat and TTS models the conversation uses are downloaded or exported separately, the default ASR `sensevoice-small-fp16-onnx` being a local export, see above; the group wake word model goes in `voice-cascade/kws` with `python -m scripts.local.fetch_kws_model`)

`depth-anything-v2-metric-indoor-small-onnx` and `sensevoice-small-fp16-onnx` have no published package; export them locally with the commands above.

`qwen3-4b-instruct-2507-int4-onnx` has no published ONNX package; export it from the pinned revision with the commands in [`configs/providers/chat/qwen3-chat.yaml`](configs/providers/chat/qwen3-chat.yaml) into `workdir/models/qwen3-4b-instruct-2507-int4-onnx`.

`qwen3-tts-0.6b-onnx` (the realtime voice's default TTS) has no published package either: export it from a pinned revision (export environment, graphs, INT8 quality comparison and latency numbers in [`docs/qwen3-tts.md`](docs/qwen3-tts.md)):

```bash
hf download Qwen/Qwen3-TTS-12Hz-0.6B-Base --revision 5d83992436eae1d760afd27aff78a71d676296fc --local-dir <src>
python -m scripts.local.qwen3_tts_export --source-model-dir <src> \
    --output-dir workdir/models/qwen3-tts-0.6b-onnx --builder-python <python with onnxruntime-genai>
```

On an RTX 4090 a frame (80 ms of speech) takes about 5.6 ms and a clause starts playing about 50 ms after it is asked for; in the realtime voice the answer starts about 0.9–1.0 s after the speaker stops (0.6 s of which is the VAD's end-of-speech silence).

`unlimited-ocr-onnx` has no published package either; export it locally from the PyTorch checkpoint:

```bash
hf download baidu/Unlimited-OCR --revision 07dea832e22aefee32ad281d4b80551282e1c168 --local-dir <src>
python -m scripts.local.unlimited_ocr_export export --source <src> --out workdir/models/unlimited-ocr-onnx
```

The export runs on Python 3.11 with the upstream pins `torch==2.10.0` (the CPU build is enough), `torchvision==0.25.0` and `transformers==4.57.1`, plus `onnx onnxruntime-gpu==1.30.0 einops addict easydict safetensors pillow matplotlib`; the `parity` subcommand compares the package with the PyTorch model token by token. The default int8 experts run only on CUDA (build the worker with `--features cuda`). On an RTX 4090 a page decodes at about 145 tokens/s in about 6.6 GB of VRAM; the 14-page paper PDF averages 6.8 s per page, against 35 s per page for the official transformers implementation on the same GPU.

### 5. Using it from an agent

Give agents only the inference MCP:

- Inference: `http://127.0.0.1:17892/mcp/infer`
- Admin: `http://127.0.0.1:17892/mcp/admin`, configured separately and only when the agent really needs to download, enable or disable models

A common Streamable HTTP MCP configuration looks like this; file and field names vary slightly between agents:

```json
{
  "mcpServers": {
    "local-multimodal": {
      "type": "streamable-http",
      "url": "http://127.0.0.1:17892/mcp/infer",
      "headers": {
        "Authorization": "Bearer replace-with-your-infer-token"
      }
    }
  }
}
```

If `LOCAL_MCP_INFER_TOKENS` is empty, drop `headers`. For the admin tools, add a separate MCP server entry with the URL changed to `/mcp/admin` and `LOCAL_ADMIN_TOKEN`; do not reuse the inference token.

Once connected, an agent can call `object_detect`, `ocr_recognize`, `asr_transcribe`, `tts_synthesize`, `text_embed` and `text_rerank` directly. For images or audio the agent cannot reach directly, use `create_task` → upload to the returned signed URL → `start_task` → `wait_task`; no host file paths need to be shared with the worker.

Every MCP tool that returns an inference result accepts `with_url_result`:

- `auto` (default): when the result's UTF-8 serialized text is at most 1000 bytes, it is returned inline in its original structure; otherwise a preview of the first 1000 bytes (cut at a UTF-8 boundary) and a download URL are returned.
- `on`: always returns a preview of at most 1000 UTF-8 bytes and a download URL; for short results the preview is the full text.
- `off`: always returns the full original result inline through MCP.

In URL mode the full result is stored as a `.txt` artifact with `text/plain; charset=utf-8`, and the response includes `preview`, `truncated`, `download_url`, `artifact_uri`, `size_bytes`, `sha256` and `expires_at`. These text artifacts are managed by the artifact center: they are cleaned up after 24 hours by default, identical content reuses the same artifact and renews it, and signed download URLs are valid for 10 minutes by default; call again for a new URL.

To integrate with OpenAI-style clients, point the base URL at `http://127.0.0.1:17890/v1` and use any token from `LOCAL_MCP_INFER_TOKENS` as the API key / Bearer token. This interface implements only the local capabilities listed above; it is not the full OpenAI API.

### 6. Verify the deployment

The repository includes a smoke harness that builds, starts the services, waits for health, sends real requests and cleans up the processes:

```bash
python -m scripts.local.smoke --tests rpc \
  --workdir ./workdir --model-dir ./workdir/models

python -m scripts.local.smoke --tests mcp \
  --workdir ./workdir --model-dir ./workdir/models
```

The `mcp` tests need the official `mcp` SDK in the current Python environment. For release verification, run `cargo build --release --bins` first and add `--skip-build --release` to the smoke harness.

Both `rpc` and `mcp` include OCR (`--tests ocr` runs it alone); it is reported as skipped when there is no local `unlimited-ocr-onnx` export or the worker has no usable CUDA. To cover OCR, build with `cargo build --release --bins -p local-cli --features cuda` and run `python -m scripts.local.smoke --skip-build --release --tests ocr --workdir ./workdir --model-dir ./workdir/models --request-timeout 300`.

## References

### Model repositories

| Use | Repository | Pinned revision |
| --- | --- | --- |
| YOLO11n ONNX | [aaurelions/yolo11n.onnx](https://huggingface.co/aaurelions/yolo11n.onnx) | `f46d9b72aa9a0f02bc00484446e2310b1a549bce` |
| YOLO COCO labels | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics/blob/eba96641b5cea142e21641909d6400fef7134244/ultralytics/cfg/datasets/coco.yaml) `coco.yaml` | `eba96641b5cea142e21641909d6400fef7134244` |
| SenseVoiceSmall ONNX | [haixuantao/SenseVoiceSmall-onnx](https://huggingface.co/haixuantao/SenseVoiceSmall-onnx) | `c4c8747214bed7ebbf2557e0412c19efa540023c` |
| FSMN-VAD ONNX | [funasr/fsmn-vad-onnx](https://huggingface.co/funasr/fsmn-vad-onnx) | `f6e9fbb4cefa7397216c763f21307993f147f585` |
| FSMN-VAD config | [MoYoYoTech/Translator](https://huggingface.co/MoYoYoTech/Translator) | `58fbad4088820ed1253955c8faf1444cd0b2dc69` |
| CAM++ Speaker | [welcomyou/campplus-3dspeaker-200k-onnx](https://huggingface.co/welcomyou/campplus-3dspeaker-200k-onnx) | `6265ff7af2a104d745b4389026ed9815c6c1c6ff` |
| IndexTTS 1.5 ONNX | [ModaLeap/indextts-1.5-onnx](https://huggingface.co/ModaLeap/indextts-1.5-onnx) | `3f1a422cd97a0b7dbb9b6ad4698dc0fde66796d1` |
| IndexTTS 2.5 ONNX FP16 | [ModaLeap/indextts-2.5-onnx](https://huggingface.co/ModaLeap/indextts-2.5-onnx) | `fd246cb6c2cf046113cd3400565edf681ac1b68b` |
| IndexTTS Mandarin frontend (WeText + g2pW) | [ModaLeap/zh-tts-frontend](https://huggingface.co/ModaLeap/zh-tts-frontend) | `ba6b85aeb17ebc58d2d3d73121096f9495ee710e` |
| multilingual-e5-small | [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | `614241f622f53c4eeff9890bdc4f31cfecc418b3` |
| mMARCO MiniLM reranker | [cross-encoder/mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) | `1427fd652930e4ba29e8149678df786c240d8825` |
| PP-OCRv5 mobile detection / recognition (ONNX) | [ilaylow/PP_OCRv5_mobile_onnx](https://huggingface.co/ilaylow/PP_OCRv5_mobile_onnx), dictionary [PaddleOCR `ppocrv5_dict.txt`](https://github.com/PaddlePaddle/PaddleOCR/blob/a38c087bcb2579f9ccc2068aea02ec893b1c2311/ppocr/utils/dict/ppocrv5_dict.txt) | `f97b337b3ac256f9dffcac5fc53955082d919d58` / `a38c087bcb2579f9ccc2068aea02ec893b1c2311` |
| Unlimited-OCR (source of the local ONNX export) | [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) | `07dea832e22aefee32ad281d4b80551282e1c168` |
| Qwen3-TTS-12Hz-0.6B-Base (source of the local ONNX export) | [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) | `5d83992436eae1d760afd27aff78a71d676296fc` |
| Qwen3-4B-Instruct-2507 (source of the local INT4 export) | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Depth Anything V2 Metric Indoor Small (source of the local ONNX export) | [depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf) | `8078d68a9c75a972131914f6afd0c1723be0da7f` |
| SenseVoiceSmall (source of the local float16 ONNX export) | [FunAudioLLM/SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) | `3847d57b6bdf2dd8875cb1508d2af43d80a16bf7` |
| Silero VAD v6.2.3 | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | `5cd7945676eb32225748052e2e6a0580e4686a08` |
| sherpa-onnx KWS zipformer zh-en 3M (wake words) | [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models) | `sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20` |

The exact files, revisions and SHA-256 sums are the ones in [`configs/providers`](configs/providers) (one directory per category): Hugging Face artifacts are pinned to a commit and URL artifacts carry a SHA-256, which the `local-registry` tests check. The IndexTTS 1.5 and 2.5 configs both download the Mandarin frontend (`ModaLeap/zh-tts-frontend`, about 177 MB; per-file licenses in its `NOTICE`) into `zh-tts-frontend/` inside their model directories; `scripts/local/zh_frontend_export.py` can also rebuild it locally.

### Reference code

- [modelscope/FunASR](https://github.com/modelscope/FunASR): reference for SenseVoice ONNX preprocessing, inference and the FSMN-VAD pipeline;
- [FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice): the SenseVoice model and official implementation;
- [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics): YOLO preprocessing, output decoding and the COCO labels;
- [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR): the PP-OCRv5 models, reference for DB detection postprocessing and CTC decoding;
- [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR): the Unlimited-OCR model and official implementation (preprocessing, prompt, R-SWA and the no-repeat sampler);
- [index-tts/index-tts](https://github.com/index-tts/index-tts): the official IndexTTS implementation;
- [DakeQQ/Text-to-Speech-TTS-ONNX](https://github.com/DakeQQ/Text-to-Speech-TTS-ONNX): reference for IndexTTS ONNX export and inference;
- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS): the official Qwen3-TTS implementation (`qwen-tts`), reference for the export and its checks;
- [snakers4/silero-vad](https://github.com/snakers4/silero-vad): the Silero VAD model for realtime voice;
- [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx): the wake word model's release, reference for the keyword spotter port (streaming zipformer2, keyword-boosted beam search, ContextGraph);
- [k2-fsa/icefall](https://github.com/k2-fsa/icefall): the wake word model's training recipe and results;
- [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2): the Depth Anything V2 metric depth models and official implementation;
- [microsoft/onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai): Qwen3 INT4 ONNX export (`models.builder`);
- [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime): CPU / CUDA inference runtime;
- [modelcontextprotocol/rust-sdk](https://github.com/modelcontextprotocol/rust-sdk): the standard MCP Rust SDK.

### Project documents

- [Implementation notes](docs/implementation-notes.md)
- [Realtime voice](docs/realtime-voice.md)
- [Qwen3-TTS export and runtime](docs/qwen3-tts.md)
- [Development and verification rules](AGENTS.md)
- [CPU Compose](docker-compose.yml)
- [NVIDIA CUDA Compose](docker-compose-nvidia.yml)
- [Environment variable example](.env.example)
