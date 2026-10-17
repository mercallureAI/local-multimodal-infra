# local-multimodal-infra

> 将本机及内网的 CPU / NVIDIA GPU 算力封装为可供 Agent 调用的多模态推理服务。

[English](README.en.md)

语音识别、语音合成、对话、向量与重排、视觉、OCR、内容审核、实时语音，全部在本地通过 ONNX Runtime 运行，以标准 MCP 和 OpenAI 兼容 API 提供给 Agent 与应用。

- **一套服务，多种接入**：标准 MCP（推理 / 管理两个独立目录）、OpenAI 兼容 API（含实时语音 WebSocket 与内容审核）、旧版 JSON-RPC。
- **controller / worker 架构**：controller 管理模型目录、文件与任务并对外提供全部接口，worker 加载模型推理，可分布在多台机器。
- **模型按需加载**：懒加载、并发限制、空闲卸载；CUDA 优先，CUDA 不可用或加载失败时回退 CPU（仅 GPU 的模型除外）。
- **可复现的模型**：Hugging Face 文件固定到 commit，URL 文件固定 SHA-256 与大小；没有发布包的模型提供导出脚本。
- **Rust 服务**：controller / worker 均为 Rust，运行时不依赖 Python（Python 脚本只用于导出模型和测试）；ONNX Runtime 在运行时动态加载。

## 支持的模型

获取方式：**下载** = 管理接口一键下载；**导出** = 没有发布的 ONNX 包，按文档从源模型本地导出。

### 语音识别 · [文档](docs/models/speech-recognition.md)

| 模型 | 说明 | 精度 | 获取 | 硬件 |
| --- | --- | --- | --- | --- |
| `sensevoice-small-onnx` | SenseVoiceSmall + FSMN-VAD + CAM++：文本、时间轴、语言、情绪、发言人 | INT8 | 下载 | CPU / GPU |
| `sensevoice-small-int8-gpu-onnx` | 同上，全程在 GPU 上 + CUDA graph，1.5 秒语音约 16 ms；实时语音的默认 ASR | INT8 权重 | 导出 | GPU |
| `fun-asr-nano-onnx` | Fun-ASR-Nano（SenseVoice 编码器 + Qwen3-0.6B）+ 同一管线：比 SenseVoice 更准，支持热词；1.5 秒语音约 30–45 ms | INT8 权重 + FP16 | 导出 | GPU |

### 语音合成 · [文档](docs/models/speech-synthesis.md)

| 模型 | 说明 | 精度 | 获取 | 硬件 |
| --- | --- | --- | --- | --- |
| `indextts-1.5-onnx` | IndexTTS 1.5，参考音频克隆声音 | FP32 | 下载 | CPU / GPU |
| `indextts-2.5-onnx` | IndexTTS 2.5，克隆声音并可控制情绪 | FP16 | 下载 | GPU |
| `qwen3-tts-0.6b-onnx` | Qwen3-TTS 0.6B，流式输出，首段音频约 50 ms；实时语音的默认 TTS | INT8 权重 + FP16 | 导出 | GPU |

### 对话 · [文档](docs/models/chat.md)

| 模型 | 说明 | 精度 | 获取 | 硬件 |
| --- | --- | --- | --- | --- |
| `qwen3-4b-instruct-2507-int4-onnx` | Qwen3-4B，流式、工具调用、KV 前缀复用 | INT4 | 导出 | GPU |

### 文本向量与重排 · [文档](docs/models/embedding-rerank.md)

| 模型 | 说明 | 精度 | 获取 | 硬件 |
| --- | --- | --- | --- | --- |
| `multilingual-e5-small-onnx` | 多语言文本向量，384 维 | INT8 (CPU) / FP16 (GPU) | 下载 | CPU / GPU |
| `mmarco-minilm-l12-onnx` | 多语言重排（cross-encoder） | INT8 (CPU) / FP16 (GPU) | 下载 | CPU / GPU |

### 视觉 · [文档](docs/models/vision.md)

| 模型 | 说明 | 精度 | 获取 | 硬件 |
| --- | --- | --- | --- | --- |
| `yolo11n.onnx` | YOLO11n 目标检测，COCO 80 类 | FP32 | 下载 | CPU / GPU |
| `depth-anything-v2-metric-indoor-small-onnx` | Depth Anything V2 单目米制深度（室内） | FP32 | 导出 | CPU / GPU |

### OCR · [文档](docs/models/ocr.md)

| 模型 | 说明 | 精度 | 获取 | 硬件 |
| --- | --- | --- | --- | --- |
| `ppocrv5-mobile-onnx` | PP-OCRv5 文字行：场景文字、界面、名牌；也用于读取待审核图片中的文字 | FP32 | 下载 | CPU / GPU |
| `unlimited-ocr-onnx` | Unlimited-OCR 整页文档：版面坐标、HTML 表格 | FP16 + INT8 专家 | 导出 | 仅 GPU |

### 内容审核 · [文档](docs/models/moderation.md)

| 模型 | 说明 | 精度 | 获取 | 硬件 |
| --- | --- | --- | --- | --- |
| `qwen3guard-gen-0.6b-int4-onnx` | Qwen3Guard 文本审核，9 个类别（FP16 版 `qwen3guard-gen-0.6b-onnx` 可选） | INT4 | 导出 | GPU |
| `freepik-nsfw-image-detector-onnx` | 图片 NSFW 分级（Falconsai 版 `falconsai-nsfw-image-detection-onnx` 可选） | FP16 | 导出 | CPU / GPU |

### 实时语音 · [文档](docs/models/realtime-voice.md)

| 模型 | 说明 | 精度 | 获取 | 硬件 |
| --- | --- | --- | --- | --- |
| `voice-cascade` | WebSocket 语音对话：Silero VAD → 语音识别 → 对话 → 语音合成，支持唤醒词 | FP32 (VAD) | 下载（依赖上面的模型） | GPU |

“GPU”指 NVIDIA CUDA；标为 GPU 的模型按 GPU 设计与测量，CPU 只是回退（很慢），“仅 GPU”的模型拒绝在 CPU 上运行。“INT8 权重”指仅权重量化为 INT8，计算仍为浮点。模型规格按分类放在 [`configs/providers/<分类>/`](configs/providers)；配置字段、Provider 回退和如何覆盖模型配置见 [模型配置](docs/models/README.md)。

## 快速开始

```bash
cp .env.example .env    # 替换其中的占位 token

docker compose up -d --build                                                    # CPU
ORT_CUDA_VERSION=12 docker compose -f docker-compose-nvidia.yml up -d --build   # 或 NVIDIA GPU

curl --fail http://127.0.0.1:17890/health
```

NVIDIA 版需要 Linux x86_64、支持 CUDA 12 的驱动、NVIDIA Container Toolkit 和 Compose 2.30+；只有 worker 使用 GPU（CUDA 12.8 + cuDNN 镜像），controller 用 CPU 镜像。模型不打进镜像，放在 `./workdir/models`。

下载一个模型：

```bash
curl --fail-with-body http://127.0.0.1:17890/rpc/admin \
  -H 'content-type: application/json' -H "x-local-admin-token: $LOCAL_ADMIN_TOKEN" \
  --data '{"jsonrpc":"2.0","id":1,"method":"download_model","params":{"id":"sensevoice-small-onnx"}}'
```

端口暴露、源码构建与部署验证见 [部署指南](docs/getting-started.md)。

## 接入

| 方式 | 地址 | 鉴权 |
| --- | --- | --- |
| MCP 推理（推荐给 Agent） | `http://127.0.0.1:17892/mcp/infer` | 推理 token（若已配置） |
| MCP 管理 | `http://127.0.0.1:17892/mcp/admin` | 管理 token |
| OpenAI 兼容 API | `http://127.0.0.1:17890/v1` | 推理 token（若已配置；`GET /v1/models` 不需要） |
| JSON-RPC（旧版） | `http://127.0.0.1:17890/rpc/infer`、`/rpc/admin` | 推理 / 管理 token |

**Token**：管理调用必须提供 `LOCAL_ADMIN_TOKEN`；`LOCAL_MCP_INFER_TOKENS`（逗号分隔）非空时保护所有推理接口，为空则推理不鉴权。均以 `Authorization: Bearer <token>` 发送（JSON-RPC 与 OpenAI 接口也接受 `x-local-infer-token`，管理接口接受 `x-local-admin-token`）。Compose 把 17890、17891 发布到所有网卡、MCP 端口只发布到本机回环；`/v1/models`、`/health` 不受推理 token 保护，请再用网络访问控制保护 controller 端口。

### OpenAI 兼容与 HTTP 接口

只实现了下列接口，不是完整的 OpenAI API：

| 接口 | 用途 |
| --- | --- |
| `GET /v1/models` | 模型列表；除 OpenAI 字段外，每个模型还有 `name`、`task_kinds`（可执行的任务类型）、`enabled` 与 `downloaded`（文件已就位） |
| `POST /v1/chat/completions` | 对话，SSE 流式，工具调用 |
| `POST /v1/embeddings` | 文本向量 |
| `POST /rerank`、`/v1/rerank`、`/v2/rerank` | 重排（vLLM / Jina / Cohere 风格） |
| `POST /v1/audio/transcriptions` | 语音识别（JSON，传 worker 可读的文件路径，返回 `{"text"}`） |
| `POST /v1/audio/speech` | 语音合成（JSON，返回 WAV 文件引用） |
| `POST /v1/moderations` | 文本与图片审核（OpenAI 格式；图片为 base64 `data:` URL，做 NSFW 分级并审核图中文字；输入为多个 part 时另有 `parts`，给出每个 part 各自的判定） |
| `GET /v1/realtime` | 实时语音 WebSocket，见下 |
| `POST /v1/ocr/lines` | PP-OCRv5 文字行，请求体为图片 |
| `POST /v1/detect/objects` | YOLO 目标检测，请求体为图片 |
| `POST /v1/depth` | 米制深度网格，请求体为图片 |
| `GET /v1/inferences` | 各 worker 最近的推理记录与分阶段耗时（含实时语音每轮的 `voice.turn`） |

**实时语音** `/v1/realtime`：客户端持续推送 16 kHz 16-bit 单声道 PCM，服务端回送 24 kHz 音频与 JSON 事件。两种模式：

- **级联模式**（默认）：服务端完成 VAD → SenseVoice → Qwen3 → TTS 的整个对话，支持插话打断、群聊唤醒词和客户端提供的发言人标注；需要客户端处理的任务以 `tool.call`（`backend_task`）交给客户端，客户端用 `tool.result` / `note` 回传。
- **音频模式**（`"mode": "audio"`）：服务端只负责听和说，对话由客户端自己的模型完成（Codex 的本地语音用的就是此模式）。客户端收到带 `id` / `replaces` / `respond` 的 `input.transcript` 和 `state`，用 `response.delta` / `response.end` / `response.cancel` 流式驱动服务端说话，每个回复结束时收到 `response.done`（`spoken` 为实际播出的部分）；可开启 `moderation`，转写交给客户端前先审核。

完整事件与配置见 [实时语音协议](docs/realtime-voice.md)。

### MCP

推理目录 `/mcp/infer` 只有通用任务工具：`list_models`（已安装的模型、各自的任务类型，以及每种任务的文件、参数和输出）、`run_task`、`create_task`、`start_task`、`wait_task`、`get_task`、`sign_assets`。管理目录 `/mcp/admin` 负责下载、启用模型和查看集群。只把推理目录交给 Agent，不要给推理服务器配置管理 token：

```json
{
  "mcpServers": {
    "local-multimodal": {
      "type": "streamable-http",
      "url": "http://127.0.0.1:17892/mcp/infer",
      "headers": { "Authorization": "Bearer <推理 token>" }
    }
  }
}
```

`LOCAL_MCP_INFER_TOKENS` 为空时去掉 `headers`。旧版 JSON-RPC 的 `/rpc/infer` 另保留 `asr_transcribe`、`tts_synthesize` 等直接方法。全部接口、任务类型与大结果的 URL 返回见 [API 文档](docs/api.md)。

## GPU 与 ONNX Runtime

- 使用官方 ONNX Runtime 1.31.0，运行时动态加载（ort `load-dynamic`），编译时不需要它。Docker 镜像已内置：CPU 版，以及 GPU 版的 CUDA 12 构建（CUDA 12.8 + cuDNN 镜像）。
- 源码构建（Rust 1.92+）：GPU 版必须加 `--features cuda`，否则所有模型都在 CPU 上运行；再用 `fetch_onnxruntime` 把校验过 SHA-256 的 ONNX Runtime 放到二进制旁边：

  ```bash
  cargo build --release --bins --features cuda   # CPU 版去掉 --features cuda
  python -m scripts.local.fetch_onnxruntime      # CPU 版加 --cpu
  ```

- worker 优先加载 `ORT_DYLIB_PATH`，否则加载二进制旁的 ONNX Runtime。Windows 上需要在 PATH 中提供 CUDA 12.8+ 运行时（cudart、cuBLAS）与 cuDNN 9，且不要让它加载 `C:\Windows\System32\onnxruntime.dll`。
- `/health` 只说明服务在运行；模型实际使用的 provider 见 worker 日志 `lazy loading model`。

## 文档

- [部署指南](docs/getting-started.md)
- [API](docs/api.md)
- [模型配置](docs/models/README.md)：每个分类的获取、导出、参数与调用
- 深入：[实时语音协议](docs/realtime-voice.md) · [内容审核](docs/moderation.md) · [Qwen3-TTS 导出](docs/qwen3-tts.md) · [IndexTTS 延迟测试](docs/indextts-latency-benchmark.md) · [实现说明](docs/implementation-notes.md)
- 开发：[AGENTS.md](AGENTS.md)（约束、测试命令）

## 许可证

代码采用 [MIT](LICENSE)。各模型沿用其上游许可证，见对应模型文档。

## 致谢

模型来源、固定版本与许可证见各[模型文档](docs/models/README.md)，确切的文件、revision 与 SHA-256 以 [`configs/providers`](configs/providers) 为准（`local-registry` 的测试检查每个 Hugging Face 文件都固定到 commit、每个 URL 文件都带 SHA-256 与大小）。本项目移植或参考了以下上游项目：

- [modelcontextprotocol/rust-sdk](https://github.com/modelcontextprotocol/rust-sdk)：标准 MCP Rust SDK
- [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)、[pykeio/ort](https://github.com/pykeio/ort)：CPU / CUDA 推理运行时及其 Rust 绑定
- [microsoft/onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai)：Qwen3、Qwen3Guard 的 INT4 导出（`models.builder`）
- [modelscope/FunASR](https://github.com/modelscope/FunASR)、[FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)：SenseVoice 前处理、推理与 FSMN-VAD 管线
- [QwenAudio/Fun-ASR](https://github.com/QwenAudio/Fun-ASR)：Fun-ASR-Nano 的模型结构、提示词与推理流程
- [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)：YOLO 预处理、输出解码与 COCO 标签
- [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)：PP-OCRv5、DB 检测后处理与 CTC 解码
- [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR)：预处理、提示词、R-SWA 与防重复采样
- [index-tts/index-tts](https://github.com/index-tts/index-tts)、[DakeQQ/Text-to-Speech-TTS-ONNX](https://github.com/DakeQQ/Text-to-Speech-TTS-ONNX)：IndexTTS 及其 ONNX 导出与推理
- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)：Qwen3-TTS（`qwen-tts`），导出与校验的参考
- [PaddlePaddle/PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech)、[wetext-rs](https://github.com/SpenserCai/wetext-rs)（[wetext](https://github.com/pengzhendong/wetext)）、[jieba-rs](https://github.com/messense/jieba-rs)：中文 TTS 前端（见[实现说明](docs/implementation-notes.md#mandarin-text-frontend-zh-tts-frontend)）
- [snakers4/silero-vad](https://github.com/snakers4/silero-vad)：实时语音的 VAD
- [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)、[k2-fsa/icefall](https://github.com/k2-fsa/icefall)：唤醒词模型、关键词检测（流式 zipformer2、关键词加分束搜索、ContextGraph）与训练配方
- [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)：米制深度
