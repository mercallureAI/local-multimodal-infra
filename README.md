# local-multimodal-infra

> 将本机及内网的 CPU / NVIDIA GPU 算力封装为可供 Agent 调用的多模态推理服务。

[English](README.en.md)

语音识别、语音合成、对话、向量与重排、视觉、OCR、内容审核、实时语音，全部在本地通过 ONNX Runtime 运行，以标准 MCP 和 OpenAI 兼容 API 提供给 Agent 与应用。

- **一套服务，多种接入**：标准 MCP（推理 / 管理分离）、OpenAI 兼容 API、JSON-RPC。
- **controller / worker 架构**：controller 管理模型、文件与任务，worker 加载模型推理，可分布在多台机器。
- **模型按需加载**：懒加载、并发限制、空闲卸载；CUDA 优先，不可用时自动回退 CPU。
- **可复现的模型**：下载固定到 commit 并校验 SHA-256；没有发布包的模型提供导出脚本。
- **纯 Rust**：运行时不依赖 Python。

## 支持的模型

获取方式：**下载** = 管理接口一键下载；**导出** = 没有发布的 ONNX 包，按文档从源模型本地导出。

### 语音识别 · [文档](docs/models/speech-recognition.md)

| 模型 | 说明 | 精度 | 获取 | 硬件 |
| --- | --- | --- | --- | --- |
| `sensevoice-small-onnx` | SenseVoiceSmall + FSMN-VAD + CAM++：文本、时间轴、语言、情绪、发言人 | INT8 | 下载 | CPU / GPU |
| `sensevoice-small-int8-gpu-onnx` | 同上，全程在 GPU 上 + CUDA graph，1.5 秒语音约 16 ms | INT8 权重 | 导出 | GPU |

### 语音合成 · [文档](docs/models/speech-synthesis.md)

| 模型 | 说明 | 精度 | 获取 | 硬件 |
| --- | --- | --- | --- | --- |
| `indextts-1.5-onnx` | IndexTTS 1.5，参考音频克隆声音 | FP32 | 下载 | CPU / GPU |
| `indextts-2.5-onnx` | IndexTTS 2.5，可控制情绪 | FP16 | 下载 | GPU |
| `qwen3-tts-0.6b-onnx` | Qwen3-TTS 0.6B，流式输出，首段音频约 50 ms | INT8 权重 + FP16 | 导出 | GPU |

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
| `ppocrv5-mobile-onnx` | PP-OCRv5 文字行：场景文字、界面、名牌 | FP32 | 下载 | CPU / GPU |
| `unlimited-ocr-onnx` | Unlimited-OCR 整页文档：版面坐标、HTML 表格 | FP16 + INT8 专家 | 导出 | 仅 GPU |

### 内容审核 · [文档](docs/models/moderation.md)

| 模型 | 说明 | 精度 | 获取 | 硬件 |
| --- | --- | --- | --- | --- |
| `qwen3guard-gen-0.6b-int4-onnx` | Qwen3Guard 文本审核（FP16 版可选） | INT4 | 导出 | GPU |
| `freepik-nsfw-image-detector-onnx` | 图片 NSFW 分级（Falconsai 版可选） | FP16 | 导出 | CPU / GPU |

### 实时语音 · [文档](docs/models/realtime-voice.md)

| 模型 | 说明 | 精度 | 获取 | 硬件 |
| --- | --- | --- | --- | --- |
| `voice-cascade` | WebSocket 语音对话：VAD → 语音识别 → 对话 → 语音合成，支持唤醒词 | FP32 (VAD) | 下载（依赖上面的模型） | GPU |

“GPU”指 NVIDIA CUDA；“INT8 权重”指仅权重量化为 INT8，计算仍为浮点。配置字段、Provider 回退和如何覆盖模型配置见 [模型配置](docs/models/README.md)。

## 快速开始

```bash
cp .env.example .env    # 替换其中的占位 token

docker compose up -d --build                                                    # CPU
ORT_CUDA_VERSION=12 docker compose -f docker-compose-nvidia.yml up -d --build   # 或 NVIDIA GPU

curl --fail http://127.0.0.1:17890/health
```

下载一个模型：

```bash
curl --fail-with-body http://127.0.0.1:17890/rpc/admin \
  -H 'content-type: application/json' -H "x-local-admin-token: $LOCAL_ADMIN_TOKEN" \
  --data '{"jsonrpc":"2.0","id":1,"method":"download_model","params":{"id":"sensevoice-small-onnx"}}'
```

端口暴露、源码构建与部署验证见 [部署指南](docs/getting-started.md)。

## 接入

| 方式 | 地址 |
| --- | --- |
| MCP（推荐给 Agent） | `http://127.0.0.1:17892/mcp/infer`（管理：`/mcp/admin`） |
| OpenAI 兼容 API | `http://127.0.0.1:17890/v1` |
| JSON-RPC | `http://127.0.0.1:17890/rpc/infer`、`/rpc/admin` |

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

全部接口、鉴权与大结果的 URL 返回见 [API 文档](docs/api.md)。

## 文档

- [部署指南](docs/getting-started.md)
- [API](docs/api.md)
- [模型配置](docs/models/README.md)：每个分类的获取、导出、参数与调用
- 深入：[实时语音协议](docs/realtime-voice.md) · [内容审核](docs/moderation.md) · [Qwen3-TTS 导出](docs/qwen3-tts.md) · [实现说明](docs/implementation-notes.md)
- 开发：[AGENTS.md](AGENTS.md)（约束、测试命令）

## 许可证

代码采用 [MIT](LICENSE)。各模型沿用其上游许可证，见对应模型文档。

## 致谢

模型来源、固定版本与许可证见各[模型文档](docs/models/README.md)，确切的文件、revision 与 SHA-256 以 [`configs/providers`](configs/providers) 为准。本项目移植或参考了以下上游项目：

- [modelcontextprotocol/rust-sdk](https://github.com/modelcontextprotocol/rust-sdk)：标准 MCP Rust SDK
- [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)、[pykeio/ort](https://github.com/pykeio/ort)：CPU / CUDA 推理运行时及其 Rust 绑定
- [microsoft/onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai)：Qwen3、Qwen3Guard 的 INT4 导出（`models.builder`）
- [modelscope/FunASR](https://github.com/modelscope/FunASR)、[FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)：SenseVoice 前处理、推理与 FSMN-VAD 管线
- [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)：YOLO 预处理、输出解码与 COCO 标签
- [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)：PP-OCRv5、DB 检测后处理与 CTC 解码
- [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR)：预处理、提示词、R-SWA 与防重复采样
- [index-tts/index-tts](https://github.com/index-tts/index-tts)、[DakeQQ/Text-to-Speech-TTS-ONNX](https://github.com/DakeQQ/Text-to-Speech-TTS-ONNX)：IndexTTS 及其 ONNX 导出与推理
- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)：Qwen3-TTS 导出与对齐的参考
- [PaddlePaddle/PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech)、[wetext-rs](https://github.com/SpenserCai/wetext-rs)（[wetext](https://github.com/pengzhendong/wetext)）、[jieba-rs](https://github.com/messense/jieba-rs)：中文 TTS 前端
- [snakers4/silero-vad](https://github.com/snakers4/silero-vad)：实时语音的 VAD
- [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)、[k2-fsa/icefall](https://github.com/k2-fsa/icefall)：唤醒词模型、关键词检测（流式 zipformer2、关键词加分束搜索、ContextGraph）与训练配方
- [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)：米制深度
