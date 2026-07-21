# local-multimodal-infra

> 将本机或内网机器的 CPU / NVIDIA GPU 转换为 Agent 可调用的多模态推理服务。

[English README](README.en.md)

## 定义

`local-multimodal-infra` 是一套**本地多模态基础设施**。它统一管理模型、文件、任务和运行时，并通过标准 MCP、legacy JSON-RPC 与部分 OpenAI-compatible API 向 Agent 或应用提供能力。

项目采用 controller / worker 架构：

| 组件 | 职责 | 默认地址 |
| --- | --- | --- |
| Controller | 模型与任务管理、文件上传、API、任务调度 | `http://127.0.0.1:17890` |
| Standard MCP Server | 隔离的管理与推理工具目录 | `http://127.0.0.1:17892/mcp/admin`、`http://127.0.0.1:17892/mcp/infer` |
| Worker | 加载 ONNX 模型并执行推理 | `http://127.0.0.1:17891` |

运行时数据统一存放在 `workdir`：

- `workdir/models`：模型工件；
- `workdir/data`：SQLite、上传文件、生成结果、日志与临时文件。

模型和输入素材不会打包进镜像。本地配置默认绑定 loopback；当前 Docker Compose 会把 controller `17890` 和 worker `17891` 发布到宿主机所有接口，而 MCP `17892` 仅发布到 loopback。部署前应根据使用范围调整端口绑定，并同时配置鉴权和网络访问控制。

## 提供的功能

### 推理能力

| 能力 | 默认模型 | 状态 | 主要输出 |
| --- | --- | --- | --- |
| 图片目标检测 | `yolo11n.onnx` | 默认启用 | 目标类别、置信度、边界框 |
| 语音识别 | `sensevoice-small-onnx` | 默认启用 | 文本、时间轴、语言、情绪、发言人 |
| 语音合成 | `indextts-1.5-onnx` | 实验性，默认禁用 | WAV 音频 |
| 文本向量 | `multilingual-e5-small-onnx` | 默认启用 | 384 维归一化向量 |
| 文本重排 | `mmarco-minilm-l12-onnx` | 默认启用 | 文档相关性排序与分数 |

所有模型均通过 ONNX Runtime 运行。模型配置表达 CUDA 优先、CPU 回退；实际 provider 仍取决于构建方式、运行环境和具体模型算子支持情况。

SenseVoice ASR 集成 FSMN-VAD 和 CAM++ 发言人识别，默认返回纯文本、约 10 秒粒度的 `timestamped_text`、`segments[].speaker` 和 `speakers[]`。可通过 `timestamps`、`timestamp_granularity_sec`、`token_timestamps`、`speaker_diarization` 调整或关闭这些结果。

### 接入接口

| 接口 | 用途 | 鉴权 |
| --- | --- | --- |
| `POST /rpc/admin` | legacy JSON-RPC 模型、节点与资产管理 | 必须配置 `LOCAL_ADMIN_TOKEN` |
| `POST /rpc/infer` | legacy JSON-RPC 推理与通用任务 | 配置 `LOCAL_MCP_INFER_TOKENS` 后启用鉴权 |
| `/mcp/admin` | 标准 MCP 管理工具 | 必须配置 `LOCAL_ADMIN_TOKEN` |
| `/mcp/infer` | 标准 MCP 推理工具 | 配置 `LOCAL_MCP_INFER_TOKENS` 后启用鉴权 |
| `/v1/models` | OpenAI-compatible 模型列表 | 无额外鉴权 |
| `/v1/audio/transcriptions` | OpenAI-compatible ASR | `LOCAL_MCP_INFER_TOKENS` |
| `/v1/audio/speech` | OpenAI-compatible TTS | `LOCAL_MCP_INFER_TOKENS` |
| `/v1/embeddings` | OpenAI-compatible Embeddings | `LOCAL_MCP_INFER_TOKENS` |
| `/rerank`、`/v1/rerank`、`/v2/rerank` | vLLM / Jina / Cohere 风格重排 | `LOCAL_MCP_INFER_TOKENS` |

Admin 和所有 MCP、RPC、OpenAI-compatible 推理接口接受 `Authorization: Bearer <token>`；legacy JSON-RPC 与 OpenAI-compatible 推理也接受 `x-local-infer-token`，Admin 接口接受 `x-local-admin-token`。

### 基础设施能力

- 模型配置、异步下载、SHA-256 校验、下载状态查询与并发下载去重；
- 模型启用、禁用、懒加载、并发限制和空闲卸载；
- controller / worker 调度，以及 CPU / CUDA provider 选择与回退；
- 签名上传 URL、任务输入、生成产物和本地资产管理；
- 标准 MCP direct tools 与“创建任务 → 上传文件 → 启动 → 等待结果”的通用任务流程；
- release、RPC、MCP 和真实模型调用链 smoke harness。

## 快速部署

### 1. 准备配置

需要 Docker 与 Docker Compose。模型不会随镜像发布，首次启动后需要下载到 `workdir/models`。

```bash
cp .env.example .env
```

编辑 `.env`，至少替换以下占位值：

```dotenv
LOCAL_WORKER_REGISTRATION_TOKEN=replace-with-a-long-random-worker-registration-token
LOCAL_UPLOAD_SIGNING_SECRET=replace-with-a-long-random-upload-signing-secret
LOCAL_ADMIN_TOKEN=replace-with-a-long-random-admin-token
LOCAL_MCP_INFER_TOKENS=
LOCAL_PUBLIC_BASE_URL=http://127.0.0.1:17890
```

`LOCAL_MCP_INFER_TOKENS` 为空时推理接口不鉴权；设置为逗号分隔的 token 后，MCP、JSON-RPC 和 OpenAI-compatible 推理接口均要求其中任意一个 token。

如服务仅供本机使用，建议把 Compose 中的 `17890:17890` 和 `17891:17891` 改为 `127.0.0.1:17890:17890`、`127.0.0.1:17891:17891`。`/v1/models`、健康检查和部分资产路由不属于推理鉴权范围，仍应使用网络访问控制保护 controller 端口。

### 2. 启动 CPU 服务

```bash
docker compose up -d --build
docker compose ps
curl --fail http://127.0.0.1:17890/health
```

### 3. 启动 NVIDIA CUDA 服务

CUDA 部署需要 NVIDIA 驱动、NVIDIA Container Toolkit 和可用的 `nvidia-smi`。当前镜像使用 CUDA 12 的 ONNX Runtime 包，支持 Linux x86_64 容器：

```bash
nvidia-smi
ORT_CUDA_VERSION=12 docker compose -f docker-compose-nvidia.yml up -d --build
docker compose -f docker-compose-nvidia.yml exec worker nvidia-smi
```

CUDA Compose 只向 worker 分配 GPU；controller 继续运行 CPU 镜像。`/health` 只表示服务可用，不能证明某个模型已经在 CUDA 上完成推理。

### 4. 下载模型

先查看配置中的模型及下载状态：

```bash
curl --fail-with-body http://127.0.0.1:17890/rpc/admin \
  -H 'content-type: application/json' \
  -H 'x-local-admin-token: replace-with-your-admin-token' \
  --data '{"jsonrpc":"2.0","id":"models","method":"list_models","params":{}}'
```

提交异步下载任务，并查询逐文件状态：

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

其他默认模型 ID：

- `yolo11n.onnx`
- `multilingual-e5-small-onnx`
- `mmarco-minilm-l12-onnx`
- `indextts-1.5-onnx`（实验性，下载后还需调用 `enable_model`）

### 5. Agent 如何使用

推荐只向 Agent 配置推理 MCP：

- 推理：`http://127.0.0.1:17892/mcp/infer`
- 管理：`http://127.0.0.1:17892/mcp/admin`，仅在 Agent 确实需要下载、启用或禁用模型时单独配置

下面是常见的 Streamable HTTP MCP 配置形状；不同 Agent 的配置文件名或字段名可能略有差异：

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

如果 `LOCAL_MCP_INFER_TOKENS` 为空，可以删除 `headers`。需要管理工具时，另建一个 MCP Server 配置，将 URL 改为 `/mcp/admin`，并使用 `LOCAL_ADMIN_TOKEN`，不要与推理 token 混用。

接入后，Agent 可以直接调用 `object_detect`、`asr_transcribe`、`tts_synthesize`、`text_embed` 和 `text_rerank`。对于 Agent 无法直接访问的图片或音频，使用 `create_task` → 上传到返回的签名 URL → `start_task` → `wait_task`，不需要与 worker 共享宿主机文件路径。

需要与 OpenAI 风格客户端集成时，将 base URL 指向 `http://127.0.0.1:17890/v1`，并把 `LOCAL_MCP_INFER_TOKENS` 中的任意一个 token 作为 API key / Bearer token。该接口只实现上表列出的本地能力，不是完整 OpenAI API。

### 6. 验证部署

仓库内置 smoke harness，会负责构建、启动、等待健康状态、执行真实请求并清理进程：

```bash
python -m scripts.local.smoke --tests rpc \
  --workdir ./workdir --model-dir ./workdir/models

python -m scripts.local.smoke --tests mcp \
  --workdir ./workdir --model-dir ./workdir/models
```

`mcp` 测试需要当前 Python 环境安装官方 `mcp` SDK。release 验证可先运行 `cargo build --release --bins`，再为 smoke harness 增加 `--skip-build --release`。

## 参考资料

### 模型仓库

| 用途 | 仓库 | 当前固定版本 |
| --- | --- | --- |
| YOLO11n ONNX | [aaurelions/yolo11n.onnx](https://huggingface.co/aaurelions/yolo11n.onnx) | `f46d9b72aa9a0f02bc00484446e2310b1a549bce` |
| SenseVoiceSmall ONNX | [haixuantao/SenseVoiceSmall-onnx](https://huggingface.co/haixuantao/SenseVoiceSmall-onnx) | `c4c8747214bed7ebbf2557e0412c19efa540023c` |
| FSMN-VAD ONNX | [funasr/fsmn-vad-onnx](https://huggingface.co/funasr/fsmn-vad-onnx) | `f6e9fbb4cefa7397216c763f21307993f147f585` |
| FSMN-VAD 配置 | [MoYoYoTech/Translator](https://huggingface.co/MoYoYoTech/Translator) | `58fbad4088820ed1253955c8faf1444cd0b2dc69` |
| CAM++ Speaker | [welcomyou/campplus-3dspeaker-200k-onnx](https://huggingface.co/welcomyou/campplus-3dspeaker-200k-onnx) | `6265ff7af2a104d745b4389026ed9815c6c1c6ff` |
| IndexTTS 1.5 ONNX | [ModaLeap/indextts-1.5-onnx](https://huggingface.co/ModaLeap/indextts-1.5-onnx) | 配置暂未固定 revision |
| multilingual-e5-small | [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | `614241f622f53c4eeff9890bdc4f31cfecc418b3` |
| mMARCO MiniLM reranker | [cross-encoder/mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) | `1427fd652930e4ba29e8149678df786c240d8825` |

实际下载文件、revision 与 SHA-256 以 [`configs/models.d`](configs/models.d) 中的配置为准。

### 参考代码仓库

- [modelscope/FunASR](https://github.com/modelscope/FunASR)：SenseVoice ONNX 前处理、推理与 FSMN-VAD 管线参考；
- [FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)：SenseVoice 模型与官方实现；
- [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)：YOLO 预处理、输出解码与 COCO 标签来源；
- [index-tts/index-tts](https://github.com/index-tts/index-tts)：IndexTTS 官方实现；
- [DakeQQ/Text-to-Speech-TTS-ONNX](https://github.com/DakeQQ/Text-to-Speech-TTS-ONNX)：IndexTTS ONNX 导出与推理参考；
- [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)：CPU / CUDA 推理运行时；
- [modelcontextprotocol/rust-sdk](https://github.com/modelcontextprotocol/rust-sdk)：标准 MCP Rust SDK。

### 项目文档

- [实现说明](docs/implementation-notes.md)
- [开发与验证约束](AGENTS.md)
- [CPU Compose](docker-compose.yml)
- [NVIDIA CUDA Compose](docker-compose-nvidia.yml)
- [环境变量示例](.env.example)
