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
| 单目米制深度 | `depth-anything-v2-metric-indoor-small-onnx` | 本地导出后启用 | 每格平均深度（米）的网格 |
| 文档 OCR | `unlimited-ocr-onnx` | 本地导出后启用（int8 专家，需 NVIDIA GPU） | 页面文本（Markdown/HTML 表格），带版面类别与坐标 |
| 画面文字行 OCR | `ppocrv5-mobile-onnx` | 默认启用 | 每行文字、置信度与像素框（场景文字、界面、名牌） |
| 语音识别 | `sensevoice-small-onnx` | 默认启用 | 文本、时间轴、语言、情绪、发言人 |
| 语音识别（低延迟） | `sensevoice-small-fp16-onnx` | 本地导出后启用 | 同上；float16 图，CUDA 上每句约 40–90 ms |
| 语音合成 | `indextts-1.5-onnx` | 默认启用 | WAV 音频 |
| 语音合成 | `indextts-2.5-onnx` | 默认启用（FP16，建议 NVIDIA GPU） | WAV 音频，支持情绪控制 |
| 语音合成 | `qwen3-tts-0.6b-onnx` | 本地导出后启用（INT8 权重，建议 NVIDIA GPU） | 流式 24 kHz 音频，3 秒参考音频克隆声音，可边接收文字边合成 |
| 文本向量 | `multilingual-e5-small-onnx` | 默认启用 | 384 维归一化向量 |
| 文本重排 | `mmarco-minilm-l12-onnx` | 默认启用 | 文档相关性排序与分数 |
| 内容审核（文本） | `qwen3guard-gen-0.6b-int4-onnx` | 本地导出后启用（INT4；FP16 版 `qwen3guard-gen-0.6b-onnx` 可选） | 安全 / 有争议 / 不安全的概率与类别（暴力、违法、色情、个人信息、自残、政治敏感等） |
| 内容审核（图片） | `freepik-nsfw-image-detector-onnx` | 本地导出后启用（FP16） | NSFW 概率与分级（neutral / low / medium / high） |
| 对话补全 | `qwen3-4b-instruct-2507-int4-onnx` | 本地导出后启用 | 流式文本与工具调用（Qwen3 模板，KV 前缀复用） |
| 实时语音 | `voice-cascade` | 默认启用，依赖 ASR、对话与 TTS 模型 | `/v1/realtime` WebSocket 语音对话（Silero VAD + SenseVoice + Qwen3 + Qwen3-TTS，TTS 也可换成 IndexTTS，见 `docs/realtime-voice.md`） |
| 唤醒词 | 实时语音内置（sherpa-onnx KWS zipformer zh-en 3M） | 模型用 `scripts.local.fetch_kws_model` 放入后启用 | 群聊会话中检出机器人名字等唤醒词（`input.wake`），只有叫到它的话要回应，其余作为上下文 |

所有模型均通过 ONNX Runtime 运行（运行时加载官方 ONNX Runtime 1.30，见 `docs/implementation-notes.md`）。模型配置表达 CUDA 优先、CPU 回退；实际 provider 仍取决于构建方式、运行环境和具体模型算子支持情况。

文档 OCR 使用 [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR)（DeepEncoder + DeepSeek-V2 MoE，R-SWA 注意力），每次请求识别一页图片（base 模式，缩放填充到 1024×1024）。结果为整页文本，每个版面块前带 `<|det|>类别 [x1, y1, x2, y2]<|/det|>` 标签，坐标按 0–999 归一化，表格为 HTML。可通过 MCP / legacy RPC 的 `ocr_recognize`（传 `image` FileRef 或 `image_path`）或通用任务 `ocr.recognize`（上传 `image`）调用。

画面文字行（场景文字、界面标签、名牌等）使用 PaddleOCR 的 PP-OCRv5 mobile 检测与识别模型（中、英、日等文字同一个模型，两个 ONNX 合计约 21 MB），返回每行文字及其像素框，按从上到下、从左到右排列。逐帧调用可直接 `POST /v1/ocr/lines[?model=ppocrv5-mobile-onnx]`，请求体就是图片（PNG/JPEG/BMP），立即返回 `{"lines": [{"text", "confidence", "bbox"}]}`，受推理 token 保护；也可用通用任务 `ocr.lines`（上传 `image`）。目标检测同样有逐帧接口 `POST /v1/detect/objects[?model=yolo11n.onnx]`，返回 `{"objects": [{"label", "confidence", "bbox"}]}`。

内容审核用于把内容发给云端 LLM API 之前，检查其中是否有违反平台政策的内容：`POST /v1/moderations` 兼容 OpenAI 的审核接口，`input` 可以是字符串、字符串数组，或 `text` / `image_url`（base64 `data:` URL，不拉取远程图片）组成的多模态数组。文本由 Qwen3Guard-Gen-0.6B（INT4）判定（一次 prefill 读出安全 / 有争议 / 不安全的概率，非安全时再解码类别）；图片先由 NSFW 分类器判定，未被拦下的再用 PP-OCRv5 读出图中文字一并审核。默认 `1 - p(safe) > 0.9` 或 NSFW 概率 > 0.5 时 `flagged`，可用 `threshold` / `nsfw_threshold` 调整，`nsfw_labels` 选择哪些图片档位算违规（如只算 `medium`、`high`），`categories` 选择哪些文字类别拦截。多模态数组的结果另有 `parts`，按输入顺序给出每一项的判定，便于调用方只去掉违规的部分。ToxicChat 上召回 0.81、误报 2.8%，600 段本地技术文档与代码 0 误报；三个模型合计显存峰值约 1.6 GB，RTX 4090 上一张图端到端约 120 ms。模型导出、接口细节与评测数据见 [`docs/moderation.md`](docs/moderation.md)。

单目米制深度使用 Depth Anything V2 Metric Indoor Small（ViT-S，室内 Hypersim 微调，最大 20 m，Apache-2.0），由 `python -m scripts.local.depth_anything_export --size 308x546 --output-dir <models>/depth-anything-v2-metric-indoor-small-onnx` 从固定 revision 导出为固定输入尺寸的 ONNX（约 99 MB）；图片被拉伸到该尺寸，因此按 16:9 画面（如 1280×720 游戏画面）导出，其他宽高比的图片会变形、深度有偏差，需按其宽高比另行导出（`--size`）。逐帧接口 `POST /v1/depth[?model=&cols=&rows=]`（默认 64×36）返回 `{"cols", "rows", "max_depth", "depth": [米，自上而下逐行]}`，每格是该区域的平均深度；通用任务 `depth.estimate`（上传 `image`，`params.cols/rows`）同样可用。RTX 4090 上一帧 1280×720 约 30 ms（含解码）。

最近的推理记录：`GET /v1/inferences[?limit=100&kind=&exclude=]`（受推理 token 保护）返回所有 worker 内存里最近的推理（每个 worker 保留 1000 条，默认取最新 100 条），每条有任务种类、模型、开始与结束时间（Unix 毫秒）、总耗时和各阶段耗时（排队、加载、执行、首个输出）。实时语音每次回复另记一条 `voice.turn`：从说话结束到回复首段音频，分为 `vad`（等静音判定结束）、`asr`、`gate`（等唤醒词判断）、`llm`（到回复首段文字）、`tts`（到首段音频）。`kind` 只取这些种类，`exclude` 去掉这些种类或模型（逗号分隔），如 `exclude=ocr.lines` 去掉高频的 OCR。

低延迟语音识别 `sensevoice-small-fp16-onnx` 与 `sensevoice-small-onnx` 是同一个 SenseVoiceSmall：后者的 int8 图里 281 个 `DynamicQuantizeLinear` 在 CUDA provider 下回落到 CPU，每层在 CPU 与 GPU 间往返，一句话约 200–350 ms；float16 图全程在 GPU 上，约 40–90 ms，转写相同。由 `python -m scripts.local.sensevoice_fp16_export --base-dir <models>/sensevoice-small-onnx --output-dir <models>/sensevoice-small-fp16-onnx` 从固定 revision 导出（约 470 MB），其余文件取自 `sensevoice-small-onnx`；元数据 `asr_model_file` 指定 `asr/` 中的图。实时语音级联在 `voice-cascade.yaml` 的 `asr_model` 中选用；默认即为它，未导出时会话启动时自动改用 `asr_fallback_model`（`sensevoice-small-onnx`）。

实时语音的群聊会话用唤醒词决定哪些话要回应：k2-fsa 开放词表的中英混合 zipformer transducer 唤醒词模型（`sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20`，Apache-2.0，3.3M 参数，fp32 chunk-16 约 13 MB），由 `local-adapter-kws-zipformer` 移植 sherpa-onnx 的检测流程（Kaldi fbank、320 ms 一块的流式编码器、沿唤醒词加分的束搜索）在 CPU 上运行，每路约占 2% 的核，词说完后 0.2–0.6 s 检出，输入先做音量归一；转写中某句以唤醒词开头或结尾的话也算叫到（ASR 能听出检测器漏掉的名字）。唤醒词直接写文字（机器人名字、别名、`wake_words`），中文按拼音、英文按模型词典、字母与数字各有读法；用 `python -m scripts.local.fetch_kws_model` 下载并校验发布包，放到 `<models>/voice-cascade/kws`。详见 `docs/realtime-voice.md` 的 Wake words。

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
| `/v1/chat/completions` | OpenAI-compatible Chat（`stream: true` 为 SSE） | `LOCAL_MCP_INFER_TOKENS` |
| `/v1/realtime` | 实时语音 WebSocket（VAD→ASR→对话→TTS，见 `docs/realtime-voice.md`） | `LOCAL_MCP_INFER_TOKENS` |
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
- `ppocrv5-mobile-onnx`
- `multilingual-e5-small-onnx`
- `mmarco-minilm-l12-onnx`
- `indextts-1.5-onnx`
- `indextts-2.5-onnx`（FP16，约 2.8 GB）
- `voice-cascade`（只下载 Silero VAD；对话所用的 ASR、对话与 TTS 模型需各自下载或导出，默认 ASR `sensevoice-small-fp16-onnx` 需本地导出，见上文；群聊唤醒词模型用 `python -m scripts.local.fetch_kws_model` 放到 `voice-cascade/kws`）

`depth-anything-v2-metric-indoor-small-onnx` 与 `sensevoice-small-fp16-onnx` 没有发布的包，按上文的导出命令本地导出。

`qwen3guard-gen-0.6b-int4-onnx`、`qwen3guard-gen-0.6b-onnx`、`freepik-nsfw-image-detector-onnx` 与 `falconsai-nsfw-image-detection-onnx` 没有发布的包，按 [`configs/providers/moderation`](configs/providers/moderation) 中的命令本地导出（NSFW 分类器用 `python -m scripts.local.nsfw_export`）。

`qwen3-4b-instruct-2507-int4-onnx` 没有发布的 ONNX 包，需按 [`configs/providers/chat/qwen3-chat.yaml`](configs/providers/chat/qwen3-chat.yaml) 中的命令从固定 revision 本地导出到 `workdir/models/qwen3-4b-instruct-2507-int4-onnx`。

`qwen3-tts-0.6b-onnx`（实时语音默认的 TTS）同样没有发布的包，需从固定 revision 本地导出（导出环境、图结构、INT8 音质对比与延迟数据见 [`docs/qwen3-tts.md`](docs/qwen3-tts.md)）：

```bash
hf download Qwen/Qwen3-TTS-12Hz-0.6B-Base --revision 5d83992436eae1d760afd27aff78a71d676296fc --local-dir <src>
python -m scripts.local.qwen3_tts_export --source-model-dir <src> \
    --output-dir workdir/models/qwen3-tts-0.6b-onnx --builder-python <装有 onnxruntime-genai 的 python>
```

在 RTX 4090 上每帧（80 ms 语音）约 5.6 ms，一句话的首段音频约 50 ms 后开始播放；实时语音从说话人停下到听到回答约 0.9–1.0 s（其中 0.6 s 是 VAD 判断说完所需的静音）。

`unlimited-ocr-onnx` 同样没有发布的包，需从 PyTorch 检查点本地导出：

```bash
hf download baidu/Unlimited-OCR --revision 07dea832e22aefee32ad281d4b80551282e1c168 --local-dir <src>
python -m scripts.local.unlimited_ocr_export export --source <src> --out workdir/models/unlimited-ocr-onnx
```

导出环境为 Python 3.11，版本与上游一致：`torch==2.10.0`（CPU 版即可）、`torchvision==0.25.0`、`transformers==4.57.1`，另需 `onnx onnxruntime-gpu==1.30.0 einops addict easydict safetensors pillow matplotlib`；`parity` 子命令可与 PyTorch 模型逐 token 对比。默认导出的 int8 专家只能在 CUDA 上运行（worker 需以 `--features cuda` 构建）。在 RTX 4090 上单页约 145 tokens/s，显存约 6.6 GB；论文 14 页 PDF 平均 6.8 秒/页，同卡上官方 transformers 实现为 35 秒/页。

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

接入后，Agent 可以直接调用 `object_detect`、`ocr_recognize`、`asr_transcribe`、`tts_synthesize`、`text_embed` 和 `text_rerank`。对于 Agent 无法直接访问的图片或音频，使用 `create_task` → 上传到返回的签名 URL → `start_task` → `wait_task`，不需要与 worker 共享宿主机文件路径。

所有会返回推理结果的 MCP tools 都接受 `with_url_result`：

- `auto`（默认）：结果的 UTF-8 序列化文本不超过 1000 字节时保持原有结构内联返回；超过时返回 UTF-8 安全边界内的前 1000 字节预览和下载 URL。
- `on`：始终返回最多 1000 UTF-8 字节的预览并生成下载 URL；短结果的预览包含全文。
- `off`：始终通过 MCP 接口内联返回完整的原有结果。

URL 模式的完整结果以 `text/plain; charset=utf-8` 的 `.txt` 产物保存，响应包含 `preview`、`truncated`、`download_url`、`artifact_uri`、`size_bytes`、`sha256` 和 `expires_at`。这些文本产物由产物中心统一管理：默认 24 小时后自动清理，相同内容复用同一产物并续期；签名下载 URL 默认有效 10 分钟，再次调用可获得新 URL。

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

`rpc` 与 `mcp` 都包含 OCR（`--tests ocr` 可单独运行）：没有本地导出的 `unlimited-ocr-onnx` 或 worker 没有可用 CUDA 时会标记为跳过。要实际覆盖 OCR，先用 `cargo build --release --bins -p local-cli --features cuda` 构建，再运行 `python -m scripts.local.smoke --skip-build --release --tests ocr --workdir ./workdir --model-dir ./workdir/models --request-timeout 300`。

## 参考资料

### 模型仓库

| 用途 | 仓库 | 当前固定版本 |
| --- | --- | --- |
| YOLO11n ONNX | [aaurelions/yolo11n.onnx](https://huggingface.co/aaurelions/yolo11n.onnx) | `f46d9b72aa9a0f02bc00484446e2310b1a549bce` |
| YOLO COCO 标签 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics/blob/eba96641b5cea142e21641909d6400fef7134244/ultralytics/cfg/datasets/coco.yaml) `coco.yaml` | `eba96641b5cea142e21641909d6400fef7134244` |
| SenseVoiceSmall ONNX | [haixuantao/SenseVoiceSmall-onnx](https://huggingface.co/haixuantao/SenseVoiceSmall-onnx) | `c4c8747214bed7ebbf2557e0412c19efa540023c` |
| FSMN-VAD ONNX | [funasr/fsmn-vad-onnx](https://huggingface.co/funasr/fsmn-vad-onnx) | `f6e9fbb4cefa7397216c763f21307993f147f585` |
| FSMN-VAD 配置 | [MoYoYoTech/Translator](https://huggingface.co/MoYoYoTech/Translator) | `58fbad4088820ed1253955c8faf1444cd0b2dc69` |
| CAM++ Speaker | [welcomyou/campplus-3dspeaker-200k-onnx](https://huggingface.co/welcomyou/campplus-3dspeaker-200k-onnx) | `6265ff7af2a104d745b4389026ed9815c6c1c6ff` |
| IndexTTS 1.5 ONNX | [ModaLeap/indextts-1.5-onnx](https://huggingface.co/ModaLeap/indextts-1.5-onnx) | `3f1a422cd97a0b7dbb9b6ad4698dc0fde66796d1` |
| IndexTTS 2.5 ONNX FP16 | [ModaLeap/indextts-2.5-onnx](https://huggingface.co/ModaLeap/indextts-2.5-onnx) | `fd246cb6c2cf046113cd3400565edf681ac1b68b` |
| IndexTTS 中文前端（WeText + g2pW） | [ModaLeap/zh-tts-frontend](https://huggingface.co/ModaLeap/zh-tts-frontend) | `ba6b85aeb17ebc58d2d3d73121096f9495ee710e` |
| multilingual-e5-small | [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | `614241f622f53c4eeff9890bdc4f31cfecc418b3` |
| mMARCO MiniLM reranker | [cross-encoder/mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) | `1427fd652930e4ba29e8149678df786c240d8825` |
| PP-OCRv5 mobile 检测 / 识别（ONNX） | [ilaylow/PP_OCRv5_mobile_onnx](https://huggingface.co/ilaylow/PP_OCRv5_mobile_onnx)，字典 [PaddleOCR `ppocrv5_dict.txt`](https://github.com/PaddlePaddle/PaddleOCR/blob/a38c087bcb2579f9ccc2068aea02ec893b1c2311/ppocr/utils/dict/ppocrv5_dict.txt) | `f97b337b3ac256f9dffcac5fc53955082d919d58` / `a38c087bcb2579f9ccc2068aea02ec893b1c2311` |
| Unlimited-OCR（本地导出 ONNX 的源模型） | [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) | `07dea832e22aefee32ad281d4b80551282e1c168` |
| Qwen3-TTS-12Hz-0.6B-Base（本地导出 ONNX 的源模型） | [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) | `5d83992436eae1d760afd27aff78a71d676296fc` |
| Qwen3-4B-Instruct-2507（本地导出 INT4 的源模型） | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Qwen3Guard-Gen-0.6B（本地导出 FP16 的源模型） | [Qwen/Qwen3Guard-Gen-0.6B](https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B) | `fada3b2f655b89601929198343c94cd2f64d93cc` |
| Freepik NSFW image detector（本地导出 ONNX 的源模型） | [Freepik/nsfw_image_detector](https://huggingface.co/Freepik/nsfw_image_detector) | `15b85477e4fd2000db76ae9aae0f89a72f95e2e3` |
| Falconsai NSFW image detection（本地导出 ONNX 的源模型） | [Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection) | `96cb0d0342c7afb80cab76ecc58b265fa44da256` |
| Depth Anything V2 Metric Indoor Small（本地导出 ONNX 的源模型） | [depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf) | `8078d68a9c75a972131914f6afd0c1723be0da7f` |
| SenseVoiceSmall（本地导出 float16 ONNX 的源模型） | [FunAudioLLM/SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) | `3847d57b6bdf2dd8875cb1508d2af43d80a16bf7` |
| Silero VAD v6.2.3 | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | `5cd7945676eb32225748052e2e6a0580e4686a08` |
| sherpa-onnx KWS zipformer zh-en 3M（唤醒词） | [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models) | `sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20` |

实际下载文件、revision 与 SHA-256 以 [`configs/providers`](configs/providers)（按分类分目录）中的配置为准：Hugging Face 工件固定到 commit，URL 工件附带 SHA-256，`local-registry` 的测试会检查这两点。IndexTTS 1.5 与 2.5 的配置都会把中文前端（`ModaLeap/zh-tts-frontend`，约 177 MB，各文件许可见其 `NOTICE`）下载到各自模型目录下的 `zh-tts-frontend/`；也可用 `scripts/local/zh_frontend_export.py` 本地重新生成。

### 参考代码仓库

- [modelscope/FunASR](https://github.com/modelscope/FunASR)：SenseVoice ONNX 前处理、推理与 FSMN-VAD 管线参考；
- [FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)：SenseVoice 模型与官方实现；
- [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)：YOLO 预处理、输出解码与 COCO 标签来源；
- [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)：PP-OCRv5 模型、DB 检测后处理与 CTC 解码的参考；
- [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR)：Unlimited-OCR 模型与官方实现（预处理、提示词、R-SWA 与防重复采样）；
- [index-tts/index-tts](https://github.com/index-tts/index-tts)：IndexTTS 官方实现；
- [DakeQQ/Text-to-Speech-TTS-ONNX](https://github.com/DakeQQ/Text-to-Speech-TTS-ONNX)：IndexTTS ONNX 导出与推理参考；
- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)：Qwen3-TTS 官方实现（`qwen-tts`），导出与对齐的参考；
- [snakers4/silero-vad](https://github.com/snakers4/silero-vad)：实时语音的 Silero VAD 模型；
- [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)：唤醒词模型的发布包，关键词检测（流式 zipformer2、关键词加分束搜索、ContextGraph）的移植参考；
- [k2-fsa/icefall](https://github.com/k2-fsa/icefall)：唤醒词模型的训练配方与评测结果；
- [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)：Depth Anything V2 米制深度模型与官方实现；
- [microsoft/onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai)：Qwen3 INT4 ONNX 导出（`models.builder`）；
- [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)：CPU / CUDA 推理运行时；
- [modelcontextprotocol/rust-sdk](https://github.com/modelcontextprotocol/rust-sdk)：标准 MCP Rust SDK。

### 项目文档

- [实现说明](docs/implementation-notes.md)
- [实时语音](docs/realtime-voice.md)
- [Qwen3-TTS 导出与运行](docs/qwen3-tts.md)
- [开发与验证约束](AGENTS.md)
- [CPU Compose](docker-compose.yml)
- [NVIDIA CUDA Compose](docker-compose-nvidia.yml)
- [环境变量示例](.env.example)
