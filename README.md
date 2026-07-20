# local-multimodal-infra

> 把本地算力变成 Agent 可调用的多模态服务。它提供本地 MCP Server 和 OpenAI-compatible API Server，让 Agent 能使用本地图片理解、语音识别、语音合成等能力。

[English README](README.en.md)

## 项目介绍

`local-multimodal-infra` 是一个本地多模态算力网关。它的目标很简单：**把你本机或内网机器上的算力，转换成可轻易部署的模型能力，以最少配置暴露给 Agent**。

你可以把它理解为：

- **Agent 的本地工具箱**：Agent 不需要关心模型怎么加载、文件放在哪里、任务在哪台机器上执行。
- **标准 MCP Server**：运行在独立端口，使用官方 SDK 的 Streamable HTTP 协议。
- **Legacy JSON-RPC API**：运行在 controller 端口，提供模型管理、文件上传、推理任务和结果查询。
- **OpenAI-compatible API Server**：提供部分 OpenAI 风格接口，方便已有应用或 Agent 工作流接入。

## 适合的场景

- 让你的 Agent 调用本机或是闲置机器上的算力，实现一些多模态能力。
- 把一台有 CPU/GPU/大内存的机器变成内网 Agent 可用的推理节点。
- 希望图片、音频、模型工件留在本地，不交给云服务。
- 需要一个本地 API Server 接入 Agent 框架。
- 在有限的算力下，动态装卸不同的模型，以最大化生产力与性价比。
- 想使用一个完整的可维护的多模态本地推理基础设施，而不是维护多个零散模型脚本。
- 打造一个云端与本地协同的 Agent。

## Agent 如何使用

项目提供三类入口：

| 入口 | 面向对象 | 用途 |
| --- | --- | --- |
| Legacy JSON-RPC API | Agent / 工具调用 | `POST /rpc/admin` 使用 `LOCAL_ADMIN_TOKEN`；`POST /rpc/infer` 在配置 `LOCAL_MCP_INFER_TOKENS` 后要求其中任一密钥。 |
| 标准 MCP Server | Agent / 标准 MCP 客户端 | Admin：`http://127.0.0.1:17892/mcp/admin`；推理：`http://127.0.0.1:17892/mcp/infer`。 |
| OpenAI-compatible API Server | 应用 / OpenAI 风格客户端 | 列出模型、语音识别、语音合成等有限兼容接口。 |

默认服务地址：

- Controller / API Server / legacy JSON-RPC: `http://127.0.0.1:17890`
- Legacy JSON-RPC：`POST /rpc/admin`、`POST /rpc/infer`
- 标准 MCP Streamable HTTP：`http://127.0.0.1:17892/mcp/admin`、`http://127.0.0.1:17892/mcp/infer`
- Admin MCP/RPC：必须配置 `LOCAL_ADMIN_TOKEN`，客户端使用 `Authorization: Bearer <token>` 或 `x-local-admin-token`
- 推理 MCP/RPC：`LOCAL_MCP_INFER_TOKENS=token-a,token-b`；未配置或为空时不鉴权，配置后客户端必须提供列表中的任一密钥，可使用 Bearer 或 `x-local-infer-token`
- 模型管理：`list_models` / `get_model` 返回 `downloaded` 与 `download_state`；`download_model` 只提交异步任务，同模型在下载中或已完整下载时会去重；使用 `get_model_download_status` 查询模型及逐文件状态。
- Worker: `http://127.0.0.1:17891`

外部 Agent 推荐使用“创建任务、上传文件、等待结果”的方式接入，这样不需要共享宿主机文件路径。

## 当前能力状态

| 能力 | 状态 |
| --- | --- |
| 图片目标检测 | `yolo11n.onnx` |
| 语音识别 | `qwen3-asr-0.6b-onnx` |
| 语音合成 | `indextts-1.5-onnx` |
| 产物管理 | 轻量内网存储，统一管理发送给模型的素材与模型生成的产物 |
| 模型管理 | 支持配置声明、下载、启用、禁用和状态查询。 |
| 本地验证 | 提供 smoke harness 验证 Docker、服务和 API 调用链。 |

## 快速开始

本项目提供了 Docker Compose ，可以快速开始使用，当然你也可以直接运行本地二进制。

```bash
cp .env.example .env
docker compose up --build
```

使用 Docker Compose 启动后会得到：

- 一个 controller：对外提供 legacy JSON-RPC、标准 MCP Server 和 OpenAI-compatible API Server。
- 一个 worker：负责加载模型并执行本地推理。
- 一个本地目录 `./workdir`：保存模型、数据、上传文件和运行日志。

真实模型不会打包进镜像。首次启动后，需要通过管理接口下载默认模型，或提前把模型放入 `workdir/models`。

### NVIDIA GPU Compose

上面的默认命令仍然使用 CPU 编译，并且不会申请 GPU。NVIDIA 部署需要先安装
较新的 NVIDIA 驱动和
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)，
以及 Docker Compose，并确认前置条件：

```bash
docker compose version
nvidia-smi
```

然后运行：

```bash
cp .env.example .env
ORT_CUDA_VERSION=12 docker compose -f docker-compose-nvidia.yml up --build
```

`ORT_CUDA_VERSION` 默认为且目前仅支持 `12`，它在构建时选择 ORT 的 CUDA 12
二进制包。该路径目前仅支持 Linux x86_64，与 rc.12 发布的 CUDA 包一致。
只有 NVIDIA worker 通过 `deploy.resources.reservations.devices` 预留一个 GPU，并使用独立的
`local-multimodal-infra:nvidia-cuda12` 镜像；controller 继续使用 CPU 镜像且
不会获得 GPU。

CPU 与 NVIDIA Compose 共同使用唯一的 `configs/models.d`。YOLO、Qwen ASR、
FP32 IndexTTS、multilingual-e5-small 和 mMARCO MiniLM reranker 表达
`[cuda, cpu]` 意图。E5 在 CPU 上优先选择派生的 INT8 pooled 图，在可用 CUDA
上优先选择派生的 O4 pooled 图；mMARCO 仍直接选择官方对应量化图。session 加载前，runtime availability
解析会在 CUDA 未编译时立即将其变为 `[cpu]`；CUDA 构建会执行并缓存一次微型
CUDA session 探针，若 EP 无法注册和初始化，则不会尝试模型 CUDA session。
探针成功时保持 CUDA 优先和 CPU 回退，但它不验证每个模型或算子，特定模型的
CUDA 加载仍可能回退 CPU。IndexTTS policy 及 A、B、C、D、E、E-prefill、F
共七个 session 都传递此顺序并报告实际 provider，但真实 NVIDIA 硬件与工件
验证仍未完成。TensorRT 不受支持且不在本任务范围内。

不要把 `/health` 当成 GPU 推理证明。先运行
`docker compose -f docker-compose-nvidia.yml exec worker nvidia-smi` 验证容器
可见 GPU。下载并启用 YOLO 工件后，把仓库样例复制到 worker 可见的共享
workdir，再发送真实的受支持请求：

```bash
mkdir -p workdir/data
cp scripts/assets/yolo-input.jpg workdir/data/yolo-input.jpg
curl --fail-with-body http://127.0.0.1:17890/rpc/infer \
  -H 'content-type: application/json' \
  --data '{"jsonrpc":"2.0","id":"gpu-yolo","method":"object_detect","params":{"model":"yolo11n.onnx","image":{"path":"/app/workdir/data/yolo-input.jpg","mime":"image/jpeg"}}}'
docker compose -f docker-compose-nvidia.yml logs worker |
  grep 'lazy loading model'
docker compose -f docker-compose-nvidia.yml exec worker nvidia-smi dmon -s pucvmet
```

在另一个终端中于请求期间启动 `nvidia-smi dmon`。lazy-load 日志显示 effective
provider order，dmon 可显示同时段 GPU 活动；两者都不能证明逐节点 GPU placement。
硬件快照中的 `has_cuda` 当前仍会是 `false`，因为控制面报告尚未探测 NVML；
它不决定 ORT CUDA EP 的实际选择。

## 本地开发

不使用 Docker 时，可以直接运行 controller 和 worker。具体命令、服务约束和 smoke harness 用法见 [AGENTS.md](AGENTS.md) 与 [Implementation notes](docs/implementation-notes.md)。

Smoke harness 常用别名：

- `mcp`：运行标准 MCP SDK 组：分别通过 `/mcp/admin` 和 `/mcp/infer` 验证隔离后的工具列表、鉴权、admin/catalog/assets、generic task flow，以及本地资源可用时的 direct inference。
- `all`：同时展开 `rpc` 和 `mcp` 两组，并继续尊重 `--skip-yolo`、`--skip-qwen-asr`、`--skip-indextts` 等跳过参数。
- `mcp_standard`：只用官方 Python MCP SDK 验证标准 MCP 工具列表、admin/catalog/assets、generic/direct 工具调用，不向 `/rpc/*` 伪装 MCP。
- `text`：用真实模型验证 OpenAI `/v1/embeddings` 和 vLLM `/v1/rerank`，包括 384 维、归一化、batch 顺序、`top_n`、相关性排序和 token usage。


示例：

```bash
python -m scripts.local.smoke --tests rpc --workdir ./workdir --model-dir ./workdir/models
python -m scripts.local.smoke --tests mcp --workdir ./workdir --model-dir ./workdir/models
python -m scripts.local.smoke --tests all --workdir ./workdir --model-dir ./workdir/models
python -m scripts.local.smoke --tests text --workdir ./workdir --model-dir ./workdir/models
```

下载 E5 官方图后，生成把 masked mean pooling 与 L2 normalization 合入图内的
派生文件。原图不会被覆盖；若派生文件不存在，adapter 会兼容回退到 Rust host
pooling：

```bash
uv run --with onnx --python 3.12 \
  python -m scripts.local.e5_pooling_export --model-dir workdir/models
```

CUDA pooled 图使用固定 shape 的 pinned-host I/O binding，输入/输出缓冲在重复请求间
复用。按 batch 与文本长度运行 release 端到端基准：

```bash
python -m scripts.local.benchmark_text_embeddings --mode cpu \
  --controller-bin target/release/controller --worker-bin target/release/worker
python -m scripts.local.benchmark_text_embeddings --mode gpu \
  --controller-bin target/release/controller --worker-bin target-cuda/release/worker
```

## 文本检索 API

Embedding 使用 OpenAI 兼容端点；`input_type` 是 E5 的可选扩展，取值为
`query` 或 `passage`，默认 `passage`：

```bash
curl http://127.0.0.1:17890/v1/embeddings \
  -H 'content-type: application/json' \
  --data '{"model":"multilingual-e5-small-onnx","input":["你好","hello"],"input_type":"query"}'
```

Rerank 兼容 vLLM/Jina/Cohere 形状，并注册 `/rerank`、`/v1/rerank`、
`/v2/rerank` 三个别名：

```bash
curl http://127.0.0.1:17890/v1/rerank \
  -H 'content-type: application/json' \
  --data '{"model":"mmarco-minilm-l12-onnx","query":"法国首都","documents":["巴黎","巴西利亚"],"top_n":1}'
```

常用配置入口：

- `configs/controller.yaml`
- `configs/worker.yaml`
- `configs/models.d`

## 目录约定

- `workdir/models`：真实模型工件。
- `workdir/data`：SQLite、上传文件、日志和生成结果。
- `docker-compose.yml` / `Dockerfile`：CPU 容器启动入口。
- `docker-compose-nvidia.yml` / `Dockerfile.nvidia`：NVIDIA CUDA 12 worker 启动入口。
- `scripts/local/smoke.py`：本地 smoke harness。
- `scripts/local/e5_pooling_export.py`：生成 E5 图内 pooling/L2 派生图。
- `scripts/local/benchmark_text_embeddings.py`：batch/长度矩阵 release 基准。
- `docs/implementation-notes.md`：更多实现细节。

## 路线图

- [ ] 完善标准 MCP 实现并保持 legacy JSON-RPC 边界清晰
- [ ] 更成熟的 API Server
- [ ] 更稳定的运行时管理
- [ ] IndexTTS 本地能力恢复
- [ ] 可选硬件加速路径
...

## 更多文档

- [Implementation notes](docs/implementation-notes.md)
- [AGENTS.md](AGENTS.md)
- [Dockerfile](Dockerfile)
- [docker-compose.yml](docker-compose.yml)
- [docker-compose-nvidia.yml](docker-compose-nvidia.yml)
- [.env.example](.env.example)

Release smoke examples:

```bash
cargo build --release --bins
python -m scripts.local.smoke --skip-build --release --tests mcp --workdir ./workdir --model-dir ./workdir/models --ready-timeout 60 --request-timeout 600
python -m scripts.local.smoke --skip-build --release --tests rpc --workdir ./workdir --model-dir ./workdir/models --ready-timeout 60 --request-timeout 600
```
