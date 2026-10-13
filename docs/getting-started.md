# Getting started

Two services: the **controller** (API, MCP, model catalog, task scheduling;
`:17890`, MCP on `:17892`) and one or more **workers** (load the models and
run them; `:17891`). Runtime data lives in `workdir/`: models in
`workdir/models`, SQLite, uploads, results and logs in `workdir/data`. Models
are never baked into images.

## Docker

Needs Docker with Compose v2.

```bash
cp .env.example .env
```

Replace the placeholders in `.env`:

| Variable | Purpose |
| --- | --- |
| `LOCAL_WORKER_REGISTRATION_TOKEN` | shared by controller and worker |
| `LOCAL_UPLOAD_SIGNING_SECRET` | signs upload URLs (random per process if unset: old URLs fail after a restart) |
| `LOCAL_ADMIN_TOKEN` | required by the admin API and MCP (`/rpc/admin`, `/mcp/admin`) |
| `LOCAL_MCP_INFER_TOKENS` | comma-separated inference tokens; empty = inference is unauthenticated |
| `LOCAL_PUBLIC_BASE_URL` | the controller URL clients see, used in upload URLs |

CPU:

```bash
docker compose up -d --build
curl --fail http://127.0.0.1:17890/health
```

NVIDIA GPU (Linux x86_64; NVIDIA driver for CUDA 12, NVIDIA Container Toolkit,
Compose 2.30+). Only the worker gets the GPU; the controller is the CPU image:

```bash
ORT_CUDA_VERSION=12 docker compose -f docker-compose-nvidia.yml up -d --build
docker compose -f docker-compose-nvidia.yml exec worker nvidia-smi
```

`/health` says the services are up, not that a model runs on the GPU: the
worker's `lazy loading model` log line shows the providers a model uses.

**Network exposure.** The Compose files publish the controller (`17890`) and
worker (`17891`) on all interfaces and MCP (`17892`) on loopback only. For
local use change them to `127.0.0.1:17890:17890` and `127.0.0.1:17891:17891`.
`/v1/models`, `/health` and some asset routes are outside the inference token,
so protect the controller port with network access control too.

## Models

Download the published ones and export the rest; see [models](models/README.md).
For example:

```bash
curl --fail-with-body http://127.0.0.1:17890/rpc/admin \
  -H 'content-type: application/json' -H "x-local-admin-token: $LOCAL_ADMIN_TOKEN" \
  --data '{"jsonrpc":"2.0","id":1,"method":"download_model","params":{"id":"yolo11n.onnx"}}'
```

## From source

Rust 1.92 or newer. ONNX Runtime is loaded at run time (the official 1.30
build; see [implementation notes](implementation-notes.md#ort-only-boundary)),
so building needs none:

```bash
cargo build --release --bins                 # CPU
cargo build --release --bins --features cuda # NVIDIA GPU
python -m scripts.local.fetch_onnxruntime    # ONNX Runtime 1.30 beside the binaries (--cpu for the CPU build)
```

Then start both (in the background; they serve until stopped):

```bash
target/release/controller configs/controller.yaml --workdir ./workdir --model-dir ./workdir/models
target/release/worker configs/worker.yaml --workdir ./workdir --model-dir ./workdir/models
```

with the same `LOCAL_*` variables as above in their environment. The worker
loads `ORT_DYLIB_PATH` if set, else the ONNX Runtime beside it; on Windows
never let it pick up `C:\Windows\System32\onnxruntime.dll`.

Each model category is a cargo feature of the binaries (all on by default):
an audio-only worker is `--no-default-features --features audio,cuda`.

## Checking a deployment

The smoke harness builds, starts both services, sends real requests and
cleans up:

```bash
python -m scripts.local.smoke --tests rpc --workdir ./workdir --model-dir ./workdir/models
python -m scripts.local.smoke --tests mcp --workdir ./workdir --model-dir ./workdir/models  # needs the Python `mcp` SDK
```

`--skip-build --release` reuses `cargo build --release --bins`. Tests whose
models are missing are skipped; `python -m scripts.local.smoke --help` lists
them all (`ocr`, `ppocr`, `depth`, `moderation`, ...).
