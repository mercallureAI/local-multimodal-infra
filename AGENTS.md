# AGENTS.md

Repo-specific instructions for future OpenCode agents. Higher-priority user instructions override.

## Architecture boundaries

- Rust Cargo workspace. Service bins `controller` and `worker` live in `crates/cli`; the `local-multimodal-infra` main only prints a hint.
- `crates/controller` may depend on API/files/model-store/registry/scheduler, but must not depend on runtime/backend-ort/adapters. It schedules/forwards and does not load models.
- `crates/worker` registers, heartbeats, and exposes protected `/internal/infer`.

## Naming rules

- Avoid generic capability-only names and broad family-only names when the adapter is task-specific.

## Config, routes, and storage

- Default configs: `configs/controller.yaml`, `configs/worker.yaml`; model specs: `configs/models.d/*.yaml`.
- Default addresses: controller HTTP API and legacy JSON-RPC `127.0.0.1:17890`, worker `127.0.0.1:17891`, standard MCP Streamable HTTP `127.0.0.1:17892/mcp`.
- Standard MCP exposes admin and mutating tools; keep it loopback-only by default. Do not use `--mcp-bind 0.0.0.0:17892` or otherwise expose it on a shared network unless a future admin-token/ACL layer is implemented and enabled.
- Start services with explicit storage args: `--workdir ./workdir --model-dir ./workdir/models`.
- Runtime artifacts: real models only in `workdir/models`; data/logs/uploads/generated/temp only in `workdir/data`; SQLite default `workdir/data/local.db`. Do not commit or delete `workdir/`; do not commit `target/`.
- Controller routes worth remembering: `/health`, `/assets`, `/assets/sign`, `/files/upload/...`, legacy JSON-RPC only at `POST /rpc/admin` and `POST /rpc/infer` (old JSON-RPC `/mcp/admin` and `/mcp/infer` compatibility aliases are removed), standard MCP Streamable HTTP only on the separate bind `127.0.0.1:17892/mcp` (verify only with official MCP SDK / rmcp client semantics, not raw HTTP JSON-RPC), OpenAI `GET /v1/models`, `POST /v1/audio/transcriptions`, `POST /v1/audio/speech`. Worker routes: `/health`, protected `/internal/infer`.

## Service execution rules

- Never run controller/worker/dev/HTTP services as blocking foreground commands; background them, set explicit command timeouts, poll readiness with a bounded deadline, collect logs on timeout, and clean up.
- Do not use PowerShell to start/manage/stop services or bypass smoke harnesses with `curl`/`Invoke-WebRequest`. PowerShell is okay only for short non-service commands unless the user forbids it. Every shell command needs an explicit timeout.
- Manual service forms exist, but must be run only under the async/cleanup rules above: `cargo run --bin controller -- configs/controller.yaml --workdir ./workdir --model-dir ./workdir/models ...`; `cargo run --bin worker -- configs/worker.yaml --workdir ./workdir --model-dir ./workdir/models ...`.
- After tests, verify ports `17890`, `17891`, and `17892` are no longer listening, or stop only the processes started for the test.

## Smoke harness

- Prefer the harness over one-off scripts/curl for service/API/MCP smoke. Primary local smoke: `python -m scripts.local.smoke --tests yolo,qwen-asr,indextts --workdir ./workdir --model-dir ./workdir/models`.
- Other useful harness commands:
  - `python -m scripts.local.smoke --tests mcp --workdir ./workdir --model-dir ./workdir/models` (standard MCP SDK group on `127.0.0.1:17892/mcp`: tool listing, admin/catalog/assets, generic task flow, direct inference where resources/artifacts are available)
  - `python -m scripts.local.smoke --tests all --workdir ./workdir --model-dir ./workdir/models` (both groups; skip flags still apply)
  - `python -m scripts.local.smoke --tests assets,yolo,qwen-asr,indextts --workdir ./workdir --model-dir ./workdir/models`
  - `python -m scripts.local.smoke --tests indextts_asr --indextts-frontend auto --workdir ./workdir --model-dir ./workdir/models`
  - `python scripts/smoke_api_mcp.py --tests yolo,qwen-asr --workdir ./workdir --model-dir ./workdir/models`
  - `python -m scripts.local.smoke --tests mcp_standard --workdir ./workdir --model-dir ./workdir/models`
- Use `--skip-build` only when existing `target/debug/controller(.exe)` and `target/debug/worker(.exe)` are valid.
- `scripts/smoke_api_mcp.py` is a thin compatibility entrypoint; implementation is under `scripts/local/` (`smoke.py`, `processes.py`, `http_client.py`, `paths.py`). The harness builds via `cargo build --bins` unless skipped, starts services with Python `subprocess.Popen`, prints PIDs, writes `workdir/data/*.{stdout,stderr}.log`, waits for `/health`, and cleans up.
- Harness defaults: controller URL `http://127.0.0.1:17890`, standard MCP URL `http://127.0.0.1:17892/mcp`, worker URL `http://127.0.0.1:17891`, smoke registration/admin env tokens, ready timeout 30s, request timeout 15s, build timeout 600s, cleanup port wait 8s.
- Update `scripts/local/smoke.py` when endpoints, legacy RPC methods, standard MCP tools, OpenAI APIs, task upload, or assets behavior changes.

## Tokens, uploads, and task flow

- Controller and worker must share a registration token: controller `--worker-registration-token`, worker `--registration-token`, or `LOCAL_WORKER_REGISTRATION_TOKEN`.
- Upload URLs derive from `--public-base-url` / `public_base_url` / `LOCAL_PUBLIC_BASE_URL`; local default is `http://127.0.0.1:17890`.
- Signed upload HMAC uses `--upload-signing-secret` / `upload_signing_secret` / `LOCAL_UPLOAD_SIGNING_SECRET`; if absent, a random per-process secret makes old upload URLs fail after restart.
- Generic legacy RPC upload flow: `create_task` via `/rpc/infer` -> raw-byte `POST /files/upload/<task_id>/<slot>?expires=...&sig=...` -> `start_task` / `wait_task`.
- Standard MCP upload flow uses MCP tools (`create_task`, `start_task`, `wait_task`, `get_task`) through the official SDK; raw HTTP is acceptable only for data transfer to signed `upload_url` / `download_url`, not for MCP protocol calls.

## Verification commands

- Cheap/default Rust checks: `cargo check --workspace --all-targets`, `cargo build --bins`, `cargo test --workspace`.
- Opt-in real Qwen artifact test: `LOCAL_QWEN_ASR_MODEL_DIR=workdir/models/qwen3-asr-0.6b-onnx cargo test -p local-adapter-qwen-asr real_model_smoke_if_env_set -- --nocapture` (do not use PowerShell if the user forbids it).

## Script entrypoints

- Help: `python -m scripts.local.smoke --help`, `python -m scripts.local.indextts_export --help`, `python scripts/indextts_export.py --help`.
- Standard MCP validation client: `python -m scripts.local.mcp_standard_client --url http://127.0.0.1:17892/mcp --full` (requires the official Python `mcp` SDK in that interpreter).
- IndexTTS export top-level entrypoint `scripts/indextts_export.py` delegates to `scripts.local.indextts_export`; do not use old `tools/indextts` paths.

Release smoke examples:

```bash
cargo build --release --bins
python -m scripts.local.smoke --skip-build --release --tests mcp --workdir ./workdir --model-dir ./workdir/models --ready-timeout 60 --request-timeout 600
python -m scripts.local.smoke --skip-build --release --tests rpc --workdir ./workdir --model-dir ./workdir/models --ready-timeout 60 --request-timeout 600
```
