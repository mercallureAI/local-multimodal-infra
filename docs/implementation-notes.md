# Implementation notes

## ORT-only boundary

This MVP intentionally implements only the ORT backend seam. Candle, Python, C++, sidecar, and external-process alternatives are not implemented. If a path would require a non-ORT backend it must return `Unsupported`, `NeedUserConfirmation`, or `NeedImplementation`.

ONNX Runtime is loaded at run time (the `ort` crate's `load-dynamic` feature), so building and checking need no ONNX Runtime at all. The worker loads the library named by `ORT_DYLIB_PATH`, else `onnxruntime.dll` / `libonnxruntime.so` beside its executable, else from the library search path. The supported runtime is the official ONNX Runtime **1.30.0** (the `ort` 2.0.0-rc.12 bindings request C API 24, which newer runtimes keep serving): the CPU Dockerfile installs the official CPU build and `Dockerfile.nvidia` the official CUDA 12 build, both pinned by digest, and set `ORT_DYLIB_PATH`. 1.30 is required, not only newer: the onnxruntime-genai decoder exports used for chat (GroupQueryAttention with a shared past/present buffer, shared quantized embeddings) produce wrong tokens on CUDA with the 1.24.x build `ort` used to bundle and with 1.28, and correct ones with 1.30. The backend remains ORT-specific at the API/config layer; CUDA/DML provider features are opt-in and must be validated against the active ORT execution providers before use.

On Windows, never rely on the search path: `C:\Windows\System32\onnxruntime.dll` (an older Windows ML copy) would be found. Put the 1.30 DLLs beside `worker.exe` or set `ORT_DYLIB_PATH`; `cargo test` binaries run from `target/<profile>/deps`, so tests that create sessions need `ORT_DYLIB_PATH`. The official Windows CUDA 12 zip needs a newer CUDA 12 runtime than 12.4 (its CUDA provider then fails to load with error 127); the DLLs of the `onnxruntime-gpu==1.30.0` Python wheel (`onnxruntime/capi/`) work with CUDA 12.4 and cuDNN 9 on `PATH`.


## Providers: CPU, CUDA, DML, TensorRT

`backend-ort` exposes `ProviderKind::{Cpu,Cuda,Dml,Trt}`, `ProviderOptions`, and `ProviderSelection`.

- CPU: the portable fallback in the shared checked-in model specs.
- CUDA: preferred by shared YOLO, Qwen ASR, IndexTTS, E5 embedding, and mMARCO reranker specs when runtime availability confirms it.
- DML: configurable as an opt-in provider on Windows builds with the backend feature enabled. Failures return an explicit reason and can fall back to CPU when configured.
- TensorRT: an existing optional backend feature, but it is not enabled, configured, or included by the NVIDIA Compose deployment.

Model/provider differences are handled by model config `runtime.provider_order`. CPU and NVIDIA Compose use the sole `configs/models.d`, where all supported models express `[cuda, cpu]`. Before any adapter/session is constructed, runtime availability resolution filters providers against actual ORT usability: no CUDA feature becomes `[cpu]` immediately; a CUDA build performs and caches one process-level probe that registers the CUDA EP and creates a tiny in-memory FP32 session. Missing CUDA runtime dependencies, driver/device access, or provider registration therefore becomes `[cpu]` without a model CUDA load attempt, while a successful probe preserves `[cuda, cpu]`. E5 prefers a derived pooled qint8 graph when CPU is effective or a derived pooled O4 graph when CUDA remains first; if those files have not been generated, it falls back to the official graph and host pooling. mMARCO selects its official quantized graph when CPU is effective or its official O4 graph on CUDA. The tiny probe does not validate every model/operator or prove graph-node CUDA placement. If a particular CUDA model session subsequently fails, the backend's existing provider loop selects CPU and records `cpu_fallback_used=true`. TensorRT is unsupported and out of scope for this deployment.

## Text embedding and reranking

`multilingual-e5-small-onnx` implements `text.embed` with E5 `query:`/`passage:` prefixes, attention-mask average pooling, and L2 normalization. `scripts/local/e5_pooling_export.py` appends the masked mean and normalization operations to both the official qint8 and O4 graphs, changes the graph output from `[batch, sequence, 384]` to `[batch, 384]`, and leaves the upstream files intact. The adapter prefers these `_pooled.onnx` files and retains host pooling as a compatibility fallback. CUDA pooled inference uses reusable pinned-host I64 input buffers and a pinned FP32 output through ORT I/O binding; the binding is recreated when batch/sequence shape changes. `/v1/embeddings` accepts OpenAI string or string-array input and returns the OpenAI list/item/usage envelope. The output dimension is fixed at 384.

Run `python -m scripts.local.benchmark_text_embeddings` with the CPU and CUDA release workers to compare batch `1/8/32/128` over short and tokenizer-truncated 512-token inputs. Reports include HTTP end-to-end latency, throughput, token counts, norms, samples, and a first vector for cross-provider parity checks.

`mmarco-minilm-l12-onnx` implements `text.rerank` by pair-tokenizing each query/document, applying sigmoid to the single cross-encoder logit, sorting descending, and applying optional `top_n`. `/rerank`, `/v1/rerank`, and `/v2/rerank` return the vLLM-compatible `id`, `model`, `usage.total_tokens`, and `results[{index,document:{text},relevance_score}]` envelope. Both capabilities are also available through generic tasks and direct MCP/legacy-RPC methods `text_embed` and `text_rerank`.

### NVIDIA Compose and ORT binary compatibility

`docker compose -f docker-compose-nvidia.yml up --build` builds only the worker
with `cargo build --locked --release -p local-cli --bin worker --features cuda`
(`ORT_CUDA_VERSION` stays `12`). The controller uses the ordinary CPU
Dockerfile/image and has no GPU request. Only the worker declares Compose
`gpus: all`, which requires Docker Compose 2.30.0 or newer; check with
`docker compose version`. The worker is constrained to Linux x86_64, the target
of the official CUDA build.

The worker image adds `onnxruntime-linux-x64-gpu_cuda12-1.30.0.tgz` from the
ONNX Runtime GitHub release (SHA-256
`f9886932ee7bb0b4d3fcab736a392d4ff5efaa0672b47f19f0cec03437cf64f1`):
`libonnxruntime.so*`, `libonnxruntime_providers_shared.so` and
`libonnxruntime_providers_cuda.so` go to `/usr/local/lib`, and
`ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so`. The runtime base stays
NVIDIA's full tag `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04` with manifest
digest `sha256:ac55d124da4882b497f732d8dfd9a702d5447a5f29d08d56da6f64f0a1eb34bc`
(CUDA 12, cuDNN 9, Ubuntu 24.04 glibc 2.39). ONNX Runtime's CUDA EP
documentation states that CUDA 12.x builds are compatible within the CUDA 12
major family while cuDNN major versions must match. The final image runs `ldd`
on the worker, the ORT core and both providers and fails the build if any
library is missing.

Evidence:

- ONNX Runtime 1.30.0 release assets and digests: <https://github.com/microsoft/onnxruntime/releases/tag/v1.30.0>
- ONNX Runtime CUDA/cuDNN compatibility: <https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements>
- Official NVIDIA image/tag and Container Toolkit requirement: <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/12.8.1-cudnn-runtime-ubuntu24.04>
- NVIDIA CUDA minor-version/driver compatibility: <https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html>

Reproducible inspection used for this choice:

```bash
# The release API lists each asset's sha256 digest:
curl -s https://api.github.com/repos/microsoft/onnxruntime/releases/tags/v1.30.0 \
  | jq -r '.assets[] | select(.name | test("linux-x64")) | "\(.name) \(.digest)"'
tar -xzf onnxruntime-linux-x64-gpu_cuda12-1.30.0.tgz
readelf -d onnxruntime-linux-x64-gpu_cuda12-1.30.0/lib/libonnxruntime_providers_cuda.so | grep NEEDED
# Resolve the official tag's multi-arch manifest digest:
docker buildx imagetools inspect nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
```

The pinned digest is the multi-architecture manifest-list digest. Compose sets
`platform: linux/amd64`, so Docker selects its amd64 child manifest while the
tag+digest still pins the official immutable manifest list.

The host needs an NVIDIA driver compatible with CUDA 12 (NVIDIA documents
driver 525 or newer for the CUDA 12 family) and NVIDIA Container Toolkit.
`/health` proves only service health. Verify GPU visibility with
`docker compose -f docker-compose-nvidia.yml exec worker nvidia-smi`, then run
a real YOLO request against the running Compose deployment:

```bash
mkdir -p workdir/data
cp scripts/assets/yolo-input.jpg workdir/data/yolo-input.jpg
curl --fail-with-body http://127.0.0.1:17890/rpc/infer \
  -H 'content-type: application/json' \
  --data '{"jsonrpc":"2.0","id":"gpu-yolo","method":"object_detect","params":{"model":"yolo11n.onnx","image":{"path":"/app/workdir/data/yolo-input.jpg","mime":"image/jpeg"}}}'
docker compose -f docker-compose-nvidia.yml logs worker |
  grep 'lazy loading model'
```

This requires downloaded/enabled YOLO artifacts. Run
`docker compose -f docker-compose-nvidia.yml exec worker nvidia-smi dmon -s pucvmet`
concurrently to sample GPU activity. The lazy-load log exposes the effective
provider order and dmon can show activity, but neither proves per-node GPU
placement. IndexTTS CUDA policy is enabled because all six A, B, C, D, E, and F
sessions (E also performs the prompt prefill with empty KV caches)
are created from the same provider selection and no concrete code/operator
blocker is known; real NVIDIA artifact smoke remains unverified. TensorRT is not
built or configured. The control-plane hardware snapshot still reports
`has_cuda: false` because it does not probe NVML; that reporting limitation is
independent of actual ORT EP selection.


## Chat completion (Qwen3)

`qwen3-4b-instruct-2507-int4-onnx` implements `chat.complete` with the `qwen3_chat` adapter on decoder graphs exported by the onnxruntime-genai model builder:

```bash
python -m onnxruntime_genai.models.builder -i Qwen/Qwen3-4B-Instruct-2507 -o <dir>   -p int4 -e cuda --extra_options int4_algo_config=rtn_last prune_lm_head=true
```

The directory (`<model_dir>/qwen3-4b-instruct-2507-int4-onnx`, a `local` artifact) holds `model.onnx(.data)`, `genai_config.json` (layer/KV-head/head-size, input and output names, EOS ids), `tokenizer.json` and `chat_template.jinja` (or `tokenizer_config.json`). `rtn_last` keeps the LM head in int8 and shares it with the embedding (2.5 GB); with `k_quant*` the model tends to end tool calls without `</tool_call>`. Other Qwen3-family decoder exports with the same I/O work the same way.

- **Prompt**: the model's own Jinja chat template is rendered with minijinja as Hugging Face does (trim/lstrip blocks, `tojson` with Python separators and key order), including tools, assistant tool calls and tool results.
- **KV cache**: `SharedKvBinding` (backend-ort) allocates one `[1, kv_heads, max_context, head_size]` tensor per layer once, on the session's device, and binds it both as `past_*` input and `present_*` output, so GroupQueryAttention updates it in place; the valid length is the `attention_mask` length. `metadata.max_context` sets the capacity (default 8192 tokens, ~144 KB/token for Qwen3-4B).
- **Prefix reuse**: the adapter remembers which tokens the cache holds; a request prefills only the tokens after its common prefix with them (system prompt, tools and earlier turns of a conversation), then decodes token by token. `usage.prompt_tokens_details.cached_tokens` reports the reused part.
- **Decoding**: temperature (0 = greedy), top-k, top-p, presence/frequency penalty, seed, stop strings, OpenAI `logit_bias`, and two extensions: `tool_call_bias` (offset on the `<tool_call>` token) and `tool_bias` (tool name -> offset on the first token of that name when the model names the tool).
- **Tool calls**: `<tool_call>{"name", "arguments"}</tool_call>` blocks become OpenAI `tool_calls` (finish reason `tool_calls`); a call ending right after complete JSON still counts.
- **Streaming**: `stream: true` answers with server-sent events (`chat.completion.chunk`, then `[DONE]`; `stream_options.include_usage` adds a usage chunk; the final chunk carries `timings` with `prefill_ms`, `first_token_ms`, `decode_ms`). Internally the controller forwards to the worker's `/internal/infer_stream` (newline-delimited `InferenceEvent`s) and the runtime passes each token to a sink while the model is held; when the client disconnects, the worker's sink fails and generation stops at the next token (finish reason `cancelled`), freeing the model for the next request.
- Generic tasks accept `chat.complete` with `params.messages`, `params.tools` and `params.options` (`ChatOptions`).

Opt-in real-model test: `LOCAL_QWEN3_CHAT_MODEL_DIR=<dir> ORT_DYLIB_PATH=<onnxruntime 1.30> cargo test --release -p local-adapter-qwen3-chat --features cuda real_model -- --nocapture`.

## Qwen ASR limitations

The adapter validates the known `qwen3-asr-0.6b-onnx` artifact layout and establishes interfaces for WAV read/resampling, 128-bin feature extraction, tokenizer JSON loading, embeddings/KV-cache, and decoder loop orchestration. INT4 artifacts may require ORT contrib/custom-op support for `MatMulNBits`; use `LOCAL_QWEN_ASR_MODEL_DIR=<model-dir> cargo test -p local-adapter-qwen-asr real_model_smoke_if_env_set -- --nocapture` as an opt-in real-artifact smoke test.


## IndexTTS FP32 and text normalization boundary

IndexTTS ONNX support uses root FP32 artifacts with CUDA-first, CPU-fallback intent. The default catalog downloads the explicit `IndexTTS_A.onnx` through `IndexTTS_F.onnx` (no separate E-prefill graph: `IndexTTS_E.onnx` performs the prompt prefill when fed zero-length KV caches), `bpe.model`, and manifest files from `ModaLeap/indextts-1.5-onnx` into `workdir/models/indextts-1.5-onnx`; export/package tooling can also write the same root layout. Runtime validation loads that root directly and no longer auto-selects `fp16/` for CUDA or `q4/` for CPU. Existing `q4/` or `fp16/` model caches may remain on disk but are ignored by current code and docs. All six A, B, C, D, E, and F sessions are loaded from one `OrtBackend` built from `spec.runtime.provider_order` and are included in its provider report; code/policy support is present, while real NVIDIA hardware validation is not.

The official IndexTTS 1.5 frontend (`workdir/models/index-tts-v1.5/indextts/utils/front.py` and `common.py`) uses WeTextProcessing/pynini TN when available, but its tokenizer path does **not** convert arbitrary Hanzi to pinyin. It protects explicit tone-number pinyin and Chinese-name placeholders around TN, expands a small English `'s` contraction pattern, applies a punctuation replacement map, then calls `tokenize_by_CJK_char`, which splits each CJK character and uppercases non-CJK segments before SentencePiece.

The default runtime path is the Rust frontend plus local SentencePiece. Explicit token ids (`text_token_ids`, `pretokenized_text_ids`, or `indextts_text_token_ids`) remain available for oracle/debug use; when present, the adapter validates a non-empty integer list in a sane range and feeds those ids to graph B directly, skipping local tokenization but preserving the A/B/C/D/E/F graph flow.

Long input is planned into ordered, punctuation-aware segments (120 model tokens by default, with a hard token split when no punctuation is available). Graph A processes the reference once per request; graphs B-F process each segment, and successful waveforms are joined with 200 ms of silence only between segments. Manifest/model metadata may override `max_text_tokens_per_segment`, `inter_segment_silence_ms`, `max_generate_length`, and generation start/stop tokens; old manifests retain the defaults. Artifact manifest values are loaded first and explicit model metadata is applied second, so deployment metadata intentionally has final precedence. The checked-in catalog does not redundantly set generation values, allowing an artifact's exported safety limits (for example, its `max_generate_length`) to take effect. Every present canonical or alias field must be a representable integer or integer string; malformed values fail model loading rather than silently reverting to a default.

On CPU-only Arch Linux, IndexTTS sessions use sequential ORT graph execution and a bounded intra-op pool. `LOCAL_INDEXTTS_ORT_INTRA_THREADS` overrides its default of `min(logical CPUs, 8)`, and `LOCAL_INDEXTTS_ORT_INTER_THREADS` overrides the default `1`; invalid or zero values fail model loading clearly. These settings apply only to IndexTTS-created CPU sessions and make no Intel/CUDA assumptions. For a Ryzen 7 5800H (16 logical CPUs), start with:

```bash
LOCAL_INDEXTTS_ORT_INTRA_THREADS=8 LOCAL_INDEXTTS_ORT_INTER_THREADS=1 <service command>
```

Benchmark representative short, punctuation-rich long, and punctuation-free long inputs after warm-up, recording wall time, generated audio duration, real-time factor, and CPU utilization. Compare nearby intra-op values such as 6, 8, and 10 one at a time; topology, thermals, memory bandwidth, and ORT builds vary, so no throughput gain is promised without measurement on the target machine.

The Rust adapter frontend follows the official structure without vendoring pynini: `OfficialLike` ports `tokenize_by_CJK_char`/`de_tokenized_by_CJK_char`, official punctuation replacement maps/order, the official English contraction subset, tone-number pinyin protection/correction (`<pinyin_a>`, `ju4` -> `JV4`), name placeholders (`<n_a>`), TextTokenizer-style encode/decode and sentence split helpers. It deliberately leaves Hanzi as Hanzi by default (`你好` -> `你 好`). Placeholder names match official `a..z`; beyond 26 protected items Rust uses a collision-safe alphabetic extension instead of Python's `chr(ord('a') + i)` punctuation spillover. Lightweight TN now covers fullwidth ASCII, Chinese/Arabic digit runs, `YYYY年MM月DD日`, `YYYY/MM/DD`, `HH:MM` with optional AM/PM, percentages, currency signs, email protection, plus forms, and common units such as `km/h`, `km`, `kg`, `g`, `GB/MB`, `m/s`, and `℃`. Remaining gaps are concrete WeTextProcessing/pynini FST classes: exhaustive Chinese/English cardinal/ordinal morphology, phone/address/fraction rules, rich currency expressions (`RMB 20`, ranges, cents), context-sensitive abbreviation expansion, and locale-specific cases from the official TN graphs. The old deterministic Hanzi-to-pinyin behavior remains available only for experiments via `LOCAL_INDEXTTS_TEXT_FRONTEND=pinyin_explicit` or `preprocess_text_for_index_tts_with_mode(..., PinyinExplicit)`; it uses the `pinyin` crate (`with_tone_num_end`, no default feature set) and has the known single-reading/polyphone limitation.

For official parity research without starting services, run `python -m scripts.local.indextts_text_parity --text "你好 OpenAI"`. The helper imports the official source tree from `workdir/models/index-tts-v1.5`, prefers `workdir/models/IndexTTS-1.5/bpe.model` and falls back to `workdir/models/indextts-1.5-onnx/bpe.model`, runs the non-service Rust dump binary, and writes normalized/tokenized/token-id equality plus summary counts under `workdir/data/indextts-text-parity-<timestamp>.json`. It supports repeated `--text`, `--input-json` (including stdin with `-`), and `--batch-file`; use `--no-rust-frontend` to skip the Rust comparator. If dependencies are missing, normal runs still write a missing-dependency report with setup hints; use `--fail-on-missing` in CI. On Windows, `pynini`/WeTextProcessing installation is commonly limited, so prefer Linux/WSL or conda-forge (`conda install -c conda-forge pynini`, then install the official project requirements) when exact official TN is required.

Optional IndexTTS ASR cross-validation lives in the Python harness, not in ad-hoc curl scripts. Run `python -m scripts.local.smoke --tests indextts_asr --indextts-frontend auto --workdir ./workdir --model-dir ./workdir/models` or add `--indextts-asr-check` to an existing smoke run. The flow enables IndexTTS, uses the Rust frontend by default (official Python only when explicitly requested), synthesizes a WAV through generic `create_task`/upload/`start_task`, transcribes that WAV with the Qwen ASR generic task path, and saves `workdir/data/smoke-indextts-asr-<timestamp>.json` containing the source text, frontend mode, token-id source, normalized expected text, WAV path/URL, ASR text, simple similarity/coverage, and missing/extra character summaries.



## Legacy JSON-RPC API

The controller exposes the legacy JSON-RPC API on port `17890` only at canonical `POST /rpc/admin` for admin/model operations and canonical `POST /rpc/infer` for inference/task operations. `/rpc/admin` requires `LOCAL_ADMIN_TOKEN`; `/rpc/infer` is open when `LOCAL_MCP_INFER_TOKENS` is empty and otherwise accepts any token in that comma-separated list. The same inference-token policy guards OpenAI-compatible ASR, TTS, embeddings, and all rerank aliases; `GET /v1/models` remains an open catalog route. These legacy JSON-RPC routes are not the standard MCP protocol.

```json
{"jsonrpc":"2.0","id":1,"result":{}}
```

Errors use:

```json
{"jsonrpc":"2.0","id":1,"error":{"code":-32000,"message":"..."}}
```


The API accepts core `ModelSpec` JSON, not MCP/OpenAI-specific schemas. `list_models` and `get_model` add the computed `downloaded` and `download_state` fields without persisting those runtime fields into `ModelSpec`. `download_model` only queues background work and immediately returns `accepted`, `deduplicated`, and the aggregate status. Calls for the same model are deduplicated while a download is active, and calls for an already complete model do not start another task. Query `get_model_download_status` on either the admin MCP catalog or legacy `/rpc/admin` for aggregate and per-artifact state. A destination is reused only when it has a matching persisted completion record or passes its configured SHA-256 check; partial URL downloads are written to a side file and replaced only after the complete response is durable. Artifact configuration changes transactionally invalidate obsolete download rows, and successful files remain reusable when a failed multi-file download is retried.

## Standard MCP Streamable HTTP API

The controller also starts official SDK-backed standard MCP services on a separate bind: admin tools at `http://127.0.0.1:17892/mcp/admin` and inference tools at `http://127.0.0.1:17892/mcp/infer`. The catalogs are disjoint and cross-catalog `tools/call` requests are rejected. Override only the bind with `--mcp-bind` or `LOCAL_MCP_BIND`.


Authentication is shared across standard MCP, legacy RPC, and OpenAI-compatible inference routes. Admin always requires `LOCAL_ADMIN_TOKEN`. Inference optionally uses the comma-separated `LOCAL_MCP_INFER_TOKENS`; a configured non-empty list is enforced everywhere. Both support Bearer authentication plus their dedicated `x-local-admin-token` / `x-local-infer-token` headers. Keep the host publish loopback-only unless the surrounding network policy is intentional.

Validate a running controller with the official Python SDK client. Do not validate this endpoint with raw HTTP JSON-RPC; use official Python MCP SDK or rmcp client semantics for MCP protocol calls. Raw `urllib`/HTTP is used by the smoke client only for asset bytes uploaded/downloaded through signed URLs returned by MCP tools.

```bash
python -m scripts.local.mcp_standard_client --admin-token "$LOCAL_ADMIN_TOKEN" --full
```

The smoke harness has aliases that start controller/worker and run the same client or the legacy RPC helpers:

```bash
python -m scripts.local.smoke --tests rpc --workdir ./workdir --model-dir ./workdir/models
python -m scripts.local.smoke --tests mcp --workdir ./workdir --model-dir ./workdir/models
python -m scripts.local.smoke --tests all --workdir ./workdir --model-dir ./workdir/models
```




- `mcp` expands to standard MCP SDK coverage on `/mcp/admin` and `/mcp/infer`: authentication, isolated tool listings, admin/catalog/assets, generic task flow, and direct inference where local resources/artifacts are available.
- `all` expands both groups and still respects sensible skip flags.
- `qwen-asr` is the canonical Qwen ASR smoke alias.


```bash
python -m scripts.local.smoke --tests qwen-asr --workdir ./workdir --model-dir ./workdir/models
```

## Model catalog, SQLite, and workdir layout

Startup initializes a SQLite store at `workdir/data/local.db` by default and creates `workdir/models` for artifacts. The controller and worker accept `--workdir` and `--model-dir` while preserving the existing positional config path. Config files may also set `workdir`, `data_dir`, `database_path`, `model_dir`, and `models_conf_dir`.

Built-in defaults are code-defined in `local-registry` and seeded into SQLite first. YAML specs from `configs/models.d` are loaded afterwards and upsert by id, so they override or extend the built-in catalog. Existing database enabled/disabled state is preserved when built-ins are re-seeded, while YAML/admin upserts can explicitly change the enabled flag.

Model upserts are normalized by `SqliteModelStore` before persistence. The adapter-facing `ModelArtifact.path` is always forced under `model_dir/<model_id>/...`:

- Local external paths are preserved as `ModelArtifact.source_path` and imported/copied into the stable destination by `download_model`.
- Hugging Face multi-file artifacts use `model_dir/<model_id>` as the root and download matched/explicit files below it while preserving repo-relative file paths. Single-file HF artifacts use `model_dir/<model_id>/<filename>`.
- URL artifacts use `model_dir/<model_id>/<relative-path-or-url-basename>`.

Destination components reject absolute paths, `.` and `..` traversal. Absolute paths are accepted only as Local `source_path` values. `download_model` rewrites and persists the normalized spec before doing any artifact work, so subsequent controller/worker startups use stable paths.

Persisted tables cover:

- `models`: `ModelSpec` JSON plus an indexed enabled column.
- `artifact_downloads`: per-artifact download state, path, optional expected sha256, and message.
- `workers`: last registered/heartbeat `NodeStatus` JSON.
- `jobs`: minimal task/job state (`queued`, `running`, `succeeded`, `failed`) keyed by `InferenceTask.id`.

The controller depends on the store for metadata/status only. It still does not depend on runtime or adapters and does not load models.

## Default model choices

- ASR: `andrewleech/qwen3-asr-0.6b-onnx` at revision `4fc24a1402e74db89c4d2ef256875e71680128c4`; enabled because it is ONNX/ORT. The int4 file subset is downloaded into `<model_dir>/qwen3-asr-0.6b-onnx`. The real CPU ORT encoder/decoder/tokenizer path is implemented; real INT4 execution still depends on ORT contrib `MatMulNBits` support and should be verified with the `LOCAL_QWEN_ASR_MODEL_DIR`-gated smoke test.
- Object detection: `aaurelions/yolo11n.onnx` at revision `f46d9b72aa9a0f02bc00484446e2310b1a549bce`; enabled. The model file downloads to `<model_dir>/yolo11n.onnx/yolo11n.onnx`. COCO labels are a separate URL artifact from Ultralytics raw GitHub because the HF repository does not provide labels.
- TTS/IndexTTS: `ModaLeap/indextts-1.5-onnx`; disabled by default while the FP32 ORT adapter remains experimental. The explicit A-F ONNX, `bpe.model`, `manifest.yaml`, and `manifest.json` subset downloads to `<model_dir>/indextts-1.5-onnx`.

Remote downloads use Hugging Face resolve URLs or direct URLs. `HF_TOKEN`/`HUGGINGFACE_HUB_TOKEN` is used for Hugging Face metadata and file requests when present. Explicit HF `files` remain supported; `allow_patterns` are expanded by reading HF model metadata siblings and matching simple `*`/`?` globs. SHA-256 is verified only when configured; otherwise status explicitly records that verification was skipped. No Candle/Python/C++/sidecar path is implemented.

Release smoke examples:

```bash
cargo build --release --bins
python -m scripts.local.smoke --skip-build --release --tests mcp --workdir ./workdir --model-dir ./workdir/models --ready-timeout 60 --request-timeout 600
python -m scripts.local.smoke --skip-build --release --tests rpc --workdir ./workdir --model-dir ./workdir/models --ready-timeout 60 --request-timeout 600
```
