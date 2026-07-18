# Implementation notes

## ORT-only boundary

This MVP intentionally implements only the ORT backend seam. Candle, Python, C++, sidecar, and external-process alternatives are not implemented. If a path would require a non-ORT backend it must return `Unsupported`, `NeedUserConfirmation`, or `NeedImplementation`.

The `ort` crate is configured with downloaded/copied binaries so build/check does not require a system-wide ONNX Runtime install. The CPU Dockerfile resolves the CPU distribution; `Dockerfile.nvidia` enables the CUDA feature and defaults `ORT_CUDA_VERSION` to `12`. The backend remains ORT-specific at the API/config layer; CUDA/DML provider features are opt-in and must be validated against the active ORT execution providers before use.


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
with `cargo build --locked --release -p local-cli --bin worker --features cuda`;
its `ORT_CUDA_VERSION` build argument defaults to `12`. The controller uses the
ordinary CPU Dockerfile/image and has no GPU request. Only the worker declares
Compose `gpus: all`, which requires Docker Compose 2.30.0 or newer; check with
`docker compose version`. The worker is constrained to Linux x86_64 because
that is the CUDA target published in rc.12's distribution table.

The locked `ort-sys` 2.0.0-rc.12 package embeds an exact distribution row for
`ms@1.24.2/x86_64-unknown-linux-gnu+cu12` (SHA-256
`6e7848acdb7284feb44e2781583a90e820839767459ad8fa2abf7dd63b731fd9`).
Inspection of that provider shows dependencies on `libcudart.so.12`,
`libcublas.so.12`, `libcublasLt.so.12`, `libcufft.so.11`,
`libcurand.so.10`, and `libcudnn.so.9`, with a maximum observed glibc symbol
version of `GLIBC_2.38`. Therefore the runtime is pinned to NVIDIA's real full
tag `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04` and manifest digest
`sha256:ac55d124da4882b497f732d8dfd9a702d5447a5f29d08d56da6f64f0a1eb34bc`.
It supplies CUDA 12, cuDNN 9, and Ubuntu 24.04 glibc 2.39. ONNX Runtime's CUDA
EP documentation states that CUDA 12.x builds are compatible within the CUDA
12 major family while cuDNN major versions must match.

The rc.12 Linux CUDA archive contains a static `libonnxruntime.a`, not a
runtime `libonnxruntime.so`; the worker therefore contains the ORT core through
static linking. The final image checks the worker itself and the dynamically
loaded CUDA provider with `ldd`, and fails the build if either reports a missing
library. It also requires `libonnxruntime_providers_cuda.so` to be present.

Evidence:

- rc.12 package distribution metadata: <https://docs.rs/crate/ort-sys/2.0.0-rc.12/source/build/download/dist.txt>
- ONNX Runtime CUDA/cuDNN compatibility: <https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements>
- Official NVIDIA image/tag and Container Toolkit requirement: <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/12.8.1-cudnn-runtime-ubuntu24.04>
- NVIDIA CUDA minor-version/driver compatibility: <https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html>

Reproducible metadata inspection used for this choice:

```bash
# From the ort-sys 2.0.0-rc.12 crate source:
grep '^cu12.*x86_64-unknown-linux-gnu' ort-sys-2.0.0-rc.12/build/download/dist.txt
# Download the URL printed above and verify its embedded checksum:
sha256sum x86_64-unknown-linux-gnu+cu12.tar.lzma2
# ort-sys uses raw LZMA2 with a 64 MiB dictionary:
python -c "import lzma,pathlib; p=pathlib.Path('x86_64-unknown-linux-gnu+cu12.tar.lzma2'); pathlib.Path('ort-cu12.tar').write_bytes(lzma.decompress(p.read_bytes(),format=lzma.FORMAT_RAW,filters=[{'id':lzma.FILTER_LZMA2,'dict_size':1<<26}]))"
tar -xf ort-cu12.tar
readelf -d libonnxruntime_providers_cuda.so | grep NEEDED
readelf --version-info libonnxruntime_providers_cuda.so | grep -o 'GLIBC_[0-9.]*' | sort -V | tail -1
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
placement. IndexTTS CUDA policy is enabled because all seven A, B, C, D, E,
E-prefill, and F sessions
are created from the same provider selection and no concrete code/operator
blocker is known; real NVIDIA artifact smoke remains unverified. TensorRT is not
built or configured. The control-plane hardware snapshot still reports
`has_cuda: false` because it does not probe NVML; that reporting limitation is
independent of actual ORT EP selection.


## Qwen ASR limitations

The adapter validates the known `qwen3-asr-0.6b-onnx` artifact layout and establishes interfaces for WAV read/resampling, 128-bin feature extraction, tokenizer JSON loading, embeddings/KV-cache, and decoder loop orchestration. INT4 artifacts may require ORT contrib/custom-op support for `MatMulNBits`; use `LOCAL_QWEN_ASR_MODEL_DIR=<model-dir> cargo test -p local-adapter-qwen-asr real_model_smoke_if_env_set -- --nocapture` as an opt-in real-artifact smoke test.


## IndexTTS FP32 and text normalization boundary

IndexTTS ONNX support uses root FP32 artifacts with CUDA-first, CPU-fallback intent. The default catalog downloads the explicit `IndexTTS_A.onnx` through `IndexTTS_F.onnx`, including the separate E-prefill graph, `bpe.model`, and manifest files from `ModaLeap/indextts-1.5-onnx` into `workdir/models/indextts-1.5-onnx`; export/package tooling can also write the same root layout. Runtime validation loads that root directly and no longer auto-selects `fp16/` for CUDA or `q4/` for CPU. Existing `q4/` or `fp16/` model caches may remain on disk but are ignored by current code and docs. All seven A, B, C, D, E, E-prefill, and F sessions are loaded from one `OrtBackend` built from `spec.runtime.provider_order` and are included in its provider report; code/policy support is present, while real NVIDIA hardware validation is not.

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

The controller exposes the legacy JSON-RPC API on port `17890` only at canonical `POST /rpc/admin` for admin/model operations and canonical `POST /rpc/infer` for inference/task operations. Legacy JSON-RPC `/mcp/admin` and `/mcp/infer` compatibility aliases are removed and must remain absent. These legacy JSON-RPC routes are not the standard MCP protocol. Requests and responses use JSON-RPC 2.0, for example:

```json
{"jsonrpc":"2.0","id":1,"result":{}}
```

Errors use:

```json
{"jsonrpc":"2.0","id":1,"error":{"code":-32000,"message":"..."}}
```


The API accepts core `ModelSpec` JSON, not MCP/OpenAI-specific schemas. `download_model` returns persisted per-artifact `DownloadStatus` values.

## Standard MCP Streamable HTTP API

The controller also starts an official SDK-backed standard MCP server on a separate bind to avoid path/protocol conflicts with the legacy JSON-RPC endpoints. The default and only documented local endpoint is `http://127.0.0.1:17892/mcp`; override the bind only for controlled local development with `--mcp-bind <addr>`, `--mcp-bind=<addr>`, or `LOCAL_MCP_BIND`.


Security boundary: the standard MCP endpoint exposes admin and mutating tools. Keep the bind loopback-only for local automation (`127.0.0.1:17892` or `[::1]:17892`). Do **not** bind it to `0.0.0.0:17892` or another shared-network address. Any future non-loopback deployment must add an admin token and/or ACL in front of the standard MCP transport before exposing these tools.

Validate a running controller with the official Python SDK client. Do not validate this endpoint with raw HTTP JSON-RPC; use official Python MCP SDK or rmcp client semantics for MCP protocol calls. Raw `urllib`/HTTP is used by the smoke client only for asset bytes uploaded/downloaded through signed URLs returned by MCP tools.

```bash
python -m scripts.local.mcp_standard_client --url http://127.0.0.1:17892/mcp --full
```

The smoke harness has aliases that start controller/worker and run the same client or the legacy RPC helpers:

```bash
python -m scripts.local.smoke --tests rpc --workdir ./workdir --model-dir ./workdir/models
python -m scripts.local.smoke --tests mcp --workdir ./workdir --model-dir ./workdir/models
python -m scripts.local.smoke --tests all --workdir ./workdir --model-dir ./workdir/models
```




- `mcp` expands to standard MCP SDK coverage on `http://127.0.0.1:17892/mcp`: tool listing, admin/catalog/assets, generic task flow, and direct inference where local resources/artifacts are available.
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
