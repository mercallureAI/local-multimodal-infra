# Implementation notes

## ORT-only boundary

This MVP intentionally implements only the ORT backend seam. Candle, Python, C++, sidecar, and external-process alternatives are not implemented. If a path would require a non-ORT backend it must return `Unsupported`, `NeedUserConfirmation`, or `NeedImplementation`.

The `ort` crate is configured with downloaded/copied CPU binaries so build/check does not require a system-wide ONNX Runtime install. The backend remains ORT-specific at the API/config layer; CUDA/DML provider features are opt-in and must be validated against the local ORT execution providers before use.


## Providers: CPU, CUDA, DML, TensorRT

`backend-ort` exposes `ProviderKind::{Cpu,Cuda,Dml,Trt}`, `ProviderOptions`, and `ProviderSelection`.

- CPU: supported as the default and primary provider in built-in and checked-in YAML model specs.
- CUDA: configurable only as an opt-in provider after building the backend feature and validating ORT CUDA EP availability. CUDA failures return an explicit reason and fall back to CPU only when CPU fallback is configured.
- DML: configurable as an opt-in provider on Windows builds with the backend feature enabled. Failures return an explicit reason and can fall back to CPU when configured.
- TensorRT: optional behind a backend cargo feature. Parser spellings `trt` and `tensorrt` are accepted. Session loading mirrors CUDA/DML in the ORT backend, but the runtime does **not** yet model a same-session TensorRT+CUDA stack, so provider order should usually be `[trt, cuda, cpu]` rather than expecting both EPs to coexist within one session registration.

Model/provider differences are handled by model config `runtime.provider_order`. The checked-in Qwen ASR, YOLO, and IndexTTS specs keep `[cpu]` on disk by policy. At runtime, the executor may derive an effective provider order for those known integrated model ids when the stored order is exactly `[cpu]`: it prefers only providers that are both conservatively validated for that model in this repo and actually available in the active ORT runtime process (not merely enabled by cargo features), while leaving the stored spec/YAML unchanged and preserving any explicit non-default order.


## Qwen ASR limitations

The adapter validates the known `qwen3-asr-0.6b-onnx` artifact layout and establishes interfaces for WAV read/resampling, 128-bin feature extraction, tokenizer JSON loading, embeddings/KV-cache, and decoder loop orchestration. INT4 artifacts may require ORT contrib/custom-op support for `MatMulNBits`; use `LCOAL_QWEN_ASR_MODEL_DIR=<model-dir> cargo test -p lcoal-adapter-qwen-asr real_model_smoke_if_env_set -- --nocapture` as an opt-in real-artifact smoke test.


## IndexTTS FP32 and text normalization boundary

IndexTTS ONNX support is FP32-only. The default catalog downloads the explicit `IndexTTS_A.onnx` through `IndexTTS_F.onnx`, `bpe.model`, and manifest files from `ModaLeap/indextts-1.5-onnx` into `workdir/models/indextts-1.5-onnx`; export/package tooling can also write the same root layout. Runtime validation loads that root directly and no longer auto-selects `fp16/` for CUDA or `q4/` for CPU. Existing `q4/` or `fp16/` model caches may remain on disk but are ignored by current code and docs.

The official IndexTTS 1.5 frontend (`workdir/models/index-tts-v1.5/indextts/utils/front.py` and `common.py`) uses WeTextProcessing/pynini TN when available, but its tokenizer path does **not** convert arbitrary Hanzi to pinyin. It protects explicit tone-number pinyin and Chinese-name placeholders around TN, expands a small English `'s` contraction pattern, applies a punctuation replacement map, then calls `tokenize_by_CJK_char`, which splits each CJK character and uppercases non-CJK segments before SentencePiece.

The default runtime path is the Rust frontend plus local SentencePiece. Explicit token ids (`text_token_ids`, `pretokenized_text_ids`, or `indextts_text_token_ids`) remain available for oracle/debug use; when present, the adapter validates a non-empty integer list in a sane range and feeds those ids to graph B directly, skipping local tokenization but preserving the A/B/C/D/E/F graph flow.

The Rust adapter frontend follows the official structure without vendoring pynini: `OfficialLike` ports `tokenize_by_CJK_char`/`de_tokenized_by_CJK_char`, official punctuation replacement maps/order, the official English contraction subset, tone-number pinyin protection/correction (`<pinyin_a>`, `ju4` -> `JV4`), name placeholders (`<n_a>`), TextTokenizer-style encode/decode and sentence split helpers. It deliberately leaves Hanzi as Hanzi by default (`你好` -> `你 好`). Placeholder names match official `a..z`; beyond 26 protected items Rust uses a collision-safe alphabetic extension instead of Python's `chr(ord('a') + i)` punctuation spillover. Lightweight TN now covers fullwidth ASCII, Chinese/Arabic digit runs, `YYYY年MM月DD日`, `YYYY/MM/DD`, `HH:MM` with optional AM/PM, percentages, currency signs, email protection, plus forms, and common units such as `km/h`, `km`, `kg`, `g`, `GB/MB`, `m/s`, and `℃`. Remaining gaps are concrete WeTextProcessing/pynini FST classes: exhaustive Chinese/English cardinal/ordinal morphology, phone/address/fraction rules, rich currency expressions (`RMB 20`, ranges, cents), context-sensitive abbreviation expansion, and locale-specific cases from the official TN graphs. The old deterministic Hanzi-to-pinyin behavior remains available only for experiments via `LCOAL_INDEXTTS_TEXT_FRONTEND=pinyin_explicit` or `preprocess_text_for_index_tts_with_mode(..., PinyinExplicit)`; it uses the `pinyin` crate (`with_tone_num_end`, no default feature set) and has the known single-reading/polyphone limitation.

For official parity research without starting services, run `python -m scripts.lcoal.indextts_text_parity --text "你好 OpenAI"`. The helper imports the official source tree from `workdir/models/index-tts-v1.5`, prefers `workdir/models/IndexTTS-1.5/bpe.model` and falls back to `workdir/models/indextts-1.5-onnx/bpe.model`, runs the non-service Rust dump binary, and writes normalized/tokenized/token-id equality plus summary counts under `workdir/data/indextts-text-parity-<timestamp>.json`. It supports repeated `--text`, `--input-json` (including stdin with `-`), and `--batch-file`; use `--no-rust-frontend` to skip the Rust comparator. If dependencies are missing, normal runs still write a missing-dependency report with setup hints; use `--fail-on-missing` in CI. On Windows, `pynini`/WeTextProcessing installation is commonly limited, so prefer Linux/WSL or conda-forge (`conda install -c conda-forge pynini`, then install the official project requirements) when exact official TN is required.

Optional IndexTTS ASR cross-validation lives in the Python harness, not in ad-hoc curl scripts. Run `python -m scripts.lcoal.smoke --tests indextts_asr --indextts-frontend auto --workdir ./workdir --model-dir ./workdir/models` or add `--indextts-asr-check` to an existing smoke run. The flow enables IndexTTS, uses the Rust frontend by default (official Python only when explicitly requested), synthesizes a WAV through generic `create_task`/upload/`start_task`, transcribes that WAV with the Qwen ASR generic task path, and saves `workdir/data/smoke-indextts-asr-<timestamp>.json` containing the source text, frontend mode, token-id source, normalized expected text, WAV path/URL, ASR text, simple similarity/coverage, and missing/extra character summaries.



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

The controller also starts an official SDK-backed standard MCP server on a separate bind to avoid path/protocol conflicts with the legacy JSON-RPC endpoints. The default and only documented local endpoint is `http://127.0.0.1:17892/mcp`; override the bind only for controlled local development with `--mcp-bind <addr>`, `--mcp-bind=<addr>`, or `LCOAL_MCP_BIND`.


Security boundary: the standard MCP endpoint exposes admin and mutating tools. Keep the bind loopback-only for local automation (`127.0.0.1:17892` or `[::1]:17892`). Do **not** bind it to `0.0.0.0:17892` or another shared-network address. Any future non-loopback deployment must add an admin token and/or ACL in front of the standard MCP transport before exposing these tools.

Validate a running controller with the official Python SDK client. Do not validate this endpoint with raw HTTP JSON-RPC; use official Python MCP SDK or rmcp client semantics for MCP protocol calls. Raw `urllib`/HTTP is used by the smoke client only for asset bytes uploaded/downloaded through signed URLs returned by MCP tools.

```bash
python -m scripts.lcoal.mcp_standard_client --url http://127.0.0.1:17892/mcp --full
```

The smoke harness has aliases that start controller/worker and run the same client or the legacy RPC helpers:

```bash
python -m scripts.lcoal.smoke --tests rpc --workdir ./workdir --model-dir ./workdir/models
python -m scripts.lcoal.smoke --tests mcp --workdir ./workdir --model-dir ./workdir/models
python -m scripts.lcoal.smoke --tests all --workdir ./workdir --model-dir ./workdir/models
```




- `mcp` expands to standard MCP SDK coverage on `http://127.0.0.1:17892/mcp`: tool listing, admin/catalog/assets, generic task flow, and direct inference where local resources/artifacts are available.
- `all` expands both groups and still respects sensible skip flags.
- `qwen-asr` is the canonical Qwen ASR smoke alias.


```bash
python -m scripts.lcoal.smoke --tests qwen-asr --workdir ./workdir --model-dir ./workdir/models
```

## Model catalog, SQLite, and workdir layout

Startup initializes a SQLite store at `workdir/data/lcoal.db` by default and creates `workdir/models` for artifacts. The controller and worker accept `--workdir` and `--model-dir` while preserving the existing positional config path. Config files may also set `workdir`, `data_dir`, `database_path`, `model_dir`, and `models_conf_dir`.

Built-in defaults are code-defined in `lcoal-registry` and seeded into SQLite first. YAML specs from `configs/models.d` are loaded afterwards and upsert by id, so they override or extend the built-in catalog. Existing database enabled/disabled state is preserved when built-ins are re-seeded, while YAML/admin upserts can explicitly change the enabled flag.

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

- ASR: `andrewleech/qwen3-asr-0.6b-onnx` at revision `4fc24a1402e74db89c4d2ef256875e71680128c4`; enabled because it is ONNX/ORT. The int4 file subset is downloaded into `<model_dir>/qwen3-asr-0.6b-onnx`. The real CPU ORT encoder/decoder/tokenizer path is implemented; real INT4 execution still depends on ORT contrib `MatMulNBits` support and should be verified with the `LCOAL_QWEN_ASR_MODEL_DIR`-gated smoke test.
- Object detection: `aaurelions/yolo11n.onnx` at revision `f46d9b72aa9a0f02bc00484446e2310b1a549bce`; enabled. The model file downloads to `<model_dir>/yolo11n.onnx/yolo11n.onnx`. COCO labels are a separate URL artifact from Ultralytics raw GitHub because the HF repository does not provide labels.
- TTS/IndexTTS: `ModaLeap/indextts-1.5-onnx`; disabled by default while the FP32 ORT adapter remains experimental. The explicit A-F ONNX, `bpe.model`, `manifest.yaml`, and `manifest.json` subset downloads to `<model_dir>/indextts-1.5-onnx`.

Remote downloads use Hugging Face resolve URLs or direct URLs. `HF_TOKEN`/`HUGGINGFACE_HUB_TOKEN` is used for Hugging Face metadata and file requests when present. Explicit HF `files` remain supported; `allow_patterns` are expanded by reading HF model metadata siblings and matching simple `*`/`?` globs. SHA-256 is verified only when configured; otherwise status explicitly records that verification was skipped. No Candle/Python/C++/sidecar path is implemented.

Release smoke examples:

```bash
cargo build --release --bins
python -m scripts.lcoal.smoke --skip-build --release --tests mcp --workdir ./workdir --model-dir ./workdir/models --ready-timeout 60 --request-timeout 600
python -m scripts.lcoal.smoke --skip-build --release --tests rpc --workdir ./workdir --model-dir ./workdir/models --ready-timeout 60 --request-timeout 600
```
