# Models

Every model is one YAML spec in [`configs/providers/<category>/`](../../configs/providers).
The controller and worker read that directory at startup (`providers_conf_dir`
in `configs/{controller,worker}.yaml`) and keep the catalog in SQLite.

| Category | Doc | Specs |
| --- | --- | --- |
| Speech recognition | [speech-recognition.md](speech-recognition.md) | [`asr/`](../../configs/providers/asr) |
| Speech synthesis | [speech-synthesis.md](speech-synthesis.md) | [`tts/`](../../configs/providers/tts) |
| Chat | [chat.md](chat.md) | [`chat/`](../../configs/providers/chat) |
| Embedding and reranking | [embedding-rerank.md](embedding-rerank.md) | [`embedding/`](../../configs/providers/embedding), [`rerank/`](../../configs/providers/rerank) |
| Vision (detection, depth) | [vision.md](vision.md) | [`detect/`](../../configs/providers/detect) |
| OCR | [ocr.md](ocr.md) | [`ocr/`](../../configs/providers/ocr) |
| Content moderation | [moderation.md](moderation.md) | [`moderation/`](../../configs/providers/moderation) |
| Realtime voice | [realtime-voice.md](realtime-voice.md) | [`realtime/`](../../configs/providers/realtime) |

## Getting a model

Each spec's `artifacts` say where its files come from:

- `hugging_face` / `url`: published files, pinned to a commit (URL artifacts
  also by SHA-256 and size). Download them through the admin API, then the
  model loads on first use:

  ```bash
  curl --fail-with-body http://127.0.0.1:17890/rpc/admin \
    -H 'content-type: application/json' -H "x-local-admin-token: $LOCAL_ADMIN_TOKEN" \
    --data '{"jsonrpc":"2.0","id":1,"method":"download_model","params":{"id":"sensevoice-small-onnx"}}'
  # progress, per file:
  curl --fail-with-body http://127.0.0.1:17890/rpc/admin \
    -H 'content-type: application/json' -H "x-local-admin-token: $LOCAL_ADMIN_TOKEN" \
    --data '{"jsonrpc":"2.0","id":2,"method":"get_model_download_status","params":{"id":"sensevoice-small-onnx"}}'
  ```

  The same methods are admin MCP tools (`download_model`,
  `get_model_download_status`, `list_models`). `HF_TOKEN` is used when set.
  Files already in place count without a download: status checks that they
  exist and have their pinned size, never hashes them (models are large);
  `download_model` verifies SHA-256 when it writes or reuses a file, and
  replaces a file of the wrong size.
- `local`: there is no published ONNX package. Export it from the upstream
  checkpoint with the command in the model's doc (and in the spec's comments)
  into `workdir/models/<model id>/`. Nothing needs downloading afterwards.

Files always land in `<model_dir>/<model id>/` (`workdir/models` by default).

## Spec fields

```yaml
id: sensevoice-small-onnx          # the name requests use (`model`)
name: FunASR Full ASR ONNX
enabled: true                      # also: enable_model / disable_model admin calls
task_kinds: [asr.transcribe]
adapter: sense_voice_asr           # the Rust adapter that runs it
backend: ort
artifacts: [...]                   # see above
runtime:
  provider_order: [cuda, cpu]      # tried in order; see below
  max_concurrency: 1               # requests run at once on this model
  idle_ttl_sec: 300                # unload after this long unused
resources:
  min_ram_mb: 1536
  min_vram_mb: 0
load_policy:
  lazy: true                       # load on first request
  evictable: true
  pin: false
metadata:                          # adapter-specific knobs, listed in each model doc
  auto_select: false               # only when a request names the model
```

- **Providers.** `provider_order` is filtered against what the worker can
  actually use: a worker built without `--features cuda`, or whose CUDA
  provider fails a one-time probe, runs everything on the CPU. If a model's
  CUDA session fails to load, the next provider is tried. Models marked
  "CUDA only" below refuse the CPU rather than give wrong results.
- **Default model of a task.** A request that names no model gets an enabled
  model of that task kind, never one with `metadata.auto_select: false`.
- **Overriding.** Edit the YAML, or add another spec file with the same `id`:
  YAML specs upsert by id over the built-in catalog at every startup.
- **Worker builds.** Each category is a cargo feature of the worker (all on
  by default); an audio-only worker is
  `cargo build --release --bins --no-default-features --features audio,cuda`.

## Licenses

Each model keeps its upstream license, named in its doc. The upstream code
this project ports or follows is listed in the README's [acknowledgements](../../README.en.md#acknowledgements).
