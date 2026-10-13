# API

The controller serves every interface; requests are forwarded to a worker that
has the model.

| Interface | Address | Auth |
| --- | --- | --- |
| Standard MCP, inference tools | `http://127.0.0.1:17892/mcp/infer` | inference token, if configured |
| Standard MCP, admin tools | `http://127.0.0.1:17892/mcp/admin` | admin token |
| OpenAI-compatible | `http://127.0.0.1:17890/v1` | inference token, if configured (`GET /v1/models` is open) |
| JSON-RPC (legacy) | `POST /rpc/infer`, `POST /rpc/admin` | inference / admin token |

**Tokens.** `LOCAL_ADMIN_TOKEN` is always required for admin calls.
`LOCAL_MCP_INFER_TOKENS` (comma-separated) guards every inference route when
it is not empty. Send either as `Authorization: Bearer <token>`; JSON-RPC and
the OpenAI routes also take `x-local-infer-token`, admin routes
`x-local-admin-token`.

## OpenAI-compatible and HTTP endpoints

| Endpoint | Task | Models |
| --- | --- | --- |
| `POST /v1/chat/completions` | chat, streaming (SSE), tool calls | [chat](models/chat.md) |
| `POST /v1/embeddings` | text embedding | [embedding](models/embedding-rerank.md) |
| `POST /rerank`, `/v1/rerank`, `/v2/rerank` | reranking (vLLM / Jina / Cohere style) | [rerank](models/embedding-rerank.md) |
| `POST /v1/audio/transcriptions` | speech recognition (JSON with a file path, returns `{"text"}`) | [ASR](models/speech-recognition.md) |
| `POST /v1/audio/speech` | speech synthesis (JSON, returns a WAV file ref) | [TTS](models/speech-synthesis.md) |
| `POST /v1/moderations` | text and image moderation | [moderation](moderation.md) |
| `GET /v1/realtime` | realtime voice WebSocket | [realtime voice](realtime-voice.md) |
| `POST /v1/ocr/lines` | text lines; image as the body | [OCR](models/ocr.md) |
| `POST /v1/detect/objects` | object detection; image as the body | [vision](models/vision.md) |
| `POST /v1/depth` | metric depth grid; image as the body | [vision](models/vision.md) |
| `GET /v1/inferences` | recent inferences and their timings | below |
| `GET /v1/models` | model list | |

Only these are implemented; it is not the full OpenAI API. Point an OpenAI
client at `http://127.0.0.1:17890/v1` with an inference token as the API key.

## MCP for agents

Give an agent the inference catalog only; add the admin catalog as a separate
server only if it must download or enable models.

```json
{
  "mcpServers": {
    "local-multimodal": {
      "type": "streamable-http",
      "url": "http://127.0.0.1:17892/mcp/infer",
      "headers": { "Authorization": "Bearer <inference token>" }
    }
  }
}
```

Drop `headers` when `LOCAL_MCP_INFER_TOKENS` is empty. Never use the admin
token for the inference server.

- **Inference tools:** `object_detect`, `ocr_recognize`, `asr_transcribe`,
  `tts_synthesize`, `text_embed`, `text_rerank`, and the generic task tools
  `create_task`, `start_task`, `get_task`, `wait_task`, `run_task`,
  `sign_assets`.
- **Admin tools:** `list_models`, `get_model`, `add_model`, `upsert_model`,
  `download_model`, `get_model_download_status`, `enable_model`,
  `disable_model`, `list_nodes`, `get_cluster_status`, `list_assets`,
  `search_assets`, `sign_asset_urls`.

**Files the agent cannot share by path** (images, audio): `create_task` →
upload the bytes to the returned signed URL → `start_task` → `wait_task`. The
worker never needs the agent's file system.

**Large results.** Tools that return inference results take `with_url_result`:

- `auto` (default): results up to 1000 bytes of UTF-8 are returned inline;
  longer ones as a 1000-byte preview plus a download URL.
- `on`: always a preview plus a URL. `off`: always the whole result inline.

URL results are saved as `.txt` artifacts (`preview`, `truncated`,
`download_url`, `artifact_uri`, `size_bytes`, `sha256`, `expires_at`), kept 24
hours (identical content reuses and renews one artifact); signed URLs last 10
minutes, and calling again gives a new one.

Check a running controller with the official Python SDK:
`python -m scripts.local.mcp_standard_client --admin-token <token> --full`.

## JSON-RPC (legacy)

`POST /rpc/infer` takes the inference tools above as methods (plus the
generic task methods); `POST /rpc/admin` the admin ones. Requests are plain
JSON-RPC 2.0:

```bash
curl --fail-with-body http://127.0.0.1:17890/rpc/admin \
  -H 'content-type: application/json' -H "x-local-admin-token: $LOCAL_ADMIN_TOKEN" \
  --data '{"jsonrpc":"2.0","id":1,"method":"list_models","params":{}}'
```

Generic task flow: `create_task` → raw-byte `POST` to
`/files/upload/<task_id>/<slot>?expires=...&sig=...` → `start_task` /
`wait_task`. Task kinds: `asr.transcribe`, `tts.synthesize`, `chat.complete`,
`text.embed`, `text.rerank`, `object.detect`, `depth.estimate`,
`ocr.recognize`, `ocr.lines`, `text.moderate`, `image.nsfw`.

## Recent inferences

`GET /v1/inferences[?limit=100&kind=&exclude=]` (inference token) lists the
latest inferences of every worker (each keeps 1000; 100 returned by default):
task kind, model, start and end (Unix ms), total time and stages (queue, load,
run, first output). Each realtime voice reply adds a `voice.turn` record from
the end of speech to the reply's first audio, split into `vad`, `asr`, `gate`
(wake word decision), `llm` (first text) and `tts` (first audio). `kind` keeps
only the given kinds, `exclude` drops kinds or models (comma-separated), such
as `exclude=ocr.lines`.
