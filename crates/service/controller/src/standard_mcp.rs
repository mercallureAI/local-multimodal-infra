use super::*;
use rmcp::{
    model::*,
    service::RequestContext,
    transport::streamable_http_server::{
        session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
    },
    ErrorData, RoleServer, ServerHandler,
};
use serde::Deserialize;
use serde_json::{Map, Value};

const MCP_RESULT_PREVIEW_MAX_BYTES: usize = 1_000;

const INFER_INSTRUCTIONS: &str = "Local multimodal inference as tasks. \
1. list_models: the installed models, their task kinds, and what each task kind takes (files, params) and returns. \
2. Without files: run_task {task_kind, model, params} returns the result. \
3. With files: create_task {task_kind, model, files, params}, POST each file's raw bytes to its uploads[].upload_url, then start_task {task_id} (it waits and returns the result). \
Alternatively upload through sign_assets and pass the returned uri as files[].asset_uri to run_task. \
Results are in output (output.type names its shape); generated files (speech audio) carry a download URL. \
Long results come back as a preview plus download_url unless with_url_result is \"off\".";

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
enum WithUrlResult {
    #[default]
    Auto,
    On,
    Off,
}

impl ControllerState {
    pub fn standard_mcp_app(self) -> Router {
        let admin = standard_mcp_router(
            "/mcp/admin",
            self.clone(),
            McpAccess::Admin,
            ApiAuth::Required {
                tokens: self.admin_token.clone().into_iter().collect(),
                header_name: "x-local-admin-token",
                missing_configuration: true,
            },
        );
        let infer_auth = ApiAuth::inference(&self.mcp_infer_tokens);
        let infer = standard_mcp_router("/mcp/infer", self, McpAccess::Infer, infer_auth);
        Router::new().merge(admin).merge(infer)
    }
}

fn standard_mcp_router(
    path: &'static str,
    state: ControllerState,
    access: McpAccess,
    auth: ApiAuth,
) -> Router {
    let server = StandardMcpServer { state, access };
    let mut config = StreamableHttpServerConfig::default();
    config.stateful_mode = false;
    let service = StreamableHttpService::new(
        move || Ok(server.clone()),
        LocalSessionManager::default().into(),
        config,
    );
    Router::new()
        .nest_service(path, service)
        .layer(axum::middleware::from_fn_with_state(auth, authorize_api))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum McpAccess {
    Admin,
    Infer,
}

#[derive(Clone)]
pub(super) enum ApiAuth {
    Open,
    Required {
        tokens: Vec<String>,
        header_name: &'static str,
        missing_configuration: bool,
    },
}

impl ApiAuth {
    pub(super) fn inference(tokens: &[String]) -> Self {
        if tokens.is_empty() {
            Self::Open
        } else {
            Self::Required {
                tokens: tokens.to_vec(),
                header_name: "x-local-infer-token",
                missing_configuration: false,
            }
        }
    }
}

pub(super) async fn authorize_api(
    State(auth): State<ApiAuth>,
    headers: HeaderMap,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let ApiAuth::Required {
        tokens,
        header_name,
        missing_configuration,
    } = auth
    else {
        return next.run(request).await;
    };
    if missing_configuration && tokens.is_empty() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({ "error": "admin token is not configured" })),
        )
            .into_response();
    }
    let provided = headers
        .get(header_name)
        .and_then(|value| value.to_str().ok())
        .or_else(|| bearer_token(&headers));
    if provided.is_some_and(|provided| tokens.iter().any(|token| token == provided)) {
        next.run(request).await
    } else {
        (
            StatusCode::UNAUTHORIZED,
            [(header::WWW_AUTHENTICATE, "Bearer")],
            Json(json!({ "error": "missing or invalid API token" })),
        )
            .into_response()
    }
}

fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
}

#[derive(Clone)]
struct StandardMcpServer {
    state: ControllerState,
    access: McpAccess,
}

impl ServerHandler for StandardMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::default(),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: format!("local-controller-{}-mcp", self.access.label()),
                title: Some(format!("local Controller {} MCP", self.access.label())),
                version: env!("CARGO_PKG_VERSION").to_string(),
                description: Some("Standard MCP adapter for the local controller".to_string()),
                icons: None,
                website_url: None,
            },
            instructions: Some(match self.access {
                McpAccess::Infer => INFER_INSTRUCTIONS.to_string(),
                McpAccess::Admin => {
                    "Standard MCP admin tools for the local controller: models, downloads, nodes and assets."
                        .to_string()
                }
            }),
        }
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> std::result::Result<ListToolsResult, ErrorData> {
        Ok(ListToolsResult {
            tools: tool_definitions(self.access),
            next_cursor: None,
            meta: None,
        })
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> std::result::Result<CallToolResult, ErrorData> {
        let arguments = Value::Object(request.arguments.unwrap_or_default());
        match self.call_tool_json(request.name.as_ref(), arguments).await {
            Ok(value) => Ok(success_json(value)?),
            Err(err) if is_unknown_tool_error(&err) => Err(ErrorData::new(
                ErrorCode::METHOD_NOT_FOUND,
                err.to_string(),
                None,
            )),
            Err(err) => Ok(CallToolResult::error(vec![Content::text(err.to_string())])),
        }
    }

    fn get_tool(&self, name: &str) -> Option<Tool> {
        tool_definitions(self.access)
            .into_iter()
            .find(|tool| tool.name.as_ref() == name)
    }
}

impl StandardMcpServer {
    async fn call_tool_json(&self, name: &str, arguments: Value) -> Result<Value> {
        if !self.access.allows(name) {
            return Err(InfraError::BadRequest(format!("unknown MCP tool `{name}`")));
        }
        let (with_url_result, arguments) = result_url_policy(name, arguments)?;
        let value = match name {
            "create_task" => {
                let request = serde_json::from_value::<CreateTaskRequest>(arguments)?;
                Ok(json!(
                    InferenceApi::create_task(&self.state, request).await?
                ))
            }
            "start_task" => {
                let request = serde_json::from_value::<StartTaskRequest>(arguments)?;
                Ok(json!(InferenceApi::start_task(&self.state, request).await?))
            }
            "get_task" => {
                let task_id = task_id_param(&arguments)?;
                Ok(json!(InferenceApi::get_task(&self.state, task_id).await?))
            }
            "wait_task" => {
                let request = serde_json::from_value::<WaitTaskRequest>(arguments)?;
                Ok(json!(InferenceApi::wait_task(&self.state, request).await?))
            }
            "run_task" => {
                let request = serde_json::from_value::<CreateTaskRequest>(arguments)?;
                Ok(json!(InferenceApi::run_task(&self.state, request).await?))
            }
            "sign_assets" => {
                let request = serde_json::from_value::<AssetSignRequest>(arguments)?;
                Ok(json!(
                    InferenceApi::sign_assets(&self.state, request).await?
                ))
            }
            "list_models" if self.access == McpAccess::Infer => self.installed_models().await,
            "list_models" => Ok(json!(AdminApi::list_models(&self.state).await?)),
            "get_model" => Ok(json!(
                AdminApi::get_model(&self.state, required_id_value(&arguments)).await?
            )),
            "add_model" | "upsert_model" => {
                let spec = parse_model_spec_value(&arguments)?;
                Ok(json!(AdminApi::upsert_model(&self.state, spec).await?))
            }
            "download_model" => Ok(json!(
                AdminApi::download_model(&self.state, required_id_value(&arguments)).await?
            )),
            "get_model_download_status" => Ok(json!(
                AdminApi::get_model_download_status(&self.state, required_id_value(&arguments))
                    .await?
            )),
            "enable_model" => Ok(json!(
                AdminApi::enable_model(&self.state, required_id_value(&arguments)).await?
            )),
            "disable_model" => Ok(json!(
                AdminApi::disable_model(&self.state, required_id_value(&arguments)).await?
            )),
            "list_nodes" => Ok(json!(AdminApi::list_nodes(&self.state).await?)),
            "get_cluster_status" => AdminApi::get_cluster_status(&self.state).await,
            "list_assets" | "search_assets" => {
                let query = serde_json::from_value::<AssetListQuery>(arguments)?;
                Ok(json!(AdminApi::list_assets(&self.state, query).await?))
            }
            other => Err(InfraError::BadRequest(format!(
                "unknown MCP tool `{other}`"
            ))),
        }?;
        match with_url_result {
            Some(policy) => {
                let mut value = value;
                self.refresh_asset_urls(&mut value);
                self.present_inference_result(value, policy)
            }
            None => Ok(value),
        }
    }

    /// Gives every stored asset in a task result (an `assets://` uri, such as
    /// speech a task generated) a fresh signed download URL: the URL saved
    /// with the task stops working DEFAULT_ASSET_URL_TTL_SECS after the task
    /// ends, and a result is often read later.
    fn refresh_asset_urls(&self, value: &mut Value) {
        match value {
            Value::Object(object) => {
                let uri = object
                    .get("uri")
                    .and_then(Value::as_str)
                    .filter(|uri| uri.starts_with("assets://"))
                    .map(str::to_string);
                if let Some(uri) = uri {
                    let signed = self.state.sign_asset_url_public(AssetSignItem {
                        operation: AssetUrlOperation::Download,
                        kind: None,
                        path: None,
                        uri: Some(uri),
                        content_type: None,
                        ttl_sec: None,
                        expires: None,
                        url_ttl_sec: None,
                    });
                    if let Ok(signed) = signed {
                        object.insert("url".to_string(), json!(signed.signed_url));
                    }
                }
                for child in object.values_mut() {
                    self.refresh_asset_urls(child);
                }
            }
            Value::Array(items) => {
                for item in items {
                    self.refresh_asset_urls(item);
                }
            }
            _ => {}
        }
    }

    /// The models a task can use now (enabled, downloaded) with their task
    /// kinds, and what each of those kinds takes and returns.
    async fn installed_models(&self) -> Result<Value> {
        let mut models = Vec::new();
        let mut task_kinds = Map::new();
        for info in AdminApi::list_models(&self.state).await? {
            // Downloaded: every artifact recorded, or present with its
            // pinned size (the store's status, which does not hash).
            if !info.spec.enabled || !info.downloaded {
                continue;
            }
            let kinds = info
                .spec
                .task_kinds
                .iter()
                .copied()
                // Served over the /v1/realtime WebSocket, not as a task.
                .filter(|kind| *kind != local_core::TaskKind::VoiceRealtime)
                .collect::<Vec<_>>();
            if kinds.is_empty() {
                continue;
            }
            for kind in &kinds {
                if let Value::String(name) = json!(kind) {
                    task_kinds
                        .entry(name)
                        .or_insert_with(|| task_kind_guide(*kind));
                }
            }
            models.push(json!({
                "id": info.spec.id,
                "name": info.spec.name,
                "task_kinds": kinds,
                "auto_select": info.spec.auto_selectable(),
            }));
        }
        Ok(json!({ "models": models, "task_kinds": task_kinds }))
    }

    fn present_inference_result(&self, value: Value, policy: WithUrlResult) -> Result<Value> {
        let text = serde_json::to_string(&value)?;
        let needs_url = match policy {
            WithUrlResult::Auto => text.len() > MCP_RESULT_PREVIEW_MAX_BYTES,
            WithUrlResult::On => true,
            WithUrlResult::Off => false,
        };
        if !needs_url {
            return Ok(value);
        }

        let bytes = text.as_bytes();
        let sha256 = sha256_hex(bytes);
        let asset_path = format!("mcp/results/{sha256}.txt");
        let expires_at = Utc::now() + chrono::Duration::seconds(DEFAULT_ARTIFACT_ASSET_TTL_SECS);
        let (mut record, _) = self.state.assets.put_or_reuse_by_sha256_at_path(
            AssetKind::Artifact,
            &asset_path,
            Some("text/plain; charset=utf-8".to_string()),
            Some(expires_at),
            bytes,
            sha256,
        )?;
        self.state.decorate_asset(&mut record);
        let preview = utf8_safe_prefix(&text, MCP_RESULT_PREVIEW_MAX_BYTES);
        Ok(json!({
            "preview": preview,
            "truncated": preview.len() < text.len(),
            "download_url": record.download_url,
            "artifact_uri": record.uri,
            "content_type": record.content_type,
            "size_bytes": record.size,
            "sha256": record.sha256,
            "expires_at": record.expires_at,
        }))
    }
}

fn result_url_policy(name: &str, mut arguments: Value) -> Result<(Option<WithUrlResult>, Value)> {
    if !is_inference_result_tool(name) {
        return Ok((None, arguments));
    }
    let raw = arguments
        .as_object_mut()
        .and_then(|object| object.remove("with_url_result"));
    let policy = raw
        .map(serde_json::from_value)
        .transpose()?
        .unwrap_or_default();
    Ok((Some(policy), arguments))
}

fn is_inference_result_tool(name: &str) -> bool {
    matches!(
        name,
        "create_task" | "start_task" | "get_task" | "wait_task" | "run_task"
    )
}

fn utf8_safe_prefix(value: &str, max_bytes: usize) -> &str {
    let mut end = value.len().min(max_bytes);
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

impl McpAccess {
    fn label(self) -> &'static str {
        match self {
            Self::Admin => "admin",
            Self::Infer => "inference",
        }
    }

    fn allows(self, name: &str) -> bool {
        match self {
            Self::Admin => matches!(
                name,
                "list_models"
                    | "get_model"
                    | "add_model"
                    | "upsert_model"
                    | "download_model"
                    | "get_model_download_status"
                    | "enable_model"
                    | "disable_model"
                    | "list_nodes"
                    | "get_cluster_status"
                    | "list_assets"
                    | "search_assets"
            ),
            Self::Infer => matches!(
                name,
                "list_models"
                    | "create_task"
                    | "start_task"
                    | "get_task"
                    | "wait_task"
                    | "run_task"
                    | "sign_assets"
            ),
        }
    }
}

fn is_unknown_tool_error(err: &InfraError) -> bool {
    matches!(err, InfraError::BadRequest(message) if message.starts_with("unknown MCP tool `"))
}

fn success_json(value: Value) -> std::result::Result<CallToolResult, ErrorData> {
    Ok(CallToolResult::success(vec![Content::json(value)?]))
}

fn task_id_param(params: &Value) -> Result<String> {
    params
        .get("task_id")
        .and_then(Value::as_str)
        .or_else(|| params.as_str())
        .map(str::to_string)
        .ok_or_else(|| InfraError::BadRequest("params.task_id is required".to_string()))
}

fn parse_model_spec_value(params: &Value) -> Result<ModelSpec> {
    if let Some(spec) = params.get("spec") {
        Ok(serde_json::from_value(spec.clone())?)
    } else {
        Ok(serde_json::from_value(params.clone())?)
    }
}

fn required_id_value(params: &Value) -> &str {
    params
        .get("id")
        .and_then(Value::as_str)
        .or_else(|| params.get("model_id").and_then(Value::as_str))
        .unwrap_or("")
}

fn tool_definitions(access: McpAccess) -> Vec<Tool> {
    let task_request = || {
        object_schema(&[
            ("task_kind", task_kind_schema()),
            (
                "model",
                described(
                    string_schema(),
                    "Model id from list_models; omit for the default model of task_kind.",
                ),
            ),
            (
                "files",
                described(
                    array_schema(),
                    "Input files, e.g. [{\"name\": \"photo.jpg\", \"role\": \"image\", \"mime\": \"image/jpeg\", \"required\": true}]; give an \"asset_uri\" from sign_assets instead of uploading to the task.",
                ),
            ),
            (
                "params",
                described(
                    object_schema(&[]),
                    "Task inputs and options of task_kind, as list_models describes.",
                ),
            ),
            ("wait_timeout_sec", number_schema()),
            ("with_url_result", with_url_result_schema()),
        ])
    };
    let list_models = match access {
        McpAccess::Infer => tool(
            "list_models",
            "List the installed models (enabled, files present) with the task kinds each runs, and for every task kind the files and params it takes and the output it returns. Call this first.",
            object_schema(&[]),
        ),
        McpAccess::Admin => tool("list_models", "List configured models.", object_schema(&[])),
    };
    vec![
        list_models,
        tool(
            "run_task",
            "Run a task in one call: create it, start it and wait for its result. For tasks without files (texts, chat, speech without a reference voice) or with files given as asset_uri from sign_assets.",
            task_request(),
        ),
        tool(
            "create_task",
            "Create a task that takes files. Returns uploads[]: POST each file's raw bytes to its upload_url, then call start_task.",
            task_request(),
        ),
        tool(
            "start_task",
            "Start a created task once its files are uploaded; wait=true (the default) returns the result.",
            object_schema(&[
                ("task_id", string_schema()),
                ("wait", bool_schema()),
                ("timeout_sec", number_schema()),
                ("with_url_result", with_url_result_schema()),
            ]),
        ),
        tool(
            "wait_task",
            "Wait until a started task finishes; returns its state, output and files.",
            object_schema(&[
                ("task_id", string_schema()),
                ("timeout_sec", number_schema()),
                ("with_url_result", with_url_result_schema()),
            ]),
        ),
        tool(
            "get_task",
            "Return a task's current state, upload slots, output and files without waiting.",
            object_schema(&[
                ("task_id", string_schema()),
                ("with_url_result", with_url_result_schema()),
            ]),
        ),
        tool(
            "sign_assets",
            "Create signed URLs: operation \"upload\" (POST the bytes to the URL, then pass the returned uri as a task file's asset_uri) or \"download\" (GET an asset by its uri).",
            asset_sign_schema(),
        ),
        tool(
            "get_model",
            "Get a configured model by id/model_id.",
            id_schema(),
        ),
        tool(
            "add_model",
            "Alias for upsert_model.",
            object_schema(&[("spec", object_schema(&[]))]),
        ),
        tool(
            "upsert_model",
            "Insert or update a ModelSpec.",
            object_schema(&[("spec", object_schema(&[]))]),
        ),
        tool(
            "download_model",
            "Queue an asynchronous artifact download for a model. Concurrent and already-complete requests are deduplicated.",
            id_schema(),
        ),
        tool(
            "get_model_download_status",
            "Return aggregate and per-artifact local download status for a model id/model_id.",
            id_schema(),
        ),
        tool(
            "enable_model",
            "Enable a model by id/model_id.",
            id_schema(),
        ),
        tool(
            "disable_model",
            "Disable a model by id/model_id.",
            id_schema(),
        ),
        tool(
            "list_nodes",
            "List registered worker nodes.",
            object_schema(&[]),
        ),
        tool(
            "get_cluster_status",
            "Return controller cluster status.",
            object_schema(&[]),
        ),
        tool(
            "list_assets",
            "List known material/artifact assets.",
            asset_list_schema(),
        ),
        tool(
            "search_assets",
            "Alias for list_assets.",
            asset_list_schema(),
        ),
    ]
    .into_iter()
    .filter(|tool| access.allows(tool.name.as_ref()))
    .collect()
}

fn tool(name: &'static str, description: &'static str, schema: JsonObject) -> Tool {
    Tool::new(name, description, schema)
}

fn object_schema(properties: &[(&str, JsonObject)]) -> JsonObject {
    let mut schema = Map::new();
    schema.insert("type".to_string(), json!("object"));
    let props = properties
        .iter()
        .map(|(key, value)| ((*key).to_string(), Value::Object(value.clone())))
        .collect();
    schema.insert("properties".to_string(), Value::Object(props));
    schema.insert("additionalProperties".to_string(), json!(true));
    schema
}

fn asset_sign_schema() -> JsonObject {
    object_schema(&[(
        "items",
        described(
            array_schema(),
            "e.g. [{\"operation\": \"upload\", \"kind\": \"material\", \"path\": \"photo.jpg\", \"content_type\": \"image/jpeg\"}]",
        ),
    )])
}

/// Task kinds a task can run (`voice.realtime` is a WebSocket).
const TASK_KINDS: [local_core::TaskKind; 11] = {
    use local_core::TaskKind::*;
    [
        AsrTranscribe,
        TtsSynthesize,
        ChatComplete,
        TextEmbed,
        TextRerank,
        ObjectDetect,
        DepthEstimate,
        OcrRecognize,
        OcrLines,
        TextModerate,
        ImageNsfw,
    ]
};

fn task_kind_schema() -> JsonObject {
    let mut schema = described(
        string_schema(),
        "What to run; list_models says which task kinds the installed models serve.",
    );
    schema.insert("enum".to_string(), json!(TASK_KINDS));
    schema
}

/// What a task kind takes and returns, for agents choosing how to call it.
fn task_kind_guide(kind: local_core::TaskKind) -> Value {
    use local_core::TaskKind::*;
    let image = |what: &str| json!([{ "role": "image", "required": true, "what": what }]);
    match kind {
        AsrTranscribe => json!({
            "does": "Speech to text with timeline, language, emotion and speakers.",
            "files": [{ "role": "audio", "required": true, "what": "WAV, FLAC, MP3, AAC/M4A or Ogg audio" }],
            "params": {
                "timestamps": "bool, default true: timestamped_text and timed segments",
                "timestamp_granularity_sec": "number, default 10: length of the timeline's segments",
                "token_timestamps": "bool, default false: per-token times",
                "speaker_diarization": "bool, default true: segments[].speaker and speakers[]",
            },
            "output": "asr_transcription: text, timestamped_text, segments[{start_ms, end_ms, text, speaker, language, emotion}], speakers",
        }),
        TtsSynthesize => json!({
            "does": "Text to speech in the voice of a reference recording.",
            "files": [{ "role": "reference_audio", "required": false, "what": "WAV of the voice to clone (a few seconds)" }],
            "params": {
                "text": "string, required: what to say",
                "language": "string: zh, en, ja, ... (IndexTTS) or chinese, english, ... (Qwen3-TTS); guessed when absent",
                "reference_text": "string (Qwen3-TTS): what the reference audio says, for a closer voice",
                "emotion_vector": "IndexTTS-2.5: 8 weights (happy, angry, sad, afraid, disgusted, melancholic, surprised, calm) or an object keyed by those names",
                "emotion_alpha": "number (IndexTTS-2.5): emotion strength, default 1",
                "speed": "number (IndexTTS-2.5): 2 speaks twice as fast",
                "seed": "integer: sampling seed",
            },
            "output": "tts_audio: audio, a WAV file with a download URL",
        }),
        ChatComplete => json!({
            "does": "Chat completion with tool calls.",
            "files": [],
            "params": {
                "messages": "required: [{role: system|user|assistant|tool, content, tool_calls, tool_call_id}]",
                "tools": "OpenAI-style function tools: [{type: function, function: {name, description, parameters}}]",
                "options": "{max_tokens, temperature (0 = greedy), top_p, top_k, presence_penalty, frequency_penalty, seed, stop}",
            },
            "output": "chat_completion: content, tool_calls, finish_reason, usage",
        }),
        TextEmbed => json!({
            "does": "Text embeddings for search.",
            "files": [],
            "params": {
                "input": "string or array of strings, required",
                "input_type": "query or passage (default)",
            },
            "output": "text_embeddings: embeddings (one vector per input)",
        }),
        TextRerank => json!({
            "does": "Order documents by relevance to a query.",
            "files": [],
            "params": {
                "query": "string, required",
                "documents": "array of strings, required",
                "top_n": "integer: keep the best n",
            },
            "output": "text_rerank: results[{index, relevance_score, document}], best first",
        }),
        ObjectDetect => json!({
            "does": "Objects in an image (80 COCO classes).",
            "files": image("PNG, JPEG or BMP"),
            "params": {},
            "output": "object_detections: objects[{label, confidence, bbox{x, y, width, height}}]",
        }),
        DepthEstimate => json!({
            "does": "Metric depth of an image (meters), averaged over a grid.",
            "files": image("PNG, JPEG or BMP"),
            "params": { "cols": "integer, default 64", "rows": "integer, default 36" },
            "output": "depth_map: cols, rows, max_depth, depth (meters, row by row from the top)",
        }),
        OcrRecognize => json!({
            "does": "Document OCR of one page: its text with layout, tables as HTML.",
            "files": image("one page image"),
            "params": {},
            "output": "ocr_text: text, each block preceded by <|det|>category [x1, y1, x2, y2]<|/det|> (coordinates 0-999)",
        }),
        OcrLines => json!({
            "does": "Text lines in a photo, screenshot or game frame, with boxes.",
            "files": image("PNG, JPEG or BMP"),
            "params": {},
            "output": "ocr_lines: lines[{text, confidence, bbox{x, y, width, height}}], top to bottom",
        }),
        TextModerate => json!({
            "does": "Safety verdict of texts (Qwen3Guard categories).",
            "files": [],
            "params": { "input": "string or array of strings, required" },
            "output": "text_moderations: results[{safe, controversial, unsafe, categories}]",
        }),
        ImageNsfw => json!({
            "does": "NSFW probability of an image.",
            "files": image("PNG, JPEG, BMP, WebP or GIF"),
            "params": {},
            "output": "image_nsfw: nsfw (probability), scores[{label, score}]",
        }),
        VoiceRealtime => json!({
            "does": "Voice conversation over the /v1/realtime WebSocket; not a task.",
        }),
    }
}

fn described(mut schema: JsonObject, description: &str) -> JsonObject {
    schema.insert("description".to_string(), json!(description));
    schema
}

fn asset_list_schema() -> JsonObject {
    object_schema(&[
        ("kind", string_schema()),
        ("prefix", string_schema()),
        ("contains", string_schema()),
        ("include_expired", bool_schema()),
    ])
}

fn id_schema() -> JsonObject {
    object_schema(&[("id", string_schema()), ("model_id", string_schema())])
}

fn string_schema() -> JsonObject {
    typed_schema("string")
}

fn number_schema() -> JsonObject {
    typed_schema("number")
}

fn with_url_result_schema() -> JsonObject {
    let mut schema = string_schema();
    schema.insert("enum".to_string(), json!(["auto", "on", "off"]));
    schema.insert("default".to_string(), json!("auto"));
    schema.insert(
        "description".to_string(),
        json!("auto stores results over 1000 UTF-8 bytes as a downloadable text artifact; on always creates the artifact; off always returns the complete result inline."),
    );
    schema
}

fn bool_schema() -> JsonObject {
    typed_schema("boolean")
}

fn array_schema() -> JsonObject {
    typed_schema("array")
}

fn typed_schema(kind: &str) -> JsonObject {
    let mut schema = Map::new();
    schema.insert("type".to_string(), json!(kind));
    schema
}

#[cfg(test)]
mod tests {
    use super::*;
    use local_registry::ModelRegistry;

    #[test]
    fn inference_tools_expose_url_result_policy() {
        for tool in tool_definitions(McpAccess::Infer)
            .into_iter()
            .filter(|tool| is_inference_result_tool(tool.name.as_ref()))
        {
            let schema = serde_json::to_value(&tool).expect("serialize tool");
            let policy = &schema["inputSchema"]["properties"]["with_url_result"];
            assert_eq!(policy["default"], "auto", "{}: {schema}", tool.name);
            assert_eq!(policy["enum"], json!(["auto", "on", "off"]));
        }
    }

    #[test]
    fn url_result_policy_is_removed_before_inference_dispatch() {
        let (policy, arguments) = result_url_policy(
            "run_task",
            json!({"task_kind": "text.embed", "params": {"input": ["q"]}, "with_url_result": "on"}),
        )
        .expect("valid policy");
        assert_eq!(policy, Some(WithUrlResult::On));
        assert!(arguments.get("with_url_result").is_none());

        let err = result_url_policy(
            "get_task",
            json!({"task_id": "t", "with_url_result": "sometimes"}),
        )
        .expect_err("invalid policy must fail");
        assert!(err.to_string().contains("unknown variant"), "{err}");
    }

    #[test]
    fn utf8_preview_never_splits_a_code_point() {
        let value = format!("{}界tail", "a".repeat(999));
        let preview = utf8_safe_prefix(&value, MCP_RESULT_PREVIEW_MAX_BYTES);
        assert_eq!(preview, "a".repeat(999));
        assert!(value.is_char_boundary(preview.len()));
    }

    #[test]
    fn url_result_modes_create_deduplicated_expiring_text_artifacts() {
        let dir = tempfile::tempdir().expect("temp data dir");
        let state = ControllerState::new_with_options(
            ModelRegistry::from_models(Vec::new()),
            None,
            ControllerOptions {
                data_dir: dir.path().to_path_buf(),
                upload_signing_secret: Some("result-test-secret".to_string()),
                asset_cleanup_interval: None,
                max_upload_bytes: DEFAULT_MAX_UPLOAD_BYTES,
                ..ControllerOptions::default()
            },
        );
        let server = StandardMcpServer {
            state: state.clone(),
            access: McpAccess::Infer,
        };
        let long_value = json!({"text": format!("{}界tail", "a".repeat(990))});
        let full_text = serde_json::to_string(&long_value).expect("serialize result");

        let short_value = json!({"text": "ok"});
        let short_auto = server
            .present_inference_result(short_value.clone(), WithUrlResult::Auto)
            .expect("short auto result");
        assert_eq!(short_auto, short_value);

        let inline = server
            .present_inference_result(long_value.clone(), WithUrlResult::Off)
            .expect("off result");
        assert_eq!(inline, long_value);
        assert!(state
            .assets
            .list(&AssetListQuery::default())
            .expect("list assets")
            .assets
            .is_empty());

        let first = server
            .present_inference_result(long_value.clone(), WithUrlResult::Auto)
            .expect("auto result");
        assert_eq!(first["truncated"], true);
        let preview = first["preview"].as_str().expect("preview");
        assert!(preview.len() <= MCP_RESULT_PREVIEW_MAX_BYTES);
        assert!(full_text.is_char_boundary(preview.len()));
        assert!(first["download_url"]
            .as_str()
            .is_some_and(|url| url.starts_with("http://127.0.0.1:17890/assets/artifact/")));
        let uri = first["artifact_uri"].as_str().expect("artifact URI");
        let (record, bytes) = state.assets.read_bytes(uri).expect("read text artifact");
        assert_eq!(bytes, full_text.as_bytes());
        assert_eq!(
            record.content_type.as_deref(),
            Some("text/plain; charset=utf-8")
        );
        assert!(record.path.ends_with(".txt"));
        let first_expiry = record.expires_at.expect("artifact expiry");

        let second = server
            .present_inference_result(long_value, WithUrlResult::On)
            .expect("on result");
        assert_eq!(second["artifact_uri"], first["artifact_uri"]);
        let assets = state
            .assets
            .list(&AssetListQuery::default())
            .expect("list deduplicated assets")
            .assets;
        assert_eq!(assets.len(), 1);
        assert!(assets[0].expires_at.expect("renewed expiry") >= first_expiry);

        let short = server
            .present_inference_result(short_value, WithUrlResult::On)
            .expect("short on result");
        assert_eq!(short["truncated"], false);
        assert!(short["download_url"].is_string());
    }

    #[test]
    fn inference_catalog_is_the_generic_task_tools_and_list_models() {
        let infer = tool_definitions(McpAccess::Infer)
            .into_iter()
            .map(|tool| tool.name.to_string())
            .collect::<std::collections::BTreeSet<_>>();
        let expected = [
            "create_task",
            "get_task",
            "list_models",
            "run_task",
            "sign_assets",
            "start_task",
            "wait_task",
        ];
        assert_eq!(infer, expected.map(String::from).into_iter().collect());
        for name in expected {
            assert!(McpAccess::Infer.allows(name), "{name}");
        }
        for removed in ["asr_transcribe", "object_detect", "text_embed", "sign_asset_urls"] {
            assert!(!McpAccess::Infer.allows(removed), "{removed}");
        }
        // The catalogs share only list_models, which differs per catalog.
        let admin = tool_definitions(McpAccess::Admin)
            .into_iter()
            .map(|tool| tool.name.to_string())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            admin.intersection(&infer).collect::<Vec<_>>(),
            vec!["list_models"]
        );
        assert!(admin.contains("enable_model") && !infer.contains("enable_model"));
    }

    #[tokio::test]
    async fn inference_list_models_shows_installed_models_and_their_task_kinds() {
        let dir = tempfile::tempdir().expect("tempdir");
        let models_dir = dir.path().join("models");
        let store = local_model_store::SqliteModelStore::new(
            dir.path().join("data/local.db"),
            models_dir.clone(),
        )
        .expect("store");
        // A local artifact counts as present when its file exists.
        let model = |id: &str, kinds: Vec<local_core::TaskKind>, present: bool| {
            let path = models_dir.join(id).join("model.onnx");
            if present {
                std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
                std::fs::write(&path, b"onnx").expect("write model");
            }
            let mut spec = crate::tests::test_model();
            spec.id = id.to_string();
            spec.task_kinds = kinds;
            spec.artifacts = vec![local_core::ModelArtifact {
                kind: local_core::ArtifactKind::Local,
                path,
                source_path: None,
                sha256: None,
                size: None,
                url: None,
                repo_id: None,
                revision: None,
                files: Vec::new(),
                allow_patterns: Vec::new(),
                metadata: Default::default(),
            }];
            spec
        };
        use local_core::TaskKind::*;
        let mut named_only = model("ocr-export", vec![OcrRecognize], true);
        named_only
            .metadata
            .insert("auto_select".to_string(), json!(false));
        let mut disabled = model("disabled", vec![TextEmbed], true);
        disabled.enabled = false;
        // A downloadable artifact copied into the model directory, with no
        // download record, is installed too.
        let mut copied = model("copied", vec![TextRerank], false);
        let copied_file = models_dir.join("copied").join("model.onnx");
        std::fs::create_dir_all(copied_file.parent().expect("parent")).expect("mkdir");
        std::fs::write(&copied_file, b"onnx").expect("write copied");
        copied.artifacts[0].kind = local_core::ArtifactKind::Url;
        copied.artifacts[0].url = Some("https://example.invalid/model.onnx".to_string());
        let specs = [
            model("detector", vec![ObjectDetect], true),
            named_only,
            model("missing", vec![TtsSynthesize], false),
            model("voice", vec![VoiceRealtime], true),
            disabled,
            copied,
        ]
        .map(|spec| store.upsert_model(spec).expect("upsert"));
        let server = StandardMcpServer {
            state: ControllerState::with_store(ModelRegistry::from_models(specs.to_vec()), store),
            access: McpAccess::Infer,
        };

        let listed = server
            .call_tool_json("list_models", json!({}))
            .await
            .expect("list_models");
        let ids = listed["models"]
            .as_array()
            .expect("models")
            .iter()
            .map(|model| model["id"].as_str().expect("id"))
            .collect::<Vec<_>>();
        assert_eq!(ids, ["copied", "detector", "ocr-export"], "{listed}");
        assert_eq!(listed["models"][1]["task_kinds"], json!(["object.detect"]));
        assert_eq!(listed["models"][1]["auto_select"], true);
        assert_eq!(listed["models"][2]["auto_select"], false);
        let kinds = listed["task_kinds"].as_object().expect("task kinds");
        // Key order depends on serde_json's preserve_order feature, which
        // other crates in the workspace turn on.
        let mut names = kinds.keys().collect::<Vec<_>>();
        names.sort();
        assert_eq!(names, ["object.detect", "ocr.recognize", "text.rerank"]);
        assert_eq!(kinds["object.detect"]["files"][0]["role"], "image");

        // The admin catalog still lists every configured model.
        let admin = StandardMcpServer {
            access: McpAccess::Admin,
            ..server
        };
        let all = admin
            .call_tool_json("list_models", json!({}))
            .await
            .expect("admin list_models");
        assert_eq!(all.as_array().expect("models").len(), 6);
    }

    #[tokio::test]
    async fn url_results_and_the_assets_in_them_download_over_http() {
        let dir = tempfile::tempdir().expect("temp data dir");
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test server");
        let base_url = format!("http://{}", listener.local_addr().expect("local addr"));
        let state = ControllerState::new_with_options(
            ModelRegistry::from_models(Vec::new()),
            None,
            ControllerOptions {
                public_base_url: base_url.clone(),
                data_dir: dir.path().to_path_buf(),
                upload_signing_secret: Some("download-test-secret".to_string()),
                asset_cleanup_interval: None,
                max_upload_bytes: DEFAULT_MAX_UPLOAD_BYTES,
                ..ControllerOptions::default()
            },
        );
        let app = state.clone().app();
        let server_task = tokio::spawn(async move {
            axum::serve(listener, app).await.expect("serve test app");
        });
        let server = StandardMcpServer {
            state: state.clone(),
            access: McpAccess::Infer,
        };
        // Speech a task generated, as its stored result names it: the URL
        // signed when the task ended has expired.
        let speech = b"RIFF....WAVEfmt speech bytes";
        let record = state
            .assets
            .put(
                AssetKind::Artifact,
                "tasks/t1/outputs/speech.wav",
                Some("audio/wav".to_string()),
                None,
                speech,
            )
            .expect("store speech");
        let mut result = json!({
            "task_id": "t1",
            "state": "succeeded",
            "output": {"type": "tts_audio", "audio": {"uri": record.uri, "url": "http://stale.invalid/x"}},
            "files": [{"uri": record.uri}],
        });
        server.refresh_asset_urls(&mut result);
        let client = reqwest::Client::new();
        for url in [&result["output"]["audio"]["url"], &result["files"][0]["url"]] {
            let url = url.as_str().expect("fresh url");
            assert!(url.starts_with(&base_url), "{url}");
            let response = client.get(url).send().await.expect("download speech");
            assert_eq!(response.status(), StatusCode::OK, "{url}");
            assert_eq!(response.bytes().await.expect("speech bytes").as_ref(), speech);
        }

        // A result too long to inline comes back as a text artifact whose
        // download URL serves the whole result.
        let long = json!({"text": "界".repeat(600), "audio": result["output"]["audio"].clone()});
        let wrapped = server
            .present_inference_result(long.clone(), WithUrlResult::Auto)
            .expect("url result");
        let url = wrapped["download_url"].as_str().expect("download_url");
        let response = client.get(url).send().await.expect("download result");
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "text/plain; charset=utf-8"
        );
        let downloaded: Value =
            serde_json::from_slice(&response.bytes().await.expect("result bytes")).expect("json");
        assert_eq!(downloaded, long);
        server_task.abort();
    }

    #[test]
    fn task_tools_describe_task_kinds_and_files() {
        let run = tool_definitions(McpAccess::Infer)
            .into_iter()
            .find(|tool| tool.name.as_ref() == "run_task")
            .expect("run_task");
        let schema = serde_json::to_value(run).expect("serialize tool");
        let kinds = &schema["inputSchema"]["properties"]["task_kind"]["enum"];
        assert!(kinds.as_array().is_some_and(|kinds| kinds.contains(&json!("asr.transcribe"))));
        assert!(!kinds.as_array().unwrap().contains(&json!("voice.realtime")));
        for name in ["model", "files", "params"] {
            assert!(
                schema["inputSchema"]["properties"][name]["description"].is_string(),
                "{name}: {schema}"
            );
        }
        for kind in TASK_KINDS {
            let guide = task_kind_guide(kind);
            for field in ["does", "files", "params", "output"] {
                assert!(guide.get(field).is_some(), "{kind:?} lacks {field}");
            }
        }
    }

    #[tokio::test]
    async fn rpc_routes_share_admin_and_inference_auth_policies() {
        let state = test_state(Some("admin-secret"), &["infer-a", "infer-b"]);
        let (base_url, server) = serve(state.app()).await;
        let client = reqwest::Client::new();
        let admin_body = json!({"jsonrpc":"2.0","id":1,"method":"list_models","params":{}});
        let infer_body =
            json!({"jsonrpc":"2.0","id":2,"method":"get_task","params":{"task_id":"missing"}});

        assert_eq!(
            client
                .post(format!("{base_url}/rpc/admin"))
                .json(&admin_body)
                .send()
                .await
                .expect("admin without token")
                .status(),
            reqwest::StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            client
                .post(format!("{base_url}/rpc/admin"))
                .bearer_auth("admin-secret")
                .json(&admin_body)
                .send()
                .await
                .expect("admin with token")
                .status(),
            reqwest::StatusCode::OK
        );
        assert_eq!(
            client
                .post(format!("{base_url}/rpc/infer"))
                .json(&infer_body)
                .send()
                .await
                .expect("infer without token")
                .status(),
            reqwest::StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            client
                .post(format!("{base_url}/rpc/infer"))
                .bearer_auth("not-listed")
                .json(&infer_body)
                .send()
                .await
                .expect("infer with wrong token")
                .status(),
            reqwest::StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            client
                .post(format!("{base_url}/rpc/infer"))
                .header("x-local-infer-token", "infer-b")
                .json(&infer_body)
                .send()
                .await
                .expect("infer with listed token")
                .status(),
            reqwest::StatusCode::OK
        );
        server.abort();
    }

    #[tokio::test]
    async fn openai_inference_routes_use_inference_tokens() {
        let state = test_state(Some("admin-secret"), &["infer-a", "infer-b"]);
        let (base_url, server) = serve(state.app()).await;
        let client = reqwest::Client::new();
        let inference_paths = [
            "/v1/audio/transcriptions",
            "/v1/audio/speech",
            "/v1/embeddings",
            "/rerank",
            "/v1/rerank",
            "/v2/rerank",
        ];

        for path in inference_paths {
            assert_eq!(
                client
                    .post(format!("{base_url}{path}"))
                    .json(&json!({}))
                    .send()
                    .await
                    .expect("OpenAI inference without token")
                    .status(),
                reqwest::StatusCode::UNAUTHORIZED,
                "{path} must reject a missing inference token"
            );
            assert_eq!(
                client
                    .post(format!("{base_url}{path}"))
                    .bearer_auth("not-listed")
                    .json(&json!({}))
                    .send()
                    .await
                    .expect("OpenAI inference with wrong token")
                    .status(),
                reqwest::StatusCode::UNAUTHORIZED,
                "{path} must reject an unlisted inference token"
            );
            assert_ne!(
                client
                    .post(format!("{base_url}{path}"))
                    .bearer_auth("infer-a")
                    .json(&json!({}))
                    .send()
                    .await
                    .expect("OpenAI inference with listed token")
                    .status(),
                reqwest::StatusCode::UNAUTHORIZED,
                "{path} must accept a listed inference token"
            );
        }

        assert_eq!(
            client
                .get(format!("{base_url}/v1/models"))
                .send()
                .await
                .expect("OpenAI model list without token")
                .status(),
            reqwest::StatusCode::OK,
            "/v1/models is a catalog route, not an inference route"
        );
        assert_eq!(
            client
                .get(format!("{base_url}/mcp/infer"))
                .send()
                .await
                .expect("standard MCP path on controller port")
                .status(),
            reqwest::StatusCode::NOT_FOUND,
            "OpenAI route auth must not intercept unmatched controller paths"
        );
        server.abort();
    }

    #[tokio::test]
    async fn inference_auth_is_optional_but_admin_auth_is_not() {
        let state = test_state(None, &[]);
        let (base_url, server) = serve(state.app()).await;
        let client = reqwest::Client::new();
        let body =
            json!({"jsonrpc":"2.0","id":1,"method":"get_task","params":{"task_id":"missing"}});

        assert_eq!(
            client
                .post(format!("{base_url}/rpc/infer"))
                .json(&body)
                .send()
                .await
                .expect("open infer")
                .status(),
            reqwest::StatusCode::OK
        );
        assert_ne!(
            client
                .post(format!("{base_url}/v1/embeddings"))
                .json(&json!({}))
                .send()
                .await
                .expect("open OpenAI inference")
                .status(),
            reqwest::StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            client
                .post(format!("{base_url}/rpc/admin"))
                .json(&body)
                .send()
                .await
                .expect("unconfigured admin")
                .status(),
            reqwest::StatusCode::SERVICE_UNAVAILABLE
        );
        server.abort();
    }

    #[tokio::test]
    async fn both_standard_mcp_paths_enforce_their_auth_policy() {
        let state = test_state(Some("admin-secret"), &["infer-secret"]);
        let (base_url, server) = serve(state.standard_mcp_app()).await;
        let client = reqwest::Client::new();

        for path in ["/mcp/admin", "/mcp/infer"] {
            assert_eq!(
                client
                    .post(format!("{base_url}{path}"))
                    .send()
                    .await
                    .expect("MCP without token")
                    .status(),
                reqwest::StatusCode::UNAUTHORIZED
            );
        }
        assert_ne!(
            client
                .post(format!("{base_url}/mcp/admin"))
                .bearer_auth("admin-secret")
                .send()
                .await
                .expect("admin MCP with token")
                .status(),
            reqwest::StatusCode::UNAUTHORIZED
        );
        assert_ne!(
            client
                .post(format!("{base_url}/mcp/infer"))
                .bearer_auth("infer-secret")
                .send()
                .await
                .expect("infer MCP with token")
                .status(),
            reqwest::StatusCode::UNAUTHORIZED
        );
        server.abort();
    }

    fn test_state(admin_token: Option<&str>, infer_tokens: &[&str]) -> ControllerState {
        ControllerState::new_with_options(
            ModelRegistry::from_models(Vec::new()),
            None,
            ControllerOptions {
                admin_token: admin_token.map(str::to_string),
                mcp_infer_tokens: infer_tokens.iter().map(|token| token.to_string()).collect(),
                asset_cleanup_interval: None,
                max_upload_bytes: DEFAULT_MAX_UPLOAD_BYTES,
                ..ControllerOptions::default()
            },
        )
    }

    async fn serve(app: Router) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test server");
        let base_url = format!("http://{}", listener.local_addr().expect("local addr"));
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.expect("serve test app");
        });
        (base_url, server)
    }
}
