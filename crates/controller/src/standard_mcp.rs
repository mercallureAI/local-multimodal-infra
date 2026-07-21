use super::*;
use rmcp::{
    model::*,
    service::RequestContext,
    transport::streamable_http_server::{
        session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
    },
    ErrorData, RoleServer, ServerHandler,
};
use serde_json::{Map, Value};

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
        let infer_auth = if self.mcp_infer_tokens.is_empty() {
            ApiAuth::Open
        } else {
            ApiAuth::Required {
                tokens: self.mcp_infer_tokens.clone(),
                header_name: "x-local-infer-token",
                missing_configuration: false,
            }
        };
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
            instructions: Some(
                format!(
                    "Standard MCP {} tools for the local controller. Results preserve the legacy JSON-RPC payloads where possible.",
                    self.access.label()
                ),
            ),
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
        match name {
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
            "sign_assets" | "sign_asset_urls" => {
                let request = serde_json::from_value::<AssetSignRequest>(arguments)?;
                Ok(json!(
                    InferenceApi::sign_assets(&self.state, request).await?
                ))
            }
            "asr_transcribe" | "object_detect" | "tts_synthesize" | "text_embed"
            | "text_rerank" => {
                let task = task_from_method(name, &arguments)?;
                Ok(json!(InferenceApi::dispatch(&self.state, task).await?))
            }
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
        }
    }
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
                "create_task"
                    | "start_task"
                    | "get_task"
                    | "wait_task"
                    | "run_task"
                    | "sign_assets"
                    | "sign_asset_urls"
                    | "asr_transcribe"
                    | "object_detect"
                    | "tts_synthesize"
                    | "text_embed"
                    | "text_rerank"
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

fn task_from_method(method: &str, params: &Value) -> Result<InferenceTask> {
    let model_id = params
        .get("model")
        .and_then(Value::as_str)
        .or_else(|| params.get("model_id").and_then(Value::as_str))
        .map(ToOwned::to_owned);
    match method {
        "asr_transcribe" => {
            let audio = direct_file_ref(params, "audio", "audio_path", "asr_transcribe")?;
            let mut task = InferenceTask::new(
                local_core::TaskKind::AsrTranscribe,
                model_id,
                InferenceInput::AsrTranscribe { audio },
            );
            task.params = params
                .as_object()
                .map(|object| {
                    object
                        .iter()
                        .map(|(key, value)| (key.clone(), value.clone()))
                        .collect()
                })
                .unwrap_or_default();
            Ok(task)
        }
        "object_detect" => {
            let image = direct_file_ref(params, "image", "image_path", "object_detect")?;
            Ok(InferenceTask::new(
                local_core::TaskKind::ObjectDetect,
                model_id,
                InferenceInput::ObjectDetect { image },
            ))
        }
        "tts_synthesize" => {
            let text = params
                .get("text")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    InfraError::BadRequest("tts_synthesize params.text is required".to_string())
                })?
                .to_string();
            let reference_audio = optional_file_ref(params.get("reference_audio"))?
                .or_else(|| path_file_ref(params.get("reference_audio_path")))
                .or_else(|| path_file_ref(params.get("reference_path")));
            let mut task = InferenceTask::new(
                local_core::TaskKind::TtsSynthesize,
                model_id,
                InferenceInput::TtsSynthesize {
                    text,
                    reference_audio,
                },
            );
            task.params = params
                .as_object()
                .map(|object| {
                    object
                        .iter()
                        .map(|(key, value)| (key.clone(), value.clone()))
                        .collect()
                })
                .unwrap_or_default();
            Ok(task)
        }
        "text_embed" => {
            let texts = value_text_list(params, &["input", "texts", "text"])?;
            let input_type = params
                .get("input_type")
                .cloned()
                .map(serde_json::from_value)
                .transpose()?
                .unwrap_or_default();
            Ok(InferenceTask::new(
                local_core::TaskKind::TextEmbed,
                model_id,
                InferenceInput::TextEmbed { texts, input_type },
            ))
        }
        "text_rerank" => {
            let query = params
                .get("query")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| {
                    InfraError::BadRequest("text_rerank params.query is required".to_string())
                })?
                .to_string();
            let documents = value_text_list(params, &["documents"])?;
            let top_n = params
                .get("top_n")
                .and_then(Value::as_u64)
                .map(|value| value as usize);
            Ok(InferenceTask::new(
                local_core::TaskKind::TextRerank,
                model_id,
                InferenceInput::TextRerank {
                    query,
                    documents,
                    top_n,
                },
            ))
        }
        other => Err(InfraError::BadRequest(format!(
            "unknown MCP tool `{other}`"
        ))),
    }
}

fn value_text_list(params: &Value, keys: &[&str]) -> Result<Vec<String>> {
    let value = keys
        .iter()
        .find_map(|key| params.get(*key))
        .ok_or_else(|| InfraError::BadRequest(format!("params.{} is required", keys[0])))?;
    let texts = match value {
        Value::String(text) => vec![text.clone()],
        Value::Array(values) => values
            .iter()
            .map(|value| {
                value.as_str().map(str::to_string).ok_or_else(|| {
                    InfraError::BadRequest(format!("params.{} must contain strings", keys[0]))
                })
            })
            .collect::<Result<Vec<_>>>()?,
        _ => {
            return Err(InfraError::BadRequest(format!(
                "params.{} must be a string or array of strings",
                keys[0]
            )))
        }
    };
    if texts.is_empty() || texts.iter().any(|text| text.trim().is_empty()) {
        return Err(InfraError::BadRequest(format!(
            "params.{} must contain non-empty strings",
            keys[0]
        )));
    }
    Ok(texts)
}

fn file_ref(value: &Value) -> Result<FileRef> {
    serde_json::from_value(value.clone()).map_err(InfraError::from)
}

fn optional_file_ref(value: Option<&Value>) -> Result<Option<FileRef>> {
    value.map(file_ref).transpose()
}

fn direct_file_ref(
    params: &Value,
    object_key: &str,
    path_key: &str,
    method: &str,
) -> Result<FileRef> {
    if let Some(file) = optional_file_ref(params.get(object_key))? {
        return Ok(file);
    }
    if let Some(file) = path_file_ref(params.get(path_key)) {
        return Ok(file);
    }
    let file = file_ref(params)?;
    if file.file_id.is_some() || file.path.is_some() || file.url.is_some() || file.uri.is_some() {
        return Ok(file);
    }
    Err(InfraError::BadRequest(format!(
        "{method} params.{object_key} or params.{path_key} is required"
    )))
}

fn path_file_ref(value: Option<&Value>) -> Option<FileRef> {
    value
        .and_then(Value::as_str)
        .map(|path| FileRef::local(std::path::PathBuf::from(path)))
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
    vec![
        tool(
            "create_task",
            "Create a generic inference task and return upload slots.",
            object_schema(&[
                ("task_kind", string_schema()),
                ("model", string_schema()),
                ("model_id", string_schema()),
                ("files", array_schema()),
                ("params", object_schema(&[])),
                ("wait_timeout_sec", number_schema()),
            ]),
        ),
        tool(
            "start_task",
            "Start a previously created generic inference task.",
            object_schema(&[
                ("task_id", string_schema()),
                ("wait", bool_schema()),
                ("timeout_sec", number_schema()),
            ]),
        ),
        tool(
            "get_task",
            "Fetch generic task status by task_id.",
            object_schema(&[("task_id", string_schema())]),
        ),
        tool(
            "wait_task",
            "Wait for a generic task to leave running states.",
            object_schema(&[
                ("task_id", string_schema()),
                ("timeout_sec", number_schema()),
            ]),
        ),
        tool(
            "run_task",
            "Create and start a generic task, waiting for completion by default.",
            object_schema(&[
                ("task_kind", string_schema()),
                ("model", string_schema()),
                ("model_id", string_schema()),
                ("files", array_schema()),
                ("params", object_schema(&[])),
                ("wait_timeout_sec", number_schema()),
            ]),
        ),
        tool(
            "sign_assets",
            "Create signed asset upload/download URLs.",
            asset_sign_schema(),
        ),
        tool(
            "sign_asset_urls",
            "Alias for sign_assets.",
            asset_sign_schema(),
        ),
        tool(
            "asr_transcribe",
            "Run direct ASR. By default returns ~10s timestamped_text/segments plus segments[].speaker and speakers[]; set timestamps=false to disable the timeline.",
            object_schema(&[
                ("model", string_schema()),
                ("model_id", string_schema()),
                ("audio", file_ref_schema()),
                ("audio_path", string_schema()),
                ("timestamps", bool_schema()),
                ("timestamp_granularity_sec", timestamp_granularity_schema()),
                ("token_timestamps", bool_schema()),
                ("speaker_diarization", bool_schema()),
            ]),
        ),
        tool(
            "object_detect",
            "Run direct object detection using model/model_id and an image FileRef.",
            object_schema(&[
                ("model", string_schema()),
                ("model_id", string_schema()),
                ("image", file_ref_schema()),
                ("image_path", string_schema()),
            ]),
        ),
        tool(
            "tts_synthesize",
            "Run direct TTS synthesis and pass all original params through task.params.",
            object_schema(&[
                ("text", string_schema()),
                ("model", string_schema()),
                ("model_id", string_schema()),
                ("reference_audio", file_ref_schema()),
                ("reference_audio_path", string_schema()),
                ("reference_path", string_schema()),
            ]),
        ),
        tool(
            "text_embed",
            "Create multilingual E5 embeddings for input/texts with query or passage input_type.",
            object_schema(&[
                ("model", string_schema()),
                ("model_id", string_schema()),
                ("input", array_schema()),
                ("texts", array_schema()),
                ("text", string_schema()),
                ("input_type", string_schema()),
            ]),
        ),
        tool(
            "text_rerank",
            "Rerank documents for a query with mMARCO MiniLM.",
            object_schema(&[
                ("model", string_schema()),
                ("model_id", string_schema()),
                ("query", string_schema()),
                ("documents", array_schema()),
                ("top_n", number_schema()),
            ]),
        ),
        tool("list_models", "List configured models.", object_schema(&[])),
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
    object_schema(&[("items", array_schema()), ("requests", array_schema())])
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

fn file_ref_schema() -> JsonObject {
    object_schema(&[
        ("file_id", string_schema()),
        ("path", string_schema()),
        ("url", string_schema()),
        ("uri", string_schema()),
        ("mime", string_schema()),
        ("sha256", string_schema()),
    ])
}

fn string_schema() -> JsonObject {
    typed_schema("string")
}

fn number_schema() -> JsonObject {
    typed_schema("number")
}

fn timestamp_granularity_schema() -> JsonObject {
    let mut schema = number_schema();
    schema.insert("default".to_string(), json!(10));
    schema.insert("minimum".to_string(), json!(1));
    schema.insert("maximum".to_string(), json!(120));
    schema.insert(
        "description".to_string(),
        json!("Target timeline segment duration in seconds; defaults to 10."),
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
    use local_core::{EmbeddingInputType, TaskKind};
    use local_registry::ModelRegistry;

    #[test]
    fn text_tools_are_registered() {
        let names = tool_definitions(McpAccess::Infer)
            .into_iter()
            .map(|tool| tool.name.to_string())
            .collect::<Vec<_>>();
        assert!(names.iter().any(|name| name == "text_embed"));
        assert!(names.iter().any(|name| name == "text_rerank"));
    }

    #[test]
    fn admin_and_infer_tool_catalogs_are_disjoint() {
        let admin = tool_definitions(McpAccess::Admin)
            .into_iter()
            .map(|tool| tool.name.to_string())
            .collect::<std::collections::BTreeSet<_>>();
        let infer = tool_definitions(McpAccess::Infer)
            .into_iter()
            .map(|tool| tool.name.to_string())
            .collect::<std::collections::BTreeSet<_>>();
        assert!(admin.is_disjoint(&infer));
        assert!(admin.contains("enable_model"));
        assert!(!admin.contains("text_embed"));
        assert!(infer.contains("text_embed"));
        assert!(!infer.contains("enable_model"));
    }

    #[test]
    fn text_tools_map_to_core_tasks() {
        let embed = task_from_method(
            "text_embed",
            &json!({
                "model": "multilingual-e5-small-onnx",
                "input": ["q"],
                "input_type": "query"
            }),
        )
        .expect("embed task");
        assert_eq!(embed.kind, TaskKind::TextEmbed);
        assert!(matches!(
            embed.input,
            InferenceInput::TextEmbed {
                input_type: EmbeddingInputType::Query,
                ..
            }
        ));

        let rerank = task_from_method(
            "text_rerank",
            &json!({
                "model": "mmarco-minilm-l12-onnx",
                "query": "q",
                "documents": ["a", "b"],
                "top_n": 1
            }),
        )
        .expect("rerank task");
        assert_eq!(rerank.kind, TaskKind::TextRerank);
        assert!(matches!(
            rerank.input,
            InferenceInput::TextRerank { top_n: Some(1), .. }
        ));
    }

    #[test]
    fn asr_tool_exposes_timeline_and_speaker_params() {
        let tool = tool_definitions(McpAccess::Infer)
            .into_iter()
            .find(|tool| tool.name.as_ref() == "asr_transcribe")
            .expect("ASR tool");
        let schema = serde_json::to_value(tool).expect("serialize tool");
        let properties = &schema["inputSchema"]["properties"];
        for name in [
            "timestamps",
            "timestamp_granularity_sec",
            "token_timestamps",
            "speaker_diarization",
        ] {
            assert!(properties.get(name).is_some(), "missing {name}: {schema}");
        }
        assert_eq!(properties["timestamp_granularity_sec"]["default"], 10);

        let task = task_from_method(
            "asr_transcribe",
            &json!({
                "model": "sensevoice-small-onnx",
                "audio_path": "./audio.wav",
                "timestamps": false,
                "timestamp_granularity_sec": 15,
                "token_timestamps": true,
                "speaker_diarization": false
            }),
        )
        .expect("ASR task");
        assert_eq!(task.kind, TaskKind::AsrTranscribe);
        assert_eq!(task.params["timestamps"], false);
        assert_eq!(task.params["timestamp_granularity_sec"], 15);
        assert_eq!(task.params["token_timestamps"], true);
        assert_eq!(task.params["speaker_diarization"], false);
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
