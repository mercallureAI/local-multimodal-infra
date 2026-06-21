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
        let server = StandardMcpServer { state: self };
        let mut config = StreamableHttpServerConfig::default();
        config.stateful_mode = false;
        let service = StreamableHttpService::new(
            move || Ok(server.clone()),
            LocalSessionManager::default().into(),
            config,
        );
        Router::new().nest_service("/mcp", service)
    }
}

#[derive(Clone)]
struct StandardMcpServer {
    state: ControllerState,
}

impl ServerHandler for StandardMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::default(),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "lcoal-controller-standard-mcp".to_string(),
                title: Some("lcoal Controller Standard MCP".to_string()),
                version: env!("CARGO_PKG_VERSION").to_string(),
                description: Some("Standard MCP adapter for the lcoal controller".to_string()),
                icons: None,
                website_url: None,
            },
            instructions: Some(
                "Standard MCP tools for the lcoal controller. Results preserve the legacy JSON-RPC payloads where possible."
                    .to_string(),
            ),
        }
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> std::result::Result<ListToolsResult, ErrorData> {
        Ok(ListToolsResult {
            tools: tool_definitions(),
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
        tool_definitions()
            .into_iter()
            .find(|tool| tool.name.as_ref() == name)
    }
}

impl StandardMcpServer {
    async fn call_tool_json(&self, name: &str, arguments: Value) -> Result<Value> {
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
            "asr_transcribe" | "object_detect" | "tts_synthesize" => {
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
            Ok(InferenceTask::new(
                lcoal_core::TaskKind::AsrTranscribe,
                model_id,
                InferenceInput::AsrTranscribe { audio },
            ))
        }
        "object_detect" => {
            let image = direct_file_ref(params, "image", "image_path", "object_detect")?;
            Ok(InferenceTask::new(
                lcoal_core::TaskKind::ObjectDetect,
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
                lcoal_core::TaskKind::TtsSynthesize,
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
        other => Err(InfraError::BadRequest(format!(
            "unknown MCP tool `{other}`"
        ))),
    }
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

fn tool_definitions() -> Vec<Tool> {
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
            "Run direct ASR transcription using model/model_id and an audio FileRef.",
            object_schema(&[
                ("model", string_schema()),
                ("model_id", string_schema()),
                ("audio", file_ref_schema()),
                ("audio_path", string_schema()),
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
            "Download or materialize artifacts for a model id/model_id.",
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
