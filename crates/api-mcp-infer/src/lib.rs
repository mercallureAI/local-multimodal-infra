use async_trait::async_trait;
use axum::{extract::State, routing::post, Json, Router};
use local_core::{
    AssetSignRequest, AssetSignResponse, CreateTaskRequest, FileRef, GenericTaskResult,
    InferenceInput, InferenceOutput, InferenceTask, StartTaskRequest, TaskKind, TaskStatus,
    WaitTaskRequest,
};
use local_error::{InfraError, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;

#[async_trait]
pub trait InferenceApi: Send + Sync + 'static {
    async fn dispatch(&self, task: InferenceTask) -> Result<InferenceOutput>;
    async fn create_task(&self, _request: CreateTaskRequest) -> Result<TaskStatus> {
        Err(InfraError::Unsupported(
            "generic create_task is not implemented by this inference service".to_string(),
        ))
    }
    async fn start_task(&self, _request: StartTaskRequest) -> Result<GenericTaskResult> {
        Err(InfraError::Unsupported(
            "generic start_task is not implemented by this inference service".to_string(),
        ))
    }
    async fn get_task(&self, _task_id: String) -> Result<TaskStatus> {
        Err(InfraError::Unsupported(
            "generic get_task is not implemented by this inference service".to_string(),
        ))
    }
    async fn wait_task(&self, _request: WaitTaskRequest) -> Result<TaskStatus> {
        Err(InfraError::Unsupported(
            "generic wait_task is not implemented by this inference service".to_string(),
        ))
    }
    async fn sign_assets(&self, _request: AssetSignRequest) -> Result<AssetSignResponse> {
        Err(InfraError::Unsupported(
            "asset URL signing is not implemented by this inference service".to_string(),
        ))
    }
    async fn run_task(&self, request: CreateTaskRequest) -> Result<GenericTaskResult> {
        let timeout_sec = request.wait_timeout_sec;
        let status = self.create_task(request).await?;
        self.start_task(StartTaskRequest {
            task_id: status.task_id,
            wait: true,
            timeout_sec,
        })
        .await
    }
}

#[derive(Clone)]
pub struct InferenceApiState {
    pub service: Arc<dyn InferenceApi>,
}

pub fn router(state: InferenceApiState) -> Router {
    Router::new()
        .route("/rpc/infer", post(handle_json_rpc))
        .with_state(state)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    #[serde(default)]
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: &'static str,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<Value>,
}

async fn handle_json_rpc(
    State(state): State<InferenceApiState>,
    Json(req): Json<JsonRpcRequest>,
) -> Json<JsonRpcResponse> {
    let id = req.id.clone();
    let generic = match req.method.as_str() {
        "create_task" => {
            let request = match serde_json::from_value::<CreateTaskRequest>(req.params.clone()) {
                Ok(request) => request,
                Err(err) => return Json(error(id, InfraError::from(err))),
            };
            Some(json_result(
                id.clone(),
                state.service.create_task(request).await,
            ))
        }
        "start_task" => {
            let request = match serde_json::from_value::<StartTaskRequest>(req.params.clone()) {
                Ok(request) => request,
                Err(err) => return Json(error(id, InfraError::from(err))),
            };
            Some(json_result(
                id.clone(),
                state.service.start_task(request).await,
            ))
        }
        "get_task" => {
            let task_id = match task_id_param(&req.params) {
                Ok(task_id) => task_id,
                Err(err) => return Json(error(id, err)),
            };
            Some(json_result(
                id.clone(),
                state.service.get_task(task_id).await,
            ))
        }
        "wait_task" => {
            let request = match serde_json::from_value::<WaitTaskRequest>(req.params.clone()) {
                Ok(request) => request,
                Err(err) => return Json(error(id, InfraError::from(err))),
            };
            Some(json_result(
                id.clone(),
                state.service.wait_task(request).await,
            ))
        }
        "run_task" => {
            let request = match serde_json::from_value::<CreateTaskRequest>(req.params.clone()) {
                Ok(request) => request,
                Err(err) => return Json(error(id, InfraError::from(err))),
            };
            Some(json_result(
                id.clone(),
                state.service.run_task(request).await,
            ))
        }
        "sign_assets" | "sign_asset_urls" => {
            let request = match serde_json::from_value::<AssetSignRequest>(req.params.clone()) {
                Ok(request) => request,
                Err(err) => return Json(error(id, InfraError::from(err))),
            };
            Some(json_result(
                id.clone(),
                state.service.sign_assets(request).await,
            ))
        }
        _ => None,
    };
    if let Some(response) = generic {
        return Json(response);
    }
    let task = match task_from_method(&req.method, &req.params) {
        Ok(task) => task,
        Err(err) => return Json(error(id, err)),
    };
    match state.service.dispatch(task).await {
        Ok(output) => Json(JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!(output)),
            error: None,
        }),
        Err(err) => Json(error(id, err)),
    }
}

fn json_result<T: Serialize>(id: Option<Value>, result: Result<T>) -> JsonRpcResponse {
    match result {
        Ok(value) => JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!(value)),
            error: None,
        },
        Err(err) => error(id, err),
    }
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
            let audio = file_ref(params.get("audio").unwrap_or(params))?;
            Ok(InferenceTask::new(
                TaskKind::AsrTranscribe,
                model_id,
                InferenceInput::AsrTranscribe { audio },
            ))
        }
        "object_detect" => {
            let image = file_ref(params.get("image").unwrap_or(params))?;
            Ok(InferenceTask::new(
                TaskKind::ObjectDetect,
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
                .or_else(|| path_file_ref(params.get("reference_path")));
            let mut task = InferenceTask::new(
                TaskKind::TtsSynthesize,
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
            let texts = text_list(params, &["input", "texts", "text"])?;
            let input_type = params
                .get("input_type")
                .cloned()
                .map(serde_json::from_value)
                .transpose()?
                .unwrap_or_default();
            Ok(InferenceTask::new(
                TaskKind::TextEmbed,
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
            let documents = text_list(params, &["documents"])?;
            let top_n = params
                .get("top_n")
                .and_then(Value::as_u64)
                .map(|value| value as usize);
            Ok(InferenceTask::new(
                TaskKind::TextRerank,
                model_id,
                InferenceInput::TextRerank {
                    query,
                    documents,
                    top_n,
                },
            ))
        }
        other => Err(InfraError::BadRequest(format!(
            "unknown inference method `{other}`"
        ))),
    }
}

fn text_list(params: &Value, keys: &[&str]) -> Result<Vec<String>> {
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

fn path_file_ref(value: Option<&Value>) -> Option<FileRef> {
    value
        .and_then(Value::as_str)
        .map(|path| FileRef::local(std::path::PathBuf::from(path)))
}

fn error(id: Option<Value>, err: InfraError) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0",
        id,
        result: None,
        error: Some(json!({ "code": -32000, "message": err.to_string() })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{to_bytes, Body},
        http::{Request, StatusCode},
    };
    use local_core::{InferenceOutput, TaskKind};
    use std::sync::{Arc, Mutex};
    use tower::ServiceExt;

    #[derive(Default)]
    struct RecordingApi {
        tasks: Mutex<Vec<InferenceTask>>,
    }

    #[async_trait::async_trait]
    impl InferenceApi for RecordingApi {
        async fn dispatch(&self, task: InferenceTask) -> Result<InferenceOutput> {
            let kind = task.kind;
            self.tasks.lock().expect("tasks lock").push(task);
            Ok(match kind {
                TaskKind::AsrTranscribe => InferenceOutput::AsrTranscription {
                    text: "ok".to_string(),
                },
                TaskKind::ObjectDetect => InferenceOutput::ObjectDetections {
                    objects: Vec::new(),
                },
                TaskKind::TtsSynthesize => InferenceOutput::Accepted {
                    job_id: "unsupported-test".to_string(),
                },
                TaskKind::TextEmbed => InferenceOutput::TextEmbeddings {
                    embeddings: vec![vec![1.0]],
                    prompt_tokens: 1,
                },
                TaskKind::TextRerank => InferenceOutput::TextRerank {
                    results: Vec::new(),
                    total_tokens: 1,
                },
            })
        }
    }

    #[tokio::test]
    async fn mcp_infer_route_is_not_registered_while_rpc_infer_remains_canonical() {
        let service = Arc::new(RecordingApi::default());
        let app = router(InferenceApiState {
            service: service.clone(),
        });

        let mcp_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/infer")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":1,"method":"object_detect","params":{"image":{"path":"./image.jpg"}}}"#,
                    ))
                    .expect("request"),
            )
            .await
            .expect("response");

        assert_eq!(mcp_response.status(), StatusCode::NOT_FOUND);

        let rpc_response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/rpc/infer")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":1,"method":"object_detect","params":{"image":{"path":"./image.jpg"}}}"#,
                    ))
                    .expect("request"),
            )
            .await
            .expect("response");

        assert_eq!(rpc_response.status(), StatusCode::OK);
        let tasks = service.tasks.lock().expect("tasks lock");
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].kind, TaskKind::ObjectDetect);
    }

    #[tokio::test]
    async fn object_detect_rpc_route_dispatches_controller_task() {
        let service = Arc::new(RecordingApi::default());
        let app = router(InferenceApiState {
            service: service.clone(),
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/rpc/infer")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":1,"method":"object_detect","params":{"model":"yolo11n.onnx","image":{"path":"./image.jpg","mime":"image/jpeg"}}}"#,
                    ))
                    .expect("request"),
            )
            .await
            .expect("response");

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        let payload: Value = serde_json::from_slice(&body).expect("json response");
        assert!(payload.get("error").is_none(), "{payload:?}");

        let tasks = service.tasks.lock().expect("tasks lock");
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].kind, TaskKind::ObjectDetect);
        assert_eq!(tasks[0].model_id.as_deref(), Some("yolo11n.onnx"));
        match &tasks[0].input {
            InferenceInput::ObjectDetect { image } => {
                assert_eq!(
                    image.path.as_deref(),
                    Some(std::path::Path::new("./image.jpg"))
                );
                assert_eq!(image.mime.as_deref(), Some("image/jpeg"));
            }
            other => panic!("unexpected input: {other:?}"),
        }
    }

    #[tokio::test]
    async fn asr_transcribe_rpc_route_dispatches_controller_task() {
        let service = Arc::new(RecordingApi::default());
        let app = router(InferenceApiState {
            service: service.clone(),
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/rpc/infer")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":2,"method":"asr_transcribe","params":{"model":"qwen3-asr-0.6b-onnx","audio":{"path":"./audio.wav","mime":"audio/wav"}}}"#,
                    ))
                    .expect("request"),
            )
            .await
            .expect("response");

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        let payload: Value = serde_json::from_slice(&body).expect("json response");
        assert!(payload.get("error").is_none(), "{payload:?}");

        let tasks = service.tasks.lock().expect("tasks lock");
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].kind, TaskKind::AsrTranscribe);
        assert_eq!(tasks[0].model_id.as_deref(), Some("qwen3-asr-0.6b-onnx"));
        match &tasks[0].input {
            InferenceInput::AsrTranscribe { audio } => {
                assert_eq!(
                    audio.path.as_deref(),
                    Some(std::path::Path::new("./audio.wav"))
                );
                assert_eq!(audio.mime.as_deref(), Some("audio/wav"));
            }
            other => panic!("unexpected input: {other:?}"),
        }
    }

    #[tokio::test]
    async fn tts_synthesize_rpc_route_dispatches_controller_task() {
        let service = Arc::new(RecordingApi::default());
        let app = router(InferenceApiState {
            service: service.clone(),
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/rpc/infer")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":3,"method":"tts_synthesize","params":{"model_id":"indextts-1.5-onnx","text":"hello","reference_audio":{"path":"./ref.wav","mime":"audio/wav"}}}"#,
                    ))
                    .expect("request"),
            )
            .await
            .expect("response");

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        let payload: Value = serde_json::from_slice(&body).expect("json response");
        assert!(payload.get("error").is_none(), "{payload:?}");

        let tasks = service.tasks.lock().expect("tasks lock");
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].kind, TaskKind::TtsSynthesize);
        assert_eq!(tasks[0].model_id.as_deref(), Some("indextts-1.5-onnx"));
        match &tasks[0].input {
            InferenceInput::TtsSynthesize {
                text,
                reference_audio,
            } => {
                assert_eq!(text, "hello");
                let reference_audio = reference_audio.as_ref().expect("reference audio");
                assert_eq!(
                    reference_audio.path.as_deref(),
                    Some(std::path::Path::new("./ref.wav"))
                );
                assert_eq!(reference_audio.mime.as_deref(), Some("audio/wav"));
            }
            other => panic!("unexpected input: {other:?}"),
        }
    }

    #[tokio::test]
    async fn tts_synthesize_accepts_reference_path_alias() {
        let service = Arc::new(RecordingApi::default());
        let app = router(InferenceApiState {
            service: service.clone(),
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/rpc/infer")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":4,"method":"tts_synthesize","params":{"model":"indextts-1.5-onnx","text":"hello","reference_path":"./ref.wav"}}"#,
                    ))
                    .expect("request"),
            )
            .await
            .expect("response");

        assert_eq!(response.status(), StatusCode::OK);
        let tasks = service.tasks.lock().expect("tasks lock");
        match &tasks[0].input {
            InferenceInput::TtsSynthesize {
                reference_audio, ..
            } => assert_eq!(
                reference_audio
                    .as_ref()
                    .and_then(|file| file.path.as_deref()),
                Some(std::path::Path::new("./ref.wav"))
            ),
            other => panic!("unexpected input: {other:?}"),
        }
    }
}
