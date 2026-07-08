use async_trait::async_trait;
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use local_core::{FileRef, InferenceInput, InferenceOutput, InferenceTask, ModelSpec, TaskKind};
use local_error::Result;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;

#[async_trait]
pub trait OpenAiApi: Send + Sync + 'static {
    async fn list_models(&self) -> Result<Vec<ModelSpec>>;
    async fn dispatch(&self, task: InferenceTask) -> Result<InferenceOutput>;
}

#[derive(Clone)]
pub struct OpenAiApiState {
    pub service: Arc<dyn OpenAiApi>,
}

pub fn router(state: OpenAiApiState) -> Router {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/audio/transcriptions", post(transcriptions))
        .route("/v1/audio/speech", post(speech))
        .with_state(state)
}

#[derive(Debug, Serialize)]
struct ModelListResponse {
    object: &'static str,
    data: Vec<ModelObject>,
}

#[derive(Debug, Serialize)]
struct ModelObject {
    id: String,
    object: &'static str,
    owned_by: &'static str,
}

async fn list_models(State(state): State<OpenAiApiState>) -> impl IntoResponse {
    match state.service.list_models().await {
        Ok(models) => (
            StatusCode::OK,
            Json(json!(ModelListResponse {
                object: "list",
                data: models
                    .into_iter()
                    .map(|m| ModelObject {
                        id: m.id,
                        object: "model",
                        owned_by: "local"
                    })
                    .collect()
            })),
        )
            .into_response(),
        Err(err) => error_response(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()),
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TranscriptionRequest {
    pub model: String,
    #[serde(default)]
    pub file: Option<FileRef>,
    #[serde(default)]
    pub path: Option<std::path::PathBuf>,
}

#[derive(Debug, Serialize)]
struct TranscriptionResponse {
    text: String,
}

async fn transcriptions(
    State(state): State<OpenAiApiState>,
    Json(req): Json<TranscriptionRequest>,
) -> impl IntoResponse {
    let audio = req.file.unwrap_or_else(|| FileRef {
        path: req.path,
        ..FileRef::default()
    });
    let task = InferenceTask::new(
        TaskKind::AsrTranscribe,
        Some(req.model),
        InferenceInput::AsrTranscribe { audio },
    );
    match state.service.dispatch(task).await {
        Ok(InferenceOutput::AsrTranscription { text }) => {
            (StatusCode::OK, Json(json!(TranscriptionResponse { text }))).into_response()
        }
        Ok(other) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("unexpected inference output: {other:?}"),
        ),
        Err(err) => error_response(StatusCode::NOT_IMPLEMENTED, err.to_string()),
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct SpeechRequest {
    pub model: String,
    #[serde(default, alias = "input")]
    pub text: Option<String>,
    #[serde(default)]
    pub reference_audio: Option<FileRef>,
    #[serde(default)]
    pub reference_path: Option<std::path::PathBuf>,
}

#[derive(Debug, Serialize)]
struct SpeechResponse {
    audio: FileRef,
}

async fn speech(
    State(state): State<OpenAiApiState>,
    Json(req): Json<SpeechRequest>,
) -> impl IntoResponse {
    let text = match req.text {
        Some(text) if !text.trim().is_empty() => text,
        _ => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "speech input/text is required".to_string(),
            )
        }
    };
    let reference_audio = req.reference_audio.or_else(|| {
        req.reference_path.map(|path| FileRef {
            path: Some(path),
            ..FileRef::default()
        })
    });
    let task = InferenceTask::new(
        TaskKind::TtsSynthesize,
        Some(req.model),
        InferenceInput::TtsSynthesize {
            text,
            reference_audio,
        },
    );
    match state.service.dispatch(task).await {
        Ok(InferenceOutput::TtsAudio { audio }) => {
            (StatusCode::OK, Json(json!(SpeechResponse { audio }))).into_response()
        }
        Ok(other) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("unexpected inference output: {other:?}"),
        ),
        Err(err) => error_response(StatusCode::NOT_IMPLEMENTED, err.to_string()),
    }
}

fn error_response(status: StatusCode, message: String) -> axum::response::Response {
    (
        status,
        Json(json!({ "error": { "message": message, "type": "local_inference_error" } })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{to_bytes, Body},
        http::{Request, StatusCode},
    };
    use local_core::{
        AdapterKind, BackendKind, LoadPolicy, ResourceRequirement, RuntimePolicy, TaskKind,
    };
    use std::{collections::BTreeMap, sync::Mutex};
    use tower::ServiceExt;

    #[derive(Default)]
    struct RecordingOpenAiApi {
        tasks: Mutex<Vec<InferenceTask>>,
    }

    #[async_trait::async_trait]
    impl OpenAiApi for RecordingOpenAiApi {
        async fn list_models(&self) -> Result<Vec<ModelSpec>> {
            Ok(vec![ModelSpec {
                id: "qwen3-asr-0.6b-onnx".to_string(),
                name: "Qwen3 ASR 0.6B ONNX INT4".to_string(),
                enabled: true,
                task_kinds: vec![TaskKind::AsrTranscribe],
                adapter: AdapterKind::QwenAsr,
                backend: BackendKind::Ort,
                artifacts: Vec::new(),
                runtime: RuntimePolicy {
                    provider_order: vec!["cpu".to_string()],
                    max_concurrency: 1,
                    idle_ttl_sec: 300,
                },
                resources: ResourceRequirement::default(),
                load_policy: LoadPolicy::default(),
                metadata: BTreeMap::new(),
            }])
        }

        async fn dispatch(&self, task: InferenceTask) -> Result<InferenceOutput> {
            let kind = task.kind;
            self.tasks.lock().expect("tasks lock").push(task);
            Ok(match kind {
                TaskKind::AsrTranscribe => InferenceOutput::AsrTranscription {
                    text: "ok".to_string(),
                },
                TaskKind::TtsSynthesize => InferenceOutput::TtsAudio {
                    audio: FileRef {
                        path: Some(std::path::PathBuf::from("workdir/data/speech.wav")),
                        mime: Some("audio/wav".to_string()),
                        ..FileRef::default()
                    },
                },
                TaskKind::ObjectDetect => InferenceOutput::ObjectDetections {
                    objects: Vec::new(),
                },
            })
        }
    }

    #[tokio::test]
    async fn audio_transcriptions_route_dispatches_asr_task() {
        let service = std::sync::Arc::new(RecordingOpenAiApi::default());
        let app = router(OpenAiApiState {
            service: service.clone(),
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/audio/transcriptions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"qwen3-asr-0.6b-onnx","path":"./audio.wav"}"#,
                    ))
                    .expect("request"),
            )
            .await
            .expect("response");

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        assert!(
            std::str::from_utf8(&body).expect("utf8").contains("ok"),
            "{}",
            std::str::from_utf8(&body).unwrap_or("<non-utf8>")
        );

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
            }
            other => panic!("unexpected input: {other:?}"),
        }
    }

    #[tokio::test]
    async fn audio_speech_route_dispatches_tts_task_and_returns_file_ref() {
        let service = std::sync::Arc::new(RecordingOpenAiApi::default());
        let app = router(OpenAiApiState {
            service: service.clone(),
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/audio/speech")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"indextts-1.5-onnx","input":"hello","reference_path":"./ref.wav"}"#,
                    ))
                    .expect("request"),
            )
            .await
            .expect("response");

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        let payload: serde_json::Value = serde_json::from_slice(&body).expect("json response");
        assert_eq!(
            payload
                .get("audio")
                .and_then(|audio| audio.get("path"))
                .and_then(serde_json::Value::as_str),
            Some("workdir/data/speech.wav")
        );

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
                assert_eq!(
                    reference_audio
                        .as_ref()
                        .and_then(|file| file.path.as_deref()),
                    Some(std::path::Path::new("./ref.wav"))
                );
            }
            other => panic!("unexpected input: {other:?}"),
        }
    }
}
