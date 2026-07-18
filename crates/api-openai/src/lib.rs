use async_trait::async_trait;
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use local_core::{
    EmbeddingInputType, FileRef, InferenceInput, InferenceOutput, InferenceTask, ModelSpec,
    TaskKind,
};
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
        .route("/v1/embeddings", post(embeddings))
        .route("/rerank", post(rerank))
        .route("/v1/rerank", post(rerank))
        .route("/v2/rerank", post(rerank))
        .with_state(state)
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Batch(Vec<String>),
}

impl EmbeddingInput {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::Single(text) => vec![text],
            Self::Batch(texts) => texts,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingsRequest {
    pub model: String,
    pub input: EmbeddingInput,
    #[serde(default)]
    pub encoding_format: Option<String>,
    #[serde(default)]
    pub dimensions: Option<usize>,
    #[serde(default)]
    pub input_type: EmbeddingInputType,
}

#[derive(Debug, Serialize)]
struct EmbeddingObject {
    object: &'static str,
    embedding: Vec<f32>,
    index: usize,
}

async fn embeddings(
    State(state): State<OpenAiApiState>,
    Json(req): Json<EmbeddingsRequest>,
) -> impl IntoResponse {
    if req
        .encoding_format
        .as_deref()
        .is_some_and(|value| value != "float")
    {
        return error_response(
            StatusCode::BAD_REQUEST,
            "only encoding_format=float is supported".to_string(),
        );
    }
    if req.dimensions.is_some_and(|dimensions| dimensions != 384) {
        return error_response(
            StatusCode::BAD_REQUEST,
            "multilingual-e5-small has a fixed 384-dimensional output".to_string(),
        );
    }
    let texts = req.input.into_vec();
    if texts.is_empty() || texts.iter().any(|text| text.trim().is_empty()) {
        return error_response(
            StatusCode::BAD_REQUEST,
            "input must contain one or more non-empty strings".to_string(),
        );
    }
    let model = req.model;
    let task = InferenceTask::new(
        TaskKind::TextEmbed,
        Some(model.clone()),
        InferenceInput::TextEmbed {
            texts,
            input_type: req.input_type,
        },
    );
    match state.service.dispatch(task).await {
        Ok(InferenceOutput::TextEmbeddings {
            embeddings,
            prompt_tokens,
        }) => {
            let data = embeddings
                .into_iter()
                .enumerate()
                .map(|(index, embedding)| EmbeddingObject {
                    object: "embedding",
                    embedding,
                    index,
                })
                .collect::<Vec<_>>();
            (
                StatusCode::OK,
                Json(json!({
                    "object": "list",
                    "data": data,
                    "model": model,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "total_tokens": prompt_tokens
                    }
                })),
            )
                .into_response()
        }
        Ok(other) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("unexpected inference output: {other:?}"),
        ),
        Err(err) => error_response(StatusCode::NOT_IMPLEMENTED, err.to_string()),
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct RerankRequest {
    pub model: String,
    pub query: String,
    pub documents: Vec<String>,
    #[serde(default)]
    pub top_n: Option<usize>,
}

async fn rerank(
    State(state): State<OpenAiApiState>,
    Json(req): Json<RerankRequest>,
) -> impl IntoResponse {
    if req.query.trim().is_empty()
        || req.documents.is_empty()
        || req
            .documents
            .iter()
            .any(|document| document.trim().is_empty())
    {
        return error_response(
            StatusCode::BAD_REQUEST,
            "query and documents must contain non-empty strings".to_string(),
        );
    }
    let model = req.model;
    let task = InferenceTask::new(
        TaskKind::TextRerank,
        Some(model.clone()),
        InferenceInput::TextRerank {
            query: req.query,
            documents: req.documents,
            top_n: req.top_n,
        },
    );
    let response_id = format!("rerank-{}", task.id);
    match state.service.dispatch(task).await {
        Ok(InferenceOutput::TextRerank {
            results,
            total_tokens,
        }) => (
            StatusCode::OK,
            Json(json!({
                "id": response_id,
                "model": model,
                "usage": { "total_tokens": total_tokens },
                "results": results.into_iter().map(|result| json!({
                    "index": result.index,
                    "document": { "text": result.document },
                    "relevance_score": result.relevance_score
                })).collect::<Vec<_>>()
            })),
        )
            .into_response(),
        Ok(other) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("unexpected inference output: {other:?}"),
        ),
        Err(err) => error_response(StatusCode::NOT_IMPLEMENTED, err.to_string()),
    }
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
                TaskKind::TextEmbed => InferenceOutput::TextEmbeddings {
                    embeddings: vec![vec![0.25; 384]],
                    prompt_tokens: 3,
                },
                TaskKind::TextRerank => InferenceOutput::TextRerank {
                    results: vec![local_core::RerankResult {
                        index: 0,
                        relevance_score: 0.9,
                        document: "doc".to_string(),
                    }],
                    total_tokens: 5,
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

    #[tokio::test]
    async fn embeddings_route_is_openai_compatible() {
        let service = std::sync::Arc::new(RecordingOpenAiApi::default());
        let app = router(OpenAiApiState {
            service: service.clone(),
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/embeddings")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"multilingual-e5-small-onnx","input":["hello"],"input_type":"query"}"#,
                    ))
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        let payload: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(payload["object"], "list");
        assert_eq!(payload["data"][0]["object"], "embedding");
        assert_eq!(
            payload["data"][0]["embedding"].as_array().unwrap().len(),
            384
        );
        assert_eq!(payload["usage"]["prompt_tokens"], 3);
        let tasks = service.tasks.lock().expect("tasks");
        assert_eq!(tasks[0].kind, TaskKind::TextEmbed);
    }

    #[tokio::test]
    async fn rerank_route_matches_vllm_shape() {
        let service = std::sync::Arc::new(RecordingOpenAiApi::default());
        let app = router(OpenAiApiState {
            service: service.clone(),
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/rerank")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"mmarco-minilm-l12-onnx","query":"q","documents":["doc"],"top_n":1}"#,
                    ))
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        let payload: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert!(payload["id"].as_str().unwrap().starts_with("rerank-"));
        assert_eq!(payload["results"][0]["document"]["text"], "doc");
        assert_eq!(payload["usage"]["total_tokens"], 5);
    }
}
