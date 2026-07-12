use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use local_core::{
    AdapterKind, BackendKind, InferenceTask, WorkerHeartbeat, WorkerRegistration,
    WorkerRegistrationResponse,
};
use local_error::{InfraError, Result};
use local_registry::ModelRegistry;
use local_runtime::{IdleMaintenanceLoopHandle, RuntimeManager, RuntimeManagerConfig};
use serde_json::json;
use std::{sync::Arc, time::Duration};
use tokio::sync::RwLock;

const WORKER_TOKEN_HEADER: &str = "x-local-worker-token";

#[derive(Clone)]
pub struct WorkerState {
    node_id: String,
    base_url: String,
    registration_token: Option<String>,
    session_token: Arc<RwLock<Option<String>>>,
    runtime: Arc<RuntimeManager>,
    _idle_maintenance_loop: Arc<IdleMaintenanceLoopHandle>,
    http: reqwest::Client,
}

impl WorkerState {
    pub async fn from_registry(
        node_id: String,
        base_url: String,
        registration_token: Option<String>,
        registry: ModelRegistry,
        runtime_config: RuntimeManagerConfig,
    ) -> Self {
        let specs = registry.list().await;
        let runtime = Arc::new(RuntimeManager::new(specs, runtime_config));
        let idle_maintenance_loop = Arc::new(runtime.clone().spawn_idle_maintenance_loop());
        Self {
            node_id,
            base_url,
            registration_token,
            session_token: Arc::new(RwLock::new(None)),
            runtime,
            _idle_maintenance_loop: idle_maintenance_loop,
            http: reqwest::Client::new(),
        }
    }

    pub fn app(self) -> Router {
        Router::new()
            .route("/health", get(health))
            .route("/internal/infer", post(infer))
            .with_state(self)
    }

    pub async fn registration(&self) -> WorkerRegistration {
        WorkerRegistration {
            node_id: self.node_id.clone(),
            base_url: self.base_url.clone(),
            registration_token: self.registration_token.clone(),
            supported_backends: vec![BackendKind::Ort],
            supported_adapters: vec![
                AdapterKind::Yolo,
                AdapterKind::QwenAsr,
                AdapterKind::IndexTts,
            ],
            resources: local_hardware::snapshot(),
        }
    }

    pub async fn heartbeat(&self) -> WorkerHeartbeat {
        WorkerHeartbeat {
            node_id: self.node_id.clone(),
            resources: local_hardware::snapshot(),
            loaded_models: self.runtime.loaded_models().await,
            queued_jobs: self.runtime.queued_jobs(),
        }
    }

    pub async fn register_with_controller(&self, controller_url: &str) -> Result<()> {
        let url = format!(
            "{}/internal/workers/register",
            controller_url.trim_end_matches('/')
        );
        let response = self
            .http
            .post(url)
            .json(&self.registration().await)
            .send()
            .await
            .map_err(|e| InfraError::Backend(format!("register worker: {e}")))?;
        if !response.status().is_success() {
            return Err(InfraError::Backend(format!(
                "controller rejected worker registration: {}",
                response.status()
            )));
        }
        let body = response
            .json::<WorkerRegistrationResponse>()
            .await
            .map_err(|e| {
                InfraError::Backend(format!("decode worker registration response: {e}"))
            })?;
        *self.session_token.write().await = Some(body.session_token);
        Ok(())
    }

    async fn session_token(&self) -> Option<String> {
        self.session_token.read().await.clone()
    }

    async fn clear_session_token(&self) {
        *self.session_token.write().await = None;
    }

    pub fn spawn_heartbeat(self, controller_url: String, interval: Duration) {
        tokio::spawn(async move {
            let url = format!(
                "{}/internal/workers/heartbeat",
                controller_url.trim_end_matches('/')
            );
            loop {
                if self.session_token().await.is_none() {
                    if let Err(err) = self.register_with_controller(&controller_url).await {
                        tracing::warn!(error = %err, "worker registration retry failed");
                        tokio::time::sleep(interval).await;
                        continue;
                    }
                }
                let heartbeat = self.heartbeat().await;
                let token = self.session_token().await.unwrap_or_default();
                match self
                    .http
                    .post(&url)
                    .header(WORKER_TOKEN_HEADER, token)
                    .json(&heartbeat)
                    .send()
                    .await
                {
                    Ok(response) if response.status().is_success() => {
                        tracing::debug!(node_id = self.node_id, "heartbeat sent")
                    }
                    Ok(response) => {
                        let status = response.status();
                        tracing::warn!(status = ?status, "controller rejected heartbeat");
                        if matches!(status.as_u16(), 400 | 401 | 404) {
                            self.clear_session_token().await;
                            tracing::warn!(
                                status = ?status,
                                "cleared worker session token; will re-register on next heartbeat"
                            );
                        }
                    }
                    Err(err) => tracing::warn!(error = %err, "failed to send heartbeat"),
                }
                tokio::time::sleep(interval).await;
            }
        });
    }
}

async fn health() -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(json!({ "status": "ok", "role": "worker" })),
    )
}

async fn infer(
    State(state): State<WorkerState>,
    headers: HeaderMap,
    Json(task): Json<InferenceTask>,
) -> impl IntoResponse {
    let handler_started = std::time::Instant::now();
    let expected = state.session_token().await;
    let provided = headers
        .get(WORKER_TOKEN_HEADER)
        .and_then(|value| value.to_str().ok());
    if expected.as_deref().is_none() || provided != expected.as_deref() {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({ "error": "missing or invalid worker session token" })),
        )
            .into_response();
    }
    let request_id = task.id;
    let result = state.runtime.infer(task).await;
    tracing::info!(
        request_id = %request_id,
        handler_total_ms = handler_started.elapsed().as_millis() as u64,
        queued_jobs = state.runtime.queued_jobs(),
        active_jobs = state.runtime.active_jobs(),
        success = result.is_ok(),
        "worker inference handler completed"
    );
    match result {
        Ok(output) => (StatusCode::OK, Json(json!(output))).into_response(),
        Err(err) => (
            StatusCode::NOT_IMPLEMENTED,
            Json(json!({ "error": err.to_string() })),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use local_core::{
        FileRef, InferenceInput, LoadPolicy, ModelSpec, ResourceRequirement, RuntimePolicy,
        TaskKind,
    };
    use local_runtime::RuntimeManagerConfig;

    #[tokio::test]
    async fn internal_infer_routes_into_runtime_and_returns_explicit_error() {
        let registry = ModelRegistry::from_models(vec![missing_yolo_model()]);
        let state = WorkerState::from_registry(
            "worker-test".to_string(),
            "http://127.0.0.1:0".to_string(),
            Some("registration".to_string()),
            registry,
            RuntimeManagerConfig::default(),
        )
        .await;
        *state.session_token.write().await = Some("session".to_string());
        let task = InferenceTask::new(
            TaskKind::ObjectDetect,
            Some("yolo11n.onnx".to_string()),
            InferenceInput::ObjectDetect {
                image: FileRef::local("image.jpg"),
            },
        );

        let mut headers = HeaderMap::new();
        headers.insert(WORKER_TOKEN_HEADER, "session".parse().expect("header"));
        let response = infer(State(state), headers, Json(task))
            .await
            .into_response();

        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        let body = std::str::from_utf8(&body).expect("utf8");
        assert!(
            body.contains("YOLO .onnx artifact path is not configured"),
            "{body}"
        );
    }

    fn missing_yolo_model() -> ModelSpec {
        ModelSpec {
            id: "yolo11n.onnx".to_string(),
            name: "YOLO11n ONNX".to_string(),
            enabled: true,
            task_kinds: vec![TaskKind::ObjectDetect],
            adapter: AdapterKind::Yolo,
            backend: BackendKind::Ort,
            artifacts: Vec::new(),
            runtime: RuntimePolicy {
                provider_order: vec!["cpu".to_string()],
                max_concurrency: 1,
                idle_ttl_sec: 300,
            },
            resources: ResourceRequirement::default(),
            load_policy: LoadPolicy::default(),
            metadata: Default::default(),
        }
    }
}
