use async_trait::async_trait;
use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, Path as AxumPath, Query, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use chrono::Utc;
use hmac::{Hmac, Mac};
use local_api_mcp_admin::{AdminApi, AdminApiState};
use local_api_mcp_infer::{InferenceApi, InferenceApiState};
use local_api_openai::{OpenAiApi, OpenAiApiState};
use local_core::{
    AssetKind, AssetListQuery, AssetRecord, AssetSignItem, AssetSignRequest, AssetSignResponse,
    AssetSignedUrl, AssetUrlOperation, CreateTaskRequest, DownloadStatus, FileRef,
    GenericTaskResult, GenericTaskState, InferenceInput, InferenceOutput, InferenceTask, JobState,
    ModelSpec, NodeStatus, StartTaskRequest, TaskStatus, TaskUploadSlot, WaitTaskRequest,
    WorkerHeartbeat, WorkerRegistration, WorkerRegistrationResponse,
};
use local_error::{InfraError, Result};
use local_files::{asset_uri, normalize_asset_path, parse_asset_uri, AssetsStore};
use local_model_store::SqliteModelStore;
use local_registry::ModelRegistry;
use local_scheduler::Scheduler;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::{collections::BTreeMap, fs, io::Write, sync::Arc, time::Duration};
use tokio::sync::RwLock;
use uuid::Uuid;

mod standard_mcp;

type HmacSha256 = Hmac<Sha256>;

const WORKER_TOKEN_HEADER: &str = "x-local-worker-token";
const DEFAULT_UPLOAD_URL_TTL_SECS: i64 = 900;
const DEFAULT_ASSET_URL_TTL_SECS: i64 = 600;
const DEFAULT_MATERIAL_ASSET_TTL_SECS: i64 = 10 * 60;
const DEFAULT_ARTIFACT_ASSET_TTL_SECS: i64 = 24 * 60 * 60;
const DEFAULT_ASSET_CLEANUP_INTERVAL: Duration = Duration::from_secs(5 * 60);

#[derive(Clone)]
pub struct ControllerState {
    registry: ModelRegistry,
    nodes: Arc<RwLock<BTreeMap<String, NodeStatus>>>,
    worker_session_tokens: Arc<RwLock<BTreeMap<String, String>>>,
    generic_tasks: Arc<RwLock<BTreeMap<String, TaskStatus>>>,
    scheduler: Scheduler,
    http: reqwest::Client,
    store: Option<SqliteModelStore>,
    worker_registration_token: Option<String>,
    public_base_url: String,
    data_dir: PathBuf,
    assets: AssetsStore,
    upload_secret: String,
    admin_token: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ControllerOptions {
    pub worker_registration_token: Option<String>,
    pub public_base_url: String,
    pub data_dir: PathBuf,
    pub upload_signing_secret: Option<String>,
    pub admin_token: Option<String>,
    pub asset_cleanup_interval: Option<Duration>,
}

impl Default for ControllerOptions {
    fn default() -> Self {
        Self {
            worker_registration_token: None,
            public_base_url: "http://127.0.0.1:17890".to_string(),
            data_dir: PathBuf::from("workdir/data"),
            upload_signing_secret: None,
            admin_token: None,
            asset_cleanup_interval: Some(DEFAULT_ASSET_CLEANUP_INTERVAL),
        }
    }
}

fn start_asset_cleanup_loop(assets: AssetsStore, interval: Duration) {
    if interval.is_zero() {
        tracing::warn!("asset cleanup loop disabled because interval is zero");
        return;
    }
    let Ok(handle) = tokio::runtime::Handle::try_current() else {
        tracing::warn!("asset cleanup loop not started because no Tokio runtime is active");
        return;
    };
    handle.spawn(async move {
        loop {
            tokio::time::sleep(interval).await;
            if let Err(err) = assets.cleanup_expired() {
                tracing::warn!(error = %err, "background asset cleanup failed");
            }
        }
    });
}

impl ControllerState {
    pub fn new(registry: ModelRegistry) -> Self {
        Self::new_with_options(registry, None, ControllerOptions::default())
    }

    pub fn with_store(registry: ModelRegistry, store: SqliteModelStore) -> Self {
        Self::new_with_options(registry, Some(store), ControllerOptions::default())
    }

    pub fn with_store_options(
        registry: ModelRegistry,
        store: SqliteModelStore,
        options: ControllerOptions,
    ) -> Self {
        Self::new_with_options(registry, Some(store), options)
    }

    pub fn new_with_options(
        registry: ModelRegistry,
        store: Option<SqliteModelStore>,
        options: ControllerOptions,
    ) -> Self {
        let upload_secret = options.upload_signing_secret.clone().unwrap_or_else(|| {
            format!(
                "{}{}",
                Uuid::new_v4().as_simple(),
                Uuid::new_v4().as_simple()
            )
        });
        let assets = AssetsStore::new(options.data_dir.clone());
        if let Err(err) = assets.cleanup_expired() {
            tracing::warn!(error = %err, "initial asset cleanup failed");
        }
        if let Some(interval) = options.asset_cleanup_interval {
            start_asset_cleanup_loop(assets.clone(), interval);
        }
        Self {
            registry,
            nodes: Arc::new(RwLock::new(BTreeMap::new())),
            worker_session_tokens: Arc::new(RwLock::new(BTreeMap::new())),
            generic_tasks: Arc::new(RwLock::new(BTreeMap::new())),
            scheduler: Scheduler,
            http: reqwest::Client::new(),
            store,
            worker_registration_token: options.worker_registration_token,
            public_base_url: options.public_base_url,
            data_dir: options.data_dir,
            assets,
            upload_secret,
            admin_token: options.admin_token,
        }
    }

    pub fn app(self) -> Router {
        let admin_service: Arc<dyn AdminApi> = Arc::new(self.clone());
        let infer_service: Arc<dyn InferenceApi> = Arc::new(self.clone());
        let openai_service: Arc<dyn OpenAiApi> = Arc::new(self.clone());
        let admin = local_api_mcp_admin::router(AdminApiState {
            service: admin_service,
        });
        let infer = local_api_mcp_infer::router(InferenceApiState {
            service: infer_service,
        });
        let openai = local_api_openai::router(OpenAiApiState {
            service: openai_service,
        });
        Router::new()
            .route("/health", get(health))
            .route("/internal/workers/register", post(register_worker))
            .route("/internal/workers/heartbeat", post(worker_heartbeat))
            .route(
                "/files/upload/:task_id/:slot",
                post(upload_task_file).layer(DefaultBodyLimit::max(512 * 1024 * 1024)),
            )
            .route(
                "/assets/:kind/*path",
                post(upload_asset)
                    .get(download_asset)
                    .delete(delete_asset)
                    .layer(DefaultBodyLimit::max(512 * 1024 * 1024)),
            )
            .route("/assets", get(list_assets))
            .route("/assets/sign", post(sign_assets))
            .with_state(self)
            .merge(admin)
            .merge(infer)
            .merge(openai)
    }

    pub async fn register_worker(
        &self,
        mut registration: WorkerRegistration,
    ) -> Result<WorkerRegistrationResponse> {
        if let Some(expected) = &self.worker_registration_token {
            if registration.registration_token.as_deref() != Some(expected.as_str()) {
                return Err(InfraError::BadRequest(
                    "invalid worker registration token".to_string(),
                ));
            }
        }
        registration.registration_token = None;
        let session_token = Uuid::new_v4().to_string();
        let status = NodeStatus {
            registration: registration.clone(),
            last_heartbeat_at: Utc::now(),
            loaded_models: Vec::new(),
            queued_jobs: 0,
        };
        self.nodes
            .write()
            .await
            .insert(registration.node_id.clone(), status.clone());
        self.worker_session_tokens
            .write()
            .await
            .insert(registration.node_id.clone(), session_token.clone());
        if let Some(store) = &self.store {
            store.record_worker_auth(&status, Some(&sha256_hex(session_token.as_bytes())))?;
        }
        Ok(WorkerRegistrationResponse {
            status,
            session_token,
        })
    }

    pub async fn heartbeat(
        &self,
        token: Option<&str>,
        heartbeat: WorkerHeartbeat,
    ) -> Result<NodeStatus> {
        self.validate_worker_session(&heartbeat.node_id, token)
            .await?;
        let mut nodes = self.nodes.write().await;
        let status = nodes.get_mut(&heartbeat.node_id).ok_or_else(|| {
            InfraError::NotFound(format!("worker `{}` is not registered", heartbeat.node_id))
        })?;
        status.last_heartbeat_at = Utc::now();
        status.registration.resources = heartbeat.resources;
        status.loaded_models = heartbeat.loaded_models;
        status.queued_jobs = heartbeat.queued_jobs;
        let status = status.clone();
        if let Some(store) = &self.store {
            store.record_worker(&status)?;
        }
        Ok(status)
    }

    async fn validate_worker_session(&self, node_id: &str, token: Option<&str>) -> Result<()> {
        let expected = self
            .worker_session_tokens
            .read()
            .await
            .get(node_id)
            .cloned()
            .ok_or_else(|| {
                InfraError::BadRequest(format!("worker `{node_id}` is not registered"))
            })?;
        if token != Some(expected.as_str()) {
            return Err(InfraError::BadRequest(
                "invalid worker session token".to_string(),
            ));
        }
        Ok(())
    }

    async fn forward_to_worker(&self, task: InferenceTask) -> Result<InferenceOutput> {
        let total_started = std::time::Instant::now();
        let request_id = task.id;
        if let Some(store) = &self.store {
            store.record_job_state(&task, JobState::Queued, None, None)?;
        }
        let models = if let Some(model_id) = &task.model_id {
            vec![self.registry.get(model_id).await.ok_or_else(|| {
                InfraError::ModelNotConfigured {
                    model_id: model_id.clone(),
                    reason: "model is not registered".to_string(),
                }
            })?]
        } else {
            self.registry.enabled_for_task(task.kind).await
        };
        let model = models.into_iter().find(|m| m.enabled).ok_or_else(|| {
            InfraError::ModelNotConfigured {
                model_id: task
                    .model_id
                    .clone()
                    .unwrap_or_else(|| "<auto>".to_string()),
                reason: "no enabled model available for task".to_string(),
            }
        })?;
        let nodes = self
            .nodes
            .read()
            .await
            .values()
            .cloned()
            .collect::<Vec<_>>();
        let worker = self.scheduler.select_worker(&model, &nodes).ok_or_else(|| InfraError::Unsupported("no registered worker can serve the requested model; controller does not load models".to_string()))?;
        let schedule_ms = total_started.elapsed().as_millis() as u64;
        let worker_token = self
            .worker_session_tokens
            .read()
            .await
            .get(&worker.registration.node_id)
            .cloned()
            .ok_or_else(|| {
                InfraError::Backend("selected worker has no session token".to_string())
            })?;
        if let Some(store) = &self.store {
            store.record_job_state(&task, JobState::Running, None, None)?;
        }
        let url = format!(
            "{}/internal/infer",
            worker.registration.base_url.trim_end_matches('/')
        );
        let http_started = std::time::Instant::now();
        let response = self
            .http
            .post(url)
            .header(WORKER_TOKEN_HEADER, worker_token)
            .json(&task)
            .send()
            .await
            .map_err(|e| InfraError::Backend(format!("forward inference task to worker: {e}")))?;
        let worker_response_headers_ms = http_started.elapsed().as_millis() as u64;
        if !response.status().is_success() {
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "<empty worker error>".to_string());
            let err = InfraError::Backend(format!("worker returned non-success response: {text}"));
            if let Some(store) = &self.store {
                store.record_job_state(&task, JobState::Failed, None, Some(&err.to_string()))?;
            }
            return Err(err);
        }
        let body_started = std::time::Instant::now();
        let output = response
            .json::<InferenceOutput>()
            .await
            .map_err(|e| InfraError::Backend(format!("decode worker inference response: {e}")))?;
        let response_body_decode_ms = body_started.elapsed().as_millis() as u64;
        if let Some(store) = &self.store {
            store.record_job_state(&task, JobState::Succeeded, Some(&output), None)?;
        }
        tracing::info!(
            request_id = %request_id,
            schedule_ms,
            worker_response_headers_ms,
            response_body_decode_ms,
            total_ms = total_started.elapsed().as_millis() as u64,
            "controller worker dispatch completed"
        );
        Ok(output)
    }

    async fn create_generic_task(&self, request: CreateTaskRequest) -> Result<TaskStatus> {
        let task_id = Uuid::new_v4().to_string();
        let expires_at = Utc::now() + chrono::Duration::seconds(DEFAULT_UPLOAD_URL_TTL_SECS);
        let uploads = request
            .files
            .iter()
            .enumerate()
            .map(|(idx, file)| {
                let slot = safe_slot(file.role.as_deref().unwrap_or(&file.name), idx);
                let file_name = safe_filename(&file.name);
                let asset_path = format!("tasks/{task_id}/inputs/{file_name}");
                let asset_uri = file
                    .asset_uri
                    .clone()
                    .unwrap_or_else(|| asset_uri(AssetKind::Material, &asset_path));
                let upload_url = if file.asset_uri.is_some() {
                    String::new()
                } else {
                    self.asset_upload_url(AssetKind::Material, &asset_path, file.mime.as_deref())
                };
                TaskUploadSlot {
                    slot,
                    file_name,
                    mime: file.mime.clone(),
                    role: file.role.clone(),
                    required: file.required,
                    upload_url,
                    asset_uri: Some(asset_uri),
                    expires_at,
                    uploaded: file.asset_uri.is_some(),
                    path: None,
                    uploaded_at: None,
                }
            })
            .collect::<Vec<_>>();
        let state = if uploads.iter().any(|slot| slot.required && !slot.uploaded) {
            GenericTaskState::WaitingForUploads
        } else {
            GenericTaskState::Ready
        };
        let now = Utc::now();
        let status = TaskStatus {
            task_id: task_id.clone(),
            task_kind: request.task_kind,
            model_id: request.effective_model_id(),
            state,
            uploads,
            params: request.params,
            output: None,
            files: Vec::new(),
            error: None,
            created_at: now,
            updated_at: now,
        };
        self.save_task(status.clone()).await?;
        Ok(status)
    }

    fn asset_upload_url(&self, kind: AssetKind, path: &str, content_type: Option<&str>) -> String {
        self.sign_asset_url_internal(AssetSignItem {
            operation: AssetUrlOperation::Upload,
            kind: Some(kind),
            path: Some(path.to_string()),
            uri: None,
            content_type: content_type.map(str::to_string),
            ttl_sec: None,
            expires: None,
            url_ttl_sec: Some(DEFAULT_UPLOAD_URL_TTL_SECS),
        })
        .map(|signed| signed.signed_url)
        .unwrap_or_default()
    }

    fn sign_assets_batch(&self, request: AssetSignRequest) -> Result<AssetSignResponse> {
        let items = request
            .items
            .into_iter()
            .map(|item| self.sign_asset_url_public(item))
            .collect::<Result<Vec<_>>>()?;
        Ok(AssetSignResponse { items })
    }

    fn sign_asset_url_public(&self, item: AssetSignItem) -> Result<AssetSignedUrl> {
        self.sign_asset_url(item, false)
    }

    fn sign_asset_url_internal(&self, item: AssetSignItem) -> Result<AssetSignedUrl> {
        self.sign_asset_url(item, true)
    }

    fn sign_asset_url(
        &self,
        item: AssetSignItem,
        allow_reserved_upload: bool,
    ) -> Result<AssetSignedUrl> {
        let (kind, path) = resolve_asset_ref(item.kind, item.path.as_deref(), item.uri.as_deref())?;
        if item.operation == AssetUrlOperation::Upload
            && !allow_reserved_upload
            && is_reserved_asset_upload_path(&path)
        {
            return Err(InfraError::BadRequest(
                "public asset signing cannot create upload URLs for reserved asset paths"
                    .to_string(),
            ));
        }
        let url_ttl = item.url_ttl_sec.unwrap_or(DEFAULT_ASSET_URL_TTL_SECS);
        if url_ttl <= 0 {
            return Err(InfraError::BadRequest(
                "url_ttl_sec must be positive".to_string(),
            ));
        }
        let capability = if item.operation == AssetUrlOperation::Upload
            && allow_reserved_upload
            && is_reserved_asset_upload_path(&path)
        {
            Some("task_upload")
        } else {
            None
        };
        let expires_at = Utc::now() + chrono::Duration::seconds(url_ttl);
        let (asset_expires_at, asset_expiry_query) = asset_expiry_for_signing(kind, &item)?;
        let signed_url = self.signed_asset_url(
            item.operation,
            kind,
            &path,
            expires_at.timestamp(),
            asset_expiry_query.as_deref(),
            capability,
            item.content_type.as_deref(),
        );
        Ok(AssetSignedUrl {
            operation: item.operation,
            uri: asset_uri(kind, &path),
            signed_url,
            method: item.operation.method().to_string(),
            expires_at,
            asset_expires_at,
            content_type: item.content_type,
        })
    }

    async fn upload_asset_bytes(
        &self,
        kind: AssetKind,
        path: String,
        query: BTreeMap<String, String>,
        headers: HeaderMap,
        body: Bytes,
    ) -> Result<AssetRecord> {
        let expires_at = asset_expires_at(kind, &query)?;
        let content_type = query
            .get("content_type")
            .cloned()
            .or_else(|| query.get("content-type").cloned())
            .or_else(|| {
                headers
                    .get(header::CONTENT_TYPE)
                    .and_then(|value| value.to_str().ok())
                    .map(str::to_string)
            });
        let mut record = self
            .assets
            .put(kind, &path, content_type, expires_at, body.as_ref())?;
        self.decorate_asset(&mut record);
        self.mark_uploaded_asset(&record).await?;
        Ok(record)
    }

    async fn mark_uploaded_asset(&self, record: &AssetRecord) -> Result<()> {
        let Some(task_id) = record
            .path
            .strip_prefix("tasks/")
            .and_then(|rest| rest.split('/').next())
            .map(str::to_string)
        else {
            return Ok(());
        };
        let Ok(mut status) = self.load_task(&task_id).await else {
            return Ok(());
        };
        let mut changed = false;
        for upload in &mut status.uploads {
            if upload.asset_uri.as_deref() == Some(record.uri.as_str()) {
                upload.uploaded = true;
                upload.uploaded_at = Some(Utc::now());
                upload.mime = upload.mime.clone().or_else(|| record.content_type.clone());
                changed = true;
            }
        }
        if changed {
            status.updated_at = Utc::now();
            status.state = if missing_required_uploads(&status).is_empty() {
                GenericTaskState::Ready
            } else {
                GenericTaskState::WaitingForUploads
            };
            self.save_task(status).await?;
        }
        Ok(())
    }

    fn decorate_asset(&self, record: &mut AssetRecord) {
        record.download_url = self
            .sign_asset_url_public(AssetSignItem {
                operation: AssetUrlOperation::Download,
                kind: Some(record.kind),
                path: Some(record.path.clone()),
                uri: None,
                content_type: record.content_type.clone(),
                ttl_sec: None,
                expires: None,
                url_ttl_sec: None,
            })
            .ok()
            .map(|signed| signed.signed_url);
    }

    async fn upload_file(
        &self,
        task_id: String,
        slot: String,
        query: BTreeMap<String, String>,
        headers: HeaderMap,
        body: Bytes,
    ) -> Result<UploadResponse> {
        let expires = query
            .get("expires")
            .ok_or_else(|| InfraError::BadRequest("upload URL is missing expires".to_string()))?
            .parse::<i64>()
            .map_err(|e| InfraError::BadRequest(format!("invalid upload expires: {e}")))?;
        let sig = query
            .get("sig")
            .ok_or_else(|| InfraError::BadRequest("upload URL is missing sig".to_string()))?;
        if Utc::now().timestamp() > expires {
            return Err(InfraError::BadRequest("upload URL has expired".to_string()));
        }
        self.verify_upload_signature(&task_id, &slot, expires, sig)?;
        let mut status = self.load_task(&task_id).await?;
        let upload = status
            .uploads
            .iter_mut()
            .find(|upload| upload.slot == slot)
            .ok_or_else(|| InfraError::NotFound(format!("upload slot `{slot}`")))?;
        let dir = self.data_dir.join("uploads").join(&task_id);
        fs::create_dir_all(&dir).map_err(|e| InfraError::io(Some(dir.clone()), e))?;
        let target = dir.join(&upload.file_name);
        let mut file =
            fs::File::create(&target).map_err(|e| InfraError::io(Some(target.clone()), e))?;
        file.write_all(&body)
            .map_err(|e| InfraError::io(Some(target.clone()), e))?;
        if upload.mime.is_none() {
            upload.mime = headers
                .get("content-type")
                .and_then(|value| value.to_str().ok())
                .map(str::to_string);
        }
        upload.path = Some(target);
        upload.uploaded = true;
        upload.uploaded_at = Some(Utc::now());
        status.updated_at = Utc::now();
        status.state = if missing_required_uploads(&status).is_empty() {
            GenericTaskState::Ready
        } else {
            GenericTaskState::WaitingForUploads
        };
        self.save_task(status.clone()).await?;
        if query
            .get("with_start_task")
            .is_some_and(|value| value.eq_ignore_ascii_case("true"))
        {
            let timeout_sec = query
                .get("timeout_sec")
                .and_then(|value| value.parse().ok());
            let result = self
                .start_generic_task(StartTaskRequest {
                    task_id,
                    wait: true,
                    timeout_sec,
                })
                .await?;
            Ok(UploadResponse::Result(result))
        } else {
            Ok(UploadResponse::Status(status))
        }
    }

    async fn start_generic_task(&self, request: StartTaskRequest) -> Result<GenericTaskResult> {
        let mut status = self.load_task(&request.task_id).await?;
        let missing = missing_required_uploads(&status);
        if !missing.is_empty() {
            let err = InfraError::BadRequest(format!(
                "task {} is missing required uploads: {}",
                status.task_id,
                missing.join(", ")
            ));
            self.fail_task_status(&mut status, &err).await?;
            return Err(err);
        }
        let task = match self.inference_task_from_status(&status).await {
            Ok(task) => task,
            Err(err) => {
                self.fail_task_status(&mut status, &err).await?;
                return Err(err);
            }
        };
        if matches!(status.state, GenericTaskState::Succeeded) {
            return Ok(result_from_status(&status));
        }
        status.state = GenericTaskState::Running;
        status.updated_at = Utc::now();
        self.save_task(status.clone()).await?;
        let request_id = task.id;
        let dispatch_started = std::time::Instant::now();
        let result: Result<GenericTaskResult> = async {
            let output = self.forward_to_worker(task).await?;
            let output = self.register_output_assets(request_id, &status.task_id, output)?;
            status.files = output_files(&output);
            status.output = Some(output);
            status.error = None;
            status.state = GenericTaskState::Succeeded;
            status.updated_at = Utc::now();
            self.save_task(status.clone()).await?;
            Ok(result_from_status(&status))
        }
        .await;
        if let Err(err) = &result {
            status.error = Some(err.to_string());
            status.state = GenericTaskState::Failed;
            status.updated_at = Utc::now();
            self.save_task(status.clone()).await?;
        }
        tracing::info!(
            request_id = %request_id,
            controller_dispatch_total_ms = dispatch_started.elapsed().as_millis() as u64,
            success = result.is_ok(),
            "controller generic task dispatch finished"
        );
        result
    }

    async fn fail_task_status(&self, status: &mut TaskStatus, err: &InfraError) -> Result<()> {
        status.error = Some(err.to_string());
        status.state = GenericTaskState::Failed;
        status.updated_at = Utc::now();
        self.save_task(status.clone()).await
    }

    async fn wait_generic_task(&self, request: WaitTaskRequest) -> Result<TaskStatus> {
        let timeout = Duration::from_secs(request.timeout_sec.unwrap_or(30));
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let status = self.load_task(&request.task_id).await?;
            if matches!(
                status.state,
                GenericTaskState::Succeeded
                    | GenericTaskState::Failed
                    | GenericTaskState::Cancelled
            ) || std::time::Instant::now() >= deadline
            {
                return Ok(status);
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }

    async fn load_task(&self, task_id: &str) -> Result<TaskStatus> {
        if let Some(status) = self.generic_tasks.read().await.get(task_id).cloned() {
            return Ok(status);
        }
        if let Some(store) = &self.store {
            if let Some(status) = store.get_generic_task(task_id)? {
                self.generic_tasks
                    .write()
                    .await
                    .insert(task_id.to_string(), status.clone());
                return Ok(status);
            }
        }
        Err(InfraError::NotFound(format!("task `{task_id}`")))
    }

    async fn save_task(&self, status: TaskStatus) -> Result<()> {
        if let Some(store) = &self.store {
            store.record_generic_task(&status)?;
        }
        self.generic_tasks
            .write()
            .await
            .insert(status.task_id.clone(), status);
        Ok(())
    }

    fn upload_signature(&self, task_id: &str, slot: &str, expires: i64) -> String {
        let mut mac = HmacSha256::new_from_slice(self.upload_secret.as_bytes())
            .expect("HMAC accepts keys of any length");
        mac.update(format!("{task_id}:{slot}:{expires}").as_bytes());
        hex(mac.finalize().into_bytes().as_slice())
    }

    fn verify_upload_signature(
        &self,
        task_id: &str,
        slot: &str,
        expires: i64,
        sig: &str,
    ) -> Result<()> {
        let expected = self.upload_signature(task_id, slot, expires);
        if expected.eq_ignore_ascii_case(sig) {
            Ok(())
        } else {
            Err(InfraError::BadRequest(
                "invalid upload URL signature".to_string(),
            ))
        }
    }

    fn signed_asset_url(
        &self,
        operation: AssetUrlOperation,
        kind: AssetKind,
        path: &str,
        url_expires: i64,
        asset_expiry_query: Option<&str>,
        capability: Option<&str>,
        content_type: Option<&str>,
    ) -> String {
        let path = path.trim_start_matches('/');
        let mut params = vec![
            ("action".to_string(), operation.as_str().to_string()),
            ("url_expires".to_string(), url_expires.to_string()),
        ];
        if let Some(value) = asset_expiry_query {
            if let Some((key, value)) = value.split_once('=') {
                params.push((key.to_string(), value.to_string()));
            }
        }
        if let Some(capability) = capability {
            params.push(("capability".to_string(), capability.to_string()));
        }
        if let Some(content_type) = content_type {
            params.push(("content_type".to_string(), content_type.to_string()));
        }
        let sig = self.asset_signature(
            operation,
            kind,
            path,
            url_expires,
            asset_expiry_query,
            capability,
            content_type,
        );
        params.push(("sig".to_string(), sig));
        let query = params
            .into_iter()
            .map(|(key, value)| format!("{}={}", key, percent_encode(&value)))
            .collect::<Vec<_>>()
            .join("&");
        format!(
            "{}/assets/{}/{}?{}",
            self.public_base_url.trim_end_matches('/'),
            kind.as_str(),
            percent_encode_path(path),
            query
        )
    }

    fn asset_signature(
        &self,
        operation: AssetUrlOperation,
        kind: AssetKind,
        path: &str,
        url_expires: i64,
        asset_expiry_query: Option<&str>,
        capability: Option<&str>,
        content_type: Option<&str>,
    ) -> String {
        let mut mac = HmacSha256::new_from_slice(self.upload_secret.as_bytes())
            .expect("HMAC accepts keys of any length");
        mac.update(
            format!(
                "asset:{}:{}:{}:{}:{}:{}:{}",
                operation.as_str(),
                kind.as_str(),
                path,
                url_expires,
                asset_expiry_query.unwrap_or(""),
                capability.unwrap_or(""),
                content_type.unwrap_or("")
            )
            .as_bytes(),
        );
        hex(mac.finalize().into_bytes().as_slice())
    }

    fn verify_asset_signature(
        &self,
        expected_operation: AssetUrlOperation,
        kind: AssetKind,
        path: &str,
        query: &BTreeMap<String, String>,
    ) -> Result<()> {
        let action = query.get("action").ok_or_else(|| {
            InfraError::BadRequest("signed asset URL is missing action".to_string())
        })?;
        if action != expected_operation.as_str() {
            return Err(InfraError::BadRequest(
                "signed asset URL action does not match request method".to_string(),
            ));
        }
        let url_expires = query
            .get("url_expires")
            .ok_or_else(|| {
                InfraError::BadRequest("signed asset URL is missing url_expires".to_string())
            })?
            .parse::<i64>()
            .map_err(|e| InfraError::BadRequest(format!("invalid url_expires: {e}")))?;
        if Utc::now().timestamp() > url_expires {
            return Err(InfraError::BadRequest(
                "signed asset URL has expired".to_string(),
            ));
        }
        let capability = query.get("capability").map(String::as_str);
        if capability.is_some_and(|value| value != "task_upload") {
            return Err(InfraError::BadRequest(
                "invalid signed asset URL capability".to_string(),
            ));
        }
        if expected_operation == AssetUrlOperation::Upload
            && is_reserved_asset_upload_path(path)
            && capability != Some("task_upload")
        {
            return Err(InfraError::BadRequest(
                "reserved asset uploads require an internal task upload capability".to_string(),
            ));
        }
        let sig = query
            .get("sig")
            .ok_or_else(|| InfraError::BadRequest("signed asset URL is missing sig".to_string()))?;
        let asset_expiry_query = canonical_asset_expiry_query(query)?;
        let expected = self.asset_signature(
            expected_operation,
            kind,
            path,
            url_expires,
            asset_expiry_query.as_deref(),
            capability,
            query.get("content_type").map(String::as_str),
        );
        if expected.eq_ignore_ascii_case(sig) {
            Ok(())
        } else {
            Err(InfraError::BadRequest(
                "invalid signed asset URL signature".to_string(),
            ))
        }
    }

    async fn inference_task_from_status(&self, status: &TaskStatus) -> Result<InferenceTask> {
        let uploaded = |role: &str| -> Option<&TaskUploadSlot> {
            status.uploads.iter().find(|upload| {
                upload.uploaded && (upload.role.as_deref() == Some(role) || upload.slot == role)
            })
        };
        let first_file = || status.uploads.iter().find(|upload| upload.uploaded);
        let to_ref = |upload: &TaskUploadSlot| self.file_ref_from_upload(upload);
        let input = match status.task_kind {
            local_core::TaskKind::ObjectDetect => InferenceInput::ObjectDetect {
                image: uploaded("image")
                    .or_else(first_file)
                    .ok_or_else(|| {
                        InfraError::BadRequest("object.detect requires an image upload".to_string())
                    })
                    .and_then(to_ref)?,
            },
            local_core::TaskKind::AsrTranscribe => InferenceInput::AsrTranscribe {
                audio: uploaded("audio")
                    .or_else(first_file)
                    .ok_or_else(|| {
                        InfraError::BadRequest(
                            "asr.transcribe requires an audio upload".to_string(),
                        )
                    })
                    .and_then(to_ref)?,
            },
            local_core::TaskKind::TtsSynthesize => {
                let text = status
                    .params
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        InfraError::BadRequest("tts.synthesize params.text is required".to_string())
                    })?
                    .to_string();
                let reference_audio = uploaded("reference_audio")
                    .or_else(|| uploaded("audio"))
                    .map(to_ref)
                    .transpose()?;
                InferenceInput::TtsSynthesize {
                    text,
                    reference_audio,
                }
            }
        };
        let mut task = InferenceTask::new(status.task_kind, status.model_id.clone(), input);
        task.params = status.params.clone();
        Ok(task)
    }

    fn file_ref_from_upload(&self, upload: &TaskUploadSlot) -> Result<FileRef> {
        if let Some(uri) = &upload.asset_uri {
            let path = self.assets.local_path(uri)?;
            Ok(FileRef {
                path: Some(path),
                uri: Some(uri.clone()),
                mime: upload.mime.clone(),
                ..FileRef::default()
            })
        } else if let Some(path) = &upload.path {
            Ok(FileRef {
                path: Some(path.clone()),
                mime: upload.mime.clone(),
                ..FileRef::default()
            })
        } else {
            Err(InfraError::BadRequest(format!(
                "upload slot `{}` has no asset",
                upload.slot
            )))
        }
    }

    fn register_output_assets(
        &self,
        request_id: Uuid,
        task_id: &str,
        output: InferenceOutput,
    ) -> Result<InferenceOutput> {
        let started = std::time::Instant::now();
        let mut source_read_ms = None;
        let mut asset_write_hash_metadata_ms = None;
        let mut error_stage = "none";
        let result = match output {
            InferenceOutput::TtsAudio { mut audio } => {
                if let Some(path) = audio.path.clone() {
                    let file_name = path
                        .file_name()
                        .and_then(|value| value.to_str())
                        .map(safe_filename)
                        .unwrap_or_else(|| "audio.wav".to_string());
                    let asset_path = format!("tasks/{task_id}/outputs/{file_name}");
                    let source_read_started = std::time::Instant::now();
                    let bytes = match std::fs::read(&path) {
                        Ok(bytes) => bytes,
                        Err(err) => {
                            source_read_ms = Some(source_read_started.elapsed().as_millis() as u64);
                            error_stage = "source_read";
                            let result = Err(InfraError::io(Some(path), err));
                            log_output_asset_completion(
                                request_id,
                                started,
                                source_read_ms,
                                asset_write_hash_metadata_ms,
                                error_stage,
                                false,
                            );
                            return result;
                        }
                    };
                    source_read_ms = Some(source_read_started.elapsed().as_millis() as u64);
                    let asset_write_started = std::time::Instant::now();
                    let mut record = match self.assets.put(
                        AssetKind::Artifact,
                        &asset_path,
                        audio.mime.clone().or_else(|| Some("audio/wav".to_string())),
                        Some(Utc::now() + chrono::Duration::hours(24)),
                        &bytes,
                    ) {
                        Ok(record) => record,
                        Err(err) => {
                            asset_write_hash_metadata_ms =
                                Some(asset_write_started.elapsed().as_millis() as u64);
                            error_stage = "asset_write_hash_metadata";
                            log_output_asset_completion(
                                request_id,
                                started,
                                source_read_ms,
                                asset_write_hash_metadata_ms,
                                error_stage,
                                false,
                            );
                            return Err(err);
                        }
                    };
                    asset_write_hash_metadata_ms =
                        Some(asset_write_started.elapsed().as_millis() as u64);
                    self.decorate_asset(&mut record);
                    audio.uri = Some(record.uri);
                    audio.url = record.download_url;
                    audio.sha256 = record.sha256;
                }
                Ok(InferenceOutput::TtsAudio { audio })
            }
            other => Ok(other),
        };
        log_output_asset_completion(
            request_id,
            started,
            source_read_ms,
            asset_write_hash_metadata_ms,
            error_stage,
            result.is_ok(),
        );
        result
    }
}

fn log_output_asset_completion(
    request_id: Uuid,
    started: std::time::Instant,
    source_read_ms: Option<u64>,
    asset_write_hash_metadata_ms: Option<u64>,
    error_stage: &'static str,
    success: bool,
) {
    tracing::info!(
        request_id = %request_id,
        source_read_ms,
        asset_write_hash_metadata_ms,
        error_stage,
        success,
        output_asset_total_ms = started.elapsed().as_millis() as u64,
        "controller output asset registration completed"
    );
}

enum UploadResponse {
    Status(TaskStatus),
    Result(GenericTaskResult),
}

impl IntoResponse for UploadResponse {
    fn into_response(self) -> axum::response::Response {
        match self {
            Self::Status(status) => (StatusCode::OK, Json(json!(status))).into_response(),
            Self::Result(result) => (StatusCode::OK, Json(json!(result))).into_response(),
        }
    }
}

async fn health() -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(json!({ "status": "ok", "role": "controller" })),
    )
}

async fn register_worker(
    State(state): State<ControllerState>,
    Json(reg): Json<WorkerRegistration>,
) -> impl IntoResponse {
    match state.register_worker(reg).await {
        Ok(response) => (StatusCode::OK, Json(json!(response))).into_response(),
        Err(err) => (
            StatusCode::UNAUTHORIZED,
            Json(json!({ "error": err.to_string() })),
        )
            .into_response(),
    }
}

async fn worker_heartbeat(
    State(state): State<ControllerState>,
    headers: HeaderMap,
    Json(hb): Json<WorkerHeartbeat>,
) -> impl IntoResponse {
    let token = headers
        .get(WORKER_TOKEN_HEADER)
        .and_then(|value| value.to_str().ok());
    match state.heartbeat(token, hb).await {
        Ok(status) => (StatusCode::OK, Json(json!(status))).into_response(),
        Err(err) => (
            StatusCode::UNAUTHORIZED,
            Json(json!({ "error": err.to_string() })),
        )
            .into_response(),
    }
}

async fn upload_task_file(
    State(state): State<ControllerState>,
    AxumPath((task_id, slot)): AxumPath<(String, String)>,
    Query(query): Query<BTreeMap<String, String>>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    match state.upload_file(task_id, slot, query, headers, body).await {
        Ok(response) => response.into_response(),
        Err(err) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": err.to_string() })),
        )
            .into_response(),
    }
}

async fn upload_asset(
    State(state): State<ControllerState>,
    AxumPath((kind, path)): AxumPath<(String, String)>,
    Query(query): Query<BTreeMap<String, String>>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    let kind = match kind.parse::<AssetKind>() {
        Ok(kind) => kind,
        Err(err) => {
            return (StatusCode::BAD_REQUEST, Json(json!({ "error": err }))).into_response();
        }
    };
    let relative = match normalize_asset_path(&path) {
        Ok(path) => path,
        Err(err) => return error_response(err),
    };
    if let Err(err) =
        state.verify_asset_signature(AssetUrlOperation::Upload, kind, &relative, &query)
    {
        return error_response(err);
    }
    match state
        .upload_asset_bytes(kind, relative, query, headers, body)
        .await
    {
        Ok(record) => (StatusCode::OK, Json(json!(record))).into_response(),
        Err(err) => error_response(err),
    }
}

async fn download_asset(
    State(state): State<ControllerState>,
    AxumPath((kind, path)): AxumPath<(String, String)>,
    Query(query): Query<BTreeMap<String, String>>,
) -> impl IntoResponse {
    let kind = match kind.parse::<AssetKind>() {
        Ok(kind) => kind,
        Err(err) => return (StatusCode::BAD_REQUEST, err).into_response(),
    };
    let relative = match normalize_asset_path(&path) {
        Ok(path) => path,
        Err(err) => return error_response(err),
    };
    if let Err(err) =
        state.verify_asset_signature(AssetUrlOperation::Download, kind, &relative, &query)
    {
        return error_response(err);
    }
    match state.assets.read_bytes(&asset_uri(kind, &relative)) {
        Ok((record, bytes)) => {
            let mut response = bytes.into_response();
            if let Some(content_type) = record
                .content_type
                .and_then(|value| HeaderValue::from_str(&value).ok())
            {
                response
                    .headers_mut()
                    .insert(header::CONTENT_TYPE, content_type);
            }
            response
        }
        Err(err) => error_response(err),
    }
}

async fn sign_assets(
    State(state): State<ControllerState>,
    Json(request): Json<AssetSignRequest>,
) -> impl IntoResponse {
    match state.sign_assets_batch(request) {
        Ok(response) => (StatusCode::OK, Json(json!(response))).into_response(),
        Err(err) => error_response(err),
    }
}

async fn list_assets(
    State(state): State<ControllerState>,
    Query(query): Query<AssetListHttpQuery>,
) -> impl IntoResponse {
    let kind = match query.kind.as_deref() {
        Some(value) => match value.parse::<AssetKind>() {
            Ok(kind) => Some(kind),
            Err(err) => {
                return (StatusCode::BAD_REQUEST, Json(json!({ "error": err }))).into_response()
            }
        },
        None => None,
    };
    let query = AssetListQuery {
        kind,
        prefix: query.prefix,
        contains: query.contains,
        include_expired: query.include_expired.unwrap_or(false),
    };
    match state.assets.list(&query) {
        Ok(mut response) => {
            for record in &mut response.assets {
                state.decorate_asset(record);
            }
            (StatusCode::OK, Json(json!(response))).into_response()
        }
        Err(err) => error_response(err),
    }
}

async fn delete_asset(
    State(state): State<ControllerState>,
    headers: HeaderMap,
    AxumPath((kind, path)): AxumPath<(String, String)>,
) -> impl IntoResponse {
    if let Err(err) = state.validate_admin(&headers) {
        return error_response(err);
    }
    let kind = match kind.parse::<AssetKind>() {
        Ok(kind) => kind,
        Err(err) => {
            return (StatusCode::BAD_REQUEST, Json(json!({ "error": err }))).into_response()
        }
    };
    let relative = match normalize_asset_path(&path) {
        Ok(path) => path,
        Err(err) => return error_response(err),
    };
    match state.assets.delete(&asset_uri(kind, &relative)) {
        Ok(()) => (StatusCode::OK, Json(json!({ "deleted": true }))).into_response(),
        Err(err) => error_response(err),
    }
}

#[derive(Debug, serde::Deserialize)]
struct AssetListHttpQuery {
    kind: Option<String>,
    prefix: Option<String>,
    contains: Option<String>,
    include_expired: Option<bool>,
}

fn asset_expires_at(
    kind: AssetKind,
    query: &BTreeMap<String, String>,
) -> Result<Option<chrono::DateTime<Utc>>> {
    if query.contains_key("expires") || query.contains_key("ttl_sec") {
        return Err(InfraError::BadRequest(
            "signed asset URLs must use asset_expires, not expires or ttl_sec".to_string(),
        ));
    }
    match query.get("asset_expires") {
        Some(value) if value == "never" => Ok(None),
        Some(value) => {
            let timestamp = value
                .parse::<i64>()
                .map_err(|e| InfraError::BadRequest(format!("invalid asset_expires: {e}")))?;
            chrono::DateTime::from_timestamp(timestamp, 0)
                .ok_or_else(|| {
                    InfraError::BadRequest("invalid asset_expires timestamp".to_string())
                })
                .map(Some)
        }
        None => {
            let ttl = default_asset_ttl(kind);
            Ok(Some(Utc::now() + chrono::Duration::seconds(ttl)))
        }
    }
}

fn resolve_asset_ref(
    kind: Option<AssetKind>,
    path: Option<&str>,
    uri: Option<&str>,
) -> Result<(AssetKind, String)> {
    if let Some(uri) = uri {
        let (uri_kind, uri_path) = parse_asset_uri(uri)?;
        if let Some(kind) = kind {
            if kind != uri_kind {
                return Err(InfraError::BadRequest(
                    "kind does not match assets:// URI".to_string(),
                ));
            }
        }
        if let Some(path) = path {
            let path = normalize_asset_path(path)?;
            if path != uri_path {
                return Err(InfraError::BadRequest(
                    "path does not match assets:// URI".to_string(),
                ));
            }
        }
        return Ok((uri_kind, uri_path));
    }
    let kind = kind.ok_or_else(|| InfraError::BadRequest("kind is required".to_string()))?;
    let path = path.ok_or_else(|| InfraError::BadRequest("path or uri is required".to_string()))?;
    Ok((kind, normalize_asset_path(path)?))
}

fn asset_expiry_for_signing(
    kind: AssetKind,
    item: &AssetSignItem,
) -> Result<(Option<chrono::DateTime<Utc>>, Option<String>)> {
    if item.operation != AssetUrlOperation::Upload {
        return Ok((None, None));
    }
    if let Some(expires) = &item.expires {
        if expires
            .as_str()
            .is_some_and(|value| matches!(value, "never" | "no-expiry"))
        {
            return Ok((None, Some("asset_expires=never".to_string())));
        }
        if let Some(ttl) = expires.as_i64() {
            return ttl_expiry(ttl)
                .map(|at| (Some(at), Some(format!("asset_expires={}", at.timestamp()))));
        }
        return Err(InfraError::BadRequest(
            "expires must be `never` or a TTL seconds number".to_string(),
        ));
    }
    let ttl = item.ttl_sec.unwrap_or_else(|| default_asset_ttl(kind));
    ttl_expiry(ttl).map(|at| (Some(at), Some(format!("asset_expires={}", at.timestamp()))))
}

fn default_asset_ttl(kind: AssetKind) -> i64 {
    match kind {
        AssetKind::Material => DEFAULT_MATERIAL_ASSET_TTL_SECS,
        AssetKind::Artifact => DEFAULT_ARTIFACT_ASSET_TTL_SECS,
    }
}

fn ttl_expiry(ttl: i64) -> Result<chrono::DateTime<Utc>> {
    if ttl <= 0 {
        return Err(InfraError::BadRequest(
            "ttl_sec must be positive".to_string(),
        ));
    }
    let timestamp = (Utc::now() + chrono::Duration::seconds(ttl)).timestamp();
    chrono::DateTime::from_timestamp(timestamp, 0)
        .ok_or_else(|| InfraError::BadRequest("invalid asset expiration timestamp".to_string()))
}

fn canonical_asset_expiry_query(query: &BTreeMap<String, String>) -> Result<Option<String>> {
    if query.contains_key("expires") || query.contains_key("ttl_sec") {
        return Err(InfraError::BadRequest(
            "signed asset URLs must use asset_expires, not expires or ttl_sec".to_string(),
        ));
    }
    if let Some(value) = query.get("asset_expires") {
        if value == "never" {
            return Ok(Some("asset_expires=never".to_string()));
        }
        let timestamp = value
            .parse::<i64>()
            .map_err(|e| InfraError::BadRequest(format!("invalid asset_expires: {e}")))?;
        if timestamp <= 0 {
            return Err(InfraError::BadRequest(
                "asset_expires must be a positive unix timestamp".to_string(),
            ));
        }
        return Ok(Some(format!("asset_expires={timestamp}")));
    }
    Ok(None)
}

fn is_reserved_asset_upload_path(path: &str) -> bool {
    matches!(
        path.split('/').next().unwrap_or_default(),
        "tasks" | "system" | ".metadata"
    )
}

impl ControllerState {
    fn validate_admin(&self, headers: &HeaderMap) -> Result<()> {
        let Some(expected) = &self.admin_token else {
            return Err(InfraError::BadRequest(
                "admin token is not configured".to_string(),
            ));
        };
        let provided = headers
            .get("x-local-admin-token")
            .and_then(|value| value.to_str().ok())
            .or_else(|| {
                headers
                    .get(header::AUTHORIZATION)
                    .and_then(|value| value.to_str().ok())
                    .and_then(|value| value.strip_prefix("Bearer "))
            });
        if provided == Some(expected.as_str()) {
            Ok(())
        } else {
            Err(InfraError::BadRequest(
                "missing or invalid admin token".to_string(),
            ))
        }
    }
}

fn error_response(err: InfraError) -> axum::response::Response {
    let status = match &err {
        InfraError::NotFound(_) => StatusCode::NOT_FOUND,
        InfraError::BadRequest(_) => StatusCode::BAD_REQUEST,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (status, Json(json!({ "error": err.to_string() }))).into_response()
}

fn missing_required_uploads(status: &TaskStatus) -> Vec<String> {
    status
        .uploads
        .iter()
        .filter(|upload| upload.required && !upload.uploaded)
        .map(|upload| upload.slot.clone())
        .collect()
}

fn output_files(output: &InferenceOutput) -> Vec<FileRef> {
    match output {
        InferenceOutput::TtsAudio { audio } => vec![audio.clone()],
        _ => Vec::new(),
    }
}

fn result_from_status(status: &TaskStatus) -> GenericTaskResult {
    GenericTaskResult {
        task_id: status.task_id.clone(),
        state: status.state,
        output: status.output.clone(),
        files: status.files.clone(),
        error: status.error.clone(),
    }
}

fn safe_slot(input: &str, index: usize) -> String {
    let candidate = sanitize_token(input);
    if candidate.is_empty() {
        format!("file{index}")
    } else {
        candidate
    }
}

fn safe_filename(input: &str) -> String {
    let name = Path::new(input)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("upload.bin");
    let sanitized = sanitize_token(name);
    if sanitized.is_empty() {
        "upload.bin".to_string()
    } else {
        sanitized
    }
}

fn sanitize_token(input: &str) -> String {
    input
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    hex(&digest)
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

fn percent_encode(value: &str) -> String {
    let mut out = String::new();
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            out.push(byte as char);
        } else {
            use std::fmt::Write as _;
            let _ = write!(&mut out, "%{byte:02X}");
        }
    }
    out
}

fn percent_encode_path(value: &str) -> String {
    value
        .split('/')
        .map(percent_encode)
        .collect::<Vec<_>>()
        .join("/")
}

#[async_trait]
impl AdminApi for ControllerState {
    async fn list_models(&self) -> Result<Vec<ModelSpec>> {
        Ok(self.registry.list().await)
    }
    async fn get_model(&self, id: &str) -> Result<ModelSpec> {
        self.registry
            .get(id)
            .await
            .ok_or_else(|| InfraError::NotFound(format!("model `{id}`")))
    }
    async fn enable_model(&self, id: &str) -> Result<ModelSpec> {
        let spec = if let Some(store) = &self.store {
            store.set_enabled(id, true)?
        } else {
            self.registry.set_enabled(id, true).await?
        };
        self.registry.upsert(spec.clone()).await;
        Ok(spec)
    }
    async fn disable_model(&self, id: &str) -> Result<ModelSpec> {
        let spec = if let Some(store) = &self.store {
            store.set_enabled(id, false)?
        } else {
            self.registry.set_enabled(id, false).await?
        };
        self.registry.upsert(spec.clone()).await;
        Ok(spec)
    }
    async fn upsert_model(&self, spec: ModelSpec) -> Result<ModelSpec> {
        let spec = if let Some(store) = &self.store {
            store.upsert_model(spec)?
        } else {
            spec
        };
        self.registry.upsert(spec.clone()).await;
        Ok(spec)
    }
    async fn download_model(&self, id: &str) -> Result<Vec<DownloadStatus>> {
        let spec = self.get_model(id).await?;
        let store = self
            .store
            .as_ref()
            .ok_or_else(|| InfraError::Config("model store is not configured".to_string()))?;
        store.download_model(&spec).await
    }
    async fn list_nodes(&self) -> Result<Vec<NodeStatus>> {
        Ok(self.nodes.read().await.values().cloned().collect())
    }
    async fn get_cluster_status(&self) -> Result<Value> {
        let models = self.registry.list().await;
        let nodes = self.nodes.read().await;
        Ok(
            json!({ "models_total": models.len(), "models_enabled": models.iter().filter(|m| m.enabled).count(), "nodes_total": nodes.len(), "nodes": nodes.values().collect::<Vec<_>>() }),
        )
    }
    async fn list_assets(&self, query: AssetListQuery) -> Result<local_core::AssetListResponse> {
        let mut response = self.assets.list(&query)?;
        for record in &mut response.assets {
            self.decorate_asset(record);
        }
        Ok(response)
    }
}

#[async_trait]
impl InferenceApi for ControllerState {
    async fn dispatch(&self, task: InferenceTask) -> Result<InferenceOutput> {
        let started = std::time::Instant::now();
        let request_id = task.id;
        let output_asset_task_id = task.id.to_string();
        let result = self
            .forward_to_worker(task.clone())
            .await
            .and_then(|output| {
                self.register_output_assets(request_id, &output_asset_task_id, output)
            });
        if let Err(err) = &result {
            if let Some(store) = &self.store {
                store.record_job_state(&task, JobState::Failed, None, Some(&err.to_string()))?;
            }
        }
        tracing::info!(
            request_id = %request_id,
            controller_dispatch_total_ms = started.elapsed().as_millis() as u64,
            success = result.is_ok(),
            "controller inference dispatch finished"
        );
        result
    }

    async fn create_task(&self, request: CreateTaskRequest) -> Result<TaskStatus> {
        self.create_generic_task(request).await
    }

    async fn start_task(&self, request: StartTaskRequest) -> Result<GenericTaskResult> {
        self.start_generic_task(request).await
    }

    async fn get_task(&self, task_id: String) -> Result<TaskStatus> {
        self.load_task(&task_id).await
    }

    async fn wait_task(&self, request: WaitTaskRequest) -> Result<TaskStatus> {
        self.wait_generic_task(request).await
    }

    async fn sign_assets(&self, request: AssetSignRequest) -> Result<AssetSignResponse> {
        self.sign_assets_batch(request)
    }
}

#[async_trait]
impl OpenAiApi for ControllerState {
    async fn list_models(&self) -> Result<Vec<ModelSpec>> {
        Ok(self.registry.list().await)
    }
    async fn dispatch(&self, task: InferenceTask) -> Result<InferenceOutput> {
        let started = std::time::Instant::now();
        let request_id = task.id;
        let output_asset_task_id = task.id.to_string();
        let result = self
            .forward_to_worker(task.clone())
            .await
            .and_then(|output| {
                self.register_output_assets(request_id, &output_asset_task_id, output)
            });
        if let Err(err) = &result {
            if let Some(store) = &self.store {
                store.record_job_state(&task, JobState::Failed, None, Some(&err.to_string()))?;
            }
        }
        tracing::info!(
            request_id = %request_id,
            controller_dispatch_total_ms = started.elapsed().as_millis() as u64,
            success = result.is_ok(),
            "controller OpenAI dispatch finished"
        );
        result
    }
}

#[cfg(test)]
mod tests;
