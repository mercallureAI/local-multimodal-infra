use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Ort,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterKind {
    Yolo,
    QwenAsr,
    IndexTts,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaskKind {
    #[serde(rename = "asr.transcribe")]
    AsrTranscribe,
    #[serde(rename = "object.detect")]
    ObjectDetect,
    #[serde(rename = "tts.synthesize")]
    TtsSynthesize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelState {
    Unavailable,
    Downloaded,
    Loading,
    Warm,
    Busy,
    Idle,
    Evicting,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Queued,
    Scheduled,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FileRef {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

impl FileRef {
    pub fn local(path: impl Into<PathBuf>) -> Self {
        Self {
            path: Some(path.into()),
            ..Self::default()
        }
    }

    pub fn asset(uri: impl Into<String>) -> Self {
        Self {
            uri: Some(uri.into()),
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssetKind {
    Material,
    Artifact,
}

impl AssetKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Material => "material",
            Self::Artifact => "artifact",
        }
    }
}

impl std::str::FromStr for AssetKind {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "material" | "materials" => Ok(Self::Material),
            "artifact" | "artifacts" => Ok(Self::Artifact),
            other => Err(format!("unknown asset kind `{other}`")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetRecord {
    pub uri: String,
    pub kind: AssetKind,
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,
    pub size: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    pub created_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub download_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upload_url: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AssetListQuery {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<AssetKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contains: Option<String>,
    #[serde(default)]
    pub include_expired: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetListResponse {
    pub assets: Vec<AssetRecord>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssetUrlOperation {
    Upload,
    Download,
}

impl AssetUrlOperation {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Upload => "upload",
            Self::Download => "download",
        }
    }

    pub fn method(self) -> &'static str {
        match self {
            Self::Upload => "POST",
            Self::Download => "GET",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetSignRequest {
    #[serde(default, alias = "requests")]
    pub items: Vec<AssetSignItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetSignItem {
    #[serde(alias = "action")]
    pub operation: AssetUrlOperation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<AssetKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_sec: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url_ttl_sec: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetSignResponse {
    pub items: Vec<AssetSignedUrl>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetSignedUrl {
    pub operation: AssetUrlOperation,
    pub uri: String,
    pub signed_url: String,
    pub method: String,
    pub expires_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asset_expires_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    Local,
    HuggingFace,
    Url,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HuggingFaceArtifact {
    pub repo_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    #[serde(default)]
    pub files: Vec<String>,
    #[serde(default)]
    pub allow_patterns: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ArtifactSource {
    Local { path: PathBuf },
    Url { url: String },
    HuggingFace(HuggingFaceArtifact),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArtifact {
    #[serde(rename = "type")]
    pub kind: ArtifactKind,
    /// Materialized local path used by adapters. For multi-file artifacts this is
    /// the model root directory; for single-file artifacts it may be the file.
    #[serde(default)]
    pub path: PathBuf,
    /// Optional external local source used for importing/copying into the
    /// stable model store layout. Adapters must use `path`, not `source_path`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    #[serde(default)]
    pub files: Vec<String>,
    #[serde(default)]
    pub allow_patterns: Vec<String>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

impl ModelArtifact {
    pub fn source(&self) -> ArtifactSource {
        match self.kind {
            ArtifactKind::Local => ArtifactSource::Local {
                path: self
                    .source_path
                    .clone()
                    .unwrap_or_else(|| self.path.clone()),
            },
            ArtifactKind::Url => ArtifactSource::Url {
                url: self.url.clone().unwrap_or_default(),
            },
            ArtifactKind::HuggingFace => ArtifactSource::HuggingFace(HuggingFaceArtifact {
                repo_id: self.repo_id.clone().unwrap_or_default(),
                revision: self.revision.clone(),
                files: self.files.clone(),
                allow_patterns: self.allow_patterns.clone(),
            }),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageLayout {
    pub workdir: PathBuf,
    pub data_dir: PathBuf,
    pub database_path: PathBuf,
    pub model_dir: PathBuf,
    pub models_conf_dir: PathBuf,
}

impl StorageLayout {
    pub fn new(
        workdir: impl Into<PathBuf>,
        data_dir: Option<PathBuf>,
        database_path: Option<PathBuf>,
        model_dir: Option<PathBuf>,
        models_conf_dir: PathBuf,
    ) -> Self {
        let workdir = workdir.into();
        let data_dir = data_dir.unwrap_or_else(|| workdir.join("data"));
        let database_path = database_path.unwrap_or_else(|| data_dir.join("lcoal.db"));
        let model_dir = model_dir.unwrap_or_else(|| workdir.join("models"));
        Self {
            workdir,
            data_dir,
            database_path,
            model_dir,
            models_conf_dir,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DownloadState {
    NotStarted,
    Downloading,
    Downloaded,
    Failed,
    Skipped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadStatus {
    pub model_id: String,
    pub artifact: String,
    pub state: DownloadState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuntimePolicy {
    #[serde(default)]
    pub provider_order: Vec<String>,
    #[serde(default = "default_concurrency")]
    pub max_concurrency: usize,
    #[serde(default = "default_idle_ttl")]
    pub idle_ttl_sec: u64,
}

fn default_concurrency() -> usize {
    1
}
fn default_idle_ttl() -> u64 {
    300
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceRequirement {
    #[serde(default)]
    pub min_ram_mb: u64,
    #[serde(default)]
    pub min_vram_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadPolicy {
    #[serde(default = "default_true")]
    pub lazy: bool,
    #[serde(default = "default_true")]
    pub evictable: bool,
    #[serde(default)]
    pub pin: bool,
}

impl Default for LoadPolicy {
    fn default() -> Self {
        Self {
            lazy: true,
            evictable: true,
            pin: false,
        }
    }
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    pub id: String,
    pub name: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub task_kinds: Vec<TaskKind>,
    pub adapter: AdapterKind,
    pub backend: BackendKind,
    #[serde(default)]
    pub artifacts: Vec<ModelArtifact>,
    #[serde(default)]
    pub runtime: RuntimePolicy,
    #[serde(default)]
    pub resources: ResourceRequirement,
    #[serde(default)]
    pub load_policy: LoadPolicy,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceTask {
    pub id: Uuid,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    pub kind: TaskKind,
    pub input: InferenceInput,
    #[serde(default)]
    pub params: BTreeMap<String, serde_json::Value>,
}

impl InferenceTask {
    pub fn new(kind: TaskKind, model_id: Option<String>, input: InferenceInput) -> Self {
        Self {
            id: Uuid::new_v4(),
            model_id,
            kind,
            input,
            params: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InferenceInput {
    AsrTranscribe {
        audio: FileRef,
    },
    ObjectDetect {
        image: FileRef,
    },
    TtsSynthesize {
        text: String,
        reference_audio: Option<FileRef>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InferenceOutput {
    AsrTranscription { text: String },
    ObjectDetections { objects: Vec<DetectedObject> },
    TtsAudio { audio: FileRef },
    Accepted { job_id: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedObject {
    pub label: String,
    pub confidence: f32,
    pub bbox: BoundingBox,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeviceSpec {
    pub has_cuda: bool,
    #[serde(default)]
    pub cuda_devices: Vec<CudaDeviceSpec>,
    pub has_dml: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaDeviceSpec {
    pub index: u32,
    pub name: String,
    pub total_vram_mb: u64,
    pub free_vram_mb: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSnapshot {
    pub cpu_cores: usize,
    pub total_ram_mb: u64,
    pub used_ram_mb: u64,
    pub devices: DeviceSpec,
    pub captured_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerRegistration {
    pub node_id: String,
    pub base_url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub registration_token: Option<String>,
    #[serde(default)]
    pub supported_backends: Vec<BackendKind>,
    #[serde(default)]
    pub supported_adapters: Vec<AdapterKind>,
    pub resources: ResourceSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerRegistrationResponse {
    pub status: NodeStatus,
    pub session_token: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerHeartbeat {
    pub node_id: String,
    pub resources: ResourceSnapshot,
    #[serde(default)]
    pub loaded_models: Vec<String>,
    #[serde(default)]
    pub queued_jobs: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStatus {
    pub registration: WorkerRegistration,
    pub last_heartbeat_at: DateTime<Utc>,
    #[serde(default)]
    pub loaded_models: Vec<String>,
    #[serde(default)]
    pub queued_jobs: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateTaskRequest {
    pub task_kind: TaskKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(default)]
    pub files: Vec<TaskFileRequirement>,
    #[serde(default)]
    pub params: BTreeMap<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wait_timeout_sec: Option<u64>,
}

impl CreateTaskRequest {
    pub fn effective_model_id(&self) -> Option<String> {
        self.model_id.clone().or_else(|| self.model.clone())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskFileRequirement {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asset_uri: Option<String>,
    #[serde(default)]
    pub required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskUploadSlot {
    pub slot: String,
    pub file_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default)]
    pub required: bool,
    pub upload_url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asset_uri: Option<String>,
    pub expires_at: DateTime<Utc>,
    #[serde(default)]
    pub uploaded: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uploaded_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenericTaskState {
    WaitingForUploads,
    Ready,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStatus {
    pub task_id: String,
    pub task_kind: TaskKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    pub state: GenericTaskState,
    #[serde(default)]
    pub uploads: Vec<TaskUploadSlot>,
    #[serde(default)]
    pub params: BTreeMap<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<InferenceOutput>,
    #[serde(default)]
    pub files: Vec<FileRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenericTaskResult {
    pub task_id: String,
    pub state: GenericTaskState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<InferenceOutput>,
    #[serde(default)]
    pub files: Vec<FileRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartTaskRequest {
    pub task_id: String,
    #[serde(default)]
    pub wait: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_sec: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaitTaskRequest {
    pub task_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_sec: Option<u64>,
}
