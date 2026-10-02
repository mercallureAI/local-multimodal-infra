use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
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
    SenseVoiceAsr,
    IndexTts,
    IndexTts2,
    /// Qwen3-TTS-12Hz Base (voice cloning), streaming.
    Qwen3Tts,
    E5Embedding,
    MmarcoReranker,
    Qwen3Chat,
    UnlimitedOcr,
    /// PP-OCRv5 mobile text lines (detection + recognition).
    Ppocrv5Mobile,
    /// Depth Anything V2 metric depth (metres per pixel).
    DepthAnythingV2,
    /// Pseudo-realtime voice (VAD, ASR, chat and TTS models of the worker),
    /// served over `/v1/realtime` only.
    VoiceCascade,
}

impl AdapterKind {
    pub const ALL: [AdapterKind; 12] = [
        AdapterKind::Yolo,
        AdapterKind::SenseVoiceAsr,
        AdapterKind::IndexTts,
        AdapterKind::IndexTts2,
        AdapterKind::Qwen3Tts,
        AdapterKind::E5Embedding,
        AdapterKind::MmarcoReranker,
        AdapterKind::Qwen3Chat,
        AdapterKind::UnlimitedOcr,
        AdapterKind::Ppocrv5Mobile,
        AdapterKind::DepthAnythingV2,
        AdapterKind::VoiceCascade,
    ];

    /// The category the adapter's models are filed and built under.
    pub fn category(self) -> ModelCategory {
        match self {
            AdapterKind::Yolo | AdapterKind::DepthAnythingV2 => ModelCategory::Detect,
            AdapterKind::SenseVoiceAsr => ModelCategory::Asr,
            AdapterKind::IndexTts | AdapterKind::IndexTts2 | AdapterKind::Qwen3Tts => {
                ModelCategory::Tts
            }
            AdapterKind::E5Embedding => ModelCategory::Embedding,
            AdapterKind::MmarcoReranker => ModelCategory::Rerank,
            AdapterKind::Qwen3Chat => ModelCategory::Chat,
            AdapterKind::UnlimitedOcr | AdapterKind::Ppocrv5Mobile => ModelCategory::Ocr,
            AdapterKind::VoiceCascade => ModelCategory::Realtime,
        }
    }
}

/// What a model is for. Model specs live in `<providers dir>/<category>/`,
/// and each category is a build feature of the worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelCategory {
    Asr,
    Tts,
    Chat,
    Embedding,
    Rerank,
    Detect,
    Ocr,
    /// Realtime voice pipelines (their VAD included).
    Realtime,
}

impl ModelCategory {
    pub const ALL: [ModelCategory; 8] = [
        ModelCategory::Asr,
        ModelCategory::Tts,
        ModelCategory::Chat,
        ModelCategory::Embedding,
        ModelCategory::Rerank,
        ModelCategory::Detect,
        ModelCategory::Ocr,
        ModelCategory::Realtime,
    ];

    /// The directory (and feature) name.
    pub fn as_str(self) -> &'static str {
        match self {
            ModelCategory::Asr => "asr",
            ModelCategory::Tts => "tts",
            ModelCategory::Chat => "chat",
            ModelCategory::Embedding => "embedding",
            ModelCategory::Rerank => "rerank",
            ModelCategory::Detect => "detect",
            ModelCategory::Ocr => "ocr",
            ModelCategory::Realtime => "realtime",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|category| category.as_str() == name)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaskKind {
    #[serde(rename = "asr.transcribe")]
    AsrTranscribe,
    #[serde(rename = "object.detect")]
    ObjectDetect,
    #[serde(rename = "tts.synthesize")]
    TtsSynthesize,
    #[serde(rename = "text.embed")]
    TextEmbed,
    #[serde(rename = "text.rerank")]
    TextRerank,
    #[serde(rename = "chat.complete")]
    ChatComplete,
    #[serde(rename = "ocr.recognize")]
    OcrRecognize,
    /// Text lines with their boxes (scene text, UI labels).
    #[serde(rename = "ocr.lines")]
    OcrLines,
    /// Metric depth of an image, pooled to a grid.
    #[serde(rename = "depth.estimate")]
    DepthEstimate,
    #[serde(rename = "voice.realtime")]
    VoiceRealtime,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingInputType {
    Query,
    #[default]
    Passage,
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
    /// The directory a Hugging Face artifact's files go to under the model
    /// directory `root`: `root` itself, or the subdirectory `path` names
    /// (relative as configured, or already materialized under `root`, where a
    /// `single_file` path ends in the file name). Not validated here.
    pub fn hugging_face_dir(&self, root: &Path, single_file: bool) -> PathBuf {
        let subdir = match self.path.strip_prefix(root) {
            Ok(relative) if single_file => relative.parent().unwrap_or(Path::new("")),
            Ok(relative) => relative,
            Err(_) if self.path.is_relative() => self.path.as_path(),
            Err(_) => Path::new(""),
        };
        if subdir.as_os_str().is_empty() {
            root.to_path_buf()
        } else {
            root.join(subdir)
        }
    }

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
    #[serde(alias = "models_conf_dir")]
    pub providers_conf_dir: PathBuf,
}

impl StorageLayout {
    pub fn new(
        workdir: impl Into<PathBuf>,
        data_dir: Option<PathBuf>,
        database_path: Option<PathBuf>,
        model_dir: Option<PathBuf>,
        providers_conf_dir: PathBuf,
    ) -> Self {
        let workdir = workdir.into();
        let data_dir = data_dir.unwrap_or_else(|| workdir.join("data"));
        let database_path = database_path.unwrap_or_else(|| data_dir.join("local.db"));
        let model_dir = model_dir.unwrap_or_else(|| workdir.join("models"));
        Self {
            workdir,
            data_dir,
            database_path,
            model_dir,
            providers_conf_dir,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDownloadStatus {
    pub model_id: String,
    pub downloaded: bool,
    pub state: DownloadState,
    #[serde(default)]
    pub artifacts: Vec<DownloadStatus>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    #[serde(flatten)]
    pub spec: ModelSpec,
    pub downloaded: bool,
    pub download_state: DownloadState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadModelResponse {
    pub accepted: bool,
    pub deduplicated: bool,
    #[serde(flatten)]
    pub status: ModelDownloadStatus,
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

impl ModelSpec {
    pub fn category(&self) -> ModelCategory {
        self.adapter.category()
    }

    /// Whether a task naming no model may get this one: not when its
    /// metadata says `auto_select: false` (a local export most installs lack,
    /// used where it is named).
    pub fn auto_selectable(&self) -> bool {
        self.metadata
            .get("auto_select")
            .and_then(serde_json::Value::as_bool)
            != Some(false)
    }
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
    TextEmbed {
        texts: Vec<String>,
        #[serde(default)]
        input_type: EmbeddingInputType,
    },
    TextRerank {
        query: String,
        documents: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        top_n: Option<usize>,
    },
    ChatComplete {
        messages: Vec<ChatMessage>,
        /// OpenAI-style tool definitions, rendered by the model's chat template.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tools: Vec<serde_json::Value>,
        #[serde(default)]
        options: ChatOptions,
    },
    OcrRecognize {
        image: FileRef,
    },
    OcrLines {
        image: FileRef,
    },
    DepthEstimate {
        image: FileRef,
        /// The grid the depth is pooled to (default: the adapter's).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        grid: Option<DepthGrid>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InferenceOutput {
    AsrTranscription {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        timestamped_text: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        segments: Vec<AsrSegment>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        speakers: Vec<AsrSpeaker>,
        /// The whole utterance's speaker (voiceprint) embedding, L2-normalised,
        /// when asked for (`speaker_embedding`): comparable across requests.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        speaker_embedding: Option<Vec<f32>>,
    },
    ObjectDetections {
        objects: Vec<DetectedObject>,
    },
    TtsAudio {
        audio: FileRef,
    },
    TextEmbeddings {
        embeddings: Vec<Vec<f32>>,
        prompt_tokens: usize,
    },
    TextRerank {
        results: Vec<RerankResult>,
        total_tokens: usize,
    },
    ChatCompletion {
        content: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ChatToolCall>,
        finish_reason: ChatFinishReason,
        usage: ChatUsage,
        #[serde(default)]
        timings: ChatTimings,
    },
    OcrText {
        text: String,
    },
    OcrLines {
        lines: Vec<OcrLine>,
    },
    /// `depth[row * cols + col]`: the mean depth (metres) of that cell of the
    /// image, rows top to bottom.
    DepthMap {
        cols: u32,
        rows: u32,
        /// The model's largest depth (metres): farther reads as this.
        max_depth: f32,
        depth: Vec<f32>,
    },
    Accepted {
        job_id: String,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ChatMessage {
    /// `system`, `user`, `assistant` or `tool`.
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ChatToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ChatToolCall {
    pub id: String,
    pub name: String,
    /// The arguments as a JSON text, as in the OpenAI API.
    pub arguments: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ChatOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    /// 0 selects greedy decoding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    /// Token id -> logit offset, applied at every step (OpenAI `logit_bias`).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub logit_bias: BTreeMap<u32, f32>,
    /// Logit offset for starting a tool call at all.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_bias: Option<f32>,
    /// Tool name -> logit offset on the name's first token when the model
    /// picks which tool to call.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub tool_bias: BTreeMap<String, f32>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatFinishReason {
    #[default]
    Stop,
    Length,
    ToolCalls,
    /// The caller stopped reading (a streaming client went away).
    Cancelled,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct ChatUsage {
    pub prompt_tokens: usize,
    /// Prompt tokens whose KV cache was reused from the previous request.
    #[serde(default)]
    pub cached_prompt_tokens: usize,
    pub completion_tokens: usize,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct ChatTimings {
    pub prefill_ms: u64,
    pub first_token_ms: u64,
    pub decode_ms: u64,
}

/// Text handed to a TTS model while it is still being written (a chat
/// model's reply, say): pieces in order, then `End`. A sender dropped before
/// `End` abandons the text: nothing more of it is spoken.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TextPiece {
    Text(String),
    End,
}

/// Incremental results of a streaming inference, in order; the last event is
/// `output` (the complete result) or `error`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InferenceEvent {
    ChatDelta {
        content: String,
    },
    ChatToolCall {
        index: usize,
        call: ChatToolCall,
    },
    /// Synthesized speech so far (mono), from TTS models that stream.
    AudioChunk {
        sample_rate: u32,
        samples: Vec<f32>,
    },
    Output {
        output: InferenceOutput,
    },
    Error {
        message: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AsrSegment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emotion: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub events: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tokens: Vec<AsrTokenTimestamp>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AsrTokenTimestamp {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AsrSpeaker {
    pub id: String,
    pub speech_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResult {
    pub index: usize,
    pub relevance_score: f32,
    pub document: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedObject {
    pub label: String,
    pub confidence: f32,
    pub bbox: BoundingBox,
}

/// Columns and rows of a depth grid over the whole image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DepthGrid {
    pub cols: u32,
    pub rows: u32,
}

/// A line of text found in an image, top to bottom then left to right.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrLine {
    pub text: String,
    /// Mean probability of the recognised characters.
    pub confidence: f32,
    /// In the image's pixels.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_auto_select_false_keeps_a_model_from_auto_selection() {
        let spec = |metadata: serde_json::Value| -> ModelSpec {
            serde_json::from_value(serde_json::json!({
                "id": "m", "name": "m", "adapter": "sense_voice_asr", "backend": "ort",
                "metadata": metadata,
            }))
            .expect("spec")
        };
        assert!(spec(serde_json::json!({})).auto_selectable());
        assert!(spec(serde_json::json!({"auto_select": true})).auto_selectable());
        assert!(!spec(serde_json::json!({"auto_select": false})).auto_selectable());
    }

    #[test]
    fn legacy_asr_text_output_deserializes_without_rich_fields() {
        let output: InferenceOutput =
            serde_json::from_str(r#"{"type":"asr_transcription","text":"hello"}"#)
                .expect("legacy output");
        let InferenceOutput::AsrTranscription {
            text,
            timestamped_text,
            segments,
            speakers,
            speaker_embedding,
        } = output
        else {
            panic!("wrong output")
        };
        assert_eq!(text, "hello");
        assert!(timestamped_text.is_none());
        assert!(segments.is_empty());
        assert!(speakers.is_empty());
        assert!(speaker_embedding.is_none());
    }
}
