use local_adapter_index_tts::IndexTtsAdapter;
use local_adapter_qwen_asr::QwenAsrAdapter;
use local_adapter_yolo::YoloAdapter;
use local_backend_ort::probe_runtime_execution_provider_availability;
use local_core::{
    AdapterKind, InferenceInput, InferenceOutput, InferenceTask, ModelSpec, ModelState, TaskKind,
};
use local_error::{InfraError, Result};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::Mutex;

pub const DEFAULT_IDLE_UNLOAD_INTERVAL: Duration = Duration::from_secs(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RuntimeProviderAvailability {
    cuda: bool,
    directml: bool,
    tensorrt: bool,
}

impl RuntimeProviderAvailability {
    fn current() -> Self {
        let availability = probe_runtime_execution_provider_availability();
        Self {
            cuda: availability.cuda,
            directml: availability.dml,
            tensorrt: availability.trt,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeManagerConfig {
    pub cache_idle_ttl: Duration,
    pub model_idle_ttl: Duration,
    pub min_residency: Duration,
    pub memory_pressure_threshold: f32,
}

impl Default for RuntimeManagerConfig {
    fn default() -> Self {
        Self {
            cache_idle_ttl: Duration::from_secs(30),
            model_idle_ttl: Duration::from_secs(600),
            min_residency: Duration::from_secs(0),
            memory_pressure_threshold: 0.85,
        }
    }
}

#[derive(Debug)]
pub struct IdleMaintenanceLoopHandle {
    handle: tokio::task::JoinHandle<()>,
}

impl Drop for IdleMaintenanceLoopHandle {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

#[derive(Debug)]
pub struct RuntimeManager {
    specs: HashMap<String, ModelSpec>,
    loaded: Mutex<HashMap<String, LoadedEntry>>,
    config: RuntimeManagerConfig,
}

impl RuntimeManager {
    pub fn new(specs: Vec<ModelSpec>, config: RuntimeManagerConfig) -> Self {
        Self {
            specs: specs
                .into_iter()
                .map(|spec| (spec.id.clone(), spec))
                .collect(),
            loaded: Mutex::new(HashMap::new()),
            config,
        }
    }

    pub async fn infer(&self, task: InferenceTask) -> Result<InferenceOutput> {
        self.maintain_idle().await?;
        let spec = self.resolve_spec(&task)?;
        let model_id = spec.id.clone();
        let mut loaded = self.loaded.lock().await;
        if !loaded.contains_key(&model_id) {
            let entry = LoadedEntry::load(&spec)?;
            loaded.insert(model_id.clone(), entry);
        }
        let entry = loaded.get_mut(&model_id).ok_or_else(|| {
            InfraError::Runtime(format!("model `{model_id}` failed to enter loaded cache"))
        })?;
        entry.state = ModelState::Busy;
        let result = entry.model.infer(&task);
        entry.last_used = Instant::now();
        entry.last_cache_released_at = None;
        entry.state = ModelState::Idle;
        result
    }

    pub fn spawn_idle_maintenance_loop(self: Arc<Self>) -> IdleMaintenanceLoopHandle {
        self.spawn_idle_maintenance_loop_with_interval(DEFAULT_IDLE_UNLOAD_INTERVAL)
    }

    pub fn spawn_idle_maintenance_loop_with_interval(
        self: Arc<Self>,
        interval: Duration,
    ) -> IdleMaintenanceLoopHandle {
        let interval = if interval.is_zero() {
            Duration::from_millis(1)
        } else {
            interval
        };
        let handle = tokio::spawn(async move {
            loop {
                tokio::time::sleep(interval).await;
                if let Err(err) = self.maintain_idle().await {
                    tracing::warn!(error = %err, "idle runtime maintenance loop failed");
                }
            }
        });
        IdleMaintenanceLoopHandle { handle }
    }

    pub async fn loaded_models(&self) -> Vec<String> {
        self.loaded.lock().await.keys().cloned().collect()
    }

    pub async fn states(&self) -> HashMap<String, ModelState> {
        self.loaded
            .lock()
            .await
            .iter()
            .map(|(k, v)| (k.clone(), v.state))
            .collect()
    }

    pub async fn unload_idle(&self) -> Result<()> {
        self.maintain_idle().await
    }

    pub async fn maintain_idle(&self) -> Result<()> {
        let mut loaded = self.loaded.lock().await;
        let now = Instant::now();
        loaded.retain(|id, entry| {
            let can_unload = entry.state == ModelState::Idle
                && now.duration_since(entry.last_used) >= self.config.model_idle_ttl
                && now.duration_since(entry.loaded_at) >= self.config.min_residency;
            if can_unload {
                tracing::info!(model_id = id, "unloading idle model");
                return false;
            }

            let can_release_cache = entry.state == ModelState::Idle
                && entry.last_cache_released_at.is_none()
                && now.duration_since(entry.last_used) >= self.config.cache_idle_ttl;
            if can_release_cache {
                entry.model.release_idle_cache(id);
                entry.last_cache_released_at = Some(now);
            }
            true
        });
        self.note_memory_pressure_policy();
        Ok(())
    }

    fn note_memory_pressure_policy(&self) {
        tracing::trace!(
            threshold = self.config.memory_pressure_threshold,
            "memory-pressure eviction policy is configured but passive in this MVP; idle cache release and model TTL unloading are the active maintenance mechanisms"
        );
    }

    fn resolve_spec(&self, task: &InferenceTask) -> Result<ModelSpec> {
        if let Some(model_id) = &task.model_id {
            let spec = self.specs.get(model_id).cloned().ok_or_else(|| {
                InfraError::ModelNotConfigured {
                    model_id: model_id.clone(),
                    reason: "model id is not present in worker registry snapshot".to_string(),
                }
            })?;
            if !spec.enabled {
                return Err(InfraError::ModelNotConfigured {
                    model_id: model_id.clone(),
                    reason: "model is disabled".to_string(),
                });
            }
            return Ok(spec);
        }
        self.specs
            .values()
            .find(|spec| spec.enabled && spec.task_kinds.contains(&task.kind))
            .cloned()
            .ok_or_else(|| InfraError::ModelNotConfigured {
                model_id: "<auto>".to_string(),
                reason: format!("no enabled model supports task {:?}", task.kind),
            })
    }
}

#[derive(Debug)]
struct LoadedEntry {
    model: LoadedModel,
    loaded_at: Instant,
    last_used: Instant,
    last_cache_released_at: Option<Instant>,
    state: ModelState,
}

impl LoadedEntry {
    fn load(spec: &ModelSpec) -> Result<Self> {
        let spec = effective_load_spec(spec);
        tracing::info!(
            model_id = spec.id,
            adapter = ?spec.adapter,
            provider_order = ?spec.runtime.provider_order,
            "lazy loading model"
        );
        let model = match spec.adapter {
            AdapterKind::Yolo => LoadedModel::Yolo(YoloAdapter::load(&spec)?),
            AdapterKind::QwenAsr => LoadedModel::QwenAsr(QwenAsrAdapter::load(&spec)?),
            AdapterKind::IndexTts => LoadedModel::IndexTts(IndexTtsAdapter::load(&spec)?),
        };
        let now = Instant::now();
        Ok(Self {
            model,
            loaded_at: now,
            last_used: now,
            last_cache_released_at: None,
            state: ModelState::Warm,
        })
    }
}

fn effective_load_spec(spec: &ModelSpec) -> ModelSpec {
    effective_load_spec_with_availability(spec, RuntimeProviderAvailability::current())
}

fn effective_load_spec_with_availability(
    spec: &ModelSpec,
    availability: RuntimeProviderAvailability,
) -> ModelSpec {
    let mut effective = spec.clone();
    if let Some(provider_order) = effective_provider_order_for_model(
        &effective.id,
        &effective.runtime.provider_order,
        availability,
    ) {
        effective.runtime.provider_order = provider_order;
    }
    effective
}

fn effective_provider_order_for_model(
    model_id: &str,
    stored_provider_order: &[String],
    availability: RuntimeProviderAvailability,
) -> Option<Vec<String>> {
    if stored_provider_order.len() != 1 || !stored_provider_order[0].eq_ignore_ascii_case("cpu") {
        return None;
    }

    let validated = validated_runtime_providers_for_model(model_id)?;
    let mut effective = Vec::new();
    for provider in ["trt", "cuda", "dml", "cpu"] {
        let available = match provider {
            "trt" => availability.tensorrt,
            "cuda" => availability.cuda,
            "dml" => availability.directml,
            "cpu" => true,
            _ => false,
        };
        if available && validated.iter().any(|candidate| *candidate == provider) {
            effective.push(provider.to_string());
        }
    }

    (effective != stored_provider_order).then_some(effective)
}

fn validated_runtime_providers_for_model(model_id: &str) -> Option<&'static [&'static str]> {
    match model_id {
        // Single-session YOLO loading has the broadest generic ORT profile today,
        // but TRT is still withheld until a real local validation path exists.
        "yolo11n.onnx" => Some(&["cuda", "dml", "cpu"]),
        // Qwen ASR is a multi-session int4-first pipeline; keep this conservative
        // and only prefer CUDA above CPU until DML/TRT are validated end-to-end.
        "qwen3-asr-0.6b-onnx" => Some(&["cuda", "cpu"]),
        // IndexTTS deliberately stays CPU-only for now; q4/fp16 paths were
        // withdrawn and no GPU runtime path is currently validated.
        "indextts-1.5-onnx" => Some(&["cpu"]),
        _ => None,
    }
}

#[derive(Debug)]
enum LoadedModel {
    Yolo(YoloAdapter),
    QwenAsr(QwenAsrAdapter),
    IndexTts(IndexTtsAdapter),
    #[cfg(test)]
    Test {
        cache_releases: Arc<std::sync::atomic::AtomicUsize>,
    },
}

impl LoadedModel {
    fn infer(&mut self, task: &InferenceTask) -> Result<InferenceOutput> {
        match (&mut *self, task.kind, &task.input) {
            (
                LoadedModel::Yolo(adapter),
                TaskKind::ObjectDetect,
                InferenceInput::ObjectDetect { image },
            ) => adapter.object_detect(image),
            (
                LoadedModel::QwenAsr(adapter),
                TaskKind::AsrTranscribe,
                InferenceInput::AsrTranscribe { audio },
            ) => adapter.transcribe(audio),
            (
                LoadedModel::IndexTts(adapter),
                TaskKind::TtsSynthesize,
                InferenceInput::TtsSynthesize {
                    text,
                    reference_audio,
                },
            ) => adapter.synthesize_with_params(text, reference_audio.as_ref(), &task.params),
            (_, kind, _) => Err(InfraError::Unsupported(format!(
                "loaded adapter does not support task {kind:?}"
            ))),
        }
    }

    fn release_idle_cache(&mut self, model_id: &str) {
        match self {
            LoadedModel::Yolo(_) | LoadedModel::QwenAsr(_) | LoadedModel::IndexTts(_) => {
                tracing::debug!(
                    model_id,
                    "idle cache release hook reached; adapters currently keep no reusable per-request cache separate from the loaded model/session"
                );
            }
            #[cfg(test)]
            LoadedModel::Test { cache_releases } => {
                cache_releases.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use local_core::{
        ArtifactKind, BackendKind, FileRef, LoadPolicy, ModelArtifact, ResourceRequirement,
        RuntimePolicy,
    };
    use std::{
        collections::BTreeMap,
        path::PathBuf,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
    };
    use tokio::time::{sleep, timeout};

    #[test]
    fn default_runtime_config_releases_cache_after_30s_and_unloads_model_after_10m() {
        let config = RuntimeManagerConfig::default();

        assert_eq!(config.cache_idle_ttl, Duration::from_secs(30));
        assert_eq!(config.model_idle_ttl, Duration::from_secs(600));
        assert_eq!(config.min_residency, Duration::from_secs(0));
    }

    #[tokio::test]
    async fn cache_release_uses_per_model_last_used_and_does_not_unload_models() {
        let runtime = RuntimeManager::new(Vec::new(), test_runtime_config(50, 500, 0));
        let due_model = "cache-due-model".to_string();
        let fresh_model = "fresh-model".to_string();
        let now = Instant::now();
        let due_releases = Arc::new(AtomicUsize::new(0));
        let fresh_releases = Arc::new(AtomicUsize::new(0));
        {
            let mut loaded = runtime.loaded.lock().await;
            loaded.insert(
                due_model.clone(),
                test_loaded_entry_with_state_and_counter(
                    now - Duration::from_millis(100),
                    ModelState::Idle,
                    due_releases.clone(),
                ),
            );
            loaded.insert(
                fresh_model.clone(),
                test_loaded_entry_with_state_and_counter(
                    now,
                    ModelState::Idle,
                    fresh_releases.clone(),
                ),
            );
        }

        runtime.maintain_idle().await.expect("maintain idle");

        let loaded = runtime.loaded_models().await;
        assert!(
            loaded.contains(&due_model),
            "30s cache release must not unload model: {loaded:?}"
        );
        assert!(
            loaded.contains(&fresh_model),
            "fresh model should remain loaded: {loaded:?}"
        );
        assert_eq!(due_releases.load(Ordering::SeqCst), 1);
        assert_eq!(fresh_releases.load(Ordering::SeqCst), 0);

        runtime.maintain_idle().await.expect("maintain idle again");
        assert_eq!(
            due_releases.load(Ordering::SeqCst),
            1,
            "cache release should only run once until the model is used again"
        );
    }

    #[tokio::test]
    async fn model_unload_uses_per_model_last_used_and_unloads_only_due_models() {
        let runtime = RuntimeManager::new(Vec::new(), test_runtime_config(50, 100, 0));
        let old_model = "old-model".to_string();
        let fresh_model = "fresh-model".to_string();
        let now = Instant::now();
        {
            let mut loaded = runtime.loaded.lock().await;
            loaded.insert(
                old_model.clone(),
                test_loaded_entry_with_state(now - Duration::from_millis(150), ModelState::Idle),
            );
            loaded.insert(
                fresh_model.clone(),
                test_loaded_entry_with_state(now - Duration::from_millis(75), ModelState::Idle),
            );
        }

        runtime.maintain_idle().await.expect("maintain idle");

        let loaded = runtime.loaded_models().await;
        assert!(
            !loaded.contains(&old_model),
            "model past model idle ttl should be unloaded: {loaded:?}"
        );
        assert!(
            loaded.contains(&fresh_model),
            "model before model idle ttl should remain loaded: {loaded:?}"
        );
    }

    #[tokio::test]
    async fn busy_models_do_not_release_cache_or_unload() {
        let runtime = RuntimeManager::new(Vec::new(), test_runtime_config(50, 100, 0));
        let busy_model = "busy-model".to_string();
        let now = Instant::now();
        let busy_releases = Arc::new(AtomicUsize::new(0));
        {
            let mut loaded = runtime.loaded.lock().await;
            loaded.insert(
                busy_model.clone(),
                test_loaded_entry_with_state_and_counter(
                    now - Duration::from_millis(150),
                    ModelState::Busy,
                    busy_releases.clone(),
                ),
            );
        }

        runtime.maintain_idle().await.expect("maintain idle");

        let loaded = runtime.loaded_models().await;
        assert!(
            loaded.contains(&busy_model),
            "busy model must not be unloaded even after both ttls: {loaded:?}"
        );
        assert_eq!(
            busy_releases.load(Ordering::SeqCst),
            0,
            "busy model cache must not be released"
        );
    }

    #[tokio::test]
    async fn idle_maintenance_loop_releases_cache_without_followup_inference() {
        let runtime = Arc::new(RuntimeManager::new(
            Vec::new(),
            test_runtime_config(20, 200, 0),
        ));
        let cache_releases = Arc::new(AtomicUsize::new(0));
        {
            let mut loaded = runtime.loaded.lock().await;
            loaded.insert(
                "idle-model".to_string(),
                test_loaded_entry_with_state_and_counter(
                    Instant::now() - Duration::from_millis(50),
                    ModelState::Idle,
                    cache_releases.clone(),
                ),
            );
        }
        let _loop = runtime
            .clone()
            .spawn_idle_maintenance_loop_with_interval(Duration::from_millis(5));

        timeout(Duration::from_secs(1), async {
            loop {
                if cache_releases.load(Ordering::SeqCst) == 1 {
                    break;
                }
                sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("idle maintenance loop should release cache promptly");
        assert!(
            runtime
                .loaded_models()
                .await
                .contains(&"idle-model".to_string()),
            "cache release ttl should not unload model"
        );
    }

    #[tokio::test]
    async fn idle_maintenance_loop_unloads_model_without_followup_inference() {
        let runtime = Arc::new(RuntimeManager::new(
            Vec::new(),
            test_runtime_config(20, 40, 0),
        ));
        {
            let mut loaded = runtime.loaded.lock().await;
            loaded.insert(
                "idle-model".to_string(),
                test_loaded_entry_with_state(
                    Instant::now() - Duration::from_millis(50),
                    ModelState::Idle,
                ),
            );
        }
        let _loop = runtime
            .clone()
            .spawn_idle_maintenance_loop_with_interval(Duration::from_millis(5));

        timeout(Duration::from_secs(1), async {
            loop {
                if runtime.loaded_models().await.is_empty() {
                    break;
                }
                sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("idle maintenance loop should unload model promptly");
    }

    #[tokio::test]
    async fn disabled_indextts_model_is_gated_before_artifact_loading() {
        let runtime = RuntimeManager::new(
            vec![index_tts_spec(false, PathBuf::from("definitely-missing"))],
            RuntimeManagerConfig::default(),
        );
        let task = InferenceTask::new(
            TaskKind::TtsSynthesize,
            Some("indextts-test".to_string()),
            InferenceInput::TtsSynthesize {
                text: "hello".to_string(),
                reference_audio: Some(FileRef::local("ref.wav")),
            },
        );

        let err = runtime.infer(task).await.expect_err("disabled model");

        assert!(err.to_string().contains("model is disabled"), "{err}");
        assert!(runtime.loaded_models().await.is_empty());
    }

    #[tokio::test]
    async fn enabled_indextts_reports_missing_local_artifacts_clearly() {
        let runtime = RuntimeManager::new(
            vec![index_tts_spec(true, PathBuf::from("definitely-missing"))],
            RuntimeManagerConfig::default(),
        );
        let task = InferenceTask::new(
            TaskKind::TtsSynthesize,
            Some("indextts-test".to_string()),
            InferenceInput::TtsSynthesize {
                text: "hello".to_string(),
                reference_audio: Some(FileRef::local("ref.wav")),
            },
        );

        let err = runtime.infer(task).await.expect_err("missing artifacts");

        assert!(
            err.to_string()
                .contains("IndexTTS artifact root is not a directory"),
            "{err}"
        );
    }

    #[test]
    fn effective_provider_order_rewrites_only_runtime_clone_for_known_model() {
        let spec = test_spec(
            "yolo11n.onnx",
            AdapterKind::Yolo,
            true,
            PathBuf::from("missing"),
        );

        let effective = effective_load_spec_with_availability(
            &spec,
            RuntimeProviderAvailability {
                cuda: true,
                directml: true,
                tensorrt: true,
            },
        );

        assert_eq!(spec.runtime.provider_order, vec!["cpu".to_string()]);
        assert_eq!(
            effective.runtime.provider_order,
            vec!["cuda".to_string(), "dml".to_string(), "cpu".to_string()]
        );
    }

    #[test]
    fn effective_provider_order_preserves_explicit_non_default_order() {
        let mut spec = test_spec(
            "qwen3-asr-0.6b-onnx",
            AdapterKind::QwenAsr,
            true,
            PathBuf::from("missing"),
        );
        spec.runtime.provider_order = vec!["cuda".to_string(), "cpu".to_string()];

        let effective = effective_load_spec_with_availability(
            &spec,
            RuntimeProviderAvailability {
                cuda: true,
                directml: true,
                tensorrt: true,
            },
        );

        assert_eq!(
            effective.runtime.provider_order,
            spec.runtime.provider_order
        );
    }

    #[test]
    fn effective_provider_order_is_model_specific_and_conservative() {
        let qwen = test_spec(
            "qwen3-asr-0.6b-onnx",
            AdapterKind::QwenAsr,
            true,
            PathBuf::from("missing"),
        );
        let indextts = test_spec(
            "indextts-1.5-onnx",
            AdapterKind::IndexTts,
            true,
            PathBuf::from("missing"),
        );
        let availability = RuntimeProviderAvailability {
            cuda: true,
            directml: true,
            tensorrt: true,
        };

        assert_eq!(
            effective_load_spec_with_availability(&qwen, availability)
                .runtime
                .provider_order,
            vec!["cuda".to_string(), "cpu".to_string()]
        );
        assert_eq!(
            effective_load_spec_with_availability(&indextts, availability)
                .runtime
                .provider_order,
            vec!["cpu".to_string()]
        );
    }

    fn index_tts_spec(enabled: bool, path: PathBuf) -> ModelSpec {
        test_spec("indextts-test", AdapterKind::IndexTts, enabled, path)
    }

    fn test_runtime_config(
        cache_idle_ttl_ms: u64,
        model_idle_ttl_ms: u64,
        min_residency_ms: u64,
    ) -> RuntimeManagerConfig {
        RuntimeManagerConfig {
            cache_idle_ttl: Duration::from_millis(cache_idle_ttl_ms),
            model_idle_ttl: Duration::from_millis(model_idle_ttl_ms),
            min_residency: Duration::from_millis(min_residency_ms),
            memory_pressure_threshold: 0.85,
        }
    }

    fn test_loaded_entry_with_state(last_used: Instant, state: ModelState) -> LoadedEntry {
        test_loaded_entry_with_state_and_counter(last_used, state, Arc::new(AtomicUsize::new(0)))
    }

    fn test_loaded_entry_with_state_and_counter(
        last_used: Instant,
        state: ModelState,
        cache_releases: Arc<AtomicUsize>,
    ) -> LoadedEntry {
        LoadedEntry {
            model: LoadedModel::Test { cache_releases },
            loaded_at: last_used,
            last_used,
            last_cache_released_at: None,
            state,
        }
    }

    fn test_spec(id: &str, adapter: AdapterKind, enabled: bool, path: PathBuf) -> ModelSpec {
        let task_kinds = match adapter {
            AdapterKind::Yolo => vec![TaskKind::ObjectDetect],
            AdapterKind::QwenAsr => vec![TaskKind::AsrTranscribe],
            AdapterKind::IndexTts => vec![TaskKind::TtsSynthesize],
        };
        ModelSpec {
            id: id.to_string(),
            name: format!("{id} Test"),
            enabled,
            task_kinds,
            adapter,
            backend: BackendKind::Ort,
            artifacts: vec![ModelArtifact {
                kind: ArtifactKind::Local,
                path,
                source_path: None,
                sha256: None,
                url: None,
                repo_id: None,
                revision: None,
                files: Vec::new(),
                allow_patterns: Vec::new(),
                metadata: BTreeMap::new(),
            }],
            runtime: RuntimePolicy {
                provider_order: vec!["cpu".to_string()],
                max_concurrency: 1,
                idle_ttl_sec: 300,
            },
            resources: ResourceRequirement::default(),
            load_policy: LoadPolicy::default(),
            metadata: BTreeMap::new(),
        }
    }
}
