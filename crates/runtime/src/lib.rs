use local_adapter_e5_embedding::E5EmbeddingAdapter;
use local_adapter_index_tts::IndexTtsAdapter;
use local_adapter_mmarco_reranker::MmarcoRerankerAdapter;
use local_adapter_sensevoice_asr::SenseVoiceAsrAdapter;
use local_adapter_yolo::YoloAdapter;
use local_backend_ort::probe_runtime_execution_provider_availability;
use local_core::{
    AdapterKind, InferenceInput, InferenceOutput, InferenceTask, ModelSpec, ModelState, TaskKind,
};
use local_error::{InfraError, Result};
use std::{
    collections::HashMap,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex as StdMutex,
    },
    time::{Duration, Instant},
};
use tokio::sync::{Mutex, Semaphore};

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
    loaded: Mutex<HashMap<String, Arc<ModelSlot>>>,
    config: RuntimeManagerConfig,
    queued_jobs: Arc<AtomicUsize>,
    active_jobs: Arc<AtomicUsize>,
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
            queued_jobs: Arc::new(AtomicUsize::new(0)),
            active_jobs: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub async fn infer(&self, task: InferenceTask) -> Result<InferenceOutput> {
        let total_started = Instant::now();
        let spec = self.resolve_spec(&task)?;
        let model_id = spec.id.clone();
        let slot = {
            let mut loaded = self.loaded.lock().await;
            loaded
                .entry(model_id.clone())
                .or_insert_with(|| Arc::new(ModelSlot::new(spec.runtime.max_concurrency)))
                .clone()
        };

        let queued = QueuedGuard::new(self.queued_jobs.clone());
        let queue_started = Instant::now();
        let permit = slot
            .admission
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| InfraError::Runtime(format!("model `{model_id}` admission closed")))?;
        let queue_wait = queue_started.elapsed();
        queued.finish();
        let active = ActiveGuard::new(self.active_jobs.clone());
        let request_id = task.id;
        let completed_model_id = model_id.clone();
        let execution_started = Instant::now();
        let result = tokio::task::spawn_blocking(move || {
            let _active = active;
            let _permit = permit;
            let acquire_started = Instant::now();
            let mut entry = slot
                .entry
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let acquire = acquire_started.elapsed();
            let load_started = Instant::now();
            if entry.is_none() {
                *entry = Some(LoadedEntry::load(&spec)?);
            }
            let load = load_started.elapsed();
            let loaded = entry.as_mut().ok_or_else(|| {
                InfraError::Runtime(format!("model `{model_id}` failed to enter loaded cache"))
            })?;
            loaded.state = ModelState::Busy;
            let infer_started = Instant::now();
            let (result, panicked) = infer_model_catching_panic(loaded, &task, &model_id);
            let execution = infer_started.elapsed();
            tracing::info!(
                request_id = %request_id,
                model_id,
                acquire_ms = acquire.as_millis() as u64,
                load_ms = load.as_millis() as u64,
                execution_ms = execution.as_millis() as u64,
                success = result.is_ok(),
                panicked,
                "runtime inference stages"
            );
            if panicked {
                // Never reuse an adapter whose mutable session state may have
                // been corrupted. The canonical slot remains in the map and
                // the next admitted request will reload it.
                let suspect = entry.take();
                drop(entry);
                drop(suspect);
            }
            result
        })
        .await
        .map_err(|err| InfraError::Runtime(format!("model inference task failed: {err}")))?;
        tracing::info!(
            request_id = %request_id,
            model_id = completed_model_id,
            queue_wait_ms = queue_wait.as_millis() as u64,
            execution_total_ms = execution_started.elapsed().as_millis() as u64,
            total_ms = total_started.elapsed().as_millis() as u64,
            success = result.is_ok(),
            "runtime inference completed"
        );
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
        let slots = self.slot_snapshot().await;
        slots
            .into_iter()
            .filter_map(|(id, slot)| match slot.entry.try_lock() {
                Ok(entry) => entry.as_ref().map(|_| id),
                // A contended slot is loading or executing and therefore is
                // truthfully considered loaded/busy without blocking heartbeat.
                Err(std::sync::TryLockError::WouldBlock) => Some(id),
                Err(std::sync::TryLockError::Poisoned(err)) => {
                    err.into_inner().as_ref().map(|_| id)
                }
            })
            .collect()
    }

    pub async fn states(&self) -> HashMap<String, ModelState> {
        let slots = self.slot_snapshot().await;
        slots
            .into_iter()
            .filter_map(|(id, slot)| match slot.entry.try_lock() {
                Ok(entry) => entry.as_ref().map(|entry| (id, entry.state)),
                Err(std::sync::TryLockError::WouldBlock) => Some((id, ModelState::Busy)),
                Err(std::sync::TryLockError::Poisoned(err)) => {
                    err.into_inner().as_ref().map(|entry| (id, entry.state))
                }
            })
            .collect()
    }

    async fn slot_snapshot(&self) -> Vec<(String, Arc<ModelSlot>)> {
        self.loaded
            .lock()
            .await
            .iter()
            .map(|(id, slot)| (id.clone(), slot.clone()))
            .collect()
    }

    pub fn queued_jobs(&self) -> usize {
        self.queued_jobs.load(Ordering::Relaxed)
    }

    pub fn active_jobs(&self) -> usize {
        self.active_jobs.load(Ordering::Relaxed)
    }

    pub async fn unload_idle(&self) -> Result<()> {
        self.maintain_idle().await
    }

    pub async fn maintain_idle(&self) -> Result<()> {
        let mut loaded = self.loaded.lock().await;
        let now = Instant::now();
        let mut removed = Vec::new();
        loaded.retain(|id, slot| {
            // Reserving admission is atomic. Arc::strong_count also prevents
            // removing a slot already handed to an inference that has not yet
            // acquired admission. Together with holding the map lock, this
            // preserves exactly one canonical slot per model.
            if Arc::strong_count(slot) != 1 {
                return true;
            }
            let Ok(_maintenance_permit) = slot.admission.clone().try_acquire_owned() else {
                return true;
            };
            let Ok(mut guard) = slot.entry.try_lock() else {
                return true;
            };
            let Some(entry) = guard.as_mut() else {
                return true;
            };
            let can_unload = entry.state == ModelState::Idle
                && now.duration_since(entry.last_used) >= self.config.model_idle_ttl
                && now.duration_since(entry.loaded_at) >= self.config.min_residency;
            if can_unload {
                tracing::info!(model_id = id, "unloading idle model");
                if let Some(entry) = guard.take() {
                    removed.push(entry);
                }
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
        drop(loaded);
        // ORT session destruction can be expensive. Never perform it while the
        // global slot-map lock is held.
        drop(removed);
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

fn infer_model_catching_panic(
    loaded: &mut LoadedEntry,
    task: &InferenceTask,
    model_id: &str,
) -> (Result<InferenceOutput>, bool) {
    match catch_unwind(AssertUnwindSafe(|| loaded.model.infer(task))) {
        Ok(result) => {
            loaded.last_used = Instant::now();
            loaded.last_cache_released_at = None;
            loaded.state = ModelState::Idle;
            (result, false)
        }
        Err(_) => (
            Err(InfraError::Runtime(format!(
                "model `{model_id}` panicked during inference; invalidating loaded instance"
            ))),
            true,
        ),
    }
}

#[derive(Debug)]
struct ModelSlot {
    entry: StdMutex<Option<LoadedEntry>>,
    admission: Arc<Semaphore>,
}

impl ModelSlot {
    fn new(configured_max_concurrency: usize) -> Self {
        if configured_max_concurrency == 0 {
            tracing::warn!(
                configured_max_concurrency,
                effective_max_concurrency = 1,
                "max_concurrency must be positive; using safe serial per-model admission"
            );
        } else if configured_max_concurrency != 1 {
            tracing::warn!(
                configured_max_concurrency,
                effective_max_concurrency = 1,
                "loaded adapters expose mutable sessions; using safe serial per-model admission"
            );
        }
        Self {
            entry: StdMutex::new(None),
            admission: Arc::new(Semaphore::new(1)),
        }
    }
}

struct QueuedGuard {
    counter: Arc<AtomicUsize>,
    active: bool,
}

impl QueuedGuard {
    fn new(counter: Arc<AtomicUsize>) -> Self {
        counter.fetch_add(1, Ordering::Relaxed);
        Self {
            counter,
            active: true,
        }
    }

    fn finish(mut self) {
        self.counter.fetch_sub(1, Ordering::Relaxed);
        self.active = false;
    }
}

impl Drop for QueuedGuard {
    fn drop(&mut self) {
        if self.active {
            self.counter.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

struct ActiveGuard(Arc<AtomicUsize>);

impl ActiveGuard {
    fn new(counter: Arc<AtomicUsize>) -> Self {
        counter.fetch_add(1, Ordering::Relaxed);
        Self(counter)
    }
}

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
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
            AdapterKind::SenseVoiceAsr => {
                LoadedModel::SenseVoiceAsr(SenseVoiceAsrAdapter::load(&spec)?)
            }
            AdapterKind::IndexTts => LoadedModel::IndexTts(IndexTtsAdapter::load(&spec)?),
            AdapterKind::E5Embedding => LoadedModel::E5Embedding(E5EmbeddingAdapter::load(&spec)?),
            AdapterKind::MmarcoReranker => {
                LoadedModel::MmarcoReranker(MmarcoRerankerAdapter::load(&spec)?)
            }
        };
        match &model {
            LoadedModel::E5Embedding(adapter) => {
                let report = adapter.provider_report();
                tracing::info!(
                    model_id = spec.id,
                    model_path = %adapter.artifacts().model.display(),
                    provider = ?report.provider,
                    cpu_fallback_used = report.cpu_fallback_used,
                    graph_has_pooling = adapter.graph_has_pooling(),
                    pinned_cuda_io_enabled = adapter.pinned_cuda_io_enabled(),
                    "text model ORT session loaded"
                );
            }
            LoadedModel::MmarcoReranker(adapter) => {
                let report = adapter.provider_report();
                tracing::info!(
                    model_id = spec.id,
                    model_path = %adapter.artifacts().model.display(),
                    provider = ?report.provider,
                    cpu_fallback_used = report.cpu_fallback_used,
                    "text model ORT session loaded"
                );
            }
            _ => {}
        }
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
    let validated = validated_runtime_providers_for_model(model_id)?;
    let mut effective = Vec::new();
    let default_order;
    let intended_order = if stored_provider_order.len() == 1
        && stored_provider_order[0].eq_ignore_ascii_case("cpu")
    {
        default_order = ["trt", "cuda", "dml", "cpu"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        &default_order
    } else {
        stored_provider_order
    };
    for provider in intended_order {
        let provider = provider.to_ascii_lowercase();
        let available = match provider {
            ref name if name == "trt" || name == "tensorrt" => availability.tensorrt,
            ref name if name == "cuda" => availability.cuda,
            ref name if name == "dml" || name == "directml" => availability.directml,
            ref name if name == "cpu" => true,
            _ => false,
        };
        let canonical = match provider.as_str() {
            "tensorrt" => "trt",
            "directml" => "dml",
            provider => provider,
        };
        if available && validated.iter().any(|candidate| *candidate == canonical) {
            effective.push(canonical.to_string());
        }
    }

    (effective != stored_provider_order).then_some(effective)
}

fn validated_runtime_providers_for_model(model_id: &str) -> Option<&'static [&'static str]> {
    match model_id {
        // Single-session YOLO loading has the broadest generic ORT profile today,
        // but TRT is still withheld until a real local validation path exists.
        "yolo11n.onnx" => Some(&["cuda", "dml", "cpu"]),
        // The FunASR ASR pipeline loads FSMN-VAD, SenseVoice, and CAM++ ONNX
        // sessions with one shared provider policy.
        "sensevoice-small-onnx" => Some(&["cuda", "cpu"]),
        // All A-F/prefill sessions receive the same provider selection. Root
        // FP32 CUDA is policy-enabled, but still needs real NVIDIA validation.
        "indextts-1.5-onnx" => Some(&["cuda", "cpu"]),
        "multilingual-e5-small-onnx" => Some(&["cuda", "cpu"]),
        "mmarco-minilm-l12-onnx" => Some(&["cuda", "cpu"]),
        _ => None,
    }
}

#[derive(Debug)]
enum LoadedModel {
    Yolo(YoloAdapter),
    SenseVoiceAsr(SenseVoiceAsrAdapter),
    IndexTts(IndexTtsAdapter),
    E5Embedding(E5EmbeddingAdapter),
    MmarcoReranker(MmarcoRerankerAdapter),
    #[cfg(test)]
    Test {
        cache_releases: Arc<std::sync::atomic::AtomicUsize>,
        panic_on_infer: Arc<std::sync::atomic::AtomicBool>,
    },
}

impl LoadedModel {
    fn infer(&mut self, task: &InferenceTask) -> Result<InferenceOutput> {
        #[cfg(test)]
        if let LoadedModel::Test { panic_on_infer, .. } = self {
            if panic_on_infer.swap(false, Ordering::SeqCst) {
                panic!("intentional test executor panic");
            }
            return Ok(InferenceOutput::Accepted {
                job_id: task.id.to_string(),
            });
        }
        match (&mut *self, task.kind, &task.input) {
            (
                LoadedModel::Yolo(adapter),
                TaskKind::ObjectDetect,
                InferenceInput::ObjectDetect { image },
            ) => adapter.object_detect(image),
            (
                LoadedModel::SenseVoiceAsr(adapter),
                TaskKind::AsrTranscribe,
                InferenceInput::AsrTranscribe { audio },
            ) => adapter.transcribe_with_params(audio, &task.params),
            (
                LoadedModel::IndexTts(adapter),
                TaskKind::TtsSynthesize,
                InferenceInput::TtsSynthesize {
                    text,
                    reference_audio,
                },
            ) => adapter.synthesize_with_request_id(
                task.id,
                text,
                reference_audio.as_ref(),
                &task.params,
            ),
            (
                LoadedModel::E5Embedding(adapter),
                TaskKind::TextEmbed,
                InferenceInput::TextEmbed { texts, input_type },
            ) => adapter.embed(texts, *input_type),
            (
                LoadedModel::MmarcoReranker(adapter),
                TaskKind::TextRerank,
                InferenceInput::TextRerank {
                    query,
                    documents,
                    top_n,
                },
            ) => adapter.rerank(query, documents, *top_n),
            (_, kind, _) => Err(InfraError::Unsupported(format!(
                "loaded adapter does not support task {kind:?}"
            ))),
        }
    }

    fn release_idle_cache(&mut self, model_id: &str) {
        match self {
            LoadedModel::Yolo(_)
            | LoadedModel::SenseVoiceAsr(_)
            | LoadedModel::IndexTts(_)
            | LoadedModel::E5Embedding(_)
            | LoadedModel::MmarcoReranker(_) => {
                tracing::debug!(
                    model_id,
                    "idle cache release hook reached; adapters currently keep no reusable per-request cache separate from the loaded model/session"
                );
            }
            #[cfg(test)]
            LoadedModel::Test { cache_releases, .. } => {
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
    async fn per_model_admission_serializes_same_model_but_not_unrelated_models() {
        let first = Arc::new(ModelSlot::new(1));
        let unrelated = Arc::new(ModelSlot::new(1));
        let held = first
            .admission
            .clone()
            .acquire_owned()
            .await
            .expect("first permit");

        assert!(
            timeout(
                Duration::from_millis(20),
                first.admission.clone().acquire_owned()
            )
            .await
            .is_err(),
            "same-model request must queue"
        );
        assert!(timeout(
            Duration::from_millis(20),
            unrelated.admission.clone().acquire_owned()
        )
        .await
        .expect("unrelated model must not queue")
        .is_ok());
        drop(held);
    }

    #[tokio::test]
    async fn maintenance_cannot_remove_a_slot_already_observed_by_inference() {
        let runtime = RuntimeManager::new(Vec::new(), test_runtime_config(0, 0, 0));
        let slot =
            test_loaded_entry_with_state(Instant::now() - Duration::from_secs(1), ModelState::Idle);
        runtime
            .loaded
            .lock()
            .await
            .insert("canonical".to_string(), slot.clone());
        let observed_by_inference = slot.clone();

        runtime.maintain_idle().await.expect("maintenance");

        let canonical = runtime
            .loaded
            .lock()
            .await
            .get("canonical")
            .cloned()
            .expect("observed slot must remain canonical");
        assert!(Arc::ptr_eq(&canonical, &observed_by_inference));
    }

    #[tokio::test]
    async fn state_queries_never_wait_for_busy_entry_mutex() {
        let runtime = RuntimeManager::new(Vec::new(), RuntimeManagerConfig::default());
        let busy = test_loaded_entry_with_state(Instant::now(), ModelState::Busy);
        let unrelated = test_loaded_entry_with_state(Instant::now(), ModelState::Idle);
        {
            let mut loaded = runtime.loaded.lock().await;
            loaded.insert("busy".to_string(), busy.clone());
            loaded.insert("unrelated".to_string(), unrelated.clone());
        }
        let _busy_guard = busy.entry.lock().expect("busy entry");

        let states = timeout(Duration::from_millis(20), runtime.states())
            .await
            .expect("state query must not block");
        assert_eq!(states.get("busy"), Some(&ModelState::Busy));
        assert_eq!(states.get("unrelated"), Some(&ModelState::Idle));
        assert!(
            unrelated.admission.clone().try_acquire_owned().is_ok(),
            "heartbeat/state query must not stop unrelated admission"
        );
    }

    #[test]
    fn panic_invalidates_test_model_without_poisoning_slot() {
        let panic_on_infer = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let mut entry = LoadedEntry {
            model: LoadedModel::Test {
                cache_releases: Arc::new(AtomicUsize::new(0)),
                panic_on_infer,
            },
            loaded_at: Instant::now(),
            last_used: Instant::now(),
            last_cache_released_at: None,
            state: ModelState::Busy,
        };
        let task = InferenceTask::new(
            TaskKind::TtsSynthesize,
            None,
            InferenceInput::TtsSynthesize {
                text: "not logged".to_string(),
                reference_audio: None,
            },
        );

        let (result, panicked) = infer_model_catching_panic(&mut entry, &task, "test");
        assert!(panicked);
        assert!(result
            .expect_err("typed panic error")
            .to_string()
            .contains("panicked"));
        // The caller invalidates this entry; proving the panic was caught also
        // proves it cannot poison the canonical slot mutex.
    }

    #[test]
    fn queue_and_active_guards_decrement_on_drop() {
        let queued = Arc::new(AtomicUsize::new(0));
        let active = Arc::new(AtomicUsize::new(0));
        {
            let _queued_guard = QueuedGuard::new(queued.clone());
            let _active_guard = ActiveGuard::new(active.clone());
            assert_eq!(queued.load(Ordering::Relaxed), 1);
            assert_eq!(active.load(Ordering::Relaxed), 1);
        }
        assert_eq!(queued.load(Ordering::Relaxed), 0);
        assert_eq!(active.load(Ordering::Relaxed), 0);
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
            "sensevoice-small-onnx",
            AdapterKind::SenseVoiceAsr,
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
    fn effective_provider_order_filters_unavailable_cuda_before_loading() {
        let mut spec = test_spec(
            "indextts-1.5-onnx",
            AdapterKind::IndexTts,
            true,
            PathBuf::from("missing"),
        );
        spec.runtime.provider_order = vec!["cuda".to_string(), "cpu".to_string()];

        let effective = effective_load_spec_with_availability(
            &spec,
            RuntimeProviderAvailability {
                cuda: false,
                directml: false,
                tensorrt: false,
            },
        );

        assert_eq!(effective.runtime.provider_order, vec!["cpu".to_string()]);
        assert_eq!(
            spec.runtime.provider_order,
            vec!["cuda".to_string(), "cpu".to_string()]
        );
    }

    #[test]
    fn effective_provider_order_preserves_cuda_when_available() {
        for (model_id, adapter) in [
            ("yolo11n.onnx", AdapterKind::Yolo),
            ("sensevoice-small-onnx", AdapterKind::SenseVoiceAsr),
            ("indextts-1.5-onnx", AdapterKind::IndexTts),
        ] {
            let mut spec = test_spec(model_id, adapter, true, PathBuf::from("missing"));
            spec.runtime.provider_order = vec!["cuda".to_string(), "cpu".to_string()];

            let effective = effective_load_spec_with_availability(
                &spec,
                RuntimeProviderAvailability {
                    cuda: true,
                    directml: false,
                    tensorrt: false,
                },
            );

            assert_eq!(
                effective.runtime.provider_order,
                vec!["cuda".to_string(), "cpu".to_string()],
                "model {model_id}"
            );
        }
    }

    #[test]
    fn effective_provider_order_is_model_specific_and_conservative() {
        let mut sensevoice = test_spec(
            "sensevoice-small-onnx",
            AdapterKind::SenseVoiceAsr,
            true,
            PathBuf::from("missing"),
        );
        let mut indextts = test_spec(
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
        sensevoice.runtime.provider_order = vec!["cuda".to_string(), "cpu".to_string()];
        indextts.runtime.provider_order = vec!["cuda".to_string(), "cpu".to_string()];

        assert_eq!(
            effective_load_spec_with_availability(&sensevoice, availability)
                .runtime
                .provider_order,
            vec!["cuda".to_string(), "cpu".to_string()]
        );
        assert_eq!(
            effective_load_spec_with_availability(&indextts, availability)
                .runtime
                .provider_order,
            vec!["cuda".to_string(), "cpu".to_string()]
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

    fn test_loaded_entry_with_state(last_used: Instant, state: ModelState) -> Arc<ModelSlot> {
        test_loaded_entry_with_state_and_counter(last_used, state, Arc::new(AtomicUsize::new(0)))
    }

    fn test_loaded_entry_with_state_and_counter(
        last_used: Instant,
        state: ModelState,
        cache_releases: Arc<AtomicUsize>,
    ) -> Arc<ModelSlot> {
        Arc::new(ModelSlot {
            entry: StdMutex::new(Some(LoadedEntry {
                model: LoadedModel::Test {
                    cache_releases,
                    panic_on_infer: Arc::new(std::sync::atomic::AtomicBool::new(false)),
                },
                loaded_at: last_used,
                last_used,
                last_cache_released_at: None,
                state,
            })),
            admission: Arc::new(Semaphore::new(1)),
        })
    }

    fn test_spec(id: &str, adapter: AdapterKind, enabled: bool, path: PathBuf) -> ModelSpec {
        let task_kinds = match adapter {
            AdapterKind::Yolo => vec![TaskKind::ObjectDetect],
            AdapterKind::SenseVoiceAsr => vec![TaskKind::AsrTranscribe],
            AdapterKind::IndexTts => vec![TaskKind::TtsSynthesize],
            AdapterKind::E5Embedding => vec![TaskKind::TextEmbed],
            AdapterKind::MmarcoReranker => vec![TaskKind::TextRerank],
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
