use lcoal_adapter_index_tts::IndexTtsAdapter;
use lcoal_adapter_qwen_asr::QwenAsrAdapter;
use lcoal_adapter_yolo::YoloAdapter;
use lcoal_core::{
    AdapterKind, InferenceInput, InferenceOutput, InferenceTask, ModelSpec, ModelState, TaskKind,
};
use lcoal_error::{InfraError, Result};
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};
use tokio::sync::Mutex;

#[derive(Debug, Clone)]
pub struct RuntimeManagerConfig {
    pub idle_ttl: Duration,
    pub min_residency: Duration,
    pub memory_pressure_threshold: f32,
}

impl Default for RuntimeManagerConfig {
    fn default() -> Self {
        Self {
            idle_ttl: Duration::from_secs(300),
            min_residency: Duration::from_secs(60),
            memory_pressure_threshold: 0.85,
        }
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
        self.unload_idle().await?;
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
        entry.state = ModelState::Idle;
        result
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
        let mut loaded = self.loaded.lock().await;
        let now = Instant::now();
        loaded.retain(|id, entry| {
            let can_unload = entry.state == ModelState::Idle
                && now.duration_since(entry.last_used) >= self.config.idle_ttl
                && now.duration_since(entry.loaded_at) >= self.config.min_residency;
            if can_unload {
                tracing::info!(model_id = id, "unloading idle model");
            }
            !can_unload
        });
        self.note_memory_pressure_policy();
        Ok(())
    }

    fn note_memory_pressure_policy(&self) {
        tracing::trace!(
            threshold = self.config.memory_pressure_threshold,
            "memory-pressure eviction policy is configured but passive in this MVP; idle TTL unloading is the active eviction mechanism"
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
    state: ModelState,
}

impl LoadedEntry {
    fn load(spec: &ModelSpec) -> Result<Self> {
        tracing::info!(model_id = spec.id, adapter = ?spec.adapter, "lazy loading model");
        let model = match spec.adapter {
            AdapterKind::Yolo => LoadedModel::Yolo(YoloAdapter::load(spec)?),
            AdapterKind::QwenAsr => LoadedModel::QwenAsr(QwenAsrAdapter::load(spec)?),
            AdapterKind::IndexTts => LoadedModel::IndexTts(IndexTtsAdapter::load(spec)?),
        };
        let now = Instant::now();
        Ok(Self {
            model,
            loaded_at: now,
            last_used: now,
            state: ModelState::Warm,
        })
    }
}

#[derive(Debug)]
enum LoadedModel {
    Yolo(YoloAdapter),
    QwenAsr(QwenAsrAdapter),
    IndexTts(IndexTtsAdapter),
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use lcoal_core::{
        ArtifactKind, BackendKind, FileRef, LoadPolicy, ModelArtifact, ResourceRequirement,
        RuntimePolicy,
    };
    use std::{collections::BTreeMap, path::PathBuf};

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

    fn index_tts_spec(enabled: bool, path: PathBuf) -> ModelSpec {
        ModelSpec {
            id: "indextts-test".to_string(),
            name: "IndexTTS Test".to_string(),
            enabled,
            task_kinds: vec![TaskKind::TtsSynthesize],
            adapter: AdapterKind::IndexTts,
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
