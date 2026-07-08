use lcoal_core::{
    AdapterKind, ArtifactKind, BackendKind, LoadPolicy, ModelArtifact, ModelSpec,
    ResourceRequirement, RuntimePolicy, TaskKind,
};
use lcoal_error::{InfraError, Result};
use serde_json::json;
use std::{collections::BTreeMap, fs, path::Path, sync::Arc};
use tokio::sync::RwLock;

#[derive(Debug, Clone, Default)]
pub struct ModelRegistry {
    inner: Arc<RwLock<BTreeMap<String, ModelSpec>>>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_models(models: Vec<ModelSpec>) -> Self {
        let map = models
            .into_iter()
            .map(|spec| (spec.id.clone(), spec))
            .collect();
        Self {
            inner: Arc::new(RwLock::new(map)),
        }
    }

    pub async fn from_models_async(models: Vec<ModelSpec>) -> Self {
        let registry = Self::new();
        registry.replace(models).await;
        registry
    }

    pub async fn load_dir(path: impl AsRef<Path>) -> Result<Self> {
        let registry = Self::new();
        registry.reload_dir(path).await?;
        Ok(registry)
    }

    pub async fn reload_dir(&self, path: impl AsRef<Path>) -> Result<()> {
        let models = load_yaml_specs(path)?;
        self.replace(models).await;
        Ok(())
    }

    pub async fn replace(&self, models: Vec<ModelSpec>) {
        *self.inner.write().await = models
            .into_iter()
            .map(|spec| (spec.id.clone(), spec))
            .collect();
    }

    pub async fn upsert(&self, spec: ModelSpec) -> ModelSpec {
        self.inner
            .write()
            .await
            .insert(spec.id.clone(), spec.clone());
        spec
    }

    pub async fn load_dir_specs(path: impl AsRef<Path>) -> Result<Vec<ModelSpec>> {
        load_yaml_specs(path)
    }

    pub async fn load_dir_into_existing(&self, path: impl AsRef<Path>) -> Result<()> {
        for spec in load_yaml_specs(path)? {
            self.upsert(spec).await;
        }
        Ok(())
    }

    pub async fn reload_dir_legacy(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let entries =
            fs::read_dir(path).map_err(|e| InfraError::io(Some(path.to_path_buf()), e))?;
        let mut models = BTreeMap::new();
        for entry in entries {
            let entry = entry.map_err(|e| InfraError::io(Some(path.to_path_buf()), e))?;
            let file_path = entry.path();
            let is_yaml = file_path
                .extension()
                .and_then(|v| v.to_str())
                .map(|v| matches!(v, "yaml" | "yml"))
                .unwrap_or(false);
            if !is_yaml {
                continue;
            }
            let bytes =
                fs::read(&file_path).map_err(|e| InfraError::io(Some(file_path.clone()), e))?;
            let spec: ModelSpec = serde_yaml::from_slice(&bytes)?;
            if models.insert(spec.id.clone(), spec).is_some() {
                return Err(InfraError::Registry(format!(
                    "duplicate model id loaded from {}",
                    file_path.display()
                )));
            }
        }
        *self.inner.write().await = models;
        Ok(())
    }

    pub async fn list(&self) -> Vec<ModelSpec> {
        self.inner.read().await.values().cloned().collect()
    }

    pub async fn get(&self, id: &str) -> Option<ModelSpec> {
        self.inner.read().await.get(id).cloned()
    }

    pub async fn set_enabled(&self, id: &str, enabled: bool) -> Result<ModelSpec> {
        let mut guard = self.inner.write().await;
        let spec = guard
            .get_mut(id)
            .ok_or_else(|| InfraError::NotFound(format!("model `{id}`")))?;
        spec.enabled = enabled;
        Ok(spec.clone())
    }

    pub async fn enabled_for_task(&self, kind: TaskKind) -> Vec<ModelSpec> {
        self.inner
            .read()
            .await
            .values()
            .filter(|spec| spec.enabled && spec.task_kinds.contains(&kind))
            .cloned()
            .collect()
    }
}

pub fn load_yaml_specs(path: impl AsRef<Path>) -> Result<Vec<ModelSpec>> {
    let path = path.as_ref();
    if !path.exists() {
        tracing::warn!(path = %path.display(), "models YAML directory does not exist; using only built-in/database models");
        return Ok(Vec::new());
    }
    let entries = fs::read_dir(path).map_err(|e| InfraError::io(Some(path.to_path_buf()), e))?;
    let mut models = BTreeMap::new();
    for entry in entries {
        let entry = entry.map_err(|e| InfraError::io(Some(path.to_path_buf()), e))?;
        let file_path = entry.path();
        let is_yaml = file_path
            .extension()
            .and_then(|v| v.to_str())
            .map(|v| matches!(v, "yaml" | "yml"))
            .unwrap_or(false);
        if !is_yaml {
            continue;
        }
        let bytes = fs::read(&file_path).map_err(|e| InfraError::io(Some(file_path.clone()), e))?;
        let spec: ModelSpec = serde_yaml::from_slice(&bytes)?;
        if models.insert(spec.id.clone(), spec).is_some() {
            return Err(InfraError::Registry(format!(
                "duplicate model id loaded from {}",
                file_path.display()
            )));
        }
    }
    Ok(models.into_values().collect())
}

pub fn default_catalog(model_dir: impl AsRef<Path>) -> Vec<ModelSpec> {
    let model_dir = model_dir.as_ref();
    vec![
        qwen_asr_default(model_dir),
        yolo_default(model_dir),
        index_tts_default(model_dir),
    ]
}

fn index_tts_default(model_dir: &Path) -> ModelSpec {
    let id = "indextts-1.5-onnx";
    let root = model_dir.join(id);
    let mut metadata = BTreeMap::new();
    metadata.insert("runtime".to_string(), json!("onnxruntime"));
    metadata.insert("model_family".to_string(), json!("index_tts"));
    metadata.insert("task".to_string(), json!("text-to-speech"));
    metadata.insert("experimental".to_string(), json!(true));
    metadata.insert("precision".to_string(), json!("cpu-fp32"));
    metadata.insert(
        "artifact_note".to_string(),
        json!("FP32 artifacts are materialized from ModaLeap/indextts-1.5-onnx or an equivalent artifact root. Runtime expects IndexTTS_A.onnx through IndexTTS_F.onnx plus bpe.model and manifest.yaml/json at the artifact root."),
    );
    metadata.insert(
        "mvp_status".to_string(),
        json!("Adapter validates FP32 artifacts materialized from Hugging Face or an equivalent artifact root, loads root A-F ORT sessions, and uses a lightweight Rust text normalizer plus bpe.model SentencePiece tokenization; FP32 artifact smoke is still required. q4cpu/fp16gpu paths have been withdrawn."),
    );
    ModelSpec {
        id: id.to_string(),
        name: "IndexTTS 1.5 ONNX (experimental)".to_string(),
        enabled: false,
        task_kinds: vec![TaskKind::TtsSynthesize],
        adapter: AdapterKind::IndexTts,
        backend: BackendKind::Ort,
        artifacts: vec![ModelArtifact {
            kind: ArtifactKind::HuggingFace,
            path: root,
            source_path: None,
            sha256: None,
            url: None,
            repo_id: Some("ModaLeap/indextts-1.5-onnx".to_string()),
            revision: None,
            files: [
                "IndexTTS_A.onnx",
                "IndexTTS_B.onnx",
                "IndexTTS_C.onnx",
                "IndexTTS_D.onnx",
                "IndexTTS_E.onnx",
                "IndexTTS_F.onnx",
                "bpe.model",
                "manifest.json",
                "manifest.yaml",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
            allow_patterns: Vec::new(),
            metadata: BTreeMap::new(),
        }],
        runtime: RuntimePolicy {
            provider_order: vec!["cpu".to_string()],
            max_concurrency: 1,
            idle_ttl_sec: 300,
        },
        resources: ResourceRequirement {
            min_ram_mb: 8192,
            min_vram_mb: 0,
        },
        load_policy: LoadPolicy::default(),
        metadata,
    }
}

pub fn materialize_artifact_paths(spec: &mut ModelSpec, model_dir: impl AsRef<Path>) {
    let root = model_dir.as_ref().join(&spec.id);
    for artifact in &mut spec.artifacts {
        if artifact.kind == ArtifactKind::Local
            && artifact.source_path.is_none()
            && !artifact.path.as_os_str().is_empty()
            && artifact.path != root
            && !artifact.path.starts_with(&root)
        {
            artifact.source_path = Some(artifact.path.clone());
        }
        artifact.path = match artifact.kind {
            ArtifactKind::Local => root.clone(),
            ArtifactKind::HuggingFace if artifact.files.len() == 1 => root.join(&artifact.files[0]),
            ArtifactKind::HuggingFace => root.clone(),
            ArtifactKind::Url => {
                let filename = artifact
                    .url
                    .as_deref()
                    .and_then(|url| url.rsplit('/').next())
                    .filter(|s| !s.is_empty())
                    .unwrap_or("artifact.bin");
                root.join(filename)
            }
        };
    }
}

fn qwen_asr_default(model_dir: &Path) -> ModelSpec {
    let id = "qwen3-asr-0.6b-onnx";
    let root = model_dir.join(id);
    let mut metadata = BTreeMap::new();
    metadata.insert("license".to_string(), json!("Apache-2.0"));
    metadata.insert("runtime".to_string(), json!("onnxruntime"));
    metadata.insert("model_family".to_string(), json!("qwen3_asr"));
    metadata.insert("task".to_string(), json!("automatic-speech-recognition"));
    metadata.insert(
        "hf_current_sha".to_string(),
        json!("4fc24a1402e74db89c4d2ef256875e71680128c4"),
    );
    metadata.insert(
        "mvp_status".to_string(),
        json!("Real CPU ORT encoder/decoder/tokenizer path is implemented; real INT4 execution depends on ORT contrib MatMulNBits support and should be verified with the LCOAL_QWEN_ASR_MODEL_DIR-gated smoke test"),
    );
    ModelSpec {
        id: id.to_string(),
        name: "Qwen3 ASR 0.6B ONNX INT4".to_string(),
        enabled: true,
        task_kinds: vec![TaskKind::AsrTranscribe],
        adapter: AdapterKind::QwenAsr,
        backend: BackendKind::Ort,
        artifacts: vec![ModelArtifact {
            kind: ArtifactKind::HuggingFace,
            path: root,
            source_path: None,
            sha256: None,
            url: None,
            repo_id: Some("andrewleech/qwen3-asr-0.6b-onnx".to_string()),
            revision: Some("4fc24a1402e74db89c4d2ef256875e71680128c4".to_string()),
            files: [
                "encoder.int4.onnx",
                "decoder_init.int4.onnx",
                "decoder_step.int4.onnx",
                "decoder_weights.int4.data",
                "embed_tokens.bin",
                "tokenizer.json",
                "config.json",
                "preprocessor_config.json",
                "tokenizer_config.json",
                "added_tokens.json",
                "vocab.json",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
            allow_patterns: Vec::new(),
            metadata: BTreeMap::new(),
        }],
        runtime: RuntimePolicy {
            provider_order: vec!["cpu".to_string()],
            max_concurrency: 1,
            idle_ttl_sec: 300,
        },
        resources: ResourceRequirement {
            min_ram_mb: 4096,
            min_vram_mb: 0,
        },
        load_policy: LoadPolicy::default(),
        metadata,
    }
}

fn yolo_default(model_dir: &Path) -> ModelSpec {
    let id = "yolo11n.onnx";
    let root = model_dir.join(id);
    let mut metadata = BTreeMap::new();
    metadata.insert("runtime".to_string(), json!("onnxruntime"));
    metadata.insert(
        "hf_current_sha".to_string(),
        json!("f46d9b72aa9a0f02bc00484446e2310b1a549bce"),
    );
    metadata.insert("used_storage_bytes".to_string(), json!(10_700_000));
    metadata.insert(
        "provider_note".to_string(),
        json!("CPU is the default provider; CUDA is optional only when the backend feature/runtime supports it"),
    );
    metadata.insert(
        "labels_source".to_string(),
        json!("COCO labels are downloaded from Ultralytics raw GitHub URL; labels are not in the HF model repo"),
    );
    ModelSpec {
        id: id.to_string(),
        name: "YOLO11n ONNX".to_string(),
        enabled: true,
        task_kinds: vec![TaskKind::ObjectDetect],
        adapter: AdapterKind::Yolo,
        backend: BackendKind::Ort,
        artifacts: vec![
            ModelArtifact {
                kind: ArtifactKind::HuggingFace,
                path: root.join("yolo11n.onnx"),
                source_path: None,
                sha256: None,
                url: None,
                repo_id: Some("aaurelions/yolo11n.onnx".to_string()),
                revision: Some("f46d9b72aa9a0f02bc00484446e2310b1a549bce".to_string()),
                files: vec!["yolo11n.onnx".to_string()],
                allow_patterns: Vec::new(),
                metadata: BTreeMap::new(),
            },
            ModelArtifact {
                kind: ArtifactKind::Url,
                path: root.join("coco.yaml"),
                source_path: None,
                sha256: None,
                url: Some("https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco.yaml".to_string()),
                repo_id: None,
                revision: None,
                files: Vec::new(),
                allow_patterns: Vec::new(),
                metadata: BTreeMap::new(),
            },
        ],
        runtime: RuntimePolicy {
            provider_order: vec!["cpu".to_string()],
            max_concurrency: 4,
            idle_ttl_sec: 600,
        },
        resources: ResourceRequirement {
            min_ram_mb: 1024,
            min_vram_mb: 0,
        },
        load_policy: LoadPolicy::default(),
        metadata,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[tokio::test]
    async fn loads_model_yaml() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut f = fs::File::create(dir.path().join("m.yaml")).expect("create file");
        writeln!(f, "id: test\nname: Test\nenabled: true\ntask_kinds: [object.detect]\nadapter: yolo\nbackend: ort\n").expect("write");
        let registry = ModelRegistry::load_dir(dir.path())
            .await
            .expect("load registry");
        assert_eq!(registry.list().await.len(), 1);
        assert_eq!(
            registry
                .enabled_for_task(TaskKind::ObjectDetect)
                .await
                .len(),
            1
        );
    }

    #[test]
    fn default_catalog_keeps_tts_disabled_and_hugging_face() {
        let dir = tempfile::tempdir().expect("tempdir");
        let specs = default_catalog(dir.path());

        assert!(!specs.is_empty());
        for spec in &specs {
            assert_eq!(
                spec.runtime.provider_order.first().map(String::as_str),
                Some("cpu"),
                "{} must keep CPU as the primary provider",
                spec.id
            );
            if spec.task_kinds.contains(&TaskKind::TtsSynthesize) {
                assert!(!spec.enabled, "{} TTS must be disabled by default", spec.id);
                assert_eq!(spec.adapter, AdapterKind::IndexTts);
                assert!(spec
                    .artifacts
                    .iter()
                    .all(|artifact| artifact.kind == ArtifactKind::HuggingFace));
                assert!(spec
                    .artifacts
                    .iter()
                    .all(|artifact| artifact.repo_id.as_deref()
                        == Some("ModaLeap/indextts-1.5-onnx")));
            }
        }
    }

    #[test]
    fn checked_in_yaml_models_are_cpu_primary_and_keep_tts_disabled() {
        let models_conf_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/models.d");
        let specs = load_yaml_specs(&models_conf_dir).expect("load checked-in YAML specs");

        assert!(!specs.is_empty());
        for spec in &specs {
            assert_eq!(
                spec.runtime.provider_order.first().map(String::as_str),
                Some("cpu"),
                "{} must keep CPU as the primary provider",
                spec.id
            );
            if spec.task_kinds.contains(&TaskKind::TtsSynthesize) {
                assert!(!spec.enabled, "{} TTS must be disabled by default", spec.id);
                assert_eq!(spec.adapter, AdapterKind::IndexTts);
                assert!(spec
                    .artifacts
                    .iter()
                    .all(|artifact| artifact.kind == ArtifactKind::HuggingFace));
                assert!(spec
                    .artifacts
                    .iter()
                    .all(|artifact| artifact.repo_id.as_deref()
                        == Some("ModaLeap/indextts-1.5-onnx")));
            }
        }
    }
}
