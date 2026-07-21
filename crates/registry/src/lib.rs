use local_core::{
    AdapterKind, ArtifactKind, BackendKind, LoadPolicy, ModelArtifact, ModelSpec,
    ResourceRequirement, RuntimePolicy, TaskKind,
};
use local_error::{InfraError, Result};
use serde_json::json;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};
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
        sensevoice_asr_default(model_dir),
        yolo_default(model_dir),
        index_tts_default(model_dir),
        e5_embedding_default(model_dir),
        mmarco_reranker_default(model_dir),
    ]
}

fn e5_embedding_default(model_dir: &Path) -> ModelSpec {
    let id = "multilingual-e5-small-onnx";
    let root = model_dir.join(id);
    let revision = "614241f622f53c4eeff9890bdc4f31cfecc418b3";
    let mut metadata = BTreeMap::new();
    metadata.insert("runtime".to_string(), json!("onnxruntime"));
    metadata.insert("model_family".to_string(), json!("multilingual_e5"));
    metadata.insert("task".to_string(), json!("feature-extraction"));
    metadata.insert("embedding_dimension".to_string(), json!(384));
    metadata.insert("max_length".to_string(), json!(512));
    metadata.insert("hf_current_sha".to_string(), json!(revision));
    metadata.insert(
        "artifact_policy".to_string(),
        json!("CUDA prefers the derived O4 pooled graph; CPU prefers the derived qint8 pooled graph; official upstream graphs remain compatibility fallbacks"),
    );
    metadata.insert(
        "pooling_export".to_string(),
        json!("uv run --with onnx --python 3.12 python -m scripts.local.e5_pooling_export --model-dir workdir/models"),
    );
    ModelSpec {
        id: id.to_string(),
        name: "multilingual-e5-small ONNX".to_string(),
        enabled: true,
        task_kinds: vec![TaskKind::TextEmbed],
        adapter: AdapterKind::E5Embedding,
        backend: BackendKind::Ort,
        artifacts: vec![ModelArtifact {
            kind: ArtifactKind::HuggingFace,
            path: root,
            source_path: None,
            sha256: None,
            url: None,
            repo_id: Some("intfloat/multilingual-e5-small".to_string()),
            revision: Some(revision.to_string()),
            files: [
                "onnx/model_O4.onnx",
                "onnx/model_qint8_avx512_vnni.onnx",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "config.json",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            allow_patterns: Vec::new(),
            metadata: BTreeMap::new(),
        }],
        runtime: RuntimePolicy {
            provider_order: vec!["cuda".to_string(), "cpu".to_string()],
            max_concurrency: 1,
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

fn mmarco_reranker_default(model_dir: &Path) -> ModelSpec {
    let id = "mmarco-minilm-l12-onnx";
    let root = model_dir.join(id);
    let revision = "1427fd652930e4ba29e8149678df786c240d8825";
    let mut metadata = BTreeMap::new();
    metadata.insert("runtime".to_string(), json!("onnxruntime"));
    metadata.insert("model_family".to_string(), json!("mmarco_minilm"));
    metadata.insert("task".to_string(), json!("text-ranking"));
    metadata.insert("max_length".to_string(), json!(512));
    metadata.insert("hf_current_sha".to_string(), json!(revision));
    metadata.insert(
        "artifact_policy".to_string(),
        json!("CUDA uses the official O4 graph; x86 CPU uses the official AVX2 uint8 graph"),
    );
    ModelSpec {
        id: id.to_string(),
        name: "mMARCO MiniLM L12 ONNX reranker".to_string(),
        enabled: true,
        task_kinds: vec![TaskKind::TextRerank],
        adapter: AdapterKind::MmarcoReranker,
        backend: BackendKind::Ort,
        artifacts: vec![ModelArtifact {
            kind: ArtifactKind::HuggingFace,
            path: root,
            source_path: None,
            sha256: None,
            url: None,
            repo_id: Some("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1".to_string()),
            revision: Some(revision.to_string()),
            files: [
                "onnx/model_O4.onnx",
                "onnx/model_quint8_avx2.onnx",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "config.json",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            allow_patterns: Vec::new(),
            metadata: BTreeMap::new(),
        }],
        runtime: RuntimePolicy {
            provider_order: vec!["cuda".to_string(), "cpu".to_string()],
            max_concurrency: 1,
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

fn index_tts_default(model_dir: &Path) -> ModelSpec {
    let id = "indextts-1.5-onnx";
    let root = model_dir.join(id);
    let mut metadata = BTreeMap::new();
    metadata.insert("runtime".to_string(), json!("onnxruntime"));
    metadata.insert("model_family".to_string(), json!("index_tts"));
    metadata.insert("task".to_string(), json!("text-to-speech"));
    metadata.insert("experimental".to_string(), json!(true));
    metadata.insert("precision".to_string(), json!("fp32"));
    metadata.insert(
        "artifact_note".to_string(),
        json!("FP32 artifacts are materialized from ModaLeap/indextts-1.5-onnx or an equivalent artifact root. Runtime expects IndexTTS_A.onnx through IndexTTS_F.onnx plus bpe.model and manifest.yaml/json at the artifact root."),
    );
    metadata.insert(
        "mvp_status".to_string(),
        json!("Root FP32 A-F ORT sessions are configured for CUDA with CPU fallback. Real NVIDIA hardware execution remains unverified; q4cpu/fp16gpu paths remain withdrawn."),
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
            provider_order: vec!["cuda".to_string(), "cpu".to_string()],
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
                if let Ok(relative) = artifact.path.strip_prefix(&root) {
                    if !relative.as_os_str().is_empty() {
                        root.join(relative)
                    } else {
                        url_artifact_default_path(&root, artifact.url.as_deref())
                    }
                } else if artifact.path.is_relative() && !artifact.path.as_os_str().is_empty() {
                    root.join(&artifact.path)
                } else {
                    url_artifact_default_path(&root, artifact.url.as_deref())
                }
            }
        };
    }
}

fn url_artifact_default_path(root: &Path, url: Option<&str>) -> PathBuf {
    let filename = url
        .and_then(|url| url.rsplit('/').next())
        .filter(|s| !s.is_empty())
        .unwrap_or("artifact.bin");
    root.join(filename)
}

fn sensevoice_asr_default(model_dir: &Path) -> ModelSpec {
    let id = "sensevoice-small-onnx";
    let root = model_dir.join(id);
    let sensevoice_revision = "c4c8747214bed7ebbf2557e0412c19efa540023c";
    let vad_revision = "f6e9fbb4cefa7397216c763f21307993f147f585";
    let vad_config_revision = "58fbad4088820ed1253955c8faf1444cd0b2dc69";
    let campplus_revision = "6265ff7af2a104d745b4389026ed9815c6c1c6ff";
    let artifact = |target: &str, url: String, sha256: &str, revision: &str| ModelArtifact {
        kind: ArtifactKind::Url,
        path: root.join(target),
        source_path: None,
        sha256: Some(sha256.to_string()),
        url: Some(url),
        repo_id: None,
        revision: Some(revision.to_string()),
        files: Vec::new(),
        allow_patterns: Vec::new(),
        metadata: BTreeMap::new(),
    };
    let mut metadata = BTreeMap::new();
    metadata.insert("license".to_string(), json!("Apache-2.0"));
    metadata.insert("runtime".to_string(), json!("onnxruntime"));
    metadata.insert("model_family".to_string(), json!("funasr_full_asr"));
    metadata.insert("task".to_string(), json!("automatic-speech-recognition"));
    metadata.insert(
        "source".to_string(),
        json!([
            "haixuantao/SenseVoiceSmall-onnx",
            "funasr/fsmn-vad-onnx",
            "MoYoYoTech/Translator",
            "3D-Speaker/CAM++"
        ]),
    );
    metadata.insert(
        "huggingface_revision".to_string(),
        json!(sensevoice_revision),
    );
    metadata.insert("vad_huggingface_revision".to_string(), json!(vad_revision));
    metadata.insert(
        "vad_config_huggingface_revision".to_string(),
        json!(vad_config_revision),
    );
    metadata.insert("speaker_diarization_default".to_string(), json!(true));
    metadata.insert("timestamps_default".to_string(), json!(true));
    metadata.insert("timestamp_granularity_sec".to_string(), json!(10));
    metadata.insert("token_timestamps_default".to_string(), json!(false));
    metadata.insert("sensevoice_max_chunk_seconds".to_string(), json!(30));
    metadata.insert("vad_max_segment_seconds".to_string(), json!(20));
    metadata.insert(
        "mvp_status".to_string(),
        json!("Full FunASR-style pipeline: FSMN-VAD, SenseVoiceSmall ONNX, configurable ~10s timeline segments, optional token timestamps, CAM++ embeddings, and speaker clustering; diarization and timeline output are enabled by default"),
    );
    ModelSpec {
        id: id.to_string(),
        name: "FunASR Full ASR ONNX".to_string(),
        enabled: true,
        task_kinds: vec![TaskKind::AsrTranscribe],
        adapter: AdapterKind::SenseVoiceAsr,
        backend: BackendKind::Ort,
        artifacts: vec![
            artifact(
                "asr/model_quant.onnx",
                format!("https://huggingface.co/haixuantao/SenseVoiceSmall-onnx/resolve/{sensevoice_revision}/model_quant.onnx?download=true"),
                "21dc965f689a78d1604717bf561e40d5a236087c85a95584567835750549e822",
                sensevoice_revision,
            ),
            artifact(
                "asr/am.mvn",
                format!("https://huggingface.co/haixuantao/SenseVoiceSmall-onnx/resolve/{sensevoice_revision}/am.mvn?download=true"),
                "29b3c740a2c0cfc6b308126d31d7f265fa2be74f3bb095cd2f143ea970896ae5",
                sensevoice_revision,
            ),
            artifact(
                "asr/config.yaml",
                format!("https://huggingface.co/haixuantao/SenseVoiceSmall-onnx/resolve/{sensevoice_revision}/config.yaml?download=true"),
                "f71e239ba36705564b5bf2d2ffd07eece07b8e3f2bbf6d2c99d8df856339ac19",
                sensevoice_revision,
            ),
            artifact(
                "asr/tokens.json",
                format!("https://huggingface.co/haixuantao/SenseVoiceSmall-onnx/resolve/{sensevoice_revision}/tokens.json?download=true"),
                "a2594fc1474e78973149cba8cd1f603ebed8c39c7decb470631f66e70ce58e97",
                sensevoice_revision,
            ),
            artifact(
                "vad/model_quant.onnx",
                format!("https://huggingface.co/funasr/fsmn-vad-onnx/resolve/{vad_revision}/model_quant.onnx?download=true"),
                "9b28837838fce9685503c63139fadbad35d6c8ed485485dafdbb32e725969660",
                vad_revision,
            ),
            artifact(
                "vad/am.mvn",
                format!("https://huggingface.co/funasr/fsmn-vad-onnx/resolve/{vad_revision}/vad.mvn?download=true"),
                "6820fef9687708c4fc3fab2530179c8fcea6262daa25514380056cd8f6eb1754",
                vad_revision,
            ),
            artifact(
                "vad/config.yaml",
                format!("https://huggingface.co/MoYoYoTech/Translator/resolve/{vad_config_revision}/moyoyo_asr_models/speech_fsmn_vad_zh-cn-16k-common-pytorch/config.yaml?download=true"),
                "486861ca26ddb79081663b6179cb204c6bfae71c52f04aafc48a9e9d8dde1e93",
                vad_config_revision,
            ),
            artifact(
                "speaker/campplus_cn_en_common_200k.onnx",
                format!("https://huggingface.co/welcomyou/campplus-3dspeaker-200k-onnx/resolve/{campplus_revision}/campplus_cn_en_common_200k.onnx"),
                "dd1740aa1e1ffa3895f96aef2166b8af2bb2ad09c00769dd275ee36aef6a2a7f",
                campplus_revision,
            ),
        ],
        runtime: RuntimePolicy {
            provider_order: vec!["cuda".to_string(), "cpu".to_string()],
            max_concurrency: 1,
            idle_ttl_sec: 300,
        },
        resources: ResourceRequirement {
            min_ram_mb: 1536,
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
        json!("CUDA is preferred when compiled and available; CPU is the portable fallback"),
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
            provider_order: vec!["cuda".to_string(), "cpu".to_string()],
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
    fn funasr_artifacts_keep_collision_free_subdirectories() {
        let dir = tempfile::tempdir().expect("tempdir");
        let spec = default_catalog(dir.path())
            .into_iter()
            .find(|spec| spec.id == "sensevoice-small-onnx")
            .expect("FunASR spec");
        let relative = spec
            .artifacts
            .iter()
            .map(|artifact| {
                artifact
                    .path
                    .strip_prefix(dir.path().join("sensevoice-small-onnx"))
                    .expect("model root")
                    .to_path_buf()
            })
            .collect::<Vec<_>>();
        assert!(relative.contains(&PathBuf::from("asr/model_quant.onnx")));
        assert!(relative.contains(&PathBuf::from("vad/model_quant.onnx")));
        assert_eq!(
            relative
                .iter()
                .filter(|path| path.file_name().is_some())
                .count(),
            8
        );
    }

    #[test]
    fn default_catalog_keeps_tts_disabled_and_hugging_face() {
        let dir = tempfile::tempdir().expect("tempdir");
        let specs = default_catalog(dir.path());

        assert!(!specs.is_empty());
        for spec in &specs {
            assert_eq!(
                spec.runtime
                    .provider_order
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
                ["cuda", "cpu"],
                "{} must prefer CUDA then CPU",
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
    fn checked_in_yaml_models_are_cuda_first_with_cpu_fallback() {
        let models_conf_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/models.d");
        let specs = load_yaml_specs(&models_conf_dir).expect("load checked-in YAML specs");

        assert!(!specs.is_empty());
        for spec in &specs {
            assert_eq!(
                spec.runtime
                    .provider_order
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
                ["cuda", "cpu"],
                "{} must prefer CUDA then CPU",
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
