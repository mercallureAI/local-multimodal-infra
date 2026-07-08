use super::*;

pub(crate) const MODEL_FILENAMES: [&str; 6] = [
    "IndexTTS_A.onnx",
    "IndexTTS_B.onnx",
    "IndexTTS_C.onnx",
    "IndexTTS_D.onnx",
    "IndexTTS_E.onnx",
    "IndexTTS_F.onnx",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum IndexTtsPrecision {
    CpuFp32,
}

impl IndexTtsPrecision {
    pub(crate) fn from_spec(spec: &ModelSpec) -> Self {
        let requested = spec
            .metadata
            .get("precision")
            .and_then(Value::as_str)
            .unwrap_or("cpu-fp32");
        match requested {
            "cpu-fp32" | "fp32" | "auto" => Self::CpuFp32,
            "cpu-q4" | "q4" | "gpu-fp16" | "fp16" => {
                tracing::warn!(
                    model_id = %spec.id,
                    precision = requested,
                    "IndexTTS quantized/fp16 precision is no longer supported; using root FP32 artifacts"
                );
                Self::CpuFp32
            }
            other => {
                tracing::warn!(
                    model_id = %spec.id,
                    precision = other,
                    "unsupported IndexTTS precision; using root FP32 artifacts"
                );
                Self::CpuFp32
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct IndexTtsArtifacts {
    pub root: PathBuf,
    pub precision: IndexTtsPrecision,
    pub a: PathBuf,
    pub b: PathBuf,
    pub c: PathBuf,
    pub d: PathBuf,
    pub e: PathBuf,
    pub f: PathBuf,
    pub bpe_model: PathBuf,
    pub manifest: Option<PathBuf>,
}

impl IndexTtsArtifacts {
    pub fn resolve(spec: &ModelSpec) -> PathBuf {
        if let Some(path) = env::var_os("LOCAL_INDEXTTS_MODEL_DIR") {
            return PathBuf::from(path);
        }
        spec.artifacts
            .first()
            .map(|artifact| artifact.path.clone())
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| PathBuf::from("workdir/models/indextts-1.5-onnx"))
    }

    pub fn validate(root: impl AsRef<Path>, precision: IndexTtsPrecision) -> Result<Self> {
        let root = root.as_ref();
        if !root.is_dir() {
            return Err(InfraError::ModelNotConfigured {
                model_id: "index-tts".to_string(),
                reason: format!(
                    "IndexTTS artifact root is not a directory: {} (set LOCAL_INDEXTTS_MODEL_DIR or export to workdir/models/indextts-1.5-onnx)",
                    root.display()
                ),
            });
        }

        let _precision = precision;
        let model_root = root.to_path_buf();
        let paths = MODEL_FILENAMES
            .iter()
            .map(|name| model_root.join(name))
            .collect::<Vec<_>>();
        for path in &paths {
            if !path.exists() {
                return Err(InfraError::ModelNotConfigured {
                    model_id: "index-tts".to_string(),
                    reason: format!(
                        "required IndexTTS ONNX artifact is missing: {}",
                        path.display()
                    ),
                });
            }
        }
        let bpe_model = root.join("bpe.model");
        if !bpe_model.exists() {
            return Err(InfraError::ModelNotConfigured {
                model_id: "index-tts".to_string(),
                reason: format!(
                    "required IndexTTS tokenizer artifact is missing: {}",
                    bpe_model.display()
                ),
            });
        }
        let manifest = ["manifest.yaml", "manifest.yml", "manifest.json"]
            .iter()
            .map(|name| root.join(name))
            .find(|path| path.exists());

        Ok(Self {
            root: root.to_path_buf(),
            precision,
            a: paths[0].clone(),
            b: paths[1].clone(),
            c: paths[2].clone(),
            d: paths[3].clone(),
            e: paths[4].clone(),
            f: paths[5].clone(),
            bpe_model,
            manifest,
        })
    }
}
