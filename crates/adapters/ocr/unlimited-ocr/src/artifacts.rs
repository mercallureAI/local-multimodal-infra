//! Package layout written by `scripts/local/unlimited_ocr_export.py export`.

use local_backend_ort::InitializerRange;
use local_core::ModelSpec;
use local_error::{InfraError, Result};
use serde::Deserialize;
use std::{
    fs,
    path::{Path, PathBuf},
};

pub const MANIFEST_FILE: &str = "manifest.json";
pub const MANIFEST_SCHEMA: &str = "local.unlimited_ocr.package.v1";

#[derive(Debug, Clone, Deserialize)]
pub struct PackageManifest {
    pub schema: String,
    pub graphs: PackageGraphs,
    pub runtime: PackageRuntime,
    #[serde(default)]
    pub precision: PackagePrecision,
    #[serde(default)]
    pub device_shared_initializers: Option<DeviceSharedInitializers>,
}

/// Weights the prefill and decode graphs both reference: uploaded once and
/// handed to both sessions.
#[derive(Debug, Clone, Deserialize)]
pub struct DeviceSharedInitializers {
    pub data_file: String,
    pub tensors: Vec<InitializerRange>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct PackagePrecision {
    /// `int8`: CUTLASS-prepacked weights that only the CUDA `QMoE` kernel reads.
    #[serde(default)]
    pub routed_experts: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PackageGraphs {
    pub vision: String,
    pub prefill: String,
    pub decode: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct PackageRuntime {
    pub image_size: u32,
    pub image_tokens_per_page: usize,
    pub image_token_id: u32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub prompt: String,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    /// Upstream R-SWA ring: generated tokens keep at most this many KV slots.
    pub sliding_window: usize,
    pub no_repeat_ngram_size: usize,
    pub ngram_window: usize,
}

#[derive(Debug, Clone)]
pub struct UnlimitedOcrArtifacts {
    pub root: PathBuf,
    pub vision: PathBuf,
    pub prefill: PathBuf,
    pub decode: PathBuf,
    pub tokenizer: PathBuf,
    pub manifest: PackageManifest,
}

impl UnlimitedOcrArtifacts {
    pub fn from_spec(spec: &ModelSpec) -> Result<Self> {
        let root = spec
            .artifacts
            .first()
            .map(|artifact| artifact.path.clone())
            .filter(|path| !path.as_os_str().is_empty())
            .ok_or_else(|| InfraError::ModelNotConfigured {
                model_id: spec.id.clone(),
                reason: "OCR model has no artifact directory".to_string(),
            })?;
        Self::open(&spec.id, root)
    }

    pub fn open(model_id: &str, root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let missing = |name: &str| InfraError::ModelNotConfigured {
            model_id: model_id.to_string(),
            reason: format!("{name} is missing below {}", root.display()),
        };
        let manifest_path = root.join(MANIFEST_FILE);
        if !manifest_path.is_file() {
            return Err(missing(MANIFEST_FILE));
        }
        let text = fs::read_to_string(&manifest_path)
            .map_err(|e| InfraError::io(Some(manifest_path.clone()), e))?;
        let manifest: PackageManifest = serde_json::from_str(&text)
            .map_err(|e| InfraError::Adapter(format!("parse {}: {e}", manifest_path.display())))?;
        if manifest.schema != MANIFEST_SCHEMA {
            return Err(InfraError::Adapter(format!(
                "{} has schema `{}`, expected `{MANIFEST_SCHEMA}`; re-export with scripts/local/unlimited_ocr_export.py",
                manifest_path.display(),
                manifest.schema
            )));
        }
        let vision = root.join(&manifest.graphs.vision);
        let prefill = root.join(&manifest.graphs.prefill);
        let decode = root.join(&manifest.graphs.decode);
        let tokenizer = root.join("tokenizer.json");
        let shared_data = manifest
            .device_shared_initializers
            .as_ref()
            .map(|shared| root.join(&shared.data_file));
        for path in [&vision, &prefill, &decode, &tokenizer]
            .into_iter()
            .chain(shared_data.as_ref())
        {
            if !path.is_file() {
                return Err(missing(
                    &path.file_name().unwrap_or_default().to_string_lossy(),
                ));
            }
        }
        Ok(Self {
            root,
            vision,
            prefill,
            decode,
            tokenizer,
            manifest,
        })
    }
}
