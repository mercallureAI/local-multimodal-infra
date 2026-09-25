//! Packaged IndexTTS-2.5 layout written by `scripts/indextts2_export.py package`.

use local_core::ModelSpec;
use local_error::{InfraError, Result};
use serde::Deserialize;
use std::{
    fs,
    path::{Path, PathBuf},
};

pub const MANIFEST_FILE: &str = "manifest.json";
pub const MANIFEST_SCHEMA: &str = "local.index_tts2.package.v1";

#[derive(Debug, Clone, Deserialize)]
pub struct PackageManifest {
    pub schema: String,
    pub model_version: String,
    pub precision: String,
    pub graphs: PackageGraphs,
    pub runtime: PackageRuntime,
    #[serde(default)]
    pub disabled_optimizers: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PackageGraphs {
    pub reference_preprocess: String,
    pub conditioning: String,
    pub target_prefill_sampling: String,
    pub decode_step_sampling: String,
    pub synthesis: String,
    pub cfm_estimator: String,
    pub decoder: String,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct PackageRuntime {
    pub in_sample_rate: u32,
    pub out_sample_rate: u32,
    pub cfm_steps: usize,
    pub max_signal_length: usize,
    pub max_text_tokens: usize,
    pub mel_code_size: usize,
    pub stop_mel_token: i32,
}

#[derive(Debug, Clone)]
pub struct IndexTts2Artifacts {
    pub root: PathBuf,
    pub manifest: PackageManifest,
}

impl IndexTts2Artifacts {
    /// The first artifact path when configured, else `<model_dir>/<id>` as
    /// materialized by the registry.
    pub fn resolve(spec: &ModelSpec) -> PathBuf {
        spec.artifacts
            .iter()
            .map(|artifact| artifact.path.clone())
            .find(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| PathBuf::from(&spec.id))
    }

    pub fn load(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let not_ready = |reason: String| InfraError::ModelNotConfigured {
            model_id: "indextts-2.5".to_string(),
            reason,
        };
        if !root.is_dir() {
            return Err(not_ready(format!(
                "IndexTTS-2.5 artifact root is not a directory: {}",
                root.display()
            )));
        }
        let manifest_path = root.join(MANIFEST_FILE);
        let bytes = fs::read(&manifest_path).map_err(|e| {
            not_ready(format!(
                "{} is missing ({e}); package the export with scripts/indextts2_export.py package",
                manifest_path.display()
            ))
        })?;
        let manifest: PackageManifest = serde_json::from_slice(&bytes).map_err(|e| {
            InfraError::Adapter(format!("parse {}: {e}", manifest_path.display()))
        })?;
        if manifest.schema != MANIFEST_SCHEMA {
            return Err(not_ready(format!(
                "{} has schema `{}`, expected `{MANIFEST_SCHEMA}`",
                manifest_path.display(),
                manifest.schema
            )));
        }
        if manifest.model_version != "2.5" {
            return Err(not_ready(format!(
                "{} declares model_version {}, expected 2.5",
                manifest_path.display(),
                manifest.model_version
            )));
        }
        let artifacts = Self { root, manifest };
        for graph in artifacts.graph_paths() {
            if !graph.is_file() {
                return Err(not_ready(format!(
                    "IndexTTS-2.5 graph is missing: {}",
                    graph.display()
                )));
            }
        }
        Ok(artifacts)
    }

    pub fn graph(&self, file: &str) -> PathBuf {
        self.root.join(file)
    }

    fn graph_paths(&self) -> [PathBuf; 7] {
        let graphs = &self.manifest.graphs;
        [
            &graphs.reference_preprocess,
            &graphs.conditioning,
            &graphs.target_prefill_sampling,
            &graphs.decode_step_sampling,
            &graphs.synthesis,
            &graphs.cfm_estimator,
            &graphs.decoder,
        ]
        .map(|file| self.graph(file))
    }
}
