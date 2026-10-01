//! The package layout written by `scripts/local/qwen3_tts_export.py`.

use local_core::ModelSpec;
use local_error::{InfraError, Result};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

pub const CONFIG_FILE: &str = "config.json";
pub const TOKENIZER_FILE: &str = "tokenizer.json";
pub const PACKAGE_SCHEMA: &str = "local.qwen3_tts.package.v1";

#[derive(Debug, Clone, Deserialize)]
pub struct PackageConfig {
    pub schema: String,
    pub hidden_size: usize,
    pub talker_layers: usize,
    pub talker_kv_heads: usize,
    pub head_dim: usize,
    pub talker_vocab_size: usize,
    pub max_position_embeddings: usize,
    pub num_code_groups: usize,
    pub codebook_size: usize,
    /// Top-k of both samplers, fixed in the talker graph.
    pub top_k: usize,
    pub sample_rate: u32,
    pub samples_per_frame: usize,
    pub tokens: SpecialTokens,
    /// Codec language tokens by lower-case language name.
    pub languages: BTreeMap<String, i64>,
    pub graphs: Graphs,
    pub precision: String,
    pub vocoder: VocoderConfig,
    pub codec_frame_samples: usize,
    pub generation: GenerationDefaults,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct SpecialTokens {
    pub tts_bos: i64,
    pub tts_eos: i64,
    pub tts_pad: i64,
    pub codec_bos: i64,
    pub codec_eos: i64,
    pub codec_pad: i64,
    pub codec_think: i64,
    pub codec_nothink: i64,
    pub codec_think_bos: i64,
    pub codec_think_eos: i64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Graphs {
    pub talker: String,
    pub vocoder: String,
    pub speaker_encoder: String,
    pub codec_encoder: String,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct VocoderConfig {
    pub layers: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub kv_window: usize,
    pub pre_ctx_channels: usize,
    pub pre_ctx_frames: usize,
    pub conv_ctx_channels: usize,
    pub conv_ctx_frames: usize,
    /// Frames the vocoder's rotary table covers: the longest stream.
    pub max_frames: usize,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct GenerationDefaults {
    pub do_sample: bool,
    pub temperature: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub subtalker_temperature: f32,
    pub subtalker_top_k: usize,
    pub min_new_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct Qwen3TtsArtifacts {
    pub root: PathBuf,
    pub config: PackageConfig,
}

impl Qwen3TtsArtifacts {
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
            model_id: "qwen3-tts".to_string(),
            reason,
        };
        if !root.is_dir() {
            return Err(not_ready(format!(
                "Qwen3-TTS artifact root is not a directory: {}",
                root.display()
            )));
        }
        let path = root.join(CONFIG_FILE);
        let bytes = fs::read(&path).map_err(|e| {
            not_ready(format!(
                "{} is missing ({e}); export it with scripts/local/qwen3_tts_export.py",
                path.display()
            ))
        })?;
        let config: PackageConfig = serde_json::from_slice(&bytes)
            .map_err(|e| InfraError::Adapter(format!("parse {}: {e}", path.display())))?;
        if config.schema != PACKAGE_SCHEMA {
            return Err(not_ready(format!(
                "{} has schema `{}`, expected `{PACKAGE_SCHEMA}`",
                path.display(),
                config.schema
            )));
        }
        let artifacts = Self { root, config };
        for file in artifacts.files() {
            if !file.is_file() {
                return Err(not_ready(format!(
                    "Qwen3-TTS file is missing: {}",
                    file.display()
                )));
            }
        }
        Ok(artifacts)
    }

    pub fn file(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    fn files(&self) -> [PathBuf; 5] {
        let graphs = &self.config.graphs;
        [
            self.file(&graphs.talker),
            self.file(&graphs.vocoder),
            self.file(&graphs.speaker_encoder),
            self.file(&graphs.codec_encoder),
            self.file(TOKENIZER_FILE),
        ]
    }
}
