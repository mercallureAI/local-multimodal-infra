//! Request parameters for Qwen3-TTS synthesis (task `params`).

use crate::artifacts::{GenerationDefaults, PackageConfig};
use local_error::{InfraError, Result};
use serde_json::Value;
use std::collections::BTreeMap;

/// Longest reference for in-context cloning unless `max_reference_seconds`
/// says otherwise (at most [`MAX_REFERENCE_SECONDS`]).
pub const DEFAULT_MAX_REFERENCE_SECONDS: f32 = 15.0;
pub const MAX_REFERENCE_SECONDS: f32 = 30.0;

#[derive(Debug, Clone, PartialEq)]
pub struct SynthesisParams {
    /// A codec language (`chinese`, `english`, ...) or `None` for auto.
    pub language: Option<String>,
    /// Transcript of the reference audio: in-context cloning when set, the
    /// x-vector alone otherwise.
    pub reference_text: Option<String>,
    pub do_sample: bool,
    pub temperature: f32,
    pub subtalker_temperature: f32,
    pub repetition_penalty: f32,
    pub max_frames: usize,
    pub max_reference_seconds: f32,
    pub seed: u64,
}

impl SynthesisParams {
    pub fn from_map(params: &BTreeMap<String, Value>, config: &PackageConfig) -> Result<Self> {
        let defaults: GenerationDefaults = config.generation;
        let language = match string(params, &["language", "lang"])? {
            None => None,
            Some(language) => {
                let language = language.to_lowercase();
                match language.as_str() {
                    "auto" | "" => None,
                    "zh" => Some("chinese".to_string()),
                    "en" => Some("english".to_string()),
                    _ if config.languages.contains_key(&language) => Some(language),
                    _ => {
                        return Err(InfraError::BadRequest(format!(
                            "language `{language}` is none of auto, {}",
                            config
                                .languages
                                .keys()
                                .cloned()
                                .collect::<Vec<_>>()
                                .join(", ")
                        )))
                    }
                }
            }
        };
        if let Some(top_k) = integer(params, &["top_k", "subtalker_top_k"])? {
            if top_k as usize != config.top_k {
                return Err(InfraError::BadRequest(format!(
                    "top_k is fixed at {} in this package",
                    config.top_k
                )));
            }
        }
        let parsed = Self {
            language,
            reference_text: string(params, &["reference_text", "ref_text", "prompt_text"])?
                .filter(|text| !text.trim().is_empty()),
            do_sample: boolean(params, &["do_sample"])?.unwrap_or(defaults.do_sample),
            temperature: number(params, &["temperature"])?.unwrap_or(defaults.temperature),
            subtalker_temperature: number(params, &["subtalker_temperature"])?
                .unwrap_or(defaults.subtalker_temperature),
            repetition_penalty: number(params, &["repetition_penalty"])?
                .unwrap_or(defaults.repetition_penalty),
            max_frames: integer(params, &["max_frames", "max_new_tokens"])?.unwrap_or(1500)
                as usize,
            max_reference_seconds: number(params, &["max_reference_seconds"])?
                .unwrap_or(DEFAULT_MAX_REFERENCE_SECONDS),
            seed: integer(params, &["seed"])?.unwrap_or_else(time_seed),
        };
        parsed.validate()?;
        Ok(parsed)
    }

    fn validate(&self) -> Result<()> {
        let bad = |message: String| Err(InfraError::BadRequest(message));
        if !(self.temperature > 0.0 && self.subtalker_temperature > 0.0) {
            return bad("temperatures must be positive".to_string());
        }
        if !(1.0..=2.0).contains(&self.repetition_penalty) {
            return bad(format!(
                "repetition_penalty must be in [1, 2], got {}",
                self.repetition_penalty
            ));
        }
        if self.max_frames == 0 {
            return bad("max_frames must be positive".to_string());
        }
        if !(1.0..=MAX_REFERENCE_SECONDS).contains(&self.max_reference_seconds) {
            return bad(format!(
                "max_reference_seconds must be in [1, {MAX_REFERENCE_SECONDS}], got {}",
                self.max_reference_seconds
            ));
        }
        Ok(())
    }
}

fn time_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x5EED)
}

fn lookup<'a>(params: &'a BTreeMap<String, Value>, keys: &[&str]) -> Option<(&'a str, &'a Value)> {
    keys.iter()
        .find_map(|key| params.get_key_value(*key))
        .map(|(key, value)| (key.as_str(), value))
        .filter(|(_, value)| !value.is_null())
}

fn string(params: &BTreeMap<String, Value>, keys: &[&str]) -> Result<Option<String>> {
    match lookup(params, keys) {
        None => Ok(None),
        Some((_, Value::String(value))) => Ok(Some(value.clone())),
        Some((key, other)) => Err(InfraError::BadRequest(format!(
            "{key} must be a string, got {other}"
        ))),
    }
}

fn number(params: &BTreeMap<String, Value>, keys: &[&str]) -> Result<Option<f32>> {
    match lookup(params, keys) {
        None => Ok(None),
        Some((key, value)) => value
            .as_f64()
            .map(|value| Some(value as f32))
            .ok_or_else(|| InfraError::BadRequest(format!("{key} must be a number, got {value}"))),
    }
}

fn integer(params: &BTreeMap<String, Value>, keys: &[&str]) -> Result<Option<u64>> {
    match lookup(params, keys) {
        None => Ok(None),
        Some((key, value)) => value.as_u64().map(Some).ok_or_else(|| {
            InfraError::BadRequest(format!("{key} must be a non-negative integer, got {value}"))
        }),
    }
}

fn boolean(params: &BTreeMap<String, Value>, keys: &[&str]) -> Result<Option<bool>> {
    match lookup(params, keys) {
        None => Ok(None),
        Some((_, Value::Bool(value))) => Ok(Some(*value)),
        Some((key, other)) => Err(InfraError::BadRequest(format!(
            "{key} must be a boolean, got {other}"
        ))),
    }
}
