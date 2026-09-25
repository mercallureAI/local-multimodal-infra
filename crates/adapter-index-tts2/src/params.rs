//! Request parameters for IndexTTS-2.5 synthesis (task `params`).

use crate::frontend::DEFAULT_MAX_TEXT_TOKENS_PER_SEGMENT;
use local_error::{InfraError, Result};
use serde_json::Value;
use std::collections::BTreeMap;

/// Emotion order of the 8-way vector, as upstream `QwenEmotion` emits it.
pub const EMOTION_NAMES: [&str; 8] = [
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "melancholic",
    "surprised",
    "calm",
];

#[derive(Debug, Clone, PartialEq)]
pub struct SynthesisParams {
    pub language: String,
    pub emotion_vector: Option<[f32; 8]>,
    pub emotion_alpha: f32,
    pub duration_factor: f32,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub cfg_rate: f32,
    pub diffusion_temperature: f32,
    pub max_mel_tokens: usize,
    pub max_text_tokens_per_segment: usize,
    pub interval_silence_ms: u64,
    pub text_normalization: bool,
    pub seed: u64,
}

impl SynthesisParams {
    pub fn from_map(params: &BTreeMap<String, Value>, text: &str) -> Result<Self> {
        let language = string(params, &["language", "lang"])?
            .map(|value| value.to_lowercase())
            .unwrap_or_else(|| detect_language(text).to_string());
        let duration_factor = match (number(params, &["duration_factor"])?, number(params, &["speed"])?) {
            (Some(factor), _) => factor,
            // OpenAI-style speed: 2.0 speaks twice as fast, i.e. half the duration.
            (None, Some(speed)) if speed > 0.0 => 1.0 / speed,
            (None, Some(speed)) => {
                return Err(InfraError::BadRequest(format!("speed must be positive, got {speed}")))
            }
            (None, None) => 1.0,
        };
        let parsed = Self {
            language,
            emotion_vector: emotion_vector(params)?,
            emotion_alpha: number(params, &["emotion_alpha", "emo_alpha"])?.unwrap_or(1.0),
            duration_factor,
            temperature: number(params, &["temperature"])?.unwrap_or(0.8),
            top_k: integer(params, &["top_k"])?.unwrap_or(20) as usize,
            top_p: number(params, &["top_p"])?.unwrap_or(0.9),
            repetition_penalty: number(params, &["repetition_penalty"])?.unwrap_or(1.2),
            cfg_rate: number(params, &["cfg_rate"])?.unwrap_or(0.7),
            diffusion_temperature: number(params, &["diffusion_temperature"])?.unwrap_or(1.0),
            max_mel_tokens: integer(params, &["max_mel_tokens"])?.unwrap_or(1500) as usize,
            max_text_tokens_per_segment: integer(params, &["max_text_tokens_per_segment"])?
                .unwrap_or(DEFAULT_MAX_TEXT_TOKENS_PER_SEGMENT as u64)
                as usize,
            interval_silence_ms: integer(params, &["interval_silence_ms"])?.unwrap_or(200),
            text_normalization: boolean(params, &["text_normalization"])?.unwrap_or(true),
            seed: integer(params, &["seed"])?.unwrap_or_else(time_seed),
        };
        parsed.validate()?;
        Ok(parsed)
    }

    fn validate(&self) -> Result<()> {
        let bad = |message: String| Err(InfraError::BadRequest(message));
        if !(0.25..=4.0).contains(&self.duration_factor) {
            return bad(format!("duration_factor must be in [0.25, 4], got {}", self.duration_factor));
        }
        if !(self.temperature > 0.0 && self.temperature <= 2.0) {
            return bad(format!("temperature must be in (0, 2], got {}", self.temperature));
        }
        if !(self.top_p > 0.0 && self.top_p <= 1.0) {
            return bad(format!("top_p must be in (0, 1], got {}", self.top_p));
        }
        if self.top_k == 0 {
            return bad("top_k must be positive".to_string());
        }
        if self.max_mel_tokens == 0 || self.max_text_tokens_per_segment == 0 {
            return bad("max_mel_tokens and max_text_tokens_per_segment must be positive".to_string());
        }
        if let Some(vector) = self.emotion_vector {
            if vector.iter().any(|value| !value.is_finite() || *value < 0.0) {
                return bad(format!("emotion_vector values must be finite and >= 0, got {vector:?}"));
            }
        }
        Ok(())
    }
}

/// `emotion_vector` as 8 numbers in [`EMOTION_NAMES`] order, or an object
/// keyed by those names (missing names are 0).
fn emotion_vector(params: &BTreeMap<String, Value>) -> Result<Option<[f32; 8]>> {
    let Some(value) = ["emotion_vector", "emo_vector"].iter().find_map(|key| params.get(*key)) else {
        return Ok(None);
    };
    let mut vector = [0.0f32; 8];
    match value {
        Value::Null => return Ok(None),
        Value::Array(items) if items.len() == 8 => {
            for (slot, item) in vector.iter_mut().zip(items) {
                *slot = item.as_f64().ok_or_else(|| {
                    InfraError::BadRequest(format!("emotion_vector entries must be numbers, got {item}"))
                })? as f32;
            }
        }
        Value::Object(map) => {
            for (name, item) in map {
                let index = EMOTION_NAMES
                    .iter()
                    .position(|candidate| candidate.eq_ignore_ascii_case(name))
                    .ok_or_else(|| {
                        InfraError::BadRequest(format!(
                            "unknown emotion `{name}`; expected one of {EMOTION_NAMES:?}"
                        ))
                    })?;
                vector[index] = item.as_f64().ok_or_else(|| {
                    InfraError::BadRequest(format!("emotion `{name}` must be a number, got {item}"))
                })? as f32;
            }
        }
        other => {
            return Err(InfraError::BadRequest(format!(
                "emotion_vector must be 8 numbers or an object keyed by {EMOTION_NAMES:?}, got {other}"
            )))
        }
    }
    Ok(Some(vector))
}

/// zh when the text contains CJK ideographs, otherwise en.
pub fn detect_language(text: &str) -> &'static str {
    if text.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c)) {
        "zh"
    } else {
        "en"
    }
}

fn first<'a>(params: &'a BTreeMap<String, Value>, keys: &[&str]) -> Option<(&'a str, &'a Value)> {
    keys.iter()
        .find_map(|key| params.get_key_value(*key))
        .filter(|(_, value)| !value.is_null())
        .map(|(key, value)| (key.as_str(), value))
}

fn number(params: &BTreeMap<String, Value>, keys: &[&str]) -> Result<Option<f32>> {
    first(params, keys)
        .map(|(key, value)| {
            value.as_f64().map(|v| v as f32).ok_or_else(|| {
                InfraError::BadRequest(format!("`{key}` must be a number, got {value}"))
            })
        })
        .transpose()
}

fn integer(params: &BTreeMap<String, Value>, keys: &[&str]) -> Result<Option<u64>> {
    first(params, keys)
        .map(|(key, value)| {
            value.as_u64().ok_or_else(|| {
                InfraError::BadRequest(format!("`{key}` must be a non-negative integer, got {value}"))
            })
        })
        .transpose()
}

fn boolean(params: &BTreeMap<String, Value>, keys: &[&str]) -> Result<Option<bool>> {
    first(params, keys)
        .map(|(key, value)| {
            value.as_bool().ok_or_else(|| {
                InfraError::BadRequest(format!("`{key}` must be a boolean, got {value}"))
            })
        })
        .transpose()
}

fn string(params: &BTreeMap<String, Value>, keys: &[&str]) -> Result<Option<String>> {
    first(params, keys)
        .map(|(key, value)| {
            value.as_str().map(str::to_string).ok_or_else(|| {
                InfraError::BadRequest(format!("`{key}` must be a string, got {value}"))
            })
        })
        .transpose()
}

fn time_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn map(value: Value) -> BTreeMap<String, Value> {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn defaults_follow_the_upstream_onnx_runtime() {
        let params = SynthesisParams::from_map(&map(json!({"seed": 1})), "你好").unwrap();
        assert_eq!(params.language, "zh");
        assert_eq!(params.emotion_vector, None);
        assert_eq!((params.top_k, params.top_p, params.cfg_rate), (20, 0.9, 0.7));
        assert_eq!(SynthesisParams::from_map(&map(json!({})), "hello").unwrap().language, "en");
    }

    #[test]
    fn emotion_vector_accepts_list_or_named_object() {
        let list = SynthesisParams::from_map(
            &map(json!({"emotion_vector": [0, 0, 0.8, 0, 0, 0, 0, 0]})),
            "x",
        )
        .unwrap();
        let named = SynthesisParams::from_map(&map(json!({"emotion_vector": {"sad": 0.8}})), "x").unwrap();
        assert_eq!(list.emotion_vector, named.emotion_vector);
        assert!(SynthesisParams::from_map(&map(json!({"emotion_vector": {"bored": 1}})), "x").is_err());
        assert!(SynthesisParams::from_map(&map(json!({"emotion_vector": [1, 2]})), "x").is_err());
    }

    #[test]
    fn speed_maps_to_inverse_duration() {
        let params = SynthesisParams::from_map(&map(json!({"speed": 2.0})), "x").unwrap();
        assert_eq!(params.duration_factor, 0.5);
        assert!(SynthesisParams::from_map(&map(json!({"speed": 0})), "x").is_err());
    }
}
