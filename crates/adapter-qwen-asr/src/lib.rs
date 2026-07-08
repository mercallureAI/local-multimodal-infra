//! Qwen3 ASR adapter provenance:
//! - ONNX artifact layout follows https://huggingface.co/andrewleech/qwen3-asr-0.6b-onnx
//! - Original upstream model card: https://huggingface.co/Qwen/Qwen3-ASR-0.6B
//! - This crate is rewritten/adapted inside this project for ONNX inference and does not
//!   directly depend on or vendor the upstream project.

use local_backend_ort::{
    OrtBackend, OrtOutput, OrtSession, OrtTensorData, OrtTensorInput, OrtTensorOutput,
    ProviderSelection, SessionMetadata, SessionProviderReport, TensorMetadata,
};
use local_core::{FileRef, InferenceOutput, ModelSpec};
use local_error::{InfraError, Result};
use serde_json::Value;
use std::{
    fs,
    path::{Path, PathBuf},
};

pub const EOS_TOKEN_IDS: [u32; 2] = [151_643, 151_645];
pub const IM_START_TOKEN_ID: u32 = 151_644;
pub const IM_END_TOKEN_ID: u32 = 151_645;
pub const AUDIO_START_TOKEN_ID: u32 = 151_669;
pub const AUDIO_END_TOKEN_ID: u32 = 151_670;
pub const AUDIO_PAD_TOKEN_ID: u32 = 151_676;
pub const NEWLINE_TOKEN_ID: u32 = 198;
const DEFAULT_HIDDEN_SIZE: usize = 1024;
const DEFAULT_VOCAB_SIZE: usize = 151_936;
const MAX_NEW_TOKENS: usize = 448;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenArtifactVariant {
    Int4,
    Fp32,
}

#[derive(Debug, Clone)]
pub struct QwenAsrArtifacts {
    pub root: PathBuf,
    pub variant: QwenArtifactVariant,
    pub encoder: PathBuf,
    pub decoder_init: PathBuf,
    pub decoder_step: PathBuf,
    pub decoder_weights: Option<PathBuf>,
    pub embed_tokens: PathBuf,
    pub tokenizer: PathBuf,
    pub config: PathBuf,
    pub preprocessor_config: PathBuf,
}

impl QwenAsrArtifacts {
    pub fn validate(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        if !root.is_dir() {
            return Err(InfraError::ModelNotConfigured {
                model_id: "qwen-asr".to_string(),
                reason: format!(
                    "Qwen ASR artifact root is not a directory: {}",
                    root.display()
                ),
            });
        }
        let int4 = root.join("encoder.int4.onnx");
        let (variant, encoder, decoder_init, decoder_step, decoder_weights) = if int4.exists() {
            (
                QwenArtifactVariant::Int4,
                int4,
                root.join("decoder_init.int4.onnx"),
                root.join("decoder_step.int4.onnx"),
                optional_file(root.join("decoder_weights.int4.data")),
            )
        } else {
            (
                QwenArtifactVariant::Fp32,
                root.join("encoder.onnx"),
                root.join("decoder_init.onnx"),
                root.join("decoder_step.onnx"),
                optional_file(root.join("decoder_weights.data")),
            )
        };
        let artifacts = Self {
            root: root.to_path_buf(),
            variant,
            encoder,
            decoder_init,
            decoder_step,
            decoder_weights,
            embed_tokens: root.join("embed_tokens.bin"),
            tokenizer: root.join("tokenizer.json"),
            config: root.join("config.json"),
            preprocessor_config: root.join("preprocessor_config.json"),
        };
        artifacts.require_files()?;
        Ok(artifacts)
    }

    fn require_files(&self) -> Result<()> {
        for path in [
            &self.encoder,
            &self.decoder_init,
            &self.decoder_step,
            &self.embed_tokens,
            &self.tokenizer,
            &self.config,
            &self.preprocessor_config,
        ] {
            if !path.exists() {
                return Err(InfraError::ModelNotConfigured {
                    model_id: "qwen-asr".to_string(),
                    reason: format!("required Qwen ASR artifact is missing: {}", path.display()),
                });
            }
        }
        Ok(())
    }
}

fn optional_file(path: PathBuf) -> Option<PathBuf> {
    path.exists().then_some(path)
}

#[derive(Debug)]
pub struct QwenAsrAdapter {
    model_id: String,
    artifacts: QwenAsrArtifacts,
    encoder: OrtSession,
    decoder_init: OrtSession,
    decoder_step: OrtSession,
    tokenizer: TokenizerSpec,
    config: QwenModelConfig,
    embeddings: EmbedTokens,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenAsrProviderReport {
    pub encoder: SessionProviderReport,
    pub decoder_init: SessionProviderReport,
    pub decoder_step: SessionProviderReport,
}

impl QwenAsrAdapter {
    pub fn load(spec: &ModelSpec) -> Result<Self> {
        let root = spec
            .artifacts
            .first()
            .map(|a| a.path.clone())
            .ok_or_else(|| InfraError::ModelNotConfigured {
                model_id: spec.id.clone(),
                reason: "Qwen ASR artifact directory is not configured".to_string(),
            })?;
        let artifacts = QwenAsrArtifacts::validate(root)?;
        let tokenizer = TokenizerSpec::load(&artifacts.tokenizer)?;
        let config = QwenModelConfig::load(&artifacts.config)?;
        let embeddings = EmbedTokens::load(&artifacts.embed_tokens, &config)?;
        let backend = OrtBackend::new(ProviderSelection::from_strings(
            &spec.runtime.provider_order,
        ));
        let encoder = backend.load_session(&artifacts.encoder)?;
        let decoder_init = backend.load_session(&artifacts.decoder_init)?;
        let decoder_step = backend.load_session(&artifacts.decoder_step)?;
        Ok(Self {
            model_id: spec.id.clone(),
            artifacts,
            encoder,
            decoder_init,
            decoder_step,
            tokenizer,
            config,
            embeddings,
        })
    }

    pub fn transcribe(&mut self, audio: &FileRef) -> Result<InferenceOutput> {
        let path = local_files::local_path(audio)?;
        let samples = audio::read_wav_mono_f32(&path)?;
        let features = features::log_mel_128(&samples, 16_000)?;
        let audio_features = self.run_encoder(&features)?;
        let token_ids = self.greedy_decode(&audio_features)?;
        let text = self.tokenizer.decode(&token_ids)?;
        Ok(InferenceOutput::AsrTranscription {
            text: strip_asr_prefix(&text).trim().to_string(),
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }
    pub fn artifacts(&self) -> &QwenAsrArtifacts {
        &self.artifacts
    }

    pub fn provider_report(&self) -> QwenAsrProviderReport {
        QwenAsrProviderReport {
            encoder: self.encoder.provider_report(),
            decoder_init: self.decoder_init.provider_report(),
            decoder_step: self.decoder_step.provider_report(),
        }
    }

    fn run_encoder(&mut self, features: &features::MelFeatures) -> Result<OrtOutput> {
        require_input(&self.encoder, "mel", "encoder")?;
        let outputs = self
            .encoder
            .run_f32(&[local_backend_ort::OrtInput {
                name: "mel".to_string(),
                shape: vec![1, features.bins, features.frames],
                data: features.data.clone(),
            }])
            .map_err(|err| run_boundary_error("encoder", &self.encoder, err))?;
        let output = find_f32_output(outputs, "audio_features", "encoder", &self.encoder)?;
        if output.shape.len() != 3
            || output.shape[0] != 1
            || output.shape[2] != self.config.hidden_size
        {
            return Err(InfraError::NeedImplementation(format!(
                "Qwen ASR encoder produced unsupported audio_features shape {:?}; expected [1, N, {}]; {}",
                output.shape,
                self.config.hidden_size,
                format_session_io("encoder", self.encoder.metadata())
            )));
        }
        Ok(output)
    }

    fn greedy_decode(&mut self, audio_features: &OrtOutput) -> Result<Vec<u32>> {
        let audio_token_count = audio_features.shape.get(1).copied().ok_or_else(|| {
            InfraError::NeedImplementation(format!(
                "Qwen ASR encoder output shape {:?} has no audio token dimension",
                audio_features.shape
            ))
        })?;
        if audio_token_count == 0 {
            return Ok(Vec::new());
        }

        let prompt = build_prompt_ids(audio_token_count);
        let audio_offset = prompt_audio_offset(&prompt)?;
        let seq_len = prompt.len();
        let init_inputs = vec![
            tensor_i64(
                "input_ids",
                vec![1, seq_len],
                prompt.iter().map(|&id| id as i64).collect(),
            ),
            tensor_i64(
                "position_ids",
                vec![1, seq_len],
                (0..seq_len).map(|pos| pos as i64).collect(),
            ),
            tensor_f32(
                "audio_features",
                audio_features.shape.clone(),
                audio_features.data.clone(),
            ),
            tensor_i64("audio_offset", vec![1], vec![audio_offset as i64]),
        ];
        require_inputs(
            &self.decoder_init,
            &[
                "input_ids",
                "position_ids",
                "audio_features",
                "audio_offset",
            ],
            "decoder_init",
        )?;
        let init_outputs = self
            .decoder_init
            .run_tensors(&init_inputs)
            .map_err(|err| run_boundary_error("decoder_init", &self.decoder_init, err))?;
        let (logits, mut keys, mut values) = decoder_outputs(
            init_outputs,
            "decoder_init",
            &self.decoder_init,
            self.config.vocab_size,
        )?;
        let mut next_token = argmax_last_vocab(&logits, self.config.vocab_size)?;
        if self.config.eos_token_ids.contains(&next_token) {
            return Ok(Vec::new());
        }

        let mut generated = vec![next_token];
        let mut position = keys.shape.get(3).copied().unwrap_or(seq_len);
        for _ in 1..MAX_NEW_TOKENS {
            let embed = self.embeddings.lookup(next_token)?;
            let step_inputs = vec![
                tensor_f32("input_embeds", vec![1, 1, self.config.hidden_size], embed),
                tensor_i64("position_ids", vec![1, 1], vec![position as i64]),
                tensor_f32("past_keys", keys.shape.clone(), take_f32_data(keys)?),
                tensor_f32("past_values", values.shape.clone(), take_f32_data(values)?),
            ];
            require_inputs(
                &self.decoder_step,
                &["input_embeds", "position_ids", "past_keys", "past_values"],
                "decoder_step",
            )?;
            let step_outputs = self
                .decoder_step
                .run_tensors(&step_inputs)
                .map_err(|err| run_boundary_error("decoder_step", &self.decoder_step, err))?;
            let (step_logits, new_keys, new_values) = decoder_outputs(
                step_outputs,
                "decoder_step",
                &self.decoder_step,
                self.config.vocab_size,
            )?;
            keys = new_keys;
            values = new_values;
            next_token = argmax_last_vocab(&step_logits, self.config.vocab_size)?;
            position += 1;

            if self.config.eos_token_ids.contains(&next_token) {
                break;
            }
            generated.push(next_token);
        }
        Ok(generated)
    }
}

#[derive(Debug, Clone)]
pub struct TokenizerSpec {
    pub raw: Value,
    tokenizer: tokenizers::Tokenizer,
}

impl TokenizerSpec {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).map_err(|e| InfraError::io(Some(path.to_path_buf()), e))?;
        let raw = serde_json::from_slice(&bytes)?;
        let tokenizer = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| InfraError::Adapter(format!("load tokenizer {}: {e}", path.display())))?;
        Ok(Self { raw, tokenizer })
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| InfraError::Adapter(format!("Qwen ASR tokenizer decode failed: {e}")))
    }
}

#[derive(Debug, Clone)]
pub struct QwenModelConfig {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub eos_token_ids: Vec<u32>,
    pub embed_tokens_dtype: String,
}

impl QwenModelConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).map_err(|e| InfraError::io(Some(path.to_path_buf()), e))?;
        let raw: Value = serde_json::from_slice(&bytes)?;
        let decoder = raw.get("decoder");
        let hidden_size = decoder
            .and_then(|v| v.get("hidden_size"))
            .or_else(|| raw.get("hidden_size"))
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_HIDDEN_SIZE as u64) as usize;
        let vocab_size = decoder
            .and_then(|v| v.get("vocab_size"))
            .or_else(|| raw.get("vocab_size"))
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_VOCAB_SIZE as u64) as usize;
        let mut eos_token_ids = match raw.get("eos_token_id") {
            Some(Value::Array(values)) => values
                .iter()
                .filter_map(|value| value.as_u64().map(|id| id as u32))
                .collect::<Vec<_>>(),
            Some(value) => value
                .as_u64()
                .map(|id| vec![id as u32])
                .unwrap_or_else(|| EOS_TOKEN_IDS.to_vec()),
            None => EOS_TOKEN_IDS.to_vec(),
        };
        for eos in EOS_TOKEN_IDS {
            if !eos_token_ids.contains(&eos) {
                eos_token_ids.push(eos);
            }
        }
        let embed_tokens_dtype = raw
            .get("embed_tokens_dtype")
            .and_then(Value::as_str)
            .unwrap_or("float16")
            .to_string();
        Ok(Self {
            hidden_size,
            vocab_size,
            eos_token_ids,
            embed_tokens_dtype,
        })
    }
}

#[derive(Debug, Clone)]
pub struct EmbedTokens {
    bytes: Vec<u8>,
    vocab_size: usize,
    hidden_size: usize,
    dtype: EmbedDtype,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EmbedDtype {
    F16,
    F32,
}

impl EmbedTokens {
    pub fn load(path: &Path, config: &QwenModelConfig) -> Result<Self> {
        let bytes = fs::read(path).map_err(|e| InfraError::io(Some(path.to_path_buf()), e))?;
        let dtype = match config.embed_tokens_dtype.as_str() {
            "float32" | "f32" => EmbedDtype::F32,
            _ => EmbedDtype::F16,
        };
        let bytes_per_value = match dtype {
            EmbedDtype::F16 => 2,
            EmbedDtype::F32 => 4,
        };
        let expected = config
            .vocab_size
            .checked_mul(config.hidden_size)
            .and_then(|values| values.checked_mul(bytes_per_value))
            .ok_or_else(|| InfraError::ModelNotConfigured {
                model_id: "qwen-asr".to_string(),
                reason: "embed_tokens shape overflows usize".to_string(),
            })?;
        if bytes.len() != expected {
            return Err(InfraError::ModelNotConfigured {
                model_id: "qwen-asr".to_string(),
                reason: format!(
                    "embed_tokens.bin has {} bytes, expected {expected} for [{}, {}] {}",
                    bytes.len(),
                    config.vocab_size,
                    config.hidden_size,
                    config.embed_tokens_dtype
                ),
            });
        }
        Ok(Self {
            bytes,
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            dtype,
        })
    }

    pub fn lookup(&self, token_id: u32) -> Result<Vec<f32>> {
        let token_id = token_id as usize;
        if token_id >= self.vocab_size {
            return Err(InfraError::Backend(format!(
                "generated token id {token_id} is outside embedding vocab {}",
                self.vocab_size
            )));
        }
        let values_per_token = self.hidden_size;
        let bytes_per_value = match self.dtype {
            EmbedDtype::F16 => 2,
            EmbedDtype::F32 => 4,
        };
        let start = token_id * values_per_token * bytes_per_value;
        let end = start + values_per_token * bytes_per_value;
        let slice = &self.bytes[start..end];
        Ok(match self.dtype {
            EmbedDtype::F16 => slice
                .chunks_exact(2)
                .map(|bytes| {
                    half::f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32()
                })
                .collect(),
            EmbedDtype::F32 => slice
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                .collect(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct KvCache {
    pub keys: OrtTensorOutput,
    pub values: OrtTensorOutput,
}

#[derive(Debug, Clone)]
pub struct DecoderLoop {
    eos_token_ids: Vec<u32>,
    audio_pad_token_id: u32,
    pub cache: Option<KvCache>,
    pub emitted: Vec<u32>,
}

impl DecoderLoop {
    pub fn new(eos_token_ids: Vec<u32>, audio_pad_token_id: u32) -> Self {
        Self {
            eos_token_ids,
            audio_pad_token_id,
            cache: None,
            emitted: Vec::new(),
        }
    }

    pub fn should_stop(&self, token_id: u32) -> bool {
        self.eos_token_ids.contains(&token_id)
    }

    pub fn audio_pad_token_id(&self) -> u32 {
        self.audio_pad_token_id
    }
}

pub mod audio;
pub mod features;

fn tensor_i64(name: &str, shape: Vec<usize>, data: Vec<i64>) -> OrtTensorInput {
    OrtTensorInput {
        name: name.to_string(),
        shape,
        data: OrtTensorData::I64(data),
    }
}

fn tensor_f32(name: &str, shape: Vec<usize>, data: Vec<f32>) -> OrtTensorInput {
    OrtTensorInput {
        name: name.to_string(),
        shape,
        data: OrtTensorData::F32(data),
    }
}

fn take_f32_data(output: OrtTensorOutput) -> Result<Vec<f32>> {
    match output.data {
        OrtTensorData::F32(data) => Ok(data),
        other => Err(InfraError::Backend(format!(
            "output `{}` is {:?}, expected f32 KV cache tensor",
            output.name,
            other.element_type()
        ))),
    }
}

fn decoder_outputs(
    outputs: Vec<OrtTensorOutput>,
    stage: &str,
    session: &OrtSession,
    vocab_size: usize,
) -> Result<(OrtTensorOutput, OrtTensorOutput, OrtTensorOutput)> {
    let (logits, keys, values) = select_decoder_outputs(outputs, stage, session)?;
    if !matches!(logits.data, OrtTensorData::F32(_)) {
        return Err(InfraError::NeedImplementation(format!(
            "Qwen ASR {stage} logits output is not f32; {}",
            format_session_io(stage, session.metadata())
        )));
    }
    if !matches!(keys.data, OrtTensorData::F32(_)) || !matches!(values.data, OrtTensorData::F32(_))
    {
        return Err(InfraError::NeedImplementation(format!(
            "Qwen ASR {stage} KV outputs are not f32; {}",
            format_session_io(stage, session.metadata())
        )));
    }
    if logits.shape.last().copied().unwrap_or(0) < vocab_size {
        return Err(InfraError::NeedImplementation(format!(
            "Qwen ASR {stage} logits shape {:?} is smaller than vocab size {vocab_size}; {}",
            logits.shape,
            format_session_io(stage, session.metadata())
        )));
    }
    Ok((logits, keys, values))
}

fn select_decoder_outputs(
    mut outputs: Vec<OrtTensorOutput>,
    stage: &str,
    session: &OrtSession,
) -> Result<(OrtTensorOutput, OrtTensorOutput, OrtTensorOutput)> {
    let logits_index = outputs.iter().position(|output| output.name == "logits");
    let keys_index = outputs
        .iter()
        .position(|output| output.name == "present_keys");
    let values_index = outputs
        .iter()
        .position(|output| output.name == "present_values");

    if let (Some(logits_index), Some(keys_index), Some(values_index)) =
        (logits_index, keys_index, values_index)
    {
        let mut indexed = [
            (logits_index, 0usize),
            (keys_index, 1usize),
            (values_index, 2usize),
        ];
        indexed.sort_by(|a, b| b.0.cmp(&a.0));
        let mut selected: [Option<OrtTensorOutput>; 3] = [None, None, None];
        for (index, slot) in indexed {
            selected[slot] = Some(outputs.remove(index));
        }
        return Ok((
            selected[0].take().expect("logits selected"),
            selected[1].take().expect("keys selected"),
            selected[2].take().expect("values selected"),
        ));
    }

    if outputs.len() >= 3 {
        let mut iter = outputs.into_iter();
        return Ok((
            iter.next().expect("logits fallback"),
            iter.next().expect("keys fallback"),
            iter.next().expect("values fallback"),
        ));
    }

    Err(InfraError::NeedImplementation(format!(
        "Qwen ASR {stage} missing decoder outputs logits/present_keys/present_values and fewer than three fallback outputs were returned; {}",
        format_session_io(stage, session.metadata())
    )))
}

fn find_f32_output(
    outputs: Vec<OrtOutput>,
    preferred_name: &str,
    stage: &str,
    session: &OrtSession,
) -> Result<OrtOutput> {
    if let Some(output) = outputs.iter().find(|output| output.name == preferred_name) {
        return Ok(output.clone());
    }
    outputs.into_iter().next().ok_or_else(|| {
        InfraError::NeedImplementation(format!(
            "Qwen ASR {stage} did not return any f32 outputs; {}",
            format_session_io(stage, session.metadata())
        ))
    })
}

fn argmax_last_vocab(output: &OrtTensorOutput, vocab_size: usize) -> Result<u32> {
    let data = match &output.data {
        OrtTensorData::F32(data) => data,
        other => {
            return Err(InfraError::Backend(format!(
                "logits output `{}` is {:?}, expected f32",
                output.name,
                other.element_type()
            )))
        }
    };
    if data.len() < vocab_size {
        return Err(InfraError::Backend(format!(
            "logits output `{}` has {} values, expected at least vocab size {vocab_size}",
            output.name,
            data.len()
        )));
    }
    let offset = data.len() - vocab_size;
    Ok(data[offset..]
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index as u32)
        .unwrap_or(0))
}

pub fn build_prompt_ids(audio_token_count: usize) -> Vec<u32> {
    let mut ids = vec![
        IM_START_TOKEN_ID,
        9125,
        NEWLINE_TOKEN_ID,
        IM_END_TOKEN_ID,
        NEWLINE_TOKEN_ID,
        IM_START_TOKEN_ID,
        882,
        NEWLINE_TOKEN_ID,
        AUDIO_START_TOKEN_ID,
    ];
    ids.extend(std::iter::repeat(AUDIO_PAD_TOKEN_ID).take(audio_token_count));
    ids.extend([
        AUDIO_END_TOKEN_ID,
        IM_END_TOKEN_ID,
        NEWLINE_TOKEN_ID,
        IM_START_TOKEN_ID,
        77091,
        NEWLINE_TOKEN_ID,
    ]);
    ids
}

fn prompt_audio_offset(prompt: &[u32]) -> Result<usize> {
    prompt
        .iter()
        .position(|id| *id == AUDIO_PAD_TOKEN_ID)
        .ok_or_else(|| InfraError::Backend("Qwen ASR prompt has no audio_pad tokens".to_string()))
}

fn strip_asr_prefix(text: &str) -> &str {
    text.split("<asr_text>").last().unwrap_or(text)
}

fn require_input(session: &OrtSession, name: &str, stage: &str) -> Result<()> {
    require_inputs(session, &[name], stage)
}

fn require_inputs(session: &OrtSession, names: &[&str], stage: &str) -> Result<()> {
    for name in names {
        if !session.inputs().iter().any(|input| input.name == *name) {
            return Err(InfraError::NeedImplementation(format!(
                "Qwen ASR {stage} tensor binding not recognized: required input `{name}` is absent; {}",
                format_session_io(stage, session.metadata())
            )));
        }
    }
    Ok(())
}

fn run_boundary_error(stage: &str, session: &OrtSession, err: InfraError) -> InfraError {
    InfraError::NeedImplementation(format!(
        "Qwen ASR {stage} ORT execution failed; this is the verified tensor/runtime/custom-op boundary. ORT error: {err}; {}",
        format_session_io(stage, session.metadata())
    ))
}

fn format_session_io(stage: &str, metadata: &SessionMetadata) -> String {
    format!(
        "{stage} inputs: [{}]; outputs: [{}]",
        metadata
            .inputs
            .iter()
            .map(format_tensor_metadata)
            .collect::<Vec<_>>()
            .join(", "),
        metadata
            .outputs
            .iter()
            .map(format_tensor_metadata)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn format_tensor_metadata(tensor: &TensorMetadata) -> String {
    format!(
        "{}:{:?}{:?}",
        tensor.name, tensor.element_type, tensor.shape
    )
}

#[cfg(test)]
mod tests;
