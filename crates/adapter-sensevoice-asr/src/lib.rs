//! Native Rust/ORT SenseVoiceSmall adapter.
//!
//! The frontend, LFR+CMVN, four-input ONNX invocation and CTC collapse follow
//! the official FunASR `runtime/onnxruntime/src/sensevoice-small.cpp` path.

mod audio;
mod features;
mod speaker;
mod vad;

use features::{Cmvn, SenseVoiceConfigFile, SenseVoiceFeatures};
use local_backend_ort::{
    OrtBackend, OrtSession, OrtTensorData, OrtTensorInput, OrtTensorOutput, ProviderSelection,
    SessionProviderReport, TensorElement,
};
use local_core::{AsrSegment, AsrSpeaker, AsrTokenTimestamp, FileRef, InferenceOutput, ModelSpec};
use local_error::{InfraError, Result};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

const MODEL_FILE: &str = "model_quant.onnx";
const CONFIG_FILE: &str = "config.yaml";
const CMVN_FILE: &str = "am.mvn";
const TOKENS_FILE: &str = "tokens.json";
const BLANK_ID: usize = 0;
const LANGUAGE_AUTO_ID: i32 = 0;
const WITH_ITN_ID: i32 = 14;
const DEFAULT_SPEAKER_BATCH_SIZE: usize = 32;
const MAX_SPEAKER_BATCH_SIZE: usize = 256;

#[derive(Debug, Clone)]
pub struct SenseVoiceArtifacts {
    pub root: PathBuf,
    pub asr_root: PathBuf,
    pub vad_root: PathBuf,
    pub speaker_root: PathBuf,
    pub model: PathBuf,
    pub config: PathBuf,
    pub cmvn: PathBuf,
    pub tokens: PathBuf,
}

impl SenseVoiceArtifacts {
    pub fn validate(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let asr_root = root.join("asr");
        let vad_root = root.join("vad");
        let speaker_root = root.join("speaker");
        let artifacts = Self {
            model: asr_root.join(MODEL_FILE),
            config: asr_root.join(CONFIG_FILE),
            cmvn: asr_root.join(CMVN_FILE),
            tokens: asr_root.join(TOKENS_FILE),
            asr_root,
            vad_root,
            speaker_root,
            root,
        };
        for path in [
            &artifacts.model,
            &artifacts.config,
            &artifacts.cmvn,
            &artifacts.tokens,
            &artifacts.vad_root.join(MODEL_FILE),
            &artifacts.vad_root.join(CONFIG_FILE),
            &artifacts.vad_root.join(CMVN_FILE),
            &artifacts
                .speaker_root
                .join("campplus_cn_en_common_200k.onnx"),
        ] {
            if !path.is_file() {
                return Err(InfraError::ModelNotConfigured {
                    model_id: "sensevoice-small-onnx".to_string(),
                    reason: format!(
                        "required SenseVoice artifact is missing: {}",
                        path.display()
                    ),
                });
            }
        }
        Ok(artifacts)
    }

    fn from_spec(spec: &ModelSpec) -> Result<Self> {
        let asr_model = spec
            .artifacts
            .iter()
            .map(|artifact| &artifact.path)
            .find(|path| {
                path.file_name().and_then(|name| name.to_str()) == Some(MODEL_FILE)
                    && path
                        .parent()
                        .and_then(Path::file_name)
                        .and_then(|name| name.to_str())
                        == Some("asr")
            });
        let root = asr_model
            .and_then(|path| path.parent())
            .and_then(Path::parent)
            .or_else(|| {
                spec.artifacts
                    .iter()
                    .find(|artifact| artifact.path.is_dir())
                    .map(|artifact| artifact.path.as_path())
            })
            .ok_or_else(|| InfraError::ModelNotConfigured {
                model_id: spec.id.clone(),
                reason: "SenseVoice artifact directory is not configured".to_string(),
            })?;
        Self::validate(root).map_err(|error| match error {
            InfraError::ModelNotConfigured { reason, .. } => InfraError::ModelNotConfigured {
                model_id: spec.id.clone(),
                reason,
            },
            other => other,
        })
    }
}

#[derive(Debug)]
pub struct SenseVoiceAsrAdapter {
    model_id: String,
    artifacts: SenseVoiceArtifacts,
    session: OrtSession,
    config: SenseVoiceConfigFile,
    cmvn: Cmvn,
    tokens: Vec<String>,
    vad: vad::FsmnVad,
    speaker: speaker::CampPlus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SenseVoicePipelineProviderReport {
    pub asr: SessionProviderReport,
    pub vad: SessionProviderReport,
    pub speaker: SessionProviderReport,
}

impl SenseVoiceAsrAdapter {
    pub fn load(spec: &ModelSpec) -> Result<Self> {
        let artifacts = SenseVoiceArtifacts::from_spec(spec)?;
        let config = SenseVoiceConfigFile::load(&artifacts.config)?;
        let feature_dim = config.frontend_conf.n_mels * config.frontend_conf.lfr_m;
        let cmvn = Cmvn::load(&artifacts.cmvn, feature_dim)?;
        let tokens: Vec<String> = serde_json::from_slice(
            &fs::read(&artifacts.tokens)
                .map_err(|e| InfraError::io(Some(artifacts.tokens.clone()), e))?,
        )?;
        if tokens.is_empty() {
            return Err(InfraError::Adapter(format!(
                "SenseVoice token list is empty: {}",
                artifacts.tokens.display()
            )));
        }
        let providers = ProviderSelection::from_strings(&spec.runtime.provider_order);
        let backend = OrtBackend::new(providers.clone());
        let session = backend.load_session(&artifacts.model)?;
        validate_session(&session)?;
        let vad_providers = metadata_provider_selection(spec, "vad_provider_order")?
            .unwrap_or_else(ProviderSelection::default);
        let speaker_providers = metadata_provider_selection(spec, "speaker_provider_order")?
            .unwrap_or_else(|| providers.clone());
        let vad = vad::FsmnVad::load(&artifacts.vad_root, &vad_providers)?;
        let speaker_batch_size = metadata_usize(
            spec,
            "speaker_batch_size",
            DEFAULT_SPEAKER_BATCH_SIZE,
            MAX_SPEAKER_BATCH_SIZE,
        )?;
        let speaker_io_binding = metadata_bool(spec, "speaker_io_binding", true)?;
        let speaker = speaker::CampPlus::load(
            &artifacts.speaker_root,
            &speaker_providers,
            speaker_batch_size,
            speaker_io_binding,
        )?;
        Ok(Self {
            model_id: spec.id.clone(),
            artifacts,
            session,
            config,
            cmvn,
            tokens,
            vad,
            speaker,
        })
    }

    pub fn transcribe(&mut self, audio: &FileRef) -> Result<InferenceOutput> {
        self.transcribe_with_params(audio, &BTreeMap::new())
    }

    pub fn transcribe_with_params(
        &mut self,
        audio: &FileRef,
        params: &BTreeMap<String, serde_json::Value>,
    ) -> Result<InferenceOutput> {
        let options = AsrOptions::from_params(params);
        let path = local_files::local_path(audio)?;
        let samples = audio::read_wav_mono_f32(&path)?;
        if samples.is_empty() {
            return Ok(InferenceOutput::AsrTranscription {
                text: String::new(),
                segments: Vec::new(),
                speakers: Vec::new(),
            });
        }
        let speech_segments = self.vad.segment(&samples)?;
        let mut speaker_labels = if options.speaker_diarization {
            self.speaker.label_segments(&samples, &speech_segments)?
        } else {
            vec![0; speech_segments.len()]
        };
        relabel_speakers_by_first_segment(&mut speaker_labels);
        let mut merged = String::new();
        let mut segments = Vec::with_capacity(speech_segments.len());
        for (index, speech_segment) in speech_segments.iter().enumerate() {
            let chunk = &samples[speech_segment.start_sample..speech_segment.end_sample];
            let features = features::extract(chunk, &self.config.frontend_conf, &self.cmvn)?;
            if features.frames == 0 {
                continue;
            }
            let decoded = self.infer_features(features)?;
            append_transcript(&mut merged, decoded.text.trim());
            let start_ms = samples_to_ms(speech_segment.start_sample);
            let end_ms = samples_to_ms(speech_segment.end_sample);
            let speaker = options
                .speaker_diarization
                .then(|| format!("speaker_{}", speaker_labels[index]));
            let tokens = if options.timestamps {
                decoded
                    .tokens
                    .into_iter()
                    .filter_map(|token| {
                        let token_start = (start_ms + token.start_ms).min(end_ms);
                        let token_end = (start_ms + token.end_ms).min(end_ms);
                        (token_end > token_start).then_some(AsrTokenTimestamp {
                            start_ms: token_start,
                            end_ms: token_end,
                            text: token.text,
                        })
                    })
                    .collect()
            } else {
                Vec::new()
            };
            segments.push(AsrSegment {
                start_ms,
                end_ms,
                text: decoded.text,
                speaker,
                language: decoded.language,
                emotion: decoded.emotion,
                events: decoded.events,
                tokens,
            });
        }
        let speakers = summarize_speakers(&segments);
        Ok(InferenceOutput::AsrTranscription {
            text: merged,
            segments,
            speakers,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn artifacts(&self) -> &SenseVoiceArtifacts {
        &self.artifacts
    }

    pub fn provider_report(&self) -> SessionProviderReport {
        self.session.provider_report()
    }

    pub fn pipeline_provider_report(&self) -> SenseVoicePipelineProviderReport {
        SenseVoicePipelineProviderReport {
            asr: self.session.provider_report(),
            vad: self.vad.provider_report(),
            speaker: self.speaker.provider_report(),
        }
    }

    fn infer_features(&mut self, features: SenseVoiceFeatures) -> Result<DecodedText> {
        let inputs = bind_inputs(&self.session, features)?;
        let outputs = self.session.run_tensors(&inputs).map_err(|error| {
            InfraError::Adapter(format!(
                "SenseVoice ORT execution failed: {error}; {}",
                format_session_io(&self.session)
            ))
        })?;
        decode_outputs(
            outputs,
            &self.tokens,
            self.config.frontend_conf.frame_shift as u64 * self.config.frontend_conf.lfr_n as u64,
        )
    }
}

fn metadata_usize(spec: &ModelSpec, key: &str, default: usize, maximum: usize) -> Result<usize> {
    let Some(value) = spec.metadata.get(key) else {
        return Ok(default);
    };
    let parsed = value
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .filter(|&value| value > 0 && value <= maximum)
        .ok_or_else(|| {
            InfraError::Adapter(format!(
                "model metadata `{key}` must be an integer in 1..={maximum}, got {value}"
            ))
        })?;
    Ok(parsed)
}

fn metadata_provider_selection(spec: &ModelSpec, key: &str) -> Result<Option<ProviderSelection>> {
    let Some(value) = spec.metadata.get(key) else {
        return Ok(None);
    };
    let values = value.as_array().ok_or_else(|| {
        InfraError::Adapter(format!(
            "model metadata `{key}` must be a non-empty array of cpu/cuda provider names"
        ))
    })?;
    if values.is_empty() {
        return Err(InfraError::Adapter(format!(
            "model metadata `{key}` must not be empty"
        )));
    }
    let mut order = Vec::with_capacity(values.len());
    for value in values {
        let provider = value.as_str().ok_or_else(|| {
            InfraError::Adapter(format!(
                "model metadata `{key}` entries must be strings, got {value}"
            ))
        })?;
        if !matches!(provider.to_ascii_lowercase().as_str(), "cpu" | "cuda") {
            return Err(InfraError::Adapter(format!(
                "model metadata `{key}` only supports cpu/cuda, got `{provider}`"
            )));
        }
        order.push(provider.to_string());
    }
    Ok(Some(ProviderSelection::from_strings(&order)))
}

fn metadata_bool(spec: &ModelSpec, key: &str, default: bool) -> Result<bool> {
    let Some(value) = spec.metadata.get(key) else {
        return Ok(default);
    };
    value.as_bool().ok_or_else(|| {
        InfraError::Adapter(format!(
            "model metadata `{key}` must be a boolean, got {value}"
        ))
    })
}

#[derive(Debug, Clone, Copy)]
struct AsrOptions {
    speaker_diarization: bool,
    timestamps: bool,
}

impl AsrOptions {
    fn from_params(params: &BTreeMap<String, serde_json::Value>) -> Self {
        Self {
            speaker_diarization: bool_param(
                params,
                &[
                    "speaker_diarization",
                    "diarization",
                    "speaker_identification",
                ],
                true,
            ),
            timestamps: bool_param(params, &["timestamps", "return_timestamps"], true),
        }
    }
}

fn bool_param(params: &BTreeMap<String, serde_json::Value>, names: &[&str], default: bool) -> bool {
    names
        .iter()
        .find_map(|name| params.get(*name).and_then(serde_json::Value::as_bool))
        .unwrap_or(default)
}

fn samples_to_ms(samples: usize) -> u64 {
    (samples as u64 * 1000) / audio::TARGET_SAMPLE_RATE as u64
}

fn summarize_speakers(segments: &[AsrSegment]) -> Vec<AsrSpeaker> {
    let mut durations = BTreeMap::<String, u64>::new();
    for segment in segments {
        if let Some(speaker) = &segment.speaker {
            *durations.entry(speaker.clone()).or_default() +=
                segment.end_ms.saturating_sub(segment.start_ms);
        }
    }
    durations
        .into_iter()
        .map(|(id, speech_ms)| AsrSpeaker { id, speech_ms })
        .collect()
}

fn relabel_speakers_by_first_segment(labels: &mut [usize]) {
    let mut seen = Vec::new();
    for label in labels {
        let mapped = seen
            .iter()
            .position(|value| value == label)
            .unwrap_or_else(|| {
                seen.push(*label);
                seen.len() - 1
            });
        *label = mapped;
    }
}

fn validate_session(session: &OrtSession) -> Result<()> {
    let inputs = session.inputs();
    if inputs.len() != 4
        || inputs
            .iter()
            .filter(|input| input.element_type == TensorElement::F32)
            .count()
            != 1
        || inputs
            .iter()
            .filter(|input| input.element_type == TensorElement::I32)
            .count()
            != 3
    {
        return Err(InfraError::Adapter(format!(
            "SenseVoice ONNX must have one F32 feature input and three I32 control inputs; {}",
            format_session_io(session)
        )));
    }
    if !session
        .outputs()
        .iter()
        .any(|output| output.element_type == TensorElement::F32)
    {
        return Err(InfraError::Adapter(format!(
            "SenseVoice ONNX has no F32 logits output; {}",
            format_session_io(session)
        )));
    }
    Ok(())
}

fn bind_inputs(session: &OrtSession, features: SenseVoiceFeatures) -> Result<Vec<OrtTensorInput>> {
    let mut unrecognized_i32 = 0;
    session
        .inputs()
        .iter()
        .map(|input| match input.element_type {
            TensorElement::F32 => Ok(OrtTensorInput {
                name: input.name.clone(),
                shape: vec![1, features.frames, features.dim],
                data: OrtTensorData::F32(features.data.clone()),
            }),
            TensorElement::I32 => {
                let lower = input.name.to_ascii_lowercase();
                let value = if lower.contains("length") || lower.contains("len") {
                    features.frames as i32
                } else if lower.contains("language") || lower.contains("lid") {
                    LANGUAGE_AUTO_ID
                } else if lower.contains("textnorm") || lower.contains("itn") {
                    WITH_ITN_ID
                } else {
                    let fallback = [features.frames as i32, LANGUAGE_AUTO_ID, WITH_ITN_ID];
                    let value = fallback.get(unrecognized_i32).copied().ok_or_else(|| {
                        InfraError::Adapter(format!(
                            "unrecognized SenseVoice input `{}`",
                            input.name
                        ))
                    })?;
                    unrecognized_i32 += 1;
                    value
                };
                Ok(OrtTensorInput {
                    name: input.name.clone(),
                    shape: vec![1],
                    data: OrtTensorData::I32(vec![value]),
                })
            }
            other => Err(InfraError::Adapter(format!(
                "unsupported SenseVoice input `{}` type {other:?}",
                input.name
            ))),
        })
        .collect()
}

#[derive(Debug)]
struct DecodedText {
    text: String,
    language: Option<String>,
    emotion: Option<String>,
    events: Vec<String>,
    tokens: Vec<RelativeTokenTimestamp>,
}

#[derive(Debug)]
struct RelativeTokenTimestamp {
    start_ms: u64,
    end_ms: u64,
    text: String,
}

fn decode_outputs(
    outputs: Vec<OrtTensorOutput>,
    tokens: &[String],
    output_frame_ms: u64,
) -> Result<DecodedText> {
    let logits = outputs
        .into_iter()
        .find(|output| output.data.element_type() == TensorElement::F32 && output.shape.len() == 3)
        .ok_or_else(|| {
            InfraError::Adapter("SenseVoice returned no rank-3 F32 logits".to_string())
        })?;
    let OrtTensorData::F32(data) = logits.data else {
        unreachable!()
    };
    let time = logits.shape[1];
    let vocab = logits.shape[2];
    if vocab == 0 || data.len() != logits.shape.iter().product::<usize>() {
        return Err(InfraError::Adapter(format!(
            "invalid SenseVoice logits shape {:?} with {} values",
            logits.shape,
            data.len()
        )));
    }
    if vocab > tokens.len() {
        return Err(InfraError::Adapter(format!(
            "SenseVoice logits vocab {vocab} exceeds token list {}",
            tokens.len()
        )));
    }
    let mut collapsed = Vec::new();
    let mut previous = usize::MAX;
    for frame in 0..time {
        let row = &data[frame * vocab..(frame + 1) * vocab];
        let id = row
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index)
            .unwrap_or(BLANK_ID);
        if id != BLANK_ID && id != previous {
            collapsed.push((id, frame));
        }
        previous = id;
    }
    decode_token_frames(&collapsed, tokens, output_frame_ms)
}

#[cfg(test)]
fn decode_tokens(ids: &[usize], tokens: &[String]) -> Result<String> {
    let decoded = decode_token_frames(
        &ids.iter()
            .copied()
            .enumerate()
            .map(|(frame, id)| (id, frame))
            .collect::<Vec<_>>(),
        tokens,
        60,
    )?;
    Ok(decoded.text)
}

fn decode_token_frames(
    ids: &[(usize, usize)],
    tokens: &[String],
    output_frame_ms: u64,
) -> Result<DecodedText> {
    let body = ids.get(4..).unwrap_or(&[]);
    let mut text = String::new();
    let mut timestamps = Vec::new();
    for &(id, frame) in body {
        let token = tokens.get(id).ok_or_else(|| {
            InfraError::Adapter(format!("SenseVoice emitted unknown token id {id}"))
        })?;
        if token.starts_with("<|") && token.ends_with("|>") {
            continue;
        }
        if let Some(word) = token.strip_prefix('▁') {
            if !text.is_empty() && !text.chars().last().is_some_and(char::is_whitespace) {
                text.push(' ');
            }
            text.push_str(word);
            timestamps.push(RelativeTokenTimestamp {
                start_ms: frame as u64 * output_frame_ms,
                end_ms: (frame as u64 + 1) * output_frame_ms,
                text: word.to_string(),
            });
        } else {
            text.push_str(token);
            timestamps.push(RelativeTokenTimestamp {
                start_ms: frame as u64 * output_frame_ms,
                end_ms: (frame as u64 + 1) * output_frame_ms,
                text: token.clone(),
            });
        }
    }
    let language = ids
        .first()
        .and_then(|(id, _)| tokens.get(*id))
        .map(String::as_str);
    let with_itn = ids
        .get(3)
        .and_then(|(id, _)| tokens.get(*id))
        .map(String::as_str)
        == Some("<|withitn|>");
    let mut text = text.trim().to_string();
    if !text.is_empty() && with_itn && !ends_with_punctuation(&text) {
        text.push(if language == Some("<|zh|>") {
            '。'
        } else {
            '.'
        });
    }
    let control = |index: usize| {
        ids.get(index)
            .and_then(|(id, _)| tokens.get(*id))
            .map(|value| {
                value
                    .trim_start_matches("<|")
                    .trim_end_matches("|>")
                    .to_string()
            })
            .filter(|value| !value.is_empty())
    };
    let event = control(2).filter(|value| !value.eq_ignore_ascii_case("speech"));
    Ok(DecodedText {
        text,
        language: control(0),
        emotion: control(1),
        events: event.into_iter().collect(),
        tokens: timestamps,
    })
}

fn append_transcript(output: &mut String, chunk: &str) {
    if chunk.is_empty() {
        return;
    }
    let needs_space = output
        .chars()
        .last()
        .is_some_and(|ch| ch.is_ascii_alphanumeric())
        && chunk
            .chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_alphanumeric());
    if needs_space {
        output.push(' ');
    }
    output.push_str(chunk);
}

fn ends_with_punctuation(text: &str) -> bool {
    text.chars().last().is_some_and(|ch| {
        matches!(
            ch,
            '.' | '!' | '?' | ',' | ';' | ':' | '。' | '！' | '？' | '，' | '；' | '：'
        )
    })
}

fn format_session_io(session: &OrtSession) -> String {
    let inputs = session
        .inputs()
        .iter()
        .map(|input| format!("{}:{:?}{:?}", input.name, input.element_type, input.shape))
        .collect::<Vec<_>>()
        .join(", ");
    let outputs = session
        .outputs()
        .iter()
        .map(|output| {
            format!(
                "{}:{:?}{:?}",
                output.name, output.element_type, output.shape
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("inputs: [{inputs}]; outputs: [{outputs}]")
}

#[cfg(test)]
mod tests;
