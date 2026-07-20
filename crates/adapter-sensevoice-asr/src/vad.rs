use crate::features::{self, Cmvn, FrontendConfig};
use local_backend_ort::{
    OrtBackend, OrtSession, OrtTensorData, OrtTensorInput, ProviderSelection,
    SessionProviderReport, TensorElement,
};
use local_error::{InfraError, Result};
use serde::Deserialize;
use std::{fs, path::Path};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeechSegment {
    pub start_sample: usize,
    pub end_sample: usize,
}

#[derive(Debug, Deserialize)]
struct VadConfigFile {
    frontend_conf: FrontendConfig,
    model_conf: VadModelConfig,
}

#[derive(Debug, Deserialize)]
struct VadModelConfig {
    max_end_silence_time: usize,
    max_single_segment_time: usize,
    speech_noise_thres: f32,
    #[serde(default = "default_window_ms")]
    window_size_ms: usize,
    #[serde(default = "default_transition_ms")]
    sil_to_speech_time_thres: usize,
    #[serde(default = "default_lookback_ms")]
    lookback_time_start_point: usize,
    #[serde(default = "default_lookahead_ms")]
    lookahead_time_end_point: usize,
    #[serde(default = "default_frame_ms")]
    frame_in_ms: usize,
}

fn default_window_ms() -> usize {
    200
}
fn default_transition_ms() -> usize {
    150
}
fn default_lookback_ms() -> usize {
    200
}
fn default_lookahead_ms() -> usize {
    100
}
fn default_frame_ms() -> usize {
    10
}

#[derive(Debug)]
pub struct FsmnVad {
    session: OrtSession,
    config: VadConfigFile,
    cmvn: Cmvn,
}

impl FsmnVad {
    pub fn load(root: &Path, providers: &ProviderSelection) -> Result<Self> {
        let config_path = root.join("config.yaml");
        let cmvn_path = root.join("am.mvn");
        let model_path = root.join("model_quant.onnx");
        let config: VadConfigFile = serde_yaml::from_slice(
            &fs::read(&config_path).map_err(|e| InfraError::io(Some(config_path.clone()), e))?,
        )?;
        let dim = config.frontend_conf.n_mels * config.frontend_conf.lfr_m;
        let cmvn = Cmvn::load(&cmvn_path, dim)?;
        let session = OrtBackend::new(providers.clone()).load_session(&model_path)?;
        validate_session(&session, dim)?;
        Ok(Self {
            session,
            config,
            cmvn,
        })
    }

    pub fn segment(&mut self, samples: &[f32]) -> Result<Vec<SpeechSegment>> {
        let feats = features::extract(samples, &self.config.frontend_conf, &self.cmvn)?;
        if feats.frames == 0 {
            return Ok(Vec::new());
        }
        let mut inputs = Vec::with_capacity(self.session.inputs().len());
        for input in self.session.inputs() {
            if input.element_type != TensorElement::F32 {
                return Err(InfraError::Adapter(format!(
                    "FSMN-VAD input `{}` is not F32",
                    input.name
                )));
            }
            if input.name == "speech" || input.shape.len() == 3 {
                inputs.push(OrtTensorInput {
                    name: input.name.clone(),
                    shape: vec![1, feats.frames, feats.dim],
                    data: OrtTensorData::F32(feats.data.clone()),
                });
            } else {
                let shape = input
                    .shape
                    .iter()
                    .map(|&dim| usize::try_from(dim).ok().filter(|&v| v > 0).unwrap_or(1))
                    .collect::<Vec<_>>();
                inputs.push(OrtTensorInput {
                    name: input.name.clone(),
                    data: OrtTensorData::F32(vec![0.0; shape.iter().product()]),
                    shape,
                });
            }
        }
        let outputs = self.session.run_tensors(&inputs).map_err(|error| {
            InfraError::Adapter(format!("FSMN-VAD ORT execution failed: {error}"))
        })?;
        let logits = outputs
            .into_iter()
            .find(|output| output.name == "logits" || output.shape.len() == 3)
            .ok_or_else(|| InfraError::Adapter("FSMN-VAD returned no rank-3 logits".to_string()))?;
        let OrtTensorData::F32(data) = logits.data else {
            return Err(InfraError::Adapter(
                "FSMN-VAD logits are not F32".to_string(),
            ));
        };
        if logits.shape.len() != 3 || logits.shape[2] < 2 {
            return Err(InfraError::Adapter(format!(
                "unexpected FSMN-VAD logits shape {:?}",
                logits.shape
            )));
        }
        Ok(decode_segments(
            &data,
            logits.shape[1],
            logits.shape[2],
            samples.len(),
            &self.config.model_conf,
        ))
    }

    pub fn provider_report(&self) -> SessionProviderReport {
        self.session.provider_report()
    }
}

fn validate_session(session: &OrtSession, feature_dim: usize) -> Result<()> {
    let speech = session
        .inputs()
        .iter()
        .find(|input| input.name == "speech" || input.shape.len() == 3)
        .ok_or_else(|| InfraError::Adapter("FSMN-VAD has no speech input".to_string()))?;
    if speech.element_type != TensorElement::F32
        || speech
            .shape
            .last()
            .copied()
            .is_some_and(|dim| dim > 0 && dim as usize != feature_dim)
    {
        return Err(InfraError::Adapter(format!(
            "FSMN-VAD speech input does not accept feature dim {feature_dim}: {:?}",
            speech.shape
        )));
    }
    Ok(())
}

fn decode_segments(
    logits: &[f32],
    frames: usize,
    classes: usize,
    sample_count: usize,
    config: &VadModelConfig,
) -> Vec<SpeechSegment> {
    let frame_ms = config.frame_in_ms.max(1);
    let window_frames = (config.window_size_ms / frame_ms).max(1);
    let start_votes = (config.sil_to_speech_time_thres / frame_ms).max(1);
    let start_latency = window_frames + config.lookback_time_start_point / frame_ms;
    let end_silence_frames = config
        .max_end_silence_time
        .saturating_sub(config.sil_to_speech_time_thres)
        / frame_ms;
    let end_lookback = end_silence_frames
        .saturating_sub(config.lookahead_time_end_point / frame_ms)
        .saturating_sub(1);
    let max_frames = (config.max_single_segment_time / frame_ms).max(1);
    let mut window = vec![false; window_frames];
    let mut window_pos = 0usize;
    let mut votes = 0usize;
    let mut in_speech_window = false;
    let mut active_start = None;
    let mut silence_run = 0usize;
    let mut result = Vec::new();

    for frame in 0..frames {
        let row = &logits[frame * classes..(frame + 1) * classes];
        let noise = row[0].clamp(0.0, 1.0);
        let speech = 1.0 - noise;
        let raw_speech = speech >= noise + config.speech_noise_thres;
        if window[window_pos] {
            votes -= 1;
        }
        window[window_pos] = raw_speech;
        if raw_speech {
            votes += 1;
        }
        window_pos = (window_pos + 1) % window_frames;

        let was_speech_window = in_speech_window;
        if !in_speech_window && votes >= start_votes {
            in_speech_window = true;
        } else if in_speech_window && votes <= start_votes {
            in_speech_window = false;
        }
        if !was_speech_window && in_speech_window && active_start.is_none() {
            active_start = Some(frame.saturating_sub(start_latency));
            silence_run = 0;
        }
        if active_start.is_some() {
            silence_run = if raw_speech { 0 } else { silence_run + 1 };
            let start = active_start.unwrap_or(0);
            let timed_out = frame.saturating_sub(start) + 1 >= max_frames;
            let ended = !in_speech_window && silence_run >= end_silence_frames;
            if timed_out || ended {
                let end = if ended {
                    frame.saturating_sub(end_lookback)
                } else {
                    frame + 1
                };
                push_segment(
                    &mut result,
                    start,
                    end.max(start + 1),
                    frame_ms,
                    sample_count,
                );
                active_start = timed_out.then_some(end);
                silence_run = 0;
            }
        }
    }
    if let Some(start) = active_start {
        push_segment(&mut result, start, frames, frame_ms, sample_count);
    }
    result
}

fn push_segment(
    result: &mut Vec<SpeechSegment>,
    start_frame: usize,
    end_frame: usize,
    frame_ms: usize,
    sample_count: usize,
) {
    let samples_per_ms = crate::audio::TARGET_SAMPLE_RATE as usize / 1000;
    let start_sample = (start_frame * frame_ms * samples_per_ms).min(sample_count);
    let end_sample = (end_frame * frame_ms * samples_per_ms).min(sample_count);
    if end_sample > start_sample {
        result.push(SpeechSegment {
            start_sample,
            end_sample,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> VadModelConfig {
        VadModelConfig {
            max_end_silence_time: 800,
            max_single_segment_time: 20_000,
            speech_noise_thres: 0.6,
            window_size_ms: 200,
            sil_to_speech_time_thres: 150,
            lookback_time_start_point: 200,
            lookahead_time_end_point: 100,
            frame_in_ms: 10,
        }
    }

    #[test]
    fn continuous_sixty_seconds_is_bounded_to_three_segments() {
        let frames = 6_000;
        let logits = (0..frames).flat_map(|_| [0.1, 0.9]).collect::<Vec<_>>();
        let segments = decode_segments(
            &logits,
            frames,
            2,
            60 * crate::audio::TARGET_SAMPLE_RATE as usize,
            &config(),
        );
        assert_eq!(segments.len(), 3);
        assert!(segments
            .iter()
            .all(|segment| segment.end_sample - segment.start_sample
                <= 20 * crate::audio::TARGET_SAMPLE_RATE as usize));
        assert_eq!(segments.last().expect("last").end_sample, 60 * 16_000);
    }
}
