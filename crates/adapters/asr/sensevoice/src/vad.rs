//! FSMN-VAD, streamed: FBANK, LFR+CMVN and the model run block by block
//! (the FSMN is causal, its four caches carried from block to block), and
//! segments come out as soon as they close, so a long recording needs memory
//! for a block, not for its whole feature matrix.
//!
//! A segment reaching the maximum length is cut where the speech is weakest
//! in its last seconds (a breath or pause) rather than at the limit itself.

use crate::features::{self, Cmvn, FrontendConfig};
use local_backend_ort::{
    OrtBackend, OrtSession, OrtTensorData, OrtTensorInput, ProviderSelection,
    SessionProviderReport, TensorElement,
};
use local_error::{InfraError, Result};
use serde::Deserialize;
use std::{collections::VecDeque, fs, path::Path};

/// LFR frames per model run (10 s).
const BLOCK_FRAMES: usize = 1_000;
/// Samples gathered before their FBANK frames are computed (1 s).
const FBANK_BLOCK_SAMPLES: usize = 16_000;
/// How far back from the maximum length a cut may go.
const SPLIT_SEARCH_MS: usize = 5_000;
/// Half the span whose mean speech probability scores a cut.
const SPLIT_SMOOTH_MS: usize = 100;
/// How much weaker (mean speech probability) an earlier cut must be.
const SPLIT_MIN_GAIN: f32 = 1e-3;

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

#[derive(Debug, Clone, Deserialize)]
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
    /// `max_segment_ms` replaces the config's `max_single_segment_time`.
    pub fn load(
        root: &Path,
        providers: &ProviderSelection,
        max_segment_ms: Option<usize>,
    ) -> Result<Self> {
        let config_path = root.join("config.yaml");
        let cmvn_path = root.join("am.mvn");
        let model_path = root.join("model_quant.onnx");
        let mut config: VadConfigFile = serde_yaml::from_slice(
            &fs::read(&config_path).map_err(|e| InfraError::io(Some(config_path.clone()), e))?,
        )?;
        if let Some(max_segment_ms) = max_segment_ms {
            config.model_conf.max_single_segment_time = max_segment_ms;
        }
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

    /// A new stream; feed it with [`Self::push`], end it with [`Self::finish`].
    pub fn start(&self) -> VadStream {
        let caches = self
            .session
            .inputs()
            .iter()
            .filter(|input| !is_speech_input(&input.name, input.shape.len()))
            .map(|input| {
                let shape = input
                    .shape
                    .iter()
                    .map(|&dim| usize::try_from(dim).ok().filter(|&v| v > 0).unwrap_or(1))
                    .collect::<Vec<_>>();
                Cache {
                    input: input.name.clone(),
                    data: vec![0.0; shape.iter().product()],
                    shape,
                }
            })
            .collect();
        VadStream {
            samples: Vec::new(),
            samples_start: 0,
            sample_count: 0,
            lfr: Lfr::default(),
            block: Vec::new(),
            block_frames: 0,
            caches,
            decoder: Decoder::new(&self.config.model_conf),
        }
    }

    /// Feeds 16 kHz mono samples; returns the segments they closed.
    pub fn push(&mut self, stream: &mut VadStream, samples: &[f32]) -> Result<Vec<SpeechSegment>> {
        stream.samples.extend_from_slice(samples);
        stream.sample_count += samples.len();
        let mut segments = Vec::new();
        if stream.samples.len() >= FBANK_BLOCK_SAMPLES {
            self.compute_fbank(stream)?;
        }
        if stream.block_frames >= BLOCK_FRAMES {
            self.run_block(stream, &mut segments)?;
        }
        Ok(segments)
    }

    /// Ends the stream; returns the segments still open.
    pub fn finish(&mut self, stream: &mut VadStream) -> Result<Vec<SpeechSegment>> {
        let mut segments = Vec::new();
        self.compute_fbank(stream)?;
        stream
            .lfr
            .finish(&self.config.frontend_conf, &self.cmvn, &mut stream.block);
        stream.block_frames = stream.block.len() / self.feature_dim();
        self.run_block(stream, &mut segments)?;
        if let Some((start, end)) = stream.decoder.finish() {
            push_segment(
                &mut segments,
                start,
                end,
                stream.decoder.frame_ms,
                stream.sample_count,
            );
        }
        Ok(segments)
    }

    /// The speech segments of a whole recording.
    #[cfg(test)]
    pub fn segment(&mut self, samples: &[f32]) -> Result<Vec<SpeechSegment>> {
        let mut stream = self.start();
        let mut segments = Vec::new();
        for chunk in samples.chunks(FBANK_BLOCK_SAMPLES) {
            segments.extend(self.push(&mut stream, chunk)?);
        }
        segments.extend(self.finish(&mut stream)?);
        Ok(segments)
    }

    pub fn provider_report(&self) -> SessionProviderReport {
        self.session.provider_report()
    }

    /// The per-frame speech probabilities of [`Self::segment`]'s stream and
    /// of one model run over the whole recording's features.
    #[cfg(test)]
    pub fn compare_one_shot(&mut self, samples: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let mut stream = self.start();
        for chunk in samples.chunks(FBANK_BLOCK_SAMPLES) {
            self.push(&mut stream, chunk)?;
        }
        self.finish(&mut stream)?;
        let streamed = std::mem::take(&mut stream.decoder.trace);
        let feats = features::extract(samples, &self.config.frontend_conf, Some(&self.cmvn))?;
        let mut stream = self.start();
        stream.sample_count = samples.len();
        stream.block = feats.data;
        stream.block_frames = feats.frames;
        self.run_block(&mut stream, &mut Vec::new())?;
        Ok((streamed, std::mem::take(&mut stream.decoder.trace)))
    }

    fn feature_dim(&self) -> usize {
        self.config.frontend_conf.n_mels * self.config.frontend_conf.lfr_m
    }

    /// FBANK of every whole frame in the buffered samples, then their LFR
    /// frames, into the model block.
    fn compute_fbank(&self, stream: &mut VadStream) -> Result<()> {
        let frontend = &self.config.frontend_conf;
        let shift = frontend.frame_shift_samples();
        let length = frontend.frame_length_samples();
        if stream.samples.len() < length {
            return Ok(());
        }
        let frames = 1 + (stream.samples.len() - length) / shift;
        let raw =
            features::fbank_frames(&stream.samples[..(frames - 1) * shift + length], frontend)?;
        debug_assert_eq!(raw.len(), frames);
        stream.samples.drain(..frames * shift);
        stream.samples_start += frames * shift;
        stream
            .lfr
            .push(raw, frontend, &self.cmvn, &mut stream.block);
        stream.block_frames = stream.block.len() / self.feature_dim();
        Ok(())
    }

    fn run_block(
        &mut self,
        stream: &mut VadStream,
        segments: &mut Vec<SpeechSegment>,
    ) -> Result<()> {
        if stream.block_frames == 0 {
            return Ok(());
        }
        let dim = self.feature_dim();
        let frames = stream.block_frames;
        let mut inputs = Vec::with_capacity(self.session.inputs().len());
        for input in self.session.inputs() {
            if input.element_type != TensorElement::F32 {
                return Err(InfraError::Adapter(format!(
                    "FSMN-VAD input `{}` is not F32",
                    input.name
                )));
            }
            if is_speech_input(&input.name, input.shape.len()) {
                inputs.push(OrtTensorInput {
                    name: input.name.clone(),
                    shape: vec![1, frames, dim],
                    data: OrtTensorData::F32(std::mem::take(&mut stream.block)),
                });
            } else {
                let cache = stream
                    .caches
                    .iter()
                    .find(|cache| cache.input == input.name)
                    .expect("one cache per non-speech input");
                inputs.push(OrtTensorInput {
                    name: input.name.clone(),
                    shape: cache.shape.clone(),
                    data: OrtTensorData::F32(cache.data.clone()),
                });
            }
        }
        stream.block_frames = 0;
        let outputs = self.session.run_tensors(&inputs).map_err(|error| {
            InfraError::Adapter(format!("FSMN-VAD ORT execution failed: {error}"))
        })?;
        let mut logits = None;
        for output in outputs {
            if output.name == "logits" {
                logits = Some(output);
                continue;
            }
            // out_cacheN feeds in_cacheN of the next block.
            let Some(index) = output.name.strip_prefix("out_cache") else {
                continue;
            };
            let input = format!("in_cache{index}");
            if let (Some(cache), OrtTensorData::F32(data)) = (
                stream.caches.iter_mut().find(|cache| cache.input == input),
                output.data,
            ) {
                if data.len() != cache.data.len() {
                    return Err(InfraError::Adapter(format!(
                        "FSMN-VAD {} has {} values, {input} wants {}",
                        output.name,
                        data.len(),
                        cache.data.len()
                    )));
                }
                cache.data = data;
            }
        }
        let logits =
            logits.ok_or_else(|| InfraError::Adapter("FSMN-VAD returned no logits".to_string()))?;
        let OrtTensorData::F32(data) = logits.data else {
            return Err(InfraError::Adapter(
                "FSMN-VAD logits are not F32".to_string(),
            ));
        };
        if logits.shape.len() != 3 || logits.shape[1] != frames || logits.shape[2] < 2 {
            return Err(InfraError::Adapter(format!(
                "unexpected FSMN-VAD logits shape {:?} for {frames} frames",
                logits.shape
            )));
        }
        let classes = logits.shape[2];
        let mut closed = Vec::new();
        for row in data.chunks_exact(classes) {
            stream.decoder.push(row[0], &mut closed);
        }
        for (start, end) in closed {
            push_segment(
                segments,
                start,
                end,
                stream.decoder.frame_ms,
                stream.sample_count,
            );
        }
        Ok(())
    }
}

fn is_speech_input(name: &str, rank: usize) -> bool {
    name == "speech" || (rank == 3 && !name.starts_with("in_cache"))
}

/// One recording's VAD state.
#[derive(Debug)]
pub struct VadStream {
    /// Samples not yet turned into FBANK frames, from `samples_start` on.
    samples: Vec<f32>,
    samples_start: usize,
    sample_count: usize,
    lfr: Lfr,
    /// LFR+CMVN frames waiting for the model.
    block: Vec<f32>,
    block_frames: usize,
    caches: Vec<Cache>,
    decoder: Decoder,
}

impl VadStream {
    /// The first sample a segment not yet returned may start at: the caller
    /// may drop the audio before it.
    pub fn retain_from_sample(&self) -> usize {
        let samples_per_frame =
            self.decoder.frame_ms * crate::audio::TARGET_SAMPLE_RATE as usize / 1000;
        self.decoder.retain_from_frame() * samples_per_frame
    }
}

#[derive(Debug)]
struct Cache {
    input: String,
    shape: Vec<usize>,
    data: Vec<f32>,
}

/// Streaming low frame rate stacking: output frame `j` is FBANK frames
/// `j * lfr_n - (lfr_m - 1) / 2 ..` (`lfr_m` of them), the first frame
/// standing in before the start and the last one past the end, as
/// [`features::extract`] does for a whole recording.
#[derive(Debug, Default)]
struct Lfr {
    /// FBANK frames from absolute index `raw_start` on.
    raw: VecDeque<Vec<f32>>,
    raw_start: usize,
    raw_count: usize,
    first: Option<Vec<f32>>,
    next_output: usize,
}

impl Lfr {
    fn push(
        &mut self,
        frames: Vec<Vec<f32>>,
        frontend: &FrontendConfig,
        cmvn: &Cmvn,
        out: &mut Vec<f32>,
    ) {
        if self.first.is_none() {
            self.first = frames.first().cloned();
        }
        self.raw_count += frames.len();
        self.raw.extend(frames);
        let padding = (frontend.lfr_m - 1) / 2;
        // Emit while the last FBANK frame an output needs has arrived.
        while self.next_output * frontend.lfr_n + frontend.lfr_m - 1 < self.raw_count + padding {
            self.emit(frontend, cmvn, out, false);
        }
        self.trim(frontend);
    }

    fn finish(&mut self, frontend: &FrontendConfig, cmvn: &Cmvn, out: &mut Vec<f32>) {
        let total = self.raw_count.div_ceil(frontend.lfr_n);
        while self.next_output < total {
            self.emit(frontend, cmvn, out, true);
        }
        self.raw.clear();
    }

    fn emit(&mut self, frontend: &FrontendConfig, cmvn: &Cmvn, out: &mut Vec<f32>, flushing: bool) {
        let padding = (frontend.lfr_m - 1) / 2;
        let start = out.len();
        for offset in 0..frontend.lfr_m {
            let padded = self.next_output * frontend.lfr_n + offset;
            let frame = if padded < padding {
                self.first.as_ref().expect("first FBANK frame")
            } else {
                let index = (padded - padding).min(self.raw_count - 1);
                debug_assert!(flushing || index == padded - padding);
                &self.raw[index - self.raw_start]
            };
            out.extend_from_slice(frame);
        }
        cmvn.apply(&mut out[start..]);
        self.next_output += 1;
    }

    fn trim(&mut self, frontend: &FrontendConfig) {
        let padding = (frontend.lfr_m - 1) / 2;
        // The next output's first frame, and the last one for the flush.
        let keep_from = (self.next_output * frontend.lfr_n)
            .saturating_sub(padding)
            .min(self.raw_count.saturating_sub(1));
        while self.raw_start < keep_from {
            self.raw.pop_front();
            self.raw_start += 1;
        }
    }
}

/// FunASR's FSMN-VAD decision, one frame at a time.
#[derive(Debug)]
struct Decoder {
    frame_ms: usize,
    speech_noise_thres: f32,
    start_votes: usize,
    start_latency: usize,
    end_silence_frames: usize,
    end_lookback: usize,
    max_frames: usize,
    split_search: usize,
    split_smooth: usize,
    window: Vec<bool>,
    window_pos: usize,
    votes: usize,
    in_speech_window: bool,
    active_start: Option<usize>,
    silence_run: usize,
    /// Speech probabilities from frame `probs_start` on.
    probs: VecDeque<f32>,
    probs_start: usize,
    frame: usize,
    /// Every frame's speech probability, for tests.
    #[cfg(test)]
    trace: Vec<f32>,
}

impl Decoder {
    fn new(config: &VadModelConfig) -> Self {
        let frame_ms = config.frame_in_ms.max(1);
        let window_frames = (config.window_size_ms / frame_ms).max(1);
        let end_silence_frames = config
            .max_end_silence_time
            .saturating_sub(config.sil_to_speech_time_thres)
            / frame_ms;
        let max_frames = (config.max_single_segment_time / frame_ms).max(2);
        Self {
            frame_ms,
            speech_noise_thres: config.speech_noise_thres,
            start_votes: (config.sil_to_speech_time_thres / frame_ms).max(1),
            start_latency: window_frames + config.lookback_time_start_point / frame_ms,
            end_silence_frames,
            end_lookback: end_silence_frames
                .saturating_sub(config.lookahead_time_end_point / frame_ms)
                .saturating_sub(1),
            max_frames,
            split_search: (SPLIT_SEARCH_MS / frame_ms).min(max_frames / 2),
            split_smooth: SPLIT_SMOOTH_MS / frame_ms,
            window: vec![false; window_frames],
            window_pos: 0,
            votes: 0,
            in_speech_window: false,
            active_start: None,
            silence_run: 0,
            probs: VecDeque::new(),
            probs_start: 0,
            frame: 0,
            #[cfg(test)]
            trace: Vec::new(),
        }
    }

    /// One frame's noise probability; closed segments (frame ranges) are
    /// appended to `closed`.
    fn push(&mut self, noise: f32, closed: &mut Vec<(usize, usize)>) {
        let frame = self.frame;
        self.frame += 1;
        let noise = noise.clamp(0.0, 1.0);
        let speech = 1.0 - noise;
        let raw_speech = speech >= noise + self.speech_noise_thres;
        self.probs.push_back(speech);
        #[cfg(test)]
        self.trace.push(speech);
        if self.window[self.window_pos] {
            self.votes -= 1;
        }
        self.window[self.window_pos] = raw_speech;
        if raw_speech {
            self.votes += 1;
        }
        self.window_pos = (self.window_pos + 1) % self.window.len();

        let was_speech_window = self.in_speech_window;
        if !self.in_speech_window && self.votes >= self.start_votes {
            self.in_speech_window = true;
        } else if self.in_speech_window && self.votes <= self.start_votes {
            self.in_speech_window = false;
        }
        if !was_speech_window && self.in_speech_window && self.active_start.is_none() {
            self.active_start = Some(frame.saturating_sub(self.start_latency));
            self.silence_run = 0;
        }
        if let Some(start) = self.active_start {
            self.silence_run = if raw_speech { 0 } else { self.silence_run + 1 };
            if !self.in_speech_window && self.silence_run >= self.end_silence_frames {
                let end = frame.saturating_sub(self.end_lookback).max(start + 1);
                closed.push((start, end));
                self.active_start = None;
                self.silence_run = 0;
            } else if frame + 1 - start >= self.max_frames {
                let cut = self.natural_cut(start, frame);
                closed.push((start, cut));
                self.active_start = Some(cut);
            }
        }
        let keep_from = self.retain_from_frame().max(self.probs_start);
        while self.probs_start < keep_from {
            self.probs.pop_front();
            self.probs_start += 1;
        }
    }

    /// Where a segment over the maximum length ends: the frame in its last
    /// `split_search` frames around which speech is weakest (the latest
    /// such frame on a tie, up to right after `frame`), so a cut falls on a
    /// pause, not inside a word.
    fn natural_cut(&self, start: usize, frame: usize) -> usize {
        let lowest = (start + 1).max((frame + 1).saturating_sub(self.split_search));
        let prob = |index: usize| {
            index
                .checked_sub(self.probs_start)
                .and_then(|index| self.probs.get(index))
                .copied()
        };
        // From the latest candidate back, moving only for clearly weaker
        // speech: equal stretches keep the segment as long as allowed.
        let mut best = (frame + 1, f32::INFINITY);
        for candidate in (lowest..=frame + 1).rev() {
            let from = candidate.saturating_sub(self.split_smooth).max(start);
            let to = (candidate + self.split_smooth).min(frame);
            let (sum, count) = (from..=to)
                .filter_map(prob)
                .fold((0.0, 0usize), |(sum, count), value| {
                    (sum + value, count + 1)
                });
            let score = if count == 0 {
                f32::INFINITY
            } else {
                sum / count as f32
            };
            if score < best.1 - SPLIT_MIN_GAIN {
                best = (candidate, score);
            }
        }
        best.0
    }

    fn finish(&mut self) -> Option<(usize, usize)> {
        let start = self.active_start.take()?;
        (self.frame > start).then_some((start, self.frame))
    }

    /// The earliest frame a segment not yet closed may start at.
    fn retain_from_frame(&self) -> usize {
        self.active_start
            .unwrap_or_else(|| self.frame.saturating_sub(self.start_latency))
    }
}

fn validate_session(session: &OrtSession, feature_dim: usize) -> Result<()> {
    let speech = session
        .inputs()
        .iter()
        .find(|input| is_speech_input(&input.name, input.shape.len()))
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
    for input in session.inputs() {
        if is_speech_input(&input.name, input.shape.len()) {
            continue;
        }
        let output = input.name.replacen("in_cache", "out_cache", 1);
        if !input.name.starts_with("in_cache")
            || !session
                .outputs()
                .iter()
                .any(|candidate| candidate.name == output)
        {
            return Err(InfraError::Adapter(format!(
                "FSMN-VAD input `{}` has no matching cache output",
                input.name
            )));
        }
    }
    if !session
        .outputs()
        .iter()
        .any(|output| output.name == "logits")
    {
        return Err(InfraError::Adapter(
            "FSMN-VAD has no `logits` output".to_string(),
        ));
    }
    Ok(())
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

    fn decode(noise: &[f32], config: &VadModelConfig) -> Vec<(usize, usize)> {
        let mut decoder = Decoder::new(config);
        let mut closed = Vec::new();
        for &value in noise {
            decoder.push(value, &mut closed);
        }
        closed.extend(decoder.finish());
        closed
    }

    #[test]
    fn continuous_speech_is_bounded_to_the_maximum_length() {
        let segments = decode(&vec![0.1; 6_000], &config());
        assert_eq!(segments.len(), 3, "{segments:?}");
        assert!(segments.iter().all(|(start, end)| end - start <= 2_000));
        assert_eq!(segments.last().expect("last").1, 6_000);
        // Contiguous: nothing is dropped at a cut.
        assert!(segments.windows(2).all(|pair| pair[0].1 == pair[1].0));
    }

    #[test]
    fn an_overlong_segment_is_cut_at_its_weakest_point() {
        // 25 s of speech with a short dip (too short to end the segment)
        // 17 s in: the cut goes there, not at 20 s.
        let mut noise = vec![0.05; 2_500];
        for value in &mut noise[1_690..1_710] {
            *value = 0.7;
        }
        let segments = decode(&noise, &config());
        assert_eq!(segments.len(), 2, "{segments:?}");
        assert!((1_690..1_710).contains(&segments[0].1), "{segments:?}");
        assert_eq!(segments[1].0, segments[0].1);
    }

    #[test]
    fn silence_closes_a_segment_and_pauses_do_not() {
        let mut noise = vec![0.95; 100];
        noise.extend(vec![0.05; 300]);
        noise.extend(vec![0.95; 30]); // 300 ms pause
        noise.extend(vec![0.05; 300]);
        noise.extend(vec![0.95; 200]); // 2 s silence
        noise.extend(vec![0.05; 300]);
        let segments = decode(&noise, &config());
        assert_eq!(segments.len(), 2, "{segments:?}");
        assert!(segments[0].0 < 100 && segments[0].1 > 730 && segments[0].1 < 930);
        assert_eq!(segments[1].1, noise.len());
    }

    #[test]
    fn lfr_stream_matches_whole_recording_lfr() {
        let frontend = FrontendConfig {
            fs: 16_000,
            window: "hamming".to_string(),
            n_mels: 2,
            frame_length: 25,
            frame_shift: 10,
            lfr_m: 5,
            lfr_n: 1,
        };
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("am.mvn");
        fs::write(
            &path,
            "<AddShift> 10 10\n<LearnRateCoef> 0 [ 0 0 0 0 0 0 0 0 0 0 ]\n<Rescale> 10 10\n<LearnRateCoef> 0 [ 1 1 1 1 1 1 1 1 1 1 ]\n",
        )
        .expect("write cmvn");
        let cmvn = Cmvn::load(&path, 10).expect("cmvn");
        for lfr_n in [1, 3] {
            let frontend = FrontendConfig {
                lfr_n,
                ..frontend.clone()
            };
            for total in [1, 2, 7, 23] {
                let raw = (0..total)
                    .map(|i| vec![i as f32, -(i as f32)])
                    .collect::<Vec<_>>();
                let whole =
                    features::lfr_cmvn_for_test(raw.clone(), &frontend, &cmvn).expect("whole lfr");
                for chunk in [1, 2, 5] {
                    let mut lfr = Lfr::default();
                    let mut out = Vec::new();
                    for part in raw.chunks(chunk) {
                        lfr.push(part.to_vec(), &frontend, &cmvn, &mut out);
                    }
                    lfr.finish(&frontend, &cmvn, &mut out);
                    assert_eq!(out, whole.data, "n={lfr_n} total={total} chunk={chunk}");
                }
            }
        }
    }
}
