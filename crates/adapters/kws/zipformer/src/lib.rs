//! Wake words: open-vocabulary keyword spotting with the sherpa-onnx
//! zipformer transducer KWS models (k2-fsa icefall's
//! `kws-zipformer-zh-en-3M-2025-12-20`, Apache-2.0: Chinese and English in
//! one model, 3.3M parameters), a port of sherpa-onnx's `KeywordSpotter`
//! on ONNX Runtime.
//!
//! Audio (16 kHz) becomes 80-bin Kaldi fbank frames; a streaming zipformer2
//! encoder reads them a chunk at a time (`T` frames, of which the last 13
//! are lookahead, `decode_chunk_len` frames on), its caches carried from
//! chunk to chunk; a modified beam search over the stateless decoder and the
//! joiner, boosted along the keywords' tokens (an Aho-Corasick graph), spots
//! a keyword once the best path has its last token followed by a blank and
//! the mean probability of its tokens reaches the keyword's threshold.
//!
//! The model directory holds `encoder.onnx`, `decoder.onnx`, `joiner.onnx`
//! (the fp32 chunk-16 exports: 320 ms chunks), `tokens.txt`, `en.phone`
//! and `pinyin.tsv` (see `keywords`). One spotter serves one audio stream:
//! small enough (13 MB, CPU, one thread, ~2 % of a core) to give every
//! conversation its own.

mod graph;
mod keywords;
mod search;

use graph::{Graph, KeywordTokens};
use kaldi_native_fbank::{
    fbank::{FbankComputer, FbankOptions},
    online::{FeatureComputer, OnlineFeature},
};
use local_backend_ort::{
    CpuSessionOptions, OrtBackend, OrtSession, OrtTensorData, OrtTensorInput, ProviderSelection,
    TensorElement,
};
use local_error::{InfraError, Result};
use search::Hyp;
use std::{collections::VecDeque, path::Path};

pub use keywords::{pinyin_tokens, Lexicon};

pub const SAMPLE_RATE: usize = 16_000;
const FEATURE_DIM: usize = 80;
/// Samples per feature frame (10 ms).
const FRAME_SHIFT: usize = SAMPLE_RATE / 100;
/// Frames past a chunk's own that the encoder reads (zipformer2: 7 + 2 * 3).
const PAD_FRAMES: usize = 13;
/// Paths the search keeps each frame. A name's first tokens score low
/// against what else the speech could be: with 4 they were dropped before
/// the rest came (recorded calls in a room's conditions: 71% spotted; 16:
/// 82%, no more false calls in 19 minutes of other speech; 32 and more
/// lost a few "M42"s).
pub const ACTIVE_PATHS: usize = 16;
/// Blanks that must follow a keyword's last token.
const TRAILING_BLANKS: usize = 1;
/// The boost (log domain) of each token along a keyword.
pub const BOOST: f32 = 1.0;
/// The mean token probability a keyword must reach; short ones (a few
/// tokens: "M3" is four) may be set to need more. With ACTIVE_PATHS kept,
/// calls score from 0.21 and other speech none at all (84% spotted at 0.18
/// or 0.20, 82% at 0.25/0.30, no false calls at any).
pub const THRESHOLD: f32 = 0.2;
pub const SHORT_THRESHOLD: f32 = 0.2;
const SHORT_TOKENS: usize = 4;
/// The input's level as the spotter hears it (`Leveler`): speech is
/// brought to about TARGET_RMS (a close, clear recording's: -25 dBFS), by
/// at most MAX_GAIN (+30 dB) up and MIN_GAIN (-6 dB) down. The model's
/// features are log energies: quiet speech (another player's voice across
/// a VRChat room, -20 to -35 dB under a close one) lost the short wake
/// words (a recording at -24 dB: "monster" missed, "M3" 0.68 -> 0.42).
const TARGET_RMS: f32 = 0.056;
const MAX_GAIN: f32 = 31.6;
const MIN_GAIN: f32 = 0.5;
/// The speech level follows a louder 10 ms block at once (half the way)
/// and falls back with this half-life; never under LEVEL_FLOOR (-60 dBFS:
/// silence is not brought up).
const LEVEL_HALF_LIFE_S: f32 = 4.0;
const LEVEL_FLOOR: f32 = 0.001;
/// The gain moves toward the wanted one this share a 10 ms block.
const GAIN_SMOOTHING: f32 = 0.2;

/// After this much trailing silence the search starts over (sherpa: 1.5 s).
const RESET_AFTER_MS: usize = 1500;
/// Feature frames kept before the features start over (seamlessly: the
/// encoder goes on where it was).
const MAX_FRAMES: usize = 30 * 100;
/// ln(FLT_EPSILON): Kaldi's floor of the log mel energies.
const LOG_FLOOR: f32 = -15.942_385;

/// A wake word heard: which, where in the stream (samples since the
/// spotter started) its first and last tokens were, and how sure (the mean
/// probability of its tokens).
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub keyword: String,
    pub start: usize,
    pub end: usize,
    pub score: f32,
}

struct State {
    input: String,
    shape: Vec<usize>,
    i64: bool,
}

pub struct KeywordSpotter {
    encoder: OrtSession,
    decoder: OrtSession,
    joiner: OrtSession,
    lexicon: Lexicon,
    state_inputs: Vec<State>,
    /// Frames an encoder call reads, and moves on.
    chunk: usize,
    shift: usize,
    vocab: usize,
    unk: Option<i64>,
    keywords: Vec<String>,
    graph: Graph,
    // The stream.
    fbank: OnlineFeature,
    /// Samples before the current fbank's first frame.
    fbank_base: usize,
    /// Samples taken so far.
    taken: usize,
    /// Recent input, to restart the features where the encoder stands.
    recent: VecDeque<f32>,
    /// Feature frames the encoder has moved past.
    processed: usize,
    states: Vec<OrtTensorData>,
    hyps: Vec<Hyp>,
    /// Milliseconds per encoder output frame (40: four feature frames).
    frame_ms: usize,
    /// The input's level, brought to the model's (none: as it comes).
    leveler: Option<Leveler>,
    /// Keywords' boost and thresholds (long, short) for `set_keywords`.
    boost: f32,
    threshold: f32,
    short_threshold: f32,
    /// Blanks after a keyword's last token before it counts.
    trailing_blanks: usize,
    /// Paths the search keeps (ACTIVE_PATHS).
    paths: usize,
}

/// An automatic gain for the spotter's input: a peak-following speech level
/// (up at once, down slowly), the gain bringing it to TARGET_RMS within
/// MIN_GAIN..MAX_GAIN, changing smoothly.
#[derive(Debug, Clone)]
pub struct Leveler {
    level: f32,
    gain: f32,
    pending: Vec<f32>,
}

impl Default for Leveler {
    fn default() -> Self {
        Leveler {
            level: TARGET_RMS,
            gain: 1.0,
            pending: Vec::new(),
        }
    }
}

impl Leveler {
    /// `samples` (16 kHz) at the level the model wants; a 10 ms block is
    /// held until whole (the output lags the input by less than 10 ms).
    pub fn apply(&mut self, samples: &[f32]) -> Vec<f32> {
        self.pending.extend_from_slice(samples);
        let whole = self.pending.len() / FRAME_SHIFT * FRAME_SHIFT;
        let decay = 0.5f32.powf(FRAME_SHIFT as f32 / SAMPLE_RATE as f32 / LEVEL_HALF_LIFE_S);
        let mut out = Vec::with_capacity(whole);
        for block in self.pending[..whole].chunks(FRAME_SHIFT) {
            let rms = (block.iter().map(|v| v * v).sum::<f32>() / block.len() as f32).sqrt();
            self.level = if rms > self.level {
                self.level + (rms - self.level) * 0.5
            } else {
                (self.level * decay).max(rms)
            };
            self.level = self.level.max(LEVEL_FLOOR);
            let want = (TARGET_RMS / self.level).clamp(MIN_GAIN, MAX_GAIN);
            let from = self.gain;
            self.gain += (want - self.gain) * GAIN_SMOOTHING;
            let n = block.len() as f32;
            out.extend(block.iter().enumerate().map(|(i, v)| {
                let g = from + (self.gain - from) * (i as f32 + 1.0) / n;
                (v * g).clamp(-1.0, 1.0)
            }));
        }
        self.pending.drain(..whole);
        out
    }

    /// The gain now (for logs).
    pub fn gain(&self) -> f32 {
        self.gain
    }
}

impl KeywordSpotter {
    pub fn load(dir: &Path) -> Result<Self> {
        let backend = OrtBackend::new(ProviderSelection::from_strings(&["cpu".to_string()]))
            .with_cpu_session_options(CpuSessionOptions {
                intra_threads: 1,
                inter_threads: 1,
            })?;
        let encoder = backend.load_session(dir.join("encoder.onnx"))?;
        let decoder = backend.load_session(dir.join("decoder.onnx"))?;
        let joiner = backend.load_session(dir.join("joiner.onnx"))?;
        let lexicon = Lexicon::load(dir)?;
        let bad =
            |what: &str| InfraError::Adapter(format!("KWS model in {}: {what}", dir.display()));
        let x = encoder
            .inputs()
            .first()
            .filter(|i| i.name == "x")
            .ok_or_else(|| bad("the encoder's first input is not `x`"))?;
        let chunk = usize::try_from(*x.shape.get(1).unwrap_or(&-1))
            .map_err(|_| bad("the encoder's chunk length is not fixed"))?;
        let shift = chunk
            .checked_sub(PAD_FRAMES)
            .filter(|s| *s > 0)
            .ok_or_else(|| bad("the encoder's chunk is too short"))?;
        let state_inputs = encoder.inputs()[1..]
            .iter()
            .map(|input| State {
                input: input.name.clone(),
                // The batch (dynamic) is one stream.
                shape: input
                    .shape
                    .iter()
                    .map(|&d| if d < 0 { 1 } else { d as usize })
                    .collect(),
                i64: input.element_type == TensorElement::I64,
            })
            .collect();
        let vocab = lexicon
            .tokens
            .values()
            .max()
            .map(|&m| m as usize + 1)
            .ok_or_else(|| bad("no tokens"))?;
        let unk = lexicon.tokens.get("<unk>").copied();
        let mut spotter = Self {
            encoder,
            decoder,
            joiner,
            lexicon,
            state_inputs,
            chunk,
            shift,
            vocab,
            unk,
            keywords: Vec::new(),
            graph: Graph::new(&[]),
            fbank: new_fbank()?,
            fbank_base: 0,
            taken: 0,
            recent: VecDeque::new(),
            processed: 0,
            states: Vec::new(),
            hyps: Vec::new(),
            frame_ms: 40,
            leveler: Some(Leveler::default()),
            boost: BOOST,
            threshold: THRESHOLD,
            short_threshold: SHORT_THRESHOLD,
            trailing_blanks: TRAILING_BLANKS,
            paths: ACTIVE_PATHS,
        };
        spotter.reset();
        Ok(spotter)
    }

    /// Brings the input to the model's level (on by default) or not.
    pub fn set_leveling(&mut self, on: bool) {
        self.leveler = on.then(Leveler::default);
    }

    /// The input gain now (1 without leveling).
    pub fn gain(&self) -> f32 {
        self.leveler.as_ref().map_or(1.0, Leveler::gain)
    }

    /// Blanks (40 ms each) that must follow a keyword's last token, more
    /// than this many.
    pub fn set_trailing_blanks(&mut self, blanks: usize) {
        self.trailing_blanks = blanks;
    }

    /// Paths the search keeps each frame (at least 1).
    pub fn set_paths(&mut self, paths: usize) {
        self.paths = paths.max(1);
    }

    /// The boost and thresholds (long, short words) `set_keywords` gives
    /// from now on.
    pub fn tune(&mut self, boost: f32, threshold: f32, short_threshold: f32) {
        (self.boost, self.threshold, self.short_threshold) = (boost, threshold, short_threshold);
    }

    /// Spots `words` from now on (in place of any before); returns those it
    /// cannot read (see `keywords`).
    pub fn set_keywords(&mut self, words: &[String]) -> Vec<String> {
        let mut entries = Vec::new();
        let mut unread = Vec::new();
        self.keywords.clear();
        for word in words.iter().map(|w| w.trim()).filter(|w| !w.is_empty()) {
            if self.keywords.iter().any(|k| k == word) {
                continue;
            }
            let variants = self.lexicon.variants(word);
            if variants.is_empty() {
                unread.push(word.to_string());
                continue;
            }
            let phrase = self.keywords.len();
            self.keywords.push(word.to_string());
            for tokens in variants {
                let threshold = if tokens.len() <= SHORT_TOKENS {
                    self.short_threshold
                } else {
                    self.threshold
                };
                entries.push(KeywordTokens {
                    tokens,
                    boost: self.boost,
                    threshold,
                    phrase,
                });
            }
        }
        self.graph = Graph::new(&entries);
        self.hyps = vec![Hyp::start()];
        unread
    }

    /// Spots keywords given as the model's tokens (sherpa-onnx's
    /// `keywords.txt` lines: `<tokens> [:<boost>] [#<threshold>] @<name>`),
    /// in place of any before; returns the lines it cannot use.
    pub fn set_keyword_lines(&mut self, lines: &[String]) -> Vec<String> {
        let mut entries = Vec::new();
        let mut unused = Vec::new();
        self.keywords.clear();
        for line in lines.iter().map(|l| l.trim()).filter(|l| !l.is_empty()) {
            let (spec, name) = line.split_once('@').unwrap_or((line, line));
            let (mut tokens, mut boost, mut threshold) = (Vec::new(), BOOST, None);
            let mut known = true;
            for part in spec.split_whitespace() {
                if let Some(value) = part.strip_prefix(':') {
                    boost = value.parse().unwrap_or(BOOST);
                } else if let Some(value) = part.strip_prefix('#') {
                    threshold = value.parse().ok();
                } else if let Some(&id) = self.lexicon.tokens.get(part) {
                    tokens.push(id);
                } else {
                    known = false;
                }
            }
            if !known || tokens.is_empty() {
                unused.push(line.to_string());
                continue;
            }
            let name = name.trim().to_string();
            let phrase = match self.keywords.iter().position(|k| *k == name) {
                Some(phrase) => phrase,
                None => {
                    self.keywords.push(name);
                    self.keywords.len() - 1
                }
            };
            let threshold = threshold.unwrap_or(if tokens.len() <= SHORT_TOKENS {
                SHORT_THRESHOLD
            } else {
                THRESHOLD
            });
            entries.push(KeywordTokens {
                tokens,
                boost,
                threshold,
                phrase,
            });
        }
        self.graph = Graph::new(&entries);
        self.hyps = vec![Hyp::start()];
        unused
    }

    pub fn keywords(&self) -> &[String] {
        &self.keywords
    }

    /// Input samples the search has gone past: a wake word ending before
    /// this (by a little: it waits for a blank after it) has been spotted.
    pub fn decoded_to(&self) -> usize {
        self.fbank_base + self.processed * FRAME_SHIFT
    }

    /// Takes `samples` (16 kHz, -1..1); returns the wake words heard.
    pub fn accept(&mut self, samples: &[f32]) -> Result<Vec<Detection>> {
        let leveled;
        let samples = match self.leveler.as_mut() {
            Some(leveler) => {
                leveled = leveler.apply(samples);
                &leveled[..]
            }
            None => samples,
        };
        self.fbank.accept_waveform(SAMPLE_RATE as f32, samples);
        self.taken += samples.len();
        self.recent.extend(samples);
        let keep = (self.chunk + 4) * FRAME_SHIFT * 2;
        while self.recent.len() > keep {
            self.recent.pop_front();
        }
        let mut found = Vec::new();
        while self.processed + self.chunk < self.fbank.num_frames_ready() {
            if self.processed > MAX_FRAMES {
                self.restart_features();
            }
            if let Some(detection) = self.decode_chunk()? {
                found.push(detection);
            }
        }
        Ok(found)
    }

    /// The encoder and the search over the next chunk.
    fn decode_chunk(&mut self) -> Result<Option<Detection>> {
        if self.keywords.is_empty() {
            self.processed += self.shift;
            return Ok(None);
        }
        // Silence long enough: start over (sherpa does so before a chunk).
        let best_blanks = self
            .hyps
            .iter()
            .max_by(|a, b| a.log_prob.total_cmp(&b.log_prob))
            .map_or(0, |h| h.trailing_blanks);
        if best_blanks * self.frame_ms > RESET_AFTER_MS {
            self.reset();
        }
        let start = self.processed;
        let mut x = Vec::with_capacity(self.chunk * FEATURE_DIM);
        for frame in start..start + self.chunk {
            // Kaldi floors energies at FLT_EPSILON before the log; this port at
            // 1e-10, which only shows on quiet input (samples are -1..1 here).
            x.extend(
                self.fbank
                    .get_frame(frame)
                    .expect("a ready frame")
                    .iter()
                    .map(|v| v.max(LOG_FLOOR)),
            );
        }
        self.processed += self.shift;
        let mut inputs = vec![OrtTensorInput {
            name: "x".to_string(),
            shape: vec![1, self.chunk, FEATURE_DIM],
            data: OrtTensorData::F32(x),
        }];
        for (state, data) in self.state_inputs.iter().zip(self.states.drain(..)) {
            inputs.push(OrtTensorInput {
                name: state.input.clone(),
                shape: state.shape.clone(),
                data,
            });
        }
        let mut outputs = self.encoder.run_tensors(&inputs)?;
        let encoder_out = outputs
            .iter()
            .position(|o| o.name == "encoder_out")
            .ok_or_else(|| {
                InfraError::Adapter("the KWS encoder gave no encoder_out".to_string())
            })?;
        let encoder_out = outputs.swap_remove(encoder_out);
        for state in &self.state_inputs {
            let name = format!("new_{}", state.input);
            let output = outputs
                .iter_mut()
                .find(|o| o.name == name)
                .ok_or_else(|| InfraError::Adapter(format!("the KWS encoder gave no {name}")))?;
            self.states.push(std::mem::replace(
                &mut output.data,
                OrtTensorData::F32(Vec::new()),
            ));
        }
        let OrtTensorData::F32(frames) = encoder_out.data else {
            return Err(InfraError::Adapter(
                "the KWS encoder output is not float".to_string(),
            ));
        };
        let dim = *encoder_out.shape.last().unwrap_or(&0);
        let out_frames = if dim == 0 { 0 } else { frames.len() / dim };
        let frames_per_out = self.shift / out_frames.max(1);
        self.frame_ms = frames_per_out * 1000 * FRAME_SHIFT / SAMPLE_RATE;
        for (t, frame) in frames.chunks(dim.max(1)).enumerate() {
            // Where (in input samples) this output frame is.
            let at = self.fbank_base + (start + t * frames_per_out) * FRAME_SHIFT;
            if let Some(found) = self.search_frame(frame, at)? {
                let keyword = self.graph.nodes[found.node]
                    .phrase
                    .and_then(|p| self.keywords.get(p))
                    .cloned()
                    .unwrap_or_default();
                // The stream starts afresh after a wake word.
                self.reset();
                return Ok(Some(Detection {
                    keyword,
                    start: found.first_time,
                    end: found.last_time + frames_per_out * FRAME_SHIFT,
                    score: found.score,
                }));
            }
        }
        Ok(None)
    }

    fn search_frame(&mut self, encoder_frame: &[f32], at: usize) -> Result<Option<search::Match>> {
        let hyps = std::mem::take(&mut self.hyps);
        let n = hyps.len();
        let ys: Vec<i64> = hyps.iter().flat_map(|h| h.context().to_vec()).collect();
        let decoder_out = self.decoder.run_tensors(&[OrtTensorInput {
            name: "y".to_string(),
            shape: vec![n, search::CONTEXT],
            data: OrtTensorData::I64(ys),
        }])?;
        let OrtTensorData::F32(decoder_out) = &decoder_out[0].data else {
            return Err(InfraError::Adapter(
                "the KWS decoder output is not float".to_string(),
            ));
        };
        let encoder_rows: Vec<f32> = (0..n).flat_map(|_| encoder_frame.iter().copied()).collect();
        let dim = encoder_frame.len();
        let logits = self.joiner.run_tensors(&[
            OrtTensorInput {
                name: "encoder_out".to_string(),
                shape: vec![n, dim],
                data: OrtTensorData::F32(encoder_rows),
            },
            OrtTensorInput {
                name: "decoder_out".to_string(),
                shape: vec![n, decoder_out.len() / n],
                data: OrtTensorData::F32(decoder_out.clone()),
            },
        ])?;
        let OrtTensorData::F32(mut log_probs) = logits
            .into_iter()
            .next()
            .map(|o| o.data)
            .unwrap_or(OrtTensorData::F32(Vec::new()))
        else {
            return Err(InfraError::Adapter(
                "the KWS joiner output is not float".to_string(),
            ));
        };
        if log_probs.len() != n * self.vocab {
            return Err(InfraError::Adapter(format!(
                "the KWS joiner gave {} logits, not {}",
                log_probs.len(),
                n * self.vocab
            )));
        }
        search::log_softmax(&mut log_probs, self.vocab);
        let (hyps, found) = search::step(
            &self.graph,
            hyps,
            &log_probs,
            search::Params {
                vocab: self.vocab,
                unk: self.unk,
                max_paths: self.paths,
                trailing_blanks: self.trailing_blanks,
            },
            at,
        );
        self.hyps = hyps;
        Ok(found)
    }

    /// The features start over from where the encoder stands, so that they
    /// do not grow without end. Frames are not snipped at the edges: frame
    /// `i` of a stream starts 120 samples before `i` * 160, so frame 2 of
    /// features started 320 samples before the next frame to encode sees
    /// exactly its samples, and the encoder goes on as if nothing happened.
    fn restart_features(&mut self) {
        let next = self.fbank_base + self.processed * FRAME_SHIFT;
        let held_from = self.taken - self.recent.len();
        let Some(from) = next
            .checked_sub(2 * FRAME_SHIFT)
            .filter(|&f| f >= held_from)
        else {
            return; // not held (more taken at once than kept): next chunk
        };
        let Ok(mut fbank) = new_fbank() else {
            return;
        };
        let rest: Vec<f32> = self.recent.iter().skip(from - held_from).copied().collect();
        fbank.accept_waveform(SAMPLE_RATE as f32, &rest);
        self.fbank = fbank;
        self.fbank_base = from;
        self.processed = 2;
    }

    /// Fresh encoder caches and search.
    fn reset(&mut self) {
        self.states = self
            .state_inputs
            .iter()
            .map(|s| {
                let len = s.shape.iter().product();
                if s.i64 {
                    OrtTensorData::I64(vec![0; len])
                } else {
                    OrtTensorData::F32(vec![0.0; len])
                }
            })
            .collect();
        self.hyps = vec![Hyp::start()];
    }
}

/// Kaldi fbank as sherpa-onnx computes it for these models: 80 bins, 25 ms
/// povey windows every 10 ms, no dither, frames not snipped at the edges,
/// mel bins up to 400 Hz below Nyquist, samples as they are (-1..1).
fn new_fbank() -> Result<OnlineFeature> {
    let mut options = FbankOptions::default();
    options.frame_opts.dither = 0.0;
    options.frame_opts.snip_edges = false;
    options.frame_opts.samp_freq = SAMPLE_RATE as f32;
    options.frame_opts.window_type = "povey".to_string();
    options.mel_opts.num_bins = FEATURE_DIM;
    options.mel_opts.low_freq = 20.0;
    options.mel_opts.high_freq = -400.0;
    options.energy_floor = 0.0;
    options.use_energy = false;
    let computer = FbankComputer::new(options)
        .map_err(|e| InfraError::Adapter(format!("initialize Kaldi FBANK: {e}")))?;
    Ok(OnlineFeature::new(FeatureComputer::Fbank(computer)))
}
