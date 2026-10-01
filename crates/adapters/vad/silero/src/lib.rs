//! Streaming voice activity detection with Silero VAD (v5/v6 ONNX, 16 kHz).
//!
//! A port of the reference `OnnxWrapper` and `VADIterator` of
//! <https://github.com/snakers4/silero-vad> (`src/silero_vad/utils_vad.py`):
//! 512-sample windows with 64 samples of context and the recurrent state
//! carried between calls, and the same speech start/end rule.

use local_backend_ort::{
    CpuSessionOptions, OrtBackend, OrtSession, OrtTensorData, OrtTensorInput, ProviderSelection,
};
use local_error::{InfraError, Result};
use std::path::Path;

pub const SAMPLE_RATE: usize = 16_000;
/// Samples per model call at 16 kHz.
pub const WINDOW: usize = 512;
const CONTEXT: usize = 64;
const STATE: usize = 2 * 128;

/// One stream's Silero VAD: the ONNX session and the stream state. Small
/// enough (2 MB, CPU, one thread) to give every stream its own.
#[derive(Debug)]
pub struct SileroVad {
    session: OrtSession,
    state: Vec<f32>,
    context: Vec<f32>,
}

impl SileroVad {
    pub fn load(model: &Path) -> Result<Self> {
        let session = OrtBackend::new(ProviderSelection::from_strings(&["cpu".to_string()]))
            .with_cpu_session_options(CpuSessionOptions {
                intra_threads: 1,
                inter_threads: 1,
            })?
            .load_session(model)?;
        for name in ["input", "state", "sr"] {
            if !session.inputs().iter().any(|input| input.name == name) {
                return Err(InfraError::Adapter(format!(
                    "Silero VAD model {} has no `{name}` input",
                    model.display()
                )));
            }
        }
        Ok(Self {
            session,
            state: vec![0.0; STATE],
            context: vec![0.0; CONTEXT],
        })
    }

    pub fn reset(&mut self) {
        self.state.fill(0.0);
        self.context.fill(0.0);
    }

    /// Speech probability of the next `WINDOW` samples.
    pub fn probability(&mut self, window: &[f32]) -> Result<f32> {
        if window.len() != WINDOW {
            return Err(InfraError::BadRequest(format!(
                "Silero VAD takes {WINDOW} samples at a time, got {}",
                window.len()
            )));
        }
        let mut input = Vec::with_capacity(CONTEXT + WINDOW);
        input.extend_from_slice(&self.context);
        input.extend_from_slice(window);
        let outputs = self.session.run_tensors(&[
            OrtTensorInput {
                name: "input".to_string(),
                shape: vec![1, CONTEXT + WINDOW],
                data: OrtTensorData::F32(input),
            },
            OrtTensorInput {
                name: "state".to_string(),
                shape: vec![2, 1, 128],
                data: OrtTensorData::F32(self.state.clone()),
            },
            OrtTensorInput {
                name: "sr".to_string(),
                shape: Vec::new(),
                data: OrtTensorData::I64(vec![SAMPLE_RATE as i64]),
            },
        ])?;
        let mut probability = None;
        for output in outputs {
            match (output.name.as_str(), output.data) {
                ("output", OrtTensorData::F32(values)) => probability = values.first().copied(),
                ("stateN", OrtTensorData::F32(values)) if values.len() == STATE => {
                    self.state = values
                }
                _ => {}
            }
        }
        self.context.copy_from_slice(&window[WINDOW - CONTEXT..]);
        probability.ok_or_else(|| InfraError::Adapter("Silero VAD returned no output".to_string()))
    }
}

/// Where speech starts or ends, in samples since the stream began.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadEvent {
    Start(usize),
    End(usize),
}

/// The reference `VADIterator`: speech starts at a window at or above
/// `threshold`, and ends after `min_silence_samples` below
/// `threshold - 0.15`; both edges are padded by `speech_pad_samples`.
#[derive(Debug, Clone)]
pub struct VadIterator {
    pub threshold: f32,
    pub min_silence_samples: usize,
    pub speech_pad_samples: usize,
    triggered: bool,
    temp_end: usize,
    current_sample: usize,
}

impl VadIterator {
    pub fn new(threshold: f32, min_silence_ms: usize, speech_pad_ms: usize) -> Self {
        Self {
            threshold,
            min_silence_samples: SAMPLE_RATE * min_silence_ms / 1000,
            speech_pad_samples: SAMPLE_RATE * speech_pad_ms / 1000,
            triggered: false,
            temp_end: 0,
            current_sample: 0,
        }
    }

    /// Whether speech is going on (started and not ended).
    pub fn triggered(&self) -> bool {
        self.triggered
    }

    /// Samples seen so far.
    pub fn position(&self) -> usize {
        self.current_sample
    }

    /// Takes the probability of the next window.
    pub fn step(&mut self, probability: f32) -> Option<VadEvent> {
        self.current_sample += WINDOW;
        if probability >= self.threshold && self.temp_end != 0 {
            self.temp_end = 0;
        }
        if probability >= self.threshold && !self.triggered {
            self.triggered = true;
            let start = self
                .current_sample
                .saturating_sub(self.speech_pad_samples + WINDOW);
            return Some(VadEvent::Start(start));
        }
        if probability < self.threshold - 0.15 && self.triggered {
            if self.temp_end == 0 {
                self.temp_end = self.current_sample;
            }
            if self.current_sample - self.temp_end < self.min_silence_samples {
                return None;
            }
            let end = (self.temp_end + self.speech_pad_samples).saturating_sub(WINDOW);
            self.temp_end = 0;
            self.triggered = false;
            return Some(VadEvent::End(end));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iterator_follows_the_reference_rule() {
        // 100 ms of silence to end, 30 ms of padding.
        let mut vad = VadIterator::new(0.5, 100, 30);
        assert_eq!(vad.step(0.1), None);
        assert_eq!(vad.step(0.9), Some(VadEvent::Start(1024 - 480 - 512)));
        assert!(vad.triggered());
        // Between threshold - 0.15 and threshold: neither speech nor silence.
        assert_eq!(vad.step(0.4), None);
        // 0.1 s = 1600 samples of silence after the first quiet window
        // (temp_end = 2048): the fifth quiet window ends it.
        for _ in 0..4 {
            assert_eq!(vad.step(0.1), None);
        }
        assert_eq!(vad.step(0.1), Some(VadEvent::End(2048 + 480 - 512)));
        assert!(!vad.triggered());
    }

    #[test]
    fn speech_during_a_pause_keeps_the_segment() {
        let mut vad = VadIterator::new(0.5, 100, 30);
        vad.step(0.9);
        vad.step(0.1);
        vad.step(0.1);
        assert_eq!(vad.step(0.8), None); // resets the pause
        for _ in 0..4 {
            assert_eq!(vad.step(0.1), None);
        }
        assert!(matches!(vad.step(0.1), Some(VadEvent::End(_))));
    }

    /// `LOCAL_SILERO_VAD_MODEL=<silero_vad.onnx> ORT_DYLIB_PATH=<onnxruntime>
    /// cargo test -p local-adapter-silero-vad real_model -- --nocapture`
    #[test]
    fn real_model_tells_tone_bursts_from_silence_if_env_set() {
        let Ok(model) = std::env::var("LOCAL_SILERO_VAD_MODEL") else {
            eprintln!("LOCAL_SILERO_VAD_MODEL not set; skipping");
            return;
        };
        let mut vad = SileroVad::load(Path::new(&model)).unwrap();
        let silence = vec![0.0f32; WINDOW];
        let quiet: Vec<f32> = (0..20)
            .map(|_| vad.probability(&silence).unwrap())
            .collect();
        assert!(quiet.iter().all(|p| *p < 0.2), "{quiet:?}");
        vad.reset();
        assert!(vad.probability(&[0.0; 10]).is_err());
    }
}
