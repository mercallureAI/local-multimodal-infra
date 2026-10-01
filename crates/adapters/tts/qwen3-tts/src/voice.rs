//! Reference voices: the speaker x-vector and the reference audio's codes.

use crate::{
    artifacts::PackageConfig,
    prompt::{Codes, GROUPS},
};
use local_backend_ort::{
    OrtBackend, OrtSession, OrtTensorData, OrtTensorInput, SessionProviderReport,
};
use local_error::{InfraError, Result};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Voice {
    /// The ECAPA x-vector of the whole reference.
    pub speaker: Vec<f32>,
    /// Codec codes of (the first `max_seconds` of) the reference.
    pub codes: Vec<Codes>,
}

#[derive(Debug)]
pub struct VoiceEncoder {
    speaker: OrtSession,
    codec: OrtSession,
    frame_samples: usize,
}

impl VoiceEncoder {
    pub fn load(
        backend: &OrtBackend,
        speaker: &Path,
        codec: &Path,
        config: &PackageConfig,
    ) -> Result<Self> {
        Ok(Self {
            speaker: backend.load_session(speaker)?,
            codec: backend.load_session(codec)?,
            frame_samples: config.codec_frame_samples,
        })
    }

    pub fn provider_reports(&self) -> [SessionProviderReport; 2] {
        [self.speaker.provider_report(), self.codec.provider_report()]
    }

    /// `audio`: mono 24 kHz. Codes cover at most `max_samples` of it.
    pub fn encode(&mut self, audio: &[f32], max_samples: usize) -> Result<Voice> {
        if audio.len() < self.frame_samples {
            return Err(InfraError::BadRequest(
                "Qwen3-TTS reference audio is shorter than one codec frame (80 ms)".to_string(),
            ));
        }
        let speaker = self.speaker.run_tensors(&[OrtTensorInput {
            name: "audio".to_string(),
            shape: vec![1, audio.len()],
            data: OrtTensorData::F32(audio.to_vec()),
        }])?;
        let speaker = match speaker.into_iter().next().map(|output| output.data) {
            Some(OrtTensorData::F32(values)) => values,
            _ => {
                return Err(InfraError::Adapter(
                    "speaker encoder returned no f32 x-vector".to_string(),
                ))
            }
        };
        // Whole frames: the codec's causal convs pad the last one with zeros.
        let clip = &audio[..audio.len().min(max_samples)];
        let frames = clip.len().div_ceil(self.frame_samples);
        let mut padded = clip.to_vec();
        padded.resize(frames * self.frame_samples, 0.0);
        let codes = self.codec.run_tensors(&[OrtTensorInput {
            name: "audio".to_string(),
            shape: vec![1, padded.len()],
            data: OrtTensorData::F32(padded),
        }])?;
        let output = codes
            .into_iter()
            .next()
            .ok_or_else(|| InfraError::Adapter("codec encoder returned nothing".to_string()))?;
        let OrtTensorData::I64(values) = output.data else {
            return Err(InfraError::Adapter(
                "codec encoder codes are not i64".to_string(),
            ));
        };
        if output.shape.len() != 3 || output.shape[1] != GROUPS || output.shape[2] != frames {
            return Err(InfraError::Adapter(format!(
                "codec encoder returned shape {:?}, expected [1, {GROUPS}, {frames}]",
                output.shape
            )));
        }
        let codes = (0..frames)
            .map(|frame| {
                let mut codes = [0; GROUPS];
                for (group, code) in codes.iter_mut().enumerate() {
                    *code = values[group * frames + frame];
                }
                codes
            })
            .collect();
        Ok(Voice { speaker, codes })
    }
}
