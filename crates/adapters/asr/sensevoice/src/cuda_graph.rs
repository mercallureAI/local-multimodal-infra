//! SenseVoice replayed as CUDA graphs, one per input length bucket.
//!
//! A SenseVoice run launches ~2400 CUDA kernels (70 SANM layers of small
//! ops) and costs ~23 ms whatever the utterance length: kernel launches, not
//! arithmetic. A captured graph replays them in one go: ~8 ms for up to 64
//! LFR frames (3.8 s), ~16 ms for 352 (21 s).
//!
//! A graph needs fixed shapes and every node on the GPU, so each bucket is a
//! session whose `feats_length` is fixed (ORT then folds the shape
//! arithmetic that otherwise runs on the CPU), fed through fixed device
//! buffers. Features are zero-padded to the bucket; `speech_lengths` masks
//! the padding (the same tokens as an unpadded run). The buckets share one
//! device copy of the weights, and return the `ctc_ids` output (argmax taken
//! on the GPU), not the logits.

use crate::features::SenseVoiceFeatures;
use local_backend_ort::{
    CudaSessionOptions, FixedTensorSpec, InitializerRange, OrtBackend, OrtSession, OrtTensorData,
    ProviderKind, ProviderSelection, SharedInitializers, StaticIoBinding, TensorElement,
};
use local_error::{InfraError, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// The greedy CTC token per output frame (int64 [B, T]).
pub const IDS_OUTPUT: &str = "ctc_ids";
/// Output frames SenseVoice puts before the speech (language, emotion,
/// event, text normalization).
const PROMPT_FRAMES: usize = 4;
/// Every bucket's graph has this id (one graph per session).
const GRAPH_ID: i64 = 0;

/// `<stem>.shared.json` beside the graph: its weights' byte ranges in the
/// external data file.
#[derive(Debug, Clone, Deserialize)]
pub struct SharedWeightsManifest {
    pub data_file: String,
    pub tensors: Vec<InitializerRange>,
}

pub fn manifest_path(model: &Path) -> PathBuf {
    let stem = model
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or_default();
    model.with_file_name(format!("{stem}.shared.json"))
}

#[derive(Debug)]
struct Bucket {
    frames: usize,
    session: OrtSession,
    binding: StaticIoBinding,
}

#[derive(Debug)]
pub struct CudaGraphRunner {
    buckets: Vec<Bucket>,
    dim: usize,
}

impl CudaGraphRunner {
    /// One captured session per entry of `frames` (sorted ascending), each
    /// warmed up so the first request replays.
    pub fn load(
        model: &Path,
        weights: &SharedInitializers,
        frames: &[usize],
        dim: usize,
        language: i32,
        textnorm: i32,
    ) -> Result<Self> {
        let mut buckets = Vec::with_capacity(frames.len());
        for &length in frames {
            let backend = OrtBackend::new(ProviderSelection::from_strings(&["cuda".to_string()]))
                .with_cuda_session_options(CudaSessionOptions {
                    cuda_graph: true,
                    ..CudaSessionOptions::default()
                })
                .with_dimension_override("batch_size", 1)
                .with_dimension_override("feats_length", length as i64);
            let session = backend.load_session_with_initializers(model, weights)?;
            if session.provider() != ProviderKind::Cuda {
                return Err(InfraError::Backend(format!(
                    "SenseVoice CUDA graph session for {length} frames is on {:?}",
                    session.provider()
                )));
            }
            let inputs = [
                FixedTensorSpec::new("speech", TensorElement::F32, &[1, length, dim]),
                FixedTensorSpec::new("speech_lengths", TensorElement::I32, &[1]),
                FixedTensorSpec::new("language", TensorElement::I32, &[1]),
                FixedTensorSpec::new("textnorm", TensorElement::I32, &[1]),
            ];
            let outputs = [
                FixedTensorSpec::new(IDS_OUTPUT, TensorElement::I64, &[1, length + PROMPT_FRAMES]),
                FixedTensorSpec::new("encoder_out_lens", TensorElement::I32, &[1]),
            ];
            let mut binding =
                session.create_static_binding(&[], [1, 1, 1, 1], &inputs, &outputs)?;
            binding.write("language", &OrtTensorData::I32(vec![language]))?;
            binding.write("textnorm", &OrtTensorData::I32(vec![textnorm]))?;
            binding.write("speech", &OrtTensorData::F32(vec![0.0; length * dim]))?;
            binding.write("speech_lengths", &OrtTensorData::I32(vec![length as i32]))?;
            buckets.push(Bucket {
                frames: length,
                session,
                binding,
            });
        }
        let mut runner = Self { buckets, dim };
        for bucket in &mut runner.buckets {
            // ORT captures after its first runs: take them now.
            for _ in 0..4 {
                bucket
                    .session
                    .run_static_fixed(&mut bucket.binding, Some(GRAPH_ID))?;
            }
        }
        Ok(runner)
    }

    pub fn frames(&self) -> Vec<usize> {
        self.buckets.iter().map(|bucket| bucket.frames).collect()
    }

    /// The CTC ids of `features` (valid frames only), from the smallest
    /// bucket that holds them; `None` when they exceed every bucket.
    pub fn run(
        &mut self,
        features: &SenseVoiceFeatures,
        language: i32,
    ) -> Result<Option<Vec<i64>>> {
        if features.dim != self.dim {
            return Err(InfraError::Adapter(format!(
                "SenseVoice features have dim {}, the CUDA graphs {}",
                features.dim, self.dim
            )));
        }
        let Some(bucket) = self
            .buckets
            .iter_mut()
            .find(|bucket| bucket.frames >= features.frames)
        else {
            return Ok(None);
        };
        let Bucket {
            frames,
            session,
            binding,
        } = bucket;
        let mut speech = Vec::with_capacity(*frames * features.dim);
        speech.extend_from_slice(&features.data);
        speech.resize(*frames * features.dim, 0.0);
        binding.write("speech", &OrtTensorData::F32(speech))?;
        binding.write(
            "speech_lengths",
            &OrtTensorData::I32(vec![features.frames as i32]),
        )?;
        binding.write("language", &OrtTensorData::I32(vec![language]))?;
        session.run_static_fixed(binding, Some(GRAPH_ID))?;
        let OrtTensorData::I32(length) = binding.read("encoder_out_lens")?.data else {
            return Err(InfraError::Adapter(
                "SenseVoice encoder_out_lens is not I32".to_string(),
            ));
        };
        let OrtTensorData::I64(mut ids) = binding.read(IDS_OUTPUT)?.data else {
            return Err(InfraError::Adapter(format!(
                "SenseVoice {IDS_OUTPUT} is not I64"
            )));
        };
        let valid = length
            .first()
            .and_then(|&length| usize::try_from(length).ok())
            .filter(|&length| length <= ids.len())
            .ok_or_else(|| {
                InfraError::Adapter(format!(
                    "SenseVoice encoder_out_lens {length:?} does not fit {} ids",
                    ids.len()
                ))
            })?;
        ids.truncate(valid);
        Ok(Some(ids))
    }
}
