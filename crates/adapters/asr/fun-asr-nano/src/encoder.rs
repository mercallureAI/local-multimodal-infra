//! The audio graph: SenseVoice encoder, adaptor and CTC decoder.
//!
//! Like SenseVoice, a run is ~2400 small kernels, so on CUDA each input
//! length bucket (LFR frames) is a fixed-length session replayed as a CUDA
//! graph, all sharing one device copy of the weights; features are padded to
//! the bucket and `speech_lengths` masks the padding. Longer inputs, and CPU
//! sessions, run the plain session.

use local_adapter_sensevoice_asr::SenseVoiceFeatures;
use local_backend_ort::{
    CudaSessionOptions, FixedTensorSpec, InitializerRange, OrtBackend, OrtSession, OrtTensorData,
    OrtTensorInput, OrtTensorOutput, ProviderKind, ProviderSelection, SessionProviderReport,
    SharedInitializers, StaticIoBinding, TensorElement,
};
use local_error::{InfraError, Result};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

const EMBEDS_OUTPUT: &str = "audio_embeds";
const IDS_OUTPUT: &str = "ctc_ids";
const LENGTHS_OUTPUT: &str = "encoder_out_lens";
/// Every bucket's graph has this id (one graph per session).
const GRAPH_ID: i64 = 0;

/// `<stem>.shared.json` beside the graph: its weights' byte ranges in the
/// external data file.
#[derive(Debug, Clone, Deserialize)]
struct SharedWeightsManifest {
    data_file: String,
    tensors: Vec<InitializerRange>,
}

fn manifest_path(model: &Path) -> PathBuf {
    let stem = model
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or_default();
    model.with_file_name(format!("{stem}.shared.json"))
}

/// One segment through the audio graph (valid frames only).
#[derive(Debug, Clone)]
pub struct AudioOutput {
    pub frames: usize,
    /// Adaptor output, `frames` x hidden.
    pub embeds: Vec<f32>,
    /// Greedy CTC token per frame.
    pub ctc_ids: Vec<i64>,
}

#[derive(Debug)]
struct Bucket {
    frames: usize,
    session: OrtSession,
    binding: StaticIoBinding,
}

#[derive(Debug)]
pub struct AudioEncoder {
    session: OrtSession,
    buckets: Vec<Bucket>,
    feature_dim: usize,
    hidden: usize,
    /// Last: the weights the sessions share outlive them.
    _weights: Option<SharedInitializers>,
}

impl AudioEncoder {
    /// The graph at `model` (its weights uploaded once when it has a
    /// `.shared.json`) and, on CUDA, one captured session per entry of
    /// `graph_frames`. `metadata` may set the CUDA memory options
    /// (`cuda_arena_same_as_requested`, `cuda_conv_max_workspace`).
    pub fn load(
        model: &Path,
        providers: &ProviderSelection,
        graph_frames: &[usize],
        feature_dim: usize,
        hidden: usize,
        metadata: &BTreeMap<String, serde_json::Value>,
    ) -> Result<Self> {
        let backend = OrtBackend::new(providers.clone()).with_cuda_memory_metadata(metadata);
        let manifest = manifest_path(model);
        let weights = if manifest.is_file() {
            let manifest: SharedWeightsManifest = serde_json::from_slice(
                &fs::read(&manifest).map_err(|e| InfraError::io(Some(manifest.clone()), e))?,
            )?;
            let data = model.with_file_name(&manifest.data_file);
            Some(backend.upload_initializers(&data, &manifest.tensors)?)
        } else {
            None
        };
        let session = match &weights {
            Some(weights) => backend.load_session_with_initializers(model, weights)?,
            None => backend.load_session(model)?,
        };
        for name in [EMBEDS_OUTPUT, IDS_OUTPUT, LENGTHS_OUTPUT] {
            if !session.outputs().iter().any(|output| output.name == name) {
                return Err(InfraError::Adapter(format!(
                    "Fun-ASR-Nano audio graph {} has no `{name}` output",
                    model.display()
                )));
            }
        }
        let mut encoder = Self {
            session,
            buckets: Vec::new(),
            feature_dim,
            hidden,
            _weights: None,
        };
        if let Some(weights) = &weights {
            if !graph_frames.is_empty()
                && weights.cuda_device().is_some()
                && encoder.session.provider() == ProviderKind::Cuda
            {
                for &frames in graph_frames {
                    encoder.buckets.push(load_bucket(
                        model,
                        weights,
                        frames,
                        feature_dim,
                        hidden,
                        metadata,
                    )?);
                }
            }
        }
        encoder._weights = weights;
        Ok(encoder)
    }

    pub fn provider_report(&self) -> SessionProviderReport {
        self.session.provider_report()
    }

    /// The input lengths (LFR frames) replayed as CUDA graphs.
    pub fn graph_frames(&self) -> Vec<usize> {
        self.buckets.iter().map(|bucket| bucket.frames).collect()
    }

    pub fn run(&mut self, features: &SenseVoiceFeatures) -> Result<AudioOutput> {
        if features.dim != self.feature_dim {
            return Err(InfraError::Adapter(format!(
                "Fun-ASR-Nano features have dim {}, the audio graph takes {}",
                features.dim, self.feature_dim
            )));
        }
        if let Some(bucket) = self
            .buckets
            .iter_mut()
            .find(|bucket| bucket.frames >= features.frames)
        {
            return run_bucket(bucket, features, self.hidden);
        }
        self.run_session(features)
    }

    /// The plain session, without the CUDA graphs.
    pub fn run_session(&mut self, features: &SenseVoiceFeatures) -> Result<AudioOutput> {
        let inputs = [
            OrtTensorInput {
                name: "speech".to_string(),
                shape: vec![1, features.frames, features.dim],
                data: OrtTensorData::F32(features.data.clone()),
            },
            OrtTensorInput {
                name: "speech_lengths".to_string(),
                shape: vec![1],
                data: OrtTensorData::I32(vec![features.frames as i32]),
            },
        ];
        let outputs = self.session.run_tensors(&inputs)?;
        let take = |name: &str| -> Result<OrtTensorOutput> {
            outputs
                .iter()
                .find(|output| output.name == name)
                .cloned()
                .ok_or_else(|| InfraError::Adapter(format!("audio graph returned no `{name}`")))
        };
        collect(
            take(EMBEDS_OUTPUT)?.data,
            take(IDS_OUTPUT)?.data,
            take(LENGTHS_OUTPUT)?.data,
            self.hidden,
        )
    }
}

fn load_bucket(
    model: &Path,
    weights: &SharedInitializers,
    frames: usize,
    dim: usize,
    hidden: usize,
    metadata: &BTreeMap<String, serde_json::Value>,
) -> Result<Bucket> {
    let backend = OrtBackend::new(ProviderSelection::from_strings(&["cuda".to_string()]))
        .with_cuda_memory_metadata(metadata)
        .with_cuda_session_options(CudaSessionOptions {
            cuda_graph: true,
            ..CudaSessionOptions::default()
        })
        .with_dimension_override("batch_size", 1)
        .with_dimension_override("feats_length", frames as i64);
    let session = backend.load_session_with_initializers(model, weights)?;
    if session.provider() != ProviderKind::Cuda {
        return Err(InfraError::Backend(format!(
            "Fun-ASR-Nano CUDA graph session for {frames} frames is on {:?}",
            session.provider()
        )));
    }
    let inputs = [
        FixedTensorSpec::new("speech", TensorElement::F32, &[1, frames, dim]),
        FixedTensorSpec::new("speech_lengths", TensorElement::I32, &[1]),
    ];
    let outputs = [
        FixedTensorSpec::new(EMBEDS_OUTPUT, TensorElement::F32, &[1, frames, hidden]),
        FixedTensorSpec::new(IDS_OUTPUT, TensorElement::I64, &[1, frames]),
        FixedTensorSpec::new(LENGTHS_OUTPUT, TensorElement::I32, &[1]),
    ];
    let mut binding = session.create_static_binding(&[], [1, 1, 1, 1], &inputs, &outputs)?;
    binding.write("speech", &OrtTensorData::F32(vec![0.0; frames * dim]))?;
    binding.write("speech_lengths", &OrtTensorData::I32(vec![frames as i32]))?;
    let mut bucket = Bucket {
        frames,
        session,
        binding,
    };
    // ORT captures after its first runs: take them now.
    for _ in 0..4 {
        bucket
            .session
            .run_static_fixed(&mut bucket.binding, Some(GRAPH_ID))?;
    }
    Ok(bucket)
}

fn run_bucket(
    bucket: &mut Bucket,
    features: &SenseVoiceFeatures,
    hidden: usize,
) -> Result<AudioOutput> {
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
    session.run_static_fixed(binding, Some(GRAPH_ID))?;
    collect(
        binding.read(EMBEDS_OUTPUT)?.data,
        binding.read(IDS_OUTPUT)?.data,
        binding.read(LENGTHS_OUTPUT)?.data,
        hidden,
    )
}

/// The valid frames of the three outputs.
fn collect(
    embeds: OrtTensorData,
    ids: OrtTensorData,
    lengths: OrtTensorData,
    hidden: usize,
) -> Result<AudioOutput> {
    let (OrtTensorData::F32(mut embeds), OrtTensorData::I64(mut ids), OrtTensorData::I32(lengths)) =
        (embeds, ids, lengths)
    else {
        return Err(InfraError::Adapter(
            "Fun-ASR-Nano audio graph outputs have unexpected types".to_string(),
        ));
    };
    let frames = lengths
        .first()
        .and_then(|&length| usize::try_from(length).ok())
        .filter(|&frames| frames <= ids.len() && frames * hidden <= embeds.len())
        .ok_or_else(|| {
            InfraError::Adapter(format!(
                "Fun-ASR-Nano encoder_out_lens {lengths:?} does not fit {} frames",
                ids.len()
            ))
        })?;
    ids.truncate(frames);
    embeds.truncate(frames * hidden);
    Ok(AudioOutput {
        frames,
        embeds,
        ctc_ids: ids,
    })
}
