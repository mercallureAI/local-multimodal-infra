//! Native Rust/ORT SenseVoiceSmall adapter.
//!
//! The frontend, LFR+CMVN, four-input ONNX invocation and CTC collapse follow
//! the official FunASR `runtime/onnxruntime/src/sensevoice-small.cpp` path.
//!
//! The FunASR pipeline around the recognizer (streamed decoding, FSMN-VAD,
//! FBANK/LFR features, CAM++ speakers, the timeline) is [`SpeechPipeline`];
//! other FunASR recognizers (Fun-ASR-Nano) plug into it through
//! [`SegmentRecognizer`].

mod audio;
mod cuda_graph;
mod features;
mod speaker;
mod vad;

pub use audio::{read_audio_mono_16k, AudioStream, Resampler, TARGET_SAMPLE_RATE};
use features::{Cmvn, SenseVoiceConfigFile};
pub use features::{FrontendConfig, SenseVoiceFeatures};
use local_backend_ort::{
    OrtBackend, OrtSession, OrtTensorData, OrtTensorInput, OrtTensorOutput, ProviderKind,
    ProviderSelection, SessionProviderReport, SharedInitializers, TensorElement,
};
use local_core::{AsrSegment, AsrSpeaker, AsrTokenTimestamp, FileRef, InferenceOutput, ModelSpec};
use local_error::{InfraError, Result};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::mpsc,
};

const MODEL_FILE: &str = "model_quant.onnx";
/// Metadata naming the SenseVoice graph in `asr/` instead of [`MODEL_FILE`]
/// (the dynamic int8 graph): e.g. the weight-only int8 export, several times
/// faster on CUDA.
const ASR_MODEL_FILE_KEY: &str = "asr_model_file";
/// The float16 graph earlier installs exported: inf/NaN on CUDA for loud
/// speech, replaced by `model_gpu_int8.onnx`.
const FLOAT16_MODEL_FILE: &str = "model_fp16.onnx";
const CONFIG_FILE: &str = "config.yaml";
const CMVN_FILE: &str = "am.mvn";
const TOKENS_FILE: &str = "tokens.json";
const BLANK_ID: usize = 0;
const LANGUAGE_AUTO_ID: i32 = 0;
/// SenseVoice's language prompt ids (FunASR's `lid_dict`).
const LANGUAGE_IDS: [(&str, i32); 6] = [
    ("auto", LANGUAGE_AUTO_ID),
    ("zh", 3),
    ("en", 4),
    ("yue", 7),
    ("ja", 11),
    ("ko", 12),
];
const WITH_ITN_ID: i32 = 14;
const DEFAULT_SPEAKER_BATCH_SIZE: usize = 32;
const MAX_SPEAKER_BATCH_SIZE: usize = 256;
const DEFAULT_TIMESTAMP_GRANULARITY_MS: u64 = 10_000;
const MIN_TIMESTAMP_GRANULARITY_MS: u64 = 1_000;
const MAX_TIMESTAMP_GRANULARITY_MS: u64 = 120_000;
/// SenseVoice was trained on utterances up to about 30 s.
const MAX_SEGMENT_SECONDS: f64 = 30.0;
/// Metadata listing the LFR frame counts (input lengths) SenseVoice runs as
/// CUDA graphs for, e.g. `[64, 128, 256, 352]`; longer inputs run normally.
/// Needs a graph exported with `ctc_ids` and its weights' `.shared.json`.
const CUDA_GRAPH_FRAMES_KEY: &str = "asr_cuda_graph_frames";
const MAX_CUDA_GRAPH_FRAMES: usize = 1_024;

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
        Self::validate_with_model(root, MODEL_FILE)
    }

    /// As [`Self::validate`], the SenseVoice graph being `asr/<model_file>`.
    pub fn validate_with_model(root: impl AsRef<Path>, model_file: &str) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let asr_root = root.join("asr");
        let vad_root = root.join("vad");
        let speaker_root = root.join("speaker");
        let artifacts = Self {
            model: asr_root.join(model_file),
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
                // An install from before the float16 graph was replaced.
                let hint = if path == &artifacts.model
                    && artifacts.asr_root.join(FLOAT16_MODEL_FILE).is_file()
                {
                    format!(
                        "; convert the {FLOAT16_MODEL_FILE} there with `python -m scripts.local.sensevoice_cuda_export from-fp16 --model-dir {}`",
                        artifacts.root.display()
                    )
                } else {
                    String::new()
                };
                return Err(InfraError::ModelNotConfigured {
                    model_id: "sensevoice-small-onnx".to_string(),
                    reason: format!(
                        "required SenseVoice artifact is missing: {}{hint}",
                        path.display()
                    ),
                });
            }
        }
        Ok(artifacts)
    }

    fn from_spec(spec: &ModelSpec) -> Result<Self> {
        let model_file = match spec.metadata.get(ASR_MODEL_FILE_KEY) {
            None => MODEL_FILE,
            Some(value) => value
                .as_str()
                .filter(|name| {
                    !name.is_empty() && Path::new(name).file_name() == Some(name.as_ref())
                })
                .ok_or_else(|| InfraError::ModelNotConfigured {
                    model_id: spec.id.clone(),
                    reason: format!("metadata `{ASR_MODEL_FILE_KEY}` must be a file name in asr/"),
                })?,
        };
        let asr_model = spec
            .artifacts
            .iter()
            .map(|artifact| &artifact.path)
            .find(|path| {
                path.file_name().and_then(|name| name.to_str()) == Some(model_file)
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
        Self::validate_with_model(root, model_file).map_err(|error| match error {
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
    pipeline: SpeechPipeline,
    asr: SenseVoiceRecognizer,
    /// Last: the weights the recognizer's sessions share outlive them.
    _weights: Option<SharedInitializers>,
}

/// One speech segment's recognizer: its FBANK/LFR features in, text out;
/// `None` when it judges the segment not to be speech (it is then left out,
/// as a segment too short for one feature frame is).
pub trait SegmentRecognizer {
    fn recognize(&mut self, features: SenseVoiceFeatures) -> Result<Option<RecognizedText>>;
}

/// The FunASR pipeline around a recognizer: streamed audio decoding and
/// FSMN-VAD, FBANK/LFR (+ CMVN) features, CAM++ speaker labels and the
/// timestamped output. The VAD and speaker graphs are `vad/` and `speaker/`
/// of the model directory.
#[derive(Debug)]
pub struct SpeechPipeline {
    config: SenseVoiceConfigFile,
    cmvn: Option<Cmvn>,
    vad: vad::FsmnVad,
    speaker: speaker::CampPlus,
}

/// The files [`SpeechPipeline::load`] reads from a model directory besides
/// the frontend configuration.
pub fn pipeline_files(root: &Path) -> [PathBuf; 4] {
    let vad = root.join("vad");
    [
        vad.join(MODEL_FILE),
        vad.join(CONFIG_FILE),
        vad.join(CMVN_FILE),
        root.join("speaker").join("campplus_cn_en_common_200k.onnx"),
    ]
}

impl SpeechPipeline {
    /// The pipeline of the model directory `root`: the frontend of `config`
    /// (a FunASR `config.yaml`), CMVN from `cmvn` when given, and the spec's
    /// VAD/speaker metadata (`vad_provider_order`, `vad_max_segment_seconds`,
    /// `speaker_provider_order`, `speaker_batch_size`, `speaker_io_binding`).
    pub fn load(spec: &ModelSpec, root: &Path, config: &Path, cmvn: Option<&Path>) -> Result<Self> {
        let config = SenseVoiceConfigFile::load(config)?;
        let feature_dim = config.frontend_conf.n_mels * config.frontend_conf.lfr_m;
        let cmvn = cmvn.map(|path| Cmvn::load(path, feature_dim)).transpose()?;
        let providers = ProviderSelection::from_strings(&spec.runtime.provider_order);
        let vad_providers = metadata_provider_selection(spec, "vad_provider_order")?
            .unwrap_or_else(ProviderSelection::default);
        let speaker_providers =
            metadata_provider_selection(spec, "speaker_provider_order")?.unwrap_or(providers);
        let vad_max_segment_ms = metadata_seconds(spec, "vad_max_segment_seconds")?
            .map(|seconds| (seconds * 1000.0).round() as usize);
        let vad = vad::FsmnVad::load(&root.join("vad"), &vad_providers, vad_max_segment_ms)?;
        let speaker_batch_size = metadata_usize(
            spec,
            "speaker_batch_size",
            DEFAULT_SPEAKER_BATCH_SIZE,
            MAX_SPEAKER_BATCH_SIZE,
        )?;
        let speaker_io_binding = metadata_bool(spec, "speaker_io_binding", true)?;
        let speaker = speaker::CampPlus::load(
            &root.join("speaker"),
            &speaker_providers,
            speaker_batch_size,
            speaker_io_binding,
        )?;
        Ok(Self {
            config,
            cmvn,
            vad,
            speaker,
        })
    }

    pub fn frontend(&self) -> &FrontendConfig {
        &self.config.frontend_conf
    }

    /// The recognizer input for `samples` (16 kHz mono, one segment).
    pub fn extract_features(&self, samples: &[f32]) -> Result<SenseVoiceFeatures> {
        features::extract(samples, &self.config.frontend_conf, self.cmvn.as_ref())
    }

    /// The length of one LFR frame (one recognizer output frame for CTC).
    pub fn output_frame_ms(&self) -> u64 {
        let frontend = &self.config.frontend_conf;
        frontend.frame_shift as u64 * frontend.lfr_n as u64
    }

    pub fn vad_provider_report(&self) -> SessionProviderReport {
        self.vad.provider_report()
    }

    /// The longest speech segment a recognizer gets (the VAD cuts there).
    pub fn max_segment_ms(&self) -> usize {
        self.vad.max_segment_ms()
    }

    pub fn speaker_provider_report(&self) -> SessionProviderReport {
        self.speaker.provider_report()
    }

    /// Transcribes a recording as a stream: decoding + FSMN-VAD, features,
    /// and the recognizer/CAM++ runs are three threads joined by short
    /// queues, so they overlap and memory follows the longest speech segment
    /// rather than the recording. Only the speaker clustering waits for the
    /// end.
    pub fn transcribe(
        &mut self,
        audio: &FileRef,
        params: &BTreeMap<String, serde_json::Value>,
        recognizer: &mut dyn SegmentRecognizer,
    ) -> Result<InferenceOutput> {
        let options = AsrOptions::from_params(params)?;
        let path = local_files::local_path(audio)?;
        let with_speakers = options.speaker_diarization || options.speaker_embedding;
        let Self {
            config,
            cmvn,
            vad,
            speaker,
        } = self;
        let frontend = &config.frontend_conf;
        let cmvn = cmvn.as_ref();
        let (decoded, chunks, embeddings) = std::thread::scope(|scope| {
            let (segment_tx, segment_rx) = mpsc::sync_channel::<SegmentAudio>(PIPELINE_QUEUE);
            let (prepared_tx, prepared_rx) = mpsc::sync_channel::<PreparedSegment>(PIPELINE_QUEUE);
            let reader = scope.spawn(move || read_segments(&path, vad, segment_tx));
            let preparer = scope.spawn(move || -> Result<()> {
                for SegmentAudio { segment, samples } in segment_rx {
                    let features = features::extract(&samples, frontend, cmvn)?;
                    let chunks = if with_speakers {
                        speaker::segment_chunks(&samples, segment)?
                    } else {
                        Vec::new()
                    };
                    let prepared = PreparedSegment {
                        segment,
                        features,
                        chunks,
                    };
                    if prepared_tx.send(prepared).is_err() {
                        break;
                    }
                }
                Ok(())
            });
            let decoded = decode_segments(recognizer, speaker, prepared_rx);
            // A failed stage closes its queues, which ends the others.
            let read = reader.join().expect("audio reader thread panicked");
            let prepare = preparer.join().expect("feature thread panicked");
            read.and(prepare).and(decoded)
        })?;
        // Segments left out (no text) take no part in the speakers.
        let kept = decoded
            .iter()
            .map(|segment| segment.text.is_some())
            .collect::<Vec<_>>();
        let (chunks, embeddings) = kept_speaker_chunks(&decoded, chunks, embeddings);

        let mut speaker_labels = if options.speaker_diarization {
            speaker::label_segments(
                &decoded
                    .iter()
                    .map(|segment| segment.segment)
                    .collect::<Vec<_>>(),
                &chunks,
                &embeddings,
            )
        } else {
            vec![0; decoded.len()]
        };
        number_kept_speakers(&mut speaker_labels, &kept);
        let speaker_embedding = options
            .speaker_embedding
            .then(|| speaker::mean_embedding(&embeddings))
            .flatten();
        let mut merged = String::new();
        let mut segments = Vec::with_capacity(decoded.len());
        for (index, DecodedSegment { segment, text }) in decoded.into_iter().enumerate() {
            let Some(text) = text else {
                continue;
            };
            append_transcript(&mut merged, text.text.trim());
            let speaker = options
                .speaker_diarization
                .then(|| format!("speaker_{}", speaker_labels[index]));
            segments.extend(split_decoded_segment(
                samples_to_ms(segment.start_sample),
                samples_to_ms(segment.end_sample),
                speaker,
                text,
                options.timestamp_granularity_ms,
                options.token_timestamps,
            ));
        }
        let speakers = summarize_speakers(&segments);
        if !options.timestamps {
            segments.clear();
        } else {
            segments = merge_timeline_segments(segments, options.timestamp_granularity_ms);
        }
        let timestamped_text = options
            .timestamps
            .then(|| format_timestamped_text(&segments));
        let timestamped_text = timestamped_text.filter(|text| !text.is_empty());
        Ok(InferenceOutput::AsrTranscription {
            text: merged,
            timestamped_text,
            segments,
            speakers,
            speaker_embedding,
        })
    }
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
        let pipeline = SpeechPipeline::load(
            spec,
            &artifacts.root,
            &artifacts.config,
            Some(&artifacts.cmvn),
        )?;
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
        let graph_frames = metadata_frames(spec, CUDA_GRAPH_FRAMES_KEY)?;
        // Weights uploaded once and shared by every session of the graph.
        let manifest = cuda_graph::manifest_path(&artifacts.model);
        let weights = if manifest.is_file() {
            let manifest: cuda_graph::SharedWeightsManifest = serde_json::from_slice(
                &fs::read(&manifest).map_err(|e| InfraError::io(Some(manifest.clone()), e))?,
            )?;
            Some(backend.upload_initializers(
                &artifacts.asr_root.join(&manifest.data_file),
                &manifest.tensors,
            )?)
        } else {
            None
        };
        let session = match &weights {
            Some(weights) => backend.load_session_with_initializers(&artifacts.model, weights)?,
            None => backend.load_session(&artifacts.model)?,
        };
        validate_session(&session)?;
        let frontend = pipeline.frontend();
        let feature_dim = frontend.n_mels * frontend.lfr_m;
        let graphs = match &weights {
            Some(weights)
                if !graph_frames.is_empty()
                    && weights.cuda_device().is_some()
                    && session.provider() == ProviderKind::Cuda
                    && session
                        .outputs()
                        .iter()
                        .any(|output| output.name == cuda_graph::IDS_OUTPUT) =>
            {
                Some(cuda_graph::CudaGraphRunner::load(
                    &artifacts.model,
                    weights,
                    &graph_frames,
                    feature_dim,
                    LANGUAGE_AUTO_ID,
                    WITH_ITN_ID,
                )?)
            }
            _ => None,
        };
        let asr = SenseVoiceRecognizer {
            session,
            graphs,
            tokens,
            output_frame_ms: pipeline.output_frame_ms(),
            language: LANGUAGE_AUTO_ID,
        };
        Ok(Self {
            model_id: spec.id.clone(),
            artifacts,
            pipeline,
            asr,
            _weights: weights,
        })
    }

    pub fn transcribe(&mut self, audio: &FileRef) -> Result<InferenceOutput> {
        self.transcribe_with_params(audio, &BTreeMap::new())
    }

    /// Transcribes a recording as a stream (see [`SpeechPipeline::transcribe`]).
    pub fn transcribe_with_params(
        &mut self,
        audio: &FileRef,
        params: &BTreeMap<String, serde_json::Value>,
    ) -> Result<InferenceOutput> {
        self.pipeline.transcribe(audio, params, &mut self.asr)
    }

    /// One speech segment (16 kHz mono samples, no VAD): its text, tags and
    /// token times (relative to the segment start); empty for audio too
    /// short for one feature frame.
    pub fn recognize_samples(&mut self, samples: &[f32]) -> Result<RecognizedText> {
        let features = self.pipeline.extract_features(samples)?;
        if features.frames == 0 {
            return Ok(RecognizedText::default());
        }
        Ok(self.asr.recognize(features)?.unwrap_or_default())
    }

    /// [`Self::recognize_samples`] told the spoken language (`zh`, `en`,
    /// `yue`, `ja`, `ko`; `None` or `auto` detects it).
    pub fn recognize_samples_with_language(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
    ) -> Result<RecognizedText> {
        let id = language_id(language)?;
        let features = self.pipeline.extract_features(samples)?;
        if features.frames == 0 {
            return Ok(RecognizedText::default());
        }
        self.asr.language = id;
        let recognized = self.asr.recognize(features);
        self.asr.language = LANGUAGE_AUTO_ID;
        Ok(recognized?.unwrap_or_default())
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn artifacts(&self) -> &SenseVoiceArtifacts {
        &self.artifacts
    }

    pub fn provider_report(&self) -> SessionProviderReport {
        self.asr.session.provider_report()
    }

    /// The input lengths (LFR frames) replayed as CUDA graphs.
    pub fn cuda_graph_frames(&self) -> Vec<usize> {
        self.asr
            .graphs
            .as_ref()
            .map(cuda_graph::CudaGraphRunner::frames)
            .unwrap_or_default()
    }

    pub fn pipeline_provider_report(&self) -> SenseVoicePipelineProviderReport {
        SenseVoicePipelineProviderReport {
            asr: self.asr.session.provider_report(),
            vad: self.pipeline.vad_provider_report(),
            speaker: self.pipeline.speaker_provider_report(),
        }
    }

    #[cfg(test)]
    fn infer_features(&mut self, features: SenseVoiceFeatures) -> Result<RecognizedText> {
        Ok(self
            .asr
            .recognize(features)?
            .expect("SenseVoice always gives text"))
    }
}

/// Queue length between pipeline stages (speech segments of up to ~20 s).
const PIPELINE_QUEUE: usize = 4;

/// One speech segment and its audio.
struct SegmentAudio {
    segment: vad::SpeechSegment,
    samples: Vec<f32>,
}

/// One speech segment ready for the models.
struct PreparedSegment {
    segment: vad::SpeechSegment,
    features: SenseVoiceFeatures,
    chunks: Vec<speaker::SpeakerChunk>,
}

struct DecodedSegment {
    segment: vad::SpeechSegment,
    /// `None` for a segment too short for one feature frame.
    text: Option<RecognizedText>,
}

/// Stage one: decodes `path` and sends each speech segment with its audio
/// as soon as the VAD closes it, keeping only the audio a segment still
/// open may need.
fn read_segments(
    path: &Path,
    vad: &mut vad::FsmnVad,
    segments: mpsc::SyncSender<SegmentAudio>,
) -> Result<()> {
    let mut stream = audio::AudioStream::open(path)?;
    let mut vad_stream = vad.start();
    // Audio from sample `kept_start` on.
    let mut kept = Vec::<f32>::new();
    let mut kept_start = 0usize;
    let mut chunk = Vec::new();
    loop {
        chunk.clear();
        let more = stream.read(&mut chunk)?;
        kept.extend_from_slice(&chunk);
        let closed = if more {
            vad.push(&mut vad_stream, &chunk)?
        } else {
            vad.finish(&mut vad_stream)?
        };
        for segment in closed {
            let samples =
                kept[segment.start_sample - kept_start..segment.end_sample - kept_start].to_vec();
            if segments.send(SegmentAudio { segment, samples }).is_err() {
                return Ok(());
            }
        }
        if !more {
            return Ok(());
        }
        let retain = vad_stream.retain_from_sample();
        // Drain in steps of at least a second, not every packet.
        if retain >= kept_start + audio::TARGET_SAMPLE_RATE as usize {
            kept.drain(..retain - kept_start);
            kept_start = retain;
        }
    }
}

/// Stage three: SenseVoice per segment and CAM++ per `batch_size` speaker
/// chunks; returns the segments' texts, the chunks (their FBANK dropped)
/// and the chunks' embeddings.
#[allow(clippy::type_complexity)]
fn decode_segments(
    asr: &mut dyn SegmentRecognizer,
    speaker: &mut speaker::CampPlus,
    prepared: mpsc::Receiver<PreparedSegment>,
) -> Result<(
    Vec<DecodedSegment>,
    Vec<speaker::SpeakerChunk>,
    Vec<Vec<f32>>,
)> {
    let mut decoded = Vec::new();
    let mut chunks = Vec::new();
    let mut pending_chunks = Vec::new();
    let mut embeddings = Vec::new();
    for PreparedSegment {
        segment,
        features,
        chunks: segment_chunks,
    } in prepared
    {
        let text = if features.frames == 0 {
            None
        } else {
            asr.recognize(features)?
        };
        decoded.push(DecodedSegment { segment, text });
        pending_chunks.extend(segment_chunks);
        if pending_chunks.len() >= speaker.batch_size() {
            let batch = pending_chunks
                .drain(..speaker.batch_size())
                .collect::<Vec<_>>();
            embeddings.extend(speaker.embed(&batch)?);
            chunks.extend(batch.iter().map(speaker::SpeakerChunk::range));
        }
    }
    embeddings.extend(speaker.embed(&pending_chunks)?);
    chunks.extend(pending_chunks.iter().map(speaker::SpeakerChunk::range));
    Ok((decoded, chunks, embeddings))
}

/// The SenseVoice graph: CUDA graph buckets where they fit, the session
/// otherwise.
#[derive(Debug)]
struct SenseVoiceRecognizer {
    session: OrtSession,
    graphs: Option<cuda_graph::CudaGraphRunner>,
    tokens: Vec<String>,
    output_frame_ms: u64,
    /// The language prompt id of the next segments.
    language: i32,
}

impl SenseVoiceRecognizer {
    /// The session alone, without the CUDA graphs.
    fn infer_session(&mut self, features: SenseVoiceFeatures) -> Result<RecognizedText> {
        let inputs = bind_inputs(&self.session, features, self.language)?;
        let outputs = self.session.run_tensors(&inputs).map_err(|error| {
            InfraError::Adapter(format!(
                "SenseVoice ORT execution failed: {error}; {}",
                format_session_io(&self.session)
            ))
        })?;
        decode_outputs(outputs, &self.tokens, self.output_frame_ms)
    }
}

impl SegmentRecognizer for SenseVoiceRecognizer {
    fn recognize(&mut self, features: SenseVoiceFeatures) -> Result<Option<RecognizedText>> {
        if let Some(graphs) = self.graphs.as_mut() {
            if let Some(ids) = graphs.run(&features, self.language)? {
                return decode_ids(&ids, &self.tokens, self.output_frame_ms).map(Some);
            }
        }
        self.infer_session(features).map(Some)
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

/// A list of input lengths in LFR frames (1..=[`MAX_CUDA_GRAPH_FRAMES`]),
/// sorted and deduplicated; empty when absent.
fn metadata_frames(spec: &ModelSpec, key: &str) -> Result<Vec<usize>> {
    let Some(value) = spec.metadata.get(key) else {
        return Ok(Vec::new());
    };
    let invalid = || {
        InfraError::Adapter(format!(
            "model metadata `{key}` must be a list of frame counts in 1..={MAX_CUDA_GRAPH_FRAMES}, got {value}"
        ))
    };
    let mut frames = value
        .as_array()
        .ok_or_else(invalid)?
        .iter()
        .map(|frames| {
            frames
                .as_u64()
                .and_then(|frames| usize::try_from(frames).ok())
                .filter(|frames| (1..=MAX_CUDA_GRAPH_FRAMES).contains(frames))
                .ok_or_else(invalid)
        })
        .collect::<Result<Vec<_>>>()?;
    frames.sort_unstable();
    frames.dedup();
    Ok(frames)
}

/// A positive number of seconds, at most [`MAX_SEGMENT_SECONDS`].
fn metadata_seconds(spec: &ModelSpec, key: &str) -> Result<Option<f64>> {
    let Some(value) = spec.metadata.get(key) else {
        return Ok(None);
    };
    value
        .as_f64()
        .filter(|seconds| seconds.is_finite() && *seconds >= 1.0 && *seconds <= MAX_SEGMENT_SECONDS)
        .map(Some)
        .ok_or_else(|| {
            InfraError::Adapter(format!(
                "model metadata `{key}` must be a number of seconds in 1..={MAX_SEGMENT_SECONDS}, got {value}"
            ))
        })
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
    speaker_embedding: bool,
    timestamps: bool,
    timestamp_granularity_ms: u64,
    token_timestamps: bool,
}

impl AsrOptions {
    fn from_params(params: &BTreeMap<String, serde_json::Value>) -> Result<Self> {
        Ok(Self {
            speaker_diarization: bool_param(
                params,
                &[
                    "speaker_diarization",
                    "diarization",
                    "speaker_identification",
                ],
                true,
            )?,
            speaker_embedding: bool_param(params, &["speaker_embedding"], false)?,
            timestamps: bool_param(params, &["timestamps", "return_timestamps"], true)?,
            timestamp_granularity_ms: timestamp_granularity_ms(params)?,
            token_timestamps: bool_param(
                params,
                &["token_timestamps", "return_token_timestamps"],
                false,
            )?,
        })
    }
}

fn bool_param(
    params: &BTreeMap<String, serde_json::Value>,
    names: &[&str],
    default: bool,
) -> Result<bool> {
    let Some((name, value)) = names
        .iter()
        .find_map(|name| params.get(*name).map(|value| (*name, value)))
    else {
        return Ok(default);
    };
    value.as_bool().ok_or_else(|| {
        InfraError::Adapter(format!(
            "ASR parameter `{name}` must be a boolean, got {value}"
        ))
    })
}

fn timestamp_granularity_ms(params: &BTreeMap<String, serde_json::Value>) -> Result<u64> {
    if let Some(value) = params.get("timestamp_granularity_ms") {
        let value = value.as_u64().filter(|value| {
            (MIN_TIMESTAMP_GRANULARITY_MS..=MAX_TIMESTAMP_GRANULARITY_MS).contains(value)
        });
        return value.ok_or_else(|| {
            InfraError::Adapter(format!(
                "ASR parameter `timestamp_granularity_ms` must be an integer in {MIN_TIMESTAMP_GRANULARITY_MS}..={MAX_TIMESTAMP_GRANULARITY_MS}"
            ))
        });
    }
    let Some((name, value)) = [
        "timestamp_granularity_sec",
        "timestamp_granularity_seconds",
        "segment_duration_sec",
    ]
    .iter()
    .find_map(|name| params.get(*name).map(|value| (*name, value))) else {
        return Ok(DEFAULT_TIMESTAMP_GRANULARITY_MS);
    };
    let seconds = value.as_f64().filter(|seconds| {
        seconds.is_finite()
            && *seconds >= MIN_TIMESTAMP_GRANULARITY_MS as f64 / 1000.0
            && *seconds <= MAX_TIMESTAMP_GRANULARITY_MS as f64 / 1000.0
    });
    seconds
        .map(|seconds| (seconds * 1000.0).round() as u64)
        .ok_or_else(|| {
            InfraError::Adapter(format!(
                "ASR parameter `{name}` must be a number in {}..={} seconds",
                MIN_TIMESTAMP_GRANULARITY_MS / 1000,
                MAX_TIMESTAMP_GRANULARITY_MS / 1000
            ))
        })
}

fn samples_to_ms(samples: usize) -> u64 {
    (samples as u64 * 1000) / audio::TARGET_SAMPLE_RATE as u64
}

fn split_decoded_segment(
    start_ms: u64,
    end_ms: u64,
    speaker: Option<String>,
    decoded: RecognizedText,
    granularity_ms: u64,
    include_tokens: bool,
) -> Vec<AsrSegment> {
    let absolute_tokens = decoded
        .tokens
        .into_iter()
        .filter_map(|mut token| {
            token.start_ms = (start_ms + token.start_ms).min(end_ms);
            token.end_ms = (start_ms + token.end_ms).min(end_ms);
            (token.end_ms > token.start_ms).then_some(token)
        })
        .collect::<Vec<_>>();
    if absolute_tokens.is_empty() || end_ms.saturating_sub(start_ms) <= granularity_ms {
        return vec![timeline_segment(
            start_ms,
            end_ms,
            decoded.text,
            speaker,
            decoded.language,
            decoded.emotion,
            decoded.events,
            &absolute_tokens,
            include_tokens,
        )];
    }

    let mut output = Vec::new();
    let mut token_begin = 0;
    let mut chunk_start_ms = start_ms;
    for index in 0..absolute_tokens.len() - 1 {
        let token_end_ms = absolute_tokens[index].end_ms;
        if token_end_ms.saturating_sub(chunk_start_ms) < granularity_ms
            || end_ms.saturating_sub(token_end_ms) < granularity_ms / 2
            || !is_safe_timeline_boundary(&absolute_tokens[index], &absolute_tokens[index + 1])
        {
            continue;
        }
        let tokens = &absolute_tokens[token_begin..=index];
        output.push(timeline_segment(
            chunk_start_ms,
            token_end_ms,
            render_timestamp_tokens(tokens),
            speaker.clone(),
            decoded.language.clone(),
            decoded.emotion.clone(),
            decoded.events.clone(),
            tokens,
            include_tokens,
        ));
        token_begin = index + 1;
        chunk_start_ms = token_end_ms;
    }
    let tokens = &absolute_tokens[token_begin..];
    output.push(timeline_segment(
        chunk_start_ms,
        end_ms,
        render_timestamp_tokens(tokens),
        speaker,
        decoded.language,
        decoded.emotion,
        decoded.events,
        tokens,
        include_tokens,
    ));
    output
}

#[allow(clippy::too_many_arguments)]
fn timeline_segment(
    start_ms: u64,
    end_ms: u64,
    text: String,
    speaker: Option<String>,
    language: Option<String>,
    emotion: Option<String>,
    events: Vec<String>,
    tokens: &[TokenTiming],
    include_tokens: bool,
) -> AsrSegment {
    AsrSegment {
        start_ms,
        end_ms,
        text,
        speaker,
        language,
        emotion,
        events,
        tokens: include_tokens
            .then(|| {
                tokens
                    .iter()
                    .map(|token| AsrTokenTimestamp {
                        start_ms: token.start_ms,
                        end_ms: token.end_ms,
                        text: token.text.clone(),
                    })
                    .collect()
            })
            .unwrap_or_default(),
    }
}

fn render_timestamp_tokens(tokens: &[TokenTiming]) -> String {
    let mut text = String::new();
    for token in tokens {
        if token.starts_word
            && !text.is_empty()
            && !text.chars().last().is_some_and(char::is_whitespace)
        {
            text.push(' ');
        }
        text.push_str(&token.text);
    }
    text.trim().to_string()
}

fn is_safe_timeline_boundary(current: &TokenTiming, next: &TokenTiming) -> bool {
    next.starts_word
        || current
            .text
            .chars()
            .last()
            .is_some_and(|ch| !ch.is_ascii_alphanumeric())
}

fn merge_timeline_segments(segments: Vec<AsrSegment>, granularity_ms: u64) -> Vec<AsrSegment> {
    let mut output: Vec<AsrSegment> = Vec::with_capacity(segments.len());
    let maximum_span_ms = granularity_ms.saturating_add(granularity_ms / 2);
    for mut segment in segments {
        let Some(current) = output.last_mut() else {
            output.push(segment);
            continue;
        };
        let compatible = current.speaker == segment.speaker
            && current.language == segment.language
            && current.emotion == segment.emotion
            && current.events == segment.events;
        let current_span_ms = current.end_ms.saturating_sub(current.start_ms);
        let merged_span_ms = segment.end_ms.saturating_sub(current.start_ms);
        if compatible && current_span_ms < granularity_ms && merged_span_ms <= maximum_span_ms {
            append_transcript(&mut current.text, segment.text.trim());
            current.end_ms = segment.end_ms;
            current.tokens.append(&mut segment.tokens);
        } else {
            output.push(segment);
        }
    }
    output
}

fn format_timestamped_text(segments: &[AsrSegment]) -> String {
    segments
        .iter()
        .filter(|segment| !segment.text.trim().is_empty())
        .map(|segment| {
            format!(
                "[{}] {} [{}]",
                format_timestamp_ms(segment.start_ms),
                segment.text.trim(),
                format_timestamp_ms(segment.end_ms)
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_timestamp_ms(value: u64) -> String {
    let hours = value / 3_600_000;
    let minutes = (value / 60_000) % 60;
    let seconds = (value / 1_000) % 60;
    let millis = value % 1_000;
    format!("{hours:02}:{minutes:02}:{seconds:02}.{millis:03}")
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

/// The speaker chunks (and their embeddings) of the segments that are kept.
fn kept_speaker_chunks(
    decoded: &[DecodedSegment],
    chunks: Vec<speaker::SpeakerChunk>,
    embeddings: Vec<Vec<f32>>,
) -> (Vec<speaker::SpeakerChunk>, Vec<Vec<f32>>) {
    let ranges = decoded
        .iter()
        .filter(|segment| segment.text.is_some())
        .map(|segment| (segment.segment.start_sample, segment.segment.end_sample))
        .collect::<Vec<_>>();
    chunks
        .into_iter()
        .zip(embeddings)
        .filter(|(chunk, _)| {
            ranges
                .iter()
                .any(|&(start, end)| start <= chunk.start_sample && chunk.end_sample <= end)
        })
        .unzip()
}

/// Speakers numbered by first appearance among the kept segments (`kept`);
/// the others' labels are left as they are (they are not output).
fn number_kept_speakers(labels: &mut [usize], kept: &[bool]) {
    let indices = (0..labels.len())
        .filter(|&index| kept[index])
        .collect::<Vec<_>>();
    let mut kept_labels = indices
        .iter()
        .map(|&index| labels[index])
        .collect::<Vec<_>>();
    relabel_speakers_by_first_segment(&mut kept_labels);
    for (&index, &label) in indices.iter().zip(&kept_labels) {
        labels[index] = label;
    }
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

fn language_id(language: Option<&str>) -> Result<i32> {
    let Some(language) = language
        .map(str::trim)
        .filter(|language| !language.is_empty())
    else {
        return Ok(LANGUAGE_AUTO_ID);
    };
    LANGUAGE_IDS
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case(language))
        .map(|&(_, id)| id)
        .ok_or_else(|| {
            InfraError::Adapter(format!(
                "SenseVoice language must be auto, zh, en, yue, ja or ko, got {language:?}"
            ))
        })
}

fn bind_inputs(
    session: &OrtSession,
    features: SenseVoiceFeatures,
    language: i32,
) -> Result<Vec<OrtTensorInput>> {
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
                    language
                } else if lower.contains("textnorm") || lower.contains("itn") {
                    WITH_ITN_ID
                } else {
                    let fallback = [features.frames as i32, language, WITH_ITN_ID];
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

/// A speech segment's text, its tags and its tokens' times (relative to the
/// segment start).
#[derive(Debug, Clone, Default)]
pub struct RecognizedText {
    pub text: String,
    pub language: Option<String>,
    pub emotion: Option<String>,
    pub events: Vec<String>,
    pub tokens: Vec<TokenTiming>,
}

#[derive(Debug, Clone)]
pub struct TokenTiming {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    /// Starts a word (a space goes before it when rendered).
    pub starts_word: bool,
}

fn decode_outputs(
    outputs: Vec<OrtTensorOutput>,
    tokens: &[String],
    output_frame_ms: u64,
) -> Result<RecognizedText> {
    // The argmax taken in the graph, when it has one.
    if let Some(output) = outputs
        .iter()
        .find(|output| output.name == cuda_graph::IDS_OUTPUT)
    {
        let OrtTensorData::I64(ids) = &output.data else {
            return Err(InfraError::Adapter(format!(
                "SenseVoice {} is not I64",
                cuda_graph::IDS_OUTPUT
            )));
        };
        return decode_ids(ids, tokens, output_frame_ms);
    }
    let logits = outputs
        .into_iter()
        .find(|output| output.data.element_type() == TensorElement::F32 && output.shape.len() == 3)
        .ok_or_else(|| {
            InfraError::Adapter("SenseVoice returned no rank-3 F32 logits".to_string())
        })?;
    let OrtTensorData::F32(data) = logits.data else {
        unreachable!()
    };
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
    let ids = data
        .chunks_exact(vocab)
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map_or(BLANK_ID as i64, |(index, _)| index as i64)
        })
        .collect::<Vec<_>>();
    decode_ids(&ids, tokens, output_frame_ms)
}

/// Greedy CTC: one token id per output frame, repeats and blanks dropped.
fn decode_ids(ids: &[i64], tokens: &[String], output_frame_ms: u64) -> Result<RecognizedText> {
    let mut collapsed = Vec::new();
    let mut previous = usize::MAX;
    for (frame, &id) in ids.iter().enumerate() {
        let id = usize::try_from(id).map_err(|_| {
            InfraError::Adapter(format!("SenseVoice emitted negative token id {id}"))
        })?;
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
) -> Result<RecognizedText> {
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
            timestamps.push(TokenTiming {
                start_ms: frame as u64 * output_frame_ms,
                end_ms: (frame as u64 + 1) * output_frame_ms,
                text: word.to_string(),
                starts_word: true,
            });
        } else {
            text.push_str(token);
            timestamps.push(TokenTiming {
                start_ms: frame as u64 * output_frame_ms,
                end_ms: (frame as u64 + 1) * output_frame_ms,
                text: token.clone(),
                starts_word: false,
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
        let punctuation = if language == Some("<|zh|>") {
            '。'
        } else {
            '.'
        };
        text.push(punctuation);
        if let Some(last) = timestamps.last() {
            timestamps.push(TokenTiming {
                start_ms: last.start_ms,
                end_ms: last.end_ms,
                text: punctuation.to_string(),
                starts_word: false,
            });
        }
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
    Ok(RecognizedText {
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
