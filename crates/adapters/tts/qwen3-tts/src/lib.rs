//! Qwen3-TTS-12Hz-0.6B-Base (voice cloning) over ONNX Runtime.
//!
//! Provenance: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base and the
//! official `qwen-tts` package (https://github.com/QwenLM/Qwen3-TTS); graphs
//! exported by `scripts/local/qwen3_tts_export.py` for this project's
//! `backend-ort` runtime (neither depends on nor vendors upstream code).
//!
//! Per request: the reference voice (x-vector + codes, cached per file) ->
//! the prompt -> one talker run per 80 ms frame (CUDA graph replay on CUDA)
//! -> the streaming vocoder every few frames, so audio is available while
//! the rest is still being generated. All of it runs on one thread of the
//! adapter's own (see `engine`).

mod artifacts;
pub mod audio;
mod engine;
mod params;
mod prompt;
mod talker;
mod units;
mod vocoder;
mod voice;

pub use artifacts::{PackageConfig, Qwen3TtsArtifacts, PACKAGE_SCHEMA};
pub use params::SynthesisParams;
pub use prompt::{build_prompt, Codes, Prompt, Reference, GROUPS};
pub use voice::Voice;

use engine::Engine;
use local_backend_ort::SessionProviderReport;
use local_core::{FileRef, InferenceEvent, InferenceOutput, ModelSpec, TextPiece};
use local_error::{InfraError, Result};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    env, fs,
    panic::{catch_unwind, AssertUnwindSafe},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    thread::JoinHandle,
};
use uuid::Uuid;

#[derive(Debug, Clone, Copy)]
pub struct Qwen3TtsProviderReport {
    pub talker: SessionProviderReport,
    pub talker_cuda_graph: bool,
    pub vocoder: SessionProviderReport,
    pub vocoder_cuda_graph: bool,
    pub vocoder_chunk_frames: usize,
    pub speaker_encoder: SessionProviderReport,
    pub codec_encoder: SessionProviderReport,
}

/// What one synthesis took.
#[derive(Debug, Clone, Default)]
pub struct SynthesisStats {
    pub prompt_positions: usize,
    pub frames: usize,
    pub samples: usize,
    pub voice_ms: u64,
    pub first_audio_ms: u64,
    pub talker_ms: u64,
    pub vocoder_ms: u64,
    pub total_ms: u64,
    /// The consumer asked to stop before the end.
    pub stopped: bool,
    /// From the moment the prompt's text was there (the first token of a
    /// stream) to the first audio.
    pub first_audio_after_text_ms: u64,
    /// Time spent waiting for a stream's text mid-speech.
    pub text_wait_ms: u64,
    pub text_tokens: usize,
    /// Cut short, with its text: far more speech than the text needs (see
    /// `MAX_FRAMES_PER_TEXT_TOKEN`).
    pub runaway: Option<String>,
}

enum JobText {
    Whole(String),
    Stream(mpsc::Receiver<TextPiece>),
}

enum Job {
    Synthesize {
        text: JobText,
        reference: Option<PathBuf>,
        params: BTreeMap<String, Value>,
        chunks: mpsc::SyncSender<Chunk>,
        /// Set when the caller stops listening: the engine stops, also while
        /// it waits for streamed text.
        cancel: Arc<AtomicBool>,
    },
}

enum Chunk {
    Audio(Vec<f32>),
    Done(Result<SynthesisStats>),
    /// The engine panicked (its thread ends).
    Panicked(String),
}

/// A handle to the engine thread.
pub struct Qwen3TtsAdapter {
    model_id: String,
    root: PathBuf,
    sample_rate: u32,
    report: Qwen3TtsProviderReport,
    output_dir: PathBuf,
    jobs: Option<mpsc::Sender<Job>>,
    /// The latest job's cancel flag (set on drop).
    cancel: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for Qwen3TtsAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3TtsAdapter")
            .field("model_id", &self.model_id)
            .field("root", &self.root)
            .field("report", &self.report)
            .finish()
    }
}

impl Qwen3TtsAdapter {
    /// Loads (and warms up) the model on a thread of its own.
    pub fn load(spec: &ModelSpec) -> Result<Self> {
        let (ready_tx, ready_rx) = mpsc::channel();
        let (jobs_tx, jobs_rx) = mpsc::channel::<Job>();
        let engine_spec = spec.clone();
        let thread = std::thread::Builder::new()
            .name(format!("qwen3-tts:{}", spec.id))
            .spawn(move || {
                let mut engine = match Engine::load(&engine_spec) {
                    Ok(engine) => {
                        let _ = ready_tx.send(Ok((engine.provider_report(), engine.sample_rate())));
                        engine
                    }
                    Err(err) => {
                        let _ = ready_tx.send(Err(err));
                        return;
                    }
                };
                while let Ok(Job::Synthesize {
                    text,
                    reference,
                    params,
                    chunks,
                    cancel,
                }) = jobs_rx.recv()
                {
                    let source = match &text {
                        JobText::Whole(text) => engine::TextSource::Whole(text),
                        JobText::Stream(pieces) => engine::TextSource::Stream(pieces),
                    };
                    let result = catch_unwind(AssertUnwindSafe(|| {
                        engine.synthesize_stream(
                            source,
                            reference.as_deref(),
                            &params,
                            &cancel,
                            &mut |samples| chunks.send(Chunk::Audio(samples.to_vec())).is_ok(),
                        )
                    }));
                    match result {
                        Ok(result) => {
                            let _ = chunks.send(Chunk::Done(result));
                        }
                        Err(panic) => {
                            // The sessions' state is suspect: end the thread;
                            // the caller passes the panic on so the runtime
                            // reloads the model.
                            let _ = chunks.send(Chunk::Panicked(panic_message(&*panic)));
                            return;
                        }
                    }
                }
            })
            .map_err(|e| InfraError::Adapter(format!("spawn the Qwen3-TTS thread: {e}")))?;
        let (report, sample_rate) = ready_rx.recv().map_err(|_| {
            InfraError::Adapter("the Qwen3-TTS thread ended while loading".to_string())
        })??;
        Ok(Self {
            model_id: spec.id.clone(),
            root: Qwen3TtsArtifacts::resolve(spec),
            sample_rate,
            report,
            output_dir: env::var_os("LOCAL_DATA_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("workdir/data")),
            jobs: Some(jobs_tx),
            cancel: Arc::new(AtomicBool::new(false)),
            thread: Some(thread),
        })
    }

    pub fn provider_report(&self) -> Qwen3TtsProviderReport {
        self.report
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Synthesizes `text` into a WAV file under the data directory.
    pub fn synthesize(
        &mut self,
        request_id: Uuid,
        text: &str,
        reference_audio: Option<&FileRef>,
        params: &BTreeMap<String, Value>,
    ) -> Result<InferenceOutput> {
        self.synthesize_with_events(request_id, text, reference_audio, params, &mut |_| true)
    }

    /// Like [`Self::synthesize`], and each audio chunk also goes to `sink` as
    /// an [`InferenceEvent::AudioChunk`] as soon as it is decoded; `sink`
    /// returning false (the consumer went away) stops the synthesis.
    pub fn synthesize_with_events(
        &mut self,
        request_id: Uuid,
        text: &str,
        reference_audio: Option<&FileRef>,
        params: &BTreeMap<String, Value>,
        sink: &mut dyn FnMut(InferenceEvent) -> bool,
    ) -> Result<InferenceOutput> {
        self.synthesize_job_with_events(
            request_id,
            JobText::Whole(text.to_string()),
            reference_audio,
            params,
            sink,
        )
    }

    /// Like [`Self::synthesize_with_events`] for text still being written:
    /// speech starts once its first tokens are there (in-context cloning
    /// needs as many as the reference audio's frames, minus its transcript)
    /// and the rest is fed as it comes, one token per 80 ms frame, the way
    /// the official streaming mode feeds text; generation waits for a
    /// stream that falls behind.
    pub fn synthesize_text_stream_with_events(
        &mut self,
        request_id: Uuid,
        text: mpsc::Receiver<TextPiece>,
        reference_audio: Option<&FileRef>,
        params: &BTreeMap<String, Value>,
        sink: &mut dyn FnMut(InferenceEvent) -> bool,
    ) -> Result<InferenceOutput> {
        self.synthesize_job_with_events(
            request_id,
            JobText::Stream(text),
            reference_audio,
            params,
            sink,
        )
    }

    fn synthesize_job_with_events(
        &mut self,
        request_id: Uuid,
        text: JobText,
        reference_audio: Option<&FileRef>,
        params: &BTreeMap<String, Value>,
        sink: &mut dyn FnMut(InferenceEvent) -> bool,
    ) -> Result<InferenceOutput> {
        let sample_rate = self.sample_rate;
        let mut samples = Vec::new();
        let stats = self.synthesize_job(text, reference_audio, params, &mut |chunk| {
            samples.extend_from_slice(chunk);
            sink(InferenceEvent::AudioChunk {
                sample_rate,
                samples: chunk.to_vec(),
            })
        })?;
        fs::create_dir_all(&self.output_dir)
            .map_err(|e| InfraError::io(Some(self.output_dir.clone()), e))?;
        let path = self
            .output_dir
            .join(format!("qwen3tts-{}.wav", Uuid::new_v4()));
        audio::write_wav_i16(&path, &samples, sample_rate)?;
        tracing::info!(
            request_id = %request_id,
            model_id = self.model_id,
            text_tokens = stats.text_tokens,
            prompt = stats.prompt_positions,
            frames = stats.frames,
            audio_ms = stats.samples as u64 * 1000 / sample_rate as u64,
            voice_ms = stats.voice_ms,
            first_audio_ms = stats.first_audio_ms,
            first_audio_after_text_ms = stats.first_audio_after_text_ms,
            text_wait_ms = stats.text_wait_ms,
            talker_ms = stats.talker_ms,
            vocoder_ms = stats.vocoder_ms,
            total_ms = stats.total_ms,
            stopped = stats.stopped,
            "Qwen3-TTS synthesized"
        );
        if let Some(text) = &stats.runaway {
            tracing::warn!(
                request_id = %request_id,
                text,
                frames = stats.frames,
                text_tokens = stats.text_tokens,
                "Qwen3-TTS kept speaking far past its text; cut short"
            );
        }
        let mut file = FileRef::local(path);
        file.mime = Some("audio/wav".to_string());
        Ok(InferenceOutput::TtsAudio { audio: file })
    }

    /// Synthesizes `text`, handing audio (mono, [`Self::sample_rate`]) to
    /// `on_audio` chunk by chunk as it is decoded; `on_audio` returning false
    /// stops the synthesis.
    pub fn synthesize_stream(
        &mut self,
        text: &str,
        reference_audio: Option<&FileRef>,
        params: &BTreeMap<String, Value>,
        on_audio: &mut dyn FnMut(&[f32]) -> bool,
    ) -> Result<SynthesisStats> {
        self.synthesize_job(
            JobText::Whole(text.to_string()),
            reference_audio,
            params,
            on_audio,
        )
    }

    /// [`Self::synthesize_stream`] for text still being written (see
    /// [`Self::synthesize_text_stream_with_events`]).
    pub fn synthesize_text_stream(
        &mut self,
        text: mpsc::Receiver<TextPiece>,
        reference_audio: Option<&FileRef>,
        params: &BTreeMap<String, Value>,
        on_audio: &mut dyn FnMut(&[f32]) -> bool,
    ) -> Result<SynthesisStats> {
        self.synthesize_job(JobText::Stream(text), reference_audio, params, on_audio)
    }

    fn synthesize_job(
        &mut self,
        text: JobText,
        reference_audio: Option<&FileRef>,
        params: &BTreeMap<String, Value>,
        on_audio: &mut dyn FnMut(&[f32]) -> bool,
    ) -> Result<SynthesisStats> {
        let reference = reference_audio.map(local_files::local_path).transpose()?;
        let (chunks_tx, chunks) = mpsc::sync_channel(16);
        let cancel = Arc::new(AtomicBool::new(false));
        self.cancel = cancel.clone();
        let sent = self.jobs.as_ref().is_some_and(|jobs| {
            jobs.send(Job::Synthesize {
                text,
                reference,
                params: params.clone(),
                chunks: chunks_tx,
                cancel: cancel.clone(),
            })
            .is_ok()
        });
        if !sent {
            // The engine thread ended (it panicked before): this adapter is
            // dead; a panic makes the runtime reload it.
            panic!("the Qwen3-TTS engine thread of `{}` is gone", self.model_id);
        }
        let mut samples = 0usize;
        loop {
            match chunks.recv() {
                Ok(Chunk::Audio(audio)) => {
                    samples += audio.len();
                    if !on_audio(&audio) {
                        // The engine stops at its next frame (or while it
                        // waits for text).
                        cancel.store(true, Ordering::Relaxed);
                        return Ok(SynthesisStats {
                            samples,
                            stopped: true,
                            ..SynthesisStats::default()
                        });
                    }
                }
                Ok(Chunk::Done(result)) => return result,
                Ok(Chunk::Panicked(message)) => {
                    panic!(
                        "the Qwen3-TTS engine of `{}` panicked: {message}",
                        self.model_id
                    )
                }
                Err(_) => panic!(
                    "the Qwen3-TTS engine thread of `{}` ended mid-synthesis",
                    self.model_id
                ),
            }
        }
    }
}

fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    panic
        .downcast_ref::<&str>()
        .map(|text| text.to_string())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_string())
}

impl Drop for Qwen3TtsAdapter {
    fn drop(&mut self) {
        // Closing the job queue ends the thread, which drops the sessions;
        // a job still waiting for text stops.
        self.cancel.store(true, Ordering::Relaxed);
        self.jobs.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Rng;

    #[test]
    fn gumbel_noise_has_the_right_mean() {
        let mut rng = Rng::new(7);
        let n = 200_000;
        let mean = (0..n).map(|_| rng.gumbel() as f64).sum::<f64>() / n as f64;
        // Euler-Mascheroni constant.
        assert!((mean - 0.5772).abs() < 0.01, "mean {mean}");
    }

    /// `LOCAL_QWEN3_TTS_MODEL_DIR` + `LOCAL_QWEN3_TTS_REFERENCE`
    /// (+ `LOCAL_QWEN3_TTS_REFERENCE_TEXT` for ICL): synthesizes and checks
    /// the audio is plausible.
    #[test]
    fn real_model_smoke_if_env_set() {
        let (Ok(model_dir), Ok(reference)) = (
            env::var("LOCAL_QWEN3_TTS_MODEL_DIR"),
            env::var("LOCAL_QWEN3_TTS_REFERENCE"),
        ) else {
            return;
        };
        let out = tempfile::tempdir().unwrap();
        env::set_var("LOCAL_DATA_DIR", out.path());
        let spec: ModelSpec = serde_json::from_value(serde_json::json!({
            "id": "qwen3-tts-0.6b-onnx",
            "name": "Qwen3-TTS",
            "adapter": "qwen3_tts",
            "backend": "ort",
            "artifacts": [{"type": "local", "path": model_dir}],
            "runtime": {"provider_order": ["cuda", "cpu"]},
        }))
        .unwrap();
        let mut adapter = Qwen3TtsAdapter::load(&spec).unwrap();
        eprintln!("{:?}", adapter.provider_report());
        let mut params = BTreeMap::new();
        params.insert("seed".to_string(), Value::from(1));
        params.insert("language".to_string(), Value::from("chinese"));
        if let Ok(text) = env::var("LOCAL_QWEN3_TTS_REFERENCE_TEXT") {
            params.insert("reference_text".to_string(), Value::from(text));
        }
        let reference = FileRef::local(PathBuf::from(reference));
        for _ in 0..3 {
            let mut samples = 0usize;
            let stats = adapter
                .synthesize_stream(
                    "你好，我是你的语音助手，今天有什么可以帮你的吗？",
                    Some(&reference),
                    &params,
                    &mut |chunk| {
                        samples += chunk.len();
                        true
                    },
                )
                .unwrap();
            eprintln!("{stats:?}");
            let seconds = samples as f32 / 24_000.0;
            assert!((2.0..10.0).contains(&seconds), "{seconds} s of audio");
        }
    }
}
