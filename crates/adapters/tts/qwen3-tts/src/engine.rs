//! The sessions and the synthesis loop. One thread owns an [`Engine`]: ORT's
//! CUDA execution provider keeps CUDA graphs per thread, so runs from other
//! threads would capture the decode frame again (and stall every other GPU
//! user meanwhile).

use crate::{
    artifacts::{self, Qwen3TtsArtifacts},
    audio,
    params::SynthesisParams,
    prompt::{build_prompt, Codes, Reference, GROUPS},
    talker::{Sampling, Talker},
    vocoder::Vocoder,
    voice::{Voice, VoiceEncoder},
    Qwen3TtsProviderReport, SynthesisStats,
};
use local_backend_ort::{CudaSessionOptions, OrtBackend, ProviderSelection};
use local_core::ModelSpec;
use local_error::{InfraError, Result};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::{Instant, SystemTime},
};
use tokenizers::Tokenizer;

/// Talker KV positions unless the spec's metadata sets `max_context`.
const DEFAULT_MAX_CONTEXT: usize = 2048;
/// Frames per vocoder run (and before the first audio) unless the metadata
/// sets `vocoder_chunk_frames`: 4 frames = 320 ms of audio.
const DEFAULT_VOCODER_CHUNK: usize = 4;

#[derive(Debug)]
struct CachedVoice {
    key: (PathBuf, u64, Option<SystemTime>, u64),
    voice: Voice,
}

pub(crate) struct Engine {
    model_id: String,
    artifacts: Qwen3TtsArtifacts,
    tokenizer: Tokenizer,
    talker: Talker,
    vocoder: Vocoder,
    encoder: VoiceEncoder,
    cached_voice: Option<CachedVoice>,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("model_id", &self.model_id)
            .field("root", &self.artifacts.root)
            .field("talker", &self.talker)
            .field("vocoder", &self.vocoder)
            .finish()
    }
}

impl Engine {
    pub(crate) fn load(spec: &ModelSpec) -> Result<Self> {
        let started = Instant::now();
        let artifacts = Qwen3TtsArtifacts::load(Qwen3TtsArtifacts::resolve(spec))?;
        let config = &artifacts.config;
        let selection = ProviderSelection::from_strings(&spec.runtime.provider_order);
        let metadata_flag = |key: &str| spec.metadata.get(key).and_then(Value::as_bool);
        let cuda_graph = metadata_flag("cuda_graph").unwrap_or(true);
        let capacity = spec
            .metadata
            .get("max_context")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .unwrap_or(DEFAULT_MAX_CONTEXT)
            .min(config.max_position_embeddings);
        // Weights go straight to the device allocator (no power-of-two arena
        // rounding).
        let base = OrtBackend::new(selection.clone())
            .with_config_entry("session.use_device_allocator_for_initializers", "1");
        let talker_backend = base.clone().with_cuda_session_options(CudaSessionOptions {
            cuda_graph,
            tf32: None,
        });
        let talker = Talker::load(
            &talker_backend,
            &artifacts.file(&config.graphs.talker),
            config,
            capacity,
            cuda_graph,
        )?;
        let chunk = spec
            .metadata
            .get("vocoder_chunk_frames")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .unwrap_or(DEFAULT_VOCODER_CHUNK);
        // The vocoder graph keeps shape arithmetic on the CPU, which CUDA
        // graph capture refuses: plain runs.
        let vocoder_backend = base.clone();
        // The stream: a primed reference (at most ~30 s = 375 frames) and the
        // talker's longest output.
        let vocoder = Vocoder::load(
            &vocoder_backend,
            &artifacts.file(&config.graphs.vocoder),
            config,
            chunk,
            capacity + 400,
            false,
        )?;
        // FP32 encoders: TF32 flips residual-VQ code choices.
        let encoder_backend = base.with_cuda_session_options(CudaSessionOptions {
            cuda_graph: false,
            tf32: Some(false),
        });
        let encoder = VoiceEncoder::load(
            &encoder_backend,
            &artifacts.file(&config.graphs.speaker_encoder),
            &artifacts.file(&config.graphs.codec_encoder),
            config,
        )?;
        let tokenizer_path = artifacts.file(artifacts::TOKENIZER_FILE);
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| InfraError::Adapter(format!("load {}: {e}", tokenizer_path.display())))?;
        let mut adapter = Self {
            model_id: spec.id.clone(),
            artifacts,
            tokenizer,
            talker,
            vocoder,
            encoder,
            cached_voice: None,
        };
        tracing::info!(
            model_id = adapter.model_id,
            root = %adapter.artifacts.root.display(),
            providers = ?adapter.provider_report(),
            capacity,
            load_ms = started.elapsed().as_millis() as u64,
            "Qwen3-TTS sessions loaded"
        );
        let warm_started = Instant::now();
        adapter.warm_up()?;
        tracing::info!(
            model_id = adapter.model_id,
            warm_ms = warm_started.elapsed().as_millis() as u64,
            "Qwen3-TTS graphs warmed up"
        );
        Ok(adapter)
    }

    /// Runs the talker and the vocoder at their decode shapes once, so the
    /// first request neither builds cuDNN plans nor captures the CUDA graph
    /// (a capture blocks every other GPU call in the process meanwhile).
    fn warm_up(&mut self) -> Result<()> {
        let config = self.artifacts.config.clone();
        let ids =
            self.encode("<|im_start|>assistant\n你好。<|im_end|>\n<|im_start|>assistant\n")?;
        let prompt = build_prompt(&config.tokens, &ids, None, false, None)?;
        let seen = vec![0.0f32; config.talker_vocab_size];
        let noise = vec![0.0f32; GROUPS * config.top_k];
        let sampling = [1.0, 1.0, 1.0, 0.0];
        let mut codes = self
            .talker
            .prefill(&prompt, None, &seen, &noise, sampling)?;
        for _ in 0..6 {
            codes = self
                .talker
                .step(config.tokens.tts_pad, &codes, &seen, &noise, sampling)?;
        }
        self.vocoder.reset()?;
        let frames = vec![[0i64; GROUPS]; self.vocoder.chunk()];
        for _ in 0..3 {
            self.vocoder.decode(&frames)?;
        }
        self.vocoder.reset()
    }

    pub(crate) fn provider_report(&self) -> Qwen3TtsProviderReport {
        let [speaker_encoder, codec_encoder] = self.encoder.provider_reports();
        Qwen3TtsProviderReport {
            talker: self.talker.provider_report(),
            talker_cuda_graph: self.talker.uses_cuda_graph(),
            vocoder: self.vocoder.provider_report(),
            vocoder_cuda_graph: self.vocoder.uses_cuda_graph(),
            vocoder_chunk_frames: self.vocoder.chunk(),
            speaker_encoder,
            codec_encoder,
        }
    }

    pub(crate) fn sample_rate(&self) -> u32 {
        self.artifacts.config.sample_rate
    }

    /// Synthesizes `text`, handing audio (mono, [`Self::sample_rate`]) to
    /// `on_audio` chunk by chunk as it is decoded; `on_audio` returning false
    /// stops the synthesis.
    pub(crate) fn synthesize_stream(
        &mut self,
        text: &str,
        reference_audio: Option<&Path>,
        params: &BTreeMap<String, Value>,
        on_audio: &mut dyn FnMut(&[f32]) -> bool,
    ) -> Result<SynthesisStats> {
        let started = Instant::now();
        let config = self.artifacts.config.clone();
        let params = SynthesisParams::from_map(params, &config)?;
        let reference_path = reference_audio.ok_or_else(|| {
            InfraError::BadRequest(
                "Qwen3-TTS (Base) synthesis requires reference_audio".to_string(),
            )
        })?;
        let voice_started = Instant::now();
        let max_samples = (params.max_reference_seconds * config.sample_rate as f32) as usize;
        self.ensure_voice(reference_path, max_samples)?;
        let cached = self.cached_voice.as_ref().expect("voice just cached");
        let voice_key = format!("{:?}", cached.key);
        let voice = &cached.voice;
        let voice_ms = voice_started.elapsed().as_millis() as u64;

        let ids = self.encode(&format!(
            "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        ))?;
        let ref_ids = match &params.reference_text {
            Some(reference) => {
                Some(self.encode(&format!("<|im_start|>assistant\n{reference}<|im_end|>\n"))?)
            }
            None => None,
        };
        let language = params.language.as_ref().map(|name| config.languages[name]);
        let reference = ref_ids.as_deref().map(|ids| Reference {
            ids,
            codes: &voice.codes,
        });
        let prompt = build_prompt(&config.tokens, &ids, language, true, reference)?;
        let max_frames = params
            .max_frames
            .min(self.talker.capacity().saturating_sub(prompt.len() + 1));
        let speaker = voice.speaker.clone();
        let icl_codes = reference.map(|reference| reference.codes.to_vec());

        let mut stats = SynthesisStats {
            prompt_positions: prompt.len(),
            voice_ms,
            ..SynthesisStats::default()
        };
        // The vocoder continues the reference audio in ICL mode, as upstream
        // decodes reference and generated codes together (primed once per
        // voice).
        let vocoder_started = Instant::now();
        match &icl_codes {
            Some(codes) => self.vocoder.prime(&voice_key, codes)?,
            None => self.vocoder.reset()?,
        }
        let chunk = self.vocoder.chunk();
        let mut vocoder_time = vocoder_started.elapsed();
        let mut talker_time = std::time::Duration::ZERO;

        let mut rng = Rng::new(params.seed);
        let mut seen = vec![0.0f32; config.talker_vocab_size];
        let mut noise = vec![0.0f32; GROUPS * config.top_k];
        let sampling = |frame: usize| -> Sampling {
            [
                1.0 / params.temperature,
                1.0 / params.subtalker_temperature,
                params.repetition_penalty,
                if frame < config.generation.min_new_tokens {
                    0.0
                } else {
                    1.0
                },
            ]
        };
        let mut fill_noise = |noise: &mut [f32]| {
            if params.do_sample {
                noise.iter_mut().for_each(|value| *value = rng.gumbel());
            }
        };
        let eos = config.tokens.codec_eos;
        let pad = config.tokens.tts_pad;
        let mut pending: Vec<Codes> = Vec::with_capacity(chunk);
        let mut first_chunk = true;

        let talker_started = Instant::now();
        fill_noise(&mut noise);
        let mut codes = self
            .talker
            .prefill(&prompt, Some(&speaker), &seen, &noise, sampling(0))?;
        talker_time += talker_started.elapsed();
        for frame in 0..max_frames {
            if codes[0] == eos {
                break;
            }
            stats.frames += 1;
            pending.push(codes);
            if pending.len() == chunk {
                let decode_started = Instant::now();
                let samples = self.vocoder.decode(&pending)?;
                vocoder_time += decode_started.elapsed();
                pending.clear();
                if first_chunk {
                    stats.first_audio_ms = started.elapsed().as_millis() as u64;
                    first_chunk = false;
                }
                stats.samples += samples.len();
                if !on_audio(&samples) {
                    stats.stopped = true;
                    break;
                }
            }
            if frame + 1 == max_frames {
                break;
            }
            let step_started = Instant::now();
            seen[codes[0] as usize] = 1.0;
            let text_id = prompt.trailing.get(frame).copied().unwrap_or(pad);
            fill_noise(&mut noise);
            codes = self
                .talker
                .step(text_id, &codes, &seen, &noise, sampling(frame + 1))?;
            talker_time += step_started.elapsed();
        }
        if !pending.is_empty() && !stats.stopped {
            let decode_started = Instant::now();
            let samples = self.vocoder.decode(&pending)?;
            vocoder_time += decode_started.elapsed();
            if first_chunk {
                stats.first_audio_ms = started.elapsed().as_millis() as u64;
            }
            stats.samples += samples.len();
            stats.stopped = !on_audio(&samples);
        }
        stats.talker_ms = talker_time.as_millis() as u64;
        stats.vocoder_ms = vocoder_time.as_millis() as u64;
        stats.total_ms = started.elapsed().as_millis() as u64;
        Ok(stats)
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|e| InfraError::Adapter(format!("tokenize: {e}")))
    }

    fn ensure_voice(&mut self, path: &Path, max_samples: usize) -> Result<()> {
        let metadata =
            fs::metadata(path).map_err(|e| InfraError::io(Some(path.to_path_buf()), e))?;
        let key = (
            path.to_path_buf(),
            metadata.len(),
            metadata.modified().ok(),
            max_samples as u64,
        );
        if self
            .cached_voice
            .as_ref()
            .is_some_and(|cached| cached.key == key)
        {
            return Ok(());
        }
        let audio = audio::read_wav_mono(path, self.sample_rate())?;
        let voice = self.encoder.encode(&audio, max_samples)?;
        tracing::info!(
            model_id = self.model_id,
            reference = %path.display(),
            seconds = audio.len() as f32 / self.sample_rate() as f32,
            frames = voice.codes.len(),
            "Qwen3-TTS reference voice encoded"
        );
        self.cached_voice = Some(CachedVoice { key, voice });
        Ok(())
    }
}

/// SplitMix64; Gumbel noise for the in-graph Gumbel-max samplers.
pub(crate) struct Rng {
    state: u64,
}

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// -ln(-ln(u)), u uniform in (0, 1).
    pub(crate) fn gumbel(&mut self) -> f32 {
        let u = ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        (-(-u.ln()).ln()) as f32
    }
}
