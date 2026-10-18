//! Fun-ASR-Nano-2512 (FunAudioLLM) on ONNX Runtime CUDA.
//!
//! The FunASR pipeline of the SenseVoice adapter (streamed decoding,
//! FSMN-VAD, FBANK/LFR features without CMVN, CAM++ speakers, the timeline)
//! with Fun-ASR-Nano recognizing each speech segment: the audio graph
//! (SenseVoice's encoder, the adaptor, the CTC decoder) and greedy decoding
//! of Qwen3-0.6B over upstream's prompt, the adaptor output in place of
//! placeholder tokens. The package is `scripts/local/fun_asr_nano_export.py`'s.
//!
//! The CTC branch guards the LLM: when it hears no speech the segment has no
//! text (the LLM would otherwise make some up for noise), and when the LLM
//! runs into its token limit (a repetition loop) the CTC text replaces its
//! output.

mod ctc;
mod embed;
mod encoder;
mod llm;

use local_adapter_sensevoice_asr::{
    pipeline_files, RecognizedText, SegmentRecognizer, SenseVoiceFeatures, SpeechPipeline,
};
use local_backend_ort::{CudaSessionOptions, OrtBackend, ProviderSelection, SessionProviderReport};
use local_core::{FileRef, InferenceOutput, ModelSpec};
use local_error::{InfraError, Result};
use serde::Deserialize;
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};
use tokenizers::Tokenizer;

pub const MANIFEST_SCHEMA: &str = "local.fun_asr_nano.package.v1";
const MANIFEST_FILE: &str = "manifest.json";
const CONFIG_FILE: &str = "config.yaml";
const TOKENIZER_FILE: &str = "tokenizer.json";
const CTC_TOKENS_FILE: &str = "ctc_tokens.json";
const DEFAULT_MAX_CONTEXT: usize = 1024;
const MAX_CONTEXT: usize = 8192;
const MAX_CUDA_GRAPH_FRAMES: usize = 1024;
/// Generated tokens allowed for a segment: a base plus this many per second
/// of speech (fast speech is ~5 tokens/s; upstream allows 512 in all).
const MAX_NEW_TOKENS_BASE: usize = 16;
const MAX_NEW_TOKENS_PER_SECOND: f64 = 12.0;
const HOTWORDS_PROMPT: &str = "请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n\n**上下文信息：**\n\n\n";
const MAX_HOTWORDS: usize = 200;

#[derive(Debug, Clone, Deserialize)]
pub struct PackageManifest {
    pub schema: String,
    pub audio_encoder: String,
    pub llm: String,
    pub embed_tokens: String,
    pub feature_dim: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub eos_token_ids: Vec<i64>,
    pub prompt: PromptManifest,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PromptManifest {
    pub system: String,
    pub user: String,
    pub prefix_ids: Vec<i64>,
    pub suffix_ids: Vec<i64>,
}

#[derive(Debug, Clone)]
pub struct FunAsrNanoArtifacts {
    pub root: PathBuf,
    pub asr_root: PathBuf,
    pub manifest: PackageManifest,
}

impl FunAsrNanoArtifacts {
    pub fn validate(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let asr_root = root.join("asr");
        let not_configured = |reason: String| InfraError::ModelNotConfigured {
            model_id: "fun-asr-nano-onnx".to_string(),
            reason,
        };
        let manifest_path = asr_root.join(MANIFEST_FILE);
        let manifest: PackageManifest = serde_json::from_slice(
            &fs::read(&manifest_path)
                .map_err(|_| not_configured(format!("missing {}", manifest_path.display())))?,
        )?;
        if manifest.schema != MANIFEST_SCHEMA {
            return Err(not_configured(format!(
                "{} has schema {:?}, expected {MANIFEST_SCHEMA}",
                manifest_path.display(),
                manifest.schema
            )));
        }
        let mut required = vec![
            asr_root.join(CONFIG_FILE),
            asr_root.join(TOKENIZER_FILE),
            asr_root.join(CTC_TOKENS_FILE),
            asr_root.join(&manifest.audio_encoder),
            asr_root.join(&manifest.llm),
            asr_root.join(&manifest.embed_tokens),
        ];
        required.extend(pipeline_files(&root));
        if let Some(missing) = required.iter().find(|path| !path.is_file()) {
            return Err(not_configured(format!(
                "required Fun-ASR-Nano artifact is missing: {}",
                missing.display()
            )));
        }
        Ok(Self {
            root,
            asr_root,
            manifest,
        })
    }

    fn from_spec(spec: &ModelSpec) -> Result<Self> {
        let root = spec
            .artifacts
            .iter()
            .find(|artifact| artifact.path.is_dir())
            .map(|artifact| artifact.path.clone())
            .ok_or_else(|| InfraError::ModelNotConfigured {
                model_id: spec.id.clone(),
                reason: "Fun-ASR-Nano package directory is not configured".to_string(),
            })?;
        Self::validate(root).map_err(|error| match error {
            InfraError::ModelNotConfigured { reason, .. } => InfraError::ModelNotConfigured {
                model_id: spec.id.clone(),
                reason,
            },
            other => other,
        })
    }

    fn file(&self, name: &str) -> PathBuf {
        self.asr_root.join(name)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FunAsrNanoProviderReport {
    pub audio: SessionProviderReport,
    pub llm: SessionProviderReport,
    pub vad: SessionProviderReport,
    pub speaker: SessionProviderReport,
}

/// Where the last request's recognizer time went.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RecognizerStats {
    pub segments: usize,
    pub audio_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub generated_tokens: usize,
    /// Segments the CTC branch heard no speech in.
    pub silent_segments: usize,
    /// Segments whose LLM output hit the token limit (CTC text used).
    pub token_limit_segments: usize,
}

#[derive(Debug)]
pub struct FunAsrNanoAdapter {
    model_id: String,
    artifacts: FunAsrNanoArtifacts,
    pipeline: SpeechPipeline,
    recognizer: NanoRecognizer,
}

impl FunAsrNanoAdapter {
    pub fn load(spec: &ModelSpec) -> Result<Self> {
        let started = Instant::now();
        let artifacts = FunAsrNanoArtifacts::from_spec(spec)?;
        let manifest = &artifacts.manifest;
        let pipeline =
            SpeechPipeline::load(spec, &artifacts.root, &artifacts.file(CONFIG_FILE), None)?;
        let frontend = pipeline.frontend();
        let feature_dim = frontend.n_mels * frontend.lfr_m;
        if feature_dim != manifest.feature_dim {
            return Err(InfraError::Adapter(format!(
                "Fun-ASR-Nano frontend gives {feature_dim} features, the package takes {}",
                manifest.feature_dim
            )));
        }
        let providers = ProviderSelection::from_strings(&spec.runtime.provider_order);
        let graph_frames = metadata_frames(spec, "asr_cuda_graph_frames")?;
        let encoder = encoder::AudioEncoder::load(
            &artifacts.file(&manifest.audio_encoder),
            &providers,
            &graph_frames,
            feature_dim,
            manifest.hidden_size,
            &spec.metadata,
        )?;
        let cuda_graph = metadata_bool(spec, "cuda_graph", true)?;
        let capacity = metadata_usize(spec, "max_context", DEFAULT_MAX_CONTEXT, MAX_CONTEXT)?;
        let llm_backend = OrtBackend::new(providers)
            .with_cuda_memory_metadata(&spec.metadata)
            .with_config_entry("session.use_device_allocator_for_initializers", "1")
            .with_cuda_session_options(CudaSessionOptions {
                cuda_graph,
                ..CudaSessionOptions::default()
            });
        let llm = llm::Llm::load(
            &llm_backend,
            &artifacts.file(&manifest.llm),
            manifest.num_layers,
            manifest.num_kv_heads,
            manifest.head_dim,
            manifest.hidden_size,
            capacity,
            cuda_graph,
        )?;
        let tokenizer_path = artifacts.file(TOKENIZER_FILE);
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| InfraError::Adapter(format!("load {}: {e}", tokenizer_path.display())))?;
        let ctc = ctc::CtcVocab::load(&artifacts.file(CTC_TOKENS_FILE))?;
        let embeddings = embed::EmbeddingTable::load(
            &artifacts.file(&manifest.embed_tokens),
            manifest.vocab_size,
            manifest.hidden_size,
        )?;
        let recognizer = NanoRecognizer {
            encoder,
            llm,
            embeddings,
            tokenizer,
            ctc,
            prompt: manifest.prompt.clone(),
            prefix: manifest.prompt.prefix_ids.clone(),
            hidden: manifest.hidden_size,
            eos: manifest.eos_token_ids.clone(),
            output_frame_ms: pipeline.output_frame_ms(),
            ctc_guard: metadata_bool(spec, "ctc_guard", true)?,
            stats: RecognizerStats::default(),
        };
        let mut adapter = Self {
            model_id: spec.id.clone(),
            artifacts,
            pipeline,
            recognizer,
        };
        adapter
            .check_prompt_room(adapter.recognizer.prompt.prefix_ids.len())
            .map_err(|error| InfraError::ModelNotConfigured {
                model_id: spec.id.clone(),
                reason: error.to_string(),
            })?;
        adapter.warm_up()?;
        tracing::info!(
            model_id = adapter.model_id,
            root = %adapter.artifacts.root.display(),
            providers = ?adapter.provider_report(),
            capacity,
            graph_frames = ?adapter.recognizer.encoder.graph_frames(),
            load_ms = started.elapsed().as_millis() as u64,
            "Fun-ASR-Nano sessions loaded"
        );
        Ok(adapter)
    }

    pub fn transcribe(&mut self, audio: &FileRef) -> Result<InferenceOutput> {
        self.transcribe_with_params(audio, &BTreeMap::new())
    }

    /// Transcribes a recording; besides the SenseVoice pipeline's parameters
    /// (timestamps, speakers), `hotwords` (a list, or a comma-separated
    /// string) and `language` (e.g. 中文, 英文, 日文; default: any) go into
    /// the prompt, and `itn: false` asks for text without normalization.
    pub fn transcribe_with_params(
        &mut self,
        audio: &FileRef,
        params: &BTreeMap<String, Value>,
    ) -> Result<InferenceOutput> {
        let prefix = self
            .recognizer
            .prompt_prefix(&PromptOptions::from_params(params)?)?;
        self.check_prompt_room(prefix.len())?;
        self.recognizer.prefix = prefix;
        self.recognizer.stats = RecognizerStats::default();
        let output = self
            .pipeline
            .transcribe(audio, params, &mut self.recognizer);
        self.recognizer.prefix = self.recognizer.prompt.prefix_ids.clone();
        output
    }

    /// One speech segment (16 kHz mono samples, no VAD) with the default
    /// prompt: its text (empty when the CTC guard hears no speech).
    pub fn recognize_samples(&mut self, samples: &[f32]) -> Result<RecognizedText> {
        self.recognize_samples_with_params(samples, &BTreeMap::new())
    }

    /// As [`Self::recognize_samples`], with the prompt parameters of
    /// [`Self::transcribe_with_params`] (`hotwords`, `language`, `itn`).
    pub fn recognize_samples_with_params(
        &mut self,
        samples: &[f32],
        params: &BTreeMap<String, Value>,
    ) -> Result<RecognizedText> {
        let prefix = self
            .recognizer
            .prompt_prefix(&PromptOptions::from_params(params)?)?;
        self.check_prompt_room(prefix.len())?;
        let features = self.pipeline.extract_features(samples)?;
        if features.frames == 0 {
            return Ok(RecognizedText::default());
        }
        self.recognizer.prefix = prefix;
        let recognized = self.recognizer.recognize(features);
        self.recognizer.prefix = self.recognizer.prompt.prefix_ids.clone();
        Ok(recognized?.unwrap_or_default())
    }

    /// Fails unless a prompt of `prefix` tokens before the audio leaves room
    /// in the LLM context for the longest segment the VAD cuts and its
    /// output (hotwords make the prefix long).
    fn check_prompt_room(&self, prefix: usize) -> Result<()> {
        let max_ms = self.pipeline.max_segment_ms() as u64;
        let frames = (max_ms / self.pipeline.output_frame_ms() + 1) as usize;
        let needed = prefix
            + fake_token_len(frames)
            + self.recognizer.prompt.suffix_ids.len()
            + max_new_tokens(max_ms as f64 / 1000.0)
            + 1;
        let capacity = self.recognizer.llm.capacity();
        if needed > capacity {
            return Err(InfraError::BadRequest(format!(
                "the Fun-ASR-Nano prompt takes {prefix} tokens before the audio: with a {:.0} s speech segment and its output that is {needed} positions, over the LLM context of {capacity} (fewer or shorter hotwords, or a larger max_context)",
                max_ms as f64 / 1000.0
            )));
        }
        Ok(())
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn artifacts(&self) -> &FunAsrNanoArtifacts {
        &self.artifacts
    }

    pub fn last_stats(&self) -> RecognizerStats {
        self.recognizer.stats
    }

    pub fn provider_report(&self) -> FunAsrNanoProviderReport {
        FunAsrNanoProviderReport {
            audio: self.recognizer.encoder.provider_report(),
            llm: self.recognizer.llm.provider_report(),
            vad: self.pipeline.vad_provider_report(),
            speaker: self.pipeline.speaker_provider_report(),
        }
    }

    /// The input lengths (LFR frames) the audio graph replays as CUDA graphs.
    pub fn cuda_graph_frames(&self) -> Vec<usize> {
        self.recognizer.encoder.graph_frames()
    }

    /// Whether the LLM's decode step replays as a CUDA graph.
    pub fn llm_cuda_graph(&self) -> bool {
        self.recognizer.llm.uses_cuda_graph()
    }

    /// Captures the LLM's decode graph (and the first prefill allocations)
    /// on a short silent segment, so the first request does not pay for it.
    fn warm_up(&mut self) -> Result<()> {
        let dim = self.artifacts.manifest.feature_dim;
        let frames = 16;
        let features = SenseVoiceFeatures {
            frames,
            dim,
            data: vec![0.0; frames * dim],
        };
        let audio = self.recognizer.encoder.run(&features)?;
        let embeds = self.recognizer.prompt(&audio)?;
        let mut token = self.recognizer.llm.prefill(embeds)?;
        for _ in 0..4 {
            let embed = self.recognizer.embeddings.row(token)?;
            token = self.recognizer.llm.step(embed)?;
        }
        Ok(())
    }
}

/// Upstream's name for a `language` parameter (Fun-ASR-Nano-2512 knows
/// 中文, 英文, 日文; the codes and English names map to them); `None` for
/// any language.
fn prompt_language(language: &str) -> Result<Option<String>> {
    let name = match language.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => return Ok(None),
        "中文" | "zh" | "zh-cn" | "cmn" | "chinese" | "mandarin" => "中文",
        "英文" | "en" | "english" => "英文",
        "日文" | "ja" | "jp" | "japanese" => "日文",
        _ => {
            return Err(InfraError::BadRequest(format!(
            "ASR parameter `language` must be 中文, 英文 or 日文 (or zh, en, ja), got {language:?}"
        )))
        }
    };
    Ok(Some(name.to_string()))
}

/// Text that would put a chat-template token into the prompt.
fn has_special_token(text: &str) -> bool {
    text.contains("<|") || text.contains("|>")
}

/// Per-request prompt options.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct PromptOptions {
    hotwords: Vec<String>,
    language: Option<String>,
    itn: bool,
}

impl PromptOptions {
    fn from_params(params: &BTreeMap<String, Value>) -> Result<Self> {
        let hotwords = match params.get("hotwords") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::String(text)) => text
                .split([',', '，', '\n'])
                .map(str::trim)
                .filter(|word| !word.is_empty())
                .map(str::to_string)
                .collect(),
            Some(Value::Array(words)) => words
                .iter()
                .map(|word| {
                    word.as_str()
                        .map(|word| word.trim().to_string())
                        .ok_or_else(|| {
                            InfraError::BadRequest(format!(
                                "ASR parameter `hotwords` must hold strings, got {word}"
                            ))
                        })
                })
                .filter(|word| !matches!(word, Ok(word) if word.is_empty()))
                .collect::<Result<Vec<_>>>()?,
            Some(other) => {
                return Err(InfraError::BadRequest(format!(
                    "ASR parameter `hotwords` must be a list or a string, got {other}"
                )))
            }
        };
        if let Some(word) = hotwords.iter().find(|word| has_special_token(word)) {
            return Err(InfraError::BadRequest(format!(
                "ASR hotword {word:?} holds a special token (`<|` or `|>`)"
            )));
        }
        if hotwords.len() > MAX_HOTWORDS {
            return Err(InfraError::BadRequest(format!(
                "at most {MAX_HOTWORDS} hotwords, got {}",
                hotwords.len()
            )));
        }
        let language = match params.get("language") {
            None | Some(Value::Null) => None,
            Some(Value::String(language)) => prompt_language(language)?,
            Some(other) => {
                return Err(InfraError::BadRequest(format!(
                    "ASR parameter `language` must be a string, got {other}"
                )))
            }
        };
        let itn = match params.get("itn") {
            None | Some(Value::Null) => true,
            Some(Value::Bool(itn)) => *itn,
            Some(other) => {
                return Err(InfraError::BadRequest(format!(
                    "ASR parameter `itn` must be a boolean, got {other}"
                )))
            }
        };
        Ok(Self {
            hotwords,
            language,
            itn,
        })
    }

    /// Upstream `get_prompt`: the user turn before the audio.
    fn user_prompt(&self) -> String {
        let mut prompt = String::new();
        if !self.hotwords.is_empty() {
            prompt.push_str(HOTWORDS_PROMPT);
            prompt.push_str(&format!("热词列表：[{}]\n", self.hotwords.join(", ")));
        }
        match &self.language {
            Some(language) => prompt.push_str(&format!("语音转写成{language}")),
            None => prompt.push_str("语音转写"),
        }
        if !self.itn {
            prompt.push_str("，不进行文本规整");
        }
        prompt.push('：');
        prompt
    }
}

#[derive(Debug)]
struct NanoRecognizer {
    encoder: encoder::AudioEncoder,
    llm: llm::Llm,
    embeddings: embed::EmbeddingTable,
    tokenizer: Tokenizer,
    ctc: ctc::CtcVocab,
    prompt: PromptManifest,
    /// This request's prompt ids before the audio.
    prefix: Vec<i64>,
    hidden: usize,
    eos: Vec<i64>,
    output_frame_ms: u64,
    ctc_guard: bool,
    stats: RecognizerStats,
}

impl NanoRecognizer {
    /// The ids before the audio for `options` (the package's own for the
    /// default prompt).
    fn prompt_prefix(&self, options: &PromptOptions) -> Result<Vec<i64>> {
        if *options == PromptOptions::from_params(&BTreeMap::new())? {
            return Ok(self.prompt.prefix_ids.clone());
        }
        let text = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}",
            self.prompt.system,
            options.user_prompt()
        );
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| InfraError::Adapter(format!("tokenize the Fun-ASR-Nano prompt: {e}")))?;
        Ok(encoding.get_ids().iter().map(|&id| id as i64).collect())
    }

    /// The prompt's embeddings for one segment: the prefix tokens, the
    /// adaptor output in place of upstream's placeholders, the suffix tokens.
    fn prompt(&self, audio: &encoder::AudioOutput) -> Result<Vec<f32>> {
        let used = fake_token_len(audio.frames).min(audio.frames);
        let positions = self.prefix.len() + used + self.prompt.suffix_ids.len();
        let mut embeds = Vec::with_capacity(positions * self.hidden);
        for &id in &self.prefix {
            self.embeddings.append_row(id, &mut embeds)?;
        }
        embeds.extend_from_slice(&audio.embeds[..used * self.hidden]);
        for &id in &self.prompt.suffix_ids {
            self.embeddings.append_row(id, &mut embeds)?;
        }
        Ok(embeds)
    }
}

impl SegmentRecognizer for NanoRecognizer {
    fn recognize(&mut self, features: SenseVoiceFeatures) -> Result<Option<RecognizedText>> {
        let started = Instant::now();
        let audio = self.encoder.run(&features)?;
        let audio_ms = ms(started);
        self.stats.segments += 1;
        self.stats.audio_ms += audio_ms;
        let ctc_text = self.ctc.decode(&audio.ctc_ids);
        if self.ctc_guard && ctc_text.is_empty() {
            self.stats.silent_segments += 1;
            return Ok(None);
        }
        let embeds = self.prompt(&audio)?;
        let positions = embeds.len() / self.hidden;
        let seconds = (audio.frames as u64 * self.output_frame_ms) as f64 / 1000.0;
        let limit = max_new_tokens(seconds).min(self.llm.room_after(positions));
        let at = Instant::now();
        let mut token = self.llm.prefill(embeds)?;
        let prefill_ms = ms(at);
        let at = Instant::now();
        let mut generated = Vec::new();
        // Up to `limit` tokens and then their EOS (`room_after` keeps a
        // position for that last step).
        let stopped = loop {
            if self.eos.contains(&token) {
                break true;
            }
            if generated.len() == limit {
                break false;
            }
            generated.push(token);
            token = self.llm.step(self.embeddings.row(token)?)?;
        };
        let decode_ms = ms(at);
        self.stats.prefill_ms += prefill_ms;
        self.stats.decode_ms += decode_ms;
        self.stats.generated_tokens += generated.len();
        let ids = generated
            .iter()
            .map(|&id| u32::try_from(id))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| InfraError::Adapter("Fun-ASR-Nano generated a negative id".to_string()))?;
        let text = self
            .tokenizer
            .decode(&ids, true)
            .map_err(|e| InfraError::Adapter(format!("decode Fun-ASR-Nano output: {e}")))?;
        let mut text = clean_text(&text);
        if !stopped && self.ctc_guard {
            tracing::warn!(
                frames = audio.frames,
                tokens = generated.len(),
                "Fun-ASR-Nano hit its token limit; using the CTC text"
            );
            self.stats.token_limit_segments += 1;
            text = ctc_text;
        }
        tracing::debug!(
            frames = audio.frames,
            prompt = self.prefix.len(),
            tokens = generated.len(),
            audio_ms,
            prefill_ms,
            decode_ms,
            "Fun-ASR-Nano segment"
        );
        Ok(Some(RecognizedText {
            text,
            ..RecognizedText::default()
        }))
    }
}

/// Audio placeholders in the prompt for `frames` LFR frames (upstream
/// `use_low_frame_rate`: three halvings of a 3-wide, stride-2 window). The
/// adaptor attends over every frame and was trained to put the utterance in
/// its first ~1/8 outputs; feeding more makes the LLM loop.
pub fn fake_token_len(frames: usize) -> usize {
    // 1 + (length - 3 + 2) // 2
    let halve = |length: usize| if length == 0 { 0 } else { 1 + (length - 1) / 2 };
    halve(halve(halve(frames)))
}

/// Generated tokens allowed for `seconds` of speech.
fn max_new_tokens(seconds: f64) -> usize {
    MAX_NEW_TOKENS_BASE + (seconds * MAX_NEW_TOKENS_PER_SECOND).ceil() as usize
}

/// Upstream post-processing: `/sil` marks as spaces, runs of white space
/// collapsed.
fn clean_text(text: &str) -> String {
    text.replace("/sil", " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn metadata_bool(spec: &ModelSpec, key: &str, default: bool) -> Result<bool> {
    match spec.metadata.get(key) {
        None => Ok(default),
        Some(value) => value.as_bool().ok_or_else(|| {
            InfraError::Adapter(format!(
                "model metadata `{key}` must be a boolean, got {value}"
            ))
        }),
    }
}

fn metadata_usize(spec: &ModelSpec, key: &str, default: usize, maximum: usize) -> Result<usize> {
    match spec.metadata.get(key) {
        None => Ok(default),
        Some(value) => value
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .filter(|&value| value > 0 && value <= maximum)
            .ok_or_else(|| {
                InfraError::Adapter(format!(
                    "model metadata `{key}` must be an integer in 1..={maximum}, got {value}"
                ))
            }),
    }
}

/// A list of input lengths in LFR frames, sorted and deduplicated; empty
/// when absent.
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

#[cfg(test)]
mod tests;
