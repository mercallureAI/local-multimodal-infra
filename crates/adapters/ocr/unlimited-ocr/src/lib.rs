//! Document OCR with baidu/Unlimited-OCR (DeepSeek-OCR architecture: SAM +
//! CLIP vision encoder, DeepSeek-V2 MoE decoder) on ONNX Runtime.
//!
//! The package (`scripts/local/unlimited_ocr_export.py`) has a vision graph and
//! prefill and decode graphs of the LLM. The sessions are loaded once; the LLM
//! weights are uploaded to the device once and shared by both LLM sessions.
//! The KV cache stays on the device, bound to both LLM sessions, and is updated
//! in place at the slots this adapter chooses, which reproduces upstream's
//! R-SWA attention: the prompt (BOS, image tokens, instruction) keeps its KV,
//! and generated tokens share a ring of `sliding_window` slots.

mod artifacts;
mod preprocess;
mod sampler;

pub use artifacts::{PackageManifest, PackageRuntime, UnlimitedOcrArtifacts};

use local_backend_ort::{
    CudaMemoryOptions, OrtBackend, OrtSession, OrtTensorData, OrtTensorInput, ProviderKind,
    ProviderSelection, SessionProviderReport, SharedInitializers, SharedKvBinding, SharedKvPair,
};
use local_core::{FileRef, InferenceOutput, ModelSpec};
use local_error::{InfraError, Result};
use std::time::Instant;
use tokenizers::Tokenizer;

/// Generated tokens unless the model spec sets `max_new_tokens`.
pub const DEFAULT_MAX_NEW_TOKENS: usize = 8192;
pub const MAX_NEW_TOKENS_ENV: &str = "LOCAL_UNLIMITED_OCR_MAX_NEW_TOKENS";
/// In the error for an int8 package without CUDA; the smoke scripts match it.
pub const NEEDS_CUDA: &str = "packed for CUDA";

#[derive(Debug, Clone, Default)]
pub struct OcrTimings {
    pub vision_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
}

#[derive(Debug)]
pub struct UnlimitedOcrAdapter {
    model_id: String,
    artifacts: UnlimitedOcrArtifacts,
    tokenizer: Tokenizer,
    /// Prompt text ids around the image block: `[BOS] + before`, `after`.
    prompt_head: Vec<u32>,
    prompt_tail: Vec<u32>,
    max_new_tokens: usize,
    vision: OrtSession,
    prefill: OrtSession,
    decode: OrtSession,
    cache: Option<KvCache>,
    last_timings: OcrTimings,
    /// Device weights of `prefill` and `decode`: must outlive both sessions,
    /// so it is the last field.
    _shared_weights: Option<SharedInitializers>,
}

/// One cache, bound to the prefill and the decode session.
#[derive(Debug)]
struct KvCache {
    prefill: SharedKvBinding,
    decode: SharedKvBinding,
}

impl UnlimitedOcrAdapter {
    pub fn load(spec: &ModelSpec) -> Result<Self> {
        let artifacts = UnlimitedOcrArtifacts::from_spec(spec)?;
        let runtime = &artifacts.manifest.runtime;
        let tokenizer = Tokenizer::from_file(&artifacts.tokenizer).map_err(|e| {
            InfraError::Adapter(format!(
                "load tokenizer {}: {e}",
                artifacts.tokenizer.display()
            ))
        })?;
        let (before, after) = runtime.prompt.split_once("<image>").ok_or_else(|| {
            InfraError::Adapter(format!(
                "OCR prompt `{}` has no <image> marker",
                runtime.prompt
            ))
        })?;
        let encode = |text: &str| -> Result<Vec<u32>> {
            Ok(tokenizer
                .encode(text, false)
                .map_err(|e| InfraError::Adapter(format!("tokenize OCR prompt: {e}")))?
                .get_ids()
                .to_vec())
        };
        let mut prompt_head = vec![runtime.bos_token_id];
        prompt_head.extend(encode(before)?);
        let prompt_tail = encode(after)?;
        let max_new_tokens = max_new_tokens(spec)?;

        // int8 experts are CUTLASS-prepacked: only the CUDA `QMoE` kernel reads
        // them (the CPU kernel rejects or misreads them), so such a package
        // never falls back to CPU.
        let int8_experts = artifacts.manifest.precision.routed_experts.as_deref() == Some("int8");
        let mut selection = ProviderSelection::from_strings(&spec.runtime.provider_order);
        if int8_experts {
            selection
                .order
                .retain(|provider| provider.kind != ProviderKind::Cpu);
            selection.fallback_to_cpu = false;
        }
        // The SAM encoder's activations (global attention over 4096 patches)
        // are several GB: grow the arena by what is asked and give the vision
        // run's memory back after each page.
        let backend = OrtBackend::new(selection).with_cuda_memory_options(CudaMemoryOptions {
            arena_same_as_requested: true,
            conv_max_workspace: false,
        });
        let needs_cuda = |found: &str| {
            InfraError::Unsupported(format!(
                "model `{}`: the int8 experts of {} are {NEEDS_CUDA} and cannot run on {found}; \
                 use a CUDA worker or re-export with --expert-precision fp16",
                spec.id,
                artifacts.root.display()
            ))
        };
        if int8_experts && backend.preferred_cuda_device().is_none() {
            return Err(needs_cuda("this worker (no usable CUDA device)"));
        }
        let started = Instant::now();
        let vision = backend.load_session(&artifacts.vision)?;
        let shared_weights =
            match &artifacts.manifest.device_shared_initializers {
                Some(shared) if !shared.tensors.is_empty() => Some(backend.upload_initializers(
                    &artifacts.root.join(&shared.data_file),
                    &shared.tensors,
                )?),
                _ => None,
            };
        let load_llm = |path: &std::path::Path| match &shared_weights {
            Some(shared) => backend.load_session_with_initializers(path, shared),
            None => backend.load_session(path),
        };
        let prefill = load_llm(&artifacts.prefill)?;
        let decode = load_llm(&artifacts.decode)?;
        // Defensive: with CPU removed from the order the sessions can only be
        // CUDA, but never run the CUTLASS layout through another provider.
        if int8_experts
            && (prefill.provider() != ProviderKind::Cuda || decode.provider() != ProviderKind::Cuda)
        {
            return Err(needs_cuda(&format!("the {:?} provider", decode.provider())));
        }
        tracing::info!(
            model_id = spec.id,
            root = %artifacts.root.display(),
            vision_provider = ?vision.provider(),
            llm_provider = ?decode.provider(),
            shared_weight_mib = shared_weights.as_ref().map(|w| w.bytes() >> 20),
            load_ms = started.elapsed().as_millis() as u64,
            max_new_tokens,
            "Unlimited-OCR loaded"
        );
        Ok(Self {
            model_id: spec.id.clone(),
            artifacts,
            tokenizer,
            prompt_head,
            prompt_tail,
            max_new_tokens,
            vision,
            prefill,
            decode,
            cache: None,
            last_timings: OcrTimings::default(),
            _shared_weights: shared_weights,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn artifacts(&self) -> &UnlimitedOcrArtifacts {
        &self.artifacts
    }

    pub fn provider_report(&self) -> SessionProviderReport {
        self.decode.provider_report()
    }

    pub fn last_timings(&self) -> &OcrTimings {
        &self.last_timings
    }

    pub fn ocr_recognize(&mut self, image: &FileRef) -> Result<InferenceOutput> {
        let path = local_files::local_path(image)?;
        let page = preprocess::load_page(&path)?;
        let text = self.recognize_pages(&[page])?;
        Ok(InferenceOutput::OcrText { text })
    }

    /// OCR of one image (base mode). Several pages would use upstream's
    /// multi-page prompt; only one page is wired up for now.
    pub fn recognize_pages(&mut self, pages: &[image::RgbImage]) -> Result<String> {
        if pages.len() != 1 {
            return Err(InfraError::Unsupported(
                "Unlimited-OCR accepts one page per request".to_string(),
            ));
        }
        let runtime = self.artifacts.manifest.runtime.clone();
        let size = runtime.image_size as usize;
        let mut timings = OcrTimings::default();

        let started = Instant::now();
        let pixels = preprocess::page_tensor(&pages[0], runtime.image_size);
        let features = self
            .vision
            .run_tensors_releasing_memory(&[OrtTensorInput {
                name: "pixel_values".to_string(),
                shape: vec![1, 3, size, size],
                data: OrtTensorData::F32(pixels),
            }])?
            .into_iter()
            .next()
            .ok_or_else(|| InfraError::Backend("vision graph returned no output".to_string()))?;
        let OrtTensorData::F32(features) = features.data else {
            return Err(InfraError::Backend(
                "vision graph image_features is not f32".to_string(),
            ));
        };
        let image_tokens = runtime.image_tokens_per_page * pages.len();
        if features.len() != image_tokens * runtime.hidden_size {
            return Err(InfraError::Backend(format!(
                "vision graph returned {} values, expected {image_tokens} x {}",
                features.len(),
                runtime.hidden_size
            )));
        }
        timings.vision_ms = ms(started);

        let mut ids = self.prompt_head.clone();
        ids.extend(std::iter::repeat(runtime.image_token_id).take(image_tokens));
        ids.extend(&self.prompt_tail);
        let mut mask = vec![false; ids.len()];
        mask[self.prompt_head.len()..self.prompt_head.len() + image_tokens].fill(true);
        let prompt = ids.len();
        let schedule = RingSchedule::new(prompt, runtime.sliding_window, self.max_new_tokens);
        timings.prompt_tokens = prompt;

        self.ensure_cache(schedule.capacity, &runtime)?;
        let cache = self.cache.as_mut().expect("cache is allocated");

        let started = Instant::now();
        let prefill = prefill_inputs(
            ids.iter().map(|&id| i64::from(id)).collect(),
            mask,
            features,
            runtime.hidden_size,
            (0..prompt as i64).collect(),
            (0..prompt as i64).collect(),
            schedule.prefill_bias(),
        );
        let mut logits = logits_f32(
            self.prefill
                .run_shared_kv_binding_releasing_memory(&mut cache.prefill, prefill)?,
        )?;
        timings.prefill_ms = ms(started);

        let started = Instant::now();
        let mut history = ids;
        let mut generated = Vec::new();
        for step in 0..self.max_new_tokens {
            let banned = sampler::banned_tokens(
                &history,
                runtime.no_repeat_ngram_size,
                runtime.ngram_window,
            );
            let token = sampler::greedy(&logits, &banned);
            if token == runtime.eos_token_id {
                break;
            }
            generated.push(token);
            history.push(token);
            if step + 1 == self.max_new_tokens {
                break;
            }
            let inputs = decode_inputs(
                i64::from(token),
                (prompt + step) as i64,
                schedule.slot(step) as i64,
                schedule.decode_bias(step),
            );
            logits = logits_f32(
                self.decode
                    .run_shared_kv_binding(&mut cache.decode, inputs)?,
            )?;
        }
        timings.decode_ms = ms(started);
        timings.generated_tokens = generated.len();

        let text = self
            .tokenizer
            .decode(&generated, false)
            .map_err(|e| InfraError::Adapter(format!("detokenize OCR output: {e}")))?;
        tracing::info!(
            model_id = %self.model_id,
            prompt_tokens = timings.prompt_tokens,
            generated_tokens = timings.generated_tokens,
            vision_ms = timings.vision_ms,
            prefill_ms = timings.prefill_ms,
            decode_ms = timings.decode_ms,
            "Unlimited-OCR page recognized"
        );
        self.last_timings = timings;
        Ok(text.trim().to_string())
    }

    /// The cache is kept between requests and rebuilt only when the capacity
    /// changes. Stale slots are harmless: every slot is written before the
    /// attention bias unmasks it, and the first allocation is zero-filled.
    fn ensure_cache(&mut self, capacity: usize, runtime: &PackageRuntime) -> Result<()> {
        if self
            .cache
            .as_ref()
            .is_some_and(|cache| cache.prefill.capacity() == capacity)
        {
            return Ok(());
        }
        self.cache = None;
        let pairs = (0..runtime.num_hidden_layers)
            .flat_map(|layer| {
                ["key", "value"].map(|kind| SharedKvPair {
                    past_input: format!("past_key_values.{layer}.{kind}"),
                    present_output: format!("present.{layer}.{kind}"),
                })
            })
            .collect::<Vec<_>>();
        let prefill = self.prefill.create_zeroed_shared_kv_binding(
            &pairs,
            [1, runtime.num_key_value_heads, capacity, runtime.head_dim],
            "logits",
        )?;
        let decode = self.decode.share_kv_binding(&prefill, "logits")?;
        self.cache = Some(KvCache { prefill, decode });
        Ok(())
    }
}

/// Where each token's KV goes and which slots it may attend to.
///
/// Prefill writes the prompt to slots `[0, prompt)` with a causal bias.
/// Decode step `t` (the token at position `prompt + t`) writes slot
/// `prompt + t % window` and attends to the prompt plus every ring slot written
/// so far, itself included: upstream's warm-up append and ring overwrite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RingSchedule {
    prompt: usize,
    window: usize,
    capacity: usize,
}

impl RingSchedule {
    fn new(prompt: usize, window: usize, max_new_tokens: usize) -> Self {
        let window = window.max(1);
        Self {
            prompt,
            window,
            capacity: prompt + window.min(max_new_tokens.max(1)),
        }
    }

    fn slot(&self, step: usize) -> usize {
        self.prompt + step % self.window
    }

    fn prefill_bias(&self) -> Vec<f32> {
        let mut bias = vec![f32::NEG_INFINITY; self.prompt * self.capacity];
        for row in 0..self.prompt {
            bias[row * self.capacity..row * self.capacity + row + 1].fill(0.0);
        }
        bias
    }

    fn decode_bias(&self, step: usize) -> Vec<f32> {
        let mut bias = vec![f32::NEG_INFINITY; self.capacity];
        let visible = self.prompt + (step + 1).min(self.window);
        bias[..visible.min(self.capacity)].fill(0.0);
        bias
    }
}

fn prefill_inputs(
    ids: Vec<i64>,
    image_mask: Vec<bool>,
    image_features: Vec<f32>,
    hidden: usize,
    positions: Vec<i64>,
    write_index: Vec<i64>,
    bias: Vec<f32>,
) -> Vec<OrtTensorInput> {
    let seq = ids.len();
    let capacity = bias.len() / seq;
    let tensor = |name: &str, shape: Vec<usize>, data: OrtTensorData| OrtTensorInput {
        name: name.to_string(),
        shape,
        data,
    };
    vec![
        tensor("input_ids", vec![1, seq], OrtTensorData::I64(ids)),
        tensor(
            "images_seq_mask",
            vec![1, seq],
            OrtTensorData::Bool(image_mask),
        ),
        tensor(
            "image_features",
            vec![image_features.len() / hidden, hidden],
            OrtTensorData::F32(image_features),
        ),
        tensor("position_ids", vec![1, seq], OrtTensorData::I64(positions)),
        tensor("write_index", vec![seq], OrtTensorData::I64(write_index)),
        tensor(
            "attention_bias",
            vec![1, 1, seq, capacity],
            OrtTensorData::F32(bias),
        ),
    ]
}

fn decode_inputs(token: i64, position: i64, slot: i64, bias: Vec<f32>) -> Vec<OrtTensorInput> {
    let capacity = bias.len();
    vec![
        OrtTensorInput {
            name: "input_ids".to_string(),
            shape: vec![1, 1],
            data: OrtTensorData::I64(vec![token]),
        },
        OrtTensorInput {
            name: "position_ids".to_string(),
            shape: vec![1, 1],
            data: OrtTensorData::I64(vec![position]),
        },
        OrtTensorInput {
            name: "write_index".to_string(),
            shape: vec![1],
            data: OrtTensorData::I64(vec![slot]),
        },
        OrtTensorInput {
            name: "attention_bias".to_string(),
            shape: vec![1, 1, 1, capacity],
            data: OrtTensorData::F32(bias),
        },
    ]
}

fn logits_f32(output: local_backend_ort::OrtTensorOutput) -> Result<Vec<f32>> {
    match output.data {
        OrtTensorData::F32(data) => Ok(data),
        other => Err(InfraError::Backend(format!(
            "OCR logits are {:?}, expected f32",
            other.element_type()
        ))),
    }
}

fn max_new_tokens(spec: &ModelSpec) -> Result<usize> {
    if let Ok(value) = std::env::var(MAX_NEW_TOKENS_ENV) {
        return value
            .trim()
            .parse::<usize>()
            .map_err(|e| InfraError::BadRequest(format!("{MAX_NEW_TOKENS_ENV}={value:?}: {e}")));
    }
    Ok(spec
        .metadata
        .get("max_new_tokens")
        .and_then(|value| value.as_u64())
        .map(|value| value as usize)
        .unwrap_or(DEFAULT_MAX_NEW_TOKENS))
}

fn ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1e3
}

#[cfg(test)]
mod tests;
