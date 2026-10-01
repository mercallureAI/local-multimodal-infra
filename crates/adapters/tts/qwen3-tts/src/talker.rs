//! The talker frame graph: one run per audio frame.
//!
//! The KV cache (28 layers, FP16, fixed capacity) is bound once with past and
//! present sharing memory. A prompt is prefilled with host inputs; decode runs
//! bind only fixed device buffers, so on CUDA the frame replays as a CUDA graph
//! and per frame only a few small inputs go up and the 16 codes come back.

use crate::{
    artifacts::PackageConfig,
    prompt::{Codes, Prompt, GROUPS},
};
use half::f16;
use local_backend_ort::{
    FixedTensorSpec, OrtBackend, OrtSession, OrtTensorData, OrtTensorInput, ProviderKind,
    SessionProviderReport, SharedKvPair, StaticIoBinding, TensorElement,
};
use local_error::{InfraError, Result};
use std::path::Path;

/// `gpu_graph_id` of the decode frame.
const DECODE_GRAPH: i64 = 1;
/// Runs that must not touch the captured graph (prefill).
const NO_GRAPH: i64 = -1;

/// `[1/temperature, 1/subtalker_temperature, repetition_penalty, eos_allowed]`.
pub type Sampling = [f32; 4];

pub struct Talker {
    session: OrtSession,
    binding: StaticIoBinding,
    cuda_graph: bool,
    hidden: usize,
    vocab: usize,
    noise_len: usize,
    /// Host copy of the decode attention mask (`[1, capacity]`).
    mask: Vec<i64>,
    /// Positions in the cache.
    length: usize,
    sampling: Option<Sampling>,
}

impl std::fmt::Debug for Talker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Talker")
            .field("provider", &self.session.provider())
            .field("cuda_graph", &self.cuda_graph)
            .field("capacity", &self.capacity())
            .finish()
    }
}

impl Talker {
    pub fn load(
        backend: &OrtBackend,
        path: &Path,
        config: &PackageConfig,
        capacity: usize,
        cuda_graph: bool,
    ) -> Result<Self> {
        let session = backend.load_session(path)?;
        let cuda_graph = cuda_graph && session.provider() == ProviderKind::Cuda;
        let kv = (0..config.talker_layers)
            .flat_map(|i| {
                ["key", "value"].map(|part| SharedKvPair {
                    past_input: format!("past_key_values.{i}.{part}"),
                    present_output: format!("present.{i}.{part}"),
                })
            })
            .collect::<Vec<_>>();
        let hidden = config.hidden_size;
        let vocab = config.talker_vocab_size;
        let noise_len = GROUPS * config.top_k;
        let fixed_inputs = [
            FixedTensorSpec::new("text_ids", TensorElement::I64, &[1, 1]),
            FixedTensorSpec::new("codec_ids", TensorElement::I64, &[1, 1, GROUPS]),
            FixedTensorSpec::new("extra_embeds", TensorElement::F16, &[1, 1, hidden]),
            FixedTensorSpec::new("attention_mask", TensorElement::I64, &[1, capacity]),
            FixedTensorSpec::new("seen_tokens", TensorElement::F32, &[1, vocab]),
            FixedTensorSpec::new("noise", TensorElement::F32, &[GROUPS, config.top_k]),
            FixedTensorSpec::new("sampling", TensorElement::F32, &[4]),
        ];
        let fixed_outputs = [FixedTensorSpec::new(
            "codes",
            TensorElement::I64,
            &[1, 1, GROUPS],
        )];
        let mut binding = session.create_static_binding(
            &kv,
            [1, config.talker_kv_heads, capacity, config.head_dim],
            &fixed_inputs,
            &fixed_outputs,
        )?;
        binding.write("extra_embeds", &OrtTensorData::F16(vec![f16::ZERO; hidden]))?;
        Ok(Self {
            session,
            binding,
            cuda_graph,
            hidden,
            vocab,
            noise_len,
            mask: vec![0; capacity],
            length: 0,
            sampling: None,
        })
    }

    pub fn capacity(&self) -> usize {
        self.binding.kv_capacity()
    }

    pub fn provider_report(&self) -> SessionProviderReport {
        self.session.provider_report()
    }

    pub fn uses_cuda_graph(&self) -> bool {
        self.cuda_graph
    }

    /// Starts a new utterance: prefills `prompt` and returns the first frame.
    pub fn prefill(
        &mut self,
        prompt: &Prompt,
        speaker: Option<&[f32]>,
        seen: &[f32],
        noise: &[f32],
        sampling: Sampling,
    ) -> Result<Codes> {
        let s = prompt.len();
        if s + 1 > self.capacity() {
            return Err(InfraError::BadRequest(format!(
                "Qwen3-TTS prompt of {s} positions exceeds the talker context of {}",
                self.capacity()
            )));
        }
        self.check_lengths(seen, noise)?;
        let mut extra = vec![f16::ZERO; s * self.hidden];
        if let (Some(position), Some(speaker)) = (prompt.speaker_position, speaker) {
            if speaker.len() != self.hidden {
                return Err(InfraError::Adapter(format!(
                    "speaker embedding has {} values, expected {}",
                    speaker.len(),
                    self.hidden
                )));
            }
            for (slot, value) in extra[position * self.hidden..(position + 1) * self.hidden]
                .iter_mut()
                .zip(speaker)
            {
                *slot = f16::from_f32(*value);
            }
        }
        let inputs = vec![
            input(
                "text_ids",
                vec![1, s],
                OrtTensorData::I64(prompt.text_ids.clone()),
            ),
            input(
                "codec_ids",
                vec![1, s, GROUPS],
                OrtTensorData::I64(prompt.codec_ids.iter().flatten().copied().collect()),
            ),
            input(
                "extra_embeds",
                vec![1, s, self.hidden],
                OrtTensorData::F16(extra),
            ),
            input("attention_mask", vec![1, s], OrtTensorData::I64(vec![1; s])),
            input(
                "seen_tokens",
                vec![1, self.vocab],
                OrtTensorData::F32(seen.to_vec()),
            ),
            input(
                "noise",
                vec![GROUPS, self.noise_len / GROUPS],
                OrtTensorData::F32(noise.to_vec()),
            ),
            input("sampling", vec![4], OrtTensorData::F32(sampling.to_vec())),
        ];
        let graph = self.cuda_graph.then_some(NO_GRAPH);
        self.session
            .run_static_host(&mut self.binding, inputs, &[], graph)?;
        self.mask.fill(0);
        self.mask[..s].fill(1);
        self.length = s;
        self.sampling = None;
        self.read_codes()
    }

    /// Feeds the previous frame (`codes`) plus its text token and returns the
    /// next frame.
    pub fn step(
        &mut self,
        text_id: i64,
        codes: &Codes,
        seen: &[f32],
        noise: &[f32],
        sampling: Sampling,
    ) -> Result<Codes> {
        if self.length + 1 > self.capacity() {
            return Err(InfraError::Runtime(format!(
                "Qwen3-TTS talker context of {} positions is full",
                self.capacity()
            )));
        }
        self.check_lengths(seen, noise)?;
        let write_started = std::time::Instant::now();
        self.mask[self.length] = 1;
        self.binding
            .write("text_ids", &OrtTensorData::I64(vec![text_id]))?;
        self.binding
            .write("codec_ids", &OrtTensorData::I64(codes.to_vec()))?;
        self.binding
            .write("attention_mask", &OrtTensorData::I64(self.mask.clone()))?;
        self.binding
            .write("seen_tokens", &OrtTensorData::F32(seen.to_vec()))?;
        self.binding
            .write("noise", &OrtTensorData::F32(noise.to_vec()))?;
        if self.sampling != Some(sampling) {
            self.binding
                .write("sampling", &OrtTensorData::F32(sampling.to_vec()))?;
            self.sampling = Some(sampling);
        }
        let write_us = write_started.elapsed().as_micros() as u64;
        let graph = self.cuda_graph.then_some(DECODE_GRAPH);
        let run_started = std::time::Instant::now();
        self.session.run_static_fixed(&mut self.binding, graph)?;
        let run_us = run_started.elapsed().as_micros() as u64;
        self.length += 1;
        let codes = self.read_codes();
        tracing::trace!(
            position = self.length,
            write_us,
            run_us,
            total_us = run_started.elapsed().as_micros() as u64,
            "talker frame"
        );
        codes
    }

    fn read_codes(&mut self) -> Result<Codes> {
        let output = self.binding.read("codes")?;
        match output.data {
            OrtTensorData::I64(values) if values.len() == GROUPS => {
                let mut codes = [0; GROUPS];
                codes.copy_from_slice(&values);
                Ok(codes)
            }
            other => Err(InfraError::Adapter(format!(
                "talker codes output is {:?} of {} values",
                other.element_type(),
                other.len()
            ))),
        }
    }

    fn check_lengths(&self, seen: &[f32], noise: &[f32]) -> Result<()> {
        if seen.len() != self.vocab || noise.len() != self.noise_len {
            return Err(InfraError::Adapter(format!(
                "talker sampling inputs have {} / {} values, expected {} / {}",
                seen.len(),
                noise.len(),
                self.vocab,
                self.noise_len
            )));
        }
        Ok(())
    }
}

fn input(name: &str, shape: Vec<usize>, data: OrtTensorData) -> OrtTensorInput {
    OrtTensorInput {
        name: name.to_string(),
        shape,
        data,
    }
}
