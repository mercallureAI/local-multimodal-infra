//! The Qwen3-0.6B step graph: one run per prompt (prefill) or token.
//!
//! The KV cache (28 layers, FP16, fixed capacity) is bound once with past and
//! present sharing memory. A prompt is prefilled with host inputs; decode runs
//! bind only fixed device buffers, so on CUDA the step replays as a CUDA graph
//! and per token only the token's embedding and the attention mask go up and
//! the next id comes back. Embeddings are looked up by the caller (the table
//! stays in host memory).

use local_backend_ort::{
    FixedTensorSpec, OrtBackend, OrtSession, OrtTensorData, OrtTensorInput, ProviderKind,
    SessionProviderReport, SharedKvPair, StaticIoBinding, TensorElement,
};
use local_error::{InfraError, Result};
use std::path::Path;

/// `gpu_graph_id` of the decode step.
const DECODE_GRAPH: i64 = 1;
/// Runs that must not touch the captured graph (prefill).
const NO_GRAPH: i64 = -1;

pub struct Llm {
    session: OrtSession,
    binding: StaticIoBinding,
    cuda_graph: bool,
    hidden: usize,
    /// Host copy of the decode attention mask (`[1, capacity]`).
    mask: Vec<i64>,
    /// Positions in the cache.
    length: usize,
}

impl std::fmt::Debug for Llm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Llm")
            .field("provider", &self.session.provider())
            .field("cuda_graph", &self.cuda_graph)
            .field("capacity", &self.capacity())
            .finish()
    }
}

impl Llm {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        backend: &OrtBackend,
        path: &Path,
        layers: usize,
        kv_heads: usize,
        head_dim: usize,
        hidden: usize,
        capacity: usize,
        cuda_graph: bool,
    ) -> Result<Self> {
        let session = backend.load_session(path)?;
        if session.provider() != ProviderKind::Cuda {
            // The builder's GroupQueryAttention fuses Qwen3's q/k norms, which
            // only the CUDA kernel implements.
            return Err(InfraError::Unsupported(format!(
                "Fun-ASR-Nano's LLM graph runs on CUDA only, got {:?}",
                session.provider()
            )));
        }
        let kv = (0..layers)
            .flat_map(|i| {
                ["key", "value"].map(|part| SharedKvPair {
                    past_input: format!("past_key_values.{i}.{part}"),
                    present_output: format!("present.{i}.{part}"),
                })
            })
            .collect::<Vec<_>>();
        let fixed_inputs = [
            FixedTensorSpec::new("embeds", TensorElement::F32, &[1, 1, hidden]),
            FixedTensorSpec::new("attention_mask", TensorElement::I64, &[1, capacity]),
        ];
        let fixed_outputs = [FixedTensorSpec::new("next_id", TensorElement::I64, &[1, 1])];
        let binding = session.create_static_binding(
            &kv,
            [1, kv_heads, capacity, head_dim],
            &fixed_inputs,
            &fixed_outputs,
        )?;
        Ok(Self {
            session,
            binding,
            cuda_graph,
            hidden,
            mask: vec![0; capacity],
            length: 0,
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

    /// Starts a new sequence: prefills the prompt's embeddings (positions x
    /// hidden) and returns the first generated id.
    pub fn prefill(&mut self, embeds: Vec<f32>) -> Result<i64> {
        if embeds.is_empty() || !embeds.len().is_multiple_of(self.hidden) {
            return Err(InfraError::Adapter(format!(
                "Fun-ASR-Nano prompt embeddings have {} values, not a multiple of {}",
                embeds.len(),
                self.hidden
            )));
        }
        let s = embeds.len() / self.hidden;
        if s + 1 > self.capacity() {
            return Err(InfraError::BadRequest(format!(
                "Fun-ASR-Nano prompt of {s} positions exceeds the LLM context of {}",
                self.capacity()
            )));
        }
        let inputs = vec![
            input(
                "embeds",
                vec![1, s, self.hidden],
                OrtTensorData::F32(embeds),
            ),
            input("attention_mask", vec![1, s], OrtTensorData::I64(vec![1; s])),
        ];
        let graph = self.cuda_graph.then_some(NO_GRAPH);
        self.session
            .run_static_host(&mut self.binding, inputs, &[], graph)?;
        self.mask.fill(0);
        self.mask[..s].fill(1);
        self.length = s;
        self.read_next()
    }

    /// Feeds the last generated token's embedding and returns the next id.
    pub fn step(&mut self, embed: Vec<f32>) -> Result<i64> {
        if self.length + 1 > self.capacity() {
            return Err(InfraError::Runtime(format!(
                "Fun-ASR-Nano LLM context of {} positions is full",
                self.capacity()
            )));
        }
        self.mask[self.length] = 1;
        self.binding.write("embeds", &OrtTensorData::F32(embed))?;
        self.binding
            .write("attention_mask", &OrtTensorData::I64(self.mask.clone()))?;
        let graph = self.cuda_graph.then_some(DECODE_GRAPH);
        self.session.run_static_fixed(&mut self.binding, graph)?;
        self.length += 1;
        self.read_next()
    }

    /// Room left for generated tokens after a prompt of `prompt` positions.
    pub fn room_after(&self, prompt: usize) -> usize {
        self.capacity().saturating_sub(prompt + 1)
    }

    fn read_next(&mut self) -> Result<i64> {
        match self.binding.read("next_id")?.data {
            OrtTensorData::I64(values) if values.len() == 1 => Ok(values[0]),
            other => Err(InfraError::Adapter(format!(
                "Fun-ASR-Nano next_id output is {:?} of {} values",
                other.element_type(),
                other.len()
            ))),
        }
    }
}

fn input(name: &str, shape: Vec<usize>, data: OrtTensorData) -> OrtTensorInput {
    OrtTensorInput {
        name: name.to_string(),
        shape,
        data,
    }
}
