//! Streaming codec decoder: codes of new frames in, their 24 kHz audio out.
//!
//! Decoding a stream chunk by chunk equals decoding it at once: the graph
//! carries the pre-conv context, the conv stack's left context (masked out at
//! a stream's start) and each pre-transformer layer's sliding-window KV cache
//! (bound in place, like the talker's).
//!
//! Every run has the same shape — `chunk` frames, the last one padded — because
//! ORT re-plans every cuDNN convolution (about 40 ms) whenever a shape
//! changes; fixed shapes also let the run replay as a CUDA graph. The state
//! moves from outputs to inputs on the device.

use crate::{
    artifacts::PackageConfig,
    prompt::{Codes, GROUPS},
};
use half::f16;
use local_backend_ort::{
    FixedTensorSpec, OrtBackend, OrtSession, OrtTensorData, OrtTensorInput, ProviderKind,
    SessionProviderReport, SharedKvPair, StaticIoBinding, TensorElement,
};
use local_error::{InfraError, Result};
use std::path::Path;

const GRAPH: i64 = 1;

/// The stream state after priming with a reference voice.
#[derive(Debug, Clone)]
struct Primed {
    key: String,
    pre_ctx: Vec<f16>,
    conv_ctx: Vec<f16>,
    length: usize,
}

pub struct Vocoder {
    session: OrtSession,
    binding: StaticIoBinding,
    chunk: usize,
    samples_per_frame: usize,
    pre_len: usize,
    conv_len: usize,
    cuda_graph: bool,
    /// Frames in the stream (positions in the KV cache).
    length: usize,
    context_valid: Option<bool>,
    primed: Option<Primed>,
}

impl std::fmt::Debug for Vocoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vocoder")
            .field("provider", &self.session.provider())
            .field("chunk", &self.chunk)
            .field("cuda_graph", &self.cuda_graph)
            .field("capacity", &self.binding.kv_capacity())
            .finish()
    }
}

impl Vocoder {
    pub fn load(
        backend: &OrtBackend,
        path: &Path,
        config: &PackageConfig,
        chunk: usize,
        capacity: usize,
        cuda_graph: bool,
    ) -> Result<Self> {
        let v = config.vocoder;
        if chunk < v.conv_ctx_frames {
            return Err(InfraError::ModelNotConfigured {
                model_id: "qwen3-tts".to_string(),
                reason: format!(
                    "vocoder chunks must be at least {} frames, got {chunk}",
                    v.conv_ctx_frames
                ),
            });
        }
        let capacity = capacity.min(v.max_frames);
        let session = backend.load_session(path)?;
        let cuda_graph = cuda_graph && session.provider() == ProviderKind::Cuda;
        let kv = (0..v.layers)
            .flat_map(|i| {
                ["key", "value"].map(|part| SharedKvPair {
                    past_input: format!("past_{part}.{i}"),
                    present_output: format!("present_{part}.{i}"),
                })
            })
            .collect::<Vec<_>>();
        let pre = [1, v.pre_ctx_channels, v.pre_ctx_frames];
        let conv = [1, v.conv_ctx_channels, v.conv_ctx_frames];
        let inputs = [
            FixedTensorSpec::new("codes", TensorElement::I64, &[1, GROUPS, chunk]),
            FixedTensorSpec::new("pre_ctx", TensorElement::F16, &pre),
            FixedTensorSpec::new("conv_ctx", TensorElement::F16, &conv),
            FixedTensorSpec::new("context_valid", TensorElement::F32, &[1]),
            FixedTensorSpec::new("seqlens_k", TensorElement::I32, &[1]),
        ];
        let outputs = [
            FixedTensorSpec::new(
                "audio",
                TensorElement::F32,
                &[1, chunk * config.samples_per_frame],
            ),
            FixedTensorSpec::new("next_pre_ctx", TensorElement::F16, &pre),
            FixedTensorSpec::new("next_conv_ctx", TensorElement::F16, &conv),
        ];
        let mut binding = session.create_static_binding(
            &kv,
            [1, v.kv_heads, capacity, v.head_dim],
            &inputs,
            &outputs,
        )?;
        // GroupQueryAttention reads the total length on the host: the cache
        // capacity (the valid length is `seqlens_k`).
        binding.bind_constant(OrtTensorInput {
            name: "total_sequence_length".to_string(),
            shape: vec![],
            data: OrtTensorData::I32(vec![capacity as i32]),
        })?;
        let mut vocoder = Self {
            session,
            binding,
            chunk,
            samples_per_frame: config.samples_per_frame,
            pre_len: pre.iter().product(),
            conv_len: conv.iter().product(),
            cuda_graph,
            length: 0,
            context_valid: None,
            primed: None,
        };
        vocoder.reset()?;
        Ok(vocoder)
    }

    pub fn provider_report(&self) -> SessionProviderReport {
        self.session.provider_report()
    }

    pub fn chunk(&self) -> usize {
        self.chunk
    }

    pub fn uses_cuda_graph(&self) -> bool {
        self.cuda_graph
    }

    /// Starts a new stream from silence.
    pub fn reset(&mut self) -> Result<()> {
        self.binding.write(
            "pre_ctx",
            &OrtTensorData::F16(vec![f16::ZERO; self.pre_len]),
        )?;
        self.binding.write(
            "conv_ctx",
            &OrtTensorData::F16(vec![f16::ZERO; self.conv_len]),
        )?;
        self.set_context_valid(false)?;
        self.length = 0;
        Ok(())
    }

    /// Starts a new stream that continues `codes` (a reference voice), as
    /// upstream decodes reference and generated codes together. The state
    /// after the reference is kept for the next stream with the same `key`.
    /// Whole chunks only: the oldest `len % chunk` frames are left out.
    pub fn prime(&mut self, key: &str, codes: &[Codes]) -> Result<()> {
        if let Some(primed) = self.primed.clone().filter(|primed| primed.key == key) {
            if self.binding.restore_kv(key)? {
                self.binding
                    .write("pre_ctx", &OrtTensorData::F16(primed.pre_ctx))?;
                self.binding
                    .write("conv_ctx", &OrtTensorData::F16(primed.conv_ctx))?;
                self.set_context_valid(primed.length > 0)?;
                self.length = primed.length;
                return Ok(());
            }
        }
        self.reset()?;
        let skip = codes.len() % self.chunk;
        for chunk in codes[skip..].chunks(self.chunk) {
            self.decode(chunk)?;
        }
        // One saved voice at a time (a saved copy is the whole KV cache).
        self.binding.clear_kv_snapshots();
        self.binding.save_kv(key)?;
        self.primed = Some(Primed {
            key: key.to_string(),
            pre_ctx: f16_data(self.binding.read("pre_ctx")?.data, "pre_ctx")?,
            conv_ctx: f16_data(self.binding.read("conv_ctx")?.data, "conv_ctx")?,
            length: self.length,
        });
        Ok(())
    }

    /// Decodes the next frames (at most one chunk) of the stream.
    pub fn decode(&mut self, frames: &[Codes]) -> Result<Vec<f32>> {
        let n = frames.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if n > self.chunk {
            return Err(InfraError::Adapter(format!(
                "{n} frames exceed the vocoder chunk of {}",
                self.chunk
            )));
        }
        let capacity = self.binding.kv_capacity();
        if self.length + self.chunk > capacity {
            return Err(InfraError::Runtime(format!(
                "Qwen3-TTS vocoder stream exceeds {capacity} frames"
            )));
        }
        // [1, 16, chunk], codebook-major; a short last chunk repeats its last
        // frame (causal: the padding never reaches the real frames' audio).
        let mut codes = Vec::with_capacity(GROUPS * self.chunk);
        for group in 0..GROUPS {
            codes.extend((0..self.chunk).map(|i| frames[i.min(n - 1)][group]));
        }
        self.binding.write("codes", &OrtTensorData::I64(codes))?;
        self.binding.write(
            "seqlens_k",
            &OrtTensorData::I32(vec![(self.length + self.chunk - 1) as i32]),
        )?;
        let run_started = std::time::Instant::now();
        self.session
            .run_static_fixed(&mut self.binding, self.cuda_graph.then_some(GRAPH))?;
        let run_us = run_started.elapsed().as_micros() as u64;
        self.binding.copy_fixed("next_pre_ctx", "pre_ctx")?;
        self.binding.copy_fixed("next_conv_ctx", "conv_ctx")?;
        self.set_context_valid(true)?;
        self.length += self.chunk;
        let audio = self.binding.read("audio")?;
        tracing::trace!(
            frames = n,
            run_us,
            total_us = run_started.elapsed().as_micros() as u64,
            "vocoder chunk"
        );
        match audio.data {
            OrtTensorData::F32(mut samples) => {
                samples.truncate(n * self.samples_per_frame);
                Ok(samples)
            }
            other => Err(InfraError::Adapter(format!(
                "vocoder audio is {:?}, expected f32",
                other.element_type()
            ))),
        }
    }

    fn set_context_valid(&mut self, valid: bool) -> Result<()> {
        if self.context_valid != Some(valid) {
            self.binding.write(
                "context_valid",
                &OrtTensorData::F32(vec![if valid { 1.0 } else { 0.0 }]),
            )?;
            self.context_valid = Some(valid);
        }
        Ok(())
    }
}

fn f16_data(data: OrtTensorData, name: &str) -> Result<Vec<f16>> {
    match data {
        OrtTensorData::F16(values) => Ok(values),
        other => Err(InfraError::Adapter(format!(
            "vocoder `{name}` is {:?}, expected f16",
            other.element_type()
        ))),
    }
}
