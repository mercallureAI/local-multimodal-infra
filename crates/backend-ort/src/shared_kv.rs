//! Fixed-capacity KV cache for decoder graphs whose attention reads `past_*`
//! and writes `present_*` in place (ORT `GroupQueryAttention` with a shared
//! past/present buffer, as exported by the onnxruntime-genai model builder).
//!
//! Each layer's cache is one `[batch, kv_heads, capacity, head_size]` tensor
//! allocated once on the session's device. The same memory is bound as the
//! `past` input (a borrowed view) and the `present` output (the owning
//! tensor), so decode steps never copy or reallocate the cache; the valid
//! length is carried by the `attention_mask` the caller binds for each run.

use super::*;
use crate::io_binding::owned_tensor;
use half::f16;
use ort::{
    memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType},
    session::{IoBinding, SharedSessionInner},
    value::{DynTensorValueType, DynValue, Shape, Tensor, TensorRefMut},
};
use std::sync::Arc;

/// One cache layer: the graph input that reads it and the output that
/// writes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedKvPair {
    pub past_input: String,
    pub present_output: String,
}

#[derive(Debug)]
pub struct SharedKvBinding {
    binding: IoBinding,
    session: Arc<SharedSessionInner>,
    shape: [usize; 4],
    element: TensorElement,
    logits_output: String,
    // The owning cache tensors are bound as outputs (held by `binding`); the
    // input views borrow their memory. Views and binding must be released
    // before the allocator that owns the device memory: fields drop in
    // declaration order, so the allocator is last.
    _allocator: Allocator,
}

impl SharedKvBinding {
    /// `[batch, kv_heads, capacity, head_size]` of every cache tensor.
    pub fn shape(&self) -> [usize; 4] {
        self.shape
    }

    /// The maximum number of tokens the cache holds.
    pub fn capacity(&self) -> usize {
        self.shape[2]
    }

    pub fn element(&self) -> TensorElement {
        self.element
    }
}

impl OrtSession {
    /// Allocates the KV cache on the session's device and binds it in place.
    ///
    /// Every `past_input` must have shape `[batch, kv_heads, dynamic,
    /// head_size]` with the given batch/heads/head size, and all layers must
    /// share one element type (FP16 or FP32). `logits_output` is returned to
    /// host memory on every run.
    pub fn create_shared_kv_binding(
        &self,
        pairs: &[SharedKvPair],
        shape: [usize; 4],
        logits_output: &str,
    ) -> Result<SharedKvBinding> {
        if pairs.is_empty() {
            return Err(InfraError::Backend(
                "shared KV binding needs at least one cache layer".to_string(),
            ));
        }
        if shape.iter().any(|dim| *dim == 0) {
            return Err(InfraError::Backend(format!(
                "shared KV cache shape {shape:?} has an empty dimension"
            )));
        }
        let metadata = self.metadata();
        let element = shared_kv_element(metadata, pairs, shape)?;
        if !metadata
            .outputs
            .iter()
            .any(|output| output.name == logits_output)
        {
            return Err(InfraError::Backend(format!(
                "shared KV binding logits output `{logits_output}` is not a graph output"
            )));
        }

        let device_memory = match (self.provider, self.device_id) {
            (ProviderKind::Cuda, Some(device_id)) if !self.whole_session_cpu_fallback_used() => {
                MemoryInfo::new(
                    AllocationDevice::CUDA,
                    device_id as i32,
                    AllocatorType::Device,
                    MemoryType::Default,
                )
                .map_err(map_ort_err)?
            }
            (ProviderKind::Cpu, _) => MemoryInfo::new(
                AllocationDevice::CPU,
                0,
                AllocatorType::Device,
                MemoryType::Default,
            )
            .map_err(map_ort_err)?,
            (provider, device_id) => {
                return Err(InfraError::Unsupported(format!(
                    "shared KV binding supports CPU and CUDA sessions; got {provider:?} on device {device_id:?} (whole-session CPU retry: {})",
                    self.whole_session_cpu_fallback_used()
                )))
            }
        };
        let allocator =
            Allocator::new(&self.real.session, device_memory.clone()).map_err(map_ort_err)?;
        let mut binding = self.real.session.create_binding().map_err(map_ort_err)?;
        let dims = Shape::new(shape.iter().map(|dim| *dim as i64));
        for pair in pairs {
            match element {
                TensorElement::F16 => {
                    bind_layer::<f16>(&mut binding, &allocator, &device_memory, pair, dims.clone())?
                }
                TensorElement::F32 => {
                    bind_layer::<f32>(&mut binding, &allocator, &device_memory, pair, dims.clone())?
                }
                other => {
                    return Err(InfraError::Backend(format!(
                        "shared KV cache element type {other:?} is not supported"
                    )))
                }
            }
        }
        binding
            .bind_output_to_device(logits_output, &host_output_memory()?)
            .map_err(map_ort_err)?;
        Ok(SharedKvBinding {
            binding,
            session: self.real.session.inner(),
            shape,
            element,
            logits_output: logits_output.to_string(),
            _allocator: allocator,
        })
    }

    /// Runs one step: `host_inputs` (token ids, attention mask, ...) are
    /// copied in, the cache is updated in place, and the logits come back.
    pub fn run_shared_kv_binding(
        &mut self,
        binding: &mut SharedKvBinding,
        host_inputs: Vec<OrtTensorInput>,
    ) -> Result<OrtTensorOutput> {
        if !Arc::ptr_eq(&binding.session, &self.real.session.inner()) {
            return Err(InfraError::Backend(
                "shared KV binding was used with a different session".to_string(),
            ));
        }
        // The cache inputs are bound already: check only that these exist.
        for input in &host_inputs {
            if !self
                .metadata()
                .inputs
                .iter()
                .any(|meta| meta.name == input.name)
            {
                return Err(InfraError::Backend(format!(
                    "shared KV run input `{}` is not a graph input",
                    input.name
                )));
            }
        }
        for input in host_inputs {
            let name = input.name.clone();
            let value = owned_tensor(input)?;
            // Re-binding a name replaces the previous value; the cache views
            // bound at creation stay in place.
            binding
                .binding
                .bind_input(name, &value)
                .map_err(map_ort_err)?;
        }
        // Logits are host outputs whose shape may follow the input length
        // (graphs without a pruned LM head); rebind so ORT allocates anew.
        binding
            .binding
            .bind_output_to_device(&binding.logits_output, &host_output_memory()?)
            .map_err(map_ort_err)?;
        let outputs = self
            .real
            .session
            .run_binding(&binding.binding)
            .map_err(map_ort_err)?;
        let mut logits = None;
        for (name, value) in outputs {
            if name == binding.logits_output {
                logits = Some(extract_host_logits(name, value)?);
            }
        }
        logits.ok_or_else(|| {
            InfraError::Backend(format!(
                "shared KV run returned no `{}` output",
                binding.logits_output
            ))
        })
    }
}

fn host_output_memory() -> Result<MemoryInfo> {
    MemoryInfo::new(
        AllocationDevice::CPU,
        0,
        AllocatorType::Device,
        MemoryType::CPUOutput,
    )
    .map_err(map_ort_err)
}

fn bind_layer<T>(
    binding: &mut IoBinding,
    allocator: &Allocator,
    memory: &MemoryInfo,
    pair: &SharedKvPair,
    dims: Shape,
) -> Result<()>
where
    T: ort::value::PrimitiveTensorElementType + std::fmt::Debug + 'static,
{
    let mut owner = Tensor::<T>::new(allocator, dims.clone()).map_err(map_ort_err)?;
    let data = owner.data_ptr_mut();
    // SAFETY: `data` is the device allocation of `owner`, which is moved into
    // the binding as the present output right below and lives as long as the
    // binding; the view is also held by the binding and released with it.
    let view =
        unsafe { TensorRefMut::<T>::from_raw(memory.clone(), data, dims) }.map_err(map_ort_err)?;
    binding
        .bind_input(pair.past_input.as_str(), &*view)
        .map_err(map_ort_err)?;
    binding
        .bind_output(pair.present_output.as_str(), owner)
        .map_err(map_ort_err)?;
    Ok(())
}

fn shared_kv_element(
    metadata: &SessionMetadata,
    pairs: &[SharedKvPair],
    shape: [usize; 4],
) -> Result<TensorElement> {
    let mut element = None;
    for pair in pairs {
        let input = metadata
            .inputs
            .iter()
            .find(|input| input.name == pair.past_input)
            .ok_or_else(|| {
                InfraError::Backend(format!(
                    "shared KV input `{}` is not a graph input",
                    pair.past_input
                ))
            })?;
        if !metadata
            .outputs
            .iter()
            .any(|output| output.name == pair.present_output)
        {
            return Err(InfraError::Backend(format!(
                "shared KV output `{}` is not a graph output",
                pair.present_output
            )));
        }
        if input.shape.len() != 4 {
            return Err(InfraError::Backend(format!(
                "shared KV input `{}` has rank {}, expected 4",
                input.name,
                input.shape.len()
            )));
        }
        for axis in [1usize, 3] {
            let dim = input.shape[axis];
            if dim > 0 && dim as usize != shape[axis] {
                return Err(InfraError::Backend(format!(
                    "shared KV input `{}` axis {axis} is {dim}, expected {}",
                    input.name, shape[axis]
                )));
            }
        }
        match element {
            None => element = Some(input.element_type),
            Some(existing) if existing != input.element_type => {
                return Err(InfraError::Backend(format!(
                    "shared KV inputs mix {existing:?} and {:?}",
                    input.element_type
                )))
            }
            Some(_) => {}
        }
    }
    element.ok_or_else(|| InfraError::Backend("shared KV binding has no layers".to_string()))
}

fn extract_host_logits(name: &str, value: DynValue) -> Result<OrtTensorOutput> {
    let value = value
        .downcast::<DynTensorValueType>()
        .map_err(map_ort_err)?;
    if !value.memory_info().is_cpu_accessible() {
        return Err(InfraError::Backend(format!(
            "logits output `{name}` is not CPU accessible"
        )));
    }
    if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
        return Ok(OrtTensorOutput {
            name: name.to_string(),
            shape: shape_to_usize(name, shape)?,
            data: OrtTensorData::F32(data.to_vec()),
        });
    }
    if let Ok((shape, data)) = value.try_extract_tensor::<f16>() {
        return Ok(OrtTensorOutput {
            name: name.to_string(),
            shape: shape_to_usize(name, shape)?,
            data: OrtTensorData::F16(data.to_vec()),
        });
    }
    Err(InfraError::Backend(format!(
        "logits output `{name}` is neither FP32 nor FP16"
    )))
}
