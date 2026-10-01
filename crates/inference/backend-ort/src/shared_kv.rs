//! Fixed-capacity KV cache for decoder graphs whose attention reads `past_*`
//! and writes `present_*` in place (ORT `GroupQueryAttention` with a shared
//! past/present buffer, as exported by the onnxruntime-genai model builder).
//!
//! Each layer's cache is one `[batch, kv_heads, capacity, head_size]` tensor
//! allocated once on the session's device. The same memory is bound as the
//! `past` input (a borrowed view) and the `present` output (the owning
//! tensor), so decode steps never copy or reallocate the cache; the valid
//! length is carried by the `attention_mask` the caller binds for each run.
//!
//! The cache can be bound to several sessions of the same model (a prefill
//! and a decode graph): [`OrtSession::share_kv_binding`] binds another session
//! to the same device memory.

use super::*;
use crate::io_binding::owned_tensor;
use crate::shared_initializers::{device_view, element_size};
use half::f16;
use ort::{
    memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType},
    session::{IoBinding, SharedSessionInner},
    value::{DynTensorValueType, DynValue, Shape, Tensor},
};
use std::sync::Arc;

/// One cache layer: the graph input that reads it and the output that
/// writes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedKvPair {
    pub past_input: String,
    pub present_output: String,
}

/// The cache tensors, shared by every binding made over them.
#[derive(Debug)]
struct KvStore {
    /// Owning device tensors, one per pair.
    layers: Vec<(SharedKvPair, DynValue)>,
    shape: [usize; 4],
    element: TensorElement,
    device: (ProviderKind, Option<u32>),
    // Fields drop in declaration order: the tensors before their allocator.
    _allocator: Allocator,
}

// SAFETY: a `KvStore` is never mutated after construction. Bindings only read
// the tensors' device pointers and shapes through `&KvStore` (to create views),
// and the ORT allocator, tensors and memory info are released once, when the
// last `Arc` drops. Sharing it between threads therefore races on nothing.
unsafe impl Sync for KvStore {}
unsafe impl Send for KvStore {}

#[derive(Debug)]
pub struct SharedKvBinding {
    // Holds views into `store`, so it is released first.
    binding: IoBinding,
    session: Arc<SharedSessionInner>,
    logits_output: String,
    store: Arc<KvStore>,
}

impl SharedKvBinding {
    /// `[batch, kv_heads, capacity, head_size]` of every cache tensor.
    pub fn shape(&self) -> [usize; 4] {
        self.store.shape
    }

    /// The maximum number of tokens the cache holds.
    pub fn capacity(&self) -> usize {
        self.store.shape[2]
    }

    pub fn element(&self) -> TensorElement {
        self.store.element
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
        self.shared_kv_binding(pairs, shape, logits_output, false)
    }

    /// [`Self::create_shared_kv_binding`] with the cache zero-filled, for
    /// graphs that read every slot and mask the unused ones with an additive
    /// bias: masked slots must still hold finite values, as `0 * NaN` is NaN.
    pub fn create_zeroed_shared_kv_binding(
        &self,
        pairs: &[SharedKvPair],
        shape: [usize; 4],
        logits_output: &str,
    ) -> Result<SharedKvBinding> {
        self.shared_kv_binding(pairs, shape, logits_output, true)
    }

    fn shared_kv_binding(
        &self,
        pairs: &[SharedKvPair],
        shape: [usize; 4],
        logits_output: &str,
        zeroed: bool,
    ) -> Result<SharedKvBinding> {
        if pairs.is_empty() {
            return Err(InfraError::Backend(
                "shared KV binding needs at least one cache layer".to_string(),
            ));
        }
        if shape.contains(&0) {
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
        // Allocates (and zeroes) the cache on the device.
        let _gate = crate::gpu_shared();
        let allocator = Allocator::new(&self.real.session, device_memory).map_err(map_ort_err)?;
        let dims = Shape::new(shape.iter().map(|dim| *dim as i64));
        let mut layers = Vec::with_capacity(pairs.len());
        for pair in pairs {
            let owner = match element {
                TensorElement::F16 => cache_tensor::<f16>(&allocator, dims.clone(), zeroed)?,
                TensorElement::F32 => cache_tensor::<f32>(&allocator, dims.clone(), zeroed)?,
                other => {
                    return Err(InfraError::Backend(format!(
                        "shared KV cache element type {other:?} is not supported"
                    )))
                }
            };
            layers.push((pair.clone(), owner));
        }
        let store = Arc::new(KvStore {
            layers,
            shape,
            element,
            device: (self.provider, self.device_id),
            _allocator: allocator,
        });
        self.bind_kv_store(store, logits_output)
    }

    /// Binds this session to the cache of `other` (made for another session
    /// of the same model on the same device): runs of either session read and
    /// update the same memory. Each binding must only run on its own session,
    /// and bindings of one cache must never run concurrently (they use
    /// different CUDA streams): keep them behind one owner that runs them in
    /// turn, as an adapter's `&mut self` does.
    pub fn share_kv_binding(
        &self,
        other: &SharedKvBinding,
        logits_output: &str,
    ) -> Result<SharedKvBinding> {
        let store = Arc::clone(&other.store);
        if store.device != (self.provider, self.device_id) || self.whole_session_cpu_fallback_used()
        {
            return Err(InfraError::Backend(format!(
                "shared KV cache lives on {:?}; this session runs on {:?} {:?}",
                store.device, self.provider, self.device_id
            )));
        }
        let pairs = store
            .layers
            .iter()
            .map(|(pair, _)| pair.clone())
            .collect::<Vec<_>>();
        let element = shared_kv_element(self.metadata(), &pairs, store.shape)?;
        if element != store.element {
            return Err(InfraError::Backend(format!(
                "shared KV cache is {:?}; this session expects {element:?}",
                store.element
            )));
        }
        self.bind_kv_store(store, logits_output)
    }

    fn bind_kv_store(&self, store: Arc<KvStore>, logits_output: &str) -> Result<SharedKvBinding> {
        if !self
            .metadata()
            .outputs
            .iter()
            .any(|output| output.name == logits_output)
        {
            return Err(InfraError::Backend(format!(
                "shared KV binding logits output `{logits_output}` is not a graph output"
            )));
        }
        let bytes = store.shape.iter().product::<usize>()
            * element_size(store.element).expect("cache element is F16 or F32");
        let mut binding = self.real.session.create_binding().map_err(map_ort_err)?;
        for (pair, owner) in &store.layers {
            // Views borrow the store's memory; the binding holds them and is
            // dropped before the store.
            let past = device_view(owner, &store.shape, bytes, &pair.past_input)?;
            binding
                .bind_input(pair.past_input.as_str(), &past)
                .map_err(map_ort_err)?;
            let present = device_view(owner, &store.shape, bytes, &pair.present_output)?;
            binding
                .bind_output(pair.present_output.as_str(), present)
                .map_err(map_ort_err)?;
        }
        binding
            .bind_output_to_device(logits_output, &host_output_memory()?)
            .map_err(map_ort_err)?;
        Ok(SharedKvBinding {
            binding,
            session: self.real.session.inner(),
            logits_output: logits_output.to_string(),
            store,
        })
    }

    /// Runs one step: `host_inputs` (token ids, attention mask, ...) are
    /// copied in, the cache is updated in place, and the logits come back.
    pub fn run_shared_kv_binding(
        &mut self,
        binding: &mut SharedKvBinding,
        host_inputs: Vec<OrtTensorInput>,
    ) -> Result<OrtTensorOutput> {
        self.run_kv_binding(binding, host_inputs, None)
    }

    /// [`Self::run_shared_kv_binding`], then shrinks this session's device
    /// arena so the run's activations go back to the driver (effective with
    /// [`CudaMemoryOptions::arena_same_as_requested`]): for a long prefill
    /// whose activations the following decode steps do not need.
    pub fn run_shared_kv_binding_releasing_memory(
        &mut self,
        binding: &mut SharedKvBinding,
        host_inputs: Vec<OrtTensorInput>,
    ) -> Result<OrtTensorOutput> {
        let shrink = match (self.provider, self.device_id) {
            (ProviderKind::Cuda, Some(device)) if !self.whole_session_cpu_fallback_used() => {
                Some(format!("gpu:{device}"))
            }
            _ => None,
        };
        self.run_kv_binding(binding, host_inputs, shrink.as_deref())
    }

    fn run_kv_binding(
        &mut self,
        binding: &mut SharedKvBinding,
        host_inputs: Vec<OrtTensorInput>,
        shrink_arenas: Option<&str>,
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
        let _gate = crate::gpu_shared();
        let outputs = match shrink_arenas {
            None => self
                .real
                .session
                .run_binding(&binding.binding)
                .map_err(map_ort_err)?,
            Some(arenas) => {
                let mut options = ort::session::RunOptions::new().map_err(map_ort_err)?;
                options
                    .add_config_entry("memory.enable_memory_arena_shrinkage", arenas)
                    .map_err(map_ort_err)?;
                self.real
                    .session
                    .run_binding_with_options(&binding.binding, &options)
                    .map_err(map_ort_err)?
            }
        };
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

fn cache_tensor<T>(allocator: &Allocator, dims: Shape, zeroed: bool) -> Result<DynValue>
where
    T: ort::value::PrimitiveTensorElementType + std::fmt::Debug + Default + Clone + 'static,
{
    let mut owner = Tensor::<T>::new(allocator, dims.clone()).map_err(map_ort_err)?;
    if zeroed {
        let len = dims.iter().product::<i64>() as usize;
        let zeros =
            Tensor::<T>::from_array((dims, vec![T::default(); len])).map_err(map_ort_err)?;
        zeros.copy_into(&mut owner).map_err(map_ort_err)?;
    }
    Ok(owner.into_dyn())
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
