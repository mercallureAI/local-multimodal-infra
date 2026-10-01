//! I/O binding for decode loops replayed as CUDA graphs.
//!
//! A [`StaticIoBinding`] owns a past/present-shared KV cache (as
//! [`crate::SharedKvBinding`] does) plus *fixed* tensors: device buffers
//! allocated once whose addresses never change. Runs come in two kinds:
//!
//! * host runs ([`OrtSession::run_static_host`]) bind owned host inputs of any
//!   shape (a prompt prefill) and return the requested outputs to the host;
//! * fixed runs ([`OrtSession::run_static_fixed`]) bind only fixed tensors, so
//!   the shapes and addresses a CUDA graph captured stay valid; the caller
//!   updates fixed inputs with [`StaticIoBinding::write`] and reads fixed
//!   outputs with [`StaticIoBinding::read`].
//!
//! Fixed outputs stay bound in both kinds, so a prefill can write the first
//! decode step's inputs.

use super::*;
use crate::io_binding::owned_tensor;
use half::f16;
use ort::{
    memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType},
    session::{IoBinding, RunOptions, SharedSessionInner},
    value::{DynTensor, DynTensorRefMut, DynTensorValueType, Shape, Tensor, TensorRefMut},
};
use std::sync::Arc;

/// A device buffer bound at a fixed address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedTensorSpec {
    pub name: String,
    pub element: TensorElement,
    pub shape: Vec<usize>,
}

impl FixedTensorSpec {
    pub fn new(name: impl Into<String>, element: TensorElement, shape: &[usize]) -> Self {
        Self {
            name: name.into(),
            element,
            shape: shape.to_vec(),
        }
    }
}

#[derive(Debug)]
enum DeviceTensor {
    /// An input buffer, owned here and bound by reference.
    Input(DynTensor),
    /// A view of an output buffer the binding owns (`bind_output` takes
    /// ownership); used to copy the result out.
    Output(DynTensorRefMut<'static>),
}

#[derive(Debug)]
struct FixedSlot {
    spec: FixedTensorSpec,
    device: DeviceTensor,
    /// Host copy of the same shape, reused for every transfer.
    staging: DynTensor,
}

impl FixedSlot {
    fn is_input(&self) -> bool {
        matches!(self.device, DeviceTensor::Input(_))
    }

    fn device(&self) -> &DynTensor {
        match &self.device {
            DeviceTensor::Input(tensor) => tensor,
            DeviceTensor::Output(view) => view,
        }
    }
}

/// Run options selecting one CUDA graph (`gpu_graph_id`).
struct GraphRunOptions {
    id: i64,
    options: Arc<RunOptions>,
}

impl std::fmt::Debug for GraphRunOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphRunOptions")
            .field("id", &self.id)
            .finish()
    }
}

#[derive(Debug)]
pub struct StaticIoBinding {
    binding: IoBinding,
    session: Arc<SharedSessionInner>,
    kv_shape: [usize; 4],
    fixed: Vec<FixedSlot>,
    /// Views of the KV cache layers (the binding owns the memory).
    kv_views: Vec<DynTensorRefMut<'static>>,
    /// Saved copies of the whole KV cache, by key.
    kv_snapshots: Vec<(String, Vec<DynTensor>)>,
    /// Host inputs bound once for every run (e.g. a CPU-side scalar).
    constants: Vec<DynTensor>,
    /// Whether the fixed inputs are the ones currently bound.
    fixed_bound: bool,
    graph_options: Vec<GraphRunOptions>,
    // Bound values reference this allocator's memory: dropped last.
    _allocator: Allocator,
}

impl StaticIoBinding {
    /// The KV cache capacity in positions.
    pub fn kv_capacity(&self) -> usize {
        self.kv_shape[2]
    }

    fn slot(&self, name: &str) -> Result<usize> {
        self.fixed
            .iter()
            .position(|slot| slot.spec.name == name)
            .ok_or_else(|| {
                InfraError::Backend(format!("`{name}` is not a fixed tensor of this binding"))
            })
    }

    /// Copies `data` into the fixed input `name` (its full shape).
    pub fn write(&mut self, name: &str, data: &OrtTensorData) -> Result<()> {
        let index = self.slot(name)?;
        let slot = &mut self.fixed[index];
        let DeviceTensor::Input(device) = &mut slot.device else {
            return Err(InfraError::Backend(format!(
                "fixed tensor `{name}` is an output"
            )));
        };
        fill_staging(&mut slot.staging, &slot.spec, data)?;
        slot.staging.copy_into(device).map_err(map_ort_err)
    }

    /// Copies the fixed tensor `name` (input or output) to the host.
    pub fn read(&mut self, name: &str) -> Result<OrtTensorOutput> {
        let index = self.slot(name)?;
        let slot = &mut self.fixed[index];
        let FixedSlot {
            device, staging, ..
        } = slot;
        let source: &DynTensor = match device {
            DeviceTensor::Input(tensor) => tensor,
            DeviceTensor::Output(view) => view,
        };
        source.copy_into(staging).map_err(map_ort_err)?;
        read_staging(&slot.staging, &slot.spec)
    }

    /// Copies fixed tensor `from` into fixed input `to` on the device (same
    /// element type and shape), e.g. a state output into its input.
    pub fn copy_fixed(&mut self, from: &str, to: &str) -> Result<()> {
        let (source, target) = (self.slot(from)?, self.slot(to)?);
        if source == target || self.fixed[source].spec.shape != self.fixed[target].spec.shape {
            return Err(InfraError::Backend(format!(
                "cannot copy fixed `{from}` into `{to}`"
            )));
        }
        // Split borrows: the two slots are distinct.
        let (a, b) = if source < target {
            let (left, right) = self.fixed.split_at_mut(target);
            (&left[source], &mut right[0])
        } else {
            let (left, right) = self.fixed.split_at_mut(source);
            (&right[0], &mut left[target])
        };
        let DeviceTensor::Input(device) = &mut b.device else {
            return Err(InfraError::Backend(format!(
                "fixed tensor `{to}` is an output"
            )));
        };
        a.device().copy_into(device).map_err(map_ort_err)
    }

    /// Binds a host input once, for every later run (host or fixed).
    pub fn bind_constant(&mut self, input: OrtTensorInput) -> Result<()> {
        let name = input.name.clone();
        let value = owned_tensor(input)?;
        self.binding.bind_input(name, &value).map_err(map_ort_err)?;
        self.constants.push(value);
        Ok(())
    }

    /// Saves a device copy of the whole KV cache under `key`.
    pub fn save_kv(&mut self, key: &str) -> Result<()> {
        let mut copies = Vec::with_capacity(self.kv_views.len());
        for view in &self.kv_views {
            let source: &DynTensor = view;
            let mut copy = DynTensor::new(
                &self._allocator,
                *source.data_type(),
                source.shape().clone(),
            )
            .map_err(map_ort_err)?;
            source.copy_into(&mut copy).map_err(map_ort_err)?;
            copies.push(copy);
        }
        self.kv_snapshots.retain(|(existing, _)| existing != key);
        self.kv_snapshots.push((key.to_string(), copies));
        Ok(())
    }

    /// Restores the KV cache saved under `key`; false when there is none.
    pub fn restore_kv(&mut self, key: &str) -> Result<bool> {
        let Some((_, copies)) = self
            .kv_snapshots
            .iter()
            .find(|(existing, _)| existing == key)
        else {
            return Ok(false);
        };
        for (copy, view) in copies.iter().zip(self.kv_views.iter_mut()) {
            let target: &mut DynTensor = view;
            copy.copy_into(target).map_err(map_ort_err)?;
        }
        Ok(true)
    }

    /// Drops every saved KV cache.
    pub fn clear_kv_snapshots(&mut self) {
        self.kv_snapshots.clear();
    }

    fn options(&mut self, graph_id: Option<i64>) -> Result<Option<Arc<RunOptions>>> {
        let Some(id) = graph_id else {
            return Ok(None);
        };
        if let Some(existing) = self.graph_options.iter().find(|options| options.id == id) {
            return Ok(Some(existing.options.clone()));
        }
        let mut options = RunOptions::new().map_err(map_ort_err)?;
        options
            .add_config_entry("gpu_graph_id", id.to_string())
            .map_err(map_ort_err)?;
        let options = Arc::new(options);
        self.graph_options.push(GraphRunOptions {
            id,
            options: options.clone(),
        });
        Ok(Some(options))
    }
}

impl OrtSession {
    /// Allocates the KV cache (`kv` pairs of `kv_shape`, bound in place) and
    /// the fixed tensors on this session's device.
    pub fn create_static_binding(
        &self,
        kv: &[SharedKvPair],
        kv_shape: [usize; 4],
        fixed_inputs: &[FixedTensorSpec],
        fixed_outputs: &[FixedTensorSpec],
    ) -> Result<StaticIoBinding> {
        let memory = match (self.provider, self.device_id) {
            (ProviderKind::Cuda, Some(device_id)) if !self.whole_session_cpu_fallback_used() => MemoryInfo::new(
                AllocationDevice::CUDA,
                device_id as i32,
                AllocatorType::Device,
                MemoryType::Default,
            )
            .map_err(map_ort_err)?,
            (ProviderKind::Cpu, _) => {
                MemoryInfo::new(AllocationDevice::CPU, 0, AllocatorType::Device, MemoryType::Default)
                    .map_err(map_ort_err)?
            }
            (provider, device_id) => {
                return Err(InfraError::Unsupported(format!(
                    "static I/O binding supports CPU and CUDA sessions; got {provider:?} on device {device_id:?}"
                )))
            }
        };
        let metadata = self.metadata();
        for spec in fixed_inputs {
            check_fixed(&metadata.inputs, spec, "input")?;
        }
        for spec in fixed_outputs {
            check_fixed(&metadata.outputs, spec, "output")?;
        }
        let allocator = Allocator::new(&self.real.session, memory.clone()).map_err(map_ort_err)?;
        let mut binding = self.real.session.create_binding().map_err(map_ort_err)?;
        let mut kv_views = Vec::with_capacity(kv.len());
        if !kv.is_empty() {
            let element = kv_element(metadata, kv)?;
            let dims = Shape::new(kv_shape.iter().map(|dim| *dim as i64));
            for pair in kv {
                match element {
                    TensorElement::F16 => kv_views.push(bind_kv::<f16>(
                        &mut binding,
                        &allocator,
                        &memory,
                        pair,
                        dims.clone(),
                    )?),
                    TensorElement::F32 => kv_views.push(bind_kv::<f32>(
                        &mut binding,
                        &allocator,
                        &memory,
                        pair,
                        dims.clone(),
                    )?),
                    other => {
                        return Err(InfraError::Backend(format!(
                            "static binding KV element type {other:?} is not supported"
                        )))
                    }
                }
            }
        }
        let mut fixed = Vec::with_capacity(fixed_inputs.len() + fixed_outputs.len());
        for spec in fixed_inputs {
            fixed.push(FixedSlot {
                spec: spec.clone(),
                device: DeviceTensor::Input(new_tensor(&allocator, spec)?),
                staging: new_tensor(&Allocator::default(), spec)?,
            });
        }
        for spec in fixed_outputs {
            let view = bind_fixed_output(&mut binding, &allocator, &memory, spec)?;
            fixed.push(FixedSlot {
                spec: spec.clone(),
                device: DeviceTensor::Output(view),
                staging: new_tensor(&Allocator::default(), spec)?,
            });
        }
        Ok(StaticIoBinding {
            binding,
            session: self.real.session.inner(),
            kv_shape,
            fixed,
            kv_views,
            kv_snapshots: Vec::new(),
            constants: Vec::new(),
            fixed_bound: false,
            graph_options: Vec::new(),
            _allocator: allocator,
        })
    }

    /// Runs with `host_inputs` bound (any shapes) and the fixed inputs not
    /// named there; returns `host_outputs` in host memory. Use
    /// `graph_id = Some(-1)` on a session with CUDA graphs enabled so this run
    /// is neither captured nor replayed.
    pub fn run_static_host(
        &mut self,
        binding: &mut StaticIoBinding,
        host_inputs: Vec<OrtTensorInput>,
        host_outputs: &[&str],
        graph_id: Option<i64>,
    ) -> Result<Vec<OrtTensorOutput>> {
        self.check_static(binding)?;
        // KV and fixed inputs are bound already: check only that these exist.
        for input in &host_inputs {
            if !self.inputs().iter().any(|meta| meta.name == input.name) {
                return Err(InfraError::Backend(format!(
                    "static binding input `{}` is not a graph input",
                    input.name
                )));
            }
        }
        let host_memory = MemoryInfo::new(
            AllocationDevice::CPU,
            0,
            AllocatorType::Device,
            MemoryType::CPUOutput,
        )
        .map_err(map_ort_err)?;
        let named = host_inputs
            .iter()
            .map(|input| input.name.clone())
            .collect::<Vec<_>>();
        for input in host_inputs {
            let name = input.name.clone();
            let value = owned_tensor(input)?;
            binding
                .binding
                .bind_input(name, &value)
                .map_err(map_ort_err)?;
        }
        for slot in binding.fixed.iter().filter(|slot| slot.is_input()) {
            if !named.contains(&slot.spec.name) {
                binding
                    .binding
                    .bind_input(slot.spec.name.as_str(), slot.device())
                    .map_err(map_ort_err)?;
            }
        }
        binding.fixed_bound = false;
        for name in host_outputs {
            binding
                .binding
                .bind_output_to_device(*name, &host_memory)
                .map_err(map_ort_err)?;
        }
        let options = binding.options(graph_id)?;
        let outputs = match &options {
            Some(options) => self
                .real
                .session
                .run_binding_with_options(&binding.binding, options.as_ref()),
            None => self.real.session.run_binding(&binding.binding),
        }
        .map_err(map_ort_err)?;
        let mut result = Vec::with_capacity(host_outputs.len());
        for name in host_outputs {
            let value = outputs.get(*name).ok_or_else(|| {
                InfraError::Backend(format!("static binding run returned no `{name}`"))
            })?;
            result.push(host_output(name, value)?);
        }
        Ok(result)
    }

    /// Runs with only the fixed inputs bound: the shapes and addresses a CUDA
    /// graph (`graph_id`) captures on its first runs and then replays.
    pub fn run_static_fixed(
        &mut self,
        binding: &mut StaticIoBinding,
        graph_id: Option<i64>,
    ) -> Result<()> {
        self.check_static(binding)?;
        if !binding.fixed_bound {
            for slot in binding.fixed.iter().filter(|slot| slot.is_input()) {
                binding
                    .binding
                    .bind_input(slot.spec.name.as_str(), slot.device())
                    .map_err(map_ort_err)?;
            }
            binding.fixed_bound = true;
        }
        let options = binding.options(graph_id)?;
        match &options {
            Some(options) => self
                .real
                .session
                .run_binding_with_options(&binding.binding, options.as_ref()),
            None => self.real.session.run_binding(&binding.binding),
        }
        .map_err(map_ort_err)?;
        Ok(())
    }

    fn check_static(&self, binding: &StaticIoBinding) -> Result<()> {
        if !Arc::ptr_eq(&binding.session, &self.real.session.inner()) {
            return Err(InfraError::Backend(
                "static I/O binding was used with a different session".to_string(),
            ));
        }
        Ok(())
    }
}

fn check_fixed(declared: &[TensorMetadata], spec: &FixedTensorSpec, kind: &str) -> Result<()> {
    let meta = declared
        .iter()
        .find(|meta| meta.name == spec.name)
        .ok_or_else(|| {
            InfraError::Backend(format!(
                "fixed {kind} `{}` is not a graph {kind}",
                spec.name
            ))
        })?;
    if meta.element_type != spec.element {
        return Err(InfraError::Backend(format!(
            "fixed {kind} `{}` is {:?} in the graph, not {:?}",
            spec.name, meta.element_type, spec.element
        )));
    }
    let rank_ok = meta.shape.len() == spec.shape.len()
        && meta
            .shape
            .iter()
            .zip(&spec.shape)
            .all(|(declared, wanted)| *declared < 0 || *declared as usize == *wanted);
    if !rank_ok || spec.shape.contains(&0) {
        return Err(InfraError::Backend(format!(
            "fixed {kind} `{}` shape {:?} does not fit the graph's {:?}",
            spec.name, spec.shape, meta.shape
        )));
    }
    Ok(())
}

fn new_tensor(allocator: &Allocator, spec: &FixedTensorSpec) -> Result<DynTensor> {
    let shape = spec.shape.clone();
    let tensor = match spec.element {
        TensorElement::F32 => Tensor::<f32>::new(allocator, shape).map(|t| t.upcast()),
        TensorElement::F16 => Tensor::<f16>::new(allocator, shape).map(|t| t.upcast()),
        TensorElement::I64 => Tensor::<i64>::new(allocator, shape).map(|t| t.upcast()),
        TensorElement::I32 => Tensor::<i32>::new(allocator, shape).map(|t| t.upcast()),
        other => {
            return Err(InfraError::Backend(format!(
                "fixed tensor `{}` element {other:?} is not supported",
                spec.name
            )))
        }
    };
    tensor.map_err(map_ort_err)
}

fn bind_fixed_output(
    binding: &mut IoBinding,
    allocator: &Allocator,
    memory: &MemoryInfo,
    spec: &FixedTensorSpec,
) -> Result<DynTensorRefMut<'static>> {
    fn bind<T>(
        binding: &mut IoBinding,
        allocator: &Allocator,
        memory: &MemoryInfo,
        spec: &FixedTensorSpec,
    ) -> Result<DynTensorRefMut<'static>>
    where
        T: ort::value::PrimitiveTensorElementType + std::fmt::Debug + 'static,
    {
        let dims = Shape::new(spec.shape.iter().map(|dim| *dim as i64));
        let mut owner = Tensor::<T>::new(allocator, dims.clone()).map_err(map_ort_err)?;
        let data = owner.data_ptr_mut();
        // SAFETY: `data` is `owner`'s allocation; `owner` moves into the
        // binding, which outlives the view (both live in the same
        // `StaticIoBinding`, whose allocator is dropped last). The view never
        // frees the memory.
        let view = unsafe { TensorRefMut::<T>::from_raw(memory.clone(), data, dims) }
            .map_err(map_ort_err)?;
        binding
            .bind_output(spec.name.as_str(), owner)
            .map_err(map_ort_err)?;
        view.into_dyn()
            .downcast::<DynTensorValueType>()
            .map_err(map_ort_err)
    }
    match spec.element {
        TensorElement::F32 => bind::<f32>(binding, allocator, memory, spec),
        TensorElement::F16 => bind::<f16>(binding, allocator, memory, spec),
        TensorElement::I64 => bind::<i64>(binding, allocator, memory, spec),
        TensorElement::I32 => bind::<i32>(binding, allocator, memory, spec),
        other => Err(InfraError::Backend(format!(
            "fixed output `{}` element {other:?} is not supported",
            spec.name
        ))),
    }
}

fn bind_kv<T>(
    binding: &mut IoBinding,
    allocator: &Allocator,
    memory: &MemoryInfo,
    pair: &SharedKvPair,
    dims: Shape,
) -> Result<DynTensorRefMut<'static>>
where
    T: ort::value::PrimitiveTensorElementType + std::fmt::Debug + 'static,
{
    let mut owner = Tensor::<T>::new(allocator, dims.clone()).map_err(map_ort_err)?;
    let data = owner.data_ptr_mut();
    // SAFETY: as in `shared_kv`: the owner is moved into the binding below,
    // which lives as long as the view (both in one `StaticIoBinding`).
    let view =
        unsafe { TensorRefMut::<T>::from_raw(memory.clone(), data, dims) }.map_err(map_ort_err)?;
    binding
        .bind_input(pair.past_input.as_str(), &*view)
        .map_err(map_ort_err)?;
    binding
        .bind_output(pair.present_output.as_str(), owner)
        .map_err(map_ort_err)?;
    view.into_dyn()
        .downcast::<DynTensorValueType>()
        .map_err(map_ort_err)
}

fn kv_element(metadata: &SessionMetadata, pairs: &[SharedKvPair]) -> Result<TensorElement> {
    let mut element = None;
    for pair in pairs {
        let input = metadata
            .inputs
            .iter()
            .find(|input| input.name == pair.past_input)
            .ok_or_else(|| {
                InfraError::Backend(format!(
                    "KV input `{}` is not a graph input",
                    pair.past_input
                ))
            })?;
        if !metadata
            .outputs
            .iter()
            .any(|output| output.name == pair.present_output)
        {
            return Err(InfraError::Backend(format!(
                "KV output `{}` is not a graph output",
                pair.present_output
            )));
        }
        match element {
            None => element = Some(input.element_type),
            Some(existing) if existing != input.element_type => {
                return Err(InfraError::Backend(format!(
                    "KV inputs mix {existing:?} and {:?}",
                    input.element_type
                )))
            }
            Some(_) => {}
        }
    }
    element.ok_or_else(|| InfraError::Backend("no KV layers".to_string()))
}

fn fill_staging(
    staging: &mut DynTensor,
    spec: &FixedTensorSpec,
    data: &OrtTensorData,
) -> Result<()> {
    let expected: usize = spec.shape.iter().product();
    if data.len() != expected {
        return Err(InfraError::Backend(format!(
            "fixed input `{}` takes {expected} values, got {}",
            spec.name,
            data.len()
        )));
    }
    macro_rules! fill {
        ($ty:ty, $source:expr) => {{
            let (_, target) = staging
                .try_extract_tensor_mut::<$ty>()
                .map_err(map_ort_err)?;
            target.copy_from_slice($source);
            Ok(())
        }};
    }
    match (spec.element, data) {
        (TensorElement::F32, OrtTensorData::F32(source)) => fill!(f32, source),
        (TensorElement::F16, OrtTensorData::F16(source)) => fill!(f16, source),
        (TensorElement::I64, OrtTensorData::I64(source)) => fill!(i64, source),
        (TensorElement::I32, OrtTensorData::I32(source)) => fill!(i32, source),
        (element, data) => Err(InfraError::Backend(format!(
            "fixed input `{}` is {element:?}, got {:?}",
            spec.name,
            data.element_type()
        ))),
    }
}

fn read_staging(staging: &DynTensor, spec: &FixedTensorSpec) -> Result<OrtTensorOutput> {
    let data = match spec.element {
        TensorElement::F32 => OrtTensorData::F32(
            staging
                .try_extract_tensor::<f32>()
                .map_err(map_ort_err)?
                .1
                .to_vec(),
        ),
        TensorElement::F16 => OrtTensorData::F16(
            staging
                .try_extract_tensor::<f16>()
                .map_err(map_ort_err)?
                .1
                .to_vec(),
        ),
        TensorElement::I64 => OrtTensorData::I64(
            staging
                .try_extract_tensor::<i64>()
                .map_err(map_ort_err)?
                .1
                .to_vec(),
        ),
        TensorElement::I32 => OrtTensorData::I32(
            staging
                .try_extract_tensor::<i32>()
                .map_err(map_ort_err)?
                .1
                .to_vec(),
        ),
        other => {
            return Err(InfraError::Backend(format!(
                "fixed element {other:?} is not supported"
            )))
        }
    };
    Ok(OrtTensorOutput {
        name: spec.name.clone(),
        shape: spec.shape.clone(),
        data,
    })
}

fn host_output(name: &str, value: &ort::value::DynValue) -> Result<OrtTensorOutput> {
    let shape = |s: &Shape| shape_to_usize(name, s);
    if let Ok((s, data)) = value.try_extract_tensor::<f32>() {
        return Ok(OrtTensorOutput {
            name: name.to_string(),
            shape: shape(s)?,
            data: OrtTensorData::F32(data.to_vec()),
        });
    }
    if let Ok((s, data)) = value.try_extract_tensor::<f16>() {
        return Ok(OrtTensorOutput {
            name: name.to_string(),
            shape: shape(s)?,
            data: OrtTensorData::F16(data.to_vec()),
        });
    }
    if let Ok((s, data)) = value.try_extract_tensor::<i64>() {
        return Ok(OrtTensorOutput {
            name: name.to_string(),
            shape: shape(s)?,
            data: OrtTensorData::I64(data.to_vec()),
        });
    }
    if let Ok((s, data)) = value.try_extract_tensor::<i32>() {
        return Ok(OrtTensorOutput {
            name: name.to_string(),
            shape: shape(s)?,
            data: OrtTensorData::I32(data.to_vec()),
        });
    }
    Err(InfraError::Backend(format!(
        "host output `{name}` has an unsupported element type"
    )))
}
