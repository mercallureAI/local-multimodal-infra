use super::*;
use ort::{
    memory::{AllocationDevice, Allocator, AllocatorType, DeviceType, MemoryInfo, MemoryType},
    session::{IoBinding, SharedSessionInner},
    value::{DynTensor, DynTensorValueType, DynValue, Tensor, TensorElementType},
};
use std::sync::Arc;

/// Reusable pinned-host I/O buffers for a CUDA session with I64 inputs and an
/// FP32 output. The fixed-shape binding removes per-run tensor allocations and
/// makes the host/device transfer boundary explicit. It is intended for
/// encoder-style CPU -> CUDA -> CPU inference where every request changes.
#[derive(Debug)]
pub struct PinnedCudaIoBinding {
    binding: IoBinding,
    session: Arc<SharedSessionInner>,
    device_id: u32,
    input_shape: Vec<usize>,
    inputs: Vec<(String, Tensor<i64>)>,
    output_name: String,
    output_shape: Vec<usize>,
    // ORT tensors allocated by a session allocator must be released before
    // that allocator. These fields are intentionally declared last because
    // Rust drops struct fields in declaration order.
    _input_allocator: Allocator,
    _output_allocator: Allocator,
}

impl PinnedCudaIoBinding {
    pub fn matches_shapes(&self, input_shape: &[usize], output_shape: &[usize]) -> bool {
        self.input_shape == input_shape && self.output_shape == output_shape
    }

    pub fn device_id(&self) -> u32 {
        self.device_id
    }
}

/// A CUDA-resident FP32 IndexTTS KV tensor. This deliberately is not `Clone`:
/// cloning an ORT tensor performs a device copy, while recurrence only needs to
/// transfer ownership of the current generation.
#[derive(Debug)]
pub struct ResidentCudaTensor {
    value: DynValue,
    shape: [usize; 4],
    device_id: u32,
}

impl ResidentCudaTensor {
    pub fn shape(&self) -> [usize; 4] {
        self.shape
    }

    pub fn sequence_len(&self) -> usize {
        self.shape[2]
    }

    pub fn device_id(&self) -> u32 {
        self.device_id
    }
}

#[derive(Debug)]
pub struct ResidentTensorInput<'a> {
    name: &'a str,
    tensor: &'a ResidentCudaTensor,
}

impl<'a> ResidentTensorInput<'a> {
    pub fn new(name: &'a str, tensor: &'a ResidentCudaTensor) -> Self {
        Self { name, tensor }
    }
}

#[derive(Debug)]
pub struct ResidentIoBinding {
    binding: IoBinding,
    session: Arc<SharedSessionInner>,
    device_id: u32,
    cuda_memory: MemoryInfo,
    cpu_memory: MemoryInfo,
    cuda_outputs: Vec<String>,
    cpu_outputs: Vec<String>,
}

#[derive(Debug)]
pub struct ResidentBindingOutputs {
    cuda: Vec<(String, ResidentCudaTensor)>,
    cpu: Vec<OrtTensorOutput>,
}

impl ResidentBindingOutputs {
    pub fn take_cuda(&mut self, name: &str) -> Result<ResidentCudaTensor> {
        let index = self
            .cuda
            .iter()
            .position(|(candidate, _)| candidate == name)
            .ok_or_else(|| {
                InfraError::Backend(format!(
                    "resident binding did not return CUDA output `{name}`"
                ))
            })?;
        Ok(self.cuda.swap_remove(index).1)
    }

    pub fn take_cpu(&mut self, name: &str) -> Result<OrtTensorOutput> {
        let index = self
            .cpu
            .iter()
            .position(|output| output.name == name)
            .ok_or_else(|| {
                InfraError::Backend(format!(
                    "resident binding did not return CPU output `{name}`"
                ))
            })?;
        Ok(self.cpu.swap_remove(index))
    }
}

impl OrtSession {
    /// Creates reusable CUDA-pinned host buffers for all-I64 model inputs and a
    /// single FP32 output. Inputs and output must have fixed, non-zero runtime
    /// shapes for this binding instance; callers can cache one per shape.
    pub fn create_pinned_cuda_binding(
        &self,
        device_id: u32,
        input_shape: &[usize],
        output_name: &str,
        output_shape: &[usize],
    ) -> Result<PinnedCudaIoBinding> {
        let selected_device = self.device_id.unwrap_or(0);
        if self.provider != ProviderKind::Cuda
            || self.whole_session_cpu_fallback_used()
            || selected_device != device_id
        {
            return Err(InfraError::Backend(format!(
                "pinned CUDA I/O binding requires a CUDA-selected session with no whole-session CPU retry on device {device_id}; got {:?}, whole_session_cpu_retry={}, device={selected_device}",
                self.provider,
                self.whole_session_cpu_fallback_used()
            )));
        }
        validate_nonzero_shape("pinned CUDA input", input_shape)?;
        validate_nonzero_shape("pinned CUDA output", output_shape)?;
        if self
            .inputs()
            .iter()
            .any(|input| input.element_type != TensorElement::I64)
        {
            return Err(InfraError::Backend(
                "pinned CUDA I/O binding currently requires every session input to be I64"
                    .to_string(),
            ));
        }
        let output = self
            .outputs()
            .iter()
            .find(|candidate| candidate.name == output_name)
            .ok_or_else(|| {
                InfraError::Backend(format!(
                    "pinned CUDA output `{output_name}` is absent; available outputs: {}",
                    format_names(self.outputs().iter().map(|output| output.name.as_str()))
                ))
            })?;
        if self.outputs().len() != 1 || output.element_type != TensorElement::F32 {
            return Err(InfraError::Backend(format!(
                "pinned CUDA I/O binding requires exactly one FP32 output; available outputs: {}",
                format_names(self.outputs().iter().map(|output| output.name.as_str()))
            )));
        }

        let input_memory = MemoryInfo::new(
            AllocationDevice::CUDA_PINNED,
            device_id as i32,
            AllocatorType::Device,
            MemoryType::CPUInput,
        )
        .map_err(map_ort_err)?;
        let output_memory = MemoryInfo::new(
            AllocationDevice::CUDA_PINNED,
            device_id as i32,
            AllocatorType::Device,
            MemoryType::CPUOutput,
        )
        .map_err(map_ort_err)?;
        let input_allocator =
            Allocator::new(&self.real.session, input_memory).map_err(map_ort_err)?;
        let output_allocator =
            Allocator::new(&self.real.session, output_memory).map_err(map_ort_err)?;
        let mut inputs = Vec::with_capacity(self.inputs().len());
        for input in self.inputs() {
            let tensor =
                Tensor::<i64>::new(&input_allocator, input_shape.to_vec()).map_err(map_ort_err)?;
            inputs.push((input.name.clone(), tensor));
        }

        let mut binding = self.real.session.create_binding().map_err(map_ort_err)?;
        let output_tensor =
            Tensor::<f32>::new(&output_allocator, output_shape.to_vec()).map_err(map_ort_err)?;
        binding
            .bind_output(output_name, output_tensor)
            .map_err(map_ort_err)?;
        Ok(PinnedCudaIoBinding {
            binding,
            session: self.real.session.inner(),
            device_id,
            input_shape: input_shape.to_vec(),
            inputs,
            output_name: output_name.to_string(),
            output_shape: output_shape.to_vec(),
            _input_allocator: input_allocator,
            _output_allocator: output_allocator,
        })
    }

    pub fn run_pinned_cuda_binding(
        &mut self,
        binding: &mut PinnedCudaIoBinding,
        inputs: &[OrtTensorInput],
    ) -> Result<Vec<OrtTensorOutput>> {
        if !Arc::ptr_eq(&binding.session, &self.real.session.inner()) {
            return Err(InfraError::Backend(
                "pinned CUDA I/O binding was used with a different target session".to_string(),
            ));
        }
        self.real.validate_inputs(inputs)?;
        binding.binding.clear_inputs();
        for (name, tensor) in &mut binding.inputs {
            let input = inputs
                .iter()
                .find(|candidate| candidate.name == *name)
                .ok_or_else(|| InfraError::Backend(format!("missing pinned input `{name}`")))?;
            if input.shape != binding.input_shape {
                return Err(InfraError::Backend(format!(
                    "pinned input `{name}` shape {:?} does not match cached shape {:?}",
                    input.shape, binding.input_shape
                )));
            }
            let source = match &input.data {
                OrtTensorData::I64(data) => data,
                other => {
                    return Err(InfraError::Backend(format!(
                        "pinned input `{name}` must be I64, got {:?}",
                        other.element_type()
                    )))
                }
            };
            let (_, target) = tensor
                .try_extract_tensor_mut::<i64>()
                .map_err(map_ort_err)?;
            if source.len() != target.len() {
                return Err(InfraError::Backend(format!(
                    "pinned input `{name}` data length {} does not match buffer length {}",
                    source.len(),
                    target.len()
                )));
            }
            target.copy_from_slice(source);
        }
        for (name, tensor) in &binding.inputs {
            binding
                .binding
                .bind_input(name.clone(), tensor)
                .map_err(map_ort_err)?;
        }

        let outputs = self
            .real
            .session
            .run_binding(&binding.binding)
            .map_err(map_ort_err)?;
        let output = outputs.get(&binding.output_name).ok_or_else(|| {
            InfraError::Backend(format!(
                "pinned binding did not return output `{}`",
                binding.output_name
            ))
        })?;
        let (_, data) = output.try_extract_tensor::<f32>().map_err(map_ort_err)?;
        let result = OrtTensorOutput {
            name: binding.output_name.clone(),
            shape: binding.output_shape.clone(),
            data: OrtTensorData::F32(data.to_vec()),
        };
        drop(outputs);
        binding.binding.clear_inputs();
        Ok(vec![result])
    }

    /// Creates a binding whose dynamic KV outputs stay on CUDA while
    /// explicitly selected control outputs return to host memory.
    pub fn create_resident_cuda_binding(
        &self,
        device_id: u32,
        cuda_outputs: &[String],
        cpu_outputs: &[String],
    ) -> Result<ResidentIoBinding> {
        if self.provider != ProviderKind::Cuda
            || self.whole_session_cpu_fallback_used()
            || self.device_id != Some(device_id)
        {
            return Err(InfraError::Backend(format!(
                "resident CUDA binding requires a CUDA-selected session with no whole-session CPU retry on device {device_id}; got {:?}, whole_session_cpu_retry={}, device={:?}",
                self.provider, self.whole_session_cpu_fallback_used(), self.device_id
            )));
        }
        validate_resident_output_names(
            self.outputs().iter().map(|output| output.name.as_str()),
            cuda_outputs.iter().map(String::as_str),
            cpu_outputs.iter().map(String::as_str),
        )?;

        let cuda = MemoryInfo::new(
            AllocationDevice::CUDA,
            device_id as i32,
            AllocatorType::Device,
            MemoryType::Default,
        )
        .map_err(map_ort_err)?;
        let cpu = MemoryInfo::new(
            AllocationDevice::CPU,
            0,
            AllocatorType::Device,
            MemoryType::CPUOutput,
        )
        .map_err(map_ort_err)?;
        let mut binding = self.real.session.create_binding().map_err(map_ort_err)?;
        // Insertion order is intentional and is the order ORT uses for bound
        // output values. Never derive this from model/native map ordering.
        for name in cuda_outputs {
            binding
                .bind_output_to_device(name, &cuda)
                .map_err(map_ort_err)?;
        }
        for name in cpu_outputs {
            binding
                .bind_output_to_device(name, &cpu)
                .map_err(map_ort_err)?;
        }
        Ok(ResidentIoBinding {
            binding,
            session: self.real.session.inner(),
            device_id,
            cuda_memory: cuda,
            cpu_memory: cpu,
            cuda_outputs: cuda_outputs.to_vec(),
            cpu_outputs: cpu_outputs.to_vec(),
        })
    }

    pub fn run_resident_binding(
        &mut self,
        binding: &mut ResidentIoBinding,
        host_inputs: Vec<OrtTensorInput>,
        resident_inputs: &[ResidentTensorInput<'_>],
    ) -> Result<ResidentBindingOutputs> {
        if !Arc::ptr_eq(&binding.session, &self.real.session.inner()) {
            return Err(InfraError::Backend(
                "resident I/O binding was used with a different target session".to_string(),
            ));
        }
        if self.device_id != Some(binding.device_id) {
            return Err(InfraError::Backend(
                "resident I/O binding CUDA device no longer matches its session".to_string(),
            ));
        }
        self.real.validate_input_names(
            host_inputs
                .iter()
                .map(|input| input.name.as_str())
                .chain(resident_inputs.iter().map(|input| input.name)),
        )?;
        let host_values = host_inputs
            .into_iter()
            .map(|input| {
                let name = input.name.clone();
                owned_tensor(input).map(|value| (name, value))
            })
            .collect::<Result<Vec<_>>>()?;
        for input in resident_inputs {
            if input.tensor.device_id != binding.device_id {
                return Err(InfraError::Backend(format!(
                    "resident input `{}` is on CUDA device {}, expected {}",
                    input.name, input.tensor.device_id, binding.device_id
                )));
            }
        }

        // Dynamic KV sequence shapes change on every decode. ORT otherwise
        // attempts to reuse a previous bound output allocation and rejects the
        // new shape. Ping-pong guarantees this binding's prior outputs are no
        // longer current before it is selected again, so release and recreate
        // only the output bindings (not tensor data) for this generation.
        binding.binding.clear_outputs();
        for name in &binding.cuda_outputs {
            binding
                .binding
                .bind_output_to_device(name, &binding.cuda_memory)
                .map_err(map_ort_err)?;
        }
        for name in &binding.cpu_outputs {
            binding
                .binding
                .bind_output_to_device(name, &binding.cpu_memory)
                .map_err(map_ort_err)?;
        }
        binding.binding.clear_inputs();
        for (name, value) in host_values {
            if let Err(err) = binding.binding.bind_input(name, &value) {
                binding.binding.clear_inputs();
                return Err(map_ort_err(err));
            }
        }
        for input in resident_inputs {
            if let Err(err) = binding.binding.bind_input(input.name, &input.tensor.value) {
                binding.binding.clear_inputs();
                return Err(map_ort_err(err));
            }
        }

        let result = (|| {
            let outputs = self
                .real
                .session
                .run_binding(&binding.binding)
                .map_err(map_ort_err)?;
            let expected_names = binding
                .cuda_outputs
                .iter()
                .chain(&binding.cpu_outputs)
                .map(String::as_str)
                .collect::<Vec<_>>();
            let actual_names = outputs.keys().collect::<Vec<_>>();
            if actual_names != expected_names {
                return Err(InfraError::Backend(format!(
                    "resident binding output order mismatch: expected {expected_names:?}, got {actual_names:?}"
                )));
            }

            let mut cuda = Vec::with_capacity(binding.cuda_outputs.len());
            let mut cpu = Vec::with_capacity(binding.cpu_outputs.len());
            for (name, value) in outputs {
                if binding.cuda_outputs.iter().any(|expected| expected == name) {
                    cuda.push((
                        name.to_string(),
                        validate_resident_cuda_tensor(name, value, binding.device_id)?,
                    ));
                } else {
                    cpu.push(extract_cpu_output(name, value)?);
                }
            }
            Ok(ResidentBindingOutputs { cuda, cpu })
        })();
        // Once returned values have been detached into owned Arc-backed values,
        // release all consumer references. A binding is not run again until its
        // old producer values have also been dropped by the adapter ping-pong.
        binding.binding.clear_inputs();
        result
    }
}

fn validate_nonzero_shape(label: &str, shape: &[usize]) -> Result<()> {
    if shape.is_empty() || shape.contains(&0) {
        return Err(InfraError::Backend(format!(
            "{label} shape must be non-empty and non-zero, got {shape:?}"
        )));
    }
    shape.iter().try_fold(1usize, |size, dim| {
        size.checked_mul(*dim)
            .ok_or_else(|| InfraError::Backend(format!("{label} shape overflows usize: {shape:?}")))
    })?;
    Ok(())
}

fn validate_resident_cuda_tensor(
    name: &str,
    value: DynValue,
    expected_device_id: u32,
) -> Result<ResidentCudaTensor> {
    let value = value
        .downcast::<DynTensorValueType>()
        .map_err(map_ort_err)?;
    let is_f32 = value.data_type() == &TensorElementType::Float32;
    let runtime_shape = shape_to_usize(name, value.shape())?;
    let memory = value.memory_info();
    let shape = validate_resident_tensor_facts(
        name,
        is_f32,
        &runtime_shape,
        memory.allocation_device(),
        memory.device_type(),
        memory.is_cpu_accessible(),
        memory.device_id(),
        expected_device_id,
    )?;
    Ok(ResidentCudaTensor {
        value: value.into_dyn(),
        shape,
        device_id: expected_device_id,
    })
}

#[allow(clippy::too_many_arguments)]
fn validate_resident_tensor_facts(
    name: &str,
    is_f32: bool,
    runtime_shape: &[usize],
    allocation_device: AllocationDevice,
    device_type: DeviceType,
    cpu_accessible: bool,
    device_id: i32,
    expected_device_id: u32,
) -> Result<[usize; 4]> {
    if !is_f32 {
        return Err(InfraError::Backend(format!(
            "resident output `{name}` is not FP32"
        )));
    }
    let shape: [usize; 4] = runtime_shape.try_into().map_err(|_| {
        InfraError::Backend(format!(
            "resident output `{name}` must be rank 4, got {runtime_shape:?}"
        ))
    })?;
    if shape[0] != 1 || shape[1] != 20 || shape[2] == 0 || shape[3] != 64 {
        return Err(InfraError::Backend(format!(
            "resident output `{name}` has invalid KV shape {shape:?}; expected [1, 20, nonzero_seq, 64]"
        )));
    }
    if allocation_device != AllocationDevice::CUDA
        || device_type != DeviceType::GPU
        || cpu_accessible
        || device_id != expected_device_id as i32
    {
        return Err(InfraError::Backend(format!(
            "resident output `{name}` placement mismatch: allocation={}, type={device_type:?}, cpu_accessible={cpu_accessible}, device={device_id}; expected CUDA GPU device {expected_device_id}",
            allocation_device.as_str()
        )));
    }
    Ok(shape)
}

fn validate_resident_output_names<'a>(
    available: impl IntoIterator<Item = &'a str>,
    cuda: impl IntoIterator<Item = &'a str>,
    cpu: impl IntoIterator<Item = &'a str>,
) -> Result<()> {
    let available = available.into_iter().collect::<HashSet<_>>();
    let requested = cuda.into_iter().chain(cpu).collect::<Vec<_>>();
    let mut unique = HashSet::with_capacity(requested.len());
    for name in &requested {
        if !available.contains(name) {
            return Err(InfraError::Backend(format!(
                "resident binding output `{name}` is absent; available outputs: {}",
                format_names(available.iter().copied())
            )));
        }
        if !unique.insert(*name) {
            return Err(InfraError::Backend(format!(
                "resident binding output `{name}` was requested more than once"
            )));
        }
    }
    if unique.len() != available.len() {
        return Err(InfraError::Backend(format!(
            "resident binding must map every session output exactly once; requested: {}, available: {}",
            format_names(requested),
            format_names(available.iter().copied())
        )));
    }
    Ok(())
}

fn owned_tensor(input: OrtTensorInput) -> Result<DynTensor> {
    let expected_len = input.shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim).ok_or_else(|| {
            InfraError::Backend(format!(
                "input `{}` shape {:?} overflows usize",
                input.name, input.shape
            ))
        })
    })?;
    if expected_len != input.data.len() {
        return Err(InfraError::Backend(format!(
            "input `{}` data length {} does not match shape {:?} (expected {expected_len})",
            input.name,
            input.data.len(),
            input.shape
        )));
    }
    let shape = input
        .shape
        .iter()
        .map(|dim| i64::try_from(*dim).map_err(|_| shape_overflow(&input.name, *dim)))
        .collect::<Result<Vec<_>>>()?;
    match input.data {
        OrtTensorData::F32(data) => Tensor::from_array((shape, data.into_boxed_slice()))
            .map(|tensor| tensor.upcast())
            .map_err(map_ort_err),
        OrtTensorData::F16(data) => Tensor::from_array((shape, data.into_boxed_slice()))
            .map(|tensor| tensor.upcast())
            .map_err(map_ort_err),
        OrtTensorData::Bool(data) => Tensor::from_array((shape, data.into_boxed_slice()))
            .map(|tensor| tensor.upcast())
            .map_err(map_ort_err),
        OrtTensorData::I8(data) => Tensor::from_array((shape, data.into_boxed_slice()))
            .map(|tensor| tensor.upcast())
            .map_err(map_ort_err),
        OrtTensorData::I16(data) => Tensor::from_array((shape, data.into_boxed_slice()))
            .map(|tensor| tensor.upcast())
            .map_err(map_ort_err),
        OrtTensorData::I32(data) => Tensor::from_array((shape, data.into_boxed_slice()))
            .map(|tensor| tensor.upcast())
            .map_err(map_ort_err),
        OrtTensorData::I64(data) => Tensor::from_array((shape, data.into_boxed_slice()))
            .map(|tensor| tensor.upcast())
            .map_err(map_ort_err),
    }
}

fn extract_cpu_output(name: &str, value: DynValue) -> Result<OrtTensorOutput> {
    let value = value
        .downcast::<DynTensorValueType>()
        .map_err(map_ort_err)?;
    if !value.memory_info().is_cpu_accessible() {
        return Err(InfraError::Backend(format!(
            "host output `{name}` is not CPU accessible"
        )));
    }
    if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
        return Ok(OrtTensorOutput {
            name: name.to_string(),
            shape: shape_to_usize(name, shape)?,
            data: OrtTensorData::F32(data.to_vec()),
        });
    }
    Err(InfraError::Backend(format!(
        "resident host output `{name}` is not an FP32 tensor"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_output_names_require_exact_unique_partition() {
        assert!(validate_resident_output_names(["kv", "logits"], ["kv"], ["logits"]).is_ok());
        assert!(validate_resident_output_names(["kv", "logits"], ["kv"], ["kv"]).is_err());
        assert!(validate_resident_output_names(["kv", "logits"], ["kv"], []).is_err());
        assert!(validate_resident_output_names(["kv", "logits"], ["missing"], ["logits"]).is_err());
    }

    #[test]
    fn pinned_shapes_must_be_nonzero_and_not_overflow() {
        assert!(validate_nonzero_shape("input", &[8, 512]).is_ok());
        assert!(validate_nonzero_shape("input", &[]).is_err());
        assert!(validate_nonzero_shape("input", &[1, 0]).is_err());
        assert!(validate_nonzero_shape("input", &[usize::MAX, 2]).is_err());
    }

    #[test]
    fn resident_tensor_facts_reject_type_rank_shape_and_placement_mismatches() {
        let valid = || {
            validate_resident_tensor_facts(
                "out_key_0",
                true,
                &[1, 20, 7, 64],
                AllocationDevice::CUDA,
                DeviceType::GPU,
                false,
                0,
                0,
            )
        };
        assert_eq!(valid().unwrap(), [1, 20, 7, 64]);
        assert!(validate_resident_tensor_facts(
            "out_key_0",
            false,
            &[1, 20, 7, 64],
            AllocationDevice::CUDA,
            DeviceType::GPU,
            false,
            0,
            0,
        )
        .is_err());
        for shape in [vec![1, 20, 7], vec![1, 20, 0, 64], vec![1, 19, 7, 64]] {
            assert!(validate_resident_tensor_facts(
                "out_key_0",
                true,
                &shape,
                AllocationDevice::CUDA,
                DeviceType::GPU,
                false,
                0,
                0,
            )
            .is_err());
        }
        assert!(validate_resident_tensor_facts(
            "out_key_0",
            true,
            &[1, 20, 7, 64],
            AllocationDevice::CPU,
            DeviceType::CPU,
            true,
            0,
            0,
        )
        .is_err());
        assert!(validate_resident_tensor_facts(
            "out_key_0",
            true,
            &[1, 20, 7, 64],
            AllocationDevice::CUDA,
            DeviceType::GPU,
            false,
            1,
            0,
        )
        .is_err());
    }
}
