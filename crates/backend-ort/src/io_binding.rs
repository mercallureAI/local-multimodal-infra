use super::*;
use ort::{
    memory::{AllocationDevice, AllocatorType, DeviceType, MemoryInfo, MemoryType},
    session::{IoBinding, SharedSessionInner},
    value::{DynTensor, DynTensorValueType, DynValue, Tensor, TensorElementType},
};
use std::sync::Arc;

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
