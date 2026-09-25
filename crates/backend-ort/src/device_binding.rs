//! Generic device-resident I/O binding.
//!
//! Multi-graph pipelines (IndexTTS-2.5) pass intermediate tensors of any
//! element type and rank between sessions and recur on KV caches. A
//! [`DeviceBinding`] routes each selected output either to the session's
//! execution device (kept as a [`DeviceTensor`] and fed back without a host
//! round trip) or to host memory. On CPU sessions the "device" is host memory,
//! so callers keep one code path for both providers.

use super::*;
use ort::{
    memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
    session::{IoBinding, SharedSessionInner},
    value::{DynTensorValueType, DynValue},
};
use std::sync::Arc;

/// An output kept where the session produced it. Not `Clone`: cloning an ORT
/// value copies device memory, while recurrence only transfers ownership.
#[derive(Debug)]
pub struct DeviceTensor {
    value: DynValue,
    shape: Vec<usize>,
    element: TensorElement,
}

impl DeviceTensor {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn element(&self) -> TensorElement {
        self.element
    }
}

#[derive(Debug)]
pub struct DeviceBinding {
    binding: IoBinding,
    session: Arc<SharedSessionInner>,
    /// CUDA device for device outputs; `None` keeps them in host memory.
    /// `MemoryInfo` is not `Send`, so it is rebuilt per run.
    cuda_device: Option<i32>,
    device_outputs: Vec<String>,
    host_outputs: Vec<String>,
}

#[derive(Debug, Default)]
pub struct DeviceBindingOutputs {
    device: Vec<(String, DeviceTensor)>,
    host: Vec<OrtTensorOutput>,
}

impl DeviceBindingOutputs {
    pub fn take_device(&mut self, name: &str) -> Result<DeviceTensor> {
        let index = self
            .device
            .iter()
            .position(|(candidate, _)| candidate == name)
            .ok_or_else(|| {
                InfraError::Backend(format!("device binding did not return device output `{name}`"))
            })?;
        Ok(self.device.swap_remove(index).1)
    }

    pub fn take_host(&mut self, name: &str) -> Result<OrtTensorOutput> {
        let index = self
            .host
            .iter()
            .position(|output| output.name == name)
            .ok_or_else(|| {
                InfraError::Backend(format!("device binding did not return host output `{name}`"))
            })?;
        Ok(self.host.swap_remove(index))
    }
}

impl OrtSession {
    /// Creates a binding that keeps `device_outputs` on this session's
    /// execution device and copies `host_outputs` to host memory. Every
    /// session output must be listed exactly once.
    pub fn create_device_binding(
        &self,
        device_outputs: &[&str],
        host_outputs: &[&str],
    ) -> Result<DeviceBinding> {
        validate_binding_output_names(
            self.outputs().iter().map(|output| output.name.as_str()),
            device_outputs.iter().copied(),
            host_outputs.iter().copied(),
        )?;
        let cuda_device = match (self.provider, self.device_id) {
            (ProviderKind::Cuda, Some(device_id)) if !self.whole_session_cpu_fallback_used() => {
                Some(device_id as i32)
            }
            _ => None,
        };
        let binding = self.real.session.create_binding().map_err(map_ort_err)?;
        Ok(DeviceBinding {
            binding,
            session: self.real.session.inner(),
            cuda_device,
            device_outputs: device_outputs.iter().map(|name| name.to_string()).collect(),
            host_outputs: host_outputs.iter().map(|name| name.to_string()).collect(),
        })
    }

    /// Runs `binding` with owned host inputs plus borrowed device tensors.
    /// Outputs are rebound on every run because dynamic shapes (KV length,
    /// mel frames) change between calls and ORT would otherwise try to reuse
    /// the previous allocation.
    pub fn run_device_binding(
        &mut self,
        binding: &mut DeviceBinding,
        host_inputs: Vec<OrtTensorInput>,
        device_inputs: &[(&str, &DeviceTensor)],
    ) -> Result<DeviceBindingOutputs> {
        if !Arc::ptr_eq(&binding.session, &self.real.session.inner()) {
            return Err(InfraError::Backend(
                "device binding was used with a different session".to_string(),
            ));
        }
        self.real.validate_input_names(
            host_inputs
                .iter()
                .map(|input| input.name.as_str())
                .chain(device_inputs.iter().map(|(name, _)| *name)),
        )?;
        let host_values = host_inputs
            .into_iter()
            .map(|input| {
                let name = input.name.clone();
                owned_input_tensor(input).map(|value| (name, value))
            })
            .collect::<Result<Vec<_>>>()?;

        let device_memory = match binding.cuda_device {
            Some(device_id) => MemoryInfo::new(
                AllocationDevice::CUDA,
                device_id,
                AllocatorType::Device,
                MemoryType::Default,
            ),
            None => MemoryInfo::new(
                AllocationDevice::CPU,
                0,
                AllocatorType::Device,
                MemoryType::Default,
            ),
        }
        .map_err(map_ort_err)?;
        let host_memory = MemoryInfo::new(
            AllocationDevice::CPU,
            0,
            AllocatorType::Device,
            MemoryType::CPUOutput,
        )
        .map_err(map_ort_err)?;
        binding.binding.clear_outputs();
        for name in &binding.device_outputs {
            binding
                .binding
                .bind_output_to_device(name, &device_memory)
                .map_err(map_ort_err)?;
        }
        for name in &binding.host_outputs {
            binding
                .binding
                .bind_output_to_device(name, &host_memory)
                .map_err(map_ort_err)?;
        }
        binding.binding.clear_inputs();
        let bound = (|| {
            for (name, value) in &host_values {
                binding.binding.bind_input(name, value).map_err(map_ort_err)?;
            }
            for (name, tensor) in device_inputs {
                binding
                    .binding
                    .bind_input(*name, &tensor.value)
                    .map_err(map_ort_err)?;
            }
            Ok(())
        })();
        if let Err(err) = bound {
            binding.binding.clear_inputs();
            return Err(err);
        }

        let result = (|| {
            let outputs = self
                .real
                .session
                .run_binding(&binding.binding)
                .map_err(map_ort_err)?;
            let mut collected = DeviceBindingOutputs::default();
            for (name, value) in outputs {
                if binding.device_outputs.iter().any(|expected| expected == name) {
                    collected.device.push((name.to_string(), device_tensor(name, value)?));
                } else {
                    collected
                        .host
                        .push(host_tensor_output(name, &value, &self.real.metadata.outputs)?);
                }
            }
            Ok(collected)
        })();
        // Returned values own their memory; release the consumer references
        // so producers from a previous run can be dropped by the caller.
        binding.binding.clear_inputs();
        result
    }
}

fn device_tensor(name: &str, value: DynValue) -> Result<DeviceTensor> {
    let tensor = value
        .downcast::<DynTensorValueType>()
        .map_err(map_ort_err)?;
    let element = tensor_element(*tensor.data_type());
    let shape = shape_to_usize(name, tensor.shape())?;
    Ok(DeviceTensor {
        value: tensor.into_dyn(),
        shape,
        element,
    })
}

fn owned_input_tensor(input: OrtTensorInput) -> Result<DynTensor> {
    let expected_len = input.shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim).ok_or_else(|| shape_overflow(&input.name, *dim))
    })?;
    if expected_len != input.data.len() {
        return Err(InfraError::Backend(format!(
            "input `{}` data length {} does not match shape {:?}",
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
    fn boxed<T: ort::value::PrimitiveTensorElementType + Clone + std::fmt::Debug + 'static>(
        shape: Vec<i64>,
        data: Vec<T>,
    ) -> Result<DynTensor> {
        Tensor::from_array((shape, data.into_boxed_slice()))
            .map(|tensor| tensor.upcast())
            .map_err(map_ort_err)
    }
    match input.data {
        OrtTensorData::F32(data) => boxed(shape, data),
        OrtTensorData::F16(data) => boxed(shape, data),
        OrtTensorData::Bool(data) => boxed(shape, data),
        OrtTensorData::I8(data) => boxed(shape, data),
        OrtTensorData::I16(data) => boxed(shape, data),
        OrtTensorData::I32(data) => boxed(shape, data),
        OrtTensorData::I64(data) => boxed(shape, data),
    }
}

fn validate_binding_output_names<'a>(
    available: impl IntoIterator<Item = &'a str>,
    device: impl IntoIterator<Item = &'a str>,
    host: impl IntoIterator<Item = &'a str>,
) -> Result<()> {
    let available = available.into_iter().collect::<HashSet<_>>();
    let requested = device.into_iter().chain(host).collect::<Vec<_>>();
    let mut unique = HashSet::with_capacity(requested.len());
    for name in &requested {
        if !available.contains(name) {
            return Err(InfraError::Backend(format!(
                "device binding output `{name}` is absent; available outputs: {}",
                format_names(available.iter().copied())
            )));
        }
        if !unique.insert(*name) {
            return Err(InfraError::Backend(format!(
                "device binding output `{name}` was requested more than once"
            )));
        }
    }
    if unique.len() != available.len() {
        return Err(InfraError::Backend(format!(
            "device binding must route every session output exactly once; requested: {}, available: {}",
            format_names(requested),
            format_names(available.iter().copied())
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_routing_requires_each_output_exactly_once() {
        let available = ["a", "b", "c"];
        assert!(validate_binding_output_names(available, ["a"], ["b", "c"]).is_ok());
        assert!(validate_binding_output_names(available, ["a"], ["b"]).is_err());
        assert!(validate_binding_output_names(available, ["a", "a"], ["b", "c"]).is_err());
        assert!(validate_binding_output_names(available, ["a"], ["b", "d"]).is_err());
    }
}
