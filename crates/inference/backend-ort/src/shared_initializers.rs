//! Initializers shared by several sessions.
//!
//! ORT copies every model initializer into each session. Graph sets that repeat
//! the same weights (IndexTTS-2.5's GPT prefill and decode graphs) instead
//! upload those ranges once and pass the same values to every session via
//! `AddInitializer`, which ORT uses in place.
//!
//! `AddInitializer` only accepts values whose buffer the caller owns, so a
//! device copy made by ORT (`Tensor::to`) is kept as the owner and sessions get
//! a non-owning view of the same device memory.

use super::*;
use ort::{memory::AllocationDevice, value::DynValue, AsPointer};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    sync::Arc,
};

/// One tensor stored as a raw little-endian byte range of an external-data file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InitializerRange {
    pub name: String,
    pub element: TensorElement,
    pub shape: Vec<usize>,
    pub offset: u64,
    pub length: u64,
}

/// Values must outlive every session built from them: keep this struct alive
/// (for example as the last field of the owning adapter) until those sessions
/// are dropped.
#[derive(Debug, Clone, Default)]
pub struct SharedInitializers {
    values: Arc<Vec<(String, Arc<DynValue>)>>,
    /// Device buffers the views in `values` point into.
    _owners: Arc<Vec<DynValue>>,
    cuda_device: Option<u32>,
    bytes: u64,
}

impl SharedInitializers {
    pub(crate) fn values(&self) -> &[(String, Arc<DynValue>)] {
        &self.values
    }

    /// CUDA device holding the values, or `None` for host memory.
    pub fn cuda_device(&self) -> Option<u32> {
        self.cuda_device
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl OrtBackend {
    /// Reads `ranges` from `data_file` and places them on this backend's
    /// preferred CUDA device (host memory when CUDA is not selected). Sessions
    /// that should fall back to CPU must not receive CUDA-resident values.
    pub fn upload_initializers(
        &self,
        data_file: &Path,
        ranges: &[InitializerRange],
    ) -> Result<SharedInitializers> {
        let cuda_device = self.preferred_cuda_device();
        let mut file = File::open(data_file)
            .map_err(|e| InfraError::io(Some(data_file.to_path_buf()), e))?;
        let mut values = Vec::with_capacity(ranges.len());
        let mut owners = Vec::new();
        let mut bytes = 0u64;
        let mut buffer = Vec::new();
        for range in ranges {
            let expected = element_size(range.element)
                .and_then(|size| {
                    range.shape.iter().try_fold(size, |acc, dim| acc.checked_mul(*dim))
                })
                .ok_or_else(|| {
                    InfraError::Backend(format!(
                        "shared initializer `{}` has unsupported element {:?} or shape {:?}",
                        range.name, range.element, range.shape
                    ))
                })?;
            if expected as u64 != range.length {
                return Err(InfraError::Backend(format!(
                    "shared initializer `{}` length {} does not match {:?} {:?} ({expected} bytes)",
                    range.name, range.length, range.element, range.shape
                )));
            }
            buffer.resize(expected, 0);
            file.seek(SeekFrom::Start(range.offset))
                .and_then(|_| file.read_exact(&mut buffer))
                .map_err(|e| InfraError::io(Some(data_file.to_path_buf()), e))?;
            let host = host_value(range, &buffer)?;
            let value = match cuda_device {
                Some(device) => {
                    let owner = to_cuda(host, device as i32, &range.name)?;
                    let view = borrowed_view(&owner, range)?;
                    owners.push(owner);
                    view
                }
                // Host tensors are built over Rust-owned memory already.
                None => host,
            };
            values.push((range.name.clone(), Arc::new(value)));
            bytes += range.length;
        }
        Ok(SharedInitializers {
            values: Arc::new(values),
            _owners: Arc::new(owners),
            cuda_device,
            bytes,
        })
    }
}

fn element_size(element: TensorElement) -> Option<usize> {
    Some(match element {
        TensorElement::F32 | TensorElement::I32 => 4,
        TensorElement::F16 | TensorElement::I16 => 2,
        TensorElement::I64 => 8,
        TensorElement::I8 | TensorElement::Bool => 1,
        TensorElement::Other => return None,
    })
}

fn host_value(range: &InitializerRange, bytes: &[u8]) -> Result<DynValue> {
    fn build<T: ort::value::PrimitiveTensorElementType + Clone + std::fmt::Debug + 'static>(
        shape: &[usize],
        data: Vec<T>,
    ) -> Result<DynValue> {
        let shape = shape.iter().map(|dim| *dim as i64).collect::<Vec<_>>();
        Tensor::from_array((shape, data.into_boxed_slice()))
            .map(|tensor| tensor.into_dyn())
            .map_err(map_ort_err)
    }
    let shape = &range.shape;
    match range.element {
        TensorElement::F32 => build(shape, le_values(bytes, f32::from_le_bytes)),
        TensorElement::F16 => build(
            shape,
            le_values(bytes, |raw: [u8; 2]| half::f16::from_bits(u16::from_le_bytes(raw))),
        ),
        TensorElement::I64 => build(shape, le_values(bytes, i64::from_le_bytes)),
        TensorElement::I32 => build(shape, le_values(bytes, i32::from_le_bytes)),
        TensorElement::I16 => build(shape, le_values(bytes, i16::from_le_bytes)),
        TensorElement::I8 => build(shape, bytes.iter().map(|byte| *byte as i8).collect()),
        TensorElement::Bool => build(shape, bytes.iter().map(|byte| *byte != 0).collect()),
        TensorElement::Other => Err(InfraError::Backend(format!(
            "shared initializer `{}` has an unsupported element type",
            range.name
        ))),
    }
}

fn le_values<const N: usize, T>(bytes: &[u8], convert: impl Fn([u8; N]) -> T) -> Vec<T> {
    bytes
        .chunks_exact(N)
        .map(|chunk| convert(chunk.try_into().expect("exact chunk")))
        .collect()
}

fn to_cuda(host: DynValue, device: i32, name: &str) -> Result<DynValue> {
    let tensor = host
        .downcast::<ort::value::DynTensorValueType>()
        .map_err(map_ort_err)?;
    tensor
        .to(AllocationDevice::CUDA, device)
        .map(|tensor| tensor.into_dyn())
        .map_err(|err| {
            InfraError::Backend(format!(
                "upload shared initializer `{name}` to CUDA device {device}: {err}"
            ))
        })
}

/// A value over `owner`'s memory that does not own it.
fn borrowed_view(owner: &DynValue, range: &InitializerRange) -> Result<DynValue> {
    let tensor = owner
        .downcast_ref::<ort::value::DynTensorValueType>()
        .map_err(map_ort_err)?;
    let element: ort::value::TensorElementType = *tensor.data_type();
    let shape = range.shape.iter().map(|dim| *dim as i64).collect::<Vec<_>>();
    let api = ort::api();
    let mut out: *mut ort::sys::OrtValue = std::ptr::null_mut();
    // SAFETY: the pointer, byte length, shape and element type all describe
    // `owner`'s live device buffer; `SharedInitializers` keeps `owner` alive
    // alongside the view.
    let status = unsafe {
        (api.CreateTensorWithDataAsOrtValue)(
            tensor.memory_info().ptr(),
            tensor.data_ptr() as *mut _,
            range.length as usize,
            shape.as_ptr(),
            shape.len(),
            element.into(),
            &mut out,
        )
    };
    if !status.0.is_null() {
        // SAFETY: a non-null status is a valid OrtStatus we must release.
        let message = unsafe {
            let text = std::ffi::CStr::from_ptr((api.GetErrorMessage)(status.0))
                .to_string_lossy()
                .into_owned();
            (api.ReleaseStatus)(status.0);
            text
        };
        return Err(InfraError::Backend(format!(
            "wrap shared initializer `{}`: {message}",
            range.name
        )));
    }
    let out = std::ptr::NonNull::new(out).ok_or_else(|| {
        InfraError::Backend(format!("wrap shared initializer `{}` returned null", range.name))
    })?;
    // SAFETY: `out` is a fresh OrtValue owned by the returned Value.
    Ok(unsafe { DynValue::from_ptr(out, None) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_upload_reads_little_endian_ranges() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("blob.data");
        let mut bytes = vec![0xAAu8; 3];
        for value in [1.5f32, -2.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&path, &bytes).expect("write blob");
        let backend = OrtBackend::new(ProviderSelection::from_strings(&["cpu".to_string()]));
        let shared = backend
            .upload_initializers(
                &path,
                &[InitializerRange {
                    name: "w".to_string(),
                    element: TensorElement::F32,
                    shape: vec![2],
                    offset: 3,
                    length: 8,
                }],
            )
            .expect("upload");
        assert_eq!((shared.len(), shared.bytes(), shared.cuda_device()), (1, 8, None));
        let (_, data) = shared.values()[0]
            .1
            .try_extract_tensor::<f32>()
            .expect("f32 tensor");
        assert_eq!(data, &[1.5, -2.0]);
    }

    #[test]
    fn length_mismatch_is_rejected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("blob.data");
        std::fs::write(&path, [0u8; 16]).expect("write blob");
        let backend = OrtBackend::new(ProviderSelection::from_strings(&["cpu".to_string()]));
        let range = InitializerRange {
            name: "w".to_string(),
            element: TensorElement::F16,
            shape: vec![3],
            offset: 0,
            length: 8,
        };
        assert!(backend.upload_initializers(&path, &[range]).is_err());
    }
}
