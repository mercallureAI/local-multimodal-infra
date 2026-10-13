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
    /// Last: held while the fields above are released (see `Drop`).
    drop_gate: crate::gpu_gate::DropGate,
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

/// Offsets every range of a packed data file is aligned to: such a file is
/// uploaded as one device buffer, the tensors views into it.
pub const PACKED_ALIGNMENT: u64 = 256;

impl OrtBackend {
    /// Reads `ranges` from `data_file` and places them on this backend's
    /// preferred CUDA device (host memory when CUDA is not selected). Sessions
    /// that should fall back to CPU must not receive CUDA-resident values.
    ///
    /// When every range starts at a multiple of [`PACKED_ALIGNMENT`], the
    /// span they cover goes up in one copy and each tensor is a view into it:
    /// copying hundreds of small tensors one by one leaves each copy's
    /// staging buffer in the copy session's arena (SenseVoice's 1198 tensors,
    /// 266 MB, took ~5.7 GB that way).
    pub fn upload_initializers(
        &self,
        data_file: &Path,
        ranges: &[InitializerRange],
    ) -> Result<SharedInitializers> {
        let cuda_device = self.preferred_cuda_device();
        if let Some(device) = cuda_device.filter(|_| {
            !ranges.is_empty()
                && ranges
                    .iter()
                    .all(|range| range.offset % PACKED_ALIGNMENT == 0)
        }) {
            return upload_packed(data_file, ranges, device);
        }
        let mut file =
            File::open(data_file).map_err(|e| InfraError::io(Some(data_file.to_path_buf()), e))?;
        let mut values = Vec::with_capacity(ranges.len());
        let mut owners = Vec::new();
        let mut bytes = 0u64;
        let mut buffer = Vec::new();
        // Each copy stages the host tensor in a temporary device buffer of the
        // copy session's arena, which keeps the memory. Largest first, every
        // later staging buffer fits in the first one, so the arena retains one
        // tensor's worth instead of a growing set of buffers.
        let mut order = (0..ranges.len()).collect::<Vec<_>>();
        order.sort_by_key(|&i| std::cmp::Reverse(ranges[i].length));
        let mut uploaded = Vec::with_capacity(ranges.len());
        for range in order.into_iter().map(|i| (i, &ranges[i])) {
            let (index, range) = range;
            let expected = element_size(range.element)
                .and_then(|size| {
                    range
                        .shape
                        .iter()
                        .try_fold(size, |acc, dim| acc.checked_mul(*dim))
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
            uploaded.push((index, (range.name.clone(), Arc::new(value))));
            bytes += range.length;
        }
        uploaded.sort_by_key(|(index, _)| *index);
        values.extend(uploaded.into_iter().map(|(_, value)| value));
        Ok(SharedInitializers {
            drop_gate: Default::default(),
            values: Arc::new(values),
            _owners: Arc::new(owners),
            cuda_device,
            bytes,
        })
    }
}

/// The span of `data_file` that `ranges` cover, uploaded in one copy, each
/// range a view into it.
fn upload_packed(
    data_file: &Path,
    ranges: &[InitializerRange],
    device: u32,
) -> Result<SharedInitializers> {
    let start = ranges.iter().map(|range| range.offset).min().unwrap_or(0);
    let end = ranges
        .iter()
        .map(|range| range.offset + range.length)
        .max()
        .unwrap_or(0);
    for range in ranges {
        let expected = element_size(range.element).and_then(|size| {
            range
                .shape
                .iter()
                .try_fold(size, |acc, dim| acc.checked_mul(*dim))
        });
        if expected.map(|bytes| bytes as u64) != Some(range.length) {
            return Err(InfraError::Backend(format!(
                "shared initializer `{}` length {} does not match {:?} {:?}",
                range.name, range.length, range.element, range.shape
            )));
        }
    }
    let mut file =
        File::open(data_file).map_err(|e| InfraError::io(Some(data_file.to_path_buf()), e))?;
    let mut span = vec![0u8; (end - start) as usize];
    file.seek(SeekFrom::Start(start))
        .and_then(|_| file.read_exact(&mut span))
        .map_err(|e| InfraError::io(Some(data_file.to_path_buf()), e))?;
    let host = Tensor::from_array((vec![span.len() as i64], span.into_boxed_slice()))
        .map(|tensor| tensor.into_dyn())
        .map_err(map_ort_err)?;
    let owner = to_cuda(host, device as i32, &data_file.display().to_string())?;
    let tensor = owner
        .downcast_ref::<ort::value::DynTensorValueType>()
        .map_err(map_ort_err)?;
    let base = tensor.data_ptr() as *mut u8;
    let mut values = Vec::with_capacity(ranges.len());
    for range in ranges {
        let shape = range
            .shape
            .iter()
            .map(|dim| *dim as i64)
            .collect::<Vec<_>>();
        // SAFETY: the range lies inside the uploaded span (offset and length
        // checked against it above), which `owner` keeps alive alongside the
        // views; the element type and shape describe its bytes.
        let view = unsafe {
            wrap_memory(
                tensor.memory_info(),
                &ort::memory::Allocator::default(),
                base.add((range.offset - start) as usize) as *mut _,
                range.length as usize,
                &shape,
                ort_element(range.element, &range.name)?,
                &range.name,
            )
        }?;
        values.push((range.name.clone(), Arc::new(view)));
    }
    Ok(SharedInitializers {
        drop_gate: Default::default(),
        values: Arc::new(values),
        _owners: Arc::new(vec![owner]),
        cuda_device: Some(device),
        bytes: ranges.iter().map(|range| range.length).sum(),
    })
}

fn ort_element(element: TensorElement, name: &str) -> Result<ort::value::TensorElementType> {
    use ort::value::TensorElementType as Ort;
    Ok(match element {
        TensorElement::F32 => Ort::Float32,
        TensorElement::F16 => Ort::Float16,
        TensorElement::Bool => Ort::Bool,
        TensorElement::I8 => Ort::Int8,
        TensorElement::U8 => Ort::Uint8,
        TensorElement::I16 => Ort::Int16,
        TensorElement::I32 => Ort::Int32,
        TensorElement::I64 => Ort::Int64,
        TensorElement::Other => {
            return Err(InfraError::Backend(format!(
                "shared initializer `{name}` has an unsupported element type"
            )))
        }
    })
}

pub(crate) fn element_size(element: TensorElement) -> Option<usize> {
    Some(match element {
        TensorElement::F32 | TensorElement::I32 => 4,
        TensorElement::F16 | TensorElement::I16 => 2,
        TensorElement::I64 => 8,
        TensorElement::I8 | TensorElement::U8 | TensorElement::Bool => 1,
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
            le_values(bytes, |raw: [u8; 2]| {
                half::f16::from_bits(u16::from_le_bytes(raw))
            }),
        ),
        TensorElement::I64 => build(shape, le_values(bytes, i64::from_le_bytes)),
        TensorElement::I32 => build(shape, le_values(bytes, i32::from_le_bytes)),
        TensorElement::I16 => build(shape, le_values(bytes, i16::from_le_bytes)),
        TensorElement::I8 => build(shape, bytes.iter().map(|byte| *byte as i8).collect()),
        TensorElement::U8 => build(shape, bytes.to_vec()),
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
    let _gate = crate::gpu_shared();
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
    device_view(owner, &range.shape, range.length as usize, &range.name)
}

/// A value of `shape` over the first `bytes` of `owner`'s buffer that does not
/// own the memory: the caller keeps `owner` alive for as long as the view.
pub(crate) fn device_view(
    owner: &DynValue,
    shape: &[usize],
    bytes: usize,
    what: &str,
) -> Result<DynValue> {
    let tensor = owner
        .downcast_ref::<ort::value::DynTensorValueType>()
        .map_err(map_ort_err)?;
    let shape = shape.iter().map(|dim| *dim as i64).collect::<Vec<_>>();
    // SAFETY: the pointer, byte length, shape and element type all describe
    // `owner`'s live buffer, which the caller keeps alive alongside the view.
    unsafe {
        // Only ever bound to sessions, never copied: see `wrap_memory`.
        wrap_memory(
            tensor.memory_info(),
            &ort::memory::Allocator::default(),
            tensor.data_ptr() as *mut _,
            bytes,
            &shape,
            *tensor.data_type(),
            what,
        )
    }
}

/// A value of `shape` and `element` over `bytes` at `data` in `memory`, not
/// owning it. ort's `TensorRefMut::from_raw` makes only typed views that
/// cannot become owned values; session initializers and fixed bindings need
/// an owned `DynValue` whose element type is known at run time.
///
/// ORT runs the value in `memory`; ort itself describes it by `allocator`
/// (`Value::memory_info`, and so the device `copy_into` copies on), which it
/// keeps alive but never frees the data with. Views that are copied to or from
/// need an allocator of `memory`'s device (a session's); views only bound to a
/// session can take `Allocator::default()`.
///
/// # Safety
/// `data` must point to `bytes` live bytes in `memory`, kept alive for as long
/// as the returned value.
pub(crate) unsafe fn wrap_memory(
    memory: &ort::memory::MemoryInfo<'_>,
    allocator: &ort::memory::Allocator,
    data: *mut std::ffi::c_void,
    bytes: usize,
    shape: &[i64],
    element: ort::value::TensorElementType,
    what: &str,
) -> Result<DynValue> {
    let api = ort::api();
    let mut out: *mut ort::sys::OrtValue = std::ptr::null_mut();
    // SAFETY: per this function's contract.
    let status = unsafe {
        (api.CreateTensorWithDataAsOrtValue)(
            memory.ptr(),
            data as *mut _,
            bytes,
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
        return Err(InfraError::Backend(format!("wrap `{what}`: {message}")));
    }
    let out = std::ptr::NonNull::new(out)
        .ok_or_else(|| InfraError::Backend(format!("wrap `{what}` returned null")))?;
    // SAFETY: `out` is a fresh OrtValue owned by the returned Value.
    Ok(unsafe { DynValue::from_ptr(out, Some(allocator)) })
}

impl Drop for SharedInitializers {
    fn drop(&mut self) {
        // The last clone frees the device buffers: not during a capture.
        self.drop_gate.hold();
    }
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
        assert_eq!(
            (shared.len(), shared.bytes(), shared.cuda_device()),
            (1, 8, None)
        );
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
