use local_error::{InfraError, Result};
#[cfg(any(
    feature = "cuda",
    all(feature = "directml", target_os = "windows"),
    feature = "tensorrt"
))]
use ort::ep::ExecutionProvider;
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::{DynTensor, Tensor, TensorElementType, ValueType},
};
use serde::{Deserialize, Serialize};
#[cfg(any(feature = "cuda", test))]
use std::sync::OnceLock;
use std::{
    borrow::Cow,
    collections::HashSet,
    path::{Path, PathBuf},
};

mod io_binding;
pub use io_binding::{
    ResidentBindingOutputs, ResidentCudaTensor, ResidentIoBinding, ResidentTensorInput,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    Cpu,
    Cuda,
    Dml,
    Trt,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderOptions {
    pub kind: ProviderKind,
    #[serde(default)]
    pub device_id: Option<u32>,
}

impl ProviderOptions {
    pub fn cpu() -> Self {
        Self {
            kind: ProviderKind::Cpu,
            device_id: None,
        }
    }
    pub fn cuda(device_id: Option<u32>) -> Self {
        Self {
            kind: ProviderKind::Cuda,
            device_id,
        }
    }
    pub fn dml(device_id: Option<u32>) -> Self {
        Self {
            kind: ProviderKind::Dml,
            device_id,
        }
    }
    pub fn trt(device_id: Option<u32>) -> Self {
        Self {
            kind: ProviderKind::Trt,
            device_id,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderSelection {
    pub order: Vec<ProviderOptions>,
    #[serde(default = "default_true")]
    pub fallback_to_cpu: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RuntimeExecutionProviderAvailability {
    pub cpu: bool,
    pub cuda: bool,
    pub dml: bool,
    pub trt: bool,
}

impl RuntimeExecutionProviderAvailability {
    pub fn is_available(self, provider: ProviderKind) -> bool {
        match provider {
            ProviderKind::Cpu => self.cpu,
            ProviderKind::Cuda => self.cuda,
            ProviderKind::Dml => self.dml,
            ProviderKind::Trt => self.trt,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionProviderReport {
    pub provider: ProviderKind,
    pub cpu_fallback_used: bool,
}

/// Probe actual ONNX Runtime execution-provider availability in the active
/// process/binary rather than relying on compile-time cargo features alone.
///
/// CPU is always reported as available. Optional GPU providers are reported as
/// `false` whenever support is not compiled in, the current target is
/// unsupported, ORT reports the provider unavailable, or the availability probe
/// itself errors.
pub fn probe_runtime_execution_provider_availability() -> RuntimeExecutionProviderAvailability {
    RuntimeExecutionProviderAvailability {
        cpu: true,
        cuda: probe_cuda_runtime_availability(),
        dml: probe_dml_runtime_availability(),
        trt: probe_trt_runtime_availability(),
    }
}

#[cfg(feature = "cuda")]
static CUDA_RUNTIME_USABILITY: OnceLock<bool> = OnceLock::new();

fn probe_cuda_runtime_availability() -> bool {
    #[cfg(feature = "cuda")]
    {
        cached_cuda_runtime_usability_with(&CUDA_RUNTIME_USABILITY, || {
            probe_cuda_runtime_usability_with(
                || ort::ep::CUDA::default().is_available().map_err(map_ort_err),
                create_cuda_usability_probe_session,
            )
        })
    }

    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

#[cfg(any(feature = "cuda", test))]
fn cached_cuda_runtime_usability_with(
    cache: &OnceLock<bool>,
    probe: impl FnOnce() -> Result<()>,
) -> bool {
    *cache.get_or_init(|| match probe() {
        Ok(()) => {
            tracing::debug!(
                "CUDA ORT execution provider passed process-level session creation probe"
            );
            true
        }
        Err(err) => {
            tracing::warn!(
                error = %err,
                "CUDA ORT execution provider is compiled in but unusable; CUDA will be filtered before model loading"
            );
            false
        }
    })
}

#[cfg(any(feature = "cuda", test))]
fn probe_cuda_runtime_usability_with(
    check_compiled_provider: impl FnOnce() -> Result<bool>,
    create_probe_session: impl FnOnce() -> Result<()>,
) -> Result<()> {
    if !check_compiled_provider()? {
        return Err(InfraError::Unsupported(
            "CUDA execution provider is not included in the active ONNX Runtime binary".to_string(),
        ));
    }
    create_probe_session()
}

#[cfg(feature = "cuda")]
fn create_cuda_usability_probe_session() -> Result<()> {
    // A deterministic one-node FP32 Identity graph. Loading it from memory avoids
    // filesystem lifetime and cleanup concerns. Disabling graph optimization
    // preserves the node while ORT registers and initializes the CUDA EP.
    const CUDA_PROBE_ONNX: &[u8] = &[
        0x08, 0x08, 0x12, 0x1c, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x2d, 0x62, 0x61, 0x63, 0x6b, 0x65,
        0x6e, 0x64, 0x2d, 0x6f, 0x72, 0x74, 0x2d, 0x63, 0x75, 0x64, 0x61, 0x2d, 0x70, 0x72, 0x6f,
        0x62, 0x65, 0x3a, 0x4a, 0x0a, 0x10, 0x0a, 0x01, 0x78, 0x12, 0x01, 0x79, 0x22, 0x08, 0x49,
        0x64, 0x65, 0x6e, 0x74, 0x69, 0x74, 0x79, 0x12, 0x14, 0x63, 0x75, 0x64, 0x61, 0x5f, 0x75,
        0x73, 0x61, 0x62, 0x69, 0x6c, 0x69, 0x74, 0x79, 0x5f, 0x70, 0x72, 0x6f, 0x62, 0x65, 0x5a,
        0x0f, 0x0a, 0x01, 0x78, 0x12, 0x0a, 0x0a, 0x08, 0x08, 0x01, 0x12, 0x04, 0x0a, 0x02, 0x08,
        0x01, 0x62, 0x0f, 0x0a, 0x01, 0x79, 0x12, 0x0a, 0x0a, 0x08, 0x08, 0x01, 0x12, 0x04, 0x0a,
        0x02, 0x08, 0x01, 0x42, 0x02, 0x10, 0x0d,
    ];

    let cuda = ort::ep::CUDA::default();
    Session::builder()
        .map_err(map_ort_err)?
        .with_execution_providers([cuda.build().error_on_failure()])
        .map_err(map_ort_err)?
        .with_optimization_level(GraphOptimizationLevel::Disable)
        .map_err(map_ort_err)?
        .commit_from_memory(CUDA_PROBE_ONNX)
        .map_err(map_ort_err)?;
    Ok(())
}

fn probe_dml_runtime_availability() -> bool {
    #[cfg(all(feature = "directml", target_os = "windows"))]
    {
        match ort::ep::DirectML::default().is_available() {
            Ok(available) => available,
            Err(err) => {
                let err = map_ort_err(err);
                tracing::warn!(error = %err, "failed to probe DirectML ORT execution provider availability");
                false
            }
        }
    }

    #[cfg(not(all(feature = "directml", target_os = "windows")))]
    {
        false
    }
}

fn probe_trt_runtime_availability() -> bool {
    #[cfg(feature = "tensorrt")]
    {
        let trt = ort::ep::TensorRT::default();
        if !trt.supported_by_platform() {
            return false;
        }
        match trt.is_available() {
            Ok(available) => available,
            Err(err) => {
                let err = map_ort_err(err);
                tracing::warn!(error = %err, "failed to probe TensorRT ORT execution provider availability");
                false
            }
        }
    }

    #[cfg(not(feature = "tensorrt"))]
    {
        false
    }
}

impl Default for ProviderSelection {
    fn default() -> Self {
        Self {
            order: vec![ProviderOptions::cpu()],
            fallback_to_cpu: true,
        }
    }
}

fn default_true() -> bool {
    true
}

impl ProviderSelection {
    pub fn from_strings(order: &[String]) -> Self {
        let providers = order
            .iter()
            .filter_map(|name| match name.to_ascii_lowercase().as_str() {
                "cpu" => Some(ProviderOptions::cpu()),
                "cuda" => Some(ProviderOptions::cuda(None)),
                "dml" | "directml" => Some(ProviderOptions::dml(None)),
                "trt" | "tensorrt" => Some(ProviderOptions::trt(None)),
                _ => None,
            })
            .collect::<Vec<_>>();
        if providers.is_empty() {
            Self::default()
        } else {
            Self {
                order: providers,
                fallback_to_cpu: true,
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct OrtInput {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct OrtOutput {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorElement {
    F32,
    F16,
    Bool,
    I8,
    I16,
    I32,
    I64,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMetadata {
    pub name: String,
    pub element_type: TensorElement,
    /// ONNX Runtime reports unknown/dynamic dimensions as -1. Symbol names, when
    /// present in the ONNX graph, are preserved in `dimension_symbols`.
    pub shape: Vec<i64>,
    pub dimension_symbols: Vec<Option<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    pub inputs: Vec<TensorMetadata>,
    pub outputs: Vec<TensorMetadata>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OrtTensorData {
    F32(Vec<f32>),
    F16(Vec<half::f16>),
    Bool(Vec<bool>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
}

impl OrtTensorData {
    pub fn element_type(&self) -> TensorElement {
        match self {
            Self::F32(_) => TensorElement::F32,
            Self::F16(_) => TensorElement::F16,
            Self::Bool(_) => TensorElement::Bool,
            Self::I8(_) => TensorElement::I8,
            Self::I16(_) => TensorElement::I16,
            Self::I32(_) => TensorElement::I32,
            Self::I64(_) => TensorElement::I64,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::F32(data) => data.len(),
            Self::F16(data) => data.len(),
            Self::Bool(data) => data.len(),
            Self::I8(data) => data.len(),
            Self::I16(data) => data.len(),
            Self::I32(data) => data.len(),
            Self::I64(data) => data.len(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OrtTensorInput {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: OrtTensorData,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OrtTensorOutput {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: OrtTensorData,
}

impl From<OrtInput> for OrtTensorInput {
    fn from(input: OrtInput) -> Self {
        Self {
            name: input.name,
            shape: input.shape,
            data: OrtTensorData::F32(input.data),
        }
    }
}

impl TryFrom<OrtTensorOutput> for OrtOutput {
    type Error = InfraError;

    fn try_from(output: OrtTensorOutput) -> Result<Self> {
        match output.data {
            OrtTensorData::F32(data) => Ok(Self {
                name: output.name,
                shape: output.shape,
                data,
            }),
            other => Err(InfraError::Backend(format!(
                "output `{}` is {:?}; run_f32 only accepts f32 outputs",
                output.name,
                other.element_type()
            ))),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OrtBackend {
    selection: ProviderSelection,
    cpu_session_options: Option<CpuSessionOptions>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuSessionOptions {
    pub intra_threads: usize,
    pub inter_threads: usize,
}

impl OrtBackend {
    pub fn new(selection: ProviderSelection) -> Self {
        Self {
            selection,
            cpu_session_options: None,
        }
    }

    pub fn with_cpu_session_options(mut self, options: CpuSessionOptions) -> Result<Self> {
        if options.intra_threads == 0 || options.inter_threads == 0 {
            return Err(InfraError::Backend(
                "ORT CPU thread counts must be greater than zero".to_string(),
            ));
        }
        self.cpu_session_options = Some(options);
        Ok(self)
    }

    pub fn load_session(&self, model_path: impl AsRef<Path>) -> Result<OrtSession> {
        OrtSession::load_with_cpu_options(
            model_path.as_ref(),
            self.selection.clone(),
            self.cpu_session_options,
        )
    }
}

#[derive(Debug)]
pub struct OrtSession {
    model_path: PathBuf,
    provider: ProviderKind,
    cpu_fallback_used: bool,
    device_id: Option<u32>,
    real: RealSession,
}

impl OrtSession {
    pub fn load(model_path: &Path, selection: ProviderSelection) -> Result<Self> {
        Self::load_with_cpu_options(model_path, selection, None)
    }

    fn load_with_cpu_options(
        model_path: &Path,
        selection: ProviderSelection,
        cpu_session_options: Option<CpuSessionOptions>,
    ) -> Result<Self> {
        Self::load_with_provider_loaders(
            model_path,
            selection,
            cpu_session_options,
            RealSession::load_cpu,
            RealSession::load_cuda,
        )
    }

    fn load_with_provider_loaders<LoadCpu, LoadCuda>(
        model_path: &Path,
        selection: ProviderSelection,
        cpu_session_options: Option<CpuSessionOptions>,
        mut load_cpu: LoadCpu,
        mut load_cuda: LoadCuda,
    ) -> Result<Self>
    where
        LoadCpu: FnMut(&Path, Option<CpuSessionOptions>) -> Result<RealSession>,
        LoadCuda: FnMut(&Path, &ProviderOptions) -> Result<RealSession>,
    {
        if !model_path.exists() {
            return Err(InfraError::ModelNotConfigured {
                model_id: "unknown".to_string(),
                reason: format!("ONNX file not found: {}", model_path.display()),
            });
        }

        let mut attempted_cpu = false;
        let mut errors = Vec::new();

        for provider in &selection.order {
            match provider.kind {
                ProviderKind::Cpu => {
                    attempted_cpu = true;
                    match load_cpu(model_path, cpu_session_options) {
                        Ok(real) => {
                            return Ok(Self {
                                model_path: model_path.to_path_buf(),
                                provider: ProviderKind::Cpu,
                                cpu_fallback_used: !errors.is_empty(),
                                device_id: None,
                                real,
                            });
                        }
                        Err(err) => errors.push(format!("cpu: {err}")),
                    }
                }
                ProviderKind::Cuda => match load_cuda(model_path, provider) {
                    Ok(real) => {
                        return Ok(Self {
                            model_path: model_path.to_path_buf(),
                            provider: ProviderKind::Cuda,
                            cpu_fallback_used: false,
                            device_id: Some(provider.device_id.unwrap_or(0)),
                            real,
                        });
                    }
                    Err(err) => {
                        tracing::warn!(error = %err, "CUDA provider was requested but was not selected");
                        errors.push(format!("cuda: {err}"));
                    }
                },
                ProviderKind::Dml => match RealSession::load_dml(model_path, provider) {
                    Ok(real) => {
                        return Ok(Self {
                            model_path: model_path.to_path_buf(),
                            provider: ProviderKind::Dml,
                            cpu_fallback_used: false,
                            device_id: provider.device_id,
                            real,
                        });
                    }
                    Err(err) => {
                        tracing::warn!(error = %err, "DML provider was requested but was not selected");
                        errors.push(format!("dml: {err}"));
                    }
                },
                ProviderKind::Trt => match RealSession::load_trt(model_path, provider) {
                    Ok(real) => {
                        return Ok(Self {
                            model_path: model_path.to_path_buf(),
                            provider: ProviderKind::Trt,
                            cpu_fallback_used: false,
                            device_id: provider.device_id,
                            real,
                        });
                    }
                    Err(err) => {
                        tracing::warn!(error = %err, "TensorRT provider was requested but was not selected");
                        errors.push(format!("tensorrt: {err}"));
                    }
                },
            }
        }

        if selection.fallback_to_cpu && !attempted_cpu {
            tracing::warn!("falling back to CPU provider after configured providers failed");
            match load_cpu(model_path, cpu_session_options) {
                Ok(real) => {
                    return Ok(Self {
                        model_path: model_path.to_path_buf(),
                        provider: ProviderKind::Cpu,
                        cpu_fallback_used: true,
                        device_id: None,
                        real,
                    });
                }
                Err(err) => errors.push(format!("cpu fallback: {err}")),
            }
        }

        Err(InfraError::Backend(format!(
            "no usable ORT execution provider selected for {}: {}",
            model_path.display(),
            errors.join("; ")
        )))
    }

    pub fn provider(&self) -> ProviderKind {
        self.provider
    }
    pub fn cpu_fallback_used(&self) -> bool {
        self.cpu_fallback_used
    }
    /// Whether selecting this session required abandoning a requested
    /// execution provider and recreating the whole session on CPU.
    ///
    /// This does not claim that every node in a CUDA-selected graph executes
    /// on CUDA. ORT may intentionally assign shape/control nodes to CPU.
    pub fn whole_session_cpu_fallback_used(&self) -> bool {
        self.cpu_fallback_used
    }
    pub fn device_id(&self) -> Option<u32> {
        self.device_id
    }
    pub fn provider_report(&self) -> SessionProviderReport {
        SessionProviderReport {
            provider: self.provider(),
            cpu_fallback_used: self.cpu_fallback_used(),
        }
    }
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    pub fn metadata(&self) -> &SessionMetadata {
        self.real.metadata()
    }

    pub fn inputs(&self) -> &[TensorMetadata] {
        &self.real.metadata().inputs
    }

    pub fn outputs(&self) -> &[TensorMetadata] {
        &self.real.metadata().outputs
    }

    pub fn run_f32(&mut self, inputs: &[OrtInput]) -> Result<Vec<OrtOutput>> {
        let typed_inputs = inputs
            .iter()
            .cloned()
            .map(OrtTensorInput::from)
            .collect::<Vec<_>>();
        self.real
            .run_tensors(&typed_inputs)?
            .into_iter()
            .map(OrtOutput::try_from)
            .collect()
    }

    pub fn run_tensors(&mut self, inputs: &[OrtTensorInput]) -> Result<Vec<OrtTensorOutput>> {
        self.real.run_tensors(inputs)
    }
}

#[derive(Debug)]
struct RealSession {
    session: Session,
    metadata: SessionMetadata,
}

impl RealSession {
    fn load_cpu(model_path: &Path, options: Option<CpuSessionOptions>) -> Result<Self> {
        // The `ort` dependency is configured with `download-binaries` and
        // `copy-dylibs`, so build/check does not depend on a system-wide ORT
        // installation. At runtime, ORT's downloaded CPU dylib is copied beside
        // test/app binaries; deployments can still override with ort-supported
        // environment variables such as ORT_LIB_PATH.
        let mut builder = Session::builder().map_err(map_ort_err)?;
        if let Some(options) = options {
            builder = builder
                .with_intra_threads(options.intra_threads)
                .map_err(map_ort_err)?
                .with_inter_threads(options.inter_threads)
                .map_err(map_ort_err)?
                .with_parallel_execution(false)
                .map_err(map_ort_err)?;
        }
        Self::load_with_builder(builder, model_path)
    }

    fn load_cuda(model_path: &Path, provider: &ProviderOptions) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let cuda = ort::ep::CUDA::default();
            if !cuda.is_available().map_err(map_ort_err)? {
                return Err(InfraError::Unsupported(format!(
                    "CUDA ORT execution provider is not available in the active ONNX Runtime binary (requested device {:?}); falling back to CPU if configured",
                    provider.device_id
                )));
            }
            let mut cuda = cuda;
            if let Some(device_id) = provider.device_id {
                cuda = cuda.with_device_id(device_id as i32);
            }
            let builder = Session::builder()
                .map_err(map_ort_err)?
                .with_execution_providers([cuda.build().error_on_failure()])
                .map_err(map_ort_err)?;
            return Self::load_with_builder(builder, model_path);
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = model_path;
            Err(InfraError::Unsupported(format!(
                "CUDA ORT execution provider support is not compiled into local-backend-ort (requested device {:?}); falling back to CPU if configured",
                provider.device_id
            )))
        }
    }

    fn load_dml(model_path: &Path, provider: &ProviderOptions) -> Result<Self> {
        #[cfg(all(feature = "directml", target_os = "windows"))]
        {
            let dml = ort::ep::DirectML::default();
            if !dml.is_available().map_err(map_ort_err)? {
                return Err(InfraError::Unsupported(format!(
                    "DirectML ORT execution provider is not available in the active ONNX Runtime binary (requested device {:?}); falling back to CPU if configured",
                    provider.device_id
                )));
            }
            let mut dml = dml;
            if let Some(device_id) = provider.device_id {
                dml = dml.with_device_id(device_id as i32);
            }
            let builder = Session::builder()
                .map_err(map_ort_err)?
                .with_execution_providers([dml.build().error_on_failure()])
                .map_err(map_ort_err)?;
            return Self::load_with_builder(builder, model_path);
        }

        #[cfg(not(all(feature = "directml", target_os = "windows")))]
        {
            let _ = model_path;
            Err(InfraError::Unsupported(format!(
                "DirectML ORT execution provider support is not compiled into local-backend-ort for this target (requested device {:?}); falling back to CPU if configured",
                provider.device_id
            )))
        }
    }

    fn load_trt(model_path: &Path, provider: &ProviderOptions) -> Result<Self> {
        #[cfg(feature = "tensorrt")]
        {
            let trt = ort::ep::TensorRT::default();
            if !trt.supported_by_platform() {
                return Err(InfraError::Unsupported(format!(
                    "TensorRT ORT execution provider is not supported on this target by the active ort build (requested device {:?}); local-backend-ort does not yet model a same-session TensorRT+CUDA stack, so prefer provider_order [trt, cuda, cpu] across whole-session retries when available",
                    provider.device_id
                )));
            }
            if !trt.is_available().map_err(map_ort_err)? {
                return Err(InfraError::Unsupported(format!(
                    "TensorRT ORT execution provider is not available in the active ONNX Runtime binary (requested device {:?}); local-backend-ort does not yet model a same-session TensorRT+CUDA stack, so prefer provider_order [trt, cuda, cpu] and fall back to CPU if configured",
                    provider.device_id
                )));
            }
            let mut trt = trt;
            if let Some(device_id) = provider.device_id {
                trt = trt.with_device_id(device_id as i32);
            }
            let builder = Session::builder()
                .map_err(map_ort_err)?
                .with_execution_providers([trt.build().error_on_failure()])
                .map_err(map_ort_err)?;
            return Self::load_with_builder(builder, model_path);
        }

        #[cfg(not(feature = "tensorrt"))]
        {
            let _ = model_path;
            Err(InfraError::Unsupported(format!(
                "TensorRT ORT execution provider support is not compiled into local-backend-ort (requested device {:?}); local-backend-ort does not yet model a same-session TensorRT+CUDA stack, so typical provider_order is [trt, cuda, cpu] only in builds that enable both features",
                provider.device_id
            )))
        }
    }

    fn load_with_builder(
        builder: ort::session::builder::SessionBuilder,
        model_path: &Path,
    ) -> Result<Self> {
        let session = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(map_ort_err)?
            .commit_from_file(model_path)
            .map_err(map_ort_err)?;
        let metadata = SessionMetadata {
            inputs: session.inputs().iter().map(tensor_metadata).collect(),
            outputs: session.outputs().iter().map(tensor_metadata).collect(),
        };
        Ok(Self { session, metadata })
    }

    fn metadata(&self) -> &SessionMetadata {
        &self.metadata
    }

    fn run_tensors(&mut self, inputs: &[OrtTensorInput]) -> Result<Vec<OrtTensorOutput>> {
        self.validate_inputs(inputs)?;

        let mut values = Vec::<(Cow<'_, str>, DynTensor)>::with_capacity(inputs.len());
        for input in inputs {
            let shape = input
                .shape
                .iter()
                .map(|dim| i64::try_from(*dim).map_err(|_| shape_overflow(&input.name, *dim)))
                .collect::<Result<Vec<_>>>()?;
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

            let tensor = match &input.data {
                OrtTensorData::F32(data) => {
                    Tensor::from_array((shape, data.clone().into_boxed_slice()))
                        .map_err(map_ort_err)?
                        .upcast()
                }
                OrtTensorData::F16(data) => {
                    Tensor::from_array((shape, data.clone().into_boxed_slice()))
                        .map_err(map_ort_err)?
                        .upcast()
                }
                OrtTensorData::Bool(data) => {
                    Tensor::from_array((shape, data.clone().into_boxed_slice()))
                        .map_err(map_ort_err)?
                        .upcast()
                }
                OrtTensorData::I8(data) => {
                    Tensor::from_array((shape, data.clone().into_boxed_slice()))
                        .map_err(map_ort_err)?
                        .upcast()
                }
                OrtTensorData::I16(data) => {
                    Tensor::from_array((shape, data.clone().into_boxed_slice()))
                        .map_err(map_ort_err)?
                        .upcast()
                }
                OrtTensorData::I32(data) => {
                    Tensor::from_array((shape, data.clone().into_boxed_slice()))
                        .map_err(map_ort_err)?
                        .upcast()
                }
                OrtTensorData::I64(data) => {
                    Tensor::from_array((shape, data.clone().into_boxed_slice()))
                        .map_err(map_ort_err)?
                        .upcast()
                }
            };
            values.push((Cow::Owned(input.name.clone()), tensor));
        }

        let outputs = self.session.run(values).map_err(map_ort_err)?;
        outputs
            .into_iter()
            .map(|(name, value)| {
                if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
                    return Ok(OrtTensorOutput {
                        name: name.to_string(),
                        shape: shape_to_usize(name, shape)?,
                        data: OrtTensorData::F32(data.to_vec()),
                    });
                }
                if let Ok((shape, data)) = value.try_extract_tensor::<half::f16>() {
                    return Ok(OrtTensorOutput {
                        name: name.to_string(),
                        shape: shape_to_usize(name, shape)?,
                        data: OrtTensorData::F16(data.to_vec()),
                    });
                }
                if let Ok((shape, data)) = value.try_extract_tensor::<bool>() {
                    return Ok(OrtTensorOutput {
                        name: name.to_string(),
                        shape: shape_to_usize(name, shape)?,
                        data: OrtTensorData::Bool(data.to_vec()),
                    });
                }
                if let Ok((shape, data)) = value.try_extract_tensor::<i8>() {
                    return Ok(OrtTensorOutput {
                        name: name.to_string(),
                        shape: shape_to_usize(name, shape)?,
                        data: OrtTensorData::I8(data.to_vec()),
                    });
                }
                if let Ok((shape, data)) = value.try_extract_tensor::<i16>() {
                    return Ok(OrtTensorOutput {
                        name: name.to_string(),
                        shape: shape_to_usize(name, shape)?,
                        data: OrtTensorData::I16(data.to_vec()),
                    });
                }
                if let Ok((shape, data)) = value.try_extract_tensor::<i32>() {
                    return Ok(OrtTensorOutput {
                        name: name.to_string(),
                        shape: shape_to_usize(name, shape)?,
                        data: OrtTensorData::I32(data.to_vec()),
                    });
                }
                if let Ok((shape, data)) = value.try_extract_tensor::<i64>() {
                    return Ok(OrtTensorOutput {
                        name: name.to_string(),
                        shape: shape_to_usize(name, shape)?,
                        data: OrtTensorData::I64(data.to_vec()),
                    });
                }
                Err(InfraError::Backend(format!(
                    "output `{name}` has unsupported tensor type; extractable output element types are f32, f16, bool, i8, i16, i32, and i64; available outputs: {}",
                    format_names(self.metadata.outputs.iter().map(|output| output.name.as_str()))
                )))
            })
            .collect()
    }

    fn validate_input_names<'a>(&self, requested: impl IntoIterator<Item = &'a str>) -> Result<()> {
        validate_requested_input_names(&self.metadata, requested)
    }

    fn validate_inputs(&self, inputs: &[OrtTensorInput]) -> Result<()> {
        self.validate_input_names(inputs.iter().map(|input| input.name.as_str()))
    }
}

fn validate_requested_input_names<'a>(
    metadata: &SessionMetadata,
    requested: impl IntoIterator<Item = &'a str>,
) -> Result<()> {
    let requested = requested.into_iter().collect::<Vec<_>>();
    let mut unique = HashSet::with_capacity(requested.len());
    for name in &requested {
        if !unique.insert(*name) {
            return Err(InfraError::Backend(format!(
                "duplicate input name `{name}`; requested inputs: {}",
                format_names(requested.iter().copied())
            )));
        }
    }
    let available = metadata
        .inputs
        .iter()
        .map(|input| input.name.as_str())
        .collect::<HashSet<_>>();
    for name in &requested {
        if !available.contains(name) {
            return Err(input_name_error(metadata, &requested, name));
        }
    }
    for input in &metadata.inputs {
        if !requested.iter().any(|name| *name == input.name) {
            return Err(input_name_error(metadata, &requested, &input.name));
        }
    }
    Ok(())
}

fn tensor_metadata(outlet: &ort::value::Outlet) -> TensorMetadata {
    let (element_type, shape, dimension_symbols) = match outlet.dtype() {
        ValueType::Tensor {
            ty,
            shape,
            dimension_symbols,
        } => (
            tensor_element(*ty),
            shape.iter().copied().collect::<Vec<_>>(),
            dimension_symbols
                .iter()
                .map(|symbol| (!symbol.is_empty()).then(|| symbol.clone()))
                .collect::<Vec<_>>(),
        ),
        _ => (TensorElement::Other, Vec::new(), Vec::new()),
    };
    TensorMetadata {
        name: outlet.name().to_string(),
        element_type,
        shape,
        dimension_symbols,
    }
}

fn tensor_element(element: TensorElementType) -> TensorElement {
    match element {
        TensorElementType::Float32 => TensorElement::F32,
        TensorElementType::Float16 => TensorElement::F16,
        TensorElementType::Bool => TensorElement::Bool,
        TensorElementType::Int8 => TensorElement::I8,
        TensorElementType::Int16 => TensorElement::I16,
        TensorElementType::Int32 => TensorElement::I32,
        TensorElementType::Int64 => TensorElement::I64,
        _ => TensorElement::Other,
    }
}

fn shape_to_usize(name: &str, shape: &[i64]) -> Result<Vec<usize>> {
    shape
        .iter()
        .map(|dim| {
            usize::try_from(*dim).map_err(|_| {
                InfraError::Backend(format!(
                    "output `{name}` has negative or too-large runtime dimension {dim} in shape {shape:?}"
                ))
            })
        })
        .collect()
}

fn shape_overflow(name: &str, dim: usize) -> InfraError {
    InfraError::Backend(format!(
        "input `{name}` dimension {dim} is too large for ORT i64 shape"
    ))
}

fn input_name_error(
    metadata: &SessionMetadata,
    requested: &[&str],
    mismatched: &str,
) -> InfraError {
    InfraError::Backend(format!(
        "input name mismatch for `{mismatched}`; available inputs: {}; requested inputs: {}; available outputs: {}",
        format_names(metadata.inputs.iter().map(|input| input.name.as_str())),
        format_names(requested.iter().copied()),
        format_names(metadata.outputs.iter().map(|output| output.name.as_str()))
    ))
}

fn format_names<'a>(names: impl IntoIterator<Item = &'a str>) -> String {
    let names = names.into_iter().collect::<Vec<_>>();
    if names.is_empty() {
        "<none>".to_string()
    } else {
        names.join(", ")
    }
}

fn map_ort_err<R>(err: ort::Error<R>) -> InfraError {
    InfraError::Backend(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        sync::atomic::{AtomicUsize, Ordering},
    };

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_runtime_probe_is_false_without_cuda_feature() {
        assert!(!probe_cuda_runtime_availability());
    }

    #[test]
    fn cuda_usability_probe_rejects_compiled_provider_when_session_creation_fails() {
        let cache = OnceLock::new();
        let usable = cached_cuda_runtime_usability_with(&cache, || {
            probe_cuda_runtime_usability_with(
                || Ok(true),
                || {
                    Err(InfraError::Backend(
                        "forced CUDA registration/session failure".to_string(),
                    ))
                },
            )
        });

        assert!(!usable);
    }

    #[test]
    fn cuda_usability_probe_accepts_successful_session_creation() {
        let cache = OnceLock::new();
        assert!(cached_cuda_runtime_usability_with(&cache, || {
            probe_cuda_runtime_usability_with(|| Ok(true), || Ok(()))
        }));
    }

    #[test]
    fn cuda_usability_probe_result_is_cached_once_per_cache() {
        let cache = OnceLock::new();
        let attempts = AtomicUsize::new(0);

        for _ in 0..3 {
            assert!(!cached_cuda_runtime_usability_with(&cache, || {
                attempts.fetch_add(1, Ordering::SeqCst);
                Err(InfraError::Backend("forced failure".to_string()))
            }));
        }

        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn parses_provider_order_with_cpu_fallback() {
        let selection = ProviderSelection::from_strings(&["cuda".into(), "cpu".into()]);
        assert_eq!(selection.order[0].kind, ProviderKind::Cuda);
        assert!(selection.fallback_to_cpu);
    }

    #[test]
    fn backend_defaults_and_scoped_cpu_options_are_distinct() {
        let default = OrtBackend::new(ProviderSelection::default());
        assert_eq!(default.cpu_session_options, None);
        let tuned = OrtBackend::new(ProviderSelection::default())
            .with_cpu_session_options(CpuSessionOptions {
                intra_threads: 8,
                inter_threads: 1,
            })
            .expect("valid options");
        assert_eq!(
            tuned.cpu_session_options,
            Some(CpuSessionOptions {
                intra_threads: 8,
                inter_threads: 1
            })
        );
        assert!(OrtBackend::new(ProviderSelection::default())
            .with_cpu_session_options(CpuSessionOptions {
                intra_threads: 0,
                inter_threads: 1
            })
            .is_err());
    }

    #[test]
    fn parses_tensorrt_aliases() {
        let selection =
            ProviderSelection::from_strings(&["trt".into(), "tensorrt".into(), "cpu".into()]);
        assert_eq!(selection.order[0].kind, ProviderKind::Trt);
        assert_eq!(selection.order[1].kind, ProviderKind::Trt);
        assert_eq!(selection.order[2].kind, ProviderKind::Cpu);
    }

    #[test]
    fn missing_model_reports_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let err = OrtSession::load(
            &dir.path().join("missing.onnx"),
            ProviderSelection::default(),
        )
        .expect_err("missing model should fail");
        assert!(err.to_string().contains("ONNX file not found"));
    }

    #[test]
    fn generated_f32_identity_prefers_cuda_then_cpu() {
        let dir = tempfile::tempdir().expect("tempdir");
        let model_path = dir.path().join("identity_f32.onnx");
        fs::write(&model_path, identity_model(1)).expect("write model");

        let mut session = OrtSession::load(
            &model_path,
            ProviderSelection {
                order: vec![ProviderOptions::cuda(Some(0)), ProviderOptions::cpu()],
                fallback_to_cpu: true,
            },
        )
        .expect("load identity model");

        assert_requested_provider_or_cpu_fallback(&session, ProviderKind::Cuda);
        assert_eq!(session.inputs()[0].name, "x");
        assert_eq!(session.inputs()[0].shape, vec![2]);
        assert_eq!(session.inputs()[0].element_type, TensorElement::F32);
        assert_eq!(session.outputs()[0].name, "y");

        let outputs = session
            .run_f32(&[OrtInput {
                name: "x".to_string(),
                shape: vec![2],
                data: vec![1.5, -2.0],
            }])
            .expect("run identity");

        assert_eq!(outputs[0].name, "y");
        assert_eq!(outputs[0].shape, vec![2]);
        assert_eq!(outputs[0].data, vec![1.5, -2.0]);
    }

    #[test]
    fn cuda_load_failure_after_selection_records_cpu_fallback() {
        let dir = tempfile::tempdir().expect("tempdir");
        let model_path = dir.path().join("identity_f32.onnx");
        fs::write(&model_path, identity_model(1)).expect("write model");
        let selection = ProviderSelection::from_strings(&["cuda".into(), "cpu".into()]);
        assert_eq!(
            selection
                .order
                .iter()
                .map(|provider| provider.kind)
                .collect::<Vec<_>>(),
            [ProviderKind::Cuda, ProviderKind::Cpu]
        );
        let mut cuda_attempted = false;

        let session = OrtSession::load_with_provider_loaders(
            &model_path,
            selection,
            None,
            RealSession::load_cpu,
            |_path, _provider| {
                cuda_attempted = true;
                Err(InfraError::Backend(
                    "forced CUDA session creation failure".to_string(),
                ))
            },
        )
        .expect("CPU fallback after forced CUDA failure");

        assert!(cuda_attempted, "CUDA loader must be attempted first");
        assert_eq!(session.provider(), ProviderKind::Cpu);
        assert!(session.cpu_fallback_used());
    }

    #[test]
    fn tensorrt_then_cpu_uses_first_usable_provider() {
        let dir = tempfile::tempdir().expect("tempdir");
        let model_path = dir.path().join("identity_f32.onnx");
        fs::write(&model_path, identity_model(1)).expect("write model");

        let session = OrtSession::load(
            &model_path,
            ProviderSelection {
                order: vec![ProviderOptions::trt(Some(0)), ProviderOptions::cpu()],
                fallback_to_cpu: true,
            },
        )
        .expect("load identity model");

        assert_requested_provider_or_cpu_fallback(&session, ProviderKind::Trt);
    }

    #[test]
    fn generated_i64_identity_runs_on_cpu() {
        let dir = tempfile::tempdir().expect("tempdir");
        let model_path = dir.path().join("identity_i64.onnx");
        fs::write(&model_path, identity_model(7)).expect("write model");

        let mut session = OrtSession::load(&model_path, ProviderSelection::default())
            .expect("load identity model");
        assert_eq!(session.inputs()[0].element_type, TensorElement::I64);

        let outputs = session
            .run_tensors(&[OrtTensorInput {
                name: "x".to_string(),
                shape: vec![2],
                data: OrtTensorData::I64(vec![4, 9]),
            }])
            .expect("run identity");

        assert_eq!(outputs[0].name, "y");
        assert_eq!(outputs[0].data, OrtTensorData::I64(vec![4, 9]));
    }

    #[test]
    fn generated_f32_identity_documents_zero_length_dimension_gate() {
        let dir = tempfile::tempdir().expect("tempdir");
        let model_path = dir.path().join("identity_zero_dim.onnx");
        fs::write(&model_path, identity_model_with_shape(1, &[1, 20, 0, 64])).expect("write model");

        let mut session = OrtSession::load(&model_path, ProviderSelection::default())
            .expect("load identity model");
        let error = session
            .run_tensors(&[OrtTensorInput {
                name: "x".to_string(),
                shape: vec![1, 20, 0, 64],
                data: OrtTensorData::F32(Vec::new()),
            }])
            .expect_err("the pinned ORT tensor constructor currently rejects zero dimensions");
        assert!(error.to_string().contains("all dimensions must be >= 1"));
    }

    #[test]
    fn generated_f16_identity_runs_on_cpu() {
        let dir = tempfile::tempdir().expect("tempdir");
        let model_path = dir.path().join("identity_f16.onnx");
        fs::write(&model_path, identity_model(10)).expect("write model");

        let mut session = OrtSession::load(&model_path, ProviderSelection::default())
            .expect("load identity model");
        assert_eq!(session.inputs()[0].element_type, TensorElement::F16);

        let data = vec![half::f16::from_f32(1.5), half::f16::from_f32(-2.0)];
        let outputs = session
            .run_tensors(&[OrtTensorInput {
                name: "x".to_string(),
                shape: vec![2],
                data: OrtTensorData::F16(data.clone()),
            }])
            .expect("run identity");

        assert_eq!(outputs[0].name, "y");
        assert_eq!(outputs[0].data, OrtTensorData::F16(data));
    }

    fn assert_requested_provider_or_cpu_fallback(session: &OrtSession, requested: ProviderKind) {
        let availability = probe_runtime_execution_provider_availability();
        let selected = session.provider();
        if availability.is_available(requested) {
            assert_eq!(
                selected, requested,
                "runtime probe reported {requested:?} available, but session selected {selected:?}"
            );
            assert!(
                !session.cpu_fallback_used(),
                "runtime probe reported {requested:?} available, so CPU fallback should not be used"
            );
        } else {
            assert_eq!(
                selected, ProviderKind::Cpu,
                "runtime probe reported {requested:?} unavailable, so session should fall back to CPU"
            );
            assert!(
                session.cpu_fallback_used(),
                "runtime probe reported {requested:?} unavailable, so CPU fallback should be recorded"
            );
        }
    }

    #[test]
    fn generated_i8_i16_i32_identities_run_on_cpu() {
        for (name, elem_type, data) in [
            ("i8", 3, OrtTensorData::I8(vec![-4, 9])),
            ("i16", 5, OrtTensorData::I16(vec![-1024, 2048])),
            ("i32", 6, OrtTensorData::I32(vec![-65_536, 65_537])),
        ] {
            let dir = tempfile::tempdir().expect("tempdir");
            let model_path = dir.path().join(format!("identity_{name}.onnx"));
            fs::write(&model_path, identity_model(elem_type)).expect("write model");

            let mut session = OrtSession::load(&model_path, ProviderSelection::default())
                .expect("load identity model");
            assert_eq!(session.inputs()[0].element_type, data.element_type());

            let outputs = session
                .run_tensors(&[OrtTensorInput {
                    name: "x".to_string(),
                    shape: vec![2],
                    data: data.clone(),
                }])
                .expect("run identity");

            assert_eq!(outputs[0].name, "y");
            assert_eq!(outputs[0].data, data);
        }
    }

    #[test]
    fn input_mismatch_reports_available_and_requested_names() {
        let dir = tempfile::tempdir().expect("tempdir");
        let model_path = dir.path().join("identity_f32.onnx");
        fs::write(&model_path, identity_model(1)).expect("write model");
        let mut session = OrtSession::load(&model_path, ProviderSelection::default())
            .expect("load identity model");

        let err = session
            .run_f32(&[OrtInput {
                name: "wrong".to_string(),
                shape: vec![2],
                data: vec![1.0, 2.0],
            }])
            .expect_err("wrong input name");
        let msg = err.to_string();
        assert!(msg.contains("available inputs: x"), "{msg}");
        assert!(msg.contains("requested inputs: wrong"), "{msg}");
        assert!(msg.contains("available outputs: y"), "{msg}");
    }

    #[test]
    fn combined_input_names_reject_duplicates_before_native_binding() {
        let metadata = SessionMetadata {
            inputs: vec![
                TensorMetadata {
                    name: "host".to_string(),
                    element_type: TensorElement::F32,
                    shape: vec![1],
                    dimension_symbols: vec![None],
                },
                TensorMetadata {
                    name: "resident".to_string(),
                    element_type: TensorElement::F32,
                    shape: vec![1],
                    dimension_symbols: vec![None],
                },
            ],
            outputs: Vec::new(),
        };
        assert!(validate_requested_input_names(&metadata, ["host", "resident"]).is_ok());
        for names in [
            vec!["host", "host"],
            vec!["host", "resident", "resident"],
            vec!["host", "resident", "host"],
        ] {
            let err = validate_requested_input_names(&metadata, names).unwrap_err();
            assert!(err.to_string().contains("duplicate input name"));
        }
    }

    fn identity_model(elem_type: u64) -> Vec<u8> {
        identity_model_with_shape(elem_type, &[2])
    }

    fn identity_model_with_shape(elem_type: u64, dimensions: &[u64]) -> Vec<u8> {
        let node = message(|node| {
            string_field(node, 1, "x");
            string_field(node, 2, "y");
            string_field(node, 4, "Identity");
        });
        let input = value_info_with_shape("x", elem_type, dimensions);
        let output = value_info_with_shape("y", elem_type, dimensions);
        let graph = message(|graph| {
            bytes_field(graph, 1, &node);
            string_field(graph, 2, "identity_graph");
            bytes_field(graph, 11, &input);
            bytes_field(graph, 12, &output);
        });
        let opset = message(|opset| {
            varint_field(opset, 2, 13);
        });
        message(|model| {
            varint_field(model, 1, 8);
            string_field(model, 2, "local-backend-ort-test");
            bytes_field(model, 7, &graph);
            bytes_field(model, 8, &opset);
        })
    }

    fn value_info_with_shape(name: &str, elem_type: u64, dimensions: &[u64]) -> Vec<u8> {
        let shape = message(|shape| {
            for dimension in dimensions {
                let dim = message(|dim| varint_field(dim, 1, *dimension));
                bytes_field(shape, 1, &dim);
            }
        });
        let tensor_type = message(|tensor| {
            varint_field(tensor, 1, elem_type);
            bytes_field(tensor, 2, &shape);
        });
        let ty = message(|ty| bytes_field(ty, 1, &tensor_type));
        message(|value| {
            string_field(value, 1, name);
            bytes_field(value, 2, &ty);
        })
    }

    fn message(write: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
        let mut bytes = Vec::new();
        write(&mut bytes);
        bytes
    }

    fn varint_field(bytes: &mut Vec<u8>, field: u64, value: u64) {
        varint(bytes, field << 3);
        varint(bytes, value);
    }

    fn string_field(bytes: &mut Vec<u8>, field: u64, value: &str) {
        bytes_field(bytes, field, value.as_bytes());
    }

    fn bytes_field(bytes: &mut Vec<u8>, field: u64, value: &[u8]) {
        varint(bytes, (field << 3) | 2);
        varint(bytes, value.len() as u64);
        bytes.extend_from_slice(value);
    }

    fn varint(bytes: &mut Vec<u8>, mut value: u64) {
        while value >= 0x80 {
            bytes.push((value as u8) | 0x80);
            value >>= 7;
        }
        bytes.push(value as u8);
    }
}
