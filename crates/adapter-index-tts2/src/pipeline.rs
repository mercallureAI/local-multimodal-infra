//! IndexTTS-2.5 synthesis over the packaged ONNX graph set.
//!
//! Per request: reference preprocessing (cached per reference file) ->
//! conditioning -> for each text segment { GPT prefill + in-graph sampled
//! decode loop with a device-resident f16 KV cache -> synthesis conditioning ->
//! 25-step CFM -> BigVGAN decoder } -> segments joined with short silences.

use crate::{
    artifacts::{IndexTts2Artifacts, PackageRuntime},
    audio,
    frontend,
    params::SynthesisParams,
    tokenizer::MultilingualTokenizer,
};
use local_backend_ort::{
    DeviceBinding, DeviceTensor, OrtBackend, OrtSession, OrtTensorData, OrtTensorInput,
    OrtTensorOutput, ProviderSelection, SessionProviderReport, SharedInitializers,
};
use local_core::{FileRef, InferenceOutput, ModelSpec};
use local_error::{InfraError, Result};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
    time::{Instant, SystemTime},
};
use uuid::Uuid;

/// Upstream `_load_and_cut_audio(..., 15)`.
const MAX_REFERENCE_SECONDS: f32 = 15.0;
/// KV state names follow the upstream export: `in_key_{i}` / `out_key_{i}`.
const KV_LAYERS: usize = 24;

#[derive(Debug, Clone, Copy)]
pub struct IndexTts2ProviderReport {
    pub reference: SessionProviderReport,
    pub conditioning: SessionProviderReport,
    pub prefill: SessionProviderReport,
    pub decode: SessionProviderReport,
    pub synthesis: SessionProviderReport,
    pub cfm: SessionProviderReport,
    pub decoder: SessionProviderReport,
}

struct Graph {
    session: OrtSession,
    binding: DeviceBinding,
}

impl std::fmt::Debug for Graph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Graph")
            .field("model", &self.session.model_path())
            .field("provider", &self.session.provider())
            .finish()
    }
}

impl Graph {
    fn load(backend: &OrtBackend, path: &Path, device: &[&str]) -> Result<Self> {
        Self::load_with(backend, path, device, None)
    }

    fn load_with(
        backend: &OrtBackend,
        path: &Path,
        device: &[&str],
        shared: Option<&SharedInitializers>,
    ) -> Result<Self> {
        let session = match shared {
            Some(shared) => backend.load_session_with_initializers(path, shared)?,
            None => backend.load_session(path)?,
        };
        let host = session
            .outputs()
            .iter()
            .map(|output| output.name.as_str())
            .filter(|name| !device.contains(name))
            .collect::<Vec<_>>();
        let binding = session.create_device_binding(device, &host)?;
        Ok(Self { session, binding })
    }

    fn run(
        &mut self,
        host: Vec<OrtTensorInput>,
        device: &[(&str, &DeviceTensor)],
    ) -> Result<local_backend_ort::DeviceBindingOutputs> {
        self.session.run_device_binding(&mut self.binding, host, device)
    }
}

/// Reference-audio features reused across requests with the same voice.
#[derive(Debug)]
struct ReferenceState {
    key: (PathBuf, u64, Option<SystemTime>),
    speaker_features: DeviceTensor,
    speaker_frames: usize,
    style: DeviceTensor,
    reference_hidden: DeviceTensor,
    null_hidden: DeviceTensor,
}

#[derive(Debug)]
pub struct IndexTts2Adapter {
    model_id: String,
    artifacts: IndexTts2Artifacts,
    runtime: PackageRuntime,
    tokenizer: MultilingualTokenizer,
    reference: Graph,
    conditioning: Graph,
    prefill: Graph,
    decode: Graph,
    synthesis: Graph,
    cfm: Graph,
    decoder: Graph,
    kv_out: Vec<String>,
    kv_in: Vec<String>,
    cached_reference: Option<ReferenceState>,
    output_dir: PathBuf,
    /// Device weights the prefill/decode sessions reference. Declared last so
    /// it is dropped after those sessions.
    _shared_gpt: Option<SharedInitializers>,
}

impl IndexTts2Adapter {
    pub fn load(spec: &ModelSpec) -> Result<Self> {
        let started = Instant::now();
        let artifacts = IndexTts2Artifacts::load(IndexTts2Artifacts::resolve(spec))?;
        let manifest = &artifacts.manifest;
        // Initializers go straight to the device allocator instead of the BFC
        // arena, which would round each weight region up to a power of two.
        let mut backend =
            OrtBackend::new(ProviderSelection::from_strings(&spec.runtime.provider_order))
                .with_config_entry("session.use_device_allocator_for_initializers", "1");
        if !manifest.disabled_optimizers.is_empty() {
            backend = backend.with_config_entry(
                "optimization.disable_specified_optimizers",
                manifest.disabled_optimizers.join(";"),
            );
        }
        let kv_out = kv_names("out");
        let kv_in = kv_names("in");
        let kv_out_refs = kv_out.iter().map(String::as_str).collect::<Vec<_>>();
        let with_kv = |extra: &[&'static str]| {
            let mut names = kv_out_refs.clone();
            names.extend_from_slice(extra);
            names
        };
        let graphs = &manifest.graphs;
        let reference = Graph::load(
            &backend,
            &artifacts.graph(&graphs.reference_preprocess),
            &["semantic_features", "style", "reference_hidden", "null_hidden"],
        )?;
        let conditioning = Graph::load(
            &backend,
            &artifacts.graph(&graphs.conditioning),
            &["speaker_latent", "emotion_vector"],
        )?;
        let shared_gpt = match &manifest.device_shared_initializers {
            Some(shared) if !shared.tensors.is_empty() => {
                let data_file = shared.data_file.as_deref().ok_or_else(|| {
                    InfraError::Adapter("device_shared_initializers has no data_file".to_string())
                })?;
                let uploaded = backend.upload_initializers(&artifacts.graph(data_file), &shared.tensors)?;
                tracing::info!(
                    tensors = uploaded.len(),
                    mib = uploaded.bytes() / (1 << 20),
                    cuda_device = ?uploaded.cuda_device(),
                    "IndexTTS-2.5 GPT weights uploaded once for prefill and decode"
                );
                Some(uploaded)
            }
            _ => None,
        };
        let prefill = Graph::load_with(
            &backend,
            &artifacts.graph(&graphs.target_prefill_sampling),
            &with_kv(&["last_hidden_state"]),
            shared_gpt.as_ref(),
        )?;
        let decode = Graph::load_with(
            &backend,
            &artifacts.graph(&graphs.decode_step_sampling),
            &with_kv(&["last_hidden_state"]),
            shared_gpt.as_ref(),
        )?;
        let synthesis = Graph::load(
            &backend,
            &artifacts.graph(&graphs.synthesis),
            &["static_hidden", "cfg_scales", "cfg_scale_sum", "target_mask"],
        )?;
        let cfm = Graph::load(
            &backend,
            &artifacts.graph(&graphs.cfm_estimator),
            &["next_mel_features"],
        )?;
        let decoder = Graph::load(&backend, &artifacts.graph(&graphs.decoder), &[])?;
        let tokenizer = MultilingualTokenizer::load(&artifacts.root)?;
        let output_dir = env::var_os("LOCAL_DATA_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("workdir/data"));
        let adapter = Self {
            model_id: spec.id.clone(),
            runtime: manifest.runtime,
            artifacts,
            tokenizer,
            reference,
            conditioning,
            prefill,
            decode,
            synthesis,
            cfm,
            decoder,
            kv_out,
            kv_in,
            cached_reference: None,
            output_dir,
            _shared_gpt: shared_gpt,
        };
        tracing::info!(
            model_id = adapter.model_id,
            root = %adapter.artifacts.root.display(),
            precision = adapter.artifacts.manifest.precision,
            providers = ?adapter.provider_report(),
            load_ms = started.elapsed().as_millis() as u64,
            "IndexTTS-2.5 sessions loaded"
        );
        Ok(adapter)
    }

    pub fn provider_report(&self) -> IndexTts2ProviderReport {
        IndexTts2ProviderReport {
            reference: self.reference.session.provider_report(),
            conditioning: self.conditioning.session.provider_report(),
            prefill: self.prefill.session.provider_report(),
            decode: self.decode.session.provider_report(),
            synthesis: self.synthesis.session.provider_report(),
            cfm: self.cfm.session.provider_report(),
            decoder: self.decoder.session.provider_report(),
        }
    }

    pub fn artifacts(&self) -> &IndexTts2Artifacts {
        &self.artifacts
    }

    pub fn synthesize(
        &mut self,
        request_id: Uuid,
        text: &str,
        reference_audio: Option<&FileRef>,
        params: &BTreeMap<String, Value>,
    ) -> Result<InferenceOutput> {
        let started = Instant::now();
        let params = SynthesisParams::from_map(params, text)?;
        let reference_audio = reference_audio.ok_or_else(|| {
            InfraError::BadRequest("IndexTTS-2.5 synthesis requires reference_audio".to_string())
        })?;
        let reference_path = local_files::local_path(reference_audio)?;
        let (language, language_id) = self.tokenizer.language(&params.language);
        let max_segment_tokens = params
            .max_text_tokens_per_segment
            .min(self.runtime.max_text_tokens.saturating_sub(2));
        let segments = frontend::prepare_segments(
            &self.tokenizer,
            text,
            &language,
            max_segment_tokens,
            params.text_normalization,
        );
        if segments.iter().all(|segment| segment.text.trim().is_empty()) {
            return Err(InfraError::BadRequest(
                "IndexTTS-2.5 text is empty after normalization".to_string(),
            ));
        }

        let reference_started = Instant::now();
        self.ensure_reference(&reference_path)?;
        let reference_ms = reference_started.elapsed().as_millis() as u64;
        let (speaker_latent, emotion_vector) = self.run_conditioning(&params)?;

        let mut rng = Rng::new(params.seed);
        let mut audio = Vec::new();
        let silence =
            vec![0.0f32; (self.runtime.out_sample_rate as u64 * params.interval_silence_ms / 1000) as usize];
        let mut total_codes = 0usize;
        for (index, segment) in segments.iter().enumerate() {
            let segment_started = Instant::now();
            let (codes, accepted) =
                self.generate_codes(&segment.ids, language_id, &params, &speaker_latent, &emotion_vector)?;
            let generate_ms = segment_started.elapsed().as_millis() as u64;
            let waveform = self.render_segment(
                &segment.ids,
                codes,
                accepted,
                &params,
                &speaker_latent,
                &emotion_vector,
                &mut rng,
            )?;
            tracing::info!(
                request_id = %request_id,
                segment = index + 1,
                segments = segments.len(),
                text = segment.text,
                text_tokens = segment.ids.len(),
                codes = accepted,
                generate_ms,
                segment_ms = segment_started.elapsed().as_millis() as u64,
                audio_ms = waveform.len() as u64 * 1000 / self.runtime.out_sample_rate as u64,
                "IndexTTS-2.5 segment synthesized"
            );
            total_codes += accepted;
            if index > 0 {
                audio.extend_from_slice(&silence);
            }
            audio.extend(waveform);
        }
        let output = self.write_output(&audio)?;
        tracing::info!(
            request_id = %request_id,
            language,
            segments = segments.len(),
            codes = total_codes,
            reference_ms,
            audio_ms = audio.len() as u64 * 1000 / self.runtime.out_sample_rate as u64,
            total_ms = started.elapsed().as_millis() as u64,
            "IndexTTS-2.5 synthesis complete"
        );
        Ok(InferenceOutput::TtsAudio { audio: output })
    }

    fn ensure_reference(&mut self, path: &Path) -> Result<()> {
        let metadata = fs::metadata(path)
            .map_err(|e| InfraError::BadRequest(format!("reference audio {}: {e}", path.display())))?;
        let key = (path.to_path_buf(), metadata.len(), metadata.modified().ok());
        if self.cached_reference.as_ref().is_some_and(|state| state.key == key) {
            return Ok(());
        }
        self.cached_reference = None;
        let samples = audio::read_wav_mono(path, self.runtime.in_sample_rate, MAX_REFERENCE_SECONDS)?;
        // The in-graph 16 kHz fbank needs at least one 25 ms frame plus a shift.
        let minimum = (560 * self.runtime.in_sample_rate as usize).div_ceil(16_000);
        if samples.len() < minimum {
            return Err(InfraError::BadRequest(format!(
                "reference audio {} is too short ({} samples at {} Hz)",
                path.display(),
                samples.len(),
                self.runtime.in_sample_rate
            )));
        }
        let len = samples.len();
        let mut outputs = self.reference.run(
            vec![host_f32("audio", vec![1, 1, len], samples)],
            &[],
        )?;
        let speaker_features = outputs.take_device("semantic_features")?;
        let speaker_frames = *speaker_features.shape().get(1).ok_or_else(|| {
            InfraError::Backend("semantic_features has no frame axis".to_string())
        })?;
        self.cached_reference = Some(ReferenceState {
            key,
            speaker_features,
            speaker_frames,
            style: outputs.take_device("style")?,
            reference_hidden: outputs.take_device("reference_hidden")?,
            null_hidden: outputs.take_device("null_hidden")?,
        });
        Ok(())
    }

    /// Emotion from an explicit 8-way vector (scaled by `emotion_alpha`, as
    /// upstream) or, by default, from the speaker reference itself.
    fn run_conditioning(&mut self, params: &SynthesisParams) -> Result<(DeviceTensor, DeviceTensor)> {
        let state = self
            .cached_reference
            .as_ref()
            .ok_or_else(|| InfraError::Backend("IndexTTS-2.5 reference state missing".to_string()))?;
        let weights = match params.emotion_vector {
            Some(vector) => {
                let scale = params.emotion_alpha.clamp(0.0, 1.0);
                vector.map(|value| (value * scale * 10_000.0).trunc() / 10_000.0)
            }
            None => [0.0; 8],
        };
        let frames = state.speaker_frames as i64;
        let mut outputs = self.conditioning.run(
            vec![
                host_i64("speaker_lengths", vec![1], vec![frames]),
                host_i64("emotion_lengths", vec![1], vec![frames]),
                host_f32("emotion_alpha", vec![1], vec![1.0]),
                host_f32("emotion_weights", vec![8], weights.to_vec()),
            ],
            &[
                ("speaker_features", &state.speaker_features),
                ("emotion_features", &state.speaker_features),
                ("style", &state.style),
            ],
        )?;
        Ok((
            outputs.take_device("speaker_latent")?,
            outputs.take_device("emotion_vector")?,
        ))
    }

    /// GPT prefill plus the sampled decode loop. Returns the upstream
    /// `save_ids` buffer and the number of accepted (non-stop) codes.
    fn generate_codes(
        &mut self,
        text_ids: &[i32],
        language_id: i64,
        params: &SynthesisParams,
        speaker_latent: &DeviceTensor,
        emotion_vector: &DeviceTensor,
    ) -> Result<(OrtTensorOutput, usize)> {
        let controls = |repetition: bool| {
            let mut inputs = vec![
                host_f32("temperature", vec![1], vec![params.temperature]),
                host_i64("top_k", vec![1], vec![params.top_k.min(self.runtime.mel_code_size) as i64]),
                host_f32("top_p", vec![1], vec![params.top_p]),
            ];
            if repetition {
                inputs.push(host_f32("repetition_penalty", vec![1], vec![params.repetition_penalty]));
            }
            inputs
        };
        let mut host = controls(false);
        host.push(host_i32("text_ids", vec![1, text_ids.len()], text_ids.to_vec()));
        host.push(host_i64("language_id", vec![1], vec![language_id]));
        let mut outputs = self.prefill.run(
            host,
            &[("speaker_latent", speaker_latent), ("emotion_vector", emotion_vector)],
        )?;
        let mut kv = take_kv(&mut outputs, &self.kv_out)?;
        let mut token = outputs.take_host("next_token")?;
        let mut history = outputs.take_host("kv_sequence_length")?;
        let prefill_length = scalar_i64(&history)? as usize;
        let mut save_ids = OrtTensorOutput {
            name: "save_ids".to_string(),
            shape: vec![1, 1],
            data: token.data.clone(),
        };
        let max_tokens = params
            .max_mel_tokens
            .min(self.runtime.max_signal_length.saturating_sub(prefill_length));
        let mut accepted = 0usize;
        loop {
            if scalar_i32(&token)? == self.runtime.stop_mel_token {
                break;
            }
            accepted += 1;
            if accepted >= max_tokens {
                tracing::warn!(max_tokens, "IndexTTS-2.5 hit the mel token limit without a stop token");
                break;
            }
            let mut host = controls(true);
            host.push(renamed(token, "current_token"));
            host.push(renamed(save_ids, "save_ids_in"));
            host.push(renamed(history, "history_length"));
            let device = self
                .kv_in
                .iter()
                .zip(&kv)
                .map(|(name, tensor)| (name.as_str(), tensor))
                .collect::<Vec<_>>();
            let mut outputs = self.decode.run(host, &device)?;
            drop(device);
            kv = take_kv(&mut outputs, &self.kv_out)?;
            token = outputs.take_host("next_token")?;
            save_ids = outputs.take_host("save_ids_out")?;
            history = outputs.take_host("kv_sequence_length")?;
        }
        Ok((save_ids, accepted))
    }

    #[allow(clippy::too_many_arguments)]
    fn render_segment(
        &mut self,
        text_ids: &[i32],
        save_ids: OrtTensorOutput,
        accepted: usize,
        params: &SynthesisParams,
        speaker_latent: &DeviceTensor,
        emotion_vector: &DeviceTensor,
        rng: &mut Rng,
    ) -> Result<Vec<f32>> {
        if accepted == 0 {
            return Ok(Vec::new());
        }
        // Upstream pads the synthesis text with the stop-text token (id 1).
        let mut padded = text_ids.to_vec();
        padded.push(1);
        // Field-level borrow so `self.synthesis` can be borrowed mutably.
        let state = self
            .cached_reference
            .as_ref()
            .ok_or_else(|| InfraError::Backend("IndexTTS-2.5 reference state missing".to_string()))?;
        let mut outputs = self.synthesis.run(
            vec![
                host_i32("text_ids", vec![1, padded.len()], padded),
                renamed(save_ids, "save_ids"),
                host_i64("accepted_length", vec![1], vec![accepted as i64]),
                host_f32("cfg_rate", vec![1], vec![params.cfg_rate]),
                host_f32("duration_factor", vec![1], vec![params.duration_factor]),
            ],
            &[
                ("speaker_latent", speaker_latent),
                ("emotion_vector", emotion_vector),
                ("style", &state.style),
                ("reference_hidden", &state.reference_hidden),
                ("null_hidden", &state.null_hidden),
            ],
        )?;
        let static_hidden = outputs.take_device("static_hidden")?;
        let cfg_scales = outputs.take_device("cfg_scales")?;
        let cfg_scale_sum = outputs.take_device("cfg_scale_sum")?;
        let target_mask = outputs.take_device("target_mask")?;
        let target_length = outputs.take_host("target_length")?;
        let total_frames = *static_hidden.shape().get(1).ok_or_else(|| {
            InfraError::Backend("static_hidden has no frame axis".to_string())
        })?;

        let noise = (0..total_frames * 80)
            .map(|_| rng.normal() * params.diffusion_temperature)
            .collect::<Vec<_>>();
        let mut mel: Option<DeviceTensor> = None;
        for step in 0..self.runtime.cfm_steps {
            let mut host = vec![host_i64("step_index", vec![1], vec![step as i64])];
            let mut device = vec![
                ("static_hidden", &static_hidden),
                ("cfg_scales", &cfg_scales),
                ("cfg_scale_sum", &cfg_scale_sum),
                ("target_mask", &target_mask),
            ];
            match &mel {
                Some(current) => device.push(("mel_features", current)),
                None => host.push(host_f32("mel_features", vec![1, total_frames, 80], noise.clone())),
            }
            let mut outputs = self.cfm.run(host, &device)?;
            drop(device);
            mel = Some(outputs.take_device("next_mel_features")?);
        }
        let mel = mel.ok_or_else(|| InfraError::Backend("CFM ran zero steps".to_string()))?;
        let mut outputs = self.decoder.run(
            vec![renamed(target_length, "target_length")],
            &[("mel_features", &mel)],
        )?;
        match outputs.take_host("waveform")?.data {
            OrtTensorData::F32(samples) => Ok(samples),
            other => Err(InfraError::Backend(format!(
                "IndexTTS-2.5 decoder returned {:?}, expected f32",
                other.element_type()
            ))),
        }
    }

    fn write_output(&self, samples: &[f32]) -> Result<FileRef> {
        fs::create_dir_all(&self.output_dir)
            .map_err(|e| InfraError::io(Some(self.output_dir.clone()), e))?;
        let path = self.output_dir.join(format!("indextts2-{}.wav", Uuid::new_v4()));
        audio::write_wav_i16(&path, samples, self.runtime.out_sample_rate)?;
        let mut file = FileRef::local(path);
        file.mime = Some("audio/wav".to_string());
        Ok(file)
    }
}

fn kv_names(direction: &str) -> Vec<String> {
    (0..KV_LAYERS)
        .map(|index| format!("{direction}_key_{index}"))
        .chain((0..KV_LAYERS).map(|index| format!("{direction}_value_{index}")))
        .collect()
}

fn take_kv(
    outputs: &mut local_backend_ort::DeviceBindingOutputs,
    names: &[String],
) -> Result<Vec<DeviceTensor>> {
    names.iter().map(|name| outputs.take_device(name)).collect()
}

fn host_f32(name: &str, shape: Vec<usize>, data: Vec<f32>) -> OrtTensorInput {
    OrtTensorInput { name: name.to_string(), shape, data: OrtTensorData::F32(data) }
}

fn host_i32(name: &str, shape: Vec<usize>, data: Vec<i32>) -> OrtTensorInput {
    OrtTensorInput { name: name.to_string(), shape, data: OrtTensorData::I32(data) }
}

fn host_i64(name: &str, shape: Vec<usize>, data: Vec<i64>) -> OrtTensorInput {
    OrtTensorInput { name: name.to_string(), shape, data: OrtTensorData::I64(data) }
}

fn renamed(output: OrtTensorOutput, name: &str) -> OrtTensorInput {
    OrtTensorInput { name: name.to_string(), shape: output.shape, data: output.data }
}

fn scalar_i32(output: &OrtTensorOutput) -> Result<i32> {
    match &output.data {
        OrtTensorData::I32(values) if !values.is_empty() => Ok(values[0]),
        other => Err(InfraError::Backend(format!(
            "`{}` is {:?}, expected a non-empty i32 tensor",
            output.name,
            other.element_type()
        ))),
    }
}

fn scalar_i64(output: &OrtTensorOutput) -> Result<i64> {
    match &output.data {
        OrtTensorData::I64(values) if !values.is_empty() => Ok(values[0]),
        other => Err(InfraError::Backend(format!(
            "`{}` is {:?}, expected a non-empty i64 tensor",
            output.name,
            other.element_type()
        ))),
    }
}

/// SplitMix64 with Box-Muller normals for the CFM initial noise.
struct Rng {
    state: u64,
    spare: Option<f32>,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed, spare: None }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f32 {
        if let Some(value) = self.spare.take() {
            return value;
        }
        let radius = (-2.0 * self.unit().ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * self.unit();
        self.spare = Some((radius * angle.sin()) as f32);
        (radius * angle.cos()) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_names_match_upstream_layout() {
        let names = kv_names("in");
        assert_eq!(names.len(), 48);
        assert_eq!(names[0], "in_key_0");
        assert_eq!(names[23], "in_key_23");
        assert_eq!(names[24], "in_value_0");
    }

    #[test]
    fn normals_are_roughly_standard() {
        let mut rng = Rng::new(7);
        let values = (0..20_000).map(|_| rng.normal() as f64).collect::<Vec<_>>();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        assert!(mean.abs() < 0.03, "{mean}");
        assert!((var - 1.0).abs() < 0.05, "{var}");
    }

}

/// Opt-in end-to-end synthesis against a real package:
/// `LOCAL_INDEXTTS2_MODEL_DIR=<package> LOCAL_INDEXTTS2_REFERENCE=<wav>
///  cargo test -p local-adapter-index-tts2 --features cuda real_model -- --nocapture`
#[cfg(test)]
mod real_model {
    use super::*;
    use local_core::{
        AdapterKind, ArtifactKind, BackendKind, LoadPolicy, ModelArtifact, ResourceRequirement,
        RuntimePolicy, TaskKind,
    };
    use serde_json::json;

    #[test]
    fn real_model_synthesis_if_env_set() {
        let (Ok(model_dir), Ok(reference)) = (
            env::var("LOCAL_INDEXTTS2_MODEL_DIR"),
            env::var("LOCAL_INDEXTTS2_REFERENCE"),
        ) else {
            return;
        };
        let out_dir = tempfile::tempdir().expect("tempdir");
        env::set_var("LOCAL_DATA_DIR", out_dir.path());
        let spec = ModelSpec {
            id: "indextts-2.5-onnx".to_string(),
            name: "IndexTTS 2.5 test".to_string(),
            enabled: true,
            task_kinds: vec![TaskKind::TtsSynthesize],
            adapter: AdapterKind::IndexTts2,
            backend: BackendKind::Ort,
            artifacts: vec![ModelArtifact {
                kind: ArtifactKind::Local,
                path: PathBuf::from(model_dir),
                source_path: None,
                sha256: None,
                url: None,
                repo_id: None,
                revision: None,
                files: Vec::new(),
                allow_patterns: Vec::new(),
                metadata: BTreeMap::new(),
            }],
            runtime: RuntimePolicy {
                provider_order: vec!["cuda".to_string(), "cpu".to_string()],
                max_concurrency: 1,
                idle_ttl_sec: 60,
            },
            resources: ResourceRequirement::default(),
            load_policy: LoadPolicy::default(),
            metadata: BTreeMap::new(),
        };
        let load_started = Instant::now();
        let mut adapter = IndexTts2Adapter::load(&spec).expect("load IndexTTS-2.5 package");
        eprintln!("load {:?}; providers {:?}", load_started.elapsed(), adapter.provider_report());
        let reference = FileRef::local(PathBuf::from(reference));
        let cases = [
            ("zh", "大家好，我现在正在体验 IndexTTS 二点五的 Rust 推理。", json!({"seed": 9527})),
            ("zh-sad", "对不起嘛！我的记性真的不太好。", json!({"seed": 9527, "emotion_vector": {"sad": 0.8}})),
            ("en", "Hello! This sentence was synthesized by the Rust pipeline.", json!({"seed": 9527})),
            ("ja", "今日はいい天気ですね。一緒に公園へ散歩に行きませんか？", json!({"seed": 9527})),
            (
                "es",
                "Hola, ¿cómo estás? Hoy hace muy buen tiempo para pasear.",
                json!({"seed": 9527, "language": "es"}),
            ),
            // One segment near the 120-token budget: the VRAM worst case.
            (
                "zh-long",
                "春天来了，公园里的花都开了，红的像火，粉的像霞，白的像雪。小朋友们在草地上放风筝，老人们坐在长椅上聊天晒太阳，年轻人沿着湖边慢慢地跑步。微风吹过，柳枝轻轻摇摆，湖面泛起一圈圈涟漪，远处传来悠扬的歌声，让人觉得格外舒服和安心。",
                json!({"seed": 9527}),
            ),
        ];
        for (label, text, params) in cases {
            let params: BTreeMap<String, Value> = serde_json::from_value(params).unwrap();
            let started = Instant::now();
            let output = adapter
                .synthesize(Uuid::new_v4(), text, Some(&reference), &params)
                .expect("synthesize");
            let InferenceOutput::TtsAudio { audio } = output else {
                panic!("unexpected output");
            };
            let path = audio.path.expect("wav path");
            let reader = hound::WavReader::open(&path).expect("open output wav");
            let seconds = reader.duration() as f32 / reader.spec().sample_rate as f32;
            let elapsed = started.elapsed().as_secs_f32();
            eprintln!("{label}: {seconds:.2}s audio in {elapsed:.2}s (RTF {:.3})", elapsed / seconds);
            assert!(seconds > 0.5, "{label} produced {seconds}s of audio");
            if let Ok(keep) = env::var("LOCAL_INDEXTTS2_KEEP_DIR") {
                fs::copy(&path, Path::new(&keep).join(format!("rust_{label}.wav"))).expect("keep wav");
            }
        }
    }
}
