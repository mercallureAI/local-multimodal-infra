//! ONNX Runtime adapter for `intfloat/multilingual-e5-small`.

use local_backend_ort::{
    OrtBackend, OrtSession, OrtTensorData, OrtTensorInput, OrtTensorOutput, PinnedCudaIoBinding,
    ProviderKind, ProviderSelection, SessionProviderReport,
};
use local_core::{EmbeddingInputType, InferenceOutput, ModelSpec};
use local_error::{InfraError, Result};
use std::path::{Path, PathBuf};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

const MAX_LENGTH: usize = 512;

#[derive(Debug, Clone)]
pub struct E5Artifacts {
    pub root: PathBuf,
    pub model: PathBuf,
    pub tokenizer: PathBuf,
}

impl E5Artifacts {
    pub fn from_spec(spec: &ModelSpec) -> Result<Self> {
        let root = artifact_root(spec)?;
        let gpu_preferred = spec
            .runtime
            .provider_order
            .first()
            .is_some_and(|provider| !provider.eq_ignore_ascii_case("cpu"));
        let model_candidates = model_candidates(gpu_preferred);
        let model = first_existing(&root, model_candidates).ok_or_else(|| {
            InfraError::ModelNotConfigured {
                model_id: spec.id.clone(),
                reason: format!(
                    "multilingual-e5-small ONNX graph is missing below {}",
                    root.display()
                ),
            }
        })?;
        let tokenizer = first_existing(&root, &["tokenizer.json", "onnx/tokenizer.json"])
            .ok_or_else(|| InfraError::ModelNotConfigured {
                model_id: spec.id.clone(),
                reason: format!("tokenizer.json is missing below {}", root.display()),
            })?;
        Ok(Self {
            root,
            model,
            tokenizer,
        })
    }
}

fn model_candidates(gpu_preferred: bool) -> &'static [&'static str] {
    if gpu_preferred {
        &[
            "onnx/model_O4_pooled.onnx",
            "model_O4_pooled.onnx",
            "onnx/model_O4.onnx",
            "model_O4.onnx",
            "onnx/model.onnx",
            "model.onnx",
        ]
    } else {
        &[
            "onnx/model_qint8_avx512_vnni_pooled.onnx",
            "model_qint8_avx512_vnni_pooled.onnx",
            "onnx/model_qint8_avx512_vnni.onnx",
            "model_qint8_avx512_vnni.onnx",
            "onnx/model_O4.onnx",
            "model_O4.onnx",
            "onnx/model.onnx",
            "model.onnx",
        ]
    }
}

#[derive(Debug)]
pub struct E5EmbeddingAdapter {
    model_id: String,
    artifacts: E5Artifacts,
    tokenizer: Tokenizer,
    session: OrtSession,
    graph_has_pooling: bool,
    pinned_cuda_binding: Option<PinnedCudaIoBinding>,
}

impl E5EmbeddingAdapter {
    pub fn load(spec: &ModelSpec) -> Result<Self> {
        let artifacts = E5Artifacts::from_spec(spec)?;
        let mut tokenizer = Tokenizer::from_file(&artifacts.tokenizer).map_err(|err| {
            InfraError::Adapter(format!(
                "load E5 tokenizer {}: {err}",
                artifacts.tokenizer.display()
            ))
        })?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_LENGTH,
                ..TruncationParams::default()
            }))
            .map_err(|err| InfraError::Adapter(format!("configure E5 truncation: {err}")))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..PaddingParams::default()
        }));
        let backend = OrtBackend::new(ProviderSelection::from_strings(
            &spec.runtime.provider_order,
        ));
        let session = backend.load_session(&artifacts.model)?;
        let graph_has_pooling = session
            .outputs()
            .iter()
            .any(|output| output.name == "sentence_embedding" && output.shape.len() == 2);
        Ok(Self {
            model_id: spec.id.clone(),
            artifacts,
            tokenizer,
            session,
            graph_has_pooling,
            pinned_cuda_binding: None,
        })
    }

    pub fn embed(
        &mut self,
        texts: &[String],
        input_type: EmbeddingInputType,
    ) -> Result<InferenceOutput> {
        if texts.is_empty() {
            return Err(InfraError::BadRequest(
                "text.embed requires at least one input string".to_string(),
            ));
        }
        if texts.iter().any(|text| text.trim().is_empty()) {
            return Err(InfraError::BadRequest(
                "text.embed inputs must not be empty".to_string(),
            ));
        }
        let prefixed = texts
            .iter()
            .map(|text| e5_text(text, input_type))
            .collect::<Vec<_>>();
        let encodings = self
            .tokenizer
            .encode_batch(prefixed, true)
            .map_err(|err| InfraError::Adapter(format!("E5 tokenization failed: {err}")))?;
        let prompt_tokens = encodings
            .iter()
            .flat_map(|encoding| encoding.get_attention_mask())
            .map(|&mask| mask as usize)
            .sum();
        let inputs = token_inputs(&self.session, &encodings)?;
        let outputs = if self.graph_has_pooling && self.session.provider() == ProviderKind::Cuda {
            let input_shape = inputs
                .first()
                .map(|input| input.shape.clone())
                .ok_or_else(|| InfraError::Adapter("E5 produced no ONNX inputs".to_string()))?;
            let output_shape = vec![texts.len(), 384];
            let needs_binding = self
                .pinned_cuda_binding
                .as_ref()
                .is_none_or(|binding| !binding.matches_shapes(&input_shape, &output_shape));
            if needs_binding {
                let binding = self.session.create_pinned_cuda_binding(
                    self.session.device_id().unwrap_or(0),
                    &input_shape,
                    "sentence_embedding",
                    &output_shape,
                )?;
                tracing::info!(
                    model_id = self.model_id,
                    batch = texts.len(),
                    sequence = input_shape[1],
                    device_id = binding.device_id(),
                    "E5 CUDA pinned I/O binding created"
                );
                self.pinned_cuda_binding = Some(binding);
            }
            self.session.run_pinned_cuda_binding(
                self.pinned_cuda_binding
                    .as_mut()
                    .expect("binding was created above"),
                &inputs,
            )?
        } else {
            self.session.run_tensors(&inputs)?
        };
        let embeddings = if self.graph_has_pooling {
            pooled_embeddings(outputs, texts.len())?
        } else {
            let hidden = float_output(outputs, "last_hidden_state")?;
            average_pool_and_normalize(
                &hidden.data,
                &hidden.shape,
                &encodings
                    .iter()
                    .map(|encoding| encoding.get_attention_mask().to_vec())
                    .collect::<Vec<_>>(),
            )?
        };
        Ok(InferenceOutput::TextEmbeddings {
            embeddings,
            prompt_tokens,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn artifacts(&self) -> &E5Artifacts {
        &self.artifacts
    }

    pub fn provider_report(&self) -> SessionProviderReport {
        self.session.provider_report()
    }

    pub fn graph_has_pooling(&self) -> bool {
        self.graph_has_pooling
    }

    pub fn pinned_cuda_io_enabled(&self) -> bool {
        self.graph_has_pooling && self.session.provider() == ProviderKind::Cuda
    }
}

#[derive(Debug)]
struct FloatOutput {
    shape: Vec<usize>,
    data: Vec<f32>,
}

fn e5_text(text: &str, input_type: EmbeddingInputType) -> String {
    match input_type {
        EmbeddingInputType::Query => format!("query: {text}"),
        EmbeddingInputType::Passage => format!("passage: {text}"),
    }
}

fn artifact_root(spec: &ModelSpec) -> Result<PathBuf> {
    let path = spec
        .artifacts
        .first()
        .map(|artifact| artifact.path.clone())
        .ok_or_else(|| InfraError::ModelNotConfigured {
            model_id: spec.id.clone(),
            reason: "E5 artifact root is not configured".to_string(),
        })?;
    Ok(if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or(Path::new(".")).to_path_buf()
    })
}

fn first_existing(root: &Path, candidates: &[&str]) -> Option<PathBuf> {
    candidates
        .iter()
        .map(|candidate| root.join(candidate))
        .find(|path| path.is_file())
}

fn token_inputs(
    session: &OrtSession,
    encodings: &[tokenizers::Encoding],
) -> Result<Vec<OrtTensorInput>> {
    let batch = encodings.len();
    let sequence = encodings.first().map(|e| e.len()).unwrap_or(0);
    if sequence == 0 || encodings.iter().any(|encoding| encoding.len() != sequence) {
        return Err(InfraError::Adapter(
            "E5 tokenizer produced an invalid padded batch".to_string(),
        ));
    }
    let mut inputs = Vec::new();
    for metadata in session.inputs() {
        let data = match metadata.name.as_str() {
            "input_ids" => encodings
                .iter()
                .flat_map(|encoding| encoding.get_ids())
                .map(|&value| value as i64)
                .collect(),
            "attention_mask" => encodings
                .iter()
                .flat_map(|encoding| encoding.get_attention_mask())
                .map(|&value| value as i64)
                .collect(),
            "token_type_ids" => encodings
                .iter()
                .flat_map(|encoding| encoding.get_type_ids())
                .map(|&value| value as i64)
                .collect(),
            other => {
                return Err(InfraError::NeedImplementation(format!(
                    "unsupported E5 ONNX input `{other}`"
                )))
            }
        };
        inputs.push(OrtTensorInput {
            name: metadata.name.clone(),
            shape: vec![batch, sequence],
            data: OrtTensorData::I64(data),
        });
    }
    Ok(inputs)
}

fn float_output(outputs: Vec<OrtTensorOutput>, preferred_name: &str) -> Result<FloatOutput> {
    let output = outputs
        .iter()
        .find(|output| output.name == preferred_name)
        .or_else(|| outputs.iter().find(|output| output.shape.len() == 3))
        .ok_or_else(|| {
            InfraError::Adapter(format!("E5 output `{preferred_name}` was not produced"))
        })?;
    let data = match &output.data {
        OrtTensorData::F32(data) => data.clone(),
        OrtTensorData::F16(data) => data.iter().map(|value| value.to_f32()).collect(),
        other => {
            return Err(InfraError::Adapter(format!(
                "E5 output `{}` has unsupported type {:?}",
                output.name,
                other.element_type()
            )))
        }
    };
    Ok(FloatOutput {
        shape: output.shape.clone(),
        data,
    })
}

fn pooled_embeddings(
    outputs: Vec<OrtTensorOutput>,
    expected_batch: usize,
) -> Result<Vec<Vec<f32>>> {
    let output = outputs
        .iter()
        .find(|output| output.name == "sentence_embedding")
        .or_else(|| outputs.iter().find(|output| output.shape.len() == 2))
        .ok_or_else(|| {
            InfraError::Adapter("E5 pooled graph did not produce sentence_embedding".to_string())
        })?;
    if output.shape.len() != 2 || output.shape[0] != expected_batch || output.shape[1] == 0 {
        return Err(InfraError::Adapter(format!(
            "E5 sentence_embedding has unsupported shape {:?}",
            output.shape
        )));
    }
    let data = match &output.data {
        OrtTensorData::F32(data) => data.clone(),
        OrtTensorData::F16(data) => data.iter().map(|value| value.to_f32()).collect(),
        other => {
            return Err(InfraError::Adapter(format!(
                "E5 sentence_embedding has unsupported type {:?}",
                other.element_type()
            )))
        }
    };
    let dimension = output.shape[1];
    if data.len() != expected_batch * dimension {
        return Err(InfraError::Adapter(format!(
            "E5 sentence_embedding data length {} does not match shape {:?}",
            data.len(),
            output.shape
        )));
    }
    Ok(data
        .chunks_exact(dimension)
        .map(|embedding| embedding.to_vec())
        .collect())
}

fn average_pool_and_normalize(
    hidden: &[f32],
    shape: &[usize],
    masks: &[Vec<u32>],
) -> Result<Vec<Vec<f32>>> {
    if shape.len() != 3 || shape[0] != masks.len() {
        return Err(InfraError::Adapter(format!(
            "E5 last_hidden_state has unsupported shape {shape:?}"
        )));
    }
    let (batch, sequence, dimension) = (shape[0], shape[1], shape[2]);
    if hidden.len() != batch * sequence * dimension
        || masks.iter().any(|mask| mask.len() != sequence)
    {
        return Err(InfraError::Adapter(
            "E5 output and attention-mask shapes do not match".to_string(),
        ));
    }
    let mut embeddings = Vec::with_capacity(batch);
    for (row, mask) in masks.iter().enumerate() {
        let token_count = mask.iter().filter(|&&value| value != 0).count();
        if token_count == 0 {
            return Err(InfraError::Adapter(
                "E5 attention mask contains no input tokens".to_string(),
            ));
        }
        let mut embedding = vec![0.0; dimension];
        for (token, &mask_value) in mask.iter().enumerate() {
            if mask_value == 0 {
                continue;
            }
            let offset = (row * sequence + token) * dimension;
            for dim in 0..dimension {
                embedding[dim] += hidden[offset + dim] / token_count as f32;
            }
        }
        let norm = embedding
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        if norm > 0.0 {
            for value in &mut embedding {
                *value /= norm;
            }
        }
        embeddings.push(embedding);
    }
    Ok(embeddings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applies_e5_query_and_passage_prefixes() {
        assert_eq!(e5_text("你好", EmbeddingInputType::Query), "query: 你好");
        assert_eq!(
            e5_text("document", EmbeddingInputType::Passage),
            "passage: document"
        );
    }

    #[test]
    fn masked_average_pooling_is_l2_normalized() {
        let embeddings = average_pool_and_normalize(
            &[1.0, 0.0, 3.0, 4.0, 100.0, 100.0],
            &[1, 3, 2],
            &[vec![1, 1, 0]],
        )
        .expect("pool");
        let expected = [1.0 / 2.0_f32.sqrt(), 1.0 / 2.0_f32.sqrt()];
        assert!((embeddings[0][0] - expected[0]).abs() < 1e-6);
        assert!((embeddings[0][1] - expected[1]).abs() < 1e-6);
    }

    #[test]
    fn cpu_prefers_qint8_and_gpu_prefers_o4() {
        assert!(model_candidates(false)[0].contains("qint8"));
        assert!(model_candidates(false)[0].contains("pooled"));
        assert!(model_candidates(true)[0].contains("model_O4"));
        assert!(model_candidates(true)[0].contains("pooled"));
    }

    #[test]
    fn accepts_pooled_sentence_embeddings() {
        let embeddings = pooled_embeddings(
            vec![OrtTensorOutput {
                name: "sentence_embedding".to_string(),
                shape: vec![2, 2],
                data: OrtTensorData::F32(vec![1.0, 2.0, 3.0, 4.0]),
            }],
            2,
        )
        .expect("pooled embeddings");
        assert_eq!(embeddings, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    }
}
