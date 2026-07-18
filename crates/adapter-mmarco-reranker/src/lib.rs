//! ONNX Runtime adapter for `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.

use local_backend_ort::{
    OrtBackend, OrtSession, OrtTensorData, OrtTensorInput, OrtTensorOutput, ProviderSelection,
    SessionProviderReport,
};
use local_core::{InferenceOutput, ModelSpec, RerankResult};
use local_error::{InfraError, Result};
use std::path::{Path, PathBuf};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

const MAX_LENGTH: usize = 512;

#[derive(Debug, Clone)]
pub struct MmarcoArtifacts {
    pub root: PathBuf,
    pub model: PathBuf,
    pub tokenizer: PathBuf,
}

impl MmarcoArtifacts {
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
                    "mMARCO MiniLM ONNX graph is missing below {}",
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
            "onnx/model_O4.onnx",
            "model_O4.onnx",
            "onnx/model.onnx",
            "model.onnx",
        ]
    } else {
        &[
            "onnx/model_quint8_avx2.onnx",
            "model_quint8_avx2.onnx",
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
pub struct MmarcoRerankerAdapter {
    model_id: String,
    artifacts: MmarcoArtifacts,
    tokenizer: Tokenizer,
    session: OrtSession,
}

impl MmarcoRerankerAdapter {
    pub fn load(spec: &ModelSpec) -> Result<Self> {
        let artifacts = MmarcoArtifacts::from_spec(spec)?;
        let mut tokenizer = Tokenizer::from_file(&artifacts.tokenizer).map_err(|err| {
            InfraError::Adapter(format!(
                "load mMARCO tokenizer {}: {err}",
                artifacts.tokenizer.display()
            ))
        })?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_LENGTH,
                ..TruncationParams::default()
            }))
            .map_err(|err| InfraError::Adapter(format!("configure mMARCO truncation: {err}")))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..PaddingParams::default()
        }));
        let backend = OrtBackend::new(ProviderSelection::from_strings(
            &spec.runtime.provider_order,
        ));
        let session = backend.load_session(&artifacts.model)?;
        Ok(Self {
            model_id: spec.id.clone(),
            artifacts,
            tokenizer,
            session,
        })
    }

    pub fn rerank(
        &mut self,
        query: &str,
        documents: &[String],
        top_n: Option<usize>,
    ) -> Result<InferenceOutput> {
        if query.trim().is_empty() {
            return Err(InfraError::BadRequest(
                "text.rerank query must not be empty".to_string(),
            ));
        }
        if documents.is_empty() || documents.iter().any(|document| document.trim().is_empty()) {
            return Err(InfraError::BadRequest(
                "text.rerank requires non-empty documents".to_string(),
            ));
        }
        let pairs = documents
            .iter()
            .map(|document| (query, document.as_str()).into())
            .collect::<Vec<tokenizers::EncodeInput<'_>>>();
        let encodings = self
            .tokenizer
            .encode_batch(pairs, true)
            .map_err(|err| InfraError::Adapter(format!("mMARCO tokenization failed: {err}")))?;
        let total_tokens = encodings
            .iter()
            .flat_map(|encoding| encoding.get_attention_mask())
            .map(|&mask| mask as usize)
            .sum();
        let inputs = token_inputs(&self.session, &encodings)?;
        let outputs = self.session.run_tensors(&inputs)?;
        let logits = logits(outputs, documents.len())?;
        let results = ranked_results(logits, documents, top_n);
        Ok(InferenceOutput::TextRerank {
            results,
            total_tokens,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn artifacts(&self) -> &MmarcoArtifacts {
        &self.artifacts
    }

    pub fn provider_report(&self) -> SessionProviderReport {
        self.session.provider_report()
    }
}

fn artifact_root(spec: &ModelSpec) -> Result<PathBuf> {
    let path = spec
        .artifacts
        .first()
        .map(|artifact| artifact.path.clone())
        .ok_or_else(|| InfraError::ModelNotConfigured {
            model_id: spec.id.clone(),
            reason: "mMARCO artifact root is not configured".to_string(),
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
            "mMARCO tokenizer produced an invalid padded batch".to_string(),
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
                    "unsupported mMARCO ONNX input `{other}`"
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

fn logits(outputs: Vec<OrtTensorOutput>, expected: usize) -> Result<Vec<f32>> {
    let output = outputs
        .iter()
        .find(|output| output.name == "logits")
        .or_else(|| outputs.first())
        .ok_or_else(|| InfraError::Adapter("mMARCO produced no output".to_string()))?;
    let values = match &output.data {
        OrtTensorData::F32(values) => values.clone(),
        OrtTensorData::F16(values) => values.iter().map(|value| value.to_f32()).collect(),
        other => {
            return Err(InfraError::Adapter(format!(
                "mMARCO output `{}` has unsupported type {:?}",
                output.name,
                other.element_type()
            )))
        }
    };
    if values.len() != expected {
        return Err(InfraError::Adapter(format!(
            "mMARCO logits shape {:?} contains {} scores for {expected} documents",
            output.shape,
            values.len()
        )));
    }
    Ok(values)
}

fn ranked_results(
    logits: Vec<f32>,
    documents: &[String],
    top_n: Option<usize>,
) -> Vec<RerankResult> {
    let mut results = logits
        .into_iter()
        .enumerate()
        .map(|(index, logit)| RerankResult {
            index,
            relevance_score: sigmoid(logit),
            document: documents[index].clone(),
        })
        .collect::<Vec<_>>();
    results.sort_by(|left, right| {
        right
            .relevance_score
            .total_cmp(&left.relevance_score)
            .then_with(|| left.index.cmp(&right.index))
    });
    let limit = top_n
        .filter(|&limit| limit > 0)
        .unwrap_or(documents.len())
        .min(documents.len());
    results.truncate(limit);
    results
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_by_activated_score_and_preserves_original_index() {
        let documents = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let results = ranked_results(vec![-2.0, 3.0, 1.0], &documents, Some(2));
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].index, 1);
        assert_eq!(results[0].document, "b");
        assert_eq!(results[1].index, 2);
        assert!(results[0].relevance_score > results[1].relevance_score);
    }

    #[test]
    fn sigmoid_is_stable_for_large_values() {
        assert_eq!(sigmoid(1000.0), 1.0);
        assert_eq!(sigmoid(-1000.0), 0.0);
    }

    #[test]
    fn top_n_zero_matches_vllm_default_all_semantics() {
        let documents = vec!["a".to_string(), "b".to_string()];
        assert_eq!(ranked_results(vec![0.0, 1.0], &documents, Some(0)).len(), 2);
    }

    #[test]
    fn cpu_prefers_avx2_uint8_and_gpu_prefers_o4() {
        assert!(model_candidates(false)[0].contains("quint8_avx2"));
        assert!(model_candidates(true)[0].contains("model_O4"));
    }
}
