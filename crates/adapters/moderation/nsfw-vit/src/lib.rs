//! NSFW probability of an image with a ViT-family classifier.
//!
//! The graphs come from `scripts/local/nsfw_export.py`: `model.onnx` takes
//! `pixels` float32 `[N, 3, S, S]` in 0..1 (RGB, already resized) and returns
//! `probs` `[N, C]` (normalization and softmax are in the graph);
//! `nsfw_meta.json` gives `size`, `resize` (`squash`: stretch to S x S;
//! `center`: shorter side to S, then the centre square), `labels` and
//! `nsfw_labels`, whose probabilities add up to the NSFW probability.
//!
//! Provenance: `Freepik/nsfw_image_detector` (EVA02-base 448, MIT; levels
//! neutral / low / medium / high), `Marqo/nsfw-image-detection-384` (ViT-tiny,
//! Apache-2.0) and `Falconsai/nsfw_image_detection` (ViT-base 224,
//! Apache-2.0); see `configs/providers/moderation/`.

use image::{imageops, ImageReader, RgbImage};
use local_backend_ort::{
    CudaSessionOptions, OrtBackend, OrtOutput, OrtSession, OrtTensorData, OrtTensorInput,
    ProviderSelection, SessionProviderReport,
};
use local_core::{FileRef, InferenceOutput, LabelScore, ModelSpec};
use local_error::{InfraError, Result};
use serde::Deserialize;
use std::path::Path;

const MODEL_FILE: &str = "model.onnx";
const META_FILE: &str = "nsfw_meta.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Resize {
    /// Stretch to the input square (nothing cut off).
    Squash,
    /// Shorter side to the input size, then the centre square.
    Center,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NsfwMeta {
    pub size: u32,
    pub resize: Resize,
    pub labels: Vec<String>,
    pub nsfw_labels: Vec<String>,
}

#[derive(Debug)]
pub struct NsfwVitAdapter {
    model_id: String,
    session: OrtSession,
    input_name: String,
    meta: NsfwMeta,
    nsfw_indices: Vec<usize>,
    /// `cuda_release_memory_after_run`.
    release_memory: bool,
}

impl NsfwVitAdapter {
    pub fn load(spec: &ModelSpec) -> Result<Self> {
        let missing = |reason: String| InfraError::ModelNotConfigured {
            model_id: spec.id.clone(),
            reason,
        };
        let root = spec
            .artifacts
            .first()
            .map(|artifact| artifact.path.clone())
            .filter(|path| !path.as_os_str().is_empty())
            .ok_or_else(|| missing("the model has no artifact directory".to_string()))?;
        let model_path = root.join(MODEL_FILE);
        if !model_path.is_file() {
            return Err(missing(format!(
                "{MODEL_FILE} is missing below {}",
                root.display()
            )));
        }
        let mut meta = read_meta(&root.join(META_FILE)).map_err(|err| missing(err.to_string()))?;
        // The spec may count fewer (or other) labels as NSFW than the export:
        // Freepik's model without `low` (suggestive) passes swimwear.
        if let Some(labels) = spec.metadata.get("nsfw_labels") {
            meta.nsfw_labels = serde_json::from_value(labels.clone()).map_err(|err| {
                missing(format!("metadata.nsfw_labels must be label names: {err}"))
            })?;
        }
        let nsfw_indices = meta
            .nsfw_labels
            .iter()
            .map(|label| {
                meta.labels.iter().position(|l| l == label).ok_or_else(|| {
                    missing(format!("{META_FILE}: nsfw label `{label}` is not a label"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if meta.size == 0 || nsfw_indices.is_empty() {
            return Err(missing(format!("{META_FILE} needs a size and nsfw_labels")));
        }
        let backend = OrtBackend::new(ProviderSelection::from_strings(
            &spec.runtime.provider_order,
        ))
        .with_cuda_session_options(CudaSessionOptions::default())
        .with_cuda_memory_metadata(&spec.metadata);
        let session = backend.load_session(&model_path)?;
        let input_name = session
            .inputs()
            .first()
            .map(|input| input.name.clone())
            .ok_or_else(|| missing(format!("{MODEL_FILE} has no input")))?;
        tracing::info!(
            model_id = spec.id,
            provider = ?session.provider(),
            size = meta.size,
            labels = ?meta.labels,
            "NSFW classifier loaded"
        );
        Ok(Self {
            model_id: spec.id.clone(),
            session,
            input_name,
            meta,
            nsfw_indices,
            release_memory: local_backend_ort::release_memory_after_run(&spec.metadata),
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn provider_report(&self) -> SessionProviderReport {
        self.session.provider_report()
    }

    pub fn classify_file(&mut self, image: &FileRef) -> Result<InferenceOutput> {
        let path = local_files::local_path(image)?;
        let image = ImageReader::open(&path)
            .and_then(|reader| reader.with_guessed_format())
            .map_err(|err| InfraError::BadRequest(format!("cannot read the image: {err}")))?
            .decode()
            .map_err(|err| InfraError::BadRequest(format!("cannot decode the image: {err}")))?
            .to_rgb8();
        self.classify(&image)
    }

    pub fn classify(&mut self, image: &RgbImage) -> Result<InferenceOutput> {
        let size = self.meta.size;
        let input = OrtTensorInput {
            name: self.input_name.clone(),
            shape: vec![1, 3, size as usize, size as usize],
            data: OrtTensorData::F32(pixels(&resized(image, size, self.meta.resize))),
        };
        let outputs = if self.release_memory {
            self.session.run_tensors_releasing_memory(&[input])?
        } else {
            self.session.run_tensors(&[input])?
        };
        let output = outputs
            .into_iter()
            .next()
            .ok_or_else(|| InfraError::Backend("the model returned no output".to_string()))
            .and_then(OrtOutput::try_from)?;
        if output.data.len() != self.meta.labels.len() {
            return Err(InfraError::Backend(format!(
                "the classifier returned {} values for {} labels",
                output.data.len(),
                self.meta.labels.len()
            )));
        }
        // A NaN would compare as "not NSFW": fail instead (a float16 graph
        // whose accumulations overflow gives one).
        if output.data.iter().any(|v| !v.is_finite()) {
            return Err(InfraError::Backend(
                "the classifier returned a non-finite probability".to_string(),
            ));
        }
        // A float16 graph's probabilities may add up to a hair over 1.
        let nsfw = self
            .nsfw_indices
            .iter()
            .map(|&i| output.data[i])
            .sum::<f32>()
            .min(1.0);
        let scores = self
            .meta
            .labels
            .iter()
            .zip(&output.data)
            .map(|(label, &score)| LabelScore {
                label: label.clone(),
                score,
            })
            .collect();
        Ok(InferenceOutput::ImageNsfw { nsfw, scores })
    }
}

fn read_meta(path: &Path) -> Result<NsfwMeta> {
    let text = std::fs::read_to_string(path)
        .map_err(|err| InfraError::Adapter(format!("read {}: {err}", path.display())))?;
    serde_json::from_str(&text)
        .map_err(|err| InfraError::Adapter(format!("parse {}: {err}", path.display())))
}

/// The classifier's `size` x `size` input.
pub fn resized(image: &RgbImage, size: u32, mode: Resize) -> RgbImage {
    let filter = imageops::FilterType::CatmullRom;
    match mode {
        Resize::Squash => imageops::resize(image, size, size, filter),
        Resize::Center => {
            let (w, h) = image.dimensions();
            let scale = size as f64 / w.min(h).max(1) as f64;
            let nw = ((w as f64 * scale).round() as u32).max(size);
            let nh = ((h as f64 * scale).round() as u32).max(size);
            let scaled = imageops::resize(image, nw, nh, filter);
            let (x, y) = ((nw - size) / 2, (nh - size) / 2);
            imageops::crop_imm(&scaled, x, y, size, size).to_image()
        }
    }
}

/// NCHW RGB in 0..1.
fn pixels(image: &RgbImage) -> Vec<f32> {
    let (w, h) = image.dimensions();
    let plane = (w * h) as usize;
    let mut data = vec![0.0f32; 3 * plane];
    for (i, pixel) in image.pixels().enumerate() {
        for c in 0..3 {
            data[c * plane + i] = pixel[c] as f32 / 255.0;
        }
    }
    data
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
