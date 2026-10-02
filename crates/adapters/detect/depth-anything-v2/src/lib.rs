//! Depth Anything V2 metric depth: metres for every pixel of an image,
//! answered pooled to a grid.
//!
//! Provenance:
//! - Model: Depth Anything V2 Metric Indoor Small (DINOv2 ViT-S + DPT head,
//!   fine-tuned on Hypersim; <https://github.com/DepthAnything/Depth-Anything-V2>,
//!   Apache-2.0 as `depth-anything/Depth-Anything-V2-Metric-Hypersim-Small`),
//!   exported from its transformers form by `scripts/local/depth_anything_export.py`
//!   for one input size (`model.onnx`; see `configs/providers/detect/depth-anything-v2.yaml`).
//! - Preprocessing follows its `DPTImageProcessor` (bicubic resize,
//!   ImageNet mean and std), but stretches the image to the graph's fixed
//!   input size (a traced export bakes that size in) instead of keeping its
//!   aspect ratio: export for the frames' aspect ratio (16:9 by default).

use image::{imageops, ImageReader, RgbImage};
use local_backend_ort::{
    CudaSessionOptions, OrtBackend, OrtOutput, OrtSession, OrtTensorData, OrtTensorInput,
    ProviderSelection, SessionProviderReport,
};
use local_core::{DepthGrid, FileRef, InferenceOutput, ModelSpec};
use local_error::{InfraError, Result};
use std::path::PathBuf;

const MODEL_FILE: &str = "model.onnx";
const CONFIG_FILE: &str = "config.json";
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
/// Hypersim's (indoor) range, when `config.json` does not say.
const DEFAULT_MAX_DEPTH: f32 = 20.0;
/// 16:9 cells of 20 px on a 1280x720 frame.
const DEFAULT_GRID: DepthGrid = DepthGrid { cols: 64, rows: 36 };

#[derive(Debug)]
pub struct DepthAnythingAdapter {
    model_id: String,
    session: OrtSession,
    input_name: String,
    /// The graph's input size.
    width: u32,
    height: u32,
    max_depth: f32,
    grid: DepthGrid,
}

impl DepthAnythingAdapter {
    pub fn load(spec: &ModelSpec) -> Result<Self> {
        let missing = |reason: String| InfraError::ModelNotConfigured {
            model_id: spec.id.clone(),
            reason,
        };
        let model_path = artifact_file(spec, MODEL_FILE)
            .ok_or_else(|| missing(format!("{MODEL_FILE} is missing")))?;
        let backend = OrtBackend::new(ProviderSelection::from_strings(
            &spec.runtime.provider_order,
        ))
        .with_cuda_session_options(CudaSessionOptions::default());
        let session = backend.load_session(&model_path)?;
        let input = session
            .inputs()
            .first()
            .ok_or_else(|| missing(format!("{MODEL_FILE} has no input")))?;
        let [_, 3, height, width] = input.shape[..] else {
            return Err(missing(format!(
                "{MODEL_FILE} input shape {:?} is not [1, 3, H, W]",
                input.shape
            )));
        };
        if height <= 0 || width <= 0 {
            return Err(missing(format!(
                "{MODEL_FILE} must be exported for one input size (got {:?})",
                input.shape
            )));
        }
        let input_name = input.name.clone();
        let max_depth = artifact_file(spec, CONFIG_FILE)
            .and_then(|path| std::fs::read_to_string(path).ok())
            .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
            .and_then(|config| config.get("max_depth").and_then(serde_json::Value::as_f64))
            .map_or(DEFAULT_MAX_DEPTH, |v| v as f32);
        let grid = grid_from_metadata(&spec.metadata).unwrap_or(DEFAULT_GRID);
        Ok(Self {
            model_id: spec.id.clone(),
            session,
            input_name,
            width: width as u32,
            height: height as u32,
            max_depth,
            grid,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn provider_report(&self) -> SessionProviderReport {
        self.session.provider_report()
    }

    /// The depth of an image (metres), pooled to `grid` (default: the
    /// adapter's).
    pub fn depth(&mut self, image: &FileRef, grid: Option<DepthGrid>) -> Result<InferenceOutput> {
        let path = local_files::local_path(image)?;
        let image = ImageReader::open(&path)
            .and_then(|reader| reader.with_guessed_format())
            .map_err(|err| InfraError::BadRequest(format!("cannot read the image: {err}")))?
            .decode()
            .map_err(|err| InfraError::BadRequest(format!("cannot decode the image: {err}")))?
            .to_rgb8();
        let grid = grid.unwrap_or(self.grid);
        if grid.cols == 0 || grid.rows == 0 || grid.cols > self.width || grid.rows > self.height {
            return Err(InfraError::BadRequest(format!(
                "the depth grid must be 1x1 to {}x{}",
                self.width, self.height
            )));
        }
        let map = self.depth_map(&image)?;
        let depth = pool(&map, self.width, self.height, grid);
        Ok(InferenceOutput::DepthMap {
            cols: grid.cols,
            rows: grid.rows,
            max_depth: self.max_depth,
            depth,
        })
    }

    /// The model's depth map (metres, `width` x `height`, row-major).
    pub fn depth_map(&mut self, image: &RgbImage) -> Result<Vec<f32>> {
        let resized = imageops::resize(
            image,
            self.width,
            self.height,
            imageops::FilterType::CatmullRom,
        );
        let input = OrtTensorInput {
            name: self.input_name.clone(),
            shape: vec![1, 3, self.height as usize, self.width as usize],
            data: OrtTensorData::F32(normalized(&resized)),
        };
        let output = self
            .session
            .run_tensors(&[input])?
            .into_iter()
            .next()
            .ok_or_else(|| InfraError::Backend("the model returned no output".to_string()))
            .and_then(OrtOutput::try_from)?;
        let expected = (self.width * self.height) as usize;
        if output.data.len() != expected {
            return Err(InfraError::Backend(format!(
                "depth output has {} values for a {}x{} input",
                output.data.len(),
                self.width,
                self.height
            )));
        }
        Ok(output.data)
    }
}

fn artifact_file(spec: &ModelSpec, name: &str) -> Option<PathBuf> {
    spec.artifacts.iter().find_map(|artifact| {
        let path = &artifact.path;
        if path.file_name().is_some_and(|n| n == name) {
            return Some(path.clone());
        }
        let candidate = path.join(name);
        (artifact.files.iter().any(|f| f == name) || candidate.is_file()).then_some(candidate)
    })
}

fn grid_from_metadata(
    metadata: &std::collections::BTreeMap<String, serde_json::Value>,
) -> Option<DepthGrid> {
    let get = |key: &str| metadata.get(key).and_then(serde_json::Value::as_u64);
    Some(DepthGrid {
        cols: get("grid_cols")? as u32,
        rows: get("grid_rows")? as u32,
    })
    .filter(|grid| grid.cols > 0 && grid.rows > 0)
}

/// NCHW, ImageNet-normalized RGB.
fn normalized(image: &RgbImage) -> Vec<f32> {
    let (w, h) = image.dimensions();
    let plane = (w * h) as usize;
    let mut data = vec![0.0f32; 3 * plane];
    for (i, pixel) in image.pixels().enumerate() {
        for c in 0..3 {
            data[c * plane + i] = (pixel[c] as f32 / 255.0 - MEAN[c]) / STD[c];
        }
    }
    data
}

/// The mean of each grid cell of a `width` x `height` map; cell edges are
/// spread evenly (`col * width / cols`), so every pixel counts once.
pub fn pool(map: &[f32], width: u32, height: u32, grid: DepthGrid) -> Vec<f32> {
    let (w, h) = (width as usize, height as usize);
    let (cols, rows) = (grid.cols as usize, grid.rows as usize);
    let mut out = Vec::with_capacity(cols * rows);
    for row in 0..rows {
        let (y0, y1) = (row * h / rows, (row + 1) * h / rows);
        for col in 0..cols {
            let (x0, x1) = (col * w / cols, (col + 1) * w / cols);
            let mut sum = 0.0f32;
            for y in y0..y1 {
                sum += map[y * w + x0..y * w + x1].iter().sum::<f32>();
            }
            out.push(sum / ((y1 - y0) * (x1 - x0)).max(1) as f32);
        }
    }
    out
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
