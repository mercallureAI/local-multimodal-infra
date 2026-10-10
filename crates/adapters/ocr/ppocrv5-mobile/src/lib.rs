//! PP-OCRv5 mobile text lines: DB text detection, then CTC recognition of
//! each detected line.
//!
//! Provenance:
//! - Models: PaddleOCR PP-OCRv5 mobile detection and recognition
//!   (<https://github.com/PaddlePaddle/PaddleOCR>), as ONNX conversions
//!   pinned in `configs/providers/ocr/ppocrv5-mobile.yaml`; the character
//!   dictionary is PaddleOCR's `ppocr/utils/dict/ppocrv5_dict.txt`.
//! - Pre/postprocessing follows PaddleOCR's `DetResizeForTest`,
//!   `NormalizeImage`, `DBPostProcess`, `RecResizeImg` and `CTCLabelDecode`,
//!   with one simplification: a detected region becomes its axis-aligned
//!   box (connected components of the thresholded probability map) rather
//!   than a rotated rectangle, which suits horizontal text such as labels,
//!   signs and name tags.
//!
//! Recognition runs lines in batches of similar width (sorted by width, as
//! PaddleOCR does), each padded to a multiple of `rec_width_step`: on a GPU
//! the number of runs matters more than the padding. Both sessions pick
//! cuDNN algorithms by heuristic, since their input shapes vary from image
//! to image (an exhaustive search for each new shape takes longer than many
//! runs).

use image::{imageops, ImageReader, RgbImage};
use local_backend_ort::{
    CudaSessionOptions, OrtBackend, OrtOutput, OrtSession, OrtTensorData, OrtTensorInput,
    ProviderSelection, SessionProviderReport,
};
use local_core::{BoundingBox, FileRef, InferenceOutput, ModelSpec, OcrLine};
use local_error::{InfraError, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const DET_FILE: &str = "ppocrv5_det.onnx";
const REC_FILE: &str = "ppocrv5_rec.onnx";
const DICT_FILE: &str = "ppocrv5_dict.txt";
const DET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const DET_STD: [f32; 3] = [0.229, 0.224, 0.225];
const REC_HEIGHT: u32 = 48;
/// Boxes smaller than this (detection pixels) are noise.
const MIN_BOX_SIDE: u32 = 3;

/// Thresholds and sizes, overridable in the spec's `metadata`.
#[derive(Debug, Clone, PartialEq)]
pub struct PpocrConfig {
    /// Longest image side the detector sees (larger images are scaled down).
    pub det_limit_side_len: u32,
    /// Probability above which a detection pixel is text.
    pub det_thresh: f32,
    /// Mean probability a region needs to be a box.
    pub box_thresh: f32,
    /// How far a box grows around its region (DB's unclip ratio).
    pub unclip_ratio: f32,
    /// Lines recognised with a lower mean confidence are left out.
    pub min_confidence: f32,
    pub rec_batch: usize,
    pub rec_width_step: u32,
    pub rec_max_width: u32,
    /// Most padded columns (lines x padded width) in one recognition batch:
    /// bounds the recogniser's activations, which grow with both (a dense
    /// screenshot's long lines would take ~1 GB more at 32 x 1600). `None`:
    /// batches of `rec_batch` lines whatever their width.
    pub rec_batch_columns: Option<u32>,
    /// Give each session's run memory back after an image
    /// (`cuda_release_memory_after_run`).
    pub release_memory: bool,
}

impl Default for PpocrConfig {
    fn default() -> Self {
        Self {
            det_limit_side_len: 960,
            det_thresh: 0.3,
            box_thresh: 0.6,
            unclip_ratio: 1.5,
            min_confidence: 0.5,
            rec_batch: 8,
            rec_width_step: 80,
            rec_max_width: 1600,
            rec_batch_columns: None,
            release_memory: false,
        }
    }
}

impl PpocrConfig {
    fn from_metadata(metadata: &BTreeMap<String, serde_json::Value>) -> Self {
        let mut config = Self::default();
        let f = |key: &str| metadata.get(key).and_then(serde_json::Value::as_f64);
        if let Some(v) = f("det_limit_side_len") {
            config.det_limit_side_len = (v as u32).max(32);
        }
        if let Some(v) = f("det_thresh") {
            config.det_thresh = v as f32;
        }
        if let Some(v) = f("box_thresh") {
            config.box_thresh = v as f32;
        }
        if let Some(v) = f("unclip_ratio") {
            config.unclip_ratio = v as f32;
        }
        if let Some(v) = f("min_confidence") {
            config.min_confidence = v as f32;
        }
        if let Some(v) = f("rec_batch") {
            config.rec_batch = (v as usize).max(1);
        }
        if let Some(v) = f("rec_width_step") {
            config.rec_width_step = (v as u32).max(8);
        }
        if let Some(v) = f("rec_max_width") {
            config.rec_max_width = (v as u32).max(REC_HEIGHT);
        }
        if let Some(v) = f("rec_batch_columns") {
            config.rec_batch_columns = Some((v as u32).max(config.rec_max_width));
        }
        config.release_memory = local_backend_ort::release_memory_after_run(metadata);
        config
    }
}

#[derive(Debug)]
pub struct PpocrAdapter {
    model_id: String,
    config: PpocrConfig,
    det: OrtSession,
    rec: OrtSession,
    /// Characters of recognition classes 1..=len (0 is the CTC blank, the
    /// class after the last character is a space).
    dict: Vec<String>,
}

impl PpocrAdapter {
    pub fn load(spec: &ModelSpec) -> Result<Self> {
        let missing = |reason: String| InfraError::ModelNotConfigured {
            model_id: spec.id.clone(),
            reason,
        };
        let det_path = artifact_file(spec, DET_FILE)
            .ok_or_else(|| missing(format!("{DET_FILE} is missing")))?;
        let rec_path = artifact_file(spec, REC_FILE)
            .ok_or_else(|| missing(format!("{REC_FILE} is missing")))?;
        let dict_path = artifact_file(spec, DICT_FILE)
            .ok_or_else(|| missing(format!("{DICT_FILE} is missing")))?;
        let dict = load_dict(&dict_path)?;
        let backend = OrtBackend::new(ProviderSelection::from_strings(
            &spec.runtime.provider_order,
        ))
        .with_cuda_session_options(CudaSessionOptions {
            conv_algo_heuristic: true,
            ..CudaSessionOptions::default()
        })
        .with_cuda_memory_metadata(&spec.metadata);
        let det = backend.load_session(&det_path)?;
        let rec = backend.load_session(&rec_path)?;
        // The recogniser and the dictionary must belong together.
        if let Some(classes) = rec
            .outputs()
            .first()
            .and_then(|output| output.shape.last().copied())
            .filter(|classes| *classes > 0)
        {
            if classes as usize != dict.len() + 2 {
                return Err(missing(format!(
                    "{REC_FILE} has {classes} classes; {DICT_FILE} needs {}",
                    dict.len() + 2
                )));
            }
        }
        Ok(Self {
            model_id: spec.id.clone(),
            config: PpocrConfig::from_metadata(&spec.metadata),
            det,
            rec,
            dict,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn provider_report(&self) -> (SessionProviderReport, SessionProviderReport) {
        (self.det.provider_report(), self.rec.provider_report())
    }

    /// The text lines of an image, top to bottom then left to right.
    pub fn ocr_lines(&mut self, image: &FileRef) -> Result<InferenceOutput> {
        let path = local_files::local_path(image)?;
        let image = ImageReader::open(&path)
            .and_then(|reader| reader.with_guessed_format())
            .map_err(|err| InfraError::BadRequest(format!("cannot read the image: {err}")))?
            .decode()
            .map_err(|err| InfraError::BadRequest(format!("cannot decode the image: {err}")))?
            .to_rgb8();
        let boxes = self.detect(&image)?;
        let lines = self.recognize(&image, &boxes)?;
        Ok(InferenceOutput::OcrLines { lines })
    }

    fn detect(&mut self, image: &RgbImage) -> Result<Vec<PixelBox>> {
        let (width, height) = image.dimensions();
        let (det_w, det_h) = det_size(width, height, self.config.det_limit_side_len);
        let resized = imageops::resize(image, det_w, det_h, imageops::FilterType::Triangle);
        let input = OrtTensorInput {
            name: input_name(&self.det, "x"),
            shape: vec![1, 3, det_h as usize, det_w as usize],
            data: OrtTensorData::F32(det_tensor(&resized)),
        };
        let output = if self.config.release_memory {
            self.det.run_tensors_releasing_memory(&[input])?
        } else {
            self.det.run_tensors(&[input])?
        };
        let output = first_output(output)?;
        let expected = (det_w * det_h) as usize;
        if output.data.len() != expected {
            return Err(InfraError::Backend(format!(
                "detector output has {} values for a {det_w}x{det_h} input",
                output.data.len()
            )));
        }
        Ok(text_boxes(
            &output.data,
            det_w,
            det_h,
            &self.config,
            (width as f32 / det_w as f32, height as f32 / det_h as f32),
            (width, height),
        ))
    }

    fn recognize(&mut self, image: &RgbImage, boxes: &[PixelBox]) -> Result<Vec<OcrLine>> {
        // Crops at the model's height, narrowest first so batches pad little.
        let mut crops: Vec<(usize, RgbImage)> = boxes
            .iter()
            .enumerate()
            .map(|(index, b)| {
                let crop =
                    imageops::crop_imm(image, b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0).to_image();
                let width = rec_width(crop.width(), crop.height(), self.config.rec_max_width);
                let resized =
                    imageops::resize(&crop, width, REC_HEIGHT, imageops::FilterType::Triangle);
                (index, resized)
            })
            .collect();
        crops.sort_by_key(|(_, crop)| crop.width());
        let classes = self.dict.len() + 2;
        let mut lines: Vec<Option<OcrLine>> = vec![None; boxes.len()];
        let widths: Vec<u32> = crops.iter().map(|(_, crop)| crop.width()).collect();
        let mut start = 0;
        let batches = rec_batches(&widths, &self.config);
        let last = batches.len().saturating_sub(1);
        for (b, len) in batches.into_iter().enumerate() {
            let batch = &crops[start..start + len];
            start += len;
            let padded = self
                .config
                .padded_width(batch.last().map_or(1, |(_, c)| c.width()));
            let input = OrtTensorInput {
                name: input_name(&self.rec, "x"),
                shape: vec![batch.len(), 3, REC_HEIGHT as usize, padded as usize],
                data: OrtTensorData::F32(rec_tensor(batch.iter().map(|(_, c)| c), padded)),
            };
            let output = if self.config.release_memory && b == last {
                self.rec.run_tensors_releasing_memory(&[input])?
            } else {
                self.rec.run_tensors(&[input])?
            };
            let output = first_output(output)?;
            let [n, steps, out_classes] = output.shape[..] else {
                return Err(InfraError::Backend(format!(
                    "recogniser output shape {:?} is not [batch, steps, classes]",
                    output.shape
                )));
            };
            if n != batch.len() || out_classes != classes {
                return Err(InfraError::Backend(format!(
                    "recogniser output shape {:?} does not fit {} lines and {classes} classes",
                    output.shape,
                    batch.len()
                )));
            }
            for (row, (index, _)) in batch.iter().enumerate() {
                let probs = &output.data[row * steps * classes..(row + 1) * steps * classes];
                let (text, confidence) = ctc_decode(probs, classes, &self.dict);
                if text.trim().is_empty() || confidence < self.config.min_confidence {
                    continue;
                }
                let b = &boxes[*index];
                lines[*index] = Some(OcrLine {
                    text,
                    confidence,
                    bbox: BoundingBox {
                        x: b.x0 as f32,
                        y: b.y0 as f32,
                        width: (b.x1 - b.x0) as f32,
                        height: (b.y1 - b.y0) as f32,
                    },
                });
            }
        }
        Ok(lines.into_iter().flatten().collect())
    }
}

impl PpocrConfig {
    /// A batch's input width for its widest crop (`rec_width` never exceeds
    /// `rec_max_width`, so neither does this).
    fn padded_width(&self, widest: u32) -> u32 {
        (widest.div_ceil(self.rec_width_step) * self.rec_width_step).min(self.rec_max_width)
    }
}

/// Batch sizes over crops sorted narrowest first: up to `rec_batch` lines,
/// and with `rec_batch_columns`, no more lines than keep lines x padded
/// width within it (one line always fits).
pub fn rec_batches(widths: &[u32], config: &PpocrConfig) -> Vec<usize> {
    let mut sizes = Vec::new();
    let mut start = 0;
    while start < widths.len() {
        let mut len = 1;
        while start + len < widths.len() && len < config.rec_batch {
            let padded = config.padded_width(widths[start + len]);
            if config
                .rec_batch_columns
                .is_some_and(|budget| (len as u32 + 1) * padded > budget)
            {
                break;
            }
            len += 1;
        }
        sizes.push(len);
        start += len;
    }
    sizes
}

/// A text box in image pixels (`x1`/`y1` exclusive).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PixelBox {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
    pub score: f32,
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

fn load_dict(path: &Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(path)
        .map_err(|err| InfraError::Backend(format!("cannot read {}: {err}", path.display())))?;
    // One character per line; the file ends with a line break.
    Ok(text.lines().map(str::to_string).collect())
}

fn input_name(session: &OrtSession, fallback: &str) -> String {
    session
        .inputs()
        .first()
        .map(|input| input.name.clone())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| fallback.to_string())
}

fn first_output(outputs: Vec<local_backend_ort::OrtTensorOutput>) -> Result<OrtOutput> {
    outputs
        .into_iter()
        .next()
        .ok_or_else(|| InfraError::Backend("the model returned no output".to_string()))
        .and_then(OrtOutput::try_from)
}

/// The detector's input size: the image scaled down to `limit` on its
/// longest side, each side rounded to a multiple of 32.
pub fn det_size(width: u32, height: u32, limit: u32) -> (u32, u32) {
    let scale = (limit as f32 / width.max(height) as f32).min(1.0);
    let round = |side: u32| (((side as f32 * scale / 32.0).round() as u32) * 32).max(32);
    (round(width), round(height))
}

/// NCHW, BGR channel order (PaddleOCR reads images as BGR), ImageNet
/// normalisation.
fn det_tensor(image: &RgbImage) -> Vec<f32> {
    let (w, h) = image.dimensions();
    let plane = (w * h) as usize;
    let mut data = vec![0.0f32; plane * 3];
    for (i, pixel) in image.pixels().enumerate() {
        for c in 0..3 {
            let value = pixel[2 - c] as f32 / 255.0;
            data[c * plane + i] = (value - DET_MEAN[c]) / DET_STD[c];
        }
    }
    data
}

/// The recogniser's width for a crop at its fixed height.
pub fn rec_width(width: u32, height: u32, max_width: u32) -> u32 {
    let width = (REC_HEIGHT as f32 * width as f32 / height.max(1) as f32).ceil() as u32;
    width.clamp(1, max_width)
}

/// A batch of crops as NCHW BGR in [-1, 1], right-padded with 0 to `padded`.
fn rec_tensor<'a>(crops: impl Iterator<Item = &'a RgbImage>, padded: u32) -> Vec<f32> {
    let crops: Vec<&RgbImage> = crops.collect();
    let (h, w) = (REC_HEIGHT as usize, padded as usize);
    let mut data = vec![0.0f32; crops.len() * 3 * h * w];
    for (n, crop) in crops.iter().enumerate() {
        for (x, y, pixel) in crop.enumerate_pixels() {
            if x as usize >= w {
                continue;
            }
            for c in 0..3 {
                let value = pixel[2 - c] as f32 / 255.0;
                data[((n * 3 + c) * h + y as usize) * w + x as usize] = (value - 0.5) / 0.5;
            }
        }
    }
    data
}

/// Text boxes from the detector's probability map (`width` x `height`):
/// connected regions above `det_thresh` whose mean probability reaches
/// `box_thresh`, grown by the unclip distance and scaled to the image.
pub fn text_boxes(
    prob: &[f32],
    width: u32,
    height: u32,
    config: &PpocrConfig,
    scale: (f32, f32),
    image_size: (u32, u32),
) -> Vec<PixelBox> {
    let (w, h) = (width as usize, height as usize);
    let mut seen = vec![false; w * h];
    let mut stack = Vec::new();
    let mut boxes = Vec::new();
    for start in 0..w * h {
        if seen[start] || prob[start] <= config.det_thresh {
            continue;
        }
        seen[start] = true;
        stack.push(start);
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
        let (mut sum, mut count) = (0.0f32, 0usize);
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
            sum += prob[i];
            count += 1;
            let mut visit = |j: usize| {
                if !seen[j] && prob[j] > config.det_thresh {
                    seen[j] = true;
                    stack.push(j);
                }
            };
            if x > 0 {
                visit(i - 1);
            }
            if x + 1 < w {
                visit(i + 1);
            }
            if y > 0 {
                visit(i - w);
            }
            if y + 1 < h {
                visit(i + w);
            }
        }
        let (bw, bh) = ((x1 - x0) as f32, (y1 - y0) as f32);
        let score = sum / count as f32;
        if (bw.min(bh) as u32) < MIN_BOX_SIDE || score < config.box_thresh {
            continue;
        }
        let d = bw * bh * config.unclip_ratio / (2.0 * (bw + bh));
        let (sx, sy) = scale;
        let clamp = |v: f32, max: u32| v.max(0.0).min(max as f32);
        let b = PixelBox {
            x0: clamp((x0 as f32 - d) * sx, image_size.0).floor() as u32,
            y0: clamp((y0 as f32 - d) * sy, image_size.1).floor() as u32,
            x1: clamp((x1 as f32 + d) * sx, image_size.0).ceil() as u32,
            y1: clamp((y1 as f32 + d) * sy, image_size.1).ceil() as u32,
            score,
        };
        if b.x1 > b.x0 && b.y1 > b.y0 {
            boxes.push(b);
        }
    }
    reading_order(&mut boxes);
    boxes
}

/// Boxes whose tops are this close (px) are on one row (PaddleOCR's
/// `sorted_boxes`).
const SAME_ROW_PX: u32 = 10;

/// Top to bottom, then left to right within a row: boxes on one row whose
/// tops differ by a few pixels still read left to right.
pub fn reading_order(boxes: &mut [PixelBox]) {
    boxes.sort_by_key(|b| (b.y0, b.x0));
    for i in 1..boxes.len() {
        let mut j = i;
        while j > 0
            && boxes[j].y0.abs_diff(boxes[j - 1].y0) < SAME_ROW_PX
            && boxes[j].x0 < boxes[j - 1].x0
        {
            boxes.swap(j, j - 1);
            j -= 1;
        }
    }
}

/// Greedy CTC decoding of one line (`steps` x `classes` probabilities):
/// repeats collapse, the blank (0) separates, the last class is a space.
/// Returns the text and the mean probability of its characters.
pub fn ctc_decode(probs: &[f32], classes: usize, dict: &[String]) -> (String, f32) {
    let mut text = String::new();
    let (mut sum, mut count) = (0.0f32, 0usize);
    let mut previous = 0usize;
    for step in probs.chunks_exact(classes) {
        let (best, p) = step
            .iter()
            .copied()
            .enumerate()
            .fold(
                (0, f32::MIN),
                |acc, (k, p)| if p > acc.1 { (k, p) } else { acc },
            );
        if best != 0 && best != previous {
            match dict.get(best - 1) {
                Some(ch) => text.push_str(ch),
                None => text.push(' '),
            }
            sum += p;
            count += 1;
        }
        previous = best;
    }
    (text, if count == 0 { 0.0 } else { sum / count as f32 })
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
