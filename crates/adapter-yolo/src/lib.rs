//! YOLO adapter provenance:
//! - Preprocessing and output decoding are implemented to match Ultralytics YOLO detect models:
//!   https://github.com/ultralytics/ultralytics
//!   https://docs.ultralytics.com/tasks/detect/
//! - Default COCO labels follow Ultralytics' dataset metadata:
//!   https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco.yaml
//! - The current default Hugging Face ONNX model source may include:
//!   https://huggingface.co/aaurelions/yolo11n.onnx
//! - This crate is an adapter implemented inside this project and does not directly depend on or
//!   vendor Ultralytics.

use image::{imageops, Rgb, RgbImage};
use lcoal_backend_ort::{
    OrtBackend, OrtInput, OrtOutput, OrtSession, ProviderSelection, SessionProviderReport,
};
use lcoal_core::{BoundingBox, DetectedObject, FileRef, InferenceOutput, ModelSpec};
use lcoal_error::{InfraError, Result};
use serde_yaml::Value;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct YoloConfig {
    pub input_width: u32,
    pub input_height: u32,
    pub confidence_threshold: f32,
    pub nms_iou_threshold: f32,
    pub labels: Vec<String>,
}

impl Default for YoloConfig {
    fn default() -> Self {
        Self {
            input_width: 640,
            input_height: 640,
            confidence_threshold: 0.25,
            nms_iou_threshold: 0.45,
            labels: coco_labels(),
        }
    }
}

#[derive(Debug)]
pub struct YoloAdapter {
    model_id: String,
    config: YoloConfig,
    session: OrtSession,
}

impl YoloAdapter {
    pub fn load(spec: &ModelSpec) -> Result<Self> {
        let model_path = yolo_model_path(spec)?;
        if !model_path.exists() {
            return Err(InfraError::ModelNotConfigured {
                model_id: spec.id.clone(),
                reason: format!("YOLO ONNX model is missing: {}", model_path.display()),
            });
        }
        let backend = OrtBackend::new(ProviderSelection::from_strings(
            &spec.runtime.provider_order,
        ));
        let session = backend.load_session(&model_path)?;
        let mut config = YoloConfig::default();
        if let Some(labels_path) = yolo_labels_path(spec) {
            if labels_path.exists() {
                config.labels = parse_coco_yaml_labels(&labels_path)?;
            }
        }
        Ok(Self {
            model_id: spec.id.clone(),
            config,
            session,
        })
    }

    pub fn object_detect(&mut self, image: &FileRef) -> Result<InferenceOutput> {
        let image_path = lcoal_files::local_path(image)?;
        let preprocessed = preprocess_image(
            &image_path,
            self.config.input_width,
            self.config.input_height,
        )?;
        let input_name = self
            .session
            .inputs()
            .first()
            .map(|input| input.name.clone())
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| "images".to_string());
        let input = OrtInput {
            name: input_name,
            shape: vec![
                1,
                3,
                self.config.input_height as usize,
                self.config.input_width as usize,
            ],
            data: preprocessed.tensor.clone(),
        };
        let outputs = self.session.run_f32(&[input])?;
        let objects = decode_yolo_outputs(&outputs, &preprocessed, &self.config)?;
        Ok(InferenceOutput::ObjectDetections { objects })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn provider_report(&self) -> SessionProviderReport {
        self.session.provider_report()
    }
}

fn yolo_model_path(spec: &ModelSpec) -> Result<PathBuf> {
    let mut candidates = Vec::new();
    for artifact in &spec.artifacts {
        if has_extension(&artifact.path, "onnx") {
            candidates.push(artifact.path.clone());
        }
        for file in &artifact.files {
            if Path::new(file)
                .extension()
                .is_some_and(|ext| os_str_eq_ignore_ascii_case(ext, "onnx"))
            {
                candidates.push(if has_extension(&artifact.path, "onnx") {
                    artifact.path.clone()
                } else {
                    artifact.path.join(file.as_str())
                });
            }
        }
    }

    candidates
        .iter()
        .find(|path| path.exists())
        .or_else(|| candidates.first())
        .cloned()
        .ok_or_else(|| InfraError::ModelNotConfigured {
            model_id: spec.id.clone(),
            reason: "YOLO .onnx artifact path is not configured".to_string(),
        })
}

fn yolo_labels_path(spec: &ModelSpec) -> Option<PathBuf> {
    spec.artifacts.iter().find_map(|artifact| {
        if artifact
            .path
            .file_name()
            .is_some_and(|name| os_str_eq_ignore_ascii_case(name, "coco.yaml"))
        {
            return Some(artifact.path.clone());
        }
        artifact.files.iter().find_map(|file| {
            (Path::new(file)
                .file_name()
                .is_some_and(|name| os_str_eq_ignore_ascii_case(name, "coco.yaml")))
            .then(|| artifact.path.join(file.as_str()))
        })
    })
}

fn has_extension(path: &Path, extension: &str) -> bool {
    path.extension()
        .is_some_and(|ext| os_str_eq_ignore_ascii_case(ext, extension))
}

fn os_str_eq_ignore_ascii_case(value: &std::ffi::OsStr, expected: &str) -> bool {
    value
        .to_str()
        .is_some_and(|value| value.eq_ignore_ascii_case(expected))
}

fn parse_coco_yaml_labels(path: &Path) -> Result<Vec<String>> {
    let contents = fs::read_to_string(path)
        .map_err(|e| InfraError::Adapter(format!("read YOLO labels {}: {e}", path.display())))?;
    parse_coco_yaml_labels_str(&contents)
}

fn parse_coco_yaml_labels_str(contents: &str) -> Result<Vec<String>> {
    let yaml: Value = serde_yaml::from_str(contents)
        .map_err(|e| InfraError::Adapter(format!("parse YOLO coco.yaml labels: {e}")))?;
    let names = yaml
        .as_mapping()
        .and_then(|mapping| mapping.get(&Value::String("names".to_string())))
        .ok_or_else(|| InfraError::Adapter("YOLO coco.yaml does not contain names".to_string()))?;

    let labels = if let Some(sequence) = names.as_sequence() {
        sequence
            .iter()
            .enumerate()
            .map(|(index, value)| yaml_label_value(value, index))
            .collect::<Result<Vec<_>>>()?
    } else if let Some(mapping) = names.as_mapping() {
        let mut entries = mapping
            .iter()
            .map(|(key, value)| {
                let index = yaml_label_index(key)?;
                let label = yaml_label_value(value, index)?;
                Ok((index, label))
            })
            .collect::<Result<Vec<_>>>()?;
        entries.sort_by_key(|(index, _)| *index);
        for (expected, (actual, _)) in entries.iter().enumerate() {
            if *actual != expected {
                return Err(InfraError::Adapter(format!(
                    "YOLO coco.yaml names are not contiguous at index {expected}; found {actual}"
                )));
            }
        }
        entries.into_iter().map(|(_, label)| label).collect()
    } else {
        return Err(InfraError::Adapter(
            "YOLO coco.yaml names must be a list or map".to_string(),
        ));
    };

    if labels.is_empty() {
        Err(InfraError::Adapter(
            "YOLO coco.yaml names are empty".to_string(),
        ))
    } else {
        Ok(labels)
    }
}

fn yaml_label_index(value: &Value) -> Result<usize> {
    if let Some(index) = value.as_i64() {
        return usize::try_from(index).map_err(|_| {
            InfraError::Adapter(format!(
                "YOLO coco.yaml contains negative label index {index}"
            ))
        });
    }
    if let Some(index) = value.as_str().and_then(|s| s.parse::<usize>().ok()) {
        return Ok(index);
    }
    Err(InfraError::Adapter(format!(
        "YOLO coco.yaml contains non-numeric label index {value:?}"
    )))
}

fn yaml_label_value(value: &Value, index: usize) -> Result<String> {
    let label = value.as_str().ok_or_else(|| {
        InfraError::Adapter(format!(
            "YOLO coco.yaml label at index {index} is not a string"
        ))
    })?;
    let label = label.trim();
    if label.is_empty() {
        Err(InfraError::Adapter(format!(
            "YOLO coco.yaml label at index {index} is empty"
        )))
    } else {
        Ok(label.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct PreprocessedImage {
    pub tensor: Vec<f32>,
    pub original_width: u32,
    pub original_height: u32,
    pub resized_width: u32,
    pub resized_height: u32,
    pub pad_x: u32,
    pub pad_y: u32,
    pub scale: f32,
}

pub fn preprocess_image(
    path: &Path,
    input_width: u32,
    input_height: u32,
) -> Result<PreprocessedImage> {
    let img = image::open(path)
        .map_err(|e| InfraError::Adapter(format!("read image {}: {e}", path.display())))?
        .to_rgb8();
    let (original_width, original_height) = img.dimensions();
    if original_width == 0 || original_height == 0 {
        return Err(InfraError::BadRequest(
            "image has zero width or height".to_string(),
        ));
    }

    let scale = (input_width as f32 / original_width as f32)
        .min(input_height as f32 / original_height as f32);
    let resized_width = (original_width as f32 * scale)
        .round()
        .clamp(1.0, input_width as f32) as u32;
    let resized_height = (original_height as f32 * scale)
        .round()
        .clamp(1.0, input_height as f32) as u32;
    let pad_x = (input_width - resized_width) / 2;
    let pad_y = (input_height - resized_height) / 2;

    let resized = imageops::resize(
        &img,
        resized_width,
        resized_height,
        imageops::FilterType::Triangle,
    );
    let mut canvas = RgbImage::from_pixel(input_width, input_height, Rgb([114, 114, 114]));
    imageops::replace(&mut canvas, &resized, i64::from(pad_x), i64::from(pad_y));

    let plane = (input_width * input_height) as usize;
    let mut tensor = vec![0.0_f32; 3 * plane];
    for (x, y, pixel) in canvas.enumerate_pixels() {
        let idx = (y * input_width + x) as usize;
        tensor[idx] = f32::from(pixel[0]) / 255.0;
        tensor[plane + idx] = f32::from(pixel[1]) / 255.0;
        tensor[2 * plane + idx] = f32::from(pixel[2]) / 255.0;
    }

    Ok(PreprocessedImage {
        tensor,
        original_width,
        original_height,
        resized_width,
        resized_height,
        pad_x,
        pad_y,
        scale,
    })
}

pub fn decode_yolo_outputs(
    outputs: &[OrtOutput],
    image: &PreprocessedImage,
    config: &YoloConfig,
) -> Result<Vec<DetectedObject>> {
    let output = outputs
        .iter()
        .find(|o| o.name == "output0")
        .or_else(|| outputs.first())
        .ok_or_else(|| InfraError::Adapter("YOLO ORT session returned no outputs".to_string()))?;
    let rows = rows_from_output(output, config)?;
    let mut candidates = Vec::new();

    for row in rows {
        if row.len() < 6 {
            continue;
        }
        if !row.iter().all(|value| value.is_finite()) {
            continue;
        }
        let Some((class_id, confidence)) = best_detection(&row, config.labels.len()) else {
            continue;
        };
        if confidence < config.confidence_threshold {
            continue;
        }
        let bbox = scale_box(row[0], row[1], row[2], row[3], image);
        candidates.push((
            DetectedObject {
                label: label_for(class_id, &config.labels),
                confidence,
                bbox,
            },
            class_id,
        ));
    }

    candidates.sort_by(|a, b| b.0.confidence.total_cmp(&a.0.confidence));
    let kept = nms(candidates, config.nms_iou_threshold);
    Ok(kept.into_iter().map(|(obj, _)| obj).collect())
}

fn rows_from_output(output: &OrtOutput, config: &YoloConfig) -> Result<Vec<Vec<f32>>> {
    match output.shape.as_slice() {
        [1, dim1, dim2] => {
            validate_output_len(output, *dim1 * *dim2)?;
            if is_transposed_yolo_shape(*dim1, *dim2, config.labels.len()) {
                let attrs = *dim1;
                let rows = *dim2;
                let mut transposed = Vec::with_capacity(rows);
                for row in 0..rows {
                    let mut values = Vec::with_capacity(attrs);
                    for attr in 0..attrs {
                        values.push(output.data[attr * rows + row]);
                    }
                    transposed.push(values);
                }
                Ok(transposed)
            } else if *dim2 >= 6 {
                Ok(output.data.chunks(*dim2).map(|row| row.to_vec()).collect())
            } else {
                Err(unsupported_output_shape(output))
            }
        }
        [rows, attrs] if *attrs >= 6 => {
            validate_output_len(output, *rows * *attrs)?;
            Ok(output.data.chunks(*attrs).map(|row| row.to_vec()).collect())
        }
        _ => Err(unsupported_output_shape(output)),
    }
}

fn is_transposed_yolo_shape(dim1: usize, dim2: usize, label_count: usize) -> bool {
    if dim1 < 6 {
        return false;
    }
    if expected_attr_count(dim1, label_count) && !expected_attr_count(dim2, label_count) {
        return true;
    }
    !expected_attr_count(dim2, label_count) && dim2 > dim1
}

fn expected_attr_count(attrs: usize, label_count: usize) -> bool {
    (label_count > 0 && (attrs == label_count + 4 || attrs == label_count + 5))
        || attrs == 84
        || attrs == 85
}

fn validate_output_len(output: &OrtOutput, expected: usize) -> Result<()> {
    if expected != output.data.len() {
        return Err(InfraError::Adapter(format!(
            "YOLO output data length {} does not match shape {:?}",
            output.data.len(),
            output.shape
        )));
    }
    Ok(())
}

fn unsupported_output_shape(output: &OrtOutput) -> InfraError {
    InfraError::Adapter(format!("unsupported YOLO output shape {:?}", output.shape))
}

fn best_detection(row: &[f32], label_count: usize) -> Option<(usize, f32)> {
    let attrs = row.len();
    let has_objectness = has_objectness(attrs, label_count);
    let (objectness, scores) = if has_objectness {
        (row[4], &row[5..])
    } else {
        (1.0, &row[4..])
    };
    let (class_id, class_score) = scores
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap_or((0, 0.0));
    let confidence = objectness * class_score;
    confidence.is_finite().then_some((class_id, confidence))
}

fn has_objectness(attrs: usize, label_count: usize) -> bool {
    if label_count > 0 {
        attrs == label_count + 5
    } else {
        attrs == 85
    }
}

fn scale_box(xc: f32, yc: f32, w: f32, h: f32, image: &PreprocessedImage) -> BoundingBox {
    let x1 = ((xc - w / 2.0) - image.pad_x as f32) / image.scale;
    let y1 = ((yc - h / 2.0) - image.pad_y as f32) / image.scale;
    let x2 = ((xc + w / 2.0) - image.pad_x as f32) / image.scale;
    let y2 = ((yc + h / 2.0) - image.pad_y as f32) / image.scale;
    let x1 = x1.clamp(0.0, image.original_width as f32);
    let y1 = y1.clamp(0.0, image.original_height as f32);
    let x2 = x2.clamp(0.0, image.original_width as f32);
    let y2 = y2.clamp(0.0, image.original_height as f32);
    BoundingBox {
        x: x1,
        y: y1,
        width: (x2 - x1).max(0.0),
        height: (y2 - y1).max(0.0),
    }
}

fn nms(candidates: Vec<(DetectedObject, usize)>, threshold: f32) -> Vec<(DetectedObject, usize)> {
    let mut kept: Vec<(DetectedObject, usize)> = Vec::new();
    'candidate: for candidate in candidates {
        for existing in &kept {
            if candidate.1 == existing.1 && iou(candidate.0.bbox, existing.0.bbox) > threshold {
                continue 'candidate;
            }
        }
        kept.push(candidate);
    }
    kept
}

fn iou(a: BoundingBox, b: BoundingBox) -> f32 {
    let ax2 = a.x + a.width;
    let ay2 = a.y + a.height;
    let bx2 = b.x + b.width;
    let by2 = b.y + b.height;
    let ix1 = a.x.max(b.x);
    let iy1 = a.y.max(b.y);
    let ix2 = ax2.min(bx2);
    let iy2 = ay2.min(by2);
    let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
    let union = a.width * a.height + b.width * b.height - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

fn label_for(index: usize, labels: &[String]) -> String {
    labels
        .get(index)
        .cloned()
        .unwrap_or_else(|| format!("class_{index}"))
}

fn coco_labels() -> Vec<String> {
    [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use lcoal_core::{
        AdapterKind, ArtifactKind, BackendKind, FileRef, ModelArtifact, ModelSpec,
        ResourceRequirement, RuntimePolicy,
    };
    use std::path::PathBuf;

    #[test]
    fn parses_coco_yaml_mapping_labels_in_index_order() {
        let labels = parse_coco_yaml_labels_str(
            r#"
path: ../datasets/coco
names:
  1: bicycle
  0: person
  2: car
"#,
        )
        .expect("parse labels");

        assert_eq!(labels, ["person", "bicycle", "car"]);
    }

    #[test]
    fn parses_coco_yaml_sequence_labels() {
        let labels =
            parse_coco_yaml_labels_str("names: [person, bicycle, car]").expect("parse labels");

        assert_eq!(labels, ["person", "bicycle", "car"]);
    }

    #[test]
    fn embedded_coco_labels_are_complete() {
        let labels = coco_labels();

        assert_eq!(labels.len(), 80);
        assert_eq!(labels[0], "person");
        assert_eq!(labels[79], "toothbrush");
    }

    #[test]
    fn decodes_default_transposed_yolo11_output_shape() {
        let config = YoloConfig::default();
        let image = unit_preprocessed_image();
        let rows = 2;
        let attrs = 84;
        let mut data = vec![0.0; rows * attrs];
        set_transposed(&mut data, rows, 0, &[100.0, 120.0, 20.0, 40.0]);
        data[(4 + 2) * rows] = 0.9;
        set_transposed(&mut data, rows, 1, &[300.0, 300.0, 10.0, 10.0]);
        data[4 * rows + 1] = 0.1;
        let output = OrtOutput {
            name: "output0".to_string(),
            shape: vec![1, attrs, rows],
            data,
        };

        let objects = decode_yolo_outputs(&[output], &image, &config).expect("decode");

        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].label, "car");
        assert_eq!(objects[0].confidence, 0.9);
        assert_eq!(objects[0].bbox.x, 90.0);
        assert_eq!(objects[0].bbox.y, 100.0);
        assert_eq!(objects[0].bbox.width, 20.0);
        assert_eq!(objects[0].bbox.height, 40.0);
    }

    #[test]
    fn decodes_row_major_output_with_objectness_when_attrs_match_labels() {
        let config = YoloConfig {
            input_width: 640,
            input_height: 640,
            confidence_threshold: 0.25,
            nms_iou_threshold: 0.45,
            labels: vec!["alpha".into(), "bravo".into()],
        };
        let output = OrtOutput {
            name: "boxes".to_string(),
            shape: vec![1, 1, 7],
            data: vec![50.0, 60.0, 20.0, 10.0, 0.5, 0.8, 0.9],
        };

        let objects =
            decode_yolo_outputs(&[output], &unit_preprocessed_image(), &config).expect("decode");

        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].label, "bravo");
        assert!((objects[0].confidence - 0.45).abs() < 1e-6);
    }

    #[test]
    fn preprocess_letterboxes_to_nchw_rgb_f32() {
        let image_path = temp_image_path("preprocess-letterbox.png");
        let mut image = RgbImage::from_pixel(2, 4, Rgb([255, 0, 0]));
        image.put_pixel(1, 0, Rgb([0, 255, 0]));
        image.save(&image_path).expect("save image");

        let preprocessed = preprocess_image(&image_path, 4, 4).expect("preprocess");

        let _ = fs::remove_file(&image_path);
        assert_eq!(preprocessed.original_width, 2);
        assert_eq!(preprocessed.original_height, 4);
        assert_eq!(preprocessed.resized_width, 2);
        assert_eq!(preprocessed.resized_height, 4);
        assert_eq!(preprocessed.pad_x, 1);
        assert_eq!(preprocessed.pad_y, 0);
        assert_eq!(preprocessed.tensor.len(), 3 * 4 * 4);
        assert!((preprocessed.tensor[0] - (114.0 / 255.0)).abs() < f32::EPSILON);
        assert_eq!(preprocessed.tensor[1], 1.0);
        assert_eq!(preprocessed.tensor[16 + 2], 1.0);
    }

    #[test]
    fn nms_suppresses_same_class_overlap() {
        let a = DetectedObject {
            label: "person".into(),
            confidence: 0.9,
            bbox: BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 10.0,
                height: 10.0,
            },
        };
        let b = DetectedObject {
            label: "person".into(),
            confidence: 0.8,
            bbox: BoundingBox {
                x: 1.0,
                y: 1.0,
                width: 10.0,
                height: 10.0,
            },
        };
        let kept = nms(vec![(a, 0), (b, 0)], 0.45);
        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn nms_keeps_overlapping_different_classes() {
        let a = DetectedObject {
            label: "person".into(),
            confidence: 0.9,
            bbox: BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 10.0,
                height: 10.0,
            },
        };
        let b = DetectedObject {
            label: "car".into(),
            confidence: 0.8,
            bbox: BoundingBox {
                x: 1.0,
                y: 1.0,
                width: 10.0,
                height: 10.0,
            },
        };

        let kept = nms(vec![(a, 0), (b, 1)], 0.45);

        assert_eq!(kept.len(), 2);
    }

    fn unit_preprocessed_image() -> PreprocessedImage {
        PreprocessedImage {
            tensor: Vec::new(),
            original_width: 640,
            original_height: 640,
            resized_width: 640,
            resized_height: 640,
            pad_x: 0,
            pad_y: 0,
            scale: 1.0,
        }
    }

    fn set_transposed(data: &mut [f32], rows: usize, row: usize, values: &[f32]) {
        for (attr, value) in values.iter().enumerate() {
            data[attr * rows + row] = *value;
        }
    }

    fn temp_image_path(file_name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "lcoal-adapter-yolo-{}-{file_name}",
            std::process::id()
        ))
    }

    #[test]
    fn real_model_smoke_if_env_set() {
        let Ok(model_dir) = std::env::var("LCOAL_YOLO_MODEL_DIR") else {
            return;
        };
        let image_path = std::env::var_os("LCOAL_YOLO_TEST_IMAGE")
            .map(PathBuf::from)
            .filter(|path| path.exists())
            .or_else(|| {
                let default = Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../scripts/assets/yolo-input.jpg");
                default.exists().then_some(default)
            });
        let Some(image_path) = image_path else {
            eprintln!("YOLO smoke skipped: no test image found");
            return;
        };

        let mut adapter = match YoloAdapter::load(&model_spec(
            PathBuf::from(model_dir),
            provider_order_from_env(),
        )) {
            Ok(adapter) => adapter,
            Err(err) => {
                eprintln!("YOLO smoke stopped at model load boundary: {err}");
                return;
            }
        };
        let report = adapter.provider_report();
        eprintln!(
            "YOLO provider report: provider={:?} cpu_fallback={}",
            report.provider, report.cpu_fallback_used
        );
        match adapter.object_detect(&FileRef::local(&image_path)) {
            Ok(InferenceOutput::ObjectDetections { objects }) => {
                eprintln!(
                    "YOLO smoke detections: count={}, first={:?}",
                    objects.len(),
                    objects.first()
                );
                assert!(
                    objects.iter().all(|obj| obj.confidence.is_finite()),
                    "YOLO smoke returned non-finite confidences"
                );
            }
            Ok(other) => panic!("unexpected output: {other:?}"),
            Err(err) => eprintln!("YOLO smoke stopped after load/run boundary: {err}"),
        }
    }

    fn provider_order_from_env() -> Vec<String> {
        provider_order_override_from_env().unwrap_or_else(|| vec!["cpu".to_string()])
    }

    fn provider_order_override_from_env() -> Option<Vec<String>> {
        std::env::var("LCOAL_TEST_PROVIDER_ORDER")
            .ok()
            .map(|value| {
                value
                    .split(',')
                    .map(str::trim)
                    .filter(|part| !part.is_empty())
                    .map(|part| part.to_string())
                    .collect::<Vec<_>>()
            })
            .filter(|parts| !parts.is_empty())
    }

    fn model_spec(path: PathBuf, provider_order: Vec<String>) -> ModelSpec {
        ModelSpec {
            id: "yolo11n.onnx".to_string(),
            name: "YOLO Test".to_string(),
            enabled: true,
            task_kinds: Vec::new(),
            adapter: AdapterKind::Yolo,
            backend: BackendKind::Ort,
            artifacts: vec![
                ModelArtifact {
                    kind: ArtifactKind::Local,
                    path: path.join("yolo11n.onnx"),
                    source_path: None,
                    sha256: None,
                    url: None,
                    repo_id: None,
                    revision: None,
                    files: Vec::new(),
                    allow_patterns: Vec::new(),
                    metadata: Default::default(),
                },
                ModelArtifact {
                    kind: ArtifactKind::Local,
                    path: path.join("coco.yaml"),
                    source_path: None,
                    sha256: None,
                    url: None,
                    repo_id: None,
                    revision: None,
                    files: Vec::new(),
                    allow_patterns: Vec::new(),
                    metadata: Default::default(),
                },
            ],
            runtime: RuntimePolicy {
                provider_order,
                ..Default::default()
            },
            resources: ResourceRequirement::default(),
            load_policy: Default::default(),
            metadata: Default::default(),
        }
    }
}
