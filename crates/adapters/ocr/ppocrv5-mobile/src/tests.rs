use super::*;
use local_core::{AdapterKind, ArtifactKind, BackendKind, ModelArtifact};

fn config() -> PpocrConfig {
    PpocrConfig::default()
}

#[test]
fn the_detector_input_is_scaled_down_to_multiples_of_32() {
    assert_eq!(det_size(1280, 720, 960), (960, 544));
    assert_eq!(det_size(640, 360, 960), (640, 352));
    assert_eq!(det_size(10, 10, 960), (32, 32));
}

#[test]
fn a_region_becomes_a_grown_scaled_box() {
    // 20x10 map, one 8x3 region of 0.9 at (4, 3), plus a weak blob.
    let (w, h) = (20u32, 10u32);
    let mut prob = vec![0.0f32; (w * h) as usize];
    for y in 3..6 {
        for x in 4..12 {
            prob[(y * w + x) as usize] = 0.9;
        }
    }
    for y in 7..10 {
        for x in 15..19 {
            prob[(y * w + x) as usize] = 0.35; // above det_thresh, below box_thresh
        }
    }
    let boxes = text_boxes(&prob, w, h, &config(), (2.0, 2.0), (40, 20));
    // d = 8 * 3 * 1.5 / (2 * 11) = 1.636...: grown, then doubled.
    let corners: Vec<_> = boxes.iter().map(|b| (b.x0, b.y0, b.x1, b.y1)).collect();
    assert_eq!(corners, vec![(4, 2, 28, 16)]);
    assert!((boxes[0].score - 0.9).abs() < 1e-5);
}

#[test]
fn ctc_collapses_repeats_and_maps_the_last_class_to_a_space() {
    let dict: Vec<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
    let classes = dict.len() + 2;
    // steps: a a blank a b space
    let steps = [1, 1, 0, 1, 2, 3];
    let mut probs = vec![0.0f32; steps.len() * classes];
    for (t, &k) in steps.iter().enumerate() {
        probs[t * classes + k] = 0.8;
    }
    let (text, confidence) = ctc_decode(&probs, classes, &dict);
    assert_eq!(text, "aab ");
    assert!((confidence - 0.8).abs() < 1e-6);
    assert_eq!(ctc_decode(&vec![0.0; 3 * classes], classes, &dict).1, 0.0);
}

#[test]
fn the_recogniser_width_keeps_the_aspect_ratio_within_bounds() {
    assert_eq!(rec_width(100, 24, 1600), 200);
    assert_eq!(rec_width(10_000, 10, 1600), 1600);
    assert_eq!(rec_width(1, 100, 1600), 1);
}

#[test]
fn metadata_overrides_the_defaults() {
    let mut metadata = BTreeMap::new();
    metadata.insert("box_thresh".to_string(), serde_json::json!(0.5));
    metadata.insert("rec_batch".to_string(), serde_json::json!(4));
    let config = PpocrConfig::from_metadata(&metadata);
    assert_eq!(
        config,
        PpocrConfig {
            box_thresh: 0.5,
            rec_batch: 4,
            ..PpocrConfig::default()
        }
    );
}

#[test]
fn recognition_batches_keep_within_the_column_budget() {
    // Narrow lines (game name tags) fill whole batches; wide ones (a chat
    // screenshot) split by their padded width.
    let mut config = PpocrConfig {
        rec_batch: 32,
        ..PpocrConfig::default()
    };
    let narrow = vec![150u32; 40];
    assert_eq!(rec_batches(&narrow, &config), vec![32, 8]);
    config.rec_batch_columns = Some(6400);
    assert_eq!(rec_batches(&narrow, &config), vec![32, 8]);
    let wide = vec![1500u32; 10];
    assert_eq!(rec_batches(&wide, &config), vec![4, 4, 2]);
    // Sorted narrowest first: the batch stops where the next line would
    // widen it past the budget; a single line always goes.
    config.rec_batch_columns = Some(4000);
    let mixed = [100, 100, 100, 1600, 1600];
    assert_eq!(rec_batches(&mixed, &config), vec![3, 2]);
    config.rec_batch_columns = Some(1600);
    assert_eq!(rec_batches(&[1600, 1600], &config), vec![1, 1]);
    assert!(rec_batches(&[], &config).is_empty());
}

#[test]
fn the_column_budget_is_at_least_one_line() {
    let mut metadata = BTreeMap::new();
    metadata.insert("rec_batch_columns".to_string(), serde_json::json!(10));
    assert_eq!(
        PpocrConfig::from_metadata(&metadata).rec_batch_columns,
        Some(1600)
    );
}

fn spec(dir: PathBuf, provider_order: Vec<String>) -> ModelSpec {
    ModelSpec {
        id: "ppocrv5-mobile-onnx".to_string(),
        name: "PP-OCRv5 mobile test".to_string(),
        enabled: true,
        task_kinds: Vec::new(),
        adapter: AdapterKind::Ppocrv5Mobile,
        backend: BackendKind::Ort,
        artifacts: vec![ModelArtifact {
            kind: ArtifactKind::Local,
            path: dir,
            source_path: None,
            sha256: None,
            url: None,
            repo_id: None,
            revision: None,
            files: Vec::new(),
            allow_patterns: Vec::new(),
            metadata: Default::default(),
        }],
        runtime: Default::default(),
        resources: Default::default(),
        load_policy: Default::default(),
        metadata: Default::default(),
    }
    .with_provider_order(provider_order)
}

trait WithProviderOrder {
    fn with_provider_order(self, order: Vec<String>) -> Self;
}

impl WithProviderOrder for ModelSpec {
    fn with_provider_order(mut self, order: Vec<String>) -> Self {
        self.runtime.provider_order = order;
        self
    }
}

/// `LOCAL_PPOCR_MODEL_DIR=<dir with the det/rec ONNX and the dict>
/// ORT_DYLIB_PATH=<onnxruntime> cargo test -p local-adapter-ppocrv5-mobile real_model`
/// (`LOCAL_TEST_PROVIDER_ORDER=cuda,cpu` and `--features cuda` for the GPU).
#[test]
fn real_model_reads_the_test_image_if_env_set() {
    let Ok(model_dir) = std::env::var("LOCAL_PPOCR_MODEL_DIR") else {
        return;
    };
    let provider_order = std::env::var("LOCAL_TEST_PROVIDER_ORDER")
        .map(|v| v.split(',').map(|p| p.trim().to_string()).collect())
        .unwrap_or_else(|_| vec!["cpu".to_string()]);
    let image =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../scripts/assets/ppocr-input.png");
    let mut adapter = PpocrAdapter::load(&spec(PathBuf::from(model_dir), provider_order))
        .expect("the model loads");
    let (det, rec) = adapter.provider_report();
    eprintln!("providers: det {:?}, rec {:?}", det.provider, rec.provider);
    let started = std::time::Instant::now();
    let InferenceOutput::OcrLines { lines } = adapter
        .ocr_lines(&FileRef::local(&image))
        .expect("the image is read")
    else {
        panic!("not OcrLines");
    };
    eprintln!("first run {:?}: {lines:?}", started.elapsed());
    let texts: Vec<&str> = lines.iter().map(|line| line.text.as_str()).collect();
    assert_eq!(
        texts,
        ["xkeyC", "猫猫ねこ Neko_42", "VRChat 名牌测试 Hello"]
    );
    let started = std::time::Instant::now();
    adapter.ocr_lines(&FileRef::local(&image)).expect("again");
    eprintln!("warm run {:?}", started.elapsed());
}

/// Per-stage timings over ten runs, `LOCAL_PPOCR_TIMING=1` with the real
/// model test's variables (`LOCAL_PPOCR_IMAGE` picks another image,
/// `LOCAL_PPOCR_DET_LIMIT` the detector's longest side, `LOCAL_PPOCR_REC_BATCH`
/// the recognition batch).
#[test]
fn stage_timings_if_env_set() {
    let (Ok(model_dir), Ok(_)) = (
        std::env::var("LOCAL_PPOCR_MODEL_DIR"),
        std::env::var("LOCAL_PPOCR_TIMING"),
    ) else {
        return;
    };
    let provider_order = std::env::var("LOCAL_TEST_PROVIDER_ORDER")
        .map(|v| v.split(',').map(|p| p.trim().to_string()).collect())
        .unwrap_or_else(|_| vec!["cpu".to_string()]);
    let path = std::env::var("LOCAL_PPOCR_IMAGE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../scripts/assets/ppocr-input.png")
        });
    let mut spec = spec(PathBuf::from(model_dir), provider_order);
    if let Ok(limit) = std::env::var("LOCAL_PPOCR_DET_LIMIT") {
        spec.metadata.insert(
            "det_limit_side_len".to_string(),
            serde_json::json!(limit.parse::<u32>().unwrap()),
        );
    }
    if let Ok(batch) = std::env::var("LOCAL_PPOCR_REC_BATCH") {
        spec.metadata.insert(
            "rec_batch".to_string(),
            serde_json::json!(batch.parse::<u32>().unwrap()),
        );
    }
    let mut adapter = PpocrAdapter::load(&spec).unwrap();
    let image = image::open(&path).unwrap().to_rgb8();
    eprintln!("providers: {:?}", adapter.provider_report());
    for i in 0..10 {
        let t0 = std::time::Instant::now();
        let boxes = adapter.detect(&image).unwrap();
        let t1 = std::time::Instant::now();
        let lines = adapter.recognize(&image, &boxes).unwrap();
        let t2 = std::time::Instant::now();
        if i == 0 {
            let sizes: Vec<_> = boxes.iter().map(|b| (b.x1 - b.x0, b.y1 - b.y0)).collect();
            eprintln!("boxes (w, h): {sizes:?}");
        }
        eprintln!(
            "run {i}: det {:?} ({} boxes), rec {:?} ({} lines)",
            t1 - t0,
            boxes.len(),
            t2 - t1,
            lines.len()
        );
    }
}

#[test]
fn boxes_on_one_row_read_left_to_right() {
    let at = |x0, y0| PixelBox {
        x0,
        y0,
        x1: x0 + 20,
        y1: y0 + 10,
        score: 1.0,
    };
    // The right box's top is a pixel higher, the next row far lower.
    let mut boxes = vec![at(100, 50), at(10, 51), at(5, 80)];
    reading_order(&mut boxes);
    assert_eq!(boxes, vec![at(10, 51), at(100, 50), at(5, 80)]);
}
