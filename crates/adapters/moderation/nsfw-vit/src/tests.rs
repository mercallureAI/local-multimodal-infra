use super::*;
use local_core::{AdapterKind, ArtifactKind, BackendKind, ModelArtifact};
use std::path::PathBuf;

#[test]
fn center_resize_keeps_the_middle_square() {
    // 40x20, left half black, right half white: the centre square of a
    // 10 px input is half black, half white.
    let image = RgbImage::from_fn(40, 20, |x, _| {
        if x < 20 {
            image::Rgb([0, 0, 0])
        } else {
            image::Rgb([255, 255, 255])
        }
    });
    let out = resized(&image, 10, Resize::Center);
    assert_eq!(out.dimensions(), (10, 10));
    assert!(out.get_pixel(1, 5)[0] < 30 && out.get_pixel(8, 5)[0] > 225);
    let out = resized(&image, 10, Resize::Squash);
    assert_eq!(out.dimensions(), (10, 10));
}

#[test]
fn meta_names_its_nsfw_labels() {
    let meta: NsfwMeta = serde_json::from_str(
        r#"{"size": 448, "resize": "squash", "labels": ["neutral", "low", "medium", "high"],
            "nsfw_labels": ["low", "medium", "high"]}"#,
    )
    .unwrap();
    assert_eq!(meta.resize, Resize::Squash);
    assert_eq!(meta.nsfw_labels.len(), 3);
}

/// `LOCAL_NSFW_VIT_MODEL_DIR=<export dir> ORT_DYLIB_PATH=<onnxruntime 1.30>
/// cargo test -p local-adapter-nsfw-vit real_model -- --nocapture`
/// (`LOCAL_TEST_PROVIDER_ORDER=cuda,cpu` and `--features cuda` for the GPU).
/// The repository's test photos (a street, a VRChat room) are not NSFW.
#[test]
fn real_model_passes_the_test_photos_if_env_set() {
    let Ok(model_dir) = std::env::var("LOCAL_NSFW_VIT_MODEL_DIR") else {
        return;
    };
    let provider_order: Vec<String> = std::env::var("LOCAL_TEST_PROVIDER_ORDER")
        .map(|v| v.split(',').map(|p| p.trim().to_string()).collect())
        .unwrap_or_else(|_| vec!["cpu".to_string()]);
    let spec = ModelSpec {
        id: "nsfw-test".to_string(),
        name: "NSFW classifier test".to_string(),
        enabled: true,
        task_kinds: Vec::new(),
        adapter: AdapterKind::NsfwVit,
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
            metadata: Default::default(),
        }],
        runtime: local_core::RuntimePolicy {
            provider_order,
            ..Default::default()
        },
        resources: Default::default(),
        load_policy: Default::default(),
        metadata: Default::default(),
    };
    let mut adapter = NsfwVitAdapter::load(&spec).expect("the model loads");
    eprintln!("provider {:?}", adapter.provider_report().provider);
    // The spec may narrow the NSFW labels; unknown labels are refused.
    let labels = adapter.meta.labels.clone();
    let mut narrowed = spec.clone();
    narrowed.metadata.insert(
        "nsfw_labels".to_string(),
        serde_json::json!([labels.last().unwrap()]),
    );
    let narrowed = NsfwVitAdapter::load(&narrowed).expect("narrowed labels load");
    assert_eq!(narrowed.nsfw_indices, vec![labels.len() - 1]);
    let mut unknown = spec.clone();
    unknown
        .metadata
        .insert("nsfw_labels".to_string(), serde_json::json!(["nope"]));
    assert!(NsfwVitAdapter::load(&unknown).is_err());
    let assets = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../scripts/assets");
    for name in ["yolo-input.jpg", "depth-input.jpg", "ppocr-input.png"] {
        let started = std::time::Instant::now();
        let out = adapter
            .classify_file(&FileRef::local(assets.join(name)))
            .unwrap();
        let InferenceOutput::ImageNsfw { nsfw, scores } = out else {
            panic!("not ImageNsfw");
        };
        eprintln!(
            "{name}: nsfw {nsfw:.4} {scores:?} ({} ms)",
            started.elapsed().as_millis()
        );
        assert!(nsfw < 0.2, "{name}: {nsfw}");
    }
}
