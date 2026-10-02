use super::*;
use local_core::{AdapterKind, ArtifactKind, BackendKind, ModelArtifact};
use std::path::Path;

#[test]
fn pooling_averages_each_cell_once() {
    // 4x2 map, 2x1 grid: left half 1..4 average 2.5, right half 10s.
    let map = [1.0, 2.0, 10.0, 10.0, 3.0, 4.0, 10.0, 10.0];
    assert_eq!(
        pool(&map, 4, 2, DepthGrid { cols: 2, rows: 1 }),
        vec![2.5, 10.0]
    );
    // Uneven cells (5 px over 2 columns: 2 + 3) still cover every pixel.
    let map = [1.0, 1.0, 4.0, 4.0, 4.0];
    assert_eq!(
        pool(&map, 5, 1, DepthGrid { cols: 2, rows: 1 }),
        vec![1.0, 4.0]
    );
}

#[test]
fn metadata_sets_the_default_grid() {
    let mut metadata = std::collections::BTreeMap::new();
    assert_eq!(grid_from_metadata(&metadata), None);
    metadata.insert("grid_cols".to_string(), serde_json::json!(32));
    metadata.insert("grid_rows".to_string(), serde_json::json!(18));
    assert_eq!(
        grid_from_metadata(&metadata),
        Some(DepthGrid { cols: 32, rows: 18 })
    );
}

/// `LOCAL_DEPTH_ANYTHING_MODEL_DIR=<export dir> ORT_DYLIB_PATH=<onnxruntime>
/// cargo test -p local-adapter-depth-anything-v2 real_model -- --nocapture`
/// (`LOCAL_TEST_PROVIDER_ORDER=cuda,cpu` and `--features cuda` for the GPU).
/// The test image is a VRChat room; the player in it stands a few metres
/// away, in front of a glass door to a balcony farther back.
#[test]
fn real_model_measures_the_test_image_if_env_set() {
    let Ok(model_dir) = std::env::var("LOCAL_DEPTH_ANYTHING_MODEL_DIR") else {
        return;
    };
    let provider_order: Vec<String> = std::env::var("LOCAL_TEST_PROVIDER_ORDER")
        .map(|v| v.split(',').map(|p| p.trim().to_string()).collect())
        .unwrap_or_else(|_| vec!["cpu".to_string()]);
    let spec = ModelSpec {
        id: "depth-anything-v2-metric-indoor-small-onnx".to_string(),
        name: "Depth Anything V2 test".to_string(),
        enabled: true,
        task_kinds: Vec::new(),
        adapter: AdapterKind::DepthAnythingV2,
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
    let mut adapter = DepthAnythingAdapter::load(&spec).expect("the model loads");
    eprintln!(
        "provider {:?}, input {}x{}, max depth {}",
        adapter.provider_report().provider,
        adapter.width,
        adapter.height,
        adapter.max_depth
    );
    let image =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../scripts/assets/depth-input.jpg");
    for run in 0..5 {
        let started = std::time::Instant::now();
        let InferenceOutput::DepthMap {
            cols, rows, depth, ..
        } = adapter
            .depth(
                &FileRef::local(&image),
                Some(DepthGrid { cols: 64, rows: 36 }),
            )
            .expect("the image is measured")
        else {
            panic!("not a depth map");
        };
        let elapsed = started.elapsed();
        // The player: x 570..674, y 471..718 of 1280x720 -> cols 28..33, rows 23..35.
        let cell = |col: usize, row: usize| depth[row * cols as usize + col];
        let player: Vec<f32> = (24..34)
            .flat_map(|r| (29..33).map(move |c| (c, r)))
            .map(|(c, r)| cell(c, r))
            .collect();
        let player = player.iter().sum::<f32>() / player.len() as f32;
        // The balcony behind them: x 480..620, y 320..440 -> cols 24..31, rows 16..22.
        let wall: Vec<f32> = (16..22)
            .flat_map(|r| (24..31).map(move |c| (c, r)))
            .map(|(c, r)| cell(c, r))
            .collect();
        let wall = wall.iter().sum::<f32>() / wall.len() as f32;
        eprintln!(
            "run {run}: {elapsed:?}, {cols}x{rows}, player {player:.2} m, behind {wall:.2} m"
        );
        assert!((1.0..8.0).contains(&player), "player at {player} m");
        assert!(
            wall > player + 1.0,
            "the balcony ({wall} m) is behind the player ({player} m)"
        );
    }
}
