use super::*;
use local_core::{AdapterKind, BackendKind, RuntimePolicy, TaskKind};
use std::collections::BTreeMap;

fn base_spec(id: &str, artifacts: Vec<ModelArtifact>) -> ModelSpec {
    ModelSpec {
        id: id.to_string(),
        name: id.to_string(),
        enabled: true,
        task_kinds: vec![TaskKind::ObjectDetect],
        adapter: AdapterKind::Yolo,
        backend: BackendKind::Ort,
        artifacts,
        runtime: RuntimePolicy::default(),
        resources: Default::default(),
        load_policy: Default::default(),
        metadata: BTreeMap::new(),
    }
}

fn artifact(kind: ArtifactKind, path: PathBuf) -> ModelArtifact {
    ModelArtifact {
        kind,
        path,
        source_path: None,
        sha256: None,
        url: None,
        repo_id: None,
        revision: None,
        files: Vec::new(),
        allow_patterns: Vec::new(),
        metadata: BTreeMap::new(),
    }
}

fn test_spec(id: &str, path: PathBuf) -> ModelSpec {
    base_spec(id, vec![artifact(ArtifactKind::Local, path)])
}

#[test]
fn persists_models_and_enabled_state() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = SqliteModelStore::new(dir.path().join("data/local.db"), dir.path().join("models"))
        .expect("store");
    let spec = test_spec("m", dir.path().join("model.onnx"));
    store.seed_models(vec![spec]).expect("seed");
    store.set_enabled("m", false).expect("disable");
    let mut changed = test_spec("m", dir.path().join("new.onnx"));
    changed.enabled = true;
    store.seed_models(vec![changed]).expect("seed preserves");
    let saved = store.get_model("m").expect("get").expect("model");
    assert!(!saved.enabled);
    assert_eq!(
        saved.artifacts[0].path,
        dir.path().join("models/m/new.onnx")
    );
    assert_eq!(
        saved.artifacts[0].source_path.as_ref(),
        Some(&dir.path().join("new.onnx"))
    );
}

#[tokio::test]
async fn records_local_download_status() {
    let dir = tempfile::tempdir().expect("tempdir");
    let artifact = dir.path().join("model.onnx");
    fs::write(&artifact, b"onnx").expect("write artifact");
    let store = SqliteModelStore::new(dir.path().join("data/local.db"), dir.path().join("models"))
        .expect("store");
    let spec = test_spec("m", artifact);
    let statuses = store.download_model(&spec).await.expect("download");
    assert_eq!(statuses[0].state, DownloadState::Downloaded);
    assert!(dir.path().join("models/m/model.onnx").exists());
}

#[test]
fn normalizes_local_absolute_source_to_stable_path() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = dir.path().join("outside/source.onnx");
    fs::create_dir_all(source.parent().unwrap()).expect("mkdir");
    fs::write(&source, b"onnx").expect("write");
    let store = SqliteModelStore::new(dir.path().join("data/local.db"), dir.path().join("models"))
        .expect("store");
    let saved = store
        .upsert_model(test_spec("m", source.clone()))
        .expect("upsert");
    assert_eq!(
        saved.artifacts[0].path,
        dir.path().join("models/m/source.onnx")
    );
    assert_eq!(saved.artifacts[0].source_path.as_ref(), Some(&source));
}

#[test]
fn normalizes_url_relative_path_under_model_dir() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = SqliteModelStore::new(dir.path().join("data/local.db"), dir.path().join("models"))
        .expect("store");
    let mut url = artifact(ArtifactKind::Url, PathBuf::from("labels/coco.yaml"));
    url.url = Some("https://example.test/not-used.yaml".to_string());
    let saved = store
        .upsert_model(base_spec("m", vec![url]))
        .expect("upsert");
    assert_eq!(
        saved.artifacts[0].path,
        dir.path().join("models/m/labels/coco.yaml")
    );
    assert!(saved.artifacts[0].source_path.is_none());
}

#[test]
fn rejects_already_stable_local_path_with_parent_traversal() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = SqliteModelStore::new(dir.path().join("data/local.db"), dir.path().join("models"))
        .expect("store");
    let escaping = dir.path().join("models/m/../escape.onnx");
    let err = store.upsert_model(test_spec("m", escaping)).unwrap_err();
    assert!(err.to_string().contains("parent traversal"));
}

#[test]
fn rejects_already_stable_url_path_with_parent_traversal() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = SqliteModelStore::new(dir.path().join("data/local.db"), dir.path().join("models"))
        .expect("store");
    let mut url = artifact(
        ArtifactKind::Url,
        dir.path().join("models/m/subdir/../../escape.yaml"),
    );
    url.url = Some("https://example.test/escape.yaml".to_string());
    let err = store.upsert_model(base_spec("m", vec![url])).unwrap_err();
    assert!(err.to_string().contains("parent traversal"));
}

#[test]
fn rejects_url_stable_model_root_without_file_suffix() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = SqliteModelStore::new(dir.path().join("data/local.db"), dir.path().join("models"))
        .expect("store");
    let mut url = artifact(ArtifactKind::Url, dir.path().join("models/m"));
    url.url = Some("https://example.test/labels.yaml".to_string());
    let err = store.upsert_model(base_spec("m", vec![url])).unwrap_err();
    assert!(err
        .to_string()
        .contains("must name a file below stable model root"));
}

#[test]
fn rejects_url_absolute_destination_path() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = SqliteModelStore::new(dir.path().join("data/local.db"), dir.path().join("models"))
        .expect("store");
    let mut url = artifact(ArtifactKind::Url, dir.path().join("outside.yaml"));
    url.url = Some("https://example.test/outside.yaml".to_string());
    let err = store.upsert_model(base_spec("m", vec![url])).unwrap_err();
    assert!(err.to_string().contains("URL artifact destination path"));
}

#[test]
fn normalizes_hf_prefilled_absolute_path_to_stable_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = SqliteModelStore::new(dir.path().join("data/local.db"), dir.path().join("models"))
        .expect("store");
    let mut hf = artifact(
        ArtifactKind::HuggingFace,
        dir.path().join("outside/model.onnx"),
    );
    hf.repo_id = Some("owner/repo".to_string());
    hf.files = vec!["nested/model.onnx".to_string()];
    let saved = store
        .upsert_model(base_spec("m", vec![hf]))
        .expect("upsert");
    assert_eq!(
        saved.artifacts[0].path,
        dir.path().join("models/m/nested/model.onnx")
    );
}

#[test]
fn rejects_hf_file_path_traversal() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = SqliteModelStore::new(dir.path().join("data/local.db"), dir.path().join("models"))
        .expect("store");
    let mut hf = artifact(ArtifactKind::HuggingFace, PathBuf::new());
    hf.repo_id = Some("owner/repo".to_string());
    hf.files = vec!["../evil.onnx".to_string()];
    let err = store.upsert_model(base_spec("m", vec![hf])).unwrap_err();
    assert!(err.to_string().contains("parent traversal"));
}

#[test]
fn hf_single_file_without_extension_targets_normalized_path() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = SqliteModelStore::new(dir.path().join("data/local.db"), dir.path().join("models"))
        .expect("store");
    let mut hf = artifact(ArtifactKind::HuggingFace, PathBuf::new());
    hf.repo_id = Some("owner/repo".to_string());
    hf.files = vec!["LICENSE".to_string()];
    let saved = store
        .upsert_model(base_spec("m", vec![hf]))
        .expect("upsert");
    let normalized = &saved.artifacts[0];
    assert_eq!(normalized.path, dir.path().join("models/m/LICENSE"));
    assert_eq!(hf_target_path(normalized, "LICENSE"), normalized.path);
    assert_ne!(
        hf_target_path(normalized, "LICENSE"),
        normalized.path.join("LICENSE")
    );
}

#[test]
fn simple_glob_matches_hf_allow_patterns() {
    assert!(glob_match("*.onnx", "model.onnx"));
    assert!(glob_match("onnx/*.onnx", "onnx/model.onnx"));
    assert!(glob_match("tokenizer.?son", "tokenizer.json"));
    assert!(!glob_match("*.onnx", "model.bin"));
}
