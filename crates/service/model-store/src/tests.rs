use super::*;
use local_core::{AdapterKind, BackendKind, RuntimePolicy, TaskKind};
use std::{collections::BTreeMap, io::Read as _, net::TcpListener, thread};

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
fn reports_downloaded_from_local_artifact_presence_and_tracks_queue_state() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = dir.path().join("source/model.onnx");
    fs::create_dir_all(source.parent().expect("source parent")).expect("mkdir");
    fs::write(&source, b"onnx").expect("write source");
    let store = SqliteModelStore::new(dir.path().join("data/local.db"), dir.path().join("models"))
        .expect("store");
    let spec = store
        .upsert_model(test_spec("m", source))
        .expect("normalized model");

    let missing = store.model_download_status(&spec).expect("missing status");
    assert!(!missing.downloaded);
    assert_eq!(missing.state, DownloadState::NotStarted);

    let queued = store.prepare_model_download(&spec).expect("queue status");
    assert!(!queued.downloaded);
    assert_eq!(queued.state, DownloadState::Downloading);
}

#[tokio::test]
async fn repeated_local_download_reuses_stable_artifact() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = dir.path().join("source/model.onnx");
    fs::create_dir_all(source.parent().expect("source parent")).expect("mkdir");
    fs::write(&source, b"onnx").expect("write source");
    let store = SqliteModelStore::new(dir.path().join("data/local.db"), dir.path().join("models"))
        .expect("store");
    let spec = store
        .upsert_model(test_spec("m", source.clone()))
        .expect("normalized model");

    store.download_model(&spec).await.expect("first download");
    fs::write(&source, b"changed source").expect("change source");
    let statuses = store
        .download_model(&spec)
        .await
        .expect("deduplicated download");

    assert_eq!(statuses[0].state, DownloadState::Downloaded);
    assert!(statuses[0]
        .message
        .as_deref()
        .is_some_and(|message| message.contains("no copy needed")));
    assert_eq!(
        fs::read(&spec.artifacts[0].path).expect("stable artifact"),
        b"onnx"
    );
    assert!(
        store
            .model_download_status(&spec)
            .expect("download status")
            .downloaded
    );
}

#[test]
fn unrecorded_url_file_is_not_reported_as_downloaded() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = SqliteModelStore::new(dir.path().join("data/local.db"), dir.path().join("models"))
        .expect("store");
    let mut url = artifact(ArtifactKind::Url, PathBuf::from("model.bin"));
    url.url = Some("https://example.test/model.bin".to_string());
    let spec = store
        .upsert_model(base_spec("m", vec![url]))
        .expect("upsert");
    fs::create_dir_all(spec.artifacts[0].path.parent().expect("model parent")).expect("mkdir");
    fs::write(&spec.artifacts[0].path, b"partial").expect("partial file");

    let status = store.model_download_status(&spec).expect("status");
    assert!(!status.downloaded);
    assert_eq!(status.state, DownloadState::NotStarted);
}

#[test]
fn unrecorded_url_file_with_matching_checksum_is_downloaded() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = SqliteModelStore::new(dir.path().join("data/local.db"), dir.path().join("models"))
        .expect("store");
    let mut url = artifact(ArtifactKind::Url, PathBuf::from("model.bin"));
    url.url = Some("https://example.test/model.bin".to_string());
    url.sha256 = Some(sha256_hex(b"complete"));
    let spec = store
        .upsert_model(base_spec("m", vec![url]))
        .expect("upsert");
    fs::create_dir_all(spec.artifacts[0].path.parent().expect("model parent")).expect("mkdir");
    fs::write(&spec.artifacts[0].path, b"complete").expect("complete file");

    let status = store.model_download_status(&spec).expect("status");
    assert!(status.downloaded);
    assert_eq!(status.state, DownloadState::Downloaded);
}

#[test]
fn artifact_spec_change_clears_old_download_status() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = SqliteModelStore::new(dir.path().join("data/local.db"), dir.path().join("models"))
        .expect("store");
    let mut url = artifact(ArtifactKind::Url, PathBuf::from("model.bin"));
    url.url = Some("https://example.test/old.bin".to_string());
    let spec = store
        .upsert_model(base_spec("m", vec![url]))
        .expect("upsert old");
    fs::create_dir_all(spec.artifacts[0].path.parent().expect("model parent")).expect("mkdir");
    fs::write(&spec.artifacts[0].path, b"old").expect("old file");
    store
        .status(
            "m",
            "url:0".to_string(),
            DownloadState::Downloaded,
            Some(spec.artifacts[0].path.clone()),
            None,
            Some("old download".to_string()),
        )
        .expect("record old status");

    let mut changed = spec;
    changed.artifacts[0].url = Some("https://example.test/new.bin".to_string());
    let changed = store.upsert_model(changed).expect("upsert changed");
    let status = store
        .model_download_status(&changed)
        .expect("changed status");

    assert!(!status.downloaded);
    assert_eq!(status.state, DownloadState::NotStarted);
}

#[test]
fn obsolete_download_rows_do_not_affect_current_model_status() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = SqliteModelStore::new(dir.path().join("data/local.db"), dir.path().join("models"))
        .expect("store");
    let spec = store
        .upsert_model(test_spec("m", dir.path().join("missing.onnx")))
        .expect("upsert");
    store
        .status(
            "m",
            "url:99".to_string(),
            DownloadState::Failed,
            Some(dir.path().join("removed.bin")),
            None,
            Some("obsolete failure".to_string()),
        )
        .expect("record obsolete status");

    let status = store.model_download_status(&spec).expect("status");
    assert_eq!(status.state, DownloadState::NotStarted);
    assert!(status
        .artifacts
        .iter()
        .all(|artifact| artifact.artifact != "url:99"));
}

#[test]
fn allow_pattern_metadata_failure_is_queryable() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = SqliteModelStore::new(dir.path().join("data/local.db"), dir.path().join("models"))
        .expect("store");
    let mut hf = artifact(ArtifactKind::HuggingFace, PathBuf::new());
    hf.repo_id = Some("owner/repo".to_string());
    hf.allow_patterns = vec!["*.onnx".to_string()];
    let spec = store
        .upsert_model(base_spec("m", vec![hf]))
        .expect("upsert");

    let queued = store.prepare_model_download(&spec).expect("prepare");
    assert_eq!(queued.state, DownloadState::Downloading);
    assert_eq!(queued.artifacts[0].artifact, "hf:0:__metadata__");
    store
        .fail_active_model_download("m", "metadata failed")
        .expect("record failure");
    let failed = store.model_download_status(&spec).expect("failed status");
    assert_eq!(failed.state, DownloadState::Failed);
    assert_eq!(
        failed.artifacts[0].message.as_deref(),
        Some("metadata failed")
    );
}

#[test]
fn retry_queue_preserves_completed_files_without_checksums() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = SqliteModelStore::new(dir.path().join("data/local.db"), dir.path().join("models"))
        .expect("store");
    let mut first = artifact(ArtifactKind::Url, PathBuf::from("first.bin"));
    first.url = Some("https://example.test/first.bin".to_string());
    let mut second = artifact(ArtifactKind::Url, PathBuf::from("second.bin"));
    second.url = Some("https://example.test/second.bin".to_string());
    let spec = store
        .upsert_model(base_spec("m", vec![first, second]))
        .expect("upsert");
    fs::create_dir_all(spec.artifacts[0].path.parent().expect("model parent")).expect("mkdir");
    fs::write(&spec.artifacts[0].path, b"complete").expect("completed file");
    store
        .status(
            "m",
            "url:0".to_string(),
            DownloadState::Downloaded,
            Some(spec.artifacts[0].path.clone()),
            None,
            Some("completed earlier".to_string()),
        )
        .expect("completed status");
    store
        .status(
            "m",
            "url:1".to_string(),
            DownloadState::Failed,
            Some(spec.artifacts[1].path.clone()),
            None,
            Some("failed earlier".to_string()),
        )
        .expect("failed status");

    let queued = store.prepare_model_download(&spec).expect("retry queue");
    let first = queued
        .artifacts
        .iter()
        .find(|status| status.artifact == "url:0")
        .expect("first status");
    let second = queued
        .artifacts
        .iter()
        .find(|status| status.artifact == "url:1")
        .expect("second status");
    assert_eq!(first.state, DownloadState::Downloaded);
    assert_eq!(second.state, DownloadState::Downloading);
}

#[tokio::test]
async fn partial_url_file_without_checksum_is_replaced_atomically() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let address = listener.local_addr().expect("address");
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        let mut request = [0_u8; 2048];
        let _ = stream.read(&mut request).expect("read request");
        stream
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 8\r\nConnection: close\r\n\r\ncomplete")
            .expect("write response");
    });
    let dir = tempfile::tempdir().expect("tempdir");
    let store = SqliteModelStore::new(dir.path().join("data/local.db"), dir.path().join("models"))
        .expect("store");
    let mut url = artifact(ArtifactKind::Url, PathBuf::from("model.bin"));
    url.url = Some(format!("http://{address}/model.bin"));
    let spec = store
        .upsert_model(base_spec("m", vec![url]))
        .expect("upsert");
    fs::create_dir_all(spec.artifacts[0].path.parent().expect("model parent")).expect("mkdir");
    fs::write(&spec.artifacts[0].path, b"partial").expect("partial file");

    store
        .download_model_artifacts(&spec)
        .await
        .expect("download replacement");
    server.join().expect("server");

    assert_eq!(
        fs::read(&spec.artifacts[0].path).expect("downloaded file"),
        b"complete"
    );
    assert!(!partial_download_path(&spec.artifacts[0].path)
        .expect("partial path")
        .exists());
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
