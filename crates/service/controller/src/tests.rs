use super::*;
use axum::{body::Bytes, http::HeaderMap, routing::post, Router};
use local_core::{
    AdapterKind, ArtifactKind, BackendKind, DeviceSpec, FileRef, InferenceInput, LoadPolicy,
    ModelArtifact, ResourceRequirement, ResourceSnapshot, RuntimePolicy, TaskKind,
};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};
use uuid::Uuid;

#[tokio::test]
async fn admin_model_download_is_async_queryable_and_deduplicated() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = dir.path().join("source/model.onnx");
    fs::create_dir_all(source.parent().expect("source parent")).expect("mkdir");
    fs::write(&source, b"onnx").expect("write source");
    let store = SqliteModelStore::new(dir.path().join("data/local.db"), dir.path().join("models"))
        .expect("store");
    let mut spec = test_model();
    spec.artifacts = vec![ModelArtifact {
        kind: ArtifactKind::Local,
        path: source,
        source_path: None,
        sha256: None,
        url: None,
        repo_id: None,
        revision: None,
        files: Vec::new(),
        allow_patterns: Vec::new(),
        metadata: BTreeMap::new(),
    }];
    let spec = store.upsert_model(spec).expect("upsert model");
    let controller =
        ControllerState::with_store(ModelRegistry::from_models(vec![spec.clone()]), store);

    let before = AdminApi::get_model(&controller, &spec.id)
        .await
        .expect("model info before download");
    assert!(!before.downloaded);
    assert_eq!(before.download_state, DownloadState::NotStarted);

    let queued = AdminApi::download_model(&controller, &spec.id)
        .await
        .expect("queue download");
    assert!(queued.accepted);
    assert!(!queued.deduplicated);
    assert_eq!(queued.status.state, DownloadState::Downloading);

    let duplicate = AdminApi::download_model(&controller, &spec.id)
        .await
        .expect("deduplicate active download");
    assert!(!duplicate.accepted);
    assert!(duplicate.deduplicated);
    assert!(!duplicate.status.downloaded);
    assert_eq!(duplicate.status.state, DownloadState::Downloading);

    let update_error = AdminApi::upsert_model(&controller, spec.clone())
        .await
        .expect_err("active model download must serialize model updates");
    assert!(update_error
        .to_string()
        .contains("while its artifacts are downloading"));

    let status = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let status = AdminApi::get_model_download_status(&controller, &spec.id)
                .await
                .expect("download status");
            if status.downloaded {
                break status;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("background download timeout");
    assert_eq!(status.state, DownloadState::Downloaded);

    let completed_duplicate = AdminApi::download_model(&controller, &spec.id)
        .await
        .expect("deduplicate completed download");
    assert!(!completed_duplicate.accepted);
    assert!(completed_duplicate.deduplicated);

    let after = AdminApi::list_models(&controller)
        .await
        .expect("model list after download");
    assert!(after[0].downloaded);
    assert_eq!(after[0].download_state, DownloadState::Downloaded);
}

#[tokio::test]
async fn mcp_dispatch_forwards_to_registered_worker_internal_infer() {
    let seen = Arc::new(Mutex::new(Vec::<InferenceTask>::new()));
    let app = Router::new()
        .route("/internal/infer", post(capture_worker_task))
        .with_state(seen.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind fake worker");
    let base_url = format!("http://{}", listener.local_addr().expect("local addr"));
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("fake worker serve");
    });

    let controller = ControllerState::new(ModelRegistry::from_models(vec![test_model()]));
    controller
        .register_worker(WorkerRegistration {
            node_id: "fake-worker".to_string(),
            base_url,
            registration_token: None,
            supported_backends: vec![BackendKind::Ort],
            supported_adapters: vec![AdapterKind::Yolo],
            resources: ResourceSnapshot {
                cpu_cores: 4,
                total_ram_mb: 8192,
                used_ram_mb: 1024,
                devices: DeviceSpec::default(),
                captured_at: Utc::now(),
            },
        })
        .await
        .expect("register worker");

    let task = InferenceTask::new(
        TaskKind::ObjectDetect,
        Some("yolo11n.onnx".to_string()),
        InferenceInput::ObjectDetect {
            image: FileRef::local("image.jpg"),
        },
    );
    let output = local_api_mcp_infer::InferenceApi::dispatch(&controller, task.clone())
        .await
        .expect("dispatch");

    assert!(matches!(output, InferenceOutput::ObjectDetections { .. }));
    let seen = seen.lock().expect("seen lock");
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].id, task.id);
    assert_eq!(seen[0].kind, TaskKind::ObjectDetect);
    assert_eq!(seen[0].model_id.as_deref(), Some("yolo11n.onnx"));
}

#[tokio::test]
async fn dispatch_without_worker_returns_explicit_controller_error() {
    let controller = ControllerState::new(ModelRegistry::from_models(vec![test_model()]));
    let task = InferenceTask::new(
        TaskKind::ObjectDetect,
        Some("yolo11n.onnx".to_string()),
        InferenceInput::ObjectDetect {
            image: FileRef::local("image.jpg"),
        },
    );

    let err = local_api_mcp_infer::InferenceApi::dispatch(&controller, task)
        .await
        .expect_err("controller should not load models locally");

    let msg = err.to_string();
    assert!(msg.contains("no registered worker"), "{msg}");
    assert!(msg.contains("controller does not load models"), "{msg}");
}

#[tokio::test]
async fn worker_registration_requires_shared_token_and_returns_session_token() {
    let controller = ControllerState::new_with_options(
        ModelRegistry::from_models(vec![test_model()]),
        None,
        ControllerOptions {
            worker_registration_token: Some("shared".to_string()),
            public_base_url: "http://127.0.0.1:17890".to_string(),
            data_dir: std::env::temp_dir(),
            upload_signing_secret: Some("test-upload-secret".to_string()),
            admin_token: None,
            mcp_infer_tokens: Vec::new(),
            asset_cleanup_interval: None,
        },
    );

    let err = controller
        .register_worker(test_registration(Some("wrong")))
        .await
        .expect_err("invalid token must be rejected");
    assert!(err.to_string().contains("registration token"), "{err}");

    let response = controller
        .register_worker(test_registration(Some("shared")))
        .await
        .expect("valid token registers");
    assert!(!response.session_token.is_empty());
    assert!(response.status.registration.registration_token.is_none());
}

#[tokio::test]
async fn generic_task_validation_failure_is_persisted_failed_not_running() {
    let controller = ControllerState::new(ModelRegistry::from_models(vec![test_model()]));
    let created = controller
        .create_generic_task(CreateTaskRequest {
            task_kind: TaskKind::TtsSynthesize,
            model: Some("indextts-1.5-onnx".to_string()),
            model_id: None,
            files: Vec::new(),
            params: std::collections::BTreeMap::new(),
            wait_timeout_sec: None,
        })
        .await
        .expect("create task");

    let err = controller
        .start_generic_task(StartTaskRequest {
            task_id: created.task_id.clone(),
            wait: true,
            timeout_sec: None,
        })
        .await
        .expect_err("missing tts text must fail validation");

    assert!(err.to_string().contains("params.text"), "{err}");
    let status = controller
        .load_task(&created.task_id)
        .await
        .expect("status");
    assert_eq!(status.state, GenericTaskState::Failed);
    assert!(status.error.unwrap_or_default().contains("params.text"));
}

#[tokio::test]
async fn optional_unuploaded_tts_reference_audio_is_not_used_as_input() {
    let seen = Arc::new(Mutex::new(Vec::<InferenceTask>::new()));
    let app = Router::new()
        .route("/internal/infer", post(capture_worker_task))
        .with_state(seen.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind fake worker");
    let base_url = format!("http://{}", listener.local_addr().expect("local addr"));
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("fake worker serve");
    });

    let controller = ControllerState::new_with_options(
        ModelRegistry::from_models(vec![test_tts_model()]),
        None,
        ControllerOptions {
            worker_registration_token: None,
            public_base_url: "http://127.0.0.1:17890".to_string(),
            data_dir: std::env::temp_dir().join(format!(
                "local-controller-test-{}",
                Uuid::new_v4().as_simple()
            )),
            upload_signing_secret: Some("test-upload-secret".to_string()),
            admin_token: None,
            mcp_infer_tokens: Vec::new(),
            asset_cleanup_interval: None,
        },
    );
    controller
        .register_worker(WorkerRegistration {
            node_id: "fake-tts-worker".to_string(),
            base_url,
            registration_token: None,
            supported_backends: vec![BackendKind::Ort],
            supported_adapters: vec![AdapterKind::IndexTts],
            resources: ResourceSnapshot {
                cpu_cores: 4,
                total_ram_mb: 8192,
                used_ram_mb: 1024,
                devices: DeviceSpec::default(),
                captured_at: Utc::now(),
            },
        })
        .await
        .expect("register worker");

    let created = controller
        .create_generic_task(CreateTaskRequest {
            task_kind: TaskKind::TtsSynthesize,
            model: Some("indextts-1.5-onnx".to_string()),
            model_id: None,
            files: vec![local_core::TaskFileRequirement {
                name: "optional-reference.wav".to_string(),
                mime: Some("audio/wav".to_string()),
                role: Some("reference_audio".to_string()),
                asset_uri: None,
                required: false,
            }],
            params: BTreeMap::from([("text".to_string(), json!("hello without reference"))]),
            wait_timeout_sec: None,
        })
        .await
        .expect("create task");
    assert_eq!(created.state, GenericTaskState::Ready);
    assert!(!created.uploads[0].uploaded);
    assert!(created.uploads[0].asset_uri.is_some());

    let task = controller
        .inference_task_from_status(&created)
        .await
        .expect("build inference task without resolving optional missing asset");
    match &task.input {
        InferenceInput::TtsSynthesize {
            reference_audio, ..
        } => assert!(reference_audio.is_none()),
        other => panic!("unexpected input: {other:?}"),
    }

    let result = controller
        .start_generic_task(StartTaskRequest {
            task_id: created.task_id.clone(),
            wait: true,
            timeout_sec: None,
        })
        .await
        .expect("start task should not resolve optional missing reference asset");
    assert_eq!(result.state, GenericTaskState::Succeeded);
    let seen = seen.lock().expect("seen lock");
    assert_eq!(seen.len(), 1);
    match &seen[0].input {
        InferenceInput::TtsSynthesize {
            reference_audio, ..
        } => assert!(reference_audio.is_none()),
        other => panic!("unexpected forwarded input: {other:?}"),
    }
    cleanup_test_controller(&controller);
}

#[tokio::test]
async fn signed_asset_upload_rejects_unsigned_ttl_params_and_persists_signed_expiration() {
    let controller = test_controller_with_temp_data_dir();
    let signed = controller
        .sign_asset_url_public(AssetSignItem {
            operation: AssetUrlOperation::Upload,
            kind: Some(AssetKind::Material),
            path: Some("user/foo.txt".to_string()),
            uri: None,
            content_type: None,
            ttl_sec: Some(DEFAULT_MATERIAL_ASSET_TTL_SECS),
            expires: None,
            url_ttl_sec: Some(DEFAULT_UPLOAD_URL_TTL_SECS),
        })
        .expect("sign upload URL");
    let query = signed_url_query(&signed.signed_url);
    assert!(query.contains_key("asset_expires"));
    assert!(!query.contains_key("ttl_sec"));
    assert!(!query.contains_key("expires"));

    controller
        .verify_asset_signature(
            AssetUrlOperation::Upload,
            AssetKind::Material,
            "user/foo.txt",
            &query,
        )
        .expect("original signature verifies");
    let expected_asset_expires = query
        .get("asset_expires")
        .expect("asset_expires")
        .parse::<i64>()
        .expect("asset_expires timestamp");
    let record = controller
        .upload_asset_bytes(
            AssetKind::Material,
            "user/foo.txt".to_string(),
            query.clone(),
            HeaderMap::new(),
            Bytes::from_static(b"hello"),
        )
        .await
        .expect("upload asset bytes");
    assert_eq!(
        record.expires_at.expect("persisted expiration").timestamp(),
        expected_asset_expires
    );

    for (key, value) in [("expires", "never"), ("ttl_sec", "999999")] {
        let mut tampered = query.clone();
        tampered.insert(key.to_string(), value.to_string());
        let err = controller
            .verify_asset_signature(
                AssetUrlOperation::Upload,
                AssetKind::Material,
                "user/foo.txt",
                &tampered,
            )
            .expect_err("unsigned TTL-affecting params must be rejected");
        assert!(err.to_string().contains("asset_expires"), "{err}");
    }
    let mut no_expiry_alias = query.clone();
    no_expiry_alias.insert("asset_expires".to_string(), "no-expiry".to_string());
    controller
        .verify_asset_signature(
            AssetUrlOperation::Upload,
            AssetKind::Material,
            "user/foo.txt",
            &no_expiry_alias,
        )
        .expect_err("only signed asset_expires=never may disable asset expiration");
    cleanup_test_controller(&controller);
}

#[test]
fn public_asset_signing_rejects_reserved_upload_paths_but_task_internal_signing_works() {
    let controller = test_controller_with_temp_data_dir();
    for (kind, path) in [
        (AssetKind::Material, "tasks/task-id/inputs/audio.wav"),
        (AssetKind::Material, "system/model-cache.bin"),
        (AssetKind::Artifact, "mcp/results/hash.txt"),
        (AssetKind::Material, ".metadata/material/hidden.json"),
    ] {
        let err = controller
            .sign_assets_batch(AssetSignRequest {
                items: vec![AssetSignItem {
                    operation: AssetUrlOperation::Upload,
                    kind: Some(kind),
                    path: Some(path.to_string()),
                    uri: None,
                    content_type: None,
                    ttl_sec: None,
                    expires: None,
                    url_ttl_sec: None,
                }],
            })
            .expect_err("reserved public upload path must be rejected");
        assert!(err.to_string().contains("reserved asset paths"), "{err}");
    }

    let internal = controller.asset_upload_url(
        AssetKind::Material,
        "tasks/task-id/inputs/audio.wav",
        Some("audio/wav"),
    );
    assert!(internal.contains("/assets/material/tasks/task-id/inputs/audio.wav?"));
    let query = signed_url_query(&internal);
    assert!(query.contains_key("asset_expires"));
    assert_eq!(
        query.get("capability").map(String::as_str),
        Some("task_upload")
    );
    assert!(query.contains_key("sig"));
    controller
        .verify_asset_signature(
            AssetUrlOperation::Upload,
            AssetKind::Material,
            "tasks/task-id/inputs/audio.wav",
            &query,
        )
        .expect("internal reserved upload signature verifies");

    let mut missing_capability = query.clone();
    missing_capability.remove("capability");
    controller
        .verify_asset_signature(
            AssetUrlOperation::Upload,
            AssetKind::Material,
            "tasks/task-id/inputs/audio.wav",
            &missing_capability,
        )
        .expect_err("reserved uploads require internal capability");
    cleanup_test_controller(&controller);
}

#[tokio::test]
async fn create_task_existing_required_asset_uri_is_ready_and_new_material_ttl_is_default() {
    let controller = test_controller_with_temp_data_dir();
    let created = controller
        .create_generic_task(CreateTaskRequest {
            task_kind: TaskKind::AsrTranscribe,
            model: None,
            model_id: None,
            files: vec![local_core::TaskFileRequirement {
                name: "audio.wav".to_string(),
                mime: Some("audio/wav".to_string()),
                role: Some("audio".to_string()),
                asset_uri: Some("assets://material/user/existing-audio.wav".to_string()),
                required: true,
            }],
            params: BTreeMap::new(),
            wait_timeout_sec: None,
        })
        .await
        .expect("create task with existing asset");
    assert_eq!(created.state, GenericTaskState::Ready);
    assert!(created.uploads[0].uploaded);
    assert!(created.uploads[0].upload_url.is_empty());

    let before = Utc::now().timestamp();
    let needs_upload = controller
        .create_generic_task(CreateTaskRequest {
            task_kind: TaskKind::AsrTranscribe,
            model: None,
            model_id: None,
            files: vec![local_core::TaskFileRequirement {
                name: "fresh.wav".to_string(),
                mime: Some("audio/wav".to_string()),
                role: Some("audio".to_string()),
                asset_uri: None,
                required: true,
            }],
            params: BTreeMap::new(),
            wait_timeout_sec: None,
        })
        .await
        .expect("create task requiring upload");
    let after = Utc::now().timestamp();
    assert_eq!(needs_upload.state, GenericTaskState::WaitingForUploads);
    let query = signed_url_query(&needs_upload.uploads[0].upload_url);
    let asset_expires = query
        .get("asset_expires")
        .expect("asset_expires")
        .parse::<i64>()
        .expect("asset_expires timestamp");
    let url_expires = query
        .get("url_expires")
        .expect("url_expires")
        .parse::<i64>()
        .expect("url_expires timestamp");
    assert!(!query.contains_key("ttl_sec"));
    assert!(!query.contains_key("expires"));
    assert!(
        (before + DEFAULT_MATERIAL_ASSET_TTL_SECS..=after + DEFAULT_MATERIAL_ASSET_TTL_SECS)
            .contains(&asset_expires),
        "asset_expires={asset_expires} before={before} after={after}"
    );
    assert!(
        (before + DEFAULT_UPLOAD_URL_TTL_SECS..=after + DEFAULT_UPLOAD_URL_TTL_SECS)
            .contains(&url_expires),
        "url_expires={url_expires} before={before} after={after}"
    );
    cleanup_test_controller(&controller);
}

#[tokio::test]
async fn small_task_input_reuses_existing_material_and_refreshes_expiry() {
    let controller = test_controller_with_temp_data_dir();
    let bytes = Bytes::from_static(b"reusable audio bytes");
    let existing = controller
        .assets
        .put(
            AssetKind::Material,
            "library/reference.wav",
            None,
            Some(Utc::now() + chrono::Duration::seconds(5)),
            bytes.as_ref(),
        )
        .expect("put reusable material");
    let created = controller
        .create_generic_task(CreateTaskRequest {
            task_kind: TaskKind::AsrTranscribe,
            model: None,
            model_id: None,
            files: vec![local_core::TaskFileRequirement {
                name: "copy.wav".to_string(),
                mime: Some("audio/wav".to_string()),
                role: Some("audio".to_string()),
                asset_uri: None,
                required: true,
            }],
            params: BTreeMap::new(),
            wait_timeout_sec: None,
        })
        .await
        .expect("create task requiring reusable upload");
    let slot = &created.uploads[0];
    let requested_uri = slot.asset_uri.clone().expect("requested asset uri");
    let (_, requested_path) = parse_asset_uri(&requested_uri).expect("parse requested uri");
    let query = signed_url_query(&slot.upload_url);
    let signed_expiry = query
        .get("asset_expires")
        .expect("asset_expires")
        .parse::<i64>()
        .expect("asset expiry timestamp");

    let record = controller
        .upload_asset_bytes(
            AssetKind::Material,
            requested_path.clone(),
            query,
            HeaderMap::new(),
            bytes,
        )
        .await
        .expect("reuse task input upload");

    assert_eq!(record.uri, existing.uri);
    assert_eq!(
        record.expires_at.expect("refreshed expiry").timestamp(),
        signed_expiry
    );
    assert_eq!(record.content_type.as_deref(), Some("audio/wav"));
    assert!(!controller
        .data_dir
        .join("assets")
        .join("material")
        .join(&requested_path)
        .exists());
    let status = controller
        .load_task(&created.task_id)
        .await
        .expect("load retargeted task");
    assert_eq!(status.state, GenericTaskState::Ready);
    assert!(status.uploads[0].uploaded);
    assert!(status.uploads[0].uploaded_at.is_some());
    assert_eq!(
        status.uploads[0].asset_uri.as_deref(),
        Some(existing.uri.as_str())
    );

    cleanup_test_controller(&controller);
}

#[test]
fn fast_hash_reuse_is_limited_to_small_material_task_inputs() {
    let path = "tasks/task-id/inputs/audio.wav";
    assert!(is_fast_hash_material_input(AssetKind::Material, path, 0));
    assert!(is_fast_hash_material_input(
        AssetKind::Material,
        path,
        MATERIAL_FAST_HASH_MAX_BYTES
    ));
    assert!(!is_fast_hash_material_input(
        AssetKind::Material,
        path,
        MATERIAL_FAST_HASH_MAX_BYTES + 1
    ));
    assert!(!is_fast_hash_material_input(
        AssetKind::Artifact,
        path,
        1024
    ));
    assert!(!is_fast_hash_material_input(
        AssetKind::Material,
        "user/audio.wav",
        1024
    ));
}

#[test]
fn tts_local_output_is_registered_as_downloadable_artifact_asset() {
    let controller = test_controller_with_temp_data_dir();
    std::fs::create_dir_all(&controller.data_dir).expect("create temp data dir");
    let source = controller.data_dir.join("speech.wav");
    std::fs::write(&source, b"RIFF fake wav").expect("write fake wav");

    let before = Utc::now().timestamp();
    let output = controller
        .register_output_assets(
            Uuid::new_v4(),
            "task-output-test",
            InferenceOutput::TtsAudio {
                audio: FileRef {
                    path: Some(source.clone()),
                    mime: Some("audio/wav".to_string()),
                    ..FileRef::default()
                },
            },
        )
        .expect("register output assets");
    let after = Utc::now().timestamp();
    let InferenceOutput::TtsAudio { audio } = output else {
        panic!("expected TtsAudio output");
    };
    assert_eq!(audio.path.as_deref(), Some(source.as_path()));
    assert!(audio
        .uri
        .as_deref()
        .is_some_and(|uri| uri.starts_with("assets://artifact/tasks/task-output-test/outputs/")));
    assert!(audio
        .url
        .as_deref()
        .is_some_and(|url| url.contains("action=download") && url.contains("sig=")));
    assert!(audio.sha256.is_some());
    let uri = audio.uri.as_deref().expect("artifact URI");
    let record = controller.assets.get(uri).expect("artifact record");
    let artifact_expires = record.expires_at.expect("artifact expiration").timestamp();
    assert!(
        (before + DEFAULT_ARTIFACT_ASSET_TTL_SECS..=after + DEFAULT_ARTIFACT_ASSET_TTL_SECS)
            .contains(&artifact_expires),
        "artifact_expires={artifact_expires} before={before} after={after}"
    );
    cleanup_test_controller(&controller);
}

#[test]
fn tts_output_asset_source_read_failure_is_returned() {
    let controller = test_controller_with_temp_data_dir();
    let missing = controller.data_dir.join("missing-output.wav");

    let error = controller
        .register_output_assets(
            Uuid::new_v4(),
            "task-missing-output",
            InferenceOutput::TtsAudio {
                audio: FileRef {
                    path: Some(missing),
                    mime: Some("audio/wav".to_string()),
                    ..FileRef::default()
                },
            },
        )
        .expect_err("missing source must fail registration");

    assert!(error.to_string().contains("missing-output.wav"), "{error}");
}

#[test]
fn controller_startup_runs_initial_asset_cleanup() {
    let data_dir = std::env::temp_dir().join(format!(
        "local-controller-test-{}",
        Uuid::new_v4().as_simple()
    ));
    let store = AssetsStore::new(&data_dir);
    let expired = store
        .put(
            AssetKind::Material,
            "user/stale.txt",
            Some("text/plain".to_string()),
            Some(Utc::now() - chrono::Duration::seconds(1)),
            b"stale",
        )
        .expect("write expired asset");

    let controller = ControllerState::new_with_options(
        ModelRegistry::from_models(vec![test_model()]),
        None,
        ControllerOptions {
            worker_registration_token: None,
            public_base_url: "http://127.0.0.1:17890".to_string(),
            data_dir,
            upload_signing_secret: Some("test-upload-secret".to_string()),
            admin_token: None,
            mcp_infer_tokens: Vec::new(),
            asset_cleanup_interval: None,
        },
    );

    let listed = controller
        .assets
        .list(&AssetListQuery {
            include_expired: true,
            ..AssetListQuery::default()
        })
        .expect("list including expired after startup cleanup");
    assert!(
        listed.assets.iter().all(|record| record.uri != expired.uri),
        "startup cleanup should remove expired asset"
    );
    cleanup_test_controller(&controller);
}

#[test]
fn controller_manifest_has_no_runtime_backend_or_adapter_deps() {
    let manifest = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"),
    )
    .expect("read controller manifest");

    for forbidden in [
        "local-runtime",
        "local-backend-ort",
        "local-adapter-yolo",
        "local-adapter-sensevoice-asr",
    ] {
        assert!(
            !manifest.contains(forbidden),
            "controller must not depend on {forbidden}"
        );
    }
}

async fn capture_worker_task(
    State(seen): State<Arc<Mutex<Vec<InferenceTask>>>>,
    Json(task): Json<InferenceTask>,
) -> impl IntoResponse {
    seen.lock().expect("seen lock").push(task);
    Json(InferenceOutput::ObjectDetections {
        objects: Vec::new(),
    })
}

fn test_model() -> ModelSpec {
    ModelSpec {
        id: "yolo11n.onnx".to_string(),
        name: "YOLO11n ONNX".to_string(),
        enabled: true,
        task_kinds: vec![TaskKind::ObjectDetect],
        adapter: AdapterKind::Yolo,
        backend: BackendKind::Ort,
        artifacts: Vec::new(),
        runtime: RuntimePolicy {
            provider_order: vec!["cpu".to_string()],
            max_concurrency: 1,
            idle_ttl_sec: 300,
        },
        resources: ResourceRequirement::default(),
        load_policy: LoadPolicy::default(),
        metadata: BTreeMap::new(),
    }
}

fn test_tts_model() -> ModelSpec {
    ModelSpec {
        id: "indextts-1.5-onnx".to_string(),
        name: "IndexTTS 1.5 ONNX".to_string(),
        enabled: true,
        task_kinds: vec![TaskKind::TtsSynthesize],
        adapter: AdapterKind::IndexTts,
        backend: BackendKind::Ort,
        artifacts: Vec::new(),
        runtime: RuntimePolicy {
            provider_order: vec!["cpu".to_string()],
            max_concurrency: 1,
            idle_ttl_sec: 300,
        },
        resources: ResourceRequirement::default(),
        load_policy: LoadPolicy::default(),
        metadata: BTreeMap::new(),
    }
}

fn test_registration(token: Option<&str>) -> WorkerRegistration {
    WorkerRegistration {
        node_id: "fake-worker".to_string(),
        base_url: "http://127.0.0.1:0".to_string(),
        registration_token: token.map(str::to_string),
        supported_backends: vec![BackendKind::Ort],
        supported_adapters: vec![AdapterKind::Yolo],
        resources: ResourceSnapshot {
            cpu_cores: 4,
            total_ram_mb: 8192,
            used_ram_mb: 1024,
            devices: DeviceSpec::default(),
            captured_at: Utc::now(),
        },
    }
}

fn test_controller_with_temp_data_dir() -> ControllerState {
    let data_dir = std::env::temp_dir().join(format!(
        "local-controller-test-{}",
        Uuid::new_v4().as_simple()
    ));
    ControllerState::new_with_options(
        ModelRegistry::from_models(vec![test_model()]),
        None,
        ControllerOptions {
            worker_registration_token: None,
            public_base_url: "http://127.0.0.1:17890".to_string(),
            data_dir,
            upload_signing_secret: Some("test-upload-secret".to_string()),
            admin_token: None,
            mcp_infer_tokens: Vec::new(),
            asset_cleanup_interval: None,
        },
    )
}

fn cleanup_test_controller(controller: &ControllerState) {
    if controller
        .data_dir
        .file_name()
        .and_then(|value| value.to_str())
        .is_some_and(|value| value.starts_with("local-controller-test-"))
    {
        let _ = std::fs::remove_dir_all(&controller.data_dir);
    }
}

fn signed_url_query(url: &str) -> BTreeMap<String, String> {
    url.split_once('?')
        .expect("signed URL query")
        .1
        .split('&')
        .map(|part| {
            let (key, value) = part.split_once('=').expect("query pair");
            (key.to_string(), percent_decode(value))
        })
        .collect()
}

fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut idx = 0;
    while idx < bytes.len() {
        if bytes[idx] == b'%' && idx + 2 < bytes.len() {
            if let Ok(hex) = std::str::from_utf8(&bytes[idx + 1..idx + 3]) {
                if let Ok(byte) = u8::from_str_radix(hex, 16) {
                    out.push(byte);
                    idx += 3;
                    continue;
                }
            }
        }
        out.push(bytes[idx]);
        idx += 1;
    }
    String::from_utf8(out).expect("query value UTF-8")
}

/// What the fake worker saw of an `ocr.lines` task: the task, and the image
/// file it names, read while the request is in flight.
type SeenOcr = Arc<Mutex<Vec<(InferenceTask, Option<Vec<u8>>)>>>;

async fn answer_ocr_lines(
    State(seen): State<SeenOcr>,
    Json(task): Json<InferenceTask>,
) -> impl IntoResponse {
    let image = match &task.input {
        InferenceInput::OcrLines { image } => image.path.as_ref().and_then(|p| fs::read(p).ok()),
        _ => None,
    };
    seen.lock().expect("seen lock").push((task, image));
    Json(InferenceOutput::OcrLines {
        lines: vec![local_core::OcrLine {
            text: "xkeyC".to_string(),
            confidence: 0.5,
            bbox: local_core::BoundingBox {
                x: 1.0,
                y: 2.0,
                width: 3.0,
                height: 4.0,
            },
        }],
    })
}

/// A controller serving `model` (of `adapter`), backed by the fake `worker`
/// router; returns its base URL and the data directory to keep alive.
async fn controller_with_fake_worker(
    model: &str,
    adapter: AdapterKind,
    kind: TaskKind,
    worker: Router,
) -> (String, tempfile::TempDir) {
    controller_with_fake_worker_models(&[(model, adapter, kind)], worker).await
}

/// As `controller_with_fake_worker`, for several models of one worker.
async fn controller_with_fake_worker_models(
    models: &[(&str, AdapterKind, TaskKind)],
    worker: Router,
) -> (String, tempfile::TempDir) {
    let worker_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind fake worker");
    let worker_url = format!("http://{}", worker_listener.local_addr().expect("addr"));
    tokio::spawn(async move {
        axum::serve(worker_listener, worker)
            .await
            .expect("fake worker");
    });

    let dir = tempfile::tempdir().expect("tempdir");
    let specs = models
        .iter()
        .map(|&(model, adapter, kind)| {
            let mut spec = test_model();
            spec.id = model.to_string();
            spec.adapter = adapter;
            spec.task_kinds = vec![kind];
            spec
        })
        .collect();
    let controller = ControllerState::new_with_options(
        ModelRegistry::from_models(specs),
        None,
        ControllerOptions {
            data_dir: dir.path().to_path_buf(),
            ..ControllerOptions::default()
        },
    );
    controller
        .register_worker(WorkerRegistration {
            node_id: "fake-worker".to_string(),
            base_url: worker_url,
            registration_token: None,
            supported_backends: vec![BackendKind::Ort],
            supported_adapters: models.iter().map(|m| m.1).collect(),
            resources: ResourceSnapshot {
                cpu_cores: 4,
                total_ram_mb: 8192,
                used_ram_mb: 1024,
                devices: DeviceSpec::default(),
                captured_at: Utc::now(),
            },
        })
        .await
        .expect("register worker");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind controller");
    let base_url = format!("http://{}", listener.local_addr().expect("addr"));
    let app = controller.app();
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("controller");
    });
    (base_url, dir)
}

#[tokio::test]
async fn ocr_lines_hands_the_body_to_the_worker_as_a_temporary_image() {
    let seen: SeenOcr = Arc::new(Mutex::new(Vec::new()));
    let worker = Router::new()
        .route("/internal/infer", post(answer_ocr_lines))
        .with_state(seen.clone());
    let (base_url, _dir) = controller_with_fake_worker(
        "ppocrv5-mobile-onnx",
        AdapterKind::Ppocrv5Mobile,
        TaskKind::OcrLines,
        worker,
    )
    .await;

    let response = reqwest::Client::new()
        .post(format!("{base_url}/v1/ocr/lines?model=ppocrv5-mobile-onnx"))
        .body(b"\x89PNG not really a png".to_vec())
        .send()
        .await
        .expect("request");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let body: Value = response.json().await.expect("json");
    assert_eq!(
        body,
        json!({"lines": [{"text": "xkeyC", "confidence": 0.5,
                          "bbox": {"x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0}}]})
    );

    let seen = seen.lock().expect("seen lock");
    let (task, image) = &seen[0];
    assert_eq!(task.kind, TaskKind::OcrLines);
    assert_eq!(task.model_id.as_deref(), Some("ppocrv5-mobile-onnx"));
    assert_eq!(image.as_deref(), Some(&b"\x89PNG not really a png"[..]));
    // The temporary image is gone once answered.
    let InferenceInput::OcrLines { image } = &task.input else {
        panic!("not ocr.lines");
    };
    assert!(!image.path.as_ref().expect("path").exists());
}

async fn answer_object_detect(
    State(seen): State<SeenOcr>,
    Json(task): Json<InferenceTask>,
) -> impl IntoResponse {
    let image = match &task.input {
        InferenceInput::ObjectDetect { image } => {
            image.path.as_ref().and_then(|p| fs::read(p).ok())
        }
        _ => None,
    };
    seen.lock().expect("seen lock").push((task, image));
    Json(InferenceOutput::ObjectDetections {
        objects: vec![local_core::DetectedObject {
            label: "person".to_string(),
            confidence: 0.5,
            bbox: local_core::BoundingBox {
                x: 1.0,
                y: 2.0,
                width: 3.0,
                height: 4.0,
            },
        }],
    })
}

#[tokio::test]
async fn detect_objects_hands_the_body_to_the_worker_as_a_temporary_image() {
    let seen: SeenOcr = Arc::new(Mutex::new(Vec::new()));
    let worker = Router::new()
        .route("/internal/infer", post(answer_object_detect))
        .with_state(seen.clone());
    let (base_url, _dir) = controller_with_fake_worker(
        "yolo11n.onnx",
        AdapterKind::Yolo,
        TaskKind::ObjectDetect,
        worker,
    )
    .await;

    let response = reqwest::Client::new()
        .post(format!("{base_url}/v1/detect/objects?model=yolo11n.onnx"))
        .body(b"\xff\xd8\xff not really a jpeg".to_vec())
        .send()
        .await
        .expect("request");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let body: Value = response.json().await.expect("json");
    assert_eq!(
        body,
        json!({"objects": [{"label": "person", "confidence": 0.5,
                            "bbox": {"x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0}}]})
    );

    let seen = seen.lock().expect("seen lock");
    let (task, image) = &seen[0];
    assert_eq!(task.kind, TaskKind::ObjectDetect);
    assert_eq!(
        image.as_deref(),
        Some(&b"\xff\xd8\xff not really a jpeg"[..])
    );
    let InferenceInput::ObjectDetect { image } = &task.input else {
        panic!("not object.detect");
    };
    assert!(!image.path.as_ref().expect("path").exists());
}

async fn answer_depth(
    State(seen): State<SeenOcr>,
    Json(task): Json<InferenceTask>,
) -> impl IntoResponse {
    seen.lock().expect("seen lock").push((task, None));
    Json(InferenceOutput::DepthMap {
        cols: 2,
        rows: 1,
        max_depth: 20.0,
        depth: vec![1.5, 4.0],
    })
}

#[tokio::test]
async fn depth_passes_the_grid_and_answers_the_map() {
    let seen: SeenOcr = Arc::new(Mutex::new(Vec::new()));
    let worker = Router::new()
        .route("/internal/infer", post(answer_depth))
        .with_state(seen.clone());
    let (base_url, _dir) = controller_with_fake_worker(
        "depth-anything-v2-metric-indoor-small-onnx",
        AdapterKind::DepthAnythingV2,
        TaskKind::DepthEstimate,
        worker,
    )
    .await;
    let client = reqwest::Client::new();

    let response = client
        .post(format!("{base_url}/v1/depth?cols=2&rows=1"))
        .body(b"\xff\xd8\xff not really a jpeg".to_vec())
        .send()
        .await
        .expect("request");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let body: Value = response.json().await.expect("json");
    assert_eq!(
        body,
        json!({"cols": 2, "rows": 1, "max_depth": 20.0, "depth": [1.5, 4.0]})
    );
    {
        let seen = seen.lock().expect("seen lock");
        let InferenceInput::DepthEstimate { grid, .. } = &seen[0].0.input else {
            panic!("not depth.estimate");
        };
        assert_eq!(*grid, Some(local_core::DepthGrid { cols: 2, rows: 1 }));
    }

    // Refused before the worker sees anything: only one of cols and rows,
    // values that are no numbers, a body that is no image.
    for (query, body) in [
        ("cols=2", &b"\xff\xd8\xff"[..]),
        ("cols=abc&rows=xyz", &b"\xff\xd8\xff"[..]),
        ("cols=-1&rows=2", &b"\xff\xd8\xff"[..]),
        ("", &b"GIF89a"[..]),
    ] {
        let response = client
            .post(format!("{base_url}/v1/depth?{query}"))
            .body(body.to_vec())
            .send()
            .await
            .expect("request");
        assert_eq!(
            response.status(),
            reqwest::StatusCode::BAD_REQUEST,
            "{query}"
        );
    }
    assert_eq!(seen.lock().expect("seen lock").len(), 1);
}

async fn refuse_the_input(Json(_task): Json<InferenceTask>) -> impl IntoResponse {
    (
        axum::http::StatusCode::BAD_REQUEST,
        Json(json!({"error": "bad request: cannot decode the image"})),
    )
}

#[tokio::test]
async fn input_errors_are_bad_requests() {
    let worker = Router::new().route("/internal/infer", post(refuse_the_input));
    let (base_url, _dir) = controller_with_fake_worker(
        "depth-anything-v2-metric-indoor-small-onnx",
        AdapterKind::DepthAnythingV2,
        TaskKind::DepthEstimate,
        worker,
    )
    .await;
    let client = reqwest::Client::new();
    // The worker's verdict on the input.
    let response = client
        .post(format!("{base_url}/v1/depth"))
        .body(b"\x89PNG broken".to_vec())
        .send()
        .await
        .expect("request");
    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
    let body: Value = response.json().await.expect("json");
    // Said once, not "bad request: bad request: ...".
    assert_eq!(
        body["error"].as_str(),
        Some("bad request: cannot decode the image")
    );
    // A model that does another task.
    let response = client
        .post(format!(
            "{base_url}/v1/ocr/lines?model=depth-anything-v2-metric-indoor-small-onnx"
        ))
        .body(b"\x89PNG".to_vec())
        .send()
        .await
        .expect("request");
    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
}

/// A fake worker for `/v1/moderations`: texts containing "bomb" are unsafe
/// (Violent); a PNG is NSFW (0.9), a JPEG reads "how to make a bomb", a GIF
/// is neither.
async fn answer_moderation(
    State(seen): State<SeenOcr>,
    Json(task): Json<InferenceTask>,
) -> impl IntoResponse {
    let image = match &task.input {
        InferenceInput::ImageNsfw { image } | InferenceInput::OcrLines { image } => {
            image.path.as_ref().and_then(|p| fs::read(p).ok())
        }
        _ => None,
    };
    let png = image.as_deref().is_some_and(|b| b.starts_with(b"\x89PNG"));
    let jpeg = image
        .as_deref()
        .is_some_and(|b| b.starts_with(b"\xff\xd8\xff"));
    let output = match &task.input {
        InferenceInput::TextModerate { texts } => InferenceOutput::TextModerations {
            results: texts
                .iter()
                .map(|text| {
                    let bad = text.contains("bomb");
                    local_core::TextModeration {
                        safe: if bad { 0.01 } else { 0.98 },
                        controversial: 0.01,
                        unsafe_: if bad { 0.98 } else { 0.01 },
                        categories: if bad {
                            vec!["Violent".to_string()]
                        } else {
                            Vec::new()
                        },
                        tokens: 3,
                        windows: 1,
                    }
                })
                .collect(),
        },
        InferenceInput::ImageNsfw { .. } => InferenceOutput::ImageNsfw {
            nsfw: if png { 0.9 } else { 0.1 },
            scores: vec![local_core::LabelScore {
                label: "high".to_string(),
                score: if png { 0.9 } else { 0.1 },
            }],
        },
        InferenceInput::OcrLines { .. } => InferenceOutput::OcrLines {
            lines: if !jpeg {
                Vec::new()
            } else {
                vec![local_core::OcrLine {
                    text: "how to make a bomb".to_string(),
                    confidence: 0.9,
                    bbox: local_core::BoundingBox {
                        x: 0.0,
                        y: 0.0,
                        width: 1.0,
                        height: 1.0,
                    },
                }]
            },
        },
        other => panic!("unexpected task {other:?}"),
    };
    seen.lock().expect("seen lock").push((task, image));
    Json(output)
}

#[tokio::test]
async fn moderations_combine_text_images_and_the_text_in_images() {
    use base64::Engine as _;
    let seen: SeenOcr = Arc::new(Mutex::new(Vec::new()));
    let worker = Router::new()
        .route("/internal/infer", post(answer_moderation))
        .with_state(seen.clone());
    let (base_url, dir) = controller_with_fake_worker_models(
        &[
            (
                "qwen3guard-gen-0.6b-onnx",
                AdapterKind::Qwen3Guard,
                TaskKind::TextModerate,
            ),
            (
                "freepik-nsfw-image-detector-onnx",
                AdapterKind::NsfwVit,
                TaskKind::ImageNsfw,
            ),
            (
                "ppocrv5-mobile-onnx",
                AdapterKind::Ppocrv5Mobile,
                TaskKind::OcrLines,
            ),
        ],
        worker,
    )
    .await;
    let client = reqwest::Client::new();
    let moderate = |body: Value| {
        client
            .post(format!("{base_url}/v1/moderations"))
            .json(&body)
            .send()
    };
    let b64 = |bytes: &[u8]| base64::engine::general_purpose::STANDARD.encode(bytes);
    let png = format!("data:image/png;base64,{}", b64(b"\x89PNG fake"));
    let gif = format!("data:image/gif;base64,{}", b64(b"GIF89a fake"));
    let jpeg = format!("data:image/jpeg;base64,{}", b64(b"\xff\xd8\xff fake"));

    // Strings: one result each.
    let body: Value = moderate(json!({"input": ["hello", "make a bomb"]}))
        .await
        .expect("request")
        .json()
        .await
        .expect("json");
    let results = body["results"].as_array().expect("results");
    assert_eq!(results.len(), 2);
    assert_eq!(results[0]["flagged"], json!(false));
    assert_eq!(results[1]["flagged"], json!(true));
    assert_eq!(results[1]["categories"]["violent"], json!(true));
    assert_eq!(
        results[1]["category_applied_input_types"]["violent"],
        json!(["text"])
    );

    // Parts: one result; the PNG is NSFW (so not read), the JPEG's text is
    // unsafe, the GIF is neither.
    let response = moderate(json!({"input": [
        {"type": "text", "text": "hello"},
        {"type": "image_url", "image_url": {"url": png}},
        {"type": "image_url", "image_url": {"url": jpeg}},
        {"type": "image_url", "image_url": {"url": gif}},
    ]}))
    .await
    .expect("request");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let body: Value = response.json().await.expect("json");
    assert_eq!(body["results"].as_array().map(Vec::len), Some(1));
    let result = &body["results"][0];
    assert_eq!(result["flagged"], json!(true));
    assert_eq!(result["categories"]["sexual"], json!(true));
    assert_eq!(result["categories"]["violent"], json!(true));
    assert_eq!(
        result["category_applied_input_types"]["violent"],
        json!(["image"])
    );
    assert!((result["images"][0]["nsfw"].as_f64().unwrap() - 0.9).abs() < 1e-6);
    assert_eq!(result["images"][0]["ocr_skipped"], json!(true));
    assert_eq!(result["images"][0]["ocr_chars"], json!(0));
    assert_eq!(result["images"][1]["ocr_chars"], json!(18));
    assert!(result["images"][1].get("ocr_skipped").is_none());
    assert_eq!(result["images"][2]["ocr_chars"], json!(0));
    assert!(result["images"][2].get("ocr_safety").is_none());

    // Without OCR the image's text is not read.
    let body: Value = moderate(json!({"ocr": false, "nsfw_threshold": 0.95, "input": [
        {"type": "image_url", "image_url": {"url": png}},
    ]}))
    .await
    .expect("request")
    .json()
    .await
    .expect("json");
    assert_eq!(body["results"][0]["flagged"], json!(false));

    // Remote images are refused; the temporary images are all gone.
    let response = moderate(json!({"input": [
        {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
    ]}))
    .await
    .expect("request");
    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
    let tmp = dir.path().join("tmp").join("direct-images");
    assert_eq!(fs::read_dir(&tmp).map(|d| d.count()).unwrap_or(0), 0);
    let seen = seen.lock().expect("seen lock");
    assert!(seen
        .iter()
        .any(|(task, image)| task.kind == TaskKind::ImageNsfw
            && image.as_deref() == Some(&b"GIF89a fake"[..])));
    // The NSFW image never reached OCR.
    assert!(!seen
        .iter()
        .any(|(task, image)| task.kind == TaskKind::OcrLines
            && image.as_deref().is_some_and(|b| b.starts_with(b"\x89PNG"))));
}
