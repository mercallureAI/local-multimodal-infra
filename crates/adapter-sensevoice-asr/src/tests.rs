use super::*;
use local_backend_ort::ProviderKind;
use local_core::{
    AdapterKind, ArtifactKind, BackendKind, ModelArtifact, ResourceRequirement, RuntimePolicy,
};
use std::{
    fs::{self, File},
    io::Write,
    path::PathBuf,
};

#[test]
fn validates_official_artifact_layout() {
    let dir = tempfile::tempdir().expect("tempdir");
    for subdir in ["asr", "vad", "speaker"] {
        fs::create_dir_all(dir.path().join(subdir)).expect("mkdir");
    }
    for name in [MODEL_FILE, CONFIG_FILE, CMVN_FILE, TOKENS_FILE] {
        File::create(dir.path().join("asr").join(name)).expect("create");
    }
    for name in [MODEL_FILE, CONFIG_FILE, CMVN_FILE] {
        File::create(dir.path().join("vad").join(name)).expect("create");
    }
    File::create(dir.path().join("speaker/campplus_cn_en_common_200k.onnx")).expect("create");
    let artifacts = SenseVoiceArtifacts::validate(dir.path()).expect("validate");
    assert_eq!(artifacts.model, dir.path().join("asr").join(MODEL_FILE));
}

#[test]
fn wav_reader_mixes_and_resamples_to_16k() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("stereo_8k.wav");
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 8_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&path, spec).expect("wav create");
    for _ in 0..800 {
        writer.write_sample::<i16>(16_384).expect("left");
        writer.write_sample::<i16>(0).expect("right");
    }
    writer.finalize().expect("finalize");
    let samples = audio::read_wav_mono_f32(&path).expect("read wav");
    assert_eq!(samples.len(), 1_600);
    assert!((samples[0] - 0.25).abs() < 0.01);
}

#[test]
fn cmvn_parser_and_lfr_match_official_dimensions() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join(CMVN_FILE);
    let mut file = File::create(&path).expect("create");
    writeln!(file, "<AddShift> 4 4").unwrap();
    writeln!(file, "<LearnRateCoef> 0 [ 1 2 3 4 ]").unwrap();
    writeln!(file, "<Rescale> 4 4").unwrap();
    writeln!(file, "<LearnRateCoef> 0 [ 2 2 2 2 ]").unwrap();
    let cmvn = Cmvn::load(&path, 4).expect("cmvn");
    let frontend = features::FrontendConfig {
        fs: 16_000,
        window: "hamming".to_string(),
        n_mels: 2,
        frame_length: 25,
        frame_shift: 10,
        lfr_m: 2,
        lfr_n: 1,
    };
    let output =
        features::lfr_cmvn_for_test(vec![vec![1.0, 2.0], vec![3.0, 4.0]], &frontend, &cmvn)
            .expect("lfr");
    assert_eq!(output.frames, 2);
    assert_eq!(output.dim, 4);
    assert_eq!(&output.data[..4], &[4.0, 8.0, 12.0, 16.0]);
}

#[test]
fn ctc_decode_removes_control_tokens_and_collapses_repeats() {
    let tokens = vec![
        "<blank>",
        "<|zh|>",
        "<|NEUTRAL|>",
        "<|Speech|>",
        "<|withitn|>",
        "你",
        "好",
        "▁world",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<Vec<_>>();
    let text = decode_tokens(&[1, 2, 3, 4, 5, 6], &tokens).expect("decode");
    assert_eq!(text, "你好。");
}

#[test]
fn speaker_batch_metadata_is_bounded() {
    let mut spec = model_spec(PathBuf::from("unused"));
    assert_eq!(
        metadata_usize(
            &spec,
            "speaker_batch_size",
            DEFAULT_SPEAKER_BATCH_SIZE,
            MAX_SPEAKER_BATCH_SIZE,
        )
        .expect("default batch"),
        DEFAULT_SPEAKER_BATCH_SIZE
    );
    spec.metadata
        .insert("speaker_batch_size".to_string(), serde_json::json!(64));
    assert_eq!(
        metadata_usize(
            &spec,
            "speaker_batch_size",
            DEFAULT_SPEAKER_BATCH_SIZE,
            MAX_SPEAKER_BATCH_SIZE,
        )
        .expect("configured batch"),
        64
    );
    spec.metadata
        .insert("speaker_batch_size".to_string(), serde_json::json!(0));
    assert!(metadata_usize(
        &spec,
        "speaker_batch_size",
        DEFAULT_SPEAKER_BATCH_SIZE,
        MAX_SPEAKER_BATCH_SIZE,
    )
    .is_err());
}

#[test]
fn component_provider_and_io_binding_metadata_are_validated() {
    let mut spec = model_spec(PathBuf::from("unused"));
    assert!(metadata_provider_selection(&spec, "vad_provider_order")
        .expect("missing provider override")
        .is_none());
    assert!(metadata_bool(&spec, "speaker_io_binding", true).expect("default I/O binding"));

    spec.metadata.insert(
        "vad_provider_order".to_string(),
        serde_json::json!(["cpu", "cuda"]),
    );
    let selection = metadata_provider_selection(&spec, "vad_provider_order")
        .expect("provider override")
        .expect("configured selection");
    assert_eq!(selection.order.len(), 2);
    assert_eq!(selection.order[0].kind, ProviderKind::Cpu);
    assert_eq!(selection.order[1].kind, ProviderKind::Cuda);

    spec.metadata
        .insert("speaker_io_binding".to_string(), serde_json::json!(false));
    assert!(!metadata_bool(&spec, "speaker_io_binding", true).expect("disabled I/O binding"));

    spec.metadata
        .insert("vad_provider_order".to_string(), serde_json::json!([]));
    assert!(metadata_provider_selection(&spec, "vad_provider_order").is_err());
    spec.metadata
        .insert("speaker_io_binding".to_string(), serde_json::json!("true"));
    assert!(metadata_bool(&spec, "speaker_io_binding", true).is_err());
}

#[test]
fn real_model_smoke_if_env_set() {
    let Ok(model_dir) = std::env::var("LOCAL_SENSEVOICE_ASR_MODEL_DIR") else {
        return;
    };
    let audio = std::env::var_os("LOCAL_SENSEVOICE_ASR_TEST_AUDIO")
        .map(PathBuf::from)
        .filter(|path| path.exists())
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/assets/tts-input-mon3tr.wav")
        });
    let mut adapter = SenseVoiceAsrAdapter::load(&model_spec(PathBuf::from(model_dir)))
        .expect("load real SenseVoice model");
    eprintln!(
        "SenseVoice provider report: {:?}",
        adapter.pipeline_provider_report()
    );
    let duration_samples = audio::read_wav_mono_f32(&audio)
        .expect("read smoke audio")
        .len();
    let output = adapter
        .transcribe(&FileRef::local(audio))
        .expect("transcribe real audio");
    let InferenceOutput::AsrTranscription {
        text,
        segments,
        speakers,
    } = output
    else {
        panic!("unexpected output")
    };
    eprintln!("SenseVoice text: {text:?}");
    eprintln!(
        "SenseVoice segments: {:#?}",
        segments
            .iter()
            .map(|segment| (
                segment.start_ms,
                segment.end_ms,
                segment.speaker.as_deref(),
                segment.text.len(),
                segment.tokens.len(),
            ))
            .collect::<Vec<_>>()
    );
    eprintln!("SenseVoice speakers: {speakers:#?}");
    assert!(!text.trim().is_empty());
    assert!(!segments.is_empty());
    assert!(segments.iter().all(|segment| {
        segment.end_ms > segment.start_ms
            && segment
                .speaker
                .as_deref()
                .is_some_and(|id| id.starts_with("speaker_"))
            && segment
                .tokens
                .iter()
                .all(|token| token.end_ms > token.start_ms)
    }));
    assert!(!speakers.is_empty());
    if let Ok(expected) = std::env::var("LOCAL_SENSEVOICE_ASR_EXPECT_SPEAKERS") {
        let expected = expected.parse::<usize>().expect("speaker count");
        assert!(
            speakers.len() >= expected,
            "expected at least {expected} speakers, got {speakers:?}"
        );
    }
    if duration_samples >= 30 * audio::TARGET_SAMPLE_RATE as usize {
        assert!(
            segments
                .iter()
                .all(|segment| segment.end_ms - segment.start_ms <= 20_000),
            "long audio contains an ASR segment above the FSMN-VAD bound"
        );
    }
}

#[test]
fn real_speaker_benchmark_if_env_set() {
    if std::env::var("LOCAL_SENSEVOICE_ASR_SPEAKER_BENCH").as_deref() != Ok("1") {
        return;
    }
    let model_dir = PathBuf::from(
        std::env::var("LOCAL_SENSEVOICE_ASR_MODEL_DIR").expect("benchmark model directory"),
    );
    let audio = PathBuf::from(
        std::env::var_os("LOCAL_SENSEVOICE_ASR_TEST_AUDIO").expect("benchmark audio path"),
    );
    let mut adapter = SenseVoiceAsrAdapter::load(&model_spec(model_dir)).expect("load model");
    let samples = audio::read_wav_mono_f32(&audio).expect("read benchmark audio");
    let segments = adapter.vad.segment(&samples).expect("benchmark VAD");
    let mut elapsed_ms = Vec::new();
    let mut labels = Vec::new();
    for _ in 0..2 {
        let started = std::time::Instant::now();
        labels = adapter
            .speaker
            .label_segments(&samples, &segments)
            .expect("benchmark speakers");
        elapsed_ms.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    assert_eq!(labels.len(), segments.len());
    eprintln!(
        "SenseVoice speaker benchmark provider={:?} segments={} elapsed_ms={elapsed_ms:?}",
        adapter.pipeline_provider_report(),
        segments.len(),
    );
}

fn model_spec(root: PathBuf) -> ModelSpec {
    let mut metadata = BTreeMap::new();
    if let Ok(batch_size) = std::env::var("LOCAL_SENSEVOICE_ASR_SPEAKER_BATCH_SIZE") {
        metadata.insert(
            "speaker_batch_size".to_string(),
            serde_json::json!(batch_size.parse::<u64>().expect("speaker batch size")),
        );
    }
    if let Ok(value) = std::env::var("LOCAL_SENSEVOICE_ASR_SPEAKER_IO_BINDING") {
        metadata.insert(
            "speaker_io_binding".to_string(),
            serde_json::json!(value.parse::<bool>().expect("speaker I/O binding flag")),
        );
    }
    ModelSpec {
        id: "sensevoice-small-onnx".to_string(),
        name: "SenseVoiceSmall ONNX".to_string(),
        enabled: true,
        task_kinds: Vec::new(),
        adapter: AdapterKind::SenseVoiceAsr,
        backend: BackendKind::Ort,
        artifacts: vec![ModelArtifact {
            kind: ArtifactKind::Local,
            path: root,
            source_path: None,
            sha256: None,
            url: None,
            repo_id: None,
            revision: None,
            files: Vec::new(),
            allow_patterns: Vec::new(),
            metadata: Default::default(),
        }],
        runtime: RuntimePolicy {
            provider_order: if cfg!(feature = "cuda") {
                vec!["cuda".to_string(), "cpu".to_string()]
            } else {
                vec!["cpu".to_string()]
            },
            ..Default::default()
        },
        resources: ResourceRequirement::default(),
        load_policy: Default::default(),
        metadata,
    }
}
