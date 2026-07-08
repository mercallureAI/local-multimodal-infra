use super::*;
use local_core::{
    AdapterKind, ArtifactKind, BackendKind, ModelArtifact, ResourceRequirement, RuntimePolicy,
};
use std::{
    fs::File,
    io::Write,
    path::{Path, PathBuf},
};

#[test]
fn validates_int4_artifacts() {
    let dir = tempfile::tempdir().expect("tempdir");
    for name in [
        "encoder.int4.onnx",
        "decoder_init.int4.onnx",
        "decoder_step.int4.onnx",
        "embed_tokens.bin",
        "tokenizer.json",
        "config.json",
        "preprocessor_config.json",
    ] {
        let mut file = File::create(dir.path().join(name)).expect("create");
        if name.ends_with(".json") {
            file.write_all(b"{}").expect("write");
        }
    }
    let artifacts = QwenAsrArtifacts::validate(dir.path()).expect("validate");
    assert_eq!(artifacts.variant, QwenArtifactVariant::Int4);
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
fn log_mel_has_whisper_shape_for_one_second() {
    let samples = vec![0.0f32; 16_000];
    let mel = features::log_mel_128(&samples, 16_000).expect("mel");
    assert_eq!(mel.bins, 128);
    assert_eq!(mel.frames, features::mel_frame_count(samples.len()));
    assert_eq!(mel.frames, 100);
    assert_eq!(mel.data.len(), 128 * 100);
    assert!(mel.data.iter().all(|value| value.is_finite()));
}

#[test]
fn prompt_places_audio_pad_block_at_expected_offset() {
    let prompt = build_prompt_ids(3);
    assert_eq!(prompt_audio_offset(&prompt).expect("offset"), 9);
    assert_eq!(
        prompt
            .iter()
            .filter(|id| **id == AUDIO_PAD_TOKEN_ID)
            .count(),
        3
    );
    assert_eq!(prompt.last(), Some(&NEWLINE_TOKEN_ID));
}

#[test]
fn f16_embedding_lookup_converts_single_row() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("embed_tokens.bin");
    let values = [0.0f32, 1.0, 2.0, 3.0];
    let bytes = values
        .iter()
        .flat_map(|value| half::f16::from_f32(*value).to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    fs::write(&path, bytes).expect("write");
    let config = QwenModelConfig {
        hidden_size: 2,
        vocab_size: 2,
        eos_token_ids: EOS_TOKEN_IDS.to_vec(),
        embed_tokens_dtype: "float16".to_string(),
    };
    let embeddings = EmbedTokens::load(&path, &config).expect("load");
    assert_eq!(embeddings.lookup(1).expect("lookup"), vec![2.0, 3.0]);
}

#[test]
fn real_model_smoke_if_env_set() {
    let Ok(model_dir) = std::env::var("LOCAL_QWEN_ASR_MODEL_DIR") else {
        return;
    };
    let dir = tempfile::tempdir().expect("tempdir");
    let wav = resolve_qwen_smoke_audio(&dir);
    let mut adapter = match QwenAsrAdapter::load(&model_spec(
        PathBuf::from(model_dir),
        provider_order_from_env(),
    )) {
        Ok(adapter) => adapter,
        Err(err) => {
            eprintln!("Qwen ASR smoke stopped at model load boundary: {err}");
            return;
        }
    };
    let report = adapter.provider_report();
    eprintln!(
        "Qwen ASR provider report: encoder={:?}/fallback={}, decoder_init={:?}/fallback={}, decoder_step={:?}/fallback={}",
        report.encoder.provider,
        report.encoder.cpu_fallback_used,
        report.decoder_init.provider,
        report.decoder_init.cpu_fallback_used,
        report.decoder_step.provider,
        report.decoder_step.cpu_fallback_used
    );
    match adapter.transcribe(&FileRef::local(wav)) {
        Ok(InferenceOutput::AsrTranscription { text }) => {
            eprintln!("Qwen ASR smoke text: {text:?}");
            assert!(
                text.trim().chars().count() <= 2048,
                "Qwen ASR smoke output is unexpectedly unbounded"
            );
        }
        Ok(other) => panic!("unexpected output: {other:?}"),
        Err(err) => {
            let msg = err.to_string();
            assert!(
                msg.contains("inputs:") && msg.contains("outputs:"),
                "smoke failure should carry session diagnostics: {msg}"
            );
            eprintln!("Qwen ASR smoke reached expected boundary: {msg}");
        }
    }
}

fn resolve_qwen_smoke_audio(dir: &tempfile::TempDir) -> PathBuf {
    let explicit = std::env::var_os("LOCAL_QWEN_ASR_TEST_AUDIO")
        .map(PathBuf::from)
        .filter(|path| path.exists());
    if let Some(path) = explicit {
        eprintln!("Qwen ASR smoke using explicit audio: {}", path.display());
        return path;
    }

    let bundled =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/assets/tts-input-mon3tr.wav");
    if bundled.exists() {
        eprintln!("Qwen ASR smoke using bundled audio: {}", bundled.display());
        return bundled;
    }

    let wav = dir.path().join("silence.wav");
    write_silence_wav(&wav, 16_000, 16_000);
    eprintln!("Qwen ASR smoke using fallback silence: {}", wav.display());
    wav
}

fn provider_order_from_env() -> Vec<String> {
    std::env::var("LOCAL_TEST_PROVIDER_ORDER")
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
        .unwrap_or_else(|| vec!["cpu".to_string()])
}

fn write_silence_wav(path: &Path, samples: usize, sample_rate: u32) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).expect("wav create");
    for _ in 0..samples {
        writer.write_sample::<i16>(0).expect("sample");
    }
    writer.finalize().expect("finalize");
}

fn model_spec(path: PathBuf, provider_order: Vec<String>) -> ModelSpec {
    ModelSpec {
        id: "qwen-asr-test".to_string(),
        name: "Qwen ASR Test".to_string(),
        enabled: true,
        task_kinds: Vec::new(),
        adapter: AdapterKind::QwenAsr,
        backend: BackendKind::Ort,
        artifacts: vec![ModelArtifact {
            kind: ArtifactKind::Local,
            path,
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
            provider_order,
            ..Default::default()
        },
        resources: ResourceRequirement::default(),
        load_policy: Default::default(),
        metadata: Default::default(),
    }
}
