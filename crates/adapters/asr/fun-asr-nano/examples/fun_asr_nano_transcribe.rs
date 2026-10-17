//! Transcribes audio files, one `file<TAB>ms<TAB>text` line each (no
//! timestamps or speakers, as the realtime cascade asks):
//! `cargo run --release -p local-adapter-fun-asr-nano --features cuda --example fun_asr_nano_transcribe -- <package dir> <audio or @list file>...`
//!
//! With `FUN_ASR_NANO_COMPARE_SENSEVOICE=<sensevoice-small-int8-gpu-onnx dir>`
//! each line adds that model's `ms<TAB>text` (its GPU int8 graph with CUDA
//! graph buckets), for side-by-side accuracy and latency. Per-file stage
//! times go to stderr; `FUN_ASR_NANO_HOTWORDS` and `FUN_ASR_NANO_LANGUAGE`
//! pass those parameters,
//! `FUN_ASR_NANO_GRAPH_FRAMES` sets the CUDA graph buckets and
//! `FUN_ASR_NANO_HOLD_SECS` pauses after loading and at the end (to read the
//! GPU memory in use).
use local_adapter_fun_asr_nano::FunAsrNanoAdapter;
use local_adapter_sensevoice_asr::SenseVoiceAsrAdapter;
use local_core::{
    AdapterKind, ArtifactKind, BackendKind, FileRef, InferenceOutput, ModelArtifact, ModelSpec,
    ResourceRequirement, RuntimePolicy,
};
use std::{collections::BTreeMap, path::PathBuf, time::Instant};

fn spec(id: &str, adapter: AdapterKind, root: PathBuf, metadata: serde_json::Value) -> ModelSpec {
    ModelSpec {
        id: id.to_string(),
        name: id.to_string(),
        enabled: true,
        task_kinds: Vec::new(),
        adapter,
        backend: BackendKind::Ort,
        artifacts: vec![ModelArtifact {
            kind: ArtifactKind::Local,
            path: root,
            source_path: None,
            sha256: None,
            size: None,
            url: None,
            repo_id: None,
            revision: None,
            files: Vec::new(),
            allow_patterns: Vec::new(),
            metadata: Default::default(),
        }],
        runtime: RuntimePolicy {
            provider_order: vec!["cuda".to_string(), "cpu".to_string()],
            ..Default::default()
        },
        resources: ResourceRequirement::default(),
        load_policy: Default::default(),
        metadata: serde_json::from_value(metadata).expect("metadata"),
    }
}

fn text(output: InferenceOutput) -> String {
    let InferenceOutput::AsrTranscription { text, .. } = output else {
        panic!("unexpected output");
    };
    text.replace(['\t', '\n'], " ")
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(args.next().expect("package dir"));
    let mut files = Vec::new();
    for arg in args {
        match arg.strip_prefix('@') {
            Some(list) => files.extend(
                std::fs::read_to_string(list)
                    .expect("read list")
                    .lines()
                    .filter(|line| !line.trim().is_empty())
                    .map(|line| line.trim().to_string()),
            ),
            None => files.push(arg),
        }
    }
    // `FUN_ASR_NANO_GRAPH_FRAMES=64,128` overrides the CUDA graph buckets
    // (empty: none).
    let frames = std::env::var("FUN_ASR_NANO_GRAPH_FRAMES")
        .map(|value| {
            value
                .split(',')
                .filter(|frames| !frames.trim().is_empty())
                .map(|frames| frames.trim().parse::<u64>().expect("frame count"))
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|_| vec![64, 128, 256, 352]);
    let graphs = serde_json::json!({"asr_cuda_graph_frames": frames});
    let mut nano = FunAsrNanoAdapter::load(&spec(
        "fun-asr-nano-onnx",
        AdapterKind::FunAsrNano,
        root,
        graphs.clone(),
    ))
    .expect("load Fun-ASR-Nano");
    eprintln!("Fun-ASR-Nano providers: {:?}", nano.provider_report());
    // `FUN_ASR_NANO_HOLD_SECS=n`: pause after loading and at the end (to
    // read the GPU memory in use).
    let hold = std::env::var("FUN_ASR_NANO_HOLD_SECS")
        .ok()
        .and_then(|secs| secs.parse::<u64>().ok())
        .map(std::time::Duration::from_secs);
    if let Some(hold) = hold {
        eprintln!("Fun-ASR-Nano loaded; holding");
        std::thread::sleep(hold);
    }
    let mut sensevoice = std::env::var_os("FUN_ASR_NANO_COMPARE_SENSEVOICE").map(|dir| {
        let mut metadata = graphs.clone();
        metadata["asr_model_file"] = serde_json::json!("model_gpu_int8.onnx");
        SenseVoiceAsrAdapter::load(&spec(
            "sensevoice-small-int8-gpu-onnx",
            AdapterKind::SenseVoiceAsr,
            PathBuf::from(dir),
            metadata,
        ))
        .expect("load SenseVoice")
    });
    let mut params: BTreeMap<String, serde_json::Value> = [
        ("timestamps".to_string(), serde_json::Value::Bool(false)),
        (
            "speaker_diarization".to_string(),
            serde_json::Value::Bool(false),
        ),
    ]
    .into();
    // `FUN_ASR_NANO_HOTWORDS=a,b`: Fun-ASR-Nano's hotwords parameter.
    if let Ok(hotwords) = std::env::var("FUN_ASR_NANO_HOTWORDS") {
        params.insert("hotwords".to_string(), serde_json::json!(hotwords));
    }
    // `FUN_ASR_NANO_LANGUAGE=中文`: Fun-ASR-Nano's language parameter.
    if let Ok(language) = std::env::var("FUN_ASR_NANO_LANGUAGE") {
        params.insert("language".to_string(), serde_json::json!(language));
    }
    for file in files {
        let audio = FileRef::local(PathBuf::from(&file));
        let started = Instant::now();
        let nano_text = text(
            nano.transcribe_with_params(&audio, &params)
                .expect("Fun-ASR-Nano"),
        );
        let nano_ms = started.elapsed().as_secs_f64() * 1000.0;
        let stats = nano.last_stats();
        eprintln!(
            "{file}: audio {:.1} ms, prefill {:.1} ms, decode {:.1} ms for {} tokens ({} segments, {} silent, {} at the token limit)",
            stats.audio_ms,
            stats.prefill_ms,
            stats.decode_ms,
            stats.generated_tokens,
            stats.segments,
            stats.silent_segments,
            stats.token_limit_segments
        );
        let mut line = format!("{file}\t{nano_ms:.1}\t{nano_text}");
        if let Some(sensevoice) = sensevoice.as_mut() {
            let started = Instant::now();
            let sv_text = text(
                sensevoice
                    .transcribe_with_params(&audio, &params)
                    .expect("SenseVoice"),
            );
            let sv_ms = started.elapsed().as_secs_f64() * 1000.0;
            line.push_str(&format!("\t{sv_ms:.1}\t{sv_text}"));
        }
        println!("{line}");
    }
    if let Some(hold) = hold {
        eprintln!("done; holding");
        std::thread::sleep(hold);
    }
}
