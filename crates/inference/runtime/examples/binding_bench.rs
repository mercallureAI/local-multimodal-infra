//! Latency of the adapter calls touched by ORT I/O binding, against real
//! models: `cargo run --release -p local-runtime --features cuda --example
//! binding_bench -- <model dir> <assets dir> [yolo,asr,e5,rerank,tts2] [iterations]`
//!
//! `<assets dir>` holds `yolo-input.jpg`, `asr-input.wav` and
//! `tts-reference.wav`. Each case prints p50/mean/min after a warm-up and a
//! digest of its last output, to compare runs of two builds.
use local_core::{
    AdapterKind, ArtifactKind, BackendKind, EmbeddingInputType, FileRef, InferenceOutput,
    ModelArtifact, ModelSpec, ResourceRequirement, RuntimePolicy,
};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::Instant,
};

fn spec(id: &str, adapter: AdapterKind, path: PathBuf) -> ModelSpec {
    ModelSpec {
        id: id.to_string(),
        name: id.to_string(),
        enabled: true,
        task_kinds: Vec::new(),
        adapter,
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
            provider_order: vec!["cuda".to_string(), "cpu".to_string()],
            max_concurrency: 1,
            idle_ttl_sec: 60,
        },
        resources: ResourceRequirement::default(),
        load_policy: Default::default(),
        metadata: Default::default(),
    }
}

fn measure(label: &str, iterations: usize, mut run: impl FnMut(usize) -> String) {
    let warmup = (iterations / 5).max(2);
    for i in 0..warmup {
        run(i);
    }
    let mut times = Vec::with_capacity(iterations);
    let mut digest = String::new();
    for i in 0..iterations {
        let started = Instant::now();
        digest = run(i);
        times.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(f64::total_cmp);
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    println!(
        "{label:<10} p50 {:>9.3} ms  mean {:>9.3} ms  min {:>9.3} ms  (n={iterations})  {digest}",
        times[times.len() / 2],
        mean,
        times[0]
    );
}

/// Sentences of varying length, so batches and padded lengths change between
/// requests as they do in service.
fn texts(seed: usize, count: usize) -> Vec<String> {
    const WORDS: &[&str] = &[
        "inference",
        "runtime",
        "向量",
        "检索",
        "binding",
        "device",
        "memory",
        "模型",
        "latency",
        "throughput",
        "语音",
        "识别",
        "graph",
        "session",
        "pinned",
        "buffer",
    ];
    (0..count)
        .map(|i| {
            let len = 4 + (seed * 7 + i * 13) % 120;
            (0..len)
                .map(|j| WORDS[(seed + i * 3 + j * 5) % WORDS.len()])
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let mut args = std::env::args().skip(1);
    let models = PathBuf::from(args.next().expect("model dir"));
    let assets = PathBuf::from(args.next().expect("assets dir"));
    let cases = args
        .next()
        .unwrap_or_else(|| "yolo,asr,e5,rerank,tts2".to_string());
    let iterations: usize = args.next().map_or(50, |n| n.parse().expect("iterations"));
    for case in cases.split(',') {
        match case {
            "yolo" => yolo(&models, &assets, iterations),
            "asr" => asr(&models, &assets, iterations),
            "e5" => e5(&models, iterations),
            "rerank" => rerank(&models, iterations),
            "tts2" => tts2(&models, &assets, iterations),
            other => panic!("unknown case {other}"),
        }
    }
}

fn yolo(models: &Path, assets: &Path, iterations: usize) {
    let mut adapter = local_adapter_yolo::YoloAdapter::load(&spec(
        "yolo11n.onnx",
        AdapterKind::Yolo,
        models.join("yolo11n.onnx/yolo11n.onnx"),
    ))
    .expect("load YOLO");
    println!("yolo providers {:?}", adapter.provider_report());
    let image = FileRef::local(assets.join("yolo-input.jpg"));
    adapter.object_detect(&image).expect("detect");
    println!("yolo pinned_io {}", adapter.pinned_cuda_io_enabled());
    measure("yolo", iterations, |_| {
        match adapter.object_detect(&image).expect("detect") {
            InferenceOutput::ObjectDetections { objects } => format!(
                "{} objects, first {:?}",
                objects.len(),
                objects
                    .first()
                    .map(|o| (&o.label, (o.confidence * 1000.0).round()))
            ),
            other => panic!("{other:?}"),
        }
    });
}

fn asr(models: &Path, assets: &Path, iterations: usize) {
    let mut adapter = local_adapter_sensevoice_asr::SenseVoiceAsrAdapter::load(&spec(
        "sensevoice-small-onnx",
        AdapterKind::SenseVoiceAsr,
        models.join("sensevoice-small-onnx"),
    ))
    .expect("load SenseVoice");
    println!("asr providers {:?}", adapter.pipeline_provider_report());
    let audio = FileRef::local(assets.join("asr-input.wav"));
    measure("asr", iterations, |_| {
        match adapter.transcribe(&audio).expect("transcribe") {
            InferenceOutput::AsrTranscription { text, .. } => {
                format!(
                    "{} chars: {}",
                    text.chars().count(),
                    text.chars().take(24).collect::<String>()
                )
            }
            other => panic!("{other:?}"),
        }
    });
}

fn e5(models: &Path, iterations: usize) {
    let mut adapter = local_adapter_e5_embedding::E5EmbeddingAdapter::load(&spec(
        "multilingual-e5-small-onnx",
        AdapterKind::E5Embedding,
        models.join("multilingual-e5-small-onnx"),
    ))
    .expect("load E5");
    println!(
        "e5 providers {:?} pinned_io {}",
        adapter.provider_report(),
        adapter.pinned_cuda_io_enabled()
    );
    let batches = [1usize, 4, 8, 16, 32];
    measure("e5", iterations, |i| {
        let inputs = texts(i, batches[i % batches.len()]);
        match adapter
            .embed(&inputs, EmbeddingInputType::Passage)
            .expect("embed")
        {
            InferenceOutput::TextEmbeddings { embeddings, .. } => format!(
                "batch {:>2}, v0[0..3] {:?}",
                embeddings.len(),
                &embeddings[0][..3]
            ),
            other => panic!("{other:?}"),
        }
    });
}

fn rerank(models: &Path, iterations: usize) {
    let mut adapter = local_adapter_mmarco_reranker::MmarcoRerankerAdapter::load(&spec(
        "mmarco-minilm-l12-onnx",
        AdapterKind::MmarcoReranker,
        models.join("mmarco-minilm-l12-onnx"),
    ))
    .expect("load mMARCO");
    println!("rerank providers {:?}", adapter.provider_report());
    let counts = [2usize, 8, 16, 32];
    measure("rerank", iterations, |i| {
        let documents = texts(i, counts[i % counts.len()]);
        match adapter
            .rerank("pinned memory inference latency", &documents, None)
            .expect("rerank")
        {
            InferenceOutput::TextRerank { results, .. } => format!(
                "docs {:>2}, top {:?}",
                documents.len(),
                results
                    .first()
                    .map(|r| (r.index, (r.relevance_score * 1e4).round()))
            ),
            other => panic!("{other:?}"),
        }
    });
}

fn tts2(models: &Path, assets: &Path, iterations: usize) {
    std::env::set_var("LOCAL_DATA_DIR", std::env::temp_dir().join("binding-bench"));
    let mut adapter = local_adapter_index_tts2::IndexTts2Adapter::load(&spec(
        "indextts-2.5-onnx",
        AdapterKind::IndexTts2,
        models.join("indextts-2.5-onnx"),
    ))
    .expect("load IndexTTS-2.5");
    let reference = FileRef::local(assets.join("tts-reference.wav"));
    let params: BTreeMap<String, serde_json::Value> =
        serde_json::from_value(serde_json::json!({"seed": 9527})).unwrap();
    measure("tts2", iterations, |_| {
        match adapter
            .synthesize(
                uuid::Uuid::new_v4(),
                "大家好，我现在正在测试推理绑定的性能，看看延迟有没有变化。",
                Some(&reference),
                &params,
            )
            .expect("synthesize")
        {
            InferenceOutput::TtsAudio { audio } => {
                let path = audio.path.expect("wav path");
                let reader = hound::WavReader::open(&path).expect("wav");
                let samples = reader.duration();
                let _ = std::fs::remove_file(&path);
                format!("{samples} samples")
            }
            other => panic!("{other:?}"),
        }
    });
}
