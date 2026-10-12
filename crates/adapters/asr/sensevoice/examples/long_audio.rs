//! Times one long transcription and reports the process's peak memory:
//! `cargo run --release -p local-adapter-sensevoice-asr --features cuda --example long_audio -- <model dir> <wav> [out.json]`
//! `LOCAL_SENSEVOICE_ASR_MODEL_FILE` names the graph in asr/ (e.g. model_gpu_int8.onnx),
//! `LOCAL_TEST_PROVIDER_ORDER` the providers (default cuda,cpu),
//! `LOCAL_SENSEVOICE_ASR_CUDA_GRAPH_FRAMES` the CUDA graph buckets (e.g. 64,128,256,352).
//! Prints the GPU memory in use (nvidia-smi) before loading, after loading and after the run.
use local_adapter_sensevoice_asr::SenseVoiceAsrAdapter;
use local_core::{
    AdapterKind, ArtifactKind, BackendKind, FileRef, InferenceOutput, ModelArtifact, ModelSpec,
    ResourceRequirement, RuntimePolicy,
};
use std::{collections::BTreeMap, path::PathBuf, time::Instant};

fn main() {
    let mut args = std::env::args().skip(1);
    let model_dir = PathBuf::from(args.next().expect("model dir"));
    let wav = PathBuf::from(args.next().expect("wav"));
    let out = args.next().map(PathBuf::from);
    let mut metadata = BTreeMap::new();
    if let Ok(file) = std::env::var("LOCAL_SENSEVOICE_ASR_MODEL_FILE") {
        metadata.insert("asr_model_file".to_string(), serde_json::json!(file));
    }
    if let Some(frames) = std::env::var("LOCAL_SENSEVOICE_ASR_CUDA_GRAPH_FRAMES")
        .ok()
        .filter(|frames| !frames.is_empty())
    {
        let frames = frames
            .split(',')
            .map(|value| value.trim().parse::<u64>().expect("frame count"))
            .collect::<Vec<_>>();
        metadata.insert(
            "asr_cuda_graph_frames".to_string(),
            serde_json::json!(frames),
        );
    }
    let gpu_before = gpu_memory_mb();
    let provider_order = std::env::var("LOCAL_TEST_PROVIDER_ORDER")
        .unwrap_or_else(|_| "cuda,cpu".to_string())
        .split(',')
        .map(str::to_string)
        .collect();
    let spec = ModelSpec {
        id: "sensevoice-small-onnx".to_string(),
        name: "SenseVoiceSmall ONNX".to_string(),
        enabled: true,
        task_kinds: Vec::new(),
        adapter: AdapterKind::SenseVoiceAsr,
        backend: BackendKind::Ort,
        artifacts: vec![ModelArtifact {
            kind: ArtifactKind::Local,
            path: model_dir,
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
        metadata,
    };
    let started = Instant::now();
    let mut adapter = SenseVoiceAsrAdapter::load(&spec).expect("load SenseVoice");
    eprintln!(
        "loaded in {:.1} s; providers: {:?}; CUDA graph frames {:?}",
        started.elapsed().as_secs_f64(),
        adapter.pipeline_provider_report(),
        adapter.cuda_graph_frames()
    );
    let gpu_loaded = gpu_memory_mb();
    eprintln!("peak memory after load: {} MB", peak_memory_mb());
    let started = Instant::now();
    let output = adapter
        .transcribe(&FileRef::local(wav))
        .expect("transcribe");
    let elapsed = started.elapsed().as_secs_f64();
    let InferenceOutput::AsrTranscription {
        text,
        segments,
        speakers,
        ..
    } = &output
    else {
        panic!("unexpected output");
    };
    let longest_ms = segments
        .iter()
        .map(|segment| segment.end_ms - segment.start_ms)
        .max()
        .unwrap_or(0);
    eprintln!(
        "transcribe {elapsed:.2} s, {} chars, {} timeline segments (longest {longest_ms} ms), {} speakers, peak memory {} MB",
        text.chars().count(),
        segments.len(),
        speakers.len(),
        peak_memory_mb()
    );
    let gpu_after = gpu_memory_mb();
    eprintln!(
        "GPU memory (whole device): {gpu_before} MB before load, +{} MB loaded, +{} MB after the run",
        gpu_loaded.saturating_sub(gpu_before),
        gpu_after.saturating_sub(gpu_before)
    );
    if let Some(out) = out {
        std::fs::write(&out, serde_json::to_vec_pretty(&output).expect("json")).expect("write");
    }
}

#[cfg(windows)]
fn peak_memory_mb() -> usize {
    #[repr(C)]
    #[derive(Default)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(
            process: isize,
            counters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
    }
    let mut counters = ProcessMemoryCounters {
        cb: std::mem::size_of::<ProcessMemoryCounters>() as u32,
        ..Default::default()
    };
    // SAFETY: the pseudo handle needs no closing; `counters` is a valid,
    // correctly sized PROCESS_MEMORY_COUNTERS.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    counters.peak_working_set_size / (1024 * 1024)
}

#[cfg(not(windows))]
fn peak_memory_mb() -> usize {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find_map(|line| line.strip_prefix("VmHWM:"))
                .and_then(|value| {
                    value
                        .trim()
                        .trim_end_matches(" kB")
                        .trim()
                        .parse::<usize>()
                        .ok()
                })
        })
        .map_or(0, |kb| kb / 1024)
}

/// MB of GPU memory in use on the first device (all processes).
fn gpu_memory_mb() -> usize {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .and_then(|text| text.lines().next()?.trim().parse().ok())
        .unwrap_or(0)
}
