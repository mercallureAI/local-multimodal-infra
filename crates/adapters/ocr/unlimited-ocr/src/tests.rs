use super::*;
use local_core::{
    AdapterKind, ArtifactKind, BackendKind, LoadPolicy, ModelArtifact, ResourceRequirement,
    RuntimePolicy, TaskKind,
};
use std::{collections::BTreeMap, path::PathBuf};

#[test]
fn ring_schedule_matches_upstream_rswa() {
    let schedule = RingSchedule::new(5, 3, 100);
    assert_eq!(schedule.capacity, 8);
    // Warm-up appends after the prompt, then the ring wraps.
    let slots: Vec<_> = (0..7).map(|step| schedule.slot(step)).collect();
    assert_eq!(slots, vec![5, 6, 7, 5, 6, 7, 5]);
    let visible = |step| {
        schedule
            .decode_bias(step)
            .iter()
            .filter(|bias| **bias == 0.0)
            .count()
    };
    assert_eq!(visible(0), 6);
    assert_eq!(visible(1), 7);
    assert_eq!(visible(2), 8);
    assert_eq!(visible(10), 8);
}

#[test]
fn ring_capacity_follows_a_short_budget() {
    assert_eq!(RingSchedule::new(277, 128, 16).capacity, 293);
    assert_eq!(RingSchedule::new(277, 128, 8192).capacity, 405);
}

#[test]
fn prefill_bias_is_causal_over_the_prompt() {
    let schedule = RingSchedule::new(3, 2, 10);
    let bias = schedule.prefill_bias();
    let row = |r: usize| -> Vec<bool> {
        bias[r * schedule.capacity..(r + 1) * schedule.capacity]
            .iter()
            .map(|b| *b == 0.0)
            .collect()
    };
    assert_eq!(row(0), vec![true, false, false, false, false]);
    assert_eq!(row(2), vec![true, true, true, false, false]);
}

/// `LOCAL_UNLIMITED_OCR_MODEL_DIR=<package> [LOCAL_UNLIMITED_OCR_TEST_IMAGE=<png>]
/// cargo test -p local-adapter-unlimited-ocr --features cuda real_model_smoke_if_env_set -- --nocapture`
#[test]
fn real_model_smoke_if_env_set() {
    let Ok(model_dir) = std::env::var("LOCAL_UNLIMITED_OCR_MODEL_DIR") else {
        return;
    };
    let image = std::env::var_os("LOCAL_UNLIMITED_OCR_TEST_IMAGE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(&model_dir)
                .parent()
                .expect("model dir has a parent")
                .join("unlimited-ocr-src/assets/baidu.png")
        });
    let runs: usize = std::env::var("LOCAL_UNLIMITED_OCR_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let provider = if cfg!(feature = "cuda") {
        "cuda"
    } else {
        "cpu"
    };
    let spec = ModelSpec {
        id: "unlimited-ocr-onnx".to_string(),
        name: "Unlimited-OCR".to_string(),
        enabled: true,
        task_kinds: vec![TaskKind::OcrRecognize],
        adapter: AdapterKind::UnlimitedOcr,
        backend: BackendKind::Ort,
        artifacts: vec![ModelArtifact {
            kind: ArtifactKind::Local,
            path: PathBuf::from(&model_dir),
            source_path: None,
            sha256: None,
            url: None,
            repo_id: None,
            revision: None,
            files: Vec::new(),
            allow_patterns: Vec::new(),
            metadata: BTreeMap::new(),
        }],
        runtime: RuntimePolicy {
            provider_order: vec![provider.to_string()],
            ..Default::default()
        },
        resources: ResourceRequirement::default(),
        load_policy: LoadPolicy::default(),
        metadata: BTreeMap::new(),
    };
    let started = Instant::now();
    let mut adapter = UnlimitedOcrAdapter::load(&spec).expect("load adapter");
    eprintln!(
        "loaded in {:.1}s, provider {:?}",
        started.elapsed().as_secs_f64(),
        adapter.provider_report()
    );
    if cfg!(feature = "cuda") {
        assert_eq!(
            adapter.provider_report().provider,
            local_backend_ort::ProviderKind::Cuda
        );
    }
    let page = preprocess::load_page(&image).expect("load page");
    for run in 0..runs {
        let text = adapter
            .recognize_pages(std::slice::from_ref(&page))
            .expect("ocr");
        let t = adapter.last_timings();
        eprintln!(
            "run {run}: vision {:.0} ms, prefill {:.0} ms ({} tokens), decode {:.0} ms for {} tokens ({:.1} tok/s)",
            t.vision_ms,
            t.prefill_ms,
            t.prompt_tokens,
            t.decode_ms,
            t.generated_tokens,
            t.generated_tokens as f64 / (t.decode_ms / 1e3)
        );
        if run == 0 {
            eprintln!("--- text ---\n{text}\n------------");
        }
        assert!(!text.is_empty());
    }
}

/// Loads and runs the package step by step, pausing between steps, so an
/// external per-process VRAM sampler can attribute memory to each step.
#[test]
fn staged_load_if_env_set() {
    let Ok(model_dir) = std::env::var("LOCAL_UNLIMITED_OCR_STAGED_DIR") else {
        return;
    };
    let artifacts =
        UnlimitedOcrArtifacts::open("unlimited-ocr-onnx", &model_dir).expect("artifacts");
    let runtime = artifacts.manifest.runtime.clone();
    let backend = OrtBackend::new(ProviderSelection::from_strings(&["cuda".to_string()]))
        .with_cuda_memory_options(CudaMemoryOptions {
            arena_same_as_requested: true,
            conv_max_workspace: false,
        });
    let started = Instant::now();
    let pause = |stage: &str| {
        eprintln!("[stage] {:>5.1}s {stage}", started.elapsed().as_secs_f64());
        std::thread::sleep(std::time::Duration::from_secs(3));
    };
    pause("start");
    let mut vision = backend.load_session(&artifacts.vision).expect("vision");
    pause("vision loaded (first CUDA session)");
    let shared = artifacts
        .manifest
        .device_shared_initializers
        .as_ref()
        .expect("shared");
    let weights = backend
        .upload_initializers(&artifacts.root.join(&shared.data_file), &shared.tensors)
        .expect("upload");
    pause(&format!("uploaded {} MiB", weights.bytes() >> 20));
    let mut prefill = backend
        .load_session_with_initializers(&artifacts.prefill, &weights)
        .expect("prefill");
    pause("prefill loaded");
    let mut decode = backend
        .load_session_with_initializers(&artifacts.decode, &weights)
        .expect("decode");
    pause("decode loaded");

    let image = std::env::var_os("LOCAL_UNLIMITED_OCR_TEST_IMAGE")
        .map(PathBuf::from)
        .expect("image");
    let page = preprocess::load_page(&image).expect("page");
    let size = runtime.image_size as usize;
    let features = vision
        .run_tensors_releasing_memory(&[OrtTensorInput {
            name: "pixel_values".to_string(),
            shape: vec![1, 3, size, size],
            data: OrtTensorData::F32(preprocess::page_tensor(&page, runtime.image_size)),
        }])
        .expect("vision run");
    pause("vision ran");
    let OrtTensorData::F32(features) = features.into_iter().next().unwrap().data else {
        panic!("f32 features")
    };
    let prompt = 1 + runtime.image_tokens_per_page + 3;
    let schedule = RingSchedule::new(prompt, runtime.sliding_window, 8192);
    let pairs = (0..runtime.num_hidden_layers)
        .flat_map(|layer| {
            ["key", "value"].map(|kind| SharedKvPair {
                past_input: format!("past_key_values.{layer}.{kind}"),
                present_output: format!("present.{layer}.{kind}"),
            })
        })
        .collect::<Vec<_>>();
    let plain = prefill
        .create_shared_kv_binding(
            &pairs,
            [
                1,
                runtime.num_key_value_heads,
                schedule.capacity,
                runtime.head_dim,
            ],
            "logits",
        )
        .expect("kv");
    pause("kv allocated, not zeroed");
    drop(plain);
    let mut prefill_kv = prefill
        .create_zeroed_shared_kv_binding(
            &pairs,
            [
                1,
                runtime.num_key_value_heads,
                schedule.capacity,
                runtime.head_dim,
            ],
            "logits",
        )
        .expect("kv");
    let mut decode_kv = decode
        .share_kv_binding(&prefill_kv, "logits")
        .expect("share kv");
    pause("kv allocated, zeroed");
    let mut ids = vec![0i64];
    ids.extend(
        std::iter::repeat(i64::from(runtime.image_token_id)).take(runtime.image_tokens_per_page),
    );
    ids.extend([16, 17, 18]);
    let mut mask = vec![false; prompt];
    mask[1..1 + runtime.image_tokens_per_page].fill(true);
    prefill
        .run_shared_kv_binding_releasing_memory(
            &mut prefill_kv,
            prefill_inputs(
                ids,
                mask,
                features,
                runtime.hidden_size,
                (0..prompt as i64).collect(),
                (0..prompt as i64).collect(),
                schedule.prefill_bias(),
            ),
        )
        .expect("prefill run");
    pause("prefill ran");
    for step in 0..200 {
        decode
            .run_shared_kv_binding(
                &mut decode_kv,
                decode_inputs(
                    100,
                    (prompt + step) as i64,
                    schedule.slot(step) as i64,
                    schedule.decode_bias(step),
                ),
            )
            .expect("decode run");
    }
    pause("200 decode steps");
    drop((decode_kv, prefill_kv, vision, decode, prefill, weights));
}

/// Recognizes every image of `LOCAL_UNLIMITED_OCR_PAGES_DIR` (sorted) with one
/// loaded adapter and reports per-page and total timings; with
/// `LOCAL_UNLIMITED_OCR_OUT_DIR` set, writes each page's text there.
#[test]
fn pages_dir_if_env_set() {
    let (Ok(model_dir), Ok(pages_dir)) = (
        std::env::var("LOCAL_UNLIMITED_OCR_MODEL_DIR"),
        std::env::var("LOCAL_UNLIMITED_OCR_PAGES_DIR"),
    ) else {
        return;
    };
    let out_dir = std::env::var_os("LOCAL_UNLIMITED_OCR_OUT_DIR").map(PathBuf::from);
    if let Some(dir) = &out_dir {
        std::fs::create_dir_all(dir).expect("out dir");
    }
    let mut pages: Vec<PathBuf> = std::fs::read_dir(&pages_dir)
        .expect("pages dir")
        .map(|entry| entry.expect("entry").path())
        .filter(|path| {
            matches!(
                path.extension().and_then(|e| e.to_str()),
                Some("png" | "jpg" | "jpeg")
            )
        })
        .collect();
    pages.sort();
    let spec = ModelSpec {
        id: "unlimited-ocr-onnx".to_string(),
        name: "Unlimited-OCR".to_string(),
        enabled: true,
        task_kinds: vec![TaskKind::OcrRecognize],
        adapter: AdapterKind::UnlimitedOcr,
        backend: BackendKind::Ort,
        artifacts: vec![ModelArtifact {
            kind: ArtifactKind::Local,
            path: PathBuf::from(&model_dir),
            source_path: None,
            sha256: None,
            url: None,
            repo_id: None,
            revision: None,
            files: Vec::new(),
            allow_patterns: Vec::new(),
            metadata: BTreeMap::new(),
        }],
        runtime: RuntimePolicy {
            provider_order: vec![if cfg!(feature = "cuda") {
                "cuda"
            } else {
                "cpu"
            }
            .to_string()],
            ..Default::default()
        },
        resources: ResourceRequirement::default(),
        load_policy: LoadPolicy::default(),
        metadata: BTreeMap::new(),
    };
    let started = Instant::now();
    let mut adapter = UnlimitedOcrAdapter::load(&spec).expect("load adapter");
    let load_s = started.elapsed().as_secs_f64();
    // Warm-up on the first page (cuDNN/cuBLAS autotuning, arena growth).
    let first = preprocess::load_page(&pages[0]).expect("page");
    adapter
        .recognize_pages(std::slice::from_ref(&first))
        .expect("warm-up");
    let warmup = adapter.last_timings().clone();
    eprintln!(
        "load {load_s:.1}s; warm-up page: vision {:.0} ms, prefill {:.0} ms, decode {:.0} ms",
        warmup.vision_ms, warmup.prefill_ms, warmup.decode_ms
    );
    let mut total_tokens = 0usize;
    let mut decode_ms = 0.0;
    let batch = Instant::now();
    for path in &pages {
        let page_started = Instant::now();
        let page = preprocess::load_page(path).expect("page");
        let text = adapter
            .recognize_pages(std::slice::from_ref(&page))
            .expect("ocr");
        let wall = page_started.elapsed().as_secs_f64();
        let t = adapter.last_timings();
        total_tokens += t.generated_tokens;
        decode_ms += t.decode_ms;
        eprintln!(
            "[page] {} wall {:.2}s | vision {:.0} ms, prefill {:.0} ms, decode {:.0} ms, {} tokens ({:.1} tok/s)",
            path.file_name().unwrap().to_string_lossy(),
            wall,
            t.vision_ms,
            t.prefill_ms,
            t.decode_ms,
            t.generated_tokens,
            t.generated_tokens as f64 / (t.decode_ms / 1e3)
        );
        if let Some(dir) = &out_dir {
            let name = path.with_extension("md");
            std::fs::write(dir.join(name.file_name().unwrap()), &text).expect("write text");
        }
    }
    let wall = batch.elapsed().as_secs_f64();
    eprintln!(
        "[total] {} pages in {wall:.1}s ({:.2}s/page), {total_tokens} tokens, decode {:.1} tok/s, end-to-end {:.1} tok/s",
        pages.len(),
        wall / pages.len() as f64,
        total_tokens as f64 / (decode_ms / 1e3),
        total_tokens as f64 / wall
    );
}
