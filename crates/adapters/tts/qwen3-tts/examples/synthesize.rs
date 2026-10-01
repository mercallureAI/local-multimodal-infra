//! Synthesizes texts with a Qwen3-TTS package and prints per-request timings:
//! `cargo run --release -p local-adapter-qwen3-tts --features cuda --example synthesize --
//!  <model dir> <reference.wav> [--ref-text <transcript>] [--language <name>] [--repeat N]
//!  [--out <dir>] <text>...`
use local_adapter_qwen3_tts::Qwen3TtsAdapter;
use local_core::{FileRef, ModelSpec};
use serde_json::Value;
use std::{collections::BTreeMap, path::PathBuf};

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let mut args = std::env::args().skip(1);
    let model_dir = args.next().expect("model dir");
    let reference = FileRef::local(PathBuf::from(args.next().expect("reference wav")));
    let mut params = BTreeMap::new();
    let (mut repeat, mut out, mut texts) = (1usize, None::<PathBuf>, Vec::new());
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--ref-text" => {
                params.insert(
                    "reference_text".to_string(),
                    Value::from(args.next().unwrap()),
                );
            }
            "--language" => {
                params.insert("language".to_string(), Value::from(args.next().unwrap()));
            }
            "--greedy" => {
                params.insert("do_sample".to_string(), Value::from(false));
            }
            "--repeat" => repeat = args.next().unwrap().parse().unwrap(),
            "--out" => out = Some(PathBuf::from(args.next().unwrap())),
            _ => texts.push(arg),
        }
    }
    let spec: ModelSpec = serde_json::from_value(serde_json::json!({
        "id": "qwen3-tts-0.6b-onnx",
        "name": "Qwen3-TTS",
        "adapter": "qwen3_tts",
        "backend": "ort",
        "artifacts": [{"type": "local", "path": model_dir}],
        "runtime": {"provider_order": ["cuda", "cpu"]},
    }))
    .unwrap();
    let mut adapter = Qwen3TtsAdapter::load(&spec).expect("load");
    println!("{:?}", adapter.provider_report());
    for (index, text) in texts.iter().enumerate() {
        for round in 0..repeat {
            params.insert("seed".to_string(), Value::from(round as u64));
            let mut audio = Vec::new();
            let stats = adapter
                .synthesize_stream(text, Some(&reference), &params, &mut |chunk| {
                    audio.extend_from_slice(chunk);
                    true
                })
                .expect("synthesize");
            let seconds = audio.len() as f32 / adapter.sample_rate() as f32;
            println!(
                "[{index}.{round}] frames={} audio={seconds:.2}s first_audio={}ms talker={}ms ({:.2}ms/frame) vocoder={}ms total={}ms RTF={:.3}",
                stats.frames,
                stats.first_audio_ms,
                stats.talker_ms,
                stats.talker_ms as f32 / stats.frames.max(1) as f32,
                stats.vocoder_ms,
                stats.total_ms,
                stats.total_ms as f32 / 1000.0 / seconds.max(1e-3)
            );
            if let (Some(dir), 0) = (&out, round) {
                std::fs::create_dir_all(dir).unwrap();
                local_adapter_qwen3_tts::audio::write_wav_i16(
                    &dir.join(format!("rust_{index}.wav")),
                    &audio,
                    adapter.sample_rate(),
                )
                .unwrap();
            }
        }
    }
}
