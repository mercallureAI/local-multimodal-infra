//! Synthesizes texts with a Qwen3-TTS package and prints per-request timings:
//! `cargo run --release -p local-adapter-qwen3-tts --features cuda --example synthesize --
//!  <model dir> <reference.wav> [--ref-text <transcript>] [--language <name>] [--repeat N]
//!  [--stream-cps <characters per second>] [--out <dir>] <text>...`
//!
//! `--stream-cps` writes each text a character at a time at that rate (a chat
//! model's reply, say) while it is spoken, and reports the first audio from
//! the first character; it also checks that a stream written at once speaks
//! exactly what the whole text does.
use local_adapter_qwen3_tts::Qwen3TtsAdapter;
use local_core::{FileRef, ModelSpec, TextPiece};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::mpsc,
    time::{Duration, Instant},
};

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let mut args = std::env::args().skip(1);
    let model_dir = args.next().expect("model dir");
    let reference = FileRef::local(PathBuf::from(args.next().expect("reference wav")));
    let mut params = BTreeMap::new();
    let (mut repeat, mut out, mut texts) = (1usize, None::<PathBuf>, Vec::new());
    let mut stream_cps = None::<f64>;
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
            "--stream-cps" => stream_cps = Some(args.next().unwrap().parse().unwrap()),
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
            let started = Instant::now();
            let mut first_audio = None;
            let mut on_audio = |chunk: &[f32]| {
                first_audio.get_or_insert(started.elapsed());
                audio.extend_from_slice(chunk);
                true
            };
            let stats = match stream_cps {
                None => adapter.synthesize_stream(text, Some(&reference), &params, &mut on_audio),
                Some(cps) => {
                    let pieces = write_text(text.clone(), cps);
                    adapter.synthesize_text_stream(pieces, Some(&reference), &params, &mut on_audio)
                }
            }
            .expect("synthesize");
            let seconds = audio.len() as f32 / adapter.sample_rate() as f32;
            println!(
                "[{index}.{round}] frames={} audio={seconds:.2}s first_audio={}ms (from the request: {}ms) talker={}ms ({:.2}ms/frame) vocoder={}ms text_wait={}ms total={}ms RTF={:.3}",
                stats.frames,
                stats.first_audio_ms,
                first_audio.unwrap_or_default().as_millis(),
                stats.talker_ms,
                stats.talker_ms as f32 / stats.frames.max(1) as f32,
                stats.vocoder_ms,
                stats.text_wait_ms,
                stats.total_ms,
                stats.total_ms as f32 / 1000.0 / seconds.max(1e-3)
            );
            if stream_cps.is_some() && round == 0 {
                // A stream written at once speaks exactly the whole text.
                let mut whole = Vec::new();
                adapter
                    .synthesize_stream(text, Some(&reference), &params, &mut |c| {
                        whole.extend_from_slice(c);
                        true
                    })
                    .unwrap();
                let mut streamed = Vec::new();
                adapter
                    .synthesize_text_stream(write_text(text.clone(), 0.0), Some(&reference), &params, &mut |c| {
                        streamed.extend_from_slice(c);
                        true
                    })
                    .unwrap();
                let mut again = Vec::new();
                adapter
                    .synthesize_stream(text, Some(&reference), &params, &mut |c| {
                        again.extend_from_slice(c);
                        true
                    })
                    .unwrap();
                let diff = |a: &[f32], b: &[f32]| {
                    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
                };
                println!(
                    "    stream written at once vs whole text: max diff {:.2e} ({} vs {} samples); whole vs whole again: max diff {:.2e}",
                    diff(&whole, &streamed),
                    streamed.len(),
                    whole.len(),
                    diff(&whole, &again)
                );
            }
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

/// `text` a character at a time, `cps` per second (0: all at once).
fn write_text(text: String, cps: f64) -> mpsc::Receiver<TextPiece> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        for c in text.chars() {
            if cps > 0.0 {
                std::thread::sleep(Duration::from_secs_f64(1.0 / cps));
            }
            if tx.send(TextPiece::Text(c.to_string())).is_err() {
                return;
            }
        }
        let _ = tx.send(TextPiece::End);
    });
    rx
}
