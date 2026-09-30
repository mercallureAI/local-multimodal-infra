//! Polyphone accuracy on the clean CPP test set (g2p-mix `benchmarks/data/corpora/cpp_clean`):
//! `cargo run --release --example cpp_bench -- <asset dir> <test.sent> <test.lb> <paddlespeech|mainland> [failures.tsv]`.
//!
//! Each line of `test.sent` marks one target character as `left▁X▁right`; `test.lb`
//! holds its reading. Scored like g2p-mix's benchmark: citation tones (no tone
//! sandhi), `u:` and `v` treated as the same vowel.
use local_backend_ort::{OrtBackend, ProviderSelection};
use local_zh_tts_frontend::{Options, ZhFrontend};
use std::{io::Write, path::Path, time::Instant};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [dir, sent, lb, mode, rest @ ..] = args.as_slice() else {
        panic!("usage: <asset dir> <test.sent> <test.lb> <paddlespeech|mainland> [failures.tsv]");
    };
    let options = Options {
        mainland: match mode.as_str() {
            "paddlespeech" => false,
            "mainland" => true,
            other => panic!("unknown mode {other}"),
        },
        tone_sandhi: false,
    };
    let backend = OrtBackend::new(ProviderSelection::from_strings(&["cpu".to_string()]));
    let mut frontend = ZhFrontend::load(Path::new(dir), &backend, options).expect("load frontend");
    let sentences = std::fs::read_to_string(sent).expect("read test.sent");
    let labels = std::fs::read_to_string(lb).expect("read test.lb");
    let mut failures = rest
        .first()
        .map(|path| std::fs::File::create(path).expect("create failures file"));
    let (mut total, mut correct) = (0usize, 0usize);
    let started = Instant::now();
    for (line, label) in sentences.lines().zip(labels.lines()) {
        let parts: Vec<&str> = line.split('▁').collect();
        let [left, target, right] = parts.as_slice() else {
            panic!("bad marker: {line}");
        };
        let text = format!("{left}{target}{right}");
        let index = left.chars().count();
        let readings = frontend.readings(&text).expect("readings");
        let got = readings[index]
            .clone()
            .unwrap_or_default()
            .replace("u:", "v");
        let expected = label.trim().replace("u:", "v");
        total += 1;
        if got == expected {
            correct += 1;
        } else if let Some(file) = failures.as_mut() {
            writeln!(
                file,
                "{target}\t{expected}\t{got}\t{left}【{target}】{right}"
            )
            .expect("write failure");
        }
        if total % 1000 == 0 {
            eprintln!(
                "{total}: {:.4}% ({:?})",
                100.0 * correct as f64 / total as f64,
                started.elapsed()
            );
        }
    }
    println!(
        "{mode}: {correct} / {total} = {:.4}% in {:?}",
        100.0 * correct as f64 / total as f64,
        started.elapsed()
    );
}
