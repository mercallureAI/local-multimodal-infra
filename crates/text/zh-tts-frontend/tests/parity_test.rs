//! Parity with PaddleSpeech's g2pW frontend as replayed by
//! `scripts/local/zh_frontend_oracle.py` (`tests/data/reference.jsonl`).
//! Opt-in: `LOCAL_ZH_TTS_FRONTEND_DIR=<model_dir>/zh-tts-frontend`.

use local_backend_ort::{OrtBackend, ProviderSelection};
use local_zh_tts_frontend::{Options, ZhFrontend};
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct Case {
    text: String,
    pinyin: Vec<Option<String>>,
}

#[test]
fn matches_paddlespeech_reference_if_env_set() {
    let Ok(dir) = std::env::var("LOCAL_ZH_TTS_FRONTEND_DIR") else {
        return;
    };
    let backend = OrtBackend::new(ProviderSelection::from_strings(&["cpu".to_string()]));
    let mut frontend = ZhFrontend::load(Path::new(&dir), &backend, Options::paddlespeech())
        .expect("load frontend");
    let reference = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/reference.jsonl");
    let text = std::fs::read_to_string(reference).expect("read reference");
    let (mut lines, mut chars, mut char_diffs) = (0usize, 0usize, 0usize);
    let mut failures = Vec::new();
    for line in text.lines() {
        let case: Case = serde_json::from_str(line).expect("parse case");
        let actual = frontend.readings(&case.text).expect("readings");
        lines += 1;
        chars += case.pinyin.len();
        let diffs: Vec<String> = case
            .text
            .chars()
            .zip(case.pinyin.iter().zip(&actual))
            .enumerate()
            .filter(|(_, (_, (expected, got)))| expected != got)
            .map(|(i, (c, (expected, got)))| format!("{i}:{c} py={expected:?} rs={got:?}"))
            .collect();
        if !diffs.is_empty() || actual.len() != case.pinyin.len() {
            char_diffs += diffs.len();
            failures.push(format!("{}\n    {}", case.text, diffs.join("; ")));
        }
    }
    let report = format!(
        "{} of {lines} lines differ ({char_diffs} of {chars} characters)",
        failures.len()
    );
    eprintln!("{report}");
    for failure in failures.iter().take(40) {
        eprintln!("{failure}");
    }
    assert!(failures.is_empty(), "{report}");
}
