//! Parity with the Python `wetext` package IndexTTS uses (reference built by
//! `scripts/local/wetext_parity.py`). Opt-in: set `LOCAL_WETEXT_FST_DIR` to a
//! directory holding `zh/tn/*.fst` and `en/tn/*.fst`.

use std::path::Path;

use serde::Deserialize;
use wetext::{Language, Normalizer, NormalizerConfig, Operator};

#[derive(Deserialize)]
struct Case {
    lang: String,
    text: String,
    expected: String,
}

#[test]
fn matches_python_wetext_if_env_set() {
    let Ok(fst_dir) = std::env::var("LOCAL_WETEXT_FST_DIR") else {
        return;
    };
    let reference = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/tn_reference.jsonl");
    let lines = std::fs::read_to_string(reference).expect("read reference");
    // IndexTTS front.py: zh keeps erhua, en uses defaults; wetext 0.1.0 has no
    // contraction fixing.
    let config = |lang| {
        NormalizerConfig::new()
            .with_lang(lang)
            .with_operator(Operator::Tn)
            .with_fix_contractions(false)
    };
    let mut zh = Normalizer::new(&fst_dir, config(Language::Zh));
    let mut en = Normalizer::new(&fst_dir, config(Language::En));
    let mut failures = Vec::new();
    let mut total = 0;
    for line in lines.lines().skip(1) {
        let case: Case = serde_json::from_str(line).expect("parse case");
        let normalizer = if case.lang == "zh" { &mut zh } else { &mut en };
        let actual = normalizer
            .normalize(&case.text)
            .unwrap_or_else(|err| format!("<error: {err}>"));
        total += 1;
        if actual != case.expected {
            failures.push(format!(
                "[{}] {:?}\n    python: {:?}\n    rust:   {:?}",
                case.lang, case.text, case.expected, actual
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {total} cases differ from Python wetext:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
