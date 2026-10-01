//! The Mandarin frontend in the IndexTTS text pipeline.
//! Opt-in: `LOCAL_ZH_TTS_FRONTEND_DIR=<model_dir>/zh-tts-frontend` (and
//! `ORT_DYLIB_PATH` for g2pW).

use local_adapter_index_tts::{normalize_text_with, MandarinFrontend, PinyinAnnotation};
use std::path::Path;

#[test]
fn polyphones_and_numbers_if_env_set() {
    let Ok(dir) = std::env::var("LOCAL_ZH_TTS_FRONTEND_DIR") else {
        return;
    };
    let frontend = MandarinFrontend::load(Path::new(&dir)).expect("load Mandarin frontend");
    let cases = [
        // (input, 1.5 inline must contain, 2.5 tagged must contain)
        ("他在银行工作了3年，很了解这一行。", ["HANG2", "LIAO3"], ["<行|HANG2>", "<了|LIAO3>"]),
        ("这个东西还给你，我还要睡觉。", ["HUAN2", "JIAO4"], ["<还|HUAN2>", "<觉|JIAO4>"]),
        ("会议于2024年3月15日举行，长度为2.5米。", ["CHANG2", "二零二四"], ["<长|CHANG2>", "二零二四"]),
    ];
    for (input, inline, tagged) in cases {
        let one = normalize_text_with(input, Some((&frontend, PinyinAnnotation::Inline)));
        let two = normalize_text_with(input, Some((&frontend, PinyinAnnotation::Tagged)));
        eprintln!("{input}\n  1.5: {one}\n  2.5: {two}");
        for expected in inline {
            assert!(one.contains(expected), "{one} lacks {expected}");
        }
        for expected in tagged {
            assert!(two.contains(expected), "{two} lacks {expected}");
        }
    }
    // Readings equal to the dictionary reading are left to the model, and
    // g2pW's Taiwan readings never reach it (星期 qi2, 垃圾 le4 se4).
    let plain = normalize_text_with(
        "我们星期三去银行研究垃圾分类的危险。",
        Some((&frontend, PinyinAnnotation::Tagged)),
    );
    for untouched in ["<银|", "<们|", "<期|", "<垃|", "<圾|", "<危|", "<究|"] {
        assert!(!plain.contains(untouched), "{plain}");
    }
    // English goes through WeText en.
    let english = normalize_text_with("It costs $5.", Some((&frontend, PinyinAnnotation::Inline)));
    assert!(english.contains("five dollars"), "{english}");
}

/// Prints the 2.5 annotations of `LOCAL_ZH_TTS_FRONTEND_SAMPLE` (a text
/// file), to review how much gets annotated.
#[test]
fn print_sample_annotations_if_env_set() {
    let (Ok(dir), Ok(sample)) = (
        std::env::var("LOCAL_ZH_TTS_FRONTEND_DIR"),
        std::env::var("LOCAL_ZH_TTS_FRONTEND_SAMPLE"),
    ) else {
        return;
    };
    let frontend = MandarinFrontend::load(Path::new(&dir)).expect("load Mandarin frontend");
    let text = std::fs::read_to_string(sample).expect("read sample");
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let tagged = normalize_text_with(line, Some((&frontend, PinyinAnnotation::Tagged)));
        eprintln!("{tagged}");
    }
}
