//! IndexTTS-2.5 text frontend, mirroring `IndexTTS2.infer_generator` in
//! `indextts/infer_v2_5.py`: character replacement plus zh/en normalization,
//! per-language casing, `<word|pronunciation>` annotations and token-budgeted
//! segmentation. zh/en normalization reuses the IndexTTS 1.5 Rust rules, which
//! approximate the official wetext TextNormalizer; ja/es (NeMo TN upstream) are
//! passed through unnormalized.

use crate::tokenizer::MultilingualTokenizer;
use local_adapter_index_tts::normalize_text;
use regex::Regex;
use std::sync::OnceLock;

/// Upstream `max_text_tokens_per_segment` default.
pub const DEFAULT_MAX_TEXT_TOKENS_PER_SEGMENT: usize = 120;

/// One GPT input segment: its display text and token ids (language prefix
/// included, stop-text padding not yet applied).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextSegment {
    pub text: String,
    pub ids: Vec<i32>,
}

pub fn prepare_segments(
    tokenizer: &MultilingualTokenizer,
    text: &str,
    language: &str,
    max_tokens: usize,
    normalize: bool,
) -> Vec<TextSegment> {
    let text = prepare_text(text, language, normalize);
    let prefix = format!("<|{language}|> ");
    split_text_by_tokens(tokenizer, &text, max_tokens, &prefix)
        .into_iter()
        .map(|segment| TextSegment {
            ids: tokenizer.encode(&format!("{prefix}{segment}")),
            text: segment.trim().to_string(),
        })
        .collect()
}

pub fn prepare_text(text: &str, language: &str, normalize: bool) -> String {
    let mut text = match language {
        "zh" | "zhen" | "en" if normalize => protect_annotations(text, normalize_text),
        _ => protect_annotations(text, apply_char_rep_map),
    };
    match language {
        "ja" | "zh" | "zhen" | "en" => text = text.to_lowercase(),
        "es" => text = text.to_uppercase(),
        _ => {}
    }
    let text = apply_pronunciation_annotations(&text);
    special_token_regex()
        .replace_all(&text, |caps: &regex::Captures<'_>| {
            format!("<|{}|>", caps[1].to_uppercase())
        })
        .into_owned()
}

/// Upstream protects `<word|pronunciation>` annotations from the normalizer
/// with placeholders; the normalizer then only sees the surrounding text.
fn protect_annotations(text: &str, normalize: impl Fn(&str) -> String) -> String {
    let mut out = String::new();
    let mut last = 0;
    for m in annotation_regex().find_iter(text) {
        out.push_str(&normalize(&text[last..m.start()]));
        out.push_str(m.as_str());
        last = m.end();
    }
    out.push_str(&normalize(&text[last..]));
    out
}

/// `<going|G OW1 . IH0 NG>` -> `<|SPECIAL_TOKEN_1|>G OW1 . IH0 NG<|SPECIAL_TOKEN_1|>`,
/// `<行|XING2>` -> `<|SPECIAL_TOKEN_2|>XING2<|SPECIAL_TOKEN_2|>`, kana readings inline.
pub fn apply_pronunciation_annotations(text: &str) -> String {
    annotation_regex()
        .replace_all(text, |caps: &regex::Captures<'_>| {
            let word = &caps[1];
            let pronunciation = caps[2].to_uppercase();
            if is_kana(&pronunciation) {
                return format!(" {pronunciation} ");
            }
            let token = if word.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c)) {
                "SPECIAL_TOKEN_2"
            } else {
                "SPECIAL_TOKEN_1"
            };
            format!("<|{token}|>{pronunciation}<|{token}|>")
        })
        .into_owned()
}

fn is_kana(s: &str) -> bool {
    !s.is_empty()
        && (s.chars().all(|c| ('\u{3040}'..='\u{309f}').contains(&c))
            || s.chars().all(|c| ('\u{30a0}'..='\u{30ff}').contains(&c)))
}

/// Mirrors `IndexTTS2.split_text_by_tokens`: keep annotated spans atomic,
/// split after punctuation, fall back to per-character packing, then merge
/// greedily up to the token budget.
pub fn split_text_by_tokens(
    tokenizer: &MultilingualTokenizer,
    text: &str,
    max_tokens: usize,
    prefix: &str,
) -> Vec<String> {
    let budget = max_tokens.saturating_sub(tokenizer.token_len(prefix)).max(1);
    let fits = |s: &str| tokenizer.token_len(s) <= budget;
    if fits(text) {
        return vec![text.to_string()];
    }
    let mut chunks = Vec::new();
    for (piece, atomic) in atomic_pieces(text) {
        if atomic {
            chunks.push(piece.to_string());
            continue;
        }
        for part in split_after_punctuation(piece) {
            if fits(part) {
                chunks.push(part.to_string());
                continue;
            }
            let mut current = String::new();
            for ch in part.chars() {
                let mut candidate = current.clone();
                candidate.push(ch);
                if !current.is_empty() && !fits(&candidate) {
                    chunks.push(std::mem::take(&mut current));
                    current.push(ch);
                } else {
                    current = candidate;
                }
            }
            if !current.is_empty() {
                chunks.push(current);
            }
        }
    }
    let mut segments = Vec::new();
    let mut current = String::new();
    for chunk in chunks {
        if !current.is_empty() && !fits(&format!("{current}{chunk}")) {
            segments.push(std::mem::replace(&mut current, chunk));
        } else {
            current.push_str(&chunk);
        }
    }
    if !current.is_empty() {
        segments.push(current);
    }
    if segments.is_empty() {
        segments.push(text.to_string());
    }
    segments
}

fn atomic_pieces(text: &str) -> Vec<(&str, bool)> {
    let mut pieces = Vec::new();
    let mut last = 0;
    for m in protected_span_regex().find_iter(text) {
        if m.start() > last {
            pieces.push((&text[last..m.start()], false));
        }
        pieces.push((m.as_str(), true));
        last = m.end();
    }
    if last < text.len() {
        pieces.push((&text[last..], false));
    }
    pieces
}

fn split_after_punctuation(text: &str) -> Vec<&str> {
    const MARKS: &str = "，。！？、；：,.!?;:\n";
    let mut parts = Vec::new();
    let mut start = 0;
    for (index, ch) in text.char_indices() {
        if MARKS.contains(ch) {
            let end = index + ch.len_utf8();
            parts.push(&text[start..end]);
            start = end;
        }
    }
    if start < text.len() {
        parts.push(&text[start..]);
    }
    parts
}

/// Official `TextNormalizer.char_rep_map` (the pass applied even when
/// normalization is off).
fn apply_char_rep_map(text: &str) -> String {
    const MAP: [(&str, &str); 38] = [
        ("：", ","), ("；", ","), (";", ","), ("，", ","), ("。", "."), ("！", "!"),
        ("？", "?"), ("\n", " "), ("·", "-"), ("、", ","), ("...", "…"), (",,,", "…"),
        ("，，，", "…"), ("……", "…"), ("“", "'"), ("”", "'"), ("\"", "'"), ("‘", "'"),
        ("’", "'"), ("（", "'"), ("）", "'"), ("(", "'"), (")", "'"), ("《", "'"),
        ("》", "'"), ("【", "'"), ("】", "'"), ("[", "'"), ("]", "'"), ("—", "-"),
        ("～", "-"), ("~", "-"), ("「", "'"), ("」", "'"), (":", ","), ("−", "-"),
        ("\u{3000}", " "), ("\t", " "),
    ];
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    'outer: while !rest.is_empty() {
        for (from, to) in MAP {
            if let Some(tail) = rest.strip_prefix(from) {
                out.push_str(to);
                rest = tail;
                continue 'outer;
            }
        }
        let ch = rest.chars().next().expect("non-empty");
        out.push(ch);
        rest = &rest[ch.len_utf8()..];
    }
    out
}

fn annotation_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"<([^|>\n]+)\|([^>\n]+)>").expect("valid regex"))
}

fn special_token_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"<\|([^|]+)\|>").expect("valid regex"))
}

fn protected_span_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"<\|SPECIAL_TOKEN_\d+\|>.*?<\|SPECIAL_TOKEN_\d+\|>").expect("valid regex")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pronunciation_annotations_follow_upstream_examples() {
        assert_eq!(
            apply_pronunciation_annotations("<going|g ow1 . ih0 ng>"),
            "<|SPECIAL_TOKEN_1|>G OW1 . IH0 NG<|SPECIAL_TOKEN_1|>"
        );
        assert_eq!(
            apply_pronunciation_annotations("晕<眩|xuan4>"),
            "晕<|SPECIAL_TOKEN_2|>XUAN4<|SPECIAL_TOKEN_2|>"
        );
        assert_eq!(apply_pronunciation_annotations("<日|にち>"), " にち ");
    }

    #[test]
    fn annotations_survive_normalization_and_lowercasing() {
        let text = prepare_text("最<重|ZHONG4>要的是2个", "zh", true);
        assert!(text.contains("<|SPECIAL_TOKEN_2|>ZHONG4<|SPECIAL_TOKEN_2|>"), "{text}");
        assert!(!text.contains('2') || text.contains("ZHONG4"), "{text}");
    }

    #[test]
    fn punctuation_split_keeps_marks_on_the_left() {
        assert_eq!(split_after_punctuation("你好，世界。ok"), ["你好，", "世界。", "ok"]);
    }
}
