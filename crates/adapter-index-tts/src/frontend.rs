use super::*;

/// Text frontend mode for the IndexTTS adapter.
///
/// `OfficialLike` is the default and mirrors the parts of the official
/// IndexTTS 1.5 frontend that are safe to keep in Rust without adding a full
/// WeTextProcessing/pynini dependency: placeholder protection for explicit
/// tone-number pinyin and Chinese names, official punctuation replacement,
/// the small English contraction rule, and `tokenize_by_CJK_char`-style CJK
/// char splitting plus uppercasing. It deliberately does **not** convert
/// arbitrary Hanzi to pinyin.
///
/// `PinyinExplicit` keeps the earlier deterministic Hanzi-to-pinyin path for
/// experiments and backwards comparisons. It uses the `pinyin` crate's single
/// reading and therefore cannot resolve polyphones/context; do not use it as
/// the default official-parity frontend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexTtsTextFrontendMode {
    OfficialLike,
    PinyinExplicit,
}

impl IndexTtsTextFrontendMode {
    pub(crate) fn from_env() -> Self {
        match env::var("LCOAL_INDEXTTS_TEXT_FRONTEND") {
            Ok(value) if value.eq_ignore_ascii_case("pinyin_explicit") => Self::PinyinExplicit,
            Ok(value) if value.eq_ignore_ascii_case("pinyin-explicit") => Self::PinyinExplicit,
            Ok(value) if value.eq_ignore_ascii_case("official_like") => Self::OfficialLike,
            Ok(value) if value.eq_ignore_ascii_case("official-like") => Self::OfficialLike,
            Ok(value) => {
                tracing::warn!(
                    value,
                    "unknown LCOAL_INDEXTTS_TEXT_FRONTEND; using official_like"
                );
                Self::OfficialLike
            }
            Err(_) => Self::OfficialLike,
        }
    }
}

pub fn normalize_text(text: &str) -> String {
    let text = text.replace('嗯', "恩").replace('呣', "母");
    let text = expand_english_contractions(&text);
    let (text, pinyin_tones) = save_pinyin_tones(&text);
    let (text, names) = save_names(&text);
    let use_chinese_rules = use_chinese_normalizer_rules(text.as_str());
    let text = lightweight_tn_placeholder_pass(&text, use_chinese_rules);
    let text = restore_names(&text, &names);
    let text = restore_pinyin_tones(&text, &pinyin_tones);
    let map = if use_chinese_rules {
        &ZH_CHAR_REP_MAP[..]
    } else {
        &CHAR_REP_MAP[..]
    };
    let (text, time_colons) = save_time_colons(&text);
    let text = collapse_spaces(&apply_char_rep_map(&text, map));
    restore_time_colons(&text, &time_colons)
}

pub fn preprocess_text_for_index_tts(text: &str) -> String {
    preprocess_text_for_index_tts_with_mode(text, IndexTtsTextFrontendMode::from_env())
}

pub fn preprocess_text_for_index_tts_with_mode(
    text: &str,
    mode: IndexTtsTextFrontendMode,
) -> String {
    match mode {
        IndexTtsTextFrontendMode::OfficialLike => preprocess_text_official_like(text),
        IndexTtsTextFrontendMode::PinyinExplicit => preprocess_text_pinyin_explicit(text),
    }
}

pub(crate) fn preprocess_text_official_like(text: &str) -> String {
    tokenize_by_cjk_char(&normalize_text(text), true)
}

pub(crate) fn preprocess_text_pinyin_explicit(text: &str) -> String {
    let mut tokens = Vec::new();
    let mut chunk = String::new();
    for ch in text.chars() {
        let mapped = normalize_punctuation_char(ch);
        if mapped.is_whitespace() {
            flush_text_chunk(&mut chunk, &mut tokens);
            continue;
        }
        if let Some(pinyin) = hanzi_to_pinyin_token(mapped) {
            flush_text_chunk(&mut chunk, &mut tokens);
            tokens.push(pinyin);
            continue;
        }
        if is_text_chunk_char(mapped) {
            chunk.push(mapped);
            continue;
        }
        flush_text_chunk(&mut chunk, &mut tokens);
        tokens.push(mapped.to_string());
    }
    flush_text_chunk(&mut chunk, &mut tokens);
    tokens.join(" ").to_uppercase()
}

pub(crate) fn use_chinese_normalizer_rules(s: &str) -> bool {
    let has_chinese = s.chars().any(is_hanzi);
    let has_alpha = s.chars().any(|ch| ch.is_ascii_alphabetic());
    let is_email = looks_like_simple_email(s);
    has_chinese || !has_alpha || is_email || contains_explicit_pinyin_tone(s)
}

pub(crate) fn looks_like_simple_email(s: &str) -> bool {
    let Some((local, domain)) = s.split_once('@') else {
        return false;
    };
    let Some((domain, tld)) = domain.rsplit_once('.') else {
        return false;
    };
    !local.is_empty()
        && !domain.is_empty()
        && !tld.is_empty()
        && local.chars().all(|ch| ch.is_ascii_alphanumeric())
        && domain.chars().all(|ch| ch.is_ascii_alphanumeric())
        && tld.chars().all(|ch| ch.is_ascii_alphabetic())
}

pub(crate) fn contains_explicit_pinyin_tone(s: &str) -> bool {
    let mut pos = 0;
    while pos < s.len() {
        if parse_pinyin_at_word_boundary(s, pos).is_some() {
            return true;
        }
        let Some(ch) = s[pos..].chars().next() else {
            break;
        };
        pos += ch.len_utf8();
    }
    false
}

pub(crate) fn expand_english_contractions(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut token = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphabetic() || ch == '\'' {
            token.push(ch);
        } else {
            flush_contraction_token(&mut token, &mut out);
            out.push(ch);
        }
    }
    flush_contraction_token(&mut token, &mut out);
    out
}

pub(crate) fn flush_contraction_token(token: &mut String, out: &mut String) {
    if token.is_empty() {
        return;
    }
    let lower = token.to_ascii_lowercase();
    if let Some(prefix) = lower.strip_suffix("'s") {
        if ENGLISH_IS_CONTRACTION_PREFIXES.contains(&prefix) {
            let keep_prefix_len = token.len().saturating_sub(2);
            out.push_str(&token[..keep_prefix_len]);
            out.push_str(" is");
            token.clear();
            return;
        }
    }
    out.push_str(token);
    token.clear();
}

pub(crate) fn save_pinyin_tones(original_text: &str) -> (String, Vec<String>) {
    let mut originals = Vec::new();
    let mut pos = 0;
    while pos < original_text.len() {
        if let Some((pinyin, end)) = parse_pinyin_at_word_boundary(original_text, pos) {
            if !originals.iter().any(|item| item == &pinyin) {
                originals.push(pinyin);
            }
            pos = end;
            continue;
        }
        let ch = original_text[pos..]
            .chars()
            .next()
            .expect("valid char boundary");
        pos += ch.len_utf8();
    }
    let mut transformed = original_text.to_string();
    for (index, pinyin) in originals.iter().enumerate() {
        transformed = transformed.replace(pinyin, &pinyin_placeholder(index));
    }
    (transformed, originals)
}

pub(crate) fn restore_pinyin_tones(
    normalized_text: &str,
    original_pinyin_list: &[String],
) -> String {
    let mut transformed = normalized_text.to_string();
    for (i, pinyin) in original_pinyin_list.iter().enumerate() {
        let placeholder = pinyin_placeholder(i);
        let replacement = correct_pinyin(pinyin);
        transformed = transformed.replace(&placeholder, &replacement);
    }
    transformed
}

pub fn correct_pinyin(pinyin: &str) -> String {
    let Some(first) = pinyin.chars().next() else {
        return pinyin.to_string();
    };
    if !matches!(first, 'j' | 'q' | 'x' | 'J' | 'Q' | 'X') {
        return pinyin.to_string();
    }
    normalize_toned_pinyin_syllable(pinyin).unwrap_or_else(|| pinyin.to_ascii_uppercase())
}

pub(crate) fn parse_pinyin_at_word_boundary(chunk: &str, pos: usize) -> Option<(String, usize)> {
    if pos > 0 {
        if let Some(previous) = chunk[..pos].chars().next_back() {
            if previous.is_ascii_alphabetic() {
                return None;
            }
        }
    }
    parse_pinyin_at(chunk, pos).map(|(_normalized, end)| (chunk[pos..end].to_string(), end))
}

pub(crate) fn save_names(original_text: &str) -> (String, Vec<String>) {
    let mut originals = Vec::new();
    let chars = original_text.chars().collect::<Vec<_>>();
    let mut index = 0;
    while index < chars.len() {
        if let Some(end) = chinese_name_end(&chars, index) {
            let name = chars[index..end].iter().collect::<String>();
            if !originals.iter().any(|item| item == &name) {
                originals.push(name);
            }
            index = end;
        } else {
            index += 1;
        }
    }
    let mut transformed = original_text.to_string();
    for (index, name) in originals.iter().enumerate() {
        transformed = transformed.replace(name, &name_placeholder(index));
    }
    (transformed, originals)
}

pub(crate) fn restore_names(normalized_text: &str, original_name_list: &[String]) -> String {
    let mut transformed = normalized_text.to_string();
    for (i, name) in original_name_list.iter().enumerate() {
        transformed = transformed.replace(&name_placeholder(i), name);
    }
    transformed
}

pub(crate) fn save_time_colons(text: &str) -> (String, Vec<String>) {
    let chars = text.chars().collect::<Vec<_>>();
    let mut transformed = String::with_capacity(text.len());
    let mut originals = Vec::new();
    let mut index = 0;
    while index < chars.len() {
        if chars[index].is_ascii_digit() {
            let start = index;
            while index < chars.len() && chars[index].is_ascii_digit() {
                index += 1;
            }
            if index < chars.len() && chars[index] == ':' {
                let colon = index;
                let minute_start = colon + 1;
                let mut minute_end = minute_start;
                while minute_end < chars.len() && chars[minute_end].is_ascii_digit() {
                    minute_end += 1;
                }
                let hour = chars[start..colon].iter().collect::<String>();
                let minute = chars[minute_start..minute_end].iter().collect::<String>();
                if is_plausible_time_pair(&hour, &minute) {
                    transformed.push_str(&hour);
                    let placeholder = time_colon_placeholder(originals.len());
                    transformed.push_str(&placeholder);
                    originals.push(":".to_string());
                    index = minute_start;
                    continue;
                }
            }
            transformed.extend(chars[start..index].iter());
            continue;
        }
        transformed.push(chars[index]);
        index += 1;
    }
    (transformed, originals)
}

pub(crate) fn restore_time_colons(text: &str, originals: &[String]) -> String {
    let mut transformed = text.to_string();
    for (index, original) in originals.iter().enumerate() {
        transformed = transformed.replace(&time_colon_placeholder(index), original);
    }
    transformed
}

pub(crate) fn time_colon_placeholder(index: usize) -> String {
    format!("<time_colon_{index}>")
}

pub(crate) fn chinese_name_end(chars: &[char], start: usize) -> Option<usize> {
    let mut index = start;
    let mut separator_count = 0;
    while index < chars.len() && is_hanzi(chars[index]) {
        index += 1;
    }
    while index + 1 < chars.len() && is_name_separator(chars[index]) && is_hanzi(chars[index + 1]) {
        separator_count += 1;
        index += 1;
        while index < chars.len() && is_hanzi(chars[index]) {
            index += 1;
        }
    }
    if (1..=2).contains(&separator_count) {
        Some(index)
    } else {
        None
    }
}

pub(crate) fn is_name_separator(ch: char) -> bool {
    matches!(ch, '-' | '·' | '—')
}

pub(crate) fn pinyin_placeholder(index: usize) -> String {
    format!("<pinyin_{}>", official_placeholder_suffix(index))
}

pub(crate) fn name_placeholder(index: usize) -> String {
    format!("<n_{}>", official_placeholder_suffix(index))
}

pub(crate) fn official_placeholder_suffix(index: usize) -> String {
    if index < 26 {
        return char::from_u32('a' as u32 + index as u32)
            .expect("ascii placeholder")
            .to_string();
    }
    // Official Python uses chr(ord('a') + i), which continues into punctuation
    // after z. Rust keeps a collision-safe alphabetic extension for large
    // synthetic inputs while preserving official <..._a> through <..._z> names.
    let mut n = index;
    let mut chars = Vec::new();
    loop {
        chars.push(char::from_u32('a' as u32 + (n % 26) as u32).expect("ascii"));
        n = n / 26;
        if n == 0 {
            break;
        }
        n -= 1;
    }
    chars.iter().rev().collect()
}

pub fn tokenize_by_cjk_char(line: &str, do_upper_case: bool) -> String {
    let mut parts = Vec::new();
    let mut current = String::new();
    for ch in line.trim().chars() {
        if is_official_cjk_split_char(ch) {
            push_tokenize_segment(&mut current, do_upper_case, &mut parts);
            parts.push(ch.to_string());
        } else {
            current.push(ch);
        }
    }
    push_tokenize_segment(&mut current, do_upper_case, &mut parts);
    parts.join(" ")
}

pub fn de_tokenized_by_cjk_char(line: &str, do_lower_case: bool) -> String {
    let mut transformed = line.to_string();
    let english_sents = collect_english_sentences(line);
    for (index, sent) in english_sents.iter().enumerate() {
        transformed = transformed.replace(sent, &format!("<sent_{index}>"));
    }
    let mut words = transformed
        .split_whitespace()
        .map(str::to_string)
        .collect::<Vec<_>>();
    for word in &mut words {
        if let Some((placeholder, index)) = find_sent_placeholder(word) {
            if let Some(sent) = english_sents.get(index) {
                let replacement = if do_lower_case {
                    sent.to_lowercase()
                } else {
                    sent.clone()
                };
                *word = word.replace(&placeholder, &replacement);
            }
        }
    }
    words.join("")
}

pub const INDEXTTS_PUNCTUATION_MARK_TOKENS: &[&str] = &[".", "!", "?", "▁.", "▁?", "▁..."];

pub fn split_sentences_by_token(
    tokenized_str: &[String],
    split_tokens: &[&str],
    max_tokens_per_sentence: usize,
) -> Vec<Vec<String>> {
    if tokenized_str.is_empty() || max_tokens_per_sentence == 0 {
        return Vec::new();
    }
    let mut sentences = Vec::<Vec<String>>::new();
    let mut current = Vec::<String>::new();
    let mut i = 0;
    while i < tokenized_str.len() {
        let token = &tokenized_str[i];
        current.push(token.clone());
        if current.len() <= max_tokens_per_sentence {
            if split_tokens.contains(&token.as_str()) && current.len() > 2 {
                if i < tokenized_str.len() - 1
                    && matches!(tokenized_str[i + 1].as_str(), "'" | "▁'")
                {
                    current.push(tokenized_str[i + 1].clone());
                    i += 1;
                }
                sentences.push(std::mem::take(&mut current));
            }
            i += 1;
            continue;
        }
        let sub = if !split_tokens.contains(&",")
            && !split_tokens.contains(&"▁,")
            && current.iter().any(|t| t == "," || t == "▁,")
        {
            split_sentences_by_token(&current, &[",", "▁,"], max_tokens_per_sentence)
        } else if !split_tokens.contains(&"-") && current.iter().any(|t| t == "-") {
            split_sentences_by_token(&current, &["-"], max_tokens_per_sentence)
        } else {
            current
                .chunks(max_tokens_per_sentence)
                .map(|chunk| chunk.to_vec())
                .collect()
        };
        sentences.extend(sub);
        current.clear();
        i += 1;
    }
    if !current.is_empty() {
        sentences.push(current);
    }
    let mut merged = Vec::<Vec<String>>::new();
    for sentence in sentences.into_iter().filter(|s| !s.is_empty()) {
        if let Some(last) = merged.last_mut() {
            if last.len() + sentence.len() <= max_tokens_per_sentence {
                last.extend(sentence);
                continue;
            }
        }
        merged.push(sentence);
    }
    merged
}

pub fn split_sentences(tokenized: &[String], max_tokens_per_sentence: usize) -> Vec<Vec<String>> {
    split_sentences_by_token(
        tokenized,
        INDEXTTS_PUNCTUATION_MARK_TOKENS,
        max_tokens_per_sentence,
    )
}

pub(crate) fn collect_english_sentences(line: &str) -> Vec<String> {
    let chars = line.chars().collect::<Vec<_>>();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if !chars[i].is_ascii_alphabetic() {
            i += 1;
            continue;
        }
        let start = i;
        i += 1;
        while i < chars.len() {
            if chars[i].is_ascii_alphabetic() || chars[i] == '-' {
                i += 1;
            } else if chars[i].is_ascii_whitespace()
                && chars
                    .get(i + 1)
                    .is_some_and(|next| next.is_ascii_alphabetic() || *next == '-')
            {
                i += 1;
            } else {
                break;
            }
        }
        out.push(chars[start..i].iter().collect());
    }
    out
}

pub(crate) fn find_sent_placeholder(word: &str) -> Option<(String, usize)> {
    let start = word.find("<sent_")?;
    let rest = &word[start + 6..];
    let end_rel = rest.find('>')?;
    let digits = &rest[..end_rel];
    let index = digits.parse::<usize>().ok()?;
    Some((word[start..start + 6 + end_rel + 1].to_string(), index))
}

pub(crate) fn push_tokenize_segment(
    current: &mut String,
    do_upper_case: bool,
    parts: &mut Vec<String>,
) {
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        if do_upper_case {
            parts.push(trimmed.to_uppercase());
        } else {
            parts.push(trimmed.to_string());
        }
    }
    current.clear();
}

const ENGLISH_IS_CONTRACTION_PREFIXES: &[&str] = &[
    "what", "where", "who", "which", "how", "there", "here", "it", "she", "he", "that", "this",
];

const CHAR_REP_MAP: &[(&str, &str)] = &[
    ("：", ","),
    ("；", ","),
    (";", ","),
    ("，", ","),
    ("。", "."),
    ("！", "!"),
    ("？", "?"),
    ("\n", " "),
    ("·", "-"),
    ("、", ","),
    ("...", "…"),
    (",,,", "…"),
    ("，，，", "…"),
    ("……", "…"),
    ("“", "'"),
    ("”", "'"),
    ("\"", "'"),
    ("‘", "'"),
    ("’", "'"),
    ("（", "'"),
    ("）", "'"),
    ("(", "'"),
    (")", "'"),
    ("《", "'"),
    ("》", "'"),
    ("【", "'"),
    ("】", "'"),
    ("[", "'"),
    ("]", "'"),
    ("—", "-"),
    ("～", "-"),
    ("~", "-"),
    ("「", "'"),
    ("」", "'"),
    (":", ","),
];

const ZH_CHAR_REP_MAP: &[(&str, &str)] = &[
    ("$", "."),
    ("：", ","),
    ("；", ","),
    (";", ","),
    ("，", ","),
    ("。", "."),
    ("！", "!"),
    ("？", "?"),
    ("\n", " "),
    ("·", "-"),
    ("、", ","),
    ("...", "…"),
    (",,,", "…"),
    ("，，，", "…"),
    ("……", "…"),
    ("“", "'"),
    ("”", "'"),
    ("\"", "'"),
    ("‘", "'"),
    ("’", "'"),
    ("（", "'"),
    ("）", "'"),
    ("(", "'"),
    (")", "'"),
    ("《", "'"),
    ("》", "'"),
    ("【", "'"),
    ("】", "'"),
    ("[", "'"),
    ("]", "'"),
    ("—", "-"),
    ("～", "-"),
    ("~", "-"),
    ("「", "'"),
    ("」", "'"),
    (":", ","),
];

pub(crate) fn normalize_punctuation_char(ch: char) -> char {
    match ch {
        '，' => ',',
        '。' => '.',
        '！' => '!',
        '？' => '?',
        '：' => ':',
        '；' => ';',
        '、' => ',',
        '（' => '(',
        '）' => ')',
        '【' | '《' => '[',
        '】' | '》' => ']',
        '“' | '”' => '"',
        '‘' | '’' => '\'',
        _ => ch,
    }
}

pub(crate) fn hanzi_to_pinyin_token(ch: char) -> Option<String> {
    let raw = ch.to_pinyin()?.with_tone_num_end();
    normalize_toned_pinyin_syllable(raw).or_else(|| {
        let mut fallback = raw.to_string();
        if !fallback.ends_with(|tone| matches!(tone, '1'..='5')) {
            fallback.push('5');
        }
        normalize_toned_pinyin_syllable(&fallback)
    })
}

pub(crate) fn flush_text_chunk(chunk: &mut String, tokens: &mut Vec<String>) {
    if chunk.is_empty() {
        return;
    }
    tokens.extend(split_mixed_text_chunk(chunk));
    chunk.clear();
}

pub(crate) fn split_mixed_text_chunk(chunk: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut raw = String::new();
    let mut pos = 0;
    while pos < chunk.len() {
        if let Some((pinyin, end)) = parse_pinyin_at(chunk, pos) {
            if !raw.is_empty() {
                out.push(std::mem::take(&mut raw));
            }
            out.push(pinyin);
            pos = end;
            continue;
        }
        let ch = chunk[pos..]
            .chars()
            .next()
            .expect("valid char boundary while splitting mixed text");
        raw.push(ch);
        pos += ch.len_utf8();
    }
    if !raw.is_empty() {
        out.push(raw);
    }
    out
}

pub(crate) fn parse_pinyin_at(chunk: &str, pos: usize) -> Option<(String, usize)> {
    let mut letters_end = pos;
    for (offset, ch) in chunk[pos..].char_indices() {
        if is_pinyin_base_char(ch) {
            letters_end = pos + offset + ch.len_utf8();
            continue;
        }
        break;
    }
    if letters_end == pos {
        return None;
    }
    let tone = chunk[letters_end..].chars().next()?;
    if !matches!(tone, '1'..='5') {
        return None;
    }
    let end = letters_end + tone.len_utf8();
    normalize_pinyin_base(&chunk[pos..letters_end], tone).map(|base| (base, end))
}

pub(crate) fn normalize_toned_pinyin_syllable(token: &str) -> Option<String> {
    let mut chars = token.chars().collect::<Vec<_>>();
    let tone = *chars.last()?;
    if !matches!(tone, '1'..='5') || chars.len() < 2 {
        return None;
    }
    chars.pop();
    normalize_pinyin_base(&chars.into_iter().collect::<String>(), tone)
}

pub(crate) fn normalize_pinyin_base(base: &str, tone: char) -> Option<String> {
    if base.is_empty() || !base.chars().all(is_pinyin_base_char) {
        return None;
    }
    let mut normalized = base
        .to_ascii_lowercase()
        .replace('ü', "v")
        .replace('Ü', "v");
    if let Some(initial) = normalized.chars().next() {
        if matches!(initial, 'j' | 'q' | 'x') && normalized[initial.len_utf8()..].starts_with('u') {
            let initial_len = initial.len_utf8();
            normalized.replace_range(initial_len..initial_len + 1, "v");
        }
    }
    if !is_valid_pinyin_base(&normalized) {
        return None;
    }
    Some(format!("{}{}", normalized.to_ascii_uppercase(), tone))
}

pub(crate) fn is_pinyin_base_char(ch: char) -> bool {
    ch.is_ascii_alphabetic() || matches!(ch, 'ü' | 'Ü')
}

pub(crate) fn is_text_chunk_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch.is_alphabetic() || matches!(ch, 'ü' | 'Ü')
}

pub(crate) fn is_valid_pinyin_base(base: &str) -> bool {
    if matches!(base, "er" | "ng" | "m") {
        return true;
    }
    if is_yw_pinyin_base(base) {
        return true;
    }
    for initial in PINYIN_INITIALS {
        if let Some(final_part) = base.strip_prefix(initial) {
            return !final_part.is_empty() && PINYIN_FINALS.contains(&final_part);
        }
    }
    PINYIN_FINALS.contains(&base)
}

pub(crate) fn is_yw_pinyin_base(base: &str) -> bool {
    matches!(
        base,
        "yi" | "ya"
            | "yan"
            | "yang"
            | "yao"
            | "ye"
            | "yo"
            | "yin"
            | "ying"
            | "yong"
            | "you"
            | "yu"
            | "yue"
            | "yuan"
            | "yun"
            | "wu"
            | "wa"
            | "wai"
            | "wan"
            | "wang"
            | "wei"
            | "wen"
            | "weng"
            | "wo"
    )
}

const PINYIN_INITIALS: &[&str] = &[
    "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r",
    "z", "c", "s",
];

const PINYIN_FINALS: &[&str] = &[
    "a", "ai", "an", "ang", "ao", "e", "ei", "en", "eng", "er", "o", "ou", "ong", "i", "ia", "ian",
    "iang", "iao", "ie", "in", "ing", "iong", "iu", "u", "ua", "uai", "uan", "uang", "ui", "un",
    "uo", "v", "ve", "van", "vn",
];

pub fn split_cjk_minimal(text: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut current_is_cjk: Option<bool> = None;
    for ch in text.chars() {
        let is_cjk = is_cjk(ch);
        let boundary = ch.is_ascii_punctuation()
            || matches!(ch, ',' | '.' | '!' | '?' | ':' | ';')
            || current_is_cjk.is_some_and(|prev| prev != is_cjk);
        if boundary && !current.trim().is_empty() {
            parts.push(current.trim().to_string());
            current.clear();
        }
        if !ch.is_ascii_punctuation() && !matches!(ch, ',' | '.' | '!' | '?' | ':' | ';') {
            current.push(ch);
            current_is_cjk = Some(is_cjk);
        } else {
            current_is_cjk = None;
        }
    }
    if !current.trim().is_empty() {
        parts.push(current.trim().to_string());
    }
    if parts.is_empty() {
        vec![text.to_string()]
    } else {
        parts
    }
}

pub(crate) fn is_cjk(ch: char) -> bool {
    matches!(
        ch as u32,
        0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF | 0x3040..=0x30FF | 0xAC00..=0xD7AF
    )
}

pub(crate) fn is_hanzi(ch: char) -> bool {
    matches!(ch as u32, 0x4E00..=0x9FFF)
}

pub(crate) fn is_official_cjk_split_char(ch: char) -> bool {
    matches!(
        ch as u32,
        0x1100..=0x11FF
            | 0x2E80..=0xA4CF
            | 0xA840..=0xD7AF
            | 0xF900..=0xFAFF
            | 0xFE30..=0xFE4F
            | 0xFF65..=0xFFDC
            | 0x20000..=0x2FFFF
    )
}
