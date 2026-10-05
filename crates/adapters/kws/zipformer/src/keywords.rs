//! Wake words as the model's tokens. The zh-en model spells English with
//! ARPAbet phones (`en.phone`, CMUdict style) and Chinese with pinyin
//! initials and toned finals (pypinyin's `INITIALS` / `FINALS_TONE`, not
//! strict: "x iǎo", "y uè", "èr").
//!
//! A word is cut into runs: Han characters (read with `pinyin.tsv`, one
//! reading per character, from pypinyin's tables), letters (the word from
//! the dictionary, and short runs spelled letter by letter too: "M" is
//! "EH1 M") and digits (said in Chinese digit by digit and as a number, and
//! in English: "42" is "四二", "四十二", "FORTY TWO"). Every combination of
//! the runs' readings is one way to say the word (at most `MAX_VARIANTS`).
//! A word with letters on both sides of a digit ("Mon3tr") has no reading
//! to guess: it is left out (its pronunciations go in as other names:
//! "monster", "梦三特").

use local_error::{InfraError, Result};
use std::{collections::HashMap, fs, path::Path};

const MAX_VARIANTS: usize = 16;
/// Letter runs this short are also spelled letter by letter.
const SPELL_UP_TO: usize = 3;

const LETTERS: [&str; 26] = [
    "EY1",
    "B IY1",
    "S IY1",
    "D IY1",
    "IY1",
    "EH1 F",
    "JH IY1",
    "EY1 CH",
    "AY1",
    "JH EY1",
    "K EY1",
    "EH1 L",
    "EH1 M",
    "EH1 N",
    "OW1",
    "P IY1",
    "K Y UW1",
    "AA1 R",
    "EH1 S",
    "T IY1",
    "Y UW1",
    "V IY1",
    "D AH1 B AH0 L Y UW0",
    "EH1 K S",
    "W AY1",
    "Z IY1",
];
const ZH_DIGITS: [&str; 10] = [
    "ling2", "yi1", "er4", "san1", "si4", "wu3", "liu4", "qi1", "ba1", "jiu3",
];
const EN_ONES: [&str; 20] = [
    "ZERO",
    "ONE",
    "TWO",
    "THREE",
    "FOUR",
    "FIVE",
    "SIX",
    "SEVEN",
    "EIGHT",
    "NINE",
    "TEN",
    "ELEVEN",
    "TWELVE",
    "THIRTEEN",
    "FOURTEEN",
    "FIFTEEN",
    "SIXTEEN",
    "SEVENTEEN",
    "EIGHTEEN",
    "NINETEEN",
];
const EN_TENS: [&str; 10] = [
    "", "", "TWENTY", "THIRTY", "FORTY", "FIFTY", "SIXTY", "SEVENTY", "EIGHTY", "NINETY",
];
const INITIALS: [&str; 23] = [
    "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r",
    "z", "c", "s", "y", "w",
];

/// What spells words in the model's tokens.
#[derive(Debug, Clone, Default)]
pub struct Lexicon {
    pub tokens: HashMap<String, i64>,
    english: HashMap<String, String>,
    pinyin: HashMap<char, String>,
}

impl Lexicon {
    /// `tokens.txt` ("<token> <id>" lines), `en.phone` ("WORD PH PH ..."),
    /// `pinyin.tsv` ("<char>\t<reading with tone digit>").
    pub fn load(dir: &Path) -> Result<Self> {
        let read = |name: &str| {
            let path = dir.join(name);
            fs::read_to_string(&path).map_err(|e| InfraError::io(Some(path), e))
        };
        let mut lexicon = Lexicon::default();
        for line in read("tokens.txt")?.lines() {
            let mut parts = line.split_whitespace();
            if let (Some(token), Some(Ok(id))) = (parts.next(), parts.next().map(str::parse)) {
                lexicon.tokens.insert(token.to_string(), id);
            }
        }
        for line in read("en.phone")?.lines() {
            if let Some((word, phones)) = line.split_once(char::is_whitespace) {
                // The first pronunciation of a word.
                lexicon
                    .english
                    .entry(word.to_uppercase())
                    .or_insert_with(|| phones.trim().to_string());
            }
        }
        for line in read("pinyin.tsv")?.lines() {
            if let Some((c, reading)) = line.split_once('\t') {
                let mut chars = c.chars();
                if let (Some(c), None) = (chars.next(), chars.next()) {
                    lexicon.pinyin.insert(c, reading.trim().to_string());
                }
            }
        }
        if lexicon.tokens.is_empty() {
            return Err(InfraError::Adapter(format!(
                "{} has no tokens",
                dir.join("tokens.txt").display()
            )));
        }
        Ok(lexicon)
    }

    /// The ways to say `word`, as token ids (none: nothing to go by).
    pub fn variants(&self, word: &str) -> Vec<Vec<i64>> {
        let mut variants: Vec<Vec<String>> = vec![Vec::new()];
        for run in runs(word) {
            let Some(readings) = self.run_readings(&run) else {
                return Vec::new();
            };
            if readings.is_empty() {
                continue;
            }
            variants = variants
                .iter()
                .flat_map(|prefix| {
                    readings.iter().map(move |reading| {
                        let mut v = prefix.clone();
                        v.extend(reading.iter().cloned());
                        v
                    })
                })
                .take(MAX_VARIANTS)
                .collect();
        }
        let mut out: Vec<Vec<i64>> = Vec::new();
        for tokens in variants {
            let ids: Option<Vec<i64>> =
                tokens.iter().map(|t| self.tokens.get(t).copied()).collect();
            if let Some(ids) = ids.filter(|ids| !ids.is_empty()) {
                if !out.contains(&ids) {
                    out.push(ids);
                }
            }
        }
        out
    }

    /// The readings of one run (tokens each); `None`: the word cannot be
    /// read; empty: nothing to say (punctuation, spaces).
    fn run_readings(&self, run: &Run) -> Option<Vec<Vec<String>>> {
        match run {
            Run::Han(text) => {
                let mut tokens = Vec::new();
                for c in text.chars() {
                    tokens.extend(pinyin_tokens(self.pinyin.get(&c)?));
                }
                Some(vec![tokens])
            }
            Run::Letters(text) => {
                let upper = text.to_uppercase();
                let mut readings = Vec::new();
                if upper.len() > 1 {
                    if let Some(phones) = self.english.get(&upper) {
                        readings.push(split(phones));
                    }
                }
                if upper.len() <= SPELL_UP_TO || readings.is_empty() {
                    let spelled: Vec<String> = upper
                        .bytes()
                        .flat_map(|b| split(LETTERS[(b - b'A') as usize]))
                        .collect();
                    if upper.len() <= SPELL_UP_TO || readings.is_empty() {
                        readings.push(spelled);
                    }
                }
                Some(readings)
            }
            Run::Digits(digits) => {
                let mut readings = Vec::new();
                let zh_each: Vec<String> = digits
                    .bytes()
                    .flat_map(|b| pinyin_tokens(ZH_DIGITS[(b - b'0') as usize]))
                    .collect();
                readings.push(zh_each);
                if let Ok(n) = digits.parse::<usize>() {
                    if (10..100).contains(&n) && digits.len() == 2 {
                        let mut zh = Vec::new();
                        if n / 10 > 1 {
                            zh.extend(pinyin_tokens(ZH_DIGITS[n / 10]));
                        }
                        zh.extend(pinyin_tokens("shi2"));
                        if n % 10 != 0 {
                            zh.extend(pinyin_tokens(ZH_DIGITS[n % 10]));
                        }
                        readings.push(zh);
                    }
                    if digits.len() <= 2 {
                        let words = english_number(n);
                        let phones: Option<Vec<Vec<String>>> = words
                            .iter()
                            .map(|w| self.english.get(*w).map(|p| split(p)))
                            .collect();
                        if let Some(phones) = phones {
                            readings.push(phones.concat());
                        }
                    }
                }
                Some(readings)
            }
            Run::Mixed => None,
            Run::Other => Some(Vec::new()),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Run {
    Han(String),
    Letters(String),
    Digits(String),
    /// Letters on both sides of a digit within one word ("Mon3tr").
    Mixed,
    Other,
}

/// `word` cut into runs of one kind.
fn runs(word: &str) -> Vec<Run> {
    // An ASCII word with letters, a digit, then letters again: no reading.
    for part in word.split(|c: char| !c.is_ascii_alphanumeric()) {
        let first_digit = part.find(|c: char| c.is_ascii_digit());
        if let Some(at) = first_digit {
            let before = part[..at].chars().any(|c| c.is_ascii_alphabetic());
            let after = part[at..].chars().any(|c| c.is_ascii_alphabetic());
            if before && after {
                return vec![Run::Mixed];
            }
        }
    }
    let kind = |c: char| {
        if is_han(c) {
            0
        } else if c.is_ascii_alphabetic() {
            1
        } else if c.is_ascii_digit() {
            2
        } else {
            3
        }
    };
    let mut out = Vec::new();
    let mut current = String::new();
    let mut current_kind = None;
    for c in word.chars() {
        let k = kind(c);
        if current_kind != Some(k) && !current.is_empty() {
            out.push(make_run(
                current_kind.unwrap(),
                std::mem::take(&mut current),
            ));
        }
        current_kind = Some(k);
        current.push(c);
    }
    if let Some(k) = current_kind.filter(|_| !current.is_empty()) {
        out.push(make_run(k, current));
    }
    out
}

fn make_run(kind: u8, text: String) -> Run {
    match kind {
        0 => Run::Han(text),
        1 => Run::Letters(text),
        2 => Run::Digits(text),
        _ => Run::Other,
    }
}

fn is_han(c: char) -> bool {
    matches!(c as u32, 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF | 0x20000..=0x2FA1F)
}

fn split(phones: &str) -> Vec<String> {
    phones.split_whitespace().map(str::to_string).collect()
}

/// English words for 0..=99.
fn english_number(n: usize) -> Vec<&'static str> {
    match n {
        0..=19 => vec![EN_ONES[n]],
        _ if n.is_multiple_of(10) => vec![EN_TENS[n / 10]],
        _ => vec![EN_TENS[n / 10], EN_ONES[n % 10]],
    }
}

/// A pinyin syllable with its tone digit ("xiao3", "lv4", "er2", "n2") as
/// tokens: its initial (y and w count) and its final with the tone mark
/// ("x", "iǎo"); a syllable without a final ("n2") is one token ("ń").
pub fn pinyin_tokens(syllable: &str) -> Vec<String> {
    let (body, tone) = match syllable.char_indices().last() {
        Some((at, c)) if c.is_ascii_digit() => (&syllable[..at], c.to_digit(10).unwrap_or(5)),
        _ => (syllable, 5),
    };
    let body = body.replace('v', "ü").replace("u:", "ü");
    let initial = INITIALS
        .iter()
        .find(|i| body.starts_with(**i))
        .copied()
        .unwrap_or("");
    let final_ = &body[initial.len()..];
    if final_.is_empty() {
        // A syllabic consonant ("n2", "m2"): marked itself.
        return vec![mark(&body, tone)];
    }
    let mut out = Vec::new();
    if !initial.is_empty() {
        out.push(initial.to_string());
    }
    out.push(mark(final_, tone));
    out
}

/// `final_` with tone `tone` marked (on a, else e, else the o of "ou", else
/// the last vowel; tone 5: none).
fn mark(final_: &str, tone: u32) -> String {
    if !(1..=4).contains(&tone) {
        return final_.to_string();
    }
    let chars: Vec<char> = final_.chars().collect();
    let at = chars
        .iter()
        .position(|&c| c == 'a')
        .or_else(|| chars.iter().position(|&c| c == 'e'))
        .or_else(|| {
            final_
                .contains("ou")
                .then(|| chars.iter().position(|&c| c == 'o'))
                .flatten()
        })
        .or_else(|| chars.iter().rposition(|c| "iouü".contains(*c)))
        .or_else(|| chars.iter().position(|c| "nm".contains(*c)));
    let Some(at) = at else {
        return final_.to_string();
    };
    let marked = match (chars[at], tone) {
        ('a', t) => ['ā', 'á', 'ǎ', 'à'][t as usize - 1],
        ('e', t) => ['ē', 'é', 'ě', 'è'][t as usize - 1],
        ('i', t) => ['ī', 'í', 'ǐ', 'ì'][t as usize - 1],
        ('o', t) => ['ō', 'ó', 'ǒ', 'ò'][t as usize - 1],
        ('u', t) => ['ū', 'ú', 'ǔ', 'ù'][t as usize - 1],
        ('ü', t) => ['ǖ', 'ǘ', 'ǚ', 'ǜ'][t as usize - 1],
        ('n', t) => ['n', 'ń', 'ň', 'ǹ'][t as usize - 1],
        ('m', t) => ['m', 'ḿ', 'm', 'm'][t as usize - 1],
        (c, _) => c,
    };
    chars
        .iter()
        .enumerate()
        .map(|(i, &c)| if i == at { marked } else { c })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinyin_syllables_are_initials_and_toned_finals() {
        let t = |s: &str| pinyin_tokens(s).join(" ");
        assert_eq!(t("xiao3"), "x iǎo");
        assert_eq!(t("san1"), "s ān");
        assert_eq!(t("yue4"), "y uè");
        assert_eq!(t("lv4"), "l ǜ");
        assert_eq!(t("er4"), "èr");
        assert_eq!(t("an1"), "ān");
        assert_eq!(t("shi2"), "sh í");
        assert_eq!(t("liu4"), "l iù");
        assert_eq!(t("gui4"), "g uì");
        assert_eq!(t("dou1"), "d ōu");
        assert_eq!(t("n2"), "ń");
        assert_eq!(t("ma5"), "m a");
    }

    #[test]
    fn words_are_cut_into_runs() {
        assert_eq!(
            runs("小M42"),
            vec![
                Run::Han("小".to_string()),
                Run::Letters("M".to_string()),
                Run::Digits("42".to_string())
            ]
        );
        assert_eq!(runs("Mon3tr"), vec![Run::Mixed]);
        assert_eq!(
            runs("M3"),
            vec![Run::Letters("M".to_string()), Run::Digits("3".to_string())]
        );
    }

    fn lexicon() -> Lexicon {
        let tokens = [
            "<blk>", "EH1", "M", "TH", "R", "IY1", "F", "AO1", "T", "IY0", "UW1", "AA1", "N", "S",
            "ER0", "s", "ān", "ì", "èr", "sh", "í", "x", "iǎo", "y", "ī",
        ];
        Lexicon {
            tokens: tokens
                .iter()
                .enumerate()
                .map(|(i, t)| (t.to_string(), i as i64))
                .collect(),
            english: [
                ("THREE", "TH R IY1"),
                ("FORTY", "F AO1 R T IY0"),
                ("TWO", "T UW1"),
                ("MONSTER", "M AA1 N S T ER0"),
            ]
            .iter()
            .map(|(w, p)| (w.to_string(), p.to_string()))
            .collect(),
            pinyin: [('小', "xiao3".to_string())].into_iter().collect(),
        }
    }

    #[test]
    fn a_name_has_every_reading_of_its_runs() {
        let lex = lexicon();
        let spelled = |v: &Vec<i64>| {
            let by_id: HashMap<i64, &String> = lex.tokens.iter().map(|(t, i)| (*i, t)).collect();
            v.iter()
                .map(|i| by_id[i].as_str())
                .collect::<Vec<_>>()
                .join(" ")
        };
        let m3: Vec<String> = lex.variants("M3").iter().map(spelled).collect();
        assert_eq!(m3, vec!["EH1 M s ān", "EH1 M TH R IY1"]);
        let m42: Vec<String> = lex.variants("M42").iter().map(spelled).collect();
        assert!(
            m42.contains(&"EH1 M F AO1 R T IY0 T UW1".to_string()),
            "{m42:?}"
        );
        assert_eq!(
            lex.variants("小M").iter().map(spelled).collect::<Vec<_>>(),
            vec!["x iǎo EH1 M"]
        );
        assert_eq!(
            lex.variants("monster")
                .iter()
                .map(spelled)
                .collect::<Vec<_>>(),
            vec!["M AA1 N S T ER0"]
        );
        assert!(lex.variants("Mon3tr").is_empty());
    }
}
