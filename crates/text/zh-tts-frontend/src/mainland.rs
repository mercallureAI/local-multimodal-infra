//! Mainland corrections of g2pW's Taiwan-leaning readings, from g2p-mix
//! (pengzhendong/g2p-mix 0893439, Apache-2.0, `backends/mandarin.py`
//! `_normalize_g2pw_pinyin`), plus local ones marked as such.

use crate::pinyin::read_tsv;
use local_error::Result;
use std::{collections::HashMap, path::Path};

const CORRECTIONS: [(char, &str, &str); 16] = [
    ('和', "han4", "he2"),
    ('崖', "yai2", "ya2"),
    ('儿', "er1", "er2"),
    ('兒', "er1", "er2"),
    ('削', "xue4", "xue1"),
    ('嵌', "qian1", "qian4"),
    ('炔', "jue2", "que1"),
    ('综', "zong4", "zong1"),
    ('綜', "zong4", "zong1"),
    ('缉', "qi4", "ji1"),
    ('緝', "qi4", "ji1"),
    ('识', "shi4", "shi2"),
    ('識', "shi4", "shi2"),
    ('朴', "pu2", "pu3"),
    ('湮', "yin1", "yan1"),
    ('晟', "cheng2", "sheng4"),
];
/// Standard Mainland readings per character (`mainland/readings.tsv`:
/// 通用规范汉字字典, else 现代汉语词典).
#[derive(Debug, Default)]
pub struct MainlandReadings(HashMap<char, Vec<String>>);

impl MainlandReadings {
    pub fn load(path: &Path) -> Result<Self> {
        Ok(Self(
            read_tsv(path)?
                .into_iter()
                .filter_map(|(word, readings)| {
                    let mut chars = word.chars();
                    let c = chars.next().filter(|_| chars.next().is_none())?;
                    Some((c, readings.split_whitespace().map(str::to_string).collect()))
                })
                .collect(),
        ))
    }

    /// Whether `reading` is a Mainland reading of `c`; a neutral tone counts
    /// for any tone of the syllable. True for characters it does not list.
    pub fn allows(&self, c: char, reading: &str) -> bool {
        let Some(readings) = self.0.get(&c) else {
            return true;
        };
        let base = |r: &str| r.trim_end_matches(|d: char| d.is_ascii_digit()).to_string();
        readings.iter().any(|r| r == reading)
            || (reading.ends_with('5') && readings.iter().any(|r| base(r) == base(reading)))
    }

    /// `reading` (g2pW's, for `c`) as a Mainland reading: kept if it is one,
    /// else the Mainland tone of the same syllable (期 qi2 -> qi1), else the
    /// only Mainland reading (圾 se4 -> ji1), else `fallback` (pypinyin in
    /// context) if Mainland, else unchanged.
    pub fn resolve(&self, c: char, reading: &str, fallback: Option<&str>) -> String {
        let Some(readings) = self.0.get(&c) else {
            return reading.to_string();
        };
        if self.allows(c, reading) {
            return reading.to_string();
        }
        let base = |r: &str| r.trim_end_matches(|d: char| d.is_ascii_digit()).to_string();
        let same_syllable: Vec<&String> = readings
            .iter()
            .filter(|r| base(r) == base(reading))
            .collect();
        if let [one] = same_syllable.as_slice() {
            return one.to_string();
        }
        if let [one] = readings.as_slice() {
            return one.clone();
        }
        match fallback {
            Some(fallback) if self.allows(c, fallback) => fallback.to_string(),
            _ => reading.to_string(),
        }
    }
}

const LIU2_PHRASES: [&str; 16] = [
    "分馏", "分餾", "干馏", "干餾", "乾馏", "乾餾", "精馏", "精餾", "蒸馏", "蒸餾", "馏出", "餾出",
    "馏分", "餾分", "馏程", "餾程",
];
const JUN1_PHRASES: [&str; 34] = [
    "古菌", "抗菌", "杆菌", "桿菌", "杀菌", "殺菌", "球菌", "病菌", "真菌", "细菌", "細菌", "腐菌",
    "菌丝", "菌絲", "菌体", "菌體", "菌株", "菌核", "菌根", "菌盖", "菌蓋", "菌种", "菌種", "菌类",
    "菌類", "菌群", "菌落", "菌褶", "菌门", "菌門", "菌齿", "菌齒", "菌管", "菌酶",
];
const BAI4_COMPOUNDS: [&str; 6] = ["梵呗", "梵唄", "美呗", "美唄", "赞呗", "讚唄"];
/// Local: Mainland reads 着 zhāo only in chess-move words; g2pW gives
/// Taiwan's zhao1 for 着急 and 着凉 (Mainland zháo).
const ZHAO1_PHRASES: [&str; 8] = [
    "着数", "着儿", "高着", "支着", "绝着", "花着", "损着", "招着",
];

fn in_phrase(text: &[char], index: usize, phrases: &[&str]) -> bool {
    phrases.iter().any(|phrase| {
        let p: Vec<char> = phrase.chars().collect();
        (0..text.len())
            .filter(|&start| text[start..].starts_with(&p))
            .any(|start| start <= index && index < start + p.len())
    })
}

fn is_spacing_or_punctuation(c: char) -> bool {
    c.is_whitespace() || (!c.is_alphanumeric() && !c.is_control())
}

/// `_normalize_g2pw_pinyin(text, index, syllable)`.
pub fn normalize_g2pw(text: &[char], index: usize, syllable: &str) -> String {
    let c = text[index];
    if let Some((_, _, corrected)) = CORRECTIONS
        .iter()
        .find(|(ch, from, _)| *ch == c && *from == syllable)
    {
        return corrected.to_string();
    }
    if "馏餾".contains(c) && syllable == "liu4" && in_phrase(text, index, &LIU2_PHRASES) {
        return "liu2".to_string();
    }
    if c == '菌' && syllable == "jun4" && in_phrase(text, index, &JUN1_PHRASES) {
        return "jun1".to_string();
    }
    if c == '着' && syllable == "zhao1" && !in_phrase(text, index, &ZHAO1_PHRASES) {
        return "zhao2".to_string();
    }
    if c == '耶' && syllable == "ye2" {
        if let Some(&next) = text.get(index + 1) {
            if next != '非' && !is_spacing_or_punctuation(next) {
                return "ye1".to_string();
            }
        }
    }
    if "呗唄".contains(c)
        && syllable == "bai4"
        && text[index + 1..]
            .iter()
            .all(|&n| is_spacing_or_punctuation(n))
        && !in_phrase(text, index, &BAI4_COMPOUNDS)
    {
        return "bei5".to_string();
    }
    syllable.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corrects_known_taiwan_readings() {
        let text: Vec<char> = "我和你，细菌".chars().collect();
        assert_eq!(normalize_g2pw(&text, 1, "han4"), "he2");
        assert_eq!(normalize_g2pw(&text, 5, "jun4"), "jun1");
        assert_eq!(normalize_g2pw(&text, 2, "ni3"), "ni3");
        let readings = MainlandReadings(
            [('期', "qi1 ji1"), ('圾', "ji1"), ('个', "ge3 ge4")]
                .into_iter()
                .map(|(c, r)| (c, r.split(' ').map(str::to_string).collect()))
                .collect(),
        );
        assert_eq!(readings.resolve('期', "qi2", None), "qi1");
        assert_eq!(readings.resolve('期', "ji1", None), "ji1");
        assert_eq!(readings.resolve('圾', "se4", None), "ji1");
        assert_eq!(readings.resolve('个', "ge5", None), "ge5");
        assert_eq!(readings.resolve('你', "ni3", None), "ni3");
        let text: Vec<char> = "别着急，高着儿".chars().collect();
        assert_eq!(normalize_g2pw(&text, 1, "zhao1"), "zhao2");
        assert_eq!(normalize_g2pw(&text, 5, "zhao1"), "zhao1");
    }
}
