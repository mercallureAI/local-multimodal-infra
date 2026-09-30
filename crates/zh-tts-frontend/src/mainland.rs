//! Mainland corrections of g2pW's Taiwan-leaning readings, from g2p-mix
//! (pengzhendong/g2p-mix 0893439, Apache-2.0, `backends/mandarin.py`
//! `_normalize_g2pw_pinyin`).

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
    }
}
