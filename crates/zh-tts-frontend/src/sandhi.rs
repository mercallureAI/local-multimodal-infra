//! Mandarin tone sandhi, ported from PaddleSpeech `tone_sandhi.py`
//! (`ToneSandhi.pre_merge_for_modify` / `modified_tone`).
//!
//! Syllables stand in for PaddleSpeech's "finals": every rule only reads or
//! rewrites the trailing tone digit, so whole syllables ("hao3") behave the
//! same. Non-syllable entries (punctuation kept as itself) pass through the
//! same string operations as in Python and are discarded by the caller.

use crate::pinyin::{is_hans, PinyinDict};
use crate::sandhi_words::{CJK_NUMERIC, MUST_NEURAL_TONE_WORDS, MUST_NOT_NEURAL_TONE_WORDS, PUNC};
use jieba_rs::Jieba;
use std::collections::HashSet;

pub struct ToneSandhi {
    must_neural: HashSet<&'static str>,
    must_not_neural: HashSet<&'static str>,
}

impl std::fmt::Debug for ToneSandhi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ToneSandhi")
    }
}

impl Default for ToneSandhi {
    fn default() -> Self {
        Self {
            must_neural: MUST_NEURAL_TONE_WORDS.into_iter().collect(),
            must_not_neural: MUST_NOT_NEURAL_TONE_WORDS.into_iter().collect(),
        }
    }
}

/// One `(word, pos)` segment.
pub type Seg = (String, String);

fn chars(s: &str) -> Vec<char> {
    s.chars().collect()
}

fn len(s: &str) -> usize {
    s.chars().count()
}

/// `finals[i][:-1] + tone`
fn set_tone(final_: &mut String, tone: char) {
    final_.pop();
    final_.push(tone);
}

fn last_char(s: &str) -> Option<char> {
    s.chars().last()
}

fn tone_is(s: &str, tone: char) -> bool {
    last_char(s) == Some(tone)
}

fn all_tone_three(finals: &[String]) -> bool {
    finals.iter().all(|f| tone_is(f, '3'))
}

/// Python `str.isnumeric()` for one character.
fn is_numeric(c: char) -> bool {
    c.is_numeric() || CJK_NUMERIC.contains(c)
}

impl ToneSandhi {
    fn in_must_neural(&self, word: &str) -> bool {
        let c = chars(word);
        let tail: String = c[c.len().saturating_sub(2)..].iter().collect();
        self.must_neural.contains(word) || self.must_neural.contains(tail.as_str())
    }

    /// `_split_word`: split a word at its shortest `cut_for_search` sub-word.
    fn split_word(&self, jieba: &Jieba, word: &str) -> [String; 2] {
        let mut subwords: Vec<&str> = jieba
            .cut_for_search(word, true)
            .into_iter()
            .map(|t| t.word)
            .collect();
        subwords.sort_by_key(|w| len(w));
        let first = subwords.first().copied().unwrap_or(word);
        match word.find(first) {
            Some(0) => [first.to_string(), word[first.len()..].to_string()],
            // Python drops len(first) characters from the end whether or not
            // `first` really is the suffix.
            _ => {
                let w = chars(word);
                let keep = w.len().saturating_sub(len(first));
                [w[..keep].iter().collect(), first.to_string()]
            }
        }
    }

    // The branches mirror PaddleSpeech's `elif` chain one to one.
    #[allow(clippy::if_same_then_else)]
    fn neural_sandhi(
        &self,
        jieba: &Jieba,
        word: &str,
        pos: &str,
        mut finals: Vec<String>,
    ) -> Vec<String> {
        if self.must_not_neural.contains(word) {
            return finals;
        }
        let w = chars(word);
        let n = w.len();
        if pos.starts_with(['n', 'v', 'a']) {
            for j in 1..n {
                if w[j] == w[j - 1] {
                    set_tone(&mut finals[j], '5');
                }
            }
        }
        let ge = w.iter().position(|&c| c == '个');
        let last = n.checked_sub(1);
        if let Some(last) =
            last.filter(|&l| "吧呢啊呐噻嘛吖嗨呐哦哒滴哩哟喽啰耶喔诶".contains(w[l]))
        {
            set_tone(&mut finals[last], '5');
        } else if let Some(last) = last.filter(|&l| "的地得".contains(w[l])) {
            set_tone(&mut finals[last], '5');
        } else if n == 1 && "了着过".contains(w[0]) && matches!(pos, "ul" | "uz" | "ug") {
            set_tone(&mut finals[0], '5');
        } else if n > 1 && "们子".contains(w[n - 1]) && matches!(pos, "r" | "n") {
            set_tone(&mut finals[n - 1], '5');
        } else if n > 1 && "上下".contains(w[n - 1]) && matches!(pos, "s" | "l" | "f") {
            set_tone(&mut finals[n - 1], '5');
        } else if n > 1 && "来去".contains(w[n - 1]) && "上下进出回过起开".contains(w[n - 2])
        {
            set_tone(&mut finals[n - 1], '5');
        } else if let Some(ge) = ge.filter(|&g| {
            (g >= 1 && (is_numeric(w[g - 1]) || "几有两半多各整每做是".contains(w[g - 1])))
                || word == "个"
        }) {
            set_tone(&mut finals[ge], '5');
        } else if self.in_must_neural(word) {
            if let Some(last) = last {
                set_tone(&mut finals[last], '5');
            }
        }

        let [first, second] = self.split_word(jieba, word);
        let split = len(&first).min(finals.len());
        let (mut head, mut tail) = (finals[..split].to_vec(), finals[split..].to_vec());
        for (sub, part) in [(&first, &mut head), (&second, &mut tail)] {
            if self.in_must_neural(sub) {
                if let Some(last) = part.last_mut() {
                    set_tone(last, '5');
                }
            }
        }
        head.extend(tail);
        head
    }

    fn bu_sandhi(&self, word: &str, mut finals: Vec<String>) -> Vec<String> {
        let w = chars(word);
        if w.len() == 3 && w[1] == '不' {
            set_tone(&mut finals[1], '5');
        } else {
            for i in 0..w.len() {
                if w[i] == '不' && i + 1 < w.len() && tone_is(&finals[i + 1], '4') {
                    set_tone(&mut finals[i], '2');
                }
            }
        }
        finals
    }

    fn yi_sandhi(&self, word: &str, mut finals: Vec<String>) -> Vec<String> {
        let w = chars(word);
        if w.contains(&'一') && w.iter().filter(|&&c| c != '一').all(|&c| is_numeric(c)) {
            return finals;
        }
        if w.len() == 3 && w[1] == '一' && w[0] == w[2] {
            set_tone(&mut finals[1], '5');
        } else if word.starts_with("第一") {
            set_tone(&mut finals[1], '1');
        } else {
            for i in 0..w.len() {
                if w[i] == '一' && i + 1 < w.len() {
                    if tone_is(&finals[i + 1], '4') || tone_is(&finals[i + 1], '5') {
                        set_tone(&mut finals[i], '2');
                    } else if !PUNC.contains(w[i + 1]) {
                        set_tone(&mut finals[i], '4');
                    }
                }
            }
        }
        finals
    }

    fn three_sandhi(&self, jieba: &Jieba, word: &str, mut finals: Vec<String>) -> Vec<String> {
        let n = len(word);
        if n == 2 && all_tone_three(&finals) {
            set_tone(&mut finals[0], '2');
        } else if n == 3 {
            let [first, _] = self.split_word(jieba, word);
            let split = len(&first).min(finals.len());
            if all_tone_three(&finals) {
                if split == 2 {
                    set_tone(&mut finals[0], '2');
                    set_tone(&mut finals[1], '2');
                } else if split == 1 {
                    set_tone(&mut finals[1], '2');
                }
            } else {
                let mut parts = [finals[..split].to_vec(), finals[split..].to_vec()];
                for i in 0..2 {
                    let sub_three = all_tone_three(&parts[i]);
                    if sub_three && parts[i].len() == 2 {
                        set_tone(&mut parts[i][0], '2');
                    } else if i == 1
                        && !sub_three
                        && parts[1].first().is_some_and(|f| tone_is(f, '3'))
                        && parts[0].last().is_some_and(|f| tone_is(f, '3'))
                    {
                        if let Some(last) = parts[0].last_mut() {
                            set_tone(last, '2');
                        }
                    }
                    finals = parts.concat();
                }
            }
        } else if n == 4 {
            let mut out = Vec::with_capacity(finals.len());
            for mut sub in [finals[..2].to_vec(), finals[2..].to_vec()] {
                if all_tone_three(&sub) {
                    set_tone(&mut sub[0], '2');
                }
                out.extend(sub);
            }
            finals = out;
        }
        finals
    }

    /// `modified_tone`: 不, 一, neutral tone, then third-tone sandhi.
    pub fn modified_tone(
        &self,
        jieba: &Jieba,
        word: &str,
        pos: &str,
        finals: Vec<String>,
    ) -> Vec<String> {
        if finals.len() != len(word) || finals.is_empty() {
            return finals;
        }
        let finals = self.bu_sandhi(word, finals);
        let finals = self.yi_sandhi(word, finals);
        let finals = self.neural_sandhi(jieba, word, pos, finals);
        self.three_sandhi(jieba, word, finals)
    }

    /// Only the neutral-tone rules of `modified_tone` (`_neural_sandhi`).
    pub fn neutral_tone(
        &self,
        jieba: &Jieba,
        word: &str,
        pos: &str,
        finals: Vec<String>,
    ) -> Vec<String> {
        if finals.len() != len(word) || finals.is_empty() {
            return finals;
        }
        self.neural_sandhi(jieba, word, pos, finals)
    }

    /// `pre_merge_for_modify`.
    pub fn pre_merge(&self, pinyin: &PinyinDict, seg: Vec<Seg>) -> Vec<Seg> {
        let seg = merge_bu(seg);
        let seg = merge_yi(seg);
        let seg = merge_reduplication(seg);
        let seg = merge_continuous_three_tones(pinyin, seg, |prev, cur| {
            all_tone_three(prev) && all_tone_three(cur)
        });
        let seg = merge_continuous_three_tones(pinyin, seg, |prev, cur| {
            prev.last().is_some_and(|f| tone_is(f, '3'))
                && cur.first().is_some_and(|f| tone_is(f, '3'))
        });
        merge_er(seg)
    }
}

fn merge_bu(seg: Vec<Seg>) -> Vec<Seg> {
    let mut out = Vec::with_capacity(seg.len());
    let mut last_word = String::new();
    for (mut word, pos) in seg {
        if last_word == "不" {
            word = format!("{last_word}{word}");
        }
        if word != "不" {
            out.push((word.clone(), pos));
        }
        last_word = word;
    }
    if last_word == "不" {
        out.push((last_word, "d".to_string()));
    }
    out
}

fn merge_yi(seg: Vec<Seg>) -> Vec<Seg> {
    let mut first: Vec<Seg> = Vec::with_capacity(seg.len());
    let mut skip_next = false;
    for i in 0..seg.len() {
        if skip_next {
            skip_next = false;
            continue;
        }
        let (word, pos) = &seg[i];
        if i >= 1
            && word == "一"
            && i + 1 < seg.len()
            && seg[i - 1].0 == seg[i + 1].0
            && seg[i - 1].1 == "v"
        {
            if let Some(last) = first.last_mut() {
                last.0 = format!("{}一{}", last.0, seg[i + 1].0);
            }
            skip_next = true;
        } else {
            first.push((word.clone(), pos.clone()));
        }
    }
    let mut out: Vec<Seg> = Vec::with_capacity(first.len());
    for (word, pos) in first {
        match out.last_mut() {
            Some(last) if last.0 == "一" => last.0.push_str(&word),
            _ => out.push((word, pos)),
        }
    }
    out
}

fn merge_reduplication(seg: Vec<Seg>) -> Vec<Seg> {
    let mut out: Vec<Seg> = Vec::with_capacity(seg.len());
    for (word, pos) in seg {
        match out.last_mut() {
            Some(last) if last.0 == word => last.0.push_str(&word),
            _ => out.push((word, pos)),
        }
    }
    out
}

fn is_reduplication(word: &str) -> bool {
    let w = chars(word);
    w.len() == 2 && w[0] == w[1]
}

/// `lazy_pinyin(word, style=FINALS_TONE3, neutral_tone_with_five=True)` with
/// the 嗯 -> n2 fix: one entry per Han character, one per non-Han run.
fn lazy_finals(pinyin: &PinyinDict, word: &str) -> Vec<String> {
    let w = chars(word);
    let readings = pinyin.readings(word);
    let mut out = Vec::new();
    let mut i = 0;
    while i < w.len() {
        if is_hans(w[i]) {
            out.push(readings[i].clone().unwrap_or_else(|| w[i].to_string()));
            i += 1;
        } else {
            let start = i;
            while i < w.len() && !is_hans(w[i]) {
                i += 1;
            }
            out.push(w[start..i].iter().collect());
        }
    }
    for (index, c) in w.iter().enumerate() {
        if *c == '嗯' && index < out.len() {
            out[index] = "n2".to_string();
        }
    }
    out
}

/// `_merge_continuous_three_tones` / `_merge_continuous_three_tones_2`.
fn merge_continuous_three_tones(
    pinyin: &PinyinDict,
    seg: Vec<Seg>,
    joins: impl Fn(&[String], &[String]) -> bool,
) -> Vec<Seg> {
    let finals: Vec<Vec<String>> = seg
        .iter()
        .map(|(word, _)| lazy_finals(pinyin, word))
        .collect();
    let mut merged = vec![false; seg.len()];
    let mut out: Vec<Seg> = Vec::with_capacity(seg.len());
    for i in 0..seg.len() {
        let (word, pos) = &seg[i];
        if i >= 1 && joins(&finals[i - 1], &finals[i]) && !merged[i - 1] {
            if !is_reduplication(&seg[i - 1].0) && len(&seg[i - 1].0) + len(word) <= 3 {
                if let Some(last) = out.last_mut() {
                    last.0.push_str(word);
                }
                merged[i] = true;
            } else {
                out.push((word.clone(), pos.clone()));
            }
        } else {
            out.push((word.clone(), pos.clone()));
        }
    }
    out
}

fn merge_er(seg: Vec<Seg>) -> Vec<Seg> {
    let mut out: Vec<Seg> = Vec::with_capacity(seg.len());
    for (i, (word, pos)) in seg.into_iter().enumerate() {
        match out.last_mut() {
            Some(last) if i >= 1 && word == "儿" => last.0.push_str(&word),
            _ => out.push((word, pos)),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finals(tones: &[&str]) -> Vec<String> {
        tones.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn bu_and_yi_follow_paddlespeech() {
        let sandhi = ToneSandhi::default();
        assert_eq!(
            sandhi.bu_sandhi("看不懂", finals(&["kan4", "bu4", "dong3"]))[1],
            "bu5"
        );
        assert_eq!(sandhi.bu_sandhi("不怕", finals(&["bu4", "pa4"]))[0], "bu2");
        assert_eq!(
            sandhi.yi_sandhi("一段", finals(&["yi1", "duan4"]))[0],
            "yi2"
        );
        assert_eq!(
            sandhi.yi_sandhi("一天", finals(&["yi1", "tian1"]))[0],
            "yi4"
        );
        assert_eq!(sandhi.yi_sandhi("第一", finals(&["di4", "yi1"]))[1], "yi1");
        assert_eq!(
            sandhi.yi_sandhi("看一看", finals(&["kan4", "yi1", "kan4"]))[1],
            "yi5"
        );
        assert_eq!(
            sandhi.yi_sandhi("一零一", finals(&["yi1", "ling2", "yi1"]))[0],
            "yi1"
        );
    }

    #[test]
    fn merges_bu_yi_and_reduplication() {
        let seg = |pairs: &[(&str, &str)]| -> Vec<Seg> {
            pairs
                .iter()
                .map(|(w, p)| (w.to_string(), p.to_string()))
                .collect()
        };
        assert_eq!(
            merge_bu(seg(&[("不", "d"), ("好", "a")])),
            seg(&[("不好", "a")])
        );
        assert_eq!(
            merge_yi(seg(&[("听", "v"), ("一", "m"), ("听", "v")])),
            seg(&[("听一听", "v")])
        );
        assert_eq!(
            merge_reduplication(seg(&[("试", "v"), ("试", "v")])),
            seg(&[("试试", "v")])
        );
    }
}
