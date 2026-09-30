//! pypinyin's reading lookup, as PaddleSpeech configures it.
//!
//! Port of `pypinyin.pinyin(..., style=TONE3, neutral_tone_with_five=True,
//! heteronym=False)`: split text into Han / non-Han runs (`simpleseg`), cut Han
//! runs with pypinyin's strict forward maximum matching (`mmseg`,
//! `no_non_phrases=True`) and read each piece from the phrase table, falling
//! back to the first reading of each character. Readings come pre-converted
//! from `pinyin/chars.tsv` and `pinyin/phrases.tsv`, exported by pypinyin
//! itself after PaddleSpeech's `_init_pypinyin` (large_pinyin, custom phrases,
//! the 地 override).

use local_error::{InfraError, Result};
use std::{collections::HashMap, fs, path::Path};

pub struct PinyinDict {
    chars: HashMap<char, Box<str>>,
    /// Sorted by phrase; binary search answers both phrase lookup and
    /// pypinyin's prefix-set membership without materializing every prefix.
    phrases: Vec<(Box<str>, Box<str>)>,
}

impl std::fmt::Debug for PinyinDict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinyinDict")
            .field("chars", &self.chars.len())
            .field("phrases", &self.phrases.len())
            .finish()
    }
}

impl PinyinDict {
    pub fn load(dir: &Path) -> Result<Self> {
        let mut chars = HashMap::new();
        for (key, value) in read_tsv(&dir.join("chars.tsv"))? {
            let mut it = key.chars();
            if let (Some(c), None) = (it.next(), it.next()) {
                chars.insert(c, value.into_boxed_str());
            }
        }
        let mut phrases: Vec<(Box<str>, Box<str>)> = read_tsv(&dir.join("phrases.tsv"))?
            .into_iter()
            .map(|(k, v)| (k.into_boxed_str(), v.into_boxed_str()))
            .collect();
        phrases.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(Self { chars, phrases })
    }

    #[cfg(test)]
    pub(crate) fn from_parts(chars: &[(char, &str)], phrases: &[(&str, &str)]) -> Self {
        let mut phrases: Vec<(Box<str>, Box<str>)> = phrases
            .iter()
            .map(|(k, v)| ((*k).into(), (*v).into()))
            .collect();
        phrases.sort_by(|a, b| a.0.cmp(&b.0));
        Self {
            chars: chars.iter().map(|(c, v)| (*c, (*v).into())).collect(),
            phrases,
        }
    }

    /// First reading of one character (`None` for characters without one).
    pub fn char_reading(&self, c: char) -> Option<&str> {
        self.chars.get(&c).map(|v| &**v)
    }

    fn phrase(&self, word: &str) -> Option<&str> {
        self.phrases
            .binary_search_by(|(k, _)| (**k).cmp(word))
            .ok()
            .map(|i| &*self.phrases[i].1)
    }

    fn is_prefix(&self, prefix: &str) -> bool {
        let index = self.phrases.partition_point(|(k, _)| **k < *prefix);
        self.phrases
            .get(index)
            .is_some_and(|(k, _)| k.starts_with(prefix))
    }

    /// One reading per character of `text`; `None` where pypinyin has none
    /// (non-Han characters). Unlike pypinyin's own list, which keeps each
    /// non-Han run as one item, the result stays aligned to characters.
    pub fn readings(&self, text: &str) -> Vec<Option<String>> {
        let chars: Vec<char> = text.chars().collect();
        let mut out = Vec::with_capacity(chars.len());
        let mut start = 0;
        while start < chars.len() {
            let han = is_hans(chars[start]);
            let mut end = start + 1;
            while end < chars.len() && is_hans(chars[end]) == han {
                end += 1;
            }
            if han {
                for word in self.cut(&chars[start..end]) {
                    out.extend(self.word_readings(&word));
                }
            } else {
                out.extend(std::iter::repeat_n(None, end - start));
            }
            start = end;
        }
        out
    }

    fn word_readings(&self, word: &[char]) -> Vec<Option<String>> {
        let text: String = word.iter().collect();
        if let Some(reading) = self.phrase(&text) {
            let parts: Vec<&str> = reading.split(' ').collect();
            if parts.len() == word.len() {
                return parts.into_iter().map(|p| Some(p.to_string())).collect();
            }
        }
        word.iter()
            .map(|c| self.char_reading(*c).map(str::to_string))
            .collect()
    }

    /// `pypinyin.seg.mmseg.Seg.cut` with `no_non_phrases=True`.
    fn cut(&self, text: &[char]) -> Vec<Vec<char>> {
        let mut words = Vec::new();
        let mut remain = text;
        while !remain.is_empty() {
            let mut last_valid = 0usize;
            let mut broke = false;
            for index in 0..remain.len() {
                let word: String = remain[..=index].iter().collect();
                if self.is_prefix(&word) {
                    if self.phrase(&word).is_some() {
                        last_valid = index + 1;
                    }
                } else {
                    if last_valid > 0 {
                        words.push(remain[..last_valid].to_vec());
                        remain = &remain[last_valid..];
                    } else {
                        words.push(vec![remain[0]]);
                        remain = &remain[1..];
                    }
                    broke = true;
                    break;
                }
            }
            if !broke {
                if last_valid > 0 {
                    words.push(remain[..last_valid].to_vec());
                    remain = &remain[last_valid..];
                } else {
                    let all: String = remain.iter().collect();
                    if self.phrase(&all).is_some() {
                        words.push(remain.to_vec());
                    } else {
                        words.extend(remain.iter().map(|c| vec![*c]));
                    }
                    break;
                }
            }
        }
        words
    }
}

/// pypinyin `RE_HANS` (wide build): characters that have a Mandarin reading.
pub fn is_hans(c: char) -> bool {
    matches!(c as u32,
        0x3007 | 0xE815..=0xE864 | 0xFA18 | 0x3400..=0x4DBF | 0x4E00..=0x9FFF
        | 0xF900..=0xFAFF | 0x20000..=0x2A6DF | 0x2A703..=0x2B73F | 0x2B740..=0x2B81D
        | 0x2B825..=0x2BF6E | 0x2C029..=0x2CE93 | 0x2D016 | 0x2D11B..=0x2EBD9
        | 0x2F80A..=0x2FA1F | 0x30000..=0x3134A | 0x31350..=0x32389)
}

pub(crate) fn read_tsv(path: &Path) -> Result<Vec<(String, String)>> {
    let text = fs::read_to_string(path).map_err(|e| InfraError::io(Some(path.to_path_buf()), e))?;
    Ok(text
        .lines()
        .filter_map(|line| line.split_once('\t'))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_mmseg_prefers_longest_phrase_and_splits_non_phrases() {
        let dict = PinyinDict::from_parts(
            &[
                ('银', "yin2"),
                ('行', "xing2"),
                ('长', "zhang3"),
                ('大', "da4"),
            ],
            &[
                ("银行", "yin2 hang2"),
                ("长大", "zhang3 da4"),
                ("长大成人", "zhang3 da4 cheng2 ren2"),
            ],
        );
        let words: Vec<String> = dict
            .cut(&"银行长大".chars().collect::<Vec<_>>())
            .into_iter()
            .map(|w| w.into_iter().collect())
            .collect();
        assert_eq!(words, ["银行", "长大"]);
        assert_eq!(
            dict.readings("去银行!"),
            [None, Some("yin2".into()), Some("hang2".into()), None]
        );
    }
}
