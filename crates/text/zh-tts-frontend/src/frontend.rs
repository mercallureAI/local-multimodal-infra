//! PaddleSpeech `Frontend._g2p` with `g2p_model="g2pW"`, returning one
//! reading per input character instead of phonemes.
//!
//! Per clause: `jieba.posseg` segmentation, `ToneSandhi.pre_merge_for_modify`,
//! whole-clause g2pW, `Polyphonic.correct_pronunciation` per word, then
//! `ToneSandhi.modified_tone`. PaddleSpeech drops ASCII letters before G2P;
//! the result is mapped back onto the original characters. Erhua merging is
//! left out: the TTS models read "儿" themselves.
//!
//! Two optional layers go beyond PaddleSpeech, both off in `Options::paddlespeech`:
//! - `mainland`: g2p-mix's Mainland corrections of g2pW's Taiwan-leaning
//!   readings plus its phrase readings (applied last, over sandhi); pypinyin
//!   instead of g2pW's Taiwan single-reading table for the characters g2pW
//!   does not disambiguate, and its predictions mapped onto the Mainland
//!   standard readings (`mainland/readings.tsv`: 星期 qi1, 垃圾 la1 ji1);
//! - user phrases: caller-provided readings for words, applied last.

use crate::g2pw::G2pw;
use crate::mainland::MainlandReadings;
use crate::pinyin::{read_tsv, PinyinDict};
use crate::sandhi::{Seg, ToneSandhi};
use jieba_rs::Jieba;
use local_backend_ort::OrtBackend;
use local_error::{InfraError, Result};
use std::{collections::HashMap, path::Path};

/// Which of PaddleSpeech's `ToneSandhi.modified_tone` rules apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sandhi {
    /// 不, 一, neutral tone and third-tone sandhi: the tones as spoken.
    Full,
    /// Only the neutral-tone words (东西, 事情, 们, 了…); 不/一 and third
    /// tones keep their citation tones.
    NeutralTone,
    /// Citation tones (g2pW's own neutral tones stay).
    Off,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Options {
    pub mainland: bool,
    pub sandhi: Sandhi,
}

impl Options {
    /// Exactly PaddleSpeech's pipeline (parity reference).
    pub fn paddlespeech() -> Self {
        Self {
            mainland: false,
            sandhi: Sandhi::Full,
        }
    }
}

impl Default for Options {
    fn default() -> Self {
        Self {
            mainland: true,
            sandhi: Sandhi::Full,
        }
    }
}

pub struct ZhFrontend {
    jieba: Jieba,
    pinyin: PinyinDict,
    g2pw: G2pw,
    sandhi: ToneSandhi,
    polyphonic: HashMap<String, Vec<String>>,
    /// Words whose readings are fixed after sandhi (Mainland + user phrases).
    overrides: HashMap<String, Vec<String>>,
    /// Mainland standard readings (with `mainland`).
    readings: Option<MainlandReadings>,
    options: Options,
}

impl std::fmt::Debug for ZhFrontend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZhFrontend")
            .field("pinyin", &self.pinyin)
            .field("g2pw", &self.g2pw)
            .field("overrides", &self.overrides.len())
            .field("options", &self.options)
            .finish()
    }
}

/// Clause boundaries of PaddleSpeech `TextNormalizer.SENTENCE_SPLITOR`:
/// one of `：、，；。？！,;?!`, optionally followed by `”` or `’`.
fn split_clauses(chars: &[char]) -> Vec<std::ops::Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < chars.len() {
        if "：、，；。？！,;?!".contains(chars[i]) {
            let mut end = i + 1;
            if end < chars.len() && "”’".contains(chars[end]) {
                end += 1;
            }
            ranges.push(start..end);
            start = end;
            i = end;
        } else {
            i += 1;
        }
    }
    if start < chars.len() {
        ranges.push(start..chars.len());
    }
    ranges
}

fn parse_readings(value: &str) -> Vec<String> {
    value.split_whitespace().map(str::to_string).collect()
}

fn is_syllable(s: &str) -> bool {
    let bytes = s.as_bytes();
    bytes.len() >= 2
        && bytes[..bytes.len() - 1].iter().all(u8::is_ascii_lowercase)
        && (b'1'..=b'5').contains(&bytes[bytes.len() - 1])
}

impl ZhFrontend {
    /// Loads `g2pw/`, `pinyin/`, `paddlespeech/` and (with `mainland`)
    /// `mainland/` from the asset directory. g2pW runs on `backend`.
    pub fn load(dir: &Path, backend: &OrtBackend, options: Options) -> Result<Self> {
        if !dir.is_dir() {
            return Err(InfraError::ModelNotConfigured {
                model_id: "zh-tts-frontend".to_string(),
                reason: format!("frontend asset directory is missing: {}", dir.display()),
            });
        }
        let pinyin = PinyinDict::load(&dir.join("pinyin"))?;
        let g2pw = G2pw::load(&dir.join("g2pw"), &dir.join("pinyin/t2s.tsv"), backend)?;
        let polyphonic = read_tsv(&dir.join("paddlespeech/polyphonic.tsv"))?
            .into_iter()
            .map(|(word, value)| (word, parse_readings(&value)))
            .collect();
        let mut frontend = Self {
            jieba: Jieba::new(),
            pinyin,
            g2pw,
            sandhi: ToneSandhi::default(),
            polyphonic,
            overrides: HashMap::new(),
            readings: None,
            options,
        };
        if options.mainland {
            let readings = dir.join("mainland/readings.tsv");
            if readings.is_file() {
                frontend.readings = Some(MainlandReadings::load(&readings)?);
            } else {
                tracing::warn!(path = %readings.display(), "Mainland readings missing; g2pW keeps its Taiwan readings");
            }
            let phrases = read_tsv(&dir.join("mainland/phrases.tsv"))?
                .into_iter()
                .map(|(word, value)| (word, parse_readings(&value)))
                .collect::<Vec<_>>();
            frontend.add_phrases(phrases);
        }
        Ok(frontend)
    }

    /// Fixes the readings of whole words (e.g. names, product terms): they are
    /// added to the segmenter and override every other stage. Entries whose
    /// reading count differs from the word length are ignored.
    pub fn add_phrases(&mut self, phrases: impl IntoIterator<Item = (String, Vec<String>)>) {
        for (word, readings) in phrases {
            if readings.len() != word.chars().count() || !readings.iter().all(|r| is_syllable(r)) {
                continue;
            }
            self.jieba.add_word(&word, None, None);
            self.overrides.insert(word, readings);
        }
    }

    /// The dictionary (pypinyin) reading of `c` on its own: what a reader
    /// without context would say.
    pub fn dictionary_reading(&self, c: char) -> Option<&str> {
        self.pinyin.char_reading(c)
    }

    /// PaddleSpeech's neutral-tone rules rewrite only the tone digit, by
    /// word and tag: a verb 得 (dei3) or 着 (zhao2) mistagged as a particle
    /// becomes dei5 / zhao5. A rewritten reading g2pW does not know for the
    /// character keeps its tone.
    fn keep_possible_readings(
        &self,
        chars: &[char],
        before: Vec<String>,
        after: Vec<String>,
    ) -> Vec<String> {
        before
            .into_iter()
            .zip(after)
            .zip(chars)
            .map(|((before, after), &c)| {
                let candidates = self.g2pw.candidates(c);
                if after != before
                    && !candidates.is_empty()
                    && !candidates.contains(&after.as_str())
                {
                    before
                } else {
                    after
                }
            })
            .collect()
    }

    /// One reading (TONE3, neutral tone 5) per character of `text`; `None`
    /// for characters without a Mandarin reading.
    pub fn readings(&mut self, text: &str) -> Result<Vec<Option<String>>> {
        let chars: Vec<char> = text.chars().collect();
        let kept: Vec<usize> = (0..chars.len())
            .filter(|&i| !chars[i].is_ascii_alphabetic())
            .collect();
        let stripped: Vec<char> = kept.iter().map(|&i| chars[i]).collect();
        let mut result: Vec<Option<String>> = Vec::with_capacity(stripped.len());
        for range in split_clauses(&stripped) {
            let clause: String = stripped[range].iter().collect();
            result.extend(self.clause(&clause)?);
        }
        let mut full = vec![None; chars.len()];
        for (index, value) in kept.into_iter().zip(result) {
            full[index] = value;
        }
        Ok(full)
    }

    fn clause(&mut self, clause: &str) -> Result<Vec<Option<String>>> {
        let seg: Vec<Seg> = self
            .jieba
            .posseg_cut(clause)
            .into_iter()
            .map(|(word, tag)| (word.to_string(), tag.to_string()))
            .collect();
        let seg = self.sandhi.pre_merge(&self.pinyin, seg);
        let mut predicted = self
            .g2pw
            .predict(clause, &self.pinyin, self.readings.as_ref())?;
        let clause_chars: Vec<char> = clause.chars().collect();
        if self.options.mainland {
            for (index, reading) in predicted.iter_mut().enumerate() {
                if let Some(syllable) = reading.as_mut() {
                    *syllable = crate::mainland::normalize_g2pw(&clause_chars, index, syllable);
                }
            }
        }
        let mut out = Vec::with_capacity(clause_chars.len());
        let mut position = 0;
        for (word, pos) in seg {
            let word_chars: Vec<char> = word.chars().collect();
            let end = (position + word_chars.len()).min(predicted.len());
            if pos == "eng" {
                out.extend(std::iter::repeat_n(None, word_chars.len()));
                position = end;
                continue;
            }
            let word_readings: Vec<Option<String>> = match self.polyphonic.get(&word) {
                Some(fixed) => fixed.iter().cloned().map(Some).collect(),
                None => predicted[position..end].to_vec(),
            };
            let finals: Vec<String> = word_readings
                .iter()
                .zip(&word_chars)
                .map(|(reading, c)| reading.clone().unwrap_or_else(|| c.to_string()))
                .collect();
            let finals = match self.options.sandhi {
                Sandhi::Full => self.sandhi.modified_tone(&self.jieba, &word, &pos, finals),
                Sandhi::NeutralTone => {
                    let toned = self
                        .sandhi
                        .neutral_tone(&self.jieba, &word, &pos, finals.clone());
                    self.keep_possible_readings(&word_chars, finals, toned)
                }
                Sandhi::Off => finals,
            };
            let finals = match self.overrides.get(&word) {
                Some(fixed) => fixed.clone(),
                None => finals,
            };
            out.extend(
                finals
                    .into_iter()
                    .zip(&word_chars)
                    .map(|(f, c)| (is_syllable(&f) && f != c.to_string()).then_some(f)),
            );
            position = end;
        }
        out.resize(clause_chars.len(), None);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clauses_split_after_punctuation_and_closing_quotes() {
        let chars: Vec<char> = "他说：“好。”我们走，吧".chars().collect();
        let pieces: Vec<String> = split_clauses(&chars)
            .into_iter()
            .map(|r| chars[r].iter().collect())
            .collect();
        assert_eq!(pieces, ["他说：", "“好。”", "我们走，", "吧"]);
    }

    #[test]
    fn syllables_are_lowercase_letters_plus_tone() {
        assert!(is_syllable("hang2") && is_syllable("lv4") && is_syllable("er5"));
        assert!(!is_syllable("，") && !is_syllable("5") && !is_syllable("Hang2"));
    }
}
