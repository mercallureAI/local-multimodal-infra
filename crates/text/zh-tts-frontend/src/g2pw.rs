//! g2pW polyphone disambiguation, ported from PaddleSpeech's
//! `G2PWOnnxConverter` (`g2pw/onnx_api.py`, `dataset.py`, `utils.py`).
//!
//! Each clause is converted to Taiwan-standard characters (OpenCC `s2tw`, the
//! model's training script), every polyphonic character becomes one query
//! over the whole clause (`window_size=None`), and the INT8 graph scores the
//! query's candidate readings (`phoneme_mask`). Other characters take their
//! monophonic reading or pypinyin's.

use crate::mainland::MainlandReadings;
use crate::pinyin::{read_tsv, PinyinDict};
use ferrous_opencc::{config::BuiltinConfig, OpenCC};
use local_backend_ort::{OrtBackend, OrtSession, OrtTensorData, OrtTensorInput};
use local_error::{InfraError, Result};
use serde::Deserialize;
use std::{collections::HashMap, fs, path::Path};
use unicode_normalization::UnicodeNormalization;

#[derive(Deserialize)]
struct Meta {
    labels: Vec<String>,
    label_pinyin: Vec<Option<String>>,
    chars: Vec<String>,
    char_phonemes: HashMap<String, Vec<usize>>,
    query_chars: Vec<String>,
    monophonic: HashMap<String, Option<String>>,
    use_mask: bool,
    max_len: usize,
}

pub struct G2pw {
    session: OrtSession,
    tokenizer: WordPiece,
    label_count: usize,
    label_pinyin: Vec<Option<String>>,
    char_index: HashMap<char, i64>,
    char_phonemes: HashMap<char, Vec<usize>>,
    query_chars: std::collections::HashSet<char>,
    monophonic: HashMap<char, String>,
    use_mask: bool,
    max_len: usize,
    s2tw: OpenCC,
    t2s: HashMap<char, char>,
}

impl std::fmt::Debug for G2pw {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("G2pw")
            .field("labels", &self.label_count)
            .field("query_chars", &self.query_chars.len())
            .finish_non_exhaustive()
    }
}

fn single_char(s: &str) -> Option<char> {
    let mut it = s.chars();
    match (it.next(), it.next()) {
        (Some(c), None) => Some(c),
        _ => None,
    }
}

impl G2pw {
    pub fn load(dir: &Path, t2s_path: &Path, backend: &OrtBackend) -> Result<Self> {
        let meta_path = dir.join("meta.json");
        let meta: Meta = serde_json::from_slice(
            &fs::read(&meta_path).map_err(|e| InfraError::io(Some(meta_path.clone()), e))?,
        )
        .map_err(|e| InfraError::Adapter(format!("parse {}: {e}", meta_path.display())))?;
        let session = backend.load_session(dir.join("g2pw_int8.onnx"))?;
        let tokenizer = WordPiece::load(&dir.join("vocab.txt"))?;
        let char_index = meta
            .chars
            .iter()
            .enumerate()
            .filter_map(|(i, c)| single_char(c).map(|c| (c, i as i64)))
            .collect();
        let char_phonemes = meta
            .char_phonemes
            .into_iter()
            .filter_map(|(c, v)| single_char(&c).map(|c| (c, v)))
            .collect();
        let query_chars = meta
            .query_chars
            .iter()
            .filter_map(|c| single_char(c))
            .collect();
        let monophonic = meta
            .monophonic
            .into_iter()
            .filter_map(|(c, p)| Some((single_char(&c)?, p?)))
            .collect();
        let s2tw = OpenCC::from_config(BuiltinConfig::S2tw)
            .map_err(|e| InfraError::Adapter(format!("load OpenCC s2tw: {e}")))?;
        let t2s = read_tsv(t2s_path)?
            .into_iter()
            .filter_map(|(t, s)| Some((single_char(&t)?, single_char(&s)?)))
            .collect();
        Ok(Self {
            session,
            tokenizer,
            label_count: meta.labels.len(),
            label_pinyin: meta.label_pinyin,
            char_index,
            char_phonemes,
            query_chars,
            monophonic,
            use_mask: meta.use_mask,
            max_len: meta.max_len,
            s2tw,
            t2s,
        })
    }

    /// The readings g2pW chooses among for `c`; empty if it does not
    /// disambiguate `c`.
    pub fn candidates(&self, c: char) -> Vec<&str> {
        let translated = self.s2tw.convert(&c.to_string());
        let Some(c) = single_char(&translated) else {
            return Vec::new();
        };
        self.char_phonemes
            .get(&c)
            .map(|labels| {
                labels
                    .iter()
                    .filter_map(|&label| self.label_pinyin[label].as_deref())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// One reading per character of `sentence` (`None` without one).
    ///
    /// g2pW's labels and single-reading table come from a Taiwan dictionary
    /// (期 qi2, 危 wei2, 垃圾 le4 se4). With `mainland`, both are mapped onto
    /// Mainland readings (`MainlandReadings::resolve`, after g2p-mix's
    /// corrections).
    pub fn predict(
        &mut self,
        sentence: &str,
        pinyin: &PinyinDict,
        mainland: Option<&MainlandReadings>,
    ) -> Result<Vec<Option<String>>> {
        let translated = self.s2tw.convert(sentence);
        let translated: Vec<char> = if translated.chars().count() == sentence.chars().count() {
            translated.chars().collect()
        } else {
            // PaddleSpeech asserts equal length; keep the input rather than fail.
            sentence.chars().collect()
        };
        // pypinyin reads Simplified better (PaddleSpeech `_prepare_data`).
        let simplified: String = translated
            .iter()
            .map(|c| *self.t2s.get(c).unwrap_or(c))
            .collect();
        let fallback = pinyin.readings(&simplified);
        let sentence_chars: Vec<char> = sentence.chars().collect();
        let mut result: Vec<Option<String>> = vec![None; translated.len()];
        let mut queries = Vec::new();
        for (index, c) in translated.iter().enumerate() {
            if self.query_chars.contains(c) {
                queries.push(index);
            } else if let Some(reading) = self.monophonic.get(c) {
                result[index] = Some(match (mainland, sentence_chars.get(index)) {
                    (Some(m), Some(&original)) => m.resolve(
                        original,
                        reading,
                        fallback.get(index).cloned().flatten().as_deref(),
                    ),
                    _ => reading.clone(),
                });
            } else {
                result[index] = fallback.get(index).cloned().flatten();
            }
        }
        if queries.is_empty() {
            return Ok(result);
        }
        let lowered: Vec<char> = translated.iter().flat_map(|c| c.to_lowercase()).collect();
        if lowered.len() != translated.len() {
            return Ok(result);
        }
        let (tokens, text2token) = self.tokenizer.tokenize_and_map(&lowered);
        let (tokens, text2token, offset) = truncate(self.max_len, &queries, tokens, text2token);
        let mut ids = Vec::with_capacity(tokens.len() + 2);
        ids.push(self.tokenizer.id("[CLS]"));
        ids.extend(tokens.iter().map(|t| self.tokenizer.id(t)));
        ids.push(self.tokenizer.id("[SEP]"));
        let width = ids.len();
        let batch = queries.len();

        let mut phoneme_mask = Vec::with_capacity(batch * self.label_count);
        let mut char_ids = Vec::with_capacity(batch);
        let mut position_ids = Vec::with_capacity(batch);
        let mut kept = Vec::with_capacity(batch);
        for &query in &queries {
            let c = lowered[query];
            let (Some(char_id), Some(candidates), Some(Some(token))) = (
                self.char_index.get(&c),
                self.char_phonemes.get(&c),
                text2token.get(query.wrapping_sub(offset)),
            ) else {
                continue;
            };
            let mut mask = vec![if self.use_mask { 0.0f32 } else { 1.0 }; self.label_count];
            if self.use_mask {
                for &label in candidates {
                    mask[label] = 1.0;
                }
            }
            phoneme_mask.extend(mask);
            char_ids.push(*char_id);
            position_ids.push(*token as i64 + 1);
            kept.push(query);
        }
        if kept.is_empty() {
            return Ok(result);
        }
        let batch = kept.len();
        let input_ids: Vec<i64> = ids.iter().cycle().take(batch * width).copied().collect();
        let outputs = self.session.run_tensors(&[
            tensor_i64("input_ids", vec![batch, width], input_ids),
            tensor_i64("token_type_ids", vec![batch, width], vec![0; batch * width]),
            tensor_i64("attention_mask", vec![batch, width], vec![1; batch * width]),
            OrtTensorInput {
                name: "phoneme_mask".to_string(),
                shape: vec![batch, self.label_count],
                data: OrtTensorData::F32(phoneme_mask),
            },
            tensor_i64("char_ids", vec![batch], char_ids),
            tensor_i64("position_ids", vec![batch], position_ids),
        ])?;
        let probs = match &outputs.first().map(|o| &o.data) {
            Some(OrtTensorData::F32(values)) => values.clone(),
            _ => {
                return Err(InfraError::Backend(
                    "g2pW returned no f32 probabilities".to_string(),
                ))
            }
        };
        for (row, &query) in kept.iter().enumerate() {
            let scores = &probs[row * self.label_count..(row + 1) * self.label_count];
            // np.argmax: first maximum wins ties.
            let mut best = 0;
            for (label, score) in scores.iter().enumerate() {
                if *score > scores[best] {
                    best = label;
                }
            }
            result[query] = self.label_pinyin.get(best).cloned().flatten();
            if let (Some(m), Some(reading), Some(&c)) = (
                mainland,
                result[query].as_deref(),
                sentence_chars.get(query),
            ) {
                // g2p-mix's corrections first: they know 缉 qi4 is ji1.
                let reading = crate::mainland::normalize_g2pw(&sentence_chars, query, reading);
                let fallback = fallback.get(query).cloned().flatten();
                result[query] = Some(m.resolve(c, &reading, fallback.as_deref()));
            }
        }
        Ok(result)
    }
}

fn tensor_i64(name: &str, shape: Vec<usize>, data: Vec<i64>) -> OrtTensorInput {
    OrtTensorInput {
        name: name.to_string(),
        shape,
        data: OrtTensorData::I64(data),
    }
}

/// `dataset._truncate` for sentences longer than `max_len - 2` tokens, applied
/// around the first query (PaddleSpeech truncates per query; clauses are
/// short, so one shared window is kept and out-of-window queries skipped).
fn truncate(
    max_len: usize,
    queries: &[usize],
    tokens: Vec<String>,
    text2token: Vec<Option<usize>>,
) -> (Vec<String>, Vec<Option<usize>>, usize) {
    let limit = max_len.saturating_sub(2);
    if tokens.len() <= limit {
        return (tokens, text2token, 0);
    }
    let anchor = text2token[queries[0]].unwrap_or(0);
    let start = anchor.saturating_sub(limit / 2).min(tokens.len() - limit);
    let end = start + limit;
    let text2token = text2token
        .into_iter()
        .map(|t| t.and_then(|t| (start..end).contains(&t).then(|| t - start)))
        .collect();
    (tokens[start..end].to_vec(), text2token, 0)
}

/// BERT `BasicTokenizer` + `WordpieceTokenizer` (lowercase, strip accents) over
/// the `wordize_and_map` units g2pW feeds it: runs of `[a-zA-Z0-9]`, spaces,
/// or single characters.
struct WordPiece {
    vocab: HashMap<String, i64>,
    unk: i64,
}

impl WordPiece {
    fn load(path: &Path) -> Result<Self> {
        let text =
            fs::read_to_string(path).map_err(|e| InfraError::io(Some(path.to_path_buf()), e))?;
        let vocab: HashMap<String, i64> = text
            .lines()
            .enumerate()
            .map(|(i, token)| (token.to_string(), i as i64))
            .collect();
        let unk = *vocab
            .get("[UNK]")
            .ok_or_else(|| InfraError::Adapter("g2pW vocab has no [UNK]".to_string()))?;
        Ok(Self { vocab, unk })
    }

    fn id(&self, token: &str) -> i64 {
        self.vocab.get(token).copied().unwrap_or(self.unk)
    }

    /// `utils.tokenize_and_map`: tokens plus, per text character, its token.
    fn tokenize_and_map(&self, text: &[char]) -> (Vec<String>, Vec<Option<usize>>) {
        let mut tokens = Vec::new();
        let mut text2token = vec![None; text.len()];
        let mut index = 0;
        while index < text.len() {
            let c = text[index];
            if c == ' ' {
                index += 1;
                continue;
            }
            let end = if c.is_ascii_alphanumeric() {
                let mut end = index;
                while end < text.len() && text[end].is_ascii_alphanumeric() {
                    end += 1;
                }
                end
            } else {
                index + 1
            };
            let word: String = text[index..end].iter().collect();
            let pieces = self.word_pieces(&word);
            if pieces.is_empty() || pieces == ["[UNK]"] {
                for slot in &mut text2token[index..end] {
                    *slot = Some(tokens.len());
                }
                tokens.push("[UNK]".to_string());
            } else {
                let mut position = index;
                for piece in pieces {
                    let length = piece.trim_start_matches("##").chars().count();
                    for slot in text2token.iter_mut().skip(position).take(length) {
                        *slot = Some(tokens.len());
                    }
                    position += length;
                    tokens.push(piece);
                }
            }
            index = end;
        }
        (tokens, text2token)
    }

    /// `BertTokenizer.tokenize` on one unit. Units are single characters or
    /// ASCII alphanumeric runs, so BasicTokenizer's punctuation split never
    /// applies; what remains is cleaning, lowercasing and accent stripping.
    fn word_pieces(&self, word: &str) -> Vec<String> {
        let cleaned: String = word
            .chars()
            .filter(|c| !(*c == '\0' || *c == '\u{fffd}' || is_control(*c)))
            .collect();
        if cleaned.chars().all(char::is_whitespace) {
            return Vec::new();
        }
        let lowered: String = cleaned
            .to_lowercase()
            .nfd()
            .filter(|c| !is_mark(*c))
            .collect();
        if lowered.is_empty() {
            return Vec::new();
        }
        self.wordpiece(&lowered)
    }

    fn wordpiece(&self, word: &str) -> Vec<String> {
        let chars: Vec<char> = word.chars().collect();
        if chars.len() > 100 {
            return vec!["[UNK]".to_string()];
        }
        let mut pieces = Vec::new();
        let mut start = 0;
        while start < chars.len() {
            let mut end = chars.len();
            let mut found = None;
            while start < end {
                let mut candidate: String = chars[start..end].iter().collect();
                if start > 0 {
                    candidate = format!("##{candidate}");
                }
                if self.vocab.contains_key(&candidate) {
                    found = Some(candidate);
                    break;
                }
                end -= 1;
            }
            match found {
                Some(piece) => {
                    pieces.push(piece);
                    start = end;
                }
                None => return vec!["[UNK]".to_string()],
            }
        }
        pieces
    }
}

fn is_control(c: char) -> bool {
    if c == '\t' || c == '\n' || c == '\r' {
        return false;
    }
    c.is_control()
}

fn is_mark(c: char) -> bool {
    matches!(c as u32, 0x0300..=0x036F | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF | 0x20D0..=0x20FF | 0xFE20..=0xFE2F)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wordpiece(tokens: &[&str]) -> WordPiece {
        let vocab: HashMap<String, i64> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.to_string(), i as i64))
            .collect();
        WordPiece {
            unk: vocab["[UNK]"],
            vocab,
        }
    }

    #[test]
    fn maps_characters_to_wordpiece_tokens() {
        let tokenizer = wordpiece(&[
            "[UNK]", "[CLS]", "[SEP]", "银", "行", "index", "##tts", "，",
        ]);
        let text: Vec<char> = "银行 indextts，".chars().collect();
        let (tokens, map) = tokenizer.tokenize_and_map(&text);
        assert_eq!(tokens, ["银", "行", "index", "##tts", "，"]);
        assert_eq!(map[0], Some(0));
        assert_eq!(map[2], None);
        assert_eq!(map[3], Some(2));
        assert_eq!(map[8], Some(3));
        assert_eq!(map[11], Some(4));
    }

    #[test]
    fn unknown_units_become_one_unk_token() {
        let tokenizer = wordpiece(&["[UNK]", "a"]);
        let (tokens, map) = tokenizer.tokenize_and_map(&"龘a".chars().collect::<Vec<_>>());
        assert_eq!(tokens, ["[UNK]", "a"]);
        assert_eq!(map, [Some(0), Some(1)]);
    }
}
