//! IndexTTS-2.5 multilingual text tokenizer.
//!
//! Mirrors `indextts.utils.tokenizer.get_encoding("multilingual_zh_ja_yue_char_del")`:
//! a whisper-style tiktoken BPE whose special tokens (language tags, TTS vocal
//! tags, `SPECIAL_TOKEN_n` pronunciation markers, timestamps) are appended after
//! the mergeable ranks. The special-token table and language ids are read from
//! `tokenizer.json`, which the exporter writes from the official Python tables so
//! the two sides cannot drift.

use base64::Engine;
use local_error::{InfraError, Result};
use rustc_hash::FxHashMap;
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    fs,
    path::Path,
};
use tiktoken_rs::CoreBPE;

pub const TOKENIZER_MANIFEST_FILE: &str = "tokenizer.json";
pub const TOKENIZER_MANIFEST_SCHEMA: &str = "local.index_tts2.tokenizer.v1";

#[derive(Debug, Clone, Deserialize)]
struct TokenizerManifest {
    schema: String,
    tiktoken_file: String,
    pattern: String,
    special_tokens: Vec<(String, u32)>,
    languages: BTreeMap<String, i64>,
    fallback_language: String,
}

pub struct MultilingualTokenizer {
    bpe: CoreBPE,
    languages: BTreeMap<String, i64>,
    fallback_language: String,
}

impl std::fmt::Debug for MultilingualTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultilingualTokenizer")
            .field("languages", &self.languages.len())
            .field("fallback_language", &self.fallback_language)
            .finish_non_exhaustive()
    }
}

impl MultilingualTokenizer {
    pub fn load(root: &Path) -> Result<Self> {
        let manifest_path = root.join(TOKENIZER_MANIFEST_FILE);
        let manifest: TokenizerManifest = serde_json::from_slice(
            &fs::read(&manifest_path).map_err(|e| missing(&manifest_path, e))?,
        )
        .map_err(|e| {
            InfraError::Adapter(format!(
                "parse IndexTTS2 tokenizer manifest {}: {e}",
                manifest_path.display()
            ))
        })?;
        if manifest.schema != TOKENIZER_MANIFEST_SCHEMA {
            return Err(InfraError::Adapter(format!(
                "IndexTTS2 tokenizer manifest {} has schema `{}`, expected `{TOKENIZER_MANIFEST_SCHEMA}`",
                manifest_path.display(),
                manifest.schema
            )));
        }
        if !manifest.languages.contains_key(&manifest.fallback_language) {
            return Err(InfraError::Adapter(format!(
                "IndexTTS2 tokenizer fallback language `{}` is not in the language table",
                manifest.fallback_language
            )));
        }
        let ranks_path = root.join(&manifest.tiktoken_file);
        let ranks = read_tiktoken_ranks(&ranks_path)?;
        let first_special = ranks.len() as u32;
        for (index, (name, id)) in manifest.special_tokens.iter().enumerate() {
            if *id != first_special + index as u32 {
                return Err(InfraError::Adapter(format!(
                    "IndexTTS2 special token `{name}` has id {id}, expected {} after {} mergeable ranks",
                    first_special + index as u32,
                    ranks.len()
                )));
            }
        }
        let specials: FxHashMap<String, u32> = manifest.special_tokens.into_iter().collect();
        let bpe = CoreBPE::new(ranks, specials, &manifest.pattern).map_err(|e| {
            InfraError::Adapter(format!("build IndexTTS2 tiktoken BPE: {e}"))
        })?;
        Ok(Self {
            bpe,
            languages: manifest.languages,
            fallback_language: manifest.fallback_language,
        })
    }

    /// Encodes `text` with every special token allowed, like
    /// `encode(text, allowed_special="all")` upstream.
    pub fn encode(&self, text: &str) -> Vec<i32> {
        self.bpe
            .encode_with_special_tokens(text)
            .into_iter()
            .map(|id| id as i32)
            .collect()
    }

    pub fn token_len(&self, text: &str) -> usize {
        self.bpe.encode_with_special_tokens(text).len()
    }

    /// Returns the lowercased language code and the GPT language-embedding id.
    /// As upstream, the code itself is kept for the `<|code|>` text prefix even
    /// when unknown; only the embedding id falls back to `common`
    /// (`lang_to_token`).
    pub fn language(&self, code: &str) -> (String, i64) {
        let code = code.trim().to_lowercase();
        let id = self
            .languages
            .get(&code)
            .copied()
            .unwrap_or(self.languages[&self.fallback_language]);
        (code, id)
    }
}

fn read_tiktoken_ranks(path: &Path) -> Result<FxHashMap<Vec<u8>, u32>> {
    let text = fs::read_to_string(path).map_err(|e| missing(path, e))?;
    let engine = base64::engine::general_purpose::STANDARD;
    let mut ranks = FxHashMap::default();
    for (line_no, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let (Some(token), Some(rank), None) = (parts.next(), parts.next(), parts.next()) else {
            return Err(InfraError::Adapter(format!(
                "{}:{} is not a `<base64> <rank>` tiktoken line",
                path.display(),
                line_no + 1
            )));
        };
        // The upstream vocabulary contains a bare `=` line; Python's lenient
        // b64decode maps it to an empty token that still occupies a rank.
        let token = if token.trim_end_matches('=').is_empty() {
            Vec::new()
        } else {
            engine.decode(token).map_err(|e| {
                InfraError::Adapter(format!("{}:{}: {e}", path.display(), line_no + 1))
            })?
        };
        let rank = rank.parse::<u32>().map_err(|e| {
            InfraError::Adapter(format!("{}:{}: {e}", path.display(), line_no + 1))
        })?;
        ranks.insert(token, rank);
    }
    Ok(ranks)
}

fn missing(path: &Path, err: std::io::Error) -> InfraError {
    InfraError::ModelNotConfigured {
        model_id: "index-tts2".to_string(),
        reason: format!("{}: {err}", path.display()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Opt-in parity check against ids produced by the official Python
    /// `get_encoding(...).encode(text, allowed_special="all")`. Point
    /// `LOCAL_INDEXTTS2_MODEL_DIR` at a packaged IndexTTS-2.5 directory.
    #[test]
    fn matches_official_encoding_if_env_set() {
        let Ok(root) = std::env::var("LOCAL_INDEXTTS2_MODEL_DIR") else {
            return;
        };
        let tokenizer = MultilingualTokenizer::load(Path::new(&root)).expect("load tokenizer");
        let cases: [(&str, &[i32]); 3] = [
            (
                "<|zh|> 你好，世界！IndexTTS 2.5 发布了。",
                &[
                    58839, 220, 48934, 50371, 58827, 48721, 53743, 58824, 21214, 3099, 51, 7213,
                    568, 13, 20, 220, 49550, 50888, 48789, 1542,
                ],
            ),
            (
                "<|en|> hello world, it's 3:00 pm.",
                &[58838, 7627, 1002, 11, 309, 311, 805, 25, 628, 22406, 13],
            ),
            (
                "<|zh|> 晕<|SPECIAL_TOKEN_2|>XUAN4<|SPECIAL_TOKEN_2|>是一种感觉",
                &[
                    58839, 220, 51990, 58959, 55, 52, 1766, 19, 58959, 51971, 48706, 54284, 51305,
                    56210,
                ],
            ),
        ];
        for (text, expected) in cases {
            assert_eq!(tokenizer.encode(text), expected, "{text}");
        }
        assert_eq!(tokenizer.language("ZH"), ("zh".to_string(), 1));
        assert_eq!(tokenizer.language("klingon"), ("klingon".to_string(), 105));
    }
}
