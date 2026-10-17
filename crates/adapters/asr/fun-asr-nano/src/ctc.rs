//! The CTC branch's vocabulary (SenseVoice's multilingual tiktoken) and its
//! greedy decoding.

use base64::Engine as _;
use local_error::{InfraError, Result};
use serde::Deserialize;
use std::{fs, path::Path};

#[derive(Debug, Deserialize)]
struct CtcTokensFile {
    blank_id: usize,
    /// Ids from here on are special tokens (`<|zh|>`, `<|nospeech|>`, ...).
    special_start: usize,
    /// Each token's bytes, base64 (byte-level BPE pieces need not be UTF-8).
    tokens: Vec<String>,
}

#[derive(Debug)]
pub struct CtcVocab {
    blank: usize,
    special_start: usize,
    tokens: Vec<Vec<u8>>,
}

impl CtcVocab {
    pub fn load(path: &Path) -> Result<Self> {
        let file: CtcTokensFile = serde_json::from_slice(
            &fs::read(path).map_err(|e| InfraError::io(Some(path.to_path_buf()), e))?,
        )?;
        let tokens = file
            .tokens
            .iter()
            .map(|token| base64::engine::general_purpose::STANDARD.decode(token))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| InfraError::Adapter(format!("{}: invalid token: {e}", path.display())))?;
        if file.blank_id >= tokens.len() || file.special_start > tokens.len() {
            return Err(InfraError::Adapter(format!(
                "{}: blank {} / special start {} outside {} tokens",
                path.display(),
                file.blank_id,
                file.special_start,
                tokens.len()
            )));
        }
        Ok(Self {
            blank: file.blank_id,
            special_start: file.special_start,
            tokens,
        })
    }

    #[cfg(test)]
    pub fn from_parts(blank: usize, special_start: usize, tokens: Vec<Vec<u8>>) -> Self {
        Self {
            blank,
            special_start,
            tokens,
        }
    }

    /// Greedy CTC text of per-frame ids: repeats collapsed, blanks and
    /// special tokens dropped.
    pub fn decode(&self, ids: &[i64]) -> String {
        let mut bytes = Vec::new();
        let mut previous = None;
        for &id in ids {
            let Ok(index) = usize::try_from(id) else {
                previous = None;
                continue;
            };
            if Some(index) != previous && index != self.blank && index < self.special_start {
                if let Some(token) = self.tokens.get(index) {
                    bytes.extend_from_slice(token);
                }
            }
            previous = Some(index);
        }
        String::from_utf8_lossy(&bytes).trim().to_string()
    }
}
