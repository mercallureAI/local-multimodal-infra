//! Qwen3's token embedding table, in host memory: the LLM graph takes
//! embeddings, and a decode step needs one row (keeping the 311 MB table off
//! the GPU).

use half::f16;
use local_error::{InfraError, Result};
use std::{fs, path::Path};

#[derive(Debug)]
pub struct EmbeddingTable {
    rows: Vec<f16>,
    vocab: usize,
    hidden: usize,
}

impl EmbeddingTable {
    /// `path`: `vocab` x `hidden` little-endian FP16, row-major.
    pub fn load(path: &Path, vocab: usize, hidden: usize) -> Result<Self> {
        let bytes = fs::read(path).map_err(|e| InfraError::io(Some(path.to_path_buf()), e))?;
        if bytes.len() != vocab * hidden * 2 {
            return Err(InfraError::Adapter(format!(
                "{} has {} bytes, expected {vocab} x {hidden} FP16",
                path.display(),
                bytes.len()
            )));
        }
        let rows = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| f16::from_le_bytes(*pair))
            .collect();
        Ok(Self {
            rows,
            vocab,
            hidden,
        })
    }

    /// Appends token `id`'s embedding (FP32) to `out`.
    pub fn append_row(&self, id: i64, out: &mut Vec<f32>) -> Result<()> {
        let index = usize::try_from(id)
            .ok()
            .filter(|&index| index < self.vocab)
            .ok_or_else(|| {
                InfraError::Adapter(format!(
                    "Fun-ASR-Nano token id {id} outside the vocabulary of {}",
                    self.vocab
                ))
            })?;
        out.extend(
            self.rows[index * self.hidden..(index + 1) * self.hidden]
                .iter()
                .map(|value| value.to_f32()),
        );
        Ok(())
    }

    pub fn row(&self, id: i64) -> Result<Vec<f32>> {
        let mut row = Vec::with_capacity(self.hidden);
        self.append_row(id, &mut row)?;
        Ok(row)
    }
}
