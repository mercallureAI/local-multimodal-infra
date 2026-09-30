//! FST-based text normalizer
//!
//! This module provides FST (Finite State Transducer) based text normalization,
//! equivalent to kaldifst.TextNormalizer in Python.

use std::path::Path;

use rustfst::algorithms::compose::compose;
use rustfst::algorithms::queues::AutoQueue;
use rustfst::algorithms::tr_filters::AnyTrFilter;
use rustfst::algorithms::tr_sort;
use rustfst::algorithms::Queue;

use rustfst::fst_impls::VectorFst;
use rustfst::fst_traits::SerializableFst;
use rustfst::prelude::*;
use rustfst::semirings::TropicalWeight;
use rustfst::utils::acceptor;
use rustfst::{Label, EPS_LABEL};

use crate::error::{Result, WeTextError};

/// FST-based text normalizer
///
/// Equivalent to kaldifst.TextNormalizer in Python
pub struct FstTextNormalizer {
    fst: VectorFst<TropicalWeight>,
}

impl FstTextNormalizer {
    /// Load FST from file
    ///
    /// # Arguments
    /// * `path` - Path to the FST file (OpenFST binary format)
    ///
    /// # Returns
    /// A new FstTextNormalizer instance
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(WeTextError::FstNotFound(path.display().to_string()));
        }

        // VectorFst::read() loads OpenFST binary format
        // This method comes from the SerializableFst trait
        let mut fst = VectorFst::<TropicalWeight>::read(path)
            .map_err(|e| WeTextError::FstLoadError(e.to_string()))?;
        // Composition matches the rule FST on its input side with a sorted
        // matcher; unsorted arcs silently drop paths, which then yields a
        // valid but non-optimal verbalization ("111" -> "one eleven").
        // kaldifst's TextNormalizer arc-sorts on load the same way.
        tr_sort(&mut fst, ILabelCompare {});

        Ok(Self { fst })
    }

    /// Apply FST for text transformation
    ///
    /// Implementation flow:
    /// 1. Convert input string to linear FST (acceptor) using UTF-8 bytes
    /// 2. Compose with the loaded FST
    /// 3. Find shortest path
    /// 4. Extract output string from the path
    ///
    /// # Arguments
    /// * `input` - Input text to normalize
    ///
    /// # Returns
    /// Normalized text string
    pub fn normalize(&self, input: &str) -> Result<String> {
        if input.is_empty() {
            return Ok(String::new());
        }

        // Step 1: Convert input string to linear FST using UTF-8 bytes
        // WeText FSTs use UTF-8 byte encoding for labels
        let labels: Vec<Label> = input.as_bytes().iter().map(|&b| b as Label).collect();
        let input_fst: VectorFst<TropicalWeight> = acceptor(&labels, TropicalWeight::one());

        // Step 2: Compose with the normalizer FST
        // Note: compose() requires output type to implement AllocableFst
        // Explicitly specify all type parameters for compose
        let composed: VectorFst<TropicalWeight> = compose::<
            TropicalWeight,
            VectorFst<TropicalWeight>,
            VectorFst<TropicalWeight>,
            VectorFst<TropicalWeight>,
            _,
            _,
        >(&input_fst, &self.fst)
        .map_err(|e| WeTextError::FstOperationError(format!("compose failed: {}", e)))?;

        // Check if compose result is empty (no match)
        if composed.num_states() == 0 {
            // If no match, return original input (same as kaldifst behavior)
            return Ok(input.to_string());
        }

        // Step 3: Find the best path. rustfst's `shortest_path` compares
        // tropical weights with `==`, which rustfst defines as equality within
        // KDELTA (1/1024); WeText grammars rank alternatives by ~1e-4 weight
        // differences ("one hundred and one" 0.9999 vs "one hundred one"
        // 1.00001), so that picks whichever path relaxed first. OpenFst (and
        // kaldifst) compare exactly; `best_path_olabels` does the same.
        let Some(olabels) = best_path_olabels(&composed)? else {
            return Ok(input.to_string());
        };

        // Step 4: kaldifst's FstToString keeps each non-epsilon output label
        // as one byte.
        let bytes: Vec<u8> = olabels
            .into_iter()
            .filter(|&label| label != EPS_LABEL)
            .map(|label| label as u8)
            .collect();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}

/// Output labels of the single best path, mirroring OpenFst's
/// `SingleShortestPath` (same queue discipline as rustfst's port) but with
/// exact tropical comparisons: a path replaces the current one only when it
/// is strictly cheaper, so exact ties keep the first path relaxed, as in
/// OpenFst. Returns `None` when no final state is reachable.
fn best_path_olabels(fst: &VectorFst<TropicalWeight>) -> Result<Option<Vec<Label>>> {
    let op_error = |e: anyhow::Error| WeTextError::FstOperationError(e.to_string());
    let Some(start) = fst.start() else {
        return Ok(None);
    };
    let states = fst.num_states();
    let mut distance = vec![f32::INFINITY; states];
    let mut parent: Vec<Option<(StateId, usize)>> = vec![None; states];
    let mut enqueued = vec![false; states];
    let mut queue = AutoQueue::new(fst, None, &AnyTrFilter {}).map_err(op_error)?;
    let mut best_final: Option<StateId> = None;
    let mut best_final_distance = f32::INFINITY;

    distance[start as usize] = 0.0;
    enqueued[start as usize] = true;
    queue.enqueue(start);
    while let Some(state) = queue.dequeue() {
        enqueued[state as usize] = false;
        let here = distance[state as usize];
        if let Some(final_weight) = fst.final_weight(state).map_err(op_error)? {
            let total = here + *final_weight.value();
            if total < best_final_distance {
                best_final_distance = total;
                best_final = Some(state);
            }
        }
        for (position, tr) in fst.get_trs(state).map_err(op_error)?.trs().iter().enumerate() {
            let next = tr.nextstate as usize;
            let candidate = here + *tr.weight.value();
            if candidate < distance[next] {
                distance[next] = candidate;
                parent[next] = Some((state, position));
                if enqueued[next] {
                    queue.update(tr.nextstate);
                } else {
                    queue.enqueue(tr.nextstate);
                    enqueued[next] = true;
                }
            }
        }
    }

    let Some(mut state) = best_final else {
        return Ok(None);
    };
    let mut olabels = Vec::new();
    while let Some((previous, position)) = parent[state as usize] {
        let trs = fst.get_trs(previous).map_err(op_error)?;
        olabels.push(trs.trs()[position].olabel);
        state = previous;
    }
    olabels.reverse();
    Ok(Some(olabels))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acceptor_creation() {
        let labels: Vec<Label> = "hello".chars().map(|c| c as Label).collect();
        let fst: VectorFst<TropicalWeight> = acceptor(&labels, TropicalWeight::one());
        assert_eq!(fst.num_states(), 6); // 5 chars + 1 (start state)
    }
}
