//! The latest inferences, in memory: when each started and ended and how
//! long its stages took, for finding where time goes (`GET /v1/inferences`).
//! Realtime pipelines add their own records (a voice turn's stages).

use std::collections::{BTreeMap, VecDeque};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

/// Records kept (the query takes the latest it wants of them).
pub const KEPT: usize = 1000;

/// One inference, or one stage-timed span of a pipeline.
#[derive(Clone, Debug, Serialize)]
pub struct InferenceRecord {
    pub id: String,
    /// The task kind (`asr.transcribe`, `ocr.lines`, ...) or a pipeline's
    /// (`voice.turn`).
    pub kind: String,
    pub model: String,
    /// Unix time, milliseconds.
    pub started_at_ms: u64,
    pub ended_at_ms: u64,
    pub total_ms: u64,
    /// Its stages, milliseconds (queue, load, execution, first_output; a
    /// voice turn's vad, asr, gate, llm, tts).
    pub stages_ms: BTreeMap<String, u64>,
    pub ok: bool,
}

/// Unix time now, milliseconds.
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

#[derive(Debug, Default)]
pub struct Recent {
    records: Mutex<VecDeque<InferenceRecord>>,
}

impl Recent {
    pub fn push(&self, record: InferenceRecord) {
        let mut records = self
            .records
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        records.push_back(record);
        while records.len() > KEPT {
            records.pop_front();
        }
    }

    /// Newest first.
    pub fn list(&self) -> Vec<InferenceRecord> {
        let records = self
            .records
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        records.iter().rev().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(n: usize) -> InferenceRecord {
        InferenceRecord {
            id: n.to_string(),
            kind: "asr.transcribe".into(),
            model: "m".into(),
            started_at_ms: n as u64,
            ended_at_ms: n as u64,
            total_ms: 0,
            stages_ms: BTreeMap::new(),
            ok: true,
        }
    }

    #[test]
    fn keeps_the_latest_newest_first() {
        let recent = Recent::default();
        for n in 0..KEPT + 5 {
            recent.push(record(n));
        }
        let list = recent.list();
        assert_eq!(list.len(), KEPT);
        assert_eq!(list[0].id, (KEPT + 4).to_string());
        assert_eq!(list.last().unwrap().id, "5");
    }
}
