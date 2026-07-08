//! IndexTTS adapter provenance:
//! - https://huggingface.co/ThreadAbort/IndexTTS-Rust
//! - https://github.com/DakeQQ/Text-to-Speech-TTS-ONNX/blob/main/IndexTTS/Export_IndexTTS.py
//! - https://github.com/DakeQQ/Text-to-Speech-TTS-ONNX/blob/main/IndexTTS/Inference_IndexTTS_ONNX.py
//! - https://github.com/index-tts/index-tts
//! - This crate is rewritten/adapted inside this project for its `backend-ort` runtime and does
//!   not directly depend on or vendor the upstream projects.

use local_backend_ort::{
    OrtBackend, OrtSession, OrtTensorData, OrtTensorInput, OrtTensorOutput, ProviderSelection,
    SessionMetadata, SessionProviderReport, TensorElement, TensorMetadata,
};
use local_core::{FileRef, InferenceOutput, ModelSpec};
use local_error::{InfraError, Result};
use pinyin::ToPinyin;
use sentencepiece_rs::SentencePieceProcessor;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
};
use uuid::Uuid;

pub const TARGET_SAMPLE_RATE: u32 = 24_000;
pub const START_TOKEN: i32 = 8192;
pub const STOP_TOKEN: i32 = 8193;
pub const MAX_GENERATE_LENGTH: usize = 800;
/// IndexTTS E graph applies repeat suppression by multiplying logits by
/// `repeat_penality`, so repeated token columns must be below 1.0.
pub const DEFAULT_REPEAT_PENALTY: f32 = 0.9;
pub const DEFAULT_REPEAT_WINDOW: usize = 16;
pub const DEFAULT_MEL_CODE_SIZE: usize = 8194;
pub const MAX_TEXT_TOKEN_ID: i64 = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexTtsProviderReport {
    pub a: SessionProviderReport,
    pub b: SessionProviderReport,
    pub c: SessionProviderReport,
    pub d: SessionProviderReport,
    pub e: SessionProviderReport,
    pub f: SessionProviderReport,
}

mod artifacts;
pub mod audio;
mod config;
mod frontend;
mod normalization_rules;
mod onnx;
mod pipeline;
mod tokenizer;

pub use artifacts::{IndexTtsArtifacts, IndexTtsPrecision};
pub use config::IndexTtsModelConfig;
pub use frontend::{
    correct_pinyin, de_tokenized_by_cjk_char, normalize_text, preprocess_text_for_index_tts,
    preprocess_text_for_index_tts_with_mode, split_cjk_minimal, split_sentences,
    split_sentences_by_token, tokenize_by_cjk_char, IndexTtsTextFrontendMode,
    INDEXTTS_PUNCTUATION_MARK_TOKENS,
};
pub use onnx::{apply_repeat_penalty_token, concatenate_hidden_states, e_loop_control_lengths};
pub use pipeline::IndexTtsAdapter;
pub use tokenizer::{
    dump_text_frontend, explicit_text_token_ids_from_params, index_tts_wav_filename,
    prepare_text_ids, prepare_text_ids_with_mode, IndexTextTokenizer, IndexTtsTextFrontendDump,
    SentencePieceTokenizer,
};

#[cfg(test)]
pub(crate) use artifacts::MODEL_FILENAMES;
pub(crate) use frontend::*;
pub(crate) use normalization_rules::*;
pub(crate) use onnx::*;
pub(crate) use pipeline::EStep;

#[cfg(test)]
mod tests;
