//! IndexTTS-2.5 adapter.
//!
//! Provenance:
//! - https://github.com/index-tts/index-tts (`indextts/infer_v2_5.py`)
//! - https://github.com/DakeQQ/Text-to-Speech-TTS-ONNX/tree/main/Index_TTS/v2
//! - Rewritten for this project's `backend-ort` runtime; it neither depends on nor
//!   vendors the upstream projects. IndexTTS 1.5 lives in `local-adapter-index-tts`.

mod artifacts;
pub mod audio;
pub mod frontend;
mod params;
mod pipeline;
mod tokenizer;

pub use artifacts::{IndexTts2Artifacts, PackageManifest, PackageRuntime};
pub use params::{detect_language, SynthesisParams, EMOTION_NAMES};
pub use pipeline::{IndexTts2Adapter, IndexTts2ProviderReport};
pub use tokenizer::{MultilingualTokenizer, TOKENIZER_MANIFEST_FILE, TOKENIZER_MANIFEST_SCHEMA};
