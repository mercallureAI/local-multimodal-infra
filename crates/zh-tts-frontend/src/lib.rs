//! Mandarin TTS text front end.
//!
//! - Text normalization: WeText FSTs through `local-wetext` (the files IndexTTS
//!   itself uses), see [`TextNormalizer`].
//! - Readings: PaddleSpeech's g2pW frontend (`zh_frontend.py`,
//!   `tone_sandhi.py`, `polyphonic.yaml`, `g2pw/onnx_api.py`; PaddlePaddle/
//!   PaddleSpeech 6b25a40, Apache-2.0) ported to Rust, see [`ZhFrontend`].
//!
//! Every data file comes from the asset directory written by
//! `scripts/local/zh_frontend_export.py` (`<model_dir>/zh-tts-frontend`).

mod frontend;
mod g2pw;
mod mainland;
mod normalizer;
mod pinyin;
mod sandhi;
mod sandhi_words;

pub use frontend::{Options, Sandhi, ZhFrontend};
pub use mainland::MainlandReadings;
pub use normalizer::TextNormalizer;
pub use pinyin::{is_hans, PinyinDict};

/// Directory name of the asset bundle under the model directory.
pub const ASSET_DIR_NAME: &str = "zh-tts-frontend";
