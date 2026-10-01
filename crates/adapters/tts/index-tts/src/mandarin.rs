//! Optional Mandarin frontend (`crates/text/zh-tts-frontend`): WeText text
//! normalization in place of the lightweight Rust rules, and g2pW polyphone
//! readings passed to the model as pinyin.
//!
//! Loaded from `LOCAL_ZH_TTS_FRONTEND_DIR`, else from `zh-tts-frontend` next
//! to the model's artifact directory (`<model_dir>/zh-tts-frontend`, built by
//! `scripts/local/zh_frontend_export.py`). Without it, or with
//! `LOCAL_ZH_TTS_FRONTEND=off`, the adapters keep their built-in frontend.
//!
//! Only characters whose reading in context differs from their dictionary
//! reading are annotated (银行 `HANG2`, 了解 `LIAO3`, 东西 `XI5`): the model
//! reads the rest well on its own. Tones are g2pW's (which already gives
//! 一 its spoken tone in places: 一个 `YI2`) plus PaddleSpeech's neutral-tone
//! words; third-tone sandhi is left to the model.

use crate::frontend::normalize_toned_pinyin_syllable;
use local_backend_ort::{OrtBackend, ProviderSelection};
use local_error::{InfraError, Result};
use local_zh_tts_frontend::{Options, Sandhi, TextNormalizer, ZhFrontend, ASSET_DIR_NAME};
use std::{
    env,
    path::{Path, PathBuf},
    sync::Mutex,
};

/// Word readings a deployment fixes (names, product terms): `word<TAB>pinyin
/// pinyin…` per line, in the asset directory.
pub const USER_PHRASES_FILE: &str = "user_phrases.tsv";

/// How a polyphone reading is written into the model's text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PinyinAnnotation {
    /// IndexTTS 1.5 / 2: the character is replaced by its pinyin (`银 HANG2`).
    Inline,
    /// IndexTTS 2.5: `<行|HANG2>`.
    Tagged,
}

pub struct MandarinFrontend {
    dir: PathBuf,
    normalizer: Mutex<TextNormalizer>,
    g2p: Mutex<ZhFrontend>,
}

impl std::fmt::Debug for MandarinFrontend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MandarinFrontend")
            .field("dir", &self.dir)
            .finish()
    }
}

impl MandarinFrontend {
    /// The asset directory for a model whose artifacts are in `artifact_root`,
    /// if the frontend is enabled and present.
    pub fn locate(artifact_root: &Path) -> Option<PathBuf> {
        if env::var("LOCAL_ZH_TTS_FRONTEND").is_ok_and(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "0" | "off" | "false" | "no"
            )
        }) {
            return None;
        }
        if let Some(dir) = env::var_os("LOCAL_ZH_TTS_FRONTEND_DIR") {
            return Some(PathBuf::from(dir));
        }
        let dir = artifact_root.parent()?.join(ASSET_DIR_NAME);
        dir.is_dir().then_some(dir)
    }

    /// `locate` + `load`, logging instead of failing: the built-in frontend
    /// still works without it.
    pub fn load_for(artifact_root: &Path) -> Option<Self> {
        let Some(dir) = Self::locate(artifact_root) else {
            tracing::info!(
                artifact_root = %artifact_root.display(),
                "Mandarin frontend (zh-tts-frontend) not found; using the built-in text rules"
            );
            return None;
        };
        match Self::load(&dir) {
            Ok(frontend) => {
                tracing::info!(dir = %dir.display(), "Mandarin frontend loaded (WeText + g2pW)");
                Some(frontend)
            }
            Err(err) => {
                tracing::warn!(dir = %dir.display(), error = %err, "Mandarin frontend failed to load; using the built-in text rules");
                None
            }
        }
    }

    pub fn load(dir: &Path) -> Result<Self> {
        let normalizer = TextNormalizer::load(&dir.join("wetext"))?;
        // g2pW INT8 is small and fast on CPU (tens of ms a sentence); it
        // stays off the GPU the TTS model uses.
        let backend = OrtBackend::new(ProviderSelection::from_strings(&["cpu".to_string()]));
        let options = Options {
            mainland: true,
            sandhi: Sandhi::NeutralTone,
        };
        let mut g2p = ZhFrontend::load(dir, &backend, options)?;
        let phrases = dir.join(USER_PHRASES_FILE);
        if phrases.is_file() {
            let text = std::fs::read_to_string(&phrases)
                .map_err(|e| InfraError::io(phrases.clone(), e))?;
            let entries: Vec<(String, Vec<String>)> = text
                .lines()
                .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
                .filter_map(|line| {
                    let (word, readings) = line.split_once('\t')?;
                    Some((
                        word.trim().to_string(),
                        readings.split_whitespace().map(str::to_lowercase).collect(),
                    ))
                })
                .collect();
            tracing::info!(count = entries.len(), file = %phrases.display(), "Mandarin frontend user phrases");
            g2p.add_phrases(entries);
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            normalizer: Mutex::new(normalizer),
            g2p: Mutex::new(g2p),
        })
    }

    /// WeText `zh` or `en` normalization, as upstream `TextNormalizer`.
    /// Falls back to the built-in rules if WeText fails.
    pub fn normalize(&self, text: &str, chinese: bool) -> String {
        let mut normalizer = self.normalizer.lock().unwrap_or_else(|e| e.into_inner());
        let result = if chinese {
            normalizer.normalize_zh(text)
        } else {
            normalizer.normalize_en(text)
        };
        result.unwrap_or_else(|err| {
            tracing::warn!(error = %err, "WeText normalization failed; using the built-in rules");
            crate::normalization_rules::lightweight_tn_placeholder_pass(text, chinese)
        })
    }

    /// `text` with each polyphone whose reading in context differs from its
    /// dictionary reading written as pinyin (`style`).
    pub fn annotate(&self, text: &str, style: PinyinAnnotation) -> String {
        let mut g2p = self.g2p.lock().unwrap_or_else(|e| e.into_inner());
        let readings = match g2p.readings(text) {
            Ok(readings) => readings,
            Err(err) => {
                tracing::warn!(error = %err, "g2pW failed; text left unannotated");
                return text.to_string();
            }
        };
        let mut out = String::with_capacity(text.len() * 2);
        for (c, reading) in text.chars().zip(readings) {
            let pinyin = reading
                .filter(|reading| g2p.dictionary_reading(c) != Some(reading.as_str()))
                .and_then(|reading| normalize_toned_pinyin_syllable(&reading));
            match (pinyin, style) {
                (Some(pinyin), PinyinAnnotation::Inline) => {
                    out.push(' ');
                    out.push_str(&pinyin);
                    out.push(' ');
                }
                (Some(pinyin), PinyinAnnotation::Tagged) => {
                    out.push('<');
                    out.push(c);
                    out.push('|');
                    out.push_str(&pinyin);
                    out.push('>');
                }
                (None, _) => out.push(c),
            }
        }
        out
    }
}
