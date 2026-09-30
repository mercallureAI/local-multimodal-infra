//! WeText TN configured as IndexTTS `TextNormalizer.load()` does on
//! Windows/macOS: `Normalizer(lang="zh", operator="tn", remove_erhua=False)`
//! and `Normalizer(lang="en", operator="tn")`.

use local_error::{InfraError, Result};
use std::path::Path;
use wetext::{Language, Normalizer, NormalizerConfig, Operator};

pub struct TextNormalizer {
    zh: Normalizer,
    en: Normalizer,
}

impl std::fmt::Debug for TextNormalizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("TextNormalizer")
    }
}

impl TextNormalizer {
    /// `fst_dir` holds `zh/tn/*.fst` and `en/tn/*.fst`.
    pub fn load(fst_dir: &Path) -> Result<Self> {
        for relative in [
            "zh/tn/tagger.fst",
            "zh/tn/verbalizer.fst",
            "en/tn/tagger.fst",
            "en/tn/verbalizer.fst",
        ] {
            if !fst_dir.join(relative).is_file() {
                return Err(InfraError::ModelNotConfigured {
                    model_id: "zh-tts-frontend".to_string(),
                    reason: format!(
                        "WeText FST is missing: {}",
                        fst_dir.join(relative).display()
                    ),
                });
            }
        }
        let config = |lang| {
            NormalizerConfig::new()
                .with_lang(lang)
                .with_operator(Operator::Tn)
                .with_fix_contractions(false)
        };
        Ok(Self {
            zh: Normalizer::new(fst_dir, config(Language::Zh)),
            en: Normalizer::new(fst_dir, config(Language::En)),
        })
    }

    pub fn normalize_zh(&mut self, text: &str) -> Result<String> {
        self.zh
            .normalize(text)
            .map_err(|e| InfraError::Adapter(format!("WeText zh TN: {e}")))
    }

    pub fn normalize_en(&mut self, text: &str) -> Result<String> {
        self.en
            .normalize(text)
            .map_err(|e| InfraError::Adapter(format!("WeText en TN: {e}")))
    }
}
