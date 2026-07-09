use super::*;

const EXPLICIT_TEXT_TOKEN_ID_KEYS: [&str; 3] = [
    "text_token_ids",
    "pretokenized_text_ids",
    "indextts_text_token_ids",
];

#[derive(Debug, Clone)]
pub struct SentencePieceTokenizer {
    path: PathBuf,
    processor: SentencePieceProcessor,
}

impl SentencePieceTokenizer {
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Err(InfraError::ModelNotConfigured {
                model_id: "index-tts".to_string(),
                reason: format!("bpe.model is missing: {}", path.display()),
            });
        }
        let processor = SentencePieceProcessor::open(path).map_err(|e| {
            InfraError::Adapter(format!(
                "load IndexTTS SentencePiece bpe.model {}: {e}",
                path.display()
            ))
        })?;
        Ok(Self {
            path: path.to_path_buf(),
            processor,
        })
    }

    pub fn encode_pieces(&self, text: &str) -> Result<Vec<String>> {
        self.processor.encode(text).map_err(|e| {
            InfraError::Adapter(format!(
                "IndexTTS SentencePiece encode pieces failed for {}: {e}",
                self.path.display()
            ))
        })
    }

    pub fn decode_ids(&self, ids: &[i32], do_lower_case: bool) -> Result<String> {
        let ids = ids
            .iter()
            .map(|id| {
                usize::try_from(*id).map_err(|_| {
                    InfraError::Adapter(format!(
                        "IndexTTS SentencePiece id {id} from {} is negative",
                        self.path.display()
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let decoded = self.processor.decode_ids(&ids).map_err(|e| {
            InfraError::Adapter(format!(
                "IndexTTS SentencePiece decode failed for {}: {e}",
                self.path.display()
            ))
        })?;
        Ok(de_tokenized_by_cjk_char(&decoded, do_lower_case))
    }
}

pub trait IndexTextTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<i32>>;
}

impl IndexTextTokenizer for SentencePieceTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<i32>> {
        self.processor
            .encode_to_ids(text)
            .map_err(|e| {
                InfraError::Adapter(format!(
                    "IndexTTS SentencePiece encode failed for {}: {e}",
                    self.path.display()
                ))
            })?
            .into_iter()
            .map(|id| {
                i32::try_from(id).map_err(|_| {
                    InfraError::Adapter(format!(
                        "IndexTTS SentencePiece id {id} from {} does not fit i32",
                        self.path.display()
                    ))
                })
            })
            .collect()
    }
}

pub fn prepare_text_ids(tokenizer: &impl IndexTextTokenizer, text: &str) -> Result<Vec<i32>> {
    prepare_text_ids_with_mode(tokenizer, text, IndexTtsTextFrontendMode::from_env())
}

pub fn prepare_text_ids_with_mode(
    tokenizer: &impl IndexTextTokenizer,
    text: &str,
    mode: IndexTtsTextFrontendMode,
) -> Result<Vec<i32>> {
    let normalized = preprocess_text_for_index_tts_with_mode(text, mode);
    ensure_index_tts_text_has_speakable_content(&normalized)?;
    tokenizer.encode(&normalized)
}

pub fn ensure_index_tts_text_has_speakable_content(text: &str) -> Result<()> {
    if index_tts_text_has_speakable_content(text) {
        Ok(())
    } else {
        Err(InfraError::BadRequest(
            "IndexTTS text must contain speakable content".to_string(),
        ))
    }
}

pub fn index_tts_text_has_speakable_content(text: &str) -> bool {
    text.chars().any(is_index_tts_speakable_char)
}

fn is_index_tts_speakable_char(ch: char) -> bool {
    ch.is_alphanumeric() || matches!(ch, 'ü' | 'Ü')
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexTtsTextFrontendDump {
    pub input: String,
    pub normalized: String,
    pub tokenized: String,
    pub tokens: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sentencepiece_tokens: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_ids: Option<Vec<i32>>,
}

pub fn dump_text_frontend(
    text: &str,
    tokenizer: Option<&SentencePieceTokenizer>,
) -> Result<IndexTtsTextFrontendDump> {
    let normalized = normalize_text(text);
    let tokenized = tokenize_by_cjk_char(&normalized, true);
    let tokens = tokenized.split_whitespace().map(str::to_string).collect();
    let (sentencepiece_tokens, token_ids) = if let Some(tokenizer) = tokenizer {
        (
            Some(tokenizer.encode_pieces(&tokenized)?),
            Some(tokenizer.encode(&tokenized)?),
        )
    } else {
        (None, None)
    };
    Ok(IndexTtsTextFrontendDump {
        input: text.to_string(),
        normalized,
        tokenized,
        tokens,
        sentencepiece_tokens,
        token_ids,
    })
}

pub fn explicit_text_token_ids_from_params(
    params: &BTreeMap<String, Value>,
) -> Result<Option<Vec<i32>>> {
    for key in EXPLICIT_TEXT_TOKEN_ID_KEYS {
        if let Some(value) = params.get(key) {
            return parse_explicit_text_token_ids(key, value).map(Some);
        }
    }
    Ok(None)
}

fn parse_explicit_text_token_ids(key: &str, value: &Value) -> Result<Vec<i32>> {
    let Some(items) = value.as_array() else {
        return Err(InfraError::BadRequest(format!(
            "IndexTTS {key} must be a non-empty JSON array of integer token ids"
        )));
    };
    if items.is_empty() {
        return Err(InfraError::BadRequest(format!(
            "IndexTTS {key} must not be empty"
        )));
    }
    items
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let Some(id) = value.as_i64() else {
                return Err(InfraError::BadRequest(format!(
                    "IndexTTS {key}[{index}] must be an integer token id"
                )));
            };
            if !(0..=MAX_TEXT_TOKEN_ID).contains(&id) {
                return Err(InfraError::BadRequest(format!(
                    "IndexTTS {key}[{index}] token id {id} is outside the accepted range 0..={MAX_TEXT_TOKEN_ID}"
                )));
            }
            i32::try_from(id).map_err(|_| {
                InfraError::BadRequest(format!(
                    "IndexTTS {key}[{index}] token id {id} does not fit i32"
                ))
            })
        })
        .collect()
}

pub fn index_tts_wav_filename(id: Uuid) -> String {
    format!("indextts-{id}.wav")
}
