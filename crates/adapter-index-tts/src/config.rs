use super::*;

#[derive(Debug, Clone)]
pub struct IndexTtsModelConfig {
    pub mel_code_size: Option<usize>,
    pub vocab_size: Option<usize>,
    pub max_generate_length: usize,
    pub generation_start_token: i32,
    pub generation_stop_token: i32,
    pub max_text_tokens_per_segment: usize,
    pub inter_segment_silence_ms: u32,
    pub max_consecutive_silence_tokens: usize,
}

impl Default for IndexTtsModelConfig {
    fn default() -> Self {
        Self {
            mel_code_size: None,
            vocab_size: None,
            max_generate_length: MAX_GENERATE_LENGTH,
            generation_start_token: START_TOKEN,
            generation_stop_token: STOP_TOKEN,
            max_text_tokens_per_segment: DEFAULT_MAX_TEXT_TOKENS_PER_SEGMENT,
            inter_segment_silence_ms: DEFAULT_INTER_SEGMENT_SILENCE_MS,
            max_consecutive_silence_tokens: DEFAULT_MAX_CONSECUTIVE_SILENCE_TOKENS,
        }
    }
}

impl IndexTtsModelConfig {
    pub fn load(artifacts: &IndexTtsArtifacts, spec: &ModelSpec) -> Result<Self> {
        let mut config = Self::default();
        if let Some(path) = &artifacts.manifest {
            config.merge_value(read_manifest_value(path)?)?;
        }
        // Model metadata is the explicit deployment-level override and is
        // therefore applied after artifact-provided manifest values.
        config.merge_map(&spec.metadata)?;
        config.validate()?;
        Ok(config)
    }

    fn merge_value(&mut self, value: Value) -> Result<()> {
        if let Some(object) = value.as_object() {
            if let Some(value) =
                parse_usize_fields(|name| object.get(name), &["mel_code_size"], "mel_code_size")?
            {
                self.mel_code_size = Some(value);
            }
            if let Some(value) =
                parse_usize_fields(|name| object.get(name), &["vocab_size"], "vocab_size")?
            {
                self.vocab_size = Some(value);
            }
            self.merge_generation_fields(|name| object.get(name))?;
        }
        Ok(())
    }

    fn merge_map(&mut self, map: &BTreeMap<String, Value>) -> Result<()> {
        if let Some(value) =
            parse_usize_fields(|name| map.get(name), &["mel_code_size"], "mel_code_size")?
        {
            // Preserve the established mel/vocabulary inference precedence:
            // artifact metadata wins, while malformed deployment metadata is
            // still rejected above rather than ignored.
            if self.mel_code_size.is_none() {
                self.mel_code_size = Some(value);
            }
        }
        if let Some(value) =
            parse_usize_fields(|name| map.get(name), &["vocab_size"], "vocab_size")?
        {
            if self.vocab_size.is_none() {
                self.vocab_size = Some(value);
            }
        }
        self.merge_generation_fields(|name| map.get(name))
    }

    pub(crate) fn configured_mel_code_size(&self) -> Option<usize> {
        self.mel_code_size.or(self.vocab_size)
    }

    fn merge_generation_fields<'a>(
        &mut self,
        get: impl Fn(&str) -> Option<&'a Value> + Copy,
    ) -> Result<()> {
        if let Some(value) =
            parse_usize_fields(get, &["max_generate_length"], "max_generate_length")?
        {
            self.max_generate_length = value;
        }
        if let Some(value) = parse_usize_fields(
            get,
            &["max_text_tokens_per_segment", "max_text_tokens"],
            "max_text_tokens_per_segment/max_text_tokens",
        )? {
            self.max_text_tokens_per_segment = value;
        }
        if let Some(value) = parse_u32_fields(
            get,
            &["inter_segment_silence_ms"],
            "inter_segment_silence_ms",
        )? {
            self.inter_segment_silence_ms = value;
        }
        if let Some(value) = parse_usize_fields(
            get,
            &["max_consecutive_silence_tokens"],
            "max_consecutive_silence_tokens",
        )? {
            self.max_consecutive_silence_tokens = value;
        }
        if let Some(value) = parse_i32_fields(
            get,
            &["generation_start_token", "start_token"],
            "generation_start_token/start_token",
        )? {
            self.generation_start_token = value;
        }
        if let Some(value) = parse_i32_fields(
            get,
            &["generation_stop_token", "stop_token"],
            "generation_stop_token/stop_token",
        )? {
            self.generation_stop_token = value;
        }
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        if self.max_generate_length < 2 || self.max_generate_length > 100_000 {
            return Err(InfraError::BadRequest(format!(
                "IndexTTS max_generate_length must be in 2..=100000, got {}",
                self.max_generate_length
            )));
        }
        if self.max_text_tokens_per_segment == 0 || self.max_text_tokens_per_segment > 4096 {
            return Err(InfraError::BadRequest(format!(
                "IndexTTS max_text_tokens_per_segment must be in 1..=4096, got {}",
                self.max_text_tokens_per_segment
            )));
        }
        if self.inter_segment_silence_ms > 10_000 {
            return Err(InfraError::BadRequest(format!(
                "IndexTTS inter_segment_silence_ms must be <=10000, got {}",
                self.inter_segment_silence_ms
            )));
        }
        // Zero disables the safety mechanism and very large values recreate the
        // decode-budget failure it is intended to prevent.
        if !(1..=120).contains(&self.max_consecutive_silence_tokens) {
            return Err(InfraError::BadRequest(format!(
                "IndexTTS max_consecutive_silence_tokens must be in 1..=120, got {}",
                self.max_consecutive_silence_tokens
            )));
        }
        Ok(())
    }
}

fn read_manifest_value(path: &Path) -> Result<Value> {
    let bytes = fs::read(path).map_err(|e| InfraError::io(Some(path.to_path_buf()), e))?;
    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
    {
        Ok(serde_json::from_slice(&bytes)?)
    } else {
        Ok(serde_yaml::from_slice(&bytes)?)
    }
}

fn parse_usize_fields<'a>(
    get: impl Fn(&str) -> Option<&'a Value>,
    names: &[&str],
    label: &str,
) -> Result<Option<usize>> {
    parse_fields(get, names, label, value_as_usize)
}

fn parse_u32_fields<'a>(
    get: impl Fn(&str) -> Option<&'a Value>,
    names: &[&str],
    label: &str,
) -> Result<Option<u32>> {
    parse_fields(get, names, label, |value| {
        value_as_u64(value).and_then(|value| u32::try_from(value).ok())
    })
}

fn parse_i32_fields<'a>(
    get: impl Fn(&str) -> Option<&'a Value>,
    names: &[&str],
    label: &str,
) -> Result<Option<i32>> {
    parse_fields(get, names, label, value_as_i32)
}

fn parse_fields<'a, T: Copy>(
    get: impl Fn(&str) -> Option<&'a Value>,
    names: &[&str],
    label: &str,
    parse: impl Fn(&Value) -> Option<T>,
) -> Result<Option<T>> {
    let mut selected = None;
    for name in names {
        let Some(raw) = get(name) else {
            continue;
        };
        let value = parse(raw).ok_or_else(|| {
            InfraError::BadRequest(format!(
                "IndexTTS {name} ({label}) must be a representable integer or integer string, got {raw}"
            ))
        })?;
        if selected.is_none() {
            selected = Some(value);
        }
    }
    Ok(selected)
}

fn value_as_u64(value: &Value) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| value.as_str().and_then(|value| value.parse::<u64>().ok()))
}

fn value_as_usize(value: &Value) -> Option<usize> {
    value_as_u64(value).and_then(|value| usize::try_from(value).ok())
}

fn value_as_i32(value: &Value) -> Option<i32> {
    value
        .as_i64()
        .and_then(|value| i32::try_from(value).ok())
        .or_else(|| value.as_str().and_then(|value| value.parse::<i32>().ok()))
}
