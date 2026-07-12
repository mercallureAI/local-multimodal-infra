use super::*;

#[derive(Debug, Clone)]
pub struct IndexTtsModelConfig {
    pub mel_code_size: Option<usize>,
    pub vocab_size: Option<usize>,
    pub max_generate_length: usize,
    pub split_contract_version: usize,
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
            split_contract_version: 0,
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
        validate_deployment_config(&artifacts.root)?;
        if let Some(path) = &artifacts.manifest {
            let primary = read_manifest_value(path)?;
            for sibling in ["manifest.yaml", "manifest.yml", "manifest.json"]
                .iter()
                .map(|name| artifacts.root.join(name))
                .filter(|candidate| candidate.exists() && candidate != path)
            {
                let other = read_manifest_value(&sibling)?;
                if other != primary {
                    return Err(InfraError::BadRequest(format!(
                        "IndexTTS manifests are not semantically equal: {} and {}",
                        path.display(),
                        sibling.display()
                    )));
                }
            }
            validate_v2_manifest(&primary)?;
            config.merge_value(primary)?;
        }
        reject_contract_metadata_overrides(&spec.metadata)?;
        config.merge_map(&spec.metadata)?;
        config.validate()?;
        Ok(config)
    }

    fn merge_value(&mut self, value: Value) -> Result<()> {
        if let Some(object) = value.as_object() {
            if let Some(value) = parse_usize_fields(
                |name| object.get(name),
                &["split_contract_version"],
                "split_contract_version",
            )? {
                self.split_contract_version = value;
            }
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
        self.merge_operational_fields(|name| map.get(name))
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

    fn merge_operational_fields<'a>(
        &mut self,
        get: impl Fn(&str) -> Option<&'a Value> + Copy,
    ) -> Result<()> {
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
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        if self.split_contract_version != 2 {
            return Err(InfraError::ModelNotConfigured {
                model_id: "index-tts".to_string(),
                reason: format!(
                    "legacy or unsupported IndexTTS split_contract_version {}; re-export the original checkpoint with split contract v2",
                    self.split_contract_version
                ),
            });
        }
        if self.max_generate_length != MAX_GENERATE_LENGTH {
            return Err(InfraError::BadRequest(format!(
                "IndexTTS v2 max_generate_length must be exactly {MAX_GENERATE_LENGTH}, got {}",
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

fn validate_deployment_config(root: &Path) -> Result<()> {
    let path = root.join("config.json");
    if !path.exists() {
        return Ok(());
    }
    let canonical_root =
        fs::canonicalize(root).map_err(|e| InfraError::io(Some(root.to_path_buf()), e))?;
    let value: Value = serde_json::from_slice(
        &fs::read(&path).map_err(|e| InfraError::io(Some(path.clone()), e))?,
    )?;
    for (pointer, expected) in [
        ("/schema", "local.index_tts.deployment.v1"),
        ("/model_id", "indextts-1.5-onnx"),
        ("/source/repo_id", "IndexTeam/IndexTTS-1.5"),
        (
            "/source/revision",
            "25851a6036dfd3095bb70fb3c8f49217104672c3",
        ),
        (
            "/source/download_method",
            "huggingface_hub.snapshot_download",
        ),
        ("/official_code/tag", "v1.5.0"),
        (
            "/official_code/commit",
            "9098497272d5803bae46cbaf5154cf2ba48f6866",
        ),
        (
            "/official_code/tree",
            "aa0335ccaba54ac42d6d209dac56bb9a8b2e80a7",
        ),
        ("/contract/cache_mode", "prefill_decode"),
    ] {
        if value.pointer(pointer).and_then(Value::as_str) != Some(expected) {
            return Err(InfraError::ModelNotConfigured {
                model_id: "index-tts".to_string(),
                reason: format!(
                    "IndexTTS deployment config {} `{pointer}` must be `{expected}`",
                    path.display()
                ),
            });
        }
    }
    if value.pointer("/contract/version").and_then(Value::as_u64) != Some(2) {
        return Err(InfraError::ModelNotConfigured {
            model_id: "index-tts".to_string(),
            reason: format!(
                "IndexTTS deployment config {} contract version must be 2",
                path.display()
            ),
        });
    }
    const REQUIRED_WARNING: &str = "The pinned official code contains INDEX_MODEL_LICENSE with non-commercial and other restrictions. Preserve and review it; do not rely only on Hugging Face apache-2.0 metadata.";
    for (pointer, expected) in [
        ("/license/warning", REQUIRED_WARNING),
        (
            "/source_provenance_reference/note",
            "Informational hash of the separately retained source provenance record; that record is intentionally outside this 14-file runtime package.",
        ),
    ] {
        if value.pointer(pointer).and_then(Value::as_str) != Some(expected) {
            return deployment_config_error(&path, format!("`{pointer}` must equal the supported value"));
        }
    }
    for pointer in ["/source_provenance_reference/runtime_verifiable"] {
        if value.pointer(pointer).and_then(Value::as_bool) != Some(false) {
            return deployment_config_error(&path, format!("`{pointer}` must be false"));
        }
    }
    let hash_files = [
        ("/contract/manifest_json_sha256", "manifest.json"),
        ("/contract/manifest_yaml_sha256", "manifest.yaml"),
        ("/contract/index_tts_e_sha256", "IndexTTS_E.onnx"),
        (
            "/contract/index_tts_e_prefill_sha256",
            "IndexTTS_E_Prefill.onnx",
        ),
    ];
    for (pointer, filename) in hash_files {
        let expected = required_sha256(&value, pointer, &path)?;
        let target = canonical_root.join(filename);
        let canonical_target =
            fs::canonicalize(&target).map_err(|e| InfraError::io(Some(target.clone()), e))?;
        if !canonical_target.starts_with(&canonical_root) {
            return deployment_config_error(
                &path,
                format!(
                    "fixed integrity target escapes model root: {}",
                    target.display()
                ),
            );
        }
        let actual = local_files::sha256_file(&canonical_target)?;
        if actual != expected {
            return deployment_config_error(
                &path,
                format!(
                    "integrity mismatch for {}: expected {expected}, got {actual}",
                    canonical_target.display()
                ),
            );
        }
    }
    let provenance = required_sha256(&value, "/source_provenance_reference/sha256", &path)?;
    if provenance != "3bfb39cc326d834be4fda72000e4cc53ebbb6c52e154150f1fdbe7323d1e909c" {
        return deployment_config_error(
            &path,
            "source provenance reference must match the pinned informational hash".to_string(),
        );
    }
    Ok(())
}

fn required_sha256<'a>(value: &'a Value, pointer: &str, path: &Path) -> Result<&'a str> {
    let hash = value
        .pointer(pointer)
        .and_then(Value::as_str)
        .ok_or_else(|| InfraError::ModelNotConfigured {
            model_id: "index-tts".to_string(),
            reason: format!(
                "IndexTTS deployment config {} missing `{pointer}`",
                path.display()
            ),
        })?;
    if hash.len() != 64
        || !hash
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return deployment_config_error(
            path,
            format!("`{pointer}` must be exactly 64 lowercase hexadecimal characters"),
        );
    }
    Ok(hash)
}

fn deployment_config_error<T>(path: &Path, reason: String) -> Result<T> {
    Err(InfraError::ModelNotConfigured {
        model_id: "index-tts".to_string(),
        reason: format!("IndexTTS deployment config {} {reason}", path.display()),
    })
}

fn reject_contract_metadata_overrides(map: &BTreeMap<String, Value>) -> Result<()> {
    for name in [
        "split_contract_version",
        "graph_contract",
        "generation_policy",
        "max_generate_length",
        "mel_code_size",
        "vocab_size",
        "generation_start_token",
        "start_token",
        "generation_stop_token",
        "stop_token",
    ] {
        if map.contains_key(name) {
            return Err(InfraError::BadRequest(format!(
                "IndexTTS deployment metadata cannot override v2 ABI/policy field `{name}`"
            )));
        }
    }
    Ok(())
}

fn validate_v2_manifest(value: &Value) -> Result<()> {
    if value.get("status").and_then(Value::as_str) != Some("ready") {
        return Err(InfraError::ModelNotConfigured {
            model_id: "index-tts".to_string(),
            reason: format!(
                "IndexTTS artifact manifest status must be `ready`, got {}",
                value.get("status").unwrap_or(&Value::Null)
            ),
        });
    }
    let expected: [(&str, Value); 28] = [
        ("/split_contract_version", Value::from(2)),
        ("/graph_contract/cache_mode", Value::from("prefill_decode")),
        ("/graph_contract/cache_layout", Value::from("hf_bhsd")),
        ("/graph_contract/layers", Value::from(24)),
        ("/graph_contract/dtype", Value::from("float32")),
        ("/graph_contract/batch", Value::from(1)),
        ("/graph_contract/heads", Value::from(20)),
        ("/graph_contract/cache_sequence_axis", Value::from(2)),
        ("/graph_contract/head_dim", Value::from(64)),
        ("/graph_contract/initial_cache_length", Value::from(0)),
        ("/graph_contract/attention_mask/rank", Value::from(2)),
        ("/graph_contract/attention_mask/dtype", Value::from("int64")),
        ("/graph_contract/logits/name", Value::from("raw_logits")),
        ("/graph_contract/logits/rank", Value::from(3)),
        (
            "/graph_contract/logits/shape",
            serde_json::json!([1, 1, 8194]),
        ),
        ("/graph_contract/logits/selection", Value::from("runtime")),
        ("/generation_policy/version", Value::from(1)),
        (
            "/generation_policy/algorithm",
            Value::from("seeded_top_k_top_p"),
        ),
        ("/generation_policy/num_beams", Value::from(1)),
        ("/generation_policy/source_num_beams", Value::from(3)),
        ("/generation_policy/top_k", Value::from(30)),
        ("/generation_policy/top_p", Value::from(0.8)),
        ("/generation_policy/temperature", Value::from(1.0)),
        ("/generation_policy/repetition_penalty", Value::from(10.0)),
        (
            "/generation_policy/repetition_scope",
            Value::from("mel_bos_and_full_generated_history"),
        ),
        ("/generation_policy/max_new_mel_tokens", Value::from(600)),
        ("/generation_policy/start_token", Value::from(8192)),
        ("/generation_policy/stop_token", Value::from(8193)),
    ];
    for (path, expected) in expected {
        let actual = value.pointer(path).ok_or_else(|| {
            InfraError::BadRequest(format!("IndexTTS v2 manifest is missing `{path}`"))
        })?;
        if actual != &expected {
            return Err(InfraError::BadRequest(format!(
                "IndexTTS v2 manifest `{path}` must be {expected}, got {actual}"
            )));
        }
    }
    for (path, expected) in [
        ("/generation_policy/default_seed", 0),
        ("/generation_policy/silence_token", 52),
        ("/max_generate_length", 600),
        ("/mel_code_size", 8194),
        ("/vocab_size", 8194),
    ] {
        if value.pointer(path).and_then(Value::as_i64) != Some(expected) {
            return Err(InfraError::BadRequest(format!(
                "IndexTTS v2 manifest `{path}` must be {expected}"
            )));
        }
    }
    Ok(())
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
