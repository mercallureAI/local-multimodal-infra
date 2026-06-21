use super::*;

#[derive(Debug, Clone, Default)]
pub struct IndexTtsModelConfig {
    pub mel_code_size: Option<usize>,
    pub vocab_size: Option<usize>,
}

impl IndexTtsModelConfig {
    pub fn load(artifacts: &IndexTtsArtifacts, spec: &ModelSpec) -> Result<Self> {
        let mut config = Self::default();
        if let Some(path) = &artifacts.manifest {
            config.merge_value(read_manifest_value(path)?);
        }
        config.merge_map(&spec.metadata);
        Ok(config)
    }

    fn merge_value(&mut self, value: Value) {
        if let Some(object) = value.as_object() {
            self.mel_code_size = self
                .mel_code_size
                .or_else(|| usize_field(object, "mel_code_size"));
            self.vocab_size = self
                .vocab_size
                .or_else(|| usize_field(object, "vocab_size"));
        }
    }

    fn merge_map(&mut self, map: &BTreeMap<String, Value>) {
        self.mel_code_size = self
            .mel_code_size
            .or_else(|| map.get("mel_code_size").and_then(value_as_usize));
        self.vocab_size = self
            .vocab_size
            .or_else(|| map.get("vocab_size").and_then(value_as_usize));
    }

    pub(crate) fn configured_mel_code_size(&self) -> Option<usize> {
        self.mel_code_size.or(self.vocab_size)
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

fn usize_field(object: &serde_json::Map<String, Value>, name: &str) -> Option<usize> {
    object.get(name).and_then(value_as_usize)
}

fn value_as_usize(value: &Value) -> Option<usize> {
    value
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .or_else(|| value.as_str().and_then(|value| value.parse::<usize>().ok()))
}
