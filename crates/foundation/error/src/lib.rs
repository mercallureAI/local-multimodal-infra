use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, InfraError>;

#[derive(Debug, thiserror::Error)]
pub enum InfraError {
    #[error("configuration error: {0}")]
    Config(String),
    #[error("registry error: {0}")]
    Registry(String),
    #[error("model `{model_id}` is not configured: {reason}")]
    ModelNotConfigured { model_id: String, reason: String },
    #[error("unsupported operation: {0}")]
    Unsupported(String),
    #[error("need user confirmation: {0}")]
    NeedUserConfirmation(String),
    #[error("need implementation: {0}")]
    NeedImplementation(String),
    #[error("backend error: {0}")]
    Backend(String),
    #[error("adapter error: {0}")]
    Adapter(String),
    #[error("runtime error: {0}")]
    Runtime(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("bad request: {0}")]
    BadRequest(String),
    #[error("I/O error at {path:?}: {source}")]
    Io {
        path: Option<PathBuf>,
        #[source]
        source: std::io::Error,
    },
    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

impl InfraError {
    pub fn io(path: impl Into<Option<PathBuf>>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}
