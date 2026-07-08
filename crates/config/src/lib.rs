use local_core::StorageLayout;
use local_error::{InfraError, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControllerConfig {
    #[serde(default = "default_controller_bind")]
    pub bind: String,
    #[serde(default = "default_workdir")]
    pub workdir: PathBuf,
    #[serde(default)]
    pub data_dir: Option<PathBuf>,
    #[serde(default)]
    pub database_path: Option<PathBuf>,
    #[serde(default)]
    pub model_dir: Option<PathBuf>,
    #[serde(default = "default_models_conf_dir")]
    pub models_conf_dir: PathBuf,
    #[serde(default)]
    pub admin_token: Option<String>,
    #[serde(default)]
    pub worker_registration_token: Option<String>,
    #[serde(default)]
    pub public_base_url: Option<String>,
    #[serde(default)]
    pub file_upload_base_url: Option<String>,
    #[serde(default)]
    pub upload_signing_secret: Option<String>,
}

impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            bind: default_controller_bind(),
            workdir: default_workdir(),
            data_dir: None,
            database_path: None,
            model_dir: None,
            models_conf_dir: default_models_conf_dir(),
            admin_token: None,
            worker_registration_token: None,
            public_base_url: None,
            file_upload_base_url: None,
            upload_signing_secret: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerConfig {
    #[serde(default = "default_node_id")]
    pub node_id: String,
    #[serde(default = "default_worker_bind")]
    pub bind: String,
    #[serde(default = "default_worker_base_url")]
    pub base_url: String,
    #[serde(default = "default_controller_url")]
    pub controller_url: String,
    #[serde(default = "default_workdir")]
    pub workdir: PathBuf,
    #[serde(default)]
    pub data_dir: Option<PathBuf>,
    #[serde(default)]
    pub database_path: Option<PathBuf>,
    #[serde(default)]
    pub model_dir: Option<PathBuf>,
    #[serde(default = "default_models_conf_dir")]
    pub models_conf_dir: PathBuf,
    #[serde(default = "default_heartbeat_secs")]
    pub heartbeat_interval_sec: u64,
    #[serde(default)]
    pub runtime: RuntimeManagerConfig,
    #[serde(default)]
    pub registration_token: Option<String>,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            node_id: default_node_id(),
            bind: default_worker_bind(),
            base_url: default_worker_base_url(),
            controller_url: default_controller_url(),
            workdir: default_workdir(),
            data_dir: None,
            database_path: None,
            model_dir: None,
            models_conf_dir: default_models_conf_dir(),
            heartbeat_interval_sec: default_heartbeat_secs(),
            runtime: RuntimeManagerConfig::default(),
            registration_token: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeManagerConfig {
    #[serde(default = "default_idle_ttl")]
    pub idle_ttl_sec: u64,
    #[serde(default = "default_min_residency")]
    pub min_residency_sec: u64,
    #[serde(default = "default_memory_pressure")]
    pub memory_pressure_threshold: f32,
}

impl Default for RuntimeManagerConfig {
    fn default() -> Self {
        Self {
            idle_ttl_sec: default_idle_ttl(),
            min_residency_sec: default_min_residency(),
            memory_pressure_threshold: default_memory_pressure(),
        }
    }
}

pub fn load_yaml<T>(path: impl AsRef<Path>) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    let path = path.as_ref();
    let bytes = fs::read(path).map_err(|e| InfraError::io(Some(path.to_path_buf()), e))?;
    Ok(serde_yaml::from_slice(&bytes)?)
}

impl ControllerConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        load_yaml(path)
    }

    pub fn layout(&self) -> StorageLayout {
        StorageLayout::new(
            self.workdir.clone(),
            self.data_dir.clone(),
            self.database_path.clone(),
            self.model_dir.clone(),
            self.models_conf_dir.clone(),
        )
    }
}

impl WorkerConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        load_yaml(path)
    }

    pub fn layout(&self) -> StorageLayout {
        StorageLayout::new(
            self.workdir.clone(),
            self.data_dir.clone(),
            self.database_path.clone(),
            self.model_dir.clone(),
            self.models_conf_dir.clone(),
        )
    }
}

fn default_controller_bind() -> String {
    "127.0.0.1:17890".to_string()
}
fn default_worker_bind() -> String {
    "127.0.0.1:17891".to_string()
}
fn default_worker_base_url() -> String {
    "http://127.0.0.1:17891".to_string()
}
fn default_controller_url() -> String {
    "http://127.0.0.1:17890".to_string()
}
fn default_models_conf_dir() -> PathBuf {
    PathBuf::from("configs/models.d")
}
fn default_workdir() -> PathBuf {
    PathBuf::from("workdir")
}
fn default_node_id() -> String {
    "local-worker".to_string()
}
fn default_heartbeat_secs() -> u64 {
    5
}
fn default_idle_ttl() -> u64 {
    300
}
fn default_min_residency() -> u64 {
    60
}
fn default_memory_pressure() -> f32 {
    0.85
}
