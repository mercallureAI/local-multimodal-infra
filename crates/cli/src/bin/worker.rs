use local_config::WorkerConfig;
use local_error::{InfraError, Result};
use local_model_store::SqliteModelStore;
use local_registry::ModelRegistry;
use local_runtime::RuntimeManagerConfig;
use local_worker::WorkerState;
use std::{net::SocketAddr, path::PathBuf, time::Duration};

#[derive(Debug, Default)]
struct CliArgs {
    config_path: Option<String>,
    workdir: Option<PathBuf>,
    model_dir: Option<PathBuf>,
    registration_token: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    local_telemetry::init("worker");
    let args = parse_args();
    let config_path = args
        .config_path
        .unwrap_or_else(|| "configs/worker.yaml".to_string());
    let mut config = WorkerConfig::load(&config_path)?;
    if let Some(workdir) = args.workdir {
        config.workdir = workdir;
    }
    if let Some(model_dir) = args.model_dir {
        config.model_dir = Some(model_dir);
    }
    if let Ok(token) = std::env::var("LOCAL_WORKER_REGISTRATION_TOKEN") {
        config.registration_token = Some(token);
    }
    if let Some(token) = args.registration_token {
        config.registration_token = Some(token);
    }
    let layout = config.layout();
    let store = SqliteModelStore::new(&layout.database_path, &layout.model_dir)?;
    store.seed_models(local_registry::default_catalog(&layout.model_dir))?;
    store.seed_models(local_registry::load_yaml_specs(&layout.models_conf_dir)?)?;
    let registry = ModelRegistry::from_models(store.list_models()?);
    let runtime_config = RuntimeManagerConfig {
        cache_idle_ttl: Duration::from_secs(config.runtime.cache_idle_ttl_sec),
        model_idle_ttl: Duration::from_secs(config.runtime.idle_ttl_sec),
        min_residency: Duration::from_secs(config.runtime.min_residency_sec),
        memory_pressure_threshold: config.runtime.memory_pressure_threshold,
    };
    let state = WorkerState::from_registry(
        config.node_id.clone(),
        config.base_url.clone(),
        config.registration_token.clone(),
        registry,
        runtime_config,
    )
    .await;
    if let Err(err) = state.register_with_controller(&config.controller_url).await {
        tracing::warn!(error = %err, "initial worker registration failed; heartbeat loop will keep trying after controller is available");
    }
    state.clone().spawn_heartbeat(
        config.controller_url.clone(),
        Duration::from_secs(config.heartbeat_interval_sec),
    );
    let addr: SocketAddr = config.bind.parse().map_err(|e| {
        InfraError::Config(format!("invalid worker bind address {}: {e}", config.bind))
    })?;
    tracing::info!(%addr, database = %layout.database_path.display(), model_dir = %layout.model_dir.display(), "starting worker");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| InfraError::io(None, e))?;
    axum::serve(listener, state.app())
        .await
        .map_err(|e| InfraError::Backend(format!("worker server error: {e}")))
}

fn parse_args() -> CliArgs {
    let mut parsed = CliArgs::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--workdir" => parsed.workdir = args.next().map(PathBuf::from),
            "--model-dir" => parsed.model_dir = args.next().map(PathBuf::from),
            "--registration-token" => parsed.registration_token = args.next(),
            value if value.starts_with("--workdir=") => {
                parsed.workdir = Some(PathBuf::from(value.trim_start_matches("--workdir=")))
            }
            value if value.starts_with("--model-dir=") => {
                parsed.model_dir = Some(PathBuf::from(value.trim_start_matches("--model-dir=")))
            }
            value if value.starts_with("--registration-token=") => {
                parsed.registration_token = Some(
                    value
                        .trim_start_matches("--registration-token=")
                        .to_string(),
                )
            }
            value if value.starts_with('-') => {
                tracing::warn!(arg = value, "unknown CLI flag ignored")
            }
            value => {
                if parsed.config_path.is_none() {
                    parsed.config_path = Some(value.to_string());
                }
            }
        }
    }
    parsed
}
