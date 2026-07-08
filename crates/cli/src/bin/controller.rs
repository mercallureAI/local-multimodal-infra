use local_config::ControllerConfig;
use local_controller::{ControllerOptions, ControllerState};
use local_error::{InfraError, Result};
use local_model_store::SqliteModelStore;
use local_registry::ModelRegistry;
use std::{net::SocketAddr, path::PathBuf};

#[derive(Debug, Default)]
struct CliArgs {
    config_path: Option<String>,
    workdir: Option<PathBuf>,
    model_dir: Option<PathBuf>,
    worker_registration_token: Option<String>,
    public_base_url: Option<String>,
    upload_signing_secret: Option<String>,
    admin_token: Option<String>,
    mcp_bind: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    local_telemetry::init("controller");
    let args = parse_args();
    let config_path = args
        .config_path
        .unwrap_or_else(|| "configs/controller.yaml".to_string());
    let mut config = ControllerConfig::load(&config_path)?;
    if let Some(workdir) = args.workdir {
        config.workdir = workdir;
    }
    if let Some(model_dir) = args.model_dir {
        config.model_dir = Some(model_dir);
    }
    if let Ok(token) = std::env::var("LOCAL_WORKER_REGISTRATION_TOKEN") {
        config.worker_registration_token = Some(token);
    }
    if let Ok(url) = std::env::var("LOCAL_PUBLIC_BASE_URL") {
        config.public_base_url = Some(url);
    }
    if let Ok(secret) = std::env::var("LOCAL_UPLOAD_SIGNING_SECRET") {
        config.upload_signing_secret = Some(secret);
    }
    if let Ok(token) = std::env::var("LOCAL_ADMIN_TOKEN") {
        config.admin_token = Some(token);
    }
    let mut mcp_bind = std::env::var("LOCAL_MCP_BIND")
        .ok()
        .unwrap_or_else(|| "127.0.0.1:17892".to_string());
    if let Some(token) = args.worker_registration_token {
        config.worker_registration_token = Some(token);
    }
    if let Some(url) = args.public_base_url {
        config.public_base_url = Some(url);
    }
    if let Some(secret) = args.upload_signing_secret {
        config.upload_signing_secret = Some(secret);
    }
    if let Some(token) = args.admin_token {
        config.admin_token = Some(token);
    }
    if let Some(bind) = args.mcp_bind {
        mcp_bind = bind;
    }
    let layout = config.layout();
    let store = SqliteModelStore::new(&layout.database_path, &layout.model_dir)?;
    store.delete_models(["glm-ocr-onnx-q4f16", "glm-ocr-onnx-fp16"])?;
    store.seed_models(local_registry::default_catalog(&layout.model_dir))?;
    store.seed_models(local_registry::load_yaml_specs(&layout.models_conf_dir)?)?;
    let registry = ModelRegistry::from_models(store.list_models()?);
    let state = ControllerState::with_store_options(
        registry,
        store,
        ControllerOptions {
            worker_registration_token: config.worker_registration_token.clone(),
            public_base_url: config
                .public_base_url
                .clone()
                .or_else(|| config.file_upload_base_url.clone())
                .unwrap_or_else(|| default_public_base_url(&config.bind)),
            data_dir: layout.data_dir.clone(),
            upload_signing_secret: config.upload_signing_secret.clone(),
            admin_token: config.admin_token.clone(),
            ..ControllerOptions::default()
        },
    );
    let addr: SocketAddr = config.bind.parse().map_err(|e| {
        InfraError::Config(format!(
            "invalid controller bind address {}: {e}",
            config.bind
        ))
    })?;
    let mcp_addr: SocketAddr = mcp_bind.parse().map_err(|e| {
        InfraError::Config(format!("invalid standard MCP bind address {mcp_bind}: {e}"))
    })?;
    tracing::info!(%addr, database = %layout.database_path.display(), model_dir = %layout.model_dir.display(), "starting controller HTTP API");
    tracing::info!(%mcp_addr, path = "/mcp", "starting standard MCP server");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| InfraError::io(None, e))?;
    let mcp_listener = tokio::net::TcpListener::bind(mcp_addr)
        .await
        .map_err(|e| InfraError::io(None, e))?;
    let http_state = state.clone();
    let http_server = async move {
        axum::serve(listener, http_state.app())
            .await
            .map_err(|e| InfraError::Backend(format!("controller server error: {e}")))
    };
    let mcp_server = async move {
        axum::serve(mcp_listener, state.standard_mcp_app())
            .await
            .map_err(|e| InfraError::Backend(format!("standard MCP server error: {e}")))
    };
    tokio::try_join!(http_server, mcp_server)?;
    Ok(())
}

fn parse_args() -> CliArgs {
    let mut parsed = CliArgs::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--workdir" => parsed.workdir = args.next().map(PathBuf::from),
            "--model-dir" => parsed.model_dir = args.next().map(PathBuf::from),
            "--worker-registration-token" => parsed.worker_registration_token = args.next(),
            "--public-base-url" => parsed.public_base_url = args.next(),
            "--upload-signing-secret" => parsed.upload_signing_secret = args.next(),
            "--admin-token" => parsed.admin_token = args.next(),
            "--mcp-bind" => parsed.mcp_bind = args.next(),
            value if value.starts_with("--workdir=") => {
                parsed.workdir = Some(PathBuf::from(value.trim_start_matches("--workdir=")))
            }
            value if value.starts_with("--model-dir=") => {
                parsed.model_dir = Some(PathBuf::from(value.trim_start_matches("--model-dir=")))
            }
            value if value.starts_with("--worker-registration-token=") => {
                parsed.worker_registration_token = Some(
                    value
                        .trim_start_matches("--worker-registration-token=")
                        .to_string(),
                )
            }
            value if value.starts_with("--public-base-url=") => {
                parsed.public_base_url =
                    Some(value.trim_start_matches("--public-base-url=").to_string())
            }
            value if value.starts_with("--upload-signing-secret=") => {
                parsed.upload_signing_secret = Some(
                    value
                        .trim_start_matches("--upload-signing-secret=")
                        .to_string(),
                )
            }
            value if value.starts_with("--admin-token=") => {
                parsed.admin_token = Some(value.trim_start_matches("--admin-token=").to_string())
            }
            value if value.starts_with("--mcp-bind=") => {
                parsed.mcp_bind = Some(value.trim_start_matches("--mcp-bind=").to_string())
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

fn default_public_base_url(bind: &str) -> String {
    let port = bind
        .rsplit(':')
        .next()
        .filter(|value| value.chars().all(|ch| ch.is_ascii_digit()))
        .unwrap_or("17890");
    format!("http://127.0.0.1:{port}")
}
