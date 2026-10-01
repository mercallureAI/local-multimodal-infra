use tracing_subscriber::{fmt, EnvFilter};

pub fn init(service_name: &str) {
    // ORT logs two INFO lines per CUDA graph replay (every decode frame).
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,local=debug,ort=warn"));
    let _ = fmt().with_env_filter(filter).with_target(true).try_init();
    tracing::info!(service = service_name, "telemetry initialized");
}
