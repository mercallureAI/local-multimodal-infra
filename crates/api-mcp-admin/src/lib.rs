use async_trait::async_trait;
use axum::{extract::State, routing::post, Json, Router};
use lcoal_core::{AssetListQuery, AssetListResponse, DownloadStatus, ModelSpec, NodeStatus};
use lcoal_error::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;

#[async_trait]
pub trait AdminApi: Send + Sync + 'static {
    async fn list_models(&self) -> Result<Vec<ModelSpec>>;
    async fn get_model(&self, id: &str) -> Result<ModelSpec>;
    async fn upsert_model(&self, spec: ModelSpec) -> Result<ModelSpec>;
    async fn download_model(&self, id: &str) -> Result<Vec<DownloadStatus>>;
    async fn enable_model(&self, id: &str) -> Result<ModelSpec>;
    async fn disable_model(&self, id: &str) -> Result<ModelSpec>;
    async fn list_nodes(&self) -> Result<Vec<NodeStatus>>;
    async fn get_cluster_status(&self) -> Result<Value>;
    async fn list_assets(&self, _query: AssetListQuery) -> Result<AssetListResponse> {
        Err(lcoal_error::InfraError::Unsupported(
            "asset listing is not implemented by this admin service".to_string(),
        ))
    }
}

#[derive(Clone)]
pub struct AdminApiState {
    pub service: Arc<dyn AdminApi>,
}

pub fn router(state: AdminApiState) -> Router {
    Router::new()
        .route("/mcp/admin", post(handle_json_rpc))
        .with_state(state)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    #[serde(default)]
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: &'static str,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
}

async fn handle_json_rpc(
    State(state): State<AdminApiState>,
    Json(req): Json<JsonRpcRequest>,
) -> Json<JsonRpcResponse> {
    let id = req.id.clone();
    let result = match req.method.as_str() {
        "list_models" => state.service.list_models().await.map(|v| json!(v)),
        "get_model" => state
            .service
            .get_model(required_id(&req.params))
            .await
            .map(|v| json!(v)),
        "add_model" | "upsert_model" => match parse_model_spec(&req.params) {
            Ok(spec) => state.service.upsert_model(spec).await.map(|v| json!(v)),
            Err(err) => Err(err),
        },
        "download_model" => state
            .service
            .download_model(required_id(&req.params))
            .await
            .map(|v| json!(v)),
        "enable_model" => state
            .service
            .enable_model(required_id(&req.params))
            .await
            .map(|v| json!(v)),
        "disable_model" => state
            .service
            .disable_model(required_id(&req.params))
            .await
            .map(|v| json!(v)),
        "list_nodes" => state.service.list_nodes().await.map(|v| json!(v)),
        "get_cluster_status" => state.service.get_cluster_status().await,
        "list_assets" | "search_assets" => {
            match serde_json::from_value::<AssetListQuery>(req.params.clone()) {
                Ok(query) => state.service.list_assets(query).await.map(|v| json!(v)),
                Err(err) => Err(lcoal_error::InfraError::from(err)),
            }
        }
        other => Err(lcoal_error::InfraError::BadRequest(format!(
            "unknown admin method `{other}`"
        ))),
    };
    Json(match result {
        Ok(value) => JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(value),
            error: None,
        },
        Err(err) => JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(JsonRpcError {
                code: -32000,
                message: err.to_string(),
            }),
        },
    })
}

fn parse_model_spec(params: &Value) -> Result<ModelSpec> {
    if let Some(spec) = params.get("spec") {
        Ok(serde_json::from_value(spec.clone())?)
    } else {
        Ok(serde_json::from_value(params.clone())?)
    }
}

fn required_id(params: &Value) -> &str {
    params
        .get("id")
        .and_then(Value::as_str)
        .or_else(|| params.get("model_id").and_then(Value::as_str))
        .unwrap_or("")
}
