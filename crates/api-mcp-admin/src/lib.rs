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
        .route("/rpc/admin", post(handle_json_rpc))
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

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{to_bytes, Body},
        http::{Request, StatusCode},
    };
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use tower::ServiceExt;

    #[derive(Default)]
    struct RecordingAdminApi {
        list_models_calls: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl AdminApi for RecordingAdminApi {
        async fn list_models(&self) -> Result<Vec<ModelSpec>> {
            self.list_models_calls.fetch_add(1, Ordering::SeqCst);
            Ok(Vec::new())
        }

        async fn get_model(&self, id: &str) -> Result<ModelSpec> {
            Err(lcoal_error::InfraError::Unsupported(format!(
                "unexpected get_model in test: {id}"
            )))
        }

        async fn upsert_model(&self, _spec: ModelSpec) -> Result<ModelSpec> {
            Err(lcoal_error::InfraError::Unsupported(
                "unexpected upsert_model in test".to_string(),
            ))
        }

        async fn download_model(&self, id: &str) -> Result<Vec<DownloadStatus>> {
            Err(lcoal_error::InfraError::Unsupported(format!(
                "unexpected download_model in test: {id}"
            )))
        }

        async fn enable_model(&self, id: &str) -> Result<ModelSpec> {
            Err(lcoal_error::InfraError::Unsupported(format!(
                "unexpected enable_model in test: {id}"
            )))
        }

        async fn disable_model(&self, id: &str) -> Result<ModelSpec> {
            Err(lcoal_error::InfraError::Unsupported(format!(
                "unexpected disable_model in test: {id}"
            )))
        }

        async fn list_nodes(&self) -> Result<Vec<NodeStatus>> {
            Ok(Vec::new())
        }

        async fn get_cluster_status(&self) -> Result<Value> {
            Ok(json!({ "ok": true }))
        }
    }

    async fn post_list_models(app: Router, uri: &str) -> Value {
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":1,"method":"list_models","params":{}}"#,
                    ))
                    .expect("request"),
            )
            .await
            .expect("response");

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        serde_json::from_slice(&body).expect("json response")
    }

    #[tokio::test]
    async fn canonical_rpc_admin_route_handles_json_rpc() {
        let service = Arc::new(RecordingAdminApi::default());
        let app = router(AdminApiState {
            service: service.clone(),
        });

        let payload = post_list_models(app, "/rpc/admin").await;

        assert!(payload.get("error").is_none(), "{payload:?}");
        assert_eq!(payload.get("result"), Some(&json!([])));
        assert_eq!(service.list_models_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn mcp_admin_route_is_not_registered_and_does_not_call_service() {
        let service = Arc::new(RecordingAdminApi::default());
        let app = router(AdminApiState {
            service: service.clone(),
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/admin")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":1,"method":"list_models","params":{}}"#,
                    ))
                    .expect("request"),
            )
            .await
            .expect("response");

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_eq!(service.list_models_calls.load(Ordering::SeqCst), 0);
    }
}
