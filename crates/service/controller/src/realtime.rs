//! `/v1/realtime`: the realtime voice WebSocket, forwarded to a worker
//! serving the voice cascade (the conversation runs there, beside its
//! models; see `local_voice_cascade::protocol`).

use super::{ControllerState, WORKER_TOKEN_HEADER};
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Query, State,
    },
    http::{header::ORIGIN, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::json;
use tokio_tungstenite::tungstenite::{self, client::IntoClientRequest};

#[derive(Debug, Deserialize)]
pub(super) struct RealtimeQuery {
    #[serde(default)]
    model: Option<String>,
}

pub(super) async fn realtime(
    State(state): State<ControllerState>,
    headers: HeaderMap,
    Query(query): Query<RealtimeQuery>,
    upgrade: WebSocketUpgrade,
) -> Response {
    let model = query.model.unwrap_or_else(|| "voice-cascade".to_string());
    let error = |status: StatusCode, message: String| {
        (
            status,
            Json(json!({ "error": { "message": message, "type": "local_inference_error" } })),
        )
            .into_response()
    };
    // WebSockets skip CORS: without inference tokens, a web page could open
    // this socket to a local server. Voice clients are not browsers.
    if state.mcp_infer_tokens.is_empty() && headers.contains_key(ORIGIN) {
        return error(
            StatusCode::FORBIDDEN,
            "browser origins may not open realtime sessions".to_string(),
        );
    }
    if model.is_empty()
        || !model
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    {
        return error(StatusCode::BAD_REQUEST, format!("bad model id {model:?}"));
    }
    let (worker_url, worker_token) = match state
        .select_worker(Some(&model), local_core::TaskKind::VoiceRealtime)
        .await
    {
        Ok(worker) => worker,
        Err(err) => return error(StatusCode::SERVICE_UNAVAILABLE, err.to_string()),
    };
    let url = format!(
        "{}/internal/realtime?model={model}",
        worker_url.replacen("http", "ws", 1)
    );
    let mut request = match url.into_client_request() {
        Ok(request) => request,
        Err(err) => return error(StatusCode::BAD_GATEWAY, format!("worker URL: {err}")),
    };
    match worker_token.parse() {
        Ok(value) => {
            request.headers_mut().insert(WORKER_TOKEN_HEADER, value);
        }
        Err(_) => return error(StatusCode::BAD_GATEWAY, "bad worker token".to_string()),
    }
    // Connect first: a worker refusing the model answers here, as HTTP.
    let upstream = match tokio_tungstenite::connect_async(request).await {
        Ok((upstream, _)) => upstream,
        Err(tungstenite::Error::Http(response)) => {
            let body = response
                .body()
                .as_deref()
                .map(|body| String::from_utf8_lossy(body).to_string())
                .unwrap_or_default();
            return error(
                StatusCode::from_u16(response.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
                format!("worker refused the realtime session: {body}"),
            );
        }
        Err(err) => return error(StatusCode::BAD_GATEWAY, format!("connect worker: {err}")),
    };
    upgrade.on_upgrade(move |client| forward(client, upstream))
}

/// Copies messages both ways until either side closes.
async fn forward(
    client: WebSocket,
    upstream: tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
) {
    let (mut client_tx, mut client_rx) = client.split();
    let (mut upstream_tx, mut upstream_rx) = upstream.split();
    let to_worker = async {
        while let Some(Ok(message)) = client_rx.next().await {
            let message = match message {
                Message::Text(text) => tungstenite::Message::Text(text),
                Message::Binary(bytes) => tungstenite::Message::Binary(bytes),
                Message::Ping(bytes) => tungstenite::Message::Ping(bytes),
                Message::Pong(bytes) => tungstenite::Message::Pong(bytes),
                Message::Close(_) => break,
            };
            if upstream_tx.send(message).await.is_err() {
                break;
            }
        }
        let _ = upstream_tx.close().await;
    };
    let to_client = async {
        while let Some(Ok(message)) = upstream_rx.next().await {
            let message = match message {
                tungstenite::Message::Text(text) => Message::Text(text),
                tungstenite::Message::Binary(bytes) => Message::Binary(bytes),
                tungstenite::Message::Ping(bytes) => Message::Ping(bytes),
                tungstenite::Message::Pong(bytes) => Message::Pong(bytes),
                tungstenite::Message::Close(_) => break,
                tungstenite::Message::Frame(_) => continue,
            };
            if client_tx.send(message).await.is_err() {
                break;
            }
        }
        let _ = client_tx.close().await;
    };
    tokio::select! {
        _ = to_worker => {}
        _ = to_client => {}
    }
}
