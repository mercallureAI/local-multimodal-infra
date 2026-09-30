//! `/internal/realtime`: realtime voice conversations (see
//! `local_voice_cascade::protocol`), in workers built with `realtime`.

use crate::{authorized, unauthorized, WorkerState};
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Query, State,
    },
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    Json,
};
use futures_util::{SinkExt, StreamExt};
use local_core::AdapterKind;
use local_runtime::RuntimeManager;
use local_voice_cascade::{
    protocol::{ClientEvent, ServerEvent},
    CascadeModels, Inbound, Outbound,
};
use serde_json::json;
use std::{sync::Arc, time::Duration};
use tokio::sync::mpsc;

#[derive(Debug, serde::Deserialize)]
pub(crate) struct RealtimeQuery {
    #[serde(default)]
    model: Option<String>,
}

/// A realtime voice conversation (see `local_voice_cascade::protocol`) on a
/// `voice_cascade` model, `voice-cascade` unless `?model=` names another.
pub(crate) async fn realtime(
    State(state): State<WorkerState>,
    headers: HeaderMap,
    Query(query): Query<RealtimeQuery>,
    upgrade: WebSocketUpgrade,
) -> axum::response::Response {
    if !authorized(&state, &headers).await {
        return unauthorized();
    }
    let model_id = query.model.unwrap_or_else(|| "voice-cascade".to_string());
    let spec = state
        .specs
        .get(&model_id)
        .filter(|spec| spec.enabled && spec.adapter == AdapterKind::VoiceCascade);
    let Some(spec) = spec else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("no enabled voice cascade `{model_id}`") })),
        )
            .into_response();
    };
    let models = match CascadeModels::from_spec(spec, &state.data_dir) {
        Ok(models) => models,
        Err(err) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({ "error": err.to_string() })),
            )
                .into_response()
        }
    };
    let runtime = state.runtime.clone();
    upgrade.on_upgrade(move |socket| serve_realtime(runtime, models, socket))
}

async fn serve_realtime(runtime: Arc<RuntimeManager>, models: CascadeModels, socket: WebSocket) {
    let (mut sink, mut stream) = socket.split();
    let (inbound, inbound_rx) = mpsc::channel(256);
    let (out, mut out_rx) = mpsc::unbounded_channel::<Outbound>();
    let reader_out = out.clone();
    let reader = tokio::spawn(async move {
        while let Some(Ok(message)) = stream.next().await {
            let inbound_message = match message {
                Message::Text(text) => match serde_json::from_str::<ClientEvent>(&text) {
                    Ok(event) => Inbound::Event(event),
                    Err(err) => {
                        let _ = reader_out.send(Outbound::Event(ServerEvent::Error {
                            message: format!("bad event: {err}"),
                        }));
                        continue;
                    }
                },
                Message::Binary(bytes) => Inbound::Audio(bytes),
                Message::Close(_) => break,
                _ => continue,
            };
            if inbound.send(inbound_message).await.is_err() {
                break;
            }
        }
    });
    let writer = tokio::spawn(async move {
        while let Some(outbound) = out_rx.recv().await {
            let message = match outbound {
                Outbound::Event(event) => match serde_json::to_string(&event) {
                    Ok(text) => Message::Text(text),
                    Err(_) => continue,
                },
                Outbound::Audio(bytes) => Message::Binary(bytes),
            };
            if sink.send(message).await.is_err() {
                return;
            }
        }
        let _ = sink.close().await;
    });
    if let Err(err) = local_voice_cascade::run(runtime, models, inbound_rx, out.clone()).await {
        tracing::warn!(error = %err, "realtime voice session ended with an error");
        let _ = out.send(Outbound::Event(ServerEvent::Error {
            message: err.to_string(),
        }));
    }
    drop(out);
    reader.abort();
    // Replies still finishing hold the sender a moment; the socket closes
    // once the last event (an error, if any) is sent.
    let abort = writer.abort_handle();
    if tokio::time::timeout(Duration::from_secs(2), writer)
        .await
        .is_err()
    {
        abort.abort();
        tracing::debug!("realtime writer did not finish; socket dropped");
    }
}
