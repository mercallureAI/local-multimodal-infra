//! `POST /v1/chat/completions`, with `stream: true` as server-sent events.

use crate::{error_response, OpenAiApiState};
use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
    Json,
};
use local_core::{
    ChatFinishReason, ChatMessage, ChatOptions, ChatTimings, ChatToolCall, ChatUsage,
    InferenceEvent, InferenceInput, InferenceOutput, InferenceTask, TaskKind,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<RequestMessage>,
    #[serde(default)]
    pub tools: Vec<Value>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Extension: top-k sampling.
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Option<Stop>,
    /// Token id (as a string key) -> bias, as in the OpenAI API.
    #[serde(default)]
    pub logit_bias: BTreeMap<String, f32>,
    /// Extension: logit offset for starting a tool call.
    #[serde(default)]
    pub tool_call_bias: Option<f32>,
    /// Extension: tool name -> logit offset when the model names the tool.
    #[serde(default)]
    pub tool_bias: BTreeMap<String, f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Stop {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct RequestMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<Value>,
    #[serde(default)]
    pub tool_calls: Vec<RequestToolCall>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RequestToolCall {
    #[serde(default)]
    pub id: String,
    pub function: RequestFunction,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RequestFunction {
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
}

impl ChatCompletionRequest {
    fn into_task(self) -> Result<(InferenceTask, bool, bool), String> {
        if self.messages.is_empty() {
            return Err("messages must not be empty".to_string());
        }
        if let Some(choice) = &self.tool_choice {
            if !matches!(choice.as_str(), Some("auto") | Some("none")) {
                return Err("only tool_choice \"auto\" and \"none\" are supported".to_string());
            }
        }
        let tools = if matches!(
            self.tool_choice.as_ref().and_then(Value::as_str),
            Some("none")
        ) {
            Vec::new()
        } else {
            self.tools
        };
        let messages = self
            .messages
            .into_iter()
            .map(|message| {
                Ok(ChatMessage {
                    role: message.role,
                    content: message.content.map(content_text).transpose()?,
                    tool_calls: message
                        .tool_calls
                        .into_iter()
                        .map(|call| ChatToolCall {
                            id: call.id,
                            name: call.function.name,
                            arguments: match call.function.arguments {
                                Value::String(text) => text,
                                Value::Null => "{}".to_string(),
                                other => other.to_string(),
                            },
                        })
                        .collect(),
                    tool_call_id: message.tool_call_id,
                    name: message.name,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let logit_bias = self
            .logit_bias
            .into_iter()
            .map(|(token, bias)| {
                token
                    .parse::<u32>()
                    .map(|token| (token, bias))
                    .map_err(|_| format!("logit_bias key `{token}` is not a token id"))
            })
            .collect::<Result<BTreeMap<_, _>, String>>()?;
        let options = ChatOptions {
            max_tokens: self.max_completion_tokens.or(self.max_tokens),
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            seed: self.seed,
            stop: match self.stop {
                None => Vec::new(),
                Some(Stop::One(stop)) => vec![stop],
                Some(Stop::Many(stops)) => stops,
            },
            logit_bias,
            tool_call_bias: self.tool_call_bias,
            tool_bias: self.tool_bias,
        };
        let include_usage = self
            .stream_options
            .is_some_and(|options| options.include_usage);
        let task = InferenceTask::new(
            TaskKind::ChatComplete,
            Some(self.model),
            InferenceInput::ChatComplete {
                messages,
                tools,
                options,
            },
        );
        Ok((task, self.stream, include_usage))
    }
}

/// Text of a message content: a string or an array of `{"type": "text"}`
/// parts.
fn content_text(content: Value) -> Result<String, String> {
    match content {
        Value::String(text) => Ok(text),
        Value::Null => Ok(String::new()),
        Value::Array(parts) => parts
            .into_iter()
            .map(|part| match part.get("type").and_then(Value::as_str) {
                Some("text") => Ok(part
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string()),
                other => Err(format!(
                    "message content part type {other:?} is not supported"
                )),
            })
            .collect(),
        other => Err(format!("message content {other} is not text")),
    }
}

pub(crate) async fn chat_completions(
    State(state): State<OpenAiApiState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let model = req.model.clone();
    let (task, stream, include_usage) = match req.into_task() {
        Ok(parts) => parts,
        Err(message) => return error_response(StatusCode::BAD_REQUEST, message),
    };
    let id = format!("chatcmpl-{}", task.id.simple());
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    if !stream {
        return match state.service.dispatch(task).await {
            Ok(InferenceOutput::ChatCompletion {
                content,
                tool_calls,
                finish_reason,
                usage,
                timings,
            }) => Json(json!({
                "id": id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": message_json(&content, &tool_calls),
                    "finish_reason": finish_reason_str(finish_reason),
                }],
                "usage": usage_json(usage),
                "timings": timings_json(timings),
            }))
            .into_response(),
            Ok(other) => error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("unexpected inference output: {other:?}"),
            ),
            Err(err) => error_response(StatusCode::NOT_IMPLEMENTED, err.to_string()),
        };
    }

    let receiver = match state.service.dispatch_stream(task).await {
        Ok(receiver) => receiver,
        Err(err) => return error_response(StatusCode::NOT_IMPLEMENTED, err.to_string()),
    };
    let chunk = move |delta: Value, finish: Option<&str>| {
        json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        })
    };
    let first = chunk(json!({"role": "assistant", "content": ""}), None);
    let state = StreamState {
        receiver,
        pending: vec![Event::default().data(first.to_string())],
        chunk: Box::new(chunk),
        include_usage,
        done: false,
    };
    let events = futures_util::stream::unfold(state, |mut state| async move {
        loop {
            if !state.pending.is_empty() {
                let event = state.pending.remove(0);
                return Some((Ok::<_, std::convert::Infallible>(event), state));
            }
            if state.done {
                return None;
            }
            match state.receiver.recv().await {
                Some(event) => state.on_event(event),
                None => {
                    state.pending.push(Event::default().data(
                        json!({"error": {"message": "inference ended without a result", "type": "local_inference_error"}}).to_string(),
                    ));
                    state.done = true;
                }
            }
        }
    });
    Sse::new(events).into_response()
}

type ChunkFn = dyn Fn(Value, Option<&str>) -> Value + Send;

struct StreamState {
    receiver: tokio::sync::mpsc::Receiver<InferenceEvent>,
    pending: Vec<Event>,
    chunk: Box<ChunkFn>,
    include_usage: bool,
    done: bool,
}

impl StreamState {
    fn push(&mut self, value: Value) {
        self.pending.push(Event::default().data(value.to_string()));
    }

    fn on_event(&mut self, event: InferenceEvent) {
        match event {
            InferenceEvent::ChatDelta { content } => {
                let chunk = (self.chunk)(json!({"content": content}), None);
                self.push(chunk);
            }
            InferenceEvent::ChatToolCall { index, call } => {
                let chunk = (self.chunk)(
                    json!({"tool_calls": [{
                        "index": index,
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments},
                    }]}),
                    None,
                );
                self.push(chunk);
            }
            InferenceEvent::Output { output } => {
                if let InferenceOutput::ChatCompletion {
                    finish_reason,
                    usage,
                    timings,
                    ..
                } = output
                {
                    let mut last = (self.chunk)(json!({}), Some(finish_reason_str(finish_reason)));
                    last["timings"] = timings_json(timings);
                    self.push(last);
                    if self.include_usage {
                        let mut usage_chunk = (self.chunk)(json!({}), None);
                        usage_chunk["choices"] = json!([]);
                        usage_chunk["usage"] = usage_json(usage);
                        self.push(usage_chunk);
                    }
                }
                self.pending.push(Event::default().data("[DONE]"));
                self.done = true;
            }
            InferenceEvent::Error { message } => {
                self.push(json!({"error": {"message": message, "type": "local_inference_error"}}));
                self.done = true;
            }
        }
    }
}

fn message_json(content: &str, tool_calls: &[ChatToolCall]) -> Value {
    let mut message = json!({"role": "assistant", "content": content});
    if !tool_calls.is_empty() {
        message["tool_calls"] = tool_calls
            .iter()
            .map(|call| {
                json!({
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                })
            })
            .collect();
    }
    message
}

fn finish_reason_str(reason: ChatFinishReason) -> &'static str {
    match reason {
        ChatFinishReason::Stop => "stop",
        ChatFinishReason::Length => "length",
        ChatFinishReason::ToolCalls => "tool_calls",
        ChatFinishReason::Cancelled => "cancelled",
    }
}

fn usage_json(usage: ChatUsage) -> Value {
    json!({
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.prompt_tokens + usage.completion_tokens,
        "prompt_tokens_details": {"cached_tokens": usage.cached_prompt_tokens},
    })
}

fn timings_json(timings: ChatTimings) -> Value {
    json!({
        "prefill_ms": timings.prefill_ms,
        "first_token_ms": timings.first_token_ms,
        "decode_ms": timings.decode_ms,
    })
}
