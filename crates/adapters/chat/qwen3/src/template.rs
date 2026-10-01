//! The model's own chat template (Jinja), rendered as Hugging Face
//! `apply_chat_template` does.

use local_core::ChatMessage;
use local_error::{InfraError, Result};
use minijinja::{context, Environment, Error, ErrorKind, Value};
use serde_json::Value as Json;

const TEMPLATE_NAME: &str = "chat";

#[derive(Debug)]
pub struct ChatTemplate {
    env: Environment<'static>,
}

impl ChatTemplate {
    pub fn new(source: String) -> Result<Self> {
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.add_filter("tojson", tojson);
        env.add_function("raise_exception", raise_exception);
        env.add_template_owned(TEMPLATE_NAME, source)
            .map_err(|err| InfraError::Adapter(format!("parse chat template: {err:#}")))?;
        Ok(Self { env })
    }

    /// The prompt for `messages`, ending with the assistant turn opening.
    pub fn render(&self, messages: &[ChatMessage], tools: &[Json]) -> Result<String> {
        let template = self
            .env
            .get_template(TEMPLATE_NAME)
            .map_err(|err| InfraError::Adapter(format!("chat template: {err:#}")))?;
        let messages = messages.iter().map(message_json).collect::<Vec<_>>();
        let tools = (!tools.is_empty()).then_some(tools);
        template
            .render(context! {
                messages => messages,
                tools => tools,
                add_generation_prompt => true,
                enable_thinking => false,
            })
            .map_err(|err| InfraError::BadRequest(format!("render chat template: {err:#}")))
    }
}

/// A message as the template sees it (the OpenAI shape).
fn message_json(message: &ChatMessage) -> Json {
    let mut out = serde_json::Map::new();
    out.insert("role".into(), Json::String(message.role.clone()));
    out.insert(
        "content".into(),
        message
            .content
            .clone()
            .map(Json::String)
            .unwrap_or(Json::Null),
    );
    if !message.tool_calls.is_empty() {
        let calls = message
            .tool_calls
            .iter()
            .map(|call| {
                // Object arguments render with the template's `tojson`, like
                // the calls the model was trained on; others stay verbatim.
                let arguments = serde_json::from_str::<Json>(&call.arguments)
                    .ok()
                    .filter(Json::is_object)
                    .unwrap_or_else(|| Json::String(call.arguments.clone()));
                serde_json::json!({
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": arguments},
                })
            })
            .collect();
        out.insert("tool_calls".into(), Json::Array(calls));
    }
    if let Some(id) = &message.tool_call_id {
        out.insert("tool_call_id".into(), Json::String(id.clone()));
    }
    if let Some(name) = &message.name {
        out.insert("name".into(), Json::String(name.clone()));
    }
    Json::Object(out)
}

fn raise_exception(message: String) -> std::result::Result<Value, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, message))
}

/// Python `json.dumps(value, ensure_ascii=False)`, as Hugging Face's
/// `tojson`: `", "` and `": "` separators, keys in their given order.
fn tojson(value: Value) -> std::result::Result<Value, Error> {
    let json = serde_json::to_value(&value)
        .map_err(|err| Error::new(ErrorKind::InvalidOperation, err.to_string()))?;
    let mut out = String::new();
    python_json(&json, &mut out);
    Ok(Value::from_safe_string(out))
}

pub(crate) fn python_json(value: &Json, out: &mut String) {
    match value {
        Json::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                python_json(item, out);
            }
            out.push(']');
        }
        Json::Object(map) => {
            out.push('{');
            for (index, (key, item)) in map.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                out.push_str(&Json::String(key.clone()).to_string());
                out.push_str(": ");
                python_json(item, out);
            }
            out.push('}');
        }
        other => out.push_str(&other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use local_core::ChatToolCall;

    const QWEN3: &str = include_str!("../tests/qwen3_chat_template.jinja");

    fn user(text: &str) -> ChatMessage {
        ChatMessage {
            role: "user".into(),
            content: Some(text.into()),
            ..ChatMessage::default()
        }
    }

    #[test]
    fn renders_system_user_and_generation_prompt() {
        let template = ChatTemplate::new(QWEN3.to_string()).unwrap();
        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: Some("你是小乐。".into()),
                ..ChatMessage::default()
            },
            user("你好"),
        ];
        assert_eq!(
            template.render(&messages, &[]).unwrap(),
            "<|im_start|>system\n你是小乐。<|im_end|>\n<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn renders_tools_like_hugging_face() {
        let template = ChatTemplate::new(QWEN3.to_string()).unwrap();
        let tools: Vec<Json> = vec![serde_json::from_str(
            r#"{"type": "function", "function": {"name": "silence", "description": "保持沉默。", "parameters": {"type": "object", "properties": {}, "required": []}}}"#,
        )
        .unwrap()];
        let prompt = template.render(&[user("嗯")], &tools).unwrap();
        assert!(prompt.contains(
            "<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"silence\", \"description\": \"保持沉默。\", \"parameters\": {\"type\": \"object\", \"properties\": {}, \"required\": []}}}\n</tools>"
        ));
        assert!(prompt.ends_with("<|im_start|>user\n嗯<|im_end|>\n<|im_start|>assistant\n"));
    }

    #[test]
    fn renders_assistant_tool_calls_and_tool_results() {
        let template = ChatTemplate::new(QWEN3.to_string()).unwrap();
        let messages = vec![
            user("几点了"),
            ChatMessage {
                role: "assistant".into(),
                content: Some(String::new()),
                tool_calls: vec![ChatToolCall {
                    id: "call_1".into(),
                    name: "backend_task".into(),
                    arguments: r#"{"task":"查询时间"}"#.into(),
                }],
                ..ChatMessage::default()
            },
            ChatMessage {
                role: "tool".into(),
                content: Some("下午三点".into()),
                tool_call_id: Some("call_1".into()),
                ..ChatMessage::default()
            },
        ];
        let prompt = template.render(&messages, &[]).unwrap();
        assert!(prompt.contains(
            "<|im_start|>assistant\n<tool_call>\n{\"name\": \"backend_task\", \"arguments\": {\"task\": \"查询时间\"}}\n</tool_call><|im_end|>\n"
        ));
        assert!(prompt
            .contains("<|im_start|>user\n<tool_response>\n下午三点\n</tool_response><|im_end|>\n"));
    }
}
