//! Chat completion with Qwen3 decoder graphs exported by the
//! onnxruntime-genai model builder (GroupQueryAttention + MatMulNBits).
//!
//! One conversation's KV cache stays on the device between requests: a new
//! request only prefills the tokens after the prefix it shares with the
//! previous one (system prompt, tools and history of a voice conversation).
//! Tokens stream to a caller-supplied sink, which can stop the generation.

mod sampler;
mod template;

use local_backend_ort::{
    OrtBackend, OrtSession, OrtTensorData, OrtTensorInput, ProviderSelection,
    SessionProviderReport, SharedKvBinding, SharedKvPair,
};
use local_core::{
    ChatFinishReason, ChatMessage, ChatOptions, ChatTimings, ChatToolCall, ChatUsage,
    InferenceEvent, InferenceOutput, ModelSpec,
};
use local_error::{InfraError, Result};
use sampler::{apply_penalties, sample, Rng, SamplerConfig};
use serde_json::Value as Json;
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use template::ChatTemplate;
use tokenizers::Tokenizer;

pub use template::ChatTemplate as Qwen3ChatTemplate;

/// KV cache capacity (tokens) unless the model spec sets `max_context`.
const DEFAULT_MAX_CONTEXT: usize = 8192;
/// Completion tokens unless the request sets `max_tokens`.
const DEFAULT_MAX_TOKENS: usize = 1024;
const TOOL_CALL_START: &str = "<tool_call>";
const TOOL_CALL_END: &str = "</tool_call>";

#[derive(Debug, Clone)]
pub struct Qwen3ChatArtifacts {
    pub root: PathBuf,
    pub model: PathBuf,
    pub genai_config: PathBuf,
    pub tokenizer: PathBuf,
    pub chat_template: PathBuf,
}

impl Qwen3ChatArtifacts {
    pub fn from_spec(spec: &ModelSpec) -> Result<Self> {
        let root = spec
            .artifacts
            .first()
            .map(|artifact| artifact.path.clone())
            .filter(|path| !path.as_os_str().is_empty())
            .ok_or_else(|| InfraError::ModelNotConfigured {
                model_id: spec.id.clone(),
                reason: "chat model has no artifact directory".to_string(),
            })?;
        let missing = |name: &str| InfraError::ModelNotConfigured {
            model_id: spec.id.clone(),
            reason: format!("{name} is missing below {}", root.display()),
        };
        let genai_config = root.join("genai_config.json");
        if !genai_config.is_file() {
            return Err(missing("genai_config.json"));
        }
        let tokenizer = root.join("tokenizer.json");
        if !tokenizer.is_file() {
            return Err(missing("tokenizer.json"));
        }
        let chat_template = root.join("chat_template.jinja");
        let tokenizer_config = root.join("tokenizer_config.json");
        let chat_template = if chat_template.is_file() {
            chat_template
        } else if tokenizer_config.is_file() {
            tokenizer_config
        } else {
            return Err(missing("chat_template.jinja (or tokenizer_config.json)"));
        };
        let config = GenaiConfig::load(&genai_config)?;
        let model = root.join(&config.filename);
        if !model.is_file() {
            return Err(missing(&config.filename));
        }
        Ok(Self {
            root,
            model,
            genai_config,
            tokenizer,
            chat_template,
        })
    }
}

/// The parts of onnxruntime-genai's `genai_config.json` this adapter uses.
#[derive(Debug, Clone)]
struct GenaiConfig {
    filename: String,
    layers: usize,
    kv_heads: usize,
    head_size: usize,
    past_key: String,
    past_value: String,
    present_key: String,
    present_value: String,
    input_ids: String,
    attention_mask: String,
    position_ids: Option<String>,
    logits: String,
    eos: Vec<u32>,
}

impl GenaiConfig {
    fn load(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .map_err(|err| InfraError::Adapter(format!("read {}: {err}", path.display())))?;
        let json: Json = serde_json::from_str(&text)
            .map_err(|err| InfraError::Adapter(format!("parse {}: {err}", path.display())))?;
        let model = &json["model"];
        let decoder = &model["decoder"];
        let str_at = |value: &Json, default: &str| value.as_str().unwrap_or(default).to_string();
        let usize_at = |value: &Json, name: &str| {
            value.as_u64().map(|v| v as usize).ok_or_else(|| {
                InfraError::Adapter(format!("genai_config.json lacks decoder.{name}"))
            })
        };
        let inputs = &decoder["inputs"];
        let outputs = &decoder["outputs"];
        let eos = match &model["eos_token_id"] {
            Json::Array(ids) => ids
                .iter()
                .filter_map(Json::as_u64)
                .map(|id| id as u32)
                .collect(),
            Json::Number(id) => id.as_u64().map(|id| vec![id as u32]).unwrap_or_default(),
            _ => Vec::new(),
        };
        Ok(Self {
            filename: str_at(&decoder["filename"], "model.onnx"),
            layers: usize_at(&decoder["num_hidden_layers"], "num_hidden_layers")?,
            kv_heads: usize_at(&decoder["num_key_value_heads"], "num_key_value_heads")?,
            head_size: usize_at(&decoder["head_size"], "head_size")?,
            past_key: str_at(&inputs["past_key_names"], "past_key_values.%d.key"),
            past_value: str_at(&inputs["past_value_names"], "past_key_values.%d.value"),
            present_key: str_at(&outputs["present_key_names"], "present.%d.key"),
            present_value: str_at(&outputs["present_value_names"], "present.%d.value"),
            input_ids: str_at(&inputs["input_ids"], "input_ids"),
            attention_mask: str_at(&inputs["attention_mask"], "attention_mask"),
            position_ids: inputs["position_ids"].as_str().map(str::to_string),
            logits: str_at(&outputs["logits"], "logits"),
            eos,
        })
    }

    fn layer_name(pattern: &str, layer: usize) -> String {
        pattern.replace("%d", &layer.to_string())
    }
}

#[derive(Debug)]
pub struct Qwen3ChatAdapter {
    model_id: String,
    artifacts: Qwen3ChatArtifacts,
    config: GenaiConfig,
    tokenizer: Tokenizer,
    template: ChatTemplate,
    session: OrtSession,
    cache: SharedKvBinding,
    /// Tokens whose keys/values are in `cache`, in order.
    cached: Vec<u32>,
    tool_call_start: Option<u32>,
    tool_call_end: Option<u32>,
}

impl Qwen3ChatAdapter {
    pub fn load(spec: &ModelSpec) -> Result<Self> {
        let artifacts = Qwen3ChatArtifacts::from_spec(spec)?;
        let config = GenaiConfig::load(&artifacts.genai_config)?;
        let tokenizer = Tokenizer::from_file(&artifacts.tokenizer).map_err(|err| {
            InfraError::Adapter(format!(
                "load tokenizer {}: {err}",
                artifacts.tokenizer.display()
            ))
        })?;
        let template = ChatTemplate::new(read_chat_template(&artifacts.chat_template)?)?;
        let backend = OrtBackend::new(ProviderSelection::from_strings(
            &spec.runtime.provider_order,
        ));
        let session = backend.load_session(&artifacts.model)?;
        let capacity = spec
            .metadata
            .get("max_context")
            .and_then(Json::as_u64)
            .map(|value| value as usize)
            .unwrap_or(DEFAULT_MAX_CONTEXT);
        let pairs = (0..config.layers)
            .flat_map(|layer| {
                [
                    SharedKvPair {
                        past_input: GenaiConfig::layer_name(&config.past_key, layer),
                        present_output: GenaiConfig::layer_name(&config.present_key, layer),
                    },
                    SharedKvPair {
                        past_input: GenaiConfig::layer_name(&config.past_value, layer),
                        present_output: GenaiConfig::layer_name(&config.present_value, layer),
                    },
                ]
            })
            .collect::<Vec<_>>();
        let cache = session.create_shared_kv_binding(
            &pairs,
            [1, config.kv_heads, capacity, config.head_size],
            &config.logits,
        )?;
        let tool_call_start = tokenizer.token_to_id(TOOL_CALL_START);
        let tool_call_end = tokenizer.token_to_id(TOOL_CALL_END);
        tracing::info!(
            model_id = spec.id,
            model_path = %artifacts.model.display(),
            provider = ?session.provider(),
            capacity,
            kv_element = ?cache.element(),
            layers = config.layers,
            "Qwen3 chat model loaded"
        );
        Ok(Self {
            model_id: spec.id.clone(),
            artifacts,
            config,
            tokenizer,
            template,
            session,
            cache,
            cached: Vec::new(),
            tool_call_start,
            tool_call_end,
        })
    }

    pub fn artifacts(&self) -> &Qwen3ChatArtifacts {
        &self.artifacts
    }

    pub fn provider_report(&self) -> SessionProviderReport {
        self.session.provider_report()
    }

    /// Generates the assistant turn for `messages`. `sink` receives content
    /// deltas and completed tool calls as they are produced; returning
    /// `false` stops the generation (finish reason `cancelled`).
    pub fn complete(
        &mut self,
        messages: &[ChatMessage],
        tools: &[Json],
        options: &ChatOptions,
        sink: &mut dyn FnMut(InferenceEvent) -> bool,
    ) -> Result<InferenceOutput> {
        if messages.is_empty() {
            return Err(InfraError::BadRequest(
                "chat.complete requires at least one message".to_string(),
            ));
        }
        let started = Instant::now();
        let prompt = self.template.render(messages, tools)?;
        let prompt_ids = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|err| InfraError::Adapter(format!("tokenize prompt: {err}")))?
            .get_ids()
            .to_vec();
        let capacity = self.cache.capacity();
        if prompt_ids.len() >= capacity {
            return Err(InfraError::BadRequest(format!(
                "prompt has {} tokens; the model's context holds {capacity}",
                prompt_ids.len()
            )));
        }
        let max_tokens = options
            .max_tokens
            .unwrap_or(DEFAULT_MAX_TOKENS)
            .min(capacity - prompt_ids.len());
        let sampler = SamplerConfig {
            temperature: options.temperature.unwrap_or(0.7).max(0.0),
            top_p: options.top_p.unwrap_or(0.8),
            top_k: options.top_k.unwrap_or(20),
            presence_penalty: options.presence_penalty.unwrap_or(0.0),
            frequency_penalty: options.frequency_penalty.unwrap_or(0.0),
        };
        let mut rng = Rng::new(options.seed.unwrap_or_else(time_seed));
        let tool_first_tokens = self.tool_first_tokens(&options.tool_bias);

        // Reuse the cached prefix; at least one token is fed to get logits.
        let reused = common_prefix(&self.cached, &prompt_ids).min(prompt_ids.len() - 1);
        let mut logits = match self.feed(&prompt_ids[reused..], reused) {
            Ok(logits) => logits,
            Err(err) => {
                self.cached.clear();
                return Err(err);
            }
        };
        self.cached.truncate(reused);
        self.cached.extend_from_slice(&prompt_ids[reused..]);
        let prefill_ms = started.elapsed().as_millis() as u64;

        let mut state = Decoding::new(&options.stop);
        let mut counts = HashMap::<u32, usize>::new();
        let mut first_token_ms = 0;
        let decode_started = Instant::now();
        let mut finish = ChatFinishReason::Length;
        for step in 0..max_tokens {
            if let Some(bias) = options.tool_call_bias {
                if let (Some(id), false) = (self.tool_call_start, state.in_tool_call()) {
                    add_bias(&mut logits, id, bias);
                }
            }
            for (&id, &bias) in &options.logit_bias {
                add_bias(&mut logits, id, bias);
            }
            if !tool_first_tokens.is_empty() && state.at_tool_name(&self.tokenizer) {
                for (&id, &bias) in &tool_first_tokens {
                    add_bias(&mut logits, id, bias);
                }
            }
            apply_penalties(&mut logits, &counts, &sampler);
            let token = sample(&logits, &sampler, &mut rng);
            if step == 0 {
                first_token_ms = started.elapsed().as_millis() as u64;
            }
            if self.config.eos.contains(&token) {
                finish = ChatFinishReason::Stop;
                break;
            }
            *counts.entry(token).or_default() += 1;
            let event = state.push(
                token,
                self.tool_call_start,
                self.tool_call_end,
                &self.tokenizer,
            )?;
            let mut keep_going = true;
            for event in event.events {
                keep_going &= sink(event);
            }
            if event.stopped {
                finish = ChatFinishReason::Stop;
                break;
            }
            if !keep_going {
                finish = ChatFinishReason::Cancelled;
                break;
            }
            if self.cached.len() + 1 >= capacity || step + 1 == max_tokens {
                break;
            }
            let position = self.cached.len();
            logits = match self.feed(&[token], position) {
                Ok(logits) => logits,
                Err(err) => {
                    self.cached.clear();
                    return Err(err);
                }
            };
            self.cached.push(token);
        }
        let (content, tail, tool_calls) = state.finish(&self.tokenizer)?;
        if finish != ChatFinishReason::Cancelled {
            // What was held back (a possible stop string's start) is content.
            if !tail.is_empty() {
                sink(InferenceEvent::ChatDelta { content: tail });
            }
            for (index, call) in tool_calls.iter().enumerate().skip(state.emitted_calls) {
                sink(InferenceEvent::ChatToolCall {
                    index,
                    call: call.clone(),
                });
            }
        }
        if !tool_calls.is_empty() && finish == ChatFinishReason::Stop {
            finish = ChatFinishReason::ToolCalls;
        }
        let completion_tokens = counts.values().sum();
        tracing::info!(
            model_id = self.model_id,
            prompt_tokens = prompt_ids.len(),
            cached_prompt_tokens = reused,
            completion_tokens,
            prefill_ms,
            first_token_ms,
            decode_ms = decode_started.elapsed().as_millis() as u64,
            finish = ?finish,
            "chat completion finished"
        );
        Ok(InferenceOutput::ChatCompletion {
            content,
            tool_calls,
            finish_reason: finish,
            usage: ChatUsage {
                prompt_tokens: prompt_ids.len(),
                cached_prompt_tokens: reused,
                completion_tokens,
            },
            timings: ChatTimings {
                prefill_ms,
                first_token_ms,
                decode_ms: decode_started.elapsed().as_millis() as u64,
            },
        })
    }

    /// Runs `tokens` starting at cache position `past`; returns the logits
    /// of the last one.
    fn feed(&mut self, tokens: &[u32], past: usize) -> Result<Vec<f32>> {
        let total = past + tokens.len();
        let mut inputs = vec![
            OrtTensorInput {
                name: self.config.input_ids.clone(),
                shape: vec![1, tokens.len()],
                data: OrtTensorData::I64(tokens.iter().map(|&id| id as i64).collect()),
            },
            OrtTensorInput {
                name: self.config.attention_mask.clone(),
                shape: vec![1, total],
                data: OrtTensorData::I64(vec![1; total]),
            },
        ];
        if let Some(name) = &self.config.position_ids {
            inputs.push(OrtTensorInput {
                name: name.clone(),
                shape: vec![1, tokens.len()],
                data: OrtTensorData::I64((past..total).map(|p| p as i64).collect()),
            });
        }
        let output = self
            .session
            .run_shared_kv_binding(&mut self.cache, inputs)?;
        last_row(output.data, output.shape.last().copied().unwrap_or(0))
    }

    fn tool_first_tokens(
        &self,
        biases: &std::collections::BTreeMap<String, f32>,
    ) -> HashMap<u32, f32> {
        biases
            .iter()
            .filter_map(|(name, bias)| {
                let ids = self.tokenizer.encode(name.as_str(), false).ok()?;
                ids.get_ids().first().map(|id| (*id, *bias))
            })
            .collect()
    }
}

fn read_chat_template(path: &Path) -> Result<String> {
    let text = fs::read_to_string(path)
        .map_err(|err| InfraError::Adapter(format!("read {}: {err}", path.display())))?;
    if path.extension().is_some_and(|ext| ext == "json") {
        let json: Json = serde_json::from_str(&text)
            .map_err(|err| InfraError::Adapter(format!("parse {}: {err}", path.display())))?;
        return json["chat_template"]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| {
                InfraError::Adapter(format!("{} has no chat_template", path.display()))
            });
    }
    Ok(text)
}

fn last_row(data: OrtTensorData, vocab: usize) -> Result<Vec<f32>> {
    let row = |len: usize| {
        if vocab == 0 || len < vocab {
            Err(InfraError::Adapter(format!(
                "logits of {len} values do not hold a {vocab}-token row"
            )))
        } else {
            Ok(len - vocab)
        }
    };
    match data {
        OrtTensorData::F32(values) => {
            let start = row(values.len())?;
            Ok(values[start..].to_vec())
        }
        OrtTensorData::F16(values) => {
            let start = row(values.len())?;
            Ok(values[start..].iter().map(|v| v.to_f32()).collect())
        }
        other => Err(InfraError::Adapter(format!(
            "logits are {:?}, expected FP32 or FP16",
            other.element_type()
        ))),
    }
}

fn add_bias(logits: &mut [f32], id: u32, bias: f32) {
    if let Some(logit) = logits.get_mut(id as usize) {
        *logit += bias;
    }
}

fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

fn time_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x5EED)
}

/// A tool call id unique in this process (and unlikely to repeat across
/// restarts), independent of the sampling seed.
fn new_call_id() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let serial = NEXT.fetch_add(1, Ordering::Relaxed);
    format!(
        "call_{:x}{:04x}",
        time_seed() & 0xffff_ffff,
        serial & 0xffff
    )
}

// ---------------------------------------------------------------- decoding state

/// Splits generated tokens into streamed content and tool calls.
#[derive(Debug)]
struct Decoding {
    /// Content tokens (outside tool calls).
    content: Vec<u32>,
    /// Characters of decoded content already sent.
    sent_chars: usize,
    /// Tokens of the tool call being generated.
    call: Option<Vec<u32>>,
    calls: Vec<ChatToolCall>,
    emitted_calls: usize,
    stop: Vec<String>,
    hold_chars: usize,
}

struct Pushed {
    events: Vec<InferenceEvent>,
    stopped: bool,
}

impl Decoding {
    fn new(stop: &[String]) -> Self {
        let stop = stop
            .iter()
            .filter(|s| !s.is_empty())
            .cloned()
            .collect::<Vec<_>>();
        let hold_chars = stop
            .iter()
            .map(|s| s.chars().count().saturating_sub(1))
            .max()
            .unwrap_or(0);
        Self {
            content: Vec::new(),
            sent_chars: 0,
            call: None,
            calls: Vec::new(),
            emitted_calls: 0,
            stop,
            hold_chars,
        }
    }

    fn in_tool_call(&self) -> bool {
        self.call.is_some()
    }

    /// Whether the tool call so far is `{"name": "` (the name comes next).
    fn at_tool_name(&self, tokenizer: &Tokenizer) -> bool {
        let Some(call) = &self.call else {
            return false;
        };
        let Ok(text) = tokenizer.decode(call, false) else {
            return false;
        };
        let rest = text.trim_start();
        let Some(rest) = rest.strip_prefix('{') else {
            return false;
        };
        let rest = rest.trim_start();
        let Some(rest) = rest.strip_prefix("\"name\"") else {
            return false;
        };
        let rest = rest.trim_start();
        let Some(rest) = rest.strip_prefix(':') else {
            return false;
        };
        rest.trim_start() == "\""
    }

    fn push(
        &mut self,
        token: u32,
        call_start: Option<u32>,
        call_end: Option<u32>,
        tokenizer: &Tokenizer,
    ) -> Result<Pushed> {
        let mut events = Vec::new();
        if Some(token) == call_start && self.call.is_none() {
            self.call = Some(Vec::new());
            return Ok(Pushed {
                events,
                stopped: false,
            });
        }
        if let Some(call) = &mut self.call {
            if Some(token) == call_end {
                let text = tokenizer
                    .decode(call, false)
                    .map_err(|err| InfraError::Adapter(format!("decode tool call: {err}")))?;
                self.call = None;
                if let Some(parsed) = parse_tool_call(&text) {
                    events.push(InferenceEvent::ChatToolCall {
                        index: self.calls.len(),
                        call: parsed.clone(),
                    });
                    self.calls.push(parsed);
                    self.emitted_calls = self.calls.len();
                }
            } else {
                call.push(token);
            }
            return Ok(Pushed {
                events,
                stopped: false,
            });
        }
        self.content.push(token);
        let text = decode_content(tokenizer, &self.content)?;
        if let Some(cut) = self.stop.iter().filter_map(|s| text.find(s.as_str())).min() {
            let kept = text[..cut].chars().count();
            if kept > self.sent_chars {
                events.push(InferenceEvent::ChatDelta {
                    content: text[..cut].chars().skip(self.sent_chars).collect(),
                });
                self.sent_chars = kept;
            }
            return Ok(Pushed {
                events,
                stopped: true,
            });
        }
        // Hold back an incomplete UTF-8 sequence and a possible stop prefix.
        let complete = text.trim_end_matches('\u{FFFD}');
        let ready = complete.chars().count().saturating_sub(self.hold_chars);
        if ready > self.sent_chars {
            events.push(InferenceEvent::ChatDelta {
                content: complete
                    .chars()
                    .skip(self.sent_chars)
                    .take(ready - self.sent_chars)
                    .collect(),
            });
            self.sent_chars = ready;
        }
        Ok(Pushed {
            events,
            stopped: false,
        })
    }

    /// The content and tool calls of the whole turn, and the content not
    /// streamed yet (held back as a possible stop string). A tool call cut
    /// short (the model sometimes ends right after the JSON) still counts
    /// when its JSON is complete.
    fn finish(&mut self, tokenizer: &Tokenizer) -> Result<(String, String, Vec<ChatToolCall>)> {
        if let Some(call) = self.call.take() {
            let text = tokenizer
                .decode(&call, false)
                .map_err(|err| InfraError::Adapter(format!("decode tool call: {err}")))?;
            if let Some(parsed) = parse_tool_call(&text) {
                self.calls.push(parsed);
            }
        }
        let mut content = decode_content(tokenizer, &self.content)?;
        if let Some(cut) = self
            .stop
            .iter()
            .filter_map(|s| content.find(s.as_str()))
            .min()
        {
            content.truncate(cut);
        }
        let complete = content.trim_end_matches('\u{FFFD}');
        let tail: String = complete.chars().skip(self.sent_chars).collect();
        self.sent_chars += tail.chars().count();
        Ok((content.trim().to_string(), tail, self.calls.clone()))
    }
}

fn decode_content(tokenizer: &Tokenizer, ids: &[u32]) -> Result<String> {
    tokenizer
        .decode(ids, true)
        .map_err(|err| InfraError::Adapter(format!("decode completion: {err}")))
}

/// `{"name": ..., "arguments": {...}}` inside `<tool_call>` tags.
fn parse_tool_call(text: &str) -> Option<ChatToolCall> {
    let json: Json = serde_json::from_str(text.trim()).ok()?;
    let name = json.get("name")?.as_str()?.to_string();
    let arguments = match json.get("arguments") {
        None | Some(Json::Null) => "{}".to_string(),
        Some(Json::String(text)) => text.clone(),
        Some(other) => other.to_string(),
    };
    Some(ChatToolCall {
        id: new_call_id(),
        name,
        arguments,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_calls_parse_with_object_or_string_arguments() {
        let call =
            parse_tool_call(r#" {"name": "backend_task", "arguments": {"task": "查询时间"}} "#)
                .unwrap();
        assert_eq!(call.name, "backend_task");
        assert_eq!(call.arguments, r#"{"task":"查询时间"}"#);
        let call = parse_tool_call(r#"{"name": "silence", "arguments": "{}"}"#).unwrap();
        assert_eq!(call.arguments, "{}");
        assert!(parse_tool_call(r#"{"name": "x""#).is_none());
    }

    /// A word-level tokenizer whose decoder concatenates tokens.
    fn tiny_tokenizer(words: &[&str]) -> Tokenizer {
        use tokenizers::{decoders::fuse::Fuse, models::wordlevel::WordLevel};
        let vocab = words
            .iter()
            .enumerate()
            .map(|(i, w)| (w.to_string(), i as u32))
            .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token(words[0].to_string())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_decoder(Some(Fuse::new()));
        tokenizer
    }

    fn deltas(events: &[InferenceEvent]) -> String {
        events
            .iter()
            .filter_map(|e| match e {
                InferenceEvent::ChatDelta { content } => Some(content.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn text_held_for_a_stop_string_is_sent_at_the_end() {
        let tokenizer = tiny_tokenizer(&["?", "你", "好", "E", "N", "D"]);
        let mut state = Decoding::new(&["END".to_string()]);
        let mut events = Vec::new();
        for token in [1, 2, 3, 4] {
            events.extend(state.push(token, None, None, &tokenizer).unwrap().events);
        }
        assert_eq!(deltas(&events), "你好"); // "EN" may start "END"
        let (content, tail, _) = state.finish(&tokenizer).unwrap();
        assert_eq!(content, "你好EN");
        assert_eq!(tail, "EN");

        let mut state = Decoding::new(&["END".to_string()]);
        let mut events = Vec::new();
        for token in [1, 3, 4, 5] {
            let pushed = state.push(token, None, None, &tokenizer).unwrap();
            events.extend(pushed.events);
            if pushed.stopped {
                break;
            }
        }
        let (content, tail, _) = state.finish(&tokenizer).unwrap();
        assert_eq!(deltas(&events) + &tail, "你");
        assert_eq!(content, "你");
    }

    #[test]
    fn streamed_tool_calls_have_ids() {
        let tokenizer = tiny_tokenizer(&["?", r#"{"name": "silence", "arguments": {}}"#]);
        let mut state = Decoding::new(&[]);
        let (start, end) = (90, 91);
        state
            .push(start, Some(start), Some(end), &tokenizer)
            .unwrap();
        state.push(1, Some(start), Some(end), &tokenizer).unwrap();
        let events = state
            .push(end, Some(start), Some(end), &tokenizer)
            .unwrap()
            .events;
        let InferenceEvent::ChatToolCall { call, .. } = &events[0] else {
            panic!("no tool call: {events:?}");
        };
        assert_eq!(call.name, "silence");
        assert!(call.id.starts_with("call_"));
        let (_, _, calls) = state.finish(&tokenizer).unwrap();
        assert_eq!(calls[0].id, call.id);
        assert_ne!(new_call_id(), new_call_id());
    }

    #[test]
    fn common_prefix_counts_equal_leading_tokens() {
        assert_eq!(common_prefix(&[1, 2, 3], &[1, 2, 4, 5]), 2);
        assert_eq!(common_prefix(&[], &[1]), 0);
        assert_eq!(common_prefix(&[1, 2], &[1, 2]), 2);
    }

    fn real_spec(dir: &str) -> ModelSpec {
        serde_json::from_value(serde_json::json!({
            "id": "qwen3-chat-test",
            "name": "Qwen3 chat test",
            "adapter": "qwen3_chat",
            "backend": "ort",
            "task_kinds": ["chat.complete"],
            "artifacts": [{"type": "local", "path": dir}],
            "runtime": {"provider_order": std::env::var("LOCAL_QWEN3_CHAT_PROVIDERS").unwrap_or_else(|_| "cuda,cpu".into()).split(',').collect::<Vec<_>>(), "max_concurrency": 1, "idle_ttl_sec": 600},
            "metadata": {"max_context": 4096},
        }))
        .unwrap()
    }

    fn msg(role: &str, text: &str) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: Some(text.into()),
            ..ChatMessage::default()
        }
    }

    /// `LOCAL_QWEN3_CHAT_MODEL_DIR=<export dir> cargo test -p
    /// local-adapter-qwen3-chat --features cuda real_model -- --nocapture`
    #[test]
    fn real_model_smoke_if_env_set() {
        let Ok(dir) = std::env::var("LOCAL_QWEN3_CHAT_MODEL_DIR") else {
            eprintln!("LOCAL_QWEN3_CHAT_MODEL_DIR not set; skipping");
            return;
        };
        let _ = tracing_subscriber::fmt().with_env_filter("info").try_init();
        let mut adapter = Qwen3ChatAdapter::load(&real_spec(&dir)).unwrap();
        eprintln!("provider: {:?}", adapter.provider_report());
        let tools: Vec<Json> = serde_json::from_str(
            r#"[{"type": "function", "function": {"name": "silence", "description": "没有需要回应的内容，保持沉默。", "parameters": {"type": "object", "properties": {}, "required": []}}},
                {"type": "function", "function": {"name": "backend_task", "description": "交给后台执行：联网搜索、查询实时信息、执行操作。", "parameters": {"type": "object", "properties": {"task": {"type": "string", "description": "要完成的任务。"}}, "required": ["task"]}}}]"#,
        )
        .unwrap();
        let system = msg(
            "system",
            "你是语音助手小乐，正在和一个人一对一语音聊天。需要实时信息（时间、天气、新闻）或执行操作时调用 backend_task；没有需要回应的内容时调用 silence；其他情况直接用简短的中文口语回答。",
        );
        let greedy = ChatOptions {
            temperature: Some(0.0),
            max_tokens: Some(40),
            ..ChatOptions::default()
        };
        let run = |adapter: &mut Qwen3ChatAdapter, messages: &[ChatMessage]| {
            let mut deltas = Vec::new();
            let out = adapter
                .complete(messages, &tools, &greedy, &mut |event| {
                    if let InferenceEvent::ChatDelta { content } = event {
                        deltas.push(content);
                    }
                    true
                })
                .unwrap();
            eprintln!("{out:?}\n  deltas={deltas:?}");
            (out, deltas)
        };

        let first = vec![system.clone(), msg("user", "现在几点了？")];
        let (out, _) = run(&mut adapter, &first);
        let InferenceOutput::ChatCompletion {
            tool_calls,
            finish_reason,
            ..
        } = out
        else {
            panic!("not a chat completion");
        };
        assert_eq!(finish_reason, ChatFinishReason::ToolCalls);
        assert_eq!(tool_calls[0].name, "backend_task");

        let second = vec![
            system.clone(),
            msg("user", "一百二十三加四百五十六等于多少？"),
        ];
        let (out, deltas) = run(&mut adapter, &second);
        let InferenceOutput::ChatCompletion { content, usage, .. } = out else {
            panic!("not a chat completion");
        };
        assert!(
            content.contains("五百七十九") || content.contains("579"),
            "{content}"
        );
        assert_eq!(deltas.concat().trim(), content);
        // The system prompt and tools were reused from the first request.
        assert!(usage.cached_prompt_tokens > 100, "{usage:?}");

        // Stopping from the sink ends the turn.
        let mut seen = 0;
        let out = adapter
            .complete(
                &[system, msg("user", "给我讲个长一点的故事。")],
                &tools,
                &greedy,
                &mut |_| {
                    seen += 1;
                    seen < 3
                },
            )
            .unwrap();
        let InferenceOutput::ChatCompletion { finish_reason, .. } = out else {
            panic!("not a chat completion");
        };
        assert_eq!(finish_reason, ChatFinishReason::Cancelled);
    }

    #[test]
    fn last_row_takes_the_final_vocab_slice() {
        let row = last_row(OrtTensorData::F32(vec![1.0, 2.0, 3.0, 4.0]), 2).unwrap();
        assert_eq!(row, vec![3.0, 4.0]);
        assert!(last_row(OrtTensorData::F32(vec![1.0]), 2).is_err());
    }
}
