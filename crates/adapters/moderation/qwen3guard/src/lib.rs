//! Safety verdicts of text with Qwen3Guard-Gen (Qwen3 decoder graphs exported
//! by the onnxruntime-genai model builder, run by the `qwen3_chat` adapter).
//!
//! The text is the user turn of the model's own moderation template, followed
//! by the answer's forced start `Safety:`; the logits of the next token over
//! ` Safe`, ` Unsafe` and ` Cont(roversial)` give the verdict's probabilities
//! in one prefill (the template's ~290-token preamble stays in the KV cache
//! between requests). Only a text not judged safe decodes its categories
//! (`Categories: Violent, PII`), a few greedy tokens.
//!
//! Provenance: `Qwen/Qwen3Guard-Gen-0.6B` (Apache-2.0); the export command is
//! in `configs/providers/moderation/qwen3guard.yaml`.

use local_adapter_qwen3_chat::Qwen3ChatAdapter;
use local_backend_ort::SessionProviderReport;
use local_core::{ChatMessage, InferenceOutput, ModelSpec, TextModeration};
use local_error::{InfraError, Result};
use serde_json::Value as Json;
use std::time::Instant;

/// Text tokens per window unless the model spec sets `window_tokens`.
const DEFAULT_WINDOW: usize = 2048;
/// Tokens shared by consecutive windows of a long text.
const WINDOW_OVERLAP: usize = 64;
/// Room the prompt and the categories need beside the text.
const PROMPT_ROOM: usize = 512;
/// Most tokens a category list takes.
const MAX_CATEGORY_TOKENS: usize = 32;
const ANSWER_START: &str = "Safety:";

#[derive(Debug)]
pub struct Qwen3GuardAdapter {
    model_id: String,
    model: Qwen3ChatAdapter,
    /// ` Safe`, ` Unsafe`, ` Cont` (first token of ` Controversial`).
    verdict_tokens: [u32; 3],
    window: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Verdict {
    Safe,
    Unsafe,
    Controversial,
}

impl Verdict {
    fn word(self) -> &'static str {
        match self {
            Verdict::Safe => "Safe",
            Verdict::Unsafe => "Unsafe",
            Verdict::Controversial => "Controversial",
        }
    }
}

/// One window's verdict: probabilities of safe, unsafe, controversial.
#[derive(Debug, Clone, Copy)]
struct Judged {
    probs: [f32; 3],
}

impl Judged {
    fn verdict(&self) -> Verdict {
        let [safe, unsafe_, controversial] = self.probs;
        if safe >= unsafe_ && safe >= controversial {
            Verdict::Safe
        } else if unsafe_ >= controversial {
            Verdict::Unsafe
        } else {
            Verdict::Controversial
        }
    }
}

impl Qwen3GuardAdapter {
    pub fn load(spec: &ModelSpec) -> Result<Self> {
        let model = Qwen3ChatAdapter::load(spec)?;
        let tokenizer = model.tokenizer();
        // The verdict word is the token after `Safety:` in each answer.
        let verdict_token = |word: &str| -> Result<u32> {
            let prefix = encode(tokenizer, ANSWER_START)?;
            let ids = encode(tokenizer, &format!("{ANSWER_START} {word}"))?;
            if ids.len() <= prefix.len() || ids[..prefix.len()] != prefix[..] {
                return Err(InfraError::ModelNotConfigured {
                    model_id: spec.id.clone(),
                    reason: format!("the tokenizer splits `{ANSWER_START} {word}` unexpectedly"),
                });
            }
            Ok(ids[prefix.len()])
        };
        let verdict_tokens = [
            verdict_token("Safe")?,
            verdict_token("Unsafe")?,
            verdict_token("Controversial")?,
        ];
        let room = model.capacity().saturating_sub(PROMPT_ROOM);
        let window = spec
            .metadata
            .get("window_tokens")
            .and_then(Json::as_u64)
            .map_or(DEFAULT_WINDOW, |v| v as usize)
            .min(room);
        if window <= WINDOW_OVERLAP * 2 {
            return Err(InfraError::ModelNotConfigured {
                model_id: spec.id.clone(),
                reason: format!(
                    "a {}-token context leaves no room for text (max_context too small)",
                    model.capacity()
                ),
            });
        }
        tracing::info!(model_id = spec.id, window, "Qwen3Guard model loaded");
        Ok(Self {
            model_id: spec.id.clone(),
            model,
            verdict_tokens,
            window,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn provider_report(&self) -> SessionProviderReport {
        self.model.provider_report()
    }

    /// A verdict for each text, in order.
    pub fn moderate(&mut self, texts: &[String]) -> Result<InferenceOutput> {
        if texts.is_empty() {
            return Err(InfraError::BadRequest(
                "text.moderate requires at least one text".to_string(),
            ));
        }
        let results = texts
            .iter()
            .map(|text| self.moderate_one(text))
            .collect::<Result<Vec<_>>>()?;
        Ok(InferenceOutput::TextModerations { results })
    }

    fn moderate_one(&mut self, text: &str) -> Result<TextModeration> {
        let started = Instant::now();
        let ids = encode(self.model.tokenizer(), text)?;
        let windows = self.windows(text, &ids)?;
        let mut worst: Option<(Judged, &str)> = None;
        for window in &windows {
            let judged = self.judge(window)?;
            if worst.is_none_or(|(w, _)| judged.probs[0] < w.probs[0]) {
                worst = Some((judged, window));
            }
        }
        let (judged, window) = worst.expect("a text has at least one window");
        let verdict = judged.verdict();
        let categories = if verdict == Verdict::Safe {
            Vec::new()
        } else {
            self.categories(window, verdict)?
        };
        tracing::debug!(
            model_id = self.model_id,
            tokens = ids.len(),
            windows = windows.len(),
            safe = judged.probs[0],
            elapsed_ms = started.elapsed().as_millis() as u64,
            "text moderated"
        );
        Ok(TextModeration {
            safe: judged.probs[0],
            unsafe_: judged.probs[1],
            controversial: judged.probs[2],
            categories,
            tokens: ids.len(),
            windows: windows.len(),
        })
    }

    /// The text itself when it fits one window, else overlapping windows of
    /// its tokens.
    fn windows<'a>(&self, text: &'a str, ids: &[u32]) -> Result<Vec<std::borrow::Cow<'a, str>>> {
        if ids.len() <= self.window {
            return Ok(vec![std::borrow::Cow::Borrowed(text)]);
        }
        let stride = self.window - WINDOW_OVERLAP;
        let mut out = Vec::new();
        let mut start = 0;
        loop {
            let end = (start + self.window).min(ids.len());
            let piece = self
                .model
                .tokenizer()
                .decode(&ids[start..end], false)
                .map_err(|err| InfraError::Adapter(format!("decode a text window: {err}")))?;
            out.push(std::borrow::Cow::Owned(piece));
            if end == ids.len() {
                break;
            }
            start += stride;
        }
        Ok(out)
    }

    /// The model's moderation prompt for `text`, up to the answer's start.
    fn prompt(&self, text: &str) -> Result<String> {
        let message = ChatMessage {
            role: "user".to_string(),
            content: Some(text.to_string()),
            ..Default::default()
        };
        Ok(self.model.template().render(&[message], &[])? + ANSWER_START)
    }

    fn judge(&mut self, text: &str) -> Result<Judged> {
        let ids = encode(self.model.tokenizer(), &self.prompt(text)?)?;
        let (logits, _) = self.model.prefill(&ids)?;
        Ok(Judged {
            probs: softmax3(&logits, self.verdict_tokens)?,
        })
    }

    /// The categories the model names after `verdict` (greedy).
    fn categories(&mut self, text: &str, verdict: Verdict) -> Result<Vec<String>> {
        let prompt = format!("{} {}\nCategories:", self.prompt(text)?, verdict.word());
        let ids = encode(self.model.tokenizer(), &prompt)?;
        let (mut logits, _) = self.model.prefill(&ids)?;
        let mut out = Vec::new();
        for _ in 0..MAX_CATEGORY_TOKENS {
            let token = argmax(&logits);
            if self.model.eos_tokens().contains(&token) {
                break;
            }
            out.push(token);
            let so_far = self
                .model
                .tokenizer()
                .decode(&out, false)
                .map_err(|err| InfraError::Adapter(format!("decode categories: {err}")))?;
            if so_far.contains('\n') {
                break;
            }
            logits = self.model.step(token)?;
        }
        let text = self
            .model
            .tokenizer()
            .decode(&out, false)
            .map_err(|err| InfraError::Adapter(format!("decode categories: {err}")))?;
        Ok(parse_categories(&text))
    }
}

/// `" Violent, PII\n..."` -> `["Violent", "PII"]` (`None` names none).
fn parse_categories(text: &str) -> Vec<String> {
    text.lines()
        .next()
        .unwrap_or("")
        .split(',')
        .map(str::trim)
        .filter(|c| !c.is_empty() && *c != "None")
        .map(str::to_string)
        .collect()
}

fn encode(tokenizer: &tokenizers::Tokenizer, text: &str) -> Result<Vec<u32>> {
    Ok(tokenizer
        .encode(text, false)
        .map_err(|err| InfraError::Adapter(format!("tokenize: {err}")))?
        .get_ids()
        .to_vec())
}

fn softmax3(logits: &[f32], ids: [u32; 3]) -> Result<[f32; 3]> {
    let pick = |id: u32| {
        logits.get(id as usize).copied().ok_or_else(|| {
            InfraError::Adapter(format!("token {id} is outside the model's vocabulary"))
        })
    };
    let raw = [pick(ids[0])?, pick(ids[1])?, pick(ids[2])?];
    let max = raw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp = raw.map(|v| (v - max).exp());
    let sum: f32 = exp.iter().sum();
    let probs = exp.map(|v| v / sum);
    // A NaN verdict would read as neither safe nor unsafe: fail instead.
    if probs.iter().any(|p| !p.is_finite()) {
        return Err(InfraError::Backend(format!(
            "the verdict logits are not finite: {raw:?}"
        )));
    }
    Ok(probs)
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i as u32)
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
