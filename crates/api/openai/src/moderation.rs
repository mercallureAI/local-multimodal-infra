//! `POST /v1/moderations`: OpenAI's moderation API on local models.
//!
//! Text goes to a `text.moderate` model (Qwen3Guard). An image (a `data:`
//! URL; remote URLs are not fetched) goes to an `image.nsfw` classifier and,
//! unless `ocr` is false or the classifier flags it already, to `ocr.lines`,
//! whose text is moderated like the rest: a screenshot of a chat says what
//! its text says.
//!
//! As in OpenAI's API, `input` is a string (one result), an array of strings
//! (a result each) or an array of `{"type": "text", "text"}` /
//! `{"type": "image_url", "image_url": {"url"}}` parts (one result for them
//! all). Beyond OpenAI's fields, a result has `safety` (the least safe text's
//! probabilities, OCR text included) and `images` (per image: NSFW
//! probability, label scores, OCR characters and their safety); a result of
//! parts also has `parts`, each part's own verdict in input order, so a
//! caller can drop just the parts that were flagged.
//!
//! A text is flagged when `1 - p(safe)` exceeds `threshold` (default 0.9:
//! ~84% recall at ~2.8% false positives on ToxicChat, 0 of 600 technical
//! documents flagged), an image when its NSFW probability exceeds
//! `nsfw_threshold` (default 0.5). Category scores are the flagging scores of
//! the texts (or images, for `sexual`) that name the category.

use crate::{error_response, OpenAiApiState};
use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use base64::Engine as _;
use local_core::{InferenceInput, InferenceOutput, InferenceTask, TaskKind, TextModeration};
use local_error::{InfraError, Result};
use serde::{Deserialize, Serialize};
#[cfg(test)]
use serde_json::json;
use serde_json::Value;
use std::collections::BTreeMap;

pub const DEFAULT_THRESHOLD: f32 = 0.9;
pub const DEFAULT_NSFW_THRESHOLD: f32 = 0.5;
/// Images one request may carry.
const MAX_IMAGES: usize = 16;

/// Qwen3Guard's categories under short names; `sexual` also takes images.
const CATEGORIES: [(&str, &str); 9] = [
    ("Violent", "violent"),
    ("Non-violent Illegal Acts", "illegal"),
    ("Sexual Content or Sexual Acts", "sexual"),
    ("PII", "pii"),
    ("Suicide & Self-Harm", "self-harm"),
    ("Unethical Acts", "unethical"),
    ("Politically Sensitive Topics", "political"),
    ("Copyright Violation", "copyright"),
    ("Jailbreak", "jailbreak"),
];

#[derive(Debug, Clone, Deserialize)]
pub struct ModerationRequest {
    pub input: Value,
    /// The `text.moderate` model (default: any that does it).
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub nsfw_model: Option<String>,
    #[serde(default)]
    pub ocr_model: Option<String>,
    /// Read the text in images (default true).
    #[serde(default)]
    pub ocr: Option<bool>,
    #[serde(default)]
    pub threshold: Option<f32>,
    #[serde(default)]
    pub nsfw_threshold: Option<f32>,
    /// The classifier labels whose probabilities make up an image's NSFW
    /// probability (default: the model's `nsfw_labels`), e.g. `["medium",
    /// "high"]` to leave suggestive images (`low`) unflagged.
    #[serde(default)]
    pub nsfw_labels: Option<Vec<String>>,
    /// The text categories that flag (short names, e.g. `["violent",
    /// "sexual"]`; default all). A text judged unsafe only for others is not
    /// flagged; one judged unsafe without a category still is. Applies to the
    /// text read in images too, not to the NSFW classifier.
    #[serde(default)]
    pub categories: Option<Vec<String>>,
}

/// One result's inputs.
#[derive(Debug, Default)]
struct Item {
    texts: Vec<String>,
    images: Vec<Vec<u8>>,
    /// The parts in input order, for an array of parts (empty otherwise).
    order: Vec<PartKind>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PartKind {
    Text,
    Image,
}

#[derive(Debug, Serialize)]
pub struct ModerationResponse {
    pub id: String,
    pub model: String,
    pub results: Vec<ModerationResult>,
}

#[derive(Debug, Serialize)]
pub struct ModerationResult {
    pub flagged: bool,
    pub categories: BTreeMap<String, bool>,
    pub category_scores: BTreeMap<String, f32>,
    pub category_applied_input_types: BTreeMap<String, Vec<&'static str>>,
    /// The least safe text's verdict (OCR text included); absent without text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety: Option<Safety>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<ImageResult>,
    /// Each part's own verdict, in input order (an array of parts only).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub parts: Vec<PartResult>,
}

/// One input part's verdict.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PartResult {
    /// `text` or `image_url`, as in the input.
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub flagged: bool,
    /// The categories it was flagged for (`sexual` for an NSFW image).
    pub categories: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Safety {
    pub safe: f32,
    pub controversial: f32,
    #[serde(rename = "unsafe")]
    pub unsafe_: f32,
}

impl From<&TextModeration> for Safety {
    fn from(m: &TextModeration) -> Self {
        Self {
            safe: m.safe,
            controversial: m.controversial,
            unsafe_: m.unsafe_,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ImageResult {
    pub nsfw: f32,
    pub scores: BTreeMap<String, f32>,
    /// Characters of text read in the image (0 without OCR).
    pub ocr_chars: usize,
    /// The image was flagged as NSFW, so its text was not read.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub ocr_skipped: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ocr_safety: Option<Safety>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub ocr_categories: Vec<String>,
}

pub(crate) async fn moderations(
    State(state): State<OpenAiApiState>,
    Json(req): Json<ModerationRequest>,
) -> impl IntoResponse {
    match moderate(&state, req).await {
        Ok(response) => Json(response).into_response(),
        Err(InfraError::BadRequest(message)) => error_response(StatusCode::BAD_REQUEST, message),
        Err(InfraError::NotFound(message)) => error_response(StatusCode::NOT_FOUND, message),
        Err(err) => error_response(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()),
    }
}

pub async fn moderate(
    state: &OpenAiApiState,
    req: ModerationRequest,
) -> Result<ModerationResponse> {
    let threshold = unit(req.threshold, DEFAULT_THRESHOLD, "threshold")?;
    let nsfw_threshold = unit(req.nsfw_threshold, DEFAULT_NSFW_THRESHOLD, "nsfw_threshold")?;
    let items = parse_input(&req.input)?;
    let image_count: usize = items.iter().map(|item| item.images.len()).sum();
    if image_count > MAX_IMAGES {
        return Err(InfraError::BadRequest(format!(
            "at most {MAX_IMAGES} images per request"
        )));
    }
    let ocr = req.ocr.unwrap_or(true);
    let text_categories = match req.categories.as_deref() {
        Some(names) if !names.is_empty() => {
            if let Some(unknown) = names
                .iter()
                .find(|n| !CATEGORIES.iter().any(|(_, short)| short == n))
            {
                let known: Vec<&str> = CATEGORIES.iter().map(|(_, short)| *short).collect();
                return Err(InfraError::BadRequest(format!(
                    "categories: unknown category {unknown:?}; the categories are {known:?}"
                )));
            }
            Some(names.to_vec())
        }
        _ => None,
    };

    // Every text of every item in one task, and every image at once.
    let texts: Vec<String> = items.iter().flat_map(|item| item.texts.clone()).collect();
    let text_future = moderate_texts(state, &req.model, texts);
    let image_futures = futures_util::future::join_all(items.iter().flat_map(|item| {
        item.images
            .iter()
            .map(|image| inspect_image(state, &req, image, ocr, nsfw_threshold))
    }));
    let (text_results, inspected) = tokio::join!(text_future, image_futures);
    let mut text_results = text_results?.into_iter();

    // Per image: NSFW scores and the text read in it.
    let mut images = Vec::with_capacity(image_count);
    let mut ocr_texts = Vec::new();
    for inspected in inspected {
        let (image, text) = inspected?;
        if image.ocr_chars > 0 {
            ocr_texts.push(text);
        }
        images.push(image);
    }
    let mut ocr_results = moderate_texts(state, &req.model, ocr_texts)
        .await?
        .into_iter();
    for image in &mut images {
        if image.ocr_chars > 0 {
            if let Some(verdict) = ocr_results.next() {
                image.ocr_safety = Some(Safety::from(&verdict));
                image.ocr_categories = verdict.categories;
            }
        }
    }

    let mut images = images.into_iter();
    let mut results = Vec::with_capacity(items.len());
    for item in &items {
        let texts: Vec<TextModeration> = text_results.by_ref().take(item.texts.len()).collect();
        let item_images: Vec<ImageResult> = images.by_ref().take(item.images.len()).collect();
        results.push(combine(
            &texts,
            item_images,
            &item.order,
            threshold,
            nsfw_threshold,
            text_categories.as_deref(),
        ));
    }
    Ok(ModerationResponse {
        id: format!("modr-{}", uuid_like()),
        model: req.model.unwrap_or_else(|| "local-moderation".to_string()),
        results,
    })
}

/// An image's NSFW scores, then (unless that already flags it, or `ocr` is
/// off) the text read in it: an image flagged as NSFW is not read.
async fn inspect_image(
    state: &OpenAiApiState,
    req: &ModerationRequest,
    image: &[u8],
    ocr: bool,
    nsfw_threshold: f32,
) -> Result<(ImageResult, String)> {
    let one = |kind: TaskKind, model: &Option<String>| {
        state
            .service
            .dispatch_image_tasks(image.to_vec(), vec![(kind, model.clone())])
    };
    let (nsfw, scores) = match one(TaskKind::ImageNsfw, &req.nsfw_model).await?.pop() {
        Some(Ok(InferenceOutput::ImageNsfw { nsfw, scores })) => (
            nsfw,
            scores.into_iter().map(|s| (s.label, s.score)).collect(),
        ),
        Some(Err(err)) => return Err(err),
        other => {
            return Err(InfraError::Backend(format!(
                "image.nsfw answered {other:?}"
            )))
        }
    };
    let nsfw = match req.nsfw_labels.as_deref() {
        Some(labels) if !labels.is_empty() => nsfw_of(&scores, labels)?,
        _ => nsfw,
    };
    let ocr_skipped = ocr && nsfw > nsfw_threshold;
    let text = if !ocr || ocr_skipped {
        String::new()
    } else {
        match one(TaskKind::OcrLines, &req.ocr_model).await?.pop() {
            Some(Ok(InferenceOutput::OcrLines { lines })) => lines
                .iter()
                .map(|line| line.text.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
            Some(Err(err)) => return Err(err),
            other => return Err(InfraError::Backend(format!("ocr.lines answered {other:?}"))),
        }
    };
    let ocr_chars = if text.trim().is_empty() {
        0
    } else {
        text.chars().count()
    };
    Ok((
        ImageResult {
            nsfw,
            scores,
            ocr_chars,
            ocr_skipped,
            ocr_safety: None,
            ocr_categories: Vec::new(),
        },
        text,
    ))
}

async fn moderate_texts(
    state: &OpenAiApiState,
    model: &Option<String>,
    texts: Vec<String>,
) -> Result<Vec<TextModeration>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }
    let count = texts.len();
    let task = InferenceTask::new(
        TaskKind::TextModerate,
        model.clone(),
        InferenceInput::TextModerate { texts },
    );
    match state.service.dispatch_direct(task).await? {
        InferenceOutput::TextModerations { results } if results.len() == count => Ok(results),
        other => Err(InfraError::Backend(format!(
            "text.moderate answered {other:?} for {count} texts"
        ))),
    }
}

/// One result from its texts' verdicts (OCR verdicts are in `images`), with
/// each part's verdict when `order` lists the parts.
fn combine(
    texts: &[TextModeration],
    images: Vec<ImageResult>,
    order: &[PartKind],
    threshold: f32,
    nsfw_threshold: f32,
    text_categories: Option<&[String]>,
) -> ModerationResult {
    let mut categories = BTreeMap::new();
    let mut scores = BTreeMap::new();
    let mut applied: BTreeMap<String, Vec<&'static str>> = BTreeMap::new();
    for (_, name) in CATEGORIES {
        categories.insert(name.to_string(), false);
        scores.insert(name.to_string(), 0.0f32);
        applied.insert(name.to_string(), Vec::new());
    }
    let counts =
        |name: &str| text_categories.is_none_or(|allowed| allowed.iter().any(|a| a == name));
    // A text's flagged categories: None when it is not flagged.
    let text_hit = |verdict: Safety, named: &[String]| -> Option<Vec<String>> {
        if 1.0 - verdict.safe <= threshold {
            return None;
        }
        let names: Vec<String> = named.iter().map(|c| category_name(c)).collect();
        if names.is_empty() {
            // Unsafe, but the model named no category: still flagged.
            return Some(names);
        }
        let kept: Vec<String> = names.into_iter().filter(|n| counts(n)).collect();
        (!kept.is_empty()).then_some(kept)
    };
    let mut flagged = false;
    let mut least_safe: Option<Safety> = None;
    let mut judge = |verdict: Safety, named: &[String], kind: &'static str| {
        let score = 1.0 - verdict.safe;
        let hit = text_hit(verdict, named).is_some();
        flagged |= hit;
        if least_safe.is_none_or(|s| verdict.safe < s.safe) {
            least_safe = Some(verdict);
        }
        for category in named {
            let name = category_name(category);
            let entry = scores.entry(name.clone()).or_insert(0.0);
            *entry = entry.max(score);
            *categories.entry(name.clone()).or_insert(false) |= hit && counts(&name);
            let kinds = applied.entry(name).or_default();
            if !kinds.contains(&kind) {
                kinds.push(kind);
            }
        }
    };
    for text in texts {
        judge(Safety::from(text), &text.categories, "text");
    }
    // The text read in an image counts as the image's.
    for image in &images {
        if let Some(verdict) = image.ocr_safety {
            judge(verdict, &image.ocr_categories, "image");
        }
    }
    for image in &images {
        let hit = image.nsfw > nsfw_threshold;
        flagged |= hit;
        let entry = scores.entry("sexual".to_string()).or_insert(0.0);
        *entry = entry.max(image.nsfw);
        *categories.entry("sexual".to_string()).or_insert(false) |= hit;
        if hit {
            let kinds = applied.entry("sexual".to_string()).or_default();
            if !kinds.contains(&"image") {
                kinds.push("image");
            }
        }
    }
    let (mut next_text, mut next_image) = (texts.iter(), images.iter());
    let flagged_names = text_hit;
    let parts = order
        .iter()
        .filter_map(|kind| match kind {
            PartKind::Text => next_text.next().map(|text| {
                let names = flagged_names(Safety::from(text), &text.categories);
                PartResult {
                    kind: "text",
                    flagged: names.is_some(),
                    categories: names.unwrap_or_default(),
                }
            }),
            PartKind::Image => next_image.next().map(|image| {
                let mut names = image
                    .ocr_safety
                    .and_then(|verdict| flagged_names(verdict, &image.ocr_categories));
                if image.nsfw > nsfw_threshold {
                    let names = names.get_or_insert_with(Vec::new);
                    if !names.iter().any(|n| n == "sexual") {
                        names.insert(0, "sexual".to_string());
                    }
                }
                PartResult {
                    kind: "image_url",
                    flagged: names.is_some(),
                    categories: names.unwrap_or_default(),
                }
            }),
        })
        .collect();
    ModerationResult {
        flagged,
        categories,
        category_scores: scores,
        category_applied_input_types: applied,
        safety: least_safe,
        images,
        parts,
    }
}

/// The NSFW probability as the sum of `labels`' probabilities (each once).
fn nsfw_of(scores: &BTreeMap<String, f32>, labels: &[String]) -> Result<f32> {
    let mut seen: Vec<&str> = Vec::new();
    let mut sum = 0.0f32;
    for label in labels {
        let Some(score) = scores.get(label) else {
            let known: Vec<&str> = scores.keys().map(String::as_str).collect();
            return Err(InfraError::BadRequest(format!(
                "nsfw_labels: unknown label {label:?}; this model's labels are {known:?}"
            )));
        };
        if !seen.contains(&label.as_str()) {
            seen.push(label);
            sum += score;
        }
    }
    Ok(sum.min(1.0))
}

fn category_name(category: &str) -> String {
    CATEGORIES
        .iter()
        .find(|(model, _)| model.eq_ignore_ascii_case(category))
        .map(|(_, name)| name.to_string())
        .unwrap_or_else(|| category.to_ascii_lowercase().replace(' ', "-"))
}

fn unit(value: Option<f32>, default: f32, name: &str) -> Result<f32> {
    match value {
        None => Ok(default),
        Some(v) if (0.0..1.0).contains(&v) => Ok(v),
        Some(v) => Err(InfraError::BadRequest(format!(
            "{name} must be at least 0 and below 1, got {v}"
        ))),
    }
}

fn parse_input(input: &Value) -> Result<Vec<Item>> {
    let bad = |message: &str| InfraError::BadRequest(format!("input: {message}"));
    match input {
        Value::String(text) => Ok(vec![Item {
            texts: vec![non_empty(text)?],
            ..Item::default()
        }]),
        Value::Array(values) if values.is_empty() => Err(bad("must not be empty")),
        Value::Array(values) if values.iter().all(Value::is_string) => values
            .iter()
            .map(|v| {
                Ok(Item {
                    texts: vec![non_empty(v.as_str().unwrap_or_default())?],
                    ..Item::default()
                })
            })
            .collect(),
        Value::Array(parts) => {
            let mut item = Item::default();
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        let text = part
                            .get("text")
                            .and_then(Value::as_str)
                            .ok_or_else(|| bad("a text part needs `text`"))?;
                        item.texts.push(non_empty(text)?);
                        item.order.push(PartKind::Text);
                    }
                    Some("image_url") => {
                        let url = part
                            .get("image_url")
                            .and_then(|image| image.get("url").or(Some(image)))
                            .and_then(Value::as_str)
                            .ok_or_else(|| bad("an image_url part needs `image_url.url`"))?;
                        item.images.push(data_url_bytes(url)?);
                        item.order.push(PartKind::Image);
                    }
                    _ if part.is_string() => {
                        item.texts
                            .push(non_empty(part.as_str().unwrap_or_default())?);
                        item.order.push(PartKind::Text);
                    }
                    _ => return Err(bad("parts are {type: text} or {type: image_url}")),
                }
            }
            Ok(vec![item])
        }
        _ => Err(bad("a string, an array of strings or an array of parts")),
    }
}

fn non_empty(text: &str) -> Result<String> {
    if text.trim().is_empty() {
        Err(InfraError::BadRequest(
            "input texts must not be empty".to_string(),
        ))
    } else {
        Ok(text.to_string())
    }
}

/// The bytes of a base64 `data:` URL (remote URLs are not fetched).
fn data_url_bytes(url: &str) -> Result<Vec<u8>> {
    let rest = url.strip_prefix("data:").ok_or_else(|| {
        InfraError::BadRequest(
            "image_url must be a data: URL (remote images are not fetched)".to_string(),
        )
    })?;
    let (meta, data) = rest
        .split_once(',')
        .ok_or_else(|| InfraError::BadRequest("malformed data: URL".to_string()))?;
    if !meta.ends_with(";base64") {
        return Err(InfraError::BadRequest(
            "image data: URLs must be base64".to_string(),
        ));
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data.trim())
        .map_err(|err| InfraError::BadRequest(format!("image data: URL: {err}")))?;
    if bytes.is_empty() {
        return Err(InfraError::BadRequest("an image is empty".to_string()));
    }
    Ok(bytes)
}

fn uuid_like() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    format!(
        "{nanos:x}{:04x}",
        NEXT.fetch_add(1, Ordering::Relaxed) & 0xffff
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verdict(safe: f32, categories: &[&str]) -> TextModeration {
        TextModeration {
            safe,
            controversial: (1.0 - safe) / 2.0,
            unsafe_: (1.0 - safe) / 2.0,
            categories: categories.iter().map(|c| c.to_string()).collect(),
            tokens: 1,
            windows: 1,
        }
    }

    #[test]
    fn input_shapes_give_openai_result_counts() {
        assert_eq!(parse_input(&json!("hi")).unwrap().len(), 1);
        assert_eq!(parse_input(&json!(["a", "b"])).unwrap().len(), 2);
        let png = base64::engine::general_purpose::STANDARD.encode([0x89, b'P', b'N', b'G']);
        let items = parse_input(&json!([
            {"type": "text", "text": "a"},
            {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{png}")}},
        ]))
        .unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!((items[0].texts.len(), items[0].images.len()), (1, 1));
        assert_eq!(items[0].images[0], [0x89, b'P', b'N', b'G']);
        for bad in [
            json!([]),
            json!(""),
            json!(42),
            json!([{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]),
            json!([{"type": "image_url", "image_url": {"url": "data:image/png,raw"}}]),
            json!([{"type": "audio"}]),
        ] {
            assert!(parse_input(&bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn flags_follow_the_thresholds_and_name_categories() {
        let texts = [verdict(0.99, &[]), verdict(0.02, &["Violent", "PII"])];
        let result = combine(&texts, Vec::new(), &[], 0.9, 0.5, None);
        assert!(result.flagged);
        assert!(result.categories["violent"] && result.categories["pii"]);
        assert!(!result.categories["sexual"]);
        assert!((result.category_scores["violent"] - 0.98).abs() < 1e-6);
        assert_eq!(result.category_applied_input_types["pii"], vec!["text"]);
        assert_eq!(result.safety.unwrap().safe, 0.02);

        // Named but under the threshold: scored, not flagged.
        let result = combine(
            &[verdict(0.2, &["Unethical Acts"])],
            Vec::new(),
            &[],
            0.9,
            0.5,
            None,
        );
        assert!(!result.flagged && !result.categories["unethical"]);
        assert!((result.category_scores["unethical"] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn images_flag_sexual_and_their_text_counts() {
        let image = |nsfw: f32, ocr: Option<(f32, &[&str])>| ImageResult {
            nsfw,
            scores: BTreeMap::new(),
            ocr_chars: ocr.map_or(0, |_| 10),
            ocr_skipped: false,
            ocr_safety: ocr.map(|(safe, c)| Safety::from(&verdict(safe, c))),
            ocr_categories: ocr.map_or(Vec::new(), |(_, c)| {
                c.iter().map(|c| c.to_string()).collect()
            }),
        };
        let result = combine(&[], vec![image(0.97, None)], &[], 0.9, 0.5, None);
        assert!(result.flagged && result.categories["sexual"]);
        assert_eq!(result.category_applied_input_types["sexual"], vec!["image"]);
        assert!(result.safety.is_none());

        let result = combine(
            &[],
            vec![image(0.01, Some((0.01, &["PII"])))],
            &[],
            0.9,
            0.5,
            None,
        );
        assert!(result.flagged && result.categories["pii"] && !result.categories["sexual"]);
        assert_eq!(result.category_applied_input_types["pii"], vec!["image"]);

        let result = combine(
            &[],
            vec![image(0.3, Some((0.95, &[])))],
            &[],
            0.9,
            0.5,
            None,
        );
        assert!(!result.flagged);
        assert!(result.parts.is_empty(), "no parts listed, none reported");
    }

    #[test]
    fn parts_get_their_own_verdicts_in_input_order() {
        let image = |nsfw: f32, ocr: Option<(f32, &[&str])>| ImageResult {
            nsfw,
            scores: BTreeMap::new(),
            ocr_chars: ocr.map_or(0, |_| 10),
            ocr_skipped: false,
            ocr_safety: ocr.map(|(safe, c)| Safety::from(&verdict(safe, c))),
            ocr_categories: ocr.map_or(Vec::new(), |(_, c)| {
                c.iter().map(|c| c.to_string()).collect()
            }),
        };
        use PartKind::{Image, Text};
        let texts = [verdict(0.99, &[]), verdict(0.01, &["Violent"])];
        let images = vec![
            image(0.9, None),
            image(0.01, Some((0.02, &["PII"]))),
            image(0.01, Some((0.99, &[]))),
        ];
        let result = combine(
            &texts,
            images,
            &[Image, Text, Image, Text, Image],
            0.9,
            0.5,
            None,
        );
        let part = |kind, flagged, categories: &[&str]| PartResult {
            kind,
            flagged,
            categories: categories.iter().map(|c| c.to_string()).collect(),
        };
        assert!(result.flagged);
        assert_eq!(
            result.parts,
            vec![
                part("image_url", true, &["sexual"]),
                part("text", false, &[]),
                part("image_url", true, &["pii"]),
                part("text", true, &["violent"]),
                part("image_url", false, &[]),
            ]
        );
    }

    #[test]
    fn only_an_array_of_parts_lists_parts() {
        assert!(parse_input(&json!("hi")).unwrap()[0].order.is_empty());
        assert!(parse_input(&json!(["a", "b"])).unwrap()[1].order.is_empty());
        let items = parse_input(&json!([{"type": "text", "text": "a"}, "b"])).unwrap();
        assert_eq!(items[0].order, [PartKind::Text, PartKind::Text]);
    }

    #[test]
    fn only_the_chosen_text_categories_flag() {
        let allowed = ["violent".to_string()];
        let texts = [verdict(0.01, &["Politically Sensitive Topics"])];
        let result = combine(
            &texts,
            Vec::new(),
            &[PartKind::Text],
            0.9,
            0.5,
            Some(&allowed),
        );
        assert!(!result.flagged && !result.categories["political"]);
        assert!(!result.parts[0].flagged);

        let texts = [verdict(0.01, &["Violent", "Politically Sensitive Topics"])];
        let result = combine(
            &texts,
            Vec::new(),
            &[PartKind::Text],
            0.9,
            0.5,
            Some(&allowed),
        );
        assert!(result.flagged && result.categories["violent"]);
        assert!(!result.categories["political"]);
        assert_eq!(result.parts[0].categories, ["violent"]);

        // Unsafe without a category, and NSFW images, still flag.
        let result = combine(
            &[verdict(0.01, &[])],
            Vec::new(),
            &[],
            0.9,
            0.5,
            Some(&allowed),
        );
        assert!(result.flagged);
        let image = ImageResult {
            nsfw: 0.9,
            scores: BTreeMap::new(),
            ocr_chars: 0,
            ocr_skipped: true,
            ocr_safety: None,
            ocr_categories: Vec::new(),
        };
        let result = combine(
            &[],
            vec![image],
            &[PartKind::Image],
            0.9,
            0.5,
            Some(&allowed),
        );
        assert!(result.flagged && result.categories["sexual"]);
    }

    #[test]
    fn nsfw_labels_choose_what_counts() {
        let scores: BTreeMap<String, f32> = [
            ("neutral", 0.1),
            ("low", 0.6),
            ("medium", 0.2),
            ("high", 0.1),
        ]
        .into_iter()
        .map(|(l, s)| (l.to_string(), s))
        .collect();
        let labels = |l: &[&str]| l.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let explicit = nsfw_of(&scores, &labels(&["medium", "high"])).unwrap();
        assert!((explicit - 0.3).abs() < 1e-6);
        // A label named twice counts once.
        let twice = nsfw_of(&scores, &labels(&["low", "low"])).unwrap();
        assert!((twice - 0.6).abs() < 1e-6);
        assert!(nsfw_of(&scores, &labels(&["porn"])).is_err());
    }

    #[test]
    fn thresholds_must_be_probabilities() {
        assert_eq!(unit(None, 0.9, "t").unwrap(), 0.9);
        assert!(unit(Some(1.0), 0.9, "t").is_err());
        assert!(unit(Some(-0.1), 0.9, "t").is_err());
    }
}
