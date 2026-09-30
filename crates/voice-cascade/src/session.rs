//! One realtime voice conversation.
//!
//! Input audio runs through Silero VAD; each utterance is recognised
//! (SenseVoice) and given to the chat model (Qwen3), which in one streamed
//! pass answers, stays silent (`silence`) or hands a task to the client
//! (`backend_task`). Its text is spoken clause by clause (IndexTTS) while it
//! is still being generated, and played at real-time pace. Talking over the
//! bot stops its speech and its generation (a new epoch).
//!
//! In audio mode there is no chat model: utterances go to the client
//! (`input.transcript`), which streams back the text to speak
//! (`response.delta`); listening, speaking and being talked over work alike.

use crate::{
    history::History,
    prompt,
    protocol::{ClientEvent, ServerEvent, SessionConfig, SessionMode, INPUT_RATE, OUTPUT_RATE},
    text::{speakable, takes_floor, ClauseSplitter},
};
use base64::Engine;
use local_adapter_silero_vad::{SileroVad, VadEvent, VadIterator, WINDOW};
use local_core::{
    ArtifactKind, ChatMessage, ChatOptions, ChatToolCall, FileRef, InferenceEvent, InferenceInput,
    InferenceOutput, InferenceTask, ModelSpec, TaskKind,
};
use local_error::{InfraError, Result};
use local_runtime::RuntimeManager;
use serde_json::Value;
use std::{
    collections::{BTreeMap, VecDeque},
    future::Future,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};
use tokio::{
    sync::{mpsc, Notify},
    task::AbortHandle,
};

/// A pause shorter than this between two utterances (e.g. after "<name>,")
/// makes them one: the second continues the first.
const JOIN_SAMPLES: usize = INPUT_RATE as usize * 3 / 2;
/// In a group, a shorter utterance (a bare "<name>,") is continued by what
/// follows it; a longer one is not joined by what others say after it.
const CALL_SAMPLES: usize = INPUT_RATE as usize * 3 / 2;
/// Shorter speech is noise (a click, a cough cut short).
const MIN_UTTERANCE_SAMPLES: usize = INPUT_RATE as usize / 4;
/// Longer speech is recognised in pieces of this length, answered once it
/// ends.
const MAX_UTTERANCE_SAMPLES: usize = INPUT_RATE as usize * 15;
/// An utterance the model reads keeps at most its last this many characters.
const MAX_UTTERANCE_CHARS: usize = 600;
/// Backend words (a task's result, a note) the model reads are cut to this.
const MAX_RELAY_CHARS: usize = 1200;
const SPEECH_PAD_MS: usize = 30;
/// Speech is sent this far ahead of real time: enough to ride out a late
/// tick, little for a client to drop on a cut.
const PLAY_LEAD: Duration = Duration::from_millis(300);
const FRAME_SAMPLES: usize = OUTPUT_RATE as usize / 50; // 20 ms
/// A task handed off within this long of the last one gets no second filler.
const FILLER_GAP: Duration = Duration::from_secs(8);
/// A relay waits at most this long for the user to finish (one who talks on
/// and on still hears a task's result).
const RELAY_WAIT: Duration = Duration::from_secs(20);
/// Input silent this long (not even silence sent) ends any utterance under
/// way, for a relay.
const INPUT_STALE: Duration = Duration::from_secs(1);
/// A spoken answer is short; this also ends a generation that loops.
const MAX_REPLY_TOKENS: usize = 160;
const START_TIMEOUT: Duration = Duration::from_secs(30);
/// First use loads every model (IndexTTS takes tens of seconds).
const WARMUP_TIMEOUT: Duration = Duration::from_secs(300);
/// Audio: how often `state` and finished responses are checked.
const STATE_TICK: Duration = Duration::from_millis(50);
/// Audio: ids of responses that are over, remembered (their late text is
/// dropped).
const OVER_KEPT: usize = 256;

pub enum Inbound {
    Event(ClientEvent),
    /// 16-bit little-endian mono PCM at `INPUT_RATE`.
    Audio(Vec<u8>),
}

pub enum Outbound {
    Event(ServerEvent),
    /// 16-bit little-endian mono PCM at `OUTPUT_RATE`.
    Audio(Vec<u8>),
}

/// Emotions IndexTTS-2.5 takes in `emotion_vector` (IndexTTS 1.5 ignores it).
const TTS_EMOTIONS: [&str; 8] = [
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "melancholic",
    "surprised",
    "calm",
];
/// Weight of the configured emotion; the reference voice's own makes up the
/// rest.
const DEFAULT_EMOTION_STRENGTH: f64 = 0.8;

/// The models of a `voice_cascade` spec: its artifact is the Silero VAD
/// model; `metadata` names the chat, ASR and TTS models and may give a
/// `default_reference_audio` (a path the worker reads) and the emotion the
/// bot speaks with by default (`tts_emotion`, calm unless set, `none` for the
/// reference voice's own; `tts_emotion_strength`, 0 to 1), which a session
/// may override.
#[derive(Debug, Clone)]
pub struct CascadeModels {
    pub vad_model: PathBuf,
    pub chat_model: String,
    pub asr_model: String,
    pub tts_model: String,
    pub default_reference_audio: Option<PathBuf>,
    /// The spec's emotion settings (`tts_emotion`, `tts_emotion_strength`).
    pub tts_emotion: BTreeMap<String, Value>,
    /// Where a conversation keeps its short-lived audio files.
    pub temp_dir: PathBuf,
}

impl CascadeModels {
    /// The models of `spec`; audio files go below `data_dir`.
    pub fn from_spec(spec: &ModelSpec, data_dir: &Path) -> Result<Self> {
        let artifact = spec
            .artifacts
            .first()
            .ok_or_else(|| InfraError::ModelNotConfigured {
                model_id: spec.id.clone(),
                reason: "voice cascade has no VAD model artifact".to_string(),
            })?;
        let vad_model = if artifact.kind == ArtifactKind::Url || artifact.path.is_file() {
            artifact.path.clone()
        } else {
            artifact.path.join("silero_vad.onnx")
        };
        if !vad_model.is_file() {
            return Err(InfraError::ModelNotConfigured {
                model_id: spec.id.clone(),
                reason: format!("Silero VAD model {} is not downloaded", vad_model.display()),
            });
        }
        let text = |key: &str, default: &str| {
            spec.metadata
                .get(key)
                .and_then(Value::as_str)
                .unwrap_or(default)
                .to_string()
        };
        Ok(Self {
            vad_model,
            chat_model: text("chat_model", "qwen3-4b-instruct-2507-int4-onnx"),
            asr_model: text("asr_model", "sensevoice-small-onnx"),
            tts_model: text("tts_model", "indextts-1.5-onnx"),
            default_reference_audio: spec
                .metadata
                .get("default_reference_audio")
                .and_then(Value::as_str)
                .filter(|path| !path.is_empty())
                .map(PathBuf::from),
            tts_emotion: {
                let emotion: BTreeMap<String, Value> = spec
                    .metadata
                    .iter()
                    .filter(|(key, _)| key.starts_with("tts_emotion"))
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect();
                // Checked now: a typo fails the session, not every sentence.
                tts_params(&emotion).map_err(|reason| InfraError::ModelNotConfigured {
                    model_id: spec.id.clone(),
                    reason,
                })?;
                emotion
            },
            temp_dir: data_dir.join("voice-cascade"),
        })
    }
}

/// The TTS parameters the emotion settings (`tts_emotion`,
/// `tts_emotion_strength`) ask for; an error says what is wrong with them.
fn tts_params(
    settings: &BTreeMap<String, Value>,
) -> std::result::Result<BTreeMap<String, Value>, String> {
    let emotion = match settings.get("tts_emotion") {
        None | Some(Value::Null) => "calm",
        Some(Value::String(emotion)) => emotion.as_str(),
        Some(other) => return Err(format!("tts_emotion must be a string, got {other}")),
    };
    let mut params = BTreeMap::new();
    if emotion.is_empty() || emotion == "none" {
        return Ok(params);
    }
    if !TTS_EMOTIONS.contains(&emotion) {
        return Err(format!(
            "tts_emotion `{emotion}` is none of {TTS_EMOTIONS:?} (or `none`)"
        ));
    }
    let strength = match settings.get("tts_emotion_strength") {
        None | Some(Value::Null) => DEFAULT_EMOTION_STRENGTH,
        Some(value) => value
            .as_f64()
            .filter(|strength| (0.0..=1.0).contains(strength))
            .ok_or_else(|| format!("tts_emotion_strength must be 0 to 1, got {value}"))?,
    };
    params.insert(
        "emotion_vector".to_string(),
        serde_json::json!({ emotion: strength }),
    );
    Ok(params)
}

/// One count of a counter, given back when dropped (also when the task
/// holding it is aborted, even before it ran).
struct Counted(Arc<AtomicUsize>);

impl Counted {
    fn new(counter: &Arc<AtomicUsize>) -> Self {
        counter.fetch_add(1, Ordering::SeqCst);
        Self(counter.clone())
    }
}

impl Drop for Counted {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// A file removed when dropped (also when its task is aborted).
struct TempFile(PathBuf);

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// `samples` in a new WAV file below `dir`. The file is owned from the
/// start: it goes even when the caller is aborted during the write.
async fn wav_file(dir: &Path, samples: Vec<f32>) -> Result<TempFile> {
    let file = TempFile(dir.join(format!("{}-in.wav", uuid::Uuid::new_v4())));
    blocking(move || {
        write_wav(&file.0, &samples)?;
        Ok(file)
    })
    .await
}

async fn blocking<T: Send + 'static>(
    work: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<T> {
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|e| InfraError::Runtime(format!("blocking task failed: {e}")))?
}

/// Runs a conversation until the client stops it or goes away.
pub async fn run(
    runtime: Arc<RuntimeManager>,
    models: CascadeModels,
    mut inbound: mpsc::Receiver<Inbound>,
    out: mpsc::UnboundedSender<Outbound>,
) -> Result<()> {
    let config = match tokio::time::timeout(START_TIMEOUT, inbound.recv()).await {
        Ok(Some(Inbound::Event(ClientEvent::SessionStart { config }))) => *config,
        _ => {
            return Err(InfraError::BadRequest(
                "the first message must be session.start".to_string(),
            ))
        }
    };
    if config.name.trim().is_empty() {
        return Err(InfraError::BadRequest(
            "session.start config.name is required".to_string(),
        ));
    }
    let temp_dir = models.temp_dir.clone();
    let ref_bytes = match config.ref_audio.as_deref().filter(|s| !s.is_empty()) {
        Some(encoded) => Some(
            base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .map_err(|e| InfraError::BadRequest(format!("ref_audio is not base64: {e}")))?,
        ),
        None => None,
    };
    let own_ref = blocking(move || {
        std::fs::create_dir_all(&temp_dir)
            .map_err(|e| InfraError::io(Some(temp_dir.clone()), e))?;
        let Some(bytes) = ref_bytes else {
            return Ok(None);
        };
        let path = temp_dir.join(format!("{}-ref.wav", uuid::Uuid::new_v4()));
        std::fs::write(&path, bytes).map_err(|e| InfraError::io(Some(path.clone()), e))?;
        Ok(Some(TempFile(path)))
    })
    .await?;
    let ref_audio = match &own_ref {
        Some(file) => file.0.clone(),
        None => models.default_reference_audio.clone().ok_or_else(|| {
            InfraError::BadRequest(
                "ref_audio is required (the model has no default_reference_audio)".to_string(),
            )
        })?,
    };
    converse(runtime, models, config, ref_audio, inbound, out).await
}

async fn converse(
    runtime: Arc<RuntimeManager>,
    models: CascadeModels,
    config: SessionConfig,
    ref_audio: PathBuf,
    mut inbound: mpsc::Receiver<Inbound>,
    out: mpsc::UnboundedSender<Outbound>,
) -> Result<()> {
    // The session's emotion settings over the model's.
    let mut emotion = models.tts_emotion.clone();
    if let Some(name) = &config.tts_emotion {
        emotion.insert("tts_emotion".to_string(), Value::from(name.as_str()));
    }
    if let Some(strength) = config.tts_emotion_strength {
        emotion.insert("tts_emotion_strength".to_string(), Value::from(strength));
    }
    let tts_params = tts_params(&emotion).map_err(InfraError::BadRequest)?;
    let vad_path = models.vad_model.clone();
    let vad = blocking(move || SileroVad::load(&vad_path)).await?;
    let (speech_tx, speech_rx) = mpsc::unbounded_channel();
    let audio = config.mode == SessionMode::Audio;
    let shared = Arc::new(Shared {
        audio,
        responses: Mutex::new(Responses::default()),
        next_utterance: AtomicU64::new(0),
        runtime,
        system: prompt::system(&config),
        chat_model: config.chat_model.clone().unwrap_or(models.chat_model),
        asr_model: config.asr_model.clone().unwrap_or(models.asr_model),
        tts_model: config.tts_model.clone().unwrap_or(models.tts_model),
        tts_params,
        ref_audio,
        temp_dir: models.temp_dir,
        tool_filler: config.tool_filler.clone().unwrap_or_default(),
        out,
        epoch: AtomicU64::new(0),
        generating: AtomicBool::new(false),
        history: Mutex::new(History::default()),
        player: Player::default(),
        speech: speech_tx,
        replies: tokio::sync::Mutex::new(()),
        jobs: Mutex::new(Vec::new()),
        last_task: Mutex::new(None),
        speech_ended_at: Mutex::new(None),
        listening: AtomicUsize::new(0),
        replies_due: Arc::new(AtomicUsize::new(0)),
        clauses: AtomicUsize::new(0),
        told: AtomicU64::new(0),
        speak_due: AtomicU64::new(0),
        audio_at: Mutex::new(Instant::now()),
    });

    // Load every model before the conversation starts, and put the system
    // prompt in the chat model's prefix cache.
    let warm = async {
        let silence = wav_file(&shared.temp_dir, vec![0.0; INPUT_RATE as usize / 2]).await?;
        let asr = shared.recognize(&silence.0);
        // Audio mode has no chat model.
        let chat = async {
            if audio {
                return Ok(());
            }
            let task = shared.chat_task(vec![user_message("你好".to_string())], Steer::Free, 1);
            shared.runtime.infer(task).await.map(|_| ())
        };
        let (asr, chat, tts) = tokio::join!(asr, chat, shared.synthesize("你好。"));
        asr?;
        chat?;
        tts.map(|_| ())
    };
    tokio::time::timeout(WARMUP_TIMEOUT, warm)
        .await
        .map_err(|_| {
            InfraError::Runtime("voice cascade models took too long to load".to_string())
        })??;
    shared.emit(ServerEvent::SessionStarted {
        input_rate: INPUT_RATE,
        output_rate: OUTPUT_RATE,
    });

    let (asr_tx, asr_rx) = mpsc::unbounded_channel::<Utterance>();
    let (heard_tx, mut heard_rx) = mpsc::unbounded_channel::<Utterance>();
    let mut tasks = vec![
        tokio::spawn(play(shared.clone())),
        tokio::spawn(synthesize_clauses(shared.clone(), speech_rx)),
        tokio::spawn(recognize_utterances(shared.clone(), asr_rx, heard_tx)),
    ];
    if audio {
        tasks.push(tokio::spawn(report_state(shared.clone())));
    }
    let mut input = Input::new(vad, &config, asr_tx);
    let names: Option<Vec<String>> = config.group.then(|| {
        std::iter::once(config.name.clone())
            .chain(config.aliases.iter().cloned())
            .collect()
    });
    let mut listener = Listener::default();

    loop {
        tokio::select! {
            message = inbound.recv() => match message {
                None | Some(Inbound::Event(ClientEvent::SessionStop)) => break,
                Some(Inbound::Event(ClientEvent::SessionStart { .. })) => {
                    shared.emit(ServerEvent::Error { message: "session already started".to_string() });
                }
                Some(Inbound::Event(event)) if audio => shared.audio_event(event),
                Some(Inbound::Event(
                    ClientEvent::ResponseDelta { .. }
                    | ClientEvent::ResponseEnd { .. }
                    | ClientEvent::ResponseCancel { .. },
                )) => {
                    shared.emit(ServerEvent::Error { message: "response events need an audio mode session".to_string() });
                }
                Some(Inbound::Event(ClientEvent::ToolResult { call_id, output })) => {
                    // In the history at once, after its call; told when the
                    // bot is done speaking.
                    let message = ChatMessage {
                        role: "tool".to_string(),
                        content: Some(clip(&output, MAX_RELAY_CHARS)),
                        tool_call_id: Some(call_id.clone()),
                        ..ChatMessage::default()
                    };
                    let id = shared.history.lock().unwrap().insert_tool_result(&call_id, message);
                    shared.spawn(relay(shared.clone(), Steer::Speak, id));
                }
                Some(Inbound::Event(ClientEvent::Note { text })) => {
                    let note = prompt::NOTE.replace("{text}", &clip(&text, MAX_RELAY_CHARS));
                    let id = shared.history.lock().unwrap().push(user_message(note));
                    shared.spawn(relay(shared.clone(), Steer::SpeakOrSilence, id));
                }
                Some(Inbound::Event(ClientEvent::Say { text })) => {
                    let opening = prompt::OPENING.replace("{text}", &clip(&text, MAX_RELAY_CHARS));
                    let id = shared.history.lock().unwrap().push(user_message(opening));
                    shared.spawn(relay(shared.clone(), Steer::Speak, id));
                }
                Some(Inbound::Audio(bytes)) => {
                    *shared.audio_at.lock().unwrap() = Instant::now();
                    // The VAD runs ONNX Runtime: off the async threads' turn.
                    tokio::task::block_in_place(|| input.feed(&bytes, &shared, names.is_none()));
                }
            },
            Some(utterance) = heard_rx.recv() => {
                heard(&shared, utterance, names.as_deref(), &mut listener);
            }
        }
    }
    shared.close();
    for task in tasks {
        task.abort();
    }
    Ok(())
}

fn user_message(text: String) -> ChatMessage {
    ChatMessage {
        role: "user".to_string(),
        content: Some(text),
        ..ChatMessage::default()
    }
}

/// Input sample `at` in seconds.
fn seconds(at: usize) -> f64 {
    at as f64 / INPUT_RATE as f64
}

/// `text` cut to its first `max` characters.
fn clip(text: &str, max: usize) -> String {
    text.chars().take(max).collect()
}

/// Samples `start..end` of the input, from what `pcm` (starting at sample
/// `pcm_start`) still holds.
fn slice(pcm: &[f32], pcm_start: usize, start: usize, end: usize) -> Vec<f32> {
    let from = start.saturating_sub(pcm_start).min(pcm.len());
    let to = end.saturating_sub(pcm_start).min(pcm.len());
    pcm[from..to.max(from)].to_vec()
}

/// The input side: PCM in, utterances out to recognition.
struct Input {
    vad: SileroVad,
    iterator: VadIterator,
    asr: mpsc::UnboundedSender<Utterance>,
    barge_in: usize,
    /// A byte of a sample split between two messages.
    odd_byte: Option<u8>,
    /// Input not yet a whole VAD window.
    pending: Vec<f32>,
    /// Recent input, from sample `pcm_start`: what a speech start may reach
    /// back to, and the speech going on.
    pcm: Vec<f32>,
    pcm_start: usize,
    /// Where the speech going on (or its latest piece) starts.
    speech_start: Option<usize>,
    /// Where the utterance going on starts (before its pieces).
    utterance_start: usize,
    /// Pieces of the utterance going on were sent: its end is sent however
    /// short.
    pieces_sent: bool,
    barged: bool,
}

impl Input {
    fn new(vad: SileroVad, config: &SessionConfig, asr: mpsc::UnboundedSender<Utterance>) -> Self {
        Self {
            vad,
            iterator: VadIterator::new(
                config.vad_threshold.unwrap_or(0.5),
                config.min_silence_ms.unwrap_or(600),
                SPEECH_PAD_MS,
            ),
            asr,
            barge_in: config.barge_in_ms.unwrap_or(1200) * INPUT_RATE as usize / 1000,
            odd_byte: None,
            pending: Vec::new(),
            pcm: Vec::new(),
            pcm_start: 0,
            speech_start: None,
            utterance_start: 0,
            pieces_sent: false,
            barged: false,
        }
    }

    /// Takes `bytes` of input. One to one (`barge_in_by_voice`), speech over
    /// the bot's for long enough stops it.
    fn feed(&mut self, bytes: &[u8], shared: &Shared, barge_in_by_voice: bool) {
        let mut data = Vec::with_capacity(bytes.len() + 1);
        data.extend(self.odd_byte.take());
        data.extend_from_slice(bytes);
        if data.len() % 2 == 1 {
            self.odd_byte = data.pop();
        }
        self.pending.extend(
            data.chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0),
        );
        let keep = SPEECH_PAD_MS * INPUT_RATE as usize / 1000 + 2 * WINDOW;
        while self.pending.len() >= WINDOW {
            let window: Vec<f32> = self.pending.drain(..WINDOW).collect();
            self.pcm.extend_from_slice(&window);
            let probability = match self.vad.probability(&window) {
                Ok(probability) => probability,
                Err(err) => {
                    tracing::warn!(error = %err, "voice cascade VAD failed");
                    0.0
                }
            };
            let position = self.iterator.position() + WINDOW;
            match self.iterator.step(probability) {
                Some(VadEvent::Start(start)) => {
                    tracing::debug!(at = seconds(start), "voice cascade speech starts");
                    self.speech_start = Some(start);
                    self.utterance_start = start;
                    self.pieces_sent = false;
                    self.barged = false;
                    shared.listening.fetch_add(1, Ordering::SeqCst);
                    shared.emit(ServerEvent::SpeechStarted);
                }
                Some(VadEvent::End(end)) => {
                    tracing::debug!(at = seconds(end), "voice cascade speech ends");
                    let sent = match self.speech_start.take() {
                        Some(start) => self.send(start, end, false),
                        None => false,
                    };
                    if !sent {
                        // Noise: nothing for `heard` to take in.
                        shared.listening.fetch_sub(1, Ordering::SeqCst);
                    }
                    *shared.speech_ended_at.lock().unwrap() = Some(Instant::now());
                    shared.emit(ServerEvent::SpeechStopped);
                }
                None => {}
            }
            if let Some(start) = self.speech_start {
                if barge_in_by_voice
                    && !self.barged
                    && position - start >= self.barge_in
                    && shared.player.speaking()
                {
                    self.barged = true;
                    shared.cut();
                }
                if position - start >= MAX_UTTERANCE_SAMPLES {
                    // A piece of a long utterance: recognised now, answered
                    // with the rest when it ends.
                    self.send(start, position, true);
                    self.speech_start = Some(position);
                }
            }
            // Keep what a speech start may reach back to, and the speech.
            let from = self
                .speech_start
                .unwrap_or(position)
                .min(position.saturating_sub(keep));
            if from > self.pcm_start {
                let drop = (from - self.pcm_start).min(self.pcm.len());
                self.pcm.drain(..drop);
                self.pcm_start += drop;
            }
        }
    }

    /// Sends input `from..to`: a piece (`continues`) or the end of an
    /// utterance. False when it is not sent (too short: noise).
    fn send(&mut self, from: usize, to: usize, continues: bool) -> bool {
        let samples = slice(&self.pcm, self.pcm_start, from, to);
        let after_pieces = std::mem::replace(&mut self.pieces_sent, continues);
        if !(continues || after_pieces || samples.len() >= MIN_UTTERANCE_SAMPLES) {
            return false;
        }
        let _ = self.asr.send(Utterance {
            start: self.utterance_start,
            end: to,
            samples,
            text: String::new(),
            continues,
            barged: self.barged,
        });
        true
    }
}

struct Utterance {
    /// Where the whole utterance starts (a piece's own samples start later).
    start: usize,
    end: usize,
    samples: Vec<f32>,
    text: String,
    /// A piece of a long utterance that goes on.
    continues: bool,
    /// It stopped the bot (one to one, by talking over it long enough).
    barged: bool,
}

#[derive(Default)]
struct Listener {
    /// Pieces of a long utterance still going on.
    held: String,
    last: Option<LastUtterance>,
}

struct LastUtterance {
    end: usize,
    text: String,
    /// A short call (a bare "<name>,") that what follows continues.
    short: bool,
    /// Its message in the history.
    message: u64,
    /// A reply to it was asked for.
    answered: bool,
}

/// `a` and `b` as one text (a space between words of a spaced script).
fn join_text(a: &str, b: &str) -> String {
    let spaced = a.ends_with(|c: char| c.is_ascii_alphanumeric())
        && b.starts_with(|c: char| c.is_ascii_alphanumeric());
    format!("{a}{}{b}", if spaced { " " } else { "" })
}

/// An utterance was recognised: the chat model decides what to do with it.
fn heard(
    shared: &Arc<Shared>,
    utterance: Utterance,
    names: Option<&[String]>,
    listener: &mut Listener,
) {
    let piece = utterance.text.trim();
    if utterance.continues {
        if !piece.is_empty() {
            listener.held = join_text(&listener.held, piece);
            shared.emit(ServerEvent::Transcript {
                text: piece.to_string(),
                partial: true,
                id: None,
                replaces: None,
                respond: None,
            });
        }
        return;
    }
    // Taken in once this returns (its reply, if any, is due by then).
    struct TakenIn<'a>(&'a AtomicUsize);
    impl Drop for TakenIn<'_> {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }
    let _taken_in = TakenIn(&shared.listening);
    let held = std::mem::take(&mut listener.held);
    let mut text = join_text(&held, piece);
    if text.is_empty() {
        return;
    }
    // The utterance continues the last one (its message, and whether a
    // reply to it was asked for).
    let mut joined = None;
    if let Some(previous) = listener.last.as_ref() {
        // One to one, the speaker only paused. In a group, voices are not
        // told apart: only a short call (a bare "<name>,") is continued;
        // what follows a longer utterance is likely someone else.
        let continues = names.is_none() || previous.short;
        if continues && utterance.start.saturating_sub(previous.end) < JOIN_SAMPLES {
            // The utterance replaces the last one.
            text = join_text(&previous.text, &text);
            joined = Some((previous.message, previous.answered));
        }
    }
    let skip = text.chars().count().saturating_sub(MAX_UTTERANCE_CHARS);
    let text: String = text.chars().skip(skip).collect();
    let busy =
        shared.player.speaking() || shared.generating.load(Ordering::SeqCst) || shared.responding();
    let restarts = matches!(joined, Some((_, true)));
    // A barge-in is answered: it cut the bot, and whatever started since.
    let answer = restarts || utterance.barged || !busy || takes_floor(&text, names);
    tracing::info!(
        text,
        joined = joined.is_some(),
        answer,
        start = seconds(utterance.start),
        end = seconds(utterance.end),
        "voice cascade heard"
    );
    let id = shared.next_utterance.fetch_add(1, Ordering::SeqCst);
    shared.emit(ServerEvent::Transcript {
        text: text.clone(),
        partial: false,
        id: shared.audio.then_some(id),
        replaces: joined
            .filter(|_| shared.audio)
            .map(|(replaced, _)| replaced),
        respond: shared.audio.then_some(answer),
    });
    if !answer && names.is_none() {
        // A backchannel while the bot speaks.
        return;
    }
    // In audio mode the client replies: a joined utterance stops only what is
    // actually going on.
    if answer && (busy || (restarts && !shared.audio)) {
        // Whatever was made of the utterance's first part goes too.
        shared.cut();
    }
    if shared.audio {
        // The client keeps the conversation: the utterance is its message.
        listener.last = Some(LastUtterance {
            end: utterance.end,
            text: text.clone(),
            short: joined.is_none() && utterance.end - utterance.start < CALL_SAMPLES,
            message: id,
            answered: answer,
        });
        return;
    }
    // In a group, talk among others while the bot speaks is kept as context.
    let message = {
        let mut history = shared.history.lock().unwrap();
        if let Some((replaced, _)) = joined {
            history.remove_utterance(replaced);
        }
        history.push(user_message(text.clone()))
    };
    listener.last = Some(LastUtterance {
        end: utterance.end,
        text: text.clone(),
        short: joined.is_none() && utterance.end - utterance.start < CALL_SAMPLES,
        message,
        answered: answer,
    });
    if answer {
        let due = Counted::new(&shared.replies_due);
        let epoch = shared.current_epoch();
        let shared_ = shared.clone();
        shared.spawn(async move {
            let _due = due;
            let _turn = shared_.replies.lock().await;
            reply(shared_.clone(), Steer::Free, Some((message, text)), epoch).await;
        });
    }
}

/// What a turn may do. Every turn offers the same tools (so the prompt's
/// prefix, cached by the chat model, stays the same); a turn that must not
/// call them is steered by logit biases.
#[derive(Clone, Copy, PartialEq)]
enum Steer {
    /// Answer, stay silent or hand off (an utterance).
    Free,
    /// Speak (a task's result, an opening): no tool call.
    Speak,
    /// Speak or stay silent, no hand-off (a note).
    SpeakOrSilence,
}

/// Has the chat model tell backend words (already in the history) once the
/// bot is done speaking, no utterance is under way and the replies to
/// utterances are done (so they answer them, with their tools), waiting at
/// most `RELAY_WAIT`. A cut meanwhile, even by noise, does not drop them.
///
/// `message` is the history id of those words: when a relay read the history
/// with it already, it has been told and this relay ends.
fn relay(shared: Arc<Shared>, steer: Steer, message: u64) -> impl Future<Output = ()> + Send {
    // At once, not when the job first runs: another relay may tell it first.
    if steer == Steer::Speak {
        shared.speak_due.fetch_max(message + 1, Ordering::SeqCst);
    }
    async move {
        let since = Instant::now();
        let idle = || {
            let input_stopped = shared.audio_at.lock().unwrap().elapsed() > INPUT_STALE;
            !shared.player.speaking()
                && (shared.listening.load(Ordering::SeqCst) == 0 || input_stopped)
                && shared.replies_due.load(Ordering::SeqCst) == 0
                // A reply's last clauses may still be on their way to the player.
                && !shared.generating.load(Ordering::SeqCst)
                && shared.clauses.load(Ordering::SeqCst) == 0
        };
        loop {
            shared.player.drained().await;
            if shared.player.is_closed() {
                return;
            }
            let late = since.elapsed() > RELAY_WAIT;
            if idle() || late {
                let turn = shared.replies.lock().await;
                // A reply may have spoken or become due while this waited.
                if idle() || late {
                    let told = shared.told.load(Ordering::SeqCst);
                    if message < told {
                        return; // an earlier relay told it
                    }
                    // This turn tells everything since the last relay (`reply`
                    // marks it told): if that includes words that must be
                    // spoken, it may not stay silent.
                    let steer = if shared.speak_due.load(Ordering::SeqCst) > told {
                        Steer::Speak
                    } else {
                        steer
                    };
                    let epoch = shared.current_epoch();
                    reply(shared.clone(), steer, None, epoch).await;
                    drop(turn);
                    return;
                }
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
}

/// One turn of the chat model (the caller holds the replies lock): its text
/// is spoken as it comes, its tool call acted on. `utterance` is what it
/// answers (its message and text), none for a relay. `epoch` is the
/// conversation's when the turn was asked for: a cut since then cancels it.
async fn reply(shared: Arc<Shared>, steer: Steer, utterance: Option<(u64, String)>, epoch: u64) {
    if shared.current_epoch() != epoch || shared.player.is_closed() {
        return;
    }
    let (answers, heard) = match utterance {
        Some((message, text)) => (Some(message), text),
        None => (None, String::new()),
    };
    let (anchor, messages) = {
        let history = shared.history.lock().unwrap();
        if answers.is_none() {
            // A relay: backend words read now are told by this turn.
            shared.told.store(history.next_id(), Ordering::SeqCst);
        }
        (history.last_id(), history.messages())
    };
    shared.generating.store(true, Ordering::SeqCst);
    let (events_tx, mut events) = mpsc::channel(64);
    let runtime = shared.runtime.clone();
    let task = shared.chat_task(messages, steer, MAX_REPLY_TOKENS);
    let inference = tokio::spawn(async move { runtime.infer_streaming(task, events_tx).await });
    let mut splitter = ClauseSplitter::default();
    let mut spoken = String::new();
    let mut cut = false;
    while let Some(event) = events.recv().await {
        if shared.current_epoch() != epoch {
            cut = true;
            break;
        }
        if let InferenceEvent::ChatDelta { content } = event {
            for clause in splitter.feed(&content) {
                if shared.speak(epoch, &clause, None) {
                    spoken.push_str(&clause);
                }
            }
        }
    }
    // Dropping the receiver stops a generation that was cut.
    drop(events);
    let output = inference.await;
    shared.generating.store(false, Ordering::SeqCst);
    let completion = match output {
        _ if cut => None,
        Ok(Ok(InferenceOutput::ChatCompletion {
            content,
            tool_calls,
            ..
        })) => Some((content, tool_calls)),
        Ok(Ok(other)) => {
            tracing::warn!(?other, "voice cascade chat returned no completion");
            None
        }
        Ok(Err(err)) => {
            tracing::warn!(error = %err, "voice cascade chat failed");
            shared.emit(ServerEvent::Error {
                message: format!("chat model failed: {err}"),
            });
            None
        }
        Err(err) => {
            tracing::warn!(error = %err, "voice cascade chat task failed");
            None
        }
    };
    if completion.is_some() {
        if let Some(rest) = splitter.flush() {
            if shared.speak(epoch, &rest, None) {
                spoken.push_str(&rest);
            }
        }
    }
    let assistant = |content: String, tool_calls: Vec<ChatToolCall>| ChatMessage {
        role: "assistant".to_string(),
        content: Some(content),
        tool_calls,
        ..ChatMessage::default()
    };
    // Under the history lock: a cut from now on comes after the turn.
    let tool_calls = {
        let mut history = shared.history.lock().unwrap();
        match completion {
            Some((content, tool_calls)) if shared.current_epoch() == epoch => {
                let tools: Vec<&str> = tool_calls.iter().map(|call| call.name.as_str()).collect();
                tracing::info!(content, ?tools, "voice cascade replied");
                if !history.insert_after(anchor, answers, assistant(content, tool_calls.clone())) {
                    return; // what it answered was replaced meanwhile
                }
                tool_calls
            }
            _ => {
                // Cut or failed: what was said stays, after what it answered.
                if !spoken.is_empty() {
                    history.insert_after(anchor, answers, assistant(spoken, Vec::new()));
                }
                return;
            }
        }
    };
    for call in tool_calls {
        match call.name.as_str() {
            "silence" => {}
            "backend_task" => {
                let arguments: Value = serde_json::from_str(&call.arguments)
                    .unwrap_or(Value::Object(Default::default()));
                let now = Instant::now();
                let mut last_task = shared.last_task.lock().unwrap();
                if !shared.tool_filler.is_empty()
                    && last_task.is_none_or(|at| now.duration_since(at) > FILLER_GAP)
                {
                    shared.speak(epoch, &shared.tool_filler, None);
                }
                *last_task = Some(now);
                shared.emit(ServerEvent::ToolCall {
                    call_id: call.id,
                    name: call.name,
                    arguments,
                    heard: heard.clone(),
                });
            }
            other => tracing::warn!(tool = other, "voice cascade: unknown tool called"),
        }
    }
}

struct Shared {
    /// Audio mode: the client runs the conversation.
    audio: bool,
    /// Audio: the responses being spoken.
    responses: Mutex<Responses>,
    /// Audio: the id of the next whole utterance.
    next_utterance: AtomicU64,
    runtime: Arc<RuntimeManager>,
    system: String,
    chat_model: String,
    asr_model: String,
    tts_model: String,
    tts_params: BTreeMap<String, Value>,
    ref_audio: PathBuf,
    temp_dir: PathBuf,
    tool_filler: String,
    out: mpsc::UnboundedSender<Outbound>,
    /// Bumped by a cut: speech and generations of an older epoch stop.
    epoch: AtomicU64,
    generating: AtomicBool,
    history: Mutex<History>,
    player: Player,
    /// Clauses to speak, with their epoch (and response, in audio mode).
    speech: mpsc::UnboundedSender<Clause>,
    /// Replies run one at a time, in order.
    replies: tokio::sync::Mutex<()>,
    /// Replies and relays running or waiting, stopped with the conversation.
    jobs: Mutex<Vec<AbortHandle>>,
    last_task: Mutex<Option<Instant>>,
    /// When the last utterance ended, until its answer starts playing.
    speech_ended_at: Mutex<Option<Instant>>,
    /// Utterances begun and not yet taken in by `heard` (spoken, or being
    /// recognised): a relay waits for them.
    listening: AtomicUsize,
    /// Replies to utterances asked for and not done: a relay goes after them.
    replies_due: Arc<AtomicUsize>,
    /// Clauses given to TTS and not yet queued to play (or dropped).
    clauses: AtomicUsize,
    /// History ids below this were in the history a relay last read: a
    /// relay for one of them has nothing new to tell.
    told: AtomicU64,
    /// One past the latest history id whose relay must speak (a task's
    /// result, an opening): a relay that tells it must not stay silent.
    speak_due: AtomicU64,
    /// When input audio last arrived (a client that stops sending mid-speech
    /// leaves an utterance begun forever).
    audio_at: Mutex<Instant>,
}

impl Shared {
    fn emit(&self, event: ServerEvent) {
        let _ = self.out.send(Outbound::Event(event));
    }

    fn current_epoch(&self) -> u64 {
        self.epoch.load(Ordering::SeqCst)
    }

    fn spawn(&self, job: impl Future<Output = ()> + Send + 'static) {
        let handle = tokio::spawn(job).abort_handle();
        let mut jobs = self.jobs.lock().unwrap();
        jobs.retain(|job| !job.is_finished());
        jobs.push(handle);
    }

    /// Stops the bot: its speech, what is queued and the generation (in
    /// audio mode, every response being spoken, each told as done).
    fn cut(&self) {
        let mut responses = self.responses.lock().unwrap();
        let events = self.cut_locked(&mut responses);
        drop(responses);
        for event in events {
            self.emit(event);
        }
    }

    /// `cut` with the responses already locked; returns the events to send
    /// once the lock is released.
    fn cut_locked(&self, responses: &mut Responses) -> Vec<ServerEvent> {
        let mut spoken = Vec::new();
        self.player.clear_with(|state| {
            self.epoch.fetch_add(1, Ordering::SeqCst);
            let heard = state.heard();
            for response in &responses.open {
                let text: String = state
                    .segments
                    .iter()
                    .filter(|segment| segment.response == response.id && segment.start < heard)
                    .map(|segment| segment.text.as_str())
                    .collect();
                spoken.push((response.id.clone(), text));
            }
        });
        // The one speaking (the others wait behind it).
        let current = responses.open.first().map(|response| response.id.clone());
        let cut: Vec<String> = responses.open.drain(..).map(|r| r.id).collect();
        for id in cut {
            responses.close(id);
        }
        let mut events = vec![ServerEvent::ResponseCut {
            response_id: current,
        }];
        events.extend(
            spoken
                .into_iter()
                .map(|(response_id, spoken)| ServerEvent::ResponseDone {
                    response_id,
                    spoken,
                    cut: true,
                }),
        );
        events
    }

    /// Ends the conversation: replies and relays stop, speech is dropped.
    fn close(&self) {
        self.player.close(|_| {
            self.epoch.fetch_add(1, Ordering::SeqCst);
        });
        for job in self.jobs.lock().unwrap().drain(..) {
            job.abort();
        }
    }

    /// Queues `clause` to be spoken (as part of `response`, in audio mode);
    /// false when it is not (nothing to say, or cut).
    fn speak(&self, epoch: u64, clause: &str, response: Option<&str>) -> bool {
        let text = speakable(clause);
        if text.is_empty() || epoch != self.current_epoch() {
            return false;
        }
        self.emit(ServerEvent::ResponseText {
            text: text.clone(),
            response_id: response.map(str::to_string),
        });
        self.clauses.fetch_add(1, Ordering::SeqCst);
        let _ = self.speech.send(Clause {
            epoch,
            text,
            response: response.map(str::to_string),
        });
        true
    }

    /// Audio: a response is being spoken (or its text is still coming).
    fn responding(&self) -> bool {
        !self.responses.lock().unwrap().open.is_empty()
    }

    /// Audio: a client event (other than `session.*`).
    fn audio_event(&self, event: ClientEvent) {
        match event {
            ClientEvent::ResponseDelta { response_id, text } => {
                self.response_text(&response_id, &text, false);
            }
            ClientEvent::ResponseEnd { response_id } => {
                self.response_text(&response_id, "", true);
            }
            ClientEvent::ResponseCancel { response_id } => {
                let mut responses = self.responses.lock().unwrap();
                let events = match response_id {
                    None if responses.open.is_empty() => Vec::new(),
                    None => self.cut_locked(&mut responses),
                    Some(id) => self.cancel_locked(&mut responses, id),
                };
                drop(responses);
                for event in events {
                    self.emit(event);
                }
            }
            ClientEvent::Say { text } => {
                let id = format!("say-{}", uuid::Uuid::new_v4());
                self.response_text(&id, &text, false);
                self.response_text(&id, "", true);
            }
            ClientEvent::ToolResult { .. } | ClientEvent::Note { .. } => {
                self.emit(ServerEvent::Error {
                    message: "tool.result and note are for cascade mode sessions".to_string(),
                });
            }
            ClientEvent::SessionStart { .. } | ClientEvent::SessionStop => {}
        }
    }

    /// Audio: stops the response `id`. The one speaking (or one whose
    /// speech is on its way) stops the bot, with what waits behind it;
    /// one still waiting its turn just goes; one not started yet never
    /// starts. Returns the events to send.
    fn cancel_locked(&self, responses: &mut Responses, id: String) -> Vec<ServerEvent> {
        if responses.is_over(&id) {
            return Vec::new();
        }
        let Some(at) = responses.open.iter().position(|r| r.id == id) else {
            responses.close(id.clone());
            return vec![ServerEvent::ResponseDone {
                response_id: id,
                spoken: String::new(),
                cut: true,
            }];
        };
        if at == 0 || responses.open[at].pending > 0 || !responses.open[at].spoken.is_empty() {
            return self.cut_locked(responses);
        }
        responses.open.remove(at);
        responses.close(id.clone());
        vec![ServerEvent::ResponseDone {
            response_id: id,
            spoken: String::new(),
            cut: true,
        }]
    }

    /// Audio: more text of `response_id` (its end, when `end`).
    fn response_text(&self, response_id: &str, text: &str, end: bool) {
        {
            let mut responses = self.responses.lock().unwrap();
            if responses.is_over(response_id) {
                return;
            }
            let at = match responses.open.iter().position(|r| r.id == response_id) {
                Some(at) => at,
                None if end => {
                    // Nothing to say: it is over at once.
                    responses.close(response_id.to_string());
                    drop(responses);
                    self.emit(ServerEvent::ResponseDone {
                        response_id: response_id.to_string(),
                        spoken: String::new(),
                        cut: false,
                    });
                    return;
                }
                None => {
                    responses.open.push(OpenResponse::new(response_id));
                    responses.open.len() - 1
                }
            };
            let response = &mut responses.open[at];
            if response.ended {
                return;
            }
            let clauses = response.splitter.feed(text);
            response.held.extend(clauses);
            if end {
                response.held.extend(response.splitter.flush());
                response.ended = true;
            }
            self.release(&mut responses);
        }
        if end {
            self.check_done();
        }
    }

    /// Audio: gives TTS the clauses whose turn it is: a response speaks once
    /// every earlier one has all its text (so responses play one after the
    /// other, not interleaved).
    fn release(&self, responses: &mut Responses) {
        let epoch = self.current_epoch();
        for response in responses.open.iter_mut() {
            for clause in std::mem::take(&mut response.held) {
                if self.speak(epoch, &clause, Some(&response.id)) {
                    response.pending += 1;
                }
            }
            if !response.ended {
                break;
            }
        }
    }

    /// Audio: a clause of `response_id` left TTS; `spoken` when it was
    /// queued to play.
    fn clause_done(&self, response_id: &str, spoken: Option<&str>) {
        {
            let mut responses = self.responses.lock().unwrap();
            if let Some(response) = responses.open.iter_mut().find(|r| r.id == response_id) {
                response.pending = response.pending.saturating_sub(1);
                if let Some(text) = spoken {
                    response.spoken.push_str(text);
                }
            }
        }
        self.check_done();
    }

    /// Audio: tells the responses whose text has ended and whose speech has
    /// been heard to the end.
    fn check_done(&self) {
        if !self.audio {
            return;
        }
        let mut done = Vec::new();
        {
            let mut responses = self.responses.lock().unwrap();
            let mut state = self.player.state.lock().unwrap();
            let heard = state.heard();
            responses.open.retain_mut(|response| {
                let finished = response.ended
                    && response.held.is_empty()
                    && response.pending == 0
                    && !state
                        .segments
                        .iter()
                        .any(|segment| segment.response == response.id && segment.end > heard);
                if finished {
                    done.push((response.id.clone(), std::mem::take(&mut response.spoken)));
                }
                !finished
            });
            state
                .segments
                .retain(|segment| !done.iter().any(|(id, _)| *id == segment.response));
            for (id, _) in &done {
                responses.close(id.clone());
            }
        }
        for (response_id, spoken) in done {
            self.emit(ServerEvent::ResponseDone {
                response_id,
                spoken,
                cut: false,
            });
        }
    }

    fn chat_task(
        &self,
        history: Vec<ChatMessage>,
        steer: Steer,
        max_tokens: usize,
    ) -> InferenceTask {
        let mut messages = Vec::with_capacity(history.len() + 1);
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: Some(self.system.clone()),
            ..ChatMessage::default()
        });
        messages.extend(history);
        InferenceTask::new(
            TaskKind::ChatComplete,
            Some(self.chat_model.clone()),
            InferenceInput::ChatComplete {
                messages,
                tools: vec![prompt::silence_tool(), prompt::backend_task_tool()],
                options: ChatOptions {
                    max_tokens: Some(max_tokens),
                    // Greedy: whether to speak is decided by the first token.
                    temperature: Some(0.0),
                    // Spoken answers loop easily (a joke retold); repeats cost.
                    presence_penalty: Some(1.0),
                    tool_call_bias: (steer == Steer::Speak).then_some(-100.0),
                    tool_bias: match steer {
                        Steer::SpeakOrSilence => [("backend_task".to_string(), -100.0)].into(),
                        _ => Default::default(),
                    },
                    ..ChatOptions::default()
                },
            },
        )
    }

    async fn recognize(&self, wav: &Path) -> Result<String> {
        let mut task = InferenceTask::new(
            TaskKind::AsrTranscribe,
            Some(self.asr_model.clone()),
            InferenceInput::AsrTranscribe {
                audio: FileRef::local(wav),
            },
        );
        task.params
            .insert("timestamps".to_string(), Value::Bool(false));
        task.params
            .insert("speaker_diarization".to_string(), Value::Bool(false));
        match self.runtime.infer(task).await? {
            InferenceOutput::AsrTranscription { text, .. } => Ok(text),
            other => Err(InfraError::Runtime(format!("ASR returned {other:?}"))),
        }
    }

    async fn synthesize(&self, text: &str) -> Result<Vec<f32>> {
        let mut task = InferenceTask::new(
            TaskKind::TtsSynthesize,
            Some(self.tts_model.clone()),
            InferenceInput::TtsSynthesize {
                text: text.to_string(),
                reference_audio: Some(FileRef::local(&self.ref_audio)),
            },
        );
        task.params.extend(self.tts_params.clone());
        let runtime = self.runtime.clone();
        // In a task of its own: the audio file goes even when the caller is
        // aborted.
        tokio::spawn(async move {
            let InferenceOutput::TtsAudio { audio } = runtime.infer(task).await? else {
                return Err(InfraError::Runtime("TTS returned no audio".to_string()));
            };
            let file = TempFile(
                audio
                    .path
                    .ok_or_else(|| InfraError::Runtime("TTS audio has no path".to_string()))?,
            );
            blocking(move || read_wav(&file.0)).await
        })
        .await
        .map_err(|e| InfraError::Runtime(format!("TTS task failed: {e}")))?
    }
}

async fn recognize_utterances(
    shared: Arc<Shared>,
    mut utterances: mpsc::UnboundedReceiver<Utterance>,
    heard: mpsc::UnboundedSender<Utterance>,
) {
    while let Some(mut utterance) = utterances.recv().await {
        let samples = std::mem::take(&mut utterance.samples);
        if samples.len() < MIN_UTTERANCE_SAMPLES {
            // The short end of a long utterance: nothing to recognise, but
            // it ends the utterance.
            let _ = heard.send(utterance);
            continue;
        }
        let result = match wav_file(&shared.temp_dir, samples).await {
            Ok(file) => shared.recognize(&file.0).await,
            Err(err) => Err(err),
        };
        match result {
            Ok(text) => {
                utterance.text = text;
                let _ = heard.send(utterance);
            }
            Err(err) => {
                tracing::warn!(error = %err, "voice cascade ASR failed");
                if !utterance.continues {
                    // The pieces held so far still get their answer.
                    let _ = heard.send(utterance);
                }
            }
        }
    }
}

/// A clause to speak.
struct Clause {
    /// The conversation's epoch when it was asked for: a cut since drops it.
    epoch: u64,
    text: String,
    /// Audio: the response it belongs to.
    response: Option<String>,
}

async fn synthesize_clauses(shared: Arc<Shared>, mut clauses: mpsc::UnboundedReceiver<Clause>) {
    while let Some(Clause {
        epoch,
        text,
        response,
    }) = clauses.recv().await
    {
        let mut queued = false;
        if epoch == shared.current_epoch() {
            match shared.synthesize(&text).await {
                Ok(samples) => {
                    let segment = response.clone().map(|response| (response, text.clone()));
                    queued = shared.player.push_if(
                        &samples,
                        || epoch == shared.current_epoch(),
                        segment,
                    );
                }
                Err(err) => {
                    tracing::warn!(error = %err, text, "voice cascade TTS failed");
                    if response.is_some() {
                        shared.emit(ServerEvent::Error {
                            message: format!("speech synthesis failed: {err}"),
                        });
                    }
                }
            }
        }
        shared.clauses.fetch_sub(1, Ordering::SeqCst);
        if let Some(response) = response {
            shared.clause_done(&response, queued.then_some(text.as_str()));
        }
    }
}

/// Audio: sends `state` when it changes, and tells finished responses.
async fn report_state(shared: Arc<Shared>) {
    let mut last = None;
    while !shared.player.is_closed() {
        shared.check_done();
        let input_stopped = shared.audio_at.lock().unwrap().elapsed() > INPUT_STALE;
        let state = (
            shared.player.speaking() || shared.responding(),
            shared.listening.load(Ordering::SeqCst) > 0 && !input_stopped,
        );
        if last != Some(state) {
            last = Some(state);
            shared.emit(ServerEvent::State {
                speaking: state.0,
                listening: state.1,
            });
        }
        tokio::time::sleep(STATE_TICK).await;
    }
}

/// Audio: the responses being spoken (oldest first) and those over.
#[derive(Default)]
struct Responses {
    open: Vec<OpenResponse>,
    /// Responses that are over (done, cut or cancelled): text or an end still
    /// coming for them is dropped.
    over: VecDeque<String>,
}

impl Responses {
    fn close(&mut self, id: String) {
        self.over.push_back(id);
        while self.over.len() > OVER_KEPT {
            self.over.pop_front();
        }
    }

    fn is_over(&self, id: &str) -> bool {
        self.over.iter().any(|over| over == id)
    }
}

struct OpenResponse {
    id: String,
    splitter: ClauseSplitter,
    /// Its text is complete.
    ended: bool,
    /// Clauses waiting for the responses before it to be given to TTS.
    held: Vec<String>,
    /// Clauses given to TTS and not yet queued to play (or dropped).
    pending: usize,
    /// The text of its clauses queued to play.
    spoken: String,
}

impl OpenResponse {
    fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            splitter: ClauseSplitter::default(),
            ended: false,
            held: Vec::new(),
            pending: 0,
            spoken: String::new(),
        }
    }
}

/// Audio: a clause's place in the speech, in samples since the start.
struct Segment {
    start: u64,
    end: u64,
    response: String,
    text: String,
}

/// The bot's speech queue, played at real-time pace.
#[derive(Default)]
struct Player {
    state: Mutex<PlayerState>,
    arrived: Notify,
    closed: AtomicBool,
}

struct PlayerState {
    queue: VecDeque<f32>,
    /// When the audio sent so far ends playing.
    until: Instant,
    /// Bumped by a clear: a frame taken before it is not sent.
    generation: u64,
    /// Samples queued since the start (the dropped ones included).
    pushed: u64,
    /// Samples sent since the start.
    sent: u64,
    /// Audio: where the clauses queued since the last clear are.
    segments: VecDeque<Segment>,
}

impl Default for PlayerState {
    fn default() -> Self {
        Self {
            queue: VecDeque::new(),
            until: Instant::now(),
            generation: 0,
            pushed: 0,
            sent: 0,
            segments: VecDeque::new(),
        }
    }
}

impl PlayerState {
    /// Samples the client has played by now: those sent, less the lead
    /// still ahead of real time.
    fn heard(&self) -> u64 {
        let ahead = self.until.saturating_duration_since(Instant::now());
        self.sent
            .saturating_sub((ahead.as_secs_f64() * OUTPUT_RATE as f64) as u64)
    }
}

impl Player {
    /// Queues `samples` (the clause `segment`, response and text, in audio
    /// mode) if `current()` holds, checked under the queue lock (a cut
    /// changes it under the same lock: nothing stale gets in after).
    /// Returns whether they were queued.
    fn push_if(
        &self,
        samples: &[f32],
        current: impl FnOnce() -> bool,
        segment: Option<(String, String)>,
    ) -> bool {
        if self.is_closed() {
            return false;
        }
        {
            let mut state = self.state.lock().unwrap();
            if !current() {
                return false;
            }
            state.queue.extend(samples);
            let start = state.pushed;
            state.pushed += samples.len() as u64;
            if let Some((response, text)) = segment {
                let end = state.pushed;
                state.segments.push_back(Segment {
                    start,
                    end,
                    response,
                    text,
                });
            }
        }
        self.arrived.notify_one();
        true
    }

    /// Drops what is queued; `then` runs under the queue lock, first.
    fn clear_with(&self, then: impl FnOnce(&PlayerState)) {
        let mut state = self.state.lock().unwrap();
        then(&state);
        state.queue.clear();
        state.until = Instant::now();
        state.generation += 1;
        state.pushed = state.sent;
        state.segments.clear();
    }

    fn close(&self, then: impl FnOnce(&PlayerState)) {
        self.closed.store(true, Ordering::SeqCst);
        self.clear_with(then);
    }

    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    fn speaking(&self) -> bool {
        let state = self.state.lock().unwrap();
        !state.queue.is_empty() || state.until > Instant::now()
    }

    /// Waits until the speech queued now has played (or the conversation
    /// ended).
    async fn drained(&self) {
        loop {
            let wait = {
                let state = self.state.lock().unwrap();
                let queued = Duration::from_secs_f64(state.queue.len() as f64 / OUTPUT_RATE as f64);
                state.until.saturating_duration_since(Instant::now()) + queued
            };
            if wait.is_zero() || self.is_closed() {
                return;
            }
            tokio::time::sleep(wait.min(Duration::from_millis(100))).await;
        }
    }
}

async fn play(shared: Arc<Shared>) {
    let player = &shared.player;
    loop {
        let taken = {
            let mut state = player.state.lock().unwrap();
            let count = state.queue.len().min(FRAME_SAMPLES);
            (count > 0).then(|| {
                let frame: Vec<f32> = state.queue.drain(..count).collect();
                let now = Instant::now();
                if state.until < now {
                    state.until = now;
                }
                (frame, state.until, state.generation)
            })
        };
        let Some((frame, until, generation)) = taken else {
            player.arrived.notified().await;
            continue;
        };
        if let Some(send_at) = until.checked_sub(PLAY_LEAD) {
            tokio::time::sleep_until(send_at.into()).await;
        }
        let bytes: Vec<u8> = frame
            .iter()
            .flat_map(|s| ((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes())
            .collect();
        {
            let mut state = player.state.lock().unwrap();
            if state.generation != generation {
                continue; // cut while waiting
            }
            state.until += Duration::from_secs_f64(frame.len() as f64 / OUTPUT_RATE as f64);
            state.sent += frame.len() as u64;
            // Sent under the lock: a cut (which clears under it) comes after.
            let _ = shared.out.send(Outbound::Audio(bytes));
        }
        if let Some(ended) = shared.speech_ended_at.lock().unwrap().take() {
            tracing::info!(
                first_audio_ms = ended.elapsed().as_millis() as u64,
                "voice cascade answer starts playing"
            );
        }
    }
}

fn write_wav(path: &Path, samples: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: INPUT_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| InfraError::Runtime(format!("write {}: {e}", path.display())))?;
    for sample in samples {
        writer
            .write_sample((sample.clamp(-1.0, 1.0) * 32767.0) as i16)
            .map_err(|e| InfraError::Runtime(format!("write {}: {e}", path.display())))?;
    }
    writer
        .finalize()
        .map_err(|e| InfraError::Runtime(format!("write {}: {e}", path.display())))
}

/// Mono samples of a WAV file at `OUTPUT_RATE`.
fn read_wav(path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|e| InfraError::Runtime(format!("read {}: {e}", path.display())))?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(|s| s.ok()).collect(),
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample.max(1) - 1)) as f32;
            reader
                .samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / scale)
                .collect()
        }
    };
    let mono: Vec<f32> = interleaved
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
        .collect();
    if spec.sample_rate == OUTPUT_RATE || mono.is_empty() {
        return Ok(mono);
    }
    // Linear resampling (IndexTTS already speaks at OUTPUT_RATE).
    let ratio = spec.sample_rate as f64 / OUTPUT_RATE as f64;
    let len = (mono.len() as f64 / ratio) as usize;
    Ok((0..len)
        .map(|i| {
            let at = i as f64 * ratio;
            let index = at as usize;
            let next = mono[(index + 1).min(mono.len() - 1)];
            let frac = (at - index as f64) as f32;
            mono[index] * (1.0 - frac) + next * frac
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_takes_what_the_buffer_still_holds() {
        let pcm: Vec<f32> = (0..10).map(|i| i as f32).collect();
        assert_eq!(slice(&pcm, 100, 102, 105), vec![2.0, 3.0, 4.0]);
        assert_eq!(slice(&pcm, 100, 90, 102), vec![0.0, 1.0]);
        assert_eq!(slice(&pcm, 100, 108, 200), vec![8.0, 9.0]);
        assert!(slice(&pcm, 100, 120, 130).is_empty());
    }

    #[test]
    fn wav_round_trip_resamples_to_the_output_rate() {
        let dir = std::env::temp_dir().join(format!("vc-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.wav");
        write_wav(&path, &vec![0.5; INPUT_RATE as usize]).unwrap();
        let samples = read_wav(&path).unwrap();
        assert_eq!(samples.len(), OUTPUT_RATE as usize);
        assert!((samples[100] - 0.5).abs() < 1e-3);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn temp_files_go_when_dropped() {
        let path = std::env::temp_dir().join(format!("vc-temp-{}", uuid::Uuid::new_v4()));
        std::fs::write(&path, b"x").unwrap();
        drop(TempFile(path.clone()));
        assert!(!path.exists());
    }

    #[tokio::test]
    async fn a_closed_player_does_not_keep_anyone_waiting() {
        let player = Player::default();
        player.push_if(&vec![0.0; OUTPUT_RATE as usize * 60], || true, None);
        assert!(player.speaking());
        player.close(|_| {});
        tokio::time::timeout(Duration::from_secs(1), player.drained())
            .await
            .expect("drained returns once closed");
        player.push_if(&[0.0; 10], || true, None);
        assert!(!player.speaking());
    }

    #[test]
    fn speech_of_a_cut_turn_is_not_queued() {
        let player = Player::default();
        player.push_if(&[0.0; 10], || false, None);
        assert!(!player.speaking());
    }

    #[test]
    fn joined_text_spaces_only_words_of_spaced_scripts() {
        assert_eq!(join_text("", "你好"), "你好");
        assert_eq!(join_text("小乐", "你好"), "小乐你好");
        assert_eq!(join_text("Jarvis", "what time"), "Jarvis what time");
        assert_eq!(join_text("Jarvis,", "what"), "Jarvis,what");
        assert_eq!(join_text("time", "?"), "time?");
    }

    #[test]
    fn a_count_is_given_back_even_by_a_task_that_never_ran() {
        let counter = Arc::new(AtomicUsize::new(0));
        let due = Counted::new(&counter);
        let job = async move {
            let _due = due;
        };
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        drop(job);
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn the_bot_speaks_calmly_unless_told_otherwise() {
        let settings =
            |value: Value| -> BTreeMap<String, Value> { serde_json::from_value(value).unwrap() };
        let calm = tts_params(&BTreeMap::new()).unwrap();
        assert_eq!(calm["emotion_vector"], serde_json::json!({"calm": 0.8}));
        let happy = tts_params(&settings(
            serde_json::json!({"tts_emotion": "happy", "tts_emotion_strength": 0.5}),
        ))
        .unwrap();
        assert_eq!(happy["emotion_vector"], serde_json::json!({"happy": 0.5}));
        assert!(
            tts_params(&settings(serde_json::json!({"tts_emotion": "none"})))
                .unwrap()
                .is_empty()
        );
        assert!(tts_params(&settings(serde_json::json!({"tts_emotion": "bored"}))).is_err());
        assert!(tts_params(&settings(
            serde_json::json!({"tts_emotion": "sad", "tts_emotion_strength": 2})
        ))
        .is_err());
    }

    /// An audio mode `Shared` without models, with what it sends and the
    /// clauses it asks TTS for.
    fn audio_shared() -> (
        Arc<Shared>,
        mpsc::UnboundedReceiver<Outbound>,
        mpsc::UnboundedReceiver<Clause>,
    ) {
        let (out, out_rx) = mpsc::unbounded_channel();
        let (speech, speech_rx) = mpsc::unbounded_channel();
        let shared = Arc::new(Shared {
            audio: true,
            responses: Mutex::new(Responses::default()),
            next_utterance: AtomicU64::new(0),
            runtime: Arc::new(RuntimeManager::new(Vec::new(), Default::default())),
            system: String::new(),
            chat_model: String::new(),
            asr_model: String::new(),
            tts_model: String::new(),
            tts_params: BTreeMap::new(),
            ref_audio: PathBuf::new(),
            temp_dir: PathBuf::new(),
            tool_filler: String::new(),
            out,
            epoch: AtomicU64::new(0),
            generating: AtomicBool::new(false),
            history: Mutex::new(History::default()),
            player: Player::default(),
            speech,
            replies: tokio::sync::Mutex::new(()),
            jobs: Mutex::new(Vec::new()),
            last_task: Mutex::new(None),
            speech_ended_at: Mutex::new(None),
            listening: AtomicUsize::new(0),
            replies_due: Arc::new(AtomicUsize::new(0)),
            clauses: AtomicUsize::new(0),
            told: AtomicU64::new(0),
            speak_due: AtomicU64::new(0),
            audio_at: Mutex::new(Instant::now()),
        });
        (shared, out_rx, speech_rx)
    }

    /// Does for the next clause what TTS does: `samples` of speech queued.
    fn synthesize_next(
        shared: &Shared,
        clauses: &mut mpsc::UnboundedReceiver<Clause>,
        samples: usize,
    ) -> String {
        let clause = clauses.try_recv().expect("a clause to speak");
        let response = clause.response.clone().expect("a response clause");
        let queued = shared.player.push_if(
            &vec![0.0; samples],
            || clause.epoch == shared.current_epoch(),
            Some((response.clone(), clause.text.clone())),
        );
        shared.clauses.fetch_sub(1, Ordering::SeqCst);
        shared.clause_done(&response, queued.then_some(clause.text.as_str()));
        clause.text
    }

    /// Marks `samples` more of the speech as played by the client.
    fn play(shared: &Shared, samples: usize) {
        let mut state = shared.player.state.lock().unwrap();
        let samples = (samples as u64).min(state.pushed - state.sent);
        state.queue.drain(..samples as usize);
        state.sent += samples;
        state.until = Instant::now();
    }

    fn events(out: &mut mpsc::UnboundedReceiver<Outbound>) -> Vec<serde_json::Value> {
        std::iter::from_fn(|| out.try_recv().ok())
            .filter_map(|outbound| match outbound {
                Outbound::Event(event) => Some(serde_json::to_value(event).unwrap()),
                Outbound::Audio(_) => None,
            })
            .collect()
    }

    fn delta(id: &str, text: &str) -> ClientEvent {
        ClientEvent::ResponseDelta {
            response_id: id.into(),
            text: text.into(),
        }
    }

    fn end(id: &str) -> ClientEvent {
        ClientEvent::ResponseEnd {
            response_id: id.into(),
        }
    }

    #[test]
    fn a_response_is_done_once_its_speech_is_heard() {
        let (shared, mut out, mut clauses) = audio_shared();
        shared.audio_event(delta("r1", "你好呀，今天天气不错。"));
        shared.audio_event(delta("r1", "我们出去走走吧"));
        shared.audio_event(end("r1"));
        let first = synthesize_next(&shared, &mut clauses, 100);
        let second = synthesize_next(&shared, &mut clauses, 100);
        let third = synthesize_next(&shared, &mut clauses, 100);
        assert!(clauses.try_recv().is_err());
        assert!(shared.responding(), "not heard yet");
        play(&shared, 250);
        shared.check_done();
        assert!(shared.responding(), "the last clause is still playing");
        play(&shared, 50);
        shared.check_done();
        assert!(!shared.responding());
        let events = events(&mut out);
        let texts: Vec<&str> = events
            .iter()
            .filter(|event| event["type"] == "response.text")
            .map(|event| {
                assert_eq!(event["response_id"], "r1");
                event["text"].as_str().unwrap()
            })
            .collect();
        assert_eq!(texts, [&first, &second, &third]);
        let done = events.last().unwrap();
        assert_eq!(done["type"], "response.done");
        assert_eq!(done["response_id"], "r1");
        assert_eq!(done["cut"], false);
        assert_eq!(done["spoken"], format!("{first}{second}{third}"));
    }

    #[test]
    fn a_cut_tells_what_was_heard_and_drops_the_rest() {
        let (shared, mut out, mut clauses) = audio_shared();
        shared.audio_event(delta("r1", "第一句话说完了。第二句话还没说。"));
        let first = synthesize_next(&shared, &mut clauses, 100);
        synthesize_next(&shared, &mut clauses, 100);
        play(&shared, 60);
        shared.cut();
        let told = events(&mut out);
        let cut = told.iter().find(|e| e["type"] == "response.cut").unwrap();
        assert_eq!(cut["response_id"], "r1");
        let done = told.iter().find(|e| e["type"] == "response.done").unwrap();
        assert_eq!(done["cut"], true);
        assert_eq!(done["spoken"], first);
        assert!(!shared.responding());
        assert!(!shared.player.speaking());
        // Text still coming for it is dropped.
        shared.audio_event(delta("r1", "晚到的。"));
        shared.audio_event(end("r1"));
        assert!(clauses.try_recv().is_err());
        assert!(events(&mut out).is_empty());
    }

    #[test]
    fn a_response_without_text_is_done_at_once() {
        let (shared, mut out, _clauses) = audio_shared();
        shared.audio_event(end("silent"));
        let events = events(&mut out);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["type"], "response.done");
        assert_eq!(events[0]["spoken"], "");
        assert!(!shared.responding());
    }

    #[test]
    fn a_response_cancelled_before_it_starts_is_never_spoken() {
        let (shared, mut out, mut clauses) = audio_shared();
        shared.audio_event(ClientEvent::ResponseCancel {
            response_id: Some("r2".into()),
        });
        shared.audio_event(delta("r2", "不该说出来的话。"));
        shared.audio_event(end("r2"));
        assert!(clauses.try_recv().is_err());
        // It is over at once, and only once.
        let told = events(&mut out);
        assert_eq!(told.len(), 1, "{told:?}");
        assert_eq!(told[0]["type"], "response.done");
        assert_eq!(told[0]["response_id"], "r2");
        assert_eq!(told[0]["cut"], true);
    }

    #[test]
    fn responses_are_spoken_one_after_the_other() {
        let (shared, mut out, mut clauses) = audio_shared();
        shared.audio_event(delta("r1", "第一句。"));
        shared.audio_event(delta("r2", "插进来的话。"));
        shared.audio_event(end("r2"));
        // r2 waits until r1 has all its text.
        assert_eq!(synthesize_next(&shared, &mut clauses, 10), "第一句。");
        assert!(clauses.try_recv().is_err());
        shared.audio_event(delta("r1", "第二句。"));
        shared.audio_event(end("r1"));
        assert_eq!(synthesize_next(&shared, &mut clauses, 10), "第二句。");
        assert_eq!(synthesize_next(&shared, &mut clauses, 10), "插进来的话。");
        play(&shared, 30);
        shared.check_done();
        let done: Vec<String> = events(&mut out)
            .into_iter()
            .filter(|e| e["type"] == "response.done")
            .map(|e| e["response_id"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(done, ["r1", "r2"]);
    }

    #[test]
    fn cancelling_a_waiting_response_leaves_the_one_speaking() {
        let (shared, mut out, mut clauses) = audio_shared();
        shared.audio_event(delta("r1", "正在说的话。"));
        synthesize_next(&shared, &mut clauses, 10);
        shared.audio_event(delta("r2", "排在后面的话。"));
        shared.audio_event(ClientEvent::ResponseCancel {
            response_id: Some("r2".into()),
        });
        let told = events(&mut out);
        assert!(told.iter().all(|e| e["type"] != "response.cut"), "{told:?}");
        let done = told.iter().find(|e| e["type"] == "response.done").unwrap();
        assert_eq!(done["response_id"], "r2");
        assert!(shared.player.speaking());
        shared.audio_event(end("r1"));
        shared.audio_event(end("r2"));
        assert!(clauses.try_recv().is_err());
        play(&shared, 10);
        shared.check_done();
        let told = events(&mut out);
        assert_eq!(told.len(), 1, "{told:?}");
        assert_eq!(told[0]["response_id"], "r1");
    }

    #[test]
    fn a_finished_response_stays_finished() {
        let (shared, mut out, mut clauses) = audio_shared();
        shared.audio_event(delta("r1", "说完了。"));
        shared.audio_event(end("r1"));
        synthesize_next(&shared, &mut clauses, 10);
        play(&shared, 10);
        shared.check_done();
        events(&mut out);
        // A late end or text of it changes nothing.
        shared.audio_event(end("r1"));
        shared.audio_event(delta("r1", "迟到的话。"));
        assert!(events(&mut out).is_empty());
        assert!(clauses.try_recv().is_err());
        assert!(!shared.responding());
    }

    #[test]
    fn say_speaks_the_text_as_it_is() {
        let (shared, mut out, mut clauses) = audio_shared();
        shared.audio_event(ClientEvent::Say {
            text: "您好，我是小乐。".into(),
        });
        let mut spoken = String::new();
        while !clauses.is_empty() {
            spoken.push_str(&synthesize_next(&shared, &mut clauses, 10));
        }
        assert_eq!(spoken, "您好，我是小乐。");
        play(&shared, 100);
        shared.check_done();
        let done = events(&mut out)
            .into_iter()
            .find(|e| e["type"] == "response.done")
            .unwrap();
        assert_eq!(done["spoken"], spoken);
        assert!(done["response_id"].as_str().unwrap().starts_with("say-"));
    }

    #[test]
    fn clipped_text_keeps_whole_characters() {
        assert_eq!(clip("你好世界", 2), "你好");
        assert_eq!(clip("hi", 10), "hi");
    }
}
