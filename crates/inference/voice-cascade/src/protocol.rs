//! The `/v1/realtime` WebSocket protocol.
//!
//! Text frames carry JSON events; binary frames carry audio as 16-bit
//! little-endian mono PCM: 16 kHz from the client, `output_rate` (24 kHz)
//! from the server. The server sends its speech at real-time pace (slightly
//! ahead), so a client plays what arrives.
//!
//! A session runs in one of two modes (`SessionConfig::mode`): `cascade`, the
//! whole conversation on the server (its chat model answers, hands tasks off
//! with `tool.call`), or `audio`, where the server only listens and speaks:
//! it sends what it hears (`input.transcript`, `state`) and speaks the text
//! the client streams (`response.delta`), the client's own model deciding
//! what to say.

use serde::{Deserialize, Serialize};

pub const INPUT_RATE: u32 = 16_000;
pub const OUTPUT_RATE: u32 = 24_000;

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum ClientEvent {
    /// The first message: who the bot is and how to talk.
    #[serde(rename = "session.start")]
    SessionStart { config: Box<SessionConfig> },
    /// The answer to a `tool.call`; the bot tells it in its own words.
    #[serde(rename = "tool.result")]
    ToolResult { call_id: String, output: String },
    /// Backend news for the bot, which tells it if it matters.
    #[serde(rename = "note")]
    Note { text: String },
    /// Cascade: makes the bot speak first about `text` (e.g. why it
    /// called). Audio: speaks `text` as it is.
    #[serde(rename = "say")]
    Say { text: String },
    /// Audio: text to speak, streamed; the first delta of an id starts that
    /// response (after any still playing).
    #[serde(rename = "response.delta")]
    ResponseDelta { response_id: String, text: String },
    /// Audio: the response's text is complete.
    #[serde(rename = "response.end")]
    ResponseEnd { response_id: String },
    /// Audio: stops the bot's speech (that response's, or any).
    #[serde(rename = "response.cancel")]
    ResponseCancel {
        #[serde(default)]
        response_id: Option<String>,
    },
    /// Changes settings of the running session.
    #[serde(rename = "session.update")]
    SessionUpdate { config: SessionUpdate },
    #[serde(rename = "session.stop")]
    SessionStop,
}

/// Settings a running session takes (`session.update`); those left out stay.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct SessionUpdate {
    #[serde(default)]
    pub wake: Option<bool>,
    #[serde(default)]
    pub aliases: Option<Vec<String>>,
    #[serde(default)]
    pub wake_words: Option<Vec<String>>,
    /// Who said a stretch of the input, as the client tells (e.g. by where
    /// the voice came from).
    #[serde(default)]
    pub speaker: Option<SpeakerSpan>,
}

/// A speaker's label for input samples `start..end` (16 kHz, counted since
/// the session's first audio: the clock of the utterances). A later label
/// with the same `start` replaces it (a speech segment still growing, then
/// `final`).
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct SpeakerSpan {
    /// None: someone not recognised.
    #[serde(default)]
    pub name: Option<String>,
    pub start: u64,
    pub end: u64,
    /// The segment is over: this is its last label.
    #[serde(default, rename = "final")]
    pub done: bool,
    /// How likely each one is to have said it, likeliest first (`name`
    /// None: someone not recognised). Empty: only `name`, sure.
    #[serde(default)]
    pub candidates: Vec<SpeakerCandidate>,
}

/// One of the people a speaker label may be, and how likely (0..1).
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct SpeakerCandidate {
    #[serde(default)]
    pub name: Option<String>,
    pub p: f64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionMode {
    /// The server's chat model runs the conversation.
    #[default]
    Cascade,
    /// The server only listens and speaks; the client runs the conversation.
    Audio,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SessionConfig {
    #[serde(default)]
    pub mode: SessionMode,
    /// The bot's name, which addresses it in a group.
    pub name: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    /// Several people talk with each other (a channel), rather than one
    /// person with the bot (a call, a whisper).
    #[serde(default)]
    pub group: bool,
    /// In a group: only what calls the bot by a wake word (its name, an
    /// alias, `wake_words`, spotted in the audio) wants a reply; the rest is
    /// context. False: every utterance may (one other person in the room).
    /// Default: on in a group.
    #[serde(default)]
    pub wake: Option<bool>,
    /// More wake words besides the name and the aliases (e.g. how a name of
    /// letters and digits is said: "monster", "梦三特" for "Mon3tr").
    #[serde(default)]
    pub wake_words: Vec<String>,
    /// The person talking, one to one (cascade).
    #[serde(default)]
    pub speaker: Option<String>,
    /// More instructions (a persona) appended to the bot's (cascade).
    #[serde(default)]
    pub instructions: Option<String>,
    /// The voice to speak in: a WAV file, base64. Without it the model's
    /// `default_reference_audio` is used.
    #[serde(default)]
    pub ref_audio: Option<String>,
    /// What `ref_audio` says: Qwen3-TTS then clones the voice in context
    /// (closer), else from its x-vector alone.
    #[serde(default)]
    pub ref_text: Option<String>,
    /// The language the bot speaks (Qwen3-TTS: chinese, english, ...).
    /// Without it, the model's `tts_language`.
    #[serde(default)]
    pub tts_language: Option<String>,
    /// Speak a reply's (a response's) first clause while its text is still
    /// coming, with a TTS model that takes streamed text (Qwen3-TTS).
    /// Without it, the model's `tts_stream_text`.
    #[serde(default)]
    pub tts_stream_text: Option<bool>,
    /// Said right away when a task is handed off (empty: nothing; cascade).
    #[serde(default)]
    pub tool_filler: Option<String>,
    #[serde(default)]
    pub chat_model: Option<String>,
    #[serde(default)]
    pub asr_model: Option<String>,
    #[serde(default)]
    pub tts_model: Option<String>,
    /// The emotion the bot speaks with (IndexTTS-2.5): happy, angry, sad,
    /// afraid, disgusted, melancholic, surprised, calm, or none for the
    /// reference voice's own. Without it, the model's `tts_emotion`.
    #[serde(default)]
    pub tts_emotion: Option<String>,
    /// Weight of that emotion, 0 to 1. Without it, the model's
    /// `tts_emotion_strength`.
    #[serde(default)]
    pub tts_emotion_strength: Option<f64>,
    /// Speech probability at which speech starts (Silero VAD).
    #[serde(default)]
    pub vad_threshold: Option<f32>,
    /// Silence that ends an utterance.
    #[serde(default)]
    pub min_silence_ms: Option<usize>,
    /// One to one: talking over the bot this long stops it.
    #[serde(default)]
    pub barge_in_ms: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum ServerEvent {
    /// The models are loaded; audio may flow.
    #[serde(rename = "session.started")]
    SessionStarted { input_rate: u32, output_rate: u32 },
    #[serde(rename = "input.speech_started")]
    SpeechStarted,
    #[serde(rename = "input.speech_stopped")]
    SpeechStopped,
    /// An utterance as recognised (joined with the previous one when the
    /// speaker only paused).
    #[serde(rename = "input.transcript")]
    Transcript {
        text: String,
        /// A piece of a long utterance still going on; the whole utterance
        /// follows when it ends.
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        partial: bool,
        /// Audio, whole utterances: its id.
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<u64>,
        /// Audio: the utterance continues this one (the speaker only
        /// paused), which it replaces; a reply to that one is out of date.
        #[serde(skip_serializing_if = "Option::is_none")]
        replaces: Option<u64>,
        /// Audio: whether it wants a reply. False for talk the bot is not
        /// part of (in a group, while it speaks: context) or a backchannel
        /// (one to one).
        #[serde(skip_serializing_if = "Option::is_none")]
        respond: Option<bool>,
        /// Audio, in a group: a wake word was spoken in it (or it follows
        /// a bare call).
        #[serde(skip_serializing_if = "Option::is_none")]
        called: Option<bool>,
        /// Who said it, by the client's labels (`session.update`
        /// `speaker`): the likeliest name (None: someone not recognised,
        /// or no label). Once labels came in the session, `text` starts
        /// with who said it (`Who::said`: `"<name>: "`, `"[<name> 62% /
        /// someone 30%]: "` or `"[unknown speaker]: "`).
        #[serde(skip_serializing_if = "Option::is_none")]
        speaker: Option<String>,
    },
    /// A wake word was heard (as soon as it is: its utterance may still go
    /// on).
    #[serde(rename = "input.wake")]
    Wake { word: String, score: f32 },
    /// Audio: sent when either changes. `speaking`: the bot speaks or has
    /// speech on its way; `listening`: someone is in the middle of an
    /// utterance (or it is being recognised).
    #[serde(rename = "state")]
    State { speaking: bool, listening: bool },
    /// A clause the bot is about to say.
    #[serde(rename = "response.text")]
    ResponseText {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        response_id: Option<String>,
    },
    /// The bot was stopped: drop its audio not played yet.
    #[serde(rename = "response.cut")]
    ResponseCut {
        #[serde(skip_serializing_if = "Option::is_none")]
        response_id: Option<String>,
    },
    /// Audio: a response is over: all spoken, or `cut` (stopped, by the
    /// client or by someone talking over the bot). `spoken` is what was
    /// heard of it.
    #[serde(rename = "response.done")]
    ResponseDone {
        response_id: String,
        spoken: String,
        cut: bool,
    },
    /// The bot hands a task to the client; answer with `tool.result`.
    #[serde(rename = "tool.call")]
    ToolCall {
        call_id: String,
        name: String,
        arguments: serde_json::Value,
        heard: String,
    },
    #[serde(rename = "error")]
    Error { message: String },
}
