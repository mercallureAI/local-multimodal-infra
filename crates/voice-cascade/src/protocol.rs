//! The `/v1/realtime` WebSocket protocol.
//!
//! Text frames carry JSON events; binary frames carry audio as 16-bit
//! little-endian mono PCM: 16 kHz from the client, `output_rate` (24 kHz)
//! from the server. The server sends its speech at real-time pace (slightly
//! ahead), so a client plays what arrives.

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
    /// Makes the bot speak first about `text` (e.g. why it called).
    #[serde(rename = "say")]
    Say { text: String },
    #[serde(rename = "session.stop")]
    SessionStop,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SessionConfig {
    /// The bot's name, which addresses it in a group.
    pub name: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    /// Several people talk with each other (a channel), rather than one
    /// person with the bot (a call, a whisper).
    #[serde(default)]
    pub group: bool,
    /// The person talking, one to one.
    #[serde(default)]
    pub speaker: Option<String>,
    /// More instructions (a persona) appended to the bot's.
    #[serde(default)]
    pub instructions: Option<String>,
    /// The voice to speak in: a WAV file, base64. Without it the model's
    /// `default_reference_audio` is used.
    #[serde(default)]
    pub ref_audio: Option<String>,
    /// Said right away when a task is handed off (empty: nothing).
    #[serde(default)]
    pub tool_filler: Option<String>,
    #[serde(default)]
    pub chat_model: Option<String>,
    #[serde(default)]
    pub asr_model: Option<String>,
    #[serde(default)]
    pub tts_model: Option<String>,
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
    },
    /// A clause the bot is about to say.
    #[serde(rename = "response.text")]
    ResponseText { text: String },
    /// The bot was stopped: drop its audio not played yet.
    #[serde(rename = "response.cut")]
    ResponseCut,
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
