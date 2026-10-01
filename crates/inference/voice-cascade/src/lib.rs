//! Pseudo-realtime voice on the worker's own models: Silero VAD, ASR, a chat
//! model and TTS in one loop, spoken over the `/v1/realtime` WebSocket (see
//! `protocol`).

mod history;
mod prompt;
pub mod protocol;
mod session;
mod text;

pub use session::{run, CascadeModels, Inbound, Outbound};
