# Realtime voice (`/v1/realtime`)

A pseudo-full-duplex voice conversation on the worker's own models, over one
WebSocket. The client streams microphone audio and plays what comes back; the
server does the rest:

1. **Silero VAD v6** (ONNX, CPU, one session per conversation; a port of the
   reference `VADIterator`: threshold 0.5, speech ends after `min_silence_ms`
   of silence, 30 ms padding) cuts the input into utterances.
2. **SenseVoice** recognises each utterance (speakers are not told apart:
   voiceprints of short phone-call utterances proved unreliable). Utterances
   less than 1.5 s apart are joined (the speaker only paused; in a group
   only after a short call such as a bare "<name>,", since what follows a
   longer utterance is likely someone else): the joined one replaces the
   first in the conversation, and a reply to the first is stopped and asked
   again. Speech longer than 15 s is recognised in pieces (each sent as a
   `partial` transcript) and answered once it ends.
3. **Qwen3** (`chat.complete`, streamed) decides in one pass: answer, stay
   silent (`silence` tool) or hand a task to the client (`backend_task`
   tool). Every turn offers the same tools, so the server reuses the cached
   prompt prefix; turns that must speak are steered with logit biases.
4. **IndexTTS** speaks the answer clause by clause while it is still being
   generated; the server sends the audio at real-time pace (0.3 s ahead).

Talking over the bot stops it: one to one after `barge_in_ms` of speech (or
an utterance that is more than a backchannel; an utterance that stopped the
bot is always answered), in a group only when the bot's name is said. A stop
ends the bot's speech and its generation, and the server sends
`response.cut` so the client drops the little audio it has buffered.

The chat, ASR and TTS models are named by the `voice-cascade` model
(`configs/models.d/voice-cascade.yaml`, whose artifact is the VAD model) and
must be enabled. With IndexTTS-2.5 the bot speaks with a fixed emotion,
`tts_emotion` (default `calm`; `none` keeps the reference voice's own) at
`tts_emotion_strength` (default 0.8); a session may choose its own
(`session.start` config). A session loads them all before
`session.started` (IndexTTS takes tens of seconds the first time).

The conversation the chat model reads keeps the last 32 messages and about
4000 characters (utterances are cut to their last 600 characters, tool
results and notes to 1200). A reply lands after the message it answers and a
tool result after its call, whatever was said meanwhile; a joined utterance
drops the replies to its first part but not the tasks handed off for it or
their results. Tool results and notes are told once the bot has finished
speaking (its last clauses included), nobody is in the middle of an
utterance and the utterances heard so far are answered (at most 20 s later),
even when the bot was stopped meanwhile; results that arrive together are
told in one turn. Short-lived audio files go to `<data_dir>/voice-cascade`.

## Connection

`GET ws://<controller>/v1/realtime[?model=voice-cascade]` with
`Authorization: Bearer <one of LOCAL_MCP_INFER_TOKENS>` (when that list is
set). The controller forwards the socket to a worker serving the model
(`/internal/realtime`). Without inference tokens, requests with an `Origin`
header (web pages) are refused: a WebSocket is not subject to CORS, and voice
clients are not browsers.

- **Text frames**: JSON events below.
- **Binary frames**: audio, 16-bit little-endian mono PCM — 16 kHz from the
  client, `output_rate` (24 kHz) from the server. The client streams
  continuously, silence included: speech ends by the VAD hearing silence.

## Client events

| type | fields | |
| --- | --- | --- |
| `session.start` | `config` | Must be first. |
| `tool.result` | `call_id`, `output` | The answer to a `tool.call`; the bot tells it in its own words. |
| `note` | `text` | Backend news; the bot tells it if it matters, else stays silent. |
| `say` | `text` | Makes the bot speak first (e.g. why it placed a call). |
| `session.stop` | | Ends the conversation. |

`session.start.config`:

| field | default | |
| --- | --- | --- |
| `name` | (required) | The bot's name. |
| `aliases` | `[]` | Other names (homophones) that address it. |
| `group` | `false` | Several people talk with each other (a channel), rather than one person with the bot. |
| `speaker` | | The person talking, one to one. |
| `instructions` | | A persona appended to the bot's instructions. |
| `ref_audio` | model's `default_reference_audio` | The voice: a WAV file, base64. |
| `tool_filler` | none | Said right away when a task is handed off. |
| `chat_model`, `asr_model`, `tts_model` | the model's `metadata` | |
| `tts_emotion`, `tts_emotion_strength` | the model's `metadata` (`calm`, 0.8) | IndexTTS-2.5 emotion: happy, angry, sad, afraid, disgusted, melancholic, surprised, calm, or none; strength 0 to 1. |
| `vad_threshold` | 0.5 | |
| `min_silence_ms` | 600 | |
| `barge_in_ms` | 1200 | |

## Server events

| type | fields | |
| --- | --- | --- |
| `session.started` | `input_rate`, `output_rate` | Models loaded; audio may flow. |
| `input.speech_started` / `input.speech_stopped` | | VAD edges. |
| `input.transcript` | `text`, `partial`? | An utterance (joined when the speaker only paused); `partial`: a piece of a long one still going on. |
| `response.text` | `text` | A clause the bot is about to say. |
| `response.cut` | | The bot was stopped: drop its buffered audio. |
| `tool.call` | `call_id`, `name`, `arguments`, `heard` | A task for the client (`backend_task`: `arguments.task`); answer with `tool.result`. |
| `error` | `message` | A failure; after `session.start` failures the socket closes. |

## Checking it

`python -m scripts.local.realtime_e2e --audio-dir <wavs>` starts the release services, opens a session
and plays spoken test utterances (WAV files, 16 kHz) at real-time pace,
printing transcripts, tool calls, what the bot says and how long after the end
of each utterance its audio starts; it answers tool calls itself. It needs the
Python `websockets`, `numpy` and `librosa` packages.

On an RTX 4090 the answer starts 0.65–1.2 s after the end of the speaker's
audio, of which 0.6 s is the VAD's end-of-speech silence.
