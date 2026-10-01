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
4. **Qwen3-TTS** (or IndexTTS) speaks the answer clause by clause while it is
   still being generated; Qwen3-TTS streams each clause too, so a clause
   starts playing about 50 ms after it is asked for, and it speaks a reply's
   first clause while the chat model is still writing it (`tts_stream_text`;
   see `docs/qwen3-tts.md`).
   The server sends the audio at real-time pace (0.3 s ahead).
   Chunks are whole sentences (only the first may end at a comma, so speech
   starts early), merged until each is about twice as long as the one before
   it, up to 60 characters; only a sentence longer than 120 is cut, at a comma
   (a first chunk, which may be speaking already, at a word).

Talking over the bot stops it: one to one after `barge_in_ms` of speech (or
an utterance that is more than a backchannel; an utterance that stopped the
bot is always answered), in a group only when the bot's name is said. A stop
ends the bot's speech and its generation, and the server sends
`response.cut` so the client drops the little audio it has buffered.

The chat, ASR and TTS models are named by the `voice-cascade` model
(`configs/providers/realtime/voice-cascade.yaml`, whose artifact is the VAD model) and
must be enabled (`tts_model` defaults to `qwen3-tts-0.6b-onnx`, a local
export). With IndexTTS-2.5 the bot speaks with a fixed emotion,
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

## Audio mode

With `"mode": "audio"` in `session.start.config` the server only listens and
speaks; the client runs the conversation with a model of its own (a cloud
LLM, say). There is no chat model (it need not be enabled), no `tool.call`,
and `tool.result` / `note` are refused. Listening, joining utterances and being
talked over work as above; the client gets:

- `input.transcript` with an `id`, `replaces` (the id of the utterance this
  one continues and replaces: a reply to that one is out of date) and
  `respond` (false for talk the bot is not part of: in a group, others
  talking while the bot speaks, which the client may keep as context; one to
  one, a backchannel);
- `state` (`speaking`, `listening`) whenever either changes, which tells the
  client when the bot is idle (e.g. to tell a task's result);
- `response.done` once a response is over: fully heard, or `cut` (someone
  talked over the bot, or the client cancelled it), with `spoken`, the
  clauses the listener heard at least the start of (what the client should
  keep in its history). Every response gets exactly one (the server remembers
  the last 256 ids that are over), in the order the responses were started,
  a response cancelled before it started included (cut, nothing spoken).

The client speaks by streaming `response.delta` (text, split into clauses
and spoken as it comes) and `response.end` under a `response_id` of its
choosing, unique within the session; responses play one after the other (a
later one waits until every earlier one has all its text). `response.cancel`
with the id of a response still waiting its turn (none of its speech on its way yet) drops just that one;
with the id of the one speaking, of one whose speech is already on its way,
or with no id, it stops the bot (and everything waiting). A response must be
ended or cancelled: until it is, the ones after it wait. Text still arriving for a response that is over is dropped. `say`
speaks a text as it is.

The server only knows the bot is busy once the client's first delta comes:
an utterance while the client's model is still thinking gets `respond: true`
and cuts nothing, so the client should stop its own pending reply (the
`replaces` id tells it when the speaker only paused).

A worker built without the `chat` category (`--features audio`) serves only
audio mode sessions: do not register it beside workers that cascade mode
clients reach.

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
| `tool.result` | `call_id`, `output` | Cascade: the answer to a `tool.call`; the bot tells it in its own words. |
| `note` | `text` | Cascade: backend news; the bot tells it if it matters, else stays silent. |
| `say` | `text` | Cascade: makes the bot speak first (e.g. why it placed a call). Audio: speaks `text` as it is. |
| `response.delta` | `response_id`, `text` | Audio: text to speak, streamed. |
| `response.end` | `response_id` | Audio: the response's text is complete. |
| `response.cancel` | `response_id`? | Audio: stops the bot (that response, or whatever it says). |
| `session.stop` | | Ends the conversation. |

`session.start.config`:

| field | default | |
| --- | --- | --- |
| `mode` | `cascade` | `audio`: the client runs the conversation (see above). |
| `name` | (required) | The bot's name. |
| `aliases` | `[]` | Other names (homophones) that address it. |
| `group` | `false` | Several people talk with each other (a channel), rather than one person with the bot. |
| `speaker` | | Cascade: the person talking, one to one. |
| `instructions` | | Cascade: a persona appended to the bot's instructions. |
| `ref_audio` | model's `default_reference_audio` | The voice: a WAV file, base64. |
| `ref_text` | model's `default_reference_text` (with its audio) | What `ref_audio` says: Qwen3-TTS clones the voice in context (closer) instead of from its x-vector alone. |
| `tts_language` | model's `tts_language` | Qwen3-TTS: chinese, english, japanese, korean, german, french, russian, portuguese, spanish, italian (unset: the model decides). |
| `tool_filler` | none | Cascade: said right away when a task is handed off. |
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
| `input.transcript` | `text`, `partial`?, `id`?, `replaces`?, `respond`? | An utterance (joined when the speaker only paused); `partial`: a piece of a long one still going on. `id`, `replaces`, `respond`: audio mode. |
| `state` | `speaking`, `listening` | Audio: sent when either changes. |
| `response.text` | `text`, `response_id`? | A clause the bot is about to say. |
| `response.cut` | `response_id`? | The bot was stopped: drop its buffered audio. |
| `response.done` | `response_id`, `spoken`, `cut` | Audio: a response is over. |
| `tool.call` | `call_id`, `name`, `arguments`, `heard` | Cascade: a task for the client (`backend_task`: `arguments.task`); answer with `tool.result`. |
| `error` | `message` | A failure; after `session.start` failures the socket closes. |

## Checking it

`python -m scripts.local.realtime_e2e --audio-dir <wavs>` starts the release services, opens a session
and plays spoken test utterances (WAV files, 16 kHz) at real-time pace,
printing transcripts, tool calls, what the bot says and how long after the end
of each utterance its audio starts; it answers tool calls itself. With
`--mode audio` it plays the client of an audio mode session, answering with
canned text. It needs the Python `websockets`, `numpy` and `librosa` packages.

On an RTX 4090 the answer starts 0.65–1.2 s after the end of the speaker's
audio, of which 0.6 s is the VAD's end-of-speech silence.
