# Realtime voice (`/v1/realtime`)

A pseudo-full-duplex voice conversation on the worker's own models, over one
WebSocket. The client streams microphone audio and plays what comes back; the
server does the rest:

1. **Silero VAD v6** (ONNX, CPU, one session per conversation; a port of the
   reference `VADIterator`: threshold 0.5, speech ends after `min_silence_ms`
   of silence, 30 ms padding) cuts the input into utterances.
2. **SenseVoice** recognises each utterance (the server does not tell
   voices apart: voiceprints of short phone-call utterances proved
   unreliable; a client may say who spoke, see Speaker labels). Utterances
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

## Wake words

In a group most talk is not for the bot, and asking a model about every
utterance only to stay silent keeps it busy when it is called. So a group
session also runs its input through a **keyword spotter**: k2-fsa's
open-vocabulary zipformer transducer KWS model for Chinese and English
(`sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20`, Apache-2.0, 3.3M
parameters; the `local-adapter-kws-zipformer` crate ports sherpa-onnx's
keyword spotter: Kaldi fbank, the streaming encoder in 320 ms chunks, a
beam search boosted along the keywords). It runs on the CPU beside the VAD,
about 2 % of a core, and spots a wake word 0.2–0.6 s after it is said. A
recognised utterance no wake word was spotted in yet (and not otherwise
answered) waits for the spotter to get a little past its end, at most 0.8 s:
a wake word that ends it ("..., <name>?") is spotted only after the VAD has
ended it. The spotter hears its input brought to a steady level first (a
gain following the speech's peaks over a few seconds, 0.5x to 31.6x): it
misses far more of a quiet or loud voice than of the same voice at a usual
level (recorded calls through a game client: 62 % spotted as heard, 71 %
levelled, no more false calls in 19 minutes of other speech). The search
keeps 16 paths a frame (`kws_paths`): a name's first tokens score low against
what else the speech could be, and with 4 they were dropped before the rest
came (82 % spotted with 16). Spotted calls then score from about 0.21 and
other speech not at all, so a keyword needs a mean token probability of
0.20 (`kws_threshold`, and `kws_short_threshold` for words of four tokens or
fewer; 84 % spotted, still no false calls). A bigger boost along the
keywords did worse (66 % at twice the boost). What is left missed is mostly
a voice 30–42 dB under a usual level, which no gain brings back.

The wake words are the bot's `name`, its `aliases` and `wake_words`, written
as text: Chinese characters are read in pinyin, English words with the model's
dictionary, a short run of letters (or capitals the dictionary does not have:
an acronym) letter by letter ("M" is "EH1 M"; a longer word the dictionary
does not have is not guessed: it is reported as unread) and digits in Chinese
(each, and as a number) and in English ("M42" is "M 四二", "M 四十二", "M
forty-two"), every combination one way to say the word. A word with letters on
both sides of a digit ("Mon3tr") has no reading to guess: give how it is said
as other words ("monster", "梦三特").

With `wake` on (the default in a group) only an utterance a wake word was
spoken in (anywhere: "<name>, ..." or "..., <name>?"), or one that starts
within 5 s of a bare call ("<name>" alone, or after a short "嗯"; not what
follows that), wants a
reply; the others are context (`respond: false`). So does one whose
transcript has a sentence starting or ending with a wake word ("M42跳一下",
"坐下吧，M3。", "嗯嗯。M3去查一下"; a few words around it like "嗯", "吧"
passed over, case, punctuation and Chinese numerals not minded): the
recogniser hears names the spotter missed (in a game room, 165 utterances
named the bot, the spotter caught 50; with the transcript 161). A first
clause that is only the name after a word or two calls too ("这个M42，你看看
周围", "那个，M42，……"), and so does a name of letters and digits the
recogniser wrote a letter or digit off, standing alone at a sentence's
start or end ("M2", "L42" or "M12" for "M42"; for a two-character name only
its digit: "M2" for "M3", not "A3"; replaying that room's 1548 transcripts
calls 18 more, all but one or two the bot). A name in
the middle of a sentence ("我说M3它……") is talked about, not called. Each
call the transcript heard and the spotter missed keeps its audio and
transcript (`<time>.wav`, `.txt`) in `<data>/voice-cascade-missed-wakes`,
the newest 200 (`missed_wakes_kept`, 0: none): real misses to tune the
spotter on. With `wake` off (one other
person to talk with) every utterance may get one, as before. `session.update`
turns it on and off mid-conversation (a room that fills up or empties). The
model replying may still stay silent. Without the spotter's model
(`<models>/voice-cascade/kws`, see `python -m scripts.local.fetch_kws_model`)
the bot's name in the transcript is what calls it.

The chat, ASR and TTS models are named by the `voice-cascade` model
(`configs/providers/realtime/voice-cascade.yaml`, whose artifact is the VAD model) and
must be enabled (`tts_model` defaults to `qwen3-tts-0.6b-onnx`, a local
export; `asr_model` to `sensevoice-small-fp16-onnx`, a local float16 export of
SenseVoiceSmall, ~4x faster on CUDA than the downloadable int8
`sensevoice-small-onnx`; a session whose `asr_model` fails to load when it
starts falls back to `asr_fallback_model`, `sensevoice-small-onnx`). With IndexTTS-2.5 the bot speaks with a fixed emotion,
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
and spoken as it comes; with `tts_stream_text` a response's first clause is
spoken while it is still coming, as in cascade mode, so stream the text as
the model writes it; should its text stop coming for 1.5 s, it ends there
and its rest is spoken once complete) and `response.end` under a `response_id` of its
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

## Speaker labels

A client that knows who is talking (e.g. by where the voice comes from) says
so with `session.update` `{"speaker": {"name", "start", "end", "final",
"candidates"}}`: `start`/`end` are input samples (16 kHz, counted from the
session's first audio: the clock the VAD's utterances are on), `name` null for
someone not recognised. `candidates` (optional) are how likely each one is,
likeliest first: `[{"name": "Alice", "p": 0.62}, {"name": null, "p": 0.3}]`
(null: someone not recognised); without them the label is `name`, sure. A
label is the latest for its speech segment: a later one with the same `start`
replaces it (a segment still growing, relabelled every 250 ms or so, then once
more with `final: true`). The server keeps the last 60 s of them (at most 512).

Who said an utterance: each label's candidates weighed by how much of its
speech the label covers (a joined one: all its parts, not the pauses between
them, so a bare call, a pause and the request are judged by what was said),
over what the labels cover; nobody when they cover less than 30% of it.
`input.transcript` carries `speaker`, the likeliest name (null when that is
someone not recognised, or nobody). Once labels have come in a session (a
group or one to one alike), its `text` starts with who said it, which is what
the model reads (cascade: the history and `tool.call` `heard`; the prompts
explain it):

| Case | `text` |
|---|---|
| Sure: the likeliest at least 75% and 30 points ahead of the next | `Alice: 明天天气怎么样` |
| Unsure: the likeliest few (at most 3, each at least 10%), `someone` for a person not recognised | `[Alice 62% / someone 30%]: 明天天气怎么样` |
| Unknown: the likeliest is someone not recognised, or no label covers it | `[unknown speaker]: 明天天气怎么样` |

A session that never gets a label (a call) reads the text as heard. A joined
utterance is said once, whole. What the wake words, the bot's names in the
transcript and joins go by is the text without it. The formats live in one
place (`session::Who::said`), with tests.

A label may come a little after its audio. Once labels have come in a
session, an utterance they do not reach yet (to within 250 ms of its end)
waits for its own until 0.5 s after the VAD ended it (the wake words'
waiting goes on meanwhile; audio keeps flowing); in a session without labels
nothing waits.

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
| `session.update` | `config` | Changes a running session: `wake`, `aliases`, `wake_words`; `speaker` labels who said a stretch of the input (see Speaker labels). |
| `session.stop` | | Ends the conversation. |

`session.start.config`:

| field | default | |
| --- | --- | --- |
| `mode` | `cascade` | `audio`: the client runs the conversation (see above). |
| `name` | (required) | The bot's name. |
| `aliases` | `[]` | Other names (homophones) that address it. |
| `group` | `false` | Several people talk with each other (a channel), rather than one person with the bot. |
| `wake` | `true` in a group | Only what calls the bot by a wake word wants a reply (see Wake words). |
| `wake_words` | `[]` | More wake words besides `name` and `aliases`. |
| `speaker` | | Cascade: the person talking, one to one. |
| `instructions` | | Cascade: a persona appended to the bot's instructions. |
| `ref_audio` | model's `default_reference_audio` | The voice: a WAV file, base64. |
| `ref_text` | model's `default_reference_text` (with its audio) | Qwen3-TTS: what `ref_audio` says, so it clones the voice in context (closer) instead of from its x-vector alone. |
| `tts_language` | model's `tts_language` | Qwen3-TTS: chinese, english, japanese, korean, german, french, russian, portuguese, spanish, italian (unset: the model decides). |
| `tts_stream_text` | model's `tts_stream_text` (true) | Speak a reply's (audio: a response's) first clause while its text is still coming; only with a TTS model that takes streamed text (Qwen3-TTS). With `ref_text` it gains little: in-context cloning starts once the text covers the reference (see `docs/qwen3-tts.md`). |
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
| `input.transcript` | `text`, `partial`?, `id`?, `replaces`?, `respond`?, `called`?, `speaker`? | An utterance (joined when the speaker only paused); `partial`: a piece of a long one still going on. `id`, `replaces`, `respond`: audio mode; `called` (audio, group): it calls the bot (a wake word in it, or without the spotter its name; or it follows a bare call, or continues a called one). `speaker`: who said it, the likeliest name by the client's labels (once labels came, `text` starts with who said it: see Speaker labels). |
| `input.wake` | `word`, `score` | A wake word was heard (its utterance may still go on). |
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
