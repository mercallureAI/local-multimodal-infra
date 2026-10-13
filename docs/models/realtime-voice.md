# Realtime voice models

`voice-cascade` is a voice conversation over the WebSocket `/v1/realtime`:
Silero VAD → speech recognition → chat → speech synthesis, with wake words in
group sessions. Its own files are only the VAD; it runs the models it names
([spec](../../configs/providers/realtime/voice-cascade.yaml)):

| Part | Model (spec metadata) | Get it |
| --- | --- | --- |
| VAD | Silero VAD v6.2.3 (the `voice-cascade` download) | download |
| Speech recognition | `asr_model`: `sensevoice-small-int8-gpu-onnx` | local export, see [speech recognition](speech-recognition.md) |
| | `asr_fallback_model`: `sensevoice-small-onnx`, used when the above is missing | download |
| Chat | `chat_model`: `qwen3-4b-instruct-2507-int4-onnx` | local export, see [chat](chat.md) |
| Speech synthesis | `tts_model`: `qwen3-tts-0.6b-onnx` (or `indextts-1.5-onnx`, `indextts-2.5-onnx`) | see [speech synthesis](speech-synthesis.md) |
| Wake words (optional) | sherpa-onnx KWS zipformer zh-en 3M, in `kws_dir` | `python -m scripts.local.fetch_kws_model` |

With the defaults on an RTX 4090, the answer starts 0.9–1.0 s after the
speaker stops, 0.6 s of which is the silence the VAD waits for.

## Setup

1. Download `voice-cascade` and `sensevoice-small-onnx`; export the GPU int8
   SenseVoice, Qwen3 chat and Qwen3-TTS models (or switch `tts_model` to an
   IndexTTS model, which downloads).
2. For group sessions, fetch the wake word model (Apache-2.0, ~13 MB, archive
   SHA-256 pinned) into `workdir/models/voice-cascade/kws`:

   ```bash
   python -m scripts.local.fetch_kws_model   # --proxy, or --archive <already downloaded>
   ```

   Without it, the bot's name heard in the transcript still calls it.
3. Give the bot a voice: `default_reference_audio` (a WAV path the worker
   reads) and `default_reference_text` (what it says), or send `ref_audio` /
   `ref_text` when a session starts.

## Options (spec metadata)

| Key | Default | Meaning |
| --- | --- | --- |
| `tts_language` | `chinese` | the language Qwen3-TTS speaks (empty: it decides) |
| `tts_stream_text` | `true` | speak a reply's first clause while it is still being written (Qwen3-TTS) |
| `tts_emotion`, `tts_emotion_strength` | `calm`, 0.8 | the voice's emotion (IndexTTS-2.5) |
| `kws_threshold`, `kws_short_threshold`, `kws_paths` | 0.2, 0.2, 16 | wake word spotter sensitivity and beam |
| `missed_wakes_kept` | 200 | calls the transcript heard but the spotter missed, kept for review |

Session events, wake words, audio mode, speaker labels and moderation of what
the session passes on: [docs/realtime-voice.md](../realtime-voice.md).
End-to-end check: `python -m scripts.local.realtime_e2e`.
