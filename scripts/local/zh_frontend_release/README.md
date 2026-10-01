---
license: other
license_name: mixed-apache-2.0-mit-unicode-3.0-cc-by-sa-4.0
license_link: https://huggingface.co/ModaLeap/zh-tts-frontend/blob/main/NOTICE
language:
  - zh
  - en
tags:
  - g2p
  - g2pw
  - polyphone
  - pinyin
  - text-normalization
  - wetext
  - mandarin
  - tts
  - indextts
  - onnx
  - onnxruntime
---

# zh-tts-frontend

This model is used by https://github.com/mercallureAI/local-multimodal-infra .

Data and model files for a Mandarin TTS text front end: WeText text
normalization, then pinyin for the characters whose reading depends on
context (g2pW polyphone disambiguation plus PaddleSpeech's rules). They are
the assets of the `local-zh-tts-frontend` Rust crate in
[local-multimodal-infra](https://github.com/mercallureAI/local-multimodal-infra),
which uses them to give IndexTTS 1.5
([ModaLeap/indextts-1.5-onnx](https://huggingface.co/ModaLeap/indextts-1.5-onnx))
and IndexTTS 2.5
([ModaLeap/indextts-2.5-onnx](https://huggingface.co/ModaLeap/indextts-2.5-onnx))
correct readings for polyphones such as 行, 还, 长 and 了.

This is not an official release of WeText, g2pW or PaddleSpeech.

## What the front end does

1. Normalizes text with WeText (`zh` and `en` TN grammars, IndexTTS settings):
   numbers, dates, times, units and symbols become words.
2. Splits clauses and segments words (jieba part-of-speech cut).
3. Reads each clause with g2pW (INT8, CPU), falling back to pypinyin, and
   applies PaddleSpeech's `polyphonic.yaml` corrections and tone-sandhi
   rules, ported literally from PaddleSpeech's `zh_frontend.py`. For IndexTTS
   only the neutral-tone rules apply; third-tone and 一/不 sandhi are left to
   the model.
4. In Mainland mode (what IndexTTS uses), maps g2pW's readings, which follow a
   Taiwan dictionary, onto the standard readings of 通用规范汉字字典 and
   现代汉语词典, and applies g2p-mix's phrase readings.

Against a Python replay of PaddleSpeech's front end the port matches 19,969
of 19,969 characters. On the CPP polyphone test set (8,935 targets) it
scores 95.22% with the Mainland layer and 88.10% without; pypinyin alone
scores 89.38%.

## Files

| File | Role | License | Source |
| --- | --- | --- | --- |
| `wetext/{zh,en}/tn/{tagger,verbalizer}.fst` | Text normalization grammars | Apache-2.0 | [WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing) via [wetext](https://github.com/pengzhendong/wetext) 0.1.0 |
| `g2pw/g2pw_int8.onnx` | g2pW polyphone model, dynamic INT8 (152 MB) | Apache-2.0 | [g2pW](https://github.com/GitYCC/g2pW); INT8 graph from ModelScope `pengzhendong/g2pw` |
| `g2pw/vocab.txt` | BERT WordPiece vocabulary | Apache-2.0 | [bert-base-chinese](https://github.com/google-research/bert) |
| `g2pw/meta.json` | g2pW labels, per-character label masks, monophonic readings | Apache-2.0 | g2pW character tables |
| `paddlespeech/polyphonic.tsv` | Phrase reading corrections | Apache-2.0 | [PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech) `polyphonic.yaml` |
| `pinyin/t2s.tsv` | Traditional-to-simplified characters | Apache-2.0 | PaddleSpeech `char_convert.py` |
| `pinyin/chars.tsv` | Default reading per character | MIT, Unicode-3.0 | [pypinyin](https://github.com/mozillazg/python-pinyin), [pinyin-data](https://github.com/mozillazg/pinyin-data) (Unihan) |
| `pinyin/phrases.tsv` | Phrase readings (about 412,000) | **CC BY-SA 4.0**, MIT | pypinyin with [pypinyin-dict](https://github.com/mozillazg/pypinyin-dict) `large_pinyin` ([phrase-pinyin-data](https://github.com/mozillazg/phrase-pinyin-data), including [CC-CEDICT](https://cc-cedict.org/)) |
| `mainland/phrases.tsv` | Mainland phrase readings | Apache-2.0 | [g2p-mix](https://github.com/pengzhendong/g2p-mix) |
| `mainland/readings.tsv` | Mainland standard reading per character | MIT, Unicode-3.0 | pypinyin-dict, Unihan `kTGHZ2013` / `kXHC1983` |
| `manifest.json` | Source versions and the size and SHA-256 of every file | | |
| `config.json` | Package summary; the Hub counts downloads by requests for this file. Not read at run time | | |

All readings are numbered-tone pinyin (`hang2`, neutral tone `5`). Source
revisions, the changes made to each source and the attribution notices are in
[`NOTICE`](NOTICE).

## License

The files keep the licenses of their sources, listed above; the full texts are
in [`LICENSES/`](LICENSES):

- [Apache-2.0](LICENSES/Apache-2.0.txt): the WeText grammars, the g2pW model and
  its tables, the BERT vocabulary and the PaddleSpeech and g2p-mix data.
- [MIT](LICENSES/MIT.txt): pypinyin, pinyin-data, phrase-pinyin-data and
  pypinyin-dict.
- [Unicode-3.0](LICENSES/Unicode-3.0.txt): data from the Unicode Han Database.
- [CC BY-SA 4.0](LICENSES/CC-BY-SA-4.0.txt): `pinyin/phrases.tsv`, which
  contains CC-CEDICT data. If you share a modified version of that file, you
  must share it under CC BY-SA 4.0 too. The other files are separate works and
  are not affected.

## Use with local-multimodal-infra

Download the repository to `<model_dir>/zh-tts-frontend` (next to the IndexTTS
model directories), or point `LOCAL_ZH_TTS_FRONTEND_DIR` at it:

```bash
hf download ModaLeap/zh-tts-frontend --local-dir workdir/models/zh-tts-frontend
```

IndexTTS picks the front end up when it loads and otherwise falls back to its
built-in text rules. To fix the reading of a word, add
`<frontend dir>/user_phrases.tsv` with one `word<TAB>pin1 yin1` per line
(`#` starts a comment). These words are added to the segmenter and override
every other stage.

## Rebuilding

`scripts/local/zh_frontend_export.py` in local-multimodal-infra rebuilds this
directory from PaddleSpeech and g2p-mix checkouts, the ModelScope
`pengzhendong/g2pw` model and the `g2pw`, `jieba`, `pypinyin`,
`pypinyin-dict` and `wetext` Python packages, and copies in this README, the
NOTICE and the license texts.
