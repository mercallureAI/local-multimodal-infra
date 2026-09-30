"""Reference outputs for crates/zh-tts-frontend (PaddleSpeech Mandarin G2P).

Replays PaddleSpeech `zh_frontend.Frontend._g2p` with `g2p_model="g2pW"` on
the exported asset directory, without importing paddle:

- clause split: `TextNormalizer.SENTENCE_SPLITOR`;
- `jieba.posseg.lcut` + `ToneSandhi.pre_merge_for_modify` (PaddleSpeech's
  `tone_sandhi.py`, loaded from the checkout unchanged);
- whole-clause g2pW (`G2PWOnnxConverter.__call__` / `_prepare_data` /
  `prepare_onnx_input`, same INT8 graph, OpenCC s2tw) with the pypinyin
  fallback configured as `Frontend._init_pypinyin` does;
- `Polyphonic.correct_pronunciation` per word, then `ToneSandhi.modified_tone`
  applied to whole syllables (it only rewrites the trailing tone digit).

Erhua merging is skipped: IndexTTS reads "儿" itself. The output is one pinyin
(TONE3, neutral = 5) or null per character of each input line:

    python scripts/local/zh_frontend_oracle.py --paddlespeech <checkout> \\
        --assets workdir/models/zh-tts-frontend --input cases.txt --out reference.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zh_frontend_export import configure_pypinyin  # noqa: E402

SENTENCE_SPLITOR = re.compile(r"([：、，；。？！,;?!][”’]?)")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wordize_and_map(text):
    words, text2word, word2text = [], [], []
    while text:
        match_space = re.match(r"^ +", text)
        if match_space:
            text2word += [None] * len(match_space.group(0))
            text = text[len(match_space.group(0)):]
            continue
        match_en = re.match(r"^[a-zA-Z0-9]+", text)
        if match_en:
            word = match_en.group(0)
            start = len(text2word)
            word2text.append((start, start + len(word)))
            text2word += [len(words)] * len(word)
            words.append(word)
            text = text[len(word):]
        else:
            start = len(text2word)
            word2text.append((start, start + 1))
            text2word += [len(words)]
            words.append(text[0])
            text = text[1:]
    return words, text2word, word2text


def aligned_pinyin(text):
    """pypinyin readings aligned to characters, as `[[reading]]` items.

    Deliberate deviation from PaddleSpeech, which indexes pypinyin's list by
    character position although pypinyin returns each non-Han run as one
    item: any clause with a multi-character non-Han run ("20", "（）") then
    reads every later character from the wrong slot. The Rust port aligns per
    character; the reference does the same so parity checks real differences.
    """
    from pypinyin import Style, pinyin
    from pypinyin.constants import RE_HANS

    out = []
    index = 0
    for item in pinyin(text, neutral_tone_with_five=True, style=Style.TONE3):
        value = item[0]
        if index < len(text) and RE_HANS.match(text[index]):
            out.append([value])
            index += 1
        else:
            out.extend([[None]] * len(value))
            index += len(value)
    return out


class G2PW:
    def __init__(self, assets: Path):
        import onnxruntime
        import opencc
        from tokenizers import BertWordPieceTokenizer

        meta = json.loads((assets / "g2pw/meta.json").read_text(encoding="utf-8"))
        self.labels = meta["labels"]
        self.label_pinyin = meta["label_pinyin"]
        self.chars = meta["chars"]
        self.char2phonemes = meta["char_phonemes"]
        self.query_chars = set(meta["query_chars"])
        self.monophonic = meta["monophonic"]
        self.t2s = dict(line.split("\t") for line in (assets / "pinyin/t2s.tsv").read_text(encoding="utf-8").splitlines())
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = onnxruntime.InferenceSession(str(assets / "g2pw/g2pw_int8.onnx"), options, providers=["CPUExecutionProvider"])
        self.tokenizer = BertWordPieceTokenizer(str(assets / "g2pw/vocab.txt"), lowercase=True)
        self.vocab = self.tokenizer.get_vocab()
        self.cc = opencc.OpenCC("s2tw")

    def tokenize(self, word):
        tokens = self.tokenizer.encode(word, add_special_tokens=False).tokens
        return tokens

    def tokenize_and_map(self, text):
        words, text2word, word2text = wordize_and_map(text)
        tokens, token2text = [], []
        for word, (start, _end) in zip(words, word2text):
            word_tokens = self.tokenize(word)
            if len(word_tokens) == 0 or word_tokens == ["[UNK]"]:
                token2text.append((start, start + len(word)))
                tokens.append("[UNK]")
            else:
                current = start
                for token in word_tokens:
                    length = len(re.sub(r"^##", "", token))
                    token2text.append((current, current + length))
                    current += length
                    tokens.append(token)
        text2token = text2word
        for index, (start, end) in enumerate(token2text):
            for position in range(start, end):
                text2token[position] = index
        return tokens, text2token

    def __call__(self, sentence):
        from pypinyin import Style, pinyin

        translated = self.cc.convert(sentence)
        assert len(translated) == len(sentence)
        simplified = "".join(self.t2s.get(ch, ch) for ch in translated)
        fallback = aligned_pinyin(simplified)
        result = [None] * len(translated)
        queries = []
        for index, char in enumerate(translated):
            if char in self.query_chars:
                queries.append(index)
            elif char in self.monophonic:
                result[index] = self.monophonic[char]
            else:
                result[index] = fallback[index][0] if index < len(fallback) else None
        if not queries:
            return result
        text = translated.lower()
        tokens, text2token = self.tokenize_and_map(text)
        ids = [self.vocab.get(t, self.vocab["[UNK]"]) for t in ["[CLS]"] + tokens + ["[SEP]"]]
        batch = len(queries)
        feeds = {
            "input_ids": np.array([ids] * batch, dtype=np.int64),
            "token_type_ids": np.zeros((batch, len(ids)), dtype=np.int64),
            "attention_mask": np.ones((batch, len(ids)), dtype=np.int64),
            "phoneme_mask": np.array(
                [[1.0 if i in self.char2phonemes[text[q]] else 0.0 for i in range(len(self.labels))] for q in queries],
                dtype=np.float32,
            ),
            "char_ids": np.array([self.chars.index(text[q]) for q in queries], dtype=np.int64),
            "position_ids": np.array([text2token[q] + 1 for q in queries], dtype=np.int64),
        }
        probs = self.session.run([], feeds)[0]
        for query, label in zip(queries, probs.argmax(axis=1)):
            result[query] = self.label_pinyin[int(label)]
        return result


class Frontend:
    def __init__(self, paddlespeech: Path, assets: Path):
        configure_pypinyin()
        frontend = paddlespeech / "paddlespeech/t2s/frontend"
        self.sandhi = load_module(frontend / "tone_sandhi.py", "tone_sandhi").ToneSandhi()
        self.g2pw = G2PW(assets)
        self.polyphonic = {
            line.split("\t")[0]: line.split("\t")[1].split()
            for line in (assets / "paddlespeech/polyphonic.tsv").read_text(encoding="utf-8").splitlines()
        }

    def clause(self, seg):
        import jieba.posseg as psg

        seg_cut = self.sandhi.pre_merge_for_modify(psg.lcut(seg))
        pinyins = self.g2pw(seg)
        out = []
        position = 0
        for word, pos in seg_cut:
            end = position + len(word)
            if pos == "eng":
                out.extend([None] * len(word))
                position = end
                continue
            word_pinyins = pinyins[position:end]
            word_pinyins = list(self.polyphonic.get(word, word_pinyins))
            finals = [p if p is not None else char for p, char in zip(word_pinyins, word)]
            finals = self.sandhi.modified_tone(word, pos, finals)
            out.extend(p if p != char and re.fullmatch(r"[a-z]+[1-5]", p or "") else None for p, char in zip(finals, word))
            position = end
        return out

    def __call__(self, text):
        # PaddleSpeech strips English before G2P; keep the character mapping.
        kept = [i for i, ch in enumerate(text) if not re.match(r"[a-zA-Z]", ch)]
        stripped = "".join(text[i] for i in kept)
        pieces = [p for p in SENTENCE_SPLITOR.sub(r"\1\n", stripped).split("\n")]
        result = []
        for piece in pieces:
            result.extend(self.clause(piece) if piece else [])
        full = [None] * len(text)
        for index, value in zip(kept, result):
            full[index] = value
        return full


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paddlespeech", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    frontend = Frontend(args.paddlespeech, args.assets)
    lines = [line.strip() for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for text in lines:
            handle.write(json.dumps({"text": text, "pinyin": frontend(text)}, ensure_ascii=False) + "\n")
    print(f"wrote {len(lines)} reference lines to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
