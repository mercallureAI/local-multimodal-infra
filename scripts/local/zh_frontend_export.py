"""Assemble the zh-tts-frontend asset directory for crates/zh-tts-frontend.

The Rust crate ports PaddleSpeech's Mandarin frontend (`zh_frontend.py` with
`g2p_model="g2pW"`, `tone_sandhi.py`, `polyphonic.yaml`) and loads everything
data-driven from this directory:

    wetext/{zh,en}/tn/{tagger,verbalizer}.fst   WeText TN (Python wetext 0.1.0 files)
    g2pw/g2pw_int8.onnx, g2pw/vocab.txt         g2pW dynamic-INT8 graph + BERT vocab
    g2pw/meta.json                              labels, char->label masks, monophonic readings
    pinyin/chars.tsv, pinyin/phrases.tsv        pypinyin readings as PaddleSpeech configures it
    pinyin/t2s.tsv                              PaddleSpeech traditional->simplified map
    paddlespeech/polyphonic.tsv                 PaddleSpeech polyphonic.yaml corrections
    mainland/phrases.tsv                        g2p-mix Mainland phrase readings
    manifest.json

Readings are exported already converted by pypinyin itself (TONE3, neutral
tone as 5), so the Rust side never re-implements tone-mark conversion. Run in
an environment with jieba, pypinyin, pypinyin-dict, pyyaml and wetext:

    python scripts/local/zh_frontend_export.py --paddlespeech <checkout> \\
        --g2pw-model-dir <modelscope pengzhendong/g2pw> --g2pw-package-dir <site-packages/g2pw> \\
        --g2p-mix <checkout> --out workdir/models/zh-tts-frontend
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "local.zh_tts_frontend.v1"

# zh_frontend.Frontend.__init__ (PaddleSpeech): custom phrase readings loaded
# into pypinyin, written in pypinyin's "finals with tone" notation.
PADDLESPEECH_PHRASES = {
    "开户行": [["ka1i"], ["hu4"], ["hang2"]],
    "发卡行": [["fa4"], ["ka3"], ["hang2"]],
    "放款行": [["fa4ng"], ["kua3n"], ["hang2"]],
    "茧行": [["jia3n"], ["hang2"]],
    "行号": [["hang2"], ["ha4o"]],
    "各地": [["ge4"], ["di4"]],
    "借还款": [["jie4"], ["hua2n"], ["kua3n"]],
    "时间为": [["shi2"], ["jia1n"], ["we2i"]],
    "为准": [["we2i"], ["zhu3n"]],
    "色差": [["se4"], ["cha1"]],
    "嗲": [["dia3"]],
    "呗": [["bei5"]],
    "不": [["bu4"]],
    "咗": [["zuo5"]],
    "嘞": [["lei5"]],
    "掺和": [["chan1"], ["huo5"]],
}
# G2PWOnnxConverter.__init__ (PaddleSpeech)
NON_POLYPHONIC = {"一", "不", "和", "咋", "嗲", "剖", "差", "攢", "倒", "難", "奔", "勁", "拗", "肖", "瘙", "誒", "泊", "听", "噢"}
NON_MONOPHONIC = {"似", "攢"}


def configure_pypinyin():
    """Put pypinyin in the state PaddleSpeech's Frontend._init_pypinyin leaves it."""
    from pypinyin import load_phrases_dict, load_single_dict
    from pypinyin_dict.phrase_pinyin_data import large_pinyin

    large_pinyin.load()
    load_phrases_dict(PADDLESPEECH_PHRASES)
    load_single_dict({ord("地"): "de,di4"})


def export_pinyin(out: Path) -> dict:
    from pypinyin import Style
    from pypinyin.constants import PHRASES_DICT, PINYIN_DICT
    from pypinyin.converter import UltimateConverter

    converter = UltimateConverter(neutral_tone_with_five=True)

    def readings(text: str) -> list[str]:
        values = converter.convert(text, Style.TONE3, heteronym=False, errors="default", strict=True)
        return [value[0] for value in values]

    (out / "pinyin").mkdir(parents=True, exist_ok=True)
    chars = 0
    with (out / "pinyin/chars.tsv").open("w", encoding="utf-8", newline="\n") as handle:
        for code in sorted(PINYIN_DICT):
            char = chr(code)
            reading = readings(char)
            if len(reading) == 1 and reading[0] != char:
                handle.write(f"{char}\t{reading[0]}\n")
                chars += 1
    phrases = 0
    with (out / "pinyin/phrases.tsv").open("w", encoding="utf-8", newline="\n") as handle:
        for phrase in sorted(PHRASES_DICT):
            reading = readings(phrase)
            if len(reading) != len(phrase):
                continue
            handle.write(f"{phrase}\t{' '.join(reading)}\n")
            phrases += 1
    return {"chars": chars, "phrases": phrases}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export_paddlespeech(paddlespeech: Path, out: Path) -> dict:
    import yaml

    frontend = paddlespeech / "paddlespeech/t2s/frontend"
    char_convert = load_module(frontend / "zh_normalization/char_convert.py", "char_convert")
    traditional = char_convert.traditional_characters
    simplified = char_convert.simplified_charcters
    assert len(traditional) == len(simplified)
    pairs = sorted({(t, s) for t, s in zip(traditional, simplified) if t != s})
    with (out / "pinyin/t2s.tsv").open("w", encoding="utf-8", newline="\n") as handle:
        for t, s in pairs:
            handle.write(f"{t}\t{s}\n")

    polyphonic = yaml.safe_load((frontend / "polyphonic.yaml").read_text(encoding="utf-8"))["polyphonic"]
    (out / "paddlespeech").mkdir(parents=True, exist_ok=True)
    with (out / "paddlespeech/polyphonic.tsv").open("w", encoding="utf-8", newline="\n") as handle:
        for word, readings in polyphonic.items():
            assert len(readings) == len(word), (word, readings)
            handle.write(f"{word}\t{' '.join(readings)}\n")
    return {"t2s_pairs": len(pairs), "polyphonic_words": len(polyphonic), "revision": git_revision(paddlespeech)}


def export_g2pw(model_dir: Path, package_dir: Path, out: Path) -> dict:
    target = out / "g2pw"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_dir / "G2PWModel/g2pw_int8.onnx", target / "g2pw_int8.onnx")
    shutil.copy2(model_dir / "bert-base-chinese/vocab.txt", target / "vocab.txt")

    def rows(name: str) -> list[list[str]]:
        text = (model_dir / "G2PWModel" / name).read_text(encoding="utf-8").strip()
        return [line.split("\t") for line in text.split("\n")]

    polyphonic = rows("POLYPHONIC_CHARS.txt")
    monophonic = rows("MONOPHONIC_CHARS.txt")
    bopomofo_to_pinyin = json.loads((package_dir / "bopomofo_to_pinyin_wo_tune_dict.json").read_text(encoding="utf-8"))

    def to_pinyin(bopomofo: str) -> str | None:
        # G2PWOnnxConverter._convert_bopomofo_to_pinyin
        tone = bopomofo[-1]
        assert tone in "12345"
        component = bopomofo_to_pinyin.get(bopomofo[:-1])
        return component + tone if component else None

    # dataset.get_phoneme_labels (use_char_phoneme is False for this model)
    labels = sorted({phoneme for _, phoneme in polyphonic})
    char2phonemes: dict[str, list[int]] = {}
    for char, phoneme in polyphonic:
        char2phonemes.setdefault(char, []).append(labels.index(phoneme))
    chars = sorted(char2phonemes)
    query_chars = sorted(set(chars) - NON_POLYPHONIC)
    monophonic_readings = {
        char: to_pinyin(phoneme) for char, phoneme in monophonic if char not in NON_MONOPHONIC
    }
    meta = {
        "labels": labels,
        "label_pinyin": [to_pinyin(label) for label in labels],
        "chars": chars,
        "char_phonemes": {char: char2phonemes[char] for char in chars},
        "query_chars": query_chars,
        "monophonic": monophonic_readings,
        "use_mask": True,
        "max_len": 512,
        "model_version": (model_dir / "G2PWModel/version").read_text(encoding="utf-8").strip(),
    }
    (target / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return {"labels": len(labels), "chars": len(chars), "query_chars": len(query_chars), "monophonic": len(monophonic_readings)}


def export_mainland(g2p_mix: Path, out: Path) -> dict:
    (out / "mainland").mkdir(parents=True, exist_ok=True)
    source = g2p_mix / "g2p_mix/dict/phrases.txt"
    count = 0
    with (out / "mainland/phrases.tsv").open("w", encoding="utf-8", newline="\n") as handle:
        for line in source.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            word, readings = parts[0], parts[1:]
            if len(readings) != len(word):
                continue
            handle.write(f"{word}\t{' '.join(readings)}\n")
            count += 1
    return {"phrases": count, "revision": git_revision(g2p_mix)}


def export_wetext(out: Path, fst_dir: Path | None) -> dict:
    import wetext

    source = fst_dir or Path(wetext.__file__).parent / "fsts"
    for relative in ("zh/tn/tagger.fst", "zh/tn/verbalizer.fst", "en/tn/tagger.fst", "en/tn/verbalizer.fst"):
        target = out / "wetext" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    from importlib.metadata import version

    return {"package": f"wetext {version('wetext')}"}


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paddlespeech", required=True, type=Path)
    parser.add_argument("--g2pw-model-dir", required=True, type=Path)
    parser.add_argument("--g2pw-package-dir", required=True, type=Path)
    parser.add_argument("--g2p-mix", required=True, type=Path)
    parser.add_argument("--wetext-fst-dir", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    configure_pypinyin()
    from importlib.metadata import version

    sources = {
        "pinyin": {**export_pinyin(out), "pypinyin": version("pypinyin"), "pypinyin_dict": version("pypinyin-dict")},
        "paddlespeech": export_paddlespeech(args.paddlespeech, out),
        "g2pw": export_g2pw(args.g2pw_model_dir, args.g2pw_package_dir, out),
        "mainland": export_mainland(args.g2p_mix, out),
        "wetext": export_wetext(out, args.wetext_fst_dir),
    }
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {
        "schema": SCHEMA,
        "created_unix": int(time.time()),
        "python": sys.version.split()[0],
        "sources": sources,
        "licenses": {
            "wetext": "Apache-2.0 (WeTextProcessing / wetext)",
            "g2pw": "Apache-2.0 (GitYCC/g2pW; INT8 graph from ModelScope pengzhendong/g2pw)",
            "pinyin": "MIT (pypinyin, pypinyin-dict / phrase-pinyin-data)",
            "paddlespeech": "Apache-2.0 (PaddlePaddle/PaddleSpeech)",
            "mainland": "Apache-2.0 (pengzhendong/g2p-mix)",
        },
        "files": [{"path": p.relative_to(out).as_posix(), "size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in files],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(sources, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
