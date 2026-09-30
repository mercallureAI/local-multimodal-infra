"""Build the WeText TN parity reference for crates/wetext.

Runs the Python `wetext` package IndexTTS uses on Windows/macOS (0.1.0,
`Normalizer(lang=..., operator="tn")`) over WeTextProcessing's TN test inputs
plus TTS-oriented cases, and writes `crates/wetext/tests/data/tn_reference.jsonl`
({"lang", "text", "expected"}). Run in the IndexTTS export environment:

    python scripts/local/wetext_parity.py --wetextprocessing <checkout> [--fst-dir-out <dir>]

`--fst-dir-out` also copies the zh/en TN FSTs of the installed package, the
files the Rust crate must load for the reference to hold.
"""

from __future__ import annotations

import argparse
import json
import shutil
from importlib.metadata import version
from pathlib import Path

EXTRA_CASES = {
    "zh": [
        "电话：135-4567-8900",
        "客服热线400-820-8820，工作时间9:00-18:00",
        "现在是北京时间2025年01月11日 20:00",
        "2002年的第一场雪，下在了2003年",
        "速度是10km/h，温度-5℃到12℃",
        "这件衣服打8.5折，只要¥129.9",
        "增长了12.5%，达到3.2万亿元",
        "他这条视频点赞3000+，评论1000+，收藏500+",
        "IndexTTS 2.5 发布了，版本号v1.0.3",
        "第1名到第10名，共1/3的人",
        "2024-10-01开会，3:30pm结束",
        "我住在302室，房号1208",
        "iPhone 15 Pro Max售价7999元",
        "1键3连",
        "5G网络是4G网络的升级版",
        "苹果于2030/1/2发布新 iPhone 2X 系列手机，最低售价仅 ¥12999",
        "数到3就开始：1、2、3",
        "明天最高气温28度，降水概率30%",
        "共465篇，约315万字",
        "他今年25岁，身高1.78米，体重65.5公斤",
    ],
    "en": [
        "See you at 8:00 AM",
        "This sales for 2.5% off, only $12.5.",
        "Counting down 3, 2, 1, go!",
        "Call me at 555-0123 on May 5th, 2024.",
        "It costs $1,299.99 and weighs 2.5kg.",
        "The meeting is on 10/01/2024 at 3:30pm.",
    ],
}


def load_wetextprocessing_inputs(root: Path) -> dict[str, list[str]]:
    cases: dict[str, list[str]] = {"zh": [], "en": []}
    for lang, folder in (("zh", "chinese"), ("en", "english")):
        for path in sorted((root / "tn" / folder / "test" / "data").glob("*.txt")):
            if path.stem in {"normalizer_tag_oov", "preprocessor", "postprocessor"}:
                continue  # exercise options IndexTTS does not enable
            for line in path.read_text(encoding="utf-8").splitlines():
                if "=>" in line:
                    text = line.split("=>", 1)[0].strip()
                    if text:
                        cases[lang].append(text)
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wetextprocessing", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "crates/wetext/tests/data/tn_reference.jsonl")
    parser.add_argument("--fst-dir-out", type=Path)
    args = parser.parse_args()

    import wetext
    from wetext import Normalizer

    normalizers = {
        # IndexTTS front.py: zh keeps erhua; en uses defaults.
        "zh": Normalizer(lang="zh", operator="tn", remove_erhua=False),
        "en": Normalizer(lang="en", operator="tn"),
    }
    cases = load_wetextprocessing_inputs(args.wetextprocessing)
    for lang, extra in EXTRA_CASES.items():
        cases[lang].extend(extra)
    rows = []
    seen = set()
    for lang, texts in cases.items():
        for text in texts:
            if (lang, text) in seen:
                continue
            seen.add((lang, text))
            rows.append({"lang": lang, "text": text, "expected": normalizers[lang].normalize(text)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"wetext_version": version("wetext")}, ensure_ascii=False) + "\n")
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} cases (wetext {version('wetext')}) to {args.out}")

    if args.fst_dir_out:
        source = Path(wetext.__file__).parent / "fsts"
        for relative in ("zh/tn/tagger.fst", "zh/tn/verbalizer.fst", "en/tn/tagger.fst", "en/tn/verbalizer.fst"):
            target = args.fst_dir_out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, target)
        print(f"copied zh/en TN FSTs to {args.fst_dir_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
