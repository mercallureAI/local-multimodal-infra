"""Add the release files (README, NOTICE, LICENSES/) to a zh-tts-frontend directory.

`zh_frontend_export` calls this for every export. Run it alone to bring an
existing export up to date before publishing it:

    python -m scripts.local.zh_frontend_release workdir/models/zh-tts-frontend

It copies `scripts/local/zh_frontend_release/` into the directory and rewrites
`manifest.json`: the per-file licenses below, and the size and SHA-256 of every
file. The directory can then be uploaded as is, for example with
`hf upload ModaLeap/zh-tts-frontend <dir> .`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

RELEASE_DIR = Path(__file__).with_name("zh_frontend_release")

# SPDX expression and source of every exported file (directories end in "/").
# Keep in sync with README.md and NOTICE in RELEASE_DIR.
LICENSES = {
    "wetext/": "Apache-2.0 (wenet-e2e/WeTextProcessing via pengzhendong/wetext)",
    "g2pw/g2pw_int8.onnx": "Apache-2.0 (GitYCC/g2pW; INT8 graph from ModelScope pengzhendong/g2pw)",
    "g2pw/vocab.txt": "Apache-2.0 (google-research/bert bert-base-chinese)",
    "g2pw/meta.json": "Apache-2.0 (GitYCC/g2pW character tables)",
    "paddlespeech/polyphonic.tsv": "Apache-2.0 (PaddlePaddle/PaddleSpeech polyphonic.yaml)",
    "pinyin/t2s.tsv": "Apache-2.0 (PaddlePaddle/PaddleSpeech char_convert.py)",
    "pinyin/chars.tsv": "MIT AND Unicode-3.0 (mozillazg/python-pinyin, mozillazg/pinyin-data from Unihan)",
    "pinyin/phrases.tsv": "CC-BY-SA-4.0 AND MIT (mozillazg/python-pinyin, pypinyin-dict large_pinyin from phrase-pinyin-data incl. CC-CEDICT)",
    "mainland/phrases.tsv": "Apache-2.0 (pengzhendong/g2p-mix)",
    "mainland/readings.tsv": "MIT AND Unicode-3.0 (mozillazg/pypinyin-dict, Unihan kTGHZ2013/kXHC1983)",
}


def copy_release_files(out: Path) -> None:
    for source in sorted(RELEASE_DIR.rglob("*")):
        if source.is_file():
            target = out / source.relative_to(RELEASE_DIR)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_entries(out: Path) -> list[dict]:
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json")
    return [{"path": p.relative_to(out).as_posix(), "size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in files]


def check_licensed(entries: list[dict]) -> None:
    """Every data file must have a license entry; release files are exempt.

    This also keeps deployment files such as `user_phrases.tsv` out of a
    published directory.
    """
    release = {p.relative_to(RELEASE_DIR).as_posix() for p in RELEASE_DIR.rglob("*") if p.is_file()}
    unlicensed = [
        entry["path"]
        for entry in entries
        if entry["path"] not in release
        and not any(entry["path"] == key or (key.endswith("/") and entry["path"].startswith(key)) for key in LICENSES)
    ]
    if unlicensed:
        raise SystemExit(f"no license entry for (remove them or add one to LICENSES): {', '.join(unlicensed)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dir", type=Path, help="an existing zh_frontend_export output directory")
    args = parser.parse_args()

    out = args.dir.resolve()
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    copy_release_files(out)
    entries = file_entries(out)
    check_licensed(entries)
    manifest["licenses"] = LICENSES
    manifest["files"] = entries
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{out}: {len(entries)} files, licenses updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
