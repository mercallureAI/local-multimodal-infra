"""Fetches the wake word spotter's model for the realtime voice cascade.

The voice cascade spots wake words (the bot's name, aliases, `wake_words`)
in group sessions with k2-fsa's open-vocabulary zipformer transducer KWS
model for Chinese and English (sherpa-onnx release
`sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20`, Apache-2.0). This
downloads the release archive, checks its SHA-256 and lays out the files the
`local-adapter-kws-zipformer` crate reads in `<models>/voice-cascade/kws`:

- encoder.onnx, decoder.onnx, joiner.onnx: the fp32 chunk-16 exports
  (320 ms chunks; int8 proved slower on CPU and missed words);
- tokens.txt, en.phone: the model's tokens and English pronunciations;
- pinyin.tsv: one reading per Chinese character, copied from the
  zh-tts-frontend model's `pinyin/chars.tsv` (pypinyin's tables).

    python -m scripts.local.fetch_kws_model [--models workdir/models]
        [--archive local.tar.bz2] [--proxy http://127.0.0.1:7890]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from .paths import repo_root, resolve_cli_path

NAME = "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/{NAME}.tar.bz2"
SHA256 = "68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6"
FILES = {
    "encoder.onnx": "encoder-epoch-13-avg-2-chunk-16-left-64.onnx",
    "decoder.onnx": "decoder-epoch-13-avg-2-chunk-16-left-64.onnx",
    "joiner.onnx": "joiner-epoch-13-avg-2-chunk-16-left-64.onnx",
    "tokens.txt": "tokens.txt",
    "en.phone": "en.phone",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", type=Path, default=Path("workdir/models"))
    parser.add_argument("--archive", type=Path, help="an already downloaded archive")
    parser.add_argument("--proxy", help="HTTP(S) proxy for the download")
    args = parser.parse_args()
    models = resolve_cli_path(args.models, repo_root())
    dest = models / "voice-cascade" / "kws"
    pinyin = models / "zh-tts-frontend" / "pinyin" / "chars.tsv"
    if not pinyin.is_file():
        print(f"missing {pinyin}: install the zh-tts-frontend model first", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        archive = args.archive
        if archive is None:
            archive = Path(tmp) / f"{NAME}.tar.bz2"
            if args.proxy:
                urllib.request.install_opener(
                    urllib.request.build_opener(
                        urllib.request.ProxyHandler({"http": args.proxy, "https": args.proxy})
                    )
                )
            print(f"downloading {URL}")
            urllib.request.urlretrieve(URL, archive)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != SHA256:
            print(f"{archive}: sha256 {digest}, expected {SHA256}", file=sys.stderr)
            return 1
        with tarfile.open(archive, "r:bz2") as tar:
            try:
                tar.extractall(tmp, filter="data")
            except TypeError:  # Python before extraction filters: the checksum vouches for it
                tar.extractall(tmp)
        source = Path(tmp) / NAME
        dest.mkdir(parents=True, exist_ok=True)
        # Each file whole or not at all: an interrupted copy leaves no
        # half model to load.
        for name, original in [*FILES.items(), ("pinyin.tsv", pinyin)]:
            part = dest / f"{name}.part"
            shutil.copyfile(source / original, part)
            part.replace(dest / name)
    print(f"wake word model in {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
