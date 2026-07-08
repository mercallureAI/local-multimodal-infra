"""Compare official IndexTTS 1.5 text frontend outputs with Rust.

This helper is intentionally non-service and writes artifacts under
``workdir/data`` by default. It imports the official source tree from
``workdir/models/index-tts-v1.5`` when optional Python dependencies are present;
otherwise it still writes a JSON report containing actionable setup hints. The
Rust comparator is a non-service cargo binary and writes only stdout.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import re
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any


DEFAULT_TEXTS = [
    "IndexTTS 正式发布1.0版本了，效果666",
    "晕XUAN4是一种GAN3觉",
    "你好 OpenAI，世界！",
    "where's the money?",
    "约瑟夫·高登-莱维特（Joseph Gordon-Levitt is an American actor）",
    "现在是北京时间2025年01月11日 20:00，速度是10km/h",
]

DEPENDENCY_HINTS = [
    "Place the official IndexTTS 1.5 source tree at workdir/models/index-tts-v1.5 (lowercase path).",
    "Place bpe.model at workdir/models/IndexTTS-1.5/bpe.model or workdir/models/indextts-1.5-onnx/bpe.model.",
    "Install the official frontend dependencies in an isolated environment before running this bridge.",
    "Install the official project's Python requirements, including torch/PyTorch, sentencepiece, cn2an, jieba, pypinyin, WeTextProcessing, and pynini as applicable.",
    "On Windows, pynini/WeTextProcessing wheels are often unavailable; prefer conda-forge on Linux/WSL/conda: conda install -c conda-forge pynini, then pip install WeTextProcessing.",
    "Do not copy model files into the source tree; keep them under workdir/models and write reports under workdir/data.",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_source_root(root: Path | None = None) -> Path:
    root = root or repo_root()
    return root / "workdir" / "models" / "index-tts-v1.5"


def default_bpe_model(root: Path | None = None) -> Path | None:
    root = root or repo_root()
    candidates = [
        root / "workdir" / "models" / "IndexTTS-1.5" / "bpe.model",
        root / "workdir" / "models" / "indextts-1.5-onnx" / "bpe.model",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def default_tagger_cache_dir(root: Path | None = None) -> Path:
    root = root or repo_root()
    return root / "workdir" / "data" / "indextts-official-tagger-cache"


@contextmanager
def official_source_imports_without_bytecode():
    """Import official source without writing __pycache__ into workdir/models."""

    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = previous


def load_common_tokenizer(source_root: Path):
    common_path = source_root / "indextts" / "utils" / "common.py"
    spec = importlib.util.spec_from_file_location("indextts_official_common", common_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official common.py from {common_path}")
    module = importlib.util.module_from_spec(spec)
    with official_source_imports_without_bytecode():
        spec.loader.exec_module(module)
    return module.tokenize_by_CJK_char


def fallback_tokenize_by_cjk_char(line: str, do_upper_case: bool = True) -> str:
    """Local copy of official common.py tokenization for dependency-light dumps."""

    cjk_range_pattern = (
        r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF"
        r"\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])"
    )
    chars = re.split(cjk_range_pattern, line.strip())
    return " ".join([w.strip().upper() if do_upper_case else w.strip() for w in chars if w.strip()])


def load_official_normalizer(source_root: Path, tagger_cache_dir: Path):
    if not (source_root / "indextts").is_dir():
        raise RuntimeError(f"official source package not found under {source_root}")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    with official_source_imports_without_bytecode():
        from indextts.utils.front import TextNormalizer, TextTokenizer  # type: ignore

    normalizer = TextNormalizer()
    if normalizer.zh_normalizer is not None and normalizer.en_normalizer is not None:
        return normalizer, TextTokenizer
    if platform.system() == "Darwin":
        # Mirror the checked local official front.py platform branch exactly:
        # Darwin uses wetext.Normalizer; all non-Darwin platforms use tn.* below.
        from wetext import Normalizer  # type: ignore

        normalizer.zh_normalizer = Normalizer(remove_erhua=False, lang="zh", operator="tn")
        normalizer.en_normalizer = Normalizer(lang="en", operator="tn")
        return normalizer, TextTokenizer

    from tn.chinese.normalizer import Normalizer as NormalizerZh  # type: ignore
    from tn.english.normalizer import Normalizer as NormalizerEn  # type: ignore

    tagger_cache_dir.mkdir(parents=True, exist_ok=True)
    gitignore = tagger_cache_dir / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*\n", encoding="utf-8")
    normalizer.zh_normalizer = NormalizerZh(
        cache_dir=str(tagger_cache_dir),
        remove_interjections=False,
        remove_erhua=False,
        overwrite_cache=False,
    )
    # Keep official non-Darwin English behavior exactly; only the Chinese tn
    # normalizer has the source-local cache_dir override in official front.py.
    normalizer.en_normalizer = NormalizerEn(overwrite_cache=False)
    return normalizer, TextTokenizer


def load_official_frontend(source_root: Path, bpe_model: Path | None, tagger_cache_dir: Path):
    normalizer, tokenizer_cls = load_official_normalizer(source_root, tagger_cache_dir)
    tokenizer = tokenizer_cls(str(bpe_model), normalizer=normalizer) if bpe_model else None
    return normalizer, tokenizer


def load_texts(args: argparse.Namespace) -> list[str]:
    texts: list[str] = []
    if args.text:
        texts.extend(args.text)
    if args.input_json is not None:
        input_json = str(args.input_json)
        if input_json == "-":
            raw = sys.stdin.read()
        elif input_json.lstrip().startswith(("{", "[", '"')):
            raw = input_json
        else:
            raw = Path(input_json).read_text(encoding="utf-8")
        texts.extend(texts_from_json(json.loads(raw.lstrip("\ufeff"))))
    if args.batch_file:
        for line in args.batch_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                texts.append(line)
    return texts or DEFAULT_TEXTS


def texts_from_json(payload: Any) -> list[str]:
    if isinstance(payload, str):
        return [payload]
    if isinstance(payload, list):
        if not all(isinstance(item, str) for item in payload):
            raise ValueError("JSON input list must contain only strings")
        return list(payload)
    if isinstance(payload, dict):
        if isinstance(payload.get("text"), str):
            return [payload["text"]]
        if isinstance(payload.get("texts"), list) and all(isinstance(item, str) for item in payload["texts"]):
            return list(payload["texts"])
    raise ValueError("JSON input must be a string, a list of strings, or an object with text/texts")


def sentence_splits(normalizer: Any, text: str, normalized: str) -> list[str] | None:
    for name in ("split_sentence", "split_sentences", "split_text", "sentence_split"):
        method = getattr(normalizer, name, None)
        if callable(method):
            for value in (normalized, text):
                try:
                    result = method(value)
                except TypeError:
                    continue
                if isinstance(result, str):
                    return [result]
                if isinstance(result, (list, tuple)) and all(isinstance(item, str) for item in result):
                    return list(result)
    return None


def tokenizer_encode(tokenizer: Any, text: str, out_type: type) -> list[Any]:
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text, out_type=out_type))
    if callable(tokenizer):
        result = tokenizer(text)
        if isinstance(result, tuple):
            result = result[0 if out_type is str else -1]
        return list(result)
    raise RuntimeError("official TextTokenizer does not expose encode/call")


def python_rust_fallback_approx(text: str) -> dict[str, Any]:
    """Small Python mirror of the Rust fallback for human comparison only."""

    normalized = normalize_fullwidth_ascii(text).replace("嗯", "恩").replace("呣", "母")
    normalized = normalized.replace("￥", "元").replace("¥", "元").replace("$", "美元")
    for src, dst in [("，", ","), ("。", "."), ("！", "!"), ("？", "?"), ("；", ","), ("：", ","), ("（", "'"), ("）", "'"), ("“", "'"), ("”", "'"), ("·", "-")]:
        normalized = normalized.replace(src, dst)
    tokenized = fallback_tokenize_by_cjk_char(normalized, True)
    return {"normalized": normalized.strip(), "tokenized": tokenized, "tokens": tokenized.split()}


def normalize_fullwidth_ascii(text: str) -> str:
    out = []
    for ch in text:
        code = ord(ch)
        if 0xFF10 <= code <= 0xFF19:
            out.append(chr(ord("0") + code - 0xFF10))
        elif 0xFF21 <= code <= 0xFF3A:
            out.append(chr(ord("A") + code - 0xFF21))
        elif 0xFF41 <= code <= 0xFF5A:
            out.append(chr(ord("a") + code - 0xFF41))
        elif ch in {"＠": "@", "．": ".", "－": "-", "／": "/", "：": ":", "％": "%", "＋": "+", "＄": "$", "　": " "}:
            out.append({"＠": "@", "．": ".", "－": "-", "／": "/", "：": ":", "％": "%", "＋": "+", "＄": "$", "　": " "}[ch])
        else:
            out.append(ch)
    return "".join(out)


def run_rust_frontend(texts: list[str], bpe_model: Path | None, timeout: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = repo_root()
    cmd = [
        "cargo",
        "run",
        "--quiet",
        "-p",
        "local-adapter-index-tts",
        "--bin",
        "indextts_text_frontend",
        "--",
        "--input-json",
        "-",
    ]
    if bpe_model is not None and bpe_model.exists():
        cmd.extend(["--bpe-model", str(bpe_model)])
    proc = subprocess.run(
        cmd,
        input=json.dumps({"texts": texts}, ensure_ascii=False),
        text=True,
        capture_output=True,
        cwd=root,
        timeout=timeout,
        check=False,
    )
    info: dict[str, Any] = {
        "command": cmd,
        "returncode": proc.returncode,
        "stderr": proc.stderr[-4000:],
    }
    if proc.returncode != 0:
        raise RuntimeError(f"Rust frontend exited {proc.returncode}: {proc.stderr[-1000:]}")
    payload = json.loads(proc.stdout)
    rust_texts = payload.get("texts")
    if not isinstance(rust_texts, list):
        raise RuntimeError("Rust frontend JSON did not contain texts list")
    info["available"] = True
    return rust_texts, info


def compare_values(official_item: dict[str, Any], rust_item: dict[str, Any]) -> dict[str, bool | None]:
    comparisons: dict[str, bool | None] = {}
    for key in ("normalized", "tokenized", "token_ids"):
        if key in official_item and key in rust_item:
            comparisons[f"{key}_equal"] = official_item.get(key) == rust_item.get(key)
        else:
            comparisons[f"{key}_equal"] = None
    return comparisons


def build_report(
    texts: list[str],
    source_root: Path,
    bpe_model: Path | None,
    tagger_cache_dir: Path,
    *,
    compare_rust_fallback: bool = False,
    compare_rust_frontend: bool = True,
    rust_timeout: float = 60.0,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    bpe_model = bpe_model.resolve() if bpe_model else None
    tagger_cache_dir = tagger_cache_dir.resolve()
    result: dict[str, Any] = {
        "source_root": str(source_root),
        "bpe_model": str(bpe_model) if bpe_model else None,
        "tagger_cache_dir": str(tagger_cache_dir),
        "dependency_hints": DEPENDENCY_HINTS,
        "texts": [],
        "official_available": False,
        "tokenize_by_cjk_available": False,
        "rust_frontend_available": False,
    }
    if bpe_model is not None and not bpe_model.exists():
        result["bpe_model_error"] = f"bpe.model not found: {bpe_model}"
        bpe_model = None

    try:
        tokenize_by_cjk_char = load_common_tokenizer(source_root)
        result["tokenize_by_cjk_available"] = True
    except Exception as exc:  # pragma: no cover - depends on local official tree
        result["tokenize_by_cjk_error"] = f"{type(exc).__name__}: {exc}"
        tokenize_by_cjk_char = fallback_tokenize_by_cjk_char
        result["tokenize_by_cjk_fallback"] = True

    normalizer = None
    tokenizer = None
    try:
        normalizer, tokenizer = load_official_frontend(source_root, bpe_model, tagger_cache_dir)
        result["official_available"] = True
    except Exception as exc:  # pragma: no cover - optional deps often absent
        result["official_error"] = f"{type(exc).__name__}: {exc}"
        result["official_traceback"] = traceback.format_exc()

    rust_items: list[dict[str, Any]] | None = None
    if compare_rust_frontend:
        try:
            rust_items, rust_info = run_rust_frontend(texts, bpe_model, rust_timeout)
            result["rust_frontend_available"] = True
            result["rust_frontend"] = rust_info
        except Exception as exc:  # pragma: no cover - depends on local Rust toolchain
            result["rust_frontend_error"] = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, subprocess.TimeoutExpired):
                result["rust_frontend_error"] = f"TimeoutExpired: Rust frontend exceeded {rust_timeout:.1f}s"

    summary = {
        "total": len(texts),
        "normalized_equal": 0,
        "tokenized_equal": 0,
        "token_ids_equal": 0,
        "normalized_compared": 0,
        "tokenized_compared": 0,
        "token_ids_compared": 0,
    }

    for idx, text in enumerate(texts):
        item: dict[str, Any] = {"input": text}
        normalized = None
        if normalizer is not None:
            try:
                normalized = normalizer.normalize(text)
                item["normalized"] = normalized
                item["tokenized"] = tokenize_by_cjk_char(normalized)
                item["tokens"] = item["tokenized"].split()
                splits = sentence_splits(normalizer, text, normalized)
                if splits is not None:
                    item["sentence_splits"] = splits
            except Exception as exc:  # pragma: no cover - optional deps
                item["official_normalize_error"] = f"{type(exc).__name__}: {exc}"
        else:
            item["tokenized"] = tokenize_by_cjk_char(text)
            item["tokens"] = item["tokenized"].split()
        if tokenizer is not None:
            try:
                item["tokenized_string_pieces"] = tokenizer_encode(tokenizer, text, str)
                item["token_ids"] = tokenizer_encode(tokenizer, text, int)
                item["token_ids_source"] = "official_python"
            except Exception as exc:  # pragma: no cover - optional deps/model
                item["official_tokenizer_error"] = f"{type(exc).__name__}: {exc}"
        if compare_rust_fallback:
            fallback = python_rust_fallback_approx(text)
            item["rust_fallback_approx"] = fallback
            if normalized is not None:
                item["rust_fallback_matches_official_tokenized"] = fallback["tokenized"] == item.get("tokenized")
        if rust_items is not None and idx < len(rust_items):
            rust_item = rust_items[idx]
            item["rust_frontend"] = rust_item
            comparisons = compare_values(item, rust_item)
            item["parity"] = comparisons
            for key, value in comparisons.items():
                metric = key.removesuffix("_equal")
                if value is not None:
                    summary[f"{metric}_compared"] += 1
                    if value:
                        summary[key] += 1
        result["texts"].append(item)
    result["summary"] = summary
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    return build_report(
        load_texts(args),
        args.source_root,
        None if args.no_tokenizer else args.bpe_model,
        args.tagger_cache_dir,
        compare_rust_fallback=args.compare_rust_fallback,
        compare_rust_frontend=not args.no_rust_frontend,
        rust_timeout=args.rust_timeout,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=default_source_root(root), help="official IndexTTS source root")
    parser.add_argument("--bpe-model", type=Path, default=default_bpe_model(root), help="bpe.model for official TextTokenizer")
    parser.add_argument(
        "--tagger-cache-dir",
        type=Path,
        default=default_tagger_cache_dir(root),
        help="WeTextProcessing/tn cache dir; defaults to workdir/data/indextts-official-tagger-cache",
    )
    parser.add_argument("--no-tokenizer", action="store_true", help="skip official SentencePiece TextTokenizer")
    parser.add_argument("--text", action="append", help="text to dump; repeat for multiple inputs")
    parser.add_argument("--input-json", nargs="?", const="-", default=None, help="JSON string/list/object input path, or '-' / omitted value for stdin")
    parser.add_argument("--batch-file", type=Path, default=None, help="UTF-8 file with one text per non-empty line")
    parser.add_argument("--compare-rust-fallback", action="store_true", help="include a Python mirror of the Rust fallback for rough A/B comparison")
    parser.add_argument("--no-rust-frontend", action="store_true", help="do not run the Rust frontend comparator")
    parser.add_argument("--rust-timeout", type=float, default=60.0, help="timeout in seconds for the non-service Rust frontend dump")
    parser.add_argument("--fail-on-missing", action="store_true", help="exit non-zero when official frontend dependencies/source are unavailable")
    parser.add_argument("--output", type=Path, default=None, help="output JSON path; defaults to workdir/data/indextts-text-parity-<timestamp>.json")
    args = parser.parse_args(argv)
    if args.output is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        args.output = root / "workdir" / "data" / f"indextts-text-parity-{timestamp}.json"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run(args)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    if not result.get("official_available"):
        print("official frontend unavailable; see JSON for dependency/import error and setup hints")
        return 2 if args.fail_on_missing else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
