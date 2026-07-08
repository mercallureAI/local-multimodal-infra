#!/usr/bin/env python3
"""Thin entrypoint for preparing LOCAL IndexTTS ONNX artifacts."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local.indextts_export import main


if __name__ == "__main__":
    raise SystemExit(main())
