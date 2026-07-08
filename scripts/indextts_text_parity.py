#!/usr/bin/env python3
"""Thin entry for IndexTTS 1.5 official-vs-Rust text frontend parity dumps."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local.indextts_text_parity import main


if __name__ == "__main__":
    sys.exit(main())
