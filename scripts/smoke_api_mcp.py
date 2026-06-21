#!/usr/bin/env python3
"""Compatible thin entrypoint for the Python-only API/MCP smoke harness."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lcoal.smoke import main


if __name__ == "__main__":
    sys.exit(main())
