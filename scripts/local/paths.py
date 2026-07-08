"""Path helpers for LOCAL scripts."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the repository root from the installed script package location."""
    return Path(__file__).resolve().parents[2]


def resolve_cli_path(path: Path, root: Path) -> Path:
    """Resolve a CLI path relative to the repository root when not absolute."""
    return path.resolve() if path.is_absolute() else (root / path).resolve()
