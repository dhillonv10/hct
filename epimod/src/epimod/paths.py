"""Filesystem layout. Everything large lives outside the repository."""
from __future__ import annotations

import os
from pathlib import Path


def data_root() -> Path:
    """Root for downloaded data and cached arrays (override with EPIMOD_DATA)."""
    root = Path(os.environ.get("EPIMOD_DATA", Path.home() / ".cache" / "epimod"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def subdir(*parts: str) -> Path:
    p = data_root().joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p
