"""Locate read-only evidence and geometry inputs without machine-specific paths."""

from __future__ import annotations

import gzip
import os
from pathlib import Path


def repository_root() -> Path:
    """Return the source checkout, or an explicitly selected data checkout."""
    configured = os.environ.get("LPM_FRACTURE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    candidates = [Path(__file__).resolve().parents[2], Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        if (candidate / "data").is_dir() and (candidate / "provenance").is_dir():
            return candidate
    raise FileNotFoundError(
        "Cannot find the data checkout. Set LPM_FRACTURE_ROOT to its directory."
    )


def geometry_file(name: str) -> Path:
    """Resolve one geometry template, including its optional gzip suffix."""
    if Path(name).name != name:
        raise ValueError("A geometry template must be a filename, not a path.")
    directory = Path(
        os.environ.get("LPM_GEOMETRY_DIR", repository_root() / "data/geometry")
    )
    path = directory / name
    if not path.exists() and path.with_suffix(path.suffix + ".gz").exists():
        path = path.with_suffix(path.suffix + ".gz")
    return path


def open_geometry(name: str, mode: str = "r"):
    """Open immutable JSON/text geometry input; generated models belong in outputs/."""
    if mode not in {"r", "rt"}:
        raise ValueError("Packaged geometry is read-only.")
    path = geometry_file(name)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")
