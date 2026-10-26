"""Portable integrity checks with no fracture solve."""

import hashlib
import importlib
import json
from pathlib import Path
import tempfile

import numpy as np

from .data import Archive
from .analysis import reproduce
from ._core.weight_optimization import moment_rows


def validate():
    archive = Archive()
    manifest_path = archive.root / "provenance/checksums.json"
    verified = 0
    if manifest_path.exists():
        for name, expected in json.loads(
            manifest_path.read_text(encoding="utf-8")
        ).items():
            actual = hashlib.sha256((archive.root / name).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"File checksum mismatch: {name}")
            verified += 1
    for path in (Path(__file__).parent / "_core").glob("*.py"):
        importlib.import_module("lpm_fracture._core." + path.stem)
    rng = np.random.default_rng(7)
    cloud = rng.normal(size=(18, 3))
    moments = moment_rows(cloud, 1)
    weights = moments.T @ np.linalg.solve(moments @ moments.T, np.array([1, 0, 0, 0]))
    _, _, vt = np.linalg.svd(moments)
    adjusted = weights + vt[4:].T @ rng.normal(size=14)
    error = float(abs(moments @ adjusted - np.array([1, 0, 0, 0])).max())
    if error > 1e-12:
        raise ValueError("Moment-nullspace consistency check failed")
    with tempfile.TemporaryDirectory(prefix="lpm-reproduction-") as folder:
        report = reproduce(folder, archive)
    return dict(
        files_verified=verified,
        moment_error=error,
        archived_checks=len(report["checks"]),
        archived_checks_passed=report["checks_passed"],
        full_simulations_rerun=False,
    )
