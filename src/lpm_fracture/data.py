"""Read the versioned evidence archive without depending on the original checkout."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from .paths import repository_root


class Archive:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root).resolve() if root else repository_root()
        self.runs = self.json("data/cohesive_runs.json")
        self._runs = {r["id"]: r for r in self.runs}
        self._tags = {r["source_tag"]: r for r in self.runs}

    def json(self, relative: str):
        path = self.root / relative
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)

    def run(self, identifier: str) -> dict:
        if identifier in self._runs:
            return self._runs[identifier]
        if identifier in self._tags:
            return self._tags[identifier]
        raise KeyError(f"Unknown archived run: {identifier}")

    def array(self, identifier: str, field: str = "history") -> np.ndarray:
        path = self.root / self.run(identifier)["files"][field]
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as stream:
            lines = [
                line for line in stream if line.strip() and not line.startswith("#")
            ]
        if not lines:
            return np.empty((0, 0))
        # Event logs may acquire trailing diagnostic columns across solver revisions.
        width = min(len(line.split()) for line in lines)
        return np.array([[float(x) for x in line.split()[:width]] for line in lines])

    def metadata(self, identifier: str) -> dict:
        path = self.root / self.run(identifier)["files"]["history"]
        return json.loads((path.parent / "model.json").read_text(encoding="utf-8"))
