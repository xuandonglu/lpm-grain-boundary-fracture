"""Regenerate the release integrity manifest after an intentional source/data update."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
excluded = {".git", ".venv", "venv", "__pycache__", "build", "dist", "outputs"}
manifest = root / "provenance/checksums.json"
values = {}
for path in sorted(root.rglob("*")):
    if not path.is_file() or path == manifest:
        continue
    rel = path.relative_to(root)
    if any(part in excluded or part.endswith(".egg-info") for part in rel.parts):
        continue
    values[rel.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
manifest.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8", newline="\n")
print(f"Recorded {len(values)} file hashes")
