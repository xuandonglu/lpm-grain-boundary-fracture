"""Restore archived initial ghost stencils for identical physical model geometry.

Nearest-neighbor ties can select different support sets across SciPy versions.
These optional, versioned templates preserve the supports and fitted weights
used by the archived calculation. Crack-aware updates remain solver operations.
"""

import hashlib
from pathlib import Path

import numpy as np

from ._core.binary_model import read
from .paths import repository_root


def physical_fingerprint(model):
    digest = hashlib.sha256()
    for key in (
        "X",
        "fixed",
        "ubc",
        "owner",
        "shell",
        "pair",
        "flag",
        "n",
        "d0",
        "kn",
        "xi",
        "sptr",
        "gmul",
        "hdr",
    ):
        value = np.asarray(model[key])
        if key == "flag":
            # Protection uses the selected far support; it is restored with the
            # initial stencil. Ghost and pre-cut bits still enter the fingerprint.
            value = value & np.uint8(5)
        if value.dtype.kind == "f":
            value = np.round(value, 12)
            value[value == 0] = 0.0
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def restore_initial_stencil(model_path, template):
    model = read(model_path)
    path = repository_root() / template
    with np.load(path, allow_pickle=False) as saved:
        if str(saved["physical_fingerprint"]) != physical_fingerprint(model):
            raise ValueError(
                "Archived stencil does not match this physical model. Remove initial_stencil from the configuration to refit the modified model."
            )
        indices, weights = saved["indices"], saved["weights"]
        flags, cohesive = saved["flags"], saved["cohesive"]
    if len(indices) != model["nsup"] or len(weights) != model["nsup"]:
        raise ValueError("Archived stencil has incompatible sparse dimensions")
    # See binary_model.read: skip all fields preceding the support-index array.
    offset = 64 + (16 if model["ver"] == 2 else 8) * 8
    offset += 51 * model["N"] + 61 * model["nb"] + 8 * (model["nb"] + 1)
    with Path(model_path).open("r+b") as stream:
        flag_offset = 64 + (16 if model["ver"] == 2 else 8) * 8
        flag_offset += 51 * model["N"] + 12 * model["nb"]
        stream.seek(flag_offset)
        stream.write(np.asarray(flags, dtype=np.uint8).tobytes())
        stream.seek(offset)
        stream.write(np.asarray(indices, dtype=np.int32).tobytes())
        stream.write(np.asarray(weights, dtype=np.float64).tobytes())
        if model["ver"] == 2:
            stream.write(np.asarray(cohesive, dtype=np.uint8).tobytes())
