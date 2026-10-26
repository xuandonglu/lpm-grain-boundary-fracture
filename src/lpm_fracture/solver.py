"""Launch the optional native solver with an explicit, bounded thread count."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def run_solver(executable, model, output, config, threads=2, check=False):
    if not 1 <= threads <= 8:
        raise ValueError("threads must be between 1 and 8")
    cfg = json.loads(Path(config).read_text(encoding="utf-8"))
    parameters = cfg.get("solver", {})
    output = Path(output).resolve()
    if any(output.parent.glob(output.name + "*")):
        raise FileExistsError(f"Output prefix already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("GH_"):
            del env[key]
    env.update(
        {
            key: str(threads)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "GH_THREADS",
            )
        }
    )
    env.update(KMP_BLOCKTIME="0", GH_TN="1", GH_EMODE="1", GH_CAENR="1", GH_CALATCH="0")
    env["GH_CA"] = "1" if cfg["method"] == "ghost" else "0"
    env["GH_CATHR"] = str(parameters.get("partition_threshold", 0.25))
    env["GH_TMAX"] = str(parameters.get("time_limit_seconds", 3600))
    args = [str(Path(executable).resolve()), str(Path(model).resolve()), str(output)]
    if check:
        args += ["checkE", "0.95"]
    else:
        args += [
            str(parameters.get(k, v))
            for k, v in (
                ("start_load", 0.5),
                ("end_load", 2.5),
                ("load_step", 0.01),
                ("fine_step", 0.005),
            )
        ]
        kfield = Path(model).resolve().with_suffix(".kf")
        if kfield.exists():
            args.append(str(kfield))
    with (output.parent / (output.name + ".log")).open("w", encoding="utf-8") as stream:
        subprocess.run(
            args, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True
        )
