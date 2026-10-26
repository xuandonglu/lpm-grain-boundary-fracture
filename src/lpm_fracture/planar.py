"""Explicit 2D bond-softening runs using the archived numerical implementation."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from ._core import bond_softening_2d as fracture
from ._core.planar_geometry import microstructure
from .analysis import boundary_distance


def simulate(config, output):
    cfg = json.loads(Path(config).read_text(encoding="utf-8"))
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    os.environ["GBMEAN"] = "1"
    os.environ["INTERP"] = "mls"
    method = cfg["method"]
    if method not in {"ghost", "merging", "reference"}:
        raise ValueError("Unknown 2D method")
    width, count, seed = cfg["width"], cfg["grains"], cfg["seed"]
    seeds, theta, shift = microstructure(width, count, seed)
    theta[:] = np.radians(cfg["angle_deg"])
    if method == "reference":
        shift[:] = 0
    kind = "j04" if method in {"merging", "reference"} else "ghost"
    if method == "reference":
        os.environ["GBMEAN"] = "0"
    model = fracture.Frac2D(
        width,
        seeds,
        theta,
        shift,
        fracture.CAL,
        kind,
        a_g=0.001,
        a_gb=0.001,
        nsup=12,
        grip=3,
        notch=0,
    )
    if method == "ghost":
        fracture.wopt(model, damp=cfg.get("weight_damping", 0.03))
        model.enable_gfc()
    model.set_softening(cfg.get("softening_ratio", 10), cfg.get("softening_stages", 6))
    if method == "ghost":
        model.pair_ghosts(mode="mean", rmax=0.6, cosmin=0.9)
    model.relax_R = 0.0
    model.use_vis = method == "ghost"
    model.vis_notch, model.vis_kill = True, True
    model.vis_facet = 0.7
    model.cut_notch(cfg["notch_length"])
    result = fracture.run(model, max_events=cfg.get("max_events", 4000))
    broken = np.asarray(result["broken"], dtype=int)
    midpoints = (
        model.X[model.I[broken]] + 0.5 * model.n[broken] * model.L0[broken, None]
    )
    result["midpoints"] = midpoints.tolist()
    result["configuration"] = cfg
    result["boundary_fraction"] = (
        float((boundary_distance(midpoints, seeds) < 0.75).mean())
        if len(broken)
        else None
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    simulate(args.config, args.output)
