"""Recompute published measures from archived histories and separated-pair locations."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .data import Archive
from ._core.planar_geometry import microstructure


def boundary_distance(points, seeds):
    """Distance to the nearest Voronoi facet; coordinates share the same length unit."""
    points, seeds = np.asarray(points), np.asarray(seeds)
    dsq = ((points[:, None, :2] - seeds[None]) ** 2).sum(-1)
    order = np.argsort(dsq, axis=1)[:, :2]
    near = np.take_along_axis(dsq, order, axis=1)
    return (near[:, 1] - near[:, 0]) / (
        2 * np.linalg.norm(seeds[order[:, 1]] - seeds[order[:, 0]], axis=1)
    )


def crossing_measure(history, boundary_x):
    initial = np.flatnonzero(history[:, 2] > 0)
    crossed = np.flatnonzero(history[:, 10] >= boundary_x + 2)
    initiation = float(history[initial[0], 1]) if initial.size else None
    propagation = float(history[: crossed[0] + 1, 1].max()) if crossed.size else None
    return dict(
        initiation=initiation,
        propagation=propagation,
        crossing_ratio=propagation / initiation if initiation and propagation else None,
        reached_crossing=bool(crossed.size),
        maximum_load=float(history[:, 1].max()),
    )


def main_crack(events, meta):
    """Archived radius-3 periodic connectivity and pre-crack root rule."""
    xyz = events[:, 3:6].copy()
    xyz[:, :2] -= xyz[:, :2].min(axis=0)
    xyz[:, 2] = np.mod(xyz[:, 2], meta["Lz"])
    pairs = cKDTree(xyz, boxsize=[1e6, 1e6, meta["Lz"]]).query_pairs(
        3.0, output_type="ndarray"
    )
    graph = csr_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(xyz), len(xyz))
    )
    _, labels = connected_components(graph, directed=False)
    root = np.flatnonzero(
        (events[:, 3] < meta["a0"] + 1) & (abs(events[:, 4] - meta["y0"]) < 1)
    )
    mask = labels == labels[root[0]] if root.size else np.zeros(len(xyz), bool)
    return mask & (events[:, 3] > meta["a0"] + 0.5)


def multigrain_measure(history, events, meta, reference=None, scale=1.0):
    selected = main_crack(events, meta)
    points = events[selected, 3:5]
    if not len(points):
        return dict(main_pairs=0, final_load=float(history[-1, 1]), holds=None), points
    start = np.flatnonzero(history[:, 2] > 100)
    valid = np.arange(len(history)) >= (start[0] if start.size else len(history))
    hold = (
        valid
        & (history[:, 3] == 0)
        & (history[:, 10] > 0.15 * meta["W"])
        & (history[:, 10] < meta["W"] - 10)
    )
    out = dict(
        main_pairs=len(points),
        final_load=float(history[-1, 1]),
        holds=int(hold.sum()),
        held_load=float(history[hold, 1].max()) if hold.any() else None,
    )
    if reference is not None:
        limit = min(points[:, 0].max(), reference[:, 0].max())
        distance = cKDTree(reference).query(points[points[:, 0] <= limit])[0]
        out["projected_reference_distance"] = float(scale * distance.mean())
    return out, points


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def reproduce(output, archive=None):
    archive = archive or Archive()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, checks = [], []
    control = archive.json("data/results/bicrystal_controls.json")
    groups = [("reference", "reference", control["reference"])]
    for group in ("crossing", "crack on the junction", "parallel, junction 2 h above"):
        for method, values in control[group].items():
            for value in values if isinstance(values, list) else [values]:
                if value is not None:
                    groups.append((group, method, value))
    for group, method, expected in groups:
        h = archive.array(expected["tag"])
        result = crossing_measure(h, 16.0)
        result.update(
            group=group, construction=method, run=archive.run(expected["tag"])["id"]
        )
        # A remote junction failure is reported as a location event, not a crack-crossing ratio.
        if group.startswith("parallel") and "plain" in method:
            result["crossing_ratio"] = None
            result["interpretation"] = "remote junction separation"
        for key, source_key in [("initiation", "init"), ("propagation", "cross")]:
            target = expected.get(source_key)
            if target is not None and result[key] is not None:
                checks.append(
                    dict(
                        check=f"{result['run']}:{key}",
                        difference=result[key] - target,
                        passed=bool(abs(result[key] - target) < 2e-6),
                    )
                )
        rows.append(result)
    write_csv(output / "bicrystal_measures.csv", rows)
    polyrows = []
    for seed in (1, 2, 3, 4):
        ref = f"pxt0_s{seed}_naive_e19"
        _, points = multigrain_measure(
            archive.array(ref), archive.array(ref, "events"), archive.metadata(ref)
        )
        for method, tag in [
            ("reference", ref),
            ("merging", f"pxtJm_s{seed}_naive_e19"),
            ("ghost", f"pxtEm1_s{seed}_ghost_e19cm025"),
        ]:
            value, _ = multigrain_measure(
                archive.array(tag),
                archive.array(tag, "events"),
                archive.metadata(tag),
                points,
            )
            value.update(seed=seed, resolution="coarse", method=method)
            polyrows.append(value)
            expected = archive.json("data/results/multigrain_controls.json")[
                f"-_s{seed}"
            ]
            key = {
                "reference": "single crystal",
                "merging": "merged, energy-matched",
                "ghost": "coupling, mu 1e-1",
            }[method]
            for field, target in [("final_load", "lam_final"), ("holds", "holds")]:
                if target in expected[key]:
                    checks.append(
                        dict(
                            check=f"multigrain:{seed}:{method}:{field}",
                            difference=value[field] - expected[key][target],
                            passed=bool(
                                abs(value[field] - expected[key][target]) < 2e-6
                            ),
                        )
                    )
    for seed in (1, 2, 3):
        ref = f"pxtR0_s{seed}_naive_e19"
        _, points = multigrain_measure(
            archive.array(ref), archive.array(ref, "events"), archive.metadata(ref)
        )
        for method, tag in [
            ("reference", ref),
            ("ghost", f"pxtREm1_s{seed}_ghost_e19cm025"),
        ]:
            value, _ = multigrain_measure(
                archive.array(tag),
                archive.array(tag, "events"),
                archive.metadata(tag),
                points,
                0.5,
            )
            value.update(seed=seed, resolution="fine", method=method)
            polyrows.append(value)
    write_csv(output / "multigrain_measures.csv", polyrows)
    planar = []
    for run in archive.json("data/bond_softening_runs.json"):
        data = archive.json(run["path"])
        seeds, _, _ = microstructure(data["W"], data["NG"], data["SEED"])
        distances = boundary_distance(np.asarray(data["mid"]), seeds)
        fraction = float((distances < 0.75).mean())
        checks.append(
            dict(
                check=f"{run['id']}:boundary_fraction",
                difference=fraction - run["boundary_fraction"],
                passed=bool(abs(fraction - run["boundary_fraction"]) < 1e-12),
            )
        )
        planar.append(
            dict(
                run=run["id"],
                resolution=run["resolution"],
                angle_deg=run["angle_deg"],
                seed=run["seed"],
                method=run["method"],
                boundary_fraction=fraction,
                double_threshold_fraction=float((distances < 1.5).mean()),
                peak_load=data["peak"],
            )
        )
    write_csv(output / "bond_softening_measures.csv", planar)
    means = {
        method: float(
            np.mean(
                [
                    r["boundary_fraction"]
                    for r in planar
                    if r["method"] == method and r["resolution"] == "coarse"
                ]
            )
        )
        for method in ("reference", "merging", "ghost")
    }
    # Experimental comparisons are re-normalized from raw records, retaining failure-mode groups.
    qa = archive.json("data/results/fe_si_transmission.json")
    experiment_rows = []
    for row in qa["cases"]:
        for method, key in [
            ("ghost", "coupling"),
            ("merging", "merged, energy-matched"),
        ]:
            expected = row[key]
            if not expected:
                continue
            single = qa["single"][str(int(row["Lz"]))]
            measure = crossing_measure(archive.array(expected["tag"]), 16)
            base = crossing_measure(archive.array(single["tag"]), 16)
            normalized = (
                measure["crossing_ratio"] / base["crossing_ratio"]
                if measure["crossing_ratio"]
                else None
            )
            experiment_rows.append(
                dict(case=row["case"], method=method, normalized_crossing=normalized)
            )
            if normalized is not None:
                checks.append(
                    dict(
                        check=f"FeSi:{row['case']}:{method}",
                        difference=normalized - expected["rel_single"],
                        passed=bool(abs(normalized - expected["rel_single"]) < 2e-6),
                    )
                )
    write_csv(output / "fe_si_measures.csv", experiment_rows)
    mo = archive.json("data/results/molybdenum_strength.json")
    strength_rows = []
    base_force = float((0.5 * archive.array(mo["runs"]["single"]["tag"])[:, 20]).max())
    for name, expected in mo["runs"].items():
        force = float((0.5 * archive.array(expected["tag"])[:, 20]).max())
        ratio = force / base_force
        strength_rows.append(
            dict(case=name, maximum_force=force, normalized_strength=ratio)
        )
        checks.append(
            dict(
                check=f"Mo:{name}",
                difference=ratio - expected["strength_ratio"],
                passed=bool(abs(ratio - expected["strength_ratio"]) < 2e-6),
            )
        )
    write_csv(output / "molybdenum_measures.csv", strength_rows)
    with (archive.root / "data/experiments/mo_ikeda_1999_fracture_strength.csv").open(
        encoding="utf-8-sig"
    ) as stream:
        measured = list(
            csv.DictReader(line for line in stream if not line.startswith("#"))
        )
    norm = float(
        np.mean([float(r["sigma_f_MPa"]) for r in measured if float(r["phi_deg"]) == 0])
    )
    measured_rows = []
    for angle in sorted({float(r["phi_deg"]) for r in measured}):
        for mode in ("GB", "TG"):
            values = [
                float(r["sigma_f_MPa"])
                for r in measured
                if float(r["phi_deg"]) == angle and r["mode"] == mode
            ]
            if values:
                measured_rows.append(
                    dict(
                        angle_deg=angle,
                        mode=mode,
                        specimens=len(values),
                        mean_stress_MPa=float(np.mean(values)),
                        normalized_mean=float(np.mean(values) / norm),
                        interpretation=(
                            "lower bound" if mode == "TG" else "boundary fracture"
                        ),
                    )
                )
    write_csv(output / "molybdenum_experiment_groups.csv", measured_rows)
    report = dict(
        checks=checks,
        checks_passed=all(c["passed"] for c in checks),
        coarse_2d_boundary_fraction_means=means,
        scope="Reanalysis of archived records; no full fracture simulation was run.",
    )
    (output / "reproduction_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if not report["checks_passed"]:
        raise RuntimeError(
            "Archived-data comparison failed; see reproduction_report.json"
        )
    return report
