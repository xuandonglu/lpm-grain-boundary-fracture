"""Compact numerical result panels, generated solely from the bundled evidence."""

from pathlib import Path
import csv
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import Archive

COLORS = {"reference": "#333333", "merging": "#C96736", "ghost": "#2166AC"}
LABELS = {
    "reference": "Single crystal",
    "merging": "Energy-matched merging",
    "ghost": "Ghost coupling",
}


def plot_results(output, archive=None):
    archive = archive or Archive()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "mathtext.fontset": "stix",
            "legend.fontsize": 9,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.1,
        }
    )
    files = []

    def finish(fig, ax, name):
        ax.tick_params(direction="in", top=True, right=True)
        ax.legend(loc="best", framealpha=0.92, edgecolor="none")
        fig.tight_layout(pad=0.2)
        for ext in ("png", "svg"):
            path = output / (name + "." + ext)
            fig.savefig(path, dpi=250, bbox_inches="tight", pad_inches=0.025)
            files.append(path.name)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    for method, tag in [
        ("reference", "sc13r_e18"),
        ("merging", "shRm1_e19"),
        ("ghost", "shRW1_e19cm025"),
    ]:
        h = archive.array(tag)
        ax.plot(h[:, 10], h[:, 1], color=COLORS[method], label=LABELS[method])
    ax.axvline(16, color="0.65", ls="--", lw=0.7)
    ax.set(xlabel=r"Crack-tip position $x/h$", ylabel=r"Load $\lambda$", xlim=(10, 48))
    finish(fig, ax, "bicrystal_crossing")

    curves = archive.json("data/results/junction_traction_curves.json")
    for quantity in ("traction",):
        fig, ax = plt.subplots(figsize=(4.8, 3.1))
        keys = [
            ("cross/zero/plain", "Bulk pair response", "#333333"),
            ("cross/s1/plain", "Direct reconnection", "#9D9D9D"),
            ("cross/s1/emc", "Energy-matched merging", COLORS["merging"]),
        ]
        for key, label, color in keys:
            ax.plot(curves["delta"], curves["T"][key], label=label, color=color)
        if not ax.lines:
            plt.close(fig)
            continue
        ax.set(xlabel=r"Opening $\delta/h$", ylabel=quantity.capitalize())
        finish(fig, ax, "junction_" + quantity)

    for seed in (1, 2, 3, 4):
        fig, ax = plt.subplots(figsize=(5.0, 2.7))
        for method, tag in [
            ("reference", f"pxt0_s{seed}_naive_e19"),
            ("merging", f"pxtJm_s{seed}_naive_e19"),
            ("ghost", f"pxtEm1_s{seed}_ghost_e19cm025"),
        ]:
            e = archive.array(tag, "events")
            meta = archive.metadata(tag)
            ax.scatter(
                e[:, 3],
                e[:, 4] - meta["y0"],
                s=2,
                color=COLORS[method],
                label=LABELS[method],
            )
        ax.set(xlabel=r"$x/h$", ylabel=r"$(y-y_0)/h$", xlim=(0, 128))
        finish(fig, ax, f"multigrain_paths_s{seed}")

    for seed in (1, 3):
        fig, ax = plt.subplots(figsize=(4.8, 3.1))
        for method, tag in [
            ("reference", f"pxtR0_s{seed}_naive_e19"),
            ("ghost", f"pxtREm1_s{seed}_ghost_e19cm025"),
        ]:
            h = archive.array(tag)
            ax.plot(0.5 * h[:, 10], h[:, 1], color=COLORS[method], label=LABELS[method])
        ax.set(
            xlabel=r"Crack-tip position $x/h$ (coarse $h$)", ylabel=r"Load $\lambda$"
        )
        finish(fig, ax, f"refined_load_history_s{seed}")

    qa = archive.json("data/results/fe_si_transmission.json")
    for family, component in (("tilt", "psi"), ("twist", "phi")):
        fig, ax = plt.subplots(figsize=(4.8, 3.1))
        for method, key in (
            ("ghost", "coupling"),
            ("merging", "merged, energy-matched"),
        ):
            rows = [
                r
                for r in qa["cases"]
                if family in r["case"].lower()
                and r[key]
                and r[key].get("rel_single") is not None
            ]
            rows.sort(key=lambda r: r[component])
            ax.plot(
                [r[component] for r in rows],
                [r[key]["rel_single"] for r in rows],
                "o-",
                ms=4,
                color=COLORS[method],
                label=LABELS[method],
            )
        ax.axhline(1, color="0.6", lw=0.7)
        ax.set(
            xlabel=f"{family.capitalize()} angle (deg)",
            ylabel="Normalized crossing measure",
        )
        finish(fig, ax, "fe_si_" + family)

    mo = archive.json("data/results/molybdenum_strength.json")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    exp_path = archive.root / "data/experiments/mo_ikeda_1999_fracture_strength.csv"
    with exp_path.open(encoding="utf-8-sig") as stream:
        exp = list(csv.DictReader(line for line in stream if not line.startswith("#")))
    single = float(
        np.mean([float(r["sigma_f_MPa"]) for r in exp if float(r["phi_deg"]) == 0])
    )
    for mode, marker in (("GB", "o"), ("TG", "^")):
        rows = [r for r in exp if r["mode"] == mode]
        ax.scatter(
            [float(r["phi_deg"]) for r in rows],
            [float(r["sigma_f_MPa"]) / single for r in rows],
            marker=marker,
            s=23,
            edgecolors="0.25",
            facecolors="white",
            label="Experiment: " + mode,
        )
    for suffix, label, color, ls in [
        ("coupling", "Ghost coupling", COLORS["ghost"], "-"),
        ("merged reg0", "Merging: registration 0", COLORS["merging"], "-"),
        ("merged reg1", "Merging: registration 1", COLORS["merging"], "--"),
    ]:
        rows = sorted(
            [v for k, v in mo["runs"].items() if k.endswith(suffix)],
            key=lambda r: r["psi"],
        )
        ax.plot(
            [r["psi"] for r in rows],
            [r["strength_ratio"] for r in rows],
            "o",
            ls=ls,
            ms=4,
            color=color,
            label=label,
        )
    ax.set(
        xlabel="Misorientation (deg)",
        ylabel="Normalized fracture strength",
        ylim=(0.2, 1.55),
    )
    finish(fig, ax, "molybdenum_strength")
    (output / "index.json").write_text(
        json.dumps(files, indent=2) + "\n", encoding="utf-8"
    )
    return files
