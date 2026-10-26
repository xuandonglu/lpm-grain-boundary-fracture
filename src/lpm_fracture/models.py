"""Configuration-driven model export. Each export runs in its own Python process."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

METHODS = {"ghost", "merging", "reference", "direct_reconnection"}


def export_model(config: str | Path, output: str | Path) -> Path:
    config, output = Path(config).resolve(), Path(output).resolve()
    if output.suffix != ".bin":
        raise ValueError("The exported model must have the .bin suffix")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Material selection in the archived kernels changes module-level constants.
    # A new interpreter prevents one model's material from affecting another.
    subprocess.run(
        [sys.executable, "-m", "lpm_fracture.models", str(config), str(output)],
        check=True,
    )
    return output


def _export(config: Path, output: Path) -> None:
    from ._core import model_export as exporter
    from ._core import energy_matching, polycrystal_geometry
    from ._core import bicrystal_export, tensile_strip_export, material_selection

    cfg = json.loads(config.read_text(encoding="utf-8"))
    method = cfg["method"]
    if method not in METHODS:
        raise ValueError(f"method must be one of {sorted(METHODS)}")
    material = cfg.get("material", "Fe")
    if material not in {"Fe", "Mo"}:
        raise ValueError("Supported publication materials: Fe, Mo")
    # Prevent unrelated caller environment settings from silently changing a case.
    for key in list(os.environ):
        if key.startswith("GH_") or key in {"MAT", "JONLY", "LOAD", "USTRIP"}:
            del os.environ[key]
    if material == "Mo":
        os.environ["MAT"] = "Mo"
        material_selection.use("Mo")
    if method == "ghost":
        os.environ["GH_WOPT"] = str(cfg.get("regularization", 0.1))
    options = dict(cfg.get("geometry", {}))
    family = cfg["family"]
    if "rotation_angle_deg" in options:
        import numpy as np

        angle = float(options.pop("rotation_angle_deg"))
        options["sigma"] = 9000
        exporter.TRIPLES[9000] = (np.sin(np.radians(angle)), np.cos(np.radians(angle)))
        if "Lz" not in options:
            raise ValueError(
                "An arbitrary tilt angle requires an explicit periodic thickness Lz"
            )
    path = str(output)
    if family == "bicrystal":
        if method == "reference":
            options["mode"] = "single"
            options.pop("shift", None)
            exporter.case_bx(path, **options)
        elif method == "ghost":
            exporter.case_bx(path, **options)
        elif method == "merging":
            polycrystal_geometry.set_poly_gb_law = energy_matching._bxn_law
            exporter.case_bxn(path, **options)
        else:
            raise ValueError(
                "Use the parallel_bicrystal family for direct reconnection"
            )
    elif family == "parallel_bicrystal":
        kind = {"ghost": "ghost", "merging": "emc", "direct_reconnection": "plain"}.get(
            method
        )
        if kind is None:
            raise ValueError(
                "Export a bicrystal/reference case for the single-crystal control"
            )
        if cfg.get("loading") == "tension":
            os.environ["LOAD"] = "strip"
            os.environ["USTRIP"] = str(cfg.get("grip_displacement", 3))
        bicrystal_export.export(kind, path, **options)
    elif family in {"polycrystal", "tensile_strip"}:
        if method not in {"ghost", "merging", "reference"}:
            raise ValueError("Unsupported polycrystal construction")
        kind = "ghost" if method == "ghost" else "naive"
        disc = options.pop("template")
        if options:
            raise ValueError(f"Unknown template options: {options}")
        # The public merging baseline always uses exact separation-work matching.
        # The original PX normalization remains internal and is never selected here.
        exporter.disc_gb_g = energy_matching.disc_gb_g_emc
        if family == "tensile_strip":
            tensile_strip_export.case_pxu_t(kind, path, disc)
        else:
            exporter.case_pxk(kind, path, disc)
    else:
        raise ValueError(f"Unknown model family: {family}")
    if cfg.get("initial_stencil"):
        from .stencils import restore_initial_stencil

        restore_initial_stencil(output, cfg["initial_stencil"])
    (output.parent / (output.stem + ".config.json")).write_text(
        json.dumps(cfg, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    _export(Path(sys.argv[1]), Path(sys.argv[2]))
