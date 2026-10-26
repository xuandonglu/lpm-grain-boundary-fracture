# Grain-boundary fracture with rotated lattice particles

Code and data for **“Grain-lattice junction artifacts in polycrystal fracture and a crack-aware coupling”** (manuscript, 2026).

The implementation connects independently rotated grain lattices using ghost interpolation, energy-derived reactions, regularized weights and crack-aware jump enrichment. The primary 3D comparison is separation-work-matched particle merging. A separate 2D implementation uses bond softening, paired damage and a local force correction.

[Reproduction guide](docs/REPRODUCTION.md) · [Data dictionary](docs/DATA.md) · [Native solver build](docs/BUILD.md) · [Verification record](docs/VERIFICATION.md)

## Quick start

Use Python 3.10 or later, from the repository root:

```sh
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -e .
python -m lpm_fracture validate
python -m lpm_fracture reproduce --output outputs/reproduced
python -m lpm_fracture plot --output outputs/figures
python -m unittest discover -s tests
```

These commands check and reanalyze the included records and draw numerical result panels. They do not launch new fracture simulations. Work from this source checkout; the versioned `data/` archive is intentionally separate from the Python wheel.

## Contents

| Directory | Contents |
|---|---|
| `src/lpm_fracture/` | Python geometry, interpolation, calibration, model export and analysis |
| `src/native/` | C cohesive-fracture solver and PARDISO wrapper |
| `examples/` | Explicit model configurations and run settings |
| `data/raw/` | 105 selected 3D histories/event sets and 36 2D result files |
| `data/geometry/` | Compressed geometry templates used for model regeneration |
| `data/results/` | Curated numerical summaries, including corrected low-start records |
| `data/tables/` | All 14 tables from the saved manuscript used for this package |
| `data/experiments/` | Attributed transcribed/digitized experimental values |
| `data/continuum/` | Independent continuum comparison outputs |
| `figures/` | All 35 manuscript panels, editable diagrams and their index |
| `provenance/` | Source mapping, hashes and tested environment |

The descriptive public API is in `models.py`, `solver.py`, `analysis.py` and `cli.py`. Numerical kernels are grouped by topic under `_core/`; internal legacy keys remain where required to read archived records. Original source-file mappings are in `provenance/source_files.json`.

## Generate a model

```sh
python -m lpm_fracture export --config examples/smoke_ghost.json --output outputs/smoke/model.bin
```

This creates a small check model and its sidecars. To calculate a load history, first build the native solver using [BUILD.md](docs/BUILD.md), then run it explicitly:

```sh
python -m lpm_fracture solve --config examples/smoke_ghost.json --solver build/cohesive_solver --model outputs/smoke/model.bin --output outputs/smoke/check --threads 2 --check
```

On Windows the executable is normally `build/cohesive_solver.exe`. Remove `--check` only when a new load-history calculation is intended. Larger example cases require substantially more time and memory. Use one run at a time; `--threads` is capped at eight.

## Evidence and scope

Transparent controls compare junction constructions against an orientation-matched single crystal. The independent continuum calculations use their own continuum discretization. Fe–Si and Mo comparisons retain the experimental orientation and failure-mode information described in [DATA.md](docs/DATA.md).

Current primary summaries use the corrected low-start initiation records and the energy-matched 3D merging baseline. Superseded low-start comparisons are explicitly labeled in the correction audit. Withdrawn PX-calibrated fracture results and regression-based experimental extrapolations are excluded from the primary archive. Refinement is in plane, and projected path distance is a one-way nearest-point measure.

## Citation and licensing

Citation metadata is supplied in [CITATION.cff](CITATION.cff). The associated manuscript is unpublished; no DOI is asserted. Experimental data sources are cited in [NOTICE.md](NOTICE.md) and the CSV headers. Article PDFs, proprietary libraries and compiler runtimes are not bundled.

**An open-source license has not yet been selected.** This package does not grant a redistribution or reuse license. Source and data attributions remain applicable.

