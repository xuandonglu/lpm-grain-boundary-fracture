# Reproduction guide

Run commands from the repository root after `python -m pip install -e .`. To use the archive with a non-editable installation, set `LPM_FRACTURE_ROOT` to the checkout containing `data/` and `provenance/`.

## 1. Reanalyze archived results

```sh
python -m lpm_fracture validate
python -m lpm_fracture reproduce --output outputs/reproduced
python -m lpm_fracture plot --output outputs/figures
```

Reanalysis writes CSVs for bicrystal initiation/propagation, multigrain path distance and holding, planar boundary-following fractions, Fe–Si crossing measures and Mo strengths. Mo experimental stresses are regrouped by angle and failure mode, then normalized by the measured single-crystal mean. `reproduction_report.json` lists every numerical comparison against the archived summaries. A failed comparison returns a nonzero exit status.

`validate` also checks the versioned file hashes, imports the kernels and verifies moment preservation under a nullspace correction. It writes temporary analysis outputs outside the archive and removes those temporary outputs when finished.

### Manuscript coverage

| Evidence | Bundled inputs and analysis |
|---|---|
| Tables 1–2, single junction | Raw histories/events, corrected low-start records, `bicrystal_measures.csv` |
| Tables 3–4, transparent multigrain | Coarse/fine histories, geometry and projected-path/holding reanalysis |
| Table 5, single-junction in-plane refinement | Exact table and archived refinement histories/templates; no full rerun in release QA |
| Table 6, tensile initiation | Histories/events and curated initiation summary |
| Tables 7–8, 2D softening | All 36 retained result files, both distance thresholds recalculated |
| Table 9, Fe–Si | Measured Table 2 transcription, model inputs and normalized crossing reanalysis |
| Table 10, Mo | Digitized strength/energy inputs, full retained histories, force-maximum normalization |
| Tables 11–12, weights/cost | Exact tables and archived cost/solver summaries; timing is a recorded measurement |
| Tables 13–14, support controls | Exact tables, selected original diagnostic histories and method configurations |
| Independent continuum checks | Python implementation and archived cohesive/Griffith histories, verification outputs |
| Figures 1–17 | All 35 exact reference panels; editable source files where located; numerical replot command |

Exact manuscript tables are exports of the author's saved v4 identified by SHA256 in `data/manuscript_tables.json`. Table exports and summaries are reference evidence; only measures explicitly calculated by the reanalysis code are described as recomputed. The numerical plot command generates clean stand-alone panels and does not rebuild Word layouts, schematic diagrams or every manuscript figure pixel for pixel.

## 2. Regenerate 3D models

```sh
python -m lpm_fracture export --config examples/bicrystal_ghost.json --output outputs/bicrystal_ghost/model.bin
python -m lpm_fracture export --config examples/bicrystal_merging.json --output outputs/bicrystal_merging/model.bin
python -m lpm_fracture export --config examples/bicrystal_reference.json --output outputs/bicrystal_reference/model.bin
```

Every export uses a fresh interpreter. This isolates the material-specific elastic and surface-energy constants used by the archived kernels. Output files are not overwritten. Sidecars include geometry metadata, the exact public configuration, grain identifiers where relevant and a K-field file for crack-tip loading.

Selected archived ghost configurations also specify `initial_stencil`: the exact original support indices, fitted weights and support-derived boundary protection, stored separately from the solver. This resolves equidistant-neighbor choices that can vary across SciPy versions. A physical-model fingerprint verifies geometry, interactions, material parameters, prescribed degrees of freedom and pre-cut/ghost flags before applying the template. When changing a model, remove this option to fit fresh weights; a mismatching archived template raises an error. Crack-aware support updates during fracture are still calculated by the solver.

`ghost` selects regularized interpolation. `merging` selects separation-work-matched reconnection in 3D. `reference` uses the specified single-crystal orientation, or the explicitly supplied single-crystal multigrain template. `direct_reconnection`, available for the parallel-bicrystal family, selects the secondary uncalibrated reconnection control. Geometry templates encode the orientations, shifts and periodic thickness of the retained cases.

The small `smoke_ghost.json` is a software check model. The `bicrystal_*`, `multigrain_*`, `fe_si_*` and `molybdenum_*` configurations represent the retained study settings. The `solver` section sets the load interval and a wall-time budget. Actual complete archived records may have used a longer budget or restarts; a run ending at its time limit is not a completed fracture result.

## 3. Run the native solver explicitly

After [building](BUILD.md), substitute the executable location below:

```sh
python -m lpm_fracture solve --config examples/bicrystal_ghost.json --solver build/cohesive_solver --model outputs/bicrystal_ghost/model.bin --output outputs/bicrystal_ghost/run --threads 2
```

Use `.exe` on Windows when needed. The launcher passes the model's `.kf` sidecar for K loading; a tensile-strip model has prescribed grip displacements instead. It selects the energy gradient, truncated Newton solver, jump enrichment and partition threshold from the public configuration. The default latch is off, consistent with the retained production branch. `--check` requests a finite-difference gradient/tangent check instead of a load history.

The executable retains historical `GH_*` options for numerical compatibility. Public launches clear inherited `GH_*` settings before specifying the selected options, so unrelated shell settings do not alter the case. The launcher accepts one to eight threads. Coordinate multiple processes to respect the total available thread budget.

The nonlinear fracture branch can depend on floating-point libraries, tolerances, load backoffs and separation ordering. Archive reanalysis is deterministic; a new nonlinear simulation should be assessed through the reported measures and stopping criteria.

## 4. Run a 2D softening case explicitly

```sh
python -m lpm_fracture.planar --config examples/planar_coarse_angle_17_s1_ghost.json --output outputs/planar_case.json
```

The 36 `planar_*` configurations give the corresponding size, seed, common orientation, notch, softening stages and weight damping. The ghost construction uses the archived visibility rule, reverse-half-bond pairing and local force correction. The 2D merging control uses the original distance-based construction. This branch has its own formulation and does not apply the 3D separation-work matching.

Coarse width is 64 lattice spacings, fine width is 128; the physical specimen and grain pattern are matched by scaling. The archived law keeps its nominal critical elongation ratio as spacing changes. As a result, the refinement comparisons report the law's observed sensitivity rather than a separately recalibrated continuum fracture-energy limit.

## 5. Independent continuum calculations

```sh
python -m lpm_fracture.continuum.cohesive_reference --case bicrystal --h 0.5 --front-end 24
python -m lpm_fracture.continuum.griffith_reference --case polycrystal --h 1 --advance 2 --tip-end 118
```

These are optional, potentially expensive computations; they are not executed by `validate` or `reproduce`. Set `LPM_CONTINUUM_OUTPUT` to change the output directory. The cohesive model follows a prescribed straight opening plane. The Griffith model compares candidate crack extensions through continuum elastic energy and directional separation work. Both load the included material/K-field metadata through the archive, without using the original workspace.

## 6. Updating the release

Change a record only with its source and reason documented. Regenerate `provenance/checksums.json` after an intentional update and rerun validation. Keep missing crossings as missing values with their run status. Preserve both favorable and unfavorable observations. Exact table/figure exports should be refreshed if the saved manuscript changes.

