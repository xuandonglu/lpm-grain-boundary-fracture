# Release verification

Prepared on 2026-10-25 from the author's saved Paper B v4. Manuscript snapshot SHA256:

`430c0a7918a293d20895e4faac8208f0ec7a69e00bb0609fd0a26013573ad1d5`

The manuscript and original scientific source files were not changed by packaging. This package reorganizes the scientific routines, adds portable inputs and public commands, and curates the evidence used by that snapshot.

## Completed checks

| Check | Result |
|---|---|
| Scientific module imports | All extracted numerical modules and independent continuum modules import |
| Archived-data reanalysis | 123 comparisons passed, including corrected initiation, coarse multigrain histories, 36 planar cases, Fe–Si normalization and Mo force maxima |
| Moment preservation | Maximum affine-moment error 1.33 × 10⁻¹⁵ under a nullspace correction |
| Small ghost model against the original exporter | Binary SHA256 identical before any archived-stencil restoration |
| Main bicrystal ghost and Fe–Si 3° models | All decoded fields match the archived model, maximum difference 6.94 × 10⁻¹⁸, with explicitly restored initial stencils |
| Energy-matched bicrystal and multigrain models | Geometry, material, constraints, bond data and pair groups match after canonical connection ordering; maximum field difference 2.22 × 10⁻¹⁵ |
| Mo ghost model | Exact-angle geometry and initial stencil restoration reproduce all decoded archived fields exactly |
| Model export paths | Small ghost, single crystal, energy-matched bicrystal, energy-matched multigrain and Mo exports completed |
| Invalid stencil/model pairing | Rejected before modifying the model |
| Native solver build | Windows GNU-target Zig/Clang-compatible build succeeded with external oneMKL/OpenMP |
| Native energy check | 4,800 particles, 64,580 half-bonds; tangent-action relative discrepancy 2.564 × 10⁻¹⁰; directional energy-gradient discrepancy 1.16 × 10⁻¹⁰ |
| 2D executable path | Eight-event ghost/softening/visibility/pairing smoke calculation completed |
| Unit tests | Three tests passed: corrected low-start values, continuum energy limits and complete planar cohort |
| Numerical result plotting | Eleven PNG/SVG panel pairs regenerated; all PNG panels visually inspected |
| Packaging/install | Editable installation and analysis tested from a relocated checkout in a separate virtual environment, with isolated Python |
| Source layout | Python sources normalized with Black; source provenance and release file hashes supplied |
| Public contents | Text and decompressed JSON scanned for personal absolute paths, credential patterns, malformed JSON and oversized files |

## What was not rerun

Full paper-scale fracture histories, parameter sweeps and independent continuum propagation studies were not rerun during packaging. Their selected raw records are included. Runtime/cost tables remain measurements of the archived runs. The CMake build instructions have not been independently tested on another operating system or a fresh oneAPI installation.

The relocated virtual environment used the installed scientific dependencies from the recorded local runtime. It verifies installation and independence from the original research paths; it is not a fresh network installation on another operating system. Exact tested versions are in `provenance/test_environment.json`.

## Dependency-sensitive ordering

Distance-query ties can change connection/support ordering across versions. Merged models were checked by physical connection keys and pair-group equivalence. For ten archived ghost examples, the configuration explicitly uses a saved initial stencil and its derived protection flags, verified against a physical-model fingerprint. Modified geometries can instead refit weights by removing `initial_stencil`.

These choices support both exact archive reconstruction and new model construction while keeping the two operations visible in the configuration. Fracture evolution after initialization is computed by the solver.

