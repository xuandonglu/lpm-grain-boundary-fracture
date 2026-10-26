# Data dictionary and provenance

## Archive organization

`data/cohesive_runs.json` maps a descriptive `id` to an original `source_tag`, context and compressed raw files. Original tags serve only as stable provenance identifiers. `scope: appendix` marks diagnostic controls. Other retained records support the main comparisons or their correction audit. `data/bond_softening_runs.json` describes all 36 planar results by resolution, common angle, seed and construction.

`data/results/` contains curated summaries. Numerical summaries retain the original field names to support exact cross-checking; missing/nonfinite summary values are represented by JSON `null`. The low-start correction audit explicitly labels its original and corrected runs. Primary bicrystal summaries use the corrected results.

`provenance/data_files.json` lists each source path relative to the original research checkout, its SHA256, the portable destination, destination SHA256 and any transformation. A provenance path documents origin and is never used as a runtime dependency. `provenance/checksums.json` covers release files except itself and ignored generated outputs.

## 3D history columns

Compressed `history.txt.gz` files retain the original header and numerical rows. The following indices are zero-based:

| Index | Field | Meaning |
|---:|---|---|
| 0 | `step` | Accepted loading step |
| 1 | `lam` | Applied load factor λ |
| 2 | `nsep` | Cumulative separated pairs |
| 3 | `new` | Newly separated pairs in this step |
| 4–5 | `its`, `err` | Solver iteration/residual record |
| 6 | `energy` | Model energy at the accepted state |
| 7–8 | `backoffs`, `newton_total` | Load backoffs and iteration work |
| 9 | `t` | Elapsed wall time, seconds |
| 10–11 | `tipx`, `tipy` | Main-crack tip coordinates in that model's lattice units |
| 12–13 | `kxc`, `kyc` | K-field center |
| 14–16 | `recentres`, `nmain`, `nclus` | Recentering and crack-cluster diagnostics |
| 17–19 | `Rx`, `Ry`, `Rz` | Net prescribed-particle reaction components |
| 20 | `sum|R|` | Sum of reaction magnitudes; half this sum is the strip force used for Mo |

The header in a file is authoritative for its solver revision. `events.txt.gz` columns are `step, lam, pair, xm, ym, zm, shell, ghost, owner`: separation step, load, pair identifier, midpoint coordinates, shell, ghost flag and owning particle. Events record separated pairs rather than a reconstructed smooth crack surface.

`solver_summary.txt` retains setup and final diagnostic lines without personal path values. `model.json` gives geometry/material/loading metadata. `.kf` contains the K normalization, original crack center and complex Stroh-field coefficients. It is unnecessary for grip-loaded strips. Some multigrain metadata are taken from the corresponding geometry template, as recorded in provenance.

## Measures and units

- Lengths use the model's lattice spacing. A fine in-plane model uses twice as many lattice intervals in each in-plane direction; reported path distances convert to the coarse spacing where specified. Periodic thickness is recorded separately.
- Elastic constants are in GPa. Surface and boundary energy inputs are in J/m² before their recorded lattice normalization. K-field factors are dimensionless relative to the model's cleavage normalization; see `K0_GPa_sqrt_h` in the template.
- The bicrystal crossing measure is the largest load reached by the first tip arrival at `xg + 2`, divided by the first recorded separation load. A remote junction failure is separately identified; its load ratio is not interpreted as main-crack crossing.
- Multigrain cracks use the archived periodic radius-3 midpoint connectivity and pre-crack root rule. `d_ref` is a **one-way nearest-point distance in projected xy coordinates**, after limiting to the common attained x range.
- Held steps use the original no-new-separation criterion after the crack has grown, with the specified interior tip window. Held-load excess uses the matched single-crystal final load as denominator.
- Mo numerical strength is the maximum half-sum of prescribed-particle reaction magnitudes, normalized by that model's single-crystal maximum. Experimental strength is normalized by the measured single-crystal mean, with boundary and intragranular failures kept distinct.

## 2D files

Each `data/raw/bond_softening/*.json.gz` file contains geometry settings (`W`, `NG`, `SEED`, `AR`, `notch`), interpolation/softening options, peak load, the saved decimated load history and every separated-bond midpoint in `mid`. Recompute geometric boundary-following fractions from these midpoints and the reproducible Voronoi seeds. `ig_geom` uses distance < 0.75 lattice spacings; reanalysis also reports distance < 1.5. The same nominal Voronoi boundaries are used when evaluating the single-crystal chance fraction.

The saved `hist` is decimated by the original solver and is not a complete event log. The archived `peak` is the original complete-run maximum. Unavailable metadata are encoded as `null` in the portable JSON files; numerical arrays and finite values retain their original values.

## Experimental sources and comparison scope

`fe_si_qiao_argon_2003.csv` transcribes 17 measured tilt/twist pairs and normalized crossing toughnesses from Qiao–Argon Table 2. The model families are idealized pure tilt or twist configurations. The measured specimens generally combine both components. The table's orientation components remain available for interpretation.

`mo_ikeda_1999_fracture_strength.csv` digitizes the purified 77 K data from Ikeda et al. Fig. 8. `GB` means grain-boundary fracture. `TG` means intragranular fracture, making the associated boundary strength a lower bound. The numerical model uses a tensile strip, while the experimental strength was measured in four-point bending. The boundary-energy digitization supplies the energy-based model input; its `kind` column distinguishes measured/calculated source series and encodes the value's units.

The data package preserves these comparison scopes. It does not reinterpret an idealized orientation family or a normalized strength comparison as an exactly reconstructed experimental specimen. Source references and digitization uncertainties are in [NOTICE.md](../NOTICE.md) and the CSV headers.

## Geometry and binary model format

`data/geometry/*.json.gz` stores dimensions, seeds, rotations, shifts, material-law metadata and source calibration. Companion text templates contain particle/interaction information consumed by `model_export.py`. These are read through `paths.open_geometry`, so no uncompressed source folder is required.

`data/stencils/*.npz` contains exact initial ghost-support indices, fitted weights and support-derived boundary-protection/cohesive flags for ten archived example models. Each file includes a SHA256 fingerprint of physical geometry, interactions, parameters, prescribed degrees of freedom and pre-cut/ghost bits, with floating arrays rounded to 12 decimals to accommodate roundoff. Protection flags are restored with the support choice because the exporter identifies a far particle using the fitted support weights. `provenance/stencil_sources.json` records the original binary-model hash and the extraction. These templates preserve the original nearest-neighbor tie choices; they contain no equilibrium trajectory or fitted experimental response.

The native binary format is implemented by `_core/model_export.write` and `_core/binary_model.read`. It stores 64-bit integer headers, model dimensions/parameters, positions, prescribed degrees of freedom, bond owners/shells/pair identifiers/flags, directions, stiffnesses and sparse support weights; version 2 adds cohesive flags and energy multipliers. The current format uses native little-endian NumPy/C types on the tested x86-64 platform. Cross-endian interchange requires conversion.

