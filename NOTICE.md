# Attribution and release status

This is a research source-and-data package associated with the manuscript named in `CITATION.cff`. No new open-source license is granted by this file. The authors should select an appropriate license for their contributions before distributing them under an open-source license. Original attributions remain applicable.

## Experimental numerical values

- Qiao, Y. and Argon, A. S. (2003), *Cleavage cracking resistance of high angle grain boundaries in Fe–3%Si alloy*, Mechanics of Materials 35, 313–331. [DOI:10.1016/S0167-6636(02)00284-3](https://doi.org/10.1016/S0167-6636(02)00284-3). `data/experiments/fe_si_qiao_argon_2003.csv` transcribes Table 2.
- Ikeda, K., Morita, H., Nakashima, H. and Abe, H. (1999), Journal of the Japan Institute of Metals 63, 179–186. [DOI:10.2320/jinstmet1952.63.2_179](https://doi.org/10.2320/jinstmet1952.63.2_179). The two Mo CSV files contain digitized numerical values from Figs. 5 and 8. [Institutional record](https://hdl.handle.net/2115/75547).

These are attributed numerical transcriptions/digitizations, with the precision described in the CSV files. No article text or scanned article figures are redistributed. Dataset attribution does not assign rights to the underlying publications.

## Dependencies and scientific source mapping

Python dependencies (NumPy, SciPy, Matplotlib), Intel oneMKL, OpenMP and compilers are obtained separately under their own licenses. No third-party binary libraries are included.

`provenance/source_files.json`, `continuum_sources.json` and `data_files.json` identify the project files from which this package was extracted. The kernel reorganization changes import paths, names and input/output locations; verification distinguishes model equivalence, numerical checks and reanalysis of archived outputs.

The 2D distance-based merging construction follows the Chen et al. lattice-particle approach identified in the associated manuscript. The exported energy-matched 3D comparison applies the separation-work calibration implemented in `energy_matching.py`. Experimental source values, model predictions and independently assembled continuum comparisons are stored separately.

