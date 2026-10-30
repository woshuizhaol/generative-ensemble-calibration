# generative-ensemble-calibration

Code for the study *Hydrogen-exchange free energies benchmark generative protein conformational ensembles* (Communications Chemistry).

Residue-level opening free energies, ΔG_open, measured by hydrogen exchange are compared with the values computed from conformational ensembles with the fixed-coefficient Best–Vendruscolo (BV) model:

```
ln PF_i   = β_C · ⟨N_C,i⟩ + β_H · ⟨N_H,i⟩      (ensemble average; β_C = 0.35, β_H = 2.0)
ΔG_open,i = RT · ln PF_i                        (EX2 regime)
```

`N_C` counts heavy atoms within 6.5 Å of the amide nitrogen and `N_H` the backbone hydrogen bonds of the amide. The model is also available as the Python package **hxprobe** (`pip install hxprobe`, https://github.com/woshuizhaol/hxprobe).

## Repository

```
revision/
  scripts/                             analysis pipeline, one prefix per analysis stage (below)
  scripts/lib/                         shared functions: BV scoring, ensembles, statistics, paths
  config/paths.yaml, seeds.yaml        all input paths; random seeds of every stochastic step
  inputs/operator/                     BV model: forward.py, differentiable form diff_forward.py, parameters
  inputs/code/utils/                   shared metrics of the first analysis
  code/original/                       scripts of the first analysis (calibration, learned scoring models, global stability)
environment/rev_environment.yml        conda environment used for the analyses
```

| Stage | Scripts (`revision/scripts/`) | Content |
|---|---|---|
| 0–1 | `s0_*`, `s1_*` | reproduction of the first analysis; data accounting; residue numbering |
| 2 | `s2_*` | overlap of the twelve proteins with generator training data |
| 3 | `s3_*` | generator settings, compute cost, molecular-dynamics protocol |
| 4 | `s4_*` | per-conformer contact and hydrogen-bond counts (PDBFixer protonation at pH 7.0), frozen random-draw indices |
| 5 | `s5_*` | ensembles of the deposited structures, generators and noise ensembles |
| 6–8 | `s6*`, `s7_*`, `s8*` | absolute scale and coefficients, tail-weighted averaging, exposure baselines, leave-one-protein-out models, noise tests |
| 9 | `s9*` | subsets, training-data overlap, cytochrome c with and without heme c (parameters, simulation, analysis) |
| 10 | `s10*` | global fold stability |
| 13 | `s13_*` | replicate short-MD trajectories from the deposited structures |
| 14 | `s14_*` | short-MD trajectories started from BioEmu conformations |

## Data

The data are archived on Zenodo, https://doi.org/10.5281/zenodo.20938502. Extract the archives into `revision/` (the content of `analysis_inputs.tar.gz` is added to `revision/inputs/`):

| Archive | Content | Extracted to |
|---|---|---|
| `result_tables.tar.gz` | result tables of all analyses | `tables/` |
| `per_conformer_counts.tar.gz` | per-conformer contact and hydrogen-bond counts; frozen random-draw indices | `raw_outputs/s4/` |
| `analysis_inputs.tar.gz` | processed experimental tables, structures, ensemble count tables, short-MD trajectories of the deposited structures | `inputs/` (merged with the files of this repository) |
| `short_md_replicates.tar.gz` | protein trajectories, initial and final coordinates, logs and input scripts of the replicate short-MD trajectories, starting conformations from BioEmu | `md_replicates/`, `md_altstart/` |
| `cytochrome_c_heme.tar.gz` | heme c parameters, quantum-chemical inputs and trajectories of cytochrome c with and without heme c | `md_cytc_heme/` |

`Source_Data.xlsx` (data underlying the figures) and `Supplementary_Data_1.xlsx` (Supplementary Tables 1–17 in machine-readable form) are in the same archive record.

## Reproduction

```
conda env create -p ./env_rev -f environment/rev_environment.yml
cd revision
export CUDA_VISIBLE_DEVICES=""                           # scoring and statistics use the CPU only
export OPENMM_CPU_THREADS=1                              # before OpenMM is loaded; required for deterministic re-scoring
../env_rev/bin/python scripts/s7_invpf_form.py           # tail-weighted versus mean-count averaging -> tables/S7_*.csv
../env_rev/bin/python scripts/s9a_subsets.py             # paired tests in subsets of proteins -> tables/S9_subset_tests.csv
../env_rev/bin/python scripts/s14_altstart_rho.py        # short-MD from BioEmu conformations -> tables/S14_*.csv
```

Paths are resolved from `revision/config/paths.yaml` (`project_root: inputs`, `revision_root: .`). Per-frame scoring, random draws, bootstrap and permutation tests use the seeds in `revision/config/seeds.yaml` and the result tables list the seeds used. Generation of ensembles (stage 5) needs the generator repositories and weights and a GPU; molecular dynamics (stages 9, 13, 14) needs OpenMM 8 with CUDA.

## Citing

> Wang, Y., Cui, H., He, Y., Kang, J., Fu, Y., Li, J., Yang, K. & Han, W. Hydrogen-exchange free energies benchmark generative protein conformational ensembles: code and processed data. Zenodo. https://doi.org/10.5281/zenodo.20938502

## License

MIT (see `LICENSE`).
