# When causes co-occur: Overcoming distortion in multidimensional predictive representation — analysis code

Euan Prentis & Akram Bakkour

This repository contains the analysis code for the manuscript: the scripts that run the Bayesian
transition-influence regressions and the power analysis, plus the Jupyter notebooks that analyse
the simulated agents and the human participants and produce the figures.

| Resource | Location |
|---|---|
| Data (simulations and human participants) | https://osf.io/nqjy2/?view_only=72ad1546d2df460f9d8b574416c10644 |
| Task code and stimuli | https://github.com/BakkourLabRepo/feat-predict_robot-builder-task-psychopy |
| Computational model code (simulations and model fitting) | https://github.com/BakkourLabRepo/feat-predict_models |
| Analysis code (this repository) | https://github.com/BakkourLabRepo/feat-predict_analysis |

---

## Repository structure

```
feat-predict_analysis/
├── transition-influence.py        Bayesian transition-influence regressions (PyMC)
├── power-analysis.py              Simulation-based power analysis (sets the exp_2 sample size)
├── configs/                       One analysis config per experiment / analysis level
│   ├── exp_1.py, exp_1_trial.py   Online one-step experiment
│   ├── exp_2.py, exp_2_trial.py   Main multi-step experiment
│   └── exp_mri.py, exp_mri_trial.py   BIDS-formatted MRI behavioural data
├── src/utils.py                   Config loader
├── notebooks/
│   ├── human/                     Human-experiment analyses and figures
│   └── simulations/               Simulated-agent analyses and figures
└── environment.yml                Conda environment
```

---

## Installation

```bash
git clone https://github.com/BakkourLabRepo/feat-predict_analysis.git
cd feat-predict_analysis
conda env create -f environment.yml
conda activate feat-predict-analysis
```

The environment pins Python 3.12, PyMC 5.16.2, Bambi 0.14.0, ArviZ 0.19.0, and the scientific and
plotting stack used for the manuscript.

---

## Getting the data

Download the archives from OSF and extract them:

```bash
# Human experiments
tar -xzf exp_1.tar.gz      # -> exp_1/{data,results}/
tar -xzf exp_2.tar.gz      # -> exp_2/{data,results}/

# Simulations (each extracts to a folder named by its number, e.g. exp-1/)
tar -xzf exp-1_sf-feature.tar.gz
```

The OSF archives already contain every result that `transition-influence.py` and `power-analysis.py`
produce (`results/transition-predictions/`, `results/transition-influence/`,
`exp_1/results/power-analysis.csv`), plus the computational model fits (`results/model_fits.csv`).
**You only need the scripts if you want to regenerate those results.** To reproduce the figures,
go straight to the [notebooks](#notebooks).

---

## Experiment numbering

The simulation numbers (`exp-<N>`) follow the internal numbering used during the project, so they
are not consecutive: there is no `exp-2` or `exp-5` simulation in the final manuscript. Human
experiments are numbered separately (`exp_1`, `exp_2`). Each simulation notebook carries the same
number(s) as the archive(s) it analyses.

---

## Notebooks

### Human experiments (`notebooks/human/`)

| Notebook | Experiment | Manuscript |
|---|---|---|
| `exp-2_two-step.ipynb` | Main multi-step experiment (in person, N = 102) | Main text (Figs. 3–6) and Supplementary Information |
| `exp-1_one-step.ipynb` | Online one-step experiment (preregistered, N = 100) | Supplementary Results: Online One-Step Experiment (Figs. S1–S3) |
| `exp-1_one-step_power-analysis.ipynb` | Power analysis based on `exp_1` | Supplementary Information: Power Analysis |

These notebooks cover task performance, computational model comparison and parameter fits, and
the transition-influence results. Both main notebooks drop the excluded participants listed in
the OSF data README.

Paths are set at the top of each notebook, relative to the notebook's location:

```python
DATA_PATH = '../data'
TRANSITION_RESULTS_PATH = '../results/transition-influence'
MODEL_RESULTS_PATH = '../results/model_fits.csv'
FIG_PATH = '../figs'
```

To run them unchanged, put each notebook in a folder next to the extracted `data/` and `results/`
folders, for example:

```
exp_2/
├── data/
├── results/
├── figs/                          created for figure output
└── notebooks/exp-2_two-step.ipynb
```

Alternatively, edit the path constants to point to wherever you extracted the data.

### Simulations (`notebooks/simulations/`)

| Notebook | Archive(s) | Contents | Manuscript figures |
|---|---|---|---|
| `exp_1-4_sf-mb-feature.ipynb` | `exp-1`, `exp-4` | Feature-based SF and MB learners, regular vs. causal learners | Fig. 2; Figs. S13, S15A |
| `exp_3_model-recovery.ipynb` | `exp-3` | Model and parameter recovery on the multi-step task design | Figs. S4–S9 |
| `exp_6_sf-forgetting.ipynb` | `exp-6` | Forgetting (decay parameter varied) | Figs. S10A, S11A |
| `exp_7_sf-l1-regularization.ipynb` | `exp-7` | L1 regularization | Figs. S10B, S11B |
| `exp_8_sf-nonlinear-inference.ipynb` | `exp-8` | Non-linear value inference | Figs. S10C, S11C |
| `exp_9_sf-causal-bias.ipynb` | `exp-9` | Causal bias varied continuously | Figs. S10D, S11D, S14 |
| `exp_10-11_sf-mb-feature-constant-freq.ipynb` | `exp-10`, `exp-11` | As `exp-1`/`exp-4`, with constant target frequencies | Fig. S15B&C |
| `exp_12-13_sf-mb-conjunctive.ipynb` | `exp-12`, `exp-13` | Conjunctive learners with and without a memory sampler | Fig. S16 |

Before running a simulation notebook, set the raw-data path constant(s) at the top to the
extracted archive's `data/` folder:

```python
DATA_PATH = '/path/to/exp-6/data'          # single-archive notebooks
DATA_PATH_EXP1 = '/path/to/exp-1/data'     # paired notebooks (exp_1-4, exp_10-11, exp_12-13)
DATA_PATH_EXP4 = '/path/to/exp-4/data'
```

On first run, the notebooks combine the per-agent CSVs into `training_df.csv` and `test_df.csv`
and cache them under `FORMATTED_DATA_PATH` (`../data`, plus `../../exp_<N>/data` for the second
archive in the paired notebooks). Later runs load the cache unless `OVERWRITE = True`. Figures are
written to `FIG_PATH` (`../figs`).

**Model recovery (`exp_3_model-recovery.ipynb`).** The `exp-3` archive extracts to `exp_2/` (named
after the human experiment whose design was simulated). Point `DATA_PATH` at that folder. The
notebook also reads model fits to the simulated agents from
`RESULTS_PATH/<ModelClass>/<model_label>/model_fits.csv` (default `RESULTS_PATH = '../results/exp_2'`).
These fits are not part of the OSF archive. Generate them by fitting the models to the simulated
agents with [`feat-predict_models`](https://github.com/BakkourLabRepo/feat-predict_models).

---

## Scripts

Both scripts take the name of a config module in `configs/` as their only argument and must be
run from the repository root, so that `configs` and `src` are importable:

```bash
python transition-influence.py exp_2
python power-analysis.py exp_1
```

### Configuration

Each config file defines an `analysis_config` dictionary. **Edit `data_path` and `results_path`
(and `fig_path` in `exp_1.py`) to point to your extracted data before running.** For example:

```python
'data_path': '/path/to/exp_2/data',
'results_path': '/path/to/exp_2/results',
```

| Key | Description |
|---|---|
| `data_path` | Folder containing `training/` and `test/` trial CSVs (or `sub-<id>/` folders when `bids=True`) |
| `results_path` | Output folder for `transition-predictions/` and `transition-influence/` |
| `instance_tmat` | Instance-level transition matrix of the task |
| `feature_tmat` | Feature-level causal transition matrices, one set per counterbalancing condition (and one per causal step in `exp_2`) |
| `model_type` | `'agent'`: one regression per participant, across trials. `'trial'`: one regression per trial and congruency condition, across participants |
| `run_test_analysis` | Also predict test-phase choices from evidence observed during training |
| `overwrite` | Recompute outputs that already exist |
| `save_residuals` | Also save trial-wise residuals (`transition-influence-residuals/`) |
| `group_labels` | Restrict the analysis to named subgroup folders. Leave empty to detect them automatically, or when there are none |
| `n_cores` | Number of CPU cores. If divisible by 4, each regression samples its 4 chains in parallel; otherwise chains run sequentially and regressions run in parallel |
| `bids` | Whether the data are organised in BIDS format |
| `ids_to_exclude` (`exp_1.py` only) | Participants excluded under the preregistered criterion (used by the power analysis) |
| `power_analysis_config` (`exp_1.py` only) | Power-analysis settings (see below) |

The `*_trial.py` configs are the same as their counterparts except that `model_type = 'trial'` and
the test analysis is off.

### `transition-influence.py`

For each participant, a Bayesian multinomial logistic regression predicts the composition chosen
on each trial from previously observed causal and spurious transitions (see main text and Methods:
Causal and Spurious Transition Evidence). The script has two stages:

1. **Transition predictions.** Compute the causal and spurious transition evidence for every
   possible composition on every trial and write the regression inputs to
   `results/transition-predictions/{training,test}/{agent,trial}/*.csv`.
2. **Transition influence.** Fit the PyMC regression (4 chains × 2000 draws, 1000 tuning steps)
   and pickle the ArviZ `InferenceData` to
   `results/transition-influence/{training,test}/{agent,trial}/*.pkl`.

To reproduce all of the shared results:

```bash
python transition-influence.py exp_1         # per-participant, training + test
python transition-influence.py exp_1_trial   # per-trial, training
python transition-influence.py exp_2
python transition-influence.py exp_2_trial
```

### `power-analysis.py`

Sets the `exp_2` sample size (Supplementary Information: Power Analysis). The script simulates
datasets from the posterior of the `exp_1` second-level transition-influence model, refits the
model to each simulated dataset, and computes power as the proportion of simulations in which the
95% HDI for the transition type × category-congruency interaction excludes 0. It needs the `exp_1`
per-participant training posteriors (`results/transition-influence/training/agent/`) and writes
`results/power-analysis.csv` and `figs/power-analysis.svg`.

`power_analysis_config` in `configs/exp_1.py` controls the simulation grid:

| Key | Description |
|---|---|
| `n_per_group_levels` | Simulated sample sizes per congruency condition |
| `shrink_factor_levels` | Multipliers applied to the `exp_1` interaction effect size |
| `n_sims` | Simulations per cell |
| `draws`, `tune` | MCMC draws and tuning steps per refit |
| `n_workers` | Parallel worker processes, each with its own PyTensor compile directory |

`exp-1_one-step_power-analysis.ipynb` contains the same analysis in interactive form.

---

## License

MIT. See [`LICENSE`](LICENSE).
