# Cross-regional epidemic spillover: code and data

Code and saved inputs for *Decision robustness under non-unique time-varying epidemic mechanisms* (PLOS Computational Biology revision). The primary analysis uses 100 diverse, dynamically and observationally admissible mechanisms. The nested 30- and 50-member sets are ensemble-size sensitivity analyses; these sets are not posterior or probability samples.

## Repository contents

- `data/input/`: `Data_all_sm.xlsx` (236-day smoothed baseline input) and `Data_all.xlsx` (matching unsmoothed daily observations), with DATE, GD and GX columns. The spreadsheet copies omit author/last-modifier metadata and an embedded local path, without changing worksheet entries.
- `data/baseline/`: actual baseline configuration, checkpoint, exported eight time-varying functions, fitted beta, fixed constants and baseline diagnostics.
- `model/`: two-region UDE training, fitted-function export and independent ODE validation. See `model/README.md` for scope and commands.
- `analysis/`: policy-blind 30-dimensional maximin reconstruction, acceptance, original nine strategies, fixed risk sets and a solver-free saved-results checker. `analysis/README.md` describes the fresh-run entry and output isolation.
- `results/`: saved 30/50/100 scenario tables, distance matrices, manifests and audit summaries; the primary 100 member NPZ/JSON files and saved daily strategy trajectories are included once. The 30/50 ensembles are nested in the 100-member set; the historic directory name `main_30` does not designate the paper's main analysis.
- `figures/main/`: MATLAB scripts and a saved display snapshot for the main figures and tables. `diagram/` includes an editable Fig. 1 diagram. The scripts produce the original individual panels for Figs. 3–5; curated composite layout in the submitted manuscript is a separate presentation step.
- `figures/supplementary/`: saved-data MATLAB rendering scripts and data for supplementary Figs. S1–S3 and the ensemble diagnostics and ranking tables. The standalone table names S1/S2 in these plotting scripts correspond to the ensemble tables numbered S3/S4 in the manuscript.
- `revision/`: saved inputs and rendering code for the revised supplementary Figs. S4–S7, and research-source snapshots/configurations for sensitivity analyses. The historical T2–T6 full computational orchestration still depends on its original project layout and has not been validated as a standalone fresh run here. See `revision/README.md` for this important distinction.

## Environment

Python 3.10 was used for testing; install the numerical dependencies with:

```bash
python -m venv .venv
# Activate .venv using your operating system's instructions.
python -m pip install -r requirements.txt
```

The UDE training additionally requires PyTorch; install the appropriate CPU or GPU build following the official PyTorch installation instructions. Full training and sensitivity searches can take substantial time; numerical output may vary across software, hardware and random seeds. The original trained checkpoint and scientific results are provided to reproduce saved-result analyses without retraining. Main and supplementary MATLAB rendering was tested with MATLAB R2024b.

## Reproduce principal checks from saved results

Run from the repository root:

```bash
python model/check_inputs.py
python model/run_baseline.py validate --output outputs/baseline_validation
python analysis/verify_saved_results.py --output outputs/paper_statistics
python revision/scripts/test_scientific_components.py
python revision/scripts/render_supplement.py
```

The first command reads the two case workbooks; the second independently integrates the saved baseline with RK4; the third recomputes 30/50/100 ensemble summaries and all 36 pairwise strategy comparisons for both outcomes from saved scenario CSVs. The revision component test checks external-input balance, subsequent-detection accounting and the last five-day WAPE interval. S4–S7 rendering reads saved member-level data; it does not search, retrain or integrate.

## Regenerate the core original analysis

```bash
python model/run_baseline.py train --output outputs/new_baseline --device cpu
python analysis/run_fresh.py --output outputs/new_ensemble --n-solutions 100 --seed 31415 --n-starts 3 --max-iterations 60
```

The training command performs the original three-configuration screen and final warm-start training. The ensemble command uses the packaged original reference baseline and fixed risk protocol, not the output of a newly trained baseline; it generates a fresh ensemble and subsequently simulates the nine strategies. Outputs must be new paths outside `results/`. A two-member short-budget end-to-end smoke test was performed to check generation, freezing and strategy execution; it is not a replacement for the published 100 mechanisms and no full retraining or full 100-member fresh search was repeated for this release. Solver checks for the original nine strategies use fine/coarse RK4; DOP853 strategy checks cover HH and LL. All-infections is a model diagnostic by regional infection-production channel, not reassignment by eventual residence.

## Regenerate figures

From MATLAB, after changing to the repository root:

```matlab
addpath(fullfile(pwd,'figures','main'));
run_all_main_results(fullfile(pwd,'outputs','main_figures'));
addpath(fullfile(pwd,'figures','supplementary'));
run_all_supplementary(fullfile(pwd,'outputs','supplementary_figures'));
```

`python revision/scripts/render_supplement.py` writes S4–S7 outputs to `revision/figures/`. Figures and results use saved values; no outputs should be mixed with or replace the archived scientific inputs. Pairwise ranking fractions describe the constructed ensembles, not probabilities.

## Reproduction scope

The relevant core mathematical code and frozen results are included; the T2–T6 source snapshots are supplied for inspection and figure reproduction, but their original full-run automation is not yet a tested portable workflow. The pre-existing `matlab/` directory and root `Data_all_sm.xlsx` are retained as earlier repository history; the authoritative revision inputs for these commands are under `data/input/`.

## License

The author-generated code in this repository is released under the MIT License. See the LICENSE file for details. This license applies to author-generated code, not third-party data or other third-party materials.
