# Revision analyses

The `data/` directory contains the saved member-level values, representative trajectories and observation-model fit needed to regenerate Supplementary Figures S4–S7. From the repository root, install `numpy`, `pandas`, `matplotlib`, and `openpyxl`, then run:

```bash
python revision/scripts/render_supplement.py
python revision/scripts/test_scientific_components.py
```

The first command writes editable SVG, PDF and PNG files under `revision/figures/`; it reads saved results and does not train, search, or solve an ODE. The second is a short algebraic smoke check of external-input population balance, subsequent-detection reporting and the inclusion of the last five days in the unsmoothed-data WAPE. These checks do not rerun the complete sensitivity experiments. `figure_source_registry.json` identifies the relative inputs and group/member keys. They are internal identifiers, not posterior sample labels.

`source_snapshots/` contains the scientific Python components and experiment settings used in the alternate-baseline, variable-transmission, alternative-basis, modified-initial-state, external-infection-pressure, subsequent-detection, unsmoothed-data, policy-sensitivity, and extreme-contrast analyses. These research-code snapshots retain their scientific logic, but their full historical orchestration expects the original project tree, accepted-baseline validation and frozen input contexts. They are supplied for inspection and provenance: they have **not** been verified as standalone commands in this release. Paths marked `SOURCE_PROJECT/` in settings are provenance placeholders and are not executable file paths. For example, the unsmoothed-data ensemble runner expects `results/baseline_independent_validation.json` and an accepted baseline training directory; these historical inputs are not included in the release. The original training and validation workflow must be recreated in its expected layout before running that ensemble. Reconstructing all ensembles from raw data would require additional integration and substantial computation; the figure reproduction command above does not require those steps.

The subsequent-detection source includes the seed-27182 training script and its pre-registered 150-iteration L-BFGS continuation, because the latter produced the accepted baseline; their filenames reflect the historical run. Failed-run logs and output directories are not included.
