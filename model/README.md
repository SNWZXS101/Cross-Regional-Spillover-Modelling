# Baseline model

`train_ude.py` is the original two-region training implementation with data/output paths adapted for this repository and its stale introductory boundary ranges corrected to match `Config`. The network, equations, loss, three-candidate screening, warm start and optimization logic are unchanged. The original final run configuration and outputs, including the fitted network checkpoint and exported daily functions, are in `../data/baseline/`; use that saved config rather than interpreting starting defaults as the selected fit.

From the repository root, after installing the dependencies documented in the main README:

```bash
python model/check_inputs.py
python model/run_baseline.py validate --output outputs/baseline_validation
python model/run_baseline.py train --output outputs/baseline_training --device cpu
```

The second command performs the standalone independent RK4 validation of saved functions. The third runs the original three-configuration screening and the final warm-start fit; training is computationally intensive and exact floating-point results can vary with software and hardware. A full retraining was not repeated for release validation. All generated files go under the specified output directory, not into the saved baseline inputs.

`../data/input/Data_all_sm.xlsx` contains the 236-day smoothed input used for the baseline; `Data_all.xlsx` contains the matching unsmoothed daily series. Columns are `DATE`, `GD` and `GX`. To remove account identifiers, creator/last-modified-by metadata was removed from each workbook copy and an embedded local filesystem path was removed from the raw workbook; worksheet data and formulas are unchanged. Underlying primary surveillance sources, case definitions and public sharing permissions should be confirmed by the authors before a publication claim is made.
