# Primary ensemble and policy calculations

Run `python analysis/verify_saved_results.py --output path/to/new/summary` from the repository root. The command uses only the saved CSV files: it verifies scenario counts, avoided-case arithmetic and the published reported-case ordering counts, then recomputes all 36 unordered pairwise comparisons for reported and diagnostic all-infections burdens. It writes `recomputed_summary.json` and `pairwise_rankings.csv` to the specified output. Strict comparisons are counts within the constructed ensemble, not probabilities.

The `results/main_30`, `results/sensitivity_50`, and `results/sensitivity_100` folders contain the saved input and outputs for the published ensemble-size comparison. The `sensitivity_100/solutions` folder contains all 100 saved members; the original 30 and 50 members are nested subsets, with matching source NPZ byte hashes checked before packaging. `sensitivity_100/trajectories` contains daily trajectories for the nine strategies for each of the 100 members. The number 100 is the manuscript's primary ensemble; 30 and 50 are sensitivity subsets.

To initiate a new, expensive calculation with the original 30-dimensional family, the original scientific acceptance checks and the original intervention protocol, run for example:

```sh
python analysis/run_fresh.py --output runs/new_100 --n-solutions 100 --seed 31415 --n-starts 3 --max-iterations 60 --workers 1
```

Choose a new output directory. The command runs an independent baseline audit, policy-blind candidate generation, a freeze, and the nine strategies. `--through audit_baseline` stops after baseline validation; `--help` lists options. The adaptive start schedule is 3+3+6 for the default 3 starts, with 60 iterations per start; the scientific thresholds are unchanged. The path-adapted algorithm is under `scripts/`, and the actual 236-day high-risk sets are in `protocol/fixed_risk_protocol.json`. Reported cases are the primary outcome; all-infections is a diagnostic measure. All nine strategies use fine/coarse RK4 checks; independent DOP853 checks apply to HH and LL only.

A brief two-member integration smoke test was run with one start per round and one optimizer iteration per start; it produced two admissible members and all 36 member/scenario/measure rows. This test is not part of the manuscript ensemble and does not establish bit-identical reproduction of a full search. The published 100-member search was not rerun as part of release packaging.

For the UDE training code and baseline inputs see `../model/` and `../data/`. Full training and 100-member reconstruction are computationally intensive and may differ across hardware/software. The bundled member data and scenario results reproduce the paper's reported ensemble statistics without recomputation.

The `baseline` field in each published `baseline_audit.json` was changed to the relative path `data/baseline` when packaging; numerical audit contents were not altered. Original frozen member `.npz` files and scenario CSV files remain byte-identical to the source files.
