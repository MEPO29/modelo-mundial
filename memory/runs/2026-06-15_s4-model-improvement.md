# Model Improvement Run — S2→S4 + close-out (2026-06-15)

## Protocol
Validated on the 275-match walk-forward backtest (5 reference tournaments), LOTO,
with paired bootstrap CIs. The expensive Bayesian SVI fits were run once and
component predictions cached (`data/interim/backtest_predictions.parquet`) so
post-hoc pool/calibration experiments were cheap.

## Baseline (production)
| layer | RPS | logloss | ECE |
|---|---|---|---|
| Dixon-Coles (gate) | 0.1967 | 0.9735 | 0.035 |
| Bayes | 0.1949 | 0.9659 | 0.047 |
| GBM | 0.2023 | 1.0073 | 0.056 |
| Ensemble | 0.1959 | 0.9708 | 0.039 |

## What was tested (all within bootstrap noise)
- Post-hoc: temperature (ll/rps), Dirichlet scaling, drop-GBM, bayes-only, market-inclusion.
- Backbone refits: history window, sigma_rw, period granularity, friendly weight, home_adv prior, svi steps, num_samples.

## Outcome
- **No statistically significant improvement available** — model at the efficient-market frontier, ECE already 0.039.
- **Shipped:** (1) Bayesian prior scales → tunable module constants (refactor); (2) `FRIENDLY_WEIGHT 0.5→0.7`, `SIGMA_RW_PRIOR 0.1→0.2`. Confirmed: bayes RPS 0.1949→0.1947, logloss 0.9659→0.9651, ensemble 0.1959→0.1953, 4/5 tournaments better, **gate holds** (0.1953<0.1967). Tradeoff: ECE 0.047→0.076.
- Tests: 49 passed. Reproducibility: seeds fixed (bayes seed=0; experiment scripts default_rng(0)); backtest seed-stable.

## Artifacts
- `artifacts/reports/model_improvement_2026-06-15.md` (full report)
- `artifacts/reports/backtest_baseline_20260615.txt`, `backtest_tuned_20260615.txt`, `backbone_sweep_20260615.txt`
- `notebooks/03_improvement_experiments.py`, `notebooks/04_backbone_sweep.py`
- Decision: `memory/decisions/2026-06-15_s4-backbone-refinement.md`

## Concerns for live operation
- Revert `SIGMA_RW_PRIOR` to 0.1 first if in-tournament calibration degrades.
- GBM is dead weight (confirmed); retire from the live cycle to save compute when convenient.
