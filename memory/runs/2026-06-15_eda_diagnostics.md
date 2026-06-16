# EDA Diagnostics Run — 2026-06-15

## Summary

Scored 12 pre-logged WC 2026 predictions against actual results (June 11-14).

## Aggregate Metrics

| Layer | RPS | Brier | Log-Loss | ECE | n |
|-------|-----|-------|----------|-----|---|
| Dixon-Coles | 0.1918 | 0.6203 | 1.0099 | 0.248 | 12 |
| Bayesian | 0.1919 | 0.6172 | 0.9954 | 0.177 | 12 |
| GBM | 0.2472 | 0.7270 | 1.1667 | 0.395 | 12 |
| Market | 0.2053 | 0.6867 | 1.0972 | 0.286 | 10 |
| Pool | 0.1970 | 0.6301 | 1.0168 | 0.276 | 12 |

## Top Findings

1. **Bayesian backbone dominates**: Dixon-Coles has the best RPS (0.1918), consistent with backtest weights giving bayes ~100%.
2. **Pool does not beat best layer**: Pool RPS (0.1970) vs best layer (0.1918) — the ensemble is not adding value over bayes alone on this sample.
3. **GBM is dead weight**: GBM RPS (0.2472) is the worst of all layers, confirming backtest findings (weight ~0).
4. **Market layer underperforms bayes**: On the 10 matches with market data, market RPS (0.2053) is worse than bayes. The 0.35 market prior may be too high.
5. **Weight discrepancy confirmed**: online_state.json has bayes=0.55/market=0.35, but code INITIAL_WEIGHTS has bayes=0.35/market=0.55. The live pool is already more bayes-heavy than intended.
6. **Strong home advantage in WC 2026**: 7/12 (58%) home wins vs historical 89/204 (44%). Model may be underestimating home advantage.
7. **High-confidence predictions are well-calibrated**: Pool predictions with >55% confidence show better accuracy than low-confidence ones.

## Concerns for S3

- **Small sample (n=12)**: All metrics have high variance. No layer differences are statistically significant.
- **GBM layer needs diagnosis**: Is it a feature problem, fitting problem, or signal-to-noise? Consider dropping or reducing its influence.
- **Market prior may be too high**: The 0.35 initial weight for market seems excessive given bayes outperforms it. Hedge updates should correct this, but starting from a better prior would help.
- **Home advantage underestimation**: The model may need a venue-specific home advantage adjustment for the unusual 2026 tri-country setup.
- **Weight discrepancy needs resolution**: Before running `make update`, decide whether to trust online_state.json or the code constant.

## Artifacts Generated

- `artifacts/plots/layer_rps_comparison.png`
- `artifacts/plots/calibration_plot.png`
- `artifacts/plots/match_rps_scatter.png`
- `artifacts/plots/prediction_vs_outcome.png`
