---
stage: S2
script_path: notebooks/02_eda_diagnostics_v2.py
target: "1X2 match outcome (0=home win, 1=draw, 2=away win)"
n_rows_analyzed: 12
random_seed: 42
findings_count: 7
---

# EDA Run Record — WC 2026 Diagnostics v2 (2026-06-15)

**Supersedes:** `memory/runs/2026-06-15_eda_diagnostics.md` (prior run had unverified data)
**Script:** `notebooks/02_eda_diagnostics_v2.py`
**Data:** `data/raw/martj42/2026-06-15/results.csv`, `artifacts/predictions_log.parquet`
**Verified ground truth:** 7H / 4D / 1A from inner join on (date, home_team, away_team)

## Aggregate Metrics Table

All stats computed fresh from raw data. n=12 for dc/bayes/gbm/pool; n=10 for market (2 June 11 opening matches had null odds).

| Layer | n | RPS | Brier | Log-Loss | ECE | Note |
|-------|---|-----|-------|----------|-----|------|
| Dixon-Coles | 12 | 0.1918 | 0.6203 | 1.0099 | 0.248 | gate |
| Bayesian | 12 | 0.1919 | 0.6172 | 0.9954 | 0.177 |  |
| GBM | 12 | 0.2472 | 0.7270 | 1.1667 | 0.395 |  |
| Market | 10 | 0.2053 | 0.6867 | 1.0972 | 0.286 | aspirational benchmark (n=10) |
| Pool | 12 | 0.1970 | 0.6301 | 1.0168 | 0.276 | primary metric |

## Gate and Benchmark Status

- **DC gate (pool RPS < Dixon-Coles RPS):** NOT MET — Pool 0.1970 vs DC 0.1918
- **Market aspirational (gap ≤ 0.01, n=10):** MET — Pool 0.1969 vs Market 0.2053, gap=-0.0084

## Top Findings

1. **Gate NOT met: Pool RPS (0.1970) does not beat Dixon-Coles (0.1918).** The ensemble fails the project's primary performance threshold on n=12 WC 2026 matches. With only 12 matches this is not statistically conclusive, but directionally concerning.
   [Plot: `artifacts/plots/v2_layer_rps_comparison.png`]

2. **Bayesian and Dixon-Coles layers perform nearly identically (RPS 0.1919 vs 0.1918).** The Bayesian backbone's Dixon-Coles low-score correction and per-team Gaussian random walks are not producing a clear discriminative advantage over the simpler DC model on this small sample. However, the pool weights (bayes=0.55) rightly favor Bayes based on backtest evidence.
   [Plot: `artifacts/plots/v2_layer_rps_comparison.png`]

3. **GBM is the worst layer by a wide margin (RPS 0.2472, Brier 0.7270).** This confirms the backtest result (GBM weight ≈ 0). GBM carries only 5% pool weight but is still actively dragging ensemble performance on misses like Qatar-Switzerland and Brazil-Morocco. Root cause is unknown from EDA alone — this is the top priority for S3/S4 investigation (feature problem? overfitting?).
   [Plot: `artifacts/plots/v2_match_rps_heatmap.png`]

4. **Market layer underperforms both DC and Bayes (RPS 0.2053 on n=10).** De-vigged closing odds are the aspirational benchmark, yet on this sample they perform worse than the Bayesian model. The Qatar-Switzerland match (market had ~75% for Switzerland, actual draw) was a significant miss. This may reflect genuine upset territory where markets mis-price, or simply small-sample noise.
   [Plot: `artifacts/plots/v2_layer_rps_comparison.png`]

5. **Draw underestimation is a systematic pattern across all layers.** The 4 matches that ended in draws (Canada-Bosnia, Qatar-Switzerland, Brazil-Morocco, Netherlands-Japan) showed mean pool P(draw) of 0.244, versus a draw rate of 0.333 in this sample and a neutral-WC historical base rate of 0.228. Every layer assigned lower draw probability than the draw actually occurred. Qatar-Switzerland and Brazil-Morocco were the worst misses (pool predicted ≥58% home or away win).
   [Plot: `artifacts/plots/v2_prediction_vs_outcome.png`]

6. **WC 2026 home-win rate (58.3%, n=12) is above the neutral-venue WC historical baseline (43.3%, n=843).** All WC 2026 venues are on neutral tri-national host ground (USA/Mexico/Canada). The correct comparison is neutral-venue WC matches, not all WC matches (which include non-neutral host-nation advantages). The pool's mean P(home win) = 0.424 was below the actual 0.583 rate. However, with n=12 this may be sample noise — the sign and magnitude are directionally important but not conclusive.
   [Plot: `artifacts/plots/v2_home_advantage_context.png`]

7. **The bayes/market weight swap in online_state.json appears to have helped on n=12.** The live pool uses bayes=0.55/market=0.35 (state file) vs the code's INITIAL_WEIGHTS of bayes=0.35/market=0.55. On these 12 matches, the actual (bayes-heavy) pool scored RPS=0.1970 vs counterfactual (market-heavy) pool RPS=0.1992. The bayes-heavy configuration was better by 0.0021 RPS points, consistent with bayes outperforming market on this sample. This suggests the hand-edited state file may have been intentional or lucky.
   [Plot: `artifacts/plots/v2_weight_counterfactual.png`]

## Concerns Carrying Into S3

| Priority | Concern | Blocks S3? |
|----------|---------|-----------|
| HIGH | GBM is actively hurting ensemble RPS. Diagnose root cause before next weight update. | No — but weight reduction to 0.0 recommended |
| HIGH | Weight discrepancy (bayes/market swap) must be resolved before `make update` activates Hedge. Starting Hedge from wrong prior is a persistent bias. | Yes for update cycle activation |
| MEDIUM | Draw probability is systematically underestimated by all layers. The Poisson model's correlation parameter (Dixon-Coles λ correction) may need recalibration for WC group stage. | No |
| MEDIUM | Market layer scored worse than Bayes on n=10. Re-evaluate the 0.35 initial market weight — lower to 0.20 is defensible. | No |
| LOW | n=12 sample — all findings are directional, not statistically significant. Await 20+ matches before drawing firm conclusions. | No |
| LOW | Qatar-Switzerland is the single worst-miss match for all layers (pool RPS=0.2817 if Qatar in match). Extreme upset; check whether similar calibration errors occurred at WC 2022/2018 for strong-vs-weak matchups. | No |

## Plots Generated

- `artifacts/plots/v2_layer_rps_comparison.png` — bar charts: RPS, Brier, log-loss by layer
- `artifacts/plots/v2_match_rps_heatmap.png` — per-match RPS heatmap (layers × matches)
- `artifacts/plots/v2_calibration_reliability.png` — reliability diagrams for pool and bayes
- `artifacts/plots/v2_prediction_vs_outcome.png` — predicted probabilities vs actual outcomes
- `artifacts/plots/v2_home_advantage_context.png` — historical WC home-win context with neutral breakdown
- `artifacts/plots/v2_weight_counterfactual.png` — per-match RPS under actual vs code INITIAL_WEIGHTS vs bayes-only
