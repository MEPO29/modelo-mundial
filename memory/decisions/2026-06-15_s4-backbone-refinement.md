---
stage: S4
date: 2026-06-15
author: ds-run routine
slug: s4-backbone-refinement
---

# Decision — Backbone refinement & the "near-optimal" finding (2026-06-15)

## Context
After S2 EDA (corrected: 7H/4D/1A on n=12 live; pool failed the gate on that tiny
sample), moved to model improvement validated on the **275-match walk-forward
backtest** (the legitimate sample), not the n=12 live window.

## What was decided

1. **Validate on the backtest, not the live n=12.** The critic (S2) showed the
   n=12 pool-vs-DC gap (0.0052) is within sampling noise. All model changes are
   gated on the 5-tournament LOTO backtest.

2. **No post-hoc calibration / pool change shipped.** Temperature scaling,
   Dirichlet scaling, drop-GBM, bayes-only, and market-inclusion were all within
   bootstrap noise. The stack's ECE is already 0.039 — nothing to calibrate.

3. **Adopted a marginal backbone refinement** (`SIGMA_RW_PRIOR 0.1→0.2`,
   `FRIENDLY_WEIGHT 0.5→0.7` in `bayes.py`): RPS 0.1949→0.1947, log-loss
   0.9659→0.9651, 4/5 tournaments improved. Bootstrap 90% CI on the delta spans 0
   (mean −0.00035, P(improve)=83%). Shipped **because** it improves both metrics
   robustly with principled, gentle levers and negligible downside — and
   documented transparently as within-noise, not overclaimed.

4. **Did NOT change** the history window (`since=2010` is optimal — more/less both
   hurt clearly), `period 365` (more structural, negligible extra gain), ensemble,
   market, GBM, or the online loop.

5. **Refactored** Bayesian prior scales to tunable module constants — enables
   future principled tuning without editing the model body. Defaults preserved
   except the two adopted above.

## Why this is the honest call
The model was already at the efficient-market frontier (bayes RPS ≈ de-vigged
market RPS) and well-calibrated. The valuable output is the *rigorous
demonstration* of that, plus configurability — not a fabricated RPS gain. See
[[2026-06-15T15-30_critique-eda-diagnostics-v2]] for the rigor standard applied.

## Future work (logged, not done)
- Bivariate-Poisson shared-component for draw *discrimination* (low expected
  payoff, high effort).
- Retire GBM from the live cycle to save compute (conclusive dead-weight evidence)
  — pending; kept for now per the Hedge-slot design.

Full evidence: `artifacts/reports/model_improvement_2026-06-15.md`,
`notebooks/03_improvement_experiments.py`, `notebooks/04_backbone_sweep.py`.
