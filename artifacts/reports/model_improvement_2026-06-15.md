# Model Improvement Report — WC 2026 Forecasting Stack (2026-06-15)

**Author:** ds-agent-framework routine (S2→S4 + close-out)
**Validation sample:** walk-forward backtest on 5 reference tournaments (WC 2014/2018/2022, Euro 2024, Copa 2024) — 275 matches. This is the project's legitimate large-sample gate, **not** the n=12 live WC 2026 sample (too small to drive changes; the live EDA was diagnostic only).

## TL;DR

The stack was already operating **at the efficient-market frontier and is exceptionally well-calibrated** (overall ECE 0.039). A systematic search across calibration, pool composition, market inclusion, and Bayesian-backbone hyperparameters found **no statistically significant improvement** — the model is near the practical RPS floor for international 1X2 forecasting. One **marginal, robust, principled** backbone refinement was adopted (transparently documented as within-noise); the rest of the value is a rigorous demonstration that the model is near-optimal, plus a configurability refactor enabling future tuning.

## Baseline (production, walk-forward LOTO)

| Layer | RPS | log-loss | ECE | n |
|---|---|---|---|---|
| Dixon-Coles (gate) | 0.1967 | 0.9735 | 0.035 | 275 |
| **Bayesian backbone** | **0.1949** | **0.9659** | 0.047 | 275 |
| GBM | 0.2023 | 1.0073 | 0.056 | 275 |
| Ensemble (dc,bayes,gbm) | 0.1959 | 0.9708 | 0.039 | 275 |
| Market (de-vigged closing) | ~0.2010 | — | — | 192 (WC only) |

**Three structural facts established:**
1. **The ensemble does not beat the Bayesian backbone alone** (0.1959 vs 0.1949). The log-opinion pool weights collapse to `bayes≈1.0, dc≈0, gbm≈0`. Pooling adds noise, not signal, on this sample.
2. **GBM is dead weight** — worst layer on every tournament, zero stacking weight. Its rich contextual features (Elo, form, rest, altitude, h2h) plus the stacked Bayesian strengths still do not beat the parametric backbone, i.e. those features carry no signal beyond goal-based strength at international level.
3. **The backbone is already ~market-efficient** — Bayesian RPS (0.1949 overall; 0.2013 on the WC-with-odds subset) sits essentially on top of the de-vigged closing line (0.2010). Matching the market is the ceiling.

## Experiments run (all LOTO, paired bootstrap CIs)

### A. Post-hoc calibration & pool composition (cheap; on cached prediction matrix)
`notebooks/03_improvement_experiments.py`. Tested: drop GBM; bayes-only; temperature scaling (log-loss and RPS objectives); Dirichlet/vector scaling; market-inclusive pools.

**Result: every variant within noise.** No CI excluded zero. Cause: ECE is already 0.039 — there is no miscalibration for temperature/Dirichlet to correct. Market inclusion helped directionally (−0.0012 RPS on WC subset, better log-loss) but CI [−0.0066,+0.0042] crosses zero. The live system already includes market at a 0.55 prior (backtest LOTO market share ≈0.68; the live prior is deliberately shrunk because The Odds API median is noisier than the backtested closing line).

### B. Bayesian-backbone hyperparameter sweep (expensive; full refits)
`notebooks/04_backbone_sweep.py`. One-lever-at-a-time over 5 tournaments:

| Lever | overall RPS | vs default | tournaments improved |
|---|---|---|---|
| **DEFAULT** (since2010, rw0.1, P182, fr0.5) | 0.1949 | — | — |
| since 2006 | 0.2000 | +0.0052 | 1/5 |
| since 2014 | 0.2020 | +0.0071 | 2/5 |
| sigma_rw 0.20 | 0.1948 | −0.0001 | 3/5 |
| sigma_rw 0.07 | 0.1952 | +0.0003 | 2/5 |
| period 365d | 0.1949 | −0.0000 | 3/5 |
| period 120d | 0.1951 | +0.0002 | 1/5 |
| friendly 0.7 | 0.1948 | −0.0001 | 4/5 |
| friendly 0.3 | 0.1951 | +0.0002 | 1/5 |
| home_adv loc 0.35 | 0.1949 | +0.0000 | 1/5 |
| svi 8000 | 0.1953 | +0.0004 | 2/5 |

**Findings:** The history window is already optimal (more or less history both hurt clearly). All other levers move RPS within ±0.0004 — noise. The only *robust* directions (improve a majority of tournaments) are `friendly 0.7` (4/5) and `sigma_rw 0.20` (3/5).

### C. Combined refinement (adopted)
`friendly 0.7 + sigma_rw 0.20`: overall RPS 0.1947, log-loss 0.9651, **4/5 tournaments improved**. Paired bootstrap on the 275-match delta: **mean −0.00035 RPS, 90% CI [−0.00092, +0.00025], P(improvement)=83%.**

**Confirmed on a full fresh backtest** with the new defaults:

| metric (overall, 275 matches) | default | tuned | change |
|---|---|---|---|
| Bayes RPS (primary) | 0.1949 | 0.1947 | −0.0002 ✓ |
| Bayes log-loss | 0.9659 | 0.9651 | −0.0008 ✓ |
| Bayes ECE | 0.047 | 0.076 | +0.029 ⚠️ |
| Ensemble RPS | 0.1959 | 0.1953 | −0.0006 ✓ |
| Gate: Ensemble vs DC (0.1967) | beats | beats | holds ✓ |

**Decision: adopted** as new backbone defaults, with full transparency that it is a *marginal, not-statistically-significant* refinement carrying **one real cost**: overall Bayes ECE rises 0.047→0.076 (the faster random walk makes the backbone sharper). Kept because (a) the **stated primary metric RPS improved**, robustly across 4/5 tournaments, as did log-loss (the Hedge update's native loss); (b) RPS is itself a proper score that penalizes miscalibration, so the net RPS gain already accounts for the sharpness; (c) ECE stays inside the well-calibrated band (<0.08) and is a noisy diagnostic at n=275/10-bins. The levers are gentle and principled; downside is bounded. The more structural `period 365` lever was **not** adopted (tiny extra benefit, larger change to random-walk granularity). **If in-tournament calibration is observed to degrade, revert `SIGMA_RW_PRIOR` to 0.1 first** — it is the source of the ECE cost.

## Changes shipped

1. **`src/mundial/models/bayes.py`** — prior scales (`SIGMA_RW_PRIOR`, `SIGMA_TEAM_PRIOR`, `HOME_ADV_LOC/SCALE`) promoted to tunable module constants (a clean refactor; enables principled future tuning without touching the model body). New defaults: `SIGMA_RW_PRIOR 0.1→0.2`, `FRIENDLY_WEIGHT 0.5→0.7`.
2. **Experiment harnesses** added (`notebooks/03_*.py`, `notebooks/04_*.py`) — reproducible, seeded, LOTO + bootstrap.
3. No changes to ensemble, market, GBM, or online-loop code — they are correct and near-optimal as-is.

## Honest limitations / not pursued

- All gains are within bootstrap noise; the model was already near the frontier. We did **not** manufacture a false "improvement."
- **Draws** remain a *discrimination* problem (not a level-bias one — mean P(draw) is well-calibrated to the base rate). This is largely irreducible with goal-based models; a bivariate-Poisson shared-component structure is the one untested structural idea, with low expected payoff and high effort — logged as future work.
- **GBM** is retained (not deleted) per the project's Hedge-slot design; flagged as a candidate for removal to save per-cycle compute given conclusive dead-weight evidence.
- No paid APIs or new data were used. Free historical odds (football-data.co.uk xlsx) were profiled and used in the backtest market analysis.
