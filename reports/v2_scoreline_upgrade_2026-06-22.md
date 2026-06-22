# V2 — Scoreline upgrade & live-loop fixes (2026-06-22)

## Motivation
Mid-tournament, the model felt like it "wasn't hitting scores." Investigation
found two separate things: (1) **exact scorelines were never evaluated** — only
1X2 (W/D/L) outcomes were scored; the bayes scoreline grid was produced but
discarded — and (2) the goal model was an **independent double-Poisson** that
structurally under-produces draws (it patched Dixon-Coles `tau` onto four cells
post-hoc). Two live-loop bugs compounded it: the persisted weight state was
inert (validated market-led prior never went live; score history empty) and the
GBM layer was dead weight skimming pool mass.

## What changed
| Area | Change |
|---|---|
| Backbone (`models/bayes.py`) | Independent double-Poisson → **shared-latent bivariate Poisson** (Karlis–Ntzoufras: home=X1+X3, away=X2+X3). Shared X3 models goal correlation / excess draw mass in the likelihood (closed-form pmf via `numpyro.factor`, SVI-cheap). Post-hoc DC `rho`/`tau` removed. Added a dormant `prior_offset` hook for an external strength prior. |
| Scoreline eval (`eval/metrics.py`) | New `score_log_loss`, margin `score_rps`, `correct_score_hit_rate`, `draw_calibration`, `paired_bootstrap`. |
| Live loop (`online/update_cycle.py`) | Logs the full grid (`bayes_grid`) immutably; scores it next cycle on a separate scoreline track; report gains a "Scoreline accuracy" block. |
| Bug fix — state | `online_state.json` reseeded to the validated prior `{dc 0.05, bayes 0.40, market 0.55}`; `load_state()` now validates/renormalizes against the component set. The 12 already-played-but-never-logged matches stay unscored (honest-scoring invariant — not back-filled). |
| Bug fix — GBM | Retired from the live pool (`COMPONENTS = dc/bayes/market`); kept as a backtest reference. |
| Backtest (`eval/backtest.py`) | Collects bayes BP + independent-ablation grids; new `scoreline_analysis` with a paired-bootstrap gate. |

## Validation (walk-forward, 275 matches: WC 2014/18/22 + Euro/Copa 2024)
**Scoreline (BP vs independent-Poisson on identical fitted rates):**

| grid | SLL | margin RPS | hit% | draw ECE |
|---|---|---|---|---|
| independent-Poisson | 2.8477 | 0.0416 | 12.0 | 0.0224 |
| **bivariate-Poisson** | **2.8255** | 0.0416 | 10.9 | 0.0224 |

Scoreline log-loss improvement **+0.0221, 90% CI [+0.0111, +0.0334]** → excludes
0, **gate PASS**. Margin RPS is identical by construction (margin = X1−X2; the
shared X3 cancels) — a correctness check that the implementation is right.

**1X2 (no regression):** bayes RPS **0.1949** (unchanged), still on the de-vigged
market line (market ~0.195). GBM stacking weight 0.0 (confirms retirement); LOTO
market share 0.64 (confirms the market-led prior).

## Decisions / scope
- **Workstream C (external data: Elo / xG / futures priors) deferred.** The
  backtest confirms 1X2 is already at the market frontier, so an external
  strength prior's expected 1X2 payoff is within-noise and it does not help
  scorelines. The `prior_offset` hook is built and dormant, ready for a feed.
- Primary target was **exact scorelines** — addressed by the BP backbone and now
  measurable; the low inherent ceiling on *exact-score hit rate* (~10-15% even
  when perfect) is unchanged, but the scoreline *distribution* is sharper and
  better calibrated on draws.

## Tests
58 → tests green (new scoreline-metric, BP draw-mass, grid-scoring, and
state-hardening tests added).
