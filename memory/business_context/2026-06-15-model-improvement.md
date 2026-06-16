---
stakeholder: "project developer (engineer)"
analytical_task: "Mid-tournament forecast evaluation and model improvement: activate the online learning loop, diagnose layer-level performance on played WC 2026 matches, and identify targeted improvements to the probabilistic forecasting pipeline."
target_variable: "1X2 match outcome (0=home win, 1=draw, 2=away win) — ordinal categorical"
success_metric: "Ranked Probability Score (RPS); secondary: Brier score, log-loss, Expected Calibration Error (ECE)"
success_threshold: "Pool RPS must beat Dixon-Coles baseline RPS on scored WC 2026 matches; aspirational target is to approach de-vigged closing market RPS within 0.01."
decision_cost_asymmetry: "symmetric — prediction quality optimization, no asymmetric decision cost"
data_sources:
  - "data/raw/martj42/2026-06-11/results.csv — historical international match results (stale since June 11; needs refresh via download_results())"
  - "artifacts/predictions_log.parquet — append-only log of 20 pre-kickoff predictions for June 11-16 WC matches"
  - "artifacts/online_state.json — Hedge pool weights and scoring history (currently empty: scored=[], score_history=[])"
  - "artifacts/pool_weights.json — backtest-fitted pool weights (bayes ~1.0, dc/gbm ~0)"
  - "data/raw/odds/ — cached The Odds API snapshots from June 11"
  - "data/reference/WorldCup2026-football-data-co-uk.xlsx — historical closing odds and xG for reference tournaments"
  - "data/reference/bracket_2026.json, groups_2026.json, venues_2026.csv, city_altitudes.csv — tournament structure and contextual features"
constraints:
  - "No leakage: features timestamped strictly before kickoff; temporal splits only"
  - "Honest scoring: only pre-kickoff logged predictions scored against results; no back-filling"
  - "Backtest gate: each layer must beat DixonColes walk-forward RPS on 5 reference tournaments"
  - "Predictions log is append-only: once a forecast is logged for a match, it is frozen"
  - "No compliance regime (public sports data, no PII)"
deadline: "2026-06-16 (before next match window opens; tournament is live and matches play daily through July 19)"
---

# Business Context: Mid-Tournament Model Improvement (2026-06-15)

## Stakeholder and Decision

The stakeholder is the project developer (an engineer) who built and operates `modelo_mundial`, a probabilistic forecasting system for the 2026 FIFA World Cup. The decision to be made is: **what specific changes should be applied to the model pipeline to improve forecast quality for the remaining ~84 tournament matches**, given that 4 days of actual match results are now available but have not yet been ingested or evaluated.

## The Situation

The tournament started on June 11, 2026. On that day, the system:
1. Pulled historical match data through June 11.
2. Ran the backtest to fit pool weights (bayes dominated at ~100%).
3. Generated and logged predictions for 20 matches (June 11-16) to `predictions_log.parquet`.
4. Initialized the online state with weights: dc=0.05, bayes=0.55, gbm=0.05, market=0.35.
5. Produced the first (and only) cycle report.

Since then, **the online update cycle (`make update`) has not been run**. This means:
- **No match results have been ingested** since June 11. Approximately 12-16 group-stage matches have been played (June 11-15) whose results exist in the upstream data source but not in our local copy.
- **No predictions have been scored.** The `scored` array and `score_history` in `online_state.json` are empty.
- **Hedge weights have not updated.** The pool is still operating on the June 11 initial weights, missing 4 days of learning signal.
- **No new predictions have been logged** for matches beyond June 16.

The system was designed with a daily update cycle that would handle all of this automatically. The cycle has been dormant.

## Analytical Task (Data-Science Terms)

This is a **forecast evaluation and iterative model improvement** task, combining:

1. **Forecast verification** — score pre-logged probabilistic predictions against observed outcomes using proper scoring rules (RPS, log-loss, Brier). This is a descriptive/evaluative task.
2. **Online learning activation** — run the Hedge multiplicative-weights update to let the ensemble self-correct based on observed component performance. This is an online optimization task.
3. **Diagnostic analysis** — decompose forecast error by layer (Dixon-Coles baseline, Bayesian backbone, GBM, market) to identify which components are well-calibrated, which are miscalibrated, and where systematic errors exist. This is an exploratory/diagnostic task.
4. **Targeted improvement identification** — based on diagnostics, propose specific changes to feature engineering, hyperparameters, or model architecture that could improve remaining predictions. This is a prescriptive task.

The task is **not** a standard classification or regression problem. It is the evaluation and improvement of an existing probabilistic forecasting system operating in a live, non-stationary environment (a tournament where team strengths, tactics, and conditions evolve match-by-match).

## Success Criteria

**Primary metric:** Ranked Probability Score (RPS) on the pool's pre-logged predictions, scored against actual match outcomes.

**Thresholds:**
- **Minimum:** Pool RPS on scored WC 2026 matches must be lower (better) than the Dixon-Coles baseline RPS on the same matches. This is the project's standing backtest gate.
- **Aspirational:** Pool RPS within 0.01 of the de-vigged closing market RPS (the efficient-market benchmark).
- **Process:** The online update cycle runs successfully, scores all played-and-logged matches, and produces updated weights and a cycle report.

**Secondary metrics:**
- Brier score (multiclass squared error)
- Log-loss (the Hedge update's native loss)
- Expected Calibration Error (ECE) — are the predicted probabilities well-calibrated?
- CUSUM drift statistic — is the model systematically worse than market?

**What "good" looks like in a sentence:** The online loop is running, we know how each layer is performing on real WC matches, and we have a prioritized list of specific, implementable improvements with estimated RPS impact.

## Decision Cost Asymmetry

**Symmetric.** This is a prediction-quality optimization problem. There is no asymmetric cost between false positives and false negatives in the traditional sense — we are optimizing a proper scoring rule over probabilistic forecasts. A miscalibrated forecast in either direction (over- or under-estimating a team) costs the same in RPS terms.

Note: if these forecasts were feeding a betting strategy, the asymmetry would depend on the market's pricing errors. But the project as specified is about forecast quality, not betting decisions.

## Data Sources

| Source | Path / Location | Status | Notes |
|--------|----------------|--------|-------|
| Historical match results | `data/raw/martj42/YYYY-MM-DD/results.csv` | **Stale** (June 11) | Needs `download_results()` refresh from martj42/international_results GitHub |
| Pre-logged predictions | `artifacts/predictions_log.parquet` | Current | 20 matches logged for June 11-16; append-only, frozen |
| Online state | `artifacts/online_state.json` | Current (empty scoring) | Weights initialized, no matches scored |
| Pool weights (backtest) | `artifacts/pool_weights.json` | Current | bayes ~1.0, dc/gbm ~0 from backtest fit |
| Market odds (cached) | `data/raw/odds/` | Stale (June 11) | 3 snapshots from The Odds API |
| Historical odds & xG | `data/reference/WorldCup2026-football-data-co-uk.xlsx` | Static reference | Closing odds for WC 2014/2018/2022 |
| Tournament structure | `data/reference/bracket_2026.json`, `groups_2026.json` | Static reference | 48-team bracket, 12 groups |
| Venue/altitude data | `data/reference/venues_2026.csv`, `city_altitudes.csv` | Static reference | Contextual features for GBM |

## Constraints

1. **No leakage.** All features must be timestamped strictly before kickoff. Temporal splits only. This is a hard invariant enforced by the architecture.
2. **Honest scoring.** Only predictions logged *before* kickoff are scored against results. The predictions log is append-only — once a forecast is logged for a match, it is frozen. No back-filling.
3. **Backtest gate.** Each component layer must beat the Dixon-Coles baseline on walk-forward RPS across 5 reference tournaments (WC 2014/2018/2022, Euro 2024, Copa 2024).
4. **No compliance regime.** Public sports data, no PII, no regulatory constraints.
5. **Time pressure.** The tournament is live. Matches play daily. Every day without the update cycle running is a day of lost learning signal. The deadline for the first improvement cycle is June 16 (before the next match window).

## Scope Creep Risks (Intentionally Out-of-Scope)

- **No real-time deployment.** The model runs as a batch pipeline, not a real-time service. We are not building a streaming system.
- **No betting strategy.** We are optimizing forecast quality, not expected return against a bookmaker.
- **No PII linkage.** No player-level data, no personal data of any kind.
- **No model architecture replacement.** The four-layer stack (Bayesian backbone, GBM, market, ensemble) is the established architecture. Improvements should be within this framework (feature additions, hyperparameter tuning, weight adjustments), not wholesale replacement.
- **No re-scoring of past predictions.** The predictions log is frozen. We evaluate what we predicted, not what we would have predicted with hindsight.

## Key Observations and Risks

1. **Weight discrepancy.** The `INITIAL_WEIGHTS` constant in `update_cycle.py` is `{dc: 0.05, bayes: 0.35, gbm: 0.05, market: 0.55}`, but `online_state.json` has `{dc: 0.05, bayes: 0.55, gbm: 0.05, market: 0.35}`. The bayes and market weights appear to have been swapped or manually adjusted. This should be investigated — if the state was hand-edited, the Hedge updates will start from a non-standard prior.

2. **Pool weights vs. online weights divergence.** The backtest-fitted `pool_weights.json` gives bayes ~100% weight (dc and gbm are effectively zero). But the online state gives bayes 55% and market 35%. These are two different weight systems serving different purposes (backtest stacking vs. live Hedge), but the divergence is worth understanding.

3. **Small sample.** Even after scoring all June 11-15 matches, we will have ~12-16 scored matches. This is a tiny sample for drawing conclusions about layer-level performance. The diagnostics will be suggestive, not definitive. The CUSUM threshold of 3.0 is calibrated for this regime.

4. **GBM may be dead weight.** Both the backtest pool weights (gbm ~0) and the online initial weights (gbm 0.05) suggest the GBM layer is contributing little. If it continues to underperform, the diagnostic should assess whether it's a feature problem, a fitting problem, or a signal-to-noise problem.

5. **Market layer dependency.** The market layer depends on The Odds API returning live odds for upcoming WC matches. If the API is unavailable or odds are missing for certain matches, the market block becomes `None` and the partial-pool rescaling kicks in. This is handled gracefully in the code but should be monitored.

## Recommended Next Steps

1. **Run `make data`** to pull fresh match results from the upstream source.
2. **Run `make update`** to activate the online learning cycle — this will score pre-logged predictions, update Hedge weights, log new predictions, and produce a cycle report.
3. **Inspect the cycle report** for weight trajectories, per-component log-loss, and the CUSUM statistic.
4. **Run diagnostic analysis** on the scored matches: RPS by layer, calibration plots, error decomposition by match type (e.g., mismatched vs. evenly-matched teams).
5. **Identify and prioritize improvements** based on diagnostics, then implement and re-run the cycle.
