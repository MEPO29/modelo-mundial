---
verdict: REVISE-WITH-NOTES
artifact_path: notebooks/02_eda_diagnostics_v2.py
reviewer: ds-critic
iteration: 1
---

# Critique: S2 EDA Diagnostics v2 — WC 2026

## Reproduction status
All headline numbers independently re-derived from `data/raw/martj42/2026-06-15/results.csv` and `artifacts/predictions_log.parquet`. RPS formula in `src/mundial/eval/metrics.py` is the standard ordered-RPS / (K-1) normalization; correct for 1X2.

| Claim | Script | Re-derived | Match |
|---|---|---|---|
| DC RPS n=12 | 0.1918 | 0.1918 | yes |
| Bayes RPS n=12 | 0.1919 | 0.1919 | yes |
| GBM RPS n=12 | 0.2472 | 0.2472 | yes |
| Market RPS n=10 | 0.2053 | 0.2053 | yes |
| Pool RPS n=12 | 0.1970 | 0.1970 | yes |
| Outcome dist | 7H/4D/1A | 7H/4D/1A | yes |
| Neutral WC H-rate | 43.3% (n=843) | 0.433 (n=843) | yes |
| All-WC H-rate | 45.5% (n=964) | 0.455 (n=964) | yes |
| Mean pool P(draw) | 0.244 | 0.2446 | yes |
| Gate (pool 0.1970 < DC 0.1918) | NOT MET | NOT MET | yes |

## Findings

### F1 (major). Weight-counterfactual reconstructs the wrong pool family
- Location: `notebooks/02_eda_diagnostics_v2.py:236-269`, `compute_pool()`.
- Issue: `src/mundial/models/ensemble.py:35` defines `pool_predict` as a **log-opinion (geometric) pool**: `p ∝ exp(Σ w_c log p_c)`. The script's `compute_pool` is a **linear (arithmetic) pool**: `p = Σ w_c p_c / Σ w_c`. These are not the same pool family. The reported counterfactual RPS (0.1992) compares a linear-pool counterfactual against the production log-opinion pool (0.1970).
- Re-derived with correct log-opinion pool and the same partial-rescaling logic: actual-weights = 0.1970 (matches logged exactly), code-weights = 0.1993. Direction (bayes-heavy better) and magnitude (~0.002 RPS) are preserved, so the *conclusion* in Finding #7 stands, but the method is wrong.
- Standard violated: internal consistency — reconstruction should use the same pooling rule as production.
- Recommended fix: swap `compute_pool` to call `mundial.models.ensemble.pool_predict` with `weights / weights.sum()` and the partial-pool subset, then re-emit Plot 6 and the numbers in Finding #7.

### F2 (major). Multiple comparisons across 5 layers with no uncertainty quantification
- Location: throughout; rankings at lines 419-432, Findings #2 and #4 in `memory/runs/2026-06-15_eda_diagnostics_v2.md`.
- Issue: Five layers are ranked on n=12 (n=10 market) with point estimates to 4 decimals, and Finding #2 says "Bayesian and Dixon-Coles perform nearly identically (0.1919 vs 0.1918)" while Finding #4 says "Market underperforms both DC and Bayes." With n=10–12 the bootstrap CI on RPS is on the order of ±0.05; the *rank order* between layers within ~0.01 of each other is not distinguishable. The run record's hedge language ("suggestive only", "directionally") helps, but specific layer comparisons (Bayes vs DC, Market vs Bayes) are stated as if real.
- Standard violated: S4 checklist — "Are bootstrap CIs reported on the primary metric? Are overlapping CIs honestly reported as ties?"
- Recommended fix: add a paired bootstrap (resample matches with replacement, 1000 reps) and report 90% CI on each layer's RPS and on the pairwise differences; explicitly call out which orderings are within CI.

### F3 (major). "Pool fails gate" framed as a Stage-2 finding when both the n and the variance preclude that conclusion
- Location: Finding #1 in run record and decisions file; `script:199-202`.
- Issue: The DC-gate is a *backtest gate* (each layer must beat DC walk-forward across 5 reference tournaments). Reading n=12 in-tournament matches as a gate failure is a category drift. The pool beats DC by -0.0052 RPS on n=12; this is well within sampling noise. Framing as "ensemble fails the project's primary performance threshold" overclaims at this sample size.
- Standard violated: S1 — the success_metric is "Pool RPS must beat Dixon-Coles RPS on scored WC 2026 matches"; the business-context file does set this gate on live matches (line 64), so the framing is technically licensed by the business context — but the conclusion "fails" should be paired with a CI showing the difference is not distinguishable from zero.
- Recommended fix: restate as "On n=12, pool RPS is 0.0052 above DC; with bootstrap 90% CI of [X, Y], this difference is not statistically distinguishable from zero. The gate is technically not met but cannot be conclusively evaluated until more matches are scored." Update both the run record (lines 31, 36) and the decisions table.

### F4 (minor). Draw-baseline figure in Finding #5 cites the wrong number
- Location: `memory/runs/2026-06-15_eda_diagnostics_v2.md:48` and `script:331`.
- Issue: Finding #5 says draw rate of 0.333 "in this sample" — correct (4/12). But the *neutral-WC historical* draw rate is stated as 0.228; re-derivation gives 0.228 (228/843). OK. However, the framing "Every layer assigned lower draw probability than the draw actually occurred" is too strong: pool mean P(draw)=0.244, sample rate=0.333, but mean P(draw | actual draw)=0.244 (no conditional discrimination — pool gave the *same* P(draw) to draws and non-draws). The interesting finding is **the pool has no draw discrimination**, not that it "underestimates" draws — its unconditional mean (0.244) is in fact close to the neutral-WC historical (0.228).
- Standard violated: S5 — narrative consistency between numbers and interpretation.
- Recommended fix: rephrase Finding #5 as "Pool's draw probability has no discrimination on this sample: mean P(draw | actual draw) = 0.244 vs mean P(draw | non-draw) = 0.244. The unconditional mean is calibrated to the neutral-WC base rate (0.228), but in this sample 4/12 matches drew. With n=4 draws, this is not yet a reliable signal of mis-calibration."

### F5 (minor). Pool-mean-P(H) vs actual-H-rate framed as bias
- Location: `script:404-411`, run record Finding #6.
- Issue: Pool mean P(H) = 0.424 vs actual WC 2026 H-rate 0.583 (n=12). The script says pool "underestimates home wins by 0.159." With n=12, the 95% binomial CI on the H-rate is roughly [0.28, 0.85]; the pool's 0.424 is inside this CI. Same caveat applies to the neutral-WC historical 0.433 — the pool's mean prediction is essentially *at* the neutral-WC historical, and the WC 2026 sample H-rate is higher than that historical (but not significantly so on n=12). Framing as "pool underestimates" presumes the sample H-rate is the truth.
- Standard violated: S5 — uncertainty as certainty.
- Recommended fix: rephrase as "Pool mean P(H) (0.424) matches the neutral-WC historical baseline (0.433). The n=12 WC 2026 sample H-rate (0.583) is higher than both, but a binomial 95% CI on n=12 spans [0.28, 0.85] — too wide to conclude the pool is mis-calibrated for home advantage."

### F6 (minor). Calibration "36 prob-outcome pairs" reuses the same 12 matches
- Location: `script:533-558`, Decision 4.
- Issue: Decision 4 calls flattening (probs, outcomes) to 36 points "class-conditional calibration", and Decision 4 notes the n=12 limitation, but the reliability diagram bin counts will look like there is more information than there is — the 36 points are not independent (each match contributes one 1 and two 0s, all correlated). This is fine if interpreted carefully but the legend label "(36 class-prob points from 12 matches)" understates the dependence. ECE values reported (Bayes 0.177, Pool 0.276) are still on the 12-match argmax-confidence definition in `metrics.py:35`, which is appropriate, but the visual on the plot could mislead.
- Standard violated: S2 — uncertainty reporting.
- Recommended fix: add a note to the plot caption that the 36 points are not independent observations.

### F7 (minor). Business-alignment gap: findings stop at "what is broken" without quantifying expected RPS impact
- Location: `memory/runs/2026-06-15_eda_diagnostics_v2.md:58-66` ("Concerns Carrying Into S3").
- Issue: The business context (line 74) defines "good": "a prioritized list of specific, implementable improvements with estimated RPS impact." The concerns table prioritizes (HIGH/MEDIUM/LOW) but does not estimate RPS impact for any change. For example: dropping GBM to 0.0 — quantifiable from the data and not done. Reducing market weight 0.35 → 0.20 — quantifiable and not done.
- Standard violated: S5/business-alignment — does the analysis answer the asked question?
- Recommended fix: add a counterfactual-pool sweep over weight settings (drop GBM; bayes ∈ {0.35, 0.45, 0.55, 0.65, 0.75}; market ∈ {0.20, 0.35, 0.55}) and report per-match RPS for each, with the caveat that this is n=12 in-sample tuning and the Hedge update will refine these once activated.

### F8 (informational, not a defect). Honest-scoring and leakage checks pass
- Checked: `predictions_log.parquet` `logged_at = 2026-06-11T22:28:38` for all 20 rows; all scored matches are dated >= 2026-06-11. Predictions were logged before any kickoff. Inner join on (date, home_team, away_team) returns exactly 12 rows. No back-filling, no post-kickoff information. The append-only invariant is preserved.

## Reflection lenses

(a) **Data-quality**: matches the corrected data dictionary exactly (7H/4D/1A, n=12, 2 null-market rows). Pool probabilities sum to 1 per dictionary. Pass.

(b) **Stat-sanity**: weakest dimension. No bootstrap CIs, no paired tests, ranks across 5 layers on n=10–12. Findings #1 and #4 in the run record make specific within-noise rank claims. The decisions file's Decision 6 correctly defers MI/KL to n=30+, but the same logic should apply to pairwise layer comparisons. **This is the primary basis for REVISE.**

(c) **Business-alignment**: partially answers the question. Identifies GBM, market weight, and draw discrimination as targets but does not quantify expected RPS deltas of any specific change. The business ask explicitly wants "a prioritized list ... with estimated RPS impact."

## Verdict: REVISE-WITH-NOTES

The work is fundamentally sound: numbers reproduce, no leakage, honest scoring, correct ground truth after the dictionary correction. The pooling-family bug in the counterfactual does not change directional conclusions but should be fixed for integrity. The main gap is uncertainty quantification (F2/F3/F5) and business-aligned counterfactual sweeps (F7).

## Recommended next action

Author should: (1) swap `compute_pool` to log-opinion to match production, (2) add bootstrap CIs to RPS comparisons, (3) add a small weight-sweep counterfactual block. None of these block S3 from starting on parallel concerns; they should be addressed before the run record is treated as a basis for weight changes in production.
