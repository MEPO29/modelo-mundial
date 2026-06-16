---
stage: S2
date: 2026-06-15
author: ds-eda-analyst
slug: eda-diagnostics
---

# EDA Decisions — WC 2026 Diagnostics (2026-06-15)

## Decision 1: Scored 12 matches using actual CSV data (not data dictionary table)

**Rationale:** The data dictionary's table of 12 scoreable predictions (section 3) contained some scores that differ from the actual `results.csv` data. Specifically:
- Brazil vs Morocco: data dictionary said 2-1 (home win), actual CSV says 1-1 (draw)
- Haiti vs Scotland appears in the actual data (0-1, away win) but was listed as "Denmark vs Cameroon" (1-0, home win) in the data dictionary

**Action:** Used the actual CSV data as ground truth. The data dictionary may have been written from a stale snapshot or with transcription errors.

## Decision 2: Used RPS formula with j in {0, 1} only (not j=2)

**Rationale:** The RPS formula for 3-class ordinal outcomes sums over cumulative differences. The j=2 term is always zero because both cumulative distributions reach 1.0. Including it would add nothing but floating-point noise.

**Action:** Implemented RPS = (1/2) * sum_{j=0}^{1} (cum_pred[j] - cum_actual[j])^2 as specified in the task.

## Decision 3: Calibration bins use all 3 outcome probabilities per match (not just the predicted outcome)

**Rationale:** For calibration analysis, each match generates 3 data points (one per outcome class): the predicted probability for that class and whether that class was the actual outcome. This gives 36 data points from 12 matches, providing better bin coverage than 12 points.

**Alternative considered:** Using only the predicted probability for the actual outcome (1 point per match). Rejected because it conflates calibration with accuracy and gives too few points for meaningful bins.

## Decision 4: Did not run mutual information or KL divergence analysis

**Rationale:** With only 12 scored matches and 5 layers producing probabilities for the same 3 outcomes, information-theoretic measures (MI, KL divergence) would not add meaningful signal beyond the direct scoring metrics (RPS, Brier, log-loss). The sample is too small for reliable distributional comparisons.

**Action:** Deferred these analyses to when more matches are scored (ideally 30+).

## Decision 5: Flagged weight discrepancy but did not attempt to resolve it

**Rationale:** The INITIAL_WEIGHTS constant in `update_cycle.py` has bayes=0.35, market=0.55, but `online_state.json` has bayes=0.55, market=0.35. This is a code/state inconsistency that should be resolved by the developer before running `make update`. The EDA analyst's role is to document, not fix.

**Action:** Documented the discrepancy clearly in the run record and concerns for S3.
