---
stage: S2
date: 2026-06-15
author: ds-eda-analyst
slug: eda-diagnostics-v2
supersedes: memory/decisions/2026-06-15_eda-diagnostics.md
---

# EDA Decisions v2 — WC 2026 Diagnostics (2026-06-15)

## Decision 1: Recompute all statistics from raw data; do not inherit prior run's numbers

**Rationale:** The task explicitly requires recomputing from raw CSV and parquet files because the prior run (v1) may have been generated before or after the data dictionary correction pass, and the ground truth (7H/4D/1A) must be verified from the inner join, not assumed.
**Action:** `assert scored.height == 12` with `assert n_hw == 7 and n_dr == 4 and n_aw == 1` to verify before any metric computation.

## Decision 2: Use v2 suffix on all plot files to avoid silent clobber of prior run

**Rationale:** The prior run's plots are in `artifacts/plots/` without versioning. Rather than decide whether they are correct or not, we write to `v2_*` filenames. Both sets of plots are preserved for comparison.
**Action:** All `plt.savefig` calls use `v2_` prefix.

## Decision 3: Historical home advantage baseline = neutral-venue WC matches only

**Rationale:** WC 2026 is hosted across USA, Mexico, and Canada — all teams are playing in a foreign country. The "home team" label in the match data is a scheduling convention (first-named team), not a geographic advantage. The relevant historical baseline for WC 2026 is neutral-venue WC matches (matches where `neutral=True` in results.csv), not all WC matches (which include host-nation advantages from WC 2014, 2018, 2022 group stages).
**Action:** Compute separate outcome distributions for neutral vs non-neutral historical WC matches; compare WC 2026 against neutral baseline.

## Decision 4: Calibration reliability diagram uses all 3 class probabilities (not just argmax)

**Rationale:** Each match generates 3 (predicted probability, actual indicator) pairs — one per class. Using all 3 triples gives 36 data points from 12 matches, which provides better bin coverage than 12 points, while still appropriately noting the n=12 limitation. This approach is sometimes called "class-conditional calibration."
**Action:** Flatten probs to (36,) and actual indicators to (36,) then bin.

## Decision 5: Weight counterfactual reconstructed via log-opinion pool with partial rescaling for null market

**Rationale:** To fairly compare the actual pool weights (bayes=0.55, market=0.35) vs code INITIAL_WEIGHTS (bayes=0.35, market=0.55), we must apply the same partial-pool rescaling logic that the actual pool uses for the 2 matches where market is null. Simply multiplying by 0.35 would give incorrect denominators.
**Action:** Implement `compute_pool()` function that skips null layers and renormalizes remaining weights.

## Decision 6: Did not run mutual information or KL divergence per prior decision (Decision 4 from v1)

**Rationale:** The decision from v1 EDA stands — n=12 is too small for meaningful MI or KL estimates. Deferred to when 30+ matches are scored.

## Decision 7: Qatar-Switzerland identified as the worst single miss but not singled out for layer-specific diagnosis

**Rationale:** Qatar vs Switzerland (pool predicted 74.6% away win, actual draw) is a clear outlier. However, decomposing why it was wrong (is Qatar stronger than expected? Is Switzerland's away form over-stated?) requires S3 feature engineering diagnostics, not EDA alone. We flag it as the worst miss and carry it forward.

## Decision 8: Weight discrepancy is documented but NOT resolved

**Rationale:** As in v1, the discrepancy between online_state.json (bayes=0.55, market=0.35) and code INITIAL_WEIGHTS (bayes=0.35, market=0.55) is a developer decision, not an EDA decision. The counterfactual analysis shows which was better on n=12, but that is suggestive evidence, not a recommendation to rewrite state.
**Action:** Flag as HIGH priority concern for S3/modeler.
