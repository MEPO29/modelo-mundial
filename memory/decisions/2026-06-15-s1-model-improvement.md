# Decision: S1 Business Framing — Model Improvement (2026-06-15)

## What was decided

Framed the user's request ("Make this model better") as a **mid-tournament forecast evaluation and iterative model improvement** task, not a from-scratch model-building exercise. The analytical work decomposes into four sub-tasks: (1) data freshness restoration, (2) honest forecast scoring against actual results, (3) online learning activation, and (4) diagnostic-driven improvement identification.

## Why this framing

- The model architecture (four-layer stack with Hedge ensemble) is already built, backtested, and validated. The user is not asking for a new model — they are asking for the existing model to perform better.
- The tournament is live (day 5 of ~40). The highest-leverage action is activating the dormant online learning loop, which the system was explicitly designed to do daily.
- The user specifically noted "we already have match results we can use to test the model" — this signals that the evaluation-then-improve workflow is the intended path, not a greenfield redesign.
- The primary metric (RPS), success thresholds (beat Dixon-Coles, approach market), and invariants (no leakage, honest scoring, append-only log) are already well-defined in the codebase and AGENTS.md.

## Alternatives considered and rejected

- **Full model rebuild:** Rejected. The architecture is sound and validated on 5 reference tournaments. A rebuild mid-tournament would invalidate the predictions log and break the honest-scoring invariant.
- **Backtest-only improvement (no live scoring):** Rejected. The user explicitly wants to use actual WC 2026 match results. Backtest improvements on historical tournaments are useful but don't address the live-tournament ask.
- **Betting-oriented optimization:** Rejected. No evidence the stakeholder wants to optimize expected return against bookmaker odds. The project is framed as forecast quality.

## Key risk flagged

Weight discrepancy between `INITIAL_WEIGHTS` in code (bayes=0.35, market=0.55) and `online_state.json` (bayes=0.55, market=0.35). This should be investigated before the first update cycle runs, as it affects the Hedge starting point.

## Next stage

Invoke `ds-data-engineer` (via `/ds-explore`) to: (1) pull fresh results, (2) run the update cycle, (3) produce scored-match diagnostics and layer-level performance decomposition.
