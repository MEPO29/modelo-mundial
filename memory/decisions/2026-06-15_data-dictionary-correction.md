---
date: 2026-06-15
stage: S1 (correction pass)
author: ds-data-engineer
---

# Data Dictionary Correction Pass — 2026-06-15

## What was corrected

### Section 3 (Scoreable Predictions)

The prior version of the data dictionary contained two errors in the scoreable-predictions table. Both were verified by re-running an inner join of `artifacts/predictions_log.parquet` against `data/raw/martj42/2026-06-15/results.csv` (filtered to `tournament == "FIFA World Cup"` and `date >= "2026-06-11"` and `home_score IS NOT NULL`).

1. **Brazil vs Morocco listed as 2–1 home win.** The actual result in results.csv is 1–1 (draw). The pool had assigned 58.0% to a home win and 26.7% to a draw. The prior dict incorrectly characterized this as a "home win" and as the most-likely outcome being correct. The correct characterization is a draw at the second-most-likely outcome (26.7%).

2. **Denmark vs Cameroon listed as a scoreable prediction.** This match does not appear in `predictions_log.parquet`. The 20-row predictions log has no row with `home_team == "Denmark"` and `away_team == "Cameroon"`. The actual June 13 prediction in the log at that slot was **Haiti vs Scotland** (Haiti home, Scotland away). The actual result was Scotland winning 0–1 (away win), which the pool correctly favoured at 63.5%.

### Outcome distribution

Corrected from "8 home wins, 3 draws, 0 away wins" to **7 home wins, 4 draws, 1 away win** (computed from the inner join).

### Section 6 (WorldCup2026-football-data-co-uk.xlsx)

The prior entry marked this file as "Could not be profiled — openpyxl not installed". With openpyxl now available, a full profile was run. The file contains 4 sheets: WC2026 qualification matches (889 rows, decimal odds + partial xG), and finished-tournament data for WC2022 (64 rows, bet365+Betfair), WC2018 (64 rows, Pinnacle), and WC2014 (64 rows, bet365+Pinnacle). No xG in the finished-tournament sheets.

### Section 8 (Quality Summary / Blockers)

Removed the openpyxl-install blocker (now resolved). Added xlsx to the quality-summary table. Added new non-blocking observations: team name mismatch ("USA" vs "United States"), xG sparsity in qualifiers.

## Decisions and rationale

- **Inner join used** (not left join). The task requires knowing which predictions genuinely have a corresponding played result. An inner join is the honest join for this purpose; a left join would include unplayed matches.
- **No retroactive score filling.** The predictions log is frozen. The correction here is to the data dictionary text only, not to any logged probability values.
- **Tool used:** polars 2026-06-15 via `.venv/bin/python`. MCP tools not used (no postgres/oracle source). Local-Python path confirmed.
