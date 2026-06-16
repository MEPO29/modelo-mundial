---
stage: "S1 — data acquisition"
author: "ds-data-engineer"
date: "2026-06-15"
---

# Data Acquisition Decisions (2026-06-15)

## 1. Data refresh: `make data` succeeded

Ran `make data` which called `mundial.ingest.results.download_results()`. The upstream source (martj42/international_results) returned 49,477 matches with a latest played date of 2026-06-14. The download created `data/raw/martj42/2026-06-15/results.csv`.

**Observation:** The upstream source includes 60 upcoming WC 2026 fixtures (through June 27) with null scores. These are scheduled matches, not results. The `download_results()` function correctly handles these as NA → null.

## 2. CSV parsing: `null_values=['NA']` required

The results CSV uses the string `NA` for null scores in upcoming matches. Polars' default parser fails on this. The correct invocation is:

```python
pl.read_csv(path, null_values=['NA'])
```

This is a known characteristic of the martj42 dataset format.

## 3. Join strategy: exact match on (date, home_team, away_team)

Used a left join from predictions to results on all three columns. Team names matched exactly between the predictions log and the results CSV — no fuzzy matching or normalization was needed. This was verified by the successful join returning 12 matched rows out of 20 predictions.

**Alternative considered:** Joining on date + sorted(team_a, team_b) to handle home/away swaps. Rejected because the predictions log and results CSV use consistent home/away conventions.

## 4. Tournament filter: `tournament == "FIFA World Cup"`

Used exact string match for "FIFA World Cup" to isolate WC matches. The dataset contains 1,036 total WC matches (historical + 2026). Qualification matches use a different tournament name ("FIFA World Cup qualification"). No ambiguity.

## 5. Could not profile historical odds Excel

The `data/reference/WorldCup2026-football-data-co-uk.xlsx` file could not be read because neither `openpyxl` nor `fastexcel` is installed in the project virtualenv. This blocks profiling of historical closing odds and xG reference data. Recommended fix: `pip install openpyxl`.

## 6. Tool path: local Python (polars) fallback

No MCP servers (postgres, SQL) are available per AGENTS.md. All profiling was done with local Python using polars 1.x in the project's `.venv/`. This is the expected path for this project.
