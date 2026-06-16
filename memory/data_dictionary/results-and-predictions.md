---
source: "martj42/international_results (GitHub) + local artifacts"
path_or_connection:
  results: "data/raw/martj42/2026-06-15/results.csv"
  predictions: "artifacts/predictions_log.parquet"
  online_state: "artifacts/online_state.json"
  pool_weights: "artifacts/pool_weights.json"
  odds: "data/raw/odds/2026-06-11T*.json (3 snapshots)"
  bracket: "data/reference/bracket_2026.json"
  groups: "data/reference/groups_2026.json"
  venues: "data/reference/venues_2026.csv"
  altitudes: "data/reference/city_altitudes.csv"
  historical_odds: "data/reference/WorldCup2026-football-data-co-uk.xlsx"
row_count:
  results: 49477
  predictions: 20
  wc2026_total: 72
  wc2026_played: 12
  wc2026_upcoming: 60
column_count:
  results: 9
  predictions: 20
  venues: 10
  altitudes: 2
last_refreshed: "2026-06-15 (via make data -> download_results())"
owner: "project developer"
pii_present: false
sensitivity: "public — no PII, no compliance regime"
freshness_sla: "daily during tournament (June 11 – July 19, 2026)"
changelog:
  - "2026-06-15 correction pass: fixed section 3 scoreable-predictions table (Brazil vs Morocco corrected to draw, Denmark vs Cameroon removed — not in predictions log, Haiti vs Scotland added as away win); corrected outcome distribution to 7H/4D/1A; profiled WorldCup2026-football-data-co-uk.xlsx (openpyxl now available); removed blocker #1 from section 8."
---

# Data Dictionary: Results & Predictions (2026-06-15 refresh)

## 1. Historical Match Results

**Path:** `data/raw/martj42/2026-06-15/results.csv`
**Source:** [martj42/international_results](https://github.com/martj42/international_results) on GitHub
**Refreshed:** 2026-06-15 via `make data` (calls `mundial.ingest.results.download_results()`)
**Rows:** 49,477 | **Columns:** 9 | **Date range:** 1872-11-30 to 2026-06-27

### Column-level schema

| Column | Dtype | Description | Null % | Example values | Known issues |
|--------|-------|-------------|--------|----------------|--------------|
| `date` | String (YYYY-MM-DD) | Match date | 0.00% | "2026-06-14", "1872-11-30" | Stored as string, not Date type; parse with `pl.col("date").str.to_date()` |
| `home_team` | String | Home team name | 0.00% | "Mexico", "Germany", "Sweden" | 327 unique teams; names may differ across eras (e.g. "West Germany" vs "Germany") |
| `away_team` | String | Away team name | 0.00% | "South Africa", "Curaçao" | 321 unique teams |
| `home_score` | Int64 | Goals scored by home team | 0.12% (60 rows) | 0, 1, 2, 7 | Null = match not yet played (upcoming fixtures through 2026-06-27) |
| `away_score` | Int64 | Goals scored by away team | 0.12% (60 rows) | 0, 1, 2 | Null = match not yet played; always null when home_score is null |
| `tournament` | String | Competition name | 0.00% | "FIFA World Cup", "Friendly", "FIFA World Cup qualification" | 200 unique tournament names |
| `city` | String | Host city | 0.00% | "Mexico City", "Houston", "Zapopan" | 2,089 unique cities |
| `country` | String | Host country | 0.00% | "Mexico", "United States", "Canada" | 269 unique countries |
| `neutral` | Boolean | Whether match was on neutral ground | 0.00% | true (13,121), false (36,356) | — |

### Quality checks

| Check | Result | Notes |
|-------|--------|-------|
| Full-row duplicates | **0** | All 49,477 rows are unique |
| Key-based duplicates (date + home + away) | **2** | (1) Gibraltar vs Cayman Islands 2026-06-06: same score (4-?) in two cities (Gibraltar, Europa Point) — likely same match logged twice with different city names. (2) Tahiti vs New Caledonia 1974-02-17: different scores (2-? vs 1-?) — possibly two matches same day or data error. Neither affects WC 2026 analysis. |
| Null pattern | home_score and away_score are always null together (60 rows) | These are upcoming WC 2026 fixtures (June 15–27). Expected behavior. |
| Score range (all played) | Home: 0–31, Away: 0–21 | Historical outliers exist (e.g. Australia 31–0 American Samoa) but are legitimate |
| Score range (WC 2026 played) | Home: 0–7, Away: 0–2 | 12 matches played; home teams averaging 2.33 goals, away 0.83 |
| WC 2026 result distribution | 7 home wins, 4 draws, 1 away win | Small sample; strong home advantage so far |
| Tournament filter | "FIFA World Cup" = 1,036 total (historical + 2026) | Use `tournament == "FIFA World Cup"` for WC matches only |
| Date parsing | String format YYYY-MM-DD throughout | No sentinel values (1900-01-01, 9999-12-31) detected |

### WC 2026 match schedule by date

| Date | Total matches | Played | Upcoming |
|------|--------------|--------|----------|
| 2026-06-11 | 2 | 2 | 0 |
| 2026-06-12 | 2 | 2 | 0 |
| 2026-06-13 | 4 | 4 | 0 |
| 2026-06-14 | 4 | 4 | 0 |
| 2026-06-15 | 4 | 0 | 4 |
| 2026-06-16 | 4 | 0 | 4 |
| 2026-06-17–23 | 4 each day | 0 | 28 |
| 2026-06-24–27 | 6 each day | 0 | 24 |

**Latest played date:** 2026-06-14. Today's matches (June 15) have not yet been scored in the upstream source.

---

## 2. Predictions Log

**Path:** `artifacts/predictions_log.parquet`
**Source:** Generated by `mundial.online.update_cycle` on 2026-06-11
**Rows:** 20 | **Columns:** 20 | **Append-only:** yes (frozen once logged)

### Column-level schema

| Column | Dtype | Description | Null % | Range / Example | Known issues |
|--------|-------|-------------|--------|-----------------|--------------|
| `date` | String | Match date (YYYY-MM-DD) | 0.00% | "2026-06-11" to "2026-06-16" | — |
| `home_team` | String | Home team name | 0.00% | "Mexico", "Germany", etc. | Must match results CSV team names exactly (verified: join works) |
| `away_team` | String | Away team name | 0.00% | "South Africa", "Curaçao", etc. | — |
| `neutral` | Boolean | Neutral venue flag | 0.00% | true/false | — |
| `dc_h`, `dc_d`, `dc_a` | Float64 | Dixon-Coles probabilities (home/draw/away) | 0.00% | h: 0.06–0.94, d: 0.05–0.36, a: 0.01–0.80 | Sum = 1.000000 for all rows |
| `bayes_h`, `bayes_d`, `bayes_a` | Float64 | Bayesian backbone probabilities | 0.00% | h: 0.08–0.97, d: 0.03–0.30, a: 0.01–0.72 | Sum = 1.000000 for all rows |
| `gbm_h`, `gbm_d`, `gbm_a` | Float64 | GBM layer probabilities | 0.00% | h: 0.06–0.94, d: 0.04–0.37, a: 0.02–0.78 | Sum = 1.000000 for all rows |
| `market_h`, `market_d`, `market_a` | Float64 | Market-implied probabilities (de-vigged) | **10.00%** (2 rows) | h: 0.06–0.94, d: 0.05–0.34, a: 0.01–0.81 | Null for Mexico vs South Africa and South Korea vs Czech Republic (June 11 opening matches — odds not available from The Odds API at prediction time). Sum = 1.000000 where non-null. |
| `pool_h`, `pool_d`, `pool_a` | Float64 | Ensemble pool probabilities (weighted combination) | 0.00% | h: 0.08–0.96, d: 0.03–0.32, a: 0.01–0.75 | Sum = 1.000000 for all rows. Uses online_state weights (dc=0.05, bayes=0.55, gbm=0.05, market=0.35) with partial-pool rescaling when market is null. |
| `logged_at` | String | Timestamp when prediction was logged | 0.00% | "2026-06-11T22:28:38" (all identical) | All 20 predictions logged at same time — batch generation |

### Predictions by date

| Date | Count |
|------|-------|
| 2026-06-11 | 2 |
| 2026-06-12 | 2 |
| 2026-06-13 | 4 |
| 2026-06-14 | 4 |
| 2026-06-15 | 4 |
| 2026-06-16 | 4 |

---

## 3. Scoreable Predictions (Join Results)

**Join key:** `(date, home_team, away_team)` — exact match on all three columns.
**Join type:** Inner join from predictions log to results (filtered to `tournament == "FIFA World Cup"` and `date >= "2026-06-11"` and `home_score IS NOT NULL`).
**Derived:** 2026-06-15 correction pass using polars; join computed from raw files, not hand-edited.

**Key finding from correction pass:** The prior version of this table contained two errors:
- "Brazil vs Morocco 2–1 (home win)" — the actual result was 1–1 (draw). Corrected.
- "Denmark vs Cameroon 1–0 (home win)" — this match was NOT in the predictions log. The prediction logged for June 13 at that slot was Haiti vs Scotland. Removed Denmark/Cameroon; Haiti vs Scotland (away win, 0–1) correctly replaces it.

### 12 scoreable predictions

| Date | Home | Away | Score | Outcome | Pool (H/D/A) |
|------|------|------|-------|---------|--------------|
| 2026-06-11 | Mexico | South Africa | 2–0 | Home win | 0.618 / 0.244 / 0.138 |
| 2026-06-11 | South Korea | Czech Republic | 2–1 | Home win | 0.310 / 0.300 / 0.389 |
| 2026-06-12 | Canada | Bosnia and Herzegovina | 1–1 | Draw | 0.473 / 0.271 / 0.256 |
| 2026-06-12 | United States | Paraguay | 4–1 | Home win | 0.430 / 0.286 / 0.284 |
| 2026-06-13 | Qatar | Switzerland | 1–1 | Draw | 0.081 / 0.173 / 0.746 |
| 2026-06-13 | Brazil | Morocco | 1–1 | **Draw** | 0.580 / 0.267 / 0.153 |
| 2026-06-13 | Haiti | Scotland | 0–1 | **Away win** | 0.142 / 0.223 / 0.635 |
| 2026-06-13 | Australia | Turkey | 2–0 | Home win | 0.253 / 0.278 / 0.469 |
| 2026-06-14 | Germany | Curaçao | 7–1 | Home win | 0.957 / 0.033 / 0.010 |
| 2026-06-14 | Ivory Coast | Ecuador | 1–0 | Home win | 0.245 / 0.317 / 0.438 |
| 2026-06-14 | Netherlands | Japan | 2–2 | Draw | 0.505 / 0.264 / 0.231 |
| 2026-06-14 | Sweden | Tunisia | 5–1 | Home win | 0.488 / 0.279 / 0.233 |

**Outcome distribution (computed):** 7 home wins, 4 draws, 1 away win.

> **Notable model misses:** Qatar vs Switzerland predicted as strong away win (74.6% Czech Republic) but ended 1–1 draw. Brazil vs Morocco predicted as home win (58.0%) but ended 1–1 draw. Haiti vs Scotland: away win correctly favoured (63.5% Scotland) and Scotland won. Australia vs Turkey: away win slightly favoured (46.9%) but Australia won 2–0. South Korea vs Czech Republic: away win most likely (38.9%) but South Korea won.

### 8 predictions not yet scoreable (matches not yet played as of 2026-06-15)

| Date | Home | Away | Pool (H/D/A) |
|------|------|------|--------------|
| 2026-06-15 | Belgium | Egypt | 0.601 / 0.243 / 0.157 |
| 2026-06-15 | Iran | New Zealand | 0.585 / 0.272 / 0.143 |
| 2026-06-15 | Spain | Cape Verde | 0.859 / 0.104 / 0.037 |
| 2026-06-15 | Saudi Arabia | Uruguay | 0.094 / 0.208 / 0.698 |
| 2026-06-16 | France | Senegal | 0.598 / 0.252 / 0.150 |
| 2026-06-16 | Iraq | Norway | 0.109 / 0.211 / 0.679 |
| 2026-06-16 | Argentina | Algeria | 0.695 / 0.205 / 0.100 |
| 2026-06-16 | Austria | Jordan | 0.680 / 0.201 / 0.118 |

---

## 4. Online State

**Path:** `artifacts/online_state.json`

| Field | Value | Notes |
|-------|-------|-------|
| `weights.dc` | 0.05 | Dixon-Coles weight |
| `weights.bayes` | 0.55 | Bayesian backbone weight |
| `weights.gbm` | 0.05 | GBM layer weight |
| `weights.market` | 0.35 | Market layer weight |
| `scored` | [] (empty) | No matches scored yet |
| `score_history` | [] (empty) | No scoring history |
| `cusum` | 0.0 | CUSUM drift statistic at zero |

**Known issue:** The `INITIAL_WEIGHTS` constant in `update_cycle.py` is `{dc: 0.05, bayes: 0.35, gbm: 0.05, market: 0.55}`, but the online state has bayes=0.55 and market=0.35 (swapped). This may indicate manual adjustment or a prior code change.

---

## 5. Pool Weights (Backtest-Fitted)

**Path:** `artifacts/pool_weights.json`

| Component | Weight | Notes |
|-----------|--------|-------|
| `dc` | ~0 (2.14e-14) | Effectively zero |
| `bayes` | 1.015 | ~100% of stacking weight |
| `gbm` | ~0 (1.18e-14) | Effectively zero |

These are the backtest-optimized stacking weights. They differ from the online Hedge weights (which include the market layer and are initialized more conservatively).

---

## 6. Reference Datasets (Static)

### bracket_2026.json
- **Structure:** Dict with keys `_source`, `round_of_32`, `round_of_16`, `quarterfinals`, `semifinals`, `final`
- **Content:** Knockout stage bracket using slot syntax (e.g. "1A" = winner of group A)
- **Source:** Wikipedia, fetched 2026-06-11
- **16 round-of-32 matches, 8 round-of-16, 4 QF, 2 SF, 1 final**

### groups_2026.json
- **Structure:** Dict with 12 group keys (A through L), each a list of 4 team names
- **Content:** Group stage assignments for all 48 teams

### venues_2026.csv
- **Rows:** 16 | **Columns:** 10
- **Columns:** city, stadium, country, lat, lon, altitude_m, roof, climate_controlled, surface, tz
- **Coverage:** All 16 WC 2026 venues across USA, Mexico, Canada

### city_altitudes.csv
- **Rows:** 40 | **Columns:** 2 (city, altitude_m)
- **Content:** Altitude data for cities worldwide (used as contextual feature for GBM)

### WorldCup2026-football-data-co-uk.xlsx

**Path:** `data/reference/WorldCup2026-football-data-co-uk.xlsx`
**Source:** football-data.co.uk (static reference file)
**Profiled:** 2026-06-15 using pandas with openpyxl engine
**Sheets:** 4 (`WorldCup2026Qualifiers`, `WorldCup2022`, `WorldCup2018`, `WorldCup2014`)

#### Sheet summary

| Sheet | Rows | Cols | Date range | Content |
|-------|------|------|------------|---------|
| WorldCup2026Qualifiers | 889 | 25 | 2023-09-07 to 2026-04-01 | WC 2026 qualification matches: scores, match stats, decimal odds, xG (partial) |
| WorldCup2022 | 64 | 40 | 2022-11-20 to 2022-12-18 | All 64 WC 2022 matches: scores by half/ET/penalties, match stats, bet365 + Betfair odds |
| WorldCup2018 | 64 | 37 | 2018-06-14 to 2018-07-15 | All 64 WC 2018 matches: scores by half/ET/penalties, match stats, Pinnacle odds |
| WorldCup2014 | 64 | 40 | 2014-06-12 to 2014-07-13 | All 64 WC 2014 matches: scores by half/ET/penalties, match stats, bet365 + Pinnacle odds |

#### WorldCup2026Qualifiers — column-level schema

| Column | Dtype | Description | Null % | Range / Notes |
|--------|-------|-------------|--------|---------------|
| `Date` | datetime64 | Match date | 0.00% | 2023-09-07 to 2026-04-01 (87 unique dates) |
| `Home` | str | Home team name | 0.00% | 201 unique teams |
| `Away` | str | Away team name | 0.00% | 201 unique teams |
| `HG` | int64 | Home goals (FT) | 0.00% | 0–11, mean=1.60 |
| `AG` | int64 | Away goals (FT) | 0.00% | 0–8, mean=1.21 |
| `H_Max` | object | Max home win decimal odds across bookmakers | 0.00% | Stored as object; 1 row contains "-" sentinel; otherwise numeric (e.g. 1.22–31) |
| `D_Max` | object | Max draw decimal odds | 0.00% | Same issue: 1 "-" sentinel row |
| `A_Max` | object | Max away win decimal odds | 0.00% | Same issue: 1 "-" sentinel row |
| `H_Avg` | object | Average home win decimal odds | 0.00% | Same issue: 1 "-" sentinel row |
| `D_Avg` | object | Average draw decimal odds | 0.00% | Same issue: 1 "-" sentinel row |
| `A_Avg` | object | Average away win decimal odds | 0.00% | Same issue: 1 "-" sentinel row |
| `HS` | float64 | Home shots | 19.3% | 0–40, mean=11.8 |
| `AS` | float64 | Away shots | 19.3% | 0–33, mean=9.5 |
| `HST` | float64 | Home shots on target | 19.3% | 0–22, mean=4.52 |
| `AST` | float64 | Away shots on target | 19.3% | 0–17, mean=3.67 |
| `HF` | float64 | Home fouls | 23.2% | 0–33, mean=12.0 |
| `AF` | float64 | Away fouls | 23.2% | 0–37, mean=12.1 |
| `HC` | float64 | Home corners | 19.3% | 0–18, mean=4.99 |
| `AC` | float64 | Away corners | 19.3% | 0–17, mean=4.06 |
| `HY` | float64 | Home yellow cards | 19.3% | 0–6, mean=1.60 |
| `AY` | float64 | Away yellow cards | 19.3% | 0–7, mean=1.81 |
| `HR` | float64 | Home red cards | 19.3% | 0–2, mean=0.054 |
| `AR` | float64 | Away red cards | 19.3% | 0–2, mean=0.071 |
| `HxG` | float64 | Home expected goals | **61.9%** | 0–9.57, mean=1.57 (where non-null); coverage: 2023=8%, 2024=6%, 2025=61%, 2026=100% |
| `AxG` | float64 | Away expected goals | **61.9%** | 0–6.34, mean=1.18 (where non-null); same coverage pattern as HxG |

#### WorldCup2022 — column-level schema (selected)

All 64 matches, 0 nulls on core columns. Detailed score breakdown by half, extra time, and penalties available.

| Column group | Columns | Notes |
|-------------|---------|-------|
| Scores | `HGFT`, `AGFT`, `HG1st`, `AG1st`, `HG2nd`, `AG2nd`, `HGET`, `AGET`, `HGP`, `HGP.1` | FT / half-time / extra-time / penalty goals. HGET/AGET/HGP null for 92.2% of rows (only knockout matches that went to ET/pens). |
| Match status | `Finished` | "90 minutes", "Extra time", "Penalties" |
| Match stats | `HS`, `AS`, `HST`, `AST`, `HF`, `AF`, `HC`, `AC`, `HY`, `AY`, `HR`, `AR` | 0.00% null. Shots, shots-on-target, fouls, corners, cards. |
| Odds | `bet365-H/D/A`, `Betfair_Exch-H/D/A`, `H-Max`, `D-Max`, `A-Max`, `H-Avg`, `D-Avg`, `A-Avg` | 0.00% null. Decimal format. Range: H ~1.11–26, D ~3.0–11, A ~1.08–23. |
| xG | — | **No xG columns in WC2022 sheet.** |

#### WorldCup2018 — column-level schema (selected)

All 64 matches. Pinnacle odds only (no bet365 or Betfair). No xG.

| Column group | Columns | Notes |
|-------------|---------|-------|
| Scores | `HGFT`, `AGFT`, `HG1st`, `AG1st`, `HG2nd`, `AG2nd`, `HGET`, `AGET`, `HGP`, `HGP.1` | HGET/AGET null 92.2%; HGP/HGP.1 null 93.8%. |
| Odds | `Pinny-H`, `Pinny-D`, `Pinny-A`, `H-Max`, `D-Max`, `A-Max`, `H-Avg`, `D-Avg`, `A-Avg` | 0.00% null. Decimal odds, range: H ~1.21–20.1. |
| xG | — | **No xG columns.** |

#### WorldCup2014 — column-level schema (selected)

All 64 matches. bet365 + Pinnacle odds. No xG.

| Column group | Columns | Notes |
|-------------|---------|-------|
| Scores | `HGFT`, `AGFT`, `HG1st`, `AG1st`, `HG2nd`, `AG2nd`, `HGET`, `AGET`, `HGP`, `HGP.1` | HGET/AGET null 87.5%; HGP/HGP.1 null 93.8%. |
| Odds | `bet365-H/D/A`, `Pinny-H/D/A`, `H-Max`, `D-Max`, `A-Max`, `H-Avg`, `D-Avg`, `A-Avg` | 0.00% null. Decimal odds, range: H ~1.14–26. |
| xG | — | **No xG columns.** |

#### Quality issues and caveats

1. **Odds stored as `object` dtype in WorldCup2026Qualifiers.** The six odds columns (`H_Max`, `D_Max`, `A_Max`, `H_Avg`, `D_Avg`, `A_Avg`) are `object` rather than float because 1 row (0.1% of 889) contains a `"-"` sentinel value instead of a number. Before numeric use, cast with `pd.to_numeric(..., errors='coerce')` to replace the sentinel with NaN.

2. **xG is sparse in WorldCup2026Qualifiers.** Only 339 of 889 rows (38.1%) have xG. Coverage is near-zero for 2023 (8%) and 2024 (6%), rising to 61% in 2025 and 100% for the 16 matches in 2026. If xG is used as a GBM feature, it cannot be applied uniformly across the qualifier history — models trained on the sparse subset may overfit the bookmaker-odds features instead.

3. **No xG in historical WC sheets (2014/2018/2022).** The three finished-tournament sheets contain only match stats (shots, corners, cards) and decimal odds. There are no xG columns. xG is only available in the qualifiers sheet (partially). This limits retrospective xG analysis to qualifier matches.

4. **Team name mismatch: "USA" vs "United States".** The WorldCup2022 sheet uses "USA" for the United States. The martj42/international_results source uses "United States". A name-mapping step is required before any join between this file and results.csv for WC 2022. (WC 2014 and 2018 do not include USA in the bracket.) Check other potential mismatches (e.g. "Bosnia & Herzegovina" in qualifiers vs "Bosnia and Herzegovina" in martj42) before joining.

5. **ET/penalty columns are mostly null.** `HGET`, `AGET`, `HGP`, `HGP.1` are non-null only for knockout matches that reached extra time or penalties: ~8% of rows (WC 2022/2018/2014). This is expected and not a data error. Use the `Finished` column to distinguish match type.

### Odds snapshots (data/raw/odds/)
- **Files:** 3 JSON snapshots from 2026-06-11 (The Odds API)
- **Structure:** List of 70 events, each with bookmaker odds (h2h market)
- **Stale:** All from June 11; no fresh odds pulled since

---

## 7. PII / Sensitivity Assessment

**PII present:** No. All data is public sports data (match results, team names, odds). No personal identifiers, no player-level data, no addresses, no financial accounts.

**Compliance regime:** None.

**Sensitivity:** Public. No restrictions on use, sharing, or logging.

---

## 8. Data Quality Summary

| Dataset | Rows | Cols | Quality flags |
|---------|------|------|---------------|
| results.csv | 49,477 | 9 | Clean. 2 benign key-duplicates (non-WC). 60 null scores = upcoming fixtures (expected). |
| predictions_log.parquet | 20 | 20 | Clean. 2 rows with null market probs (June 11 opening matches, odds unavailable). All probability triplets sum to 1.000000. |
| online_state.json | — | — | Empty scoring arrays. Weight discrepancy vs. INITIAL_WEIGHTS constant (bayes/market swapped). |
| pool_weights.json | — | — | bayes ~100%, dc/gbm ~0. Consistent with backtest findings. |
| bracket_2026.json | — | — | Well-structured. No issues. |
| groups_2026.json | — | — | 12 groups x 4 teams = 48 teams. No issues. |
| venues_2026.csv | 16 | 10 | Complete. No nulls. |
| city_altitudes.csv | 40 | 2 | Complete. No nulls. |
| WorldCup2026-football-data-co-uk.xlsx (Qualifiers) | 889 | 25 | Odds cols stored as object due to 1 "-" sentinel row. xG sparse (38% coverage, near-zero pre-2025). |
| WorldCup2026-football-data-co-uk.xlsx (WC2022) | 64 | 40 | Clean. No nulls on core columns. "USA" name diverges from martj42 "United States". No xG. |
| WorldCup2026-football-data-co-uk.xlsx (WC2018) | 64 | 37 | Clean. Pinnacle odds only. No xG. |
| WorldCup2026-football-data-co-uk.xlsx (WC2014) | 64 | 40 | Clean. bet365 + Pinnacle. No xG. |

### Blockers

1. **June 15 results not yet in upstream** — Today's 4 matches (Belgium/Egypt, Iran/New Zealand, Spain/Cape Verde, Saudi Arabia/Uruguay) show null scores. The upstream source's latest played date is 2026-06-14. These matches will become scoreable after the next `make data` refresh (expected June 16).

### Non-blocking observations

- **Weight discrepancy** between `INITIAL_WEIGHTS` in code and `online_state.json` values should be investigated by `ds-eda-analyst` or `ds-modeler` before running `make update`.
- **2 key-based duplicates** in the historical data are in Friendly matches (1974, 2026-06-06) and do not affect WC 2026 analysis.
- **Market layer nulls** for the 2 June 11 opening matches are handled by partial-pool rescaling (documented in business context). The pool probabilities for these matches are still valid.
- **Small sample size** (12 scoreable predictions) limits diagnostic power. CUSUM threshold of 3.0 is calibrated for this regime per business context.
- **Team name mismatch** ("USA" vs "United States") between xlsx and martj42 results.csv — non-blocking for current scoring task (scored predictions join directly to results.csv), but must be resolved if xlsx odds are used as GBM training features against historical results.
- **xG sparsity** in qualifier sheet (62% null overall, near-zero before 2025) — non-blocking but limits xG feature utility for historical model training.

---

## Lineage

| Dataset | Source | Last refreshed | Refresh mechanism |
|---------|--------|----------------|-------------------|
| results.csv | github.com/martj42/international_results | 2026-06-15 | `make data` → `mundial.ingest.results.download_results()` |
| predictions_log.parquet | Generated locally | 2026-06-11 | `mundial.online.update_cycle` (logged pre-kickoff) |
| online_state.json | Generated locally | 2026-06-11 | `mundial.online.update_cycle` |
| pool_weights.json | Generated locally | 2026-06-11 | `mundial.eval.backtest` |
| odds/*.json | the-odds-api.com | 2026-06-11 | API fetch (3 snapshots) |
| bracket_2026.json | Wikipedia | 2026-06-11 | Manual fetch |
| groups_2026.json | FIFA | 2026-06-11 | Manual entry |
| venues_2026.csv | Public sources | Static | Manual entry |
| city_altitudes.csv | Public sources | Static | Manual entry |
| WorldCup2026-football-data-co-uk.xlsx | football-data.co.uk | Static reference | Manual download; covers WC 2014/2018/2022 + 2026 qualifiers |
