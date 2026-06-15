# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup (one-time)
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Core operations (via Makefile; Python is at .venv/bin/python)
make data       # pull latest international results from martj42/international_results
make backtest   # walk-forward eval on WC 2014/2018/2022 + Euro 2024 + Copa 2024
make predict    # fit all models and predict upcoming WC 2026 matches (scripts/predict_next.py)
make simulate   # 100k Monte Carlo runs of the full 48-team 2026 bracket
make update     # full post-match cycle: score, refit, log forecasts, simulate, report, push artifacts
make test       # pytest -q (tests/)

# Run single test file
.venv/bin/pytest tests/test_baseline.py -q

# Run update cycle directly
.venv/bin/python -m mundial.online.update_cycle

# Morning digest (cloud; uses Telegram)
python scripts/cloud_digest.py
```

Environment variables go in `.env` (never committed): `ODDS_API_KEY` for The Odds API, `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` for digest delivery. `update_cycle.py` calls `load_env()` to read this file.

## Architecture

Four-layer prediction stack with an online learning loop running through the 2026 World Cup (kicked off June 11, 2026).

### Modeling layers (`src/mundial/models/`)

| Layer | File | Role |
|---|---|---|
| A — Bayesian backbone | `bayes.py` | `DynamicHierarchicalPoisson`: bivariate Poisson with Dixon-Coles low-score correction, per-team attack/defense strengths evolving as Gaussian random walks over 6-month periods, partially pooled by confederation (via NumPyro SVI). The only component that produces a full scoreline distribution (`score_matrix()`). |
| B — GBM layer | `gbm.py` | `GbmModel`: LightGBM 1X2 classifier consuming contextual features (rest, altitude, etc.) plus Layer A posterior means. |
| C — Market layer | `market.py` | Shin-de-vigged closing odds from The Odds API. Degrades gracefully when unavailable. |
| D — Ensemble | `ensemble.py` | Log-opinion pool: `pool_predict()` / `pool_predict_partial()`. Weights start from LOTO-fitted values and are updated in-tournament by **Hedge (multiplicative weights)** via `hedge_update()`. |
| — | `baseline.py` | `DixonColes`: the gate — every layer must beat this on backtest RPS before it ships. |
| — | `simulate.py` | `TournamentSimulator`: 100k Monte Carlo bracket runs sampling team strengths from the Bayesian posterior; handles 48-team group stage, best-third-place ranking, knockouts, and shootouts. |

### Online learning loop (`src/mundial/online/update_cycle.py`)

The cycle scores previously-logged predictions against new results (never retroactively fills in unlogged forecasts), applies the Hedge weight update, refits models, logs predictions for upcoming matches (immutably — first logged forecast per match is frozen), re-simulates, writes a markdown cycle report, and pushes `artifacts/` + `reports/` to remote.

State: `artifacts/online_state.json` (weights, score history, CUSUM). Prediction record: `artifacts/predictions_log.parquet` (append-only). CUSUM threshold 3.0 triggers a market-outperformance alarm in cycle reports.

### Data pipeline (`src/mundial/ingest/`)

- `results.py`: pulls `martj42/international_results` results.csv into dated immutable directories under `data/raw/martj42/<date>/`; `load_results()` returns played matches only, `load_fixtures()` returns rows with null scores (upcoming fixtures).
- `confederations.py`: team → confederation mapping for hierarchical pooling.

### Evaluation (`src/mundial/eval/`)

- `metrics.py`: `rps()`, `log_loss()`, `brier()`, `ece()`.
- `backtest.py`: walk-forward eval on historical tournaments; never random K-fold — all splits are strictly temporal. Persists the final stacked weights to `artifacts/`.

### Cloud delivery

`scripts/cloud_digest.py` is the self-contained morning digest: pulls results, refits models, pools with committed Hedge weights from `artifacts/pool_weights.json`, simulates, sends Telegram message. Runs daily at 12:45 UTC via `.github/workflows/digest.yml` using repo secrets. Sends a failure notice (not a silent no-op) if it breaks.

### Key invariants

- **No leakage**: every feature must be timestamped strictly before kickoff. Backtest uses `as_of = tournament_start_date`.
- **Honest scoring**: only pre-kickoff logged predictions are ever scored against results. `append_log()` is anti-joined to prevent overwriting earlier forecasts.
- **Degradation tolerance**: odds fetch failures (and any missing unconventional feature) must not block the daily run — components fall back to their absence with rescaled weights.
- **Backtest gate**: each layer must beat `DixonColes` walk-forward RPS on the 5 reference tournaments before going live.

### Reference data

`data/reference/bracket_2026.json` — 48-team bracket topology (group definitions, knockout pairings, third-place rules).  
`data/reference/groups_2026.json` — official group assignments.
