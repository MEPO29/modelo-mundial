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
| A — Bayesian backbone | `bayes.py` | `DynamicHierarchicalPoisson`: **shared-latent bivariate Poisson** (Karlis–Ntzoufras: home=X1+X3, away=X2+X3) — the shared X3 produces real goal correlation / draw mass in the likelihood (closed-form pmf via `numpyro.factor`), replacing the old independent double-Poisson + post-hoc Dixon-Coles tau. Per-team attack/defense strengths evolve as Gaussian random walks over 6-month periods, partially pooled by confederation (NumPyro SVI). The only component that produces a full scoreline distribution (`score_matrix()`); a dormant `prior_offset` hook accepts an external strength prior (Elo/ranking/futures). For knockout ties, `knockout_breakdown()` chains the three resolution stages — 90' 1X2, then extra time (`score_matrix(rate_factor=ET_RATE_FACTOR)`, 30' at 1/3 rate, conditional on a 90' draw), then a coin-flip shootout — into per-stage distributions and overall advancement probabilities, on the same assumptions as the bracket simulator. |
| B — GBM layer | `gbm.py` | `GbmModel`: LightGBM 1X2 classifier consuming contextual features (rest, altitude, etc.) plus Layer A posterior means. **Retired from the live pool** (dead weight: backtest stacking weight ~0, worst live RPS) — kept only as a backtest reference component in `eval/backtest.py`. |
| C — Market layer | `market.py` | Shin-de-vigged closing odds from The Odds API. Degrades gracefully when unavailable. |
| D — Ensemble | `ensemble.py` | Log-opinion pool: `pool_predict()` / `pool_predict_partial()`. Weights start from LOTO-fitted values and are updated in-tournament by **Hedge (multiplicative weights)** via `hedge_update()`. |
| — | `baseline.py` | `DixonColes`: the gate — every layer must beat this on backtest RPS before it ships. |
| — | `simulate.py` | `TournamentSimulator`: 100k Monte Carlo bracket runs sampling team strengths from the Bayesian posterior; handles 48-team group stage, best-third-place ranking, knockouts, and shootouts. |

### Online learning loop (`src/mundial/online/update_cycle.py`)

The cycle scores previously-logged predictions against new results (never retroactively fills in unlogged forecasts), applies the Hedge weight update, refits models, logs predictions for upcoming matches (immutably — first logged forecast per match is frozen), re-simulates, writes a markdown cycle report, and pushes `artifacts/` + `reports/` to remote. The live pool is `dc / bayes / market` (GBM retired). From the Round of 32 (`KNOCKOUT_START`, 2026-06-28), knockout fixtures also log the full FT→ET→penalties resolution (`is_knockout`, `et_*`, `pens_*`, `adv_*`, `p_reach_et`, `p_reach_pens`) and surface it in the cycle report and the morning digest; the headline pool 1X2 stays the 90' result. Each logged prediction also carries the full bayes scoreline grid (`bayes_grid`), scored next cycle on a separate scoreline track (`score_log_loss` / margin `score_rps` / draw calibration in `eval/metrics.py`); legacy rows without a grid are still 1X2-scored. `load_state()` validates/renormalizes persisted weights against the current component set so a stale state file can't silently shadow the prior.

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

## Tool budget

This project is set up for the `ds-agent-framework` (see `memory/`, `notebooks/00_template.ipynb`). MCP availability as of bootstrap (2026-06-15):

- **`mcp__jupyter__*`** — available. Notebook-first agents (EDA, modeling, interpretation) run cells here; `notebooks/` is the working surface.
- **`mcp__postgres__*`** — available, but this project has no database; the data pipeline reads `martj42/international_results` CSVs into `data/raw/`. Treat postgres tools as unused unless a DB is introduced.

Defaults from `.claude/settings.local.json`: read-only Bash/jupyter/postgres calls are pre-approved; writes, `pip install`, and SQL/cell execution are gated to "ask". Local Python lives at `.venv/bin/python` (`.venv/Scripts/python.exe` on Windows).
