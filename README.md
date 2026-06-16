# modelo_mundial

A match-outcome prediction system for the **2026 FIFA World Cup**. It produces
calibrated **1X2** (home / draw / away) and **scoreline** probabilities for
upcoming matches and **tournament-progression** odds for all 48 teams, and it
**self-corrects after every match** through an online learning loop that runs
unattended in the cloud.

Betting odds are treated as a *feature and a benchmark*, never the objective.
The primary metric is **Ranked Probability Score (RPS)**; Brier, log-loss, and
calibration error (ECE) are secondary. Success is defined as beating a
Dixon-Coles baseline and approaching the de-vigged closing market line on RPS
across the tournament's 104 matches.

Full design rationale lives in **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

---

## How it works — a four-layer stack

Predictions come from four independent components combined by an online
ensemble. Each component must beat the Dixon-Coles baseline on walk-forward
backtest before it ships.

| Layer | Module | What it does |
|---|---|---|
| **A — Bayesian backbone** | `models/bayes.py` | `DynamicHierarchicalPoisson`: bivariate Poisson with Dixon-Coles low-score correction. Per-team attack/defense strengths evolve as Gaussian random walks over 6-month periods and are partially pooled by confederation (NumPyro SVI). Prior scales (`SIGMA_RW_PRIOR`, `FRIENDLY_WEIGHT`, home-advantage terms) are module-level constants tuned by a 5-tournament walk-forward sweep. The **only** component that emits a full scoreline distribution (`score_matrix()`), so expected goals and the headline predicted score come from here. |
| **B — Contextual GBM** | `models/gbm.py` | `GbmModel`: a LightGBM 1X2 classifier over contextual features (rest, altitude, etc.) plus Layer A's posterior means. Captures interactions the parametric model can't. |
| **C — Market** | `models/market.py` | Shin-de-vigged closing odds from The Odds API. Degrades gracefully to absent when the key or feed is unavailable. |
| **D — Ensemble** | `models/ensemble.py` | Log-opinion pool (`pool_predict` / `pool_predict_partial`). Weights start from a **market-led prior** (`market 0.55`, `bayes 0.35`, `dc`/`gbm` 0.05 each) — see below — and are updated **in-tournament by Hedge (multiplicative weights)** via `hedge_update()`. |
| — *gate* | `models/baseline.py` | `DixonColes`: the bar every layer must clear on backtest RPS. |
| — *simulator* | `models/simulate.py` | `TournamentSimulator`: 100k Monte Carlo bracket runs that **sample team strengths from the Bayesian posterior** (propagating uncertainty rather than simulating from point estimates). Handles the 48-team group stage, best-third-place ranking, knockouts, and shootouts. |

### Where the starting weights come from

The ensemble's initial weights are **fit out-of-sample**, not hand-picked. `make
backtest` runs a **leave-one-tournament-out (LOTO)** evaluation over the WC
2014/2018/2022 closing-odds subset: fit the pool on two tournaments, score the
held-out one, average the fitted weights across folds. That puts the
`bayes:market` split near `0.31:0.69` and improves held-out RPS over the
market-free pool — the honest test the old `0.35` market prior never had.

The live prior **shrinks off that `0.69` point estimate to `market 0.55`**:
there are only three folds, and the live feed (The Odds API median) is noisier
than the backtested closing line. `dc` and `gbm` keep small live slots (`0.05`
each) and must earn weight through the in-tournament Hedge updates.

---

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"     # Windows: .venv\Scripts\pip install -e ".[dev]"
```

Requires Python ≥ 3.11. Core dependencies: `jax[cpu]` + `numpyro` (Bayesian
layer), `lightgbm`, `polars` + `duckdb`, `scikit-learn`, `scipy`, `pyarrow`,
`requests`.

### Environment variables (`.env`, never committed)

| Variable | Used for | Required? |
|---|---|---|
| `ODDS_API_KEY` | Market layer — closing odds from The Odds API | Optional (pipeline degrades without it) |
| `TELEGRAM_BOT_TOKEN` | Morning digest delivery | Only for the digest |
| `TELEGRAM_CHAT_ID` | Morning digest delivery | Only for the digest |

`update_cycle.py` and `cloud_digest.py` call `load_env()` to read `.env`.

---

## Usage

Run via the Makefile (`PY := .venv/bin/python`):

```bash
make data       # pull latest international results + WC 2026 fixtures
make backtest   # walk-forward eval of all layers + ensemble on 5 past tournaments
                # (also validates the market layer out-of-sample via leave-one-
                #  tournament-out on WC 14/18/22 closing odds)
make predict    # fit on everything played, predict upcoming WC matches
make simulate   # 100k Monte Carlo runs of the full 48-team 2026 bracket
make update     # full online cycle (see below)
make test       # pytest -q
```

Run a single test file directly:

```bash
.venv/bin/pytest tests/test_baseline.py -q
```

### Development

```bash
make lint        # ruff check (F/I/UP/B)
make format      # ruff format
make typecheck   # mypy (advisory)
make cov         # pytest with coverage report
```

CI (`.github/workflows/ci.yml`) runs lint + the test suite on every push and
pull request; mypy runs as a non-blocking step. Pre-commit hooks
(`.pre-commit-config.yaml`) apply ruff lint+format on commit.

---

## The online learning loop

`make update` (or `python -m mundial.online.update_cycle`) runs the post-match
cycle. After each match window it:

1. **Pulls fresh results** from `martj42/international_results`.
2. **Scores previously *logged* predictions** against newly played matches and
   applies the **Hedge** weight update to the pool. Only forecasts logged
   *before kickoff* are ever scored — matches played before a forecast was
   logged are skipped, never back-filled.
3. **Refits the component models** on the updated data and fetches fresh odds.
4. **Logs pooled predictions** for upcoming matches (immutably — the first
   forecast logged per match is frozen; the next cycle scores it).
5. **Re-runs** the 100k tournament simulation.
6. **Writes a cycle report** (`reports/cycle_<date>.md`): newly scored matches,
   weight trajectory, upcoming forecasts, and a cumulative model-vs-market
   scoreboard with a **CUSUM drift alarm** (threshold 3.0).

State lives in `artifacts/online_state.json` (weights, score history, CUSUM);
the honest, append-only forecast record in `artifacts/predictions_log.parquet`.

> **Why this matters for predictions.** Because step 3 refits on fresh results,
> a new result propagates immediately — e.g. if Ecuador loses, every team that
> played Ecuador has its strength estimate revised on the next run. Step 2,
> separately, shifts *ensemble weight* toward whichever layer has actually been
> calling 2026 right.

---

## Cloud automation (runs itself)

A GitHub Actions workflow (`.github/workflows/digest.yml`) runs the **full
learning cycle in the cloud each morning**, then sends a Telegram digest. Each
run:

1. **Restores** prior learning state from the dedicated **`model-state`** branch.
2. Runs the update cycle (`CYCLE_PUSH=0`, so it defers the git push).
3. **Persists** updated state back to `model-state` — an orphan branch that
   holds only `artifacts/` + `reports/`, keeping `main` free of daily churn.
4. Sends the morning digest (`scripts/cloud_digest.py`), which reads the
   just-updated weights and pushes a Telegram message with upcoming-match
   probabilities, predicted scorelines/xG, yesterday's results, and the title
   race. It sends a **failure notice** rather than failing silently.

### Reliable triggering

GitHub's `schedule:` cron is kept only as a **backup** — it lags under load and,
critically, auto-disables after 60 days of repo inactivity. The reliable
trigger is a **free external cron** (e.g. [cron-job.org](https://cron-job.org))
that POSTs to the `workflow_dispatch` API:

```
POST https://api.github.com/repos/MEPO29/modelo-mundial/actions/workflows/digest.yml/dispatches
Headers:
  Authorization: Bearer <FINE_GRAINED_PAT>      # repo-scoped, Actions: read/write
  Accept: application/vnd.github+json
  X-GitHub-Api-Version: 2022-11-28
Body: {"ref":"main"}
```

A `204` response means it fired. To test the whole chain manually:
**Actions → morning-digest → Run workflow**, or `gh workflow run digest.yml`.

**Confirm it actually ran:** check that the **`model-state`** branch has a fresh
`cycle: online state <today>` commit. If that branch is missing or stale while
matches are being played, the loop has stalled — the cycle report and the
Telegram digest now print a ⚠️ freshness warning in that case (played matches
that were pre-logged but never scored, or results older than ~2 days).

Required GitHub repo **secrets**: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`,
`ODDS_API_KEY`.

---

## Project layout

```
src/mundial/
  ingest/        results.py (martj42 pull → immutable data/raw/<date>/),
                 confederations.py (team → confederation for pooling)
  features/      build.py (contextual feature construction)
  models/        baseline.py, bayes.py, gbm.py, market.py, ensemble.py, simulate.py
  eval/          metrics.py (rps/log_loss/brier/ece), backtest.py (walk-forward)
  online/        update_cycle.py (the post-match learning loop)
scripts/         predict_next.py (make predict), cloud_digest.py (morning digest)
notebooks/       EDA + improvement studies (01_eda_diagnostics, 02_eda_diagnostics_v2,
                 03_improvement_experiments, 04_backbone_sweep) on the ds-agent template
data/reference/  bracket_2026.json, groups_2026.json, venues_2026.csv, city_altitudes.csv
artifacts/       online_state.json, pool_weights.json, predictions_log.parquet;
                 plots/ + reports/ (committed backtest sweeps and the model-improvement card)
reports/         per-cycle markdown reports + simulation CSVs
docs/            ARCHITECTURE.md (full blueprint and roadmap)
tests/           pytest suite
```

Immutable raw pulls under `data/raw/`, `data/interim/`, `data/features/` are
gitignored.

---

## Key invariants

- **No leakage.** Every feature is timestamped strictly before kickoff;
  backtest uses `as_of = tournament_start_date`. Splits are always temporal,
  never random K-fold.
- **Honest scoring.** Only pre-kickoff logged predictions are scored against
  results. `append_log()` is anti-joined so an earlier forecast is never
  overwritten — the Hedge updates and the model-vs-market scoreboard are honest
  by construction.
- **Degradation tolerance.** An odds-fetch failure (or any missing optional
  feature) must not block a run — components fall back to their absence with
  rescaled ensemble weights.
- **Backtest gate.** Each layer must beat `DixonColes` walk-forward RPS on the
  five reference tournaments (WC 2014/2018/2022, Euro 2024, Copa 2024) before
  going live.

---

## Data sources

- **Results & fixtures:** [`martj42/international_results`](https://github.com/martj42/international_results)
  — `load_results()` returns played matches, `load_fixtures()` returns upcoming
  rows (null scores).
- **Live odds:** [The Odds API](https://the-odds-api.com/) — Shin de-vigged in
  `models/market.py`.
- **Historical odds & xG:** `data/reference/WorldCup2026-football-data-co-uk.xlsx`
  (football-data.co.uk) — closing bookmaker odds for WC 2014/2018/2022 and xG for
  the 2026 qualifiers, parsed by `ingest/footy_odds.py`. This is what lets
  `make backtest` score the **market layer** against real closing odds.
- **Reference:** 48-team bracket topology, official group assignments, venue
  table, and altitude data in `data/reference/`.
