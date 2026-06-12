# modelo_mundial

World Cup 2026 match-outcome prediction system. Calibrated 1X2 + scoreline
probabilities and tournament simulation, updated after every match of the
tournament. Full design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Usage

```bash
make data       # pull latest international results (includes WC 2026 fixtures)
make backtest   # walk-forward baseline eval on WC 2014/2018/2022, Euro 2024, Copa 2024
make predict    # fit on everything played and predict upcoming WC matches
make test       # pytest
```

## Layout

- `src/mundial/ingest/` — data acquisition (immutable raw pulls under `data/raw/`)
- `src/mundial/models/` — `baseline.py` (Dixon-Coles), then Bayesian backbone, GBM, ensemble
- `src/mundial/eval/` — metrics (RPS/log-loss/Brier/ECE) and walk-forward backtests
- `data/reference/venues_2026.csv` — hand-curated 16-venue table (altitude, roof, surface, tz)
- `docs/ARCHITECTURE.md` — the full system blueprint and sprint roadmap
