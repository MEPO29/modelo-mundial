"""Historical international match results (martj42 dataset).

Raw pulls are stored immutably under data/raw/martj42/<date>/results.csv.
The dataset also carries upcoming fixtures as rows with NA scores, which is
how we obtain the 2026 World Cup group-stage schedule.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import requests

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "martj42"


def download_results(pull_date: dt.date | None = None) -> Path:
    """Pull the latest results.csv into an immutable dated directory."""
    pull_date = pull_date or dt.date.today()
    dest = RAW_DIR / pull_date.isoformat() / "results.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(RESULTS_URL, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def latest_raw_path() -> Path:
    pulls = sorted(RAW_DIR.glob("*/results.csv"))
    if not pulls:
        raise FileNotFoundError(
            f"No raw pulls under {RAW_DIR}; run download_results() first"
        )
    return pulls[-1]


def load_results(path: Path | None = None) -> pl.DataFrame:
    """Played matches only (rows with scores), typed and sorted by date."""
    df = _load(path)
    return df.filter(pl.col("home_score").is_not_null()).sort("date")


def load_fixtures(path: Path | None = None, tournament: str | None = None) -> pl.DataFrame:
    """Scheduled matches (NA scores). Optionally filter by tournament name."""
    df = _load(path).filter(pl.col("home_score").is_null()).sort("date")
    if tournament:
        df = df.filter(pl.col("tournament") == tournament)
    return df


def _load(path: Path | None) -> pl.DataFrame:
    path = path or latest_raw_path()
    return pl.read_csv(
        path,
        null_values=["NA"],
        schema_overrides={
            "date": pl.Date,
            "home_score": pl.Int64,
            "away_score": pl.Int64,
            "neutral": pl.Boolean,
        },
    )


if __name__ == "__main__":
    p = download_results()
    played = load_results(p)
    fixtures = load_fixtures(p, tournament="FIFA World Cup")
    print(f"pulled {p}")
    print(f"played matches: {played.height}, latest: {played['date'].max()}")
    print(f"upcoming WC fixtures: {fixtures.height}")
