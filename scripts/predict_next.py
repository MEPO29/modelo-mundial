"""Fit the baseline on everything played so far and predict upcoming WC fixtures."""

from __future__ import annotations

import datetime as dt
import sys

import polars as pl

from mundial.ingest.results import load_fixtures, load_results
from mundial.models.baseline import DixonColes


def main(days_ahead: int = 4) -> None:
    today = dt.date.today()
    results = load_results().filter(pl.col("date") >= dt.date(1990, 1, 1))
    model = DixonColes().fit(results, as_of=today + dt.timedelta(days=1))

    fixtures = load_fixtures(tournament="FIFA World Cup").filter(
        (pl.col("date") >= today) & (pl.col("date") <= today + dt.timedelta(days=days_ahead))
    )

    print(f"Dixon-Coles baseline | fit through {today} | home_adv={model.home_adv:.3f} "
          f"rho={model.rho:.3f}\n")
    print(f"{'date':<11} {'home':<22} {'away':<22} {'P(H)':>6} {'P(D)':>6} {'P(A)':>6}")
    for date, home, away, neu in fixtures.select(
        "date", "home_team", "away_team", "neutral"
    ).iter_rows():
        p = model.predict_1x2(home, away, neutral=neu)
        print(f"{str(date):<11} {home:<22} {away:<22} {p[0]:>6.1%} {p[1]:>6.1%} {p[2]:>6.1%}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
