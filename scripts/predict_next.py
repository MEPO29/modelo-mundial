"""Fit a model on everything played so far and predict upcoming WC fixtures.

Usage: predict_next.py [days_ahead] [baseline|bayes]
"""

from __future__ import annotations

import datetime as dt
import sys

import polars as pl

from mundial.ingest.results import load_fixtures, load_results
from mundial.models.baseline import DixonColes
from mundial.models.bayes import DynamicHierarchicalPoisson

MODELS = {"baseline": DixonColes, "bayes": DynamicHierarchicalPoisson}


def main(days_ahead: int = 4, model_name: str = "bayes") -> None:
    today = dt.date.today()
    results = load_results().filter(pl.col("date") >= dt.date(1990, 1, 1))
    model = MODELS[model_name]().fit(results, as_of=today + dt.timedelta(days=1))

    fixtures = load_fixtures(tournament="FIFA World Cup").filter(
        (pl.col("date") >= today) & (pl.col("date") <= today + dt.timedelta(days=days_ahead))
    )

    rho = getattr(model, "rho", None)  # only the Dixon-Coles baseline has rho
    tag = f" | rho={rho:.3f}" if rho is not None else ""
    print(f"model={model_name} | fit through {today}{tag}\n")
    print(f"{'date':<11} {'home':<22} {'away':<22} {'P(H)':>6} {'P(D)':>6} {'P(A)':>6}")
    for date, home, away, neu in fixtures.select(
        "date", "home_team", "away_team", "neutral"
    ).iter_rows():
        p = model.predict_1x2(home, away, neutral=neu)
        print(f"{str(date):<11} {home:<22} {away:<22} {p[0]:>6.1%} {p[1]:>6.1%} {p[2]:>6.1%}")


if __name__ == "__main__":
    main(
        int(sys.argv[1]) if len(sys.argv) > 1 else 4,
        sys.argv[2] if len(sys.argv) > 2 else "bayes",
    )
