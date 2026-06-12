"""Walk-forward backtest on past tournaments.

Fits strictly on matches before each tournament's start date (no leakage),
then scores the tournament's matches. The Dixon-Coles baseline is the gate:
no layer ships unless it beats the baseline's RPS on average across
tournaments. Naive references (outcome-frequency and uniform forecasts)
are reported for scale.
"""

from __future__ import annotations

import datetime as dt
from typing import Callable

import numpy as np
import polars as pl

from mundial.eval.metrics import brier, ece, log_loss, rps
from mundial.ingest.results import load_results
from mundial.models.baseline import DixonColes
from mundial.models.bayes import DynamicHierarchicalPoisson

TOURNAMENTS = {
    "WC 2014": ("FIFA World Cup", dt.date(2014, 6, 12), dt.date(2014, 7, 13)),
    "WC 2018": ("FIFA World Cup", dt.date(2018, 6, 14), dt.date(2018, 7, 15)),
    "WC 2022": ("FIFA World Cup", dt.date(2022, 11, 20), dt.date(2022, 12, 18)),
    "Euro 2024": ("UEFA Euro", dt.date(2024, 6, 14), dt.date(2024, 7, 14)),
    "Copa 2024": ("Copa América", dt.date(2024, 6, 20), dt.date(2024, 7, 14)),
}

MODELS: dict[str, Callable] = {
    "baseline": DixonColes,
    "bayes": DynamicHierarchicalPoisson,
}


def outcome_labels(df: pl.DataFrame) -> np.ndarray:
    """0 = home win, 1 = draw, 2 = away win."""
    hs, as_ = df["home_score"].to_numpy(), df["away_score"].to_numpy()
    return np.where(hs > as_, 0, np.where(hs == as_, 1, 2))


def tournament_matches(
    results: pl.DataFrame, tournament: str, start: dt.date, end: dt.date
) -> pl.DataFrame:
    return results.filter(
        (pl.col("tournament") == tournament)
        & (pl.col("date") >= start)
        & (pl.col("date") <= end)
    )


def score_model(model, test: pl.DataFrame) -> dict[str, float]:
    probs = np.array(
        [
            model.predict_1x2(h, a, neutral=neu)
            for h, a, neu in test.select("home_team", "away_team", "neutral").iter_rows()
        ]
    )
    y = outcome_labels(test)
    return {
        "n": len(y),
        "rps": rps(probs, y),
        "log_loss": log_loss(probs, y),
        "brier": brier(probs, y),
        "ece": ece(probs, y),
    }


def naive_references(results: pl.DataFrame, test: pl.DataFrame, start: dt.date) -> dict:
    y = outcome_labels(test)
    hist = results.filter(pl.col("date") < start)
    freq = np.bincount(outcome_labels(hist), minlength=3) / hist.height
    return {
        "rps_freq": rps(np.tile(freq, (len(y), 1)), y),
        "rps_uniform": rps(np.full((len(y), 3), 1 / 3), y),
    }


def main() -> None:
    results = load_results().filter(pl.col("date") >= dt.date(1990, 1, 1))
    rows = []
    for name, (tourn, start, end) in TOURNAMENTS.items():
        test = tournament_matches(results, tourn, start, end)
        if test.height == 0:
            continue
        ref = naive_references(results, test, start)
        for model_name, factory in MODELS.items():
            model = factory().fit(results, as_of=start)
            m = score_model(model, test)
            rows.append({"tournament": name, "model": model_name, **m, **ref})

    print(f"{'tournament':<11} {'model':<9} {'n':>3}  {'RPS':>6}  {'logloss':>7}  "
          f"{'brier':>6}  {'ECE':>5}  {'freq':>6}  {'unif':>6}")
    for r in rows:
        print(
            f"{r['tournament']:<11} {r['model']:<9} {r['n']:>3}  {r['rps']:>6.4f}  "
            f"{r['log_loss']:>7.4f}  {r['brier']:>6.4f}  {r['ece']:>5.3f}  "
            f"{r['rps_freq']:>6.4f}  {r['rps_uniform']:>6.4f}"
        )

    for model_name in MODELS:
        sub = [r for r in rows if r["model"] == model_name]
        total_n = sum(r["n"] for r in sub)
        w_rps = sum(r["rps"] * r["n"] for r in sub) / total_n
        w_ll = sum(r["log_loss"] * r["n"] for r in sub) / total_n
        print(f"\n{model_name}: match-weighted RPS {w_rps:.4f}, log-loss {w_ll:.4f} "
              f"over {total_n} matches")


if __name__ == "__main__":
    main()
