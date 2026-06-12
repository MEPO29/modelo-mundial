"""Walk-forward backtest of the baseline on past tournaments.

Fits strictly on matches before each tournament's start date (no leakage),
then scores the tournament's matches. Compares against a uniform forecast
and a constant historical-frequency forecast.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from mundial.eval.metrics import brier, ece, log_loss, rps
from mundial.ingest.results import load_results
from mundial.models.baseline import DixonColes

TOURNAMENTS = {
    "WC 2014": ("FIFA World Cup", dt.date(2014, 6, 12), dt.date(2014, 7, 13)),
    "WC 2018": ("FIFA World Cup", dt.date(2018, 6, 14), dt.date(2018, 7, 15)),
    "WC 2022": ("FIFA World Cup", dt.date(2022, 11, 20), dt.date(2022, 12, 18)),
    "Euro 2024": ("UEFA Euro", dt.date(2024, 6, 14), dt.date(2024, 7, 14)),
    "Copa 2024": ("Copa América", dt.date(2024, 6, 20), dt.date(2024, 7, 14)),
}


def outcome_labels(df: pl.DataFrame) -> np.ndarray:
    """0 = home win, 1 = draw, 2 = away win."""
    hs, as_ = df["home_score"].to_numpy(), df["away_score"].to_numpy()
    return np.where(hs > as_, 0, np.where(hs == as_, 1, 2))


def backtest_tournament(
    results: pl.DataFrame, tournament: str, start: dt.date, end: dt.date
) -> dict[str, float] | None:
    test = results.filter(
        (pl.col("tournament") == tournament)
        & (pl.col("date") >= start)
        & (pl.col("date") <= end)
    )
    if test.height == 0:
        return None

    model = DixonColes().fit(results, as_of=start)
    probs = np.array(
        [
            model.predict_1x2(h, a, neutral=neu)
            for h, a, neu in test.select("home_team", "away_team", "neutral").iter_rows()
        ]
    )
    y = outcome_labels(test)

    uniform = np.full_like(probs, 1 / 3)
    hist = results.filter(pl.col("date") < start)
    freq = np.bincount(outcome_labels(hist), minlength=3) / hist.height
    freq_probs = np.tile(freq, (len(y), 1))

    return {
        "n": len(y),
        "rps": rps(probs, y),
        "rps_freq": rps(freq_probs, y),
        "rps_uniform": rps(uniform, y),
        "log_loss": log_loss(probs, y),
        "brier": brier(probs, y),
        "ece": ece(probs, y),
        "home_adv": model.home_adv,
        "rho": model.rho,
    }


def main() -> None:
    results = load_results().filter(pl.col("date") >= dt.date(1990, 1, 1))
    print(f"{'tournament':<11} {'n':>3}  {'RPS':>6}  {'freq':>6}  {'unif':>6}  "
          f"{'logloss':>7}  {'brier':>6}  {'ECE':>5}")
    for name, (tourn, start, end) in TOURNAMENTS.items():
        m = backtest_tournament(results, tourn, start, end)
        if m is None:
            print(f"{name:<11} no matches found")
            continue
        print(
            f"{name:<11} {m['n']:>3}  {m['rps']:>6.4f}  {m['rps_freq']:>6.4f}  "
            f"{m['rps_uniform']:>6.4f}  {m['log_loss']:>7.4f}  {m['brier']:>6.4f}  "
            f"{m['ece']:>5.3f}"
        )


if __name__ == "__main__":
    main()
