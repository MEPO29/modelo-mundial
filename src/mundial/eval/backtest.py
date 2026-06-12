"""Walk-forward backtest on past tournaments.

For every tournament, every component model is fit strictly on matches
before the start date (no leakage) and scores the tournament's matches.
The stacked ensemble is evaluated leave-one-tournament-out, and the final
meta-learner (fit on all tournaments) is persisted to artifacts/.

The Dixon-Coles baseline is the gate: nothing ships unless it beats the
baseline's match-weighted RPS.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from mundial.eval.metrics import brier, ece, log_loss, rps
from mundial.ingest.results import load_results
from mundial.models import ensemble as ens
from mundial.models.baseline import DixonColes
from mundial.models.bayes import DynamicHierarchicalPoisson
from mundial.models.gbm import GbmModel

TOURNAMENTS = {
    "WC 2014": ("FIFA World Cup", dt.date(2014, 6, 12), dt.date(2014, 7, 13)),
    "WC 2018": ("FIFA World Cup", dt.date(2018, 6, 14), dt.date(2018, 7, 15)),
    "WC 2022": ("FIFA World Cup", dt.date(2022, 11, 20), dt.date(2022, 12, 18)),
    "Euro 2024": ("UEFA Euro", dt.date(2024, 6, 14), dt.date(2024, 7, 14)),
    "Copa 2024": ("Copa América", dt.date(2024, 6, 20), dt.date(2024, 7, 14)),
}

COMPONENTS = ["dc", "bayes", "gbm"]


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


def collect_predictions(results: pl.DataFrame) -> pl.DataFrame:
    """Component predictions for every backtest tournament match (walk-forward)."""
    frames = []
    for name, (tourn, start, end) in TOURNAMENTS.items():
        test = tournament_matches(results, tourn, start, end)
        if test.height == 0:
            continue
        models = {
            "dc": DixonColes().fit(results, as_of=start),
            "bayes": DynamicHierarchicalPoisson().fit(results, as_of=start),
            "gbm": GbmModel().fit(results, as_of=start, fixtures=test),
        }
        rows = test.select("date", "home_team", "away_team", "neutral").to_dicts()
        y = outcome_labels(test)
        for i, r in enumerate(rows):
            r["tournament_name"] = name
            r["outcome"] = int(y[i])
            for comp, model in models.items():
                p = model.predict_1x2(r["home_team"], r["away_team"], neutral=r["neutral"])
                r[f"{comp}_h"], r[f"{comp}_d"], r[f"{comp}_a"] = map(float, p)
        frames.append(pl.DataFrame(rows))
    return pl.concat(frames)


def metric_row(probs: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "rps": rps(probs, y),
        "log_loss": log_loss(probs, y),
        "brier": brier(probs, y),
        "ece": ece(probs, y),
    }


def main() -> None:
    results = load_results().filter(pl.col("date") >= dt.date(1990, 1, 1))
    preds = collect_predictions(results)
    preds.write_parquet("data/interim/backtest_predictions.parquet")
    y_all = preds["outcome"].to_numpy()

    ens_probs = ens.loto_eval(preds, COMPONENTS)

    print(f"{'tournament':<11} {'model':<9} {'n':>3}  {'RPS':>6}  {'logloss':>7}  {'ECE':>5}")
    for tname in preds["tournament_name"].unique(maintain_order=True):
        mask = (preds["tournament_name"] == tname).to_numpy()
        y = y_all[mask]
        for comp in COMPONENTS:
            p = preds.filter(pl.col("tournament_name") == tname).select(
                f"{comp}_h", f"{comp}_d", f"{comp}_a"
            ).to_numpy()
            m = metric_row(p, y)
            print(f"{tname:<11} {comp:<9} {len(y):>3}  {m['rps']:>6.4f}  "
                  f"{m['log_loss']:>7.4f}  {m['ece']:>5.3f}")
        m = metric_row(ens_probs[mask], y)
        print(f"{tname:<11} {'ENSEMBLE':<9} {len(y):>3}  {m['rps']:>6.4f}  "
              f"{m['log_loss']:>7.4f}  {m['ece']:>5.3f}")

    print(f"\n{'overall':<11} {'model':<9} {'n':>3}  {'RPS':>6}  {'logloss':>7}  {'ECE':>5}")
    for comp in COMPONENTS:
        p = preds.select(f"{comp}_h", f"{comp}_d", f"{comp}_a").to_numpy()
        m = metric_row(p, y_all)
        print(f"{'':<11} {comp:<9} {len(y_all):>3}  {m['rps']:>6.4f}  "
              f"{m['log_loss']:>7.4f}  {m['ece']:>5.3f}")
    m = metric_row(ens_probs, y_all)
    print(f"{'':<11} {'ENSEMBLE':<9} {len(y_all):>3}  {m['rps']:>6.4f}  "
          f"{m['log_loss']:>7.4f}  {m['ece']:>5.3f}")

    blocks = [preds.select(f"{c}_h", f"{c}_d", f"{c}_a").to_numpy() for c in COMPONENTS]
    weights = ens.fit_pool_weights(blocks, y_all)
    path = ens.save_weights(weights, COMPONENTS)
    print(f"\npool weights {dict(zip(COMPONENTS, weights.round(3)))} saved to {path}")


if __name__ == "__main__":
    main()
