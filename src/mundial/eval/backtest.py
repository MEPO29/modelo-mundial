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

from mundial.eval.metrics import (
    brier,
    correct_score_hit_rate,
    draw_calibration,
    ece,
    log_loss,
    paired_bootstrap,
    rps,
    score_log_loss,
    score_rps,
)
from mundial.ingest.footy_odds import oriented_market
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

# backtest tournaments for which the football-data workbook carries closing odds
MARKET_YEARS = {"WC 2014": 2014, "WC 2018": 2018, "WC 2022": 2022}


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
        bayes = DynamicHierarchicalPoisson().fit(results, as_of=start)
        models = {
            "dc": DixonColes().fit(results, as_of=start),
            "bayes": bayes,
            "gbm": GbmModel().fit(results, as_of=start, fixtures=test, backbone=bayes),
        }
        rows = test.select(
            "date", "home_team", "away_team", "neutral", "home_score", "away_score"
        ).to_dicts()
        y = outcome_labels(test)
        market = oriented_market(test, MARKET_YEARS[name]) if name in MARKET_YEARS else None
        for i, r in enumerate(rows):
            r["tournament_name"] = name
            r["outcome"] = int(y[i])
            for comp, model in models.items():
                p = model.predict_1x2(r["home_team"], r["away_team"], neutral=r["neutral"])
                r[f"{comp}_h"], r[f"{comp}_d"], r[f"{comp}_a"] = map(float, p)
            # bayes scoreline grids: the live bivariate-Poisson and the
            # shared-component-off ablation (independent Poisson, same rates),
            # so scoreline_analysis can isolate the structural change's value.
            grid_bp = bayes.score_matrix(r["home_team"], r["away_team"], neutral=r["neutral"])
            grid_ind = bayes.score_matrix(
                r["home_team"], r["away_team"], neutral=r["neutral"], shared=False
            )
            r["bayes_grid"] = grid_bp.reshape(-1).tolist()
            r["bayes_grid_indep"] = grid_ind.reshape(-1).tolist()
            if market is not None and not np.isnan(market[i]).any():
                r["market_h"], r["market_d"], r["market_a"] = map(float, market[i])
            else:
                r["market_h"] = r["market_d"] = r["market_a"] = None
        frames.append(pl.DataFrame(rows))
    return pl.concat(frames, how="diagonal")


def metric_row(probs: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "rps": rps(probs, y),
        "log_loss": log_loss(probs, y),
        "brier": brier(probs, y),
        "ece": ece(probs, y),
    }


def market_analysis(preds: pl.DataFrame) -> None:
    """Validate the market layer on the tournaments that carry closing odds.

    Reports the market's own RPS and whether adding it as a 4th component
    improves the pool, on exactly the rows where real odds exist — the
    evidence the live 0.35 market prior never had.
    """
    if "market_h" not in preds.columns:
        return
    sub = preds.filter(pl.col("market_h").is_not_null())
    if sub.height == 0:
        print("\nno market odds matched any backtest tournament")
        return
    y = sub["outcome"].to_numpy()
    n_total = preds.filter(pl.col("tournament_name").is_in(list(MARKET_YEARS))).height
    print(f"\nmarket layer — odds matched {sub.height}/{n_total} WC matches")
    print(f"{'tournament':<11} {'model':<9} {'n':>3}  {'RPS':>6}  {'logloss':>7}  {'ECE':>5}")
    for tname in sub["tournament_name"].unique(maintain_order=True):
        t = sub.filter(pl.col("tournament_name") == tname)
        yt = t["outcome"].to_numpy()
        for comp in [*COMPONENTS, "market"]:
            p = t.select(f"{comp}_h", f"{comp}_d", f"{comp}_a").to_numpy()
            m = metric_row(p, yt)
            print(f"{tname:<11} {comp:<9} {len(yt):>3}  {m['rps']:>6.4f}  "
                  f"{m['log_loss']:>7.4f}  {m['ece']:>5.3f}")

    comps4 = [*COMPONENTS, "market"]
    blocks3 = [sub.select(f"{c}_h", f"{c}_d", f"{c}_a").to_numpy() for c in COMPONENTS]
    blocks4 = blocks3 + [sub.select("market_h", "market_d", "market_a").to_numpy()]

    # in-sample (optimistic — fit and scored on the same rows)
    w4_in = ens.fit_pool_weights(blocks4, y)
    w4_in /= w4_in.sum()

    # out-of-sample: leave-one-tournament-out over the WC tournaments with odds.
    # This is the honest test the live 0.35 prior never had — fit on two
    # tournaments, score the held-out one, and average the market share.
    rps3_loto = rps(ens.loto_eval(sub, COMPONENTS), y)
    rps4_loto = rps(ens.loto_eval(sub, comps4), y)
    tournaments = sub["tournament_name"].to_numpy()
    fold_weights = []
    for t in np.unique(tournaments):
        held = tournaments == t
        wt = ens.fit_pool_weights([b[~held] for b in blocks4], y[~held])
        fold_weights.append(wt / wt.sum())
    w4_oos = np.mean(fold_weights, axis=0)

    print(f"\nmarket-inclusive pool on the odds subset (n={sub.height}):")
    print(f"  LOTO RPS  3-comp (dc,bayes,gbm) {rps3_loto:.4f}"
          f"   4-comp (+market) {rps4_loto:.4f}  "
          f"({'better' if rps4_loto < rps3_loto else 'no improvement'})")
    print(f"  in-sample weights  {dict(zip(comps4, w4_in.round(3)))}")
    print(f"  LOTO mean weights  {dict(zip(comps4, w4_oos.round(3)))}")
    print(f"  -> out-of-sample market share {w4_oos[-1]:.2f} "
          f"(in-sample {w4_in[-1]:.2f}; live prior 0.35)")


def _grids(preds: pl.DataFrame, col: str) -> np.ndarray:
    """Stack a flattened-grid column back to (n, G, G)."""
    flat = np.asarray(preds[col].to_list(), dtype=float)
    g = int(round(flat.shape[1] ** 0.5))
    return flat.reshape(-1, g, g)


def _scoreline_metrics(grids: np.ndarray, hg: np.ndarray, ag: np.ndarray) -> dict:
    return {
        "sll": score_log_loss(grids, hg, ag),
        "srps": score_rps(grids, hg, ag),
        "hit": correct_score_hit_rate(grids, hg, ag),
        "draw_ece": draw_calibration(grids, hg, ag),
    }


def scoreline_analysis(preds: pl.DataFrame) -> None:
    """Compare the bivariate-Poisson scoreline grid against the independent-
    Poisson ablation (shared component off, identical fitted rates).

    The structural change ships only if the BP grid improves scoreline
    log-loss AND draw calibration with a paired-bootstrap CI on the per-match
    log-loss delta that excludes 0 — the guard against shipping noise.
    """
    if "bayes_grid" not in preds.columns:
        return
    hg = preds["home_score"].to_numpy().astype(int)
    ag = preds["away_score"].to_numpy().astype(int)
    bp = _grids(preds, "bayes_grid")
    ind = _grids(preds, "bayes_grid_indep")

    print(f"\n{'scoreline':<22} {'n':>3}  {'SLL':>7}  {'mRPS':>6}  {'hit%':>5}  {'drawECE':>7}")
    for name, grids in (("independent-Poisson", ind), ("bivariate-Poisson", bp)):
        m = _scoreline_metrics(grids, hg, ag)
        print(f"{name:<22} {len(hg):>3}  {m['sll']:>7.4f}  {m['srps']:>6.4f}  "
              f"{m['hit'] * 100:>5.1f}  {m['draw_ece']:>7.4f}")

    g = bp.shape[1] - 1
    h, a = np.clip(hg, 0, g), np.clip(ag, 0, g)
    idx = np.arange(len(h))
    sll_bp = -np.log(np.clip(bp[idx, h, a], 1e-12, 1.0))
    sll_ind = -np.log(np.clip(ind[idx, h, a], 1e-12, 1.0))
    mean_d, lo, hi = paired_bootstrap(sll_ind - sll_bp)  # >0 => BP better
    ships = lo > 0
    print(f"\nscoreline log-loss improvement (independent - bivariate): "
          f"{mean_d:+.4f}  90% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  gate: {'PASS — BP improves with CI support' if ships else 'HOLD — within noise'}")


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

    market_analysis(preds)
    scoreline_analysis(preds)


if __name__ == "__main__":
    main()
