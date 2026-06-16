"""Targeted Bayesian-backbone hyperparameter sweep on the 5 reference tournaments.

The backbone dominates the pool (pool weights collapse to bayes~1.0), so the only
remaining RPS lever is the backbone itself. Each config changes ONE thing from the
production default and is scored walk-forward (fit strictly pre-tournament, score the
tournament) on all 5 tournaments — the same honest protocol as eval/backtest.py.

We report per-tournament and overall RPS + log-loss, and the overall RPS delta vs
the production default. To guard against tuning to noise we also report how many of
the 5 tournaments each config improves (a config that wins overall by helping one
tournament and hurting four is rejected).

Runtime ~50s per config (10s/fit x 5).  Run:
  .venv/bin/python notebooks/04_backbone_sweep.py
"""

from __future__ import annotations

import datetime as dt
import time

import numpy as np
import polars as pl

from mundial.eval.backtest import TOURNAMENTS, outcome_labels, tournament_matches
from mundial.eval.metrics import log_loss, rps
from mundial.ingest.results import load_results
from mundial.models import bayes as bayes_mod
from mundial.models.bayes import DynamicHierarchicalPoisson

RESULTS = load_results().filter(pl.col("date") >= dt.date(1990, 1, 1))


def eval_config(field_kwargs: dict, module_consts: dict) -> dict:
    """Walk-forward bayes-only eval across the 5 tournaments for one config."""
    # apply module-level prior/constant overrides (restored by caller)
    saved = {k: getattr(bayes_mod, k) for k in module_consts}
    for k, v in module_consts.items():
        setattr(bayes_mod, k, v)
    try:
        per_t, all_p, all_y = {}, [], []
        for name, (tourn, start, end) in TOURNAMENTS.items():
            test = tournament_matches(RESULTS, tourn, start, end)
            if test.height == 0:
                continue
            model = DynamicHierarchicalPoisson(**field_kwargs).fit(RESULTS, as_of=start)
            rows = test.select("home_team", "away_team", "neutral").to_dicts()
            P = np.array([model.predict_1x2(r["home_team"], r["away_team"],
                                            neutral=r["neutral"]) for r in rows])
            y = outcome_labels(test)
            per_t[name] = rps(P, y)
            all_p.append(P)
            all_y.append(y)
        P = np.vstack(all_p)
        y = np.concatenate(all_y)
        return {"overall_rps": rps(P, y), "overall_ll": log_loss(P, y), "per_t": per_t}
    finally:
        for k, v in saved.items():
            setattr(bayes_mod, k, v)


CONFIGS = [
    ("DEFAULT (since2010, rw0.1, P182, fr0.5)", {}, {}),
    # history window
    ("since 2006", {"since": dt.date(2006, 1, 1)}, {}),
    ("since 2014", {"since": dt.date(2014, 1, 1)}, {}),
    # random-walk dynamics (how fast strengths evolve)
    ("sigma_rw 0.15", {}, {"SIGMA_RW_PRIOR": 0.15}),
    ("sigma_rw 0.07", {}, {"SIGMA_RW_PRIOR": 0.07}),
    ("sigma_rw 0.20", {}, {"SIGMA_RW_PRIOR": 0.20}),
    # period granularity
    ("period 365d", {}, {"PERIOD_DAYS": 365}),
    ("period 120d", {}, {"PERIOD_DAYS": 120}),
    # friendly weight
    ("friendly 0.3", {}, {"FRIENDLY_WEIGHT": 0.3}),
    ("friendly 0.7", {}, {"FRIENDLY_WEIGHT": 0.7}),
    # home advantage prior
    ("home_adv loc 0.35", {}, {"HOME_ADV_LOC": 0.35}),
    # convergence
    ("svi 8000", {"svi_steps": 8000}, {}),
]


def main() -> None:
    base = None
    print(f"{'config':<42} {'RPS':>7} {'logloss':>8} {'dRPS':>8} {'better/5':>8}   per-tournament RPS")
    for label, fk, mc in CONFIGS:
        t0 = time.time()
        r = eval_config(fk, mc)
        if base is None:
            base = r
            dstr, win = "  --  ", "  -- "
        else:
            d = r["overall_rps"] - base["overall_rps"]
            wins = sum(r["per_t"][k] < base["per_t"][k] - 1e-5 for k in base["per_t"])
            dstr = f"{d:+.4f}"
            win = f"{wins}/{len(base['per_t'])}"
        pt = " ".join(f"{k.split()[-1]}:{v:.4f}" for k, v in r["per_t"].items())
        print(f"{label:<42} {r['overall_rps']:>7.4f} {r['overall_ll']:>8.4f} "
              f"{dstr:>8} {win:>8}   {pt}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
