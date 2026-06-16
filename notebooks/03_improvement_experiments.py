"""S3/S4 improvement experiments on the cached backtest prediction matrix.

Every experiment is evaluated leave-one-tournament-out (LOTO) on the 5 reference
tournaments, exactly mirroring eval/backtest.py's honest protocol. The expensive
component fits (Bayesian SVI x5) are NOT repeated here: we operate on the cached
per-match component probabilities in data/interim/backtest_predictions.parquet,
so we can sweep pool/calibration strategies cheaply.

Reported per strategy: match-weighted RPS, log-loss, Brier, ECE, and a paired
bootstrap 90% CI on the RPS delta vs the production baseline (log-opinion pool
over dc/bayes/gbm). A change only "wins" if its RPS CI excludes 0 in its favour.

Nothing here is leakage-prone: all calibration parameters are fit on the
training folds and applied to the held-out tournament only.

Run:  .venv/bin/python notebooks/03_improvement_experiments.py
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.optimize import minimize, minimize_scalar

from mundial.eval.metrics import brier, ece, log_loss, rps
from mundial.models import ensemble as ens

RNG = np.random.default_rng(0)
PRED_PATH = "data/interim/backtest_predictions.parquet"
EPS = 1e-9
ALL_COMPS = ["dc", "bayes", "gbm"]


def load() -> pl.DataFrame:
    return pl.read_parquet(PRED_PATH)


def blocks_of(df: pl.DataFrame, comps: list[str]) -> list[np.ndarray]:
    return [df.select(f"{c}_h", f"{c}_d", f"{c}_a").to_numpy() for c in comps]


# ---------- calibration primitives (all fit on train, applied to test) ----------

def _temperature(p: np.ndarray, t: float) -> np.ndarray:
    """Raise to power 1/t in log space then renormalise. t>1 softens, t<1 sharpens."""
    z = np.log(np.clip(p, EPS, 1.0)) / t
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(p: np.ndarray, y: np.ndarray, objective: str = "logloss") -> float:
    def obj(t):
        q = _temperature(p, t)
        return log_loss(q, y) if objective == "logloss" else rps(q, y)

    return float(minimize_scalar(obj, bounds=(0.3, 3.0), method="bounded").x)


def _dirichlet_apply(p: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vector-scaling calibration: softmax(w * log p + b). 3 scales + 3 biases."""
    z = w * np.log(np.clip(p, EPS, 1.0)) + b
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_dirichlet(p: np.ndarray, y: np.ndarray, l2: float = 0.1):
    def nll(theta):
        w, b = theta[:3], theta[3:]
        q = _dirichlet_apply(p, w, b)
        idx = np.arange(len(y))
        return -np.mean(np.log(np.clip(q[idx, y], 1e-12, 1.0))) + l2 * (
            np.sum((w - 1) ** 2) + np.sum(b**2)
        )

    x0 = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    res = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 4000})
    return res.x[:3], res.x[3:]


# ---------- LOTO drivers ----------

def loto_pool(df: pl.DataFrame, comps: list[str]) -> np.ndarray:
    """Production-style log-opinion pool, LOTO weights. Returns (N,3)."""
    return ens.loto_eval(df, comps)


def loto_calibrated_pool(
    df: pl.DataFrame, comps: list[str], method: str, objective: str = "logloss"
) -> np.ndarray:
    """Pool (LOTO weights) then apply a calibrator fit on the same train folds."""
    blocks = blocks_of(df, comps)
    y = df["outcome"].to_numpy()
    tour = df["tournament_name"].to_numpy()
    out = np.zeros((df.height, 3))
    for t in np.unique(tour):
        held = tour == t
        w = ens.fit_pool_weights([b[~held] for b in blocks], y[~held])
        p_tr = ens.pool_predict(w, [b[~held] for b in blocks])
        p_te = ens.pool_predict(w, [b[held] for b in blocks])
        if method == "none":
            q_te = p_te
        elif method == "temp":
            t_hat = fit_temperature(p_tr, y[~held], objective)
            q_te = _temperature(p_te, t_hat)
        elif method == "dirichlet":
            wv, bv = fit_dirichlet(p_tr, y[~held])
            q_te = _dirichlet_apply(p_te, wv, bv)
        else:
            raise ValueError(method)
        out[held] = q_te
    return out


def loto_pool_with_market(df: pl.DataFrame, comps: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """LOTO pool incl. market on the subset of rows where market exists.

    Folds are the tournaments that have market odds. Returns (probs, mask) where
    mask selects the market-available rows aligned to df.
    """
    mask = df["market_h"].is_not_null().to_numpy()
    sub = df.filter(pl.col("market_h").is_not_null())
    blocks = blocks_of(sub, comps)
    y = sub["outcome"].to_numpy()
    tour = sub["tournament_name"].to_numpy()
    out = np.zeros((sub.height, 3))
    for t in np.unique(tour):
        held = tour == t
        if (~held).sum() == 0:
            continue
        w = ens.fit_pool_weights([b[~held] for b in blocks], y[~held])
        out[held] = ens.pool_predict(w, [b[held] for b in blocks])
    return out, mask


# ---------- evaluation + bootstrap ----------

def metric_block(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {"rps": rps(p, y), "logloss": log_loss(p, y), "brier": brier(p, y), "ece": ece(p, y)}


def paired_rps_ci(p_new: np.ndarray, p_base: np.ndarray, y: np.ndarray, reps: int = 2000):
    """Bootstrap 90% CI on mean per-match RPS delta (new - base). Negative = better."""
    def per_match_rps(p):
        oh = np.eye(3)[y]
        cd = np.cumsum(p - oh, axis=1)[:, :2]
        return np.sum(cd**2, axis=1) / 2

    d = per_match_rps(p_new) - per_match_rps(p_base)
    n = len(d)
    means = np.array([d[RNG.integers(0, n, n)].mean() for _ in range(reps)])
    return float(d.mean()), float(np.percentile(means, 5)), float(np.percentile(means, 95))


def main() -> None:
    df = load()
    y = df["outcome"].to_numpy()
    has_market = "market_h" in df.columns and df["market_h"].is_not_null().any()
    print(f"loaded {df.height} matches across {df['tournament_name'].n_unique()} tournaments; "
          f"market column present: {has_market}\n")

    # --- baseline: production pool over dc/bayes/gbm ---
    base = loto_pool(df, ALL_COMPS)
    bm = metric_block(base, y)
    print(f"{'strategy':<34} {'RPS':>7} {'logloss':>8} {'brier':>7} {'ece':>6}  RPS d vs base [90% CI]")
    print(f"{'BASELINE pool(dc,bayes,gbm)':<34} {bm['rps']:>7.4f} {bm['logloss']:>8.4f} "
          f"{bm['brier']:>7.4f} {bm['ece']:>6.3f}   --")

    experiments = {
        "pool(dc,bayes)": lambda: loto_pool(df, ["dc", "bayes"]),
        "pool(bayes) only": lambda: loto_pool(df, ["bayes"]),
        "pool(dc,bayes,gbm)+temp[ll]": lambda: loto_calibrated_pool(df, ALL_COMPS, "temp", "logloss"),
        "pool(dc,bayes,gbm)+temp[rps]": lambda: loto_calibrated_pool(df, ALL_COMPS, "temp", "rps"),
        "pool(dc,bayes,gbm)+dirichlet": lambda: loto_calibrated_pool(df, ALL_COMPS, "dirichlet"),
        "pool(dc,bayes)+temp[ll]": lambda: loto_calibrated_pool(df, ["dc", "bayes"], "temp", "logloss"),
        "pool(dc,bayes)+dirichlet": lambda: loto_calibrated_pool(df, ["dc", "bayes"], "dirichlet"),
    }
    for name, fn in experiments.items():
        p = fn()
        m = metric_block(p, y)
        d, lo, hi = paired_rps_ci(p, base, y)
        flag = "  <-- wins" if hi < 0 else ("  (worse)" if lo > 0 else "")
        print(f"{name:<34} {m['rps']:>7.4f} {m['logloss']:>8.4f} {m['brier']:>7.4f} "
              f"{m['ece']:>6.3f}   {d:+.4f} [{lo:+.4f},{hi:+.4f}]{flag}")

    # --- market subset experiments ---
    if has_market:
        print("\n--- market subset (WC tournaments with closing odds) ---")
        sub = df.filter(pl.col("market_h").is_not_null())
        ys = sub["outcome"].to_numpy()
        base_sub = loto_pool(sub, ALL_COMPS)
        bms = metric_block(base_sub, ys)
        print(f"n={sub.height}  baseline pool(dc,bayes,gbm) RPS {bms['rps']:.4f} "
              f"logloss {bms['logloss']:.4f}")
        m4, _ = loto_pool_with_market(df, ["dc", "bayes", "gbm", "market"])
        mm = metric_block(m4, ys)
        d, lo, hi = paired_rps_ci(m4, base_sub, ys)
        flag = "  <-- wins" if hi < 0 else ("  (worse)" if lo > 0 else "")
        print(f"4-comp pool(+market)          RPS {mm['rps']:.4f} logloss {mm['logloss']:.4f}  "
              f"RPS d {d:+.4f} [{lo:+.4f},{hi:+.4f}]{flag}")
        # market alone + bayes+market
        for combo in (["bayes", "market"], ["dc", "bayes", "market"]):
            p, _ = loto_pool_with_market(df, combo)
            m = metric_block(p, ys)
            d, lo, hi = paired_rps_ci(p, base_sub, ys)
            flag = "  <-- wins" if hi < 0 else ("  (worse)" if lo > 0 else "")
            print(f"pool({','.join(combo)})".ljust(30)
                  + f"RPS {m['rps']:.4f} logloss {m['logloss']:.4f}  "
                  f"RPS d {d:+.4f} [{lo:+.4f},{hi:+.4f}]{flag}")


if __name__ == "__main__":
    main()
