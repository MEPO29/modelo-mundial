"""Layer D: log-opinion-pool ensemble over the component models.

p(k) proportional to exp( sum_c w_c * log p_c(k) ), w_c >= 0.

Three weights fit by penalized log-loss — far more stable on ~275 stacking
rows than a full logistic meta-learner (which LOTO-overfit: RPS 0.204 vs
0.196 for the pool). Validated leave-one-tournament-out; never random
K-fold. The fitted weights are the starting point for the in-tournament
Hedge (multiplicative-weights) updates: a component that is useless today
(current GBM weight is ~0) keeps its slot and earns weight only if it
starts scoring during the live tournament.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = PROJECT_ROOT / "artifacts"

EPS = 1e-9
L2_WEIGHT = 0.01


def _log_blocks(prob_blocks: list[np.ndarray]) -> np.ndarray:
    """(C, N, 3) stacked log-probabilities."""
    return np.stack([np.log(np.clip(p, EPS, 1.0)) for p in prob_blocks])


def pool_predict(weights: np.ndarray, prob_blocks: list[np.ndarray]) -> np.ndarray:
    z = np.einsum("c,cnk->nk", np.asarray(weights), _log_blocks(prob_blocks))
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def fit_pool_weights(
    prob_blocks: list[np.ndarray], y: np.ndarray, l2: float = L2_WEIGHT
) -> np.ndarray:
    logp = _log_blocks(prob_blocks)
    n = len(y)
    rows = np.arange(n)

    def nll(theta):
        w = np.exp(theta)
        z = np.einsum("c,cnk->nk", w, logp)
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        return -np.mean(np.log(np.clip(p[rows, y], 1e-12, 1.0))) + l2 * w.sum()

    x0 = np.log(np.full(len(prob_blocks), 1.0 / len(prob_blocks)))
    res = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 2000})
    return np.exp(res.x)


def loto_eval(df: pl.DataFrame, components: list[str], y_col: str = "outcome") -> np.ndarray:
    """Leave-one-tournament-out pool predictions, aligned with df rows."""
    blocks = [df.select(f"{c}_h", f"{c}_d", f"{c}_a").to_numpy() for c in components]
    y = df[y_col].to_numpy()
    tournaments = df["tournament_name"].to_numpy()
    out = np.zeros((df.height, 3))
    for t in np.unique(tournaments):
        held = tournaments == t
        w = fit_pool_weights([b[~held] for b in blocks], y[~held])
        out[held] = pool_predict(w, [b[held] for b in blocks])
    return out


def save_weights(weights: np.ndarray, components: list[str]) -> Path:
    ARTIFACTS.mkdir(exist_ok=True)
    path = ARTIFACTS / "pool_weights.json"
    path.write_text(
        json.dumps(
            {"components": components, "weights": np.asarray(weights).tolist()},
            indent=2,
        )
    )
    return path


def load_weights() -> tuple[np.ndarray, list[str]]:
    blob = json.loads((ARTIFACTS / "pool_weights.json").read_text())
    return np.array(blob["weights"]), blob["components"]
