"""Forecast quality metrics for ordered 1X2 outcomes.

Outcome encoding everywhere in this project: 0 = home win, 1 = draw, 2 = away win
(ordered for RPS as home > draw > away).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def rps(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean Ranked Probability Score. probs: (n, 3), outcomes: (n,) in {0,1,2}."""
    probs = np.asarray(probs, dtype=float)
    onehot = np.eye(3)[np.asarray(outcomes, dtype=int)]
    cum_diff = np.cumsum(probs - onehot, axis=1)[:, :2]
    return float(np.mean(np.sum(cum_diff**2, axis=1)) / 2)


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    picked = probs[np.arange(len(outcomes)), np.asarray(outcomes, dtype=int)]
    return float(-np.mean(np.log(np.clip(picked, EPS, 1.0))))


def brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Multiclass Brier score (sum of squared errors over the 3 classes)."""
    probs = np.asarray(probs, dtype=float)
    onehot = np.eye(3)[np.asarray(outcomes, dtype=int)]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def ece(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error on the max-probability class."""
    probs = np.asarray(probs, dtype=float)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == np.asarray(outcomes, dtype=int)).astype(float)
    bins = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    total = len(conf)
    err = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            err += mask.sum() / total * abs(correct[mask].mean() - conf[mask].mean())
    return float(err)


# --- Scoreline metrics -------------------------------------------------------
# A scoreline grid is grid[h, a] = P(home scores h, away scores a), shape
# (G, G) with G = MAX_GOALS + 1, rows/cols normalized to sum 1. Observed goals
# are clipped to the grid's truncation (G-1), mirroring how the goal models
# clip training scores. These score the FULL scoreline distribution that the
# 1X2 metrics above collapse away.


def _as_grids(grids: np.ndarray) -> np.ndarray:
    """Coerce a single (G,G) grid or a batch to (n, G, G)."""
    grids = np.asarray(grids, dtype=float)
    if grids.ndim == 2:
        grids = grids[None, :, :]
    return grids


def score_log_loss(grids: np.ndarray, hg: np.ndarray, ag: np.ndarray) -> float:
    """Mean -log P(observed exact scoreline). The primary scoreline metric
    (a proper scoring rule, directly comparable to log_loss on 1X2)."""
    grids = _as_grids(grids)
    g = grids.shape[1] - 1
    h = np.clip(np.asarray(hg, dtype=int), 0, g)
    a = np.clip(np.asarray(ag, dtype=int), 0, g)
    picked = grids[np.arange(len(h)), h, a]
    return float(-np.mean(np.log(np.clip(picked, EPS, 1.0))))


def _margin_dist(grids: np.ndarray) -> np.ndarray:
    """Collapse each (G,G) grid to its goal-margin distribution P(d=h-a),
    over ordered bins d = -(G-1)..(G-1). Returns (n, 2G-1)."""
    g = grids.shape[1] - 1
    bins = 2 * g + 1
    out = np.zeros((grids.shape[0], bins))
    for d in range(-g, g + 1):
        # trace offset -d sums grid[i, i+(-d)] = cells where h-a = d
        out[:, d + g] = np.trace(grids, offset=-d, axis1=1, axis2=2)
    return out


def score_rps(grids: np.ndarray, hg: np.ndarray, ag: np.ndarray) -> float:
    """Ranked Probability Score over the ordered goal-MARGIN distribution.

    The natural ordinal axis for scorelines: predicting 1-0 when 2-0 happens
    (margin off by one) is penalized less than predicting 0-1 (wrong side).
    Generalizes the 1X2 ``rps`` from 3 ordered outcomes to 2G-1 margin bins.
    """
    grids = _as_grids(grids)
    g = grids.shape[1] - 1
    margin = _margin_dist(grids)                       # (n, K)
    k = margin.shape[1]
    obs = np.clip(np.asarray(hg, int) - np.asarray(ag, int), -g, g) + g
    onehot = np.eye(k)[obs]
    cum_diff = np.cumsum(margin - onehot, axis=1)[:, : k - 1]
    return float(np.mean(np.sum(cum_diff**2, axis=1)) / (k - 1))


def correct_score_hit_rate(grids: np.ndarray, hg: np.ndarray, ag: np.ndarray) -> float:
    """Fraction of matches where the grid's modal cell is the exact score.
    A headline KPI, NOT a proper scoring rule — report-only, gates nothing."""
    grids = _as_grids(grids)
    g = grids.shape[1] - 1
    n = grids.shape[0]
    flat_argmax = grids.reshape(n, -1).argmax(axis=1)
    mh, ma = np.unravel_index(flat_argmax, (g + 1, g + 1))
    h = np.clip(np.asarray(hg, int), 0, g)
    a = np.clip(np.asarray(ag, int), 0, g)
    return float(np.mean((mh == h) & (ma == a)))


def draw_calibration(
    grids: np.ndarray, hg: np.ndarray, ag: np.ndarray, n_bins: int = 10
) -> float:
    """Expected calibration error of the model's DRAW mass P(h=a)=trace(grid)
    against the empirical draw rate. The targeted check that the bivariate
    backbone actually fixed draw discrimination (independent-Poisson under-
    predicts draws)."""
    grids = _as_grids(grids)
    p_draw = np.trace(grids, axis1=1, axis2=2)
    is_draw = (np.asarray(hg, int) == np.asarray(ag, int)).astype(float)
    bins = np.clip((p_draw * n_bins).astype(int), 0, n_bins - 1)
    total = len(p_draw)
    err = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            err += mask.sum() / total * abs(is_draw[mask].mean() - p_draw[mask].mean())
    return float(err)


def paired_bootstrap(
    delta_per_match: np.ndarray, n: int = 10000, ci: float = 0.90, seed: int = 0
) -> tuple[float, float, float]:
    """Bootstrap CI on the mean of per-match metric deltas (a - b).

    Returns (mean_delta, lo, hi). The gate ships a change only when the CI
    excludes 0 in the improving direction — the guard against shipping a
    within-noise change. Resamples matches with replacement, recomputing the
    mean each draw, then takes the central ``ci`` percentile interval.
    """
    d = np.asarray(delta_per_match, dtype=float)
    if d.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n, d.size))
    means = d[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * (1 - ci) / 2))
    hi = float(np.percentile(means, 100 * (1 + ci) / 2))
    return float(d.mean()), lo, hi
