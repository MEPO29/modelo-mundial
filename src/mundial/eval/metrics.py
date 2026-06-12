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
