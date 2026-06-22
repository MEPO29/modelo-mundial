import numpy as np

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


def _onehot_grid(h, a, g=10):
    """A degenerate scoreline grid placing all mass on cell (h, a)."""
    grid = np.zeros((g + 1, g + 1))
    grid[h, a] = 1.0
    return grid


def test_perfect_forecast_scores_zero():
    probs = np.eye(3)
    y = np.array([0, 1, 2])
    assert rps(probs, y) == 0.0
    assert brier(probs, y) == 0.0
    assert log_loss(probs, y) < 1e-10


def test_rps_respects_ordering():
    # Predicting "away win" when home wins must be worse than predicting "draw":
    # RPS penalizes by distance on the ordered H/D/A scale, Brier does not.
    y = np.array([0])
    near_miss = rps(np.array([[0.0, 1.0, 0.0]]), y)
    far_miss = rps(np.array([[0.0, 0.0, 1.0]]), y)
    assert far_miss > near_miss


def test_uniform_forecast_rps():
    y = np.array([0])
    expected = ((1 / 3 - 1) ** 2 + (2 / 3 - 1) ** 2) / 2
    assert abs(rps(np.full((1, 3), 1 / 3), y) - expected) < 1e-12


def test_ece_bounds():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3), size=200)
    y = rng.integers(0, 3, size=200)
    assert 0.0 <= ece(probs, y) <= 1.0


def test_log_loss_uniform():
    y = np.array([0, 1, 2])
    assert abs(log_loss(np.full((3, 3), 1 / 3), y) - np.log(3)) < 1e-12


# --- Scoreline metrics -------------------------------------------------------


def test_perfect_scoreline_grid_scores_zero():
    grids = np.stack([_onehot_grid(2, 1), _onehot_grid(0, 0)])
    hg, ag = np.array([2, 0]), np.array([1, 0])
    assert score_log_loss(grids, hg, ag) < 1e-9
    assert score_rps(grids, hg, ag) < 1e-12
    assert correct_score_hit_rate(grids, hg, ag) == 1.0


def test_score_rps_respects_margin_ordering():
    # observed 2-0 (margin +2). Predicting 1-0 (margin +1) must beat 0-1 (margin -1).
    hg, ag = np.array([2]), np.array([0])
    near = score_rps(_onehot_grid(1, 0), hg, ag)
    far = score_rps(_onehot_grid(0, 1), hg, ag)
    assert far > near


def test_correct_score_hit_rate_partial():
    grids = np.stack([_onehot_grid(1, 1), _onehot_grid(3, 0)])
    hg, ag = np.array([1, 0]), np.array([1, 0])  # only the first matches
    assert correct_score_hit_rate(grids, hg, ag) == 0.5


def test_draw_calibration_bounds_and_signal():
    rng = np.random.default_rng(0)
    n = 200
    grids = rng.dirichlet(np.ones(121), size=n).reshape(n, 11, 11)
    hg = rng.integers(0, 4, size=n)
    ag = rng.integers(0, 4, size=n)
    dc = draw_calibration(grids, hg, ag)
    assert 0.0 <= dc <= 1.0


def test_score_log_loss_clips_out_of_grid_score():
    # a 12-x scoreline maps to the truncated corner cell, never crashes
    grid = _onehot_grid(10, 0)
    assert score_log_loss(grid, np.array([12]), np.array([0])) < 1e-9


def test_paired_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(1)
    deltas = rng.normal(0.05, 0.02, size=500)  # clearly positive
    mean, lo, hi = paired_bootstrap(deltas, n=2000)
    assert lo <= mean <= hi
    assert lo > 0  # CI excludes 0 — a shippable signal
