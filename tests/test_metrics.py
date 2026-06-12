import numpy as np

from mundial.eval.metrics import brier, ece, log_loss, rps


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
