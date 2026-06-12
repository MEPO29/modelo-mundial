import numpy as np

from mundial.models.ensemble import (
    WEIGHT_FLOOR,
    hedge_update,
    pool_predict,
    pool_predict_partial,
)
from mundial.models.market import normalize_team


def test_hedge_rewards_better_component():
    w = np.array([0.5, 0.5])
    # component 0 predicted the outcome well (low loss), component 1 poorly
    w2 = hedge_update(w, np.array([0.3, 2.0]))
    assert w2[0] > w[0]
    assert w2[1] < w[1]
    assert abs(w2.sum() - w.sum()) < 1e-9


def test_hedge_total_mass_preserved_many_steps():
    w = np.array([0.05, 0.55, 0.05, 0.35])
    rng = np.random.default_rng(0)
    for _ in range(100):
        w = hedge_update(w, rng.uniform(0.5, 2.0, size=4))
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w >= WEIGHT_FLOOR - 1e-12).all()


def test_hedge_floor_keeps_components_alive():
    w = np.array([0.97, 0.01, 0.01, 0.01])
    for _ in range(50):
        w = hedge_update(w, np.array([0.1, 3.0, 3.0, 3.0]))
    assert (w >= WEIGHT_FLOOR - 1e-12).all()
    # a dead component can still come back
    for _ in range(30):
        w = hedge_update(w, np.array([3.0, 0.1, 3.0, 3.0]))
    assert w[1] > 0.5


def test_pool_partial_matches_full_when_all_present():
    rng = np.random.default_rng(1)
    blocks = [rng.dirichlet(np.ones(3), 10) for _ in range(3)]
    w = np.array([0.5, 0.3, 0.2])
    assert np.allclose(pool_predict_partial(w, blocks), pool_predict(w, blocks))


def test_pool_partial_rescales_missing():
    rng = np.random.default_rng(2)
    b0 = rng.dirichlet(np.ones(3), 10)
    b1 = rng.dirichlet(np.ones(3), 10)
    w = np.array([0.6, 0.2, 0.2])
    p = pool_predict_partial(w, [b0, b1, None])
    # equivalent to pooling the available two with weights scaled to total 1.0
    expected = pool_predict(np.array([0.75, 0.25]), [b0, b1])
    assert np.allclose(p, expected)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_team_aliases():
    assert normalize_team("USA") == "United States"
    assert normalize_team("Bosnia & Herzegovina") == "Bosnia and Herzegovina"
    assert normalize_team("Brazil") == "Brazil"
