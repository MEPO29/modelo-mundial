import datetime as dt

import numpy as np
import polars as pl
import pytest

from mundial.models.baseline import DixonColes, _tau


@pytest.fixture(scope="module")
def synthetic_matches():
    """Strong team beats Weak repeatedly; Mid splits with both."""
    rng = np.random.default_rng(42)
    rows = []
    teams = {"Strong": 1.6, "Mid": 1.2, "Weak": 0.7}
    names = list(teams)
    date = dt.date(2020, 1, 1)
    for k in range(600):
        h, a = rng.choice(names, size=2, replace=False)
        rows.append(
            {
                "date": date + dt.timedelta(days=k * 3),
                "home_team": h,
                "away_team": a,
                "home_score": int(rng.poisson(teams[h] / teams[a] * 1.3)),
                "away_score": int(rng.poisson(teams[a] / teams[h])),
                "neutral": bool(k % 2),
            }
        )
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted(synthetic_matches):
    return DixonColes(min_matches=5).fit(synthetic_matches, as_of=dt.date(2026, 1, 1))


def test_probabilities_sum_to_one(fitted):
    p = fitted.predict_1x2("Strong", "Weak")
    assert abs(p.sum() - 1.0) < 1e-9
    assert (p >= 0).all()


def test_strength_ordering(fitted):
    p = fitted.predict_1x2("Strong", "Weak")
    assert p[0] > 0.5, "Strong should be clear favorite over Weak"
    p_rev = fitted.predict_1x2("Weak", "Strong")
    assert p_rev[2] > 0.5


def test_home_advantage_positive_and_applied(fitted):
    assert fitted.home_adv > 0
    p_home = fitted.predict_1x2("Mid", "Strong", neutral=False)
    p_neutral = fitted.predict_1x2("Mid", "Strong", neutral=True)
    assert p_home[0] > p_neutral[0]


def test_unknown_team_maps_to_pooled(fitted):
    p = fitted.predict_1x2("Strong", "Atlantis")
    assert abs(p.sum() - 1.0) < 1e-9


def test_score_matrix_normalized(fitted):
    m = fitted.score_matrix("Strong", "Mid")
    assert abs(m.sum() - 1.0) < 1e-9


def test_tau_identity_outside_low_scores():
    t = _tau(np.array([3.0]), np.array([2.0]), np.array([1.5]), np.array([1.0]), -0.1)
    assert t[0] == 1.0
