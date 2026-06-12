import datetime as dt

import numpy as np
import polars as pl
import pytest

from mundial.ingest.confederations import team_confederations
from mundial.models.bayes import DynamicHierarchicalPoisson


@pytest.fixture(scope="module")
def synthetic_matches():
    """Strong beats Weak; played in a fake continental cup so confederations resolve."""
    rng = np.random.default_rng(7)
    rows = []
    teams = {"Strong": 1.6, "Mid": 1.2, "Weak": 0.7}
    names = list(teams)
    date = dt.date(2019, 1, 1)
    for k in range(500):
        h, a = rng.choice(names, size=2, replace=False)
        rows.append(
            {
                "date": date + dt.timedelta(days=k * 4),
                "home_team": h,
                "away_team": a,
                "home_score": int(rng.poisson(teams[h] / teams[a] * 1.3)),
                "away_score": int(rng.poisson(teams[a] / teams[h])),
                "tournament": "Friendly" if k % 3 else "UEFA Euro",
                "neutral": bool(k % 2),
            }
        )
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted(synthetic_matches):
    return DynamicHierarchicalPoisson(
        since=dt.date(2019, 1, 1), svi_steps=1000, num_samples=200
    ).fit(synthetic_matches, as_of=dt.date(2025, 1, 1))


def test_probabilities_sum_to_one(fitted):
    p = fitted.predict_1x2("Strong", "Weak")
    assert abs(p.sum() - 1.0) < 1e-9
    assert (p >= 0).all()


def test_strength_ordering(fitted):
    assert fitted.predict_1x2("Strong", "Weak")[0] > 0.5
    assert fitted.predict_1x2("Weak", "Strong")[2] > 0.5


def test_home_advantage_applied(fitted):
    p_home = fitted.predict_1x2("Mid", "Strong", neutral=False)
    p_neutral = fitted.predict_1x2("Mid", "Strong", neutral=True)
    assert p_home[0] > p_neutral[0]


def test_unseen_team_falls_back_to_confederation_mean(fitted):
    p = fitted.predict_1x2("Strong", "Atlantis")
    assert abs(p.sum() - 1.0) < 1e-9
    # Atlantis gets the OTHER-pool prior, far weaker than Strong's fitted rating
    assert p[0] > 0.4


def test_posterior_not_degenerate(fitted):
    s = fitted._samples
    assert np.abs(s["atk"]).max() < 10, "attack strengths exploded (bad init?)"
    assert s["intercept"].std() > 0


def test_confederation_mapping(synthetic_matches):
    mapping = team_confederations(synthetic_matches)
    assert mapping["Strong"] == "UEFA"
    assert mapping["Weak"] == "UEFA"
