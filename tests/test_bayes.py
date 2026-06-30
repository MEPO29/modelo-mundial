import datetime as dt

import numpy as np
import polars as pl
import pytest

from mundial.ingest.confederations import team_confederations
from mundial.models.bayes import (
    ET_RATE_FACTOR,
    DynamicHierarchicalPoisson,
    _bp_grid_np,
)


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
    # the shared bivariate-Poisson component must stay a modest correlation
    # term, not absorb the overall goal level (intercept identifiability)
    lam3 = np.exp(s["cov_intercept"])
    assert 0.01 < lam3.mean() < 1.0, "shared component absorbed the goal level"


def test_bivariate_poisson_lifts_draw_mass_at_equal_means():
    # mean-matched comparison: independent (l1,l2) vs BP (l1-l3,l2-l3,l3) share
    # the same marginal goal expectations, so any extra diagonal mass is the
    # genuine correlation the shared component adds (the draw-discrimination win
    # an independent double-Poisson structurally cannot produce).
    l1, l2, l3 = 1.4, 1.2, 0.3
    indep = _bp_grid_np(np.log([l1]), np.log([l2]), np.array([-50.0]))[0]
    bp = _bp_grid_np(np.log([l1 - l3]), np.log([l2 - l3]), np.log([l3]))[0]
    indep /= indep.sum()
    bp /= bp.sum()
    g = np.arange(indep.shape[0])
    assert abs((indep.sum(axis=1) * g).sum() - l1) < 0.05  # marginal mean preserved
    assert abs((bp.sum(axis=1) * g).sum() - l1) < 0.05
    assert np.trace(bp) > np.trace(indep) + 0.01  # excess draw mass


def test_knockout_breakdown_is_consistent(fitted):
    kb = fitted.knockout_breakdown("Strong", "Weak")
    # each stage is a valid distribution
    assert abs(kb["ft"].sum() - 1.0) < 1e-9
    assert abs(kb["et"].sum() - 1.0) < 1e-9
    assert kb["pens"].tolist() == [0.5, 0.5]
    # someone always advances; the two advancement probs partition 1
    assert abs(kb["advance"].sum() - 1.0) < 1e-9
    assert (kb["advance"] >= 0).all()
    # a tie reaches penalties strictly less often than it reaches extra time
    assert kb["p_reach_pens"] < kb["p_reach_et"]
    assert kb["p_reach_et"] == pytest.approx(kb["ft"][1])
    # Strong is favoured to go through against Weak
    assert kb["advance"][0] > 0.5


def test_extra_time_scores_fewer_goals_than_full_time(fitted):
    # ET is 30' at 1/3 of the 90' rate, so its scoreline grid concentrates far
    # more mass on 0-0 (more ET ties -> more shootouts) than the 90' grid.
    ft = fitted.score_matrix("Strong", "Weak")
    et = fitted.score_matrix("Strong", "Weak", rate_factor=ET_RATE_FACTOR)
    assert et[0, 0] > ft[0, 0]
    assert np.trace(et) > np.trace(ft)  # more draws over the shorter period


def test_confederation_mapping(synthetic_matches):
    mapping = team_confederations(synthetic_matches)
    assert mapping["Strong"] == "UEFA"
    assert mapping["Weak"] == "UEFA"
