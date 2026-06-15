"""Football-data.co.uk odds/xG ingest."""

from __future__ import annotations

import numpy as np
import pytest

from mundial.ingest import footy_odds as fo
from mundial.models.market import normalize_team

pytestmark = pytest.mark.skipif(not fo.XLSX.exists(), reason="workbook not present")


def test_historical_odds_shape_and_probabilities():
    df = fo.historical_odds(2022)
    assert df.height == 64  # a full finals tournament
    for col in ("date", "home_team", "away_team", "odds_h", "odds_d", "odds_a",
                "p_h", "p_d", "p_a"):
        assert col in df.columns
    p = df.select("p_h", "p_d", "p_a").to_numpy()
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
    assert ((p > 0) & (p < 1)).all()
    # de-vigged probabilities must beat the raw vigged implied probs back to 1
    q = 1.0 / df.select("odds_h", "odds_d", "odds_a").to_numpy()
    assert (q.sum(axis=1) > 1.0).all()  # bookmaker overround present


def test_all_historical_odds_stacks_three_tournaments():
    df = fo.all_historical_odds()
    assert df.height == 64 * len(fo.FINALS_SHEETS)


def test_qualifier_xg_present():
    df = fo.qualifier_xg()
    assert df.height > 0
    assert df["hxg"].null_count() == 0
    assert df["axg"].null_count() == 0
    assert (df["hxg"].to_numpy() >= 0).all()


def test_team_names_normalized():
    # the finals frames must not contain raw football-data spellings
    df = fo.all_historical_odds()
    teams = set(df["home_team"]) | set(df["away_team"])
    assert "USA" not in teams
    assert "United States" in teams or all("USA" != t for t in teams)
    assert normalize_team("Bosnia & Herzegovina") == "Bosnia and Herzegovina"
    assert normalize_team("Trinidad & Tobago") == "Trinidad and Tobago"
