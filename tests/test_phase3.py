import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from mundial.features.build import build_features
from mundial.models.ensemble import fit_pool_weights, pool_predict
from mundial.models.market import shin_devig
from mundial.models.simulate import load_reference, third_slot_assignments

ROOT = Path(__file__).resolve().parents[1]


# ---------------- market ----------------

def test_shin_devig_sums_to_one():
    p = shin_devig(np.array([1.5, 4.0, 6.0]))
    assert abs(p.sum() - 1.0) < 1e-6


def test_shin_devig_fair_odds_identity():
    fair = np.array([0.5, 0.3, 0.2])
    p = shin_devig(1.0 / fair)
    assert np.allclose(p, fair, atol=1e-4)


def test_shin_devig_shrinks_longshots_more():
    odds = np.array([1.4, 4.5, 9.0])
    shin = shin_devig(odds)
    q = 1.0 / odds
    proportional = q / q.sum()
    # Shin removes more implied probability from the longshot than proportional
    assert shin[2] < proportional[2]
    assert shin[0] > proportional[0]


# ---------------- ensemble pool ----------------

def test_pool_prefers_better_component():
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.integers(0, 3, n)
    good = np.full((n, 3), 0.15)
    good[np.arange(n), y] = 0.7
    noise = rng.dirichlet(np.ones(3), n)
    w = fit_pool_weights([good, noise], y)
    assert w[0] > 5 * max(w[1], 1e-6)


def test_pool_predict_normalized():
    rng = np.random.default_rng(1)
    blocks = [rng.dirichlet(np.ones(3), 50), rng.dirichlet(np.ones(3), 50)]
    p = pool_predict(np.array([0.6, 0.4]), blocks)
    assert np.allclose(p.sum(axis=1), 1.0)


# ---------------- features ----------------

@pytest.fixture(scope="module")
def small_history():
    rows = []
    date = dt.date(2023, 1, 1)
    for k in range(20):
        rows.append({
            "date": date + dt.timedelta(days=k * 10),
            "home_team": "Alpha" if k % 2 else "Beta",
            "away_team": "Beta" if k % 2 else "Alpha",
            "home_score": 2, "away_score": 0,
            "tournament": "Friendly", "city": "Quito", "neutral": False,
        })
    return pl.DataFrame(rows)


def test_features_no_leakage_first_match(small_history):
    train, _ = build_features(small_history)
    first = train.row(0, named=True)
    # before any match is observed, both teams must be at the Elo prior
    assert first["elo_h"] == 1500.0
    assert first["elo_a"] == 1500.0
    assert first["h2h_n"] == 0


def test_features_altitude_lookup(small_history):
    train, _ = build_features(small_history)
    assert train["altitude_m"][0] == 2850.0  # Quito


def test_fixture_features_continue_state(small_history):
    fixtures = pl.DataFrame([{
        "date": dt.date(2024, 1, 1), "home_team": "Alpha", "away_team": "Beta",
        "tournament": "FIFA World Cup", "city": "Unknown City", "neutral": True,
    }])
    train, fix = build_features(small_history, fixtures)
    assert fix.height == 1
    assert fix["h2h_n"][0] == 20
    assert fix["altitude_m"][0] is None


# ---------------- simulator reference data ----------------

def test_groups_match_fixture_components():
    """The Wikipedia-sourced groups must equal the team sets that actually
    play each other in the dataset's 72 group fixtures."""
    from mundial.ingest import results as results_mod
    from mundial.models.simulate import wc_matches

    if not list(results_mod.RAW_DIR.glob("*/results.csv")):
        pytest.skip("martj42 results not pulled")

    groups, _, _ = load_reference()
    played, fixtures = wc_matches()
    allm = pl.concat([played, fixtures])
    assert allm.height == 72

    pair_groups = {frozenset(g) for g in groups.values()}
    seen: dict[str, set] = {}
    for h, a in allm.select("home_team", "away_team").iter_rows():
        seen.setdefault(h, set()).update([h, a])
        seen.setdefault(a, set()).update([h, a])
    components = {frozenset(s) for s in seen.values()}
    assert components == pair_groups


def test_all_495_third_combinations_solvable():
    _, bracket, _ = load_reference()
    slot_ids, table = third_slot_assignments(bracket)
    assert len(slot_ids) == 8
    valid = table[table[:, 0] >= 0]
    assert len(valid) == 495
    for row in valid:
        assert len(set(row.tolist())) == 8  # no group used twice


def test_bracket_slots_consistent():
    groups, bracket, _ = load_reference()
    r32 = bracket["round_of_32"]
    assert len(r32) == 16
    winners = sorted(s["home"][1] for s in r32.values() if s["home"].startswith("1"))
    runners = sorted(
        [s["home"][1] for s in r32.values() if s["home"].startswith("2")]
        + [s["away"][1] for s in r32.values() if s["away"].startswith("2")]
    )
    assert winners == sorted(groups.keys())
    assert runners == sorted(groups.keys())
