"""Online learning cycle — the honest-scoring invariants and the stall guard.

These lock in the behaviour that the live loop's silent stall (results played
but never scored) violated: every played, pre-logged match is scored exactly
once, the log is immutable, and a stall is surfaced as a warning.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from mundial.online import update_cycle as uc

COMPS = ["dc", "bayes", "market"]


def _log_row(date, home, away, probs, pool, market=True):
    row = {"date": date, "home_team": home, "away_team": away}
    for c in COMPS:
        ph, pd_, pa = (probs if c != "market" else (market and probs or (None, None, None)))
        if c == "market" and not market:
            row[f"{c}_h"] = row[f"{c}_d"] = row[f"{c}_a"] = None
        else:
            row[f"{c}_h"], row[f"{c}_d"], row[f"{c}_a"] = ph, pd_, pa
    row["pool_h"], row["pool_d"], row["pool_a"] = pool
    return row


def _results(rows):
    return pl.DataFrame(
        [
            {"tournament": "FIFA World Cup", "date": d, "home_team": h,
             "away_team": a, "home_score": hs, "away_score": as_}
            for (d, h, a, hs, as_) in rows
        ]
    )


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(uc, "LOG_PATH", tmp_path / "log.parquet")
    monkeypatch.setattr(uc, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(uc, "ARTIFACTS", tmp_path)
    return tmp_path


def _seed_state():
    return {"weights": dict(uc.INITIAL_WEIGHTS), "scored": [], "score_history": [], "cusum": 0.0}


def test_scores_each_match_once_and_moves_weights(paths):
    # distinct component calls so the Hedge update actually shifts weight:
    # bayes is confidently right, market confidently wrong, on a home win.
    row = {"date": "2026-06-13", "home_team": "Brazil", "away_team": "Serbia"}
    comp_probs = {
        "dc": (0.5, 0.25, 0.25), "bayes": (0.8, 0.1, 0.1),
        "market": (0.1, 0.2, 0.7),
    }
    for c, (ph, pd_, pa) in comp_probs.items():
        row[f"{c}_h"], row[f"{c}_d"], row[f"{c}_a"] = ph, pd_, pa
    row["pool_h"], row["pool_d"], row["pool_a"] = 0.6, 0.2, 0.2
    pl.DataFrame([row]).write_parquet(uc.LOG_PATH)
    results = _results([(dt.date(2026, 6, 13), "Brazil", "Serbia", 2, 0)])

    state = _seed_state()
    scored = uc.score_new_results(state, results)
    assert len(scored) == 1
    assert len(state["score_history"]) == 1
    assert "2026-06-13|Brazil|Serbia" in state["scored"]
    # home win: the confident-correct component gains share, the wrong one loses
    assert state["weights"]["bayes"] > uc.INITIAL_WEIGHTS["bayes"]
    assert state["weights"]["market"] < uc.INITIAL_WEIGHTS["market"]

    # idempotent: re-running scores nothing new
    again = uc.score_new_results(state, results)
    assert again == []
    assert len(state["score_history"]) == 1


def test_scoreline_grid_scored_and_legacy_row_skipped(paths):
    # a grid peaked on 2-0 for a match that ends 2-0 (exact hit); a second
    # match logs NO grid (legacy) and must still be 1X2-scored but carry no
    # scoreline block.
    g = np.full((11, 11), 1e-4)
    g[2, 0] = 1.0
    g = (g / g.sum()).reshape(-1).tolist()
    rows = [
        {**_log_row("2026-06-13", "Brazil", "Serbia",
                    (0.6, 0.25, 0.15), (0.58, 0.26, 0.16)), "bayes_grid": g},
        {**_log_row("2026-06-14", "Spain", "Japan",
                    (0.6, 0.25, 0.15), (0.58, 0.26, 0.16)), "bayes_grid": None},
    ]
    pl.DataFrame(rows).write_parquet(uc.LOG_PATH)
    results = _results([
        (dt.date(2026, 6, 13), "Brazil", "Serbia", 2, 0),
        (dt.date(2026, 6, 14), "Spain", "Japan", 1, 0),
    ])

    state = _seed_state()
    uc.score_new_results(state, results)
    hist = {h["key"]: h for h in state["score_history"]}
    assert hist["2026-06-13|Brazil|Serbia"]["scoreline"]["hit"] is True
    assert "scoreline" not in hist["2026-06-14|Spain|Japan"]

    sb = uc.scoreboard(state)
    assert sb["scoreline"]["n"] == 1  # only the grid-logged match counts
    assert sb["scoreline"]["hit_rate"] == pytest.approx(1.0)


def test_played_but_unlogged_match_is_never_scored(paths):
    # log contains a different match than the one that was played
    log = pl.DataFrame([_log_row("2026-06-13", "Brazil", "Serbia",
                                 (0.6, 0.25, 0.15), (0.58, 0.26, 0.16))])
    log.write_parquet(uc.LOG_PATH)
    results = _results([(dt.date(2026, 6, 13), "Spain", "Japan", 1, 0)])

    state = _seed_state()
    scored = uc.score_new_results(state, results)
    assert scored == []  # honest scoring: no forecast logged -> not scored


def test_missing_component_inherits_pool_loss(paths):
    log = pl.DataFrame([_log_row("2026-06-13", "Brazil", "Serbia",
                                 (0.6, 0.25, 0.15), (0.58, 0.26, 0.16), market=False)])
    log.write_parquet(uc.LOG_PATH)
    results = _results([(dt.date(2026, 6, 13), "Brazil", "Serbia", 2, 0)])

    state = _seed_state()
    uc.score_new_results(state, results)
    h = state["score_history"][0]
    assert h["losses"]["market"] == pytest.approx(h["pool_loss"])


def test_append_log_freezes_first_forecast(paths):
    first = pl.DataFrame([_log_row("2026-06-13", "Brazil", "Serbia",
                                   (0.6, 0.25, 0.15), (0.58, 0.26, 0.16))])
    first = first.with_columns(pl.lit(0.58).alias("pool_h"))
    uc.append_log(first)
    # a later cycle tries to log a *different* forecast for the same match
    second = pl.DataFrame([_log_row("2026-06-13", "Brazil", "Serbia",
                                    (0.1, 0.2, 0.7), (0.12, 0.2, 0.68))])
    uc.append_log(second)

    stored = pl.read_parquet(uc.LOG_PATH)
    assert stored.height == 1
    assert stored["pool_h"][0] == pytest.approx(0.58)  # original kept


def test_update_cusum_accumulates_and_skips_no_market():
    state = {"score_history": [
        {"pool_loss": 1.2, "losses": {"market": 1.0}},      # +0.15
        {"pool_loss": 0.5, "losses": {"market": 0.6}},      # -0.15 -> floored region
        {"pool_loss": 1.0, "losses": {}},                   # no market -> skipped
    ]}
    cusum = uc.update_cusum(state)
    # step1: max(0, 0 + (1.2-1.0) - 0.05) = 0.15; step2: max(0, 0.15 + (0.5-0.6) -0.05)=0.0
    assert cusum == pytest.approx(0.0)
    assert state["cusum"] == pytest.approx(0.0)


def test_freshness_warning_flags_stall_then_silent(paths):
    log = pl.DataFrame([_log_row("2026-06-13", "Brazil", "Serbia",
                                 (0.6, 0.25, 0.15), (0.58, 0.26, 0.16))])
    log.write_parquet(uc.LOG_PATH)
    results = _results([(dt.date(2026, 6, 13), "Brazil", "Serbia", 2, 0)])

    stalled = {"scored": []}
    warn = uc.freshness_warning(results, stalled, dt.date(2026, 6, 13))
    assert warn is not None and "unscored" in warn

    healthy = {"scored": ["2026-06-13|Brazil|Serbia"]}
    assert uc.freshness_warning(results, healthy, dt.date(2026, 6, 13)) is None


def test_scoreboard_summarizes_history():
    state = {"score_history": [
        {"pool_loss": 0.5, "losses": {c: 0.4 for c in COMPS}},
        {"pool_loss": 0.7, "losses": {c: 0.6 for c in COMPS}},
    ]}
    sb = uc.scoreboard(state)
    assert sb["n_scored"] == 2
    assert sb["pool_logloss"] == pytest.approx(0.6)
    assert sb["bayes_logloss"] == pytest.approx(0.5)


def test_load_state_drops_retired_component_and_renormalizes(paths):
    # the exact rot that left the loop inert: a stale state file with a
    # now-retired 'gbm' key and weights that no longer sum to 1.
    import json
    stale = {"weights": {"dc": 0.05, "bayes": 0.55, "gbm": 0.05, "market": 0.35},
             "scored": [], "score_history": [], "cusum": 0.0}
    uc.STATE_PATH.write_text(json.dumps(stale), encoding="utf-8")

    state = uc.load_state()
    assert set(state["weights"]) == set(COMPS)  # gbm dropped
    assert sum(state["weights"].values()) == pytest.approx(1.0)  # renormalized


def test_write_report_emits_warning_and_weights(paths, monkeypatch):
    monkeypatch.setattr(uc, "REPORTS", paths)
    state = {"weights": dict(uc.INITIAL_WEIGHTS), "score_history": []}
    path = uc.write_report(
        dt.date(2026, 6, 15), [], state, pl.DataFrame(), None, 0.0, "loop stalled!"
    )
    text = path.read_text(encoding="utf-8")
    assert "Cycle report" in text
    assert "loop stalled!" in text  # the freshness warning is surfaced
    assert "Pool weights" in text
