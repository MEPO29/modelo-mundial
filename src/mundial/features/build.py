"""Streaming match featurizer.

A single chronological pass over matches, maintaining per-team state (Elo,
recent form, schedule density, head-to-head). Features for each match are
emitted BEFORE the state is updated with its result, so leakage is impossible
by construction — every feature is a function of strictly earlier matches.

The same pass extends over unplayed fixtures (state simply stops updating),
which is how inference-time features for upcoming WC matches are produced.
"""

from __future__ import annotations

import datetime as dt
from collections import deque
from pathlib import Path

import polars as pl

from mundial.ingest.confederations import CONFEDERATIONS, team_confederations

PROJECT_ROOT = Path(__file__).resolve().parents[3]

ELO_K = 32.0
ELO_HOME = 60.0
ELO_INIT = 1500.0
FORM_WINDOW = 10
REST_CAP = 60.0

TOURNAMENT_TYPE = {
    "FIFA World Cup": 0,
    "FIFA World Cup qualification": 1,
    "Friendly": 5,
}
CONTINENTAL = {"UEFA Euro", "Copa América", "African Cup of Nations", "AFC Asian Cup",
               "Gold Cup", "Oceania Nations Cup", "Confederations Cup"}

FEATURE_COLS = [
    "elo_h", "elo_a", "elo_diff",
    "rest_h", "rest_a", "rest_diff",
    "dens90_h", "dens90_a",
    "form5_gd_h", "form5_gd_a", "form10_gd_h", "form10_gd_a",
    "gf5_h", "ga5_h", "gf5_a", "ga5_a",
    "h2h_gd", "h2h_n",
    "home_flag", "altitude_m",
    "ttype", "conf_h", "conf_a",
]
CATEGORICAL_COLS = ["ttype", "conf_h", "conf_a"]


def _tournament_type(name: str) -> int:
    if name in TOURNAMENT_TYPE:
        return TOURNAMENT_TYPE[name]
    if name in CONTINENTAL:
        return 2
    if "qualification" in name:
        return 3
    if "Nations League" in name:
        return 4
    return 6


def _altitudes() -> dict[str, float]:
    df = pl.read_csv(PROJECT_ROOT / "data" / "reference" / "city_altitudes.csv")
    return dict(df.iter_rows())


class _TeamState:
    __slots__ = ("elo", "last_date", "dates", "recent")

    def __init__(self) -> None:
        self.elo = ELO_INIT
        self.last_date: dt.date | None = None
        self.dates: deque[dt.date] = deque()
        self.recent: deque[tuple[int, int]] = deque(maxlen=FORM_WINDOW)


class Featurizer:
    """One pass over matches sorted by date; call features() then update()."""

    def __init__(self, conf_of: dict[str, str]) -> None:
        self.conf_of = conf_of
        self.alt = _altitudes()
        self.state: dict[str, _TeamState] = {}
        self.h2h: dict[tuple[str, str], list[int]] = {}

    def _st(self, team: str) -> _TeamState:
        if team not in self.state:
            self.state[team] = _TeamState()
        return self.state[team]

    def features(self, date: dt.date, home: str, away: str, tournament: str,
                 city: str, neutral: bool) -> dict:
        sh, sa = self._st(home), self._st(away)

        def rest(s: _TeamState) -> float:
            return REST_CAP if s.last_date is None else min((date - s.last_date).days, REST_CAP)

        def dens(s: _TeamState) -> int:
            while s.dates and (date - s.dates[0]).days > 90:
                s.dates.popleft()
            return len(s.dates)

        def form(s: _TeamState, n: int) -> tuple[float, float, float]:
            recent = list(s.recent)[-n:]
            if not recent:
                return 0.0, 0.0, 0.0
            gf = sum(r[0] for r in recent) / len(recent)
            ga = sum(r[1] for r in recent) / len(recent)
            return gf - ga, gf, ga

        gd5h, gf5h, ga5h = form(sh, 5)
        gd5a, gf5a, ga5a = form(sa, 5)
        gd10h, _, _ = form(sh, 10)
        gd10a, _, _ = form(sa, 10)
        past = self.h2h.get((home, away), [])

        return {
            "elo_h": sh.elo,
            "elo_a": sa.elo,
            "elo_diff": sh.elo - sa.elo,
            "rest_h": rest(sh),
            "rest_a": rest(sa),
            "rest_diff": rest(sh) - rest(sa),
            "dens90_h": dens(sh),
            "dens90_a": dens(sa),
            "form5_gd_h": gd5h,
            "form5_gd_a": gd5a,
            "form10_gd_h": gd10h,
            "form10_gd_a": gd10a,
            "gf5_h": gf5h,
            "ga5_h": ga5h,
            "gf5_a": gf5a,
            "ga5_a": ga5a,
            "h2h_gd": sum(past) / len(past) if past else 0.0,
            "h2h_n": len(past),
            "home_flag": 0.0 if neutral else 1.0,
            "altitude_m": self.alt.get(city),
            "ttype": _tournament_type(tournament),
            "conf_h": CONFEDERATIONS.index(self.conf_of.get(home, "OTHER")),
            "conf_a": CONFEDERATIONS.index(self.conf_of.get(away, "OTHER")),
        }

    def update(self, date: dt.date, home: str, away: str, hg: int, ag: int,
               neutral: bool) -> None:
        sh, sa = self._st(home), self._st(away)
        exp_h = 1.0 / (1.0 + 10 ** (-((sh.elo - sa.elo) + (0 if neutral else ELO_HOME)) / 400))
        score_h = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        margin = 1.0 + 0.5 * max(abs(hg - ag) - 1, 0)  # diminishing goal-diff boost
        sh.elo += ELO_K * margin * (score_h - exp_h)
        sa.elo -= ELO_K * margin * (score_h - exp_h)

        for s, gf, ga in ((sh, hg, ag), (sa, ag, hg)):
            s.last_date = date
            s.dates.append(date)
            s.recent.append((gf, ga))
        self.h2h.setdefault((home, away), []).append(hg - ag)
        self.h2h.setdefault((away, home), []).append(ag - hg)


def build_features(
    played: pl.DataFrame, fixtures: pl.DataFrame | None = None
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Feature matrices for played matches and (optionally) upcoming fixtures.

    `played` must be sorted by date and contain only rows with scores.
    """
    conf_of = team_confederations(played)
    fz = Featurizer(conf_of)

    rows = []
    for r in played.iter_rows(named=True):
        f = fz.features(r["date"], r["home_team"], r["away_team"], r["tournament"],
                        r["city"], r["neutral"])
        hs, as_ = r["home_score"], r["away_score"]
        f["outcome"] = 0 if hs > as_ else 1 if hs == as_ else 2
        f["date"] = r["date"]
        f["home_team"], f["away_team"] = r["home_team"], r["away_team"]
        f["tournament"] = r["tournament"]
        rows.append(f)
        fz.update(r["date"], r["home_team"], r["away_team"], hs, as_, r["neutral"])
    train = pl.DataFrame(rows, schema_overrides={"altitude_m": pl.Float64})

    fix = None
    if fixtures is not None:
        frows = []
        for r in fixtures.iter_rows(named=True):
            f = fz.features(r["date"], r["home_team"], r["away_team"], r["tournament"],
                            r["city"], r["neutral"])
            f["date"] = r["date"]
            f["home_team"], f["away_team"] = r["home_team"], r["away_team"]
            f["tournament"] = r["tournament"]
            frows.append(f)
        fix = pl.DataFrame(frows, schema_overrides={"altitude_m": pl.Float64})
    return train, fix
