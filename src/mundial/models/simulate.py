"""Monte Carlo simulation of the 2026 World Cup (48 teams, 104 matches).

Every simulation run draws ONE sample from the Bayesian posterior and plays
the whole tournament under it — epistemic uncertainty about team strength
propagates into bracket probabilities instead of being averaged away (the
standard point-estimate mistake that makes favorites look too safe).

Already-played matches enter as fixed results; everything else is sampled.

Documented approximations vs the full FIFA regulations:
- Group/third-place tiebreakers: points > GD > GF > random (head-to-head and
  fair-play steps omitted).
- Third-place bracket allocation: a valid matching against each slot's
  eligible-group list (FIFA Annex C fixes one specific matching per
  combination; any valid matching is used here).
- Knockout: extra time at 1/3 of the 90' scoring rates, then a 50/50
  shootout. Hosts get the home-advantage term when playing in their country.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl

from mundial.ingest.results import load_fixtures, load_results
from mundial.models.bayes import DynamicHierarchicalPoisson

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REFERENCE = PROJECT_ROOT / "data" / "reference"

ET_RATE_FACTOR = 1.0 / 3.0
ROUNDS = ["r32", "r16", "qf", "sf", "final", "champion"]


def load_reference() -> tuple[dict, dict, dict]:
    groups = json.loads((REFERENCE / "groups_2026.json").read_text())
    bracket = json.loads((REFERENCE / "bracket_2026.json").read_text())
    venues = pl.read_csv(REFERENCE / "venues_2026.csv")
    city_country = dict(venues.select("city", "country").iter_rows())
    return groups, bracket, city_country


def third_slot_assignments(bracket: dict) -> tuple[list[str], np.ndarray]:
    """Precompute a valid third-place allocation for every 8-of-12 combination.

    Returns (slot_match_ids, table) where table[mask] is an array of 8 group
    indices (aligned with slot order), or -1s for invalid masks. mask bit g =
    group g's third qualified.
    """
    slots = [
        (mid, [ord(c) - 65 for c in spec["away"].split(":")[1]])
        for mid, spec in bracket["round_of_32"].items()
        if spec["away"].startswith("3:")
    ]
    slot_ids = [s[0] for s in slots]
    eligible = [set(s[1]) for s in slots]

    table = np.full((1 << 12, 8), -1, dtype=np.int8)
    for combo in combinations(range(12), 8):
        mask = sum(1 << g for g in combo)
        # backtracking, most-constrained slot first
        order = sorted(range(8), key=lambda i: len(eligible[i] & set(combo)))
        assign = [-1] * 8
        used: set[int] = set()

        def solve(k: int) -> bool:
            if k == 8:
                return True
            i = order[k]
            for g in sorted(eligible[i] & set(combo) - used):
                assign[i] = g
                used.add(g)
                if solve(k + 1):
                    return True
                used.discard(g)
            assign[i] = -1
            return False

        if not solve(0):
            raise RuntimeError(f"no valid third-place matching for {combo}")
        table[mask] = assign
    return slot_ids, table


class TournamentSimulator:
    def __init__(self, model: DynamicHierarchicalPoisson, n_sims: int = 100_000,
                 seed: int = 0):
        self.model = model
        self.n_sims = n_sims
        self.rng = np.random.default_rng(seed)
        self.groups, self.bracket, self.city_country = load_reference()
        self.teams = sorted(t for g in self.groups.values() for t in g)
        self.tidx = {t: i for i, t in enumerate(self.teams)}
        self.team_country = {t: t for t in ("Mexico", "Canada", "United States")}

        s = model._samples
        S = s["intercept"].shape[0]
        self.sample_of_run = self.rng.integers(0, S, size=n_sims)
        atk = np.empty((S, len(self.teams)))
        dfn = np.empty((S, len(self.teams)))
        for t, i in self.tidx.items():
            atk[:, i], dfn[:, i] = model._team_strengths(t)
        self.atk = atk[self.sample_of_run]  # (n_sims, 48)
        self.dfn = dfn[self.sample_of_run]
        self.icpt = np.asarray(s["intercept"])[self.sample_of_run]
        self.hadv = np.asarray(s["home_adv"])[self.sample_of_run]

    def _rates(self, hi: np.ndarray, ai: np.ndarray, home_flag: np.ndarray):
        r = np.arange(self.n_sims)
        lam = np.exp(self.icpt + self.atk[r, hi] - self.dfn[r, ai] + self.hadv * home_flag)
        mu = np.exp(self.icpt + self.atk[r, ai] - self.dfn[r, hi])
        return lam, mu

    def _is_home(self, team_idx: np.ndarray, city: str) -> np.ndarray:
        country = self.city_country.get(city, "")
        host_idx = self.tidx.get(country if country != "United States" else "United States", -1)
        if country == "United States":
            host_idx = self.tidx["United States"]
        return (team_idx == host_idx).astype(float)

    # ---------------- group stage ----------------

    def simulate_group_stage(self, group_fixtures: pl.DataFrame):
        """group_fixtures: 72 rows with home_team, away_team, neutral, city,
        home_score/away_score (null when unplayed)."""
        n, G = self.n_sims, len(self.groups)
        pts = np.zeros((n, 48)); gd = np.zeros((n, 48)); gf = np.zeros((n, 48))

        for r in group_fixtures.iter_rows(named=True):
            hi = np.full(n, self.tidx[r["home_team"]])
            ai = np.full(n, self.tidx[r["away_team"]])
            if r["home_score"] is not None:
                hg = np.full(n, r["home_score"]); ag = np.full(n, r["away_score"])
            else:
                lam, mu = self._rates(hi, ai, np.full(n, 0.0 if r["neutral"] else 1.0))
                hg = self.rng.poisson(lam); ag = self.rng.poisson(mu)
            h, a = hi[0], ai[0]
            pts[:, h] += np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
            pts[:, a] += np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
            gd[:, h] += hg - ag; gd[:, a] += ag - hg
            gf[:, h] += hg; gf[:, a] += ag

        key = pts * 1e9 + (gd + 100) * 1e4 + gf * 10 + self.rng.random((n, 48))
        ranks = np.zeros((n, G, 4), dtype=np.int64)  # team idx by finish position
        for g, letter in enumerate(sorted(self.groups)):
            members = np.array([self.tidx[t] for t in self.groups[letter]])
            order = np.argsort(-key[:, members], axis=1)
            ranks[:, g] = members[order]
        return ranks, key

    # ---------------- knockout ----------------

    def _ko_match(self, hi, ai, city):
        home_flag = self._is_home(hi, city) - self._is_home(ai, city)
        lam, mu = self._rates(hi, ai, home_flag)
        hg = self.rng.poisson(lam); ag = self.rng.poisson(mu)
        tie = hg == ag
        if tie.any():
            hg = hg + self.rng.poisson(lam * ET_RATE_FACTOR) * tie
            ag = ag + self.rng.poisson(mu * ET_RATE_FACTOR) * tie
        tie = hg == ag
        pens = self.rng.random(self.n_sims) < 0.5
        home_wins = np.where(tie, pens, hg > ag)
        return np.where(home_wins, hi, ai)

    def run(self) -> pl.DataFrame:
        played, fixtures = wc_matches()
        ranks, key = self.simulate_group_stage(pl.concat([played, fixtures]))

        G = len(self.groups)
        winners = {chr(65 + g): ranks[:, g, 0] for g in range(G)}
        runners = {chr(65 + g): ranks[:, g, 1] for g in range(G)}
        thirds = ranks[:, :, 2]  # (n, 12) team idx of each group's third

        third_key = np.take_along_axis(key, thirds, axis=1)
        qual_order = np.argsort(-third_key, axis=1)[:, :8]  # group indices, best 8
        mask = np.bitwise_or.reduce(1 << qual_order.astype(np.int64), axis=1)

        slot_ids, table = third_slot_assignments(self.bracket)
        slot_groups = table[mask]  # (n, 8) group idx per third-slot
        third_of_slot = {
            sid: np.take_along_axis(thirds, slot_groups[:, [j]], axis=1)[:, 0]
            for j, sid in enumerate(slot_ids)
        }

        reached = {r: np.zeros((self.n_sims, 48), dtype=bool) for r in ROUNDS}
        ko_winner: dict[str, np.ndarray] = {}

        def resolve(slot: str, mid: str) -> np.ndarray:
            if slot.startswith("3:"):
                return third_of_slot[mid]
            pos, letter = slot[0], slot[1]
            return winners[letter] if pos == "1" else runners[letter]

        rows = np.arange(self.n_sims)
        for mid, spec in self.bracket["round_of_32"].items():
            hi = resolve(spec["home"], mid)
            ai = resolve(spec["away"], mid)
            reached["r32"][rows, hi] = True
            reached["r32"][rows, ai] = True
            ko_winner[mid] = self._ko_match(hi, ai, spec["city"])

        ko_cities = {
            "97": "Foxborough", "98": "Inglewood", "99": "Miami Gardens",
            "100": "Kansas City", "101": "Arlington", "102": "Atlanta",
            "104": "East Rutherford",
        }
        for round_name, stage in [("r16", "round_of_16"), ("qf", "quarterfinals"),
                                  ("sf", "semifinals"), ("final", "final")]:
            for mid, (m1, m2) in self.bracket[stage].items():
                hi, ai = ko_winner[m1], ko_winner[m2]
                reached[round_name][rows, hi] = True
                reached[round_name][rows, ai] = True
                ko_winner[mid] = self._ko_match(hi, ai, ko_cities.get(mid, ""))
        champ = ko_winner["104"]
        reached["champion"][rows, champ] = True

        out = pl.DataFrame(
            {
                "team": self.teams,
                **{f"p_{r}": reached[r].mean(axis=0) for r in ROUNDS},
            }
        ).sort("p_champion", descending=True)
        return out


def wc_matches() -> tuple[pl.DataFrame, pl.DataFrame]:
    """(played, upcoming) 2026 WC group matches, harmonized columns."""
    cols = ["date", "home_team", "away_team", "home_score", "away_score",
            "neutral", "city"]
    played = (
        load_results()
        .filter((pl.col("tournament") == "FIFA World Cup")
                & (pl.col("date") >= dt.date(2026, 6, 11)))
        .select(cols)
    )
    fixtures = (
        load_fixtures(tournament="FIFA World Cup")
        .filter(pl.col("date") >= dt.date(2026, 6, 11))
        .select(cols)
    )
    return played, fixtures


def main(n_sims: int = 100_000) -> None:
    results = load_results()
    print("fitting Bayesian backbone...")
    model = DynamicHierarchicalPoisson().fit(
        results, as_of=dt.date.today() + dt.timedelta(days=1)
    )
    sim = TournamentSimulator(model, n_sims=n_sims)
    table = sim.run()

    out_path = PROJECT_ROOT / "reports" / f"sim_{dt.date.today().isoformat()}.csv"
    out_path.parent.mkdir(exist_ok=True)
    table.write_csv(out_path)

    print(f"\n{n_sims:,} tournament simulations | saved to {out_path}\n")
    print(f"{'team':<22} {'R32':>6} {'R16':>6} {'QF':>6} {'SF':>6} {'Final':>6} {'Champ':>6}")
    for r in table.head(20).iter_rows(named=True):
        print(f"{r['team']:<22} {r['p_r32']:>6.1%} {r['p_r16']:>6.1%} {r['p_qf']:>6.1%} "
              f"{r['p_sf']:>6.1%} {r['p_final']:>6.1%} {r['p_champion']:>6.1%}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100_000)
