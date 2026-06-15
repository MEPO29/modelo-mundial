"""Static Dixon-Coles baseline with exponential time decay.

This is the Phase-1 reference model: independent Poisson goal rates with
per-team attack/defense, a home-advantage term (suppressed at neutral venues),
the Dixon-Coles low-score correlation tau, and match weights that decay with
age. Every later layer must beat this on walk-forward RPS to ship.

Fit is two-stage (the standard practical estimator): (1) weighted Poisson
likelihood for attack/defense/home-advantage via L-BFGS with an analytic
gradient, (2) one-dimensional profile likelihood for tau's rho with rates
held fixed.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy.optimize import minimize, minimize_scalar
from scipy.special import factorial

MAX_GOALS = 10  # scoreline grid truncation
L2 = 1e-3  # ridge pinning the attack/defense level (identifiability)


@dataclass
class DixonColes:
    half_life_days: float = 900.0  # weight halves every ~2.5 years
    min_matches: int = 10  # teams below this are pooled into a generic team
    teams: list[str] = field(default_factory=list)
    attack: np.ndarray | None = None
    defense: np.ndarray | None = None
    home_adv: float = 0.25
    rho: float = -0.05
    _index: dict[str, int] = field(default_factory=dict)

    POOLED = "__other__"

    def fit(self, matches: pl.DataFrame, as_of: dt.date) -> DixonColes:
        """matches: columns date, home_team, away_team, home_score, away_score, neutral."""
        df = matches.filter(pl.col("date") < as_of)

        counts: dict[str, int] = {}
        for col in ("home_team", "away_team"):
            for t, c in df.group_by(col).len().iter_rows():
                counts[t] = counts.get(t, 0) + c
        kept = sorted(t for t, c in counts.items() if c >= self.min_matches)
        self.teams = kept + [self.POOLED]
        self._index = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)

        hi = np.array([self._index.get(t, n - 1) for t in df["home_team"]])
        ai = np.array([self._index.get(t, n - 1) for t in df["away_team"]])
        hg = np.minimum(df["home_score"].to_numpy(), MAX_GOALS).astype(float)
        ag = np.minimum(df["away_score"].to_numpy(), MAX_GOALS).astype(float)
        home_flag = 1.0 - df["neutral"].to_numpy().astype(float)
        days_ago = np.array([(as_of - d).days for d in df["date"]], dtype=float)
        w = 0.5 ** (days_ago / self.half_life_days)

        def nll_grad(x):
            atk, dfn, ha = x[:n], x[n : 2 * n], x[2 * n]
            lam = np.exp(atk[hi] - dfn[ai] + ha * home_flag)
            mu = np.exp(atk[ai] - dfn[hi])
            nll = -np.sum(w * (hg * np.log(lam) - lam + ag * np.log(mu) - mu))
            nll += L2 * (atk @ atk + dfn @ dfn)
            rh = w * (hg - lam)  # d(loglik)/d(log lam) per match
            ra = w * (ag - mu)
            g_atk = -(np.bincount(hi, rh, n) + np.bincount(ai, ra, n)) + 2 * L2 * atk
            g_dfn = np.bincount(ai, rh, n) + np.bincount(hi, ra, n) + 2 * L2 * dfn
            g_ha = -np.sum(rh * home_flag)
            return nll, np.concatenate([g_atk, g_dfn, [g_ha]])

        x0 = np.zeros(2 * n + 1)
        x0[2 * n] = 0.25
        res = minimize(nll_grad, x0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
        self.attack, self.defense, self.home_adv = res.x[:n], res.x[n : 2 * n], res.x[2 * n]

        lam = np.exp(self.attack[hi] - self.defense[ai] + self.home_adv * home_flag)
        mu = np.exp(self.attack[ai] - self.defense[hi])

        def rho_nll(rho):
            t = _tau(hg, ag, lam, mu, rho)
            return -np.sum(w * np.log(np.clip(t, 1e-10, None)))

        self.rho = minimize_scalar(rho_nll, bounds=(-0.5, 0.5), method="bounded").x
        return self

    def rates(self, home: str, away: str, neutral: bool = True) -> tuple[float, float]:
        i = self._index.get(home, len(self.teams) - 1)
        j = self._index.get(away, len(self.teams) - 1)
        ha = 0.0 if neutral else self.home_adv
        lam = float(np.exp(self.attack[i] - self.defense[j] + ha))
        mu = float(np.exp(self.attack[j] - self.defense[i]))
        return lam, mu

    def score_matrix(self, home: str, away: str, neutral: bool = True) -> np.ndarray:
        """Joint P(home_goals=h, away_goals=a) on a (MAX_GOALS+1)^2 grid."""
        lam, mu = self.rates(home, away, neutral)
        g = np.arange(MAX_GOALS + 1)
        ph = np.exp(-lam) * lam**g / factorial(g, exact=False)
        pa = np.exp(-mu) * mu**g / factorial(g, exact=False)
        m = np.outer(ph, pa)
        for h in (0, 1):
            for a in (0, 1):
                m[h, a] *= _tau(
                    np.array([h]), np.array([a]), np.array([lam]), np.array([mu]), self.rho
                )[0]
        return m / m.sum()

    def predict_1x2(self, home: str, away: str, neutral: bool = True) -> np.ndarray:
        """[P(home win), P(draw), P(away win)] — outcome order 0/1/2 as in eval.metrics."""
        m = self.score_matrix(home, away, neutral)
        return np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])


def _tau(hg, ag, lam, mu, rho):
    """Dixon-Coles low-score correlation adjustment."""
    t = np.ones_like(lam, dtype=float)
    t = np.where((hg == 0) & (ag == 0), 1 - lam * mu * rho, t)
    t = np.where((hg == 0) & (ag == 1), 1 + lam * rho, t)
    t = np.where((hg == 1) & (ag == 0), 1 + mu * rho, t)
    t = np.where((hg == 1) & (ag == 1), 1 - rho, t)
    return t
