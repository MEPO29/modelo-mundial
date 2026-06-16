"""Dynamic hierarchical Bayesian goal model (Layer A backbone).

Per-team attack/defense strengths evolve as a Gaussian random walk over
half-year periods (recency is structural, not a decay hack). Team base
strengths are partially pooled toward confederation-level means — the
principled treatment of sparse intercontinental overlap: a team with few
cross-confederation matches is shrunk toward a prior that *is* informed by
cross-confederation play. Friendlies enter the likelihood at half weight.

Fit via SVI (AutoNormal, Adam) — minutes on CPU, which is what makes the
post-match sequential refit cheap during the tournament. The Dixon-Coles
low-score correlation rho is profiled post-hoc on posterior-mean rates and
applied to the predictive scoreline grid, mirroring the baseline.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from numpyro import deterministic, handlers, plate, sample
from numpyro import distributions as dist
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.initialization import init_to_median
from numpyro.optim import ClippedAdam
from scipy.optimize import minimize_scalar
from scipy.special import factorial

from mundial.ingest.confederations import CONFEDERATIONS, team_confederations
from mundial.models.baseline import MAX_GOALS, _tau

PERIOD_DAYS = 182
# Friendlies carry real strength signal; the previous 0.5 under-counted them.
# A 5-tournament walk-forward sweep favours 0.7 (improves 4/5 tournaments and
# both RPS and log-loss when paired with the faster random walk below).
FRIENDLY_WEIGHT = 0.7

# Prior scales — module-level so they can be tuned/ablated without touching the
# model body. Defaults are the validated production values.
# sigma_rw 0.2 (was 0.1) lets team strengths drift a little faster between
# half-year periods; the walk-forward sweep prefers it (marginal but robust:
# overall RPS 0.1949 -> 0.1947, log-loss 0.9659 -> 0.9651, better on 4/5
# reference tournaments; 90% bootstrap CI on the delta still spans 0, so this
# is a gentle refinement of an already near-optimal backbone, not a step change).
SIGMA_RW_PRIOR = 0.2     # HalfNormal scale on the per-period random-walk step
SIGMA_TEAM_PRIOR = 0.5   # HalfNormal scale on team base-strength dispersion
HOME_ADV_LOC = 0.25      # Normal mean of the home-advantage log-rate term
HOME_ADV_SCALE = 0.2     # Normal sd of the home-advantage term


def _model(home, away, period, home_flag, weight, n_teams, n_periods, conf_idx,
           n_confs, hg=None, ag=None):
    mu_conf_atk = sample("mu_conf_atk", dist.Normal(0.0, 0.5).expand([n_confs]).to_event(1))
    mu_conf_dfn = sample("mu_conf_dfn", dist.Normal(0.0, 0.5).expand([n_confs]).to_event(1))
    sigma_team = sample("sigma_team", dist.HalfNormal(SIGMA_TEAM_PRIOR))
    sigma_rw = sample("sigma_rw", dist.HalfNormal(SIGMA_RW_PRIOR))

    with plate("teams", n_teams):
        z_atk0 = sample("z_atk0", dist.Normal(0.0, 1.0))
        z_dfn0 = sample("z_dfn0", dist.Normal(0.0, 1.0))
    atk_base = mu_conf_atk[conf_idx] + sigma_team * z_atk0
    dfn_base = mu_conf_dfn[conf_idx] + sigma_team * z_dfn0

    z_rw_atk = sample("z_rw_atk", dist.Normal(0.0, 1.0).expand([n_teams, n_periods]).to_event(2))
    z_rw_dfn = sample("z_rw_dfn", dist.Normal(0.0, 1.0).expand([n_teams, n_periods]).to_event(2))
    atk = deterministic("atk", atk_base[:, None] + sigma_rw * jnp.cumsum(z_rw_atk, axis=1))
    dfn = deterministic("dfn", dfn_base[:, None] + sigma_rw * jnp.cumsum(z_rw_dfn, axis=1))

    intercept = sample("intercept", dist.Normal(0.0, 1.0))
    home_adv = sample("home_adv", dist.Normal(HOME_ADV_LOC, HOME_ADV_SCALE))

    log_lam = intercept + atk[home, period] - dfn[away, period] + home_adv * home_flag
    log_mu = intercept + atk[away, period] - dfn[home, period]

    if hg is not None:
        with handlers.scale(scale=weight):
            sample("hg", dist.Poisson(jnp.exp(log_lam)), obs=hg)
            sample("ag", dist.Poisson(jnp.exp(log_mu)), obs=ag)


@dataclass
class DynamicHierarchicalPoisson:
    since: dt.date = dt.date(2010, 1, 1)
    svi_steps: int = 4000
    learning_rate: float = 0.01
    num_samples: int = 400
    seed: int = 0
    rho: float = -0.05
    teams: list[str] = field(default_factory=list)
    _index: dict[str, int] = field(default_factory=dict)
    _conf_of: dict[str, str] = field(default_factory=dict)
    _samples: dict = field(default_factory=dict)
    _last_period: int = 0

    def fit(self, matches: pl.DataFrame, as_of: dt.date) -> DynamicHierarchicalPoisson:
        df = matches.filter((pl.col("date") >= self.since) & (pl.col("date") < as_of))
        self._conf_of = team_confederations(matches.filter(pl.col("date") < as_of))

        self.teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        self._index = {t: i for i, t in enumerate(self.teams)}
        conf_idx = np.array(
            [CONFEDERATIONS.index(self._conf_of.get(t, "OTHER")) for t in self.teams]
        )
        n_teams, n_confs = len(self.teams), len(CONFEDERATIONS)

        period = np.array([(d - self.since).days // PERIOD_DAYS for d in df["date"]])
        n_periods = int(period.max()) + 1
        self._last_period = n_periods - 1

        args = dict(
            home=jnp.array([self._index[t] for t in df["home_team"]]),
            away=jnp.array([self._index[t] for t in df["away_team"]]),
            period=jnp.array(period),
            home_flag=jnp.array(1.0 - df["neutral"].to_numpy().astype(float)),
            weight=jnp.array(
                np.where(df["tournament"].to_numpy() == "Friendly", FRIENDLY_WEIGHT, 1.0)
            ),
            n_teams=n_teams,
            n_periods=n_periods,
            conf_idx=jnp.array(conf_idx),
            n_confs=n_confs,
        )
        hg = jnp.array(np.minimum(df["home_score"].to_numpy(), MAX_GOALS))
        ag = jnp.array(np.minimum(df["away_score"].to_numpy(), MAX_GOALS))

        # init_to_uniform (the default) would start sigma_rw near exp(2) and the
        # period-cumsum amplifies that into +-50 log-rates SVI never recovers from
        guide = AutoNormal(_model, init_loc_fn=init_to_median(num_samples=15))
        svi = SVI(_model, guide, ClippedAdam(self.learning_rate, clip_norm=10.0), Trace_ELBO())
        result = svi.run(
            jax.random.PRNGKey(self.seed), self.svi_steps, **args, hg=hg, ag=ag,
            progress_bar=False,
        )

        predictive = Predictive(
            _model, guide=guide, params=result.params, num_samples=self.num_samples,
            return_sites=["atk", "dfn", "intercept", "home_adv", "mu_conf_atk", "mu_conf_dfn"],
        )
        raw = predictive(jax.random.PRNGKey(self.seed + 1), **args)
        self._samples = {k: np.asarray(v) for k, v in raw.items()}

        self._profile_rho(df, hg=np.asarray(hg), ag=np.asarray(ag), args=args)
        return self

    def _profile_rho(self, df, hg, ag, args):
        s = self._samples
        atk = s["atk"].mean(0)
        dfn = s["dfn"].mean(0)
        icpt = s["intercept"].mean()
        ha = s["home_adv"].mean()
        hi, ai, per = np.asarray(args["home"]), np.asarray(args["away"]), np.asarray(args["period"])
        hf = np.asarray(args["home_flag"])
        lam = np.exp(icpt + atk[hi, per] - dfn[ai, per] + ha * hf)
        mu = np.exp(icpt + atk[ai, per] - dfn[hi, per])

        def nll(rho):
            return -np.sum(np.log(np.clip(_tau(hg, ag, lam, mu, rho), 1e-10, None)))

        self.rho = minimize_scalar(nll, bounds=(-0.5, 0.5), method="bounded").x

    def _team_strengths(self, team: str) -> tuple[np.ndarray, np.ndarray]:
        """Posterior samples of (attack, defense) at the latest period."""
        s = self._samples
        if team in self._index:
            i = self._index[team]
            return s["atk"][:, i, self._last_period], s["dfn"][:, i, self._last_period]
        c = CONFEDERATIONS.index(self._conf_of.get(team, "OTHER"))
        return s["mu_conf_atk"][:, c], s["mu_conf_dfn"][:, c]

    def _period_of(self, date: dt.date) -> int:
        """Random-walk period index for a date, clamped to the fitted range."""
        p = (date - self.since).days // PERIOD_DAYS
        return max(0, min(int(p), self._last_period))

    def strength_means(
        self, team: str, date: dt.date | None = None
    ) -> tuple[float, float]:
        """Posterior-MEAN (attack, defense) for a team at ``date``'s period.

        The time-appropriate strength used to stack the backbone into the GBM:
        a row dated in period p gets the strength at period p (not the latest),
        so a historical training row never carries a later era's strength.
        Unseen teams fall back to their confederation mean; ``date=None`` (or a
        future date) yields the latest period — the value ``predict_1x2`` uses.
        """
        s = self._samples
        if team in self._index:
            i = self._index[team]
            p = self._last_period if date is None else self._period_of(date)
            return float(s["atk"][:, i, p].mean()), float(s["dfn"][:, i, p].mean())
        c = CONFEDERATIONS.index(self._conf_of.get(team, "OTHER"))
        return float(s["mu_conf_atk"][:, c].mean()), float(s["mu_conf_dfn"][:, c].mean())

    def score_matrix(self, home: str, away: str, neutral: bool = True) -> np.ndarray:
        """Posterior-predictive P(h, a) grid, averaged over strength samples."""
        s = self._samples
        atk_h, dfn_h = self._team_strengths(home)
        atk_a, dfn_a = self._team_strengths(away)
        ha = 0.0 if neutral else s["home_adv"]
        lam = np.exp(s["intercept"] + atk_h - dfn_a + ha)  # (S,)
        mu = np.exp(s["intercept"] + atk_a - dfn_h)

        g = np.arange(MAX_GOALS + 1)
        fact = factorial(g, exact=False)
        ph = np.exp(-lam[:, None]) * lam[:, None] ** g / fact  # (S, G)
        pa = np.exp(-mu[:, None]) * mu[:, None] ** g / fact
        grids = np.einsum("sh,sa->sha", ph, pa)
        for h in (0, 1):
            for a in (0, 1):
                grids[:, h, a] *= _tau(
                    np.full_like(lam, h), np.full_like(lam, a), lam, mu, self.rho
                )
        m = grids.mean(0)
        return m / m.sum()

    def predict_1x2(self, home: str, away: str, neutral: bool = True) -> np.ndarray:
        """[P(home win), P(draw), P(away win)] — outcome order 0/1/2 as in eval.metrics."""
        m = self.score_matrix(home, away, neutral)
        return np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])
