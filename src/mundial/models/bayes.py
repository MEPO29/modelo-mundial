"""Dynamic hierarchical Bayesian goal model (Layer A backbone).

Per-team attack/defense strengths evolve as a Gaussian random walk over
half-year periods (recency is structural, not a decay hack). Team base
strengths are partially pooled toward confederation-level means — the
principled treatment of sparse intercontinental overlap: a team with few
cross-confederation matches is shrunk toward a prior that *is* informed by
cross-confederation play. Friendlies enter the likelihood at half weight.

Goals follow a SHARED-LATENT bivariate Poisson (Karlis & Ntzoufras): home =
X1 + X3, away = X2 + X3 with X3 a shared Poisson component. X3 induces the
positive goal correlation — and, crucially, the excess draw mass — that an
independent double-Poisson misses and the old Dixon-Coles post-hoc tau only
patched onto four low-score cells. The likelihood is the closed-form BP pmf,
marginalized over X3, so SVI stays as cheap as the independent model.

Fit via SVI (AutoNormal, Adam) — minutes on CPU, which is what makes the
post-match sequential refit cheap during the tournament.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from jax.scipy.special import gammaln as jgammaln
from jax.scipy.special import logsumexp as jlogsumexp
from numpyro import deterministic, factor, handlers, plate, sample
from numpyro import distributions as dist
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.initialization import init_to_median
from numpyro.optim import ClippedAdam
from scipy.special import gammaln, logsumexp

from mundial.ingest.confederations import CONFEDERATIONS, team_confederations
from mundial.models.baseline import MAX_GOALS

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
# Shared bivariate-Poisson component X3 ~ Poisson(exp(cov_intercept)). The
# prior centers lambda3 ~ exp(-2.0) ≈ 0.14 (a realistic shared goal rate) with
# a tight scale so it captures correlation/draw mass without absorbing the
# overall goal level (identifiability vs the intercept).
COV_LOC = -2.0
COV_SCALE = 0.5
# Coupling of a standardized external strength prior (FIFA/Elo rank, etc.)
PRIOR_OFFSET_SCALE = 0.3


def _bp_log_pmf(x, y, log_lam, log_mu, log_lam3, k):
    """Bivariate-Poisson log-pmf, marginalized over the shared latent X3.

    home = X1 + X3, away = X2 + X3 with rates exp(log_lam), exp(log_mu),
    exp(log_lam3). x, y, log_lam, log_mu are (n,); log_lam3 is scalar; k is a
    static (K,) arange covering 0..max(min(x,y)). The inner sum runs only to
    min(x,y) per observation — terms beyond that are masked to a finite large
    negative (not -inf) so the SVI gradient stays well-defined. Reduces exactly
    to independent Poisson as log_lam3 -> -inf.
    """
    lam, mu, lam3 = jnp.exp(log_lam), jnp.exp(log_mu), jnp.exp(log_lam3)
    c = log_lam3 - log_lam - log_mu              # (n,)
    kk = k[None, :]                              # (1, K)
    minxy = jnp.minimum(x, y)[:, None]           # (n, 1)
    valid = kk <= minxy
    xk = jnp.maximum(x[:, None] - kk + 1.0, 1.0)  # clamp so gammaln stays finite
    yk = jnp.maximum(y[:, None] - kk + 1.0, 1.0)
    g = -jgammaln(kk + 1.0) - jgammaln(xk) - jgammaln(yk) + kk * c[:, None]
    g = jnp.where(valid, g, -1e30)
    log_s = jlogsumexp(g, axis=1)                # (n,)
    return -(lam + mu + lam3) + x * log_lam + y * log_mu + log_s


def _bp_grid_np(log_lam, log_mu, log_lam3):
    """Bivariate-Poisson scoreline grid (numpy), one per posterior sample.

    log_lam/log_mu/log_lam3 are (S,). Returns P(h, a) of shape (S, G, G) with
    G = MAX_GOALS + 1 — the same closed-form pmf as ``_bp_log_pmf``, evaluated
    over the full goal grid for the predictive distribution.
    """
    g = MAX_GOALS + 1
    h = np.arange(g)
    k = np.arange(g)  # shared component up to min(h, a) <= MAX_GOALS
    lam, mu, lam3 = np.exp(log_lam), np.exp(log_mu), np.exp(log_lam3)
    c = (log_lam3 - log_lam - log_mu)[:, None, None, None]  # (S,1,1,1)
    H = h[None, :, None, None]
    A = h[None, None, :, None]
    K = k[None, None, None, :]
    valid = K <= np.minimum(H, A)
    xk = np.maximum(H - K + 1, 1)
    yk = np.maximum(A - K + 1, 1)
    gk = -gammaln(K + 1) - gammaln(xk) - gammaln(yk) + K * c
    gk = np.where(valid, gk, -np.inf)
    log_s = logsumexp(gk, axis=3)  # (S, G, G)
    base = -(lam + mu + lam3)[:, None, None]
    logp = (base + h[None, :, None] * log_lam[:, None, None]
            + h[None, None, :] * log_mu[:, None, None] + log_s)
    return np.exp(logp)


def _model(home, away, period, home_flag, weight, n_teams, n_periods, conf_idx,
           n_confs, k, prior_offset=None, hg=None, ag=None):
    mu_conf_atk = sample("mu_conf_atk", dist.Normal(0.0, 0.5).expand([n_confs]).to_event(1))
    mu_conf_dfn = sample("mu_conf_dfn", dist.Normal(0.0, 0.5).expand([n_confs]).to_event(1))
    sigma_team = sample("sigma_team", dist.HalfNormal(SIGMA_TEAM_PRIOR))
    sigma_rw = sample("sigma_rw", dist.HalfNormal(SIGMA_RW_PRIOR))

    with plate("teams", n_teams):
        z_atk0 = sample("z_atk0", dist.Normal(0.0, 1.0))
        z_dfn0 = sample("z_dfn0", dist.Normal(0.0, 1.0))
    atk_base = mu_conf_atk[conf_idx] + sigma_team * z_atk0
    dfn_base = mu_conf_dfn[conf_idx] + sigma_team * z_dfn0

    # Optional external strength prior (Workstream C): a standardized per-team
    # offset (ranking/Elo/value) shifts attack and defense base strength. Off
    # by default (prior_offset=None) so the backbone is unchanged without it.
    if prior_offset is not None:
        beta_atk = sample("beta_atk", dist.Normal(0.0, PRIOR_OFFSET_SCALE))
        beta_dfn = sample("beta_dfn", dist.Normal(0.0, PRIOR_OFFSET_SCALE))
        atk_base = atk_base + beta_atk * prior_offset
        dfn_base = dfn_base + beta_dfn * prior_offset

    z_rw_atk = sample("z_rw_atk", dist.Normal(0.0, 1.0).expand([n_teams, n_periods]).to_event(2))
    z_rw_dfn = sample("z_rw_dfn", dist.Normal(0.0, 1.0).expand([n_teams, n_periods]).to_event(2))
    atk = deterministic("atk", atk_base[:, None] + sigma_rw * jnp.cumsum(z_rw_atk, axis=1))
    dfn = deterministic("dfn", dfn_base[:, None] + sigma_rw * jnp.cumsum(z_rw_dfn, axis=1))

    intercept = sample("intercept", dist.Normal(0.0, 1.0))
    home_adv = sample("home_adv", dist.Normal(HOME_ADV_LOC, HOME_ADV_SCALE))
    cov_intercept = sample("cov_intercept", dist.Normal(COV_LOC, COV_SCALE))

    log_lam = intercept + atk[home, period] - dfn[away, period] + home_adv * home_flag
    log_mu = intercept + atk[away, period] - dfn[home, period]

    if hg is not None:
        log_pmf = _bp_log_pmf(hg, ag, log_lam, log_mu, cov_intercept, k)
        with handlers.scale(scale=weight):
            factor("obs", log_pmf)


@dataclass
class DynamicHierarchicalPoisson:
    since: dt.date = dt.date(2010, 1, 1)
    svi_steps: int = 4000
    learning_rate: float = 0.01
    num_samples: int = 400
    seed: int = 0
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
        hg_np = np.minimum(df["home_score"].to_numpy(), MAX_GOALS)
        ag_np = np.minimum(df["away_score"].to_numpy(), MAX_GOALS)
        hg = jnp.array(hg_np.astype(float))
        ag = jnp.array(ag_np.astype(float))
        # k arange for the bivariate-Poisson inner sum: 0..max over data of
        # min(home, away) goals (the shared component can be at most that).
        n_k = int(np.minimum(hg_np, ag_np).max()) + 1
        args["k"] = jnp.arange(n_k, dtype=float)

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
            return_sites=["atk", "dfn", "intercept", "home_adv", "cov_intercept",
                          "mu_conf_atk", "mu_conf_dfn"],
        )
        raw = predictive(jax.random.PRNGKey(self.seed + 1), **args)
        self._samples = {kk: np.asarray(v) for kk, v in raw.items()}
        return self

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

    def score_matrix(
        self, home: str, away: str, neutral: bool = True, shared: bool = True
    ) -> np.ndarray:
        """Posterior-predictive P(h, a) grid, averaged over strength samples.

        Uses the bivariate-Poisson pmf with the shared component, so the
        diagonal (draw) mass and low-score correlation are produced by the
        model rather than patched on afterward. ``shared=False`` zeroes the
        shared component (independent Poisson on the same fitted rates) — the
        ablation the backtest uses to isolate the structural change's value.
        """
        s = self._samples
        atk_h, dfn_h = self._team_strengths(home)
        atk_a, dfn_a = self._team_strengths(away)
        ha = 0.0 if neutral else s["home_adv"]
        log_lam = s["intercept"] + atk_h - dfn_a + ha  # (S,)
        log_mu = s["intercept"] + atk_a - dfn_h
        log_lam3 = s["cov_intercept"] if shared else np.full_like(s["intercept"], -50.0)
        grids = _bp_grid_np(log_lam, log_mu, log_lam3)  # (S, G, G)
        m = grids.mean(0)
        return m / m.sum()

    @staticmethod
    def _grid_1x2(m: np.ndarray) -> np.ndarray:
        """[P(home win), P(draw), P(away win)] from a scoreline grid."""
        return np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])

    def predict_1x2(self, home: str, away: str, neutral: bool = True) -> np.ndarray:
        """[P(home win), P(draw), P(away win)] — outcome order 0/1/2 as in eval.metrics."""
        return self._grid_1x2(self.score_matrix(home, away, neutral))

    def knockout_breakdown(
        self, home: str, away: str, neutral: bool = True
    ) -> dict[str, np.ndarray | float]:
        """Sequential knockout resolution for a single tie.

        A knockout match is decided in up to three stages: 90 minutes, then
        (if level) 30 minutes of extra time, then (if still level) a shootout.
        Returns each stage's outcome distribution PLUS the unconditional
        advancement probabilities:

        - ``ft``   : [P(H), P(draw), P(A)] over 90' (a draw sends the tie to ET)
        - ``et``   : [P(H), P(draw), P(A)] scored in the 30' of extra time,
                     CONDITIONAL on a level 90' (a draw sends it to penalties)
        - ``pens`` : [P(H), P(A)] shootout — a model-agnostic coin flip
        - ``advance`` : [P(home advances), P(away advances)], chaining the three
        - ``p_reach_et`` / ``p_reach_pens`` : P(the tie reaches that stage)

        ET uses ``ET_RATE_FACTOR`` of the 90' rates (the same approximation as
        the bracket simulator). Only the bivariate-Poisson backbone yields a
        scoreline distribution, so this is a bayes-layer product.
        """
        ft = self.predict_1x2(home, away, neutral)
        et = self._grid_1x2(
            self.score_matrix(home, away, neutral, rate_factor=ET_RATE_FACTOR)
        )
        pens = np.array([PENALTY_HOME_WIN, 1.0 - PENALTY_HOME_WIN])
        # win in 90'; else level then win the 30'; else level again then shootout
        adv_h = ft[0] + ft[1] * (et[0] + et[1] * pens[0])
        adv_a = ft[2] + ft[1] * (et[2] + et[1] * pens[1])
        return {
            "ft": ft,
            "et": et,
            "pens": pens,
            "advance": np.array([adv_h, adv_a]),
            "p_reach_et": float(ft[1]),
            "p_reach_pens": float(ft[1] * et[1]),
        }
