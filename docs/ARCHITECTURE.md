# World Cup 2026 Match-Outcome Prediction System — Architecture & Execution Blueprint

## Context

Greenfield project in `/Users/mepo/Documents/Personal/modelo_mundial` (empty directory). Goal: a state-of-the-art system producing **maximally calibrated probabilities** for match outcomes (1X2 + scoreline distributions) and tournament progression for the 2026 World Cup — which kicked off **today (June 11, 2026)**. The roadmap is therefore a **live-tournament sprint**: a usable model within ~2 days, the full online-learning loop within ~4, hardening continuously. Betting odds are a *feature and benchmark*, not the objective; primary metric is **Ranked Probability Score (RPS)**, with Brier, log-loss, and calibration error (ECE) secondary. Success = beating a Dixon-Coles baseline and approaching (ideally matching) the de-vigged closing line on RPS across the 104 matches.

Explicitly *not* a basic Elo tracker. The architecture below is a four-layer stack: a hierarchical Bayesian goal model as backbone, a gradient-boosted contextual layer, a market-information layer, and a Monte Carlo tournament simulator — glued together by an online ensemble whose weights self-correct after every match.

---

## 1. Data Sources

### 1.1 Core (table stakes)
- **All international match results since 1993** (~47k matches; the classic `results.csv` international dataset) — trains the Bayesian backbone with time decay.
- **World Cup qualifiers + warm-up friendlies 2024–2026** — highest-recency signal, weighted up.
- **Elo ratings (eloratings.net)** — *only* as a prior/feature, not the model.
- **Shot/xG data** for recent internationals and players' club seasons (FBref/Opta-derived aggregates; FBref is scrapeable, StatsBomb free data for historical WCs) — xG is the single biggest variance-reducer over raw scores.
- **Squad lists + predicted starting XIs** (official FIFA squads, lineup-news scraping).

### 1.2 Unconventional / high-leverage sources (the 2026 edition rewards these unusually well)

**Physical & physiological**
- **Altitude**: Mexico City (2,240 m), Guadalajara (1,566 m), Monterrey (540 m) vs sea-level venues. Feature = venue altitude × each squad's altitude-acclimatization proxy (home-league altitudes, days since arrival). Literature shows ~0.4 goal swing for unacclimatized teams above 2,000 m.
- **Heat stress**: kickoff-hour **wet-bulb globe temperature forecasts** (Open-Meteo API) for open-air venues (Miami, Houston-roof-open scenarios, Kansas City, Philadelphia, Monterrey) vs climate-controlled roofs (Atlanta, Dallas, Houston, Vancouver). Interacts with a team's pressing intensity (high-press teams degrade more in heat).
- **Travel & circadian load**: cumulative great-circle km between a team's venue sequence, time zones crossed (the 2026 footprint spans 4 zones), and **kickoff time relative to squad's home/club time zone** (circadian performance dips for 9pm-body-clock kickoffs).
- **Rest differential**: days since last match, minus opponent's — one of the few in-tournament features with consistently measurable effect.
- **Season fatigue**: minutes-weighted squad workload over the 2025–26 club season (FBref minutes). European players arrive off 50+ match seasons; MLS/Liga MX players arrive mid-season match-sharp. Encode per-player minutes percentile and league-calendar phase.
- **Yellow-card suspension tracking**: accumulation rules per FIFA regs — deterministic, free signal most models ignore until it bites.

**Psychological & sociological**
- **Squad cohesion matrix**: pairwise shared club minutes + shared international caps among the predicted XI → network density score. Cohesion is one of the strongest non-obvious predictors in international football (national teams train together ~2 weeks/year).
- **Coach features**: tenure (months), matches in charge, prior tournament knockout experience, observed substitution aggressiveness when trailing.
- **Camp-turmoil detector**: LLM classification over daily news headlines per team (bonus disputes, federation chaos, star-player rifts — the Cameroon/Spain-2022 pattern). Output: a 0–3 turmoil ordinal, decayed over 7 days. Cheap to run with Claude Haiku 4.5 over an RSS pull.
- **Expectation pressure**: gap between market-implied tournament odds and underlying squad quality — over-expected teams underperform; quasi-host pressure for USA/Mexico/Canada.

**Environmental & contextual**
- **De facto home crowd**: diaspora population near each venue (US Census ancestry data) + Google Trends regional interest → crowd-composition proxy. Mexico in LA or Houston ≈ home match; this materially shifts the home-advantage term, which must be *venue×team-specific* in 2026, not binary.
- **Pitch**: 2026's NFL stadiums use temporary natural grass laid over turf — narrower pitches and inconsistent surfaces penalize possession/passing teams; encode pitch dimensions + surface-type flag × team passing-volume profile.
- **Referee assignment** (published ~48h ahead): historical cards/penalties-per-match tendencies, confederation of origin vs the two teams.

**Market & wisdom-of-crowds**
- **Closing odds** from a sharp book / exchange (The Odds API; Betfair exchange if accessible), de-vigged with **Shin's method** (handles favorite–longshot bias better than proportional normalization). Used as: (a) ensemble component, (b) the benchmark to approach.
- **Prediction markets** (Polymarket/Kalshi tournament-winner contracts) — aggregates information (injuries, camp news) faster than feature pipelines can.
- **Transfermarkt squad value**, age-curve-adjusted and log-transformed — the best single cross-confederation strength proxy, which directly attacks the sparse-intercontinental-overlap problem.

**Player-level (solves national-team data sparsity via club transfer learning)**
- Minutes-weighted aggregation of club-season per-90 metrics (xG, xA, progressive passes, defensive actions) over the predicted XI, with **league-strength adjustment** (UEFA coefficients + inter-league transfer network).
- Squad age structure vs the ~26.5-year peak curve; count of key players >32 (tournament-specific decline is steep in heat).
- Goalkeeper **post-shot xG ± goals** (club season) — goalkeeper quality is underweighted by team-level models and decisive in knockouts/shootouts.

### 1.3 Acquisition notes
All sources are public/scrapeable or free-tier APIs. Store raw pulls immutably (parquet, `data/raw/<source>/<date>/`), transform with DuckDB/Polars. One scraper module per source with a common interface; degradation-tolerant (any unconventional source failing must not block the daily run — features default to training-set means with a missingness indicator).

---

## 2. Modeling Strategy — a four-layer stack

### Layer A — Hierarchical Bayesian goal model (the backbone)
**Bivariate Poisson with Dixon-Coles low-score correction, dynamic team strengths, hierarchical pooling by confederation.** Implemented in **NumPyro** (JAX; NUTS for full fits, SVI for fast in-tournament refits — runs in minutes on an M-series Mac).

- Each team has latent attack `α_t` and defense `δ_t` evolving as a **Gaussian random walk over match index** (Glickman-style dynamics) → recency is structural, not a hack.
- **Partial pooling**: team strengths drawn from confederation-level priors. This is the principled answer to *sparse intercontinental overlap* — Uzbekistan or Curaçao with few cross-confederation matches get shrunk toward an AFC/CONCACAF prior that *is* informed by cross-confederation play, instead of having garbage independent estimates. Strength priors additionally regress onto log squad value, so the market proxy anchors teams the match graph barely connects.
- **Overdispersion**: Negative Binomial (or COM-Poisson) goal margins rather than pure Poisson — fat tails are how the model respects upsets instead of being shocked by them. Dixon-Coles `τ` handles draw inflation.
- **xG-augmented likelihood**: where shot data exists, the likelihood mixes actual goals with xG ("deserved goals", ~60/40 weight tuned by backtest) — dramatically less noisy than scorelines alone.
- Covariates entering the log-rate: venue-specific home/crowd advantage, altitude×acclimatization, rest differential, heat×pressing-style.

### Layer B — Gradient-boosted contextual model
**LightGBM** (multiclass 1X2 + twin Poisson-rate regressors), consuming the full unconventional feature vector from §1.2 plus Layer A's posterior means as features. Captures interactions the parametric model can't (heat×press, pitch×passing-style, turmoil×expectation). Trained on all internationals 2000–present with tournament matches upweighted; monotonic constraints on strength features to prevent overfitting nonsense.

### Layer C — Market layer
Shin-de-vigged closing probabilities as a standalone "model". Including it honestly does two things: gives the ensemble an information-rich component, and makes the monitoring question precise — *when does our model add anything beyond the market?* (Answer from literature: most at long horizons and for teams markets price lazily; the ensemble weights will discover this empirically.)

### Layer D — Online ensemble + calibration + simulation
- **Stacking**: multinomial logistic meta-learner over Layers A/B/C probabilities, fit by walk-forward CV on 2014/2018/2022 WCs + Euros 2020/2024 + Copa América (never random K-fold — temporal leakage is the #1 failure mode here).
- **Online weight updates**: during the tournament, component weights follow an **exponentially weighted average forecaster (Hedge / multiplicative weights)** on per-match log-score, learning rate η tuned on historical tournaments. This is the formally-grounded "self-correction": whichever layer is actually predicting 2026 well accrues weight, with regret bounds rather than vibes.
- **Calibration**: temperature scaling / isotonic on held-out tournament matches, refreshed on a rolling window in-tournament. Report reliability diagrams every cycle.
- **Tournament simulator**: 100k Monte Carlo runs of the full 104-match bracket (48-team format: 12 groups of 4, 8 best third-places, round of 32). Critically, each run **samples team strengths from the Bayesian posterior** — propagating epistemic uncertainty into bracket probabilities instead of simulating from point estimates (the standard mistake that makes favorites look too safe). Knockout extra-time modeled with reduced-rate Poisson; **shootouts as ~52/48 coin flips adjusted only by goalkeeper post-shot-xG quality** (almost everything else in shootout "history" is noise — refuse the temptation).

### Why this stack
- Bayesian backbone → principled uncertainty, sparse-data pooling, structural recency. Pure ML can't do the first two; pure Elo can't do any.
- GBM layer → eats the creative features; the Bayesian model would need a prior for each.
- Deep learning (transformers over match sequences) is *rejected* deliberately: international football is a small-data regime (~100 meaningful matches/team-decade); deep nets here overfit or just rediscover Elo expensively. The deep-learning budget goes to the LLM news-ingestion pipeline instead, where it actually pays.

---

## 3. Online Learning Loop (after every match)

Automated post-match cycle (cron/launchd ~3h after each match window, when xG data lands):

1. **Ingest**: result, shots/xG, cards (update suspension ledger), minutes (update fatigue), injuries from news scrape.
2. **State update**: SVI refit of Layer A with the new match in the likelihood (the random-walk prior makes this a genuine sequential posterior update; full NUTS refit nightly). GBM features refresh; LightGBM itself is *not* refit intra-tournament (32 matches of signal would just add variance) — its inputs and its ensemble weight move instead.
3. **Self-correction**: Hedge update of ensemble weights from realized log-scores; rolling recalibration layer refresh.
4. **Context refresh for upcoming matches**: travel legs, rest differential, weather forecast pull, referee assignments, turmoil scores, lineup predictions.
5. **Re-simulate**: 100k bracket runs → updated advancement/title probabilities.
6. **Publish**: a Quarto/markdown report per cycle — next-match probability table, biggest probability movers, calibration plot, model-vs-market RPS tracker.
7. **Monitor**: CUSUM on cumulative log-score gap vs market benchmark; alert (and freeze ensemble drift toward any degrading component) if the model statistically underperforms.

---

## 4. Execution Roadmap (sprint phasing)

**Phase 0 — Scaffold (hours)**
Repo init, `uv` project, layout:
```
modelo_mundial/
  data/{raw,interim,features}/   src/ingest/  src/features/
  src/models/{bayes.py,gbm.py,market.py,ensemble.py,simulate.py}
  src/online/update_cycle.py     src/eval/backtest.py
  notebooks/   reports/   tests/   Makefile
```
Stack: Python 3.12, NumPyro+JAX, LightGBM, Polars+DuckDB, scikit-learn (calibration), Quarto, pytest.

**Phase 1 — Data backbone + baseline (Day 0–1)**
Historical internationals, Elo, squad values, fixtures/venues table (with altitude/roof/surface/timezone columns — hand-curated once, 16 venues), odds API hookup. Fit static Dixon-Coles → **this is the baseline every later layer must beat on backtest, and a usable model exists by end of Day 1.**

**Phase 2 — Backbone + backtest harness (Day 1–2)**
Dynamic hierarchical NumPyro model; walk-forward backtest over 2014/2018/2022 WCs + Euros + Copas; RPS/log-loss/ECE vs baseline and vs market. *No layer ships without beating baseline here.*

**Phase 3 — Feature layers + ensemble (Day 2–3)**
Unconventional feature pipelines (travel/heat/altitude/rest/cohesion/fatigue first — highest effect-size-per-effort; turmoil-LLM and crowd-composition next), LightGBM layer, Shin de-vig, stacked + calibrated ensemble, tournament simulator.

**Phase 4 — Online loop + automation (Day 3–4)**
`update_cycle.py` end-to-end, scheduled; Hedge weights; monitoring + report generation. From here the system runs itself through July 19.

**Phase 5 — Continuous hardening (rest of tournament)**
Knockout-specific recalibration before the round of 32, shootout module, ablation reports on which clever features actually carried weight (the post-tournament writeup writes itself from the Hedge weight trajectories).

---

## 5. Verification

- **Backtest gate**: each layer must improve walk-forward RPS on 2014–2024 tournaments over the Phase-1 baseline; ensemble must land within ~0.005 RPS of the de-vigged market on backtest to be credible.
- **Calibration gate**: ECE < 0.05 on held-out tournaments; reliability diagrams in every report.
- **Pipeline tests**: pytest on feature builders (no-leakage assertions: every feature timestamped strictly pre-kickoff), suspension-ledger logic, simulator bracket rules (48-team format edge cases: third-place ranking tiebreakers).
- **Live tracking**: cumulative RPS/log-score vs market and vs baseline, updated every cycle — the honest scoreboard.

## Approval & first action

On approval I will execute Phase 0 + start Phase 1: scaffold the repo, materialize this document as `docs/ARCHITECTURE.md`, build the venue table and the historical-results ingestion, and fit the Dixon-Coles baseline.
