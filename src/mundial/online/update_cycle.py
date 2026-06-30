"""The post-match online learning cycle (Phase 4).

Run after each match window (scheduled daily). Steps:

1. Pull fresh results.
2. Score previously LOGGED predictions against newly played matches and
   apply the Hedge multiplicative-weights update to the pool. Only
   predictions logged before kickoff are ever scored — matches that were
   played before we logged a forecast are skipped, never back-filled.
3. Refit the component models on the updated data; fetch fresh odds.
4. Log pooled predictions for upcoming matches (the next cycle scores these).
5. Re-run the tournament simulation.
6. Write a cycle report: new results, weight trajectory, upcoming matches,
   cumulative model-vs-market scoreboard with a CUSUM drift alarm.

State lives in artifacts/online_state.json; the prediction log (the honest,
append-only record) in artifacts/predictions_log.parquet.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl

from mundial.eval.metrics import correct_score_hit_rate, score_log_loss, score_rps
from mundial.ingest.results import download_results, load_fixtures, load_results
from mundial.models import ensemble as ens
from mundial.models.baseline import DixonColes
from mundial.models.bayes import DynamicHierarchicalPoisson
from mundial.models.market import fetch_odds
from mundial.models.simulate import TournamentSimulator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = PROJECT_ROOT / "artifacts"
REPORTS = PROJECT_ROOT / "reports"
STATE_PATH = ARTIFACTS / "online_state.json"
LOG_PATH = ARTIFACTS / "predictions_log.parquet"

COMPONENTS = ["dc", "bayes", "market"]
# Leave-one-tournament-out on WC 14/18/22 closing odds (eval/backtest market
# analysis) puts the bayes:market split near 0.31:0.69 and improves held-out
# RPS. The prior reflects that market-led evidence but shrinks off the 0.69
# point estimate — only 3 folds, and the live feed (The Odds API median) is
# noisier than the backtested closing line. dc keeps a live slot and must
# earn weight through the Hedge updates.
#
# GBM was RETIRED from the live pool at V2: it was worst on every live metric
# (RPS 0.247) and carried ~0 backtest stacking weight, yet the Hedge weight
# floor kept it skimming mass each match. Its 0.05 prior is reallocated to
# bayes (which now carries the bivariate-Poisson scoreline upgrade). GBM
# remains a backtest reference component (eval/backtest.py) so the retirement
# stays auditable and reversible.
INITIAL_WEIGHTS = {"dc": 0.05, "bayes": 0.40, "market": 0.55}
HORIZON_DAYS = 5
CUSUM_THRESHOLD = 3.0
N_SIMS = 100_000

# WC 2026 group stage runs Jun 11–27; the Round of 32 opens Jun 28. From this
# date a "draw" is no longer a final result — the tie continues into extra time
# and (if still level) penalties, so knockout fixtures carry the full
# FT → ET → penalties resolution in the forecast and the digest.
KNOCKOUT_START = dt.date(2026, 6, 28)


def load_env() -> None:
    env = PROJECT_ROOT / ".env"
    if env.exists():
        import os

        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _validate_weights(weights: dict) -> dict:
    """Coerce a persisted weight vector onto the current COMPONENTS.

    Guards the failure mode that left the live loop inert: a stale
    online_state.json shadowing the validated INITIAL_WEIGHTS (e.g. an old
    4-component vector with a swapped market prior, or a now-retired
    component). Unknown keys are dropped, missing keys seeded from
    INITIAL_WEIGHTS, and the result renormalized to sum 1 — so the file's
    weights can never silently diverge from the current component set again.
    """
    clean = {c: float(weights.get(c, INITIAL_WEIGHTS[c])) for c in COMPONENTS}
    extra = sorted(set(weights) - set(COMPONENTS))
    missing = [c for c in COMPONENTS if c not in weights]
    total = sum(clean.values())
    if extra or missing or abs(total - 1.0) > 1e-6:
        notes = []
        if extra:
            notes.append(f"dropped retired component(s) {extra}")
        if missing:
            notes.append(f"seeded missing {missing} from INITIAL_WEIGHTS")
        if abs(total - 1.0) > 1e-6:
            notes.append(f"renormalized (sum was {total:.3f})")
        print(f"load_state: {'; '.join(notes)}")
    if total <= 0:
        return dict(INITIAL_WEIGHTS)
    return {c: w / total for c, w in clean.items()}


def load_state() -> dict:
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        state["weights"] = _validate_weights(state.get("weights", {}))
        return state
    return {
        "weights": dict(INITIAL_WEIGHTS),
        "scored": [],
        "score_history": [],
        "cusum": 0.0,
    }


def save_state(state: dict) -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def match_key(date, home, away) -> str:
    return f"{date}|{home}|{away}"


def outcome_of(hs: int, as_: int) -> int:
    return 0 if hs > as_ else 1 if hs == as_ else 2


def _score_grid(flat, hs: int, as_: int) -> dict | None:
    """Scoreline metrics for one logged grid against the actual result.

    Returns None for legacy rows that never logged a grid — honest scoring:
    we never fabricate a scoreline distribution we didn't commit pre-kickoff.
    The grid is logged row-major (G*G,); G is recovered from its length.
    """
    if flat is None:
        return None
    arr = np.asarray(flat, dtype=float)
    if arr.size == 0:
        return None
    side = int(round(arr.size**0.5))
    grid = arr.reshape(side, side)
    hg, ag = np.array([hs]), np.array([as_])
    return {
        "score_log_loss": float(score_log_loss(grid, hg, ag)),
        "score_rps": float(score_rps(grid, hg, ag)),
        "hit": bool(correct_score_hit_rate(grid, hg, ag) == 1.0),
    }


def score_new_results(state: dict, results: pl.DataFrame) -> list[dict]:
    """Hedge-update pool weights from newly played, previously-logged matches."""
    if not LOG_PATH.exists():
        return []
    log = pl.read_parquet(LOG_PATH)
    played = {
        match_key(r["date"], r["home_team"], r["away_team"]): (r["home_score"], r["away_score"])
        for r in results.filter(
            (pl.col("tournament") == "FIFA World Cup")
            & (pl.col("date") >= dt.date(2026, 6, 11))
        ).iter_rows(named=True)
    }

    scored_rows = []
    weights = np.array([state["weights"][c] for c in COMPONENTS])
    for r in log.iter_rows(named=True):
        key = match_key(r["date"], r["home_team"], r["away_team"])
        if key in state["scored"] or key not in played:
            continue
        hs, as_ = played[key]
        y = outcome_of(hs, as_)
        losses = np.full(len(COMPONENTS), np.nan)
        for i, c in enumerate(COMPONENTS):
            p = r.get(f"{c}_{['h', 'd', 'a'][y]}")
            if p is not None:
                losses[i] = -np.log(max(p, 1e-12))
        # components without a logged forecast get the pool's own loss (no
        # free ride, no punishment)
        pool_loss = -np.log(max(r[f"pool_{['h', 'd', 'a'][y]}"], 1e-12))
        losses = np.where(np.isnan(losses), pool_loss, losses)
        weights = ens.hedge_update(weights, losses)

        state["scored"].append(key)
        entry = {
            "key": key,
            "outcome": y,
            "score": f"{hs}-{as_}",
            "pool_loss": float(pool_loss),
            "losses": {c: float(l) for c, l in zip(COMPONENTS, losses)},
            "weights_after": {c: float(w) for c, w in zip(COMPONENTS, weights)},
        }
        # score the logged scoreline grid (diagnostic track, separate from the
        # 1X2 Hedge pool above); skipped for legacy rows with no grid
        scoreline = _score_grid(r.get("bayes_grid"), hs, as_)
        if scoreline is not None:
            entry["scoreline"] = scoreline
        state["score_history"].append(entry)
        scored_rows.append({"key": key, "y": y, "hs": hs, "as": as_, "pool_loss": pool_loss})

    state["weights"] = {c: float(w) for c, w in zip(COMPONENTS, weights)}
    return scored_rows


WC_START = dt.date(2026, 6, 11)
STALE_DAYS = 2  # newest WC result older than this during the tournament = suspect


def freshness_warning(results: pl.DataFrame, state: dict, today: dt.date) -> str | None:
    """Warn when the loop looks stalled: played WC matches that were pre-logged
    but never scored, or results that have gone stale mid-tournament.

    Returns a human-readable warning string, or None when healthy. This turns
    the silent failure mode (a cycle that runs but scores nothing) into a loud
    signal in the cycle report and the morning digest.
    """
    wc = results.filter(
        (pl.col("tournament") == "FIFA World Cup")
        & (pl.col("date") >= WC_START)
        & (pl.col("date") <= today)
    )
    if wc.height == 0:
        return None  # tournament has no played matches yet — nothing to score

    msgs = []
    if LOG_PATH.exists():
        log = pl.read_parquet(LOG_PATH)
        played_keys = {
            match_key(r["date"], r["home_team"], r["away_team"])
            for r in wc.iter_rows(named=True)
        }
        logged_keys = {
            match_key(r["date"], r["home_team"], r["away_team"])
            for r in log.iter_rows(named=True)
        }
        unscored = (played_keys & logged_keys) - set(state.get("scored", []))
        if unscored:
            msgs.append(
                f"{len(unscored)} played, pre-logged match(es) still unscored - "
                "the loop may be stalled (check it is running daily)"
            )

    age = (today - wc["date"].max()).days
    if age > STALE_DAYS:
        msgs.append(f"newest WC result is {age}d old ({wc['date'].max()}) - results may be stale")
    return "; ".join(msgs) or None


def update_cusum(state: dict) -> float:
    """CUSUM of (pool log-loss - market log-loss); positive drift = model worse."""
    cusum = 0.0
    for h in state["score_history"]:
        market_loss = h["losses"].get("market")
        if market_loss is None:
            continue
        cusum = max(0.0, cusum + (h["pool_loss"] - market_loss) - 0.05)
    state["cusum"] = cusum
    return cusum


def predict_upcoming(
    results: pl.DataFrame, today: dt.date, horizon_days: int = HORIZON_DAYS
) -> tuple[pl.DataFrame, DynamicHierarchicalPoisson | None]:
    """Component + pooled predictions for WC fixtures in the horizon window."""
    fixtures = load_fixtures(tournament="FIFA World Cup").filter(
        (pl.col("date") >= today) & (pl.col("date") <= today + dt.timedelta(days=horizon_days))
    )
    if fixtures.height == 0:
        return pl.DataFrame(), None

    as_of = today + dt.timedelta(days=1)
    train = results.filter(pl.col("date") >= dt.date(1990, 1, 1))
    bayes = DynamicHierarchicalPoisson().fit(train, as_of=as_of)
    models = {
        "dc": DixonColes().fit(train, as_of=as_of),
        "bayes": bayes,
    }
    try:
        odds = fetch_odds()
    except Exception as e:  # odds are an enhancement, never a dependency
        print(f"odds fetch failed, continuing without market layer: {e}")
        odds = None
    market = {}
    if odds is not None:
        market = {
            (r["home_team"], r["away_team"]): np.array([r["p_h"], r["p_d"], r["p_a"]])
            for r in odds.iter_rows(named=True)
        }

    state = load_state()
    weights = np.array([state["weights"][c] for c in COMPONENTS])

    rows = []
    for r in fixtures.iter_rows(named=True):
        h, a, neu = r["home_team"], r["away_team"], r["neutral"]
        blocks: list[np.ndarray | None] = [
            models["dc"].predict_1x2(h, a, neutral=neu)[None, :],
            models["bayes"].predict_1x2(h, a, neutral=neu)[None, :],
            market[(h, a)][None, :] if (h, a) in market else None,
        ]
        pool = ens.pool_predict_partial(weights, blocks)[0]

        # scoreline distribution from the Bayesian layer (the only component
        # with a goal grid); expected goals from the grid mean, mode as the
        # headline "predicted score"
        grid = models["bayes"].score_matrix(h, a, neutral=neu)
        g = np.arange(grid.shape[0])
        xg_h = float((grid.sum(axis=1) * g).sum())
        xg_a = float((grid.sum(axis=0) * g).sum())
        mh, ma = np.unravel_index(grid.argmax(), grid.shape)

        row = {"date": str(r["date"]), "home_team": h, "away_team": a, "neutral": neu,
               "score_pred": f"{mh}-{ma}", "score_prob": float(grid[mh, ma]),
               "xg_h": xg_h, "xg_a": xg_a,
               # full scoreline distribution, row-major (G*G,), so the next
               # cycle can score the grid against the actual result. Logged
               # immutably alongside the 1X2 probs; legacy rows lack it.
               "bayes_grid": grid.reshape(-1).tolist()}
        for c, b in zip(COMPONENTS, blocks):
            for j, suf in enumerate(["h", "d", "a"]):
                row[f"{c}_{suf}"] = float(b[0, j]) if b is not None else None
        for j, suf in enumerate(["h", "d", "a"]):
            row[f"pool_{suf}"] = float(pool[j])

        # Knockout resolution. The headline 1X2 (pool, above) is the 90-minute
        # result; from the Round of 32 a 90' draw is not final. The FT split
        # uses the pool (so it carries the market layer); the conditional ET
        # distribution and the coin-flip shootout come from the bayes backbone.
        ko_cols = ["et_h", "et_d", "et_a", "pens_h", "pens_a",
                   "adv_h", "adv_a", "p_reach_et", "p_reach_pens"]
        if dt.date.fromisoformat(row["date"]) >= KNOCKOUT_START:
            kb = models["bayes"].knockout_breakdown(h, a, neutral=neu)
            et = kb["et"]
            pens_h = float(kb["pens"][0])
            # advancement chains the pool's FT split with the bayes ET split
            adv_h = pool[0] + pool[1] * (et[0] + et[1] * pens_h)
            adv_a = pool[2] + pool[1] * (et[2] + et[1] * (1.0 - pens_h))
            row.update({
                "is_knockout": True,
                "et_h": float(et[0]), "et_d": float(et[1]), "et_a": float(et[2]),
                "pens_h": pens_h, "pens_a": 1.0 - pens_h,
                "adv_h": float(adv_h), "adv_a": float(adv_a),
                "p_reach_et": float(pool[1]),
                "p_reach_pens": float(pool[1] * et[1]),
            })
        else:
            row["is_knockout"] = False
            for col in ko_cols:
                row[col] = None
        row["logged_at"] = dt.datetime.now().isoformat(timespec="seconds")
        rows.append(row)
    # Pin the knockout columns to Float64 so a group-stage-only frame (all
    # nulls) doesn't infer a Null dtype that clashes with a later knockout
    # frame on append.
    ko_schema = {c: pl.Float64 for c in
                 ("et_h", "et_d", "et_a", "pens_h", "pens_a",
                  "adv_h", "adv_a", "p_reach_et", "p_reach_pens")}
    return pl.DataFrame(rows, schema_overrides=ko_schema), models["bayes"]


def append_log(preds: pl.DataFrame) -> None:
    """Append predictions, keeping the EARLIEST logged forecast per match.

    Once a forecast for a match is logged it is frozen — later cycles must
    not overwrite it, or the Hedge scoring stops being honest.
    """
    if preds.height == 0:
        return
    if LOG_PATH.exists():
        old = pl.read_parquet(LOG_PATH)
        new = preds.join(
            old.select("date", "home_team", "away_team"),
            on=["date", "home_team", "away_team"],
            how="anti",
        )
        if new.height:
            # diagonal_relaxed supertypes column dtypes that differ across
            # frames (e.g. a group-only day's null knockout columns vs a
            # knockout day's Float64) instead of erroring.
            pl.concat([old, new], how="diagonal_relaxed").write_parquet(LOG_PATH)
    else:
        preds.write_parquet(LOG_PATH)


def scoreboard(state: dict) -> dict:
    hist = state["score_history"]
    if not hist:
        return {}
    n = len(hist)
    pool = np.mean([h["pool_loss"] for h in hist])
    comp_means = {
        c: float(np.mean([h["losses"][c] for h in hist])) for c in COMPONENTS
    }
    out = {"n_scored": n, "pool_logloss": float(pool),
           **{f"{c}_logloss": v for c, v in comp_means.items()}}
    sl = [h["scoreline"] for h in hist if "scoreline" in h]
    if sl:
        out["scoreline"] = {
            "n": len(sl),
            "score_log_loss": float(np.mean([s["score_log_loss"] for s in sl])),
            "score_rps": float(np.mean([s["score_rps"] for s in sl])),
            "hit_rate": float(np.mean([s["hit"] for s in sl])),
        }
    return out


def write_report(today, scored, state, preds, sim_table, cusum, warning=None) -> Path:
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"cycle_{today.isoformat()}.md"
    w = state["weights"]
    lines = [f"# Cycle report — {today}", ""]

    if warning:
        lines += [f"> ⚠️ **{warning}**", ""]

    if scored:
        lines += ["## Newly scored matches", ""]
        for s in scored:
            lines.append(f"- {s['key']}  **{s['hs']}-{s['as']}**  pool log-loss {s['pool_loss']:.3f}")
        lines.append("")
    lines += [
        "## Pool weights",
        "",
        " | ".join(f"{c}: {w[c]:.3f}" for c in COMPONENTS),
        "",
    ]
    sb = scoreboard(state)
    if sb:
        lines += [
            "## Scoreboard (cumulative log-loss, scored matches only)",
            "",
            f"- matches scored: {sb['n_scored']}",
            f"- pool: {sb['pool_logloss']:.4f}",
            *[f"- {c}: {sb[f'{c}_logloss']:.4f}" for c in COMPONENTS],
            f"- CUSUM vs market: {cusum:.2f}"
            + ("  ⚠️ ALARM — model drifting worse than market" if cusum > CUSUM_THRESHOLD else ""),
            "",
        ]
        if "scoreline" in sb:
            sl = sb["scoreline"]
            lines += [
                "## Scoreline accuracy (bayes grid, scored matches only)",
                "",
                f"- matches with logged grid: {sl['n']}",
                f"- scoreline log-loss: {sl['score_log_loss']:.4f}",
                f"- margin RPS: {sl['score_rps']:.4f}",
                f"- exact-score hit rate: {sl['hit_rate']:.1%}",
                "",
            ]
    if preds.height:
        lines += ["## Upcoming matches (pool forecast)", "",
                  "| date | match | P(H) | P(D) | P(A) | likely score | xG | market P(H/D/A) |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in preds.iter_rows(named=True):
            mk = (
                f"{r['market_h']:.0%}/{r['market_d']:.0%}/{r['market_a']:.0%}"
                if r["market_h"] is not None else "—"
            )
            lines.append(
                f"| {r['date']} | {r['home_team']} v {r['away_team']} | "
                f"{r['pool_h']:.1%} | {r['pool_d']:.1%} | {r['pool_a']:.1%} | "
                f"{r['score_pred']} ({r['score_prob']:.0%}) | "
                f"{r['xg_h']:.1f}–{r['xg_a']:.1f} | {mk} |"
            )
        lines.append("")
        ko = [r for r in preds.iter_rows(named=True) if r.get("is_knockout")]
        if ko:
            lines += [
                "## Knockout resolution",
                "",
                "1X2 above is the 90' result; below chains extra time (30' at "
                "1/3 rate) and a coin-flip shootout into who advances.",
                "",
                "| match | advance (H / A) | to ET | ET (H/D/A) | to pens |",
                "|---|---|---|---|---|",
            ]
            for r in ko:
                lines.append(
                    f"| {r['home_team']} v {r['away_team']} | "
                    f"{r['adv_h']:.1%} / {r['adv_a']:.1%} | "
                    f"{r['p_reach_et']:.1%} | "
                    f"{r['et_h']:.0%}/{r['et_d']:.0%}/{r['et_a']:.0%} | "
                    f"{r['p_reach_pens']:.1%} |"
                )
            lines.append("")
    if sim_table is not None:
        lines += ["## Title race (top 10)", "",
                  "| team | R16 | QF | SF | Final | Champion |", "|---|---|---|---|---|---|"]
        for r in sim_table.head(10).iter_rows(named=True):
            lines.append(
                f"| {r['team']} | {r['p_r16']:.1%} | {r['p_qf']:.1%} | {r['p_sf']:.1%} "
                f"| {r['p_final']:.1%} | {r['p_champion']:.1%} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def push_artifacts() -> None:
    """Best-effort: sync Hedge state and reports to the remote so cloud
    digest runs pool with current weights. Never fails the cycle.

    Set CYCLE_PUSH=0 to skip the git push entirely — used in CI, where a
    dedicated workflow step persists state to the `model-state` branch
    instead of committing to the working branch."""
    import os
    import subprocess

    if os.environ.get("CYCLE_PUSH", "1") == "0":
        print("artifact push skipped (CYCLE_PUSH=0)")
        return

    try:
        subprocess.run(["git", "add", "artifacts", "reports"], cwd=PROJECT_ROOT,
                       check=True, capture_output=True)
        r = subprocess.run(
            ["git", "commit", "-m", f"cycle: online state {dt.date.today()}"],
            cwd=PROJECT_ROOT, capture_output=True,
        )
        if r.returncode == 0:
            subprocess.run(["git", "push"], cwd=PROJECT_ROOT, check=True,
                           capture_output=True, timeout=120)
            print("artifacts pushed to remote")
    except Exception as e:  # no remote, offline, etc.
        print(f"artifact push skipped: {e}")


def main() -> None:
    load_env()
    today = dt.date.today()
    print(f"=== update cycle {today} ===")

    download_results()
    results = load_results()
    print(f"results through {results['date'].max()}")

    state = load_state()
    scored = score_new_results(state, results)
    cusum = update_cusum(state)
    print(f"scored {len(scored)} new matches; weights "
          + " ".join(f"{c}={state['weights'][c]:.3f}" for c in COMPONENTS))

    warning = freshness_warning(results, state, today)
    if warning:
        print(f"WARNING: {warning}")

    preds, bayes_model = predict_upcoming(results, today)
    append_log(preds)
    print(f"logged {preds.height} upcoming predictions")

    sim_table = None
    if bayes_model is not None:
        sim = TournamentSimulator(bayes_model, n_sims=N_SIMS)
        sim_table = sim.run()
        sim_table.write_csv(REPORTS / f"sim_{today.isoformat()}.csv")

    save_state(state)
    path = write_report(today, scored, state, preds, sim_table, cusum, warning)
    print(f"report: {path}")
    push_artifacts()


if __name__ == "__main__":
    main()
