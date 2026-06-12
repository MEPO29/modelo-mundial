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

from mundial.eval.metrics import log_loss, rps
from mundial.ingest.results import download_results, load_fixtures, load_results
from mundial.models import ensemble as ens
from mundial.models.baseline import DixonColes
from mundial.models.bayes import DynamicHierarchicalPoisson
from mundial.models.gbm import GbmModel
from mundial.models.market import fetch_odds
from mundial.models.simulate import TournamentSimulator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = PROJECT_ROOT / "artifacts"
REPORTS = PROJECT_ROOT / "reports"
STATE_PATH = ARTIFACTS / "online_state.json"
LOG_PATH = ARTIFACTS / "predictions_log.parquet"

COMPONENTS = ["dc", "bayes", "gbm", "market"]
# bayes carries the LOTO-fitted mass; market gets the literature prior;
# dc/gbm keep live slots and must earn weight through the Hedge updates.
INITIAL_WEIGHTS = {"dc": 0.05, "bayes": 0.55, "gbm": 0.05, "market": 0.35}
HORIZON_DAYS = 5
CUSUM_THRESHOLD = 3.0
N_SIMS = 100_000


def load_env() -> None:
    env = PROJECT_ROOT / ".env"
    if env.exists():
        import os

        for line in env.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {
        "weights": dict(INITIAL_WEIGHTS),
        "scored": [],
        "score_history": [],
        "cusum": 0.0,
    }


def save_state(state: dict) -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def match_key(date, home, away) -> str:
    return f"{date}|{home}|{away}"


def outcome_of(hs: int, as_: int) -> int:
    return 0 if hs > as_ else 1 if hs == as_ else 2


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
        state["score_history"].append(
            {
                "key": key,
                "outcome": y,
                "score": f"{hs}-{as_}",
                "pool_loss": float(pool_loss),
                "losses": {c: float(l) for c, l in zip(COMPONENTS, losses)},
                "weights_after": {c: float(w) for c, w in zip(COMPONENTS, weights)},
            }
        )
        scored_rows.append({"key": key, "y": y, "hs": hs, "as": as_, "pool_loss": pool_loss})

    state["weights"] = {c: float(w) for c, w in zip(COMPONENTS, weights)}
    return scored_rows


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
    models = {
        "dc": DixonColes().fit(train, as_of=as_of),
        "bayes": DynamicHierarchicalPoisson().fit(train, as_of=as_of),
        "gbm": GbmModel().fit(train, as_of=as_of, fixtures=fixtures),
    }
    odds = fetch_odds()
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
            models["gbm"].predict_1x2(h, a, neutral=neu)[None, :],
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
               "xg_h": xg_h, "xg_a": xg_a}
        for c, b in zip(COMPONENTS, blocks):
            for j, suf in enumerate(["h", "d", "a"]):
                row[f"{c}_{suf}"] = float(b[0, j]) if b is not None else None
        for j, suf in enumerate(["h", "d", "a"]):
            row[f"pool_{suf}"] = float(pool[j])
        row["logged_at"] = dt.datetime.now().isoformat(timespec="seconds")
        rows.append(row)
    return pl.DataFrame(rows), models["bayes"]


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
            pl.concat([old, new], how="diagonal").write_parquet(LOG_PATH)
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
    return {"n_scored": n, "pool_logloss": float(pool), **{f"{c}_logloss": v for c, v in comp_means.items()}}


def write_report(today, scored, state, preds, sim_table, cusum) -> Path:
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"cycle_{today.isoformat()}.md"
    w = state["weights"]
    lines = [f"# Cycle report — {today}", ""]

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
    if sim_table is not None:
        lines += ["## Title race (top 10)", "",
                  "| team | R16 | QF | SF | Final | Champion |", "|---|---|---|---|---|---|"]
        for r in sim_table.head(10).iter_rows(named=True):
            lines.append(
                f"| {r['team']} | {r['p_r16']:.1%} | {r['p_qf']:.1%} | {r['p_sf']:.1%} "
                f"| {r['p_final']:.1%} | {r['p_champion']:.1%} |"
            )
        lines.append("")
    path.write_text("\n".join(lines))
    return path


def push_artifacts() -> None:
    """Best-effort: sync Hedge state and reports to the remote so cloud
    digest runs pool with current weights. Never fails the cycle."""
    import subprocess

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

    preds, bayes_model = predict_upcoming(results, today)
    append_log(preds)
    print(f"logged {preds.height} upcoming predictions")

    sim_table = None
    if bayes_model is not None:
        sim = TournamentSimulator(bayes_model, n_sims=N_SIMS)
        sim_table = sim.run()
        sim_table.write_csv(REPORTS / f"sim_{today.isoformat()}.csv")

    save_state(state)
    path = write_report(today, scored, state, preds, sim_table, cusum)
    print(f"report: {path}")
    push_artifacts()


if __name__ == "__main__":
    main()
