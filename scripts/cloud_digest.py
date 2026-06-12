"""Self-contained morning digest for cloud routine runs.

Designed to run in a fresh sandbox with only the repo + env vars:
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID  (delivery)
  ODDS_API_KEY                          (market layer; optional)
  DIGEST_DAYS (default 2), DIGEST_SIMS (default 30000)

Pulls fresh results, refits the models, fetches odds, pools with the latest
committed Hedge weights, simulates the bracket, and sends a Telegram digest.
Sends a short failure notice instead if anything breaks.
"""

from __future__ import annotations

import datetime as dt
import os
import sys
import traceback

import requests

DAYS = int(os.environ.get("DIGEST_DAYS", "2"))
SIMS = int(os.environ.get("DIGEST_SIMS", "30000"))


def send(text: str) -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat = os.environ["TELEGRAM_CHAT_ID"]
    resp = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={"chat_id": chat, "text": text, "parse_mode": "HTML"},
        timeout=30,
    )
    resp.raise_for_status()
    if not resp.json().get("ok"):
        raise RuntimeError(f"telegram send failed: {resp.text[:200]}")


def build_digest() -> str:
    import polars as pl

    from mundial.ingest.results import download_results, load_results
    from mundial.models.simulate import TournamentSimulator
    from mundial.online.update_cycle import load_env, predict_upcoming

    load_env()
    today = dt.date.today()
    download_results()
    results = load_results()

    preds, bayes_model = predict_upcoming(results, today, horizon_days=DAYS)

    lines = [f"🏆 <b>WC2026 — {today.strftime('%a %b %d')}</b>"]

    if preds.height == 0:
        lines.append("\nNo matches in the next "
                     f"{DAYS} day(s) — knockout pairings may not be set yet.")
    else:
        current = None
        for r in preds.iter_rows(named=True):
            if r["date"] != current:
                current = r["date"]
                d = dt.date.fromisoformat(r["date"])
                lines.append(f"\n<b>{d.strftime('%A %b %d')}</b>")
            mk = (
                f"  (mkt {r['market_h']:.0%}/{r['market_d']:.0%}/{r['market_a']:.0%})"
                if r["market_h"] is not None else ""
            )
            lines.append(
                f"{r['home_team']} v {r['away_team']}\n"
                f"  {r['pool_h']:.0%} / {r['pool_d']:.0%} / {r['pool_a']:.0%}{mk}"
            )

    # yesterday's results
    played = results.filter(
        (pl.col("tournament") == "FIFA World Cup")
        & (pl.col("date") >= today - dt.timedelta(days=1))
        & (pl.col("date") < today)
    )
    if played.height:
        lines.append("\n<b>Yesterday</b>")
        for r in played.iter_rows(named=True):
            lines.append(
                f"{r['home_team']} {r['home_score']}-{r['away_score']} {r['away_team']}"
            )

    if bayes_model is not None:
        table = TournamentSimulator(bayes_model, n_sims=SIMS).run()
        lines.append("\n<b>Title race</b>")
        for r in table.head(5).iter_rows(named=True):
            lines.append(f"{r['team']}  {r['p_champion']:.1%}")

    lines.append(f"\n<i>pool = model + market ensemble · {SIMS:,} sims</i>")
    return "\n".join(lines)


def main() -> None:
    try:
        send(build_digest())
        print("digest sent")
    except Exception:
        err = traceback.format_exc()
        print(err, file=sys.stderr)
        try:
            send(f"⚠️ WC2026 digest failed:\n<pre>{err[-600:]}</pre>")
        finally:
            sys.exit(1)


if __name__ == "__main__":
    main()
