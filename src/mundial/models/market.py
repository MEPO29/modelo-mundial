"""Market layer (Layer C): de-vigged bookmaker probabilities.

Shin's method models the bookmaker margin as protection against insider
trading, which removes more vig from longshots than from favorites — the
empirically correct correction for the favorite-longshot bias, unlike
proportional normalization.

Live odds come from The Odds API when ODDS_API_KEY is set; the layer is
optional and the ensemble degrades gracefully without it.
"""

from __future__ import annotations

import datetime as dt
import os

import numpy as np
import polars as pl
import requests

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds"


def shin_devig(odds: np.ndarray) -> np.ndarray:
    """De-vig decimal odds (..., 3) -> probabilities (..., 3) via Shin's method.

    Solves for the insider fraction z such that the implied probabilities
    sum to 1: p_i = (sqrt(z^2 + 4(1-z) q_i^2 / s) - z) / (2(1-z)), where
    q_i = 1/odds_i and s = sum q_i. Bisection on z in [0, 0.2].
    """
    odds = np.asarray(odds, dtype=float)
    q = 1.0 / odds
    s = q.sum(axis=-1, keepdims=True)

    def probs(z):
        return (np.sqrt(z**2 + 4 * (1 - z) * q**2 / s) - z) / (2 * (1 - z))

    lo = np.zeros(s.shape)
    hi = np.full(s.shape, 0.2)
    for _ in range(60):
        mid = (lo + hi) / 2
        too_big = probs(mid).sum(axis=-1, keepdims=True) > 1.0
        lo = np.where(too_big, mid, lo)
        hi = np.where(too_big, hi, mid)
    p = probs((lo + hi) / 2)
    return p / p.sum(axis=-1, keepdims=True)


def fetch_odds(api_key: str | None = None) -> pl.DataFrame | None:
    """Current h2h odds for WC matches; None when no key is configured.

    Returns columns: home_team, away_team, odds_h, odds_d, odds_a, p_h, p_d, p_a
    (median odds across books, Shin de-vigged).
    """
    api_key = api_key or os.environ.get("ODDS_API_KEY")
    if not api_key:
        return None
    resp = requests.get(
        ODDS_API_URL,
        params={"apiKey": api_key, "regions": "eu", "markets": "h2h"},
        timeout=30,
    )
    resp.raise_for_status()
    rows = []
    for event in resp.json():
        home, away = event["home_team"], event["away_team"]
        per_book = []
        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                if market["key"] != "h2h":
                    continue
                prices = {o["name"]: o["price"] for o in market["outcomes"]}
                if home in prices and away in prices and "Draw" in prices:
                    per_book.append([prices[home], prices["Draw"], prices[away]])
        if not per_book:
            continue
        med = np.median(np.array(per_book), axis=0)
        p = shin_devig(med)
        rows.append(
            {
                "home_team": home, "away_team": away,
                "odds_h": med[0], "odds_d": med[1], "odds_a": med[2],
                "p_h": p[0], "p_d": p[1], "p_a": p[2],
                "fetched": dt.datetime.now().isoformat(timespec="seconds"),
            }
        )
    return pl.DataFrame(rows) if rows else None
