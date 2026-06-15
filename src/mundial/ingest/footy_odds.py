"""Football-data.co.uk World Cup odds, results and xG.

A static workbook (``data/reference/WorldCup2026-football-data-co-uk.xlsx``)
with four sheets:

- ``WorldCup2026Qualifiers`` — 889 qualifier matches (2023-09 → 2026-04) with
  average/max bookmaker odds, shots, cards and xG (``HxG``/``AxG``, populated
  for ~339 of them).
- ``WorldCup2014`` / ``WorldCup2018`` / ``WorldCup2022`` — the 64-match finals
  (the exact backtest tournaments) with closing bookmaker odds (Bet365 /
  Pinnacle / Betfair plus clean ``H/D/A-Avg``) and penalty-shootout goals.

This is what lets the backtest score the market layer against *real* closing
odds (3 of the 5 reference tournaments) and sources xG as an optional feature.
Team names are reconciled to the martj42 dataset via ``market.normalize_team``;
``unmatched_teams`` reports any name this workbook uses that a results set does
not, so the alias map can be kept honest rather than guessed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from mundial.models.market import normalize_team, shin_devig

PROJECT_ROOT = Path(__file__).resolve().parents[3]
XLSX = PROJECT_ROOT / "data" / "reference" / "WorldCup2026-football-data-co-uk.xlsx"

# year -> finals sheet (these are the WC backtest tournaments)
FINALS_SHEETS = {2014: "WorldCup2014", 2018: "WorldCup2018", 2022: "WorldCup2022"}
QUALIFIERS_SHEET = "WorldCup2026Qualifiers"


def _read_sheet(sheet: str):
    """Read one workbook sheet as a pandas frame (xlsx → openpyxl)."""
    import pandas as pd

    return pd.read_excel(XLSX, sheet_name=sheet)


def historical_odds(year: int) -> pl.DataFrame:
    """Closing average odds for a finals tournament, Shin de-vigged.

    Columns: ``date, home_team, away_team, odds_h, odds_d, odds_a, p_h, p_d,
    p_a``. Rows missing any of the three average-odds columns are dropped.
    """
    import pandas as pd

    if year not in FINALS_SHEETS:
        raise KeyError(f"no finals odds sheet for {year}; have {sorted(FINALS_SHEETS)}")
    df = _read_sheet(FINALS_SHEETS[year])
    cols = ["Date", "Home", "Away", "H-Avg", "D-Avg", "A-Avg"]
    df = df[cols].dropna(subset=["H-Avg", "D-Avg", "A-Avg"])

    odds = df[["H-Avg", "D-Avg", "A-Avg"]].to_numpy(dtype=float)
    p = shin_devig(odds)
    dates = pd.to_datetime(df["Date"]).dt.date.tolist()
    return pl.DataFrame(
        {
            "date": dates,
            "home_team": [normalize_team(str(t)) for t in df["Home"]],
            "away_team": [normalize_team(str(t)) for t in df["Away"]],
            "odds_h": odds[:, 0], "odds_d": odds[:, 1], "odds_a": odds[:, 2],
            "p_h": p[:, 0], "p_d": p[:, 1], "p_a": p[:, 2],
        },
        schema_overrides={"date": pl.Date},
    )


def all_historical_odds() -> pl.DataFrame:
    """Stacked closing odds for every finals sheet."""
    return pl.concat([historical_odds(y) for y in FINALS_SHEETS])


def qualifier_xg() -> pl.DataFrame:
    """Qualifier xG / shots-on-target for the matches that carry xG.

    Columns: ``date, home_team, away_team, hxg, axg, hst, ast``. Only rows with
    both ``HxG`` and ``AxG`` present are returned (~339 of 889).
    """
    import pandas as pd

    df = _read_sheet(QUALIFIERS_SHEET).dropna(subset=["HxG", "AxG"])
    dates = pd.to_datetime(df["Date"]).dt.date.tolist()
    return pl.DataFrame(
        {
            "date": dates,
            "home_team": [normalize_team(str(t)) for t in df["Home"]],
            "away_team": [normalize_team(str(t)) for t in df["Away"]],
            "hxg": df["HxG"].to_numpy(dtype=float),
            "axg": df["AxG"].to_numpy(dtype=float),
            "hst": df["HST"].to_numpy(dtype=float),
            "ast": df["AST"].to_numpy(dtype=float),
        },
        schema_overrides={"date": pl.Date},
    )


def oriented_market(matches: pl.DataFrame, year: int) -> np.ndarray:
    """Shin-de-vigged market probs aligned to ``matches`` rows.

    Returns an ``(N, 3)`` array of ``[p_home, p_draw, p_away]`` oriented to each
    row's own home/away (the workbook's Home/Away designation for neutral-site
    matches need not match the results dataset, so probabilities are flipped to
    the row's orientation). Rows whose pairing has no odds are left as ``nan``.
    """
    odds = historical_odds(year)
    lut = {
        frozenset((r["home_team"], r["away_team"])): r
        for r in odds.iter_rows(named=True)
    }
    out = np.full((matches.height, 3), np.nan)
    for i, m in enumerate(matches.iter_rows(named=True)):
        rec = lut.get(frozenset((m["home_team"], m["away_team"])))
        if rec is None:
            continue
        if rec["home_team"] == m["home_team"]:
            out[i] = [rec["p_h"], rec["p_d"], rec["p_a"]]
        else:  # workbook had the pairing the other way round — flip H<->A
            out[i] = [rec["p_a"], rec["p_d"], rec["p_h"]]
    return out


def unmatched_teams(known: set[str]) -> set[str]:
    """Workbook team names (after normalization) absent from ``known``.

    Pass the team set of a loaded results frame to surface aliases still needed.
    """
    seen: set[str] = set()
    for sheet in (QUALIFIERS_SHEET, *FINALS_SHEETS.values()):
        df = _read_sheet(sheet)
        for col in ("Home", "Away"):
            seen |= {normalize_team(str(t)) for t in df[col].dropna()}
    return seen - known
