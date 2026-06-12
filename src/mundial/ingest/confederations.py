"""Team -> confederation mapping, derived from the match data itself.

Each team is assigned to the confederation whose continental competitions it
has appeared in most often since 2008 (post-dating Australia's 2006 move from
OFC to AFC). Guest appearances (e.g. Qatar at the 2019 Copa América) are
outvoted by a team's regular continental schedule. Teams that never appear in
a confederation competition (CONIFA sides, micro-islands) map to OTHER and
share a pooled prior in the hierarchical model.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

CONFEDERATIONS = ["UEFA", "CONMEBOL", "CONCACAF", "AFC", "CAF", "OFC", "OTHER"]

TOURNAMENT_CONF = {
    "UEFA Euro": "UEFA",
    "UEFA Euro qualification": "UEFA",
    "UEFA Nations League": "UEFA",
    "Baltic Cup": "UEFA",
    "Copa América": "CONMEBOL",
    "Gold Cup": "CONCACAF",
    "Gold Cup qualification": "CONCACAF",
    "CONCACAF Nations League": "CONCACAF",
    "CONCACAF Nations League qualification": "CONCACAF",
    "CONCACAF Championship": "CONCACAF",
    "CONCACAF Series": "CONCACAF",
    "CFU Caribbean Cup": "CONCACAF",
    "CFU Caribbean Cup qualification": "CONCACAF",
    "UNCAF Cup": "CONCACAF",
    "Copa Centroamericana": "CONCACAF",
    "AFC Asian Cup": "AFC",
    "AFC Asian Cup qualification": "AFC",
    "AFC Challenge Cup": "AFC",
    "AFC Challenge Cup qualification": "AFC",
    "AFF Championship": "AFC",
    "AFF Championship qualification": "AFC",
    "EAFF Championship": "AFC",
    "SAFF Cup": "AFC",
    "WAFF Championship": "AFC",
    "Gulf Cup": "AFC",
    "African Cup of Nations": "CAF",
    "African Cup of Nations qualification": "CAF",
    "African Nations Championship": "CAF",
    "COSAFA Cup": "CAF",
    "CECAFA Cup": "CAF",
    "Amílcar Cabral Cup": "CAF",
    "Oceania Nations Cup": "OFC",
    "Oceania Nations Cup qualification": "OFC",
    "Pacific Games": "OFC",
    "Melanesia Cup": "OFC",
    "Polynesia Cup": "OFC",
}


def team_confederations(
    results: pl.DataFrame, since: dt.date = dt.date(2008, 1, 1)
) -> dict[str, str]:
    """Map every team appearing in `results` to a confederation (or OTHER)."""
    tagged = (
        results.filter(pl.col("date") >= since)
        .with_columns(pl.col("tournament").replace_strict(TOURNAMENT_CONF, default=None).alias("conf"))
        .filter(pl.col("conf").is_not_null())
    )
    counts: dict[str, dict[str, int]] = {}
    for col in ("home_team", "away_team"):
        for team, conf, n in tagged.group_by(col, "conf").len().iter_rows():
            d = counts.setdefault(team, {})
            d[conf] = d.get(conf, 0) + n

    all_teams = set(results["home_team"]) | set(results["away_team"])
    return {
        t: max(counts[t], key=counts[t].get) if t in counts else "OTHER"
        for t in sorted(all_teams)
    }


if __name__ == "__main__":
    from collections import Counter

    from mundial.ingest.results import load_results

    mapping = team_confederations(load_results())
    print(Counter(mapping.values()))
    for t in ["Brazil", "Mexico", "Australia", "Japan", "Qatar", "United States",
              "New Zealand", "Morocco", "Germany", "Haiti", "Uzbekistan", "Curaçao"]:
        print(f"{t:<15} {mapping.get(t)}")
