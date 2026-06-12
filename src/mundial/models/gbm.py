"""LightGBM contextual layer (Layer B).

Multiclass 1X2 on the streaming feature matrix. Captures interactions the
parametric backbone can't (form x rest, altitude x confederation, schedule
density). Friendlies are down-weighted to match the likelihood treatment in
the other layers. Trained once per walk-forward cut; NOT refit intra-
tournament — its inputs and ensemble weight move instead.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import polars as pl

from mundial.features.build import CATEGORICAL_COLS, FEATURE_COLS, build_features

PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "verbosity": -1,
    "seed": 0,
}
NUM_ROUNDS = 600
TRAIN_SINCE = dt.date(2000, 1, 1)
FRIENDLY_WEIGHT = 0.5


@dataclass
class GbmModel:
    booster: lgb.Booster | None = None
    _fixture_probs: dict[tuple, np.ndarray] = field(default_factory=dict)

    def fit(self, matches: pl.DataFrame, as_of: dt.date,
            fixtures: pl.DataFrame | None = None) -> "GbmModel":
        """Fit on played matches before `as_of`; pre-compute fixture predictions.

        The featurizer streams the full history once, so fixture features see
        exactly the state as of the last played match — no leakage.
        """
        played = matches.filter(pl.col("date") < as_of).sort("date")
        train, fix = build_features(played, fixtures)
        train = train.filter(pl.col("date") >= TRAIN_SINCE)

        X = train.select(FEATURE_COLS).to_pandas()
        for c in CATEGORICAL_COLS:
            X[c] = X[c].astype("category")
        y = train["outcome"].to_numpy()
        w = np.where(train["tournament"].to_numpy() == "Friendly", FRIENDLY_WEIGHT, 1.0)

        dtrain = lgb.Dataset(X, label=y, weight=w, categorical_feature=CATEGORICAL_COLS)
        self.booster = lgb.train(PARAMS, dtrain, num_boost_round=NUM_ROUNDS)

        self._fixture_probs = {}
        if fix is not None and fix.height:
            Xf = fix.select(FEATURE_COLS).to_pandas()
            for c in CATEGORICAL_COLS:
                Xf[c] = Xf[c].astype("category")
            probs = self.booster.predict(Xf)
            for (d, h, a), p in zip(
                fix.select("date", "home_team", "away_team").iter_rows(), probs
            ):
                self._fixture_probs[(h, a)] = p
                self._fixture_probs[(d, h, a)] = p
        return self

    def predict_1x2(self, home: str, away: str, neutral: bool = True) -> np.ndarray:
        """Look up the pre-computed fixture prediction (keyed by team pair)."""
        key = (home, away)
        if key not in self._fixture_probs:
            raise KeyError(
                f"{home} v {away} was not in the fixtures passed to fit(); "
                "GBM features are schedule-dependent and cannot be built ad hoc"
            )
        return self._fixture_probs[key]

    def feature_importance(self) -> pl.DataFrame:
        imp = self.booster.feature_importance(importance_type="gain")
        return pl.DataFrame({"feature": FEATURE_COLS, "gain": imp}).sort(
            "gain", descending=True
        )
