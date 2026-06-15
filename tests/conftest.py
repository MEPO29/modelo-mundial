"""Shared test fixtures.

A deterministic stand-in for the Bayesian backbone so the tournament simulator
and online-cycle tests run in milliseconds without fitting NumPyro.
"""

from __future__ import annotations

import numpy as np
import pytest


class FakeBayes:
    """Minimal model satisfying what ``TournamentSimulator`` consumes.

    Provides ``_samples`` (intercept, home_adv) and ``_team_strengths`` with
    deterministic per-team attack/defense draws keyed by team name, so results
    are reproducible across runs and platforms.
    """

    def __init__(self, n_samples: int = 40, spread: float = 0.4) -> None:
        self.n_samples = n_samples
        self.spread = spread
        self._samples = {
            "intercept": np.zeros(n_samples),
            "home_adv": np.full(n_samples, 0.3),
        }
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _team_strengths(self, team: str) -> tuple[np.ndarray, np.ndarray]:
        if team not in self._cache:
            r = np.random.default_rng(abs(hash(team)) % (2**32))
            self._cache[team] = (
                r.normal(0.0, self.spread, self.n_samples),
                r.normal(0.0, self.spread, self.n_samples),
            )
        return self._cache[team]


@pytest.fixture
def fake_bayes() -> FakeBayes:
    return FakeBayes()
