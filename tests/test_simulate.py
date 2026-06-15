"""Tournament simulator invariants (fast, with a deterministic fake backbone)."""

from __future__ import annotations

import numpy as np
import pytest

from mundial.ingest import results as results_mod
from mundial.models.simulate import ROUNDS, TournamentSimulator


def _has_data() -> bool:
    return bool(list(results_mod.RAW_DIR.glob("*/results.csv")))


pytestmark = pytest.mark.skipif(not _has_data(), reason="martj42 results not pulled")


def test_run_yields_valid_probability_table(fake_bayes):
    sim = TournamentSimulator(fake_bayes, n_sims=2000, seed=1)
    table = sim.run()

    assert table.height == 48  # one row per team
    probs = table.select(*[f"p_{r}" for r in ROUNDS]).to_numpy()
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
    # exactly one champion per simulation -> champion probabilities sum to ~1
    assert table["p_champion"].sum() == pytest.approx(1.0, abs=1e-6)
    # fixed bracket sizes: 32 reach the R32, then 16, 8, 4, 2, 1
    assert table["p_r32"].sum() == pytest.approx(32.0, abs=1e-6)
    assert table["p_r16"].sum() == pytest.approx(16.0, abs=1e-6)
    assert table["p_final"].sum() == pytest.approx(2.0, abs=1e-6)


def test_survival_is_monotone_per_team(fake_bayes):
    sim = TournamentSimulator(fake_bayes, n_sims=2000, seed=2)
    table = sim.run()
    cols = [f"p_{r}" for r in ROUNDS]  # r32, r16, qf, sf, final, champion
    m = table.select(*cols).to_numpy()
    # reaching a later round implies reaching every earlier one
    assert (np.diff(m, axis=1) <= 1e-9).all()


def test_seed_is_reproducible(fake_bayes):
    a = TournamentSimulator(fake_bayes, n_sims=1000, seed=7).run()
    b = TournamentSimulator(fake_bayes, n_sims=1000, seed=7).run()
    assert a.equals(b)
