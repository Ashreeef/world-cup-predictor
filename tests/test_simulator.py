"""Tests for the Monte Carlo simulator.

We use conservation invariants: across all sims there is exactly 1 champion,
2 finalists, 4 semi-finalists, 8 quarter-finalists, 16 R16 teams, 32 qualifiers
per tournament. Summing each probability column over all teams must therefore
equal those constants — a strong structural check on the bracket logic.
"""

import numpy as np
import pandas as pd

from worldcup.data.wc2026 import get_group_teams
from worldcup.features.elo import EloRatingSystem
from worldcup.models.poisson import train_poisson
from worldcup.simulation.simulator import TournamentSimulator, _seeding_order


def _toy_bundle():
    rng = np.random.default_rng(0)
    elo_diff = rng.normal(0, 200, 1500)
    feats = pd.DataFrame(
        {
            "elo_diff": elo_diff,
            "neutral": 1,
            "home_score": rng.poisson(np.exp(0.2 + 0.002 * elo_diff)),
            "away_score": rng.poisson(np.exp(0.2 - 0.002 * elo_diff)),
            "date": pd.date_range("2010-01-01", periods=1500, freq="D"),
        }
    )
    return train_poisson(feats, save=False)


def _toy_elo():
    elo = EloRatingSystem()
    teams = [t for ts in get_group_teams().values() for t in ts]
    # Gradient of strengths; make Argentina very strong, Haiti very weak.
    elo.ratings = {t: 1500.0 for t in teams}
    elo.ratings["Argentina"] = 2200.0
    elo.ratings["Haiti"] = 1100.0
    return elo


def _sim(seed=1):
    return TournamentSimulator(_toy_elo(), _toy_bundle(), seed=seed)


def test_seeding_order_is_a_permutation():
    order = _seeding_order(32)
    assert sorted(order) == list(range(1, 33))
    assert order[0] == 1 and 32 in (order[1], order[-1])


def test_output_shape_and_ranges():
    df = _sim().run(n_sims=300)
    assert len(df) == 48
    for col in ["qualify", "round_of_16", "quarter_final", "semi_final", "final", "champion"]:
        assert df[col].between(0, 1).all()


def test_conservation_invariants():
    df = _sim().run(n_sims=300)
    assert abs(df["champion"].sum() - 1) < 1e-9
    assert abs(df["final"].sum() - 2) < 1e-9
    assert abs(df["semi_final"].sum() - 4) < 1e-9
    assert abs(df["quarter_final"].sum() - 8) < 1e-9
    assert abs(df["round_of_16"].sum() - 16) < 1e-9
    assert abs(df["qualify"].sum() - 32) < 1e-9


def test_probabilities_are_monotonic_per_team():
    df = _sim().run(n_sims=300)
    assert (df["champion"] <= df["final"] + 1e-9).all()
    assert (df["final"] <= df["semi_final"] + 1e-9).all()
    assert (df["semi_final"] <= df["quarter_final"] + 1e-9).all()
    assert (df["quarter_final"] <= df["round_of_16"] + 1e-9).all()
    assert (df["round_of_16"] <= df["qualify"] + 1e-9).all()


def test_stronger_team_more_likely_to_win():
    df = _sim().run(n_sims=400).set_index("team")
    assert df.loc["Argentina", "champion"] > df.loc["Haiti", "champion"]


def test_determinism_with_seed():
    a = _sim(seed=7).run(n_sims=150)
    b = _sim(seed=7).run(n_sims=150)
    pd.testing.assert_frame_equal(a, b)
