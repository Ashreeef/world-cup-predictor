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
from worldcup.simulation.simulator import (
    TournamentSimulator,
    assign_thirds,
    build_official_bracket,
)


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


def test_assign_thirds_avoids_rematch_and_is_bijection():
    advancing = ["A", "B", "C", "D", "G", "H", "K", "L"]  # incl. all winner-slot groups
    assign = assign_thirds(advancing)
    assert len(assign) == 8
    assert sorted(assign.values()) == sorted(advancing)  # bijection
    for slot, group in assign.items():
        assert slot != group  # no group-stage rematch


def test_official_bracket_has_32_unique_teams():
    winners = {g: ord(g) - 65 for g in "ABCDEFGHIJKL"}          # 0..11
    runners = {g: 12 + ord(g) - 65 for g in "ABCDEFGHIJKL"}     # 12..23
    advancing = ["A", "B", "C", "D", "G", "H", "K", "L"]
    third_team_by_group = {g: 24 + i for i, g in enumerate(advancing)}  # 24..31
    assign = assign_thirds(advancing)
    bracket = build_official_bracket(winners, runners, third_team_by_group, assign)
    assert len(bracket) == 32
    assert len(set(bracket)) == 32  # every qualifier appears exactly once


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


def test_played_results_are_fixed():
    """A fixed result should make that outcome certain across all sims."""
    teams = [t for ts in get_group_teams().values() for t in ts]
    # Argentina hammers Jordan 5-0 (same group J) in every simulation.
    played = pd.DataFrame(
        [{"home_team": "Argentina", "away_team": "Jordan", "home_score": 5, "away_score": 0}]
    )
    sim = TournamentSimulator(_toy_elo(), _toy_bundle(), seed=1, played_matches=played)
    i, j = sim.idx["Argentina"], sim.idx["Jordan"]
    # The fixed match always returns 5-0 with Argentina winning.
    for _ in range(20):
        gi, gj, w = sim._play(i, j)
        assert (gi, gj) == (5, 0) and w == i
