"""Tests for fixed-bracket knockout simulation."""

import numpy as np
import pandas as pd

from worldcup.features.elo import EloRatingSystem
from worldcup.models.poisson import train_poisson
from worldcup.simulation.knockout import (
    bracket_team_names,
    knockout_predictions,
    load_r32_bracket,
    r32_match_table,
)
from worldcup.simulation.simulator import TournamentSimulator


def _toy_bundle():
    rng = np.random.default_rng(0)
    elo_diff = rng.normal(0, 200, 1500)
    feats = pd.DataFrame({
        "elo_diff": elo_diff, "neutral": 1,
        "home_score": rng.poisson(np.exp(0.2 + 0.002 * elo_diff)),
        "away_score": rng.poisson(np.exp(0.2 - 0.002 * elo_diff)),
        "date": pd.date_range("2010-01-01", periods=1500, freq="D"),
    })
    return train_poisson(feats, save=False)


def _toy_elo():
    from worldcup.data.wc2026 import get_group_teams

    elo = EloRatingSystem()
    teams = [t for ts in get_group_teams().values() for t in ts]
    elo.ratings = {t: 1500.0 for t in teams}
    elo.ratings["Argentina"] = 2200.0
    return elo


def test_bracket_loads_32_unique_teams():
    bracket = load_r32_bracket()
    assert len(bracket) == 16
    names = bracket_team_names(bracket)
    assert len(names) == 32 and len(set(names)) == 32


def test_knockout_conservation_invariants():
    table, _ = knockout_predictions(_toy_elo(), _toy_bundle(), n_sims=300)
    assert abs(table["champion"].sum() - 1) < 1e-9
    assert abs(table["final"].sum() - 2) < 1e-9
    assert abs(table["semi_final"].sum() - 4) < 1e-9
    assert abs(table["quarter_final"].sum() - 8) < 1e-9
    assert abs(table["round_of_16"].sum() - 16) < 1e-9
    assert abs(table["qualify"].sum() - 32) < 1e-9  # exactly the 32 bracket teams


def test_only_bracket_teams_have_nonzero_odds():
    table, _ = knockout_predictions(_toy_elo(), _toy_bundle(), n_sims=200)
    in_bracket = set(bracket_team_names())
    nonzero = set(table[table["qualify"] > 0]["team"])
    assert nonzero == in_bracket


def test_match_win_prob_and_played_result():
    bundle = _toy_bundle()
    bracket = load_r32_bracket()
    # No games played -> probabilities in (0, 1) and complementary.
    sim = TournamentSimulator(_toy_elo(), bundle, seed=1)
    mt = r32_match_table(sim, bundle, bracket)
    assert len(mt) == 16
    assert np.allclose(mt["p_home"] + mt["p_away"], 1.0)

    # A fixed result makes the win prob deterministic.
    home, away = bracket[0]
    played = pd.DataFrame([{"home_team": home, "away_team": away, "home_score": 3, "away_score": 0}])
    sim2 = TournamentSimulator(_toy_elo(), bundle, seed=1, played_matches=played)
    assert sim2.match_win_prob(home, away) == 1.0
