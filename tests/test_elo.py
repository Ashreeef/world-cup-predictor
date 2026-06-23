"""Tests for the Elo rating system — verifying the math behaves correctly."""

import pandas as pd

from worldcup.features.elo import EloRatingSystem, _goal_diff_multiplier


def test_expected_scores_sum_to_one():
    elo = EloRatingSystem()
    e_home = elo.expected_score("A", "B", neutral=True)
    e_away = elo.expected_score("B", "A", neutral=True)
    assert abs((e_home + e_away) - 1.0) < 1e-9


def test_home_advantage_raises_expectation():
    elo = EloRatingSystem()
    assert elo.expected_score("A", "B", neutral=False) > elo.expected_score("A", "B", neutral=True)


def test_win_is_zero_sum():
    elo = EloRatingSystem()
    res = elo.update_match("A", "B", 1, 0, neutral=True)
    assert res["home_change"] > 0
    assert abs(res["home_change"] + res["away_change"]) < 1e-9
    assert abs(elo.get("A") - 1500) == abs(elo.get("B") - 1500)


def test_bigger_margin_bigger_change():
    narrow = EloRatingSystem().update_match("A", "B", 1, 0, neutral=True)["home_change"]
    blowout = EloRatingSystem().update_match("A", "B", 5, 0, neutral=True)["home_change"]
    assert blowout > narrow


def test_underdog_gains_more_than_favorite():
    # Favorite (A=1800) beats underdog (B=1500): small gain.
    fav = EloRatingSystem()
    fav.ratings = {"A": 1800.0, "B": 1500.0}
    fav_gain = fav.update_match("A", "B", 1, 0, neutral=True)["home_change"]

    # Underdog (B=1500) beats favorite (A=1800): big gain.
    dog = EloRatingSystem()
    dog.ratings = {"A": 1800.0, "B": 1500.0}
    dog_gain = dog.update_match("B", "A", 1, 0, neutral=True)["home_change"]

    assert dog_gain > fav_gain


def test_goal_diff_multiplier():
    assert _goal_diff_multiplier(1) == 1.0
    assert _goal_diff_multiplier(2) == 1.5
    assert _goal_diff_multiplier(3) == 14 / 8


def test_fit_and_persistence(tmp_path):
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "home_team": ["A", "B"],
            "away_team": ["B", "C"],
            "home_score": [2, 0],
            "away_score": [0, 1],
            "neutral": [True, True],
        }
    )
    elo = EloRatingSystem().fit(matches)
    assert elo.n_matches == 2
    assert elo.get("A") > 1500  # A won

    path = tmp_path / "elo.json"
    elo.save(path)
    reloaded = EloRatingSystem.load(path)
    assert reloaded.get("A") == elo.get("A")
    assert reloaded.n_matches == 2
