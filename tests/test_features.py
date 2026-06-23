"""Tests for feature engineering — especially that features are leak-free."""

import numpy as np
import pandas as pd

from worldcup.features.build import (
    FEATURE_COLUMNS,
    build_features,
    fit_state,
    make_feature_row,
)
from worldcup.features.elo import EloRatingSystem

# A and B meet twice; A wins the first 2-0. Non-neutral, A always home.
_MATCHES = pd.DataFrame(
    {
        "date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
        "home_team": ["A", "A"],
        "away_team": ["B", "B"],
        "home_score": [2, 0],
        "away_score": [0, 0],
        "neutral": [False, False],
    }
)


def test_first_match_has_no_form_history():
    """LEAK CHECK: the first match must have NaN form (no prior data used)."""
    df = build_features(_MATCHES, dropna=False)
    assert np.isnan(df.loc[0, "form_points_diff"])
    assert np.isnan(df.loc[0, "form_gd_diff"])
    # Both teams unseen -> elo_diff is pure home advantage.
    assert df.loc[0, "elo_diff"] == 100.0
    assert df.loc[0, "result"] == "H"


def test_second_match_uses_only_prior_form():
    """Match 2's form reflects ONLY match 1 (A: +3 pts, +2 GD; B: 0 pts, -2 GD)."""
    df = build_features(_MATCHES, dropna=False)
    assert df.loc[1, "form_points_diff"] == 3.0  # 3 - 0
    assert df.loc[1, "form_gd_diff"] == 4.0  # (+2) - (-2)
    assert df.loc[1, "elo_diff"] > 100.0  # A gained Elo after winning


def test_dropna_removes_history_less_rows():
    df = build_features(_MATCHES, dropna=True)
    assert len(df) == 1  # only the 2nd match has form history
    assert set(FEATURE_COLUMNS).issubset(df.columns)


def test_result_encoding_draw():
    draw = _MATCHES.copy()
    draw.loc[0, ["home_score", "away_score"]] = [1, 1]
    assert build_features(draw, dropna=False).loc[0, "result"] == "D"


def test_make_feature_row_matches_columns():
    elo = EloRatingSystem().fit(_MATCHES)
    row = make_feature_row("A", "B", neutral=True, elo=elo)
    assert set(row.keys()) == set(FEATURE_COLUMNS)
    assert row["neutral"] == 1
    assert row["elo_diff"] > 0  # A is stronger after beating B


def test_fit_state_returns_usable_state():
    elo, form = fit_state(_MATCHES)
    assert elo.get("A") > elo.get("B")
    assert form.avg_points("A") > form.avg_points("B")
