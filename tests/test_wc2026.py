"""Tests for the WC2026 groups loader and the live-match schema validator."""

import pandas as pd
import pytest

from worldcup.data.schema import validate_match_df
from worldcup.data.teams import canonical, display
from worldcup.data.wc2026 import get_group_teams, load_groups


# ── Groups ──────────────────────────────────────────────────────────────────────
def test_groups_structure():
    df = load_groups()
    assert len(df) == 48
    assert df["group"].nunique() == 12
    assert (df.groupby("group").size() == 4).all()
    assert not df["team"].duplicated().any()


def test_groups_use_canonical_names():
    teams = set(load_groups()["team"])
    # Canonical (dataset) spellings, not the official display spellings.
    assert "Czech Republic" in teams and "Czechia" not in teams
    assert "Turkey" in teams and "Türkiye" not in teams


def test_get_group_teams():
    groups = get_group_teams()
    assert groups["J"] == ["Argentina", "Austria", "Algeria", "Jordan"]


# ── Name normalization ──────────────────────────────────────────────────────────
def test_canonical_and_display():
    assert canonical("Türkiye") == "Turkey"
    assert canonical("Czechia") == "Czech Republic"
    assert display("Turkey") == "Türkiye"


# ── Match schema ────────────────────────────────────────────────────────────────
def _match(**overrides):
    base = {
        "date": "2026-06-25",
        "home_team": "Türkiye",
        "away_team": "United States",
        "home_score": 1,
        "away_score": 2,
        "stage": "group",
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_validate_match_normalizes_names():
    out = validate_match_df(_match())
    assert out.loc[0, "home_team"] == "Turkey"  # alias normalized
    assert out["home_score"].dtype.kind == "i"


def test_validate_match_rejects_bad_stage():
    with pytest.raises(ValueError):
        validate_match_df(_match(stage="quarterfinal"))  # not in VALID_STAGES


def test_validate_match_rejects_same_team():
    with pytest.raises(ValueError):
        validate_match_df(_match(away_team="Türkiye"))


def test_validate_match_rejects_missing_column():
    df = _match().drop(columns=["stage"])
    with pytest.raises(ValueError):
        validate_match_df(df)
