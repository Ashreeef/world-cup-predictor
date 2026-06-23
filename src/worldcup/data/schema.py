"""The canonical schema for a finished-match update (`new_match.csv`).

This is the contract between you (entering match results) and the live update
pipeline. Defining it in one place means the dashboard, the validator, and the
update pipeline all agree on exactly what a match record looks like.

Required columns (must always be present):
    date            ISO date, e.g. 2026-06-25
    home_team       team name (aliases are normalized automatically)
    away_team       team name
    home_score      final goals, integer >= 0
    away_score      final goals, integer >= 0
    stage           one of VALID_STAGES below

Optional rich-stat columns (used by advanced features when available):
    home_xg, away_xg                 expected goals (float)
    home_possession, away_possession percentage 0-100
    home_shots, away_shots           integer >= 0
    home_red_cards, away_red_cards   integer >= 0
"""

from __future__ import annotations

import pandas as pd

from worldcup.data.teams import canonical

REQUIRED_COLUMNS: list[str] = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "stage",
]

OPTIONAL_STAT_COLUMNS: list[str] = [
    "home_xg",
    "away_xg",
    "home_possession",
    "away_possession",
    "home_shots",
    "away_shots",
    "home_red_cards",
    "away_red_cards",
]

VALID_STAGES: set[str] = {
    "group",
    "round_of_32",
    "round_of_16",
    "quarter_final",
    "semi_final",
    "third_place",
    "final",
}


def validate_match_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a finished-match DataFrame.

    Returns a cleaned copy: dates parsed, team names canonicalized, scores as
    ints. Raises ValueError on any contract violation so a bad file can never
    silently corrupt the ratings.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Match file missing required columns: {sorted(missing)}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    out["home_team"] = out["home_team"].map(canonical)
    out["away_team"] = out["away_team"].map(canonical)

    for col in ("home_score", "away_score"):
        out[col] = out[col].astype(int)
        if (out[col] < 0).any():
            raise ValueError(f"{col} contains negative values.")

    bad_stage = set(out["stage"]) - VALID_STAGES
    if bad_stage:
        raise ValueError(f"Invalid stage value(s): {sorted(bad_stage)}. Allowed: {sorted(VALID_STAGES)}")

    if (out["home_team"] == out["away_team"]).any():
        raise ValueError("A match cannot have the same home and away team.")

    return out
