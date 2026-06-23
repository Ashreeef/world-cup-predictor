"""Load the WC2026 tournament structure (groups & teams).

This gives the simulator its source of truth: which 48 teams are in which of
the 12 groups. Stored in data/reference/ (curated, version-controlled).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from worldcup.config import REFERENCE_DATA_DIR
from worldcup.data.teams import canonical

GROUPS_CSV: Path = REFERENCE_DATA_DIR / "wc2026_groups.csv"

N_GROUPS = 12
TEAMS_PER_GROUP = 4


def load_groups(path: Path | None = None) -> pd.DataFrame:
    """Return the groups table (columns: group, slot, team), names canonicalized.

    Raises ValueError if the structure isn't 12 groups of 4 unique teams.
    """
    csv_path = path or GROUPS_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found.")

    df = pd.read_csv(csv_path)
    expected = {"group", "slot", "team"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Groups file missing columns: {sorted(missing)}")

    df["team"] = df["team"].map(canonical)

    # Structural checks — fail loudly if the bracket is malformed.
    if df["group"].nunique() != N_GROUPS:
        raise ValueError(f"Expected {N_GROUPS} groups, found {df['group'].nunique()}.")
    sizes = df.groupby("group").size()
    if not (sizes == TEAMS_PER_GROUP).all():
        raise ValueError(f"Every group must have {TEAMS_PER_GROUP} teams. Got:\n{sizes}")
    if df["team"].duplicated().any():
        dupes = df.loc[df["team"].duplicated(), "team"].tolist()
        raise ValueError(f"Teams appear in more than one group: {dupes}")

    return df.sort_values(["group", "slot"]).reset_index(drop=True)


def get_group_teams(path: Path | None = None) -> dict[str, list[str]]:
    """Return {group_letter: [team, ...]} for convenient iteration."""
    df = load_groups(path)
    return {g: sub["team"].tolist() for g, sub in df.groupby("group")}
