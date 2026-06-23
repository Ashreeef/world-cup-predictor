"""Load and validate the historical international match results.

This is the single entry point the rest of the project uses to read match
data. Centralizing it means cleaning rules live in one place and every
downstream module (Elo, EDA, features) sees identical, validated data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from worldcup.config import RAW_DATA_DIR

RESULTS_CSV: Path = RAW_DATA_DIR / "results.csv"

# Columns guaranteed by the source dataset. We assert these exist so a
# malformed file fails loudly here rather than silently downstream.
EXPECTED_COLUMNS = {
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
}


def load_results(path: Path | None = None) -> pd.DataFrame:
    """Return the cleaned match-results table, sorted oldest -> newest.

    Parameters
    ----------
    path : Path, optional
        Override the CSV location (used by tests). Defaults to
        data/raw/results.csv.

    Raises
    ------
    FileNotFoundError
        If the file is missing (you probably need to run the downloader).
    ValueError
        If expected columns are absent.
    """
    csv_path = path or RESULTS_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Download it first:\n"
            "    python -m worldcup.data.download"
        )

    df = pd.read_csv(csv_path, parse_dates=["date"])

    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    # Drop matches without a recorded score (can't learn from them) and
    # enforce integer score types.
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Deterministic ordering is important for time-based Elo updates later.
    df = df.sort_values("date").reset_index(drop=True)
    return df
