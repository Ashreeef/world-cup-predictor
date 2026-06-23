"""Tests for the results loader using a tiny synthetic CSV.

We don't commit the real dataset (it's downloaded on demand), so these tests
exercise the loader's validation and cleaning logic against a fixture we build
in a temp directory.
"""

import pandas as pd
import pytest

from worldcup.data.load import load_results

_VALID_ROWS = pd.DataFrame(
    {
        "date": ["2022-11-20", "2018-06-14"],
        "home_team": ["Qatar", "Russia"],
        "away_team": ["Ecuador", "Saudi Arabia"],
        "home_score": [0, 5],
        "away_score": [2, 0],
        "tournament": ["FIFA World Cup", "FIFA World Cup"],
        "city": ["Al Khor", "Moscow"],
        "country": ["Qatar", "Russia"],
        "neutral": [False, False],
    }
)


def test_load_results_sorts_and_types(tmp_path):
    csv = tmp_path / "results.csv"
    _VALID_ROWS.to_csv(csv, index=False)

    df = load_results(path=csv)

    # Sorted oldest -> newest.
    assert df.iloc[0]["date"] == pd.Timestamp("2018-06-14")
    # Scores are integers.
    assert df["home_score"].dtype.kind == "i"
    assert len(df) == 2


def test_load_results_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_results(path=tmp_path / "nope.csv")


def test_load_results_missing_columns(tmp_path):
    csv = tmp_path / "bad.csv"
    _VALID_ROWS.drop(columns=["neutral"]).to_csv(csv, index=False)
    with pytest.raises(ValueError):
        load_results(path=csv)
