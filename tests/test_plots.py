"""Smoke tests for plotting helpers — they should run without error on tiny data."""

import matplotlib

matplotlib.use("Agg")  # headless backend for CI

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from worldcup.features.elo import EloRatingSystem  # noqa: E402
from worldcup.visualization import plots  # noqa: E402

_MATCHES = pd.DataFrame(
    {
        "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]),
        "home_team": ["A", "B", "C", "A"],
        "away_team": ["B", "C", "A", "C"],
        "home_score": [2, 0, 1, 3],
        "away_score": [0, 1, 1, 2],
        "neutral": [False, True, False, True],
    }
)

_GROUPS = pd.DataFrame({"group": ["A", "A", "B", "B"], "slot": [1, 2, 1, 2], "team": ["A", "B", "C", "A"]})


def test_plot_functions_return_axes():
    plots.set_style()
    elo = EloRatingSystem().fit(_MATCHES)
    assert plots.plot_goal_distribution(_MATCHES) is not None
    assert plots.plot_home_advantage(_MATCHES) is not None
    assert plots.plot_elo_calibration(_MATCHES, n_bins=3) is not None
    assert plots.plot_top_teams(elo, n=3) is not None
    assert plots.plot_group_strength(_GROUPS, elo) is not None
    plt.close("all")
