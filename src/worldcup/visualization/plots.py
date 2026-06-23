"""Reusable plotting helpers.

Every chart in the EDA notebook AND the Streamlit dashboard comes from here, so
there is a single, tested place for visualization logic. Each function accepts
an optional `ax` and returns it, making the plots easy to compose.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import poisson

from worldcup.features.elo import EloRatingSystem


def set_style() -> None:
    """Apply a consistent visual theme."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["figure.figsize"] = (9, 5)
    plt.rcParams["axes.titleweight"] = "bold"


def plot_goal_distribution(matches: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Histogram of goals scored per team-match, with a Poisson fit overlay."""
    ax = ax or plt.gca()
    goals = pd.concat([matches["home_score"], matches["away_score"]], ignore_index=True)
    max_g = int(min(goals.max(), 10))
    bins = np.arange(0, max_g + 2) - 0.5

    ax.hist(goals, bins=bins, density=True, alpha=0.6, label="Observed", color="#4C72B0")
    x = np.arange(0, max_g + 1)
    ax.plot(x, poisson.pmf(x, goals.mean()), "o-", color="#C44E52", label=f"Poisson(λ={goals.mean():.2f})")
    ax.set(xlabel="Goals scored (per team, per match)", ylabel="Probability", title="Goals follow a Poisson distribution")
    ax.legend()
    return ax


def plot_home_advantage(matches: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Compare outcomes at non-neutral venues to quantify home advantage."""
    ax = ax or plt.gca()
    non_neutral = matches[~matches["neutral"]]
    outcomes = {
        "Home win": (non_neutral["home_score"] > non_neutral["away_score"]).mean(),
        "Draw": (non_neutral["home_score"] == non_neutral["away_score"]).mean(),
        "Away win": (non_neutral["home_score"] < non_neutral["away_score"]).mean(),
    }
    colors = ["#55A868", "#CCCCCC", "#C44E52"]
    ax.bar(outcomes.keys(), outcomes.values(), color=colors)
    for i, v in enumerate(outcomes.values()):
        ax.text(i, v + 0.01, f"{v:.0%}", ha="center", fontweight="bold")
    ax.set(ylabel="Share of matches", title="Home advantage (non-neutral venues)", ylim=(0, 0.6))
    return ax


def plot_elo_calibration(matches: pd.DataFrame, n_bins: int = 10, ax: plt.Axes | None = None) -> plt.Axes:
    """Reliability curve: do Elo's expected scores match real outcomes?"""
    ax = ax or plt.gca()
    preds = EloRatingSystem().replay_predictions(matches)
    bins = np.linspace(0, 1, n_bins + 1)
    preds["bucket"] = pd.cut(preds["expected_home"], bins, include_lowest=True)
    grouped = preds.groupby("bucket", observed=True).agg(
        predicted=("expected_home", "mean"), actual=("actual_home", "mean")
    )
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(grouped["predicted"], grouped["actual"], "o-", color="#4C72B0", label="Elo")
    ax.set(xlabel="Predicted home score (Elo)", ylabel="Actual home score", title="Elo calibration", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    return ax


def plot_top_teams(elo: EloRatingSystem, n: int = 15, ax: plt.Axes | None = None) -> plt.Axes:
    """Horizontal bar chart of the top-n teams by Elo."""
    ax = ax or plt.gca()
    top = elo.top(n).iloc[::-1]
    ax.barh(top["team"], top["elo"], color="#4C72B0")
    ax.set(xlabel="Elo rating", title=f"Top {n} teams by Elo")
    ax.set_xlim(left=top["elo"].min() - 50)
    return ax


def plot_score_matrix(
    matrix, home: str = "Home", away: str = "Away", max_display: int = 6, ax: plt.Axes | None = None
) -> plt.Axes:
    """Heatmap of scoreline probabilities (home goals rows x away goals cols)."""
    ax = ax or plt.gca()
    m = matrix[: max_display + 1, : max_display + 1]
    sns.heatmap(m, annot=True, fmt=".1%", cmap="Blues", cbar=False, ax=ax)
    ax.set(xlabel=f"{away} goals", ylabel=f"{home} goals", title="Scoreline probabilities")
    return ax


def plot_feature_importance(importances: dict[str, float], ax: plt.Axes | None = None) -> plt.Axes:
    """Horizontal bar chart of model feature importances."""
    ax = ax or plt.gca()
    s = pd.Series(importances).sort_values()
    ax.barh(s.index, s.values, color="#8172B3")
    ax.set(xlabel="Importance", title="Feature importance")
    return ax


def plot_group_strength(groups: pd.DataFrame, elo: EloRatingSystem, ax: plt.Axes | None = None) -> plt.Axes:
    """Average Elo per WC2026 group — reveals the 'group of death'."""
    ax = ax or plt.gca()
    g = groups.copy()
    g["elo"] = g["team"].map(elo.get)
    avg = g.groupby("group")["elo"].mean().sort_values(ascending=False)
    colors = sns.color_palette("flare", len(avg))
    ax.bar(avg.index, avg.values, color=colors)
    ax.set(xlabel="Group", ylabel="Average Elo", title="WC2026 group strength (avg Elo)")
    ax.set_ylim(bottom=avg.min() - 50)
    return ax
