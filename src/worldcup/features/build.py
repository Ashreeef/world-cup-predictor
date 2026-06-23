"""Feature engineering — turn matches into leak-free model inputs.

THE GOLDEN RULE: every feature for a match uses only information available
*before* kickoff. We enforce this with a single chronological pass:

    for each match (oldest -> newest):
        1. read current state (Elo, recent form)  -> build the feature row
        2. THEN update the state with this match's result

Doing it in that order makes leakage structurally impossible.

Features (home-relative):
    elo_diff          home Elo - away Elo (+ home advantage if not neutral)
    form_points_diff  avg points (last N) home - away   (win=3, draw=1, loss=0)
    form_gd_diff      avg goal difference (last N) home - away
    neutral           1 if neutral venue else 0
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from worldcup.data.teams import canonical
from worldcup.features.elo import EloRatingSystem, match_weight

FEATURE_COLUMNS = ["elo_diff", "form_points_diff", "form_gd_diff", "neutral"]

DEFAULT_FORM_WINDOW = 5


def _result(home_score: int, away_score: int) -> str:
    """Match result from the home team's perspective: 'H', 'D', or 'A'."""
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def _avg(values: deque) -> float:
    """Mean of a deque, or NaN if empty (no history yet)."""
    return float(np.mean(values)) if len(values) else np.nan


class _FormTracker:
    """Maintains rolling last-N points and goal-difference per team."""

    def __init__(self, window: int) -> None:
        self.points: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self.gd: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))

    def avg_points(self, team: str) -> float:
        return _avg(self.points[team])

    def avg_gd(self, team: str) -> float:
        return _avg(self.gd[team])

    def update(self, home: str, away: str, hs: int, as_: int) -> None:
        hp = 3 if hs > as_ else (1 if hs == as_ else 0)
        ap = 3 if as_ > hs else (1 if hs == as_ else 0)
        self.points[home].append(hp)
        self.points[away].append(ap)
        self.gd[home].append(hs - as_)
        self.gd[away].append(as_ - hs)


def build_features(
    matches: pd.DataFrame,
    form_window: int = DEFAULT_FORM_WINDOW,
    dropna: bool = True,
) -> pd.DataFrame:
    """Build a leak-free feature table from a results DataFrame.

    Returns one row per match: date, home_team, away_team, the FEATURE_COLUMNS,
    and the target `result` ('H'/'D'/'A'). With dropna=True, matches where a
    team has no prior history (NaN form) are dropped.
    """
    matches = matches.sort_values("date").reset_index(drop=True)
    elo = EloRatingSystem()
    form = _FormTracker(form_window)
    rows = []

    for row in matches.itertuples(index=False):
        home, away = canonical(row.home_team), canonical(row.away_team)
        neutral = bool(getattr(row, "neutral", False))
        hs, as_ = int(row.home_score), int(row.away_score)

        # 1) READ state -> features (pre-match only)
        hfa = 0.0 if neutral else elo.home_advantage
        rows.append(
            {
                "date": row.date,
                "home_team": home,
                "away_team": away,
                "elo_diff": elo.get(home) - elo.get(away) + hfa,
                "form_points_diff": form.avg_points(home) - form.avg_points(away),
                "form_gd_diff": form.avg_gd(home) - form.avg_gd(away),
                "neutral": int(neutral),
                "home_score": hs,
                "away_score": as_,
                "result": _result(hs, as_),
            }
        )

        # 2) THEN update state with the result
        elo.update_match(home, away, hs, as_, neutral, weight=match_weight(getattr(row, "tournament", None)))
        form.update(home, away, hs, as_)

    df = pd.DataFrame(rows)
    if dropna:
        df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    return df


def make_feature_row(
    home: str,
    away: str,
    neutral: bool,
    elo: EloRatingSystem,
    form: "_FormTracker | None" = None,
) -> dict[str, float]:
    """Build a single feature row for an UPCOMING match (for prediction).

    Uses the supplied (already-fitted) Elo and optional form tracker. When form
    is None, form features are 0 (neutral) — useful for pure-Elo prediction.
    """
    home, away = canonical(home), canonical(away)
    hfa = 0.0 if neutral else elo.home_advantage
    if form is None:
        fp_diff = fg_diff = 0.0
    else:
        fp_diff = (form.avg_points(home) or 0.0) - (form.avg_points(away) or 0.0)
        fg_diff = (form.avg_gd(home) or 0.0) - (form.avg_gd(away) or 0.0)
    return {
        "elo_diff": elo.get(home) - elo.get(away) + hfa,
        "form_points_diff": fp_diff,
        "form_gd_diff": fg_diff,
        "neutral": int(neutral),
    }


def fit_state(matches: pd.DataFrame, form_window: int = DEFAULT_FORM_WINDOW):
    """Replay all matches and return the final (elo, form) state.

    Use this to obtain up-to-date Elo + form for predicting future matches.
    """
    matches = matches.sort_values("date")
    elo = EloRatingSystem()
    form = _FormTracker(form_window)
    for row in matches.itertuples(index=False):
        home, away = canonical(row.home_team), canonical(row.away_team)
        hs, as_ = int(row.home_score), int(row.away_score)
        elo.update_match(home, away, hs, as_, bool(getattr(row, "neutral", False)),
                         weight=match_weight(getattr(row, "tournament", None)))
        form.update(home, away, hs, as_)
    return elo, form
