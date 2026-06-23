"""Football Elo rating system (margin-of-victory + home advantage).

This is the "World Football Elo" variant used by eloratings.net:

    expected_home = 1 / (1 + 10 ** ((R_away - R_home - H) / 400))
    R_new        = R_old + K * G * (W - E)

where
    H  home advantage in Elo points (0 at a neutral venue)
    K  base sensitivity (how fast ratings move)
    G  goal-difference multiplier (margin of victory)
    W  actual result for the home team (1 win / 0.5 draw / 0 loss)
    E  expected result for the home team

The update is zero-sum (home gains exactly what away loses) and incremental
(one match updates two teams), which is exactly what the live pipeline needs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from worldcup.config import ARTIFACTS_DIR
from worldcup.data.teams import canonical

# Default hyper-parameters (tunable later in Phase 14).
BASE_RATING = 1500.0
K_FACTOR = 40.0
HOME_ADVANTAGE = 100.0

RATINGS_PATH: Path = ARTIFACTS_DIR / "elo_ratings.json"


def match_weight(tournament: str | None) -> float:
    """Match-importance multiplier on the K-factor (à la eloratings.net).

    Friendlies move ratings little; World Cup matches move them most.
    """
    if not tournament:
        return 1.0
    t = tournament.lower()
    if "friendly" in t:
        return 0.5
    if "qualification" in t or "qualifier" in t:
        return 1.2
    if "nations league" in t:
        return 1.2
    if "world cup" in t:  # the finals themselves (qualification handled above)
        return 2.0
    # Major continental finals and other senior competitive tournaments.
    major = ("euro", "copa", "african cup", "afc asian", "gold cup", "confederations", "nations cup")
    if any(k in t for k in major):
        return 1.5
    return 1.0  # other competitive matches


def _goal_diff_multiplier(margin: int) -> float:
    """Margin-of-victory multiplier G (eloratings.net scheme)."""
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11 + margin) / 8.0


class EloRatingSystem:
    """Holds and updates team Elo ratings."""

    def __init__(
        self,
        base_rating: float = BASE_RATING,
        k_factor: float = K_FACTOR,
        home_advantage: float = HOME_ADVANTAGE,
    ) -> None:
        self.base_rating = base_rating
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings: dict[str, float] = {}
        self.n_matches: int = 0
        self.last_date: str | None = None

    # ── Access ──────────────────────────────────────────────────────────────────
    def get(self, team: str) -> float:
        """Current rating for a team (base rating if unseen)."""
        return self.ratings.get(canonical(team), self.base_rating)

    def expected_score(self, home: str, away: str, neutral: bool = False) -> float:
        """Expected score (0-1) for the home team."""
        hfa = 0.0 if neutral else self.home_advantage
        diff = self.get(away) - self.get(home) - hfa
        return 1.0 / (1.0 + 10 ** (diff / 400.0))

    # ── Update ──────────────────────────────────────────────────────────────────
    def update_match(
        self,
        home: str,
        away: str,
        home_score: int,
        away_score: int,
        neutral: bool = False,
        date: str | None = None,
        weight: float = 1.0,
    ) -> dict[str, float]:
        """Apply one finished match and return the rating changes.

        `weight` scales the K-factor by match importance (see match_weight()).
        """
        home, away = canonical(home), canonical(away)
        e_home = self.expected_score(home, away, neutral)

        if home_score > away_score:
            w_home = 1.0
        elif home_score < away_score:
            w_home = 0.0
        else:
            w_home = 0.5

        g = _goal_diff_multiplier(abs(home_score - away_score))
        change = self.k_factor * weight * g * (w_home - e_home)

        self.ratings[home] = self.get(home) + change
        self.ratings[away] = self.get(away) - change  # zero-sum

        self.n_matches += 1
        if date is not None:
            self.last_date = str(date)[:10]

        return {"home": home, "away": away, "home_change": change, "away_change": -change}

    def fit(self, matches: pd.DataFrame) -> "EloRatingSystem":
        """Process a whole results DataFrame in chronological order."""
        matches = matches.sort_values("date")
        for row in matches.itertuples(index=False):
            self.update_match(
                home=row.home_team,
                away=row.away_team,
                home_score=int(row.home_score),
                away_score=int(row.away_score),
                neutral=bool(getattr(row, "neutral", False)),
                date=getattr(row, "date", None),
                weight=match_weight(getattr(row, "tournament", None)),
            )
        return self

    def replay_predictions(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Walk forward through matches, recording each PRE-match prediction.

        For every match (in date order) we record the expected home score
        *before* seeing the result, then apply the update. Returns a DataFrame
        with columns: expected_home, actual_home. Used for calibration (Phase 5)
        and as an Elo-only baseline (Phase 7).
        """
        rows = []
        for row in matches.sort_values("date").itertuples(index=False):
            neutral = bool(getattr(row, "neutral", False))
            e = self.expected_score(row.home_team, row.away_team, neutral)
            if row.home_score > row.away_score:
                w = 1.0
            elif row.home_score < row.away_score:
                w = 0.0
            else:
                w = 0.5
            rows.append((e, w))
            self.update_match(
                row.home_team, row.away_team, int(row.home_score), int(row.away_score), neutral,
                weight=match_weight(getattr(row, "tournament", None)),
            )
        return pd.DataFrame(rows, columns=["expected_home", "actual_home"])

    # ── Reporting ─────────────────────────────────────────────────────────────────
    def top(self, n: int = 20) -> pd.DataFrame:
        """Return the top-n teams by rating."""
        s = pd.Series(self.ratings, name="elo").sort_values(ascending=False)
        return s.head(n).round(1).rename_axis("team").reset_index()

    # ── Persistence ─────────────────────────────────────────────────────────────────
    def save(self, path: Path | None = None) -> None:
        path = path or RATINGS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "params": {
                "base_rating": self.base_rating,
                "k_factor": self.k_factor,
                "home_advantage": self.home_advantage,
            },
            "n_matches": self.n_matches,
            "last_date": self.last_date,
            "ratings": self.ratings,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | None = None) -> "EloRatingSystem":
        path = path or RATINGS_PATH
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Build it first: python -m worldcup.features.elo")
        payload = json.loads(path.read_text(encoding="utf-8"))
        p = payload.get("params", {})
        system = cls(
            base_rating=p.get("base_rating", BASE_RATING),
            k_factor=p.get("k_factor", K_FACTOR),
            home_advantage=p.get("home_advantage", HOME_ADVANTAGE),
        )
        system.ratings = {k: float(v) for k, v in payload["ratings"].items()}
        system.n_matches = payload.get("n_matches", 0)
        system.last_date = payload.get("last_date")
        return system


def build_current_ratings(save: bool = True) -> EloRatingSystem:
    """Fit Elo over all historical results and (optionally) save to artifacts/."""
    from worldcup.data.load import load_results

    df = load_results()
    system = EloRatingSystem().fit(df)
    if save:
        system.save()
        print(f"Saved {len(system.ratings)} team ratings to {RATINGS_PATH}")
        print(f"Processed {system.n_matches} matches (through {system.last_date}).")
    return system


if __name__ == "__main__":
    system = build_current_ratings()
    print("\nTop 20 teams by Elo:")
    print(system.top(20).to_string(index=False))
