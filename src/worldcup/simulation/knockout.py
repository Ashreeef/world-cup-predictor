"""Knockout-stage analysis from the fixed Round-of-32 bracket.

Once the group stage ends, the 32 qualifiers and their bracket are known (no
more simulating group standings). This module loads that official bracket and
runs the knockout: per-match win probabilities (analytic) and per-team
advancement odds (Monte Carlo), all results-aware.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from worldcup.config import REFERENCE_DATA_DIR
from worldcup.data.teams import canonical
from worldcup.models.poisson import predict_match
from worldcup.simulation.simulator import TournamentSimulator

R32_CSV: Path = REFERENCE_DATA_DIR / "wc2026_knockout_r32.csv"

# Bracket tree: which R32 match indices (0-based) feed each later round, by side.
ROUND_NAMES = ["Round of 32", "Round of 16", "Quarter-finals", "Semi-finals", "Final"]


def load_r32_bracket(path: Path | None = None) -> list[tuple[str, str]]:
    """Return the 16 Round-of-32 matchups (home, away) in bracket order."""
    df = pd.read_csv(path or R32_CSV).sort_values("position")
    return [(canonical(r.home_team), canonical(r.away_team)) for r in df.itertuples(index=False)]


def bracket_team_names(bracket: list[tuple[str, str]] | None = None) -> list[str]:
    """Flatten the bracket to 32 team names in bracket order."""
    bracket = bracket or load_r32_bracket()
    return [t for pair in bracket for t in pair]


def r32_match_table(sim: TournamentSimulator, poisson_bundle: dict,
                    bracket: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    """Per-match table for the Round of 32: win probabilities + predicted score."""
    bracket = bracket or load_r32_bracket()
    rows = []
    for pos, (home, away) in enumerate(bracket, start=1):
        p_home = sim.match_win_prob(home, away)
        i, j = sim.idx[canonical(home)], sim.idx[canonical(away)]
        elo_diff = float(sim.team_elo[i] - sim.team_elo[j])
        pred = predict_match(poisson_bundle, {"elo_diff": elo_diff, "neutral": 1})
        s = pred["most_likely_score"]
        rows.append({
            "match": pos,
            "home": home,
            "away": away,
            "p_home": p_home,
            "p_away": 1 - p_home,
            "favorite": home if p_home >= 0.5 else away,
            "likely_score": f"{s['home']}-{s['away']}",
        })
    return pd.DataFrame(rows)


def knockout_predictions(
    elo, poisson_bundle: dict, played_matches: pd.DataFrame | None = None,
    n_sims: int = 10000, seed: int = 42, bracket: list[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, TournamentSimulator]:
    """Run the fixed-bracket knockout simulation; return (per-team odds, simulator)."""
    bracket = bracket or load_r32_bracket()
    sim = TournamentSimulator(elo, poisson_bundle, seed=seed, played_matches=played_matches)
    table = sim.run_fixed_knockout(bracket_team_names(bracket), n_sims=n_sims)
    return table, sim


def _main() -> None:
    from worldcup.features.elo import EloRatingSystem
    from worldcup.models.poisson import load_model

    bracket = load_r32_bracket()
    elo, poisson_bundle = EloRatingSystem.load(), load_model()
    table, sim = knockout_predictions(elo, poisson_bundle, n_sims=10000)

    print("Round of 32 - win probabilities:")
    mt = r32_match_table(sim, poisson_bundle, bracket)
    for r in mt.itertuples():
        print(f"  M{r.match:>2}  {r.home:>22} {r.p_home:5.1%} - {r.p_away:<5.1%} {r.away:<22} "
              f"(likely {r.likely_score})")

    print("\nChampionship odds (given the bracket):")
    top = table.head(10)[["team", "semi_final", "final", "champion"]].copy()
    for c in ["semi_final", "final", "champion"]:
        top[c] = (top[c] * 100).round(1)
    print(top.to_string(index=False))


if __name__ == "__main__":
    _main()
