"""Monte Carlo tournament simulator for the 48-team World Cup.

Each simulated match samples a scoreline from the Poisson model. We play the
full tournament (group stage -> knockout) many times and aggregate how often
each team reaches each stage and wins the title.

Format (2026):
    - 12 groups of 4, single round-robin (6 matches/group, 72 total)
    - Knockout entrants: 12 group winners + 12 runners-up + 8 best 3rd-placed
      -> Round of 32 -> R16 -> QF -> SF -> Final

Performance: expected goals (lambda) for every ordered team pair are computed
once up front, so simulating a match is just two Poisson draws.

Bracket note: the official R32 slotting depends on which 8 of the 12 third-
placed teams advance (a combination lookup table). We use a strength-seeded
single-elimination bracket instead — a standard, reproducible approximation.
Plugging in the exact official mapping is a Phase 14 refinement.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from worldcup.data.wc2026 import get_group_teams
from worldcup.features.elo import EloRatingSystem
from worldcup.models.poisson import POISSON_FEATURES

# Furthest stage reached -> ordinal level (for aggregation).
STAGE_LEVEL = {
    "group": 0,
    "round_of_32": 1,
    "round_of_16": 2,
    "quarter_final": 3,
    "semi_final": 4,
    "final": 5,
    "champion": 6,
}
# Stage the WINNER of each knockout round advances to (R32 entrants start at round_of_32).
_KO_PROGRESSION = ["round_of_16", "quarter_final", "semi_final", "final", "champion"]


def _seeding_order(n: int) -> list[int]:
    """Standard tournament seeding: bracket positions for seeds 1..n so that
    seed 1 meets seed n, the top two seeds can only meet in the final, etc."""
    order = [1, 2]
    while len(order) < n:
        m = len(order) * 2
        order = [x for s in order for x in (s, m + 1 - s)]
    return order


class TournamentSimulator:
    def __init__(
        self,
        elo: EloRatingSystem,
        poisson_bundle: dict,
        groups: dict[str, list[str]] | None = None,
        seed: int | None = None,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.groups = groups or get_group_teams()
        self.teams = [t for teams in self.groups.values() for t in teams]
        self.idx = {t: i for i, t in enumerate(self.teams)}

        self._precompute_lambdas(elo, poisson_bundle)

    def _precompute_lambdas(self, elo: EloRatingSystem, bundle: dict) -> None:
        """Build LH[i, j], LA[i, j] = expected goals for team i vs team j (neutral)."""
        n = len(self.teams)
        elos = np.array([elo.get(t) for t in self.teams])

        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        X = pd.DataFrame(
            [{"elo_diff": elos[i] - elos[j], "neutral": 1} for i, j in pairs]
        )[POISSON_FEATURES]
        lh = bundle["home_model"].predict(X)
        la = bundle["away_model"].predict(X)

        self.LH = np.zeros((n, n))
        self.LA = np.zeros((n, n))
        for k, (i, j) in enumerate(pairs):
            self.LH[i, j] = lh[k]
            self.LA[i, j] = la[k]

        # Elo win probability for i over j (neutral) — used for shootouts.
        self.p_win = 1.0 / (1.0 + 10 ** ((elos[None, :] - elos[:, None]) / 400.0))

    # ── Match ─────────────────────────────────────────────────────────────────────
    def _play(self, i: int, j: int, knockout: bool = False) -> tuple[int, int, int | None]:
        """Return (goals_i, goals_j, winner_idx_or_None)."""
        gi = self.rng.poisson(self.LH[i, j])
        gj = self.rng.poisson(self.LA[i, j])
        if gi > gj:
            return gi, gj, i
        if gj > gi:
            return gi, gj, j
        if knockout:  # decide level knockout by Elo-weighted shootout
            return gi, gj, (i if self.rng.random() < self.p_win[i, j] else j)
        return gi, gj, None

    # ── Group stage ─────────────────────────────────────────────────────────────────
    def _simulate_group(self, team_idxs: list[int]) -> list[tuple[int, int, int, int]]:
        """Round-robin; return team indices ranked best-first with (pts, gd, gf)."""
        pts = {t: 0 for t in team_idxs}
        gd = {t: 0 for t in team_idxs}
        gf = {t: 0 for t in team_idxs}
        for a, b in itertools.combinations(team_idxs, 2):
            ga, gb, w = self._play(a, b)
            gf[a] += ga
            gf[b] += gb
            gd[a] += ga - gb
            gd[b] += gb - ga
            if w is None:
                pts[a] += 1
                pts[b] += 1
            else:
                pts[w] += 3
        # Rank by points, then GD, then GF, then random tiebreak.
        return sorted(
            ((t, pts[t], gd[t], gf[t]) for t in team_idxs),
            key=lambda r: (r[1], r[2], r[3], self.rng.random()),
            reverse=True,
        )

    # ── One full tournament ─────────────────────────────────────────────────────────
    def simulate_tournament(self) -> dict[int, str]:
        """Play one tournament; return {team_idx: furthest stage reached}."""
        winners, runners, thirds = [], [], []
        for team_idxs in (
            [self.idx[t] for t in teams] for teams in self.groups.values()
        ):
            ranked = self._simulate_group(team_idxs)
            winners.append(ranked[0])
            runners.append(ranked[1])
            thirds.append(ranked[2])

        # 8 best third-placed teams by (pts, gd, gf).
        best_thirds = sorted(thirds, key=lambda r: (r[1], r[2], r[3], self.rng.random()), reverse=True)[:8]

        # Seed all 32 qualifiers: group winners best, then runners-up, then thirds;
        # within each tier by (pts, gd, gf).
        def seed_key(rank_tier, rec):
            return (rank_tier, -rec[1], -rec[2], -rec[3])

        qualifiers = (
            [("w", r) for r in winners] + [("r", r) for r in runners] + [("t", r) for r in best_thirds]
        )
        tier = {"w": 0, "r": 1, "t": 2}
        qualifiers.sort(key=lambda x: seed_key(tier[x[0]], x[1]))
        seeded = [rec[0] for _, rec in qualifiers]  # team idx, best seed first

        stage = {rec[0]: "group" for group in (winners, runners, thirds) for rec in group}
        for t in seeded:
            stage[t] = "round_of_32"

        # Build bracket by standard seeding and play down to a champion.
        order = _seeding_order(32)
        current = [seeded[s - 1] for s in order]
        for round_stage in _KO_PROGRESSION:
            nxt = []
            for k in range(0, len(current), 2):
                _, _, w = self._play(current[k], current[k + 1], knockout=True)
                stage[w] = round_stage
                nxt.append(w)
            current = nxt
        return stage

    # ── Many tournaments ─────────────────────────────────────────────────────────────
    def run(self, n_sims: int = 10000) -> pd.DataFrame:
        """Run n_sims tournaments; return per-team stage probabilities."""
        n = len(self.teams)
        levels = np.zeros((n, 7), dtype=np.int64)  # counts per stage level
        for _ in range(n_sims):
            for t, st in self.simulate_tournament().items():
                levels[t, STAGE_LEVEL[st]] += 1

        # Cumulative: P(reach at least this stage).
        reach = np.cumsum(levels[:, ::-1], axis=1)[:, ::-1] / n_sims
        df = pd.DataFrame(
            {
                "team": self.teams,
                "qualify": reach[:, 1],
                "round_of_16": reach[:, 2],
                "quarter_final": reach[:, 3],
                "semi_final": reach[:, 4],
                "final": reach[:, 5],
                "champion": reach[:, 6],
            }
        )
        return df.sort_values("champion", ascending=False).reset_index(drop=True)


def run_simulation(n_sims: int = 10000, seed: int | None = None) -> pd.DataFrame:
    """Convenience: build current Elo + Poisson from data and run the simulation."""
    from worldcup.data.load import load_results
    from worldcup.features.build import build_features
    from worldcup.features.elo import build_current_ratings
    from worldcup.models.poisson import train_poisson

    elo = build_current_ratings(save=False)
    poisson_bundle = train_poisson(build_features(load_results()), save=False)
    sim = TournamentSimulator(elo, poisson_bundle, seed=seed)
    return sim.run(n_sims)


if __name__ == "__main__":
    print("Running 10,000 tournament simulations ...")
    table = run_simulation(n_sims=10000, seed=42)
    show = table.head(15).copy()
    for col in ["qualify", "round_of_16", "quarter_final", "semi_final", "final", "champion"]:
        show[col] = (show[col] * 100).round(1)
    print("\nTitle contenders (probabilities %):")
    print(show.to_string(index=False))
