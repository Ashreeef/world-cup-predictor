"""Monte Carlo tournament simulator for the 48-team World Cup (2026 format).

Each simulated match samples a scoreline from the Poisson model. We play the
full tournament (group stage -> knockout) many times and aggregate how often
each team reaches each stage and wins the title.

Results-aware: any matches already played (`played_matches`) are FIXED rather
than sampled, so the simulation is conditioned on what has actually happened —
teams get eliminated exactly as the real tournament progresses.

2026 specifics implemented:
    - 12 groups of 4; head-to-head tiebreaker (then overall GD, GF).
    - Knockout entrants: 12 winners + 12 runners-up + 8 best 3rd-placed.
    - Official Round-of-32 bracket: the exact winner-vs-runner and
      runner-vs-runner slots and the fixed left/right tree. The 8 third-placed
      teams are assigned to their winner slots avoiding a group-stage rematch
      (FIFA's full 495-combination table is not public-data here, so this is the
      faithful structural approximation).
    - No re-seeding after R32: win and advance in a fixed tree.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from itertools import groupby

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from worldcup.data.teams import canonical
from worldcup.data.wc2026 import get_group_teams
from worldcup.features.elo import EloRatingSystem
from worldcup.models.poisson import POISSON_FEATURES

STAGE_LEVEL = {
    "group": 0,
    "round_of_32": 1,
    "round_of_16": 2,
    "quarter_final": 3,
    "semi_final": 4,
    "final": 5,
    "champion": 6,
}
_KO_PROGRESSION = ["round_of_16", "quarter_final", "semi_final", "final", "champion"]

# Group-winner slots that face a 3rd-placed team in the Round of 32.
WINNER_THIRD_SLOTS = ["A", "B", "C", "D", "G", "H", "K", "L"]


def assign_thirds(advancing_groups: list[str]) -> dict[str, str]:
    """Map each winner-vs-3rd slot to one advancing 3rd-place group.

    Bijection that avoids a group-stage rematch (a 3rd from group X is not sent
    to the Winner-X slot) where possible. Deterministic.
    """
    n = len(WINNER_THIRD_SLOTS)
    cost = np.array(
        [[1.0 if g == slot else 0.0 for g in advancing_groups] for slot in WINNER_THIRD_SLOTS]
    )
    rows, cols = linear_sum_assignment(cost)
    return {WINNER_THIRD_SLOTS[i]: advancing_groups[j] for i, j in zip(rows, cols)}


def build_official_bracket(winners, runners, third_team_by_group, assign) -> list[int]:
    """Return the 32 team indices in official bracket order (consecutive pairs
    are Round-of-32 matches; left half = first 16, right half = last 16)."""
    def W(g):
        return winners[g]

    def R(g):
        return runners[g]

    def T(slot):
        return third_team_by_group[assign[slot]]

    matches = [
        # ── Left half (feeds Semi-final 1) ──
        (W("A"), T("A")), (R("B"), R("E")), (W("E"), R("A")), (W("F"), R("C")),
        (W("C"), T("C")), (R("D"), R("F")), (W("D"), T("D")), (W("G"), T("G")),
        # ── Right half (feeds Semi-final 2), mirroring the left structure ──
        (W("B"), T("B")), (R("I"), R("K")), (W("I"), R("G")), (W("J"), R("H")),
        (W("K"), T("K")), (R("J"), R("L")), (W("L"), T("L")), (W("H"), T("H")),
    ]
    return [t for m in matches for t in m]


class TournamentSimulator:
    def __init__(
        self,
        elo: EloRatingSystem,
        poisson_bundle: dict,
        groups: dict[str, list[str]] | None = None,
        seed: int | None = None,
        played_matches: pd.DataFrame | None = None,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.groups = groups or get_group_teams()
        self.teams = [t for teams in self.groups.values() for t in teams]
        self.idx = {t: i for i, t in enumerate(self.teams)}
        self._build_played(played_matches)
        self._precompute_lambdas(elo, poisson_bundle)

    def _build_played(self, played_matches: pd.DataFrame | None) -> None:
        """Lookup {(home_idx, away_idx): (home_goals, away_goals)} of fixed results."""
        self.played: dict[tuple[int, int], tuple[int, int]] = {}
        if played_matches is None:
            return
        for r in played_matches.itertuples(index=False):
            h, a = canonical(r.home_team), canonical(r.away_team)
            if h in self.idx and a in self.idx:
                self.played[(self.idx[h], self.idx[a])] = (int(r.home_score), int(r.away_score))

    def _precompute_lambdas(self, elo: EloRatingSystem, bundle: dict) -> None:
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
        self.p_win = 1.0 / (1.0 + 10 ** ((elos[None, :] - elos[:, None]) / 400.0))

    # ── Match ─────────────────────────────────────────────────────────────────────
    def _play(self, i: int, j: int, knockout: bool = False) -> tuple[int, int, int | None]:
        if (i, j) in self.played:
            gi, gj = self.played[(i, j)]
        elif (j, i) in self.played:
            gj, gi = self.played[(j, i)]
        else:
            gi = int(self.rng.poisson(self.LH[i, j]))
            gj = int(self.rng.poisson(self.LA[i, j]))
        if gi > gj:
            return gi, gj, i
        if gj > gi:
            return gi, gj, j
        if knockout:  # extra time / penalties -> Elo-weighted coin flip
            return gi, gj, (i if self.rng.random() < self.p_win[i, j] else j)
        return gi, gj, None

    # ── Group stage with head-to-head tiebreaker ────────────────────────────────────
    def _simulate_group(self, tids: list[int]) -> list[tuple[int, int, int, int]]:
        results = []
        for a, b in itertools.combinations(tids, 2):
            ga, gb, _ = self._play(a, b)
            results.append((a, b, ga, gb))
        return self._rank_group(tids, results)

    def _rank_group(self, tids, results):
        pts, gd, gf = defaultdict(int), defaultdict(int), defaultdict(int)
        for a, b, ga, gb in results:
            gf[a] += ga
            gf[b] += gb
            gd[a] += ga - gb
            gd[b] += gb - ga
            if ga > gb:
                pts[a] += 3
            elif gb > ga:
                pts[b] += 3
            else:
                pts[a] += 1
                pts[b] += 1

        def h2h_key(team, cluster):
            """Mini-table (points, GD, GF) among only the tied teams."""
            mp = mg = mf = 0
            for a, b, ga, gb in results:
                if a in cluster and b in cluster:
                    if a == team:
                        mf += ga; mg += ga - gb; mp += 3 if ga > gb else (1 if ga == gb else 0)
                    elif b == team:
                        mf += gb; mg += gb - ga; mp += 3 if gb > ga else (1 if ga == gb else 0)
            return (mp, mg, mf)

        ordered = []
        teams_by_pts = sorted(tids, key=lambda t: pts[t], reverse=True)
        for _, cluster_iter in groupby(teams_by_pts, key=lambda t: pts[t]):
            cluster = list(cluster_iter)
            if len(cluster) > 1:
                cset = set(cluster)
                cluster.sort(
                    key=lambda t: (h2h_key(t, cset), gd[t], gf[t], self.rng.random()),
                    reverse=True,
                )
            ordered.extend(cluster)
        return [(t, pts[t], gd[t], gf[t]) for t in ordered]

    # ── One full tournament ─────────────────────────────────────────────────────────
    def simulate_tournament(self) -> dict[int, str]:
        winners, runners, third_recs, third_group = {}, {}, [], {}
        stage = {}
        for g, team_names in self.groups.items():
            ranked = self._simulate_group([self.idx[t] for t in team_names])
            for rec in ranked:
                stage[rec[0]] = "group"
            winners[g], runners[g] = ranked[0][0], ranked[1][0]
            third_recs.append(ranked[2])
            third_group[ranked[2][0]] = g

        # 8 best third-placed teams (virtual table: pts, GD, GF, then random).
        best = sorted(third_recs, key=lambda r: (r[1], r[2], r[3], self.rng.random()), reverse=True)[:8]
        advancing_groups = [third_group[rec[0]] for rec in best]
        third_team_by_group = {third_group[rec[0]]: rec[0] for rec in best}
        assign = assign_thirds(advancing_groups)

        bracket = build_official_bracket(winners, runners, third_team_by_group, assign)
        for t in bracket:
            stage[t] = "round_of_32"

        current = bracket
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
        n = len(self.teams)
        levels = np.zeros((n, 7), dtype=np.int64)
        for _ in range(n_sims):
            for t, st in self.simulate_tournament().items():
                levels[t, STAGE_LEVEL[st]] += 1
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


def load_wc2026_played(extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """Played WC2026 matches between known WC teams (from the dataset + optional extra)."""
    from worldcup.data.load import load_results

    df = load_results()
    groups = get_group_teams()
    team_set = {canonical(t) for ts in groups.values() for t in ts}
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"] >= "2026-01-01")].copy()
    cols = ["home_team", "away_team", "home_score", "away_score"]
    wc = wc[wc["home_team"].map(lambda x: canonical(x) in team_set)
            & wc["away_team"].map(lambda x: canonical(x) in team_set)][cols]
    if extra is not None and len(extra):
        wc = pd.concat([wc, extra[cols]], ignore_index=True)
    return wc.reset_index(drop=True)


def current_standings(played_matches: pd.DataFrame, groups: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Real group standings from the matches played so far (points, GD, GF, played)."""
    groups = groups or get_group_teams()
    t2g = {canonical(t): g for g, ts in groups.items() for t in ts}
    pts, gd, gf, pl = (defaultdict(int) for _ in range(4))
    for r in played_matches.itertuples(index=False):
        h, a = canonical(r.home_team), canonical(r.away_team)
        if h not in t2g or a not in t2g:
            continue
        hs, as_ = int(r.home_score), int(r.away_score)
        pl[h] += 1; pl[a] += 1
        gf[h] += hs; gf[a] += as_
        gd[h] += hs - as_; gd[a] += as_ - hs
        if hs > as_:
            pts[h] += 3
        elif as_ > hs:
            pts[a] += 3
        else:
            pts[h] += 1; pts[a] += 1
    rows = [
        {"group": g, "team": canonical(t), "played": pl[canonical(t)], "pts": pts[canonical(t)],
         "gd": gd[canonical(t)], "gf": gf[canonical(t)]}
        for g, ts in groups.items() for t in ts
    ]
    return pd.DataFrame(rows).sort_values(
        ["group", "pts", "gd", "gf"], ascending=[True, False, False, False]
    ).reset_index(drop=True)


def run_simulation(
    n_sims: int = 10000,
    seed: int | None = None,
    elo: EloRatingSystem | None = None,
    poisson_bundle: dict | None = None,
    played_matches: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run the simulation, building current Elo + Poisson from data if not given."""
    if elo is None or poisson_bundle is None:
        from worldcup.data.load import load_results
        from worldcup.features.build import build_features
        from worldcup.features.elo import build_current_ratings
        from worldcup.models.poisson import train_poisson

        if elo is None:
            elo = build_current_ratings(save=False)
        if poisson_bundle is None:
            poisson_bundle = train_poisson(build_features(load_results()), save=False)

    sim = TournamentSimulator(elo, poisson_bundle, seed=seed, played_matches=played_matches)
    return sim.run(n_sims)


if __name__ == "__main__":
    from worldcup.features.elo import EloRatingSystem as _Elo
    from worldcup.models.poisson import load_model as _load_poisson

    played = load_wc2026_played()
    print(f"Conditioning on {len(played)} played WC2026 matches.")
    table = run_simulation(n_sims=10000, seed=42, elo=_Elo.load(), poisson_bundle=_load_poisson(), played_matches=played)
    show = table.head(15).copy()
    for col in ["qualify", "round_of_16", "quarter_final", "semi_final", "final", "champion"]:
        show[col] = (show[col] * 100).round(1)
    print("\nTitle contenders (probabilities %, given results so far):")
    print(show.to_string(index=False))
