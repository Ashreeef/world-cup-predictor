"""Dixon-Coles score model.

The classic football scoreline model (Dixon & Coles, 1997). Each team gets an
attack and a defence strength; there's a home-advantage term, optional time-
decay (recent matches weighted more), and a low-score correction (rho) that
fixes independent Poisson's tendency to under-predict 0-0 / 1-1 draws.

    log(lambda_home) = intercept + attack[home] + defence[away] + home_adv*(not neutral)
    log(lambda_away) = intercept + attack[away] + defence[home]
    P(x, y) = Poisson(x; lambda_home) * Poisson(y; lambda_away) * tau(x, y)

Trade-off vs the Elo-Poisson model: DC is more accurate on exact scores but keys
off team identity, so it does not auto-update with live Elo. It is offered as a
standalone / comparison model; the Elo-Poisson remains the dynamic engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

from worldcup.data.teams import canonical
from worldcup.models.poisson import most_likely_score, outcome_probs

MAX_GOALS = 10


def _tau(x, y, lh, la, rho):
    """Dixon-Coles low-score correction (vectorized over arrays)."""
    x, y = np.asarray(x), np.asarray(y)
    out = np.ones_like(lh, dtype=float)
    out = np.where((x == 0) & (y == 0), 1 - lh * la * rho, out)
    out = np.where((x == 0) & (y == 1), 1 + lh * rho, out)
    out = np.where((x == 1) & (y == 0), 1 + la * rho, out)
    out = np.where((x == 1) & (y == 1), 1 - rho, out)
    return out


class DixonColesModel:
    def __init__(self, since: str = "2014-01-01", min_matches: int = 25,
                 half_life_days: float | None = 730.0, alpha: float = 1e-3) -> None:
        self.since = since
        self.min_matches = min_matches
        self.half_life_days = half_life_days
        self.alpha = alpha

    def fit(self, matches: pd.DataFrame) -> "DixonColesModel":
        df = matches.copy()
        df = df[df["date"] >= pd.Timestamp(self.since)].copy()
        df["home_team"] = df["home_team"].map(canonical)
        df["away_team"] = df["away_team"].map(canonical)

        # Keep teams with enough matches (stable strength estimates).
        counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
        keep = set(counts[counts >= self.min_matches].index)
        df = df[df["home_team"].isin(keep) & df["away_team"].isin(keep)].reset_index(drop=True)
        self.teams = sorted(keep)

        # Long format: one row per (attacking team, defending team, goals).
        n = len(df)
        long = pd.DataFrame(
            {
                "attack": pd.concat([df["home_team"], df["away_team"]], ignore_index=True),
                "defend": pd.concat([df["away_team"], df["home_team"]], ignore_index=True),
                "goals": pd.concat([df["home_score"], df["away_score"]], ignore_index=True),
                "home": np.concatenate([(~df["neutral"]).astype(int), np.zeros(n, dtype=int)]),
                "date": pd.concat([df["date"], df["date"]], ignore_index=True),
            }
        )

        X = pd.concat(
            [pd.get_dummies(long["attack"], prefix="atk"),
             pd.get_dummies(long["defend"], prefix="def"),
             long["home"]],
            axis=1,
        ).astype(float)

        weights = None
        if self.half_life_days:
            age = (long["date"].max() - long["date"]).dt.days
            weights = 0.5 ** (age / self.half_life_days)

        model = PoissonRegressor(alpha=self.alpha, max_iter=3000).fit(X, long["goals"], sample_weight=weights)

        coefs = dict(zip(X.columns, model.coef_))
        self.intercept = float(model.intercept_)
        self.attack = {t: coefs.get(f"atk_{t}", 0.0) for t in self.teams}
        self.defence = {t: coefs.get(f"def_{t}", 0.0) for t in self.teams}
        self.home_adv = coefs.get("home", 0.0)
        self.rho = self._fit_rho(df)
        return self

    def _lambdas_array(self, df: pd.DataFrame):
        atk = df["home_team"].map(self.attack).fillna(0.0).to_numpy()
        dfd = df["away_team"].map(self.defence).fillna(0.0).to_numpy()
        atk_a = df["away_team"].map(self.attack).fillna(0.0).to_numpy()
        dfd_h = df["home_team"].map(self.defence).fillna(0.0).to_numpy()
        home = (~df["neutral"]).to_numpy().astype(float)
        lh = np.exp(self.intercept + atk + dfd + self.home_adv * home)
        la = np.exp(self.intercept + atk_a + dfd_h)
        return lh, la

    def _fit_rho(self, df: pd.DataFrame) -> float:
        lh, la = self._lambdas_array(df)
        x, y = df["home_score"].to_numpy(), df["away_score"].to_numpy()
        base = poisson.logpmf(x, lh) + poisson.logpmf(y, la)

        def nll(rho):
            t = np.clip(_tau(x, y, lh, la, rho), 1e-12, None)
            return -np.sum(base + np.log(t))

        return float(minimize_scalar(nll, bounds=(-0.2, 0.2), method="bounded").x)

    def predict_lambdas(self, home: str, away: str, neutral: bool = True) -> tuple[float, float]:
        home, away = canonical(home), canonical(away)
        h = 0.0 if neutral else self.home_adv
        lh = float(np.exp(self.intercept + self.attack.get(home, 0.0) + self.defence.get(away, 0.0) + h))
        la = float(np.exp(self.intercept + self.attack.get(away, 0.0) + self.defence.get(home, 0.0)))
        return lh, la

    def score_matrix(self, home: str, away: str, neutral: bool = True, max_goals: int = MAX_GOALS) -> np.ndarray:
        lh, la = self.predict_lambdas(home, away, neutral)
        goals = np.arange(max_goals + 1)
        base = np.outer(poisson.pmf(goals, lh), poisson.pmf(goals, la))
        xx, yy = np.meshgrid(goals, goals, indexing="ij")
        corrected = base * _tau(xx, yy, lh, la, self.rho)
        corrected = np.clip(corrected, 0, None)
        return corrected / corrected.sum()

    def predict_match(self, home: str, away: str, neutral: bool = True, max_goals: int = MAX_GOALS) -> dict:
        matrix = self.score_matrix(home, away, neutral, max_goals)
        i, j, p = most_likely_score(matrix)
        return {"outcome": outcome_probs(matrix), "most_likely_score": {"home": i, "away": j, "prob": p}, "matrix": matrix}


def _main() -> None:
    from worldcup.data.load import load_results
    from worldcup.features.build import build_features
    from worldcup.models.poisson import score_matrix as poisson_matrix
    from worldcup.models.poisson import train_poisson

    matches = load_results()
    print("Fitting Dixon-Coles ...")
    dc = DixonColesModel().fit(matches)
    print(f"  teams modelled: {len(dc.teams)}, home_adv={dc.home_adv:.3f}, rho={dc.rho:.3f}")

    # Exact-score log-likelihood on a recent test slice: DC vs Elo-Poisson.
    feats = build_features(matches)
    test = feats[feats["date"] >= "2022-01-01"]
    pois = train_poisson(feats[feats["date"] < "2022-01-01"], save=False)

    dc_ll, po_ll, n = 0.0, 0.0, 0
    for r in test.itertuples():
        home_c, away_c = canonical(r.home_team), canonical(r.away_team)
        if home_c not in dc.teams or away_c not in dc.teams:
            continue
        x, y = int(r.home_score), int(r.away_score)
        if x > MAX_GOALS or y > MAX_GOALS:
            continue
        dc_ll += np.log(max(dc.score_matrix(home_c, away_c, neutral=bool(r.neutral))[x, y], 1e-12))
        lh = pois["home_model"].predict(pd.DataFrame([{"elo_diff": r.elo_diff, "neutral": r.neutral}]))[0]
        la = pois["away_model"].predict(pd.DataFrame([{"elo_diff": r.elo_diff, "neutral": r.neutral}]))[0]
        po_ll += np.log(max(poisson_matrix(lh, la)[x, y], 1e-12))
        n += 1

    print(f"\nExact-score avg log-likelihood on {n} test matches (higher = better):")
    print(f"  Dixon-Coles : {dc_ll / n:.4f}")
    print(f"  Elo-Poisson : {po_ll / n:.4f}")


if __name__ == "__main__":
    _main()
