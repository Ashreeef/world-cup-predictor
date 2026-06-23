"""Poisson score model — probabilities for every scoreline.

Phase 5 showed goals are ~Poisson. Here we model each side's expected goals (λ)
as a function of the Elo gap, via two Poisson regressions:

    log(λ_home) = a0 + a1 * elo_diff + a2 * neutral
    log(λ_away) = b0 + b1 * elo_diff + b2 * neutral

Given λ_home and λ_away, the probability of an exact score (i, j) is

    P(i, j) = Poisson(i; λ_home) * Poisson(j; λ_away)

Arrange these into a (max_goals+1) x (max_goals+1) "score matrix" and we can read
off win/draw/loss, the most likely score, over/under totals — all consistent.
This matrix is what the Monte Carlo simulator samples from in Phase 10.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from worldcup.config import ARTIFACTS_DIR
from worldcup.models.baseline import DEFAULT_CUTOFF, time_split

POISSON_FEATURES = ["elo_diff", "neutral"]
MAX_GOALS = 10
MODEL_PATH: Path = ARTIFACTS_DIR / "poisson_model.joblib"


# ── Core probability math ─────────────────────────────────────────────────────────
def score_matrix(lambda_home: float, lambda_away: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """Matrix M where M[i, j] = P(home scores i, away scores j)."""
    goals = np.arange(max_goals + 1)
    home_pmf = poisson.pmf(goals, lambda_home)
    away_pmf = poisson.pmf(goals, lambda_away)
    matrix = np.outer(home_pmf, away_pmf)
    return matrix / matrix.sum()  # renormalize the truncated tail


def outcome_probs(matrix: np.ndarray) -> dict[str, float]:
    """Collapse a score matrix into {'H', 'D', 'A'} probabilities."""
    home_win = np.tril(matrix, -1).sum()  # i > j
    away_win = np.triu(matrix, 1).sum()  # i < j
    draw = np.trace(matrix)
    return {"H": float(home_win), "D": float(draw), "A": float(away_win)}


def most_likely_score(matrix: np.ndarray) -> tuple[int, int, float]:
    """Return (home_goals, away_goals, probability) of the modal scoreline."""
    i, j = np.unravel_index(matrix.argmax(), matrix.shape)
    return int(i), int(j), float(matrix[i, j])


# ── Model ─────────────────────────────────────────────────────────────────────────
def predict_lambdas(bundle: dict, elo_diff: float, neutral: int) -> tuple[float, float]:
    """Expected goals (λ_home, λ_away) for a matchup."""
    X = pd.DataFrame([{"elo_diff": elo_diff, "neutral": neutral}])[POISSON_FEATURES]
    lh = float(bundle["home_model"].predict(X)[0])
    la = float(bundle["away_model"].predict(X)[0])
    return lh, la


def predict_match(bundle: dict, feature_row: dict, max_goals: int = MAX_GOALS) -> dict:
    """Full prediction for one match: λ's, outcome probs, and top scorelines."""
    lh, la = predict_lambdas(bundle, feature_row["elo_diff"], int(feature_row["neutral"]))
    matrix = score_matrix(lh, la, max_goals)
    i, j, p = most_likely_score(matrix)
    return {
        "lambda_home": lh,
        "lambda_away": la,
        "outcome": outcome_probs(matrix),
        "most_likely_score": {"home": i, "away": j, "prob": p},
        "matrix": matrix,
    }


def _poisson_pipeline() -> Pipeline:
    """Standardize features (essential — elo_diff is on a ~100s scale, and the
    Poisson log-link overflows on unscaled inputs), then Poisson regression."""
    return Pipeline(
        [("scaler", StandardScaler()), ("glm", PoissonRegressor(alpha=1e-6, max_iter=1000))]
    )


def train_poisson(features: pd.DataFrame, save: bool = True) -> dict:
    """Fit home & away Poisson regressions on ALL data. Saves to artifacts/."""
    X = features[POISSON_FEATURES]
    home_model = _poisson_pipeline().fit(X, features["home_score"])
    away_model = _poisson_pipeline().fit(X, features["away_score"])
    bundle = {
        "home_model": home_model,
        "away_model": away_model,
        "features": POISSON_FEATURES,
        "trained_through": str(features["date"].max())[:10],
        "n_matches": len(features),
        "model_type": "poisson",
    }
    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, MODEL_PATH)
        print(f"Saved Poisson model to {MODEL_PATH} ({bundle['n_matches']} matches "
              f"through {bundle['trained_through']}).")
    return bundle


def evaluate_poisson(features: pd.DataFrame, cutoff: str = DEFAULT_CUTOFF) -> dict:
    """Train on <cutoff, evaluate the implied W/D/L on >=cutoff (comparable to classifiers)."""
    train, test = time_split(features, cutoff)
    bundle = train_poisson(train, save=False)

    lh = bundle["home_model"].predict(test[POISSON_FEATURES])
    la = bundle["away_model"].predict(test[POISSON_FEATURES])

    sorted_labels = ["A", "D", "H"]  # sklearn log_loss expects lexicographic columns
    proba, preds = [], []
    for h, a in zip(lh, la):
        o = outcome_probs(score_matrix(h, a))
        proba.append([o["A"], o["D"], o["H"]])
        preds.append(max(o, key=o.get))

    return {
        "n_train": len(train),
        "n_test": len(test),
        "accuracy": accuracy_score(test["result"], preds),
        "log_loss": log_loss(test["result"], np.array(proba), labels=sorted_labels),
        "mean_goals_pred": float((lh + la).mean()),
        "mean_goals_actual": float((test["home_score"] + test["away_score"]).mean()),
    }


def load_model(path: Path | None = None) -> dict:
    path = path or MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Train it first: python -m worldcup.models.poisson")
    return joblib.load(path)


def _main() -> None:
    from worldcup.data.load import load_results
    from worldcup.features.build import build_features

    features = build_features(load_results())

    print("=== Poisson model - W/D/L evaluation (time split) ===")
    m = evaluate_poisson(features)
    print(f"accuracy {m['accuracy']:.3f} | log_loss {m['log_loss']:.3f}")
    print(f"mean goals/match  predicted {m['mean_goals_pred']:.2f}  vs actual {m['mean_goals_actual']:.2f}")

    print("\n=== Training production Poisson model ===")
    bundle = train_poisson(features)

    # Demo: a strong favourite at a neutral venue.
    demo = predict_match(bundle, {"elo_diff": 200.0, "neutral": 1})
    print(f"\nDemo matchup (elo_diff=200, neutral): lam_home={demo['lambda_home']:.2f}, lam_away={demo['lambda_away']:.2f}")
    print(f"  P(H/D/A) = {demo['outcome']}")
    s = demo["most_likely_score"]
    print(f"  Most likely score: {s['home']}-{s['away']} ({s['prob']:.1%})")


if __name__ == "__main__":
    _main()
