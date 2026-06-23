"""Baseline match-outcome model: multinomial logistic regression.

Two clearly separated jobs:

  evaluate_baseline()         Honest measurement. Trains on matches BEFORE a
                              cutoff, tests on matches AFTER it (time-based
                              split). Reports accuracy + log-loss vs naive
                              baselines. Never tests on training data.

  train_production_model()    The real thing. Trains on ALL matches (incl. the
                              latest 2022-2026 results) and saves to artifacts/.
                              This is what predicts WC2026. Because Elo + form
                              are recency-aware, recent performance dominates.

Why logistic regression as the baseline? It outputs proper probabilities
P(home win / draw / away win), which the Monte Carlo simulator needs, and it's
simple enough to be a trustworthy "bar to beat" for XGBoost in Phase 8.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from worldcup.config import ARTIFACTS_DIR
from worldcup.features.build import FEATURE_COLUMNS

TARGET = "result"
DEFAULT_CUTOFF = "2022-01-01"
MODEL_PATH: Path = ARTIFACTS_DIR / "baseline_model.joblib"


def time_split(features: pd.DataFrame, cutoff: str = DEFAULT_CUTOFF):
    """Split by date: train strictly before cutoff, test on/after it."""
    train = features[features["date"] < cutoff]
    test = features[features["date"] >= cutoff]
    return train, test


def make_pipeline() -> Pipeline:
    """Standardize features, then multinomial logistic regression."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def evaluate_baseline(features: pd.DataFrame, cutoff: str = DEFAULT_CUTOFF) -> dict:
    """Train on <cutoff, evaluate on >=cutoff. Returns a metrics dict."""
    train, test = time_split(features, cutoff)
    pipe = make_pipeline().fit(train[FEATURE_COLUMNS], train[TARGET])

    proba = pipe.predict_proba(test[FEATURE_COLUMNS])
    preds = pipe.predict(test[FEATURE_COLUMNS])

    acc = accuracy_score(test[TARGET], preds)
    ll = log_loss(test[TARGET], proba, labels=list(pipe.classes_))

    # Reference baselines for context.
    always_home_acc = (test[TARGET] == "H").mean()
    uniform_ll = float(np.log(3))  # log-loss of guessing 1/3 each

    return {
        "n_train": len(train),
        "n_test": len(test),
        "accuracy": acc,
        "log_loss": ll,
        "baseline_always_home_acc": always_home_acc,
        "baseline_uniform_log_loss": uniform_ll,
        "classes": list(pipe.classes_),
        "pipeline": pipe,
    }


def train_production_model(features: pd.DataFrame, save: bool = True) -> dict:
    """Train on ALL matches (the model used for real WC2026 predictions)."""
    pipe = make_pipeline().fit(features[FEATURE_COLUMNS], features[TARGET])
    bundle = {
        "pipeline": pipe,
        "features": FEATURE_COLUMNS,
        "classes": list(pipe.classes_),
        "trained_through": str(features["date"].max())[:10],
        "n_matches": len(features),
        "model_type": "logistic_regression",
    }
    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, MODEL_PATH)
        print(f"Saved baseline model to {MODEL_PATH} (trained on {bundle['n_matches']} matches "
              f"through {bundle['trained_through']}).")
    return bundle


def load_model(path: Path | None = None) -> dict:
    path = path or MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Train it first: python -m worldcup.models.baseline")
    return joblib.load(path)


def predict_proba(bundle: dict, feature_row: dict) -> dict[str, float]:
    """Return {'H': p, 'D': p, 'A': p} for a single feature row."""
    X = pd.DataFrame([feature_row])[bundle["features"]]
    proba = bundle["pipeline"].predict_proba(X)[0]
    return {cls: float(p) for cls, p in zip(bundle["classes"], proba)}


def _main() -> None:
    from worldcup.data.load import load_results
    from worldcup.features.build import build_features

    features = build_features(load_results())

    print("=== Honest evaluation (time-based split) ===")
    m = evaluate_baseline(features)
    print(f"Train: {m['n_train']:,} matches (<{DEFAULT_CUTOFF})  |  Test: {m['n_test']:,} (>= {DEFAULT_CUTOFF})")
    print(f"Accuracy : {m['accuracy']:.3f}   (always-home baseline: {m['baseline_always_home_acc']:.3f})")
    print(f"Log-loss : {m['log_loss']:.3f}   (uniform 1/3 baseline: {m['baseline_uniform_log_loss']:.3f})")

    print("\n=== Production model (trained on ALL data) ===")
    train_production_model(features)


if __name__ == "__main__":
    _main()
