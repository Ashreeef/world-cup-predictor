"""Advanced match-outcome model: gradient-boosted trees (XGBoost).

Same interface and same time-based test split as the logistic baseline, so the
two can be compared fairly. XGBoost captures non-linear feature interactions the
linear baseline cannot.

Overfitting control: modest depth/learning-rate + early stopping on a
chronological validation tail (the model chooses its own number of trees).

Production fit is two-stage, which matters for WC2026 freshness:
    Stage 1  fit on the earliest 90%, early-stop on the latest 10% -> best #trees
    Stage 2  REFIT on 100% of the data with that fixed #trees (no early stop)
so the most recent matches also train the final model.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

from worldcup.config import ARTIFACTS_DIR
from worldcup.features.build import FEATURE_COLUMNS
from worldcup.models.baseline import DEFAULT_CUTOFF, TARGET, time_split

# XGBoost needs integer labels. Keep a fixed, explicit mapping.
CLASS_TO_INT = {"H": 0, "D": 1, "A": 2}
INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}
CLASSES_ORDER = [INT_TO_CLASS[i] for i in range(3)]  # ['H', 'D', 'A']

MODEL_PATH: Path = ARTIFACTS_DIR / "xgb_model.joblib"


def _safe_log_loss(y_true, proba: np.ndarray) -> float:
    """log_loss that aligns our [H,D,A] proba columns to sklearn's expected
    lexicographic order ['A','D','H'] (sklearn ignores column order otherwise)."""
    sorted_labels = sorted(CLASSES_ORDER)
    col_idx = [CLASSES_ORDER.index(lbl) for lbl in sorted_labels]
    return log_loss(y_true, proba[:, col_idx], labels=sorted_labels)


def _make_xgb(n_estimators: int = 1000, early_stopping_rounds: int | None = 40) -> XGBClassifier:
    """XGBoost classifier with sensible, overfit-resistant defaults."""
    kwargs = dict(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=3,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    if early_stopping_rounds:
        kwargs["early_stopping_rounds"] = early_stopping_rounds
    return XGBClassifier(**kwargs)


def _chrono_val_split(train: pd.DataFrame, val_frac: float = 0.1):
    """Split training data into earliest-(1-frac) fit and latest-frac validation."""
    train = train.sort_values("date")
    n_val = max(1, int(len(train) * val_frac))
    return train.iloc[:-n_val], train.iloc[-n_val:]


def evaluate_xgb(features: pd.DataFrame, cutoff: str = DEFAULT_CUTOFF) -> dict:
    """Train on <cutoff (with a validation tail for early stopping), test on >=cutoff."""
    train, test = time_split(features, cutoff)
    fit_df, val_df = _chrono_val_split(train)

    model = _make_xgb()
    model.fit(
        fit_df[FEATURE_COLUMNS],
        fit_df[TARGET].map(CLASS_TO_INT),
        eval_set=[(val_df[FEATURE_COLUMNS], val_df[TARGET].map(CLASS_TO_INT))],
        verbose=False,
    )

    proba = model.predict_proba(test[FEATURE_COLUMNS])
    preds = [INT_TO_CLASS[i] for i in proba.argmax(axis=1)]

    return {
        "n_train": len(train),
        "n_test": len(test),
        "accuracy": accuracy_score(test[TARGET], preds),
        "log_loss": _safe_log_loss(test[TARGET], proba),
        "best_iteration": int(model.best_iteration),
        "baseline_always_home_acc": (test[TARGET] == "H").mean(),
        "baseline_uniform_log_loss": float(np.log(3)),
        "model": model,
    }


def train_production_xgb(features: pd.DataFrame, save: bool = True) -> dict:
    """Two-stage fit (find #trees, then refit on ALL data). Saves to artifacts/."""
    features = features.sort_values("date")

    # Stage 1: find the best number of trees via early stopping.
    fit_df, val_df = _chrono_val_split(features)
    probe = _make_xgb()
    probe.fit(
        fit_df[FEATURE_COLUMNS],
        fit_df[TARGET].map(CLASS_TO_INT),
        eval_set=[(val_df[FEATURE_COLUMNS], val_df[TARGET].map(CLASS_TO_INT))],
        verbose=False,
    )
    best_trees = int(probe.best_iteration) + 1

    # Stage 2: refit on 100% of the data with that fixed tree count.
    final = _make_xgb(n_estimators=best_trees, early_stopping_rounds=None)
    final.fit(features[FEATURE_COLUMNS], features[TARGET].map(CLASS_TO_INT), verbose=False)

    bundle = {
        "model": final,
        "features": FEATURE_COLUMNS,
        "classes_order": CLASSES_ORDER,
        "n_trees": best_trees,
        "trained_through": str(features["date"].max())[:10],
        "n_matches": len(features),
        "model_type": "xgboost",
    }
    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, MODEL_PATH)
        print(f"Saved XGBoost model to {MODEL_PATH} ({best_trees} trees, "
              f"{bundle['n_matches']} matches through {bundle['trained_through']}).")
    return bundle


def load_model(path: Path | None = None) -> dict:
    path = path or MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Train it first: python -m worldcup.models.xgboost_model")
    return joblib.load(path)


def predict_proba(bundle: dict, feature_row: dict) -> dict[str, float]:
    """Return {'H': p, 'D': p, 'A': p} for a single feature row."""
    X = pd.DataFrame([feature_row])[bundle["features"]]
    proba = bundle["model"].predict_proba(X)[0]
    return {INT_TO_CLASS[i]: float(p) for i, p in enumerate(proba)}


def feature_importances(bundle: dict) -> dict[str, float]:
    """Map each feature to its XGBoost importance (gain-based, normalized)."""
    imp = bundle["model"].feature_importances_
    return {f: float(v) for f, v in zip(bundle["features"], imp)}


def _main() -> None:
    from worldcup.data.load import load_results
    from worldcup.features.build import build_features
    from worldcup.models.baseline import evaluate_baseline

    features = build_features(load_results())

    base = evaluate_baseline(features)
    xgb = evaluate_xgb(features)

    print("=== Logistic vs XGBoost (same time-based test set) ===")
    print(f"{'metric':<12}{'logistic':>12}{'xgboost':>12}")
    print(f"{'accuracy':<12}{base['accuracy']:>12.3f}{xgb['accuracy']:>12.3f}")
    print(f"{'log_loss':<12}{base['log_loss']:>12.3f}{xgb['log_loss']:>12.3f}")
    print(f"\nXGBoost stopped at {xgb['best_iteration']} trees.")

    print("\n=== Training production XGBoost (all data) ===")
    bundle = train_production_xgb(features)
    print("\nFeature importances:")
    for f, v in sorted(feature_importances(bundle).items(), key=lambda kv: -kv[1]):
        print(f"  {f:<18}{v:.3f}")


if __name__ == "__main__":
    _main()
