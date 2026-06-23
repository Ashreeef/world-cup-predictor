"""Probability calibration analysis.

Accuracy asks "is the top pick right?". Calibration asks the deeper question:
"when the model says 70%, does it happen ~70% of the time?" — which is what
makes probabilistic predictions (title odds, qualification %) trustworthy.

    Brier score      mean squared error of probabilities (0 = perfect, lower better)
    reliability      predicted probability vs observed frequency, per class
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from worldcup.models.baseline import DEFAULT_CUTOFF, TARGET, make_pipeline, time_split
from worldcup.features.build import FEATURE_COLUMNS


def multiclass_brier(y_true, proba: np.ndarray, classes: list[str]) -> float:
    """Multiclass Brier score: mean over samples of sum_k (p_k - y_k)^2.

    0 = perfect, 2 = worst possible (all confidence on the wrong class).
    """
    idx = {c: i for i, c in enumerate(classes)}
    Y = np.zeros_like(proba, dtype=float)
    for row, c in enumerate(y_true):
        Y[row, idx[c]] = 1.0
    return float(np.mean(np.sum((proba - Y) ** 2, axis=1)))


def reliability_curve(y_true, proba: np.ndarray, classes: list[str], n_bins: int = 10) -> dict:
    """Per-class reliability data: {class: (pred_mean, observed_freq, counts)}."""
    y_true = np.asarray(y_true)
    bins = np.linspace(0, 1, n_bins + 1)
    out: dict[str, tuple] = {}
    for i, c in enumerate(classes):
        p = proba[:, i]
        actual = (y_true == c).astype(float)
        bucket = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
        pred_mean, obs, cnt = [], [], []
        for k in range(n_bins):
            mask = bucket == k
            if mask.sum() == 0:
                continue
            pred_mean.append(p[mask].mean())
            obs.append(actual[mask].mean())
            cnt.append(int(mask.sum()))
        out[c] = (np.array(pred_mean), np.array(obs), np.array(cnt))
    return out


def evaluate_calibration(features: pd.DataFrame, cutoff: str = DEFAULT_CUTOFF, n_bins: int = 10) -> dict:
    """Train the baseline on <cutoff and assess calibration on >=cutoff."""
    train, test = time_split(features, cutoff)
    pipe = make_pipeline().fit(train[FEATURE_COLUMNS], train[TARGET])
    classes = list(pipe.classes_)
    proba = pipe.predict_proba(test[FEATURE_COLUMNS])
    return {
        "brier": multiclass_brier(test[TARGET].values, proba, classes),
        "reliability": reliability_curve(test[TARGET].values, proba, classes, n_bins),
        "classes": classes,
        "n_test": len(test),
    }


def _main() -> None:
    from worldcup.config import REPORTS_DIR
    from worldcup.data.load import load_results
    from worldcup.features.build import build_features
    from worldcup.visualization import plots

    features = build_features(load_results())
    result = evaluate_calibration(features)
    print(f"Brier score: {result['brier']:.3f}  (n_test={result['n_test']}, classes={result['classes']})")

    import matplotlib.pyplot as plt

    plots.set_style()
    fig, ax = plt.subplots(figsize=(6, 6))
    plots.plot_reliability(result["reliability"], ax=ax)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "calibration.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved reliability plot to {out}")


if __name__ == "__main__":
    _main()
