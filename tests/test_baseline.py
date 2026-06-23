"""Tests for the baseline model on a separable synthetic feature table."""

import numpy as np
import pandas as pd

from worldcup.models.baseline import (
    evaluate_baseline,
    load_model,
    predict_proba,
    time_split,
    train_production_model,
)


def _synthetic_features(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """elo_diff strongly determines result -> a learnable signal."""
    rng = np.random.default_rng(seed)
    elo_diff = rng.normal(0, 200, n)
    # Higher elo_diff -> more likely Home win; very negative -> Away win.
    result = np.where(elo_diff > 80, "H", np.where(elo_diff < -80, "A", "D"))
    return pd.DataFrame(
        {
            "date": pd.date_range("2010-01-01", periods=n, freq="W"),
            "elo_diff": elo_diff,
            "form_points_diff": elo_diff / 100 + rng.normal(0, 0.5, n),
            "form_gd_diff": elo_diff / 80 + rng.normal(0, 0.5, n),
            "neutral": rng.integers(0, 2, n),
            "result": result,
        }
    )


def test_time_split_is_chronological():
    df = _synthetic_features()
    train, test = time_split(df, cutoff="2018-01-01")
    assert train["date"].max() < pd.Timestamp("2018-01-01")
    assert test["date"].min() >= pd.Timestamp("2018-01-01")


def test_baseline_beats_naive_and_proba_valid():
    df = _synthetic_features()
    m = evaluate_baseline(df, cutoff="2018-01-01")
    # With a real signal, the model should beat always-home.
    assert m["accuracy"] > m["baseline_always_home_acc"]
    # And produce better (lower) log-loss than uniform guessing.
    assert m["log_loss"] < m["baseline_uniform_log_loss"]


def test_train_save_load_predict(tmp_path, monkeypatch):
    import worldcup.models.baseline as bl

    path = tmp_path / "m.joblib"
    monkeypatch.setattr(bl, "MODEL_PATH", path)

    bundle = train_production_model(_synthetic_features(), save=True)
    assert path.exists()

    reloaded = load_model(path)
    proba = predict_proba(reloaded, {"elo_diff": 300, "form_points_diff": 3, "form_gd_diff": 3, "neutral": 1})
    assert abs(sum(proba.values()) - 1.0) < 1e-9
    assert proba["H"] == max(proba.values())  # strong home favourite
