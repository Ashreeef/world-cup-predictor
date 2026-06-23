"""Tests for the XGBoost model on a separable synthetic feature table."""

import numpy as np
import pandas as pd

from worldcup.models.xgboost_model import (
    evaluate_xgb,
    feature_importances,
    load_model,
    predict_proba,
    train_production_xgb,
)


def _synthetic_features(n: int = 1200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    elo_diff = rng.normal(0, 200, n)
    result = np.where(elo_diff > 80, "H", np.where(elo_diff < -80, "A", "D"))
    return pd.DataFrame(
        {
            "date": pd.date_range("2005-01-01", periods=n, freq="W"),
            "elo_diff": elo_diff,
            "form_points_diff": elo_diff / 100 + rng.normal(0, 0.5, n),
            "form_gd_diff": elo_diff / 80 + rng.normal(0, 0.5, n),
            "neutral": rng.integers(0, 2, n),
            "result": result,
        }
    )


def test_xgb_beats_naive_baselines():
    m = evaluate_xgb(_synthetic_features(), cutoff="2020-01-01")
    assert m["accuracy"] > m["baseline_always_home_acc"]
    assert m["log_loss"] < m["baseline_uniform_log_loss"]
    assert m["best_iteration"] >= 0


def test_train_save_load_predict(tmp_path, monkeypatch):
    import worldcup.models.xgboost_model as xm

    monkeypatch.setattr(xm, "MODEL_PATH", tmp_path / "xgb.joblib")
    bundle = train_production_xgb(_synthetic_features(), save=True)
    assert (tmp_path / "xgb.joblib").exists()

    reloaded = load_model(tmp_path / "xgb.joblib")
    proba = predict_proba(reloaded, {"elo_diff": 350, "form_points_diff": 3, "form_gd_diff": 4, "neutral": 1})
    assert abs(sum(proba.values()) - 1.0) < 1e-6
    assert proba["H"] == max(proba.values())


def test_feature_importance_keys_and_dominance():
    bundle = train_production_xgb(_synthetic_features(), save=False)
    imp = feature_importances(bundle)
    assert set(imp.keys()) == set(bundle["features"])
    # elo_diff drives the synthetic labels -> should be the top feature.
    assert max(imp, key=imp.get) == "elo_diff"
