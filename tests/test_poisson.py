"""Tests for the Poisson score model."""

import numpy as np
import pandas as pd

from worldcup.models.poisson import (
    most_likely_score,
    outcome_probs,
    predict_lambdas,
    predict_match,
    score_matrix,
    train_poisson,
)


def _synthetic_features(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Goals drawn from Poisson whose mean rises with elo_diff."""
    rng = np.random.default_rng(seed)
    elo_diff = rng.normal(0, 200, n)
    lam_home = np.exp(0.2 + 0.002 * elo_diff)
    lam_away = np.exp(0.2 - 0.002 * elo_diff)
    return pd.DataFrame(
        {
            "date": pd.date_range("2005-01-01", periods=n, freq="W"),
            "elo_diff": elo_diff,
            "neutral": rng.integers(0, 2, n),
            "home_score": rng.poisson(lam_home),
            "away_score": rng.poisson(lam_away),
            "result": "H",  # unused by Poisson training
        }
    )


def test_score_matrix_is_a_distribution():
    m = score_matrix(1.5, 1.2)
    assert m.shape == (11, 11)
    assert abs(m.sum() - 1.0) < 1e-9
    assert (m >= 0).all()


def test_outcome_probs_sum_to_one_and_favour_stronger():
    strong_home = outcome_probs(score_matrix(2.5, 0.8))
    assert abs(sum(strong_home.values()) - 1.0) < 1e-9
    assert strong_home["H"] > strong_home["A"]


def test_equal_lambdas_symmetric():
    o = outcome_probs(score_matrix(1.3, 1.3))
    assert abs(o["H"] - o["A"]) < 1e-9  # symmetric -> equal win probs


def test_most_likely_score():
    i, j, p = most_likely_score(score_matrix(1.1, 0.9))
    assert isinstance(i, int) and isinstance(j, int)
    assert 0 < p < 1


def test_train_and_predict_lambdas_track_elo():
    bundle = train_poisson(_synthetic_features(), save=False)
    lh_fav, la_fav = predict_lambdas(bundle, elo_diff=300, neutral=1)
    lh_dog, la_dog = predict_lambdas(bundle, elo_diff=-300, neutral=1)
    # A strong favourite scores more and concedes less than a heavy underdog.
    assert lh_fav > lh_dog
    assert la_fav < la_dog


def test_predict_match_structure():
    bundle = train_poisson(_synthetic_features(), save=False)
    out = predict_match(bundle, {"elo_diff": 150, "neutral": 1})
    assert set(out["outcome"]) == {"H", "D", "A"}
    assert abs(sum(out["outcome"].values()) - 1.0) < 1e-9
    assert out["lambda_home"] > out["lambda_away"]
