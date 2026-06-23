"""Tests for probability calibration metrics."""

import numpy as np

from worldcup.models.calibration import multiclass_brier, reliability_curve

CLASSES = ["A", "D", "H"]


def test_brier_perfect_is_zero():
    proba = np.array([[1, 0, 0], [0, 0, 1]], dtype=float)
    assert multiclass_brier(["A", "H"], proba, CLASSES) == 0.0


def test_brier_worst_is_two():
    # All confidence on the wrong class -> (1-0)^2 + (0-1)^2 = 2 per sample.
    proba = np.array([[1, 0, 0]], dtype=float)
    assert multiclass_brier(["H"], proba, CLASSES) == 2.0


def test_brier_uniform_is_intermediate():
    proba = np.array([[1 / 3, 1 / 3, 1 / 3]], dtype=float)
    b = multiclass_brier(["A"], proba, CLASSES)
    assert 0 < b < 2


def test_reliability_curve_structure():
    rng = np.random.default_rng(0)
    n = 500
    proba = rng.dirichlet([1, 1, 1], size=n)
    y = [CLASSES[i] for i in proba.argmax(axis=1)]
    rel = reliability_curve(y, proba, CLASSES, n_bins=5)
    assert set(rel.keys()) == set(CLASSES)
    for pred, obs, cnt in rel.values():
        assert len(pred) == len(obs) == len(cnt)
        assert (pred >= 0).all() and (pred <= 1).all()
