"""Tests for the Dixon-Coles model on synthetic data with known strengths."""

import numpy as np
import pandas as pd

from worldcup.models.dixon_coles import DixonColesModel, _tau


def _synthetic(n_per_pair: int = 60, seed: int = 0) -> pd.DataFrame:
    """3 teams of clearly different strength; many neutral matches between them."""
    rng = np.random.default_rng(seed)
    strength = {"Strong": 1.3, "Mid": 0.6, "Weak": 0.0}  # log attack baseline
    teams = list(strength)
    rows = []
    date = pd.Timestamp("2015-01-01")
    for h in teams:
        for a in teams:
            if h == a:
                continue
            for _ in range(n_per_pair):
                lh = np.exp(0.1 + strength[h] - 0.5 * strength[a])
                la = np.exp(0.1 + strength[a] - 0.5 * strength[h])
                rows.append({
                    "date": date, "home_team": h, "away_team": a,
                    "home_score": rng.poisson(lh), "away_score": rng.poisson(la), "neutral": True,
                })
                date += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


def test_tau_neutral_for_high_scores():
    # No correction outside the low-score cells.
    assert _tau(np.array([3]), np.array([2]), np.array([1.5]), np.array([1.2]), 0.1)[0] == 1.0


def test_attack_strength_ranks_teams():
    dc = DixonColesModel(since="2014-01-01", min_matches=5, half_life_days=None).fit(_synthetic())
    assert dc.attack["Strong"] > dc.attack["Mid"] > dc.attack["Weak"]


def test_score_matrix_is_distribution_and_favours_strong():
    dc = DixonColesModel(since="2014-01-01", min_matches=5, half_life_days=None).fit(_synthetic())
    m = dc.score_matrix("Strong", "Weak", neutral=True)
    assert abs(m.sum() - 1.0) < 1e-9
    assert (m >= 0).all()
    o = dc.predict_match("Strong", "Weak")["outcome"]
    assert o["H"] > o["A"]


def test_predict_lambdas_strong_scores_more():
    dc = DixonColesModel(since="2014-01-01", min_matches=5, half_life_days=None).fit(_synthetic())
    lh, la = dc.predict_lambdas("Strong", "Weak", neutral=True)
    assert lh > la
