"""Refresh every model artifact and prediction snapshot from the current data.

Run this after pulling new results (`python -m worldcup.data.download --force`)
to propagate the latest matches through the whole pipeline:

    python scripts/refresh_all.py [--sims 10000]

Steps:
    1. Rebuild Elo ratings           -> artifacts/elo_ratings.json
    2. Retrain Poisson score model   -> artifacts/poisson_model.joblib
    3. Retrain baseline + XGBoost    -> artifacts/*.joblib
    4. Regenerate a results-aware prediction snapshot (conditioned on the
       matches played so far) -> data/predictions/predictions_<ts>.csv
"""

from __future__ import annotations

import argparse

from worldcup.data.load import load_results
from worldcup.features.build import build_features
from worldcup.features.elo import build_current_ratings
from worldcup.models.baseline import train_production_model
from worldcup.models.poisson import train_poisson
from worldcup.models.xgboost_model import train_production_xgb
from worldcup.simulation.predictions import generate_predictions
from worldcup.simulation.simulator import current_standings, load_wc2026_played


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh all models and predictions from current data.")
    parser.add_argument("--sims", type=int, default=10000, help="Monte Carlo simulations (default 10000).")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    args = parser.parse_args()

    print("1/4  Rebuilding Elo ratings ...")
    elo = build_current_ratings(save=True)

    print("2/4  Building features + retraining models ...")
    feats = build_features(load_results())
    poisson_bundle = train_poisson(feats, save=True)
    train_production_model(feats)   # logistic baseline
    train_production_xgb(feats)     # xgboost

    print("3/4  Loading matches played so far ...")
    played = load_wc2026_played()
    print(f"     {len(played)} WC2026 matches played.")

    # Once the group stage is over and the R32 bracket exists, predict the knockout
    # from the fixed bracket; otherwise simulate the full group + knockout.
    from worldcup.simulation.knockout import R32_CSV, load_r32_bracket

    bracket = load_r32_bracket() if R32_CSV.exists() else None
    mode = "fixed knockout bracket" if bracket else "full group + knockout"
    print(f"4/4  Running {args.sims:,} simulations ({mode}) ...")
    table, snap = generate_predictions(
        n_sims=args.sims, seed=args.seed, elo=elo, poisson_bundle=poisson_bundle,
        played_matches=played, save=True, label="full_refresh", r32_bracket=bracket,
    )

    eliminated = table[table["qualify"] < 1e-9]["team"].tolist()
    print(f"\nSaved snapshot: {snap.name}")
    print(f"Eliminated so far ({len(eliminated)}): {', '.join(eliminated) if eliminated else 'none'}")
    print("\nTop title contenders now:")
    top = table.head(8)[["team", "qualify", "champion"]].copy()
    top["qualify"] = (top["qualify"] * 100).round(1)
    top["champion"] = (top["champion"] * 100).round(1)
    print(top.to_string(index=False))
    print("\nDone. Restart the dashboard (or it will pick up the new snapshot on next launch).")


if __name__ == "__main__":
    main()
