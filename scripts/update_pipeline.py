"""Live update pipeline — apply finished match(es) and refresh predictions.

Usage:
    python scripts/update_pipeline.py --match data/live/match_updates/match_001.csv
    python scripts/update_pipeline.py --match new_match.csv --sims 10000

Flow (your dynamic requirement):
    new_match.csv
        -> validate
        -> snapshot BEFORE predictions (current Elo)
        -> incrementally update Elo with the match(es)  [no full retrain]
        -> log the applied match, save updated Elo
        -> snapshot AFTER predictions (re-run simulation)
        -> write a before/after comparison report

The Poisson model is NOT retrained: its lambda is a function of the Elo gap, so
updated ratings flow through automatically. That keeps updates fast.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from worldcup.config import LIVE_DIR
from worldcup.data.schema import validate_match_df
from worldcup.features.elo import EloRatingSystem, RATINGS_PATH, build_current_ratings
from worldcup.simulation.predictions import (
    compare,
    generate_predictions,
    write_comparison_report,
)

APPLIED_LOG = LIVE_DIR / "applied_matches.csv"


def _load_elo() -> EloRatingSystem:
    """Load saved Elo, or build it from history if this is the first run."""
    if RATINGS_PATH.exists():
        return EloRatingSystem.load()
    print("No saved Elo found — building from history (one-time).")
    return build_current_ratings(save=True)


def _load_poisson() -> dict:
    """Load saved Poisson model, or train it from history if missing."""
    from worldcup.models.poisson import MODEL_PATH, load_model, train_poisson

    if MODEL_PATH.exists():
        return load_model()
    from worldcup.data.load import load_results
    from worldcup.features.build import build_features

    return train_poisson(build_features(load_results()), save=True)


def _log_applied(matches: pd.DataFrame) -> None:
    """Append the applied matches to the running live log."""
    APPLIED_LOG.parent.mkdir(parents=True, exist_ok=True)
    header = not APPLIED_LOG.exists()
    matches.to_csv(APPLIED_LOG, mode="a", header=header, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply finished match(es) and refresh predictions.")
    parser.add_argument("--match", required=True, help="Path to the match CSV file.")
    parser.add_argument("--sims", type=int, default=10000, help="Monte Carlo simulations (default 10000).")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (kept fixed so before/after is comparable).")
    args = parser.parse_args()

    match_path = Path(args.match)
    if not match_path.exists():
        sys.exit(f"Match file not found: {match_path}")

    matches = validate_match_df(pd.read_csv(match_path))
    descriptions = [
        f"{r.home_team} {r.home_score}-{r.away_score} {r.away_team} ({r.stage})"
        for r in matches.itertuples()
    ]
    print("Applying matches:")
    for d in descriptions:
        print(f"  - {d}")

    elo = _load_elo()
    poisson_bundle = _load_poisson()

    from worldcup.simulation.simulator import load_wc2026_played

    played_before = load_wc2026_played()
    played_after = load_wc2026_played(extra=matches)

    # BEFORE snapshot (current state) — same seed as AFTER for a fair comparison.
    print("\nComputing BEFORE predictions ...")
    before_df, _ = generate_predictions(
        n_sims=args.sims, seed=args.seed, elo=elo, poisson_bundle=poisson_bundle,
        played_matches=played_before, save=False, label="before_update",
    )

    # Incrementally update Elo (no retrain) and persist; fix the new result in the sim.
    for r in matches.itertuples():
        elo.update_match(r.home_team, r.away_team, int(r.home_score), int(r.away_score),
                         neutral=True, date=str(r.date)[:10])
    elo.save()
    _log_applied(matches)

    # AFTER snapshot.
    print("Computing AFTER predictions ...")
    after_df, snap_path = generate_predictions(
        n_sims=args.sims, seed=args.seed, elo=elo, poisson_bundle=poisson_bundle,
        played_matches=played_after, save=True, label="after_update", applied_matches=descriptions,
    )

    report_path = write_comparison_report(before_df, after_df, descriptions)

    # Console summary of the biggest title-odds movers.
    movers = compare(before_df, after_df, "champion").head(6)
    print("\nBiggest title-odds movers:")
    for _, m in movers.iterrows():
        print(f"  {m['team']:<16} {m['champion_before']:.1%} -> {m['champion_after']:.1%} "
              f"({m['delta']:+.1%})")
    print(f"\nSnapshot: {snap_path}")
    print(f"Report:   {report_path}")


if __name__ == "__main__":
    main()
