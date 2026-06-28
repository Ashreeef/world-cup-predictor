"""Persist prediction snapshots and build before/after comparison reports.

A "snapshot" is one run of the simulator, saved as:
    data/predictions/predictions_<timestamp>.csv    (48 teams x stage probs)
    data/predictions/predictions_<timestamp>.json   (metadata: when, n_sims, ...)

Comparison reports (reports/) show how each team's probabilities moved between
two snapshots — e.g. after a match is applied via the live update pipeline.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from worldcup.config import PREDICTIONS_DIR, REPORTS_DIR

PREFIX = "predictions_"
PROB_COLUMNS = ["qualify", "round_of_16", "quarter_final", "semi_final", "final", "champion"]


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


# ── Snapshots ─────────────────────────────────────────────────────────────────────
def save_snapshot(df: pd.DataFrame, metadata: dict, dirpath: Path = PREDICTIONS_DIR, ts: str | None = None) -> Path:
    """Write a snapshot CSV + sibling JSON metadata. Returns the CSV path."""
    dirpath.mkdir(parents=True, exist_ok=True)
    ts = ts or _timestamp()
    csv_path = dirpath / f"{PREFIX}{ts}.csv"
    df.to_csv(csv_path, index=False)

    meta = {**metadata, "timestamp": ts, "csv": csv_path.name}
    (dirpath / f"{PREFIX}{ts}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return csv_path


def load_snapshot(csv_path: Path) -> tuple[pd.DataFrame, dict]:
    """Load a snapshot CSV and its metadata JSON."""
    df = pd.read_csv(csv_path)
    json_path = csv_path.with_suffix(".json")
    meta = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}
    return df, meta


def list_snapshots(dirpath: Path = PREDICTIONS_DIR) -> list[Path]:
    """All snapshot CSVs, oldest -> newest (timestamp sorts lexically)."""
    return sorted(dirpath.glob(f"{PREFIX}*.csv"))


def latest_snapshot(dirpath: Path = PREDICTIONS_DIR) -> Path | None:
    snaps = list_snapshots(dirpath)
    return snaps[-1] if snaps else None


def generate_predictions(
    n_sims: int = 10000,
    seed: int | None = 42,
    elo=None,
    poisson_bundle: dict | None = None,
    save: bool = True,
    label: str | None = None,
    applied_matches: list[str] | None = None,
    played_matches: pd.DataFrame | None = None,
    r32_bracket: list | None = None,
) -> tuple[pd.DataFrame, Path | None]:
    """Run the simulation (conditioned on `played_matches`) and optionally save.

    If `r32_bracket` is given (group stage finished), the fixed-bracket knockout
    simulation is used; otherwise the full group+knockout simulation runs.
    """
    if r32_bracket is not None:
        from worldcup.simulation.knockout import knockout_predictions

        df, _ = knockout_predictions(
            elo, poisson_bundle, played_matches=played_matches,
            n_sims=n_sims, seed=seed or 42, bracket=r32_bracket,
        )
        stage = "knockout"
    else:
        from worldcup.simulation.simulator import run_simulation

        df = run_simulation(
            n_sims=n_sims, seed=seed, elo=elo, poisson_bundle=poisson_bundle, played_matches=played_matches
        )
        stage = "group"

    metadata = {
        "n_sims": n_sims,
        "seed": seed,
        "label": label,
        "stage": stage,
        "elo_through": getattr(elo, "last_date", None),
        "n_played": 0 if played_matches is None else len(played_matches),
        "applied_matches": applied_matches or [],
    }
    path = save_snapshot(df, metadata) if save else None
    return df, path


# ── Comparison ─────────────────────────────────────────────────────────────────────
def compare(before: pd.DataFrame, after: pd.DataFrame, metric: str = "champion") -> pd.DataFrame:
    """Return team, before, after, delta for a metric, sorted by |delta| desc."""
    merged = before[["team", metric]].merge(after[["team", metric]], on="team", suffixes=("_before", "_after"))
    merged["delta"] = merged[f"{metric}_after"] - merged[f"{metric}_before"]
    return merged.reindex(merged["delta"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def write_comparison_report(
    before: pd.DataFrame,
    after: pd.DataFrame,
    applied_matches: list[str],
    dirpath: Path = REPORTS_DIR,
    top: int = 10,
    ts: str | None = None,
) -> Path:
    """Write a markdown before/after report (champion + qualify movers)."""
    dirpath.mkdir(parents=True, exist_ok=True)
    ts = ts or _timestamp()
    path = dirpath / f"report_{ts}.md"

    lines = ["# Prediction update report", "", f"Generated: {ts}", "", "## Matches applied"]
    lines += [f"- {m}" for m in applied_matches] or ["- (none)"]

    for metric in ("champion", "qualify"):
        diff = compare(before, after, metric).head(top)
        lines += ["", f"## Biggest movers — {metric.replace('_', ' ')}", "",
                   "| Team | Before | After | Change |", "|------|-------:|------:|-------:|"]
        for _, r in diff.iterrows():
            arrow = "🔺" if r["delta"] > 0 else ("🔻" if r["delta"] < 0 else "")
            lines.append(
                f"| {r['team']} | {r[f'{metric}_before']:.1%} | {r[f'{metric}_after']:.1%} | "
                f"{arrow} {r['delta']:+.1%} |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
