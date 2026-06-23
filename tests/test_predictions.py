"""Tests for snapshot persistence and before/after comparison reporting."""

import pandas as pd

from worldcup.simulation.predictions import (
    compare,
    latest_snapshot,
    list_snapshots,
    load_snapshot,
    save_snapshot,
    write_comparison_report,
)

PROB_COLS = ["qualify", "round_of_16", "quarter_final", "semi_final", "final", "champion"]


def _snap(champion_vals):
    teams = [f"T{i}" for i in range(len(champion_vals))]
    data = {"team": teams}
    for c in PROB_COLS:
        data[c] = [0.5] * len(teams)
    data["champion"] = champion_vals
    return pd.DataFrame(data)


def test_save_and_load_roundtrip(tmp_path):
    df = _snap([0.3, 0.7])
    csv = save_snapshot(df, {"n_sims": 100}, dirpath=tmp_path, ts="20260623T120000")
    assert csv.exists()
    assert csv.with_suffix(".json").exists()

    loaded, meta = load_snapshot(csv)
    assert loaded.equals(df)
    assert meta["n_sims"] == 100
    assert meta["timestamp"] == "20260623T120000"


def test_list_and_latest(tmp_path):
    save_snapshot(_snap([0.5, 0.5]), {}, dirpath=tmp_path, ts="20260101T000000")
    save_snapshot(_snap([0.4, 0.6]), {}, dirpath=tmp_path, ts="20260102T000000")
    assert len(list_snapshots(tmp_path)) == 2
    assert latest_snapshot(tmp_path).name.endswith("20260102T000000.csv")


def test_compare_computes_sorted_deltas():
    before = _snap([0.20, 0.10])
    after = _snap([0.25, 0.05])
    diff = compare(before, after, "champion")
    # Largest absolute change first; both moved 0.05.
    assert set(diff.columns) == {"team", "champion_before", "champion_after", "delta"}
    assert diff.iloc[0]["delta"] == 0.05 or diff.iloc[0]["delta"] == -0.05
    assert abs(diff["delta"]).is_monotonic_decreasing


def test_write_comparison_report(tmp_path):
    before = _snap([0.16, 0.20])
    after = _snap([0.21, 0.18])
    path = write_comparison_report(before, after, ["A 1-0 B (group)"], dirpath=tmp_path)
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "Biggest movers" in text
    assert "A 1-0 B (group)" in text
