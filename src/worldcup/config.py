"""Central configuration: all filesystem paths live here.

Why this file exists
--------------------
Hard-coding paths like "../data/raw/matches.csv" all over notebooks and scripts
is fragile — it breaks the moment you run code from a different folder. Instead,
every module imports paths from here. Change a location once, and the whole
project follows.
"""

from pathlib import Path

# Project root = two levels up from this file (src/worldcup/config.py -> project/)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# ── Data directories ───────────────────────────────────────────────────────────
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"               # downloaded external data (git-ignored)
REFERENCE_DATA_DIR: Path = DATA_DIR / "reference"   # curated data we maintain (tracked)
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"   # cleaned, model-ready data
LIVE_DIR: Path = DATA_DIR / "live"                  # live tournament state
MATCH_UPDATES_DIR: Path = LIVE_DIR / "match_updates"  # incoming new_match.csv files
PREDICTIONS_DIR: Path = DATA_DIR / "predictions"    # timestamped prediction snapshots

# ── Code-adjacent directories ──────────────────────────────────────────────────
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"    # trained models (rarely change)
REPORTS_DIR: Path = PROJECT_ROOT / "reports"        # before/after comparison reports
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

# Convenience: list of all directories the project expects to exist.
ALL_DIRS = [
    RAW_DATA_DIR,
    REFERENCE_DATA_DIR,
    PROCESSED_DATA_DIR,
    MATCH_UPDATES_DIR,
    PREDICTIONS_DIR,
    ARTIFACTS_DIR,
    REPORTS_DIR,
    NOTEBOOKS_DIR,
]
