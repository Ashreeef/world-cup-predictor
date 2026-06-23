"""Download historical international football results.

Two interchangeable sources for the SAME data:

  github  (default)  Pulls the CSVs straight from the upstream public repo
                     martj42/international_results. No authentication, no rate
                     limits, always current. This is the original source that
                     the Kaggle dataset mirrors.

  kaggle             Uses the Kaggle API (requires a configured API token and a
                     phone-verified Kaggle account).

Files delivered into data/raw/:
    results.csv      one row per international match (scores, date, venue, ...)
    goalscorers.csv  one row per goal (scorer, minute, penalty/own-goal)
    shootouts.csv    penalty-shootout winners for drawn knockout games

Run it:
    python -m worldcup.data.download                 # github (default)
    python -m worldcup.data.download --source kaggle
    python -m worldcup.data.download --force          # re-download
"""

from __future__ import annotations

import argparse

from worldcup.config import RAW_DATA_DIR

# Upstream public repo (the source Kaggle itself mirrors).
GITHUB_BASE = "https://raw.githubusercontent.com/martj42/international_results/master"
DATA_FILES = ("results.csv", "goalscorers.csv", "shootouts.csv")

# Kaggle dataset slug (alternative source).
KAGGLE_DATASET = "martj42/international-football-results-1872-to-2017"


def download_from_github(force: bool = False) -> None:
    """Stream the CSVs directly from the upstream GitHub repo."""
    import requests

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename in DATA_FILES:
        target = RAW_DATA_DIR / filename
        if target.exists() and not force:
            print(f"Already present: {target.name}")
            continue
        url = f"{GITHUB_BASE}/{filename}"
        print(f"Downloading {url} ...")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        target.write_bytes(resp.content)
        print(f"  saved {target.name} ({len(resp.content) // 1024} KB)")


def download_from_kaggle(force: bool = False) -> None:
    """Fetch and unzip the dataset via the Kaggle API."""
    if (RAW_DATA_DIR / "results.csv").exists() and not force:
        print("Already present: results.csv (pass --force to re-download)")
        return
    # Imported here so the package still loads without Kaggle credentials.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading '{KAGGLE_DATASET}' into {RAW_DATA_DIR} ...")
    api.dataset_download_files(KAGGLE_DATASET, path=str(RAW_DATA_DIR), unzip=True)


# Optional: FIFA world ranking history (separate Kaggle dataset). Not used as a
# feature yet — wired up here for a future Phase 14b extension. Requires Kaggle auth.
FIFA_RANKING_DATASET = "cashncarry/fifaworldranking"


def download_fifa_rankings(force: bool = False) -> None:
    """Download FIFA world ranking history from Kaggle into data/raw/ (optional)."""
    target = RAW_DATA_DIR / "fifa_ranking.csv"
    if target.exists() and not force:
        print(f"Already present: {target.name}")
        return
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading '{FIFA_RANKING_DATASET}' into {RAW_DATA_DIR} ...")
    api.dataset_download_files(FIFA_RANKING_DATASET, path=str(RAW_DATA_DIR), unzip=True)


def ensure_results() -> None:
    """Make sure data/raw/results.csv exists, downloading it if not.

    Used at app startup so a fresh deployment (where data/ is git-ignored) can
    bootstrap itself on first run.
    """
    if not (RAW_DATA_DIR / "results.csv").exists():
        download_from_github()


def download_historical_results(source: str = "github", force: bool = False) -> None:
    """Download the historical results from the chosen source."""
    if source == "github":
        download_from_github(force=force)
    elif source == "kaggle":
        download_from_kaggle(force=force)
    else:
        raise ValueError(f"Unknown source: {source!r} (use 'github' or 'kaggle')")
    print("Done. Files are in data/raw/.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download historical international match data.")
    parser.add_argument(
        "--source",
        choices=["github", "kaggle"],
        default="github",
        help="Where to fetch the data from (default: github).",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist.")
    args = parser.parse_args()
    download_historical_results(source=args.source, force=args.force)


if __name__ == "__main__":
    main()
