"""Create the project's directory structure.

Run this once after cloning the repo. Git does not track
empty folders, so this script guarantees every expected directory exists before
the pipeline runs.
"""

from worldcup.config import ALL_DIRS


def main() -> None:
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        # Drop a .gitkeep so the (otherwise empty) folder can be committed.
        gitkeep = directory / ".gitkeep"
        gitkeep.touch(exist_ok=True)
        print(f"  ✅ {directory}")
    print("\nProject structure ready.")


if __name__ == "__main__":
    main()
