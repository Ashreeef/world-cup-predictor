"""Team-name normalization.

Real-world data uses inconsistent country names ("Türkiye" vs "Turkey",
"Czechia" vs "Czech Republic"). The match dataset has its own canonical
spellings, and joins silently break if names don't match exactly.

Strategy:
    - CANONICAL names = whatever the historical match dataset uses. We store
      everything internally using these so joins always work.
    - ALIASES map any alternative/official spelling -> canonical.
    - DISPLAY_NAMES map canonical -> the nicer official name, for the UI only.
"""

from __future__ import annotations

# Alternative / official spellings -> the dataset's canonical spelling.
ALIASES: dict[str, str] = {
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Cabo Verde": "Cape Verde",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "USA": "United States",
}

# Canonical -> pretty display name (used only for presentation).
DISPLAY_NAMES: dict[str, str] = {
    "Czech Republic": "Czechia",
    "Turkey": "Türkiye",
    "Cape Verde": "Cabo Verde",
}


def canonical(name: str) -> str:
    """Return the dataset's canonical spelling for a team name."""
    return ALIASES.get(name.strip(), name.strip())


def display(name: str) -> str:
    """Return the pretty display name for a (canonical) team name."""
    return DISPLAY_NAMES.get(name, name)
