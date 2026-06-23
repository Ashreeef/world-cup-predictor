"""Smoke tests — confirm the package is installed and importable.

These trivial tests let `pytest` (and CI) pass from day one, proving the
environment is correctly set up before any real logic exists.
"""

import worldcup
from worldcup import config


def test_package_imports():
    assert worldcup.__version__ == "0.1.0"


def test_config_paths_defined():
    # Project root should contain this repo's pyproject.toml.
    assert (config.PROJECT_ROOT / "pyproject.toml").exists()
    assert config.RAW_DATA_DIR.name == "raw"
