"""Smoke test: the Streamlit app runs headlessly without raising.

Uses Streamlit's AppTest. Requires the data + artifacts to be present (Elo and
Poisson models, and at least one prediction snapshot). Skipped if missing so the
test suite stays green on a fresh checkout before the pipeline has been run.
"""

import pytest

from worldcup.features.elo import RATINGS_PATH
from worldcup.models.poisson import MODEL_PATH as POISSON_PATH
from worldcup.simulation.predictions import latest_snapshot

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

_READY = RATINGS_PATH.exists() and POISSON_PATH.exists() and latest_snapshot() is not None


@pytest.mark.skipif(not _READY, reason="Run the pipeline first (Elo/Poisson/snapshot needed).")
def test_app_runs_without_exception():
    at = AppTest.from_file("app/streamlit_app.py").run(timeout=120)
    assert not at.exception
    # The four tabs should render.
    assert len(at.tabs) == 4
    # Headline title present.
    assert any("World Cup 2026 Predictor" in str(t.value) for t in at.title)
