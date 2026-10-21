"""TabularARGN end to end on the clinic fixture (``-m slow``).

Needs its separate environment (``uv sync --project envs/tabularargn``):

    uv run pytest tests/integration/test_tabularargn.py -m slow
"""

from __future__ import annotations

import pytest

from synthdata.generation.sidecar import REPO_ROOT

from .conftest import new_run, run_full_pipeline

pytestmark = [pytest.mark.integration, pytest.mark.slow]

OVERRIDES = {
    "generation": {
        "synthcity": {"enabled": False},
        "hpo": {"enabled": False},
        "tabularargn": {"enabled": True, "params": {"max_epochs": 20}},
    },
}


@pytest.fixture(scope="module")
def argn_run(tmp_path_factory):
    if not (REPO_ROOT / "envs" / "tabularargn" / ".venv").exists():
        pytest.skip("needs the TabularARGN environment: uv sync --project envs/tabularargn")
    return run_full_pipeline(new_run(tmp_path_factory, "tabularargn", overrides=OVERRIDES))


def test_tabularargn_output_is_complete(argn_run):
    frame = argn_run.synthetic()["tabularargn"]
    assert len(frame) == argn_run.cfg.generation.n_samples
    assert not frame.isna().any().any()
    assert frame["target"].nunique() == 2


def test_tabularargn_was_evaluated(argn_run):
    assert "tabularargn" in argn_run.combined().index
