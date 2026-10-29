"""Fixtures shared by the unit tests only."""

import pytest


@pytest.fixture(autouse=True)
def one_generation_process(monkeypatch):
    """Run every generation unit in the test process.

    ``compute.workers: auto`` sizes the pool from the machine's free memory
    and cores, so on a large machine ``run_generation`` would start worker
    processes, where the tests' mocks do not apply: real models were trained
    and failures planted by a mock never happened.
    """
    monkeypatch.setattr("synthdata.generation.pipeline.resolve_workers", lambda *a, **k: 1)
