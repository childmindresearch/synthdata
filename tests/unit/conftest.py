"""Fixtures shared by the unit tests only."""

import os

import pytest

# One thread per library. LightGBM (miceforest), XGBoost and torch otherwise
# start a thread per core and wait for the slowest at every step, so on a
# machine where other work holds a few cores the tiny unit-test fits slowed
# down about 30-fold (miceforest tests: 13 s idle, 6 min 39 s with 3 of 4
# cores busy). Set before any of them is imported; a value already in the
# environment wins.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")


@pytest.fixture(autouse=True, scope="session")
def _one_thread_in_loaded_libraries():
    """Also cap the thread pools already loaded before this file (scikit-learn's
    OpenMP, the BLAS behind numpy)."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=int(os.environ["OMP_NUM_THREADS"])):
        yield


@pytest.fixture(autouse=True)
def one_generation_process(monkeypatch):
    """Run every generation unit in the test process.

    ``compute.workers: auto`` sizes the pool from the machine's free memory
    and cores, so on a large machine ``run_generation`` would start worker
    processes, where the tests' mocks do not apply: real models were trained
    and failures planted by a mock never happened.
    """
    monkeypatch.setattr("synthdata.generation.pipeline.resolve_workers", lambda *a, **k: 1)
