"""Unit tests for the shared worker-count rule."""

import os

import pytest

from synthdata.compute import resolve_workers, run_per_model
from synthdata.config import ComputeConfig

pytestmark = pytest.mark.unit


def _machine(monkeypatch, cpus, available_gib):
    monkeypatch.setattr("synthdata.compute.os.cpu_count", lambda: cpus)
    monkeypatch.setattr("synthdata.compute.available_memory_gib", lambda: available_gib)


def test_explicit_limit_obeys_model_and_max_bounds():
    cfg = ComputeConfig(workers=10, max_workers=4)
    assert resolve_workers(cfg, n_models=3, n_columns=10) == 3


def test_one_worker_runs_models_one_at_a_time():
    assert resolve_workers(ComputeConfig(workers=1), n_models=7, n_columns=10) == 1


def test_auto_uses_memory_and_cpu_bounds(monkeypatch):
    _machine(monkeypatch, cpus=24, available_gib=118.0)
    cfg = ComputeConfig(workers="auto", max_workers=8, cores_per_worker=4, memory_reserve_gib=16)
    assert resolve_workers(cfg, n_models=18, n_columns=1038) == 6


@pytest.mark.parametrize(
    ("cpus", "memory", "n_models", "expected"),
    [
        (24, 118.0, 7, 6),  # Linux box: 24 cores / 4
        (36, 500.0, 7, 7),  # never more workers than models
        (36, 500.0, 30, 8),  # max_workers
        (24, 40.0, 7, 1),  # (40 - 16) // 14 = 1
        (2, 118.0, 7, 1),  # fewer cores than one worker needs
    ],
)
def test_auto_bounds(monkeypatch, cpus, memory, n_models, expected):
    _machine(monkeypatch, cpus=cpus, available_gib=memory)
    cfg = ComputeConfig(memory_per_worker_gib=14)
    assert resolve_workers(cfg, n_models=n_models, n_columns=1000) == expected


def test_auto_does_not_read_total_memory_after_available_memory(monkeypatch):
    _machine(monkeypatch, cpus=24, available_gib=118.0)
    monkeypatch.setattr(
        "synthdata.compute.Path.read_text",
        lambda _path: pytest.fail("worker resolution must not read MemTotal"),
    )
    cfg = ComputeConfig(workers="auto", max_workers=8, cores_per_worker=4, memory_reserve_gib=16)
    assert resolve_workers(cfg, n_models=18, n_columns=1038) == 6


def _square_or_fail(x):
    if x == 3:
        raise ValueError("three")
    if x == 4:
        os._exit(9)  # a killed process (out-of-memory killer, segfault)
    return x * x, os.environ.get("OMP_NUM_THREADS")


@pytest.mark.parametrize("workers", [1, 2])
def test_run_per_model_collects_results_and_failures(workers):
    tasks = {f"m{x}": (x,) for x in (1, 2, 3)}
    run = run_per_model(
        _square_or_fail,
        tasks,
        ComputeConfig(workers=workers, cores_per_worker=1),
        n_columns=1,
        label="test",
    )
    assert list(run.results) == ["m1", "m2"]
    assert [value for value, _ in run.results.values()] == [1, 4]
    assert run.failures == {"m3": "ValueError: three"}
    if workers > 1:
        # Each process is limited to cores_per_worker threads.
        assert {threads for _, threads in run.results.values()} == {"1"}


def test_run_per_model_survives_a_killed_process():
    run = run_per_model(
        _square_or_fail,
        {"ok": (2,), "killed": (4,)},
        ComputeConfig(workers=1, cores_per_worker=1),
        n_columns=1,
        label="test",
        isolate=True,
    )
    assert run.results["ok"][0] == 4
    assert "exited with code 9" in run.failures["killed"]
