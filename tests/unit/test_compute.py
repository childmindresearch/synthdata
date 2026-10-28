"""Unit tests for the shared worker-count rule."""

import io
import os

import pytest

from synthdata import compute
from synthdata.compute import (
    _cgroup_memory_headroom_gib,
    resolve_workers,
    run_per_model,
    usable_cpus,
)
from synthdata.config import ComputeConfig
from synthdata.utils import get_logger

pytestmark = pytest.mark.unit


def _machine(monkeypatch, cpus, available_gib):
    monkeypatch.setattr("synthdata.compute.usable_cpus", lambda: cpus)
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
    return x * x, os.environ.get("OMP_NUM_THREADS"), os.environ.get("SYNTHDATA_TASK")


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
    assert [value for value, *_ in run.results.values()] == [1, 4]
    assert run.failures == {"m3": "ValueError: three"}
    if workers > 1:
        # Each process is limited to cores_per_worker threads.
        assert {threads for _, threads, _ in run.results.values()} == {"1"}
        # Each process names its model, for synthdata-run's resource monitor.
        assert [task for *_, task in run.results.values()] == ["test: m1", "test: m2"]
        assert "SYNTHDATA_TASK" not in os.environ


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


def _result_then_hang(x):
    import threading
    import time

    # A non-daemon thread left behind keeps the process from exiting.
    threading.Thread(target=time.sleep, args=(600,)).start()
    return x


def test_a_worker_that_does_not_exit_after_its_result_is_stopped(monkeypatch, caplog):
    monkeypatch.setattr(compute, "EXIT_GRACE_SECONDS", 1)
    with caplog.at_level("INFO", logger=compute.logger.name):
        run = run_per_model(
            _result_then_hang,
            {"stuck": (7,)},
            ComputeConfig(workers=1, cores_per_worker=1),
            n_columns=1,
            label="test",
            isolate=True,
        )
    assert run.results == {"stuck": 7} and not run.failures
    finished = next(r.getMessage() for r in caplog.records if "stuck finished" in r.getMessage())
    # The finish line carries what the worker measured itself.
    assert "(peak RAM " in finished and "cores on average" in finished


def test_usage_text_lists_what_was_measured():
    assert compute._usage_text({"ram_gib": 1.5, "cpu_s": 30.0, "vram_gib": 0.5}, 10) == (
        " (peak RAM 1.50 GiB, CPU 3.00 cores on average, peak GPU memory 0.50 GiB)"
    )
    assert compute._usage_text({}, 10) == ""


def test_usable_cpus_follows_the_slurm_allocation(monkeypatch):
    monkeypatch.setattr(
        "synthdata.compute.os.sched_getaffinity", lambda _pid: set(range(64)), raising=False
    )
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "12")
    assert usable_cpus() == 12
    monkeypatch.delenv("SLURM_CPUS_PER_TASK")
    monkeypatch.setattr(
        "synthdata.compute.os.sched_getaffinity", lambda _pid: set(range(8)), raising=False
    )
    assert usable_cpus() == 8  # bound to 8 CPUs of a larger node


def _cgroup(tmp_path, proc_line, files):
    tmp_path.mkdir(parents=True, exist_ok=True)
    proc = tmp_path / "proc_cgroup"
    proc.write_text(proc_line + "\n")
    root = tmp_path / "cgroup"
    for relative, value in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    return proc, root


def test_cgroup_v2_limit_gives_the_headroom_left_under_it(tmp_path):
    gib = 1024**3
    proc, root = _cgroup(
        tmp_path,
        "0::/slurm/job_7/step_0",
        {
            "slurm/job_7/memory.max": str(64 * gib),  # the job's --mem
            "slurm/job_7/memory.current": str(10 * gib),
            "slurm/job_7/step_0/memory.max": "max",
            "slurm/job_7/step_0/memory.current": str(10 * gib),
        },
    )
    assert _cgroup_memory_headroom_gib(proc, root) == pytest.approx(54.0)


def test_cgroup_v1_and_unlimited_cgroups(tmp_path):
    gib = 1024**3
    proc, root = _cgroup(
        tmp_path,
        "4:memory:/slurm/job_7",
        {
            "memory/slurm/job_7/memory.limit_in_bytes": str(32 * gib),
            "memory/slurm/job_7/memory.usage_in_bytes": str(2 * gib),
        },
    )
    assert _cgroup_memory_headroom_gib(proc, root) == pytest.approx(30.0)
    proc, root = _cgroup(
        tmp_path / "unlimited",
        "0::/user.slice",
        {"user.slice/memory.max": "max", "user.slice/memory.current": "1"},
    )
    assert _cgroup_memory_headroom_gib(proc, root) is None


def test_prefixed_stream_writes_whole_prefixed_lines():
    out = io.StringIO()
    stream = compute._Prefixed(out, "[gen: ctgan] ")
    stream.write("first\nsec")
    stream.write("ond\r\n\r 10%|#")
    stream.write("\r 20%|##")
    assert out.getvalue() == "[gen: ctgan] first\n[gen: ctgan] second\n[gen: ctgan]  10%|#\n"
    stream.close_line()  # the last update, held until its line ends
    assert out.getvalue().endswith("[gen: ctgan]  10%|#\n[gen: ctgan]  20%|##\n")


def _chatty(x):
    print(f"printed {x}")
    get_logger("synthdata.test_chatty").info("logged %s", x)
    return x


def test_worker_output_names_its_model(capfd, monkeypatch):
    messages = []
    monkeypatch.setattr(compute.logger, "info", lambda msg, *a: messages.append(msg % a))
    run = run_per_model(
        _chatty,
        {"ctgan": (1,), "tvae": (2,)},
        ComputeConfig(workers=2, cores_per_worker=1),
        n_columns=1,
        label="generation",
    )
    assert run.results == {"ctgan": 1, "tvae": 2}
    captured = capfd.readouterr()
    output = captured.out + captured.err
    assert "[generation: ctgan] printed 1" in output
    assert "[generation: tvae] printed 2" in output
    assert "[generation: ctgan] [" in output and "logged 1" in output
    assert any("finished after" in m and "2 of 2 done" in m for m in messages)
