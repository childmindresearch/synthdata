"""Unit tests for synthdata.resources."""

import csv
import os
import subprocess
import sys
import time

import pytest

from synthdata import compute, resources

pytestmark = pytest.mark.unit


def test_gpu_usage_sums_memory_and_keeps_only_the_pipelines_processes(monkeypatch):
    replies = {
        "--query-gpu=utilization.gpu,memory.used": [["90", "40000"], ["10", "2000"]],
        "--query-compute-apps=pid,used_memory": [["11", "30000"], ["99", "5000"], ["12", "1000"]],
    }
    monkeypatch.setattr(resources, "_nvidia_smi", lambda query: replies[query])
    assert resources.gpu_usage({11, 12}) == (
        {"gpu_util_pct": 50.0, "gpu_mem_used_mib": 42000, "pipeline_vram_mib": 31000},
        {11: 30000, 12: 1000},
    )


def test_gpu_usage_leaves_pipeline_vram_empty_when_per_process_memory_is_na(monkeypatch):
    # Windows (WDDM) drivers report a process's GPU memory as "[N/A]".
    replies = {
        "--query-gpu=utilization.gpu,memory.used": [["40", "3000"]],
        "--query-compute-apps=pid,used_memory": [["11", "[N/A]"], ["99", "[N/A]"]],
    }
    monkeypatch.setattr(resources, "_nvidia_smi", lambda query: replies[query])
    assert resources.gpu_usage({11}) == ({"gpu_util_pct": 40.0, "gpu_mem_used_mib": 3000}, {})


def test_gpu_usage_is_empty_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(resources.subprocess, "run", _raise(FileNotFoundError))
    assert resources.gpu_usage({1}) == ({}, {})


def _raise(error):
    def run(*args, **kwargs):
        raise error

    return run


def test_rows_and_peaks(tmp_path, monkeypatch):
    monkeypatch.setattr(resources, "gpu_usage", lambda pids: ({"gpu_mem_used_mib": 2048}, {}))
    monitor = resources.ResourceMonitor(tmp_path / "r.csv", interval=60, root_pid=os.getpid())
    monitor.set_stage("generate")
    monitor.record()
    monitor.record()
    lines = (tmp_path / "r.csv").read_text().splitlines()
    assert lines[0] == ",".join(resources.COLUMNS)
    assert len(lines) == 3
    first, second = (line.split(",") for line in lines[1:])
    assert first[0].endswith("Z") and first[1] == "generate"
    assert first[3] == "" and second[3] != ""  # CPU needs a previous row
    line = monitor.peak_line("generate")
    assert "RAM" in line and "CPU" in line and "GPU memory 2.0 GiB" in line
    assert "no samples" in monitor.peak_line("plot")


def test_use_is_split_by_the_task_a_worker_process_names(tmp_path, monkeypatch):
    env = {**os.environ, compute.TASK_ENV_VAR: "generation: ctgan"}
    script = "import time; x = bytearray(80 * 2**20); time.sleep(30)"
    worker = subprocess.Popen([sys.executable, "-c", script], env=env)
    try:
        monkeypatch.setattr(
            resources, "gpu_usage", lambda pids: ({"pipeline_vram_mib": 700}, {worker.pid: 700})
        )
        monitor = resources.ResourceMonitor(tmp_path / "r.csv", interval=60, root_pid=os.getpid())
        monitor.set_stage("generate")
        for _ in range(50):  # until the worker has allocated its memory
            monitor.record()
            rows = list(csv.DictReader(monitor.task_path.open()))
            ctgan = [r for r in rows if r["task"] == "generation: ctgan"]
            if ctgan and float(ctgan[-1]["ram_gib"]) > 0.07:
                break
            time.sleep(0.1)
    finally:
        worker.kill()
        worker.wait()
    assert monitor.task_path.name == "r_by_task.csv"
    assert {"main", "generation: ctgan"} <= {r["task"] for r in rows}
    assert ctgan[-1]["vram_mib"] == "700" and ctgan[-1]["stage"] == "generate"
    assert float(ctgan[-1]["ram_gib"]) > 0.07
    line = monitor.heaviest_line("generate")
    assert line.startswith("heaviest tasks during generate: RAM (GiB) ")
    assert "GPU memory (GiB) generation: ctgan 0.7" in line
    assert monitor.heaviest_line("plot") is None
