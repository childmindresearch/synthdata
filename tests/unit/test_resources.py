"""Unit tests for synthdata.resources."""

import os

import pytest

from synthdata import resources

pytestmark = pytest.mark.unit


def test_gpu_usage_sums_memory_and_keeps_only_the_pipelines_processes(monkeypatch):
    replies = {
        "--query-gpu=utilization.gpu,memory.used": [["90", "40000"], ["10", "2000"]],
        "--query-compute-apps=pid,used_memory": [["11", "30000"], ["99", "5000"], ["12", "1000"]],
    }
    monkeypatch.setattr(resources, "_nvidia_smi", lambda query: replies[query])
    assert resources.gpu_usage({11, 12}) == {
        "gpu_util_pct": 50.0,
        "gpu_mem_used_mib": 42000,
        "pipeline_vram_mib": 31000,
    }


def test_gpu_usage_is_empty_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(resources.subprocess, "run", _raise(FileNotFoundError))
    assert resources.gpu_usage({1}) == {}


def _raise(error):
    def run(*args, **kwargs):
        raise error

    return run


def test_rows_and_peaks(tmp_path, monkeypatch):
    monkeypatch.setattr(resources, "gpu_usage", lambda pids: {"gpu_mem_used_mib": 2048})
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
