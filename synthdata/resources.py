"""Record the RAM, CPU and GPU a pipeline run uses, for sizing machines and jobs.

``synthdata-run`` starts a :class:`ResourceMonitor` next to its log. Every
``interval`` seconds it appends one row to ``<UTC time>_resources.csv``:

* ``pipeline_ram_gib``: resident memory of the runner, its stage and every
  worker process. Pages shared between processes count once per process, so
  this is an upper bound on what a job's memory limit must allow.
* ``pipeline_cpu_cores``: CPU time those processes used per second of wall
  time since the previous row (4.0 = four cores busy).
* ``system_ram_used_gib`` / ``system_ram_total_gib``: the whole machine.
* ``gpu_util_pct`` / ``gpu_mem_used_mib``: all GPUs, as ``nvidia-smi``
  reports them (mean utilization, summed memory, including other users).
* ``pipeline_vram_mib``: GPU memory held by the pipeline's own processes.

All times are UTC. The GPU columns stay empty without ``nvidia-smi``. A
sampling error skips that row and never stops the run.
"""

import contextlib
import csv
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil

COLUMNS = [
    "time_utc",
    "stage",
    "pipeline_ram_gib",
    "pipeline_cpu_cores",
    "system_ram_used_gib",
    "system_ram_total_gib",
    "gpu_util_pct",
    "gpu_mem_used_mib",
    "pipeline_vram_mib",
]

#: Columns whose maximum per stage is written to the log.
PEAK_COLUMNS = ("pipeline_ram_gib", "pipeline_cpu_cores", "gpu_mem_used_mib", "pipeline_vram_mib")

_GIB = 1024**3


def _nvidia_smi(*query: str) -> list[list[str]] | None:
    """Rows of an ``nvidia-smi`` CSV query, or None when it can't run."""
    try:
        out = subprocess.run(
            ["nvidia-smi", *query, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return [[field.strip() for field in line.split(",")] for line in out.splitlines() if line]


def gpu_usage(pids: set[int]) -> dict:
    """GPU utilization and memory, overall and for ``pids``; empty without a GPU."""
    gpus = _nvidia_smi("--query-gpu=utilization.gpu,memory.used")
    if not gpus:
        return {}
    usage = {
        "gpu_util_pct": round(sum(float(g[0]) for g in gpus) / len(gpus), 1),
        "gpu_mem_used_mib": sum(int(float(g[1])) for g in gpus),
    }
    apps = _nvidia_smi("--query-compute-apps=pid,used_memory") or []
    usage["pipeline_vram_mib"] = sum(
        int(float(mem)) for pid, mem in apps if pid.isdigit() and int(pid) in pids
    )
    return usage


class ResourceMonitor:
    """Append resource rows for a process tree to a CSV from a background thread."""

    def __init__(self, path: Path, interval: float = 60.0, root_pid: int | None = None):
        self.path = Path(path)
        self.interval = interval
        self.root = psutil.Process(root_pid)
        self.stage = ""
        self._cpu_seen: dict[int, float] = {}
        self._last_time: float | None = None
        self._peaks: dict[str, dict[str, float]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    def _tree(self) -> list[psutil.Process]:
        try:
            return [self.root, *self.root.children(recursive=True)]
        except psutil.Error:
            return []

    def sample(self) -> dict:
        """One row: the tree's RAM and CPU, the machine's RAM, and the GPUs."""
        now = time.time()
        ram = 0
        cpu_used = 0.0
        seen: dict[int, float] = {}
        for process in self._tree():
            try:
                with process.oneshot():
                    ram += process.memory_info().rss
                    times = process.cpu_times()
            except psutil.Error:  # exited between listing and reading
                continue
            total = times.user + times.system
            seen[process.pid] = total
            # A process new since the last row counts all its CPU time so far.
            cpu_used += total - self._cpu_seen.get(process.pid, 0.0)
        elapsed = now - self._last_time if self._last_time else None
        self._cpu_seen, self._last_time = seen, now
        memory = psutil.virtual_memory()
        return {
            "time_utc": datetime.fromtimestamp(now, UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "stage": self.stage,
            "pipeline_ram_gib": round(ram / _GIB, 2),
            "pipeline_cpu_cores": round(max(cpu_used, 0.0) / elapsed, 2) if elapsed else "",
            "system_ram_used_gib": round((memory.total - memory.available) / _GIB, 2),
            "system_ram_total_gib": round(memory.total / _GIB, 2),
            **gpu_usage(set(seen)),
        }

    def record(self) -> None:
        """Sample once, append the row and update the stage's peaks."""
        row = self.sample()
        with self._lock:
            new = not self.path.exists() or self.path.stat().st_size == 0
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS, restval="")
                if new:
                    writer.writeheader()
                writer.writerow(row)
            peaks = self._peaks.setdefault(row["stage"], {})
            for column in PEAK_COLUMNS:
                value = row.get(column)
                if isinstance(value, int | float):
                    peaks[column] = max(peaks.get(column, value), value)

    def _loop(self) -> None:
        while True:
            # Monitoring must never stop the run: a failed sample skips the row.
            with contextlib.suppress(Exception):
                self.record()
            self._wake.wait(self.interval)
            self._wake.clear()
            if self._stop.is_set():
                return

    def start(self) -> "ResourceMonitor":
        self._thread = threading.Thread(target=self._loop, name="resource-monitor", daemon=True)
        self._thread.start()
        return self

    def set_stage(self, stage: str) -> None:
        """Label later rows with ``stage`` and take a row now."""
        self.stage = stage
        self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=30)

    def peak_line(self, stage: str) -> str:
        """``peak use during <stage>: ...`` for the log, from this stage's rows."""
        with self._lock:
            peaks = dict(self._peaks.get(stage, {}))
        if not peaks:
            return f"peak use during {stage}: no samples (stage shorter than {self.interval:g} s)"
        parts = []
        if "pipeline_ram_gib" in peaks:
            parts.append(f"RAM {peaks['pipeline_ram_gib']:.1f} GiB")
        if "pipeline_cpu_cores" in peaks:
            parts.append(f"CPU {peaks['pipeline_cpu_cores']:.1f} cores")
        if "gpu_mem_used_mib" in peaks:
            parts.append(
                f"GPU memory {peaks['gpu_mem_used_mib'] / 1024:.1f} GiB "
                f"({peaks.get('pipeline_vram_mib', 0) / 1024:.1f} GiB by this run)"
            )
        return f"peak use during {stage}: " + ", ".join(parts)
