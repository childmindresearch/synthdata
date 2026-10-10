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

A second file, ``<UTC time>_resources_by_task.csv``, splits the run's RAM,
CPU and GPU memory by task: each model's worker process (and anything it
starts) is named by :data:`synthdata.compute.TASK_ENV_VAR`, e.g.
``generation: ctgan`` or ``syntheval full: tvae``; everything else counts as
``main`` (the runner and the stage process, which also runs the models
itself when ``compute.workers`` is 1). After each stage the log names the
heaviest tasks.

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

from synthdata.compute import TASK_ENV_VAR

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

TASK_COLUMNS = ["time_utc", "stage", "task", "ram_gib", "cpu_cores", "vram_mib"]

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


def gpu_usage(pids: set[int]) -> tuple[dict, dict[int, int]]:
    """GPU utilization and memory, overall and for ``pids``, and memory per pid.

    Both are empty without a GPU.
    """
    gpus = _nvidia_smi("--query-gpu=utilization.gpu,memory.used")
    if not gpus:
        return {}, {}
    usage = {
        "gpu_util_pct": round(sum(float(g[0]) for g in gpus) / len(gpus), 1),
        "gpu_mem_used_mib": sum(int(float(g[1])) for g in gpus),
    }
    apps = _nvidia_smi("--query-compute-apps=pid,used_memory") or []
    ours = [(int(pid), mem) for pid, mem in apps if pid.isdigit() and int(pid) in pids]
    # Windows (WDDM) reports per-process memory as "[N/A]": leave the column empty
    # rather than fail the whole row.
    by_pid: dict[int, int] = {}
    for pid, mem in ours:
        if _is_number(mem):
            by_pid[pid] = by_pid.get(pid, 0) + int(float(mem))
    if by_pid or not ours:
        usage["pipeline_vram_mib"] = sum(by_pid.values())
    return usage, by_pid


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


class ResourceMonitor:
    """Append resource rows for a process tree to a CSV from a background thread."""

    def __init__(self, path: Path, interval: float = 60.0, root_pid: int | None = None):
        self.path = Path(path)
        self.interval = interval
        self.root = psutil.Process(root_pid)
        self.stage = ""
        self._cpu_seen: dict[int, float] = {}
        self._last_time: float | None = None
        self.task_path = self.path.with_name(f"{self.path.stem}_by_task.csv")
        self._tasks: dict[int, str] = {}
        self._peaks: dict[str, dict[str, float]] = {}
        self._task_peaks: dict[str, dict[str, dict[str, float]]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    def _tree(self) -> list[psutil.Process]:
        try:
            return [self.root, *self.root.children(recursive=True)]
        except psutil.Error:
            return []

    def _task(self, process: psutil.Process) -> str:
        """The model a process works for, from its environment; ``main`` otherwise."""
        if process.pid not in self._tasks:
            try:
                task = process.environ().get(TASK_ENV_VAR, "main")
            except psutil.Error:  # no access, or exited
                task = "main"
            self._tasks[process.pid] = task
        return self._tasks[process.pid]

    def sample(self) -> tuple[dict, list[dict]]:
        """One row for the run, and one per task: RAM, CPU and GPU memory."""
        now = time.time()
        by_task: dict[str, dict[str, float]] = {}
        seen: dict[int, float] = {}
        task_of: dict[int, str] = {}
        for process in self._tree():
            try:
                with process.oneshot():
                    rss = process.memory_info().rss
                    times = process.cpu_times()
            except psutil.Error:  # exited between listing and reading
                continue
            total = times.user + times.system
            seen[process.pid] = total
            task_of[process.pid] = task = self._task(process)
            use = by_task.setdefault(task, {"ram": 0.0, "cpu": 0.0, "vram": 0.0})
            use["ram"] += rss
            # A process new since the last row counts all its CPU time so far.
            use["cpu"] += max(total - self._cpu_seen.get(process.pid, 0.0), 0.0)
        elapsed = now - self._last_time if self._last_time else None
        self._cpu_seen, self._last_time = seen, now
        self._tasks = {pid: task for pid, task in self._tasks.items() if pid in seen}
        gpu, vram_by_pid = gpu_usage(set(seen))
        for pid, mib in vram_by_pid.items():
            by_task[task_of[pid]]["vram"] += mib
        memory = psutil.virtual_memory()
        stamp = datetime.fromtimestamp(now, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        cpu_total = sum(use["cpu"] for use in by_task.values())
        row = {
            "time_utc": stamp,
            "stage": self.stage,
            "pipeline_ram_gib": round(sum(use["ram"] for use in by_task.values()) / _GIB, 2),
            "pipeline_cpu_cores": round(cpu_total / elapsed, 2) if elapsed else "",
            "system_ram_used_gib": round((memory.total - memory.available) / _GIB, 2),
            "system_ram_total_gib": round(memory.total / _GIB, 2),
            **gpu,
        }
        has_vram = "pipeline_vram_mib" in gpu
        task_rows = [
            {
                "time_utc": stamp,
                "stage": self.stage,
                "task": task,
                "ram_gib": round(use["ram"] / _GIB, 2),
                "cpu_cores": round(use["cpu"] / elapsed, 2) if elapsed else "",
                "vram_mib": int(use["vram"]) if has_vram else "",
            }
            for task, use in sorted(by_task.items())
        ]
        return row, task_rows

    @staticmethod
    def _append(path: Path, columns: list[str], rows: list[dict]) -> None:
        new = not path.exists() or path.stat().st_size == 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, restval="")
            if new:
                writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _update(peaks: dict[str, float], row: dict, columns) -> None:
        for column in columns:
            value = row.get(column)
            if isinstance(value, int | float):
                peaks[column] = max(peaks.get(column, value), value)

    def record(self) -> None:
        """Sample once, append the rows and update the stage's peaks."""
        row, task_rows = self.sample()
        with self._lock:
            self._append(self.path, COLUMNS, [row])
            self._append(self.task_path, TASK_COLUMNS, task_rows)
            self._update(self._peaks.setdefault(row["stage"], {}), row, PEAK_COLUMNS)
            stage_tasks = self._task_peaks.setdefault(row["stage"], {})
            for task_row in task_rows:
                self._update(
                    stage_tasks.setdefault(task_row["task"], {}),
                    task_row,
                    ("ram_gib", "cpu_cores", "vram_mib"),
                )

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

    def heaviest_line(self, stage: str, top: int = 3) -> str | None:
        """``heaviest tasks during <stage>: ...``, the top tasks by peak RAM, CPU and VRAM."""
        with self._lock:
            tasks = {task: dict(peaks) for task, peaks in self._task_peaks.get(stage, {}).items()}
        if len(tasks) < 2:  # nothing to compare when everything ran in one process
            return None
        parts = []
        for column, name, scale, unit in (
            ("ram_gib", "RAM", 1, "GiB"),
            ("cpu_cores", "CPU", 1, "cores"),
            ("vram_mib", "GPU memory", 1 / 1024, "GiB"),
        ):
            ranked = sorted(
                ((peaks[column], task) for task, peaks in tasks.items() if column in peaks),
                reverse=True,
            )[:top]
            if ranked and ranked[0][0] > 0:
                listed = ", ".join(f"{task} {value * scale:.1f}" for value, task in ranked)
                parts.append(f"{name} ({unit}) {listed}")
        if not parts:
            return None
        return f"heaviest tasks during {stage}: " + "; ".join(parts)
