"""Run one process per model, as many at once as the ``compute`` config allows.

Generation (each synthcity model's search, final fit and replicates) and
evaluation (each synthetic dataset's SynthEval, synthcity, TSTR and privacy
scores) both go through :func:`run_per_model`, so one config section decides
how the machine's cores and memory are used.
"""

import dataclasses
import multiprocessing
import os
import pickle
import tempfile
import time
import traceback
from collections.abc import Callable
from pathlib import Path

from synthdata.utils import get_logger

logger = get_logger(__name__)


def usable_cpus() -> int:
    """CPUs this process may run on: its affinity mask, else the machine's count.

    A SLURM job (or a container) is often bound to fewer CPUs than the node
    has; ``os.cpu_count()`` would still report the whole node. ``SLURM_CPUS_PER_TASK``
    caps the count on clusters that allocate CPUs without binding them.
    """
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # not available on Windows or macOS
        cpus = os.cpu_count() or 1
    slurm = os.environ.get("SLURM_CPUS_PER_TASK", "")
    if slurm.isdigit() and int(slurm) > 0:
        cpus = min(cpus, int(slurm))
    return max(1, cpus)


def _cgroup_memory_headroom_gib(
    proc_cgroup: Path = Path("/proc/self/cgroup"), root: Path = Path("/sys/fs/cgroup")
) -> float | None:
    """Memory left under this process's cgroup limit (cgroup v2 or v1), or None if unlimited.

    SLURM enforces a job's ``--mem`` through its cgroup; the node's
    MemAvailable knows nothing of that limit.
    """
    try:
        lines = proc_cgroup.read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        hierarchy, controllers, path = (line.split(":", 2) + ["", ""])[:3]
        relative = path.lstrip("/")
        if hierarchy == "0" and controllers == "":  # v2 unified hierarchy
            limit_file, usage_file = "memory.max", "memory.current"
            base = root / relative
        elif "memory" in controllers.split(","):  # v1
            limit_file, usage_file = "memory.limit_in_bytes", "memory.usage_in_bytes"
            base = root / "memory" / relative
        else:
            continue
        # The tightest limit on the path from this cgroup up to the root applies.
        headroom = None
        for directory in (base, *base.parents):
            if not str(directory).startswith(str(root)):
                break
            try:
                limit = (directory / limit_file).read_text().strip()
                usage = int((directory / usage_file).read_text().strip())
            except (OSError, ValueError):
                continue
            # "max" (v2) or a huge number (v1) means no limit at this level.
            if limit.isdigit() and int(limit) < 2**60:
                free = max(0, int(limit) - usage) / 1024**3
                headroom = free if headroom is None else min(headroom, free)
        if headroom is not None:
            return headroom
    return None


def available_memory_gib() -> float | None:
    """RAM (GiB) this process can still use: Linux MemAvailable, capped by a cgroup limit.

    None where neither can be observed (not Linux).
    """
    available = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) / 1024**2
                break
    except OSError:
        pass
    cgroup = _cgroup_memory_headroom_gib()
    if cgroup is not None:
        available = cgroup if available is None else min(available, cgroup)
    return available


def resolve_workers(compute_cfg, *, n_models: int, n_columns: int) -> int:
    """Resolve a memory- and CPU-bounded number of concurrent model processes.

    CPUs are the ones this process may run on (a SLURM job's allocation, not
    the whole node). Memory is Linux ``MemAvailable``, the kernel's estimate
    of immediately allocatable memory, capped by the job's cgroup limit; the
    node's ``MemTotal`` is never used.
    """
    if not n_models:
        return 0
    requested = compute_cfg.workers
    if requested != "auto":
        return min(requested, compute_cfg.max_workers, n_models)

    cpu_bound = max(1, usable_cpus() // compute_cfg.cores_per_worker)
    per_model_gib = compute_cfg.memory_per_worker_gib or max(6.0, 0.0135 * n_columns)
    available_gib = available_memory_gib()
    if available_gib is None:
        memory_bound = 1
    else:
        budget_gib = max(0.0, available_gib - compute_cfg.memory_reserve_gib)
        memory_bound = max(1, int(budget_gib // per_model_gib))
    return max(1, min(n_models, compute_cfg.max_workers, cpu_bound, memory_bound))


#: Thread-pool sizes read by torch, OpenMP, MKL, OpenBLAS and joblib when a
#: process starts; each model's process gets ``compute.cores_per_worker``.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
)

#: Names the model a worker process runs, e.g. ``"generation: ctgan"``; read
#: by :mod:`synthdata.resources`.
TASK_ENV_VAR = "SYNTHDATA_TASK"


def _child(fn, args, result_path: str, cores: int) -> None:
    """Run ``fn(*args)`` in a spawned process and pickle its outcome to ``result_path``."""
    try:
        import torch

        torch.set_num_threads(cores)
    except ImportError:
        pass
    try:
        outcome = {"result": fn(*args)}
    except BaseException as exc:  # noqa: BLE001 - reported to the parent
        outcome = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    with open(result_path, "wb") as f:
        pickle.dump(outcome, f)


@dataclasses.dataclass
class PerModelRun:
    """Outcome of :func:`run_per_model`: results and failures, keyed by model."""

    results: dict
    #: ``{model: "ExceptionType: message"}``; a killed process (for example by
    #: the out-of-memory killer) reports its exit code.
    failures: dict


def run_per_model(
    fn: Callable,
    tasks: dict[str, tuple],
    compute_cfg,
    *,
    n_columns: int,
    label: str,
    isolate: bool = False,
) -> PerModelRun:
    """Call ``fn(*tasks[model])`` for every model, ``compute.workers`` at a time.

    With more than one worker, each call runs in a fresh spawned process (no
    state, CUDA context or memory carried over from another model) whose
    numeric libraries are limited to ``compute.cores_per_worker`` threads.
    One model's failure, even a killed process, never stops the others.
    With one worker the calls run in this process, one after another (in one
    fresh process at a time with ``isolate``), and failures are collected the
    same way. Results come back in task order.
    """
    n_workers = resolve_workers(compute_cfg, n_models=len(tasks), n_columns=n_columns)
    results, failures = {}, {}
    if n_workers <= 1 and not isolate:
        for name, args in tasks.items():
            try:
                results[name] = fn(*args)
            except Exception as exc:  # noqa: BLE001 - reported to the caller
                logger.error("[%s] %s failed: %s", label, name, exc)
                failures[name] = f"{type(exc).__name__}: {exc}"
        return PerModelRun(results, failures)

    cores = compute_cfg.cores_per_worker
    logger.info(
        "[%s] %d models, %d at a time with %d CPU threads each",
        label,
        len(tasks),
        n_workers,
        cores,
    )
    context = multiprocessing.get_context("spawn")
    pending = iter(enumerate(tasks.items()))
    active: dict[str, tuple] = {}
    outcomes: dict[str, dict] = {}
    # Spawned processes inherit the environment, so thread limits set here
    # take effect before their numeric libraries load.
    if os.name != "nt":
        # multiprocessing's helper process starts with the first worker and
        # would inherit that worker's task name; start it before naming any.
        from multiprocessing import resource_tracker

        resource_tracker.ensure_running()
    saved_env = {var: os.environ.get(var) for var in (*THREAD_ENV_VARS, TASK_ENV_VAR)}
    os.environ.update({var: str(cores) for var in THREAD_ENV_VARS})
    try:
        with tempfile.TemporaryDirectory(prefix="synthdata-") as scratch:

            def start_next() -> None:
                item = next(pending, None)
                if item is None:
                    return
                index, (name, args) = item
                result_path = str(Path(scratch) / f"{index}.pkl")
                process = context.Process(
                    target=_child, args=(fn, args, result_path, cores), name=f"{label}-{name}"
                )
                # The spawned process inherits this name, so the resource
                # monitor of synthdata-run can attribute its use to the model.
                os.environ[TASK_ENV_VAR] = f"{label}: {name}"
                process.start()
                active[name] = (process, result_path)
                logger.info("[%s] started %s (pid %s)", label, name, process.pid)

            for _ in range(max(n_workers, 1)):
                start_next()
            while active:
                done = [name for name, (process, _) in active.items() if not process.is_alive()]
                if not done:
                    time.sleep(0.2)
                    continue
                for name in done:
                    process, result_path = active.pop(name)
                    process.join()
                    try:
                        with open(result_path, "rb") as f:
                            outcomes[name] = pickle.load(f)
                    except (OSError, EOFError, pickle.UnpicklingError):
                        outcomes[name] = {
                            "error": f"process exited with code {process.exitcode} "
                            "without a result (killed, possibly out of memory)"
                        }
                    start_next()
    finally:
        for var, value in saved_env.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value

    for name in tasks:
        outcome = outcomes[name]
        if "error" in outcome:
            logger.error(
                "[%s] %s failed: %s\n%s",
                label,
                name,
                outcome["error"],
                outcome.get("traceback", ""),
            )
            failures[name] = outcome["error"]
        else:
            results[name] = outcome["result"]
    return PerModelRun(results, failures)
