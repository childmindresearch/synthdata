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


def available_memory_gib() -> float | None:
    """Return Linux MemAvailable in GiB, or None where it cannot be observed."""
    try:
        lines = Path("/proc/meminfo").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    return None


def resolve_workers(compute_cfg, *, n_models: int, n_columns: int) -> int:
    """Resolve a memory- and CPU-bounded number of concurrent model processes.

    The Linux ``MemAvailable`` value is already the kernel's estimate of
    immediately allocatable memory. It is therefore the only host-memory
    value used for auto-sizing: ``MemTotal`` can describe a smaller container
    or runner limit than the available-memory probe supplied by callers/tests,
    making worker selection depend on an unrelated second system read.
    """
    if not n_models:
        return 0
    requested = compute_cfg.workers
    if requested != "auto":
        return min(requested, compute_cfg.max_workers, n_models)

    cpu_count = os.cpu_count() or 1
    cpu_bound = max(1, cpu_count // compute_cfg.cores_per_worker)
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
    saved_env = {var: os.environ.get(var) for var in THREAD_ENV_VARS}
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
