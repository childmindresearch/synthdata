"""Share the GPUs between models that train in parallel, longest remaining work first.

When generation trains several synthcity models at once (see
:mod:`synthdata.compute`), every fit (an HPO trial, an untuned or a tuned
fit) goes to the GPU running the fewest fits at that moment, so with two
GPUs the work spreads over both as it happens, with nothing decided in
advance. A model's process holds its place on the GPU only while it fits a
generator and samples from it, not during its CPU work (TSTR scoring,
screens, saving).

With ``compute.fits_per_gpu: auto`` there is no limit on the fits sharing a
GPU: measured on the LORIS data (scripts/gpu_concurrency.py), CTGAN fits
sharing one GPU trained 1.33 times faster in total at two at once and 1.72
times at six, and a mix of DDPM and CTGAN fits got as much done as CTGAN
alone, so holding fits back only left the GPU part idle. A number caps the
fits per GPU (for example to fit GPU memory); then waiting fits go by
estimated work left, most first, so the model that would finish last is
served first.

A model's remaining work is its fits still to come (untuned and tuned fits,
HPO trials) times its seconds per training epoch: first a rough prior per
model, then the speed measured on its own finished fits. Nothing waits for a
measurement. Slots never change what a fit computes (same seeds and
settings), only when and on which GPU it runs.

GPU memory is shared the same way, from measurements rather than a formula:
a fit is admitted to a GPU only if the memory its holders need plus its own
stays within :data:`MEMORY_FRACTION` of the GPU (or the GPU is empty). A
fit's need is the largest peak measured on that model's earlier fits (in any
process) plus a margin, and a share of the GPU (:data:`PRIOR_VRAM_SHARE`)
until the first fit of the model finishes; an out-of-memory error raises the
model's estimate. A model's memory depends on its data, network size and
batch size together, and on what the caching allocator keeps, so a formula
from the config would be either loose or wrong; one measured fit is exact for
that data. After each fit the process hands its cached memory back to the
driver, so a worker doing CPU work between fits (scoring a trial) holds no
GPU memory.

The parent keeps the slot table in a temporary folder on local disk (under
``TMPDIR``) and sets :data:`SLOTS_ENV_VAR` to it for its worker processes,
which all run on the same machine; the table is guarded by a file lock, and
slots held by a process that died are freed. It is not kept in the output
folder because that is often a network drive, where file locks and
just-written files are not reliably seen by other processes (a worker's
entry could vanish from the table). Each fit is recorded in ``fits.csv`` in
the output folder (:data:`RECORDS_ENV_VAR`): model, kind, GPU, epochs,
seconds waited and held.
"""

import contextlib
import csv
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import psutil
from filelock import FileLock

from synthdata.utils import get_logger

logger = get_logger(__name__)

#: Folder of the shared slot table; set by the parent for its workers.
SLOTS_ENV_VAR = "SYNTHDATA_GPU_SLOTS"

#: Folder of ``fits.csv``; set by the parent for its workers.
RECORDS_ENV_VAR = "SYNTHDATA_GPU_SLOTS_RECORDS"

#: Rough seconds per training epoch (PATE-GAN: per iteration) used until a
#: model's own fits are measured. From profiling on the 682-column
#: simulated clinical data (scripts/profile_hpo.py); only their ratios
#: matter, and only until the first fit of each model finishes.
PRIOR_SECONDS_PER_EPOCH = {
    "ctgan": 30.0,
    "tvae": 16.0,
    "adsgan": 18.0,
    "rtvae": 19.0,
    "ddpm": 3.1,
    "pategan": 54.0,
}
_DEFAULT_PRIOR = 15.0

#: Models that train on the GPU and so take a slot. Others (ARF, Bayesian
#: network, marginals) accept a ``device`` but train on the CPU, so they
#: never wait for a slot.
GPU_MODELS = frozenset(PRIOR_SECONDS_PER_EPOCH)

#: Share of a GPU's memory a model is assumed to need until one of its fits
#: has been measured. From the 9,000-row simulated clinical run, where
#: single fits peaked at 10-16 GiB (ddpm: 1 GiB) on a 48 GiB GPU.
PRIOR_VRAM_SHARE = {"ddpm": 0.1}
_DEFAULT_VRAM_SHARE = 0.25

#: Share of a GPU's memory the fits on it may need together.
MEMORY_FRACTION = 0.9

#: A measured peak is raised by this factor, plus :data:`CONTEXT_GIB` for the
#: CUDA context, to give a fit's need.
PEAK_MARGIN = 1.2
CONTEXT_GIB = 0.6

#: Factor applied to a model's estimate after an out-of-memory error.
OOM_GROWTH = 1.5

#: A fit kept out by memory this long blocks newer fits from jumping ahead
#: of it, so a large fit is not starved by a stream of small ones.
STARVATION_SECONDS = 600.0

#: Seconds between checks while waiting for a slot.
POLL_SECONDS = 2.0

#: Attempts at reading the slot table before a parse error is raised.
READ_RETRIES = 5


#: ``slots`` value of a table without a limit per GPU.
UNLIMITED = 0


def resolve_fits_per_gpu(value) -> int | None:
    """``compute.fits_per_gpu`` as slots per GPU: None is off, :data:`UNLIMITED` no limit."""
    if value == "auto":
        return UNLIMITED
    return int(value) if value else None


def setup(
    records: Path, n_gpus: int, fits_per_gpu: int, capacity_gib: list[float] | None = None
) -> None:
    """Create the slot table for ``n_gpus`` GPUs and point worker processes at it.

    The table goes in a new local temporary folder; ``records`` is where
    ``fits.csv`` is written. ``capacity_gib`` (memory of each GPU) turns on
    memory-aware admission; without it only ``fits_per_gpu`` limits a GPU.
    """
    records.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="synthdata_gpu_slots_"))
    table = {
        "slots": fits_per_gpu,
        "gpus": {str(i): [] for i in range(n_gpus)},
        "waiting": {},
        "capacity": {str(i): float(c) for i, c in enumerate(capacity_gib or [])},
        "peaks": {},
    }
    with FileLock(str(folder / "table.lock")):
        _write_table(folder / "table.json", table)
    os.environ[SLOTS_ENV_VAR] = str(folder)
    os.environ[RECORDS_ENV_VAR] = str(records)


def teardown() -> None:
    folder = os.environ.pop(SLOTS_ENV_VAR, None)
    os.environ.pop(RECORDS_ENV_VAR, None)
    if folder:
        shutil.rmtree(folder, ignore_errors=True)


class WorkPlan:
    """A model's fits still to come, in epochs, and its measured speed."""

    def __init__(self, model: str):
        self.model = model
        self.epochs_left = 0.0
        self.measured: list[tuple[float, float]] = []  # (seconds, epochs)

    def add(self, fits: int, epochs_per_fit: float) -> None:
        self.epochs_left += fits * epochs_per_fit

    def done(self, epochs: float, seconds: float) -> None:
        self.epochs_left = max(self.epochs_left - epochs, 0.0)
        if epochs > 0:
            self.measured.append((seconds, epochs))

    def seconds_per_epoch(self) -> float:
        if self.measured:
            return sum(s for s, _ in self.measured) / sum(e for _, e in self.measured)
        return PRIOR_SECONDS_PER_EPOCH.get(self.model, _DEFAULT_PRIOR)

    def seconds_left(self) -> float:
        return self.epochs_left * self.seconds_per_epoch()


#: This process's plans, by model.
_PLANS: dict[str, WorkPlan] = {}


def plan(model: str) -> WorkPlan:
    return _PLANS.setdefault(model, WorkPlan(model))


def _alive(pid: int) -> bool:
    try:
        return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def _write_table(path: Path, table: dict) -> None:
    """Replace the table in one step, so no reader ever sees it half written.

    Writing in place truncates the file first; a worker killed between the
    truncate and the write (a stage time limit, an out-of-memory kill) left an
    empty table that every other worker then failed to parse.
    """
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(table))
    os.replace(tmp, path)


def _read_table(path: Path) -> dict:
    """Read the table, retrying a read that comes back empty or cut short.

    On a network file system (NFS) a reader can briefly see a stale or
    partial copy even after an atomic replace; a few short retries ride
    that out before the error is raised.
    """
    for attempt in range(READ_RETRIES):
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            if attempt == READ_RETRIES - 1:
                raise
            time.sleep(0.2 * (attempt + 1))
    raise AssertionError("unreachable")


def _update(folder: Path, change):
    """Apply ``change(table)`` under the lock, after freeing dead processes' slots."""
    with FileLock(str(folder / "table.lock")):
        path = folder / "table.json"
        table = _read_table(path)
        for gpu, holders in table["gpus"].items():
            table["gpus"][gpu] = [h for h in holders if _alive(h["pid"])]
        table["waiting"] = {key: w for key, w in table["waiting"].items() if _alive(w["pid"])}
        result = change(table)
        _write_table(path, table)
        return result


def need_gib(table: dict, model: str, gpu: str | None = None) -> float:
    """Memory (GiB) a fit of ``model`` is expected to need: measured peak, else a prior share."""
    peak = table.get("peaks", {}).get(model)
    if peak is not None:
        return peak * PEAK_MARGIN + CONTEXT_GIB
    capacity = table.get("capacity", {})
    size = capacity.get(gpu) if gpu is not None else max(capacity.values(), default=0.0)
    return PRIOR_VRAM_SHARE.get(model, _DEFAULT_VRAM_SHARE) * (size or 0.0)


def _used_gib(table: dict, gpu: str) -> float:
    return sum(h.get("need", 0.0) for h in table["gpus"][gpu])


def _room(table: dict, gpu: str, need: float) -> bool:
    """Whether a fit needing ``need`` GiB fits next to the fits on ``gpu``."""
    capacity = table.get("capacity", {}).get(gpu)
    if not capacity or not table["gpus"][gpu]:
        return True
    return _used_gib(table, gpu) + need <= MEMORY_FRACTION * capacity


def _try_take(
    table: dict,
    key: str,
    priority: float,
    entry: dict | None = None,
    now: float | None = None,
) -> str | None:
    """Take a slot on the least busy GPU if one is free, has the memory, and no one ahead waits.

    ``entry`` is the waiting entry to put back if it went missing from the
    table, so a lost entry costs a place in the queue rather than the fit.
    """
    if entry is not None:
        table["waiting"].setdefault(key, dict(entry))
    me = table["waiting"][key]
    me["priority"] = priority
    now = time.time() if now is None else now
    # A fit kept out by memory for long has first claim on the memory that frees up.
    if any(
        other_key != key
        and other.get("short_of_memory")
        and other["since"] < me["since"]
        and now - other["since"] > STARVATION_SECONDS
        for other_key, other in table["waiting"].items()
    ):
        return None
    with_slot = [
        gpu
        for gpu, holders in table["gpus"].items()
        if table["slots"] == UNLIMITED or len(holders) < table["slots"]
    ]
    free = [gpu for gpu in with_slot if _room(table, gpu, need_gib(table, me["model"], gpu))]
    me["short_of_memory"] = bool(with_slot) and not free
    if not free:
        return None
    if table["slots"] == UNLIMITED:
        gpu = min(free, key=lambda g: (len(table["gpus"][g]), _used_gib(table, g), int(g)))
        return _hold(table, key, gpu)
    me = (-priority, table["waiting"][key]["since"])
    ahead = [
        other
        for other_key, other in table["waiting"].items()
        if other_key != key and (-other["priority"], other["since"]) < me
    ]
    # Everyone ahead of us gets a slot first; there must be one left for us.
    if len(ahead) >= sum(table["slots"] - len(table["gpus"][gpu]) for gpu in free):
        return None
    gpu = min(free, key=lambda g: (len(table["gpus"][g]), _used_gib(table, g), int(g)))
    return _hold(table, key, gpu)


def _hold(table: dict, key: str, gpu: str) -> str:
    holder = dict(table["waiting"].pop(key))
    holder.pop("short_of_memory", None)
    holder["need"] = need_gib(table, holder["model"], gpu)
    table["gpus"][gpu].append(holder)
    return gpu


def _set_peak(table: dict, model: str, peak_gib: float, grow: bool = False) -> None:
    """Keep the largest peak measured for ``model``; ``grow`` raises it after an OOM."""
    peaks = table.setdefault("peaks", {})
    old = peaks.get(model)
    if grow:
        base = max(old or 0.0, peak_gib, need_gib(table, model))
        peaks[model] = base * OOM_GROWTH
    elif peak_gib > 0:
        peaks[model] = max(old or 0.0, peak_gib)


def note_oom(model: str) -> None:
    """Raise ``model``'s memory estimate after an out-of-memory error (no-op without a table)."""
    folder = os.environ.get(SLOTS_ENV_VAR)
    if folder:
        _update(Path(folder), lambda table: _set_peak(table, model, 0.0, grow=True))


def is_oom(exc: BaseException) -> bool:
    """Whether ``exc`` is a CUDA out-of-memory error."""
    name = type(exc).__name__
    return name == "OutOfMemoryError" or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


def free_cached_memory() -> None:
    """Hand this process's cached GPU memory back to the driver."""
    import gc

    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _start_peak(device: str | None) -> None:
    try:
        import torch
    except ImportError:
        return
    if device and str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch.device(device))


def _peak_gib(device: str | None) -> float | None:
    try:
        import torch
    except ImportError:
        return None
    if device and str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.max_memory_reserved(torch.device(device)) / 2**30
    return None


@contextlib.contextmanager
def gpu_slot(model: str, kind: str, epochs: float, device: str | None):
    """Hold a GPU slot around a fit; yields the device to use.

    Without a slot table (one model at a time, or slots off), on the CPU or
    for a model that trains on the CPU (not in :data:`GPU_MODELS`), yields
    ``device`` unchanged and only does the bookkeeping.
    """
    folder = os.environ.get(SLOTS_ENV_VAR)
    work = plan(model)
    on_gpu = model in GPU_MODELS and device is not None and str(device).startswith("cuda")
    if not folder or not on_gpu:
        started = time.monotonic()
        try:
            yield device
        finally:
            if on_gpu:
                free_cached_memory()
        work.done(epochs, time.monotonic() - started)
        return

    folder = Path(folder)
    key = f"{os.getpid()}:{kind}:{time.monotonic_ns()}"
    asked = time.time()
    entry = {
        "pid": os.getpid(),
        "model": model,
        "kind": kind,
        "priority": work.seconds_left(),
        "since": asked,
    }
    _update(folder, lambda table: table["waiting"].__setitem__(key, dict(entry)))
    try:
        while (
            gpu := _update(folder, lambda t: _try_take(t, key, work.seconds_left(), entry))
        ) is None:
            time.sleep(POLL_SECONDS)
    except BaseException:
        _update(folder, lambda table: table["waiting"].pop(key, None))
        raise
    waited = time.time() - asked
    if waited > 60:
        logger.info("[%s] %s got cuda:%s after waiting %.0f min", model, kind, gpu, waited / 60)
    started = time.monotonic()
    device = f"cuda:{gpu}"
    _start_peak(device)
    try:
        yield device
    finally:
        held = time.monotonic() - started
        peak = _peak_gib(device)
        free_cached_memory()

        def release(table):
            table["gpus"][gpu] = [
                h
                for h in table["gpus"][gpu]
                if not (h["pid"] == os.getpid() and h["since"] == asked)
            ]
            if peak:
                _set_peak(table, model, peak)

        _update(folder, release)
        work.done(epochs, held)
        _record(model, kind, gpu, epochs, waited, held, asked, peak)


def _record(model, kind, gpu, epochs, waited, held, asked, peak=None) -> None:
    records = os.environ.get(RECORDS_ENV_VAR)
    if not records:
        return
    folder = Path(records)
    path = folder / "fits.csv"
    row = {
        "model": model,
        "kind": kind,
        "gpu": gpu,
        "epochs": epochs,
        "asked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(asked)),
        "waited_s": round(waited),
        "held_s": round(held),
        "peak_vram_gib": "" if peak is None else round(peak, 2),
    }
    with FileLock(str(folder / "fits.lock")):
        new = not path.exists()
        # A file from an older version keeps its columns (no peak column).
        fields = list(row) if new else path.open(encoding="utf-8").readline().strip().split(",")
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if new:
                writer.writeheader()
            writer.writerow(row)
