"""Share the GPUs between models that train in parallel, longest remaining work first.

When generation trains several synthcity models at once (see
:mod:`synthdata.compute`), each GPU hands out ``compute.fits_per_gpu`` slots.
A model's process holds a slot only while it fits a generator and samples
from it, and gives it back for its CPU work (TSTR scoring, screens, saving),
so:

* no more than ``fits_per_gpu`` fits share a GPU at once. Six fits on one
  GPU take turns and each runs about six times slower; two keep the GPU busy
  without that;
* every fit (an HPO trial, an untuned or a tuned fit) takes a slot on
  whichever GPU has one free, so with two GPUs the work spreads over both as
  it happens, with nothing decided in advance;
* when processes wait for a slot, the model with the most estimated work
  left goes first, so the model that would finish last (the critical one)
  is not slowed down by the others.

A model's remaining work is its fits still to come (untuned and tuned fits,
HPO trials) times its seconds per training epoch: first a rough prior per
model, then the speed measured on its own finished fits. Nothing waits for a
measurement. Slots never change what a fit computes (same seeds and
settings), only when and on which GPU it runs.

The parent sets :data:`SLOTS_ENV_VAR` to a folder its worker processes
share; the slot table there is guarded by a file lock, and slots held by a
process that died are freed. Each fit is recorded in ``fits.csv`` in that
folder (model, kind, GPU, epochs, seconds waited and held).
"""

import contextlib
import csv
import json
import os
import time
from pathlib import Path

import psutil
from filelock import FileLock

from synthdata.utils import get_logger

logger = get_logger(__name__)

#: Folder of the shared slot table; set by the parent for its workers.
SLOTS_ENV_VAR = "SYNTHDATA_GPU_SLOTS"

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

#: Seconds between checks while waiting for a slot.
POLL_SECONDS = 2.0


def resolve_fits_per_gpu(value) -> int:
    """``compute.fits_per_gpu`` as a number of slots; 0 turns the slots off."""
    if value == "auto":
        return 2
    return int(value or 0)


def setup(folder: Path, n_gpus: int, fits_per_gpu: int) -> None:
    """Create the slot table for ``n_gpus`` GPUs and point worker processes at it."""
    folder.mkdir(parents=True, exist_ok=True)
    table = {"slots": fits_per_gpu, "gpus": {str(i): [] for i in range(n_gpus)}, "waiting": {}}
    with FileLock(str(folder / "table.lock")):
        (folder / "table.json").write_text(json.dumps(table))
    os.environ[SLOTS_ENV_VAR] = str(folder)


def teardown() -> None:
    os.environ.pop(SLOTS_ENV_VAR, None)


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


def _update(folder: Path, change):
    """Apply ``change(table)`` under the lock, after freeing dead processes' slots."""
    with FileLock(str(folder / "table.lock")):
        path = folder / "table.json"
        table = json.loads(path.read_text())
        for gpu, holders in table["gpus"].items():
            table["gpus"][gpu] = [h for h in holders if _alive(h["pid"])]
        table["waiting"] = {key: w for key, w in table["waiting"].items() if _alive(w["pid"])}
        result = change(table)
        path.write_text(json.dumps(table))
        return result


def _try_take(table: dict, key: str, priority: float) -> str | None:
    """Take a slot on the least busy GPU if one is free and no one ahead is waiting."""
    table["waiting"][key]["priority"] = priority
    free = [gpu for gpu, holders in table["gpus"].items() if len(holders) < table["slots"]]
    if not free:
        return None
    me = (-priority, table["waiting"][key]["since"])
    ahead = [
        other
        for other_key, other in table["waiting"].items()
        if other_key != key and (-other["priority"], other["since"]) < me
    ]
    # Everyone ahead of us gets a slot first; there must be one left for us.
    if len(ahead) >= sum(table["slots"] - len(table["gpus"][gpu]) for gpu in free):
        return None
    gpu = min(free, key=lambda g: (len(table["gpus"][g]), int(g)))
    holder = dict(table["waiting"].pop(key))
    table["gpus"][gpu].append(holder)
    return gpu


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
        yield device
        work.done(epochs, time.monotonic() - started)
        return

    folder = Path(folder)
    key = f"{os.getpid()}:{kind}:{time.monotonic_ns()}"
    asked = time.time()
    _update(
        folder,
        lambda table: table["waiting"].__setitem__(
            key,
            {
                "pid": os.getpid(),
                "model": model,
                "kind": kind,
                "priority": work.seconds_left(),
                "since": asked,
            },
        ),
    )
    try:
        while (gpu := _update(folder, lambda t: _try_take(t, key, work.seconds_left()))) is None:
            time.sleep(POLL_SECONDS)
    except BaseException:
        _update(folder, lambda table: table["waiting"].pop(key, None))
        raise
    waited = time.time() - asked
    if waited > 60:
        logger.info("[%s] %s got cuda:%s after waiting %.0f min", model, kind, gpu, waited / 60)
    started = time.monotonic()
    try:
        yield f"cuda:{gpu}"
    finally:
        held = time.monotonic() - started

        def release(table):
            table["gpus"][gpu] = [
                h
                for h in table["gpus"][gpu]
                if not (h["pid"] == os.getpid() and h["since"] == asked)
            ]

        _update(folder, release)
        work.done(epochs, held)
        _record(folder, model, kind, gpu, epochs, waited, held, asked)


def _record(folder, model, kind, gpu, epochs, waited, held, asked) -> None:
    path = folder / "fits.csv"
    row = {
        "model": model,
        "kind": kind,
        "gpu": gpu,
        "epochs": epochs,
        "asked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(asked)),
        "waited_s": round(waited),
        "held_s": round(held),
    }
    with FileLock(str(folder / "fits.lock")):
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if new:
                writer.writeheader()
            writer.writerow(row)
