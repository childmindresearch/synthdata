"""Shared low-level helpers: seeding, device selection, logging, simple CSV/JSON caching."""

import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_LOGGERS: dict = {}


def get_logger(name: str = "synthdata") -> logging.Logger:
    """Return a module-level logger with a simple stream handler (idempotent)."""
    if name in _LOGGERS:
        return _LOGGERS[name]
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    _LOGGERS[name] = logger
    return logger


def set_global_seed(seed: int) -> None:
    """Seed python/numpy/torch (if available) RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


#: Separates a model name from its replicate index in synthetic dataset names.
REPLICATE_SEPARATOR = "__rep"


def replicate_name(name: str, replicate: int) -> str:
    """Dataset name for replicate ``replicate`` (counted from 0) of ``name``: ``ctgan__rep1``, ``ctgan__rep2``, ..."""
    return f"{name}{REPLICATE_SEPARATOR}{replicate + 1}"


def split_replicate_name(name: str) -> tuple[str, int]:
    """Inverse of :func:`replicate_name`: ``"ctgan__rep2"`` -> ``("ctgan", 1)``.

    A name without the suffix (a dataset from before every replicate carried
    one) counts as the first replicate.
    """
    base, sep, index = str(name).rpartition(REPLICATE_SEPARATOR)
    if sep and base and index.isdigit() and int(index) >= 1:
        return base, int(index) - 1
    return str(name), 0


def model_sort_key(name: str) -> tuple:
    """Order of models in every figure and table: generators by name, each tuned
    model right after its untuned one, baselines last; replicates in turn.

    ``sorted(models, key=model_sort_key)`` gives the same order whatever
    order the models were scored or ranked in, so a model sits in the same
    place in every plot.
    """
    base, replicate = split_replicate_name(str(name))
    tuned = base.endswith("_hpo")
    family = base.removesuffix("_hpo")
    return (family.startswith("baseline_"), family, tuned, replicate)


def resolve_device(device: str = "auto") -> str:
    """Resolve "auto" to the best available torch device string ("cuda"/"mps"/"cpu").

    Note: several synthcity metrics require float64 support, which is not available
    on MPS, so callers that need those metrics should force "cpu" explicitly even if
    "mps" is available (see notebooks' HPO cells).
    """
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def git_commit() -> str | None:
    """Best-effort short git commit hash of the current working tree (None if unavailable)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


#: Packages whose versions decide metric values; recorded with every run.
_PROVENANCE_PACKAGES = (
    "synthcity",
    "syntheval",
    "torch",
    "xgboost",
    "scikit-learn",
    "tabpfn",
    "anonymeter",
    "numpy",
    "pandas",
)


def _git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


#: Environment variables that change results or their reproducibility.
_PROVENANCE_ENV = (
    "CUDA_VISIBLE_DEVICES",
    "CUBLAS_WORKSPACE_CONFIG",
    "PYTHONHASHSEED",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _torch_build() -> dict | None:
    """torch's version and the CUDA/cuDNN it was built with.

    Read without creating a CUDA context (no ``torch.cuda`` call that
    initializes the device), so worker processes forked later are unaffected.
    """
    try:
        import torch
    except ImportError:
        return None
    try:
        cudnn = torch.backends.cudnn.version()
    except (RuntimeError, AttributeError, OSError):
        cudnn = None
    return {"version": torch.__version__, "cuda": torch.version.cuda, "cudnn": cudnn}


def _gpus() -> list | None:
    """``nvidia-smi``'s GPU name, driver and memory per device (None without it)."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip().splitlines() if result.returncode == 0 else None


def provenance() -> dict:
    """Best-effort record of the code, packages and machine a run used.

    The root commit alone does not pin a run: the submodule checkouts or the
    working tree can differ from it, and goldens differ by platform and CUDA
    build. ``git_diff_sha256`` fingerprints uncommitted changes to tracked
    files, so two dirty runs can be told apart. ``lockfile_sha256`` pins the
    resolved dependency set (``uv.lock``, and ``envs/*/uv.lock`` for the
    generators in their own environments); ``all_packages`` lists what was
    actually installed, which can differ from the lock. Every field is
    ``None`` when it cannot be read (no git, not a checkout, package or tool
    missing); this never raises.
    """
    from importlib.metadata import PackageNotFoundError, distributions, version

    status = _git_output("status", "--porcelain", "--untracked-files=no")
    submodules = _git_output("submodule", "status")
    diff = _git_output("diff", "HEAD", "--submodule=diff") if status else None
    root = _git_output("rev-parse", "--show-toplevel")
    versions = {}
    for package in _PROVENANCE_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    installed = {}
    for dist in distributions():
        name = dist.metadata["Name"]
        if name:
            installed[name.lower()] = dist.version
    return {
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "git_diff_sha256": None if diff is None else hashlib.sha256(diff.encode()).hexdigest(),
        # One line per submodule: "<sha> <path> (<describe>)"; a leading "+"
        # means the checkout differs from the commit the root repo pins.
        "submodules": None if submodules is None else submodules.splitlines(),
        "lockfile_sha256": None if root is None else _sha256(Path(root) / "uv.lock"),
        # TabularARGN, CDTD and Desbordante run in their own environments
        # (``envs/<name>``, see ``generation.sidecar``), pinned by their own locks.
        "env_lockfiles_sha256": None
        if root is None
        else {p.parent.name: _sha256(p) for p in sorted(Path(root).glob("envs/*/uv.lock"))},
        "packages": versions,
        "all_packages": dict(sorted(installed.items())),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": _torch_build(),
        "gpus": _gpus(),
        "env": {name: os.environ.get(name) for name in _PROVENANCE_ENV},
    }


def cached_csv(
    path: str | Path,
    build_fn: Callable[[], pd.DataFrame],
    use_cache: bool = True,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Load a DataFrame from ``path`` if it exists (and use_cache), else build+save it."""
    p = Path(path)
    if use_cache and p.exists():
        return pd.read_csv(p, **read_csv_kwargs)
    df = build_fn()
    ensure_dir(p.parent)
    df.to_csv(p, index=False)
    return df


def cached_json(
    path: str | Path,
    build_fn: Callable[[], dict],
    use_cache: bool = True,
) -> dict:
    """Load a dict from a JSON file if it exists (and use_cache), else build+save it."""
    p = Path(path)
    if use_cache and p.exists():
        with open(p) as f:
            return json.load(f)
    data = build_fn()
    ensure_dir(p.parent)
    with open(p, "w") as f:
        json.dump(data, f, indent=2)
    return data


def load_json(path: str | Path, default: dict | None = None) -> dict:
    p = Path(path)
    if not p.exists():
        return {} if default is None else default
    with open(p) as f:
        return json.load(f)


def save_json(path: str | Path, data: dict) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w") as f:
        json.dump(data, f, indent=2)


#: Characters some libraries reject in column names, each with a readable
#: stand-in: LightGBM refuses ``[ ] { } " : ,`` (SynthEval's MIA fits it) and
#: XGBoost refuses ``[ ] <``.
_UNSAFE_NAME_CHARACTERS = {
    "[": "(",
    "]": ")",
    "{": "(",
    "}": ")",
    '"': "'",
    ":": " -",
    ",": ";",
    "<": "lt",
}


def library_safe_names(columns) -> dict:
    """``{name: safe name}`` for the columns whose names a library would reject.

    Each unsafe character becomes a readable stand-in (``"BP [mmHg]"`` ->
    ``"BP (mmHg)"``), so the renamed column is still recognisable. A safe name
    that would collide with another column, or with another renamed one, gets
    a ``__2``, ``__3``... suffix, so the mapping is always one to one and can be
    inverted. Columns that need no change are left out.
    """
    names = [str(c) for c in columns]
    taken = set(names)
    mapping = {}
    for name in names:
        safe = name
        for char, stand_in in _UNSAFE_NAME_CHARACTERS.items():
            safe = safe.replace(char, stand_in)
        if safe == name:
            continue
        candidate, k = safe, 2
        while candidate in taken:
            candidate, k = f"{safe}__{k}", k + 1
        taken.add(candidate)
        mapping[name] = candidate
    return mapping
