"""Shared low-level helpers: seeding, device selection, logging, simple CSV/JSON caching."""

import json
import logging
import random
import subprocess
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
    """Dataset name for replicate ``replicate`` of ``name`` (replicate 0 keeps ``name``)."""
    return name if replicate == 0 else f"{name}{REPLICATE_SEPARATOR}{replicate}"


def split_replicate_name(name: str) -> tuple[str, int]:
    """Inverse of :func:`replicate_name`: ``"ctgan__rep2"`` -> ``("ctgan", 2)``."""
    base, sep, index = str(name).rpartition(REPLICATE_SEPARATOR)
    if sep and base and index.isdigit():
        return base, int(index)
    return str(name), 0


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


def provenance() -> dict:
    """Best-effort record of the code and packages a run used.

    The root commit alone does not pin a run: the submodule checkouts or the
    working tree can differ from it. Every field is ``None`` when it cannot be
    read (no git, not a checkout, package missing); this never raises.
    """
    from importlib.metadata import PackageNotFoundError, version

    status = _git_output("status", "--porcelain", "--untracked-files=no")
    submodules = _git_output("submodule", "status")
    versions = {}
    for package in _PROVENANCE_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return {
        "git_dirty": None if status is None else bool(status),
        # One line per submodule: "<sha> <path> (<describe>)"; a leading "+"
        # means the checkout differs from the commit the root repo pins.
        "submodules": None if submodules is None else submodules.splitlines(),
        "packages": versions,
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
