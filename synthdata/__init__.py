"""synthdata: imputation, synthetic data generation, and evaluation pipeline.

This package turns the exploratory work in ``notebooks/test_hepatitis_data.ipynb``
and ``notebooks/ctgan_hpo_hepatitis.ipynb`` into reusable, config-driven modules:

- :mod:`synthdata.data` -- generic dataset loading (UCI or local CSV), typing, splitting
- :mod:`synthdata.imputation` -- TabImpute-based missing data imputation
- :mod:`synthdata.generation` -- synthcity + TabPFN + TabPFGen synthetic data generation, with
  Optuna hyperparameter optimization
- :mod:`synthdata.evaluation` -- combined synthcity + SynthEval + custom (fairness/log-disparity)
  evaluation, merged into a single ranked, multi-index table
- :mod:`synthdata.plotting` -- all figures produced across the pipeline

Everything is driven by a single YAML config file (see ``configs/config.yaml``), loaded via
:func:`synthdata.config.load_config`.
"""

import ctypes
import os
import platform
from pathlib import Path

# Load .env (TABPFN_TOKEN, HF_TOKEN, PYTORCH_CUDA_ALLOC_CONF, ...) into the
# process environment as early as possible: variables like
# PYTORCH_CUDA_ALLOC_CONF only take effect if set *before* the CUDA context is
# initialized (i.e. before anything imports torch), and merely having them in
# .env does nothing on its own -- nothing else in this package reads that file.
from dotenv import load_dotenv

from synthdata.config import Config, load_config

load_dotenv()


def _preload_system_nvrtc() -> object | None:
    """Make the system CUDA NVRTC linker name visible to KeOps on ARM64.

    KeOps 2.3 links with ``-lnvrtc``, while Torch's pip-installed CUDA 13
    runtime exposes only ``libnvrtc.so.13``. Preloading a system toolkit's
    unversioned linker symlink lets KeOps select a linkable CUDA directory
    before Torch loads its packaged runtime library.
    """
    if platform.system() != "Linux" or platform.machine() not in {"aarch64", "arm64"}:
        return

    cuda_roots = [
        Path(value)
        for variable in ("CUDA_PATH", "CUDA_HOME")
        if (value := os.environ.get(variable))
    ]
    cuda_roots.extend((Path("/usr/local/cuda"), Path("/opt/cuda")))
    target = "sbsa-linux"

    candidates = []
    for root in cuda_roots:
        candidates.extend(
            (
                root / "targets" / target / "lib" / "libnvrtc.so",
                root / "lib64" / "libnvrtc.so",
                root / "lib" / "libnvrtc.so",
            )
        )

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        try:
            return ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
    return None


_system_nvrtc_handle = _preload_system_nvrtc()

# Force a non-interactive matplotlib backend before anything else (including
# SynthEval/synthcity internals) can import matplotlib.pyplot and pick a GUI
# backend (e.g. "macosx"). This package only ever saves figures to disk
# (savefig/write_html/write_image) and never displays them interactively, so
# there is no reason for a Python/plot window (or its Dock icon) to appear.
import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)

# pgmpy's TreeSearch (used by synthcity's "bayesian_network" plugin for
# Chow-Liu/TAN structure learning) scores candidate edges with sklearn's
# mutual-information *clustering* metrics, which warn loudly whenever a
# column looks continuous/multiclass rather than a strict clustering label --
# harmless here since it's being (ab)used as an information-theoretic score,
# not an actual clustering evaluation. These warnings fire inside joblib
# worker subprocesses, so a `warnings.filterwarnings()` call in this process
# doesn't reach them; setting PYTHONWARNINGS before those subprocesses spawn
# does, since child processes inherit the environment.
os.environ.setdefault(
    "PYTHONWARNINGS", "ignore:Clustering metrics expects discrete values:UserWarning"
)

__all__ = ["Config", "load_config"]
