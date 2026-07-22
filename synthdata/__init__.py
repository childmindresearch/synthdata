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

# Load .env (TABPFN_TOKEN, HF_TOKEN, PYTORCH_CUDA_ALLOC_CONF, ...) into the
# process environment as early as possible: variables like
# PYTORCH_CUDA_ALLOC_CONF only take effect if set *before* the CUDA context is
# initialized (i.e. before anything imports torch), and merely having them in
# .env does nothing on its own -- nothing else in this package reads that file.
from dotenv import load_dotenv

load_dotenv()

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
import os  # noqa: E402

os.environ.setdefault(
    "PYTHONWARNINGS", "ignore:Clustering metrics expects discrete values:UserWarning"
)

from synthdata.config import Config, load_config  # noqa: E402

__all__ = ["Config", "load_config"]
