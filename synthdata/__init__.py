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

# Force a non-interactive matplotlib backend before anything else (including
# SynthEval/synthcity internals) can import matplotlib.pyplot and pick a GUI
# backend (e.g. "macosx"). This package only ever saves figures to disk
# (savefig/write_html/write_image) and never displays them interactively, so
# there is no reason for a Python/plot window (or its Dock icon) to appear.
import matplotlib

matplotlib.use("Agg", force=True)

from synthdata.config import Config, load_config

__all__ = ["Config", "load_config"]
