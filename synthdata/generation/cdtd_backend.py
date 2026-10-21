"""CDTD (continuous diffusion for mixed-type tabular data) as a generator, run in its own environment.

The model is the authors' ``cdtd_simple`` wrapper (Mueller, Gruber & Fok, ICLR
2025), pinned as the submodule ``submodules/cdtd_simple``. It needs Python
3.13 and torch>=2.9.1, so it runs in ``envs/cdtd`` as a subprocess
(``envs/cdtd/run_cdtd.py``, plumbing in :mod:`synthdata.generation.sidecar`).

The wrapper adds no modelling of its own. It declares the schema's categorical
columns (and a categorical target) as CDTD's categorical features and the rest
as continuous, and preprocesses them as the authors' notebook does. CDTD has no
conditional sampling, so for a categorical target with ``match_class_prior``
one fit draws a pool of ``pool_factor`` times the rows needed, and the class
quotas are filled from that pool by rejection sampling (no refit per batch).
"""

from pathlib import Path

import pandas as pd

from synthdata.generation import class_quota, sidecar

RUNNER = "run_cdtd.py"


def run_cdtd(
    train_df: pd.DataFrame,
    categorical: list,
    *,
    seed: int,
    n_samples: int,
    env_dir: str | Path,
    source_dir: str | Path,
    params: dict | None = None,
    fit_params: dict | None = None,
    sample_params: dict | None = None,
) -> pd.DataFrame:
    """Fit CDTD on ``train_df`` in its environment and return ``n_samples`` raw synthetic rows."""
    source = sidecar.resolve_env_dir(source_dir)
    if not (source / "cdtd" / "__init__.py").exists():
        raise FileNotFoundError(
            f"CDTD source not found at {source}; run `git submodule update --init`"
        )
    spec = {
        "source_dir": str(source),
        "categorical": categorical,
        "continuous": [c for c in train_df.columns if c not in set(categorical)],
        "params": params or {},
        "fit_params": fit_params or {},
        "sample_params": sample_params or {},
        "seed": int(seed),
        "n_samples": int(n_samples),
    }
    return sidecar.run_in_env(env_dir, RUNNER, train_df, spec, "cdtd")


def generate_cdtd(
    train_df: pd.DataFrame,
    categorical_columns: list,
    target_column: str,
    target_is_categorical: bool,
    n_samples: int,
    seed: int,
    env_dir: str | Path,
    source_dir: str | Path,
    params: dict | None = None,
    fit_params: dict | None = None,
    sample_params: dict | None = None,
    class_prior: pd.Series | None = None,
    pool_factor: float = 3.0,
) -> pd.DataFrame:
    """One CDTD fit on ``train_df`` and ``n_samples`` synthetic rows.

    With ``class_prior`` (a categorical target), the quota report is attached
    as ``df.attrs["class_sampling"]``.
    """
    categorical = [c for c in train_df.columns if c in set(categorical_columns)]
    if target_is_categorical and target_column not in categorical:
        categorical.append(target_column)
    encoded, value_maps = sidecar.encode_categories(train_df, categorical)

    pool_size = n_samples if class_prior is None else max(n_samples, round(pool_factor * n_samples))
    raw = run_cdtd(
        encoded,
        categorical,
        seed=seed,
        n_samples=pool_size,
        env_dir=env_dir,
        source_dir=source_dir,
        params=params,
        fit_params=fit_params,
        sample_params=sample_params,
    )
    pool = sidecar.decode_categories(raw, train_df, value_maps, "cdtd")
    if class_prior is None:
        return pool

    # The whole pool is one batch: no refit for top-ups.
    df, report = class_quota.sample_to_quota(
        lambda count, _seed, _labels: pool,
        target_column,
        class_prior,
        n_samples,
        seed,
        first_batch=pool,
        max_rounds=1,
    )
    df.attrs["class_sampling"] = report
    return df
