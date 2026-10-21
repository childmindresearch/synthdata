"""MOSTLY AI's TabularARGN (``mostlyai-engine``) as a generator, run in its own environment.

mostlyai-engine pins one torch minor per release and ``psutil<6``, which
clash with this repo's torch and with synthcity, so it lives in
``envs/tabularargn`` and runs as a subprocess (``envs/tabularargn/run_tabularargn.py``).
The subprocess plumbing is in :mod:`synthdata.generation.sidecar`.

The wrapper adds no modelling of its own. It only:

* declares each column's encoding type (categorical columns, including numeric
  codes, as ``TABULAR_CATEGORICAL``; the rest ``TABULAR_NUMERIC_AUTO``), since
  the library would otherwise treat numeric codes as numbers;
* passes categories as strings and maps them back (see :mod:`synthdata.generation.sidecar`);
* for a categorical target with ``match_class_prior``, fixes the target to the
  class quotas as seed data, so each class gets exactly its quota from one fit.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from synthdata.generation import class_quota, sidecar

RUNNER = "run_tabularargn.py"


def encoding_types(
    columns: list, categorical_columns: list, target_column: str, target_is_categorical: bool
) -> dict:
    """mostlyai encoding type for every column."""
    categorical = set(categorical_columns)
    if target_is_categorical:
        categorical.add(target_column)
    return {
        c: "TABULAR_CATEGORICAL" if c in categorical else "TABULAR_NUMERIC_AUTO" for c in columns
    }


def run_tabularargn(
    train_df: pd.DataFrame,
    encodings: dict,
    *,
    seed: int,
    device: str,
    env_dir: str | Path,
    params: dict | None = None,
    sample_params: dict | None = None,
    n_samples: int | None = None,
    target_column: str | None = None,
    labels: np.ndarray | None = None,
) -> pd.DataFrame:
    """Fit TabularARGN on ``train_df`` in its environment and return the raw synthetic rows."""
    spec = {
        "encoding_types": encodings,
        "params": params or {},
        "sample_params": sample_params or {},
        "seed": int(seed),
        "device": device,
        "n_samples": None if n_samples is None else int(n_samples),
        "target_column": target_column,
        "labels": None if labels is None else [str(v) for v in labels],
    }
    return sidecar.run_in_env(env_dir, RUNNER, train_df, spec, "tabularargn")


def generate_tabularargn(
    train_df: pd.DataFrame,
    categorical_columns: list,
    target_column: str,
    target_is_categorical: bool,
    n_samples: int,
    seed: int,
    device: str,
    env_dir: str | Path,
    params: dict | None = None,
    sample_params: dict | None = None,
    class_prior: pd.Series | None = None,
) -> pd.DataFrame:
    """One TabularARGN fit on ``train_df`` and ``n_samples`` synthetic rows.

    With ``class_prior`` (a categorical target), the target is fixed to the
    class quotas, and the quota report is attached as ``df.attrs["class_sampling"]``.
    """
    encodings = encoding_types(
        list(train_df.columns), categorical_columns, target_column, target_is_categorical
    )
    categorical = [c for c, kind in encodings.items() if kind == "TABULAR_CATEGORICAL"]
    engine_df, value_maps = sidecar.encode_categories(train_df, categorical)

    def draw(count, labels):
        raw = run_tabularargn(
            engine_df,
            encodings,
            seed=seed,
            device=device,
            env_dir=env_dir,
            params=params,
            sample_params=sample_params,
            n_samples=count,
            target_column=target_column,
            labels=labels,
        )
        return sidecar.decode_categories(raw, train_df, value_maps, "tabularargn")

    if class_prior is None:
        return draw(n_samples, None)

    # Seed data fixes every row's class, so one conditional batch fills every quota.
    df, report = class_quota.sample_to_quota(
        lambda count, _seed, labels: draw(count, labels),
        target_column,
        class_prior,
        n_samples,
        seed,
        conditional=True,
        max_rounds=1,
    )
    df.attrs["class_sampling"] = report
    return df
