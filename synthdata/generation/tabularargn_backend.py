"""MOSTLY AI's TabularARGN (``mostlyai-engine``) as a generator, run in its own environment.

mostlyai-engine pins one torch minor per release and ``psutil<6``, which
clash with this repo's torch and with synthcity, so it lives in
``envs/tabularargn`` and runs as a subprocess (``envs/tabularargn/run_tabularargn.py``).
This module writes the train split and a JSON spec, runs the script with
``uv run --project envs/tabularargn --no-sync``, and reads the synthetic rows back.

The wrapper adds no modelling of its own. It only:

* declares each column's encoding type (categorical columns, including numeric
  codes, as ``TABULAR_CATEGORICAL``; the rest ``TABULAR_NUMERIC_AUTO``), since
  the library would otherwise treat numeric codes as numbers;
* writes categorical values as strings (parquet needs one type per column) and
  maps them back to the train values afterwards, and casts numeric columns back
  to their train dtypes;
* for a categorical target with ``match_class_prior``, fixes the target to the
  class quotas as seed data, so each class gets exactly its quota from one fit.
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from synthdata.generation import class_quota
from synthdata.utils import get_logger

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = "run_tabularargn.py"


def _resolve_env_dir(env_dir: str | Path) -> Path:
    path = Path(env_dir)
    return path if path.is_absolute() else REPO_ROOT / path


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


def to_engine_frame(train_df: pd.DataFrame, encodings: dict) -> tuple[pd.DataFrame, dict]:
    """Categorical columns as strings, plus the map from each string back to its train value."""
    out = train_df.copy()
    value_maps = {}
    for col, kind in encodings.items():
        if kind != "TABULAR_CATEGORICAL":
            continue
        values = train_df[col].dropna().unique()
        value_maps[col] = {str(v): v for v in values}
        out[col] = train_df[col].map(lambda v: None if pd.isna(v) else str(v)).astype(object)
    return out, value_maps


def from_engine_frame(
    synthetic: pd.DataFrame, train_df: pd.DataFrame, value_maps: dict
) -> pd.DataFrame:
    """Undo :func:`to_engine_frame`: train values for categories, train dtypes for numbers."""
    missing = [c for c in train_df.columns if c not in synthetic]
    if missing:
        raise ValueError(f"TabularARGN output lacks columns {missing[:5]}")
    out = synthetic[list(train_df.columns)].copy()
    for col in train_df.columns:
        if col in value_maps:
            mapping = value_maps[col]
            unknown = set(out[col].dropna().astype(str)) - set(mapping)
            if unknown:
                raise ValueError(
                    f"TabularARGN made categories of {col!r} not in train: {sorted(unknown)[:5]}"
                )
            out[col] = out[col].map(lambda v, m=mapping: np.nan if pd.isna(v) else m[str(v)])
            if not out[col].isna().any():
                out[col] = out[col].astype(train_df[col].dtype)
        else:
            values = pd.to_numeric(out[col], errors="coerce")
            dtype = train_df[col].dtype
            if pd.api.types.is_integer_dtype(dtype) and not values.isna().any():
                values = values.round().astype(dtype)
            out[col] = values
    return out.reset_index(drop=True)


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
    env = _resolve_env_dir(env_dir)
    runner = env / RUNNER
    if not runner.exists():
        raise FileNotFoundError(f"TabularARGN runner not found at {runner}")
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("TabularARGN runs through uv (uv run --project envs/tabularargn)")

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
    with tempfile.TemporaryDirectory(prefix="synthdata_argn_") as tmp:
        tmp = Path(tmp)
        train_df.to_parquet(tmp / "train.parquet", index=False)
        (tmp / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
        cmd = [
            uv,
            "run",
            "--project",
            str(env),
            # Never install mid-run; the environment is set up once beforehand.
            "--no-sync",
            "python",
            str(runner),
            "--train",
            str(tmp / "train.parquet"),
            "--spec",
            str(tmp / "spec.json"),
            "--out",
            str(tmp / "synthetic.parquet"),
        ]
        # The parent's virtualenv must not leak into the child's interpreter.
        child_env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        logger.info("[tabularargn] running %s", " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=child_env,
        )
        tail: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                logger.info("[tabularargn] %s", line)
                tail = (tail + [line])[-20:]
        if proc.wait() != 0:
            raise RuntimeError(
                f"TabularARGN subprocess failed (exit {proc.returncode}); if mostlyai is "
                f"missing, run `uv sync --project {env_dir}` first:\n" + "\n".join(tail)
            )
        return pd.read_parquet(tmp / "synthetic.parquet")


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
    engine_df, value_maps = to_engine_frame(train_df, encodings)

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
        return from_engine_frame(raw, train_df, value_maps)

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
