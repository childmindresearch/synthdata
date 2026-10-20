"""Run a generator that lives in its own environment (``envs/<name>``) as a subprocess.

Some generators pin library versions that clash with this repo's (torch,
psutil, Python itself), so each gets its own uv project under ``envs/`` with a
runner script that imports nothing from synthdata. :func:`run_in_env` writes
the train rows to parquet and a JSON spec, runs
``uv run --project <env> --no-sync python <runner> --train ... --spec ... --out ...``,
streams its output to the log, and reads the synthetic rows back.

Parquet needs one type per column, so categorical columns travel as strings:
:func:`encode_categories` writes them, :func:`decode_categories` maps them
back to the train values and casts numeric columns back to their train dtypes.
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from synthdata.utils import get_logger

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_env_dir(env_dir: str | Path) -> Path:
    """``env_dir`` as an absolute path (relative paths are from the repo root)."""
    path = Path(env_dir)
    return path if path.is_absolute() else REPO_ROOT / path


def encode_categories(train_df: pd.DataFrame, categorical: list) -> tuple[pd.DataFrame, dict]:
    """Categorical columns as strings, plus the map from each string back to its train value."""
    out = train_df.copy()
    value_maps = {}
    for col in categorical:
        values = train_df[col].dropna().unique()
        value_maps[col] = {str(v): v for v in values}
        out[col] = train_df[col].map(lambda v: None if pd.isna(v) else str(v)).astype(object)
    return out, value_maps


def decode_categories(
    synthetic: pd.DataFrame, train_df: pd.DataFrame, value_maps: dict, model: str
) -> pd.DataFrame:
    """Undo :func:`encode_categories`: train values for categories, train dtypes for numbers."""
    missing = [c for c in train_df.columns if c not in synthetic]
    if missing:
        raise ValueError(f"{model} output lacks columns {missing[:5]}")
    out = synthetic[list(train_df.columns)].copy()
    for col in train_df.columns:
        if col in value_maps:
            mapping = value_maps[col]
            unknown = set(out[col].dropna().astype(str)) - set(mapping)
            if unknown:
                raise ValueError(
                    f"{model} made categories of {col!r} not in train: {sorted(unknown)[:5]}"
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


def run_in_env(
    env_dir: str | Path, runner: str, train_df: pd.DataFrame, spec: dict, model: str
) -> pd.DataFrame:
    """Run ``<env_dir>/<runner>`` on ``train_df`` with ``spec`` and return its synthetic rows."""
    env = resolve_env_dir(env_dir)
    script = env / runner
    if not script.exists():
        raise FileNotFoundError(f"{model} runner not found at {script}")
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(f"{model} runs through uv (uv run --project {env_dir})")

    with tempfile.TemporaryDirectory(prefix=f"synthdata_{model}_") as tmp:
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
            str(script),
            "--train",
            str(tmp / "train.parquet"),
            "--spec",
            str(tmp / "spec.json"),
            "--out",
            str(tmp / "synthetic.parquet"),
        ]
        # The parent's virtualenv must not leak into the child's interpreter.
        child_env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        # Progress bars redraw many times a second; one log line per minute is enough.
        child_env.setdefault("TQDM_MININTERVAL", "60")
        logger.info("[%s] running %s", model, " ".join(cmd))
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
                logger.info("[%s] %s", model, line)
                tail = (tail + [line])[-20:]
        if proc.wait() != 0:
            raise RuntimeError(
                f"{model} subprocess failed (exit {proc.returncode}); if its packages are "
                f"missing, run `uv sync --project {env_dir}` first:\n" + "\n".join(tail)
            )
        return pd.read_parquet(tmp / "synthetic.parquet")
