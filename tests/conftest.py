"""Shared pytest fixtures for the synthdata test suite.

Fixtures here are deliberately in-memory / tmp_path-rooted so unit tests never
touch the real ``data/``/``output/`` directories or require network access.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from synthdata.config import (
    Config,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    GenerationConfig,
    PlotsConfig,
)
from synthdata.data import Dataset
from synthdata.utils import ensure_dir


@pytest.fixture
def sample_mixed_df() -> pd.DataFrame:
    """A small (30-row) DataFrame mixing numeric, string-categorical, a {1,2}
    binary quirk column, and injected missingness -- for synthdata.data's
    pure column-typing/transform functions.
    """
    rng = np.random.default_rng(0)
    n = 30
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 90, size=n).astype(float),
            "score": rng.normal(50, 10, size=n),
            "smoker": rng.choice(["Light", "Heavy"], size=n),
            "binary_flag": rng.choice([1, 2], size=n),
            "group": rng.integers(0, 3, size=n),
            "target": rng.integers(0, 2, size=n),
        }
    )
    df.loc[df.index[:5], "age"] = np.nan
    df.loc[df.index[5:8], "score"] = np.nan
    return df


@pytest.fixture
def make_config(tmp_path):
    """Factory building a minimal :class:`~synthdata.config.Config` rooted at
    ``tmp_path`` (construction alone performs no I/O).
    """

    def _make_config(
        name: str = "testds",
        tag: str | None = None,
        experiment_id: str | None = None,
    ) -> Config:
        output_root = tmp_path / "output" / name
        return Config(
            name=name,
            data=DataConfig(
                source="csv",
                path=str(tmp_path / "raw.csv"),
                target_column="target",
            ),
            generation=GenerationConfig(output_dir=str(output_root / "synthetic_data")),
            evaluation=EvaluationConfig(output_dir=str(output_root / "evaluation")),
            plots=PlotsConfig(output_dir=str(output_root / "plots")),
            experiment=ExperimentConfig(tag=tag, id=experiment_id),
        )

    return _make_config


@pytest.fixture
def make_dataset(tmp_path, sample_mixed_df):
    """Factory building a :class:`~synthdata.data.Dataset` from small in-memory
    DataFrames, without going through :func:`synthdata.data.load_dataset`'s
    real I/O (UCI fetch/CSV read).
    """

    def _make_dataset(
        df: pd.DataFrame | None = None,
        target_column: str = "target",
        feature_columns: list | None = None,
        categorical_columns: list | None = None,
        sensitive_columns: list | None = None,
        name: str = "testds",
    ) -> Dataset:
        df = sample_mixed_df.copy() if df is None else df
        feature_columns = feature_columns or [c for c in df.columns if c != target_column]
        categorical_columns = categorical_columns if categorical_columns is not None else []
        sensitive_columns = sensitive_columns if sensitive_columns is not None else []
        data_dir = ensure_dir(tmp_path / "data" / name)
        train_df, test_df = train_test_split(df, train_size=0.7, random_state=0)
        return Dataset(
            name=name,
            target_column=target_column,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
            sensitive_columns=sensitive_columns,
            data_dir=data_dir,
            full_df=df,
            train_df=train_df,
            test_df=test_df,
        )

    return _make_dataset


@pytest.fixture
def real_synth_pair():
    """Small paired real/synthetic DataFrames with two protected columns and a
    binary target, with deliberately shifted proportions in ``sex`` -- for
    log_disparity/evaluation-combine tests.
    """
    rng = np.random.default_rng(42)
    n = 40
    real = pd.DataFrame(
        {
            "sex": rng.choice(["M", "F"], size=n, p=[0.5, 0.5]),
            "age_group": rng.choice(["young", "old"], size=n, p=[0.6, 0.4]),
            "outcome": rng.integers(0, 2, size=n),
        }
    )
    synth = pd.DataFrame(
        {
            "sex": rng.choice(["M", "F"], size=n, p=[0.2, 0.8]),
            "age_group": rng.choice(["young", "old"], size=n, p=[0.6, 0.4]),
            "outcome": rng.integers(0, 2, size=n),
        }
    )
    return real, synth
