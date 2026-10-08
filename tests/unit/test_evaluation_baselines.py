"""Unit tests for synthdata.evaluation.baselines (the marginals baseline needs
synthcity and is covered by the integration tests)."""

import pandas as pd
import pytest

from synthdata.evaluation.baselines import build_baselines, is_baseline

pytestmark = pytest.mark.unit


def _train():
    return pd.DataFrame({"x": range(10), "y": [0, 1] * 5})


def test_train_copy_is_real_rows_without_replacement():
    train = _train()
    out = build_baselines(["train_copy"], train, n_samples=6, seed=0)
    frame = out["baseline_train_copy"]
    assert len(frame) == 6
    assert frame["x"].is_unique
    assert set(map(tuple, frame.to_numpy())) <= set(map(tuple, train.to_numpy()))


def test_train_copy_is_capped_at_the_train_size():
    out = build_baselines(["train_copy"], _train(), n_samples=50, seed=0)
    assert len(out["baseline_train_copy"]) == 10


def test_no_baselines_requested_returns_nothing():
    assert build_baselines([], _train(), n_samples=5, seed=0) == {}


def test_is_baseline():
    assert is_baseline("baseline_marginals")
    assert not is_baseline("ctgan")


def test_marginals_keep_each_column_distribution_and_repeat_under_a_seed():
    import numpy as np

    rng = np.random.default_rng(0)
    train = pd.DataFrame(
        {
            "x": rng.exponential(1.0, 4000),
            "c": rng.choice(["a", "b", "c"], 4000, p=[0.8, 0.15, 0.05]),
            "y": rng.integers(0, 2, 4000),
        }
    )
    train.loc[:399, "x"] = np.nan
    first = build_baselines(["marginals"], train, n_samples=4000, seed=7)["baseline_marginals"]
    again = build_baselines(["marginals"], train, n_samples=4000, seed=7)["baseline_marginals"]
    other = build_baselines(["marginals"], train, n_samples=4000, seed=8)["baseline_marginals"]

    pd.testing.assert_frame_equal(first, again)
    assert not first.equals(other)
    assert list(first.dtypes) == list(train.dtypes)
    assert set(first["x"].dropna()) <= set(train["x"].dropna())
    assert abs(first["x"].isna().mean() - 0.1) < 0.02
    assert abs(first["x"].median() - train["x"].median()) < 0.1
    assert abs((first["c"] == "a").mean() - 0.8) < 0.03


def test_marginals_break_the_link_between_columns():
    train = pd.DataFrame({"x": range(1000), "y": range(1000)})
    frame = build_baselines(["marginals"], train, n_samples=1000, seed=0)["baseline_marginals"]
    assert abs(frame["x"].corr(frame["y"])) < 0.1
