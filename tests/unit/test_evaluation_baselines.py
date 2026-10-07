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
    out = build_baselines(["train_copy"], train, "y", [], n_samples=6, seed=0)
    frame = out["baseline_train_copy"]
    assert len(frame) == 6
    assert frame["x"].is_unique
    assert set(map(tuple, frame.to_numpy())) <= set(map(tuple, train.to_numpy()))


def test_train_copy_is_capped_at_the_train_size():
    out = build_baselines(["train_copy"], _train(), "y", [], n_samples=50, seed=0)
    assert len(out["baseline_train_copy"]) == 10


def test_no_baselines_requested_returns_nothing():
    assert build_baselines([], _train(), "y", [], n_samples=5, seed=0) == {}


def test_is_baseline():
    assert is_baseline("baseline_marginals")
    assert not is_baseline("ctgan")
