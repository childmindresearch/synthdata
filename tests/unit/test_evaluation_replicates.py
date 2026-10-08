"""Unit tests for seed replicates: naming and the uncertainty-aware ranking summary."""

import numpy as np
import pandas as pd
import pytest

from synthdata.evaluation.combine import summarize_replicates
from synthdata.utils import replicate_name, split_replicate_name

pytestmark = pytest.mark.unit


def test_replicate_names_round_trip():
    assert replicate_name("ctgan", 0) == "ctgan"
    assert replicate_name("ctgan", 2) == "ctgan__rep2"
    assert split_replicate_name("ctgan__rep2") == ("ctgan", 2)
    assert split_replicate_name("ctgan") == ("ctgan", 0)
    assert split_replicate_name("baseline_marginals__rep1") == ("baseline_marginals", 1)
    assert split_replicate_name("odd__repx") == ("odd__repx", 0)


def _combined(overall: dict) -> pd.DataFrame:
    df = pd.DataFrame(index=list(overall))
    for dim in ("utility", "privacy", "fairness"):
        df[("__all__", dim, "rank")] = [v / 3 for v in overall.values()]
    df[("__all__", "overall", "rank")] = list(overall.values())
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_mean_and_t_interval_per_model():
    summary = summarize_replicates(
        _combined({"a": 2.0, "a__rep1": 2.2, "a__rep2": 2.4, "b": 1.0, "b__rep1": 1.0})
    )
    assert list(summary.index) == ["a", "b"]
    a = summary.loc["a"]
    assert a["n_replicates"] == 3
    assert a["overall_mean"] == pytest.approx(2.2)
    # t(0.975, 2) * sd / sqrt(3) with sd = 0.2
    half = 4.302652729911275 * 0.2 / np.sqrt(3)
    assert a["overall_ci_low"] == pytest.approx(2.2 - half)
    assert a["overall_ci_high"] == pytest.approx(2.2 + half)
    assert list(summary["rank"]) == [1, 2]


def test_single_seed_has_no_interval_and_no_tie_verdict():
    summary = summarize_replicates(_combined({"a": 2.0, "b": 1.0}))
    assert summary["overall_ci_low"].isna().all()
    assert summary.loc["a", "tied_with_best"] is True
    assert pd.isna(summary.loc["b", "tied_with_best"])


def test_overlapping_models_are_tied_and_separated_ones_are_not():
    summary = summarize_replicates(
        _combined(
            {
                "best": 2.0, "best__rep1": 2.1, "best__rep2": 1.9,
                "close": 1.95, "close__rep1": 2.05, "close__rep2": 1.85,
                "far": 0.5, "far__rep1": 0.6, "far__rep2": 0.4,
            }
        )
    )  # fmt: skip
    assert summary.loc["best", "tied_with_best"] is True
    assert summary.loc["close", "tied_with_best"] is True
    assert summary.loc["far", "tied_with_best"] is False


def test_baselines_cannot_be_best():
    summary = summarize_replicates(
        _combined(
            {
                "baseline_train_copy": 3.0, "baseline_train_copy__rep1": 3.0,
                "best": 2.5, "best__rep1": 2.6,
                "far": 1.0, "far__rep1": 1.1,
            },
        )
    )  # fmt: skip
    assert summary.loc["baseline_train_copy", "baseline"]
    assert not summary.loc["baseline_train_copy", "eligible"]
    assert pd.isna(summary.loc["baseline_train_copy", "tied_with_best"])
    assert summary.loc["best", "eligible"]
    assert summary.loc["best", "tied_with_best"] is True
    assert "privacy_gate_pass" not in summary.columns
