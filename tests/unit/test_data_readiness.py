"""Unit tests for the data-readiness check: one known-answer frame per check."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synthdata.config import DataReadinessConfig
from synthdata.data_readiness import CHECKS, assess, save_readiness_report

pytestmark = pytest.mark.unit

N = 400


def _frame() -> pd.DataFrame:
    """A clean base: every column should pass every check."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "num": rng.normal(size=N),
            "num2": rng.normal(size=N),
            "cat": rng.integers(0, 3, N),
            "target": rng.integers(0, 2, N),
        }
    )


def _findings(make_dataset, df, **kwargs):
    nominal = kwargs.pop("nominal", ["cat"])
    cfg = DataReadinessConfig(min_rows_per_column=kwargs.pop("min_rows_per_column", None), **kwargs)
    dataset = make_dataset(df=df, nominal_columns=nominal)
    return assess(dataset, cfg)


def _flagged(findings, check):
    return set(findings.loc[findings["check"] == check, "column"])


def test_clean_frame_has_no_findings(make_dataset):
    assert _findings(make_dataset, _frame()).empty


def test_constant_and_near_zero_variance_follow_caret(make_dataset):
    df = _frame()
    df["const"] = 1
    df["nzv"] = 0
    df.loc[df.index[:5], "nzv"] = 1  # ratio ~79 > 19, 2 distinct of 400 = 0.5%
    df["balanced"] = np.arange(N) % 2  # ratio 1: not flagged
    findings = _findings(make_dataset, df, nominal=["cat", "nzv", "balanced"])
    assert _flagged(findings, "constant") == {"const"}
    assert "nzv" in _flagged(findings, "near_zero_variance")
    assert "balanced" not in _flagged(findings, "near_zero_variance")
    assert "const" not in _flagged(findings, "near_zero_variance")


def test_duplicates_are_found_up_to_relabelling(make_dataset):
    df = _frame()
    df["cat_copy"] = df["cat"].map({0: "a", 1: "b", 2: "c"})  # same partition, new labels
    df["num_neg"] = -df["num"]  # perfectly correlated, not a relabelling of a category
    findings = _findings(make_dataset, df, nominal=["cat", "cat_copy"])
    assert _flagged(findings, "duplicate") == {"cat_copy"}
    assert _flagged(findings, "high_correlation") == {"num_neg"}


def test_missingness_rare_and_many_levels(make_dataset):
    df = _frame()
    df.loc[df.index[: int(0.6 * N)], "num2"] = np.nan
    df["rare"] = np.where(np.arange(N) < 3, "x", "y")
    df["many"] = np.arange(N) % 80
    findings = _findings(make_dataset, df, nominal=["cat", "rare", "many"])
    assert "num2" in _flagged(findings, "heavy_missingness")
    assert "rare" in _flagged(findings, "rare_categories")
    assert "many" in _flagged(findings, "high_cardinality")


def test_skew_and_zero_inflation_are_info(make_dataset):
    df = _frame()
    rng = np.random.default_rng(1)
    df["skewed"] = rng.lognormal(0, 1.5, N)
    df["zeros"] = np.where(np.arange(N) < 0.7 * N, 0.0, rng.normal(size=N))
    findings = _findings(make_dataset, df)
    assert "skewed" in _flagged(findings, "skewed")
    assert "zeros" in _flagged(findings, "zero_inflated")
    assert set(findings.loc[findings["check"].isin(["skewed", "zero_inflated"]), "status"]) == {
        "info"
    }


def test_split_checks_and_small_groups(make_dataset):
    df = _frame()
    dataset = make_dataset(df=df, nominal_columns=["cat"], protected_columns=["cat"])
    test_rows = dataset.test_df.index
    dataset.test_df.loc[test_rows[:2], "cat"] = 9  # a level train never has
    dataset.test_df.loc[test_rows[2:], "cat"] = 0  # the others rarely in test
    findings = assess(dataset, DataReadinessConfig(min_rows_per_column=None))
    assert "cat" in _flagged(findings, "unseen_category")
    small = findings[findings["check"] == "small_group"]
    assert "cat" in set(small["column"]) and "in test" in small["action"].iloc[0]


def test_rows_per_column_and_identifier(make_dataset):
    df = _frame()
    df["patient_name"] = [f"p{i}" for i in range(N)]
    dataset = make_dataset(df=df, nominal_columns=["cat", "patient_name"])
    findings = assess(dataset, DataReadinessConfig(min_rows_per_column=1000))
    assert "patient_name" in _flagged(findings, "identifier_like")
    assert _flagged(findings, "rows_per_column") == {"(dataset)"}


def test_null_threshold_turns_a_check_off(make_dataset):
    df = _frame()
    df["skewed"] = np.random.default_rng(1).lognormal(0, 1.5, N)
    assert "skewed" not in _flagged(_findings(make_dataset, df, max_abs_skew=None), "skewed")


def test_every_check_has_a_meaning_and_an_action():
    assert all(len(text) == 2 and all(text) for text in CHECKS.values())


def test_report_is_written_and_can_be_turned_off(make_config, make_dataset):
    cfg = make_config()
    dataset = make_dataset(df=_frame(), nominal_columns=["cat"])
    findings = save_readiness_report(cfg, dataset)
    assert findings is not None
    written = list(Path(cfg.generation.output_dir).parent.rglob("data_readiness.csv"))
    assert len(written) == 1
    cfg.data.readiness.enabled = False
    assert save_readiness_report(cfg, dataset) is None


def test_missingness_indicators_are_left_out(make_dataset):
    df = _frame()
    df["ind_a__missing"] = (np.arange(N) % 7 == 0).astype(int)
    df["ind_b__missing"] = df["ind_a__missing"]  # missing together: identical indicators
    dataset = make_dataset(df=df, nominal_columns=["cat", "ind_a__missing", "ind_b__missing"])
    dataset.missing_indicator_columns = {"ind_a__missing": "a", "ind_b__missing": "b"}
    findings = assess(dataset, DataReadinessConfig(min_rows_per_column=None))
    assert not set(findings["column"]) & {"ind_a__missing", "ind_b__missing"}
