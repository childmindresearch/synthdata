"""Unit tests for synthdata.evaluation.custom_eval."""

import pandas as pd
import pytest

from synthdata.config import FrameworkSelectionConfig, LogDisparityConfig
from synthdata.evaluation.custom_eval import (
    build_log_disparity_summary_table,
    run_log_disparity_evaluation,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def fairness_dataset(make_dataset):
    df = pd.DataFrame(
        {
            "sex": ["M", "F", "M", "F", "M", "F", "M", "F"],
            "age": [20, 30, 40, 50, 25, 35, 45, 55],
            "target": [0, 1, 0, 1, 1, 0, 1, 0],
        }
    )
    return make_dataset(df=df, target_column="target", sensitive_columns=["sex"])


class TestRunLogDisparityEvaluation:
    def test_disabled_selection_returns_empty(self, fairness_dataset):
        reports = run_log_disparity_evaluation(
            {"model_a": fairness_dataset.train_df},
            fairness_dataset,
            LogDisparityConfig(protected_columns=["sex"]),
            FrameworkSelectionConfig(enabled=False),
        )
        assert reports == {}

    def test_no_protected_columns_warns_and_returns_empty(self, fairness_dataset):
        fairness_dataset.sensitive_columns = []
        reports = run_log_disparity_evaluation(
            {"model_a": fairness_dataset.train_df},
            fairness_dataset,
            LogDisparityConfig(protected_columns=[]),
            FrameworkSelectionConfig(enabled=True),
        )
        assert reports == {}

    def test_success_path_produces_summary_stats(self, fairness_dataset):
        good_synth = pd.DataFrame({"sex": ["M", "F", "M", "F"], "target": [0, 1, 1, 0]})
        reports = run_log_disparity_evaluation(
            {"good_model": good_synth},
            fairness_dataset,
            LogDisparityConfig(protected_columns=["sex"]),
            FrameworkSelectionConfig(enabled=True),
        )
        assert "summary_stats" in reports["good_model"]

    def test_failing_model_recorded_not_raised(self, fairness_dataset):
        # Missing the "sex" protected column entirely -> KeyError inside
        # compute_log_disparity_report, which must be caught and persisted,
        # not raised or silently dropped.
        bad_synth = pd.DataFrame({"target": [0, 1, 0, 1]})
        good_synth = pd.DataFrame({"sex": ["M", "F", "M", "F"], "target": [0, 1, 1, 0]})
        reports = run_log_disparity_evaluation(
            {"good_model": good_synth, "bad_model": bad_synth},
            fairness_dataset,
            LogDisparityConfig(protected_columns=["sex"]),
            FrameworkSelectionConfig(enabled=True),
        )
        assert "summary_stats" in reports["good_model"]
        assert reports["bad_model"]["error_type"] == "KeyError"
        assert "error" in reports["bad_model"]


class TestBuildLogDisparitySummaryTable:
    def test_success_report_extracts_summary_stats(self):
        reports = {
            "model_a": {
                "summary_stats": {
                    "mean_abs_log_disparity": 0.1,
                    "median_abs_log_disparity": 0.2,
                    "share_significant_bh": 0.3,
                }
            }
        }
        table = build_log_disparity_summary_table(reports)
        assert table.loc["model_a", "log_disparity_mean_abs"] == 0.1
        assert table.loc["model_a", "log_disparity_median_abs"] == 0.2
        assert table.loc["model_a", "log_disparity_share_significant"] == 0.3

    def test_failed_report_yields_all_nan_row_not_keyerror(self):
        reports = {"model_a": {"error": "boom", "error_type": "KeyError"}}
        table = build_log_disparity_summary_table(reports)
        assert table.loc["model_a"].isna().all()

    def test_mixed_success_and_failure(self):
        reports = {
            "good": {
                "summary_stats": {
                    "mean_abs_log_disparity": 0.5,
                    "median_abs_log_disparity": 0.4,
                    "share_significant_bh": 0.1,
                }
            },
            "bad": {"error": "boom", "error_type": "ValueError"},
        }
        table = build_log_disparity_summary_table(reports)
        assert table.loc["good"].notna().all()
        assert table.loc["bad"].isna().all()
