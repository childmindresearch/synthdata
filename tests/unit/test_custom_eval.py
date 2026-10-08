"""Unit tests for synthdata.evaluation.custom_eval."""

import pandas as pd
import pytest

from synthdata.config import FrameworkSelectionConfig, LogDisparityConfig
from synthdata.evaluation.custom_eval import (
    build_log_disparity_summary_table,
    build_tstr_table,
    run_log_disparity_evaluation,
    run_tstr_evaluation,
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
    return make_dataset(
        df=df, target_column="target", sensitive_columns=["age"], protected_columns=["sex"]
    )


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
        fairness_dataset.protected_columns = []
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

    def test_defaults_to_the_protected_columns_not_the_sensitive_ones(
        self, fairness_dataset, monkeypatch
    ):
        import synthdata.log_disparity.metric_log_disparity as ld

        seen = {}
        monkeypatch.setattr(
            ld,
            "compute_log_disparity_report",
            lambda **kwargs: seen.setdefault("protected_cols", kwargs["protected_cols"]),
        )
        run_log_disparity_evaluation(
            {"model_a": fairness_dataset.train_df},
            fairness_dataset,
            LogDisparityConfig(),
            FrameworkSelectionConfig(enabled=True),
        )
        assert seen["protected_cols"] == ["sex"]

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


class TestRunTstrEvaluation:
    @pytest.fixture
    def tstr_dataset(self, make_dataset):
        import numpy as np

        rng = np.random.default_rng(0)
        x = rng.normal(size=200)
        df = pd.DataFrame({"x": x, "target": np.digitize(x, [-0.5, 0.8])})
        dataset = make_dataset(df=df)
        dataset.train_imputed_df, dataset.test_imputed_df = dataset.train_df, dataset.test_df
        return dataset

    def test_scores_each_model_on_the_test_split_with_a_trtr_ceiling(self, tstr_dataset):
        real = tstr_dataset.train_imputed_df
        noise = real.assign(target=real["target"].sample(frac=1, random_state=0).to_numpy())
        result = run_tstr_evaluation(
            {"copy": real, "noise": noise}, tstr_dataset, FrameworkSelectionConfig(), 2, 0
        )
        assert result["classes"] == ["0", "1", "2"]
        assert result["scores"]["copy"] == result["trtr"]
        assert result["scores"]["noise"].macro_f1 < result["scores"]["copy"].macro_f1

        table = build_tstr_table(result)
        assert list(table.index) == ["copy", "noise", "trtr (real train)"]
        assert {"tstr_macro_f1", "tstr_balanced_accuracy", "tstr_macro_auprc"} <= set(table.columns)
        assert {"f1_0", "f1_1", "f1_2"} <= set(table.columns)

    def test_deselected_returns_empty(self, tstr_dataset):
        selection = FrameworkSelectionConfig(categories=["fairness"])
        assert run_tstr_evaluation({}, tstr_dataset, selection, 1, 0) == {}
        assert build_tstr_table({}).empty

    def test_log_disparity_selection_by_name_leaves_tstr_out(self, tstr_dataset):
        selection = FrameworkSelectionConfig(metrics=["log_disparity"])
        assert run_tstr_evaluation({}, tstr_dataset, selection, 1, 0) == {}
