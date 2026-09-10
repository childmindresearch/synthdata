"""Unit tests for synthdata.evaluation.combine: per-framework frame builders,
min-max scaling, and the combined ranked table.
"""

import numpy as np
import pandas as pd
import pytest

from synthdata.evaluation.combine import (
    _log_disparity_frames,
    _minmax_scale,
    _synthcity_frames,
    _syntheval_frames,
    build_combined_table,
    load_combined_table,
    validate_combined_table,
)
from synthdata.evaluation.metric_contracts import (
    MetricEvaluationContext,
    MetricStatusRecord,
    MetricValidationResult,
)
from synthdata.evaluation.synthcity_eval import validate_synthcity_report
from synthdata.evaluation.syntheval_eval import validate_syntheval_results

pytestmark = pytest.mark.unit


class TestMinMaxScale:
    def test_normal_scaling(self):
        scaled = _minmax_scale(pd.Series([0.0, 5.0, 10.0]))
        assert scaled.tolist() == pytest.approx([0.0, 0.5, 1.0])

    def test_ties_become_half(self):
        scaled = _minmax_scale(pd.Series([3.0, 3.0, 3.0]))
        assert scaled.tolist() == [0.5, 0.5, 0.5]

    def test_all_nan_returned_unchanged(self):
        col = pd.Series([np.nan, np.nan])
        scaled = _minmax_scale(col)
        assert scaled.isna().all()

    def test_nan_preserved_alongside_scaled_values(self):
        scaled = _minmax_scale(pd.Series([0.0, np.nan, 10.0]))
        assert scaled.iloc[0] == 0.0
        assert pd.isna(scaled.iloc[1])
        assert scaled.iloc[2] == 1.0


class TestSynthcityFrames:
    def test_empty_results_returns_empty_frames_indexed_by_models(self):
        raw, oriented = _synthcity_frames({}, model_names=["a", "b"])
        assert raw.empty
        assert list(raw.index) == ["a", "b"]

    def test_builds_multiindex_columns_oriented_by_direction(self):
        result = pd.DataFrame(
            {"mean": [0.5, 0.3, 0.2], "direction": ["maximize", "minimize", "minimize"]},
            index=[
                "stats.ks_test",
                "privacy.identifiability_score",
                "attack.data_leakage_linear",
            ],
        )
        raw, oriented = _synthcity_frames({"model_a": result}, model_names=["model_a"])

        assert raw.loc["model_a", ("synthcity", "utility", "stats.ks_test")] == 0.5
        assert raw.loc["model_a", ("synthcity", "privacy", "privacy.identifiability_score")] == 0.3
        assert raw.loc["model_a", ("synthcity", "privacy", "attack.data_leakage_linear")] == 0.2
        # maximize -> unchanged sign; minimize -> flipped sign.
        assert oriented.loc["model_a", ("synthcity", "utility", "stats.ks_test")] == 0.5
        assert (
            oriented.loc["model_a", ("synthcity", "privacy", "privacy.identifiability_score")]
            == -0.3
        )
        assert (
            oriented.loc["model_a", ("synthcity", "privacy", "attack.data_leakage_linear")] == -0.2
        )

    def test_failed_model_excluded_not_raising(self):
        ok_result = pd.DataFrame(
            {"mean": [0.5], "direction": ["maximize"]}, index=["stats.ks_test"]
        )
        failed_result = pd.DataFrame({"error": ["boom"], "error_type": ["ValueError"]})
        raw, oriented = _synthcity_frames(
            {"model_a": ok_result, "model_b": failed_result},
            model_names=["model_a", "model_b"],
        )
        assert raw.loc["model_b", ("synthcity", "audit", "__model_error")] == "boom"
        assert raw.loc["model_b", ("synthcity", "audit", "__model_error_type")] == "ValueError"
        assert raw.loc["model_a", ("synthcity", "utility", "stats.ks_test")] == 0.5

    def test_missing_expected_identity_is_materialized_with_model_state(self):
        result = pd.DataFrame(
            {"mean": [0.25], "direction": ["maximize"]},
            index=["stats.ks_test.marginal"],
        )
        validation = validate_synthcity_report(
            "model_a",
            result,
            expected_base_keys=["stats.ks_test", "stats.wasserstein_dist"],
            context=MetricEvaluationContext(
                role_hashes={"train": "train-hash", "test": "test-hash"}
            ),
        )

        raw, oriented = _synthcity_frames(
            {"model_a": result},
            model_names=["model_a"],
            synthcity_validations={"model_a": validation},
        )

        missing_key = (
            "synthcity",
            "utility",
            "stats.wasserstein_dist.joint",
        )
        assert pd.isna(raw.loc["model_a", missing_key])
        assert not bool(raw.loc["model_a", ("synthcity", "audit", "__model_synthcity_succeeded")])
        assert (
            raw.loc["model_a", ("synthcity", "audit", "__model_synthcity_decision_status")]
            == "indeterminate"
        )
        assert oriented.empty

    def test_failed_model_keeps_expected_raw_columns_and_failure_state(self):
        failed_result = pd.DataFrame(
            {"error": ["framework crashed"], "error_type": ["RuntimeError"]}
        )
        validation = validate_synthcity_report(
            "model_a",
            failed_result,
            expected_base_keys=["stats.ks_test"],
            context=MetricEvaluationContext(
                role_hashes={"train": "train-hash", "test": "test-hash"}
            ),
        )

        raw, oriented = _synthcity_frames(
            {"model_a": failed_result},
            model_names=["model_a"],
            synthcity_validations={"model_a": validation},
        )

        assert pd.isna(raw.loc["model_a", ("synthcity", "utility", "stats.ks_test.marginal")])
        assert raw.loc["model_a", ("synthcity", "audit", "__model_error")] == "framework crashed"
        assert (
            raw.loc["model_a", ("synthcity", "audit", "__model_synthcity_audit_status")]
            == "indeterminate"
        )
        assert raw.loc["model_a", ("synthcity", "audit", "__model_synthcity_expected_count")] == 1
        assert oriented.empty

    def test_all_models_failed_returns_empty_frame(self):
        failed_result = pd.DataFrame({"error": ["boom"], "error_type": ["ValueError"]})
        raw, oriented = _synthcity_frames({"model_a": failed_result}, model_names=["model_a"])
        assert raw.loc["model_a", ("synthcity", "audit", "__model_error")] == "boom"
        assert oriented.empty

    def test_redundant_naive_alpha_precision_submetrics_excluded(self):
        result = pd.DataFrame(
            {
                "mean": [0.9, 0.9, 0.5, 0.5],
                "direction": ["maximize", "maximize", "maximize", "maximize"],
            },
            index=[
                "stats.alpha_precision.authenticity_OC",
                "stats.alpha_precision.delta_precision_alpha_OC",
                "stats.alpha_precision.authenticity_naive",
                "stats.alpha_precision.delta_precision_alpha_naive",
            ],
        )
        raw, oriented = _synthcity_frames({"model_a": result}, model_names=["model_a"])
        raw_metrics = raw.columns.get_level_values(2)
        assert "stats.alpha_precision.authenticity_OC" in raw_metrics
        assert "stats.alpha_precision.authenticity_naive" not in raw_metrics
        assert "stats.alpha_precision.delta_precision_alpha_naive" not in raw_metrics
        assert oriented.columns.get_level_values(2).tolist() == raw_metrics.tolist()

    def test_contract_validation_keeps_audit_raw_values_out_of_policy_rank(self):
        result = pd.DataFrame(
            {"mean": [0.4], "direction": ["minimize"]},
            index=["privacy.identifiability_score.score_OC"],
        )
        validation = validate_synthcity_report(
            "model_a",
            result,
            context=MetricEvaluationContext(
                role_hashes={"train": "train-hash", "test": "test-hash"}
            ),
            requested_use="policy_rank",
        )

        raw, oriented = _synthcity_frames(
            {"model_a": result},
            model_names=["model_a"],
            synthcity_validations={"model_a": validation},
        )

        assert (
            raw.loc["model_a", ("synthcity", "privacy", "privacy.identifiability_score.score_OC")]
            == 0.4
        )
        assert oriented.empty

    def test_build_combined_table_all_audit_only_results(self):
        result = pd.DataFrame(
            {"mean": [0.4], "direction": ["minimize"]},
            index=["privacy.identifiability_score.score_OC"],
        )
        validation = validate_synthcity_report(
            "model_a",
            result,
            context=MetricEvaluationContext(
                role_hashes={"train": "train-hash", "test": "test-hash"}
            ),
            requested_use="policy_rank",
        )

        combined = build_combined_table(
            {"model_a": result},
            None,
            None,
            {},
            model_names=["model_a"],
            synthcity_validations={"model_a": validation},
        )

        assert (
            combined.loc[
                "model_a", ("synthcity", "privacy", "privacy.identifiability_score.score_OC")
            ]
            == 0.4
        )
        assert pd.isna(combined.loc["model_a", ("__all__", "overall", "rank")])


class TestSyntheEvalFrames:
    @staticmethod
    def _validation(model_name, status, raw_value):
        record = MetricStatusRecord(
            model_name=model_name,
            expected_key="auroc",
            framework="syntheval",
            status=status,
            contract_id="test.auroc",
            raw_value=raw_value,
            policy_value=raw_value,
            uncertainty=0.01 if status == "succeeded" else None,
            sample_size=10 if status == "succeeded" else None,
            allowed_uses=frozenset({"audit", "policy_rank"}),
            value_role="policy_scalar",
            lifecycle_state="operational",
            direction="minimize",
        )
        return MetricValidationResult(
            model_name=model_name,
            requested_use="policy_rank",
            contract_digest="test-digest",
            records=(record,),
        )

    def _benchmark_results(self):
        df = pd.DataFrame(index=["model_a", "model_b"])
        df[("ks_tvd_stat", "value")] = [0.1, 0.2]
        df[("equal_opportunity", "value")] = [0.05, 0.9]
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        return df

    def _benchmark_ranks(self):
        return pd.DataFrame(
            {
                "ks_tvd_stat": [0.9, 0.8],
                "equal_opportunity": [0.6, 0.1],
                "rank": [1, 2],
            },
            index=["model_a", "model_b"],
        )

    def test_none_results_returns_empty(self):
        raw, oriented = _syntheval_frames(None, None, model_names=["model_a"])
        assert raw.empty

    def test_tags_custom_fairness_metrics_separately(self):
        raw, oriented = _syntheval_frames(
            self._benchmark_results(), self._benchmark_ranks(), model_names=["model_a", "model_b"]
        )
        columns = list(raw.columns)
        assert ("syntheval", "utility", "ks_tvd_stat") in columns
        assert ("custom", "fairness", "equal_opportunity") in columns

    def test_raw_values_extracted_correctly(self):
        raw, _ = _syntheval_frames(
            self._benchmark_results(), self._benchmark_ranks(), model_names=["model_a", "model_b"]
        )
        assert raw.loc["model_a", ("syntheval", "utility", "ks_tvd_stat")] == pytest.approx(0.1)

    def test_missing_expected_identity_is_materialized_from_validation(self):
        benchmark_results = pd.DataFrame(index=["model_a"])
        benchmark_results[("avg_dwm_diff", "value")] = [0.1]
        benchmark_results.columns = pd.MultiIndex.from_tuples(benchmark_results.columns)
        benchmark_ranks = pd.DataFrame({"avg_dwm_diff": [0.9]}, index=["model_a"])
        validations = validate_syntheval_results(
            benchmark_results,
            benchmark_ranks,
            {"syntheval": ["avg_dwm_diff", "pca_eigval_diff"], "custom": []},
            role_hashes={"train": "train-hash", "test": "test-hash"},
            model_names=["model_a"],
            requested_use="audit",
        )

        raw, _oriented = _syntheval_frames(
            benchmark_results,
            benchmark_ranks,
            model_names=["model_a"],
            syntheval_validations=validations,
        )

        assert pd.isna(raw.loc["model_a", ("syntheval", "utility", "pca_eigval_diff")])
        assert not bool(
            raw.loc["model_a", ("syntheval", "audit", "__model_syntheval_main_succeeded")]
        )

    def test_pass_ownership_is_resolved_per_model(self):
        benchmark_results = pd.DataFrame(index=["model_a", "model_b"])
        benchmark_results[("auroc", "value")] = [0.1, np.nan]
        benchmark_results.columns = pd.MultiIndex.from_tuples(benchmark_results.columns)
        benchmark_ranks = pd.DataFrame(
            {"auroc": [0.9, 0.8]},
            index=["model_a", "model_b"],
        )
        validations = {
            ("syntheval", "main"): {
                "model_a": self._validation("model_a", "succeeded", 0.1),
                "model_b": self._validation("model_b", "failed", None),
            },
            ("syntheval", "binary_target"): {
                "model_a": self._validation("model_a", "succeeded", 0.2),
                "model_b": self._validation("model_b", "succeeded", 0.4),
            },
        }
        execution_passes = {
            ("syntheval", "auroc", "model_a"): "main",
            ("syntheval", "auroc", "model_b"): "binary_target",
        }

        raw, oriented = _syntheval_frames(
            benchmark_results,
            benchmark_ranks,
            model_names=["model_a", "model_b"],
            syntheval_validations=validations,
            metric_execution_passes=execution_passes,
        )

        assert raw.loc["model_a", ("syntheval", "utility", "auroc")] == 0.1
        assert raw.loc["model_b", ("syntheval", "utility", "auroc")] == 0.4
        assert oriented.loc["model_a", ("syntheval", "utility", "auroc")] == 0.9
        assert oriented.loc["model_b", ("syntheval", "utility", "auroc")] == 0.8


class TestLogDisparityFrames:
    def test_empty_reports_returns_empty(self):
        raw, oriented = _log_disparity_frames({}, model_names=["model_a"])
        assert raw.empty

    def test_unvalidated_custom_metrics_remain_raw_but_unranked(self):
        reports = {
            "model_a": {
                "summary_stats": {
                    "mean_abs_log_disparity": 0.4,
                    "median_abs_log_disparity": 0.3,
                    "share_significant_bh": 0.1,
                }
            }
        }
        raw, oriented = _log_disparity_frames(reports, model_names=["model_a"])
        raw_val = raw.loc["model_a", ("custom", "fairness", "log_disparity_mean_abs")]
        assert raw_val == pytest.approx(0.4)
        assert oriented.empty

    def test_median_abs_present_in_raw_but_excluded_from_oriented(self):
        # log_disparity_median_abs is redundant with mean_abs (same underlying
        # per-subgroup array) -- still shown in the raw table (informational)
        # but excluded from the oriented/ranked table to avoid double-counting.
        reports = {
            "model_a": {
                "summary_stats": {
                    "mean_abs_log_disparity": 0.4,
                    "median_abs_log_disparity": 0.3,
                    "share_significant_bh": 0.1,
                }
            }
        }
        raw, oriented = _log_disparity_frames(reports, model_names=["model_a"])
        assert ("custom", "fairness", "log_disparity_median_abs") in raw.columns
        assert ("custom", "fairness", "log_disparity_median_abs") not in oriented.columns

    def test_failed_model_yields_nan_row(self):
        reports = {"model_a": {"error": "boom", "error_type": "KeyError"}}
        raw, _ = _log_disparity_frames(reports, model_names=["model_a"])
        assert raw.loc["model_a"].isna().all()


class TestBuildCombinedTable:
    def test_raises_when_nothing_to_combine(self):
        with pytest.raises(ValueError, match="No evaluation results"):
            build_combined_table({}, None, None, {}, model_names=["model_a"])

    def test_combines_single_source_and_ranks(self):
        synthcity_results = {
            "model_a": pd.DataFrame(
                {"mean": [0.9], "direction": ["maximize"]}, index=["stats.ks_test"]
            ),
            "model_b": pd.DataFrame(
                {"mean": [0.1], "direction": ["maximize"]}, index=["stats.ks_test"]
            ),
        }
        combined = build_combined_table(
            synthcity_results, None, None, {}, model_names=["model_a", "model_b"]
        )
        assert ("__all__", "overall", "rank") in combined.columns
        assert ("synthcity", "utility", "rank") in combined.columns
        # model_a has the higher raw metric -> higher overall rank -> sorted first.
        assert combined.index[0] == "model_a"

    def test_load_combined_table_validates_round_trip(self, tmp_path):
        combined = build_combined_table(
            {
                "model_a": pd.DataFrame(
                    {"mean": [0.9], "direction": ["maximize"]}, index=["stats.ks_test"]
                )
            },
            None,
            None,
            {},
            model_names=["model_a"],
        )
        path = tmp_path / "combined_evaluation.csv"
        combined.to_csv(path)

        loaded = load_combined_table(str(path))

        pd.testing.assert_frame_equal(loaded, combined)

    def test_loads_empty_blocked_legacy_table(self, tmp_path):
        columns = pd.MultiIndex.from_arrays([[], [], []], names=["framework", "type", "metric"])
        combined = pd.DataFrame(index=pd.Index([], name="model"), columns=columns)
        path = tmp_path / "combined_evaluation.csv"
        combined.to_csv(path)

        loaded = load_combined_table(str(path))

        assert loaded.empty
        assert loaded.shape == (0, 0)

    def test_combined_table_requires_overall_rank(self):
        combined = pd.DataFrame(index=["model_a"])
        combined[("__all__", "utility", "rank")] = [0.5]
        combined.columns = pd.MultiIndex.from_tuples(combined.columns)

        with pytest.raises(ValueError, match="overall rank"):
            validate_combined_table(combined)

    def test_combined_table_rejects_non_finite_metric_values(self):
        combined = pd.DataFrame(index=["model_a"])
        combined[("syntheval", "utility", "metric_a")] = [np.inf]
        combined[("__all__", "overall", "rank")] = [0.5]
        combined.columns = pd.MultiIndex.from_tuples(combined.columns)

        with pytest.raises(ValueError, match="non-finite"):
            validate_combined_table(combined)

    def test_combined_table_rejects_duplicate_metric_identities(self):
        combined = pd.DataFrame([[0.5, 0.5]], index=["model_a"])
        combined.columns = pd.MultiIndex.from_tuples(
            [
                ("syntheval", "utility", "metric_a"),
                ("syntheval", "utility", "metric_a"),
            ]
        )

        with pytest.raises(ValueError, match="duplicate metric"):
            validate_combined_table(combined)

    def test_combines_multiple_sources(self):
        synthcity_results = {
            "model_a": pd.DataFrame(
                {"mean": [0.9], "direction": ["maximize"]}, index=["stats.ks_test"]
            )
        }
        log_disparity_reports = {
            "model_a": {
                "summary_stats": {
                    "mean_abs_log_disparity": 0.2,
                    "median_abs_log_disparity": 0.2,
                    "share_significant_bh": 0.0,
                }
            }
        }
        combined = build_combined_table(
            synthcity_results, None, None, log_disparity_reports, model_names=["model_a"]
        )
        assert ("custom", "fairness", "log_disparity_mean_abs") in combined.columns
        assert ("__all__", "fairness", "rank") not in combined.columns
        assert ("__all__", "utility", "rank") in combined.columns

    def test_metric_count_imbalance_does_not_dominate_type_rollup(self):
        # Unvalidated SynthEval rank values remain audit evidence and cannot
        # create a policy group, even when their raw columns are present.
        synthcity_results = {
            "model_a": pd.DataFrame(
                {"mean": [1.0] * 5, "direction": ["maximize"] * 5},
                index=[f"stats.metric_{i}" for i in range(5)],
            ),
            "model_b": pd.DataFrame(
                {"mean": [0.0] * 5, "direction": ["maximize"] * 5},
                index=[f"stats.metric_{i}" for i in range(5)],
            ),
        }
        benchmark_results = pd.DataFrame(index=["model_a", "model_b"])
        benchmark_results[("avg_F1_diff", "value")] = [0.0, 1.0]
        benchmark_results.columns = pd.MultiIndex.from_tuples(benchmark_results.columns)
        benchmark_ranks = pd.DataFrame(
            {"avg_F1_diff": [0.0, 1.0], "rank": [0.0, 1.0]}, index=["model_a", "model_b"]
        )
        combined = build_combined_table(
            synthcity_results,
            benchmark_results,
            benchmark_ranks,
            {},
            model_names=["model_a", "model_b"],
        )
        assert ("syntheval", "utility", "avg_F1_diff") in combined.columns
        assert combined.loc["model_a", ("__all__", "utility", "rank")] == pytest.approx(1.0)
        assert combined.loc["model_b", ("__all__", "utility", "rank")] == pytest.approx(0.0)

    def test_incomplete_fixed_utility_is_indeterminate_regardless_of_rank_weights(self):
        synthcity_results = {
            "model_a": pd.DataFrame(
                {"mean": [1.0], "direction": ["maximize"]}, index=["privacy.identifiability_score"]
            ),
            "model_b": pd.DataFrame(
                {"mean": [0.0], "direction": ["maximize"]}, index=["privacy.identifiability_score"]
            ),
        }
        combined = build_combined_table(
            synthcity_results,
            None,
            None,
            {},
            model_names=["model_a", "model_b"],
            rank_weights={"utility": 1.0, "privacy": 0.0, "fairness": 1.0},
        )
        assert combined[("__all__", "overall", "rank")].isna().all()

    def test_rank_weights_asymmetric_changes_sort_order(self):
        synthcity_results = {
            "model_a": pd.DataFrame(
                {"mean": [1.0, 0.0], "direction": ["maximize", "maximize"]},
                index=["stats.utility_metric", "privacy.identifiability_score"],
            ),
            "model_b": pd.DataFrame(
                {"mean": [0.0, 1.0], "direction": ["maximize", "maximize"]},
                index=["stats.utility_metric", "privacy.identifiability_score"],
            ),
        }
        combined = build_combined_table(
            synthcity_results,
            None,
            None,
            {},
            model_names=["model_a", "model_b"],
            rank_weights={"utility": 5.0, "privacy": 0.1, "fairness": 1.0},
        )
        # model_a wins on utility (weighted heavily); model_b wins on privacy
        # (weighted lightly) -- utility-heavy weighting should make model_a
        # rank first overall.
        assert combined.index[0] == "model_a"

    def test_default_rank_weights_used_when_none_passed(self):
        synthcity_results = {
            "model_a": pd.DataFrame(
                {"mean": [0.9], "direction": ["maximize"]}, index=["stats.ks_test"]
            ),
            "model_b": pd.DataFrame(
                {"mean": [0.1], "direction": ["maximize"]}, index=["stats.ks_test"]
            ),
        }
        combined = build_combined_table(
            synthcity_results, None, None, {}, model_names=["model_a", "model_b"]
        )
        assert combined.index[0] == "model_a"
