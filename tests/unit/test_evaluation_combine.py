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
)

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
        # model_b has no "mean"/"direction" data -- reindexed to an all-NaN row.
        assert raw.loc["model_b"].isna().all()
        assert raw.loc["model_a", ("synthcity", "utility", "stats.ks_test")] == 0.5

    def test_all_models_failed_returns_empty_frame(self):
        failed_result = pd.DataFrame({"error": ["boom"], "error_type": ["ValueError"]})
        raw, oriented = _synthcity_frames({"model_a": failed_result}, model_names=["model_a"])
        assert raw.empty

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


class TestSyntheEvalFrames:
    def _benchmark_results(self):
        df = pd.DataFrame(index=["model_a", "model_b"])
        df[("ks_test", "value")] = [0.1, 0.2]
        df[("equal_opportunity", "value")] = [0.05, 0.9]
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        return df

    def _benchmark_ranks(self):
        return pd.DataFrame(
            {
                "ks_test": [0.9, 0.8],
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
        assert ("syntheval", "utility", "ks_test") in columns
        assert ("custom", "fairness", "equal_opportunity") in columns

    def test_raw_values_extracted_correctly(self):
        raw, _ = _syntheval_frames(
            self._benchmark_results(), self._benchmark_ranks(), model_names=["model_a", "model_b"]
        )
        assert raw.loc["model_a", ("syntheval", "utility", "ks_test")] == pytest.approx(0.1)


class TestLogDisparityFrames:
    def test_empty_reports_returns_empty(self):
        raw, oriented = _log_disparity_frames({}, model_names=["model_a"])
        assert raw.empty

    def test_minimize_metrics_get_flipped_sign(self):
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
        oriented_val = oriented.loc["model_a", ("custom", "fairness", "log_disparity_mean_abs")]
        assert raw_val == pytest.approx(0.4)
        assert oriented_val == pytest.approx(-0.4)  # all log_disparity metrics minimize

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
        assert ("__all__", "fairness", "rank") in combined.columns
        assert ("__all__", "utility", "rank") in combined.columns

    def test_metric_count_imbalance_does_not_dominate_type_rollup(self):
        # synthcity contributes 5 utility metrics, syntheval contributes 1 --
        # under the old flat-sum scheme, synthcity's group would dominate the
        # utility rollup purely by column count. Under mean-of-means, both
        # groups' RANK contributes equally to the utility rollup regardless
        # of how many raw metrics compose each group.
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
        benchmark_results[("cls_acc", "value")] = [0.0, 1.0]
        benchmark_results.columns = pd.MultiIndex.from_tuples(benchmark_results.columns)
        benchmark_ranks = pd.DataFrame(
            {"cls_acc": [0.0, 1.0], "rank": [0.0, 1.0]}, index=["model_a", "model_b"]
        )
        combined = build_combined_table(
            synthcity_results,
            benchmark_results,
            benchmark_ranks,
            {},
            model_names=["model_a", "model_b"],
        )
        # synthcity favors model_a (scaled 1.0 vs 0.0), syntheval favors
        # model_b (scaled 0.0 vs 1.0) -- with equal group weighting these
        # exactly cancel out in the utility rollup regardless of synthcity
        # having 5x the raw metric columns.
        assert combined.loc["model_a", ("__all__", "utility", "rank")] == pytest.approx(0.5)
        assert combined.loc["model_b", ("__all__", "utility", "rank")] == pytest.approx(0.5)

    def test_rank_weights_zero_excludes_type_from_overall(self):
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
        assert combined[("__all__", "overall", "rank")].tolist() == [0.0, 0.0]

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
