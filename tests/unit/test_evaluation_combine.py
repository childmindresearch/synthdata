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
            {"mean": [0.5, 0.3], "direction": ["maximize", "minimize"]},
            index=["stats.ks_test", "privacy.identifiability_score"],
        )
        raw, oriented = _synthcity_frames({"model_a": result}, model_names=["model_a"])

        assert raw.loc["model_a", ("synthcity", "utility", "stats.ks_test")] == 0.5
        assert raw.loc["model_a", ("synthcity", "privacy", "privacy.identifiability_score")] == 0.3
        # maximize -> unchanged sign; minimize -> flipped sign.
        assert oriented.loc["model_a", ("synthcity", "utility", "stats.ks_test")] == 0.5
        assert (
            oriented.loc["model_a", ("synthcity", "privacy", "privacy.identifiability_score")]
            == -0.3
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
