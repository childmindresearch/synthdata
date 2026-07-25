"""Unit tests for the pure-function parts of synthdata.evaluation.syntheval_eval:
the binary-target collapsing helpers used to let auroc_diff/statistical_parity/
equalized_odds/equal_opportunity run against a target with more than 2 classes.
"""

import numpy as np
import pandas as pd
import pytest

from synthdata.config import FrameworkSelectionConfig
from synthdata.evaluation.syntheval_eval import (
    BINARY_ONLY_METRICS,
    build_binary_preset,
    build_binary_target_series,
    merge_binary_target_results,
)

pytestmark = pytest.mark.unit


class TestBuildBinaryTargetSeries:
    def test_maps_positive_and_negative_classes(self):
        series = pd.Series([0, 1, 2, 0, 2], name="CGAS_class")
        out = build_binary_target_series(series, positive_classes=[0, 1], negative_classes=[2])
        assert out.tolist() == [1, 1, 0, 1, 0]
        assert out.name == "CGAS_class"

    def test_output_is_int_dtype(self):
        # Must be int (not float): SynthEval's AnalysisConfig only treats
        # object/int-dtype columns as categorical -- a float output would
        # silently get classified as continuous ("num"), defeating the
        # entire point of this function.
        series = pd.Series([0, 1, 2], name="t")
        out = build_binary_target_series(series, positive_classes=[0, 1], negative_classes=[2])
        assert out.dtype == np.int64

    def test_missing_value_raises(self):
        series = pd.Series([0, np.nan, 2], name="t")
        with pytest.raises(ValueError, match="missing value"):
            build_binary_target_series(series, positive_classes=[0], negative_classes=[2])

    def test_unmapped_value_raises(self):
        series = pd.Series([0, 1, 2], name="CGAS_class")
        with pytest.raises(ValueError, match="CGAS_class"):
            build_binary_target_series(series, positive_classes=[0], negative_classes=[2])

    def test_unmapped_value_message_lists_offending_value(self):
        series = pd.Series([0, 1, 2], name="t")
        with pytest.raises(ValueError, match=r"\[1\]"):
            build_binary_target_series(series, positive_classes=[0], negative_classes=[2])


class TestBuildBinaryPreset:
    def _selection(self, **overrides) -> FrameworkSelectionConfig:
        return FrameworkSelectionConfig(**overrides)

    def test_default_selection_includes_only_binary_only_metrics(self):
        preset = build_binary_preset(self._selection())
        assert set(preset) == set(BINARY_ONLY_METRICS)

    def test_disabled_selection_returns_empty(self):
        preset = build_binary_preset(self._selection(enabled=False))
        assert preset == {}

    def test_explicit_metrics_filters_to_binary_only_subset(self):
        preset = build_binary_preset(self._selection(metrics=["auroc_diff", "cls_acc"]))
        # cls_acc is a valid metric name but not a BINARY_ONLY_METRICS one --
        # it must never show up in a binary-target-only preset.
        assert set(preset) == {"auroc_diff"}

    def test_category_selection_excludes_fairness_excludes_fairness_metrics(self):
        preset = build_binary_preset(self._selection(categories=["utility"]))
        assert set(preset) == {"auroc_diff"}


class TestMergeBinaryTargetResults:
    @staticmethod
    def _comb_df(metric_values: dict, rank: float, index=("m1",)) -> pd.DataFrame:
        """Build a DataFrame matching SynthEval.benchmark()'s real comb_df structure:
        a (metric, 'value'/'error') MultiIndex for metrics, plus a scalar 'rank'
        column that pandas pads to ('rank', '') once the MultiIndex is set.
        """
        df = pd.DataFrame(index=list(index))
        for metric, (value, error) in metric_values.items():
            df[(metric, "value")] = value
            df[(metric, "error")] = error
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        df["rank"] = rank
        return df

    def test_both_none_returns_none(self):
        results, ranks = merge_binary_target_results(None, None, None, None)
        assert results is None
        assert ranks is None

    def test_binary_none_returns_main_unchanged(self):
        main_results = self._comb_df({"dwm": ([1.0], [0.1])}, rank=[0.9])
        main_ranks = pd.DataFrame({"dwm": [1.0], "rank": [0.9]}, index=["m1"])
        results, ranks = merge_binary_target_results(main_results, main_ranks, None, None)
        assert results is main_results
        assert ranks is main_ranks

    def test_main_none_returns_binary_unchanged(self):
        binary_results = self._comb_df({"auroc_diff": ([0.5], [0.05])}, rank=[0.3])
        binary_ranks = pd.DataFrame({"auroc_diff": [0.5], "rank": [0.3]}, index=["m1"])
        results, ranks = merge_binary_target_results(None, None, binary_results, binary_ranks)
        assert results is binary_results
        assert ranks is binary_ranks

    def test_merges_new_metric_columns_without_touching_existing(self):
        main_results = self._comb_df({"dwm": ([1.0], [0.1])}, rank=[0.9])
        main_ranks = pd.DataFrame({"dwm": [1.0], "rank": [0.9]}, index=["m1"])

        # rank=[0.3] here is the mini-benchmark's OWN aggregate rank (computed
        # from only auroc_diff) -- it must NOT overwrite the main pass's rank.
        binary_results = self._comb_df({"auroc_diff": ([0.5], [0.05])}, rank=[0.3])
        binary_ranks = pd.DataFrame({"auroc_diff": [0.7], "rank": [0.3]}, index=["m1"])

        results, ranks = merge_binary_target_results(
            main_results, main_ranks, binary_results, binary_ranks
        )

        assert results[("dwm", "value")].tolist() == [1.0]
        assert results[("auroc_diff", "value")].tolist() == [0.5]
        assert results[("auroc_diff", "error")].tolist() == [0.05]
        # The main pass's own aggregate rank must be untouched by the binary pass's.
        assert results["rank"].tolist() == [0.9]
        assert ranks["auroc_diff"].tolist() == [0.7]
        assert ranks["rank"].tolist() == [0.9]
