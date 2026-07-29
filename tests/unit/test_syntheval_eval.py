"""Unit tests for the pure-function parts of synthdata.evaluation.syntheval_eval:
the binary-target collapsing helpers used to let auroc_diff/statistical_parity/
equalized_odds/equal_opportunity run against a target with more than 2 classes,
and the syntheval benchmark result caching helpers.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synthdata.config import FrameworkSelectionConfig, SynthEvalExecutionConfig
from synthdata.evaluation.catalog import FAIRNESS_METRICS_WITH_POSITIVE_CLASS, SYNTHEVAL_PRESET
from synthdata.evaluation.syntheval_eval import (
    BINARY_ONLY_METRICS,
    _atomic_parquet,
    _checkpoint_paths,
    _compute_cache_key,
    _load_syntheval_cache,
    _save_syntheval_cache,
    build_binary_preset,
    build_binary_target_series,
    build_preset,
    merge_binary_target_results,
    resolve_model_workers,
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


class TestBuildPreset:
    def _selection(self, **overrides) -> FrameworkSelectionConfig:
        return FrameworkSelectionConfig(**overrides)

    def test_default_positive_class_matches_preset_default(self):
        preset = build_preset(self._selection())
        for name in FAIRNESS_METRICS_WITH_POSITIVE_CLASS:
            assert preset[name]["positive_class"] == 1

    def test_positive_class_override_applies_to_fairness_metrics_only(self):
        preset = build_preset(self._selection(), positive_class=0)
        for name in FAIRNESS_METRICS_WITH_POSITIVE_CLASS:
            assert preset[name]["positive_class"] == 0
        # A non-fairness metric's params must be untouched.
        assert preset["dwm"] == {}

    def test_override_does_not_mutate_shared_preset_constant(self):
        # SYNTHEVAL_PRESET's nested dicts are shared module-level objects --
        # build_preset must shallow-copy before overriding, or this would
        # corrupt the global constant for every subsequent call in-process.
        build_preset(self._selection(), positive_class=0)
        for name in FAIRNESS_METRICS_WITH_POSITIVE_CLASS:
            assert SYNTHEVAL_PRESET[name]["positive_class"] == 1

    def test_disabled_selection_returns_empty(self):
        preset = build_preset(self._selection(enabled=False))
        assert preset == {}


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


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _make_results(index=("m1", "m2")) -> pd.DataFrame:
    """Minimal benchmark_results DataFrame with a MultiIndex column level."""
    df = pd.DataFrame(index=list(index))
    df[("dwm", "value")] = [0.8, 0.7]
    df[("dwm", "error")] = [0.01, 0.02]
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df["rank"] = [0.9, 0.8]
    return df


def _make_ranks(index=("m1", "m2")) -> pd.DataFrame:
    return pd.DataFrame({"dwm": [0.8, 0.7], "rank": [0.9, 0.8]}, index=list(index))


class TestComputeCacheKey:
    def test_same_inputs_produce_same_key(self):
        preset = {"dwm": {}, "cls_acc": {}}
        k1 = _compute_cache_key(preset, ["m1", "m2"], "linear")
        k2 = _compute_cache_key(preset, ["m1", "m2"], "linear")
        assert k1 == k2

    def test_different_model_order_same_key(self):
        # Model names are sorted before hashing -- order must not matter.
        preset = {"dwm": {}}
        k1 = _compute_cache_key(preset, ["m1", "m2"], "linear")
        k2 = _compute_cache_key(preset, ["m2", "m1"], "linear")
        assert k1 == k2

    def test_different_models_different_key(self):
        preset = {"dwm": {}}
        k1 = _compute_cache_key(preset, ["m1"], "linear")
        k2 = _compute_cache_key(preset, ["m1", "m2"], "linear")
        assert k1 != k2

    def test_different_preset_different_key(self):
        k1 = _compute_cache_key({"dwm": {}}, ["m1"], "linear")
        k2 = _compute_cache_key({"cls_acc": {}}, ["m1"], "linear")
        assert k1 != k2

    def test_different_ranking_strategy_different_key(self):
        preset = {"dwm": {}}
        k1 = _compute_cache_key(preset, ["m1"], "linear")
        k2 = _compute_cache_key(preset, ["m1"], "summation")
        assert k1 != k2

    def test_different_evaluation_fingerprint_different_key(self):
        k1 = _compute_cache_key({"dwm": {}}, ["m1"], "linear", "real-data-a")
        k2 = _compute_cache_key({"dwm": {}}, ["m1"], "linear", "real-data-b")
        assert k1 != k2

    def test_returns_hex_string(self):
        key = _compute_cache_key({"dwm": {}}, ["m1"], "linear")
        assert isinstance(key, str)
        int(key, 16)  # raises ValueError if not valid hex


class TestSaveLoadSynthevalCache:
    def test_roundtrip_results_and_ranks(self, tmp_path):
        results = _make_results()
        ranks = _make_ranks()
        key = _compute_cache_key({"dwm": {}}, ["m1", "m2"], "linear")
        _save_syntheval_cache(results, ranks, tmp_path, "main", key)

        loaded = _load_syntheval_cache(tmp_path, "main", key)
        assert loaded is not None
        loaded_results, loaded_ranks = loaded
        pd.testing.assert_frame_equal(loaded_results, results)
        pd.testing.assert_frame_equal(loaded_ranks, ranks)

    def test_cache_miss_when_no_files(self, tmp_path):
        key = _compute_cache_key({"dwm": {}}, ["m1"], "linear")
        assert _load_syntheval_cache(tmp_path, "main", key) is None

    def test_cache_miss_when_key_changed(self, tmp_path):
        results, ranks = _make_results(), _make_ranks()
        old_key = _compute_cache_key({"dwm": {}}, ["m1", "m2"], "linear")
        new_key = _compute_cache_key({"dwm": {}}, ["m1", "m2", "m3"], "linear")
        _save_syntheval_cache(results, ranks, tmp_path, "main", old_key)
        assert _load_syntheval_cache(tmp_path, "main", new_key) is None

    def test_cache_miss_when_meta_corrupted(self, tmp_path):
        results, ranks = _make_results(), _make_ranks()
        key = _compute_cache_key({"dwm": {}}, ["m1", "m2"], "linear")
        _save_syntheval_cache(results, ranks, tmp_path, "main", key)
        (tmp_path / "main_cache_meta.json").write_text("not json")
        assert _load_syntheval_cache(tmp_path, "main", key) is None

    def test_cache_miss_when_results_parquet_missing(self, tmp_path):
        results, ranks = _make_results(), _make_ranks()
        key = _compute_cache_key({"dwm": {}}, ["m1", "m2"], "linear")
        _save_syntheval_cache(results, ranks, tmp_path, "main", key)
        (tmp_path / "main_results.parquet").unlink()
        assert _load_syntheval_cache(tmp_path, "main", key) is None

    def test_separate_prefixes_do_not_collide(self, tmp_path):
        results_a = _make_results(index=["a1", "a2"])
        ranks_a = _make_ranks(index=["a1", "a2"])
        results_b = _make_results(index=["b1", "b2"])
        ranks_b = _make_ranks(index=["b1", "b2"])
        key = _compute_cache_key({"dwm": {}}, ["m1", "m2"], "linear")

        _save_syntheval_cache(results_a, ranks_a, tmp_path, "main", key)
        _save_syntheval_cache(results_b, ranks_b, tmp_path, "binary_target", key)

        loaded_a = _load_syntheval_cache(tmp_path, "main", key)
        loaded_b = _load_syntheval_cache(tmp_path, "binary_target", key)
        assert loaded_a is not None and loaded_b is not None
        assert list(loaded_a[0].index) == ["a1", "a2"]
        assert list(loaded_b[0].index) == ["b1", "b2"]

    def test_meta_json_contains_cache_key(self, tmp_path):
        results, ranks = _make_results(), _make_ranks()
        key = _compute_cache_key({"dwm": {}}, ["m1", "m2"], "linear")
        _save_syntheval_cache(results, ranks, tmp_path, "main", key)
        meta = json.loads((tmp_path / "main_cache_meta.json").read_text())
        assert meta["cache_key"] == key


class TestResolveModelWorkers:
    def test_explicit_limit_obeys_model_and_max_bounds(self):
        cfg = SynthEvalExecutionConfig(model_workers=10, max_model_workers=4)
        assert resolve_model_workers(cfg, n_models=3, n_columns=10) == 3

    def test_auto_uses_memory_and_cpu_bounds(self, monkeypatch):
        cfg = SynthEvalExecutionConfig(
            model_workers="auto",
            max_model_workers=8,
            cores_per_model=4,
            memory_reserve_gib=16,
        )
        monkeypatch.setattr("synthdata.evaluation.syntheval_eval.os.cpu_count", lambda: 24)
        monkeypatch.setattr(
            "synthdata.evaluation.syntheval_eval._available_memory_gib", lambda: 118.0
        )
        assert resolve_model_workers(cfg, n_models=18, n_columns=1038) == 6


class TestCheckpointPaths:
    def test_absolute_checkpoint_path_survives_plot_directory_change(self, tmp_path):
        checkpoint_root = tmp_path / "evaluation" / "syntheval_benchmark"
        model_dir, _, result_path = _checkpoint_paths(checkpoint_root.resolve(), "main", "model_a")
        model_dir.mkdir(parents=True)
        plot_dir = tmp_path / "plots" / "model_a"
        plot_dir.mkdir(parents=True)
        previous_dir = Path.cwd()
        try:
            os.chdir(plot_dir)
            _atomic_parquet(result_path, pd.DataFrame({"metric": ["dwm"], "val": [0.5]}))
        finally:
            os.chdir(previous_dir)
        assert result_path.exists()
