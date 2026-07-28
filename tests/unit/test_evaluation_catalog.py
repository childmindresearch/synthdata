"""Unit tests for synthdata.evaluation.catalog.resolve_selection (partial-
selection precedence resolver used by every evaluation framework) and the
SynthEval result-column classification helpers.
"""

import pytest

from synthdata.evaluation.catalog import (
    LOG_DISPARITY_METRICS,
    classify_syntheval_metric,
    is_custom_syntheval_metric,
    is_redundant_synthcity_submetric,
    resolve_selection,
)

pytestmark = pytest.mark.unit

ALL_METRICS = ["m1", "m2", "m3", "m4"]
TYPE_MAP = {"m1": "utility", "m2": "utility", "m3": "privacy", "m4": "fairness"}


class TestResolveSelection:
    def test_disabled_returns_empty(self):
        assert resolve_selection(False, None, None, ALL_METRICS, TYPE_MAP) == []

    def test_disabled_wins_over_explicit_metrics(self):
        assert resolve_selection(False, None, ["m1"], ALL_METRICS, TYPE_MAP) == []

    def test_explicit_metrics_take_precedence_over_categories(self):
        result = resolve_selection(True, ["privacy"], ["m1"], ALL_METRICS, TYPE_MAP)
        assert result == ["m1"]

    def test_explicit_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            resolve_selection(True, None, ["not_a_metric"], ALL_METRICS, TYPE_MAP)

    def test_categories_filter_by_type(self):
        result = resolve_selection(True, ["utility"], None, ALL_METRICS, TYPE_MAP)
        assert result == ["m1", "m2"]

    def test_multiple_categories(self):
        result = resolve_selection(True, ["privacy", "fairness"], None, ALL_METRICS, TYPE_MAP)
        assert result == ["m3", "m4"]

    def test_neither_given_returns_all(self):
        result = resolve_selection(True, None, None, ALL_METRICS, TYPE_MAP)
        assert result == ALL_METRICS

    def test_empty_metrics_list_falls_through_to_categories(self):
        # An empty (falsy) explicit list should not be treated as "given".
        result = resolve_selection(True, ["utility"], [], ALL_METRICS, TYPE_MAP)
        assert result == ["m1", "m2"]


class TestClassifySynthevalMetric:
    def test_known_metric_key_matches_dict(self):
        assert classify_syntheval_metric("statistical_parity") == "fairness"
        assert classify_syntheval_metric("dwm") == "utility"
        assert classify_syntheval_metric("nnaa") == "privacy"

    def test_auroc_diffs_actual_result_column_name_is_utility(self):
        # auroc_diff's own result column is literally "auroc", not "auroc_diff".
        assert classify_syntheval_metric("auroc") == "utility"

    def test_auroc_per_target_submetric_is_utility(self):
        assert classify_syntheval_metric("auroc_CGAS_class") == "utility"

    @pytest.mark.parametrize("prefix", ["sp_", "eo_", "eqo_"])
    def test_fairness_submetrics_are_fairness(self, prefix):
        assert classify_syntheval_metric(f"{prefix}CGAS_class_Sex") == "fairness"

    def test_unknown_metric_defaults_to_utility(self):
        assert classify_syntheval_metric("some_unrecognised_metric") == "utility"


class TestIsCustomSynthevalMetric:
    def test_primary_custom_fairness_keys(self):
        assert is_custom_syntheval_metric("equalized_odds") is True
        assert is_custom_syntheval_metric("equal_opportunity") is True

    def test_non_custom_fairness_key(self):
        assert is_custom_syntheval_metric("statistical_parity") is False

    @pytest.mark.parametrize("prefix", ["eo_", "eqo_"])
    def test_custom_submetric_prefixes(self, prefix):
        assert is_custom_syntheval_metric(f"{prefix}CGAS_class_Sex") is True

    def test_non_custom_submetric_prefix(self):
        assert is_custom_syntheval_metric("sp_CGAS_class_Sex") is False

    def test_unrelated_metric(self):
        assert is_custom_syntheval_metric("dwm") is False


class TestIsRedundantSynthcitySubmetric:
    @pytest.mark.parametrize(
        "metric_key",
        [
            "stats.alpha_precision.delta_precision_alpha_naive",
            "stats.alpha_precision.delta_coverage_beta_naive",
            "stats.alpha_precision.authenticity_naive",
        ],
    )
    def test_naive_alpha_precision_submetrics_flagged_redundant(self, metric_key):
        assert is_redundant_synthcity_submetric(metric_key) is True

    @pytest.mark.parametrize(
        "metric_key",
        [
            "stats.alpha_precision.delta_precision_alpha_OC",
            "stats.alpha_precision.delta_coverage_beta_OC",
            "stats.alpha_precision.authenticity_OC",
            "stats.ks_test.marginal",
            "privacy.identifiability_score.score_OC",
        ],
    )
    def test_non_naive_submetrics_not_flagged(self, metric_key):
        assert is_redundant_synthcity_submetric(metric_key) is False


class TestLogDisparityMetricsExcludesMedian:
    def test_median_abs_not_a_ranked_metric(self):
        # log_disparity_median_abs is computed from the exact same per-subgroup
        # value array as log_disparity_mean_abs (see metric_log_disparity.py) --
        # deliberately excluded here to avoid double-counting one signal.
        assert "log_disparity_median_abs" not in LOG_DISPARITY_METRICS

    def test_mean_abs_and_share_significant_still_ranked(self):
        assert "log_disparity_mean_abs" in LOG_DISPARITY_METRICS
        assert "log_disparity_share_significant" in LOG_DISPARITY_METRICS
