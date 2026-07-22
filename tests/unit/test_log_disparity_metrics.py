"""Unit tests for synthdata.log_disparity.metric_log_disparity's pure numeric/
statistical functions.
"""

import math

import pandas as pd
import pytest

from synthdata.log_disparity.metric_log_disparity import (
    SENTINEL_INSUFFICIENT,
    SENTINEL_NO_BASE,
    SENTINEL_NO_INFO,
    _is_integer_like_edges,
    _max_decimal_places,
    assign_equity_color,
    benjamini_hochberg_correction,
    classify_equity_outcome,
    compare_population_proportion,
    generate_bin_labels,
    log_disparate_impact,
    prepare_data_for_analysis,
    rate_calculation,
)

pytestmark = pytest.mark.unit


class TestRateCalculation:
    def test_normal_rate(self):
        assert rate_calculation(25, 100) == pytest.approx(0.25)

    def test_zero_total_returns_zero(self):
        assert rate_calculation(0, 0) == 0.0

    def test_zero_subgroup(self):
        assert rate_calculation(0, 50) == 0.0

    def test_full_subgroup(self):
        assert rate_calculation(10, 10) == 1.0


class TestLogDisparateImpact:
    def test_equal_rates_near_zero(self):
        assert log_disparate_impact(0.5, 0.5) == pytest.approx(0.0)

    def test_both_zero_returns_no_info_sentinel(self):
        assert log_disparate_impact(0.0, 0.0) == SENTINEL_NO_INFO

    def test_synthetic_zero_returns_neg_inf(self):
        assert log_disparate_impact(0.5, 0.0) == -math.inf

    def test_background_zero_returns_no_base_sentinel(self):
        assert log_disparate_impact(0.0, 0.5) == SENTINEL_NO_BASE

    def test_both_one_returns_zero(self):
        assert log_disparate_impact(1.0, 1.0) == 0.0

    def test_synthetic_one_returns_inf(self):
        assert log_disparate_impact(0.5, 1.0) == math.inf

    def test_background_one_returns_neg_inf(self):
        assert log_disparate_impact(1.0, 0.5) == -math.inf

    def test_higher_synthetic_rate_gives_positive_value(self):
        assert log_disparate_impact(0.3, 0.6) > 0

    def test_lower_synthetic_rate_gives_negative_value(self):
        assert log_disparate_impact(0.6, 0.3) < 0

    def test_for_plot_true_matches_normal_path_for_mid_range_rates(self):
        # Neither rate is 0/1, so the sentinel-skipping flag makes no
        # difference here -- both paths reach the same log-odds computation.
        assert log_disparate_impact(0.4, 0.6, for_plot=True) == pytest.approx(
            log_disparate_impact(0.4, 0.6, for_plot=False)
        )

    def test_for_plot_true_bypasses_sentinel_guard_for_zero_rates(self):
        # for_plot=True skips the sentinel branch entirely (even for exact-
        # zero rates), falling through to the raw log-odds computation --
        # which is undefined for a literal 0 rate. This documents a known
        # edge case: for_plot is only safe for slightly-perturbed rates used
        # in visualization, not literal 0/1 inputs.
        with pytest.raises(ValueError, match="math domain error"):
            log_disparate_impact(0.0, 0.0, for_plot=True)


class TestComparePopulationProportion:
    def test_valid_sample_returns_p_value_in_unit_interval(self):
        p = compare_population_proportion(50, 100, 45, 100)
        assert 0.0 <= p <= 1.0

    def test_identical_proportions_high_p_value(self):
        p = compare_population_proportion(50, 100, 50, 100)
        assert p == pytest.approx(1.0)

    def test_both_zero_observed_and_background_returns_no_info(self):
        assert compare_population_proportion(0, 100, 0, 100) == SENTINEL_NO_INFO

    def test_observed_zero_returns_neg_inf(self):
        assert compare_population_proportion(10, 100, 0, 100) == -math.inf

    def test_background_zero_returns_no_base(self):
        assert compare_population_proportion(0, 100, 10, 100) == SENTINEL_NO_BASE

    def test_small_counts_return_insufficient_sentinel(self):
        # All counts must be >=5 for the chi-squared path; here they aren't.
        assert compare_population_proportion(2, 3, 1, 2) == SENTINEL_INSUFFICIENT


class TestBenjaminiHochbergCorrection:
    def test_monotonic_and_bounded(self):
        p_values = pd.Series([0.01, 0.04, 0.03, 0.10])
        adjusted = benjamini_hochberg_correction(p_values)
        assert (adjusted >= p_values).all()
        assert (adjusted <= 1.0).all()

    def test_empty_series_returns_all_nan(self):
        adjusted = benjamini_hochberg_correction(pd.Series(dtype=float))
        assert adjusted.empty

    def test_sentinels_become_nan(self):
        p_values = pd.Series([0.02, float(SENTINEL_NO_INFO), 0.5])
        adjusted = benjamini_hochberg_correction(p_values)
        assert pd.isna(adjusted.iloc[1])
        assert pd.notna(adjusted.iloc[0])
        assert pd.notna(adjusted.iloc[2])

    def test_all_invalid_returns_all_nan(self):
        p_values = pd.Series([float(SENTINEL_NO_INFO), float(SENTINEL_NO_BASE)])
        adjusted = benjamini_hochberg_correction(p_values)
        assert adjusted.isna().all()


class TestClassifyEquityOutcome:
    @pytest.mark.parametrize(
        ("disparity", "p_value", "expected"),
        [
            (-math.inf, 0.5, "Absent"),
            (SENTINEL_NO_INFO, 0.5, "No Info"),
            (SENTINEL_NO_BASE, 0.5, "No Base Data"),
            (SENTINEL_INSUFFICIENT, 0.5, "Insufficient Data"),
            (0.3, 0.80, "Equitable(p)"),  # not significant overrides magnitude
            (-0.3, 0.01, "Highly Underrepresented"),
            (-0.15, 0.01, "Underrepresented"),
            (0.0, 0.01, "Equitable"),
            (0.15, 0.01, "Overrepresented"),
            (0.3, 0.01, "Highly Overrepresented"),
        ],
    )
    def test_all_outcome_branches(self, disparity, p_value, expected):
        assert classify_equity_outcome(disparity, p_value) == expected

    def test_nan_p_value_does_not_trigger_equitable_p(self):
        # pd.notna(nan) is False, so the "not significant" branch is skipped
        # and classification falls through to the magnitude-based branches.
        assert classify_equity_outcome(0.3, float("nan")) == "Highly Overrepresented"


class TestAssignEquityColor:
    def test_returns_mapped_color_for_known_label(self):
        color = assign_equity_color(-math.inf, 0.5)
        assert color == "#ab2328"  # "Absent"

    def test_returns_black_fallback_for_unknown_label(self, monkeypatch):
        import synthdata.log_disparity.metric_log_disparity as mod

        monkeypatch.setattr(mod, "classify_equity_outcome", lambda *a, **k: "Nonexistent")
        assert assign_equity_color(0.0, 0.5) == "#000000"


class TestBinLabelHelpers:
    def test_integer_like_edges_detected(self):
        assert _is_integer_like_edges([7.0, 18.0, 30.0])
        assert not _is_integer_like_edges([7.0, 18.5, 30.0])

    def test_max_decimal_places(self):
        assert _max_decimal_places([0, 0.5, 1.0]) == 1
        assert _max_decimal_places([0.001, 0.002]) == 3
        assert _max_decimal_places([1, 2, 3]) == 0

    def test_generate_bin_labels_integer(self):
        labels = generate_bin_labels([18, 30, 45, 60])
        assert labels == ["18-30", "31-45", "46-60"]

    def test_generate_bin_labels_float_precision(self):
        labels = generate_bin_labels([0.0, 0.5, 1.0])
        assert labels == ["0.00-0.50", "0.51-1.00"]

    def test_generate_bin_labels_requires_at_least_two_edges(self):
        with pytest.raises(ValueError, match="at least 2"):
            generate_bin_labels([1.0])

    def test_generate_bin_labels_rejects_non_list(self):
        with pytest.raises(ValueError, match="at least 2"):
            generate_bin_labels((1.0, 2.0))


class TestPrepareDataForAnalysis:
    def test_missing_target_column_raises(self):
        df = pd.DataFrame({"age": [1, 2]})
        with pytest.raises(KeyError, match="target_col"):
            prepare_data_for_analysis(df, "outcome", ["age"])

    def test_missing_protected_column_raises(self):
        df = pd.DataFrame({"outcome": [0, 1]})
        with pytest.raises(KeyError, match="protected column"):
            prepare_data_for_analysis(df, "outcome", ["age"])

    def test_empty_protected_cols_raises(self):
        df = pd.DataFrame({"outcome": [0, 1], "age": [20, 30]})
        with pytest.raises(ValueError, match="non-empty"):
            prepare_data_for_analysis(df, "outcome", [])

    def test_target_map_and_bins_mutually_exclusive(self):
        df = pd.DataFrame({"outcome": [0, 1], "age": [20, 30]})
        with pytest.raises(ValueError, match="mutually exclusive"):
            prepare_data_for_analysis(
                df,
                "outcome",
                ["age"],
                target_map={0: "no"},
                target_bins=[0, 1],
            )

    def test_protected_map_length_mismatch_raises(self):
        df = pd.DataFrame({"outcome": [0, 1], "age": [20, 30], "sex": [0, 1]})
        with pytest.raises(ValueError, match="protected_map length"):
            prepare_data_for_analysis(df, "outcome", ["age", "sex"], protected_map=[{0: "young"}])

    def test_applies_map_and_bins(self):
        df = pd.DataFrame({"outcome": [0, 1, 1], "sex": [0, 1, 0], "age": [25, 35, 65]})
        prepared, group_cols, order_maps, target_order = prepare_data_for_analysis(
            df,
            "outcome",
            ["sex", "age"],
            protected_map=[{0: "male", 1: "female"}, None],
            protected_bins=[None, [0, 30, 60, 100]],
        )
        assert group_cols == ["sex__GROUP", "age__GROUP"]
        assert prepared["sex__GROUP"].tolist() == ["male", "female", "male"]
        assert "age__GROUP" in order_maps
        assert target_order is None

    def test_target_bins_applied(self):
        df = pd.DataFrame({"outcome": [10, 40, 70], "age": [20, 30, 40]})
        prepared, _, _, target_order = prepare_data_for_analysis(
            df, "outcome", ["age"], target_bins=[0, 30, 60, 100]
        )
        assert "TARGET_LABEL" in prepared.columns
        assert target_order is not None

    def test_out_of_range_bin_value_labeled(self):
        df = pd.DataFrame({"outcome": [0, 1], "age": [200, 30]})
        prepared, group_cols, order_maps, _ = prepare_data_for_analysis(
            df, "outcome", ["age"], protected_bins=[[0, 30, 60]]
        )
        assert "OutOfRange" in prepared["age__GROUP"].tolist()
