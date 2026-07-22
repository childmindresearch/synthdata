"""Unit tests for synthdata.evaluation.catalog.resolve_selection (partial-
selection precedence resolver used by every evaluation framework).
"""

import pytest

from synthdata.evaluation.catalog import resolve_selection

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
