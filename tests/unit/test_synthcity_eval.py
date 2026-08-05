"""Unit tests for synthcity metric selection and native category names."""

import pytest

from synthdata.config import FrameworkSelectionConfig
from synthdata.evaluation.synthcity_eval import resolve_metric_config

pytestmark = pytest.mark.unit

ATTACK_METRICS = [
    "data_leakage_mlp",
    "data_leakage_xgb",
    "data_leakage_linear",
]


class TestResolveMetricConfig:
    def test_default_selection_uses_native_attack_category(self):
        result = resolve_metric_config(FrameworkSelectionConfig())

        assert result["attack"] == ATTACK_METRICS
        assert "attacks" not in result

    def test_privacy_category_includes_attack_metrics(self):
        result = resolve_metric_config(FrameworkSelectionConfig(categories=["privacy"]))

        assert result["attack"] == ATTACK_METRICS

    def test_explicit_attack_metric_uses_native_category(self):
        result = resolve_metric_config(FrameworkSelectionConfig(metrics=["data_leakage_linear"]))

        assert result == {"attack": ["data_leakage_linear"]}
