"""Unit tests for synthcity metric selection and native category names."""

import pytest

from synthdata.config import FrameworkSelectionConfig
from synthdata.evaluation.catalog import SYNTHCITY_METRIC_CONFIG
from synthdata.evaluation.synthcity_eval import resolve_metric_config

pytestmark = pytest.mark.unit


class TestResolveMetricConfig:
    def test_default_selection_uses_native_attack_category(self):
        result = resolve_metric_config(FrameworkSelectionConfig())

        assert result["attack"] == SYNTHCITY_METRIC_CONFIG["attack"]
        assert "attacks" not in result

    def test_final_evaluation_catalog_retains_domias(self):
        assert "DomiasMIA_prior" in SYNTHCITY_METRIC_CONFIG["privacy"]

    def test_privacy_category_includes_attack_metrics(self):
        result = resolve_metric_config(FrameworkSelectionConfig(categories=["privacy"]))

        assert result["attack"] == SYNTHCITY_METRIC_CONFIG["attack"]

    def test_explicit_attack_metric_uses_native_category(self):
        result = resolve_metric_config(FrameworkSelectionConfig(metrics=["data_leakage_linear"]))

        assert result == {"attack": ["data_leakage_linear"]}
