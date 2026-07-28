"""Unit tests for synthdata.evaluation.privacy_gate: absolute privacy
threshold checks against the combined evaluation table's raw metric values.
"""

import pandas as pd
import pytest

from synthdata.evaluation.privacy_gate import (
    evaluate_privacy_gate,
    merge_privacy_gate_results,
)

pytestmark = pytest.mark.unit


class _FakeGateConfig:
    def __init__(self, enabled=True, thresholds=None):
        self.enabled = enabled
        self.thresholds = thresholds if thresholds is not None else {}


def _combined(metric_name: str, values: dict, framework="syntheval", type_="privacy"):
    df = pd.DataFrame(index=list(values))
    df[(framework, type_, metric_name)] = pd.Series(values)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


class TestEvaluatePrivacyGate:
    def test_disabled_returns_none(self):
        combined = _combined("mia_recall", {"model_a": 0.5})
        cfg = _FakeGateConfig(
            enabled=False, thresholds={"mia_recall": {"bound": "max", "value": 0.6}}
        )
        assert evaluate_privacy_gate(combined, cfg) is None

    def test_no_thresholds_returns_none(self):
        combined = _combined("mia_recall", {"model_a": 0.5})
        cfg = _FakeGateConfig(thresholds={})
        assert evaluate_privacy_gate(combined, cfg) is None

    def test_metric_not_found_returns_none_and_does_not_raise(self):
        combined = _combined("mia_recall", {"model_a": 0.5})
        cfg = _FakeGateConfig(thresholds={"not_a_real_metric": {"bound": "max", "value": 0.6}})
        assert evaluate_privacy_gate(combined, cfg) is None

    def test_max_bound_pass_and_fail(self):
        combined = _combined("mia_recall", {"model_a": 0.5, "model_b": 0.9})
        cfg = _FakeGateConfig(thresholds={"mia_recall": {"bound": "max", "value": 0.6}})
        result = evaluate_privacy_gate(combined, cfg)
        assert result.loc["model_a", "pass"] is True or result.loc["model_a", "pass"] == True  # noqa: E712
        assert result.loc["model_b", "pass"] == False  # noqa: E712
        assert "mia_recall=0.9" in result.loc["model_b", "violations"]

    def test_min_bound_pass_and_fail(self):
        combined = _combined(
            "privacy.k-anonymization.syn", {"model_a": 10.0, "model_b": 2.0}, type_="privacy"
        )
        cfg = _FakeGateConfig(
            thresholds={"privacy.k-anonymization.syn": {"bound": "min", "value": 5.0}}
        )
        result = evaluate_privacy_gate(combined, cfg)
        assert result.loc["model_a", "pass"] == True  # noqa: E712
        assert result.loc["model_b", "pass"] == False  # noqa: E712

    def test_nan_value_fails_conservatively_not_silently_passes(self):
        combined = _combined("mia_recall", {"model_a": float("nan")})
        cfg = _FakeGateConfig(thresholds={"mia_recall": {"bound": "max", "value": 0.6}})
        result = evaluate_privacy_gate(combined, cfg)
        assert result.loc["model_a", "pass"] == False  # noqa: E712
        assert "could not evaluate" in result.loc["model_a", "violations"]

    def test_multiple_thresholds_all_must_pass(self):
        df = pd.DataFrame(index=["model_a", "model_b"])
        df[("syntheval", "privacy", "mia_recall")] = [0.5, 0.5]
        df[("syntheval", "privacy", "hit_rate")] = [0.01, 0.9]
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        cfg = _FakeGateConfig(
            thresholds={
                "mia_recall": {"bound": "max", "value": 0.6},
                "hit_rate": {"bound": "max", "value": 0.05},
            }
        )
        result = evaluate_privacy_gate(df, cfg)
        assert result.loc["model_a", "pass"] == True  # noqa: E712
        assert result.loc["model_b", "pass"] == False  # noqa: E712
        assert "hit_rate" in result.loc["model_b", "violations"]
        assert "mia_recall" not in result.loc["model_b", "violations"]

    def test_partial_metric_availability_only_checks_found_metrics(self):
        combined = _combined("mia_recall", {"model_a": 0.5})
        cfg = _FakeGateConfig(
            thresholds={
                "mia_recall": {"bound": "max", "value": 0.6},
                "not_computed_this_run": {"bound": "max", "value": 0.3},
            }
        )
        result = evaluate_privacy_gate(combined, cfg)
        assert result is not None
        assert result.loc["model_a", "pass"] == True  # noqa: E712


class TestMergePrivacyGateResults:
    def test_none_result_is_noop(self):
        combined = _combined("mia_recall", {"model_a": 0.5})
        merged = merge_privacy_gate_results(combined, None)
        assert merged.equals(combined)

    def test_merges_pass_and_violations_columns(self):
        combined = _combined("mia_recall", {"model_a": 0.5})
        cfg = _FakeGateConfig(thresholds={"mia_recall": {"bound": "max", "value": 0.6}})
        gate_result = evaluate_privacy_gate(combined, cfg)
        merged = merge_privacy_gate_results(combined, gate_result)
        assert ("__all__", "privacy_gate", "pass") in merged.columns
        assert ("__all__", "privacy_gate", "violations") in merged.columns
        assert merged.loc["model_a", ("__all__", "privacy_gate", "pass")] == True  # noqa: E712
