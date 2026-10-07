"""Unit tests for synthcity metric selection, native category names, and what
each synthcity ``Metrics.evaluate`` argument receives."""

import pandas as pd
import pytest

from synthdata.config import FrameworkSelectionConfig
from synthdata.evaluation import synthcity_eval
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


@pytest.fixture
def captured(monkeypatch):
    metrics_module = pytest.importorskip("synthcity.metrics")
    calls = []

    def fake_evaluate(x_gt, x_syn, x_train=None, x_ref_syn=None, x_augmented=None, **kwargs):
        loaders = {
            "gt": x_gt,
            "syn": x_syn,
            "train": x_train,
            "ref_syn": x_ref_syn,
            "augmented": x_augmented,
        }
        calls.append(
            {k: (v.dataframe() if v is not None else None) for k, v in loaders.items()} | kwargs
        )
        keys = [f"{cat}.{name}.score" for cat, names in kwargs["metrics"].items() for name in names]
        return pd.DataFrame({"mean": 0.0, "direction": "maximize"}, index=keys)

    monkeypatch.setattr(metrics_module.Metrics, "evaluate", staticmethod(fake_evaluate))
    return calls


def _frames():
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "target": [0, 1, 0, 1]})
    test = pd.DataFrame({"a": [5.0, 6.0], "target": [1, 0]})
    synthetic = pd.DataFrame({"target": [0, 1, 1], "a": [1.0, 1.0, 9.0]})
    return train, test, synthetic


METRICS = {
    "sanity": ["common_rows_proportion"],
    "stats": ["ks_test"],
    "performance": ["xgb", "xgb_augmentation"],
    "privacy": ["identifiability_score", "DomiasMIA_prior"],
    "attack": ["data_leakage_linear"],
}


def _run(captured):
    train, test, synthetic = _frames()
    result = synthcity_eval.run_synthcity_metrics(
        synthetic, test, train, "target", [], METRICS, workspace=None
    )
    on_train, on_held_out = captured
    return train, test, synthetic, result, on_train, on_held_out


def test_synthetic_rows_are_scored_as_generated(captured):
    train, _, synthetic, _, on_train, on_held_out = _run(captured)
    expected = synthetic[["a", "target"]]
    for call in (on_train, on_held_out):
        pd.testing.assert_frame_equal(call["syn"].reset_index(drop=True), expected)
    pd.testing.assert_frame_equal(on_held_out["ref_syn"].reset_index(drop=True), expected)
    assert len(on_held_out["augmented"]) == len(train) + len(synthetic)


def test_memorisation_and_fidelity_metrics_compare_against_train(captured):
    train, _, _, _, on_train, _ = _run(captured)
    pd.testing.assert_frame_equal(on_train["gt"].reset_index(drop=True), train)
    assert on_train["metrics"] == {
        "sanity": ["common_rows_proportion"],
        "stats": ["ks_test"],
        "privacy": ["identifiability_score"],
        "attack": ["data_leakage_linear"],
    }


def test_performance_and_domias_use_held_out_rows(captured):
    train, test, _, _, _, on_held_out = _run(captured)
    pd.testing.assert_frame_equal(on_held_out["gt"].reset_index(drop=True), test)
    pd.testing.assert_frame_equal(on_held_out["train"].reset_index(drop=True), train)
    assert on_held_out["metrics"] == {
        "performance": ["xgb", "xgb_augmentation"],
        "privacy": ["DomiasMIA_prior"],
    }


def test_results_of_both_passes_are_stacked(captured):
    *_, result, _, _ = _run(captured)
    assert len(result) == sum(len(names) for names in METRICS.values())


def test_failed_metrics_are_logged(monkeypatch, caplog):
    metrics_module = pytest.importorskip("synthcity.metrics")

    def drops_xgb(x_gt, x_syn, *args, metrics, **kwargs):
        keys = [
            f"{cat}.{name}.score"
            for cat, names in metrics.items()
            for name in names
            if name != "xgb"
        ]
        return pd.DataFrame({"mean": 0.0, "direction": "maximize"}, index=keys)

    monkeypatch.setattr(metrics_module.Metrics, "evaluate", staticmethod(drops_xgb))
    train, test, synthetic = _frames()
    synthcity_eval.logger.addHandler(caplog.handler)
    try:
        with caplog.at_level("WARNING"):
            synthcity_eval.run_synthcity_metrics(
                synthetic, test, train, "target", [], METRICS, workspace=None
            )
    finally:
        synthcity_eval.logger.removeHandler(caplog.handler)
    assert "'performance.xgb'" in caplog.text
    assert "xgb_augmentation" not in caplog.text
