"""Unit tests for TabPFN integration compatibility patches."""

import sys
import types

import numpy as np
import pandas as pd
import pytest

from synthdata.generation import tabpfn_backend

pytestmark = pytest.mark.unit


def _install_unsupervised_module(monkeypatch):
    extensions = types.ModuleType("tabpfn_extensions")
    extensions.__path__ = []
    unsupervised_package = types.ModuleType("tabpfn_extensions.unsupervised")
    unsupervised_package.__path__ = []
    unsupervised_module = types.ModuleType("tabpfn_extensions.unsupervised.unsupervised")

    extensions.unsupervised = unsupervised_package
    unsupervised_package.unsupervised = unsupervised_module
    monkeypatch.setitem(sys.modules, "tabpfn_extensions", extensions)
    monkeypatch.setitem(sys.modules, "tabpfn_extensions.unsupervised", unsupervised_package)
    monkeypatch.setitem(
        sys.modules,
        "tabpfn_extensions.unsupervised.unsupervised",
        unsupervised_module,
    )
    return unsupervised_module


def test_explicit_type_patch_disables_cardinality_inference(monkeypatch):
    """Continuous low-cardinality columns must not become categorical."""
    unsupervised = _install_unsupervised_module(monkeypatch)

    tabpfn_backend._patch_explicit_categorical_feature_inference()

    assert unsupervised.infer_categorical_features([[1, 2], [1, 3]], categorical_features=[1]) == [
        1
    ]
    assert unsupervised.infer_categorical_features([[1, 2], [1, 3]], categorical_features=[]) == []


def test_explicit_type_patch_is_idempotent(monkeypatch):
    unsupervised = _install_unsupervised_module(monkeypatch)

    tabpfn_backend._patch_explicit_categorical_feature_inference()
    first = unsupervised.infer_categorical_features
    tabpfn_backend._patch_explicit_categorical_feature_inference()

    assert unsupervised.infer_categorical_features is first


@pytest.mark.parametrize(
    "generator, arguments",
    [
        (
            tabpfn_backend.generate_tabpfn_standard,
            (
                pd.DataFrame({"feature": [1.0, 2.0], "target": [0.5, 1.5]}),
                ["feature"],
                [],
                "target",
                2,
            ),
        ),
        (
            tabpfn_backend.generate_tabpfn_custom,
            (pd.DataFrame({"feature": [1.0, 2.0], "target": [0.5, 1.5]}), [], "target", 2),
        ),
    ],
)
def test_tabpfn_generators_reject_continuous_target(generator, arguments):
    with pytest.raises(ValueError, match="target column 'target'.*continuous"):
        generator(*arguments, target_is_categorical=False)


def test_sample_labels_follows_class_probabilities():
    proba = np.tile([0.8, 0.15, 0.05], (20000, 1))
    labels = tabpfn_backend.sample_labels(proba, np.array(["a", "b", "c"]), seed=0)
    shares = pd.Series(labels).value_counts(normalize=True)
    # argmax would return "a" for every row
    assert shares["a"] == pytest.approx(0.8, abs=0.01)
    assert shares["b"] == pytest.approx(0.15, abs=0.01)
    assert shares["c"] == pytest.approx(0.05, abs=0.01)


def test_sample_labels_is_seeded_and_handles_certain_rows():
    proba = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    first = tabpfn_backend.sample_labels(proba, np.array([0, 1]), seed=3)
    assert list(first[:2]) == [0, 1]
    assert list(first) == list(tabpfn_backend.sample_labels(proba, np.array([0, 1]), seed=3))
