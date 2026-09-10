"""Unit tests for TabPFN integration compatibility patches."""

import sys
import types

import numpy as np
import pandas as pd
import pytest
import torch

from synthdata.data import semantic_context_digest, semantic_context_payload
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


def test_tabpfn_custom_consumes_semantic_context(make_canonical_dataset, mocker):
    dataset = make_canonical_dataset("column")
    train_frame = dataset.role_frame("train", imputed=True).reset_index(drop=True)
    semantic_context = semantic_context_payload(
        dataset,
        classification_score="macro_f1",
    )

    class FakeExperiment:
        def __init__(self):
            self.data = train_frame.copy()
            self.synthetic_X = torch.zeros((2, len(train_frame.columns)))

        def run(self, **kwargs):
            self.run_kwargs = kwargs

    experiment = FakeExperiment()

    class FakeClassifier:
        def fit(self, features, labels):
            del features, labels
            return self

        def predict(self, features):
            return np.zeros(len(features), dtype=int)

    mocker.patch.object(tabpfn_backend, "_make_experiment", return_value=(experiment, object()))
    mocker.patch("tabpfn.TabPFNClassifier", FakeClassifier)

    generated, returned_experiment = tabpfn_backend.generate_tabpfn_custom(
        train_frame,
        dataset.categorical_columns,
        dataset.target_column,
        2,
        target_is_categorical=True,
        variable_schema_fingerprint=dataset.variable_schema_fingerprint,
        semantic_context=semantic_context,
    )

    assert len(generated) == 2
    assert returned_experiment.semantic_context == semantic_context
    assert returned_experiment.semantic_context_digest == semantic_context_digest(semantic_context)
