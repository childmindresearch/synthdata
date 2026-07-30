"""Unit tests for TabPFN integration compatibility patches."""

import sys
import types

import pytest

from synthdata.generation import tabpfn_backend

pytestmark = pytest.mark.unit


def test_explicit_type_patch_disables_cardinality_inference(monkeypatch):
    """Continuous low-cardinality columns must not become categorical."""
    unsupervised = types.SimpleNamespace()
    extensions = types.SimpleNamespace(unsupervised=unsupervised)
    monkeypatch.setitem(sys.modules, "tabpfn_extensions", extensions)

    tabpfn_backend._patch_explicit_categorical_feature_inference()

    assert unsupervised.infer_categorical_features([[1, 2], [1, 3]], categorical_features=[1]) == [
        1
    ]
    assert unsupervised.infer_categorical_features([[1, 2], [1, 3]], categorical_features=[]) == []


def test_explicit_type_patch_is_idempotent(monkeypatch):
    unsupervised = types.SimpleNamespace()
    extensions = types.SimpleNamespace(unsupervised=unsupervised)
    monkeypatch.setitem(sys.modules, "tabpfn_extensions", extensions)

    tabpfn_backend._patch_explicit_categorical_feature_inference()
    first = unsupervised.infer_categorical_features
    tabpfn_backend._patch_explicit_categorical_feature_inference()

    assert unsupervised.infer_categorical_features is first
