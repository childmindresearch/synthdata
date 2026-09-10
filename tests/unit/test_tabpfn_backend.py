"""Unit tests for TabPFN integration compatibility patches."""

import sys
import types

import numpy as np
import pandas as pd
import pytest
import torch

from synthdata.data import semantic_context_digest, semantic_context_payload
from synthdata.generation import hpo as hpo_module
from synthdata.generation import tabpfgen_backend, tabpfn_backend
from synthdata.generation.pipeline import run_generation

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


def test_tabpfgen_objective_uses_supplied_shared_evaluator(mocker):
    """TabPFGen objective must score release output through shared callback."""
    train = pd.DataFrame({"feature": [0.0, 1.0, 0.0, 1.0], "target": [0, 1, 0, 1]})
    evaluated = []

    class FakeGenerator:
        def __init__(self, **kwargs):
            del kwargs

        def generate_classification(self, **kwargs):
            evaluated.append(kwargs["X_train"].copy())
            return np.asarray([[0.25], [0.75], [0.25], [0.75]]), None

    class FakeClassifier:
        def fit(self, features, labels):
            del features, labels

        def predict(self, features):
            return np.zeros(len(features), dtype=int)

    class FakeTrial:
        number = 1

        def __init__(self):
            self.user_attrs = {}

        def suggest_int(self, name, low, high, step=1):
            del name, high, step
            return low

        def suggest_float(self, name, low, high, log=False):
            del name, high, log
            return low

        def set_user_attr(self, key, value):
            self.user_attrs[key] = value

    mocker.patch.object(tabpfgen_backend, "TabPFGen", FakeGenerator)
    mocker.patch("tabpfn.TabPFNClassifier", FakeClassifier)

    candidates = []

    def shared_eval(synthetic):
        candidates.append(synthetic.copy())
        return 0.125

    objective = tabpfgen_backend.build_tabpfgen_standard_objective(
        train,
        ["feature"],
        [],
        "target",
        len(train),
        500,
        shared_eval,
        target_is_categorical=True,
    )

    assert objective(FakeTrial()) == pytest.approx(0.125)
    assert len(evaluated) == 1
    pd.testing.assert_frame_equal(
        candidates[0].drop(columns="target"), pd.DataFrame({"feature": [0.25, 0.75, 0.25, 0.75]})
    )


def test_tabpfgen_custom_objective_uses_supplied_shared_evaluator(mocker):
    train = pd.DataFrame({"feature": [0.0, 1.0, 0.0, 1.0], "target": [0, 1, 0, 1]})
    candidates = []

    class FakeGenerator:
        def __init__(self, **kwargs):
            del kwargs

        def generate_classification(self, X_train, y_train, n_samples, balance_classes):
            del X_train, y_train, balance_classes
            return (
                np.tile([[0.25], [0.75]], (n_samples // 2, 1)),
                np.tile([0, 1], n_samples // 2),
            )

    class FakeTrial:
        number = 1

        def __init__(self):
            self.user_attrs = {}

        def suggest_int(self, name, low, high, step=1):
            del name, high, step
            return low

        def suggest_float(self, name, low, high, log=False):
            del name, high, log
            return low

        def set_user_attr(self, key, value):
            self.user_attrs[key] = value

    mocker.patch.object(tabpfgen_backend, "TabPFGenSGLDLabels", FakeGenerator)
    mocker.patch.object(tabpfgen_backend, "_record_hpo_generator_metadata")

    def shared_eval(synthetic):
        candidates.append(synthetic.copy())
        return 0.25

    objective = tabpfgen_backend.build_tabpfgen_custom_objective(
        train,
        ["feature"],
        [],
        "target",
        len(train),
        500,
        shared_eval,
        target_is_categorical=True,
    )

    assert objective(FakeTrial()) == pytest.approx(0.25)
    assert len(candidates) == 1
    assert list(candidates[0].columns) == ["feature", "target"]
    assert len(candidates[0]) == len(train)


def test_canonical_hpo_evaluator_is_train_fit_tuning_only_and_excludes_holdout(mocker):
    train = pd.DataFrame({"feature": [0.0, 1.0], "target": [0, 1]})
    tuning = pd.DataFrame({"feature": [2.0, 3.0], "target": [0, 1]})
    candidate = pd.DataFrame({"feature": [0.5, 2.5], "target": [0, 1]})
    captured = {}

    def fake_canonical(train_df, tuning_df, synthetic_df, **kwargs):
        captured.update(train=train_df, tuning=tuning_df, synthetic=synthetic_df, kwargs=kwargs)
        report = pd.DataFrame(
            {"mean": [0.8, 0.2, 0.4], "direction": ["maximize", "minimize", "minimize"]},
            index=["tstr_macro_f1.v1", "mixed_mmd.v1", "elastic_net_jsd.v1"],
        )
        report.attrs["canonical_hpo"] = True
        report.attrs["canonical_hpo_keys"] = tuple(report.index)
        report.attrs["hpo_provenance"] = {
            "fit_roles": ["train"],
            "comparison_role": "tuning",
            "release_transform_digest": "release-a",
            "role_hashes": {"train": "train-a", "tuning": "tuning-a"},
            "contracts": {
                "fit_roles": ["train"],
                "comparison_role": "tuning",
                "excluded_roles": ["final_holdout"],
                "privacy": False,
                "fairness": False,
            },
            "support_provenance": {
                "fit_roles": ["train"],
                "support_contract": "train_frozen_v1",
            },
            "bandwidth_provenance": {
                "fit_roles": ["train"],
                "comparison_role": "tuning",
                "contract": "train_frozen_v1",
            },
            "objective_version": "release-utility-v1",
        }
        return report

    mocker.patch.object(hpo_module, "evaluate_canonical_hpo_metrics", fake_canonical)
    evaluate = hpo_module.build_synthetic_eval_fn(
        train,
        tuning,
        "target",
        [],
        {"task12": ["tstr_macro_f1.v1", "mixed_mmd.v1", "elastic_net_jsd.v1"]},
        seed=7,
        feature_types={"feature": "continuous", "target": "categorical"},
    )

    assert evaluate(candidate) == pytest.approx(-((0.8 + 0.8 + 0.6) / 3))
    pd.testing.assert_frame_equal(captured["train"], train)
    pd.testing.assert_frame_equal(captured["tuning"], tuning)
    pd.testing.assert_frame_equal(captured["synthetic"], candidate)
    assert captured["kwargs"]["utility_policy"]["weights"] == pytest.approx([1 / 3] * 3)


@pytest.mark.parametrize("role", ["train", "tuning", "synthetic"])
def test_canonical_hpo_evaluator_rejects_final_holdout_metadata(role):
    frames = {
        name: pd.DataFrame({"feature": [0.0, 1.0], "target": [0, 1]})
        for name in ("train", "tuning", "synthetic")
    }
    frames[role].attrs["evaluation_role"] = "final_holdout"

    with pytest.raises(ValueError, match="final_holdout"):
        hpo_module.evaluate_canonical_hpo_metrics(
            frames["train"],
            frames["tuning"],
            frames["synthetic"],
            metric_config={"task12": ["tstr_macro_f1.v1"]},
            target_column="target",
        )


@pytest.mark.parametrize("metadata_key", ["nested_privacy_metadata", "nested_fairness_metadata"])
def test_canonical_hpo_evaluator_rejects_nested_privacy_or_fairness_metadata(metadata_key):
    frames = [pd.DataFrame({"feature": [0.0, 1.0], "target": [0, 1]}) for _ in range(3)]
    frames[1].attrs[metadata_key] = {"enabled": True, "details": {"value": "forbidden"}}

    with pytest.raises(ValueError, match="forbidden objective metadata"):
        hpo_module.evaluate_canonical_hpo_metrics(
            *frames,
            metric_config={"task12": ["tstr_macro_f1.v1"]},
            target_column="target",
        )


def test_tabpfn_pipeline_excludes_hpo_and_uses_train_only(
    make_config, make_canonical_dataset, mocker
):
    cfg = make_config()
    cfg.generation.synthcity.enabled = False
    cfg.generation.tabpfgen.enabled = False
    cfg.generation.tabpfn.enabled = True
    cfg.generation.tabpfn.variants = ["custom"]
    cfg.generation.tabpfn.data_variants = ["raw"]
    cfg.generation.hpo.enabled = True
    cfg.generation.n_samples = 4
    dataset = make_canonical_dataset("column")
    train = dataset.role_frame("train", imputed=False).reset_index(drop=True)
    generated = train.head(cfg.generation.n_samples).copy()
    generate = mocker.patch(
        "synthdata.generation.pipeline.tpfn.generate_tabpfn_custom",
        return_value=(generated, None),
    )
    run_study = mocker.patch("synthdata.generation.pipeline.hpo_mod.run_study")
    evaluate = mocker.patch("synthdata.generation.pipeline.hpo_mod.evaluate_canonical_hpo_metrics")

    result = run_generation(cfg, dataset)

    run_study.assert_not_called()
    evaluate.assert_not_called()
    generate.assert_called_once()
    pd.testing.assert_frame_equal(generate.call_args.args[0], train)
    pd.testing.assert_frame_equal(result["tabpfn_custom"], generated)
