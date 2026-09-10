"""Unit tests for the explicit scope of resumable HPO artifacts."""

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest

from synthdata.config import HPOConfig
from synthdata.data import semantic_context_payload
from synthdata.evaluation.catalog import emitted_keys_for_synthcity_metrics
from synthdata.generation.hpo import (
    HPO_OBJECTIVE_METRICS,
    HPO_TRIAL_CHECKPOINT_SCHEMA_VERSION,
    BestParamsCache,
    build_hpo_context,
    build_stage_a_contract,
    build_synthetic_eval_fn,
    contextual_study_name,
    default_best_params_path,
    default_storage_url,
    hpo_context_digest,
    hpo_score,
    load_hpo_trial_checkpoint,
    persist_hpo_trial_checkpoint,
    persist_stage_a_contract,
    run_study,
    screen_stage_a,
    screen_stage_a_trial,
    validate_hpo_metric_config,
)
from synthdata.generation.synthcity_backend import build_synthcity_objective, make_loader
from tests.unit.synthcity_emitted_key_fixtures import HPO_SYNTHCITY_EMITTED_KEY_FIXTURES

pytestmark = pytest.mark.unit


def _hpo_context(**overrides):
    context = build_hpo_context(
        task_type="classification",
        metric_config={"stats": ["wasserstein_dist"]},
        registry_digest="registry-a",
        stage_a_contract_digest="stage-a",
        group_context={"group_mode": "row"},
        role_context_fingerprint="roles-a",
        role_context={"roles": {"train": {"rows": 4}, "tuning": {"rows": 2}}},
    )
    context.update(overrides)
    return context


def test_default_hpo_objective_excludes_privacy_and_diagnostics():
    config = HPOConfig()

    assert "privacy" not in config.metric_config
    validate_hpo_metric_config(config.metric_config)


def test_patient_group_hpo_rejects_row_only_objectives():
    with pytest.raises(ValueError, match="patient_group HPO requires group-safe metrics"):
        build_hpo_context(
            task_type="classification",
            metric_config={"stats": ["wasserstein_dist"]},
            registry_digest="registry-a",
            stage_a_contract_digest="stage-a",
            group_context={"group_mode": "patient_group"},
            role_context_fingerprint="roles-a",
            role_context={"roles": {"train": {"rows": 4}, "tuning": {"rows": 2}}},
        )


def test_patient_group_hpo_accepts_group_safe_objective_contract():
    context = build_hpo_context(
        task_type="classification",
        metric_config={"performance": ["xgb"]},
        registry_digest="registry-a",
        stage_a_contract_digest="stage-a",
        group_context={"group_mode": "patient_group"},
        role_context_fingerprint="roles-a",
        role_context={"roles": {"train": {"rows": 4}, "tuning": {"rows": 2}}},
    )

    assert "performance.xgb.syn_id" in context["expected_emitted_keys"]


def test_hpo_context_uses_contextual_emitted_key_manifest():
    context = build_hpo_context(
        task_type="classification",
        metric_config={"performance": ["xgb"]},
        registry_digest="registry-a",
        stage_a_contract_digest="stage-a",
        group_context={"group_mode": "row"},
        role_context_fingerprint="roles-a",
        role_context={"roles": {"train": {"rows": 4}, "tuning": {"rows": 2}}},
        variable_columns=["feature", "target"],
        attack_target_types={"protected": "categorical"},
    )

    assert context["expected_emitted_keys"] == emitted_keys_for_synthcity_metrics(
        {"performance": ["xgb"]},
        variable_columns=["feature", "target"],
        attack_target_types={"protected": "categorical"},
    )


def test_hpo_fixture_covers_every_operational_objective_metric():
    fixture_metric_names = {
        fixture_name
        for fixture_name, _metric_config, _expected_keys in HPO_SYNTHCITY_EMITTED_KEY_FIXTURES
        if fixture_name != "default"
    }

    assert fixture_metric_names == HPO_OBJECTIVE_METRICS


@pytest.mark.parametrize(
    ("fixture_name", "metric_config", "expected_keys"),
    HPO_SYNTHCITY_EMITTED_KEY_FIXTURES,
    ids=[
        fixture_name
        for fixture_name, _metric_config, _expected_keys in HPO_SYNTHCITY_EMITTED_KEY_FIXTURES
    ],
)
def test_hpo_context_matches_literal_emitted_key_fixture(
    fixture_name, metric_config, expected_keys
):
    del fixture_name
    context = build_hpo_context(
        task_type="classification",
        metric_config=metric_config,
        registry_digest="registry-a",
        stage_a_contract_digest="stage-a",
        group_context={"group_mode": "row"},
        role_context_fingerprint="roles-a",
        role_context={"roles": {"train": {"rows": 4}, "tuning": {"rows": 2}}},
        variable_columns=["feature", "target"],
        attack_target_types={"protected": "categorical"},
    )

    assert context["expected_emitted_keys"] == list(expected_keys)


def test_patient_group_tabpfgen_objective_requires_group_ids():
    with pytest.raises(ValueError, match="require group IDs for both train and tuning loaders"):
        build_synthetic_eval_fn(
            pd.DataFrame({"feature": [0, 1], "target": [0, 1]}),
            pd.DataFrame({"feature": [2, 3], "target": [0, 1]}),
            target_column="target",
            sensitive_features=[],
            metric_config={"performance": ["xgb"]},
            seed=0,
            group_context={"group_mode": "patient_group"},
        )


def test_patient_group_tabpfgen_objective_groups_all_generated_populations(mocker):
    train = pd.DataFrame({"feature": [0, 1, 2, 3], "target": [0, 1, 0, 1]})
    tuning = pd.DataFrame({"feature": [4, 5, 6, 7], "target": [0, 1, 0, 1]})
    candidate = pd.DataFrame({"feature": [8, 9], "target": [0, 1]})
    captured = {}

    def fake_evaluate(*args, **kwargs):
        captured["loaders"] = args
        captured["kwargs"] = kwargs
        return pd.DataFrame(
            {
                "mean": [0.5, 0.5, 0.5],
                "direction": ["maximize", "maximize", "maximize"],
            },
            index=[
                "performance.xgb.gt",
                "performance.xgb.syn_id",
                "performance.xgb.syn_ood",
            ],
        )

    mocker.patch("synthcity.metrics.Metrics.evaluate", staticmethod(fake_evaluate))
    eval_fn = build_synthetic_eval_fn(
        train,
        tuning,
        target_column="target",
        sensitive_features=[],
        metric_config={"performance": ["xgb"]},
        seed=11,
        group_context={"group_mode": "patient_group"},
        quasi_identifier_columns=["feature"],
        sensitive_target_types={"target": "categorical"},
        semantic_context={
            "schema_version": "semantic-context-v1",
            "task_type": "classification",
        },
        train_group_ids=["train-a", "train-a", "train-b", "train-b"],
        holdout_group_ids=["tuning-a", "tuning-a", "tuning-b", "tuning-b"],
    )

    assert eval_fn(candidate) == pytest.approx(-0.5)
    loader_groups = [loader.group_ids.tolist() for loader in captured["loaders"]]
    assert loader_groups[0] == ["tuning-a", "tuning-a", "tuning-b", "tuning-b"]
    assert loader_groups[1] == [("hpo.synthetic", 0), ("hpo.synthetic", 1)]
    assert loader_groups[2] == ["train-a", "train-a", "train-b", "train-b"]
    assert loader_groups[3] == [
        ("hpo.reference_synthetic", 0),
        ("hpo.reference_synthetic", 1),
    ]
    assert loader_groups[4][:4] == ["train-a", "train-a", "train-b", "train-b"]
    assert loader_groups[4][4:] == [("hpo.synthetic", 0), ("hpo.synthetic", 1)]
    assert captured["kwargs"]["X_syn_group_ids"] == [
        ("hpo.synthetic", 0),
        ("hpo.synthetic", 1),
    ]
    assert captured["kwargs"]["X_ref_syn_group_ids"] == [
        ("hpo.reference_synthetic", 0),
        ("hpo.reference_synthetic", 1),
    ]
    assert captured["kwargs"]["quasi_identifier_columns"] == ["feature"]
    assert captured["kwargs"]["sensitive_target_types"] == {"target": "categorical"}
    assert captured["kwargs"]["semantic_context"] == {
        "schema_version": "semantic-context-v1",
        "task_type": "classification",
    }


def test_patient_group_synthcity_xgb_objective_uses_grouped_real_loaders(mocker):
    train = pd.DataFrame({"feature": [0, 1, 2, 3], "target": [0, 1, 0, 1]})
    tuning = pd.DataFrame({"feature": [4, 5, 6, 7], "target": [0, 1, 0, 1]})
    train_loader = make_loader(
        train,
        "target",
        [],
        group_ids=["train-a", "train-a", "train-b", "train-b"],
    )
    tuning_loader = make_loader(
        tuning,
        "target",
        [],
        group_ids=["tuning-a", "tuning-a", "tuning-b", "tuning-b"],
    )
    hpo_config = HPOConfig(metric_config={"performance": ["xgb"]})
    captured = {}

    class FakePlugin:
        @staticmethod
        def sample_hyperparameters_optuna(trial):
            return {}

    def fake_evaluate(tests, fit_loader, **kwargs):
        captured["tests"] = tests
        captured["fit_loader"] = fit_loader
        captured.update(kwargs)
        report = pd.DataFrame(
            {
                "mean": [0.7, 0.8, 0.9],
                "direction": ["maximize", "maximize", "maximize"],
            },
            index=[
                "performance.xgb.gt",
                "performance.xgb.syn_id",
                "performance.xgb.syn_ood",
            ],
        )
        report.attrs["metric_metadata"] = {
            "generator.fake": {"accounting": {"effective_epsilon": 1.0}}
        }
        return {tests[0][0]: report}

    mocker.patch("synthdata.generation.synthcity_backend.get_plugin_class", return_value=FakePlugin)
    mocker.patch("synthdata.generation.synthcity_backend.plugin_accepts", return_value=False)
    mocker.patch("synthcity.benchmark.Benchmarks.evaluate", side_effect=fake_evaluate)

    objective = build_synthcity_objective(
        "xgb",
        train_loader,
        hpo_config,
        seed=0,
        tuning_loader=tuning_loader,
        task_type="classification",
        classification_score="macro_f1",
        semantic_context={
            "schema_version": "semantic-context-v1",
            "task_type": "classification",
        },
        synthetic_size=4,
        group_context={"group_mode": "patient_group"},
    )
    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    score = objective(trial)

    assert score == pytest.approx(-0.85)
    assert trial.user_attrs["metric_metadata"]["generator.fake"]["accounting"] == {
        "effective_epsilon": 1.0
    }
    assert captured["fit_loader"] is train_loader
    assert captured["X_test"] is tuning_loader
    assert captured["fit_on_X"] is True
    assert captured["classification_score"] == "macro_f1"
    assert captured["semantic_context"] == {
        "schema_version": "semantic-context-v1",
        "task_type": "classification",
    }
    assert captured["fit_loader"].group_ids.tolist() == [
        "train-a",
        "train-a",
        "train-b",
        "train-b",
    ]
    assert captured["X_test"].group_ids.tolist() == [
        "tuning-a",
        "tuning-a",
        "tuning-b",
        "tuning-b",
    ]


def test_patient_group_synthcity_objective_reaches_native_group_contract(
    mocker, monkeypatch, tmp_path
):
    from synthcity.metrics import Metrics
    from synthcity.plugins import Plugins

    train = pd.DataFrame({"feature": [0, 1, 2, 3], "target": [0, 1, 0, 1]})
    tuning = pd.DataFrame({"feature": [4, 5], "target": [0, 1]})
    train_loader = make_loader(
        train,
        "target",
        [],
        group_ids=["train-a", "train-a", "train-b", "train-b"],
    )
    tuning_loader = make_loader(
        tuning,
        "target",
        [],
        group_ids=["tuning-a", "tuning-b"],
    )
    hpo_config = HPOConfig(metric_config={"performance": ["xgb"]})
    generated_lengths = {"synthetic": 3, "reference_synthetic": 2}
    captured = {"namespaces": [], "loaders": None, "kwargs": None}

    class FakePlugin:
        @staticmethod
        def sample_hyperparameters_optuna(trial):
            del trial
            return {}

    class FakeGenerator:
        def fit(self, _fit_loader):
            return self

        def generate(self, count=None, **kwargs):
            del count
            namespace = kwargs["_group_namespace"]
            captured["namespaces"].append(namespace)
            size = generated_lengths[namespace]
            frame = pd.DataFrame(
                {
                    "feature": np.arange(size),
                    "target": [index % 2 for index in range(size)],
                }
            )
            return make_loader(
                frame,
                "target",
                [],
                group_ids=[f"__synthcity_generated__{namespace}__{index}" for index in range(size)],
            )

    def fake_get(_self, _name, **_kwargs):
        return FakeGenerator()

    def fake_metrics_evaluate(*args, **kwargs):
        captured["loaders"] = args
        captured["kwargs"] = kwargs
        return pd.DataFrame(
            {
                "mean": [0.6, 0.7, 0.8],
                "errors": [0.0, 0.0, 0.0],
                "durations": [0.0, 0.0, 0.0],
                "direction": ["maximize", "maximize", "maximize"],
            },
            index=[
                "performance.xgb.gt",
                "performance.xgb.syn_id",
                "performance.xgb.syn_ood",
            ],
        )

    mocker.patch(
        "synthdata.generation.synthcity_backend.get_plugin_class",
        return_value=FakePlugin,
    )
    mocker.patch(
        "synthdata.generation.synthcity_backend.plugin_accepts",
        return_value=False,
    )
    monkeypatch.setattr(Plugins, "get", fake_get)
    monkeypatch.setattr(Metrics, "evaluate", staticmethod(fake_metrics_evaluate))

    objective = build_synthcity_objective(
        "fake",
        train_loader,
        hpo_config,
        seed=0,
        tuning_loader=tuning_loader,
        task_type="classification",
        synthetic_size=3,
        group_context={"group_mode": "patient_group"},
    )
    study = optuna.create_study(direction="minimize")
    score = objective(study.ask())

    assert score == pytest.approx(-0.75)
    assert captured["namespaces"] == ["synthetic", "reference_synthetic"]
    assert [len(loader) for loader in captured["loaders"][:4]] == [2, 3, 4, 2]
    assert captured["loaders"][0] is tuning_loader
    assert captured["loaders"][2] is train_loader
    assert all(
        group_id.startswith("__synthcity_generated__synthetic__")
        for group_id in captured["loaders"][1].group_ids
    )
    assert all(
        group_id.startswith("__synthcity_generated__reference_synthetic__")
        for group_id in captured["loaders"][3].group_ids
    )
    assert captured["kwargs"]["group_mode"] == "patient_group"


def test_hpo_context_digest_and_study_name_change_with_objective_identity():
    context = _hpo_context()
    changed = {**context, "task_type": "regression"}
    balanced_context = {
        **context,
        "semantic_context": {"classification_score": "balanced_accuracy"},
    }
    macro_context = {
        **context,
        "semantic_context": {"classification_score": "macro_f1"},
    }

    assert hpo_context_digest(context) != hpo_context_digest(changed)
    assert contextual_study_name("hpo_model", context) != contextual_study_name(
        "hpo_model", changed
    )
    assert hpo_context_digest(balanced_context) != hpo_context_digest(macro_context)


def test_best_params_cache_rejects_mismatched_hpo_context(tmp_path):
    path = tmp_path / "hpo_best_params.json"
    context = _hpo_context()
    BestParamsCache(path, hpo_context=context).set("synthcity", "ctgan", {"n_iter": 3})

    assert BestParamsCache(path, hpo_context=context).has("synthcity", "ctgan")
    assert not BestParamsCache(
        path,
        hpo_context={**context, "registry_digest": "registry-b"},
    ).has("synthcity", "ctgan")


def test_contextual_hpo_studies_are_separate_and_resumeable(tmp_path):
    config = HPOConfig(n_trials=1, timeout_seconds=None)
    first_context = _hpo_context()
    second_context = {**first_context, "stage_a_contract_digest": "stage-b"}

    run_study(
        "hpo_model",
        lambda trial: 1.0,
        config,
        tmp_path,
        seed=0,
        drop_keys=(),
        hpo_context=first_context,
    )
    run_study(
        "hpo_model",
        lambda trial: 2.0,
        config,
        tmp_path,
        seed=0,
        drop_keys=(),
        hpo_context=second_context,
    )

    storage = default_storage_url(tmp_path)
    first_study = optuna.load_study(
        study_name=contextual_study_name("hpo_model", first_context),
        storage=storage,
    )
    second_study = optuna.load_study(
        study_name=contextual_study_name("hpo_model", second_context),
        storage=storage,
    )
    assert len(first_study.trials) == 1
    assert len(second_study.trials) == 1
    assert first_study.user_attrs["hpo_context_digest"] == hpo_context_digest(first_context)
    assert second_study.user_attrs["hpo_context_digest"] == hpo_context_digest(second_context)


def _stage_a_source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [0.0, 10.0, 0.0, 10.0],
            "group": ["a", "a", "b", "b"],
            "target": [0, 1, 0, 1],
            "derived": ["a0", "a1", "b0", "b1"],
        }
    )


def _stage_a_contract(source: pd.DataFrame):
    return build_stage_a_contract(
        source,
        expected_n_samples=4,
        target_column="target",
        categorical_columns=["group", "derived"],
        protected_columns=["group"],
        dependency_rules=({"child": "derived", "parents": ["group", "target"]},),
    )


def test_stage_a_screen_passes_without_running_attack_metrics():
    source = _stage_a_source()
    candidate = pd.DataFrame(
        {
            "age": [1.0, 9.0, 1.0, 9.0],
            "group": ["a", "a", "b", "b"],
            "target": [0, 1, 0, 1],
            "derived": ["a0", "a1", "b0", "b1"],
        }
    )

    result = screen_stage_a(candidate, _stage_a_contract(source), source)

    assert result.passed
    assert not result.prune_reasons
    assert {check["screen"] for check in result.checks} == {
        "shape_schema",
        "bounds_categories",
        "support",
        "dependencies",
        "exact_reuse",
        "subgroup_collapse",
    }


@pytest.mark.parametrize(
    ("mutate", "screen"),
    [
        (lambda frame: frame.drop(columns="derived"), "shape_schema"),
        (lambda frame: frame.assign(age=[0.0, 11.0, 1.0, 9.0]), "bounds_categories"),
        (lambda frame: frame.assign(group=["a", "a", "b", "unknown"]), "bounds_categories"),
        (lambda frame: frame.assign(group=["a", "a", "a", "a"]), "subgroup_collapse"),
        (lambda frame: frame.assign(derived=["a0", "wrong", "b0", "b1"]), "dependencies"),
    ],
)
def test_stage_a_screen_prunes_invalid_candidates(mutate, screen):
    source = _stage_a_source()
    candidate = mutate(
        pd.DataFrame(
            {
                "age": [1.0, 9.0, 1.0, 9.0],
                "group": ["a", "a", "b", "b"],
                "target": [0, 1, 0, 1],
                "derived": ["a0", "a1", "b0", "b1"],
            }
        )
    )

    result = screen_stage_a(candidate, _stage_a_contract(source), source)

    assert result.pruned
    assert any(check["screen"] == screen and not check["passed"] for check in result.checks)


def test_stage_a_screen_enforces_zero_exact_reuse_and_target_support():
    source = _stage_a_source()
    contract = _stage_a_contract(source)

    exact_result = screen_stage_a(source.copy(), contract, source)
    missing_class = source.iloc[[0, 2, 0, 2]].assign(age=[1.0, 9.0, 1.0, 9.0])
    support_result = screen_stage_a(missing_class, contract, source)

    assert any("exact reuse" in reason for reason in exact_result.prune_reasons)
    assert any("target support" in reason for reason in support_result.prune_reasons)


def test_stage_a_regression_target_uses_numeric_bounds():
    source = pd.DataFrame({"feature": [0.0, 1.0], "target": [10.0, 20.0]})
    contract = build_stage_a_contract(
        source,
        expected_n_samples=2,
        target_column="target",
        target_is_categorical=False,
    )
    candidate = pd.DataFrame({"feature": [0.25, 0.75], "target": [12.0, 18.0]})

    result = screen_stage_a(candidate, contract, source)

    assert result.passed


def test_stage_a_screen_rejects_source_frame_mismatch():
    source = _stage_a_source()
    contract = _stage_a_contract(source)
    changed_source = source.copy()
    changed_source.loc[0, "age"] = 2.0

    with pytest.raises(ValueError, match="source frame does not match"):
        screen_stage_a(source, contract, changed_source)


def test_stage_a_contract_persists_semantic_context():
    source = _stage_a_source()
    contract = build_stage_a_contract(
        source,
        expected_n_samples=4,
        target_column="target",
        categorical_columns=["group", "derived"],
        protected_columns=["group"],
        registry_digest="registry-a",
        role_context_fingerprint="roles-a",
        role_context={"roles": {"train": {"rows": 4}}},
        group_context={"group_mode": "row"},
        hpo_context={"metric_config": {"stats": ["wasserstein_dist"]}},
    )

    payload = contract.to_dict()

    assert payload["registry_digest"] == "registry-a"
    assert payload["role_context_fingerprint"] == "roles-a"
    assert payload["role_context"] == {"roles": {"train": {"rows": 4}}}
    assert payload["group_context"] == {"group_mode": "row"}
    assert payload["hpo_context"] == {"metric_config": {"stats": ["wasserstein_dist"]}}


def test_stage_a_trial_prune_and_persist_result(tmp_path):
    source = _stage_a_source()
    contract = _stage_a_contract(source)
    contract_path = persist_stage_a_contract(tmp_path, contract)
    study = optuna.create_study(direction="minimize")

    study.optimize(
        lambda trial: screen_stage_a_trial(
            trial,
            source.copy(),
            contract,
            source,
            tmp_path,
            "hpo_test",
        ),
        n_trials=1,
    )

    trial = study.trials[0]
    result_path = tmp_path / "hpo_test" / "trial-0" / "result.json"
    assert trial.state == optuna.trial.TrialState.PRUNED
    assert trial.user_attrs["stage_a_state"] == "pruned"
    assert result_path.exists()
    assert contract_path.exists()
    assert "exact_reuse" in result_path.read_text()


def test_stage_a_trial_persists_screen_exception_as_pruned(tmp_path):
    source = _stage_a_source()
    contract = _stage_a_contract(source)
    changed_source = source.copy()
    changed_source.loc[0, "age"] = 2.0
    study = optuna.create_study(direction="minimize")

    study.optimize(
        lambda trial: screen_stage_a_trial(
            trial,
            source.copy(),
            contract,
            changed_source,
            tmp_path,
            "hpo_exception",
        ),
        n_trials=1,
    )

    trial = study.trials[0]
    result_path = tmp_path / "hpo_exception" / "trial-0" / "result.json"
    payload = json.loads(result_path.read_text())

    assert trial.state == optuna.trial.TrialState.PRUNED
    assert trial.user_attrs["stage_a_state"] == "pruned"
    assert payload["state"] == "pruned"
    assert payload["checks"][0]["screen"] == "stage_a_exception"
    assert payload["checks"][0]["observed"]["exception_type"] == "ValueError"
    assert "source frame does not match" in payload["checks"][0]["observed"]["exception_message"]
    assert "source frame does not match" in payload["prune_reasons"][0]


def test_stage_a_setup_failure_is_persisted_before_trial_construction(tmp_path, mocker):
    source = _stage_a_source()
    contract = _stage_a_contract(source)
    mocker.patch(
        "synthdata.generation.tabpfgen_backend.label_encode_non_numeric_columns",
        side_effect=ValueError("categorical encoder failed"),
    )

    from synthdata.generation import tabpfgen_backend

    with pytest.raises(ValueError, match="categorical encoder failed"):
        tabpfgen_backend.build_tabpfgen_standard_objective(
            source,
            ["age", "group", "derived"],
            ["group", "derived"],
            "target",
            len(source),
            500,
            lambda _synthetic: 0.5,
            stage_a_contract=contract,
            stage_a_source_df=source,
            stage_a_root=tmp_path,
            study_name="hpo_setup_failure",
            target_is_categorical=True,
        )

    result_path = tmp_path / "hpo_setup_failure" / "construction-failure.json"
    payload = json.loads(result_path.read_text())
    assert payload["trial_number"] is None
    assert payload["checks"][0]["screen"] == "stage_a_exception"
    assert payload["checks"][0]["observed"]["exception_type"] == "ValueError"
    assert "categorical encoder failed" in payload["prune_reasons"][0]


def test_resumed_study_counts_pruned_trials_without_rerunning_them(tmp_path):
    calls = []

    def objective(_trial):
        calls.append(True)
        raise optuna.TrialPruned("screen failed")

    config = HPOConfig(n_trials=1, timeout_seconds=None)
    with pytest.raises(RuntimeError, match="produced no completed trials"):
        run_study("hpo_restart", objective, config, tmp_path, seed=0, drop_keys=())
    with pytest.raises(RuntimeError, match="produced no completed trials"):
        run_study("hpo_restart", objective, config, tmp_path, seed=0, drop_keys=())

    assert len(calls) == 1
    study = optuna.load_study(
        study_name="hpo_restart",
        storage=default_storage_url(tmp_path),
    )
    assert len(study.trials) == 1
    assert study.trials[0].state == optuna.trial.TrialState.PRUNED


def test_completed_hpo_checkpoint_is_durable_and_resume_skips_terminal_trial(tmp_path):
    calls = []

    def objective(_trial):
        calls.append(True)
        return 0.25

    config = HPOConfig(n_trials=1, timeout_seconds=None)
    run_study("hpo_checkpoint", objective, config, tmp_path, seed=0, drop_keys=())

    checkpoint_path = (
        tmp_path / "hpo_checkpoints" / "hpo_checkpoint" / "trial-0" / "checkpoint.json"
    )
    checkpoint = load_hpo_trial_checkpoint(checkpoint_path)
    assert checkpoint["schema_version"] == HPO_TRIAL_CHECKPOINT_SCHEMA_VERSION
    assert checkpoint["state"] == "complete"
    assert checkpoint["objective_value"] == pytest.approx(0.25)

    def should_not_run(_trial):
        pytest.fail("a completed HPO trial was recomputed during resume")

    run_study("hpo_checkpoint", should_not_run, config, tmp_path, seed=0, drop_keys=())

    assert calls == [True]
    assert load_hpo_trial_checkpoint(checkpoint_path) == checkpoint


def test_hpo_checkpoint_resume_rejects_changed_implementation_fingerprint(tmp_path):
    calls = []

    def objective(trial):
        calls.append(True)
        trial.set_user_attr("generator_plugin_name", "test_generator")
        trial.set_user_attr("generator_privacy_claim_type", "none")
        trial.set_user_attr("generator_metadata_state", "not_attempted")
        trial.set_user_attr("generator_implementation_fingerprint", "implementation-a")
        return 0.25

    config = HPOConfig(n_trials=1, timeout_seconds=None)
    run_study(
        "hpo_fingerprint_checkpoint",
        objective,
        config,
        tmp_path,
        seed=0,
        drop_keys=(),
        checkpoint_implementation_fingerprint="implementation-a",
    )

    checkpoint_path = (
        tmp_path / "hpo_checkpoints" / "hpo_fingerprint_checkpoint" / "trial-0" / "checkpoint.json"
    )
    checkpoint = load_hpo_trial_checkpoint(
        checkpoint_path,
        expected_implementation_fingerprint="implementation-a",
    )
    assert checkpoint["metadata"]["generator"]["implementation_fingerprint"] == "implementation-a"

    with pytest.raises(RuntimeError, match="does not match the current implementation"):
        load_hpo_trial_checkpoint(
            checkpoint_path,
            expected_implementation_fingerprint="implementation-b",
        )

    def should_not_run(_trial):
        pytest.fail("a checkpoint from another implementation was reused")

    with pytest.raises(RuntimeError, match="does not match the current implementation"):
        run_study(
            "hpo_fingerprint_checkpoint",
            should_not_run,
            config,
            tmp_path,
            seed=0,
            drop_keys=(),
            checkpoint_implementation_fingerprint="implementation-b",
        )
    assert calls == [True]


def test_legacy_hpo_generator_metadata_is_readable_but_not_current_evidence(tmp_path):
    legacy_metadata = {
        "schema_version": "generator-metadata-v1",
        "generator_context": {"privacy_claim_type": "none"},
        "plugin_name": "test_generator",
        "plugin_fqdn": "test.generator",
        "requested_parameters": {},
        "n_samples": 1,
        "random_state": 0,
        "privacy_accounting": None,
    }
    study = optuna.create_study(direction="minimize")

    def objective(trial):
        trial.set_user_attr("generator_plugin_name", "test_generator")
        trial.set_user_attr("generator_privacy_claim_type", "none")
        trial.set_user_attr("generator_metadata_state", "present")
        trial.set_user_attr("generator_metadata", legacy_metadata)
        return 0.25

    study.optimize(objective, n_trials=1)
    checkpoint_path = persist_hpo_trial_checkpoint(
        tmp_path / "legacy_checkpoints",
        "legacy_hpo",
        study.trials[0],
    )

    loaded = load_hpo_trial_checkpoint(checkpoint_path)
    assert loaded["metadata"]["generator"]["metadata"]["schema_version"] == (
        "generator-metadata-v1"
    )
    with pytest.raises(RuntimeError, match="does not match the current implementation"):
        load_hpo_trial_checkpoint(
            checkpoint_path,
            expected_implementation_fingerprint="implementation-a",
        )


def test_failed_hpo_checkpoint_persists_exception_context(tmp_path):
    def objective(_trial):
        raise RuntimeError("generator fit failed")

    config = HPOConfig(n_trials=1, timeout_seconds=None)
    with pytest.raises(RuntimeError, match="generator fit failed"):
        run_study("hpo_failed_checkpoint", objective, config, tmp_path, seed=0, drop_keys=())

    checkpoint = load_hpo_trial_checkpoint(
        tmp_path / "hpo_checkpoints" / "hpo_failed_checkpoint" / "trial-0" / "checkpoint.json"
    )
    assert checkpoint["state"] == "failed"
    assert checkpoint["metadata"]["outcome"] == {
        "error_type": "RuntimeError",
        "error_message": "generator fit failed",
        "state": "failed",
    }


def test_pategan_hpo_checkpoint_persists_accounting_and_resumes(mocker, tmp_path):
    params = {
        "epsilon": 1.0,
        "delta": 1e-6,
        "alpha": 100,
        "lamda": 0.1,
    }
    accounting = {
        "schema_version": "pate-accounting-v1",
        "privacy_claim_type": "formal_dp",
        "accountant": "pate_moments_v1",
        "requested_epsilon": 1.0,
        "requested_delta": 1e-6,
        "requested_alpha": 100,
        "requested_lamda": 0.1,
        "resolved_epsilon": 1.0,
        "resolved_delta": 1e-6,
        "resolved_alpha": 100,
        "resolved_lamda": 0.1,
        "effective_epsilon": 1.2,
        "effective_delta": 1e-6,
        "effective_alpha": 100,
        "effective_lamda": 0.1,
        "iterations": 2,
        "max_iter": 10,
        "stopping_state": "epsilon_reached",
    }

    class FakePATEGAN:
        def __init__(
            self,
            epsilon=1.0,
            delta=None,
            alpha=100,
            lamda=0.001,
        ):
            del epsilon, delta, alpha, lamda

        @staticmethod
        def sample_hyperparameters_optuna(_trial):
            return dict(params)

    def fake_evaluate(tests, _fit_loader, **_kwargs):
        report = pd.DataFrame(
            {"mean": [0.4], "direction": ["minimize"]},
            index=["stats.wasserstein_dist.joint"],
        )
        report.attrs["metric_metadata"] = {
            "generator.pategan": {
                "schema_version": "generator-runtime-metadata-v2",
                "plugin_name": "pategan",
                "plugin_fqdn": "privacy.pategan",
                "requested_parameters": {**params, "random_state": 0},
                "n_samples": 4,
                "random_state": 0,
                "privacy_claim_type": "formal_dp",
                "accounting": accounting,
            }
        }
        return {tests[0][0]: report}

    mocker.patch(
        "synthdata.generation.synthcity_backend.get_plugin_class", return_value=FakePATEGAN
    )
    mocker.patch("synthdata.generation.synthcity_backend.plugin_accepts", return_value=False)
    mocker.patch("synthcity.benchmark.Benchmarks.evaluate", side_effect=fake_evaluate)
    objective = build_synthcity_objective(
        "pategan",
        object(),
        HPOConfig(metric_config={"stats": ["wasserstein_dist"]}, n_trials=1),
        seed=0,
        tuning_loader=object(),
        synthetic_size=4,
    )

    run_study(
        "hpo_pategan_checkpoint",
        objective,
        HPOConfig(metric_config={"stats": ["wasserstein_dist"]}, n_trials=1),
        tmp_path,
        seed=0,
        drop_keys=(),
    )

    checkpoint_path = (
        tmp_path / "hpo_checkpoints" / "hpo_pategan_checkpoint" / "trial-0" / "checkpoint.json"
    )
    checkpoint = load_hpo_trial_checkpoint(checkpoint_path)
    generator = checkpoint["metadata"]["generator"]
    assert generator["state"] == "present"
    assert generator["privacy_claim_type"] == "formal_dp"
    assert generator["implementation_fingerprint"]
    assert generator["metadata"]["schema_version"] == "generator-metadata-v2"
    assert (
        generator["metadata"]["implementation_fingerprint"]
        == generator["implementation_fingerprint"]
    )
    assert generator["metadata"]["privacy_accounting"] == accounting
    assert checkpoint["metadata"]["result_metadata"] == checkpoint["metadata"]["metric_metadata"]

    def should_not_run(_trial):
        pytest.fail("a completed PATE trial was recomputed during resume")

    run_study(
        "hpo_pategan_checkpoint",
        should_not_run,
        HPOConfig(metric_config={"stats": ["wasserstein_dist"]}, n_trials=1),
        tmp_path,
        seed=0,
        drop_keys=(),
    )
    assert load_hpo_trial_checkpoint(checkpoint_path) == checkpoint


def test_pategan_hpo_missing_accounting_fails_closed_with_checkpoint(mocker, tmp_path):
    params = {"epsilon": 1.0, "delta": 1e-6, "alpha": 100, "lamda": 0.1}

    class FakePATEGAN:
        def __init__(self, epsilon=1.0, delta=None, alpha=100, lamda=0.001):
            del epsilon, delta, alpha, lamda

        @staticmethod
        def sample_hyperparameters_optuna(_trial):
            return dict(params)

    def fake_evaluate(tests, _fit_loader, **_kwargs):
        report = pd.DataFrame(
            {"mean": [0.4], "direction": ["minimize"]},
            index=["stats.wasserstein_dist.joint"],
        )
        return {tests[0][0]: report}

    mocker.patch(
        "synthdata.generation.synthcity_backend.get_plugin_class", return_value=FakePATEGAN
    )
    mocker.patch("synthdata.generation.synthcity_backend.plugin_accepts", return_value=False)
    mocker.patch("synthcity.benchmark.Benchmarks.evaluate", side_effect=fake_evaluate)
    objective = build_synthcity_objective(
        "pategan",
        object(),
        HPOConfig(metric_config={"stats": ["wasserstein_dist"]}, n_trials=1),
        seed=0,
        tuning_loader=object(),
        synthetic_size=4,
    )

    with pytest.raises(RuntimeError, match="produced no completed trials"):
        run_study(
            "hpo_pategan_missing_accounting",
            objective,
            HPOConfig(metric_config={"stats": ["wasserstein_dist"]}, n_trials=1),
            tmp_path,
            seed=0,
            drop_keys=(),
        )

    checkpoint = load_hpo_trial_checkpoint(
        tmp_path
        / "hpo_checkpoints"
        / "hpo_pategan_missing_accounting"
        / "trial-0"
        / "checkpoint.json"
    )
    assert checkpoint["state"] == "pruned"
    assert checkpoint["metadata"]["generator"]["state"] == "missing"
    assert checkpoint["metadata"]["generator"]["privacy_claim_type"] == "formal_dp"
    assert checkpoint["metadata"]["outcome"]["state"] == "pruned"


def test_pategan_hpo_incomplete_runtime_metadata_fails_closed(mocker, tmp_path):
    params = {"epsilon": 1.0, "delta": 1e-6, "alpha": 100, "lamda": 0.1}

    class FakePATEGAN:
        def __init__(
            self,
            epsilon=1.0,
            delta=None,
            alpha=100,
            lamda=0.001,
        ):
            del epsilon, delta, alpha, lamda

        @staticmethod
        def sample_hyperparameters_optuna(_trial):
            return dict(params)

    def fake_evaluate(tests, _fit_loader, **_kwargs):
        report = pd.DataFrame(
            {"mean": [0.4], "direction": ["minimize"]},
            index=["stats.wasserstein_dist.joint"],
        )
        report.attrs["metric_metadata"] = {
            "generator.pategan": {
                "schema_version": "generator-runtime-metadata-v2",
                "plugin_name": "pategan",
                "plugin_fqdn": "privacy.pategan",
                "requested_parameters": dict(params),
                "privacy_claim_type": "formal_dp",
                "accounting": {"effective_epsilon": 1.0},
            }
        }
        return {tests[0][0]: report}

    mocker.patch(
        "synthdata.generation.synthcity_backend.get_plugin_class", return_value=FakePATEGAN
    )
    mocker.patch("synthdata.generation.synthcity_backend.plugin_accepts", return_value=False)
    mocker.patch("synthcity.benchmark.Benchmarks.evaluate", side_effect=fake_evaluate)
    objective = build_synthcity_objective(
        "pategan",
        object(),
        HPOConfig(metric_config={"stats": ["wasserstein_dist"]}, n_trials=1),
        seed=0,
        tuning_loader=object(),
        synthetic_size=4,
    )

    with pytest.raises(optuna.TrialPruned, match="incomplete PATE metadata"):
        objective(optuna.create_study(direction="minimize").ask())


def test_hpo_metric_config_rejects_calibrating_privacy_metric():
    with pytest.raises(ValueError, match="not approved operational objectives"):
        validate_hpo_metric_config({"privacy": ["identifiability_score"]})


def test_hpo_builders_reject_unsafe_config_before_backend_execution():
    invalid_config = {"privacy": ["identifiability_score"]}

    with pytest.raises(ValueError, match="not approved operational objectives"):
        build_synthetic_eval_fn(
            pd.DataFrame(),
            pd.DataFrame(),
            target_column="target",
            sensitive_features=[],
            metric_config=invalid_config,
            seed=0,
        )
    with pytest.raises(ValueError, match="not approved operational objectives"):
        build_synthcity_objective(
            "tvae",
            train_loader=None,
            hpo_cfg=HPOConfig(metric_config=invalid_config),
            seed=0,
        )


def test_synthcity_candidate_construction_failure_persists_stage_a_result(mocker, tmp_path):
    source = pd.DataFrame({"feature": [0.0, 1.0], "target": [0, 1]})
    contract = build_stage_a_contract(
        source,
        expected_n_samples=2,
        target_column="target",
    )

    class FakePlugin:
        @staticmethod
        def sample_hyperparameters_optuna(_trial):
            raise ValueError("invalid plugin search space")

    class FakeTrial:
        number = 3

        def __init__(self):
            self.user_attrs = {}

        def set_user_attr(self, key, value):
            self.user_attrs[key] = value

    mocker.patch("synthdata.generation.synthcity_backend.get_plugin_class", return_value=FakePlugin)
    mocker.patch("synthdata.generation.synthcity_backend.plugin_accepts", return_value=False)
    objective = build_synthcity_objective(
        "ctgan",
        object(),
        HPOConfig(metric_config={"stats": ["wasserstein_dist"]}),
        seed=0,
        tuning_loader=object(),
        stage_a_contract=contract,
        stage_a_source_df=source,
        stage_a_root=tmp_path,
        study_name="hpo_construction_failure",
    )

    trial = FakeTrial()
    with pytest.raises(optuna.TrialPruned, match="invalid plugin search space"):
        objective(trial)

    result_path = tmp_path / "hpo_construction_failure" / "trial-3" / "result.json"
    payload = json.loads(result_path.read_text())
    assert payload["checks"][0]["screen"] == "stage_a_exception"
    assert payload["checks"][0]["observed"]["exception_type"] == "ValueError"
    assert trial.user_attrs["stage_a_state"] == "pruned"


def test_synthcity_empty_report_before_stage_a_persists_exception(mocker, tmp_path):
    source = pd.DataFrame({"feature": [0.0, 1.0], "target": [0, 1]})
    contract = build_stage_a_contract(
        source,
        expected_n_samples=2,
        target_column="target",
    )

    class FakePlugin:
        @staticmethod
        def sample_hyperparameters_optuna(_trial):
            return {}

    class FakeTrial:
        number = 4

        def __init__(self):
            self.user_attrs = {}

        def set_user_attr(self, key, value):
            self.user_attrs[key] = value

    def fake_evaluate(tests, _fit_loader, **_kwargs):
        return {tests[0][0]: pd.DataFrame()}

    mocker.patch("synthdata.generation.synthcity_backend.get_plugin_class", return_value=FakePlugin)
    mocker.patch("synthdata.generation.synthcity_backend.plugin_accepts", return_value=False)
    mocker.patch("synthcity.benchmark.Benchmarks.evaluate", side_effect=fake_evaluate)
    objective = build_synthcity_objective(
        "ctgan",
        object(),
        HPOConfig(metric_config={"stats": ["wasserstein_dist"]}),
        seed=0,
        tuning_loader=object(),
        stage_a_contract=contract,
        stage_a_source_df=source,
        stage_a_root=tmp_path,
        study_name="hpo_empty_report",
    )

    trial = FakeTrial()
    with pytest.raises(optuna.TrialPruned, match="no metric rows"):
        objective(trial)

    result_path = tmp_path / "hpo_empty_report" / "trial-4" / "result.json"
    payload = json.loads(result_path.read_text())
    assert payload["checks"][0]["screen"] == "stage_a_exception"
    assert payload["checks"][0]["observed"]["exception_type"] == "ValueError"


def test_synthcity_missing_report_before_stage_a_persists_exception(mocker, tmp_path):
    source = pd.DataFrame({"feature": [0.0, 1.0], "target": [0, 1]})
    contract = build_stage_a_contract(
        source,
        expected_n_samples=2,
        target_column="target",
    )

    class FakePlugin:
        @staticmethod
        def sample_hyperparameters_optuna(_trial):
            return {}

    class FakeTrial:
        number = 5

        def __init__(self):
            self.user_attrs = {}

        def set_user_attr(self, key, value):
            self.user_attrs[key] = value

    mocker.patch("synthdata.generation.synthcity_backend.get_plugin_class", return_value=FakePlugin)
    mocker.patch("synthdata.generation.synthcity_backend.plugin_accepts", return_value=False)
    mocker.patch("synthcity.benchmark.Benchmarks.evaluate", return_value={})
    objective = build_synthcity_objective(
        "ctgan",
        object(),
        HPOConfig(metric_config={"stats": ["wasserstein_dist"]}),
        seed=0,
        tuning_loader=object(),
        stage_a_contract=contract,
        stage_a_source_df=source,
        stage_a_root=tmp_path,
        study_name="hpo_missing_report",
    )

    trial = FakeTrial()
    with pytest.raises(optuna.TrialPruned, match="did not return report"):
        objective(trial)

    result_path = tmp_path / "hpo_missing_report" / "trial-5" / "result.json"
    payload = json.loads(result_path.read_text())
    assert payload["checks"][0]["screen"] == "stage_a_exception"
    assert payload["checks"][0]["observed"]["exception_type"] == "RuntimeError"


def test_hpo_score_rejects_diagnostic_or_failed_rows():
    diagnostic = pd.DataFrame(
        {"mean": [0.5], "direction": ["maximize"]},
        index=["stats.prdc"],
    )
    with pytest.raises(ValueError, match="not decision-eligible"):
        hpo_score(diagnostic)

    failed = pd.DataFrame(
        {
            "mean": [float("nan")],
            "direction": ["minimize"],
            "errors": [1],
            "error_types": ["ValueError"],
        },
        index=["stats.wasserstein_dist"],
    )
    with pytest.raises(ValueError, match="not decision-eligible"):
        hpo_score(failed)


def test_hpo_score_rejects_partial_static_metric_set():
    report = pd.DataFrame(
        {
            "mean": [0.25],
            "direction": ["minimize"],
        },
        index=["stats.wasserstein_dist.joint"],
    )

    with pytest.raises(ValueError, match="incomplete metric set.*missing"):
        hpo_score(
            report,
            expected_keys=(
                "stats.wasserstein_dist.joint",
                "stats.inv_kl_divergence.marginal",
            ),
        )


def test_hpo_score_rejects_duplicate_static_metric_set():
    metric_key = "stats.wasserstein_dist.joint"
    report = pd.DataFrame(
        {
            "mean": [0.25, 0.3],
            "direction": ["minimize", "minimize"],
        },
        index=[metric_key, metric_key],
    )

    with pytest.raises(ValueError, match="incomplete metric set.*duplicate"):
        hpo_score(report, expected_keys=(metric_key,))


def test_synthcity_hpo_objective_prunes_partial_metric_set(mocker):
    class FakePlugin:
        @staticmethod
        def sample_hyperparameters_optuna(trial):
            return {}

    def fake_evaluate(tests, fit_loader, **kwargs):
        return {
            tests[0][0]: pd.DataFrame(
                {"mean": [0.25], "direction": ["minimize"]},
                index=["stats.wasserstein_dist.joint"],
            )
        }

    mocker.patch("synthdata.generation.synthcity_backend.get_plugin_class", return_value=FakePlugin)
    mocker.patch("synthdata.generation.synthcity_backend.plugin_accepts", return_value=False)
    mocker.patch("synthcity.benchmark.Benchmarks.evaluate", side_effect=fake_evaluate)

    objective = build_synthcity_objective(
        "ctgan",
        object(),
        HPOConfig(
            metric_config={
                "stats": ["wasserstein_dist", "inv_kl_divergence"],
            }
        ),
        seed=7,
        tuning_loader=object(),
    )

    with pytest.raises(optuna.TrialPruned, match="incomplete metric set"):
        objective(type("Trial", (), {"number": 6})())


def test_hpo_score_accepts_only_approved_operational_rows():
    report = pd.DataFrame(
        {
            "mean": [0.25, 0.75],
            "direction": ["minimize", "maximize"],
        },
        index=["stats.wasserstein_dist.joint", "performance.xgb.mean"],
    )

    assert hpo_score(report) == pytest.approx(-0.25)


def test_hpo_score_ignores_candidate_independent_performance_baseline():
    report = pd.DataFrame(
        {
            "mean": [0.8, 0.6, 0.7],
            "direction": ["maximize", "maximize", "maximize"],
        },
        index=[
            "performance.xgb.gt",
            "performance.xgb.syn_id",
            "performance.xgb.syn_ood",
        ],
    )

    assert hpo_score(report) == pytest.approx(-0.65)


def test_hpo_score_does_not_double_count_legacy_performance_aggregate():
    report = pd.DataFrame(
        {
            "mean": [0.6, 0.7, 0.9],
            "direction": ["maximize", "maximize", "maximize"],
        },
        index=[
            "performance.xgb.syn_id",
            "performance.xgb.syn_ood",
            "performance.xgb.mean",
        ],
    )

    assert hpo_score(report) == pytest.approx(-0.65)


def test_synthcity_hpo_objective_uses_explicit_train_and_tuning_loaders(mocker):
    train_loader = object()
    tuning_loader = object()
    captured = {}

    class FakePlugin:
        @staticmethod
        def sample_hyperparameters_optuna(trial):
            return {}

    def fake_evaluate(tests, fit_loader, **kwargs):
        captured["tests"] = tests
        captured["fit_loader"] = fit_loader
        captured.update(kwargs)
        return {
            tests[0][0]: pd.DataFrame(
                {"mean": [0.25], "direction": ["minimize"]},
                index=["stats.wasserstein_dist.joint"],
            )
        }

    mocker.patch("synthdata.generation.synthcity_backend.get_plugin_class", return_value=FakePlugin)
    mocker.patch("synthdata.generation.synthcity_backend.plugin_accepts", return_value=False)
    mocker.patch("synthcity.benchmark.Benchmarks.evaluate", side_effect=fake_evaluate)

    objective = build_synthcity_objective(
        "ctgan",
        train_loader,
        HPOConfig(metric_config={"stats": ["wasserstein_dist"]}),
        seed=7,
        tuning_loader=tuning_loader,
    )

    assert objective(type("Trial", (), {"number": 3})()) == pytest.approx(0.25)
    assert captured["fit_loader"] is train_loader
    assert captured["X_test"] is tuning_loader
    assert captured["fit_on_X"] is True


def test_synthcity_hpo_group_unsafe_trial_is_pruned_and_checkpointed(mocker, tmp_path):
    frame = pd.DataFrame({"feature": [0.0, 1.0, 0.0, 1.0], "target": [0, 1, 0, 1]})
    train_loader = make_loader(
        frame,
        "target",
        [],
        group_ids=["train-a", "train-a", "train-b", "train-b"],
    )
    tuning_loader = make_loader(
        frame,
        "target",
        [],
        group_ids=["tuning-a", "tuning-a", "tuning-b", "tuning-b"],
    )
    captured = {}

    class FakePlugin:
        @staticmethod
        def sample_hyperparameters_optuna(trial):
            del trial
            return {}

    def fake_evaluate(tests, fit_loader, **kwargs):
        captured["fit_loader"] = fit_loader
        captured.update(kwargs)
        report = pd.DataFrame(
            {
                "mean": [float("nan")],
                "errors": [1],
                "durations": [0.0],
                "direction": ["minimize"],
            },
            index=["performance.xgb.syn_id"],
        )
        report.attrs["group_safety"] = {
            "schema_version": "group-safety-v1",
            "status": "group_unsafe",
            "group_mode": "patient_group",
            "task_type": "classification",
            "reason": "unsupported grouped modality",
            "loader_types": {"X": "syn_seq"},
            "metrics": ["performance.xgb.syn_id"],
        }
        return {tests[0][0]: report}

    mocker.patch("synthdata.generation.synthcity_backend.get_plugin_class", return_value=FakePlugin)
    mocker.patch("synthdata.generation.synthcity_backend.plugin_accepts", return_value=False)
    mocker.patch("synthcity.benchmark.Benchmarks.evaluate", side_effect=fake_evaluate)

    objective = build_synthcity_objective(
        "ctgan",
        train_loader,
        HPOConfig(metric_config={"performance": ["xgb"]}, n_trials=1),
        seed=7,
        tuning_loader=tuning_loader,
        synthetic_size=len(frame),
        group_context={"group_mode": "patient_group"},
    )

    with pytest.raises(RuntimeError, match="produced no completed trials"):
        run_study(
            "hpo_group_unsafe",
            objective,
            HPOConfig(metric_config={"performance": ["xgb"]}, n_trials=1),
            tmp_path,
            seed=7,
            drop_keys=(),
        )

    checkpoint = load_hpo_trial_checkpoint(
        tmp_path / "hpo_checkpoints" / "hpo_group_unsafe" / "trial-0" / "checkpoint.json"
    )
    assert captured["fit_loader"] is train_loader
    assert captured["X_test"] is tuning_loader
    assert captured["group_mode"] == "patient_group"
    assert checkpoint["state"] == "pruned"
    assert checkpoint["objective_value"] is None
    assert checkpoint["metadata"]["outcome"]["status"] == "group_unsafe"
    assert checkpoint["metadata"]["outcome"]["group_safety"]["reason"] == (
        "unsupported grouped modality"
    )


def test_synthcity_hpo_objective_screens_candidate_before_scoring(mocker, tmp_path):
    source = pd.DataFrame({"feature": [0.0, 1.0, 0.0, 1.0], "target": [0, 1, 0, 1]})
    candidate = pd.DataFrame({"feature": [0.25, 0.75, 0.25, 0.75], "target": [0, 1, 0, 1]})
    contract = build_stage_a_contract(
        source,
        expected_n_samples=len(candidate),
        target_column="target",
    )
    events = []

    class FakePlugin:
        @staticmethod
        def sample_hyperparameters_optuna(trial):
            return {}

    class FakeTrial:
        number = 4

        def __init__(self):
            self.user_attrs = {}

        def set_user_attr(self, key, value):
            self.user_attrs[key] = value

    def fake_evaluate(tests, fit_loader, **kwargs):
        events.append("before_screen")
        kwargs["candidate_screen"](candidate)
        events.append("after_screen")
        return {
            tests[0][0]: pd.DataFrame(
                {"mean": [0.25], "direction": ["minimize"]},
                index=["stats.wasserstein_dist.joint"],
            )
        }

    mocker.patch("synthdata.generation.synthcity_backend.get_plugin_class", return_value=FakePlugin)
    mocker.patch("synthdata.generation.synthcity_backend.plugin_accepts", return_value=False)
    mocker.patch("synthcity.benchmark.Benchmarks.evaluate", side_effect=fake_evaluate)
    objective = build_synthcity_objective(
        "ctgan",
        object(),
        HPOConfig(metric_config={"stats": ["wasserstein_dist"]}),
        seed=7,
        tuning_loader=object(),
        synthetic_size=len(candidate),
        stage_a_contract=contract,
        stage_a_source_df=source,
        stage_a_root=tmp_path,
        study_name="hpo_ctgan",
    )
    trial = FakeTrial()

    assert objective(trial) == pytest.approx(0.25)
    assert events == ["before_screen", "after_screen"]
    assert trial.user_attrs["stage_a_state"] == "passed"
    assert (tmp_path / "hpo_ctgan" / "contract.json").exists()
    assert (tmp_path / "hpo_ctgan" / "trial-4" / "result.json").exists()


def test_tabpfgen_hpo_objective_screens_candidate_before_eval(mocker, tmp_path):
    from synthdata.generation import tabpfgen_backend as backend

    source = pd.DataFrame({"feature": [0.0, 1.0, 0.0, 1.0], "target": [0, 1, 0, 1]})
    candidate_features = pd.DataFrame({"feature": [0.25, 0.75, 0.25, 0.75]})
    contract = build_stage_a_contract(
        source,
        expected_n_samples=len(source),
        target_column="target",
    )
    events = []

    class FakeGenerator:
        def __init__(self, **kwargs):
            pass

        def generate_classification(self, **kwargs):
            return candidate_features.to_numpy(), None

    class FakeClassifier:
        def fit(self, x_train, y_train):
            pass

        def predict(self, features):
            return source["target"].to_numpy()

    class FakeTrial:
        number = 5

        def __init__(self):
            self.user_attrs = {}

        def suggest_int(self, name, low, high, step=1):
            return low

        def suggest_float(self, name, low, high, log=False):
            return low

        def set_user_attr(self, key, value):
            self.user_attrs[key] = value

    mocker.patch.object(backend, "TabPFGen", FakeGenerator)
    mocker.patch("tabpfn.TabPFNClassifier", FakeClassifier)

    def eval_fn(synthetic):
        events.append("eval")
        return 0.5

    objective = backend.build_tabpfgen_standard_objective(
        source,
        ["feature"],
        [],
        "target",
        len(source),
        500,
        eval_fn,
        stage_a_contract=contract,
        stage_a_source_df=source,
        stage_a_root=tmp_path,
        study_name="hpo_tabpfgen_standard",
        target_is_categorical=True,
    )
    trial = FakeTrial()

    assert objective(trial) == pytest.approx(0.5)
    assert events == ["eval"]
    assert trial.user_attrs["stage_a_state"] == "passed"
    assert (tmp_path / "hpo_tabpfgen_standard" / "trial-5" / "result.json").exists()


def test_tabpfgen_standard_cardinality_oversamples_and_trims(mocker):
    from synthdata.generation import tabpfgen_backend as backend

    source = pd.DataFrame(
        {
            "feature": np.arange(6, dtype=float),
            "target": [0, 1, 2, 0, 1, 2],
        }
    )
    captured = {}

    class FakeGenerator:
        def __init__(self, **kwargs):
            del kwargs

        def generate_classification(self, **kwargs):
            captured["n_samples"] = kwargs["n_samples"]
            count = kwargs["n_samples"]
            return (
                np.arange(count, dtype=float).reshape(-1, 1),
                np.tile(np.arange(3), count // 3),
            )

    mocker.patch.object(backend, "TabPFGen", FakeGenerator)

    result = backend.generate_tabpfgen_standard(
        source,
        ["feature"],
        [],
        "target",
        5,
        target_is_categorical=True,
    )

    assert captured["n_samples"] == 6
    assert len(result) == 5


def test_tabpfgen_custom_cardinality_uses_exact_proportions(mocker):
    from synthdata.generation import tabpfgen_backend as backend

    source = pd.DataFrame(
        {
            "feature": np.arange(6, dtype=float),
            "target": [0, 1, 2, 0, 1, 2],
        }
    )
    captured = {}

    class FakeGenerator:
        def __init__(self, **kwargs):
            del kwargs

        def generate_classification(self, features, labels, n_samples, balance_classes):
            del features, labels, balance_classes
            captured["n_samples"] = n_samples
            return (
                np.arange(n_samples, dtype=float).reshape(-1, 1),
                np.tile(np.arange(3), n_samples // 3),
            )

    mocker.patch.object(backend, "TabPFGenSGLDLabels", FakeGenerator)

    result = backend.generate_tabpfgen_custom(
        source,
        ["feature"],
        [],
        "target",
        5,
        target_is_categorical=True,
    )

    assert captured["n_samples"] == 6
    assert len(result) == 5


def test_tabpfgen_standard_hpo_cardinality_oversamples_and_trims(mocker):
    from synthdata.generation import tabpfgen_backend as backend

    source = pd.DataFrame(
        {
            "feature": np.arange(6, dtype=float),
            "target": [0, 1, 2, 0, 1, 2],
        }
    )
    captured = {}
    evaluated = []

    class FakeGenerator:
        def __init__(self, **kwargs):
            del kwargs

        def generate_classification(self, **kwargs):
            captured["n_samples"] = kwargs["n_samples"]
            count = kwargs["n_samples"]
            return np.arange(count, dtype=float).reshape(-1, 1), np.tile(np.arange(3), count // 3)

    class FakeClassifier:
        def fit(self, features, labels):
            del features, labels
            return self

        def predict(self, features):
            return np.tile(np.arange(3), (len(features) + 2) // 3)[: len(features)]

    class FakeTrial:
        number = 6

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

    mocker.patch.object(backend, "TabPFGen", FakeGenerator)
    mocker.patch("tabpfn.TabPFNClassifier", FakeClassifier)

    objective = backend.build_tabpfgen_standard_objective(
        source,
        ["feature"],
        [],
        "target",
        5,
        500,
        lambda synthetic: evaluated.append(synthetic) or 0.5,
        target_is_categorical=True,
    )

    assert objective(FakeTrial()) == pytest.approx(0.5)
    assert captured["n_samples"] == 6
    assert len(evaluated) == 1
    assert len(evaluated[0]) == 5


def test_tabpfgen_hpo_checkpoint_metadata_is_durable(mocker, tmp_path):
    from synthdata.generation import tabpfgen_backend as backend

    source = pd.DataFrame(
        {
            "feature": np.arange(6, dtype=float),
            "target": [0, 1, 2, 0, 1, 2],
        }
    )
    calls = []

    class FakeGenerator:
        def __init__(self, **kwargs):
            calls.append(("construct", kwargs))

        def generate_classification(self, **kwargs):
            count = kwargs["n_samples"]
            return (
                np.arange(count, dtype=float).reshape(-1, 1),
                np.tile(np.arange(3), count // 3),
            )

    class FakeClassifier:
        def fit(self, features, labels):
            del features, labels
            return self

        def predict(self, features):
            return np.tile(np.arange(3), (len(features) + 2) // 3)[: len(features)]

    mocker.patch.object(backend, "TabPFGen", FakeGenerator)
    mocker.patch("tabpfn.TabPFNClassifier", FakeClassifier)
    objective = backend.build_tabpfgen_standard_objective(
        source,
        ["feature"],
        [],
        "target",
        6,
        500,
        lambda _synthetic: 0.5,
        seed=13,
        target_is_categorical=True,
    )
    config = HPOConfig(n_trials=1, timeout_seconds=None)

    run_study(
        "hpo_tabpfgen_checkpoint_metadata",
        objective,
        config,
        tmp_path,
        seed=13,
        drop_keys=(),
    )

    checkpoint_path = (
        tmp_path
        / "hpo_checkpoints"
        / "hpo_tabpfgen_checkpoint_metadata"
        / "trial-0"
        / "checkpoint.json"
    )
    checkpoint = load_hpo_trial_checkpoint(checkpoint_path)
    generator = checkpoint["metadata"]["generator"]
    assert generator["state"] == "present"
    assert generator["plugin_name"] == "tabpfgen_standard"
    assert generator["privacy_claim_type"] == "none"
    assert generator["implementation_fingerprint"]
    assert generator["metadata"]["schema_version"] == "generator-metadata-v2"
    assert (
        generator["metadata"]["implementation_fingerprint"]
        == generator["implementation_fingerprint"]
    )
    assert generator["metadata"]["plugin_name"] == "tabpfgen_standard"
    assert generator["metadata"]["n_samples"] == 6
    assert generator["metadata"]["random_state"] == 13

    def should_not_run(_trial):
        pytest.fail("a completed TabPFGen trial was recomputed during resume")

    run_study(
        "hpo_tabpfgen_checkpoint_metadata",
        should_not_run,
        config,
        tmp_path,
        seed=13,
        drop_keys=(),
    )
    assert len(calls) == 1
    assert load_hpo_trial_checkpoint(checkpoint_path) == checkpoint


def test_tabpfgen_custom_hpo_records_generator_metadata(mocker):
    from synthdata.generation import tabpfgen_backend as backend

    source = pd.DataFrame(
        {
            "feature": np.arange(6, dtype=float),
            "target": [0, 1, 2, 0, 1, 2],
        }
    )

    class FakeGenerator:
        def __init__(self, **kwargs):
            del kwargs

        def generate_classification(self, features, labels, n_samples, balance_classes):
            del features, labels, balance_classes
            return (
                np.arange(n_samples, dtype=float).reshape(-1, 1),
                np.tile(np.arange(3), n_samples // 3),
            )

    class FakeTrial:
        number = 7

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

    mocker.patch.object(backend, "TabPFGenSGLDLabels", FakeGenerator)
    objective = backend.build_tabpfgen_custom_objective(
        source,
        ["feature"],
        [],
        "target",
        6,
        500,
        lambda _synthetic: 0.5,
        seed=13,
        target_is_categorical=True,
    )
    trial = FakeTrial()

    assert objective(trial) == pytest.approx(0.5)
    assert trial.user_attrs["generator_metadata_state"] == "present"
    assert trial.user_attrs["generator_plugin_name"] == "tabpfgen_custom"
    assert trial.user_attrs["generator_privacy_claim_type"] == "none"
    assert trial.user_attrs["generator_metadata"]["plugin_name"] == "tabpfgen_custom"
    assert trial.user_attrs["generator_metadata"]["n_samples"] == 6
    assert trial.user_attrs["generator_metadata"]["random_state"] == 13


@pytest.mark.parametrize(
    ("function_name", "constructor_name", "parameter_name"),
    [
        ("generate_tabpfgen_standard", "TabPFGen", "tabpfgen_params"),
        ("generate_tabpfgen_custom", "TabPFGenSGLDLabels", "sgld_params"),
    ],
)
def test_tabpfgen_rejects_mismatched_semantic_context_before_generator(
    mocker,
    make_canonical_dataset,
    function_name,
    constructor_name,
    parameter_name,
):
    from synthdata.generation import tabpfgen_backend as backend

    dataset = make_canonical_dataset("column")
    train_frame = dataset.role_frame("train", imputed=True)
    semantic_context = semantic_context_payload(
        dataset,
        classification_score="balanced_accuracy",
    )
    semantic_context["target_column"] = "feature"
    constructor = mocker.patch.object(backend, constructor_name)
    generate = getattr(backend, function_name)

    with pytest.raises(ValueError, match="semantic_context target_column"):
        generate(
            train_frame,
            dataset.feature_columns,
            dataset.categorical_columns,
            dataset.target_column,
            2,
            **{parameter_name: {}},
            target_is_categorical=True,
            variable_schema_fingerprint=dataset.variable_schema_fingerprint,
            semantic_context=semantic_context,
        )

    constructor.assert_not_called()


@pytest.mark.parametrize(
    ("function_name", "constructor_name", "parameter_name"),
    [
        ("generate_tabpfgen_standard", "TabPFGen", "tabpfgen_params"),
        ("generate_tabpfgen_custom", "TabPFGenSGLDLabels", "sgld_params"),
    ],
)
def test_tabpfgen_backend_rejects_continuous_target_before_generator(
    mocker,
    function_name,
    constructor_name,
    parameter_name,
):
    from synthdata.generation import tabpfgen_backend as backend

    constructor = mocker.patch.object(backend, constructor_name)
    frame = pd.DataFrame({"feature": [0.0, 1.0], "target": [0.0, 1.0]})
    generate = getattr(backend, function_name)

    with pytest.raises(ValueError, match="TabPFGen generation requires a categorical target"):
        generate(
            frame,
            ["feature"],
            [],
            "target",
            2,
            **{parameter_name: {}},
            target_is_categorical=False,
        )

    constructor.assert_not_called()


def test_default_hpo_artifacts_live_inside_the_experiment_directory(tmp_path):
    experiment_dir = tmp_path / "output" / "dataset" / "synthetic_data" / "v2" / "exp-1"

    storage_url = default_storage_url(experiment_dir)

    assert storage_url == f"sqlite:///{experiment_dir / 'optuna_studies.db'}"
    assert default_best_params_path(experiment_dir) == experiment_dir / "hpo_best_params.json"
    assert Path(experiment_dir).exists()


def test_completed_hpo_keeps_best_and_latest_generator_checkpoints(tmp_path):
    workspace = tmp_path / "synthcity_workspace"
    output_dir = tmp_path / "output"
    values = [3.0, 1.0, 2.0]

    def objective(trial):
        for suffix in ("base_cache", "augmentation_cache"):
            path = workspace / (f"data_trial_{trial.number}_tvae_{suffix}_3.11.15_generator_0.bkp")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"checkpoint")
        return values[trial.number]

    run_study(
        "hpo_tvae",
        objective,
        HPOConfig(n_trials=3, timeout_seconds=None),
        output_dir,
        seed=0,
        drop_keys=(),
        checkpoint_workspace=workspace,
        checkpoint_plugin="tvae",
    )

    assert (workspace / "data_trial_1_tvae_base_cache_3.11.15_generator_0.bkp").exists()
    assert (workspace / "data_trial_1_tvae_augmentation_cache_3.11.15_generator_0.bkp").exists()
    assert (workspace / "data_trial_2_tvae_base_cache_3.11.15_generator_0.bkp").exists()
    assert (workspace / "data_trial_2_tvae_augmentation_cache_3.11.15_generator_0.bkp").exists()
    assert not (workspace / "data_trial_0_tvae_base_cache_3.11.15_generator_0.bkp").exists()
    assert not (workspace / "data_trial_0_tvae_augmentation_cache_3.11.15_generator_0.bkp").exists()


def test_partial_hpo_keeps_best_and_latest_generator_checkpoints(tmp_path):
    workspace = tmp_path / "synthcity_workspace"

    def objective(trial):
        path = workspace / f"data_trial_{trial.number}_tvae_cache_3.11.15_generator_0.bkp"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")
        if trial.number == 0:
            return 1.0
        raise optuna.TrialPruned()

    run_study(
        "hpo_tvae",
        objective,
        HPOConfig(n_trials=3, timeout_seconds=None),
        tmp_path / "output",
        seed=0,
        drop_keys=(),
        checkpoint_workspace=workspace,
        checkpoint_plugin="tvae",
    )

    assert (workspace / "data_trial_0_tvae_cache_3.11.15_generator_0.bkp").exists()
    assert (workspace / "data_trial_2_tvae_cache_3.11.15_generator_0.bkp").exists()
    assert not (workspace / "data_trial_1_tvae_cache_3.11.15_generator_0.bkp").exists()


def test_all_pruned_hpo_fails_without_default_fallback(tmp_path):
    def objective(trial):
        raise optuna.TrialPruned("Stage A failed")

    with pytest.raises(RuntimeError, match="produced no completed trials"):
        run_study(
            "hpo_all_pruned",
            objective,
            HPOConfig(n_trials=2, timeout_seconds=None),
            tmp_path / "output",
            seed=0,
            drop_keys=(),
        )
