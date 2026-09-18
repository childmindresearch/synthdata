"""Unit tests for the explicit scope of resumable HPO artifacts."""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest

from synthdata.config import HPOConfig, load_config
from synthdata.data import semantic_context_payload
from synthdata.generation import hpo as hpo_module
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
from synthdata.generation.synthcity_backend import build_synthcity_objective
from tests.unit.synthcity_emitted_key_fixtures import HPO_SYNTHCITY_EMITTED_KEY_FIXTURES

pytestmark = pytest.mark.unit


def _strings(*values: str) -> np.ndarray:
    """Build pandas-compatible string arrays with explicit element typing."""
    return np.asarray(values, dtype=str)


def _screen_trial_callback(
    callback: Callable[[optuna.trial.Trial], float],
) -> Callable[[optuna.trial.Trial], float]:
    def wrapped(trial: optuna.trial.Trial) -> float:
        return callback(trial)

    return wrapped


def _hpo_context(**overrides):
    context = build_hpo_context(
        task_type="classification",
        metric_config={"task12": ["mixed_mmd.v1"]},
        registry_digest="registry-a",
        stage_a_contract_digest="stage-a",
        group_context={"group_mode": "row"},
        role_context_fingerprint="roles-a",
        role_context={"roles": {"train": {"rows": 4}, "tuning": {"rows": 2}}},
        release_transform_digest="release-a",
        role_hashes={"train": "train-a", "tuning": "tuning-a"},
        contracts={
            "fit_roles": ["train"],
            "comparison_role": "tuning",
            "excluded_roles": ["final_holdout"],
            "privacy": False,
            "fairness": False,
        },
        support_provenance={"fit_roles": ["train"], "support_contract": "train_frozen_v1"},
        bandwidth_provenance={
            "fit_roles": ["train"],
            "comparison_role": "tuning",
            "contract": "train_frozen_v1",
        },
        objective_version="release-utility-v1",
    )
    context.update(overrides)
    return context


def _provenance_kwargs():
    return {
        "release_transform_digest": "release-a",
        "role_hashes": {"train": "train-a", "tuning": "tuning-a"},
        "contracts": {
            "fit_roles": ["train"],
            "comparison_role": "tuning",
            "excluded_roles": ["final_holdout"],
            "privacy": False,
            "fairness": False,
        },
        "support_provenance": {"fit_roles": ["train"], "support_contract": "train_frozen_v1"},
        "bandwidth_provenance": {
            "fit_roles": ["train"],
            "comparison_role": "tuning",
            "contract": "train_frozen_v1",
        },
        "objective_version": "release-utility-v1",
    }


def _score(report, **kwargs):
    report.attrs.setdefault("hpo_provenance", _hpo_context())
    return hpo_score(report, **kwargs)


def test_default_hpo_objective_excludes_privacy_and_diagnostics():
    config = HPOConfig()

    assert "privacy" not in config.metric_config
    validate_hpo_metric_config(config.metric_config)
    assert "privacy" not in config.metric_config


def test_loris_config_loads_canonical_versioned_equal_thirds_policy():
    config = load_config(Path(__file__).parents[2] / "configs" / "config_loris.yaml")
    metric_config = config.generation.hpo.metric_config
    metrics = [metric for values in metric_config.values() for metric in values]

    assert metric_config == {
        "canonical_objectives": [
            "tstr_macro_f1.v1",
            "mixed_mmd.v1",
            "elastic_net_jsd.v1",
        ]
    }
    assert len(metrics) == len(set(metrics)) == 3
    assert all(metric.endswith(".v1") for metric in metrics)
    assert config.generation.hpo.utility_policy == {
        "metrics": metrics,
        "weights": [1 / 3, 1 / 3, 1 / 3],
    }


def test_canonical_hpo_partial_set_fails_closed():
    report = pd.DataFrame(
        {"mean": [0.2], "direction": _strings("minimize")}, index=_strings("mixed_mmd.v1")
    )
    report.attrs["canonical_hpo"] = True
    report.attrs["canonical_hpo_keys"] = ("mixed_mmd.v1", "elastic_net_jsd.v1")
    report.attrs["hpo_provenance"] = _hpo_context()
    with pytest.raises(ValueError, match="incomplete metric set"):
        _score(report, expected_keys=("mixed_mmd.v1", "elastic_net_jsd.v1"))


def test_patient_group_hpo_rejects_row_only_objectives():
    with pytest.raises(ValueError, match="not in canonical HPO allowlist"):
        build_hpo_context(
            task_type="classification",
            metric_config={"stats": ["wasserstein_dist"]},
            registry_digest="registry-a",
            stage_a_contract_digest="stage-a",
            group_context={"group_mode": "patient_group"},
            role_context_fingerprint="roles-a",
            role_context={"roles": {"train": {"rows": 4}, "tuning": {"rows": 2}}},
            **_provenance_kwargs(),
        )


def test_patient_group_hpo_accepts_group_safe_objective_contract():
    context = build_hpo_context(
        task_type="classification",
        metric_config={"task12": ["mixed_mmd.v1"]},
        registry_digest="registry-a",
        stage_a_contract_digest="stage-a",
        group_context={"group_mode": "patient_group"},
        role_context_fingerprint="roles-a",
        role_context={"roles": {"train": {"rows": 4}, "tuning": {"rows": 2}}},
        **_provenance_kwargs(),
    )

    assert context["expected_emitted_keys"] == [
        "tstr_macro_f1.v1",
        "mixed_mmd.v1",
        "elastic_net_jsd.v1",
    ]


def test_patient_group_hpo_resolves_reordered_rows_by_stable_identity(mocker):
    train = pd.DataFrame({"x": [0.0, 1.0], "target": [0, 1]}, index=_strings("a", "b"))
    tuning = pd.DataFrame({"x": [2.0, 3.0], "target": [0, 1]}, index=_strings("c", "d"))
    groups = {"a": "p1", "b": "p2"}
    tuning_groups = {"c": "p3", "d": "p4"}
    report = pd.DataFrame(
        {"mean": [0.2, 0.3, 0.8], "direction": _strings("minimize", "minimize", "maximize")},
        index=_strings("mixed_mmd.v1", "elastic_net_jsd.v1", "tstr_macro_f1.v1"),
    )
    report.attrs["hpo_provenance"] = _hpo_context()

    evaluate = hpo_module.build_synthetic_eval_fn(
        train,
        tuning,
        "target",
        [],
        {"task12": ["mixed_mmd.v1"]},
        seed=0,
        group_context={"group_mode": "patient_group"},
        train_group_ids=groups,
        holdout_group_ids=tuning_groups,
    )
    reordered_evaluate = hpo_module.build_synthetic_eval_fn(
        train.iloc[[1, 0]],
        tuning.iloc[[1, 0]],
        "target",
        [],
        {"task12": ["mixed_mmd.v1"]},
        seed=0,
        group_context={"group_mode": "patient_group"},
        train_group_ids=groups,
        holdout_group_ids=tuning_groups,
    )
    report.attrs["hpo_provenance"] = _hpo_context()
    mocker.patch.object(hpo_module, "evaluate_canonical_hpo_metrics", return_value=report)
    assert evaluate(train) == pytest.approx(reordered_evaluate(train))


def test_patient_group_hpo_rejects_misaligned_group_rows():
    train = pd.DataFrame({"x": [0.0, 1.0], "target": [0, 1]}, index=_strings("a", "b"))
    tuning = pd.DataFrame({"x": [2.0, 3.0], "target": [0, 1]}, index=_strings("c", "d"))

    with pytest.raises(ValueError, match="do not align"):
        build_synthetic_eval_fn(
            train,
            tuning,
            "target",
            [],
            {"task12": ["mixed_mmd.v1"]},
            seed=0,
            group_context={"group_mode": "patient_group"},
            train_group_ids=pd.Series(["p1", "p2"], index=["b", "wrong"]),
            holdout_group_ids=pd.Series(["p3", "p4"], index=["c", "d"]),
        )


def test_hpo_context_uses_contextual_emitted_key_manifest():
    context = build_hpo_context(
        task_type="classification",
        metric_config={"task12": ["tstr_macro_f1.v1"]},
        registry_digest="registry-a",
        stage_a_contract_digest="stage-a",
        group_context={"group_mode": "row"},
        role_context_fingerprint="roles-a",
        role_context={"roles": {"train": {"rows": 4}, "tuning": {"rows": 2}}},
        **_provenance_kwargs(),
        variable_columns=["feature", "target"],
        attack_target_types={"protected": "categorical"},
    )

    assert context["expected_emitted_keys"] == [
        "tstr_macro_f1.v1",
        "mixed_mmd.v1",
        "elastic_net_jsd.v1",
    ]


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
        **_provenance_kwargs(),
        variable_columns=["feature", "target"],
        attack_target_types={"protected": "categorical"},
    )

    assert context["expected_emitted_keys"] == [
        "tstr_macro_f1.v1",
        "mixed_mmd.v1",
        "elastic_net_jsd.v1",
    ]
    assert set(expected_keys) <= set(context["expected_emitted_keys"])


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


@pytest.mark.parametrize(
    "field",
    [
        "release_transform_digest",
        "role_hashes",
        "contracts",
        "support_provenance",
        "bandwidth_provenance",
        "objective_version",
    ],
)
def test_hpo_context_digest_changes_for_each_required_provenance(field):
    context = _hpo_context()
    changed = dict(context)
    value = changed[field]
    if isinstance(value, dict):
        changed[field] = {**value, "changed": "yes"}
    else:
        changed[field] = "changed"
    assert hpo_context_digest(context) != hpo_context_digest(changed)


def test_hpo_context_requires_explicit_provenance():
    context = _hpo_context()
    context.pop("role_hashes")
    with pytest.raises(ValueError, match="required provenance"):
        hpo_context_digest(context)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "contracts",
            {
                "fit_roles": ["train", "tuning"],
                "comparison_role": "tuning",
                "excluded_roles": ["final_holdout"],
                "privacy": False,
                "fairness": False,
            },
        ),
        (
            "contracts",
            {
                "fit_roles": ["train"],
                "comparison_role": "tuning",
                "excluded_roles": ["final_holdout", "hidden"],
                "privacy": False,
                "fairness": False,
            },
        ),
        (
            "contracts",
            {
                "fit_roles": ["train"],
                "comparison_role": "tuning",
                "excluded_roles": ["final_holdout"],
                "privacy": True,
                "fairness": False,
            },
        ),
        (
            "contracts",
            {
                "fit_roles": ["train"],
                "comparison_role": "tuning",
                "excluded_roles": ["final_holdout"],
                "privacy": False,
                "fairness": True,
            },
        ),
        ("support_provenance", {"fit_roles": ["tuning"], "support_contract": "train_frozen_v1"}),
        ("support_provenance", {"fit_roles": ["train"], "support_contract": "mutable_v1"}),
        (
            "bandwidth_provenance",
            {"fit_roles": ["train"], "comparison_role": "train", "contract": "train_frozen_v1"},
        ),
        (
            "bandwidth_provenance",
            {"fit_roles": ["train"], "comparison_role": "tuning", "contract": "mutable_v1"},
        ),
    ],
)
def test_hpo_context_rejects_noncanonical_provenance_contract_at_construction(field, value):
    kwargs = _provenance_kwargs()
    kwargs[field] = value
    with pytest.raises(ValueError, match="contracts|support_provenance|bandwidth_provenance"):
        build_hpo_context(
            task_type="classification",
            metric_config={"task12": ["mixed_mmd.v1"]},
            registry_digest="registry-a",
            stage_a_contract_digest="stage-a",
            group_context={"group_mode": "row"},
            role_context_fingerprint="roles-a",
            role_context={"roles": {"train": {"rows": 4}, "tuning": {"rows": 2}}},
            **kwargs,
        )


@pytest.mark.parametrize(
    "field_value",
    [
        (
            "contracts",
            {
                "fit_roles": ["train", "tuning"],
                "comparison_role": "tuning",
                "excluded_roles": ["final_holdout"],
                "privacy": False,
                "fairness": False,
            },
        ),
        (
            "contracts",
            {
                "fit_roles": ["train"],
                "comparison_role": "tuning",
                "excluded_roles": [],
                "privacy": False,
                "fairness": False,
            },
        ),
        (
            "contracts",
            {
                "fit_roles": ["train"],
                "comparison_role": "tuning",
                "excluded_roles": ["final_holdout"],
                "privacy": True,
                "fairness": False,
            },
        ),
        (
            "contracts",
            {
                "fit_roles": ["train"],
                "comparison_role": "tuning",
                "excluded_roles": ["final_holdout"],
                "privacy": False,
                "fairness": True,
            },
        ),
        ("support_provenance", {"fit_roles": ["tuning"], "support_contract": "train_frozen_v1"}),
        ("support_provenance", {"fit_roles": ["train"], "support_contract": "mutable_v1"}),
        (
            "bandwidth_provenance",
            {"fit_roles": ["train"], "comparison_role": "train", "contract": "train_frozen_v1"},
        ),
        (
            "bandwidth_provenance",
            {"fit_roles": ["train"], "comparison_role": "tuning", "contract": "mutable_v1"},
        ),
    ],
)
def test_hpo_cache_rejects_noncanonical_provenance_contract(tmp_path, field_value):
    field, value = field_value
    with pytest.raises(ValueError):
        BestParamsCache(tmp_path / "cache.json", hpo_context=_hpo_context(**{field: value}))


def test_hpo_score_rejects_provenance_free_report():
    report = pd.DataFrame(
        {"mean": [0.2], "direction": _strings("minimize")}, index=_strings("mixed_mmd.v1")
    )
    with pytest.raises(ValueError, match="missing required hpo_provenance"):
        hpo_score(report)


def test_canonical_score_retains_provenance_metadata(mocker):
    provenance = {
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
        "support_provenance": {"fit_roles": ["train"], "support_contract": "train_frozen_v1"},
        "bandwidth_provenance": {
            "fit_roles": ["train"],
            "comparison_role": "tuning",
            "contract": "train_frozen_v1",
        },
        "objective_version": "release-utility-v1",
    }
    report = pd.DataFrame(
        {
            "mean": [0.2, 0.3, 0.8],
            "direction": _strings("minimize", "minimize", "maximize"),
        },
        index=_strings("mixed_mmd.v1", "elastic_net_jsd.v1", "tstr_macro_f1.v1"),
    )
    report.attrs["hpo_provenance"] = provenance
    assert _score(report) == pytest.approx(-((0.8 + 0.7 + 0.8) / 3))
    assert report.attrs["hpo_provenance"] == provenance


def test_best_params_cache_rejects_mismatched_hpo_context(tmp_path):
    path = tmp_path / "hpo_best_params.json"
    context = _hpo_context()
    BestParamsCache(path, hpo_context=context).set("synthcity", "ctgan", {"n_iter": 3})

    assert BestParamsCache(path, hpo_context=context).has("synthcity", "ctgan")
    assert not BestParamsCache(
        path,
        hpo_context={**context, "registry_digest": "registry-b"},
    ).has("synthcity", "ctgan")


@pytest.mark.parametrize(
    ("field", "value", "error_field"),
    [
        ("release_transform_digest", "   ", "release_transform_digest"),
        ("role_context_fingerprint", "\t\n", "role_context_fingerprint"),
        (
            "role_hashes",
            {"train": "train-a", "tuning": "tuning-a", "synthetic": " "},
            "role_hashes",
        ),
        ("support_provenance", {"fit_roles": [" "]}, "support_provenance"),
        (
            "bandwidth_provenance",
            {"nested": {"comparison_role": "\t"}},
            "bandwidth_provenance",
        ),
        (
            "contracts",
            {"fit_roles": ["train"], "comparison_role": "tuning", "nested": [" "]},
            "contracts",
        ),
    ],
)
def test_hpo_context_rejects_blank_provenance_strings(field, value, error_field):
    if field == "role_context_fingerprint":
        with pytest.raises(ValueError, match=error_field):
            build_hpo_context(
                task_type="classification",
                metric_config={"task12": ["mixed_mmd.v1"]},
                registry_digest="registry-a",
                stage_a_contract_digest="stage-a",
                group_context={"group_mode": "row"},
                role_context_fingerprint=value,
                role_context={},
                **_provenance_kwargs(),
            )
    else:
        context = _hpo_context(**{field: value})
        with pytest.raises(ValueError, match=error_field):
            hpo_context_digest(context)


def test_cache_rejects_legacy_payload_even_when_allowed(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"synthcity": {"ctgan": {"n_iter": 3}}}))
    with pytest.raises(ValueError, match="verified hpo_context"):
        BestParamsCache(path, role_context_fingerprint="roles-a", allow_unverified_legacy=True)


def test_checkpoint_persist_and_load_require_context(tmp_path):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda _trial: 0.1, n_trials=1)
    with pytest.raises(ValueError, match="canonical hpo_context"):
        persist_hpo_trial_checkpoint(tmp_path, "study", study.trials[0])
    with pytest.raises(ValueError, match="canonical hpo_context"):
        load_hpo_trial_checkpoint(tmp_path / "missing.json")


def test_run_study_and_resume_require_context(tmp_path):
    config = HPOConfig(n_trials=1, timeout_seconds=None)
    with pytest.raises(ValueError, match="canonical hpo_context"):
        run_study("context_required", lambda _trial: 1.0, config, tmp_path, seed=0, drop_keys=())


@pytest.mark.parametrize(
    "unsafe_name",
    ["../escape", "/absolute", "nested/name", "nested\\name", "bad\x00name", ".", "..", "   "],
)
def test_hpo_rejects_unsafe_study_names_without_echoing_input(unsafe_name):
    with pytest.raises(ValueError, match="safe relative identifier") as error:
        contextual_study_name(unsafe_name, _hpo_context())

    assert unsafe_name not in str(error.value)


def test_hpo_preserves_valid_study_name_in_contextual_name():
    context = _hpo_context()

    contextual_name = contextual_study_name("valid-model_v2", context)

    assert contextual_name == f"valid-model_v2-{hpo_context_digest(context)[:16]}"


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
        hpo_context={"metric_config": {"task12": ["mixed_mmd.v1"]}},
    )

    payload = contract.to_dict()

    assert payload["registry_digest"] == "registry-a"
    assert payload["role_context_fingerprint"] == "roles-a"
    assert payload["role_context"] == {"roles": {"train": {"rows": 4}}}
    assert payload["group_context"] == {"group_mode": "row"}
    assert payload["hpo_context"] == {"metric_config": {"task12": ["mixed_mmd.v1"]}}


def test_stage_a_trial_prune_and_persist_result(tmp_path):
    source = _stage_a_source()
    contract = _stage_a_contract(source)
    contract_path = persist_stage_a_contract(tmp_path, contract)
    study = optuna.create_study(direction="minimize")

    study.optimize(
        _screen_trial_callback(
            lambda trial: (
                screen_stage_a_trial(
                    trial,
                    source.copy(),
                    contract,
                    source,
                    tmp_path,
                    "hpo_test",
                ),
                0.0,
            )[1]
        ),
        n_trials=1,
    )

    trial = study.trials[0]
    result_path = tmp_path / "hpo_test" / "trial-0" / "result.json"
    assert trial.state == optuna.trial.TrialState.PRUNED
    assert trial.user_attrs["stage_a_state"] == "pruned"
    assert trial.user_attrs["stage_a_result_path"] == "hpo_test/trial-0/result.json"
    assert not Path(trial.user_attrs["stage_a_result_path"]).is_absolute()
    assert result_path.exists()
    assert contract_path.exists()
    assert "exact_reuse" in result_path.read_text()


def test_stage_a_trial_persists_screen_exception_as_pruned(tmp_path):
    sentinel = "source frame does not match SECRET_PATH /tmp/raw-value"
    source = _stage_a_source()
    contract = _stage_a_contract(source)
    changed_source = source.copy()
    changed_source.loc[0, "age"] = 2.0
    study = optuna.create_study(direction="minimize")

    study.optimize(
        _screen_trial_callback(
            lambda trial: (
                screen_stage_a_trial(
                    trial,
                    source.copy(),
                    contract,
                    changed_source,
                    tmp_path,
                    "hpo_exception",
                ),
                0.0,
            )[1]
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
    assert payload["checks"][0]["observed"]["exception_message"] == (
        "Stage A screen failed; exception details suppressed."
    )
    assert payload["checks"][0]["observed"]["reason_code"] == "stage_a_screen_exception"
    assert "source frame does not match" not in result_path.read_text()
    assert sentinel not in result_path.read_text()
    assert "source frame does not match" not in payload["prune_reasons"][0]


def test_stage_a_result_path_in_checkpoint_is_relative_and_safe(tmp_path):
    source = _stage_a_source()
    contract = _stage_a_contract(source)
    study = optuna.create_study(direction="minimize")

    study.optimize(
        _screen_trial_callback(
            lambda trial: (
                screen_stage_a_trial(
                    trial, source.copy(), contract, source, tmp_path.resolve(), "safe_paths"
                ),
                0.0,
            )[1]
        ),
        n_trials=1,
    )
    trial = study.trials[0]
    checkpoint = persist_hpo_trial_checkpoint(
        tmp_path / "checkpoints",
        "safe_paths",
        trial,
        hpo_context=_hpo_context(),
    )
    payload = json.loads(checkpoint.read_text())
    result_path = payload["metadata"]["stage_a"]["result_path"]
    assert result_path == "safe_paths/trial-0/result.json"
    assert not Path(result_path).is_absolute()
    assert ".." not in Path(result_path).parts


def test_stage_a_setup_failure_is_persisted_before_trial_construction(tmp_path, mocker):
    sentinel = "categorical encoder failed SECRET_ID /tmp/raw-path"
    source = _stage_a_source()
    contract = _stage_a_contract(source)
    mocker.patch(
        "synthdata.generation.tabpfgen_backend.label_encode_non_numeric_columns",
        side_effect=ValueError(sentinel),
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
    assert payload["prune_reasons"][0] == (
        "stage_a_screen_exception: Stage A screen failed; exception details suppressed."
    )
    assert sentinel not in result_path.read_text()


def test_resumed_study_counts_pruned_trials_without_rerunning_them(tmp_path):
    calls = []

    def objective(_trial):
        calls.append(True)
        raise optuna.TrialPruned("screen failed")

    config = HPOConfig(n_trials=1, timeout_seconds=None)
    context = _hpo_context()
    with pytest.raises(RuntimeError, match="produced no completed trials"):
        run_study(
            "hpo_restart", objective, config, tmp_path, seed=0, drop_keys=(), hpo_context=context
        )
    with pytest.raises(RuntimeError, match="produced no completed trials"):
        run_study(
            "hpo_restart", objective, config, tmp_path, seed=0, drop_keys=(), hpo_context=context
        )

    assert len(calls) == 1
    study = optuna.load_study(
        study_name=contextual_study_name("hpo_restart", context),
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
    context = _hpo_context()
    run_study(
        "hpo_checkpoint", objective, config, tmp_path, seed=0, drop_keys=(), hpo_context=context
    )

    checkpoint_path = (
        tmp_path
        / "hpo_checkpoints"
        / contextual_study_name("hpo_checkpoint", context)
        / "trial-0"
        / "checkpoint.json"
    )
    checkpoint = load_hpo_trial_checkpoint(checkpoint_path, hpo_context=context)
    assert checkpoint["schema_version"] == HPO_TRIAL_CHECKPOINT_SCHEMA_VERSION
    assert checkpoint["state"] == "complete"
    assert checkpoint["objective_value"] == pytest.approx(0.25)

    def should_not_run(_trial):
        raise AssertionError("a completed HPO trial was recomputed during resume")

    run_study(
        "hpo_checkpoint",
        should_not_run,
        config,
        tmp_path,
        seed=0,
        drop_keys=(),
        hpo_context=context,
    )

    assert calls == [True]
    assert load_hpo_trial_checkpoint(checkpoint_path, hpo_context=context) == checkpoint


def test_resuming_recovers_stale_running_trial_without_extra_allocation(tmp_path):
    config = HPOConfig(n_trials=2, timeout_seconds=None)
    context = _hpo_context()
    study = hpo_module.create_study(
        "hpo_stale_running", config, tmp_path, seed=0, hpo_context=context
    )
    study.optimize(lambda _trial: 0.25, n_trials=1)
    stale = study.ask()
    stale.suggest_float("learning_rate", 0.1, 0.2)
    stale_number = stale.number

    calls = []
    result = run_study(
        "hpo_stale_running",
        lambda _trial: calls.append(True) or 0.5,
        config,
        tmp_path,
        seed=0,
        drop_keys=(),
        hpo_context=context,
    )

    resumed = optuna.load_study(
        study_name=contextual_study_name("hpo_stale_running", context),
        storage=default_storage_url(tmp_path),
    )
    recovered = resumed.trials[stale_number]
    assert result == {}
    assert calls == []
    assert len(resumed.trials) == 2
    assert recovered.state == optuna.trial.TrialState.FAIL
    assert recovered.params == stale.params
    assert recovered.user_attrs["hpo_running_recovery"] == {
        "schema_version": "hpo-running-recovery-v1",
        "study_name": resumed.study_name,
        "context_digest": hpo_context_digest(context),
        "trial_number": stale_number,
        "original_state": "RUNNING",
        "reason_code": "stale_running_trial_recovery",
        "terminal_state": "FAIL",
    }


def test_stale_running_recovery_is_idempotent(tmp_path):
    config = HPOConfig(n_trials=1, timeout_seconds=None)
    context = _hpo_context()
    study = hpo_module.create_study(
        "hpo_stale_idempotent", config, tmp_path, seed=0, hpo_context=context
    )
    stale = study.ask()
    stale.suggest_int("depth", 1, 2)

    with pytest.raises(RuntimeError, match="produced no completed trials"):
        run_study(
            "hpo_stale_idempotent",
            lambda _trial: 0.5,
            config,
            tmp_path,
            seed=0,
            drop_keys=(),
            hpo_context=context,
        )
    first = optuna.load_study(
        study_name=contextual_study_name("hpo_stale_idempotent", context),
        storage=default_storage_url(tmp_path),
    ).trials[0]

    with pytest.raises(RuntimeError, match="produced no completed trials"):
        run_study(
            "hpo_stale_idempotent",
            lambda _trial: pytest.fail("recovery must not allocate a trial"),
            config,
            tmp_path,
            seed=0,
            drop_keys=(),
            hpo_context=context,
        )
    second = optuna.load_study(
        study_name=contextual_study_name("hpo_stale_idempotent", context),
        storage=default_storage_url(tmp_path),
    ).trials[0]
    assert second.state == optuna.trial.TrialState.FAIL
    assert second.params == first.params
    assert second.user_attrs == first.user_attrs


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
    context = _hpo_context()
    run_study(
        "hpo_fingerprint_checkpoint",
        objective,
        config,
        tmp_path,
        seed=0,
        drop_keys=(),
        checkpoint_implementation_fingerprint="implementation-a",
        hpo_context=context,
    )

    checkpoint_path = (
        tmp_path
        / "hpo_checkpoints"
        / contextual_study_name("hpo_fingerprint_checkpoint", context)
        / "trial-0"
        / "checkpoint.json"
    )
    checkpoint = load_hpo_trial_checkpoint(
        checkpoint_path,
        hpo_context=context,
        expected_implementation_fingerprint="implementation-a",
    )
    assert checkpoint["metadata"]["generator"]["implementation_fingerprint"] == "implementation-a"

    with pytest.raises(RuntimeError, match="does not match the current implementation"):
        load_hpo_trial_checkpoint(
            checkpoint_path,
            hpo_context=context,
            expected_implementation_fingerprint="implementation-b",
        )

    def should_not_run(_trial):
        raise AssertionError("a checkpoint from another implementation was reused")

    with pytest.raises(RuntimeError, match="does not match the current implementation"):
        run_study(
            "hpo_fingerprint_checkpoint",
            should_not_run,
            config,
            tmp_path,
            seed=0,
            drop_keys=(),
            checkpoint_implementation_fingerprint="implementation-b",
            hpo_context=context,
        )
    assert calls == [True]


def test_legacy_hpo_generator_metadata_is_rejected_as_current_evidence(tmp_path):
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
        trial.set_user_attr("generator_implementation_fingerprint", "implementation-a")
        return 0.25

    study.optimize(objective, n_trials=1)
    context = _hpo_context()
    with pytest.raises(RuntimeError, match="unsupported schema"):
        persist_hpo_trial_checkpoint(
            tmp_path / "legacy_checkpoints",
            "legacy_hpo",
            study.trials[0],
            hpo_context=context,
            expected_implementation_fingerprint="implementation-a",
        )


def test_durable_checkpoint_rejects_legacy_metadata_with_valid_context():
    context = _hpo_context()
    payload = {
        "schema_version": HPO_TRIAL_CHECKPOINT_SCHEMA_VERSION,
        "study_name": "legacy_hpo",
        "trial_number": 0,
        "state": "complete",
        "objective_value": 0.25,
        "params": {},
        "hpo_context": context,
        "hpo_context_digest": hpo_context_digest(context),
        "metadata": {
            "generator": {
                "state": "present",
                "plugin_name": "test_generator",
                "privacy_claim_type": "none",
                "implementation_fingerprint": "implementation-a",
                "metadata": {
                    "schema_version": "generator-metadata-v1",
                    "generator_context": {"privacy_claim_type": "none"},
                    "plugin_name": "test_generator",
                    "plugin_fqdn": "test.generator",
                    "requested_parameters": {},
                    "n_samples": 1,
                    "random_state": 0,
                    "privacy_accounting": None,
                },
            },
            "stage_a": {},
        },
    }
    with pytest.raises(RuntimeError, match="unsupported schema"):
        hpo_module._validate_hpo_trial_checkpoint(
            payload,
            expected_study_name="legacy_hpo",
            expected_context_digest=hpo_context_digest(context),
            expected_implementation_fingerprint="implementation-a",
        )


def test_failed_hpo_checkpoint_persists_exception_context(tmp_path):
    sentinel = "generator fit failed SECRET_VALUE /tmp/raw-path"

    def objective(_trial):
        raise RuntimeError(sentinel)

    config = HPOConfig(n_trials=1, timeout_seconds=None)
    context = _hpo_context()
    with pytest.raises(RuntimeError, match="generator fit failed"):
        run_study(
            "hpo_failed_checkpoint",
            objective,
            config,
            tmp_path,
            seed=0,
            drop_keys=(),
            hpo_context=context,
        )

    checkpoint = load_hpo_trial_checkpoint(
        tmp_path
        / "hpo_checkpoints"
        / contextual_study_name("hpo_failed_checkpoint", context)
        / "trial-0"
        / "checkpoint.json",
        hpo_context=context,
    )
    assert checkpoint["state"] == "failed"
    assert checkpoint["metadata"]["outcome"] == {
        "error_type": "RuntimeError",
        "error_message": "HPO trial failed; exception details suppressed.",
        "error_reason_code": "hpo_trial_exception",
        "state": "failed",
    }
    assert sentinel not in json.dumps(checkpoint)


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


def test_hpo_score_rejects_diagnostic_or_failed_rows():
    diagnostic = pd.DataFrame(
        {"mean": [0.5], "direction": _strings("maximize")},
        index=_strings("stats.prdc"),
    )
    with pytest.raises(ValueError, match="not decision-eligible"):
        _score(diagnostic)

    failed = pd.DataFrame(
        {
            "mean": [float("nan")],
            "direction": _strings("minimize"),
            "errors": [1],
            "error_types": ["ValueError"],
        },
        index=_strings("mixed_mmd.v1"),
    )
    with pytest.raises(ValueError, match="not decision-eligible"):
        _score(failed)


def test_hpo_score_rejects_partial_static_metric_set():
    report = pd.DataFrame(
        {
            "mean": [0.25],
            "direction": _strings("minimize"),
        },
        index=_strings("mixed_mmd.v1"),
    )

    with pytest.raises(ValueError, match="incomplete metric set.*missing"):
        _score(
            report,
            expected_keys=(
                "mixed_mmd.v1",
                "elastic_net_jsd.v1",
            ),
        )


def test_canonical_hpo_partial_report_is_indeterminate():
    report = pd.DataFrame(
        {"mean": [0.25], "direction": _strings("minimize")}, index=_strings("mixed_mmd.v1")
    )
    report.attrs["canonical_hpo"] = True
    with pytest.raises(ValueError, match="incomplete metric set"):
        _score(report)


def test_hpo_score_rejects_duplicate_static_metric_set():
    metric_key = "mixed_mmd.v1"
    report = pd.DataFrame(
        {
            "mean": [0.25, 0.3],
            "direction": _strings("minimize", "minimize"),
        },
        index=_strings(metric_key, metric_key),
    )

    with pytest.raises(ValueError, match="incomplete metric set.*duplicate"):
        _score(report, expected_keys=(metric_key,))


def test_hpo_score_accepts_only_approved_operational_rows():
    report = pd.DataFrame(
        {
            "mean": [0.25, 0.75, 0.5],
            "direction": _strings("minimize", "maximize", "minimize"),
        },
        index=_strings("mixed_mmd.v1", "tstr_macro_f1.v1", "elastic_net_jsd.v1"),
    )

    assert _score(report) == pytest.approx(-(0.75 + 0.75 + 0.5) / 3)


def test_hpo_score_rejects_extra_metric_rows_without_expected_keys():
    report = pd.DataFrame(
        {
            "mean": [0.25, 0.75, 0.5, 0.1],
            "direction": _strings("minimize", "maximize", "minimize", "minimize"),
        },
        index=_strings("mixed_mmd.v1", "tstr_macro_f1.v1", "elastic_net_jsd.v1", "unexpected.v1"),
    )

    with pytest.raises(ValueError, match="not decision-eligible"):
        _score(report)


def test_hpo_score_uses_fixed_release_formula_and_jsd_safety_orientation():
    report = pd.DataFrame(
        {
            "mean": [0.8, 0.2, 0.4],
            "direction": _strings("maximize", "minimize", "minimize"),
        },
        index=_strings("tstr_macro_f1.v1", "mixed_mmd.v1", "elastic_net_jsd.v1"),
    )
    assert _score(report) == pytest.approx(-((0.8 + 0.8 + 0.6) / 3))
    with pytest.raises(ValueError, match="fixed to equal thirds"):
        _score(report, utility_policy={"metrics": ["mixed_mmd.v1"], "weights": [1.0]})


def test_hpo_score_rejects_legacy_two_metric_utility_policy():
    report = pd.DataFrame(
        {
            "mean": [0.8, 0.2, 0.4],
            "direction": _strings("maximize", "minimize", "minimize"),
        },
        index=_strings("tstr_macro_f1.v1", "mixed_mmd.v1", "elastic_net_jsd.v1"),
    )

    with pytest.raises(ValueError, match="fixed to equal thirds"):
        _score(
            report,
            utility_policy={
                "metrics": ["mixed_mmd.v1", "elastic_net_jsd.v1"],
                "weights": [0.5, 0.5],
            },
        )


def test_hpo_score_accepts_canonical_equal_thirds_utility_policy():
    report = pd.DataFrame(
        {
            "mean": [0.8, 0.2, 0.4],
            "direction": _strings("maximize", "minimize", "minimize"),
        },
        index=_strings("tstr_macro_f1.v1", "mixed_mmd.v1", "elastic_net_jsd.v1"),
    )

    assert _score(
        report,
        utility_policy={
            "metrics": ["tstr_macro_f1.v1", "mixed_mmd.v1", "elastic_net_jsd.v1"],
            "weights": [1 / 3, 1 / 3, 1 / 3],
        },
    ) == pytest.approx(-((0.8 + 0.8 + 0.6) / 3))


def test_hpo_score_is_invariant_to_candidate_metric_order():
    report = pd.DataFrame(
        {
            "mean": [0.4, 0.8, 0.2],
            "direction": _strings("minimize", "maximize", "minimize"),
        },
        index=_strings("elastic_net_jsd.v1", "tstr_macro_f1.v1", "mixed_mmd.v1"),
    )
    reordered = report.iloc[[2, 1, 0]]

    assert _score(report) == pytest.approx(-((0.6 + 0.8 + 0.8) / 3))
    assert _score(reordered) == pytest.approx(_score(report))


def test_train_frozen_mmd_uses_train_for_fit_and_tuning_for_comparison():
    train = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    tuning = pd.DataFrame({"x": [100.0, 101.0, 102.0]})
    candidate = pd.DataFrame({"x": [100.0, 101.0, 102.0]})
    result = hpo_module._evaluate_train_frozen_mmd(
        train,
        tuning,
        candidate,
        continuous_columns=["x"],
        ordinal_columns=[],
        nominal_columns=[],
    )
    assert result["fit_role"] == "train"
    assert result["comparison_role"] == "tuning"
    assert result["bandwidth"] == pytest.approx(0.5)
    changed_tuning = tuning.assign(x=[200.0, 201.0, 202.0])
    changed = hpo_module._evaluate_train_frozen_mmd(
        train,
        changed_tuning,
        candidate,
        continuous_columns=["x"],
        ordinal_columns=[],
        nominal_columns=[],
    )
    assert changed["bandwidth"] == pytest.approx(result["bandwidth"])
    assert changed["b_mmd_clip"] != pytest.approx(result["b_mmd_clip"])


def test_canonical_jsd_uses_tuning_for_candidate_evidence_not_train():
    train = pd.DataFrame({"x": ["a", "b", "c"]})
    tuning = pd.DataFrame({"x": ["a", "a", "b"]})
    candidate = tuning.copy()

    first = hpo_module.evaluate_canonical_hpo_metrics(
        train,
        tuning,
        candidate,
        metric_config={"task12": ["elastic_net_jsd.v1"]},
        target_column="x",
        feature_types={"x": "categorical"},
    )
    changed_tuning = tuning.assign(x=["c", "c", "c"])
    second = hpo_module.evaluate_canonical_hpo_metrics(
        train,
        changed_tuning,
        candidate,
        metric_config={"task12": ["elastic_net_jsd.v1"]},
        target_column="x",
        feature_types={"x": "categorical"},
    )

    assert first.loc["elastic_net_jsd.v1", "fit_roles"] == ["train"]
    assert first.loc["elastic_net_jsd.v1", "evaluation_role"] == "tuning"
    assert second.loc["elastic_net_jsd.v1", "mean"] != pytest.approx(
        first.loc["elastic_net_jsd.v1", "mean"]
    )


def test_hpo_context_is_invariant_to_candidate_order_and_excludes_holdout():
    context = _hpo_context(
        objective_context={
            "role_hashes": {"train": "train", "tuning": "tuning"},
            "excluded_roles": ["final_holdout"],
        }
    )
    reordered = {
        **context,
        "role_context": {"roles": {"tuning": {"rows": 2}, "train": {"rows": 4}}},
    }
    assert hpo_context_digest(context) == hpo_context_digest(reordered)


def test_canonical_hpo_rejects_final_holdout_at_input_boundary():
    train = pd.DataFrame({"x": [0.0, 1.0], "target": [0, 1]})
    tuning = pd.DataFrame({"x": [0.0, 1.0], "target": [0, 1]})
    synthetic = pd.DataFrame({"x": [0.2, 0.8], "target": [0, 1]})
    tuning.attrs["source_role"] = "final_holdout"

    with pytest.raises(ValueError, match="final_holdout"):
        hpo_module.evaluate_canonical_hpo_metrics(
            train,
            tuning,
            synthetic,
            metric_config={"task12": ["mixed_mmd.v1"]},
            target_column="target",
            feature_types={"x": "continuous", "target": "categorical"},
        )


def test_canonical_hpo_rejects_tampered_release_generalization_metadata():
    frames = [pd.DataFrame({"x": [0.0, 1.0], "target": [0, 1]}) for _ in range(3)]

    with pytest.raises(ValueError, match="forbidden"):
        hpo_module.evaluate_canonical_hpo_metrics(
            *frames,
            metric_config={"task12": ["mixed_mmd.v1"]},
            target_column="target",
            feature_types={"x": "continuous", "target": "categorical"},
            release_generalization={
                "columns": {"x": {"nested": {"evaluation": {"role": "tuning"}}}}
            },
        )


@pytest.mark.parametrize(
    "provenance",
    [
        {"release_provenance": {"source_role": "final_holdout"}},
        {"release_provenance": {"evaluation": {"role": "hidden"}}},
        {"release_provenance": {"privacy": {"enabled": True}}},
        {"release_provenance": {"fairness": {"role": "tuning"}}},
        {"release_provenance": {"split": {"role": "tuning"}}},
    ],
)
def test_canonical_hpo_rejects_nested_release_provenance(provenance):
    frames = [pd.DataFrame({"x": [0.0, 1.0], "target": [0, 1]}) for _ in range(3)]
    frames[1].attrs.update(provenance)

    with pytest.raises(ValueError, match="forbidden|unsupported"):
        hpo_module.evaluate_canonical_hpo_metrics(
            *frames,
            metric_config={"task12": ["mixed_mmd.v1"]},
            target_column="target",
            feature_types={"x": "continuous", "target": "categorical"},
        )


def test_hpo_context_digest_is_invariant_to_input_order_and_holdout_contents():
    context = _hpo_context(
        role_context={"roles": {"train": {"rows": 4}, "tuning": {"rows": 2}}},
        objective_context={
            "role_hashes": ["train", "tuning"],
            "excluded_roles": ["final_holdout"],
            "holdout": {"rows": [1, 2, 3]},
        },
    )
    reordered = {
        **context,
        "objective_context": {
            "role_hashes": ["tuning", "train"],
            "excluded_roles": ["final_holdout"],
            "holdout": {"rows": [999]},
        },
    }
    assert hpo_context_digest(context) == hpo_context_digest(reordered)


def test_train_frozen_mmd_is_invariant_to_row_order():
    train = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    tuning = pd.DataFrame({"x": [100.0, 101.0, 102.0]})
    candidate = pd.DataFrame({"x": [100.0, 101.0, 102.0]})
    reordered = hpo_module._evaluate_train_frozen_mmd(
        train.iloc[[2, 0, 1]],
        tuning.iloc[[1, 2, 0]],
        candidate.iloc[[2, 1, 0]],
        continuous_columns=["x"],
        ordinal_columns=[],
        nominal_columns=[],
    )
    original = hpo_module._evaluate_train_frozen_mmd(
        train,
        tuning,
        candidate,
        continuous_columns=["x"],
        ordinal_columns=[],
        nominal_columns=[],
    )
    assert reordered["bandwidth"] == pytest.approx(original["bandwidth"])
    assert reordered["b_mmd_clip"] == pytest.approx(original["b_mmd_clip"])


def test_canonical_hpo_rejects_empty_or_privacy_metric_config():
    frames = [pd.DataFrame({"x": [0.0], "target": [0]})] * 3
    for metric_config in ({}, {"privacy": ["identifiability_score"]}):
        with pytest.raises(ValueError):
            hpo_module.evaluate_canonical_hpo_metrics(
                *frames,
                metric_config=metric_config,
                target_column="target",
            )


def test_canonical_metric_report_contains_protocol_and_orientation_provenance():
    train = pd.DataFrame({"x": [0.0, 1.0], "target": [0, 1]})
    tuning = train.copy()
    synthetic = pd.DataFrame({"x": [0.2, 0.8], "target": [0, 1]})
    report = hpo_module.evaluate_canonical_hpo_metrics(
        train,
        tuning,
        synthetic,
        metric_config={"task12": ["mixed_mmd.v1"]},
        target_column="target",
        feature_types={"x": "continuous", "target": "categorical"},
    )
    assert report.loc["mixed_mmd.v1", "orientation"] == "minimize_distance"
    provenance = report.loc["mixed_mmd.v1", "provenance"]
    assert provenance["fit_roles"] == ["train"]
    assert provenance["comparison_role"] == "tuning"
    assert provenance["release_transform_digest"]
    assert provenance["common_protocol_digest"]
    assert set(provenance["role_hashes"]) == {"synthetic", "train", "tuning"}
    assert report.loc["mixed_mmd.v1", "objective_version"]


def test_default_canonical_report_scores_with_complete_role_provenance():
    train = pd.DataFrame({"x": np.arange(20, dtype=float), "target": [0, 1] * 10})
    tuning = pd.DataFrame({"x": np.arange(20, 40, dtype=float), "target": [0, 1] * 10})
    synthetic = train.copy()
    report = hpo_module.evaluate_canonical_hpo_metrics(
        train,
        tuning,
        synthetic,
        metric_config={
            "task12": [
                "tstr_macro_f1.v1",
                "mixed_mmd.v1",
                "elastic_net_jsd.v1",
            ]
        },
        target_column="target",
        feature_types={"x": "continuous", "target": "categorical"},
    )

    provenance = report.attrs["hpo_provenance"]
    assert set(provenance["role_hashes"]) == {"synthetic", "train", "tuning"}
    assert provenance["role_hashes"]["train"]
    assert provenance["role_hashes"]["tuning"]
    assert provenance["contracts"]["fit_roles"] == ["train"]
    assert provenance["contracts"]["comparison_role"] == "tuning"
    assert np.isfinite(hpo_score(report))


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
    context = _hpo_context()

    run_study(
        "hpo_tabpfgen_checkpoint_metadata",
        objective,
        config,
        tmp_path,
        seed=13,
        drop_keys=(),
        hpo_context=context,
    )

    checkpoint_path = (
        tmp_path
        / "hpo_checkpoints"
        / contextual_study_name("hpo_tabpfgen_checkpoint_metadata", context)
        / "trial-0"
        / "checkpoint.json"
    )
    checkpoint = load_hpo_trial_checkpoint(checkpoint_path, hpo_context=context)
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
        raise AssertionError("a completed TabPFGen trial was recomputed during resume")

    run_study(
        "hpo_tabpfgen_checkpoint_metadata",
        should_not_run,
        config,
        tmp_path,
        seed=13,
        drop_keys=(),
        hpo_context=context,
    )
    assert len(calls) == 1
    assert load_hpo_trial_checkpoint(checkpoint_path, hpo_context=context) == checkpoint


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
        hpo_context=_hpo_context(),
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
        hpo_context=_hpo_context(),
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
            hpo_context=_hpo_context(),
        )


@pytest.mark.parametrize(
    "legacy",
    [
        "stats.wasserstein_dist",
        "stats.inv_kl_divergence",
        "sanity.nearest_syn_neighbor_distance",
        "performance.xgb",
    ],
)
def test_legacy_hpo_objectives_are_rejected(legacy):
    category, metric = legacy.split(".", 1)
    with pytest.raises(ValueError, match="canonical HPO allowlist"):
        validate_hpo_metric_config({category: [metric]})


def test_canonical_tstr_hpo_fails_closed():
    from synthdata.generation.hpo import evaluate_canonical_hpo_metrics

    report = evaluate_canonical_hpo_metrics(
        pd.DataFrame({"target": [0, 1]}),
        pd.DataFrame({"target": [0, 1]}),
        pd.DataFrame({"target": [0, 1]}),
        metric_config={"task12": ["tstr_macro_f1.v1"]},
        target_column="target",
    )
    assert set(report.index) == {
        "tstr_macro_f1.v1",
        "mixed_mmd.v1",
        "elastic_net_jsd.v1",
    }
    assert report.loc["tstr_macro_f1.v1", "errors"] == 1


def test_canonical_tabpfgen_eval_uses_canonical_producer_not_native_metrics(mocker):
    train = pd.DataFrame({"feature": [0.0, 1.0], "target": [0, 1]})
    tuning = train.copy()
    candidate = train.copy()
    canonical = mocker.patch(
        "synthdata.generation.hpo.evaluate_canonical_hpo_metrics",
        return_value=pd.DataFrame(
            {"mean": [0.25, 0.5, 0.75], "direction": _strings("minimize", "minimize", "maximize")},
            index=_strings("mixed_mmd.v1", "elastic_net_jsd.v1", "tstr_macro_f1.v1"),
        ),
    )
    canonical.return_value.attrs["hpo_provenance"] = _hpo_context()
    native = mocker.patch("synthcity.metrics.Metrics.evaluate")

    evaluate = build_synthetic_eval_fn(
        train,
        tuning,
        "target",
        [],
        {"task12": ["mixed_mmd.v1"]},
        seed=0,
    )

    assert evaluate(candidate) == pytest.approx(-(0.75 + 0.5 + 0.75) / 3)
    canonical.assert_called_once()
    native.assert_not_called()
