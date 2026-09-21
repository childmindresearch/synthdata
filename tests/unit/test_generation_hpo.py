"""Unit tests for the explicit scope of resumable HPO artifacts."""

import enum
import json
import math
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest
import torch

from synthdata.config import HPOConfig, load_config
from synthdata.data import dataframe_fingerprint, semantic_context_payload
from synthdata.generation import hpo as hpo_module
from synthdata.generation import synthcity_backend as synthcity_backend_module
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
    normalize_hpo_metadata,
    persist_hpo_trial_checkpoint,
    persist_stage_a_contract,
    run_study,
    screen_stage_a,
    screen_stage_a_trial,
    validate_existing_stage_a_trial_result,
    validate_hpo_metric_config,
)
from synthdata.generation.synthcity_backend import (
    _validate_native_benchmark_report,
    build_synthcity_objective,
)
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
    canonical = hpo_module.is_canonical_hpo_context(
        context["metric_config"],
        expected_keys=context["expected_emitted_keys"],
        utility_policy=context["utility_policy"],
    )
    context["canonical_hpo"] = canonical
    context["canonical_expected_keys"] = list(context["expected_emitted_keys"]) if canonical else []
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


def _canonical_metric_metadata(*, failed: bool = False):
    """Build complete bounded metadata for canonical trial evidence."""
    metrics = ("mixed_mmd.v1", "elastic_net_jsd.v1", "tstr_macro_f1.v1")
    return {
        key: {
            "metric_name": key,
            "status": "failed" if failed else "complete",
            "direction": "maximize" if key == "tstr_macro_f1.v1" else "minimize",
            "finite": not failed,
            "eligible": not failed,
            "error_reason_code": "metric_evaluation_exception" if failed else None,
            "fit_roles": ["train"],
            "evaluation_role": "tuning",
        }
        for key in metrics
    }


def _canonical_hpo_context(**overrides):
    """Build context matching exact canonical objective identities."""
    context = _hpo_context(
        metric_config={
            "canonical_objectives": [
                "tstr_macro_f1.v1",
                "mixed_mmd.v1",
                "elastic_net_jsd.v1",
            ]
        },
        **overrides,
    )
    return context


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
        {
            "canonical_objectives": [
                "tstr_macro_f1.v1",
                "mixed_mmd.v1",
                "elastic_net_jsd.v1",
            ]
        },
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
        {
            "canonical_objectives": [
                "tstr_macro_f1.v1",
                "mixed_mmd.v1",
                "elastic_net_jsd.v1",
            ]
        },
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


def test_patient_group_ids_accepts_aligned_index_for_range_index_frame():
    frame = pd.DataFrame({"x": [0.0, 1.0], "target": [0, 1]})

    assert hpo_module._validate_aligned_group_ids(frame, pd.Index(["p1", "p2"]), "train") == [
        "p1",
        "p2",
    ]


def test_patient_group_ids_index_rejects_non_range_frame_as_group_unsafe():
    train = pd.DataFrame({"x": [0.0, 1.0], "target": [0, 1]}, index=pd.Index(["a", "b"]))
    tuning = pd.DataFrame({"x": [2.0, 3.0], "target": [0, 1]}, index=pd.Index(["c", "d"]))

    with pytest.raises(hpo_module.HPOGroupUnsafeError, match="contract is malformed"):
        hpo_module.evaluate_canonical_hpo_metrics(
            train,
            tuning,
            train.copy(),
            metric_config={"canonical_objectives": list(HPO_OBJECTIVE_METRICS)},
            target_column="target",
            group_context=_patient_group_contract(train, tuning, ["p1", "p2"], ["p3", "p4"]),
            train_group_ids=pd.Index(["p1", "p2"]),
            tuning_group_ids=pd.Index(["p3", "p4"]),
        )


def _patient_group_contract(train, tuning, train_groups, tuning_groups):
    """Build bounded group evidence matching canonical evaluator inputs."""
    return {
        "group_mode": "patient_group",
        "roles": {
            role: {
                "rows": len(values),
                "groups": len(set(values)),
                "fingerprint": dataframe_fingerprint(pd.DataFrame({"group_id": values})),
                "source": "dataset_role_groups",
            }
            for role, values in (("train", train_groups), ("tuning", tuning_groups))
        },
    }


def test_canonical_patient_group_evaluator_emits_affirmative_bounded_contract():
    train = pd.DataFrame({"x": [0.0, 1.0], "target": [0, 1]})
    tuning = pd.DataFrame({"x": [2.0, 3.0], "target": [0, 1]})
    groups = ["p1", "p2"]
    train_groups = dict(zip(train.index, groups, strict=True))
    tuning_groups = dict(zip(tuning.index, ["p3", "p4"], strict=True))
    report = hpo_module.evaluate_canonical_hpo_metrics(
        train,
        tuning,
        train.copy(),
        metric_config={"canonical_objectives": list(HPO_OBJECTIVE_METRICS)},
        target_column="target",
        feature_types={"x": "continuous", "target": "categorical"},
        group_context=_patient_group_contract(train, tuning, groups, ["p3", "p4"]),
        train_group_ids=train_groups,
        tuning_group_ids=tuning_groups,
    )
    safety = report.attrs["group_safety"]
    assert safety["schema_version"] == "group-safety-v1"
    assert safety["status"] == "group_safe"
    assert safety["group_mode"] == "patient_group"
    assert set(safety["roles"]) == {"train", "tuning"}
    assert all("fingerprint" in evidence for evidence in safety["roles"].values())
    assert "p1" not in json.dumps(report.attrs)
    assert "p3" not in json.dumps(report.attrs)


@pytest.mark.parametrize(
    "contract_mutation",
    [
        lambda contract: contract.pop("roles"),
        lambda contract: contract["roles"]["train"].update({"groups": 99}),
        lambda contract: contract["roles"]["tuning"].update({"source": "raw_ids"}),
    ],
)
def test_canonical_patient_group_missing_or_contradictory_contract_fails_closed(contract_mutation):
    train = pd.DataFrame({"x": [0.0, 1.0], "target": [0, 1]})
    tuning = pd.DataFrame({"x": [2.0, 3.0], "target": [0, 1]})
    contract = _patient_group_contract(train, tuning, ["p1", "p2"], ["p3", "p4"])
    contract_mutation(contract)
    with pytest.raises(hpo_module.HPOGroupUnsafeError):
        hpo_module.evaluate_canonical_hpo_metrics(
            train,
            tuning,
            train.copy(),
            metric_config={"canonical_objectives": list(HPO_OBJECTIVE_METRICS)},
            target_column="target",
            group_context=contract,
            train_group_ids=dict(zip(train.index, ["p1", "p2"], strict=True)),
            tuning_group_ids=dict(zip(tuning.index, ["p3", "p4"], strict=True)),
        )


def test_native_report_rejects_provenance_and_metadata_leakage():
    report = pd.DataFrame(
        {"mean": [0.2], "direction": ["minimize"]}, index=_strings("mixed_mmd.v1")
    )
    report.attrs["hpo_provenance"] = {"role_hashes": {"train": "x"}}
    report.attrs["metric_metadata"] = {
        "mixed_mmd.v1": {
            "metric_name": "mixed_mmd.v1",
            "status": "complete",
            "exception": "SECRET /tmp/raw-path",
        }
    }
    with pytest.raises(hpo_module.HPOMetricNotEligibleError):
        _validate_native_benchmark_report({"trial_0": report}, "trial_0")


def test_native_report_accepts_valid_bounded_metadata_contract():
    report = pd.DataFrame(
        {"mean": [0.2], "direction": ["minimize"]}, index=_strings("mixed_mmd.v1")
    )
    report.attrs["hpo_provenance"] = {
        "release_transform_digest": "release-a",
        "role_hashes": {"train": "train-a", "tuning": "tuning-a"},
        "contracts": {
            "fit_roles": ["train"],
            "comparison_role": "tuning",
            "excluded_roles": ["final_holdout"],
        },
        "objective_version": "native-v1",
    }
    report.attrs["metric_metadata"] = {
        "mixed_mmd.v1": {
            "metric_name": "mixed_mmd.v1",
            "status": "complete",
            "finite": True,
            "eligible": True,
            "error_reason_code": None,
        }
    }
    report.attrs["result_metadata"] = dict(report.attrs["metric_metadata"])
    assert _validate_native_benchmark_report({"trial_0": report}, "trial_0") is report


def _native_group_safety(status="group_safe"):
    return {
        "schema_version": "group-safety-v1",
        "status": status,
        "group_mode": "patient_group",
        "roles": {
            role: {
                "rows": 4,
                "groups": 4,
                "fingerprint": "a" * 64 if role == "train" else "b" * 64,
                "source": "dataset_role_groups",
            }
            for role in ("train", "tuning")
        },
    }


def test_native_group_unsafe_requires_complete_bounded_contract():
    report = pd.DataFrame(
        {"mean": [0.2], "direction": ["minimize"]}, index=_strings("mixed_mmd.v1")
    )
    report.attrs["group_safety"] = _native_group_safety("group_unsafe")
    provenance = _provenance_kwargs()
    provenance["contracts"] = {
        key: provenance["contracts"][key]
        for key in ("fit_roles", "comparison_role", "excluded_roles")
    }
    report.attrs["hpo_provenance"] = {
        key: provenance[key]
        for key in ("release_transform_digest", "role_hashes", "contracts", "objective_version")
    }
    report.attrs["metric_metadata"] = {
        "mixed_mmd.v1": {
            "metric_name": "mixed_mmd.v1",
            "status": "complete",
            "finite": True,
            "eligible": True,
            "error_reason_code": None,
        }
    }
    report.attrs["result_metadata"] = dict(report.attrs["metric_metadata"])
    assert _validate_native_benchmark_report({"trial_0": report}, "trial_0") is report


def test_native_group_unsafe_malformed_contract_fails_closed():
    report = pd.DataFrame(
        {"mean": [0.2], "direction": ["minimize"]}, index=_strings("mixed_mmd.v1")
    )
    report.attrs["group_safety"] = {"status": "group_unsafe"}
    with pytest.raises(hpo_module.HPOGroupUnsafeError):
        _validate_native_benchmark_report({"trial_0": report}, "trial_0")


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
    config = HPOConfig(
        n_trials=1,
        timeout_seconds=None,
        metric_config={"task12": ["mixed_mmd.v1"]},
    )
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


def test_stage_a_exception_artifact_roundtrips_with_null_fingerprint(tmp_path):
    source = _stage_a_source()
    contract = _stage_a_contract(source)
    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    result = hpo_module._stage_a_exception_result(None, contract, ValueError("hidden"))
    result_path = hpo_module.persist_stage_a_result(
        tmp_path, "exception_roundtrip", trial.number, result
    )
    payload = json.loads(result_path.read_text())

    hpo_module._validate_stage_a_result_artifact(
        payload,
        root=tmp_path,
        context={"stage_a_contract_digest": contract.digest},
        expected_study_name="exception_roundtrip",
        expected_trial_number=trial.number,
    )
    assert payload["candidate_frame_fingerprint"] is None


def test_stage_a_result_persistence_accepts_relative_result_path(tmp_path):
    source = _stage_a_source()
    contract = _stage_a_contract(source)
    result = hpo_module._stage_a_exception_result(None, contract, ValueError("hidden"))
    root = tmp_path.resolve()
    result_path = root / "relative_result" / "trial-0" / "result.json"
    relative_path = result_path.relative_to(Path.cwd())
    result_path.parent.mkdir(parents=True)
    payload = {"study_name": "relative_result", "trial_number": 0, **result.to_dict()}
    result_path.write_text(json.dumps(payload))

    assert hpo_module._read_stage_a_json_pinned(root, relative_path) == payload


def test_stage_a_first_write_validates_persisted_artifact(tmp_path, mocker):
    source = _stage_a_source()
    contract = _stage_a_contract(source)
    result = hpo_module._stage_a_exception_result(None, contract, ValueError("hidden"))

    def write_malformed(directory_fd, name, _payload):
        malformed_fd = hpo_module.os.open(
            name, hpo_module.os.O_WRONLY | hpo_module.os.O_CREAT, 0o600, dir_fd=directory_fd
        )
        try:
            hpo_module.os.write(malformed_fd, b"{")
        finally:
            hpo_module.os.close(malformed_fd)

    mocker.patch.object(hpo_module, "_atomic_stage_a_json_fd", side_effect=write_malformed)

    with pytest.raises(RuntimeError, match="is unreadable"):
        hpo_module.persist_stage_a_result(tmp_path, "first_write", 0, result)


def test_stage_a_existing_result_read_rejects_symlinked_trial_directory(tmp_path):
    result = hpo_module._stage_a_exception_result(
        None, _stage_a_contract(_stage_a_source()), ValueError()
    )
    study_root = tmp_path / "existing_read"
    target = tmp_path / "outside"
    target.mkdir()
    (target / "result.json").write_text(json.dumps({"not": "stage-a"}))
    study_root.mkdir()
    (study_root / "trial-0").symlink_to(target, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink path|is unreadable"):
        hpo_module.persist_stage_a_result(tmp_path, "existing_read", 0, result)


def test_stage_a_result_rejects_divergent_existing_payload(tmp_path):
    source = _stage_a_source()
    result = hpo_module._stage_a_exception_result(None, _stage_a_contract(source), ValueError())
    path = hpo_module.persist_stage_a_result(tmp_path, "divergent", 0, result)
    payload = json.loads(path.read_text())
    payload["trial_number"] = 1
    path.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="does not match the current result"):
        hpo_module.persist_stage_a_result(tmp_path, "divergent", 0, result)


def test_stage_a_contract_rejects_divergent_existing_payload(tmp_path):
    contract = _stage_a_contract(_stage_a_source())
    path = persist_stage_a_contract(tmp_path, contract)
    payload = json.loads(path.read_text())
    payload["digest"] = "different"
    path.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="does not match the current contract"):
        persist_stage_a_contract(tmp_path, contract)


@pytest.mark.parametrize("trial_number", [True, False, 1.0, "1", None])
def test_stage_a_result_rejects_non_integer_trial_number_before_filesystem_access(
    tmp_path, trial_number
):
    result = hpo_module._stage_a_exception_result(
        None, _stage_a_contract(_stage_a_source()), ValueError()
    )

    with pytest.raises(ValueError, match="trial_number"):
        hpo_module.persist_stage_a_result(tmp_path, "invalid_trial", trial_number, result)
    assert not (tmp_path / "invalid_trial").exists()


def test_stage_a_artifact_rejects_cross_trial_and_malformed_bounded_fields(tmp_path):
    payload = {
        "schema_version": hpo_module.STAGE_A_SCREEN_SCHEMA_VERSION,
        "study_name": "identity-study",
        "trial_number": 2,
        "contract_digest": "stage-a",
        "candidate_shape": [1, 1],
        "candidate_columns": ["feature"],
        "candidate_frame_fingerprint": "a" * 64,
        "state": "passed",
        "passed": True,
        "pruned": False,
        "checks": [],
        "prune_reasons": [],
    }
    result_path = tmp_path / "identity-study" / "trial-2" / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="trial_number does not match"):
        hpo_module._validate_stage_a_result_artifact(
            {
                "state": "passed",
                "contract_digest": "stage-a",
                "result_path": "identity-study/trial-2/result.json",
            },
            root=tmp_path,
            context={},
            expected_study_name="identity-study",
            expected_trial_number=3,
        )

    payload["candidate_shape"] = [0, 2]
    payload["candidate_columns"] = ["feature", "feature"]
    result_path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="invalid bounded fields|shape is inconsistent"):
        hpo_module._validate_stage_a_result_artifact(
            {
                "state": "passed",
                "contract_digest": "stage-a",
                "result_path": "identity-study/trial-2/result.json",
            },
            root=tmp_path,
            context={},
            expected_study_name="identity-study",
            expected_trial_number=2,
        )


def test_stage_a_artifact_accepts_realistic_wide_schema(tmp_path):
    columns = [f"feature_{index}" for index in range(665)]
    payload = {
        "schema_version": hpo_module.STAGE_A_SCREEN_SCHEMA_VERSION,
        "study_name": "wide-study",
        "trial_number": 0,
        "contract_digest": "stage-a",
        "candidate_shape": [40, 665],
        "candidate_columns": columns,
        "candidate_frame_fingerprint": "a" * 64,
        "state": "passed",
        "passed": True,
        "pruned": False,
        "checks": [],
        "prune_reasons": [],
    }
    result_path = tmp_path / "wide-study" / "trial-0" / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(payload))

    hpo_module._validate_stage_a_result_artifact(
        {
            "state": "passed",
            "contract_digest": "stage-a",
            "result_path": "wide-study/trial-0/result.json",
        },
        root=tmp_path,
        context={},
        expected_study_name="wide-study",
        expected_trial_number=0,
    )


def test_stage_a_artifact_accepts_slash_containing_categorical_checks(tmp_path):
    payload = {
        "schema_version": hpo_module.STAGE_A_SCREEN_SCHEMA_VERSION,
        "study_name": "r4-shaped",
        "trial_number": 0,
        "contract_digest": "stage-a",
        "candidate_shape": [2, 1],
        "candidate_columns": ["shelter"],
        "candidate_frame_fingerprint": "a" * 64,
        "state": "passed",
        "passed": True,
        "pruned": False,
        "checks": [
            {
                "screen": "shape_schema",
                "passed": True,
                "expected": {"categories": ["Shelter/Hostel"]},
                "observed": {"categories": ["Group Home/Assisted Living"]},
            }
        ],
        "prune_reasons": [],
    }
    result_path = tmp_path / "r4-shaped" / "trial-0" / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(payload))

    hpo_module._validate_stage_a_result_artifact(
        {
            "state": "passed",
            "contract_digest": "stage-a",
            "result_path": "r4-shaped/trial-0/result.json",
        },
        root=tmp_path,
        context={},
        expected_study_name="r4-shaped",
        expected_trial_number=0,
    )


def test_stage_a_artifact_rejects_unsafe_nested_check_path_values(tmp_path):
    payload = {
        "schema_version": hpo_module.STAGE_A_SCREEN_SCHEMA_VERSION,
        "study_name": "unsafe-check",
        "trial_number": 0,
        "contract_digest": "stage-a",
        "candidate_shape": [2, 1],
        "candidate_columns": ["shelter"],
        "candidate_frame_fingerprint": "a" * 64,
        "state": "passed",
        "passed": True,
        "pruned": False,
        "checks": [
            {
                "screen": "shape_schema",
                "passed": True,
                "observed": {"category": "../outside"},
            }
        ],
        "prune_reasons": [],
    }
    result_path = tmp_path / "unsafe-check" / "trial-0" / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="not normalized JSON"):
        hpo_module._validate_stage_a_result_artifact(
            {
                "state": "passed",
                "contract_digest": "stage-a",
                "result_path": "unsafe-check/trial-0/result.json",
            },
            root=tmp_path,
            context={},
            expected_study_name="unsafe-check",
            expected_trial_number=0,
        )


def test_stage_a_artifact_keeps_result_path_safety_validation(tmp_path):
    with pytest.raises(RuntimeError, match="result_path is unsafe"):
        hpo_module._validate_stage_a_result_artifact(
            {
                "state": "passed",
                "contract_digest": "stage-a",
                "result_path": "../outside/result.json",
            },
            root=tmp_path,
            context={},
        )


def test_existing_stage_a_result_is_validated_without_overwrite(tmp_path):
    source = _stage_a_source()
    contract = _stage_a_contract(source)
    original = hpo_module._stage_a_exception_result(None, contract, ValueError("first"))
    result_path = hpo_module.persist_stage_a_result(tmp_path, "immutable", 0, original)
    before = result_path.read_bytes()

    assert validate_existing_stage_a_trial_result(tmp_path, "immutable", 0, contract)
    assert result_path.read_bytes() == before


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
    result_path = tmp_path / "checkpoints" / "safe_paths" / "trial-0" / "result.json"
    result_path.parent.mkdir(parents=True)
    result_payload = json.loads((tmp_path / "safe_paths" / "trial-0" / "result.json").read_text())
    result_payload["contract_digest"] = "stage-a"
    result_path.write_text(json.dumps(result_payload))
    trial.set_user_attr("stage_a_contract_digest", "stage-a")
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


def test_checkpoint_first_write_uses_sibling_stage_a_root(tmp_path):
    context = _hpo_context()
    stage_a_root = tmp_path / "hpo_stage_a"
    checkpoint_root = tmp_path / "hpo_checkpoints"
    study_name = "sibling_stage"
    result_path = stage_a_root / study_name / "trial-0" / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "schema_version": hpo_module.STAGE_A_SCREEN_SCHEMA_VERSION,
                "study_name": study_name,
                "trial_number": 0,
                "contract_digest": "stage-a",
                "candidate_shape": [4, 3],
                "candidate_columns": ["age", "group", "target"],
                "candidate_frame_fingerprint": "a" * 64,
                "state": "passed",
                "passed": True,
                "pruned": False,
                "checks": [],
                "prune_reasons": [],
            }
        )
    )
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda _trial: 0.1, n_trials=1)
    trial = study.trials[0]
    trial.set_user_attr("stage_a_state", "passed")
    trial.set_user_attr("stage_a_contract_digest", "stage-a")
    trial.set_user_attr("stage_a_result_path", f"{study_name}/trial-0/result.json")

    checkpoint_path = persist_hpo_trial_checkpoint(
        checkpoint_root,
        study_name,
        trial,
        hpo_context=context,
        stage_a_root=stage_a_root,
    )

    assert checkpoint_path.exists()
    assert (
        load_hpo_trial_checkpoint(checkpoint_path, hpo_context=context, stage_a_root=stage_a_root)[
            "trial_number"
        ]
        == 0
    )


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("delete", "artifact is missing"),
        ("corrupt", "result is unreadable"),
        ("state", "state does not match"),
        ("contract", "contract digest does not match result"),
        ("fingerprint", "candidate fingerprint is invalid"),
        ("unsafe", "result_path is unsafe"),
        ("symlink", "uses a symlink path"),
    ],
)
def test_loading_rejects_tampered_stage_a_artifact(tmp_path, tamper, message):
    context = _hpo_context()
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: 0.25, n_trials=1)
    trial = study.trials[0]
    result_path = tmp_path / "hpo_checkpoints" / "stage_load" / "trial-0" / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "schema_version": hpo_module.STAGE_A_SCREEN_SCHEMA_VERSION,
                "study_name": "stage_load",
                "trial_number": 0,
                "contract_digest": "stage-a",
                "candidate_shape": [4, 3],
                "candidate_columns": ["age", "group", "target"],
                "candidate_frame_fingerprint": "a" * 64,
                "state": "passed",
                "passed": True,
                "pruned": False,
                "checks": [],
                "prune_reasons": [],
            }
        )
    )
    trial.set_user_attr("stage_a_state", "passed")
    trial.set_user_attr("stage_a_contract_digest", "stage-a")
    trial.set_user_attr("stage_a_result_path", "stage_load/trial-0/result.json")
    checkpoint_path = persist_hpo_trial_checkpoint(
        tmp_path / "hpo_checkpoints", "stage_load", trial, hpo_context=context
    )

    if tamper == "delete":
        result_path.unlink()
    elif tamper == "corrupt":
        result_path.write_text("{")
    elif tamper == "state":
        result = json.loads(result_path.read_text())
        result["state"] = "pruned"
        result_path.write_text(json.dumps(result))
    elif tamper == "contract":
        result = json.loads(result_path.read_text())
        result["contract_digest"] = "b" * 64
        result_path.write_text(json.dumps(result))
    elif tamper == "fingerprint":
        result = json.loads(result_path.read_text())
        result["candidate_frame_fingerprint"] = "not-a-fingerprint"
        result_path.write_text(json.dumps(result))
    elif tamper == "unsafe":
        checkpoint = json.loads(checkpoint_path.read_text())
        checkpoint["metadata"]["stage_a"]["result_path"] = "../escape.json"
        checkpoint_path.write_text(json.dumps(checkpoint))
    else:
        linked_path = tmp_path / "outside-result.json"
        linked_path.write_text(result_path.read_text())
        result_path.unlink()
        result_path.symlink_to(linked_path)

    with pytest.raises(RuntimeError, match=message):
        load_hpo_trial_checkpoint(checkpoint_path, hpo_context=context)


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


@pytest.mark.parametrize("capability", ["O_DIRECTORY", "O_NOFOLLOW"])
def test_stage_a_locking_requires_descriptor_capabilities(mocker, capability):
    mocker.patch.object(hpo_module.os, capability, None)

    with pytest.raises(RuntimeError, match=r"missing: os\." + capability):
        hpo_module._require_stage_a_locking()


def test_stage_a_persistence_requires_locking_before_path_resolution(mocker, tmp_path):
    contract = _stage_a_contract(_stage_a_source())
    result = hpo_module._stage_a_exception_result(None, contract, ValueError())
    mocker.patch.object(hpo_module, "_require_stage_a_locking", side_effect=RuntimeError("locking"))
    mocker.patch.object(Path, "resolve", side_effect=AssertionError("resolved before preflight"))

    with pytest.raises(RuntimeError, match="locking"):
        hpo_module.persist_stage_a_contract(tmp_path, contract)
    with pytest.raises(RuntimeError, match="locking"):
        hpo_module.persist_stage_a_result(tmp_path, "ordering", 0, result)
    with pytest.raises(RuntimeError, match="locking"):
        hpo_module.persist_stage_a_exception(tmp_path, "ordering", contract, ValueError())


def test_stage_a_directory_rejects_symlinked_root_and_ancestor(tmp_path):
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    symlink_root = tmp_path / "symlink-root"
    symlink_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(OSError):
        hpo_module._open_stage_a_directory(symlink_root, (), create=True)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    (real_parent / "workspace").mkdir()
    symlink_parent = tmp_path / "symlink-parent"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(OSError):
        hpo_module._open_stage_a_directory(symlink_parent / "workspace", (), create=True)


def test_stage_a_construction_failure_rejects_divergent_existing_payload(tmp_path):
    contract = _stage_a_contract(_stage_a_source())
    hpo_module.persist_stage_a_exception(tmp_path, "construction_divergent", contract, ValueError())
    path = tmp_path / "construction_divergent" / "construction-failure.json"
    payload = json.loads(path.read_text())
    payload["trial_number"] = 0
    path.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="does not match the current construction failure"):
        hpo_module.persist_stage_a_exception(
            tmp_path, "construction_divergent", contract, ValueError()
        )


def test_stage_a_construction_failure_rejects_symlinked_study_directory(tmp_path):
    contract = _stage_a_contract(_stage_a_source())
    target = tmp_path / "outside-construction"
    target.mkdir()
    (tmp_path / "construction_symlink").symlink_to(target, target_is_directory=True)

    with pytest.raises(RuntimeError, match="uses a symlink path"):
        hpo_module.persist_stage_a_exception(
            tmp_path, "construction_symlink", contract, ValueError()
        )


def test_stage_a_atomic_json_handles_short_writes(tmp_path, mocker):
    directory = tmp_path / "short-write"
    directory.mkdir()
    directory_fd = hpo_module.os.open(
        directory, hpo_module.os.O_RDONLY | hpo_module.os.O_DIRECTORY | hpo_module.os.O_NOFOLLOW
    )
    original_write = hpo_module.os.write
    calls = 0

    def short_write(file_fd, data):
        nonlocal calls
        calls += 1
        return original_write(file_fd, data[:1])

    mocker.patch.object(hpo_module.os, "write", side_effect=short_write)
    try:
        hpo_module._atomic_stage_a_json_fd(directory_fd, "result.json", {"value": "evidence"})
    finally:
        hpo_module.os.close(directory_fd)

    assert calls > 1
    assert json.loads((directory / "result.json").read_text()) == {"value": "evidence"}


def test_resumed_study_counts_pruned_trials_without_rerunning_them(tmp_path):
    calls = []

    def objective(_trial):
        calls.append(True)
        raise optuna.TrialPruned("screen failed")

    config = HPOConfig(
        n_trials=1,
        timeout_seconds=None,
        metric_config={"task12": ["mixed_mmd.v1"]},
    )
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


def _canonical_checkpoint_payload(tmp_path, *, state="complete"):
    context = _hpo_context(
        metric_config={
            "canonical_objectives": [
                "tstr_macro_f1.v1",
                "mixed_mmd.v1",
                "elastic_net_jsd.v1",
            ]
        }
    )
    metadata = _canonical_metric_metadata(failed=state != "complete")

    def objective(trial):
        trial.set_user_attr("metric_metadata", metadata)
        trial.set_user_attr("result_metadata", dict(metadata))
        if state != "complete":
            error = hpo_module.HPOMetricEvaluationError("SECRET /tmp/raw-path")
            trial.set_user_attr(
                "hpo_error_provenance",
                hpo_module.hpo_exception_provenance(error, location="canonical_metric_evaluation"),
            )
            raise optuna.TrialPruned("metric evaluation failed")
        return 0.25

    with nullcontext() if state == "complete" else pytest.raises(RuntimeError):
        run_study(
            "checkpoint_metadata_shape",
            objective,
            HPOConfig(n_trials=1),
            tmp_path,
            seed=0,
            drop_keys=(),
            hpo_context=context,
        )
    path = (
        tmp_path
        / "hpo_checkpoints"
        / contextual_study_name("checkpoint_metadata_shape", context)
        / "trial-0"
        / "checkpoint.json"
    )
    return json.loads(path.read_text()), context


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["metadata"].pop("metric_metadata"),
        lambda payload: payload["metadata"]["metric_metadata"].pop("mixed_mmd.v1"),
        lambda payload: payload["metadata"]["metric_metadata"].update(
            {"unexpected.v1": payload["metadata"]["metric_metadata"]["tstr_macro_f1.v1"]}
        ),
        lambda payload: payload["metadata"].update(
            {"result_metadata": {**payload["metadata"]["metric_metadata"], "extra.v1": {}}}
        ),
        lambda payload: payload["metadata"]["result_metadata"]["mixed_mmd.v1"].update(
            {"eligible": False}
        ),
        lambda payload: payload["metadata"]["metric_metadata"]["mixed_mmd.v1"].update(
            {"fit_roles": ["tuning"]}
        ),
        lambda payload: payload["metadata"]["metric_metadata"]["mixed_mmd.v1"].update(
            {"status": "pending"}
        ),
        lambda payload: payload["metadata"]["metric_metadata"]["mixed_mmd.v1"].update(
            {"finite": "true"}
        ),
        lambda payload: payload["metadata"]["metric_metadata"]["mixed_mmd.v1"].update(
            {"error_reason_code": "unsafe_reason"}
        ),
        lambda payload: payload["metadata"]["metric_metadata"]["mixed_mmd.v1"].update(
            {
                "status": "failed",
                "finite": False,
                "eligible": False,
                "error_reason_code": "metric_evaluation_exception",
            }
        ),
    ],
    ids=[
        "absent",
        "partial-identities",
        "extra-identity",
        "unequal-identities",
        "unequal-values",
        "invalid-roles",
        "invalid-status",
        "invalid-flags",
        "invalid-error-code",
        "complete-but-failed",
    ],
)
def test_canonical_checkpoint_rejects_incomplete_or_inconsistent_metric_evidence(
    tmp_path, mutation
):
    payload, context = _canonical_checkpoint_payload(tmp_path)
    mutation(payload)

    with pytest.raises(RuntimeError):
        hpo_module._validate_hpo_trial_checkpoint(
            payload,
            expected_study_name=payload["study_name"],
            expected_context_digest=hpo_context_digest(context),
        )


def test_canonical_failed_checkpoint_requires_matching_safe_evidence(tmp_path):
    payload, context = _canonical_checkpoint_payload(tmp_path, state="pruned")
    assert "SECRET" not in json.dumps(payload)
    assert "/tmp" not in json.dumps(payload)
    payload["metadata"]["metric_metadata"]["mixed_mmd.v1"]["error_reason_code"] = (
        "hpo_metric_not_eligible"
    )

    with pytest.raises(RuntimeError, match="result_metadata"):
        hpo_module._validate_hpo_trial_checkpoint(
            payload,
            expected_study_name=payload["study_name"],
            expected_context_digest=hpo_context_digest(context),
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
        "error_location": "tracked_objective",
        "error_fingerprint": checkpoint["metadata"]["outcome"]["error_fingerprint"],
        "state": "failed",
    }
    assert sentinel not in json.dumps(checkpoint)


def test_metric_failure_metadata_and_provenance_survive_checkpoint(tmp_path):
    context = _hpo_context()

    def objective(trial):
        trial.set_user_attr("metric_metadata", _canonical_metric_metadata(failed=True))
        trial.set_user_attr("result_metadata", trial.user_attrs["metric_metadata"])
        error = hpo_module.HPOMetricEvaluationError("SECRET /tmp/patient")
        trial.set_user_attr(
            "hpo_error_provenance",
            hpo_module.hpo_exception_provenance(error, location="canonical_metric_evaluation"),
        )
        raise optuna.TrialPruned("metric evaluation failed")

    with pytest.raises(RuntimeError, match="no completed trials"):
        run_study(
            "metric_failure_checkpoint",
            objective,
            HPOConfig(n_trials=1),
            tmp_path,
            0,
            hpo_context=context,
        )

    checkpoint = load_hpo_trial_checkpoint(
        tmp_path
        / "hpo_checkpoints"
        / contextual_study_name("metric_failure_checkpoint", context)
        / "trial-0"
        / "checkpoint.json",
        hpo_context=context,
    )
    metadata = checkpoint["metadata"]
    assert checkpoint["state"] == "pruned"
    assert set(metadata["metric_metadata"]) == {
        "mixed_mmd.v1",
        "elastic_net_jsd.v1",
        "tstr_macro_f1.v1",
    }
    assert all(
        item["error_reason_code"] == "metric_evaluation_exception"
        for item in metadata["metric_metadata"].values()
    )
    assert metadata["metric_metadata"] == metadata["result_metadata"]
    assert metadata["hpo_error_provenance"]["error_reason_code"] == "metric_evaluation_exception"
    assert metadata["metric_metadata"]["mixed_mmd.v1"]["fit_roles"] == ["train"]
    assert metadata["metric_metadata"]["mixed_mmd.v1"]["evaluation_role"] == "tuning"
    assert "SECRET" not in json.dumps(checkpoint)
    assert "/tmp" not in json.dumps(checkpoint)


def test_tracked_objective_persists_provenance_for_canonical_prune(tmp_path):
    context = _hpo_context(
        metric_config={
            "canonical_objectives": [
                "tstr_macro_f1.v1",
                "mixed_mmd.v1",
                "elastic_net_jsd.v1",
            ]
        }
    )

    def objective(trial):
        metadata = _canonical_metric_metadata(failed=True)
        trial.set_user_attr("metric_metadata", metadata)
        trial.set_user_attr("result_metadata", dict(metadata))
        raise optuna.TrialPruned("stage A failed")

    with pytest.raises(RuntimeError, match="no completed trials"):
        run_study(
            "canonical_tracked_prune",
            objective,
            HPOConfig(n_trials=1),
            tmp_path,
            seed=0,
            drop_keys=(),
            hpo_context=context,
        )

    checkpoint = load_hpo_trial_checkpoint(
        tmp_path
        / "hpo_checkpoints"
        / contextual_study_name("canonical_tracked_prune", context)
        / "trial-0"
        / "checkpoint.json",
        hpo_context=context,
    )
    provenance = checkpoint["metadata"]["hpo_error_provenance"]
    assert provenance["error_reason_code"] == "hpo_trial_exception"
    assert checkpoint["metadata"]["outcome"] == {
        **provenance,
        "state": "pruned",
    }


def test_stage_a_prune_through_tracked_objective_preserves_evidence(tmp_path):
    source = _stage_a_source()
    contract = _stage_a_contract(source)
    context = _hpo_context(stage_a_contract_digest=contract.digest)
    persisted_study_name = contextual_study_name("stage_a_tracked_prune", context)

    def objective(trial):
        screen_stage_a_trial(
            trial,
            source.copy(),
            contract,
            source,
            tmp_path,
            persisted_study_name,
        )
        return 0.0

    with pytest.raises(RuntimeError, match="no completed trials"):
        run_study(
            "stage_a_tracked_prune",
            objective,
            HPOConfig(n_trials=1),
            tmp_path,
            seed=0,
            drop_keys=(),
            hpo_context=context,
            stage_a_root=tmp_path,
        )

    checkpoint = load_hpo_trial_checkpoint(
        tmp_path / "hpo_checkpoints" / persisted_study_name / "trial-0" / "checkpoint.json",
        hpo_context=context,
        stage_a_root=tmp_path,
    )
    metadata = checkpoint["metadata"]
    assert checkpoint["state"] == "pruned"
    assert metadata["stage_a"]["state"] == "pruned"
    assert metadata["stage_a"]["result_path"] == (f"{persisted_study_name}/trial-0/result.json")
    assert metadata["stage_a"]["prune_reasons"]
    assert metadata["hpo_error_provenance"]["error_reason_code"] == "hpo_trial_exception"
    assert metadata["outcome"]["state"] == "pruned"


def test_uncaught_objective_exception_produces_valid_fail_checkpoint(tmp_path):
    context = _hpo_context()

    def objective(_trial):
        raise KeyError("secret /tmp/path")

    with pytest.raises(KeyError):
        run_study(
            "uncaught_key_error",
            objective,
            HPOConfig(n_trials=1),
            tmp_path,
            seed=0,
            drop_keys=(),
            hpo_context=context,
        )

    checkpoint = load_hpo_trial_checkpoint(
        tmp_path
        / "hpo_checkpoints"
        / contextual_study_name("uncaught_key_error", context)
        / "trial-0"
        / "checkpoint.json",
        hpo_context=context,
    )
    assert checkpoint["state"] == "failed"
    assert checkpoint["metadata"]["hpo_error_provenance"]["error_type"] == "KeyError"
    assert checkpoint["metadata"]["outcome"]["state"] == "failed"


def test_hpo_exception_provenance_sanitizes_unusual_exception_type():
    class UnusualException(Exception):
        pass

    UnusualException.__name__ = "9 unsafe/" + ("x" * 200)
    provenance = hpo_module.hpo_exception_provenance(
        UnusualException(), location="unsafe location/" + ("x" * 200)
    )

    assert len(provenance["error_type"]) <= 128
    assert provenance["error_type"].startswith("Exception_")
    assert provenance["error_location"] == "unknown"
    assert hpo_module._validate_hpo_error_provenance(provenance) == provenance


def test_mixed_metric_failure_reasons_survive_checkpoint_validation(tmp_path):
    payload, context = _canonical_checkpoint_payload(tmp_path, state="pruned")
    metric_metadata = payload["metadata"]["metric_metadata"]
    result_metadata = payload["metadata"]["result_metadata"]
    for metadata in (metric_metadata, result_metadata):
        metadata["mixed_mmd.v1"]["error_reason_code"] = "hpo_metric_not_eligible"
    provenance = hpo_module.HPOMetricNotEligibleError("not eligible")
    payload["metadata"]["hpo_error_provenance"] = hpo_module.hpo_exception_provenance(
        provenance, location="canonical_metric_evaluation"
    )

    validated = hpo_module._validate_hpo_trial_checkpoint(
        payload,
        expected_study_name=payload["study_name"],
        expected_context_digest=hpo_context_digest(context),
    )

    assert (
        validated["metadata"]["metric_metadata"]["mixed_mmd.v1"]["error_reason_code"]
        == "hpo_metric_not_eligible"
    )
    assert (
        validated["metadata"]["metric_metadata"]["elastic_net_jsd.v1"]["error_reason_code"]
        == "metric_evaluation_exception"
    )


def test_historical_v2_noncanonical_context_without_new_identity_fields_is_readable(tmp_path):
    payload, context = _canonical_checkpoint_payload(tmp_path)
    historical_context = dict(context)
    historical_context["metric_config"] = {"task12": ["mixed_mmd.v1"]}
    historical_context.pop("canonical_hpo")
    historical_context.pop("canonical_expected_keys")
    payload["hpo_context"] = historical_context
    payload["hpo_context_digest"] = hpo_context_digest(historical_context)

    validated = hpo_module._validate_hpo_trial_checkpoint(
        payload,
        expected_study_name=payload["study_name"],
        expected_context_digest=hpo_context_digest(historical_context),
    )

    assert validated["hpo_context"]["schema_version"] == "hpo-context-v2"
    assert "canonical_hpo" not in validated["hpo_context"]


def test_group_unsafe_provenance_is_bounded_and_distinct():
    provenance = hpo_module.hpo_exception_provenance(
        hpo_module.HPOGroupUnsafeError("unsafe details"),
        location="grouped_metric_evaluation",
    )

    assert provenance["error_reason_code"] == "group_unsafe"
    assert provenance["error_message"] == hpo_module._safe_exception_message("group_unsafe")


def test_invalid_metric_report_has_distinct_non_eligible_provenance():
    report = pd.DataFrame(
        {"mean": [float("nan")], "direction": _strings("minimize")},
        index=_strings("mixed_mmd.v1"),
    )
    report.attrs["hpo_provenance"] = _hpo_context()
    report.attrs["canonical_hpo"] = True
    report.attrs["canonical_hpo_keys"] = ("mixed_mmd.v1",)
    with pytest.raises(hpo_module.HPOMetricNotEligibleError) as error:
        _score(report, expected_keys=("mixed_mmd.v1",))
    provenance = hpo_module.hpo_exception_provenance(error.value, location="synthcity_objective")
    assert provenance["error_reason_code"] == "hpo_metric_not_eligible"
    assert provenance["error_message"] == hpo_module._safe_exception_message(
        "hpo_metric_not_eligible"
    )


def test_hpo_metadata_normalizes_runtime_values_deterministically():
    class Mode(enum.Enum):
        FAST = "fast"

    value = {
        "device": torch.device("cpu"),
        "array": np.asarray([np.int64(3), np.float64(1.5)]),
        "identifier": "workspace/model",
        "mode": Mode.FAST,
        "nested": {"values": (np.int32(2),)},
    }

    assert normalize_hpo_metadata(value) == {
        "array": [3, 1.5],
        "device": "cpu",
        "mode": "fast",
        "nested": {"values": [2]},
        "identifier": "workspace/model",
    }


def test_hpo_metric_config_rejects_calibrating_privacy_metric():
    with pytest.raises(ValueError, match="not approved operational objectives"):
        validate_hpo_metric_config({"privacy": ["identifiability_score"]})


@pytest.mark.parametrize(
    "unsafe",
    [
        Path("workspace/model"),
        Path("/tmp/raw-data"),
        "/tmp/raw-data",
        "../secret",
        "..\\secret",
        "\\tmp\\raw-data",
        "\\\\server\\share\\raw-data",
        "C:raw-data",
        "C:\\tmp\\raw-data",
    ],
)
def test_hpo_metadata_rejects_raw_paths_without_echoing_value(unsafe):
    with pytest.raises(TypeError, match="unsupported HPO metadata value") as error:
        normalize_hpo_metadata({"path": unsafe})

    assert str(unsafe) not in str(error.value)


@pytest.mark.parametrize(
    "unsafe_key",
    [
        Path("workspace/model"),
        Path("/tmp/raw-data"),
        "/tmp/raw-data",
        "../secret",
        "..\\secret",
        "\\tmp\\raw-data",
        "\\\\server\\share\\raw-data",
        "C:raw-data",
        "C:\\tmp\\raw-data",
    ],
)
def test_hpo_metadata_rejects_unsafe_mapping_keys_without_echoing_input(unsafe_key):
    with pytest.raises(TypeError, match="unsupported HPO metadata value") as error:
        normalize_hpo_metadata({unsafe_key: "value"})

    assert str(unsafe_key) not in str(error.value)


def test_hpo_metadata_allows_safe_mapping_keys_recursively():
    assert normalize_hpo_metadata({"workspace/model": {"nested/v1": 1}, "ordinary": "value"}) == {
        "ordinary": "value",
        "workspace/model": {"nested/v1": 1},
    }


def test_hpo_metadata_normalizes_safe_nested_windows_mapping_key():
    assert normalize_hpo_metadata({r"nested\v1": "value"}) == {"nested/v1": "value"}


def test_hpo_metadata_rejects_mapping_key_collision_after_normalization():
    with pytest.raises(TypeError, match="unsupported HPO metadata value") as error:
        normalize_hpo_metadata({r"nested\v1": "backslash", "nested/v1": "slash"})

    assert r"nested\v1" not in str(error.value)
    assert "nested/v1" not in str(error.value)


def test_hpo_metadata_allows_safe_slash_bearing_identifier():
    assert normalize_hpo_metadata({"identifier": "urn:example/model/v1"}) == {
        "identifier": "urn:example/model/v1"
    }


def test_hpo_exception_provenance_maps_unknown_reason_code():
    error = RuntimeError("SECRET /tmp/raw-data")
    error.reason_code = "untrusted_reason"  # type: ignore[attr-defined]

    provenance = hpo_module.hpo_exception_provenance(error, location="test")

    assert provenance["error_reason_code"] == "hpo_trial_exception"
    assert provenance["error_message"] == "HPO trial failed; exception details suppressed."
    assert "SECRET" not in json.dumps(provenance)


def test_hpo_error_provenance_validator_recomputes_fingerprint():
    provenance = hpo_module.hpo_exception_provenance(
        RuntimeError("SECRET"), location="tracked_objective"
    )

    provenance["error_fingerprint"] = "0" * 64
    with pytest.raises(RuntimeError, match="invalid fingerprint"):
        hpo_module._validate_hpo_error_provenance(provenance)


@pytest.mark.parametrize(
    "field, value",
    [
        ("error_message", "safe message altered"),
        ("error_location", "../unsafe"),
        ("error_reason_code", "untrusted_reason"),
    ],
)
def test_hpo_error_provenance_validator_rejects_unsafe_or_tampered_fields(field, value):
    provenance = hpo_module.hpo_exception_provenance(
        RuntimeError("SECRET"), location="tracked_objective"
    )
    provenance[field] = value

    with pytest.raises(RuntimeError):
        hpo_module._validate_hpo_error_provenance(provenance)

    assert "SECRET" not in json.dumps(provenance)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("error_type"),
        lambda value: value.__setitem__("extra", "field"),
    ],
)
def test_hpo_error_provenance_validator_rejects_invalid_shape(mutation):
    provenance = hpo_module.hpo_exception_provenance(
        RuntimeError("SECRET"), location="tracked_objective"
    )
    mutation(provenance)

    with pytest.raises(RuntimeError, match="invalid shape"):
        hpo_module._validate_hpo_error_provenance(provenance)


def test_hpo_metadata_serialization_provenance_survives_checkpoint(tmp_path, monkeypatch):
    class FakePlugin:
        @staticmethod
        def sample_hyperparameters_optuna(_trial):
            return {"unsafe": Path("/secret/raw-data")}

    monkeypatch.setattr(synthcity_backend_module, "get_plugin_class", lambda _name: FakePlugin)
    monkeypatch.setattr(synthcity_backend_module, "plugin_accepts", lambda *_args: False)
    objective = build_synthcity_objective(
        "ctgan",
        train_loader=object(),
        tuning_loader=object(),
        hpo_cfg=HPOConfig(
            n_trials=1,
            timeout_seconds=None,
        ),
        seed=0,
        train_df=pd.DataFrame({"target": [0]}),
        tuning_df=pd.DataFrame({"target": [0]}),
        target_column="target",
    )
    context = _hpo_context()

    with pytest.raises(RuntimeError, match="produced no completed trials"):
        run_study(
            "metadata_serialization_failure",
            objective,
            HPOConfig(n_trials=1, timeout_seconds=None),
            tmp_path,
            seed=0,
            drop_keys=(),
            hpo_context=context,
        )

    checkpoint = load_hpo_trial_checkpoint(
        tmp_path
        / "hpo_checkpoints"
        / contextual_study_name("metadata_serialization_failure", context)
        / "trial-0"
        / "checkpoint.json",
        hpo_context=context,
    )
    provenance = checkpoint["metadata"]["hpo_error_provenance"]
    assert provenance["error_reason_code"] == "hpo_metadata_serialization_failure"
    assert provenance["error_type"] == "HPOMetadataSerializationError"
    assert provenance["error_location"] == "synthcity_objective.setup"
    assert checkpoint["metadata"]["outcome"] == {
        **provenance,
        "state": "pruned",
    }
    assert checkpoint["state"] == "pruned"
    assert "/secret/raw-data" not in json.dumps(checkpoint)


def test_synthcity_objective_persists_sampled_and_effective_parameters(tmp_path, monkeypatch):
    fit_calls = []

    class FakePlugin:
        @staticmethod
        def sample_hyperparameters_optuna(trial):
            return {"n_iter": trial.suggest_int("n_iter", 100, 100)}

    def fake_canonical_metrics(*_args, **_kwargs):
        report = pd.DataFrame(
            {
                "mean": [0.25, 0.25, 0.25],
                "direction": _strings("minimize", "minimize", "maximize"),
            },
            index=_strings("mixed_mmd.v1", "elastic_net_jsd.v1", "tstr_macro_f1.v1"),
        )
        report.attrs["metric_metadata"] = {
            key: {**value, "mean": 0.25, "errors": 0}
            for key, value in _canonical_metric_metadata().items()
        }
        report.attrs["hpo_provenance"] = _canonical_hpo_context()
        return report

    def fake_fit_generate(*args, **kwargs):
        fit_calls.append((args, kwargs))
        metadata = {
            "schema_version": "generator-metadata-v1",
            "generator_context": {"privacy_claim_type": "none"},
            "plugin_name": "ctgan",
            "plugin_fqdn": "synthcity.ctgan",
            "requested_parameters": {"n_iter": 100},
            "n_samples": 1,
            "random_state": 0,
            "privacy_accounting": None,
        }
        return pd.DataFrame({"target": [0]}), metadata

    monkeypatch.setattr(synthcity_backend_module, "get_plugin_class", lambda _name: FakePlugin)
    monkeypatch.setattr(
        synthcity_backend_module,
        "plugin_accepts",
        lambda _name, parameter: parameter in {"n_iter", "device"},
    )
    monkeypatch.setattr(synthcity_backend_module, "fit_generate", fake_fit_generate)
    monkeypatch.setattr(
        synthcity_backend_module, "evaluate_canonical_hpo_metrics", fake_canonical_metrics
    )

    config = HPOConfig(
        n_trials=1,
        timeout_seconds=None,
        n_iter_cap=40,
    )
    context = _canonical_hpo_context()
    objective = build_synthcity_objective(
        "ctgan",
        train_loader=object(),
        tuning_loader=object(),
        hpo_cfg=config,
        seed=0,
        device="cuda",
        train_df=pd.DataFrame({"target": [0]}),
        tuning_df=pd.DataFrame({"target": [0]}),
        target_column="target",
        synthetic_size=1,
    )
    run_study(
        "fake_ctgan_boundary",
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
        / contextual_study_name("fake_ctgan_boundary", context)
        / "trial-0"
        / "checkpoint.json",
        hpo_context=context,
    )
    generator = checkpoint["metadata"]["generator"]["metadata"]
    assert len(fit_calls) == 1
    assert fit_calls[0][1]["device"] == "cuda"
    assert fit_calls[0][0][1]["n_iter"] == 40
    assert isinstance(fit_calls[0][0][1]["device"], torch.device)
    assert fit_calls[0][0][1]["device"] == torch.device("cuda")
    assert generator["requested_parameters"]["n_iter"] == 100
    assert generator["effective_parameters"]["n_iter"] == 40
    assert generator["effective_parameters"]["device"] == "cuda"
    trial = optuna.load_study(
        study_name=contextual_study_name("fake_ctgan_boundary", context),
        storage=f"sqlite:///{tmp_path / 'optuna_studies.db'}",
    ).trials[0]
    assert trial.user_attrs["generator_requested_parameters"]["n_iter"] == 100
    assert trial.user_attrs["generator_effective_parameters"] == {
        "n_iter": 40,
        "random_state": 0,
        "device": "cuda",
    }
    json.dumps(checkpoint)


@pytest.mark.parametrize("report_kind", ["none", "missing_trial", "malformed_metric"])
def test_native_malformed_benchmark_report_is_persisted_as_not_eligible(
    tmp_path, monkeypatch, report_kind
):
    class FakePlugin:
        @staticmethod
        def sample_hyperparameters_optuna(_trial):
            return {}

    from synthcity.benchmark import Benchmarks

    def fake_evaluate(*_args, **_kwargs):
        if report_kind == "none":
            return None
        if report_kind == "missing_trial":
            return {}
        return {"trial_0": object()}

    monkeypatch.setattr(synthcity_backend_module, "get_plugin_class", lambda _name: FakePlugin)
    monkeypatch.setattr(synthcity_backend_module, "plugin_accepts", lambda *_args: False)
    monkeypatch.setattr(Benchmarks, "evaluate", staticmethod(fake_evaluate))

    config = HPOConfig(
        n_trials=1,
        timeout_seconds=None,
        metric_config={"task12": ["mixed_mmd.v1"]},
    )
    context = _hpo_context()
    objective = build_synthcity_objective(
        "ctgan",
        train_loader=object(),
        tuning_loader=object(),
        hpo_cfg=config,
        seed=0,
    )
    study_name = f"native_malformed_{report_kind}"
    with pytest.raises(RuntimeError, match="produced no completed trials"):
        run_study(
            study_name,
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
        / contextual_study_name(study_name, context)
        / "trial-0"
        / "checkpoint.json",
        hpo_context=context,
    )
    assert checkpoint["state"] == "pruned"
    assert checkpoint["objective_value"] is None
    provenance = checkpoint["metadata"]["hpo_error_provenance"]
    assert provenance["error_reason_code"] == "hpo_metric_not_eligible"
    assert provenance["error_location"] == "synthcity_objective"
    assert checkpoint["metadata"]["outcome"]["state"] == "pruned"
    assert "SECRET" not in json.dumps(checkpoint)
    assert "/tmp" not in json.dumps(checkpoint)


@pytest.mark.parametrize("report_kind", ["all_failed", "invalid", "successful"])
def test_synthcity_canonical_report_call_chain_persists_metric_outcome(
    tmp_path, monkeypatch, report_kind
):
    metrics = ["mixed_mmd.v1", "elastic_net_jsd.v1", "tstr_macro_f1.v1"]

    class FakePlugin:
        @staticmethod
        def sample_hyperparameters_optuna(_trial):
            return {}

    def fake_fit_generate(*_args, **_kwargs):
        return pd.DataFrame({"target": [0]}), {
            "schema_version": "generator-metadata-v1",
            "generator_context": {"privacy_claim_type": "none"},
            "plugin_name": "ctgan",
            "plugin_fqdn": "synthcity.ctgan",
            "requested_parameters": {},
            "n_samples": 1,
            "random_state": 0,
            "privacy_accounting": None,
        }

    def fake_metrics(*_args, **_kwargs):
        failed = report_kind == "all_failed"
        invalid = report_kind == "invalid"
        means = [float("nan")] * 3 if failed else [0.2, 0.3, 0.8]
        directions = (
            _strings("maximize", "maximize", "minimize")
            if invalid
            else _strings("minimize", "minimize", "maximize")
        )
        report = pd.DataFrame(
            {"mean": means, "direction": directions},
            index=_strings(*metrics),
        )
        report.attrs["hpo_provenance"] = _canonical_hpo_context()
        report.attrs["metric_metadata"] = {
            key: {
                "metric_name": key,
                "status": "failed" if failed or invalid else "complete",
                "direction": "maximize" if key == "tstr_macro_f1.v1" else "minimize",
                "mean": None if failed or invalid else mean,
                "errors": 1 if failed or invalid else 0,
                "error_reason_code": (
                    "metric_evaluation_exception"
                    if failed
                    else "hpo_metric_not_eligible"
                    if invalid
                    else None
                ),
                "fit_roles": ["train"],
                "evaluation_role": "tuning",
            }
            for key, mean in zip(metrics, means, strict=True)
        }
        return report

    monkeypatch.setattr(synthcity_backend_module, "get_plugin_class", lambda _name: FakePlugin)
    monkeypatch.setattr(synthcity_backend_module, "plugin_accepts", lambda *_args: False)
    monkeypatch.setattr(synthcity_backend_module, "fit_generate", fake_fit_generate)
    monkeypatch.setattr(synthcity_backend_module, "evaluate_canonical_hpo_metrics", fake_metrics)
    frame = pd.DataFrame({"target": [0]})
    config = HPOConfig(n_trials=1, timeout_seconds=None)
    context = _canonical_hpo_context()
    objective = build_synthcity_objective(
        "ctgan",
        object(),
        config,
        0,
        tuning_loader=object(),
        train_df=frame,
        tuning_df=frame,
        target_column="target",
        synthetic_size=1,
    )
    with (
        pytest.raises(RuntimeError, match="produced no completed trials")
        if report_kind != "successful"
        else nullcontext()
    ):
        run_study(
            f"canonical_{report_kind}",
            objective,
            config,
            tmp_path,
            0,
            drop_keys=(),
            hpo_context=context,
        )
    checkpoint = load_hpo_trial_checkpoint(
        tmp_path
        / "hpo_checkpoints"
        / contextual_study_name(f"canonical_{report_kind}", context)
        / "trial-0"
        / "checkpoint.json",
        hpo_context=context,
    )
    assert checkpoint["metadata"]["metric_metadata"]
    metric_metadata = checkpoint["metadata"]["metric_metadata"]
    assert set(metric_metadata) == set(metrics)
    assert metric_metadata == checkpoint["metadata"]["result_metadata"]
    assert "SECRET" not in json.dumps(checkpoint) and "/tmp" not in json.dumps(checkpoint)
    if report_kind == "successful":
        assert checkpoint["state"] == "complete"
        assert math.isfinite(checkpoint["objective_value"])
    else:
        assert checkpoint["state"] == "pruned"
        assert checkpoint["objective_value"] is None
        expected = (
            "metric_evaluation_exception"
            if report_kind == "all_failed"
            else "hpo_metric_not_eligible"
        )
        assert all(item["status"] == "failed" for item in metric_metadata.values())
        assert checkpoint["metadata"]["hpo_error_provenance"]["error_reason_code"] == expected


def test_canonical_pre_evaluation_failure_preserves_absent_metric_evidence(tmp_path, monkeypatch):
    class FakePlugin:
        @staticmethod
        def sample_hyperparameters_optuna(_trial):
            return {}

    monkeypatch.setattr(synthcity_backend_module, "get_plugin_class", lambda _name: FakePlugin)
    monkeypatch.setattr(synthcity_backend_module, "plugin_accepts", lambda *_args: False)
    monkeypatch.setattr(
        synthcity_backend_module,
        "fit_generate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("fit failed")),
    )
    config = HPOConfig(n_trials=1, timeout_seconds=None)
    context = _canonical_hpo_context()
    objective = build_synthcity_objective(
        "ctgan",
        object(),
        config,
        0,
        tuning_loader=object(),
        train_df=pd.DataFrame({"target": [0]}),
        tuning_df=pd.DataFrame({"target": [0]}),
        target_column="target",
    )
    with pytest.raises(RuntimeError, match="produced no completed trials"):
        run_study(
            "canonical_pre_eval_failure",
            objective,
            config,
            tmp_path,
            0,
            drop_keys=(),
            hpo_context=context,
        )
    checkpoint = load_hpo_trial_checkpoint(
        tmp_path
        / "hpo_checkpoints"
        / contextual_study_name("canonical_pre_eval_failure", context)
        / "trial-0"
        / "checkpoint.json",
        hpo_context=context,
    )
    assert checkpoint["metadata"]["metric_metadata"] is None
    assert checkpoint["metadata"]["result_metadata"] is None
    assert (
        checkpoint["metadata"]["hpo_error_provenance"]["error_reason_code"] == "hpo_trial_exception"
    )


def test_hpo_checkpoint_boundary_keeps_requested_and_effective_parameters(tmp_path):
    context = _hpo_context()
    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    trial.set_user_attr("generator_metadata_state", "present")
    trial.set_user_attr("generator_plugin_name", "ctgan")
    trial.set_user_attr("generator_privacy_claim_type", "none")
    trial.set_user_attr("generator_implementation_fingerprint", "impl")
    trial.set_user_attr(
        "generator_metadata",
        {
            "schema_version": "generator-metadata-v2",
            "generator_context": {"privacy_claim_type": "none"},
            "plugin_name": "ctgan",
            "plugin_fqdn": "synthcity.ctgan",
            "requested_parameters": {"n_iter": 100, "device": "cpu"},
            "effective_parameters": {"n_iter": 40, "device": "cpu"},
            "n_samples": 4,
            "random_state": 0,
            "privacy_accounting": None,
            "implementation_fingerprint": "impl",
        },
    )
    study.tell(trial, 0.25)

    checkpoint_path = persist_hpo_trial_checkpoint(
        tmp_path,
        "boundary",
        study.trials[0],
        hpo_context=context,
        expected_implementation_fingerprint="impl",
    )
    checkpoint = load_hpo_trial_checkpoint(
        checkpoint_path,
        hpo_context=context,
        expected_implementation_fingerprint="impl",
    )
    metadata = checkpoint["metadata"]["generator"]["metadata"]

    json.dumps(checkpoint)
    assert metadata["requested_parameters"]["n_iter"] == 100
    assert metadata["effective_parameters"]["n_iter"] == 40
    assert metadata["effective_parameters"]["device"] == "cpu"


def test_synthcity_objective_rejects_malformed_v1_generator_metadata(tmp_path, monkeypatch):
    class FakePlugin:
        @staticmethod
        def sample_hyperparameters_optuna(_trial):
            return {"n_iter": 100}

    def fake_canonical_metrics(*_args, **_kwargs):
        report = pd.DataFrame(
            {
                "mean": [0.25, 0.25, 0.25],
                "direction": _strings("minimize", "minimize", "maximize"),
            },
            index=_strings("mixed_mmd.v1", "elastic_net_jsd.v1", "tstr_macro_f1.v1"),
        )
        report.attrs["metric_metadata"] = _canonical_metric_metadata()
        report.attrs["hpo_provenance"] = _hpo_context()
        return report

    def fake_fit_generate(*_args, **_kwargs):
        return pd.DataFrame({"target": [0]}), {
            "schema_version": "generator-metadata-v1",
            "generator_context": {"privacy_claim_type": "none"},
            "plugin_name": "ctgan",
            "n_samples": 1,
            "random_state": 0,
            "privacy_accounting": None,
        }

    monkeypatch.setattr(synthcity_backend_module, "get_plugin_class", lambda _name: FakePlugin)
    monkeypatch.setattr(
        synthcity_backend_module,
        "plugin_accepts",
        lambda _name, parameter: parameter in {"n_iter", "device"},
    )
    monkeypatch.setattr(synthcity_backend_module, "fit_generate", fake_fit_generate)
    monkeypatch.setattr(
        synthcity_backend_module, "evaluate_canonical_hpo_metrics", fake_canonical_metrics
    )
    monkeypatch.setattr(synthcity_backend_module, "hpo_score", lambda *_args, **_kwargs: 0.25)

    config = HPOConfig(n_trials=1, timeout_seconds=None, n_iter_cap=40)
    context = _hpo_context()
    objective = build_synthcity_objective(
        "ctgan",
        train_loader=object(),
        tuning_loader=object(),
        hpo_cfg=config,
        seed=0,
        train_df=pd.DataFrame({"target": [0]}),
        tuning_df=pd.DataFrame({"target": [0]}),
        target_column="target",
    )

    with pytest.raises(RuntimeError, match="produced no completed trials"):
        run_study(
            "malformed_v1_metadata",
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
        / contextual_study_name("malformed_v1_metadata", context)
        / "trial-0"
        / "checkpoint.json",
        hpo_context=context,
    )
    assert checkpoint["state"] == "pruned"
    assert checkpoint["metadata"]["generator"]["state"] == "missing"
    assert (
        "generator" not in checkpoint["metadata"]
        or checkpoint["metadata"]["generator"].get("metadata") is None
    )


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
    assert all(report.attrs["metric_metadata"][key]["errors"] == 0 for key in report.index)
    assert hpo_module.sanitize_hpo_metric_metadata(
        report.attrs["metric_metadata"],
        allowed_keys=list(report.index),
    )


def test_real_canonical_evaluator_runs_through_objective_and_checkpoint(tmp_path, monkeypatch):
    class FakePlugin:
        @staticmethod
        def sample_hyperparameters_optuna(_trial):
            return {}

    def fake_fit_generate(*_args, **_kwargs):
        return frame.copy(), {
            "schema_version": "generator-metadata-v1",
            "generator_context": {"privacy_claim_type": "none"},
            "plugin_name": "ctgan",
            "plugin_fqdn": "synthcity.ctgan",
            "requested_parameters": {},
            "n_samples": 20,
            "random_state": 0,
            "privacy_accounting": None,
        }

    monkeypatch.setattr(synthcity_backend_module, "get_plugin_class", lambda _name: FakePlugin)
    monkeypatch.setattr(synthcity_backend_module, "plugin_accepts", lambda *_args: False)
    monkeypatch.setattr(synthcity_backend_module, "fit_generate", fake_fit_generate)
    frame = pd.DataFrame({"x": np.arange(20, dtype=float), "target": [0, 1] * 10})
    config = HPOConfig(n_trials=1, timeout_seconds=None)
    context = _hpo_context(
        metric_config={
            "canonical_objectives": [
                "tstr_macro_f1.v1",
                "mixed_mmd.v1",
                "elastic_net_jsd.v1",
            ]
        }
    )
    objective = build_synthcity_objective(
        "ctgan",
        train_loader=object(),
        tuning_loader=object(),
        hpo_cfg=config,
        seed=0,
        train_df=frame,
        tuning_df=frame.copy(),
        target_column="target",
        feature_types={"x": "continuous", "target": "categorical"},
        synthetic_size=len(frame),
    )

    run_study(
        "real_canonical_boundary",
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
        / contextual_study_name("real_canonical_boundary", context)
        / "trial-0"
        / "checkpoint.json",
        hpo_context=context,
    )
    assert checkpoint["state"] == "complete"
    assert math.isfinite(checkpoint["objective_value"])
    assert all(
        item["status"] == "complete" and item["eligible"]
        for item in checkpoint["metadata"]["metric_metadata"].values()
    )


@pytest.mark.parametrize("outcome", ["failed", "invalid"])
def test_real_canonical_evaluator_failure_and_invalid_evidence_checkpoint(
    tmp_path, monkeypatch, outcome
):
    """Real evaluator preserves bounded failure evidence without report fabrication."""

    class FakePlugin:
        @staticmethod
        def sample_hyperparameters_optuna(_trial):
            return {}

    frame = pd.DataFrame({"x": np.arange(4, dtype=float), "target": [0, 1, 0, 1]})

    def fake_fit_generate(*_args, **_kwargs):
        return frame.copy(), {
            "schema_version": "generator-metadata-v1",
            "generator_context": {"privacy_claim_type": "none"},
            "plugin_name": "ctgan",
            "plugin_fqdn": "synthcity.ctgan",
            "requested_parameters": {},
            "n_samples": len(frame),
            "random_state": 0,
            "privacy_accounting": None,
        }

    def lower_metric(*_args, **_kwargs):
        if outcome == "failed":
            raise RuntimeError("controlled metric failure")
        if outcome == "invalid":
            raise hpo_module.HPOMetricNotEligibleError("controlled invalid evidence")
        return {"b_mmd_clip": float("nan"), "bandwidth": 1.0}

    def lower_jsd(*_args, **_kwargs):
        if outcome == "failed":
            raise RuntimeError("controlled metric failure")
        if outcome == "invalid":
            raise hpo_module.HPOMetricNotEligibleError("controlled invalid evidence")
        return float("nan"), {}

    def fake_tstr(*_args, **_kwargs):
        if outcome == "failed":
            raise RuntimeError("controlled metric failure")
        raise hpo_module.HPOMetricNotEligibleError("controlled invalid evidence")

    monkeypatch.setattr(synthcity_backend_module, "get_plugin_class", lambda _name: FakePlugin)
    monkeypatch.setattr(synthcity_backend_module, "plugin_accepts", lambda *_args: False)
    monkeypatch.setattr(synthcity_backend_module, "fit_generate", fake_fit_generate)
    monkeypatch.setattr(hpo_module, "_evaluate_train_frozen_mmd", lower_metric)
    monkeypatch.setattr(hpo_module, "_evaluate_train_frozen_jsd", lower_jsd)
    monkeypatch.setattr(
        "synthdata.evaluation.tstr.run_tstr_evaluation",
        fake_tstr,
    )

    config = HPOConfig(n_trials=1, timeout_seconds=None)
    context = _hpo_context(
        metric_config={
            "canonical_objectives": [
                "tstr_macro_f1.v1",
                "mixed_mmd.v1",
                "elastic_net_jsd.v1",
            ]
        }
    )
    objective = build_synthcity_objective(
        "ctgan",
        object(),
        config,
        0,
        tuning_loader=object(),
        train_df=frame,
        tuning_df=frame.copy(),
        target_column="target",
        feature_types={"x": "continuous", "target": "categorical"},
        synthetic_size=len(frame),
    )

    with pytest.raises(RuntimeError, match="produced no completed trials"):
        run_study(
            f"real_canonical_{outcome}",
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
        / contextual_study_name(f"real_canonical_{outcome}", context)
        / "trial-0"
        / "checkpoint.json",
        hpo_context=context,
    )
    assert checkpoint["state"] == "pruned"
    assert checkpoint["objective_value"] is None
    metadata = checkpoint["metadata"]["metric_metadata"]
    assert set(metadata) == {
        "tstr_macro_f1.v1",
        "mixed_mmd.v1",
        "elastic_net_jsd.v1",
    }
    assert all(item["status"] == "failed" for item in metadata.values())
    expected_reason = (
        "metric_evaluation_exception" if outcome == "failed" else "hpo_metric_not_eligible"
    )
    assert checkpoint["metadata"]["hpo_error_provenance"]["error_reason_code"] == expected_reason
    assert "hpo_provenance" not in checkpoint["metadata"]
    assert "SECRET" not in json.dumps(checkpoint) and "/tmp" not in json.dumps(checkpoint)


def test_real_canonical_evaluator_mixed_metric_reasons_checkpoint(tmp_path, monkeypatch):
    """Canonical objective preserves mixed bounded metric failures durably."""

    class FakePlugin:
        @staticmethod
        def sample_hyperparameters_optuna(_trial):
            return {}

    frame = pd.DataFrame({"x": np.arange(4, dtype=float), "target": [0, 1, 0, 1]})

    def fake_fit_generate(*_args, **_kwargs):
        return frame.copy(), {
            "schema_version": "generator-metadata-v1",
            "generator_context": {"privacy_claim_type": "none"},
            "plugin_name": "ctgan",
            "plugin_fqdn": "synthcity.ctgan",
            "requested_parameters": {},
            "n_samples": len(frame),
            "random_state": 0,
            "privacy_accounting": None,
        }

    def failed_mmd(*_args, **_kwargs):
        raise RuntimeError("controlled mmd failure SECRET /tmp/raw-path")

    def ineligible_jsd(*_args, **_kwargs):
        raise hpo_module.HPOMetricNotEligibleError("controlled jsd invalid SECRET")

    def failed_tstr(*_args, **_kwargs):
        raise RuntimeError("controlled tstr failure SECRET /tmp/raw-path")

    monkeypatch.setattr(synthcity_backend_module, "get_plugin_class", lambda _name: FakePlugin)
    monkeypatch.setattr(synthcity_backend_module, "plugin_accepts", lambda *_args: False)
    monkeypatch.setattr(synthcity_backend_module, "fit_generate", fake_fit_generate)
    monkeypatch.setattr(hpo_module, "_evaluate_train_frozen_mmd", failed_mmd)
    monkeypatch.setattr(hpo_module, "_evaluate_train_frozen_jsd", ineligible_jsd)
    monkeypatch.setattr("synthdata.evaluation.tstr.run_tstr_evaluation", failed_tstr)

    config = HPOConfig(n_trials=1, timeout_seconds=None)
    context = _canonical_hpo_context()
    objective = build_synthcity_objective(
        "ctgan",
        object(),
        config,
        0,
        tuning_loader=object(),
        train_df=frame,
        tuning_df=frame.copy(),
        target_column="target",
        feature_types={"x": "continuous", "target": "categorical"},
        synthetic_size=len(frame),
    )

    with pytest.raises(RuntimeError, match="produced no completed trials"):
        run_study(
            "real_canonical_mixed_reasons",
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
        / contextual_study_name("real_canonical_mixed_reasons", context)
        / "trial-0"
        / "checkpoint.json",
        hpo_context=context,
    )
    assert checkpoint["state"] == "pruned"
    assert checkpoint["objective_value"] is None
    metadata = checkpoint["metadata"]
    metric_metadata = metadata["metric_metadata"]
    assert set(metric_metadata) == {
        "tstr_macro_f1.v1",
        "mixed_mmd.v1",
        "elastic_net_jsd.v1",
    }
    assert {key: item["error_reason_code"] for key, item in metric_metadata.items()} == {
        "tstr_macro_f1.v1": "metric_evaluation_exception",
        "mixed_mmd.v1": "metric_evaluation_exception",
        "elastic_net_jsd.v1": "hpo_metric_not_eligible",
    }
    assert all(
        item["status"] == "failed"
        and item["finite"] is False
        and item["eligible"] is False
        and item["fit_roles"] == ["train"]
        and item["evaluation_role"] == "tuning"
        for item in metric_metadata.values()
    )
    assert metric_metadata == metadata["result_metadata"]
    assert "hpo_provenance" not in metadata

    aggregate = metadata["hpo_error_provenance"]
    assert aggregate["error_reason_code"] == "hpo_metric_not_eligible"
    assert aggregate["error_message"] == (
        "HPO metric report was not eligible; exception details suppressed."
    )
    assert aggregate["error_location"] == "canonical_metric_evaluation"
    assert len(aggregate["error_fingerprint"]) == 64
    assert all(character in "0123456789abcdef" for character in aggregate["error_fingerprint"])
    assert "SECRET" not in json.dumps(checkpoint)
    assert "/tmp" not in json.dumps(checkpoint)


@pytest.mark.parametrize("outcome", ["complete", "failed", "pruned"])
def test_noncanonical_checkpoint_preserves_native_metadata_contract(tmp_path, outcome):
    context = _hpo_context(metric_config={"task12": ["mixed_mmd.v1"]})

    def objective(trial):
        if outcome == "complete":
            trial.set_user_attr("metric_metadata", {"native": {"value": 1}})
            trial.set_user_attr("result_metadata", {"native": {"value": 1}})
            return 0.25
        if outcome == "pruned":
            raise optuna.TrialPruned("native metric unavailable")
        raise RuntimeError("native metric failed")

    with pytest.raises(RuntimeError) if outcome != "complete" else nullcontext():
        run_study(
            f"noncanonical_{outcome}",
            objective,
            HPOConfig(n_trials=1, timeout_seconds=None),
            tmp_path,
            seed=0,
            drop_keys=(),
            hpo_context=context,
        )
    checkpoint = load_hpo_trial_checkpoint(
        tmp_path
        / "hpo_checkpoints"
        / contextual_study_name(f"noncanonical_{outcome}", context)
        / "trial-0"
        / "checkpoint.json",
        hpo_context=context,
    )
    assert checkpoint["state"] == outcome
    if outcome == "complete":
        assert checkpoint["metadata"]["metric_metadata"] == {"native": {"value": 1}}
        assert checkpoint["metadata"]["result_metadata"] == {"native": {"value": 1}}
    else:
        assert checkpoint["metadata"]["metric_metadata"] is None
        assert checkpoint["metadata"]["result_metadata"] is None


def test_noncanonical_checkpoint_defaults_result_metadata_to_metric_metadata(tmp_path):
    context = _hpo_context(metric_config={"task12": ["mixed_mmd.v1"]})
    study = optuna.create_study(direction="minimize")

    def objective(trial):
        trial.set_user_attr("metric_metadata", {"native": {"value": 1}})
        return 0.25

    study.optimize(objective, n_trials=1)
    path = persist_hpo_trial_checkpoint(
        tmp_path,
        "noncanonical_fallback",
        study.trials[0],
        hpo_context=context,
    )

    checkpoint = load_hpo_trial_checkpoint(path, hpo_context=context)
    assert checkpoint["metadata"]["result_metadata"] == {"native": {"value": 1}}


def test_noncanonical_checkpoint_rejects_unequal_metadata(tmp_path):
    context = _hpo_context(metric_config={"task12": ["mixed_mmd.v1"]})
    study = optuna.create_study(direction="minimize")

    def objective(trial):
        trial.set_user_attr("metric_metadata", {"native": {"value": 1}})
        trial.set_user_attr("result_metadata", {"native": {"value": 2}})
        return 0.25

    study.optimize(objective, n_trials=1)
    with pytest.raises(RuntimeError, match="result_metadata"):
        persist_hpo_trial_checkpoint(
            tmp_path,
            "noncanonical_mismatch",
            study.trials[0],
            hpo_context=context,
        )


def test_group_unsafe_objective_provenance_survives_run_study_checkpoint(tmp_path, monkeypatch):
    class FakePlugin:
        @staticmethod
        def sample_hyperparameters_optuna(_trial):
            return {}

    def fake_evaluate(*_args, **_kwargs):
        report = pd.DataFrame(
            {"mean": [0.2], "direction": _strings("minimize")},
            index=_strings("mixed_mmd.v1"),
        )
        report.attrs["group_safety"] = _native_group_safety("group_unsafe")
        provenance = _provenance_kwargs()
        provenance["contracts"] = {
            key: provenance["contracts"][key]
            for key in ("fit_roles", "comparison_role", "excluded_roles")
        }
        report.attrs["hpo_provenance"] = {
            key: provenance[key]
            for key in (
                "release_transform_digest",
                "role_hashes",
                "contracts",
                "objective_version",
            )
        }
        report.attrs["metric_metadata"] = {
            "mixed_mmd.v1": {
                "metric_name": "mixed_mmd.v1",
                "status": "complete",
                "finite": True,
                "eligible": True,
                "error_reason_code": None,
            }
        }
        report.attrs["result_metadata"] = dict(report.attrs["metric_metadata"])
        return {"trial_0": report}

    from synthcity.benchmark import Benchmarks

    monkeypatch.setattr(synthcity_backend_module, "get_plugin_class", lambda _name: FakePlugin)
    monkeypatch.setattr(synthcity_backend_module, "plugin_accepts", lambda *_args: False)
    monkeypatch.setattr(Benchmarks, "evaluate", staticmethod(fake_evaluate))

    config = HPOConfig(
        n_trials=1,
        timeout_seconds=None,
        metric_config={"task12": ["mixed_mmd.v1"]},
    )
    context = _hpo_context()
    objective = build_synthcity_objective(
        "ctgan",
        train_loader=object(),
        tuning_loader=object(),
        hpo_cfg=config,
        seed=0,
    )

    with pytest.raises(RuntimeError, match="produced no completed trials"):
        run_study(
            "group_unsafe_checkpoint",
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
        / contextual_study_name("group_unsafe_checkpoint", context)
        / "trial-0"
        / "checkpoint.json",
        hpo_context=context,
    )
    assert checkpoint["state"] == "pruned"
    assert checkpoint["objective_value"] is None
    assert checkpoint["metadata"]["metric_metadata"] is None
    assert checkpoint["metadata"]["result_metadata"] is None
    assert checkpoint["metadata"]["hpo_error_provenance"]["error_reason_code"] == "group_unsafe"
    assert checkpoint["metadata"]["outcome"]["group_safety"] == {
        "status": "group_unsafe",
        "reason_code": "group_unsafe",
    }


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
        {
            "canonical_objectives": [
                "tstr_macro_f1.v1",
                "mixed_mmd.v1",
                "elastic_net_jsd.v1",
            ]
        },
        seed=0,
    )

    assert evaluate(candidate) == pytest.approx(-(0.75 + 0.5 + 0.75) / 3)
    canonical.assert_called_once()
    native.assert_not_called()
