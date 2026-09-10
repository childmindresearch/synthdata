"""Unit tests for contract-aware absolute privacy gates."""

import dataclasses

import pandas as pd
import pytest

from synthdata.evaluation.metric_contracts import (
    MetricContract,
    MetricContractRegistry,
    MetricEvaluationContext,
    MetricObservation,
    resolve_metric_observations,
)
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


def _case(
    metrics: dict[str, dict],
    *,
    context_overrides: dict | None = None,
    observed_values: dict[str, dict] | None = None,
    requested_use: str = "gate",
):
    contracts = []
    definitions = {}
    model_names = sorted(
        {model for definition in metrics.values() for model in definition["values"]}
    )
    for emitted_key, definition in metrics.items():
        framework = definition.get("framework", "syntheval")
        contract_options = {
            "contract_id": f"test.{framework}.{emitted_key}",
            "framework": framework,
            "emitted_key_pattern": emitted_key,
            "semantic_family": "privacy",
            "direction": "minimize",
            "value_role": "policy_scalar",
            "lifecycle_state": "operational",
            "allowed_uses": frozenset({"audit", "gate"}),
            "execution_pass": definition.get("execution_pass", "main"),
            "target_view": definition.get("target_view", "native"),
            "population_unit": definition.get("population_unit", "row"),
            "group_safety": definition.get("group_safety", "row_only"),
            "required_roles": ("train", "tuning"),
            "uncertainty_field": None,
            "sample_size_field": None,
            "status_reason": "",
        }
        contract_options.update(definition.get("contract", {}))
        if contract_options["lifecycle_state"] != "operational":
            contract_options["allowed_uses"] = frozenset({"audit"})
            contract_options["status_reason"] = contract_options.get("status_reason") or (
                "Test contract is intentionally not operational"
            )
        contract = MetricContract(**contract_options)
        contracts.append(contract)
        definitions[emitted_key] = (framework, contract)

    registry = MetricContractRegistry(contracts)
    context_options = {
        "role_hashes": {"train": "train-hash", "tuning": "tuning-hash"},
        "resolved_configuration": {"protocol": "test-v1"},
    }
    context_options.update(context_overrides or {})
    context = MetricEvaluationContext(**context_options)
    combined = pd.DataFrame(index=model_names)
    validations = {}
    observed_values = observed_values or {}
    for emitted_key, definition in metrics.items():
        framework, contract = definitions[emitted_key]
        combined[(framework, "privacy", emitted_key)] = pd.Series(
            definition["values"], index=model_names
        )
    combined.columns = pd.MultiIndex.from_tuples(combined.columns)
    for model_name in model_names:
        observations = []
        for emitted_key, definition in metrics.items():
            framework, contract = definitions[emitted_key]
            observation_options = {
                "execution_pass": contract.execution_pass,
                "target_view": contract.target_view,
                "role_hashes": dict(context.role_hashes),
            }
            observation_options.update(definition.get("observation", {}))
            raw_value = observed_values.get(emitted_key, {}).get(
                model_name, definition["values"].get(model_name)
            )
            observations.append(
                MetricObservation(
                    model_name=model_name,
                    framework=framework,
                    emitted_key=emitted_key,
                    raw_value=raw_value,
                    **observation_options,
                )
            )
        for framework in {item[0] for item in definitions.values()}:
            framework_keys = [
                key
                for key, (item_framework, _) in definitions.items()
                if item_framework == framework
            ]
            framework_observations = [
                observation for observation in observations if observation.framework == framework
            ]
            validations[(framework, context.execution_pass)] = {
                **validations.get((framework, context.execution_pass), {}),
                model_name: resolve_metric_observations(
                    registry=registry,
                    model_name=model_name,
                    framework=framework,
                    expected_keys=framework_keys,
                    observations=framework_observations,
                    context=context,
                    requested_use=requested_use,
                ),
            }
    thresholds = {
        definition.get("threshold_key", emitted_key): definition.get(
            "threshold", {"bound": "max", "value": 0.6}
        )
        for emitted_key, definition in metrics.items()
    }
    cfg = _FakeGateConfig(thresholds=thresholds)
    return combined, cfg, registry, validations, context


def _evaluate(case, **kwargs):
    combined, cfg, registry, validations, context = case
    kwargs.setdefault("context", context)
    return evaluate_privacy_gate(
        combined,
        cfg,
        registry=registry,
        validation_results=validations,
        **kwargs,
    )


class TestEvaluatePrivacyGate:
    def test_enabled_release_privacy_gate_consumes_task12_validation(self):
        digest = "a" * 64
        support = {
            "support_contract": "declared_support_v1",
            "state": "valid",
            "role_population_floor": 1,
            "protected_slices": {"state": "not_applicable"},
            "roles": {
                "synthetic": {"population": 20, "population_floor": 1, "role_hash": "syn"},
                "reference": {"population": 20, "population_floor": 1, "role_hash": "tune"},
            },
        }
        case = _case(
            {
                "release_privacy.v1": {
                    "framework": "custom",
                    "values": {"model_a": 0.2},
                    "contract": {
                        "contract_id": "test.custom.release_privacy.v1",
                        "required_support": "declared_support_v1",
                        "protocol_version": "task12-evaluation-v1",
                        "required_roles": ("train", "tuning"),
                    },
                    "observation": {
                        "role_hashes": {"train": "train-hash", "tuning": "tuning-hash"},
                        "fit_roles": ("train",),
                        "support": support,
                        "provenance": {
                            "producer": "task12_release_privacy",
                            "protocol_version": "task12-evaluation-v1",
                            "seed": 0,
                            "release_transform_digest": digest,
                            "release_support": support,
                            "common_protocol_digest": digest,
                            "role_hashes": {"train": "train-hash", "tuning": "tuning-hash"},
                        },
                    },
                    "threshold": {
                        "contract_id": "test.custom.release_privacy.v1",
                        "emitted_key": "release_privacy.v1",
                        "framework": "custom",
                        "bound": "max",
                        "value": 0.1,
                    },
                }
            },
            context_overrides={
                "resolved_configuration": {
                    "protocol": "test-v1",
                    "fit_roles": ("train",),
                    "release_transform_digest": digest,
                }
            },
        )
        result = _evaluate(case)

        assert result.loc["model_a", "status"] == "failed"
        assert result.loc["model_a", "pass"] == False  # noqa: E712

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

    def test_metric_not_found_makes_all_models_indeterminate(self):
        combined = _combined("mia_recall", {"model_a": 0.5})
        cfg = _FakeGateConfig(thresholds={"not_a_real_metric": {"bound": "max", "value": 0.6}})
        result = evaluate_privacy_gate(combined, cfg)
        assert result.loc["model_a", "pass"] == False  # noqa: E712
        assert result.loc["model_a", "status"] == "indeterminate"
        assert "contract" in result.loc["model_a", "violations"]

    def test_max_bound_pass_and_fail(self):
        case = _case(
            {
                "mia_recall": {
                    "values": {"model_a": 0.5, "model_b": 0.9},
                    "threshold": {"bound": "max", "value": 0.6},
                }
            }
        )
        result = _evaluate(case)
        assert bool(result.loc["model_a", "pass"])
        assert result.loc["model_b", "pass"] == False  # noqa: E712
        assert "mia_recall=0.9" in result.loc["model_b", "violations"]

    def test_model_specific_execution_pass_context_is_used(self):
        main_contract = MetricContract(
            contract_id="test.syntheval.mia_recall.main",
            framework="syntheval",
            emitted_key_pattern="mia_recall",
            semantic_family="privacy",
            direction="minimize",
            value_role="policy_scalar",
            lifecycle_state="operational",
            allowed_uses=frozenset({"audit", "gate"}),
            execution_pass="main",
            target_view="native",
            required_roles=("train", "tuning"),
        )
        binary_contract = dataclasses.replace(
            main_contract,
            contract_id="test.syntheval.mia_recall.binary",
            execution_pass="binary_target",
            target_view="binary_collapsed",
        )
        registry = MetricContractRegistry((main_contract, binary_contract))
        role_hashes = {"train": "train-hash", "tuning": "tuning-hash"}
        main_context = MetricEvaluationContext(
            execution_pass="main",
            target_view="native",
            role_hashes=role_hashes,
            resolved_configuration={"protocol": "test-v1"},
        )
        binary_context = dataclasses.replace(
            main_context,
            execution_pass="binary_target",
            target_view="binary_collapsed",
        )

        def validation(model_name, context, raw_value, error=None):
            return resolve_metric_observations(
                registry=registry,
                model_name=model_name,
                framework="syntheval",
                expected_keys=["mia_recall"],
                observations=[
                    MetricObservation(
                        model_name=model_name,
                        framework="syntheval",
                        emitted_key="mia_recall",
                        raw_value=raw_value,
                        execution_pass=context.execution_pass,
                        target_view=context.target_view,
                        direction="minimize",
                        error=error,
                        role_hashes=role_hashes,
                    )
                ],
                context=context,
                requested_use="gate",
            )

        combined = _combined("mia_recall", {"model_a": 0.5, "model_b": 0.4})
        validation_results = {
            ("syntheval", "main"): {
                "model_a": validation("model_a", main_context, 0.5),
                "model_b": validation("model_b", main_context, None, error="main failed"),
            },
            ("syntheval", "binary_target"): {
                "model_a": validation("model_a", binary_context, 0.55),
                "model_b": validation("model_b", binary_context, 0.4),
            },
        }
        cfg = _FakeGateConfig(thresholds={"mia_recall": {"bound": "max", "value": 0.6}})

        result = evaluate_privacy_gate(
            combined,
            cfg,
            registry=registry,
            validation_results=validation_results,
            context=main_context,
            contexts={
                ("syntheval", "main"): main_context,
                ("syntheval", "binary_target"): binary_context,
            },
            execution_passes={
                ("syntheval", "mia_recall", "model_a"): "main",
                ("syntheval", "mia_recall", "model_b"): "binary_target",
            },
        )

        assert result.loc["model_a", "pass"]
        assert result.loc["model_b", "pass"]
        assert result["status"].tolist() == ["eligible", "eligible"]

    def test_min_bound_pass_and_fail(self):
        case = _case(
            {
                "privacy.k-anonymization.syn": {
                    "framework": "synthcity",
                    "values": {"model_a": 10.0, "model_b": 2.0},
                    "contract": {"direction": "maximize"},
                    "threshold": {"bound": "min", "value": 5.0},
                }
            }
        )
        result = _evaluate(case)
        assert result.loc["model_a", "pass"] == True  # noqa: E712
        assert result.loc["model_b", "pass"] == False  # noqa: E712

    def test_nan_value_fails_conservatively_not_silently_passes(self):
        case = _case(
            {
                "mia_recall": {
                    "values": {"model_a": float("nan")},
                    "threshold": {"bound": "max", "value": 0.6},
                }
            }
        )
        result = _evaluate(case)
        assert result.loc["model_a", "pass"] == False  # noqa: E712
        assert "non_finite" in result.loc["model_a", "violations"]

    def test_multiple_thresholds_all_must_pass(self):
        case = _case(
            {
                "mia_recall": {
                    "values": {"model_a": 0.5, "model_b": 0.5},
                    "threshold": {"bound": "max", "value": 0.6},
                },
                "hit_rate": {
                    "values": {"model_a": 0.01, "model_b": 0.9},
                    "threshold": {"bound": "max", "value": 0.05},
                },
            }
        )
        result = _evaluate(case)
        assert result.loc["model_a", "pass"] == True  # noqa: E712
        assert result.loc["model_b", "pass"] == False  # noqa: E712
        assert "hit_rate" in result.loc["model_b", "violations"]
        assert "mia_recall" not in result.loc["model_b", "violations"]

    def test_partial_metric_availability_is_indeterminate(self):
        case = _case(
            {
                "mia_recall": {
                    "values": {"model_a": 0.5},
                    "threshold": {"bound": "max", "value": 0.6},
                }
            }
        )
        case[1].thresholds["not_computed_this_run"] = {"bound": "max", "value": 0.3}
        result = _evaluate(case)
        assert result is not None
        assert result.loc["model_a", "pass"] == False  # noqa: E712
        assert result.loc["model_a", "status"] == "indeterminate"

    def test_calibration_only_contract_is_indeterminate(self):
        case = _case(
            {
                "mia_recall": {
                    "values": {"model_a": 0.5},
                    "contract": {"lifecycle_state": "calibrating"},
                    "threshold": {"bound": "max", "value": 0.6},
                }
            }
        )
        result = _evaluate(case)
        assert result.loc["model_a", "status"] == "indeterminate"
        assert "calibrating" in result.loc["model_a", "violations"]

    def test_audit_only_contract_is_indeterminate(self):
        case = _case(
            {
                "mia_recall": {
                    "values": {"model_a": 0.5},
                    "contract": {"lifecycle_state": "audit_only"},
                    "threshold": {"bound": "max", "value": 0.6},
                }
            }
        )
        result = _evaluate(case)
        assert result.loc["model_a", "status"] == "indeterminate"
        assert "audit_only" in result.loc["model_a", "violations"]

    def test_wrong_framework_is_indeterminate(self):
        case = _case(
            {
                "mia_recall": {
                    "values": {"model_a": 0.5},
                    "threshold": {
                        "framework": "synthcity",
                        "bound": "max",
                        "value": 0.6,
                    },
                }
            }
        )
        result = _evaluate(case)
        assert result.loc["model_a", "status"] == "indeterminate"
        assert "No metric contract" in result.loc["model_a", "violations"]

    def test_wrong_execution_pass_is_indeterminate(self):
        case = _case(
            {
                "mia_recall": {
                    "values": {"model_a": 0.5},
                    "threshold": {
                        "contract_id": "test.syntheval.mia_recall",
                        "execution_pass": "binary_target",
                        "bound": "max",
                        "value": 0.6,
                    },
                }
            }
        )
        result = _evaluate(case)
        assert result.loc["model_a", "status"] == "indeterminate"
        assert "execution_pass" in result.loc["model_a", "violations"]

    def test_wrong_target_view_is_indeterminate(self):
        case = _case(
            {
                "mia_recall": {
                    "values": {"model_a": 0.5},
                    "threshold": {"bound": "max", "value": 0.6},
                }
            },
            context_overrides={"target_view": "binary_collapsed"},
        )
        result = _evaluate(case)
        assert result.loc["model_a", "status"] == "indeterminate"
        assert "target" in result.loc["model_a", "violations"]

    def test_wrong_role_hash_is_indeterminate(self):
        case = _case(
            {
                "mia_recall": {
                    "values": {"model_a": 0.5},
                    "threshold": {"bound": "max", "value": 0.6},
                }
            }
        )
        mismatched_context = dataclasses.replace(
            case[4], role_hashes={"train": "wrong", "tuning": "tuning-hash"}
        )
        result = _evaluate(case, context=mismatched_context)
        assert result.loc["model_a", "status"] == "indeterminate"
        assert "role hash" in result.loc["model_a", "violations"]

    def test_patient_group_contract_without_group_safety_is_indeterminate(self):
        case = _case(
            {
                "mia_recall": {
                    "values": {"model_a": 0.5},
                    "population_unit": "patient_group",
                    "group_safety": "row_only",
                    "threshold": {"bound": "max", "value": 0.6},
                }
            },
            context_overrides={
                "population_unit": "patient_group",
                "group_mode": "patient_group",
            },
        )
        result = _evaluate(case)
        assert result.loc["model_a", "status"] == "indeterminate"
        assert "group" in result.loc["model_a", "violations"]

    def test_missing_protocol_context_is_indeterminate(self):
        case = _case(
            {
                "mia_recall": {
                    "values": {"model_a": 0.5},
                    "threshold": {"bound": "max", "value": 0.6},
                }
            },
            context_overrides={"resolved_configuration": {}},
        )
        result = _evaluate(case)
        assert result.loc["model_a", "status"] == "indeterminate"
        assert "configuration context" in result.loc["model_a", "violations"]

    def test_missing_uncertainty_is_indeterminate(self):
        case = _case(
            {
                "mia_recall": {
                    "values": {"model_a": 0.5},
                    "contract": {"uncertainty_field": "err"},
                    "threshold": {"bound": "max", "value": 0.6},
                }
            }
        )
        result = _evaluate(case)
        assert result.loc["model_a", "status"] == "indeterminate"
        assert "uncertainty" in result.loc["model_a", "violations"]

    def test_missing_sample_size_is_indeterminate(self):
        case = _case(
            {
                "mia_recall": {
                    "values": {"model_a": 0.5},
                    "contract": {"sample_size_field": "n_val"},
                    "threshold": {"bound": "max", "value": 0.6},
                }
            }
        )
        result = _evaluate(case)
        assert result.loc["model_a", "status"] == "indeterminate"
        assert "sample-size" in result.loc["model_a", "violations"]

    def test_declared_sample_size_allows_gate_and_enforces_minimum(self):
        case = _case(
            {
                "corr_mat_diff_v2": {
                    "values": {"model_a": 0.5},
                    "contract": {"sample_size_field": "metadata.valid_pairs"},
                    "observation": {"sample_size": 4},
                    "threshold": {
                        "bound": "max",
                        "value": 0.6,
                        "minimum_sample_size": 4,
                    },
                }
            }
        )
        result = _evaluate(case)

        assert result.loc["model_a", "pass"]
        assert result.loc["model_a", "status"] == "eligible"

        case[1].thresholds["corr_mat_diff_v2"]["minimum_sample_size"] = 5
        result = _evaluate(case)

        assert result.loc["model_a", "pass"] == False  # noqa: E712
        assert result.loc["model_a", "status"] == "indeterminate"
        assert "sample size" in result.loc["model_a", "violations"]

    def test_combined_raw_value_mismatch_is_indeterminate(self):
        case = _case(
            {
                "mia_recall": {
                    "values": {"model_a": 0.5},
                    "threshold": {"bound": "max", "value": 0.6},
                }
            },
            observed_values={"mia_recall": {"model_a": 0.4}},
        )
        result = _evaluate(case)
        assert result.loc["model_a", "status"] == "indeterminate"
        assert "differs from validated" in result.loc["model_a", "violations"]

    def test_indeterminate_status_retains_valid_threshold_violation(self):
        case = _case(
            {
                "mia_recall": {
                    "values": {"model_a": 0.9},
                    "threshold": {"bound": "max", "value": 0.6},
                }
            }
        )
        case[1].thresholds["not_computed_this_run"] = {"bound": "max", "value": 0.3}
        result = _evaluate(case)
        assert result.loc["model_a", "status"] == "indeterminate"
        assert "mia_recall=0.9" in result.loc["model_a", "violations"]
        assert "not_computed_this_run" in result.loc["model_a", "violations"]


class TestMergePrivacyGateResults:
    def test_none_result_is_noop(self):
        combined = _combined("mia_recall", {"model_a": 0.5})
        merged = merge_privacy_gate_results(combined, None)
        assert merged.equals(combined)

    def test_merges_pass_and_violations_columns(self):
        case = _case(
            {
                "mia_recall": {
                    "values": {"model_a": 0.5},
                    "threshold": {"bound": "max", "value": 0.6},
                }
            }
        )
        combined, _, _, _, _ = case
        gate_result = _evaluate(case)
        merged = merge_privacy_gate_results(combined, gate_result)
        assert ("__all__", "privacy_gate", "pass") in merged.columns
        assert ("__all__", "privacy_gate", "status") in merged.columns
        assert ("__all__", "privacy_gate", "violations") in merged.columns
        assert merged.loc["model_a", ("__all__", "privacy_gate", "pass")] == True  # noqa: E712

    def test_gate_failure_invalidates_precomputed_policy_ranks(self):
        case = _case(
            {
                "mia_recall": {
                    "values": {"model_a": 0.9, "model_b": 0.1},
                    "threshold": {"bound": "max", "value": 0.6},
                }
            }
        )
        combined, _, _, _, _ = case
        combined[("__all__", "overall", "rank")] = [0.8, 0.2]
        combined.columns = pd.MultiIndex.from_tuples(combined.columns)

        gate_result = _evaluate(case)
        merged = merge_privacy_gate_results(combined, gate_result)

        assert pd.isna(merged.loc["model_a", ("__all__", "overall", "rank")])
        assert merged.loc["model_b", ("__all__", "overall", "rank")] == pytest.approx(0.2)
