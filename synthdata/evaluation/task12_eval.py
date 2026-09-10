"""Task 12 custom evaluation producers and validation."""

from collections.abc import Mapping

import pandas as pd

from synthdata.data import Dataset
from synthdata.evaluation import custom_eval
from synthdata.evaluation.metric_contracts import (
    DEFAULT_METRIC_CONTRACT_REGISTRY,
    MetricEvaluationContext,
    MetricObservation,
    MetricValidationResult,
    is_verified_task10_tstr,
    resolve_metric_observations,
)
from synthdata.evaluation.release import release_privacy_evidence, transform_release_roles

TASK12_CUSTOM_KEYS = (
    "release_privacy.v1",
    "representation_evidence.v1",
    "equalized_odds.final.v1",
)
TASK12_PROTOCOL_VERSION = "task12-evaluation-v1"


def _task12_record(
    model_name: str,
    key: str,
    value: float | None,
    *,
    role_hashes: Mapping[str, str],
    evaluation_role: str,
    status: str = "succeeded",
    reason: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> MetricObservation:
    """Build canonical Task 12 observation, including blocked evidence metadata."""
    details = dict(metadata or {})
    raw_fit_roles = details.get("fit_roles", ())
    fit_roles = tuple(raw_fit_roles) if isinstance(raw_fit_roles, (list, tuple)) else ()
    expected_fit_roles = ("train",) if evaluation_role == "tuning" else ("train", "tuning")
    support = details.get("support", {})
    if not isinstance(support, Mapping):
        support = {"value": support}
    support_contract = (
        "all_target_protected_cells" if key == "equalized_odds.final.v1" else "declared_support_v1"
    )
    if key == "equalized_odds.final.v1" and details.get("support_slices"):
        support = {
            "state": details.get("support_state", "indeterminate"),
            "slices": details["support_slices"],
            "target_classes": details.get("target_classes", ()),
            "protected_domains": details.get("protected_domains", {}),
        }
    details["support"] = {"support_contract": support_contract, **dict(support)}
    details["support"]["support_contract"] = support_contract
    required_metadata = (
        "producer",
        "protocol_version",
        "seed",
        "release_transform_digest",
        "common_protocol_digest",
        "role_hashes",
    )
    missing_metadata = [field for field in required_metadata if field not in details]
    if missing_metadata:
        status = "blocked"
        value = None
        reason = reason or ("Missing required producer metadata: " + ", ".join(missing_metadata))
    if status != "succeeded":
        value = None
        reason = reason or f"Producer reported non-success status: {status}"
    if fit_roles != expected_fit_roles:
        status = "blocked"
        value = None
        reason = reason or (
            f"Producer fit_roles {fit_roles!r} do not match expected {expected_fit_roles!r}"
        )
    return MetricObservation(
        model_name=model_name,
        framework="custom",
        emitted_key=key,
        raw_value=value,
        execution_pass="final_audit" if key == "equalized_odds.final.v1" else "main",
        error=reason if status != "succeeded" else None,
        role_hashes=role_hashes,
        source_metadata={
            "status": status,
            "evidence_role": evaluation_role,
            "support": details["support"],
            **details,
        },
        result_metadata={
            "evaluation_role": evaluation_role,
            "evidence_role": evaluation_role,
            "status": status,
            "support": details["support"],
            **details,
        },
        fit_roles=fit_roles,
        support=details["support"],
        provenance=details,
    )


def run_task12_custom_evaluation(
    synthetic_datasets: dict[str, pd.DataFrame],
    dataset: Dataset,
    *,
    evaluation_role: str,
    generalization: Mapping[str, object] | None,
    quasi_identifiers: list[str],
    sensitive_fields: list[str],
    protected_columns: list[str],
    role_hashes: Mapping[str, str],
    k_required: int = 5,
    l_required: int = 2,
    tstr_results: Mapping[str, object] | None = None,
    seed: int = 0,
) -> dict[str, list[MetricObservation]]:
    """Produce canonical release, representation, and final fairness observations.

    Missing release artifacts remain explicit blocked observations. Candidate
    callers use ``tuning``; final callers use ``final_holdout``. Equalized odds
    is never emitted outside ``final_audit``.
    """
    reference = dataset.role_frame(evaluation_role, imputed=False)
    if reference is None:
        reference = pd.DataFrame()
    released_roles = {evaluation_role: reference}
    results: dict[str, list[MetricObservation]] = {}
    for name, frame in synthetic_datasets.items():
        observations: list[MetricObservation] = []
        try:
            released_synthetic, released, _metadata = transform_release_roles(
                frame,
                released_roles,
                generalization,
            )
            release = release_privacy_evidence(
                released_synthetic,
                released[evaluation_role],
                quasi_identifiers=quasi_identifiers,
                sensitive_fields=sensitive_fields,
                k_required=k_required,
                l_required=l_required,
                seed=seed,
            )
            release_status = release.get("status", "indeterminate")
            release_value = release.get("value")
            common_metadata = {
                "support": {
                    "support_contract": "declared_support_v1",
                    "synthetic": len(released_synthetic),
                    "reference": len(released[evaluation_role]),
                },
            }
            observations.append(
                _task12_record(
                    name,
                    "release_privacy.v1",
                    release_value,
                    role_hashes=role_hashes,
                    evaluation_role=evaluation_role,
                    status=release_status,
                    reason="; ".join(release.get("invalid_reasons", [])) or None,
                    metadata={**release, **common_metadata},
                )
            )
            representation = custom_eval.run_log_disparity_evaluation(
                {name: released_synthetic},
                dataset,
                type(
                    "Config",
                    (),
                    {
                        "protected_columns": protected_columns,
                        "target_map": None,
                        "protected_map": None,
                        "protected_bins": None,
                    },
                )(),
                type("Selection", (), {"enabled": True, "categories": [], "metrics": []})(),
                evaluation_role=evaluation_role,
            ).get(name, {})
            safety = representation.get("summary_stats", {}).get("representation_safety")
            observations.append(
                _task12_record(
                    name,
                    "representation_evidence.v1",
                    float(1.0 - safety) if safety is not None and pd.notna(safety) else None,
                    role_hashes=role_hashes,
                    evaluation_role=evaluation_role,
                    status="succeeded"
                    if safety is not None and pd.notna(safety)
                    else "indeterminate",
                    reason=representation.get("result_metadata", {}).get("release_evidence_reason"),
                    metadata={
                        **representation.get("result_metadata", {}),
                        **common_metadata,
                        "producer": "task12_representation",
                        "protocol_version": TASK12_PROTOCOL_VERSION,
                        "producer_protocol_version": "log-disparity-v1",
                        "seed": seed,
                        "release_transform_digest": released_synthetic.attrs[
                            "release_provenance"
                        ].get("release_transform_digest"),
                        "common_protocol_digest": released_synthetic.attrs[
                            "release_provenance"
                        ].get("common_protocol_digest"),
                        "role_hashes": role_hashes,
                        "fit_roles": ["train"]
                        if evaluation_role == "tuning"
                        else ["train", "tuning"],
                    },
                )
            )
        except (KeyError, ValueError, TypeError) as exc:
            for key in TASK12_CUSTOM_KEYS[:2]:
                observations.append(
                    _task12_record(
                        name,
                        key,
                        None,
                        role_hashes=role_hashes,
                        evaluation_role=evaluation_role,
                        status="blocked",
                        reason=str(exc),
                    )
                )
        if evaluation_role == "final_holdout":
            tstr = (tstr_results or {}).get(name)
            try:
                if tstr is None:
                    raise ValueError("authoritative final TSTR result is unavailable")
                envelope = getattr(tstr, "envelope", None)
                if envelope is None and isinstance(tstr, Mapping):
                    envelope = tstr
                if not isinstance(envelope, Mapping) or not isinstance(
                    envelope.get("report"), Mapping
                ):
                    raise ValueError("authoritative Task 10 envelope is unavailable")
                tstr_report = envelope["report"]
                if (
                    envelope.get("producer") != "task10_tstr"
                    or envelope.get("protocol_version") != "tstr-v1"
                ):
                    raise ValueError(
                        "authoritative Task 10 envelope producer or protocol is invalid"
                    )
                if envelope.get("result_metadata") != tstr_report.get("result_metadata"):
                    raise ValueError("authoritative Task 10 envelope metadata binding is invalid")
                if not isinstance(tstr_report, Mapping) or not is_verified_task10_tstr(tstr_report):
                    raise ValueError("Task 10 TSTR producer metadata or artifact is unverified")
                fairness = (
                    tstr_report.get("equalized_odds") if isinstance(tstr_report, Mapping) else None
                )
                value = (
                    fairness.get("macro_valid_slice_score")
                    if isinstance(fairness, Mapping) and fairness.get("state") == "complete"
                    else None
                )
                if value is None:
                    raise ValueError("verified final TSTR equalized-odds result unavailable")
                prediction_artifact = (
                    tstr_report.get("prediction_artifact")
                    if isinstance(tstr_report, Mapping)
                    else None
                )
                if not isinstance(prediction_artifact, Mapping) or not prediction_artifact.get(
                    "verified"
                ):
                    raise ValueError("authoritative final TSTR prediction artifact unavailable")
                if prediction_artifact.get("source_role") != "final_holdout":
                    raise ValueError("authoritative final TSTR artifact is not final-holdout bound")
                if prediction_artifact.get("role_hashes") != dict(role_hashes):
                    raise ValueError("authoritative final TSTR artifact role hashes are untrusted")
                if prediction_artifact.get("common_protocol_digest") != tstr_report[
                    "result_metadata"
                ].get("common_protocol_digest"):
                    raise ValueError(
                        "authoritative final TSTR artifact protocol binding mismatches"
                    )
                observations.append(
                    _task12_record(
                        name,
                        "equalized_odds.final.v1",
                        float(value),
                        role_hashes=role_hashes,
                        evaluation_role=evaluation_role,
                        status="succeeded",
                        metadata={
                            "tstr_report": tstr_report,
                            "prediction_artifact": prediction_artifact,
                            "support": tstr_report.get("class_supports", {})
                            if isinstance(tstr_report, Mapping)
                            else {},
                            "support_state": fairness.get("state")
                            if isinstance(fairness, Mapping)
                            else "indeterminate",
                            "support_slices": fairness.get("slices", [])
                            if isinstance(fairness, Mapping)
                            else [],
                            "target_classes": tstr_report.get("target_order", [])
                            if isinstance(tstr_report, Mapping)
                            else [],
                            "protected_domains": (
                                tstr_report.get("protected_domains", {})
                                if isinstance(tstr_report, Mapping)
                                else {}
                            )
                            or {
                                column: sorted(
                                    {
                                        rate.get("group")
                                        for cell in (
                                            fairness.get("slices", [])
                                            if isinstance(fairness, Mapping)
                                            else []
                                        )
                                        if cell.get("protected_column") == column
                                        for rate in cell.get("group_rates", [])
                                        if isinstance(rate, Mapping)
                                    },
                                    key=str,
                                )
                                for column in sorted(
                                    {
                                        cell.get("protected_column")
                                        for cell in (
                                            fairness.get("slices", [])
                                            if isinstance(fairness, Mapping)
                                            else []
                                        )
                                        if isinstance(cell, Mapping)
                                        and cell.get("protected_column") is not None
                                    }
                                )
                            },
                            "execution_pass": "final_audit",
                            "producer": tstr_report["result_metadata"].get("producer"),
                            "producer_protocol_version": tstr_report["result_metadata"].get(
                                "protocol_version"
                            ),
                            "protocol_version": TASK12_PROTOCOL_VERSION,
                            "seed": tstr_report["result_metadata"].get("seed"),
                            "common_protocol_digest": tstr_report["result_metadata"].get(
                                "common_protocol_digest"
                            ),
                            "release_transform_digest": tstr_report["result_metadata"].get(
                                "release_transform_digest"
                            ),
                            "target_population_identity": tstr_report["result_metadata"].get(
                                "target_population_identity"
                            ),
                            "protected_population_identity": tstr_report["result_metadata"].get(
                                "protected_population_identity"
                            ),
                            "fit_roles": tstr_report["result_metadata"].get("fit_roles"),
                        },
                    )
                )
            except (KeyError, ValueError, TypeError, AttributeError, RuntimeError) as exc:
                observations.append(
                    _task12_record(
                        name,
                        "equalized_odds.final.v1",
                        None,
                        role_hashes=role_hashes,
                        evaluation_role=evaluation_role,
                        status="blocked",
                        reason=str(exc),
                    )
                )
        else:
            observations.append(
                _task12_record(
                    name,
                    "equalized_odds.final.v1",
                    None,
                    role_hashes=role_hashes,
                    evaluation_role=evaluation_role,
                    status="blocked",
                    reason="Equalized odds requires evaluation_role=final_holdout",
                    metadata={"execution_pass": "final_audit"},
                )
            )
        results[name] = observations
    return results


def validate_task12_custom_results(
    observations_by_model: Mapping[str, list[MetricObservation]],
    *,
    role_hashes: Mapping[str, str],
    evaluation_role: str,
    population_unit: str = "row",
    group_mode: str = "row",
    requested_use: str = "audit",
) -> dict[str, MetricValidationResult]:
    """Validate canonical Task 12 custom records without dropping blocked keys."""
    expected = TASK12_CUSTOM_KEYS if evaluation_role == "final_holdout" else TASK12_CUSTOM_KEYS[:2]
    return {
        model: _validate_task12_model(
            model,
            records,
            expected,
            role_hashes,
            evaluation_role,
            population_unit,
            group_mode,
            requested_use,
        )
        for model, records in observations_by_model.items()
    }


def _validate_task12_model(
    model: str,
    records: list[MetricObservation],
    expected: tuple[str, ...],
    role_hashes: Mapping[str, str],
    evaluation_role: str,
    population_unit: str,
    group_mode: str,
    requested_use: str,
) -> MetricValidationResult:
    """Validate main-pass and final-audit records under their owning contracts."""
    contexts = {
        "main": MetricEvaluationContext(
            role_hashes=role_hashes,
            evaluation_role=evaluation_role,
            population_unit=population_unit,
            group_mode=group_mode,
            execution_pass="main",
        ),
        "final_audit": MetricEvaluationContext(
            role_hashes=role_hashes,
            evaluation_role=evaluation_role,
            population_unit=population_unit,
            group_mode=group_mode,
            execution_pass="final_audit",
        ),
    }
    partitions = {
        pass_name: [item for item in records if item.execution_pass == pass_name]
        for pass_name in contexts
    }
    expected_by_pass = {
        "main": [key for key in expected if key != "equalized_odds.final.v1"],
        "final_audit": [key for key in expected if key == "equalized_odds.final.v1"],
    }
    validated = [
        resolve_metric_observations(
            registry=DEFAULT_METRIC_CONTRACT_REGISTRY,
            model_name=model,
            framework="custom",
            expected_keys=expected_by_pass[pass_name],
            observations=partitions[pass_name],
            context=contexts[pass_name],
            requested_use=requested_use,
        )
        for pass_name in contexts
        if expected_by_pass[pass_name]
    ]
    return MetricValidationResult(
        model_name=model,
        requested_use=requested_use,
        contract_digest=DEFAULT_METRIC_CONTRACT_REGISTRY.digest(),
        records=tuple(record for result in validated for record in result.records),
        evaluation_context=contexts["final_audit"]
        if evaluation_role == "final_holdout"
        else contexts["main"],
    )
