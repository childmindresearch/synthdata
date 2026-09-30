"""Canonical release-evidence evaluation producers and validation."""

import math
import re
from collections.abc import Mapping

import pandas as pd

from synthdata.data import Dataset
from synthdata.evaluation import custom_eval
from synthdata.evaluation.metric_contracts import (
    DEFAULT_METRIC_CONTRACT_REGISTRY,
    MetricEvaluationContext,
    MetricObservation,
    MetricValidationResult,
    is_verified_authoritative_tstr,
    resolve_metric_observations,
)
from synthdata.evaluation.release import release_privacy_evidence, transform_release_roles

CANONICAL_RELEASE_EVIDENCE_KEYS = (
    "release_privacy.v1",
    "representation_evidence.v1",
    "equalized_odds.final.v1",
)
RELEASE_EVIDENCE_PROTOCOL_VERSION = "release-evidence-v2"
LEGACY_TASK12_PROTOCOL_VERSION = "task12-evaluation-v1"
_RELEASE_EVIDENCE_REASON_CODES = frozenset(
    {
        "release_evidence_release_or_representation_error",
        "release_evidence_final_fairness_error",
        "release_evidence_representation_error",
        "evaluator_exception",
    }
)
_UNTRUSTED_REASON_METADATA_KEYS = frozenset(
    {"invalid_reasons", "release_evidence_reason", "reason", "error"}
)
_SAFE_BLOCKED_METADATA_KEYS = frozenset(
    {
        "producer",
        "protocol_version",
        "producer_protocol_version",
        "seed",
        "release_transform_digest",
        "common_protocol_digest",
        "role_hashes",
        "fit_roles",
        "execution_pass",
        "support",
        "status",
        "error_code",
        "error_type",
    }
)
_SAFE_PRODUCERS = frozenset({"release_evidence_privacy", "release_evidence_representation"})
_SAFE_ERROR_TYPES = frozenset(
    {"AttributeError", "KeyError", "RuntimeError", "TypeError", "ValueError"}
)
_NON_SUCCESS_STATUSES = frozenset({"blocked", "indeterminate"})
_SAFE_PROTOCOL_VERSIONS = frozenset(
    {RELEASE_EVIDENCE_PROTOCOL_VERSION, LEGACY_TASK12_PROTOCOL_VERSION}
)
_SAFE_EXECUTION_PASSES = frozenset({"main", "final_audit"})
_SAFE_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ROLE_NAMES = frozenset({"train", "tuning", "final_holdout"})
_SAFE_SUPPORT_KEYS = frozenset(
    {
        "support_contract",
        "synthetic",
        "reference",
        "state",
        "slices",
        "target_classes",
        "protected_domains",
        "role_population_floor",
        "protected_slice_floor",
        "protected_slices",
    }
)


def _safe_exception_metadata(exc: Exception, reason_code: str) -> dict[str, str]:
    """Return non-sensitive evidence for a caught evaluation exception."""
    return {"error_code": reason_code, "error_type": type(exc).__name__}


def _bounded_reason_code(reason: object, fallback: str) -> str:
    """Return an allowlisted reason code without persisting producer text."""
    if isinstance(reason, str) and reason in _RELEASE_EVIDENCE_REASON_CODES:
        return reason
    return fallback


def _sanitize_role_hashes(value: object) -> dict[str, str]:
    """Keep only allowlisted role names paired with valid SHA-256 digests."""
    if not isinstance(value, Mapping):
        return {}
    return {
        name: digest
        for name, digest in value.items()
        if isinstance(name, str)
        and name in _SAFE_ROLE_NAMES
        and isinstance(digest, str)
        and _SAFE_DIGEST.fullmatch(digest) is not None
    }


def _sanitize_blocked_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    """Keep only bounded structural metadata for non-success observations."""
    sanitized: dict[str, object] = {}
    for key, value in metadata.items():
        if key in _UNTRUSTED_REASON_METADATA_KEYS or key not in _SAFE_BLOCKED_METADATA_KEYS:
            continue
        if key == "producer":
            if value in _SAFE_PRODUCERS:
                sanitized[key] = value
        elif key == "error_code":
            if isinstance(value, str) and value in _RELEASE_EVIDENCE_REASON_CODES:
                sanitized[key] = value
        elif key == "error_type":
            if isinstance(value, str) and value in _SAFE_ERROR_TYPES:
                sanitized[key] = value
        elif key == "role_hashes":
            valid_role_hashes = _sanitize_role_hashes(value)
            if valid_role_hashes:
                sanitized[key] = valid_role_hashes
        elif key == "fit_roles":
            if isinstance(value, (list, tuple)) and all(
                role in {"train", "tuning"} for role in value
            ):
                sanitized[key] = tuple(value)
        elif key == "support":
            if isinstance(value, Mapping):
                support = {}
                for support_key, support_value in value.items():
                    if support_key not in _SAFE_SUPPORT_KEYS:
                        continue
                    if (
                        (
                            support_key == "support_contract"
                            and support_value
                            in {"declared_support_v1", "all_target_protected_cells"}
                        )
                        or (
                            support_key in {"synthetic", "reference"}
                            and _safe_number(support_value)
                        )
                        or (
                            support_key == "state"
                            and support_value in _NON_SUCCESS_STATUSES | {"succeeded"}
                        )
                        or (
                            support_key in {"role_population_floor", "protected_slice_floor"}
                            and isinstance(support_value, int)
                            and not isinstance(support_value, bool)
                            and 1 <= support_value <= 1_000_000_000
                        )
                    ):
                        support[support_key] = support_value
                    elif support_key == "protected_slices" and isinstance(support_value, Mapping):
                        floor = support_value.get("floor")
                        if (
                            support_value.get("state") == "not_applicable"
                            and isinstance(floor, int)
                            and not isinstance(floor, bool)
                            and 1 <= floor <= 1_000_000_000
                        ):
                            support[support_key] = {"state": "not_applicable", "floor": floor}
                sanitized[key] = support
        elif key in {"protocol_version", "producer_protocol_version"}:
            if isinstance(value, str) and value in _SAFE_PROTOCOL_VERSIONS:
                sanitized[key] = value
        elif key in {"release_transform_digest", "common_protocol_digest"}:
            if isinstance(value, str) and _SAFE_DIGEST.fullmatch(value):
                sanitized[key] = value
        elif key == "execution_pass":
            if isinstance(value, str) and value in _SAFE_EXECUTION_PASSES:
                sanitized[key] = value
        elif (
            key == "seed"
            and isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value <= 2**63 - 1
        ):
            sanitized[key] = value
    return sanitized


def _safe_number(value: object) -> bool:
    """Return whether value is a bounded finite numeric support value."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and abs(value) <= 1_000_000_000
    )


def _release_evidence_record(
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
    """Build canonical release-evidence observation, including blocked metadata."""
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
            **dict(support),
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
        reason = reason or "release_evidence_release_or_representation_error"
    if status != "succeeded":
        status = status if status in _NON_SUCCESS_STATUSES else "indeterminate"
        value = None
        reason = _bounded_reason_code(reason, "release_evidence_release_or_representation_error")
        details = _sanitize_blocked_metadata(details)
    if fit_roles != expected_fit_roles:
        status = "blocked"
        value = None
        reason = _bounded_reason_code(reason, "release_evidence_release_or_representation_error")
        details = _sanitize_blocked_metadata(details)
    if status != "succeeded":
        details["status"] = status
    observation_role_hashes = (
        role_hashes if status == "succeeded" else _sanitize_role_hashes(role_hashes)
    )
    return MetricObservation(
        model_name=model_name,
        framework="custom",
        emitted_key=key,
        raw_value=value,
        execution_pass="final_audit" if key == "equalized_odds.final.v1" else "main",
        error=reason if status != "succeeded" else None,
        role_hashes=observation_role_hashes,
        source_metadata={
            "evidence_role": evaluation_role,
            "support": details["support"],
            **details,
            "status": status,
        },
        result_metadata={
            "evaluation_role": evaluation_role,
            "evidence_role": evaluation_role,
            "support": details["support"],
            **details,
            "status": status,
        },
        fit_roles=fit_roles,
        support=details["support"],
        provenance=details,
    )


def run_release_evidence_evaluation(
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
    role_population_floor: int = 20,
    protected_slice_floor: int = 1,
    tstr_results: Mapping[str, object] | None = None,
    seed: int = 0,
    release_form_inputs: tuple[pd.DataFrame, Mapping[str, pd.DataFrame], Mapping[str, object]]
    | None = None,
) -> dict[str, list[MetricObservation]]:
    """Produce canonical release, representation, and final fairness observations.

    Missing release artifacts remain explicit blocked observations. Candidate
    callers use ``tuning``; final callers use ``final_holdout``. Equalized odds
    is never emitted outside ``final_audit``.
    Population support is checked on both synthetic and reference rows.
    Protected-slice support is not applicable to privacy (no protected
    population assessed); final TSTR owns the minimum total, actual-positive,
    and actual-negative rows per protected group/OVR class view.
    """
    if release_form_inputs is not None and evaluation_role == "final_holdout":
        _, released_roles, _ = release_form_inputs
        reference = (
            released_roles.get(evaluation_role) if isinstance(released_roles, Mapping) else None
        )
        if not isinstance(reference, pd.DataFrame):
            reference = pd.DataFrame()
    else:
        reference = dataset.role_frame(evaluation_role, imputed=False)
        if reference is None:
            reference = pd.DataFrame()
    identity_metadata = dataset.role_metadata.get("identity", {})
    patient_id_column = (
        identity_metadata.get("identity_column") if isinstance(identity_metadata, Mapping) else None
    )
    released_roles = {evaluation_role: reference}
    results: dict[str, list[MetricObservation]] = {}
    for name, frame in synthetic_datasets.items():
        observations: list[MetricObservation] = []
        floor_support = {
            "role_population_floor": role_population_floor,
            "protected_slice_floor": protected_slice_floor,
        }
        privacy_support = {
            **floor_support,
            "protected_slices": {"state": "not_applicable", "floor": protected_slice_floor},
        }
        try:
            if release_form_inputs is None:
                released_synthetic, released, _metadata = transform_release_roles(
                    frame,
                    released_roles,
                    generalization,
                )
            else:
                prepared_synthetic, prepared_roles, _metadata = release_form_inputs
                released_synthetic = prepared_synthetic
                released = prepared_roles
            release = release_privacy_evidence(
                released_synthetic,
                released[evaluation_role],
                quasi_identifiers=quasi_identifiers,
                sensitive_fields=sensitive_fields,
                patient_id_column=patient_id_column,
                k_required=k_required,
                l_required=l_required,
                seed=seed,
                role_population_floor=role_population_floor,
                protected_slice_floor=protected_slice_floor,
            )
            release_status = release.get("status", "indeterminate")
            release_value = release.get("value")
            common_metadata = {
                "support": {
                    **floor_support,
                    "support_contract": "declared_support_v1",
                    "synthetic": len(released_synthetic),
                    "reference": len(released[evaluation_role]),
                },
            }
            observations.append(
                _release_evidence_record(
                    name,
                    "release_privacy.v1",
                    release_value,
                    role_hashes=role_hashes,
                    evaluation_role=evaluation_role,
                    status=release_status,
                    reason=(
                        "release_evidence_release_or_representation_error"
                        if release.get("invalid_reasons")
                        else None
                    ),
                    metadata={
                        **release,
                        **common_metadata,
                        "support": {**common_metadata["support"], **privacy_support},
                    },
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
                reference_frame=reference
                if release_form_inputs is not None and evaluation_role == "final_holdout"
                else None,
            ).get(name, {})
            safety = representation.get("summary_stats", {}).get("representation_safety")
            observations.append(
                _release_evidence_record(
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
                        "producer": "release_evidence_representation",
                        "protocol_version": RELEASE_EVIDENCE_PROTOCOL_VERSION,
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
            for key in CANONICAL_RELEASE_EVIDENCE_KEYS[:2]:
                observations.append(
                    _release_evidence_record(
                        name,
                        key,
                        None,
                        role_hashes=role_hashes,
                        evaluation_role=evaluation_role,
                        status="blocked",
                        reason="release_evidence_release_or_representation_error",
                        metadata={
                            **_safe_exception_metadata(
                                exc, "release_evidence_release_or_representation_error"
                            ),
                            "support": privacy_support
                            if key == "release_privacy.v1"
                            else floor_support,
                        },
                    )
                )
        if evaluation_role == "final_holdout":
            tstr = (tstr_results or {}).get(name)
            fairness_support = dict(floor_support)
            try:
                if tstr is None:
                    raise ValueError("authoritative final TSTR result is unavailable")
                envelope = getattr(tstr, "envelope", None)
                if envelope is None and isinstance(tstr, Mapping):
                    envelope = tstr
                if not isinstance(envelope, Mapping) or not isinstance(
                    envelope.get("report"), Mapping
                ):
                    raise ValueError("authoritative final TSTR envelope is unavailable")
                tstr_report = envelope["report"]
                if (
                    envelope.get("producer") != "authoritative_tstr"
                    or envelope.get("protocol_version") != "tstr-v1"
                ):
                    raise ValueError(
                        "authoritative final TSTR envelope producer or protocol is invalid"
                    )
                if envelope.get("result_metadata") != tstr_report.get("result_metadata"):
                    raise ValueError(
                        "authoritative final TSTR envelope metadata binding is invalid"
                    )
                if not isinstance(tstr_report, Mapping) or not is_verified_authoritative_tstr(
                    tstr_report
                ):
                    raise ValueError(
                        "authoritative TSTR producer metadata or artifact is unverified"
                    )
                fairness = (
                    tstr_report.get("equalized_odds") if isinstance(tstr_report, Mapping) else None
                )
                if isinstance(fairness, Mapping) and isinstance(fairness.get("support"), Mapping):
                    fairness_support.update(fairness["support"])
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
                    _release_evidence_record(
                        name,
                        "equalized_odds.final.v1",
                        float(value),
                        role_hashes=role_hashes,
                        evaluation_role=evaluation_role,
                        status="succeeded",
                        metadata={
                            "tstr_report": tstr_report,
                            "prediction_artifact": prediction_artifact,
                            "support": fairness_support,
                            "support_policy": tstr_report["result_metadata"].get(
                                "support_policy", {}
                            ),
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
                            "protocol_version": RELEASE_EVIDENCE_PROTOCOL_VERSION,
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
                            "role_hashes": tstr_report["result_metadata"].get("role_hashes"),
                        },
                    )
                )
            except (KeyError, ValueError, TypeError, AttributeError, RuntimeError) as exc:
                observations.append(
                    _release_evidence_record(
                        name,
                        "equalized_odds.final.v1",
                        None,
                        role_hashes=role_hashes,
                        evaluation_role=evaluation_role,
                        status="blocked",
                        reason="release_evidence_final_fairness_error",
                        metadata={
                            **_safe_exception_metadata(
                                exc, "release_evidence_final_fairness_error"
                            ),
                            "support": fairness_support,
                        },
                    )
                )
        else:
            observations.append(
                _release_evidence_record(
                    name,
                    "equalized_odds.final.v1",
                    None,
                    role_hashes=role_hashes,
                    evaluation_role=evaluation_role,
                    status="blocked",
                    reason="Equalized odds requires evaluation_role=final_holdout",
                    metadata={"execution_pass": "final_audit", "support": floor_support},
                )
            )
        results[name] = observations
    return results


def validate_release_evidence_results(
    observations_by_model: Mapping[str, list[MetricObservation]],
    *,
    role_hashes: Mapping[str, str],
    evaluation_role: str,
    population_unit: str = "row",
    group_mode: str = "row",
    requested_use: str = "audit",
) -> dict[str, MetricValidationResult]:
    """Validate canonical release-evidence records without dropping blocked keys."""
    expected = (
        CANONICAL_RELEASE_EVIDENCE_KEYS
        if evaluation_role == "final_holdout"
        else CANONICAL_RELEASE_EVIDENCE_KEYS[:2]
    )
    return {
        model: _validate_release_evidence_model(
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


def _validate_release_evidence_model(
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
