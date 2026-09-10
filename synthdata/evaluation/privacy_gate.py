"""Absolute privacy audit evidence.

The combined evaluation table's ranked/scaled columns (see
:mod:`synthdata.evaluation.combine`) only tell you which model looks *better
or worse than the other candidates in this run* -- a model can rank "best on
privacy" purely by comparison while still leaking an unacceptable absolute
amount, which is exactly the wrong thing to optimize for when working with
sensitive healthcare data. This module checks each model's RAW metric value
(from the combined table, before any min-max scaling) against a fixed
threshold configured in ``evaluation.privacy_gate`` (see
:class:`synthdata.config.PrivacyGateConfig`), producing a per-model
pass/fail verdict that is surfaced (never silently hidden, and never used to
silently drop a model from the ranked table).
"""

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

import pandas as pd

from synthdata.evaluation.metric_contracts import (
    DEFAULT_METRIC_CONTRACT_REGISTRY,
    AmbiguousMetricContractError,
    MetricContract,
    MetricEvaluationContext,
    MetricStatusRecord,
    MetricValidationResult,
    UnknownMetricContractError,
)
from synthdata.utils import get_logger

logger = get_logger(__name__)

_PASS_COL = "pass"
_VIOLATIONS_COL = "violations"
_STATUS_COL = "status"


def _combined_metric_columns(
    columns: pd.Index,
    emitted_key: str,
    framework: str | None = None,
) -> list[tuple]:
    """Return raw privacy columns for one exact emitted metric identity."""
    return [
        column
        for column in columns
        if isinstance(column, tuple)
        and len(column) == 3
        and column[1] == "privacy"
        and column[2] == emitted_key
        and (framework is None or column[0] == framework)
    ]


def _find_metric_column(columns: pd.Index, metric_name: str) -> tuple | None:
    """Find one unambiguous raw privacy column by emitted metric identity."""
    matches = _combined_metric_columns(columns, metric_name)
    return matches[0] if len(matches) == 1 else None


def _as_finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _validation_for_model(
    validation_results: Mapping[Any, Any] | None,
    model_name: str,
    framework: str,
    execution_pass: str,
) -> MetricValidationResult | None:
    """Read a validation result from grouped or direct framework evidence."""
    if validation_results is None:
        return None
    grouped = validation_results.get((framework, execution_pass))
    if isinstance(grouped, Mapping):
        validation = grouped.get(model_name)
        if isinstance(validation, MetricValidationResult):
            return validation
    direct = validation_results.get(model_name)
    if isinstance(direct, MetricValidationResult):
        return direct
    return None


def _record_for_key(
    validation: MetricValidationResult | None,
    emitted_key: str,
) -> MetricStatusRecord | None:
    if validation is None:
        return None
    matches = [
        record for record in validation.expected_records if record.expected_key == emitted_key
    ]
    return matches[0] if len(matches) == 1 else None


def _context_for_validation(
    validation: MetricValidationResult | None,
    context: MetricEvaluationContext | None,
    contexts: Mapping[tuple[str, str], MetricEvaluationContext] | None,
    framework: str,
    execution_pass: str,
) -> MetricEvaluationContext | None:
    if contexts is not None:
        selected = contexts.get((framework, execution_pass))
        if selected is not None:
            return selected
    if context is not None:
        return context
    if validation is not None:
        return validation.evaluation_context
    return None


def _execution_pass_for_metric(
    execution_passes: Mapping[tuple[str, ...], str],
    framework: str | None,
    emitted_key: str,
    model_name: Any | None,
    default: str,
) -> str:
    if framework is None:
        return default
    if model_name is not None:
        owner = execution_passes.get((framework, emitted_key, str(model_name)))
        if owner is not None:
            return owner
    legacy_owner = execution_passes.get((framework, emitted_key))
    return legacy_owner if legacy_owner is not None else default


def _context_mismatch_reason(
    *,
    record: MetricStatusRecord,
    contract: MetricContract,
    context: MetricEvaluationContext | None,
    registry_digest: str,
    validation: MetricValidationResult | None,
    spec: Mapping[str, Any],
) -> str | None:
    """Return the first evidence mismatch that blocks a gate check."""
    if validation is None:
        return "metric validation evidence is unavailable"
    if validation.contract_digest != registry_digest:
        return (
            "metric validation used a different contract registry digest "
            f"({validation.contract_digest!r} != {registry_digest!r})"
        )
    if context is None:
        return "metric evaluation context is unavailable"
    if record.status != "succeeded":
        return f"metric status is {record.status!r}"
    if record.contract_id != contract.contract_id:
        return (
            f"validation resolved contract {record.contract_id!r}, "
            f"expected {contract.contract_id!r}"
        )
    if record.lifecycle_state != contract.lifecycle_state:
        return "validation lifecycle state differs from the resolved contract"
    if "gate" not in record.allowed_uses:
        return "validated metric record does not allow gate use"
    if record.execution_pass != context.execution_pass:
        return f"execution pass mismatch ({record.execution_pass!r} != {context.execution_pass!r})"
    if contract.execution_pass != context.execution_pass:
        return (
            f"contract execution pass mismatch ({contract.execution_pass!r} != "
            f"{context.execution_pass!r})"
        )
    if record.target_view != context.target_view or contract.target_view != context.target_view:
        return (
            f"target view mismatch (contract={contract.target_view!r}, "
            f"record={record.target_view!r}, context={context.target_view!r})"
        )
    if record.population_unit != context.population_unit:
        return (
            f"population-unit mismatch ({record.population_unit!r} != {context.population_unit!r})"
        )
    if contract.population_unit != context.population_unit:
        return (
            f"contract population unit mismatch ({contract.population_unit!r} != "
            f"{context.population_unit!r})"
        )
    if record.group_mode != context.group_mode:
        return f"group-mode mismatch ({record.group_mode!r} != {context.group_mode!r})"
    if context.population_unit == "patient_group" and contract.group_safety != "group_safe":
        return "contract is not group-safe for patient_group evaluation"
    required_roles = set(record.required_roles)
    missing_roles = sorted(required_roles - set(context.role_hashes))
    if missing_roles:
        return f"required role hash(es) missing from gate context: {missing_roles}"
    for role in sorted(required_roles):
        expected_hash = context.role_hashes.get(role)
        observed_hash = record.role_hashes.get(role)
        if not expected_hash or observed_hash != expected_hash:
            return f"required role hash mismatch for {role!r}"
    required_protocol_context = spec.get("protocol_context")
    if required_protocol_context is not None:
        if not isinstance(required_protocol_context, Mapping):
            return "threshold protocol_context must be a mapping"
        actual_context = context.resolved_configuration
        mismatches = {
            key: expected
            for key, expected in required_protocol_context.items()
            if actual_context.get(key) != expected
        }
        if mismatches:
            return f"protocol context mismatch: {mismatches}"
    elif not context.resolved_configuration:
        return "resolved protocol/configuration context is unavailable"
    if contract.uncertainty_field is not None and record.uncertainty is None:
        return f"required uncertainty field {contract.uncertainty_field!r} is missing"
    if contract.sample_size_field is not None and record.sample_size is None:
        return f"required sample-size field {contract.sample_size_field!r} is missing"
    if spec.get("require_uncertainty", False) and record.uncertainty is None:
        return "threshold requires uncertainty evidence"
    if record.uncertainty is not None and _as_finite_number(record.uncertainty) is None:
        return "uncertainty is non-finite or not numeric"
    if record.sample_size is not None and (
        isinstance(record.sample_size, bool)
        or not isinstance(record.sample_size, int)
        or record.sample_size < 1
    ):
        return "sample size is not a positive integer"
    minimum_sample_size = spec.get("minimum_sample_size")
    if minimum_sample_size is not None and (
        isinstance(minimum_sample_size, bool)
        or not isinstance(minimum_sample_size, int)
        or minimum_sample_size < 1
    ):
        return "threshold minimum_sample_size must be a positive integer"
    if minimum_sample_size is not None and (
        record.sample_size is None or record.sample_size < minimum_sample_size
    ):
        return (
            f"sample size {record.sample_size!r} is below the required minimum "
            f"{minimum_sample_size}"
        )
    return None


def _resolve_gate_contract(
    metric_name: str,
    spec: Mapping[str, Any],
    combined: pd.DataFrame,
    registry,
    context: MetricEvaluationContext | None,
    execution_passes: Mapping[tuple[str, ...], str],
    model_name: Any | None = None,
) -> tuple[MetricContract | None, str | None, tuple | None, str, str | None]:
    """Resolve a threshold to one exact emitted key, column, and contract."""
    contract_id = spec.get("contract_id")
    emitted_key = spec.get("emitted_key", metric_name)
    requested_framework = spec.get("framework")
    if not isinstance(emitted_key, str) or not emitted_key:
        return None, "threshold emitted_key must be a non-empty string", None, "main", None
    if requested_framework is not None and (
        not isinstance(requested_framework, str) or not requested_framework
    ):
        return None, "threshold framework must be a non-empty string", None, "main", None

    contract = None
    if contract_id is not None:
        if not isinstance(contract_id, str) or not contract_id:
            return None, "threshold contract_id must be a non-empty string", None, "main", None
        try:
            contract = registry.get(contract_id)
        except UnknownMetricContractError as exc:
            return None, str(exc), None, "main", requested_framework
        if requested_framework is not None and requested_framework != contract.framework:
            return (
                contract,
                f"threshold framework does not match contract ({requested_framework!r} != "
                f"{contract.framework!r})",
                None,
                contract.execution_pass,
                contract.framework,
            )
        if "*" in contract.emitted_key_pattern:
            return (
                None,
                "privacy gate thresholds must name an exact emitted key; wildcard contract "
                f"{contract_id!r} is not directly gateable",
                None,
                contract.execution_pass,
                contract.framework,
            )
        emitted_key = spec.get("emitted_key", contract.emitted_key_pattern)
        requested_framework = contract.framework
    else:
        try:
            contract = registry.get(metric_name)
        except UnknownMetricContractError:
            contract = None
        if contract is not None:
            if "*" in contract.emitted_key_pattern:
                return (
                    None,
                    "privacy gate thresholds must name an exact emitted key; wildcard contract "
                    f"{metric_name!r} is not directly gateable",
                    None,
                    contract.execution_pass,
                    contract.framework,
                )
            emitted_key = spec.get("emitted_key", contract.emitted_key_pattern)
            if requested_framework is not None and requested_framework != contract.framework:
                return (
                    contract,
                    f"threshold framework does not match contract ({requested_framework!r} != "
                    f"{contract.framework!r})",
                    None,
                    contract.execution_pass,
                    contract.framework,
                )
            requested_framework = contract.framework

    if contract is None:
        matching_columns = _combined_metric_columns(
            combined.columns,
            emitted_key,
            framework=requested_framework,
        )
        if len(matching_columns) > 1:
            return (
                None,
                f"metric identity {emitted_key!r} matches multiple privacy columns",
                None,
                str(spec.get("execution_pass", "main")),
                requested_framework,
            )
        if len(matching_columns) == 1:
            requested_framework = matching_columns[0][0]
        if requested_framework is None:
            matching_contracts = [
                candidate for candidate in registry if candidate.emitted_key_pattern == emitted_key
            ]
            if len(matching_contracts) == 1:
                requested_framework = matching_contracts[0].framework
            elif len(matching_contracts) > 1:
                return (
                    None,
                    f"metric identity {emitted_key!r} has multiple framework contracts",
                    None,
                    str(spec.get("execution_pass", "main")),
                    None,
                )
        execution_pass = spec.get(
            "execution_pass",
            _execution_pass_for_metric(
                execution_passes,
                requested_framework,
                emitted_key,
                model_name,
                context.execution_pass if context is not None else "main",
            ),
        )
        if not isinstance(execution_pass, str) or not execution_pass:
            return (
                None,
                "threshold execution_pass must be a non-empty string",
                None,
                "main",
                requested_framework,
            )
        if requested_framework is not None:
            try:
                contract = registry.resolve(
                    framework=requested_framework,
                    emitted_key=emitted_key,
                    execution_pass=execution_pass,
                )
            except (UnknownMetricContractError, AmbiguousMetricContractError) as exc:
                return None, str(exc), None, execution_pass, requested_framework

    if contract is None:
        return None, f"no contract resolved for threshold {metric_name!r}", None, "main", None
    execution_pass = spec.get("execution_pass", contract.execution_pass)
    if execution_pass != contract.execution_pass:
        return (
            contract,
            f"threshold execution_pass does not match contract ({execution_pass!r} != "
            f"{contract.execution_pass!r})",
            None,
            str(execution_pass),
            contract.framework,
        )
    if contract.emitted_key_pattern != emitted_key:
        return (
            contract,
            f"threshold emitted_key does not match contract ({emitted_key!r} != "
            f"{contract.emitted_key_pattern!r})",
            None,
            contract.execution_pass,
            contract.framework,
        )
    columns = _combined_metric_columns(combined.columns, emitted_key, contract.framework)
    if len(columns) > 1:
        return (
            contract,
            f"metric identity {emitted_key!r} matches multiple privacy columns",
            None,
            contract.execution_pass,
            contract.framework,
        )
    return (
        contract,
        None,
        columns[0] if columns else None,
        contract.execution_pass,
        contract.framework,
    )


def evaluate_privacy_gate(
    combined: pd.DataFrame,
    privacy_gate_cfg,
    *,
    registry=DEFAULT_METRIC_CONTRACT_REGISTRY,
    validation_results: Mapping[Any, Any] | None = None,
    context: MetricEvaluationContext | None = None,
    contexts: Mapping[tuple[str, str], MetricEvaluationContext] | None = None,
    execution_passes: Mapping[tuple[str, ...], str] | None = None,
) -> pd.DataFrame | None:
    """Check configured raw thresholds as audit-only evidence.

    Results never block release or alter candidate selection; Stage-A
    exact-copy screening remains the sole automated privacy block.
    """
    if not privacy_gate_cfg.enabled:
        logger.info("[privacy_gate] disabled via evaluation.privacy_gate.enabled; skipping")
        return None
    thresholds = privacy_gate_cfg.thresholds
    if not thresholds:
        logger.info("[privacy_gate] no thresholds configured; skipping")
        return None

    pass_mask = pd.Series(True, index=combined.index, dtype=bool)
    indeterminate_mask = pd.Series(False, index=combined.index, dtype=bool)
    violation_lists: dict[Any, list[str]] = {model: [] for model in combined.index}
    indeterminate_lists: dict[Any, list[str]] = {model: [] for model in combined.index}
    execution_passes = execution_passes or {}
    registry_digest = registry.digest()
    checked_any = False

    for metric_name, spec in thresholds.items():
        if not isinstance(metric_name, str) or not metric_name:
            reason = "threshold metric name must be a non-empty string"
            contract = None
            column = None
            execution_pass = "main"
            framework = None
        elif not isinstance(spec, Mapping):
            reason = f"threshold specification for {metric_name!r} is not a mapping"
            contract = None
            column = None
            execution_pass = "main"
            framework = None
        else:
            contract, reason, column, execution_pass, framework = _resolve_gate_contract(
                metric_name,
                spec,
                combined,
                registry,
                context,
                execution_passes,
            )
        if reason is not None:
            logger.warning("[privacy_gate] threshold %r is indeterminate: %s", metric_name, reason)
            for model in combined.index:
                indeterminate_mask.loc[model] = True
                indeterminate_lists[model].append(f"{metric_name}: {reason}")
            continue
        if contract is None:
            reason = f"no metric contract resolved for threshold {metric_name!r}"
            for model in combined.index:
                indeterminate_mask.loc[model] = True
                indeterminate_lists[model].append(f"{metric_name}: {reason}")
            continue
        checked_any = True
        if column is None:
            reason = f"emitted metric {contract.emitted_key_pattern!r} is unavailable"
            logger.warning("[privacy_gate] threshold %r is indeterminate: %s", metric_name, reason)
            for model in combined.index:
                indeterminate_mask.loc[model] = True
                indeterminate_lists[model].append(f"{metric_name}: {reason}")
            continue

        bound = spec.get("bound")
        limit = _as_finite_number(spec.get("value"))
        if bound not in {"max", "min"} or limit is None:
            reason = f"threshold {metric_name!r} has an invalid bound/value"
            for model in combined.index:
                indeterminate_mask.loc[model] = True
                indeterminate_lists[model].append(reason)
            continue
        if contract.semantic_family != "privacy":
            reason = f"contract semantic family is {contract.semantic_family!r}, not 'privacy'"
            for model in combined.index:
                indeterminate_mask.loc[model] = True
                indeterminate_lists[model].append(f"{metric_name}: {reason}")
            continue
        if contract.lifecycle_state != "operational":
            reason = f"contract lifecycle is {contract.lifecycle_state!r}"
            logger.warning("[privacy_gate] threshold %r is indeterminate: %s", metric_name, reason)
            for model in combined.index:
                indeterminate_mask.loc[model] = True
                indeterminate_lists[model].append(f"{metric_name}: {reason}")
            continue
        if "gate" not in contract.allowed_uses:
            reason = "contract does not allow gate use"
            for model in combined.index:
                indeterminate_mask.loc[model] = True
                indeterminate_lists[model].append(f"{metric_name}: {reason}")
            continue
        if contract.value_role != "policy_scalar":
            reason = f"contract value role is {contract.value_role!r}, not 'policy_scalar'"
            for model in combined.index:
                indeterminate_mask.loc[model] = True
                indeterminate_lists[model].append(f"{metric_name}: {reason}")
            continue
        expected_bound = "max" if contract.direction == "minimize" else "min"
        if bound != expected_bound:
            reason = (
                f"threshold bound {bound!r} disagrees with contract direction "
                f"{contract.direction!r}"
            )
            for model in combined.index:
                indeterminate_mask.loc[model] = True
                indeterminate_lists[model].append(f"{metric_name}: {reason}")
            continue

        values = combined[column]
        for model in combined.index:
            selected_contract = contract
            selected_execution_pass = execution_pass
            selected_framework = framework
            if spec.get("contract_id") is None and spec.get("execution_pass") is None:
                (
                    selected_contract,
                    model_reason,
                    _selected_column,
                    selected_execution_pass,
                    selected_framework,
                ) = _resolve_gate_contract(
                    metric_name,
                    spec,
                    combined,
                    registry,
                    context,
                    execution_passes,
                    model_name=model,
                )
                if model_reason is not None:
                    pass_mask.loc[model] = False
                    indeterminate_mask.loc[model] = True
                    indeterminate_lists[model].append(f"{metric_name}: {model_reason}")
                    continue
            if selected_contract is None:
                pass_mask.loc[model] = False
                indeterminate_mask.loc[model] = True
                indeterminate_lists[model].append(
                    f"{metric_name}: no metric contract resolved for this model"
                )
                continue
            validation = _validation_for_model(
                validation_results,
                model,
                selected_framework or selected_contract.framework,
                selected_execution_pass,
            )
            record = _record_for_key(validation, selected_contract.emitted_key_pattern)
            selected_context = _context_for_validation(
                validation,
                context,
                contexts,
                selected_framework or selected_contract.framework,
                selected_execution_pass,
            )
            evidence_reason = (
                _context_mismatch_reason(
                    record=record,
                    contract=selected_contract,
                    context=selected_context,
                    registry_digest=registry_digest,
                    validation=validation,
                    spec=spec,
                )
                if record is not None
                else "expected metric validation record is unavailable"
            )
            if evidence_reason is not None:
                pass_mask.loc[model] = False
                indeterminate_mask.loc[model] = True
                indeterminate_lists[model].append(f"{metric_name}: {evidence_reason}")
                continue

            value = _as_finite_number(values.loc[model])
            record_value = _as_finite_number(record.raw_value)
            if value is None or record_value is None:
                pass_mask.loc[model] = False
                indeterminate_mask.loc[model] = True
                indeterminate_lists[model].append(
                    f"{metric_name}: NaN or non-finite raw evidence (could not evaluate)"
                )
                continue
            if not math.isclose(value, record_value, rel_tol=0.0, abs_tol=1e-12):
                pass_mask.loc[model] = False
                indeterminate_mask.loc[model] = True
                indeterminate_lists[model].append(
                    f"{metric_name}: combined raw value differs from validated evidence"
                )
                continue
            metric_pass = value <= limit if bound == "max" else value >= limit
            if not metric_pass:
                pass_mask.loc[model] = False
                violation_lists[model].append(f"{metric_name}={value:.4g} ({bound} limit {limit})")

    if not checked_any:
        logger.warning(
            "[privacy_gate] no configured threshold reached a valid contract check; "
            "marking every model indeterminate"
        )
    pass_mask &= ~indeterminate_mask
    status = pd.Series("eligible", index=combined.index, dtype="object")
    status.loc[~pass_mask] = "failed"
    status.loc[indeterminate_mask] = "indeterminate"
    for model in combined.index:
        violation_lists[model].extend(indeterminate_lists[model])

    result = pd.DataFrame(
        {
            _PASS_COL: pass_mask,
            _STATUS_COL: status,
            _VIOLATIONS_COL: pd.Series(
                {model: "; ".join(values) for model, values in violation_lists.items()},
                index=combined.index,
            ),
        },
        index=combined.index,
    )
    result.attrs["evidence_status"] = {
        "thresholds_checked": checked_any,
        "models": {
            str(model): {
                "status": result.loc[model, _STATUS_COL],
                "violations": result.loc[model, _VIOLATIONS_COL],
            }
            for model in result.index
        },
    }
    result.attrs["provenance"] = {
        "status": "audit_only",
        "automated_gate": False,
        "selection_effect": "none",
        "stage_a_exact_copy_screen": "sole_automated_privacy_block",
    }
    for model in result.index[~result[_PASS_COL]]:
        logger.warning(
            "[privacy_gate] model %r FAILED the privacy gate: %s",
            model,
            result.loc[model, _VIOLATIONS_COL],
        )
    return result


def merge_privacy_gate_results(
    combined: pd.DataFrame, gate_result: pd.DataFrame | None
) -> pd.DataFrame:
    """Merge ``evaluate_privacy_gate``'s output into ``combined`` as
    ``("__all__", "privacy_gate", "pass")`` / ``("__all__", "privacy_gate",
    "violations")`` columns. A no-op (returns ``combined`` unchanged) if the
    gate was disabled/skipped (``gate_result is None``).
    """
    if gate_result is None:
        return combined
    combined = combined.copy()
    combined[("__all__", "privacy_gate", _PASS_COL)] = gate_result[_PASS_COL]
    combined[("__all__", "privacy_gate", _STATUS_COL)] = gate_result[_STATUS_COL]
    combined[("__all__", "privacy_gate", _VIOLATIONS_COL)] = gate_result[_VIOLATIONS_COL]
    combined.attrs["privacy_gate_provenance"] = {
        "status": "audit_only",
        "automated_gate": False,
        "selection_effect": "none",
        "stage_a_exact_copy_screen": "sole_automated_privacy_block",
    }
    return combined
