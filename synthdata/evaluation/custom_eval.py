"""Custom fairness evaluation: log disparity (Bhanot et al. 2021) summary metrics.

This module owns log disparity, which has no SynthEval equivalent. Historical
release-evidence compatibility remains isolated in its deprecated shim.
"""

import hashlib
import json
from collections.abc import Mapping

import pandas as pd

from synthdata.data import Dataset
from synthdata.evaluation.catalog import LOG_DISPARITY_METRICS, resolve_selection
from synthdata.evaluation.metric_contracts import (
    DEFAULT_METRIC_CONTRACT_REGISTRY,
    MetricEvaluationContext,
    MetricObservation,
    MetricValidationResult,
    resolve_metric_observations,
)
from synthdata.utils import get_logger

logger = get_logger(__name__)

_LOG_DISPARITY_NAME = "log_disparity"
_LOG_REPORT_TABLES = (
    "leaf_results",
    "hierarchy_results",
    "subgroup_table",
    "leaf_equity_table",
    "legend_table",
    "label_counts",
)
_LOG_DISPARITY_STATES = frozenset({"succeeded", "failed", "indeterminate"})
_UNKNOWN_STATE_REASON = "report_state_missing_or_unknown"


def _indeterminate_report(reason: str, **metadata: object) -> dict:
    """Build explicit metadata-only evidence for an incomplete report."""
    result_metadata = {
        **metadata,
        "release_evidence_state": "indeterminate",
        "release_evidence_reason": reason,
    }
    return {
        "state": "indeterminate",
        "reason": reason,
        "missing_tables": metadata.get("missing_tables", list(_LOG_REPORT_TABLES)),
        "result_metadata": result_metadata,
    }


def _report_missing_tables(report: Mapping[str, object]) -> list[str]:
    """Return required report tables absent from an evaluator report."""
    return [table for table in _LOG_REPORT_TABLES if table not in report]


def _normalized_report_state(report: Mapping[str, object] | None) -> str:
    """Return canonical state, failing closed for absent or unknown values."""
    state = report.get("state") if isinstance(report, Mapping) else None
    return state if isinstance(state, str) and state in _LOG_DISPARITY_STATES else "indeterminate"


def _failed_report(exc: Exception) -> dict:
    """Build failure evidence without persisting exception details or raw data."""
    error_type = type(exc).__name__
    return {
        "state": "failed",
        "reason": "evaluator_exception",
        "error": "log-disparity evaluator raised an unexpected exception",
        "error_type": error_type,
        "missing_tables": list(_LOG_REPORT_TABLES),
        "result_metadata": {
            "release_evidence_state": "failed",
            "release_evidence_reason": "evaluator_exception",
            "error_type": error_type,
        },
    }


def _release_provenance(
    frame: pd.DataFrame,
    *,
    evaluation_role: str | None = None,
    synthetic: bool = False,
) -> dict | None:
    """Read caller-provided, verifiable release provenance from frame attrs."""
    value = frame.attrs.get("release_provenance")
    if not isinstance(value, Mapping):
        return None
    required = {
        "release_form",
        "digest",
        "role",
        "source_role",
        "protocol_version",
        "role_hash",
        "common_protocol_digest",
        "evaluation_role",
    }
    if not required.issubset(value) or value["release_form"] is not True:
        return None
    if not isinstance(value["digest"], str) or not value["digest"]:
        return None
    source_role = value["source_role"]
    if source_role not in {"tuning", "final_holdout", "synthetic"}:
        return None
    expected_source_roles = {"synthetic"} if synthetic else {evaluation_role}
    if source_role not in expected_source_roles:
        return None
    expected_public_roles = (
        {"release"}
        if synthetic
        else ({"tuning"} if evaluation_role == "tuning" else {"final", "final_holdout"})
    )
    if value["role"] not in expected_public_roles:
        return None
    if value["evaluation_role"] != evaluation_role:
        return None
    role_payload = {"role": source_role, "rows": frame.to_dict("records")}
    role_hash = hashlib.sha256(
        json.dumps(role_payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()
    if value["role_hash"] != role_hash:
        return None
    population_payload = {
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "rows": frame.to_dict("records"),
    }
    population_digest = hashlib.sha256(
        json.dumps(population_payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()
    metadata = {
        "protocol_version": value["protocol_version"],
        "role": source_role,
        "row_count": len(frame),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "generalization": value.get("generalization", {}),
    }
    digest_payload = {**metadata, "role_hash": role_hash}
    digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()
    if value["digest"] != digest:
        return None
    if not isinstance(value["common_protocol_digest"], str) or not value["common_protocol_digest"]:
        return None
    if value["role"] not in {"release", "tuning", "final", "final_holdout"}:
        return None
    result = dict(value)
    result["population_digest"] = population_digest
    return result


def _log_disparity_result_metadata(
    report: Mapping[str, object] | None,
    *,
    role_hashes: Mapping[str, str],
    evaluation_role: str,
    state: str,
) -> dict:
    supplied = report.get("result_metadata") if report is not None else None
    if supplied is None and report is not None:
        supplied = report.get("metadata")
    if supplied is not None and not isinstance(supplied, Mapping):
        raise ValueError("Custom log-disparity result_metadata must be an object")
    report_data: Mapping[str, object] = report or {}
    supplied_metadata = report_data.get("result_metadata")
    declared = (
        supplied_metadata.get("declared_protected_columns", [])
        if isinstance(supplied_metadata, Mapping)
        else []
    )
    target_order = report_data.get("target_order", [])
    metadata: dict = {
        "schema_version": "log-disparity-result-v1",
        "metric": _LOG_DISPARITY_NAME,
        "evaluation_role": evaluation_role,
        "protected_columns": list(declared),
        "target_order": list(target_order) if isinstance(target_order, list) else [],
        "population_role_hashes": dict(role_hashes),
        "report_state": state,
        "evidence_definition": "valid-applicable-BH-material-log-disparity",
        "release_form": supplied.get("synthetic_input_form", "unverified")
        if isinstance(supplied, Mapping)
        else "unverified",
    }
    if supplied is not None:
        metadata.update(dict(supplied))
    return metadata


def run_log_disparity_evaluation(
    synthetic_datasets: dict[str, pd.DataFrame],
    dataset: Dataset,
    log_disparity_cfg,
    selection_cfg,
    *,
    evaluation_role: str = "tuning",
    reference_frame: pd.DataFrame | None = None,
) -> dict[str, dict]:
    """Compute a log-disparity fairness report for every synthetic dataset.

    Returns ``{model_name: report}`` (the full dict from
    ``compute_log_disparity_report``, including the Plotly ``report_figure``),
    or ``{}`` if log_disparity is not in the configured selection.
    """
    all_names = [_LOG_DISPARITY_NAME]
    selected = resolve_selection(
        selection_cfg.enabled,
        selection_cfg.categories,
        selection_cfg.metrics,
        all_names,
        {_LOG_DISPARITY_NAME: "fairness"},
    )
    if _LOG_DISPARITY_NAME not in selected:
        return {}

    from synthdata.log_disparity.metric_log_disparity import compute_log_disparity_report

    # Representation evidence is restricted to the explicit protected-field
    # declaration.  Sensitive fields are a separate attack/evaluation role.
    declared_protected = list(dataset.protected_columns)
    configured_protected = list(log_disparity_cfg.protected_columns or [])
    protected_cols = configured_protected or declared_protected
    if not protected_cols:
        logger.warning("[custom] log_disparity requires protected columns; skipping")
        return {
            name: {
                **_indeterminate_report("missing_declared_protected_fields"),
            }
            for name in synthetic_datasets
        }
    # Legacy Dataset fixtures predate explicit protected_columns; configured
    # fields remain accepted there, while canonical datasets are strict.
    if not set(protected_cols).issubset(declared_protected):
        return {
            name: {
                **_indeterminate_report(
                    "undeclared_protected_fields",
                    declared_protected_columns=declared_protected,
                    configured_protected_columns=protected_cols,
                ),
            }
            for name in synthetic_datasets
        }
    if evaluation_role == "final_holdout":
        if reference_frame is None:
            raise ValueError(
                "Final-holdout log-disparity requires an explicit released reference frame"
            )
        if not isinstance(reference_frame, pd.DataFrame):
            raise TypeError("Explicit log-disparity reference frame must be a pandas DataFrame")
        expected_columns = list(dataset.full_df.columns)
        if list(reference_frame.columns) != expected_columns:
            raise ValueError(
                "Explicit log-disparity reference frame columns do not match the dataset schema"
            )
        raw_role = dataset.role_frame(evaluation_role, imputed=False)
        if raw_role is None:
            raise ValueError("Final-holdout log-disparity reference role is missing")
        if len(reference_frame) != len(raw_role):
            raise ValueError(
                "Explicit log-disparity reference frame row count does not match final_holdout"
            )
        real_data = reference_frame
    else:
        real_data = dataset.role_frame(evaluation_role, imputed=False)
    if real_data is None:
        return {
            name: {
                **_indeterminate_report(
                    "missing_requested_evaluation_role",
                    declared_protected_columns=protected_cols,
                    real_evidence_role=evaluation_role,
                    synthetic_input_form="unverified",
                    test_family="representation::target_by_protected_leaf",
                ),
            }
            for name in synthetic_datasets
        }

    # This metric is release evidence, not a raw legacy diagnostic.  Without
    # verified provenance on both evidence populations it cannot make a
    # release claim, regardless of Dataset compatibility mode.
    canonical = True

    reports: dict[str, dict] = {}
    real_provenance = _release_provenance(real_data, evaluation_role=evaluation_role)
    for name, syn_df in synthetic_datasets.items():
        try:
            provenance = _release_provenance(
                syn_df, evaluation_role=evaluation_role, synthetic=True
            )
            if provenance is not None:
                expected_roles = {"release"} if evaluation_role == "tuning" else {"release"}
                if provenance["role"] not in expected_roles or (
                    provenance.get("evaluation_role") is not None
                    and provenance["evaluation_role"] != evaluation_role
                ):
                    provenance = None
                if canonical and (
                    real_provenance is None
                    or provenance is None
                    or (
                        provenance["common_protocol_digest"]
                        != real_provenance["common_protocol_digest"]
                    )
                ):
                    provenance = None
            if canonical and (provenance is None or real_provenance is None):
                reports[name] = {
                    **_indeterminate_report(
                        "missing_or_invalid_release_provenance"
                        if provenance is None
                        else "missing_real_evidence_provenance",
                        declared_protected_columns=protected_cols,
                        real_evidence_role=evaluation_role,
                        synthetic_input_form="unverified",
                        test_family="representation::target_by_protected_leaf",
                        representation_safety=float("nan"),
                        worst_abs_log_disparity=float("nan"),
                    ),
                }
                continue
            synthetic_digest = provenance["population_digest"]
            real_digest = real_provenance["population_digest"]
            if synthetic_digest == real_digest:
                reports[name] = {
                    **_indeterminate_report("synthetic_real_population_alias"),
                }
                continue
            reports[name] = compute_log_disparity_report(
                real_data=real_data,
                synth_data=syn_df,
                target_col=dataset.target_column,
                protected_cols=protected_cols,
                model_name=name,
                target_map=log_disparity_cfg.target_map,
                protected_map=log_disparity_cfg.protected_map,
                protected_bins=log_disparity_cfg.protected_bins,
            )
            reports[name]["state"] = "succeeded"
            reports[name]["result_metadata"] = {
                "declared_protected_columns": protected_cols,
                "real_evidence_role": evaluation_role,
                "synthetic_input_form": "release" if provenance else "legacy_unverified",
                "release_evidence_state": "verified" if provenance else "legacy_unverified",
                "release_provenance": provenance,
                "test_family": "representation::target_by_protected_leaf",
                "representation_safety": reports[name]["summary_stats"].get(
                    "representation_safety"
                ),
                "worst_abs_log_disparity": reports[name]["summary_stats"].get(
                    "worst_abs_log_disparity"
                ),
                "synthetic_population_digest": synthetic_digest,
                "real_population_digest": real_digest,
                "synthetic_role_hash": provenance["role_hash"],
                "real_role_hash": real_provenance["role_hash"],
            }
        except Exception as exc:  # noqa: BLE001 - evaluator failures are per-model evidence
            logger.warning("[custom] log_disparity failed for %s (%s)", name, type(exc).__name__)
            reports[name] = _failed_report(exc)
    return reports


def build_log_disparity_summary_table(reports: dict[str, dict]) -> pd.DataFrame:
    """Models x {log_disparity_mean_abs, log_disparity_median_abs, log_disparity_share_significant}.

    Models whose report failed (see ``run_log_disparity_evaluation``'s
    ``{"error": ...}`` entries) get all-NaN rows here rather than being
    silently dropped, so a failure is still visible in the summary table.
    """
    rows = {}
    for name, report in reports.items():
        if _normalized_report_state(report) != "succeeded":
            rows[name] = {
                "log_disparity_mean_abs": None,
                "log_disparity_median_abs": None,
                "log_disparity_share_significant": None,
                "log_disparity_representation_safety": None,
                "log_disparity_worst_abs": None,
            }
            continue
        stats = report.get("summary_stats", {})
        rows[name] = {
            "log_disparity_mean_abs": stats.get("mean_abs_log_disparity"),
            "log_disparity_median_abs": stats.get("median_abs_log_disparity"),
            "log_disparity_share_significant": stats.get("share_significant_bh"),
            "log_disparity_representation_safety": stats.get(
                "representation_safety", stats.get("share_significant_bh")
            ),
            "log_disparity_worst_abs": stats.get(
                "worst_abs_log_disparity", stats.get("mean_abs_log_disparity")
            ),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def validate_log_disparity_results(
    reports: dict[str, dict],
    model_names: list[str],
    *,
    role_hashes: dict[str, str],
    requested_use: str = "policy_rank",
    population_unit: str = "row",
    group_mode: str = "row",
    resolved_configuration: dict | None = None,
    evaluation_role: str = "tuning",
    expected_role_hashes: Mapping[str, str] | None = None,
) -> dict[str, MetricValidationResult]:
    """Validate log-disparity summaries without dropping failed reports."""
    if expected_role_hashes is not None and dict(role_hashes) != dict(expected_role_hashes):
        raise ValueError("Custom log-disparity role hashes do not match the declared raw role map")
    expected_keys = [
        "log_disparity_mean_abs",
        "log_disparity_median_abs",
        "log_disparity_share_significant",
    ]
    validations = {}
    context = MetricEvaluationContext(
        role_hashes=role_hashes,
        evaluation_role=evaluation_role,
        population_unit=population_unit,
        group_mode=group_mode,
        resolved_configuration=resolved_configuration or {},
    )
    for model_name in model_names:
        report = reports.get(model_name)
        observations = []
        state = _normalized_report_state(report)
        if state == "indeterminate":
            known_state = (
                isinstance(report, Mapping) and report.get("state") in _LOG_DISPARITY_STATES
            )
            reason = report.get("reason") if known_state else _UNKNOWN_STATE_REASON
            malformed = dict(report) if isinstance(report, Mapping) else {}
            malformed.setdefault("reason", reason)
            result_metadata = _log_disparity_result_metadata(
                malformed,
                role_hashes=role_hashes,
                evaluation_role=evaluation_role,
                state="indeterminate",
            )
            observations = []
        elif state == "failed":
            failed_report = report if isinstance(report, Mapping) else {}
            result_metadata = _log_disparity_result_metadata(
                failed_report,
                role_hashes=role_hashes,
                evaluation_role=evaluation_role,
                state="failed",
            )
            observations = [
                MetricObservation(
                    model_name=model_name,
                    framework="custom",
                    emitted_key=key,
                    raw_value=None,
                    error=(
                        f"{failed_report.get('error_type', 'UnknownError')}: "
                        f"{failed_report.get('reason', 'log-disparity failed')}"
                    ),
                    role_hashes=role_hashes,
                    source_metadata={
                        "report_state": "failed",
                        "result_metadata": result_metadata,
                    },
                    result_metadata=result_metadata,
                )
                for key in expected_keys
            ]
        else:
            missing_tables = (
                _report_missing_tables(report)
                if isinstance(report, Mapping)
                else list(_LOG_REPORT_TABLES)
            )
            summary_stats = report.get("summary_stats") if isinstance(report, Mapping) else None
            if missing_tables or not isinstance(summary_stats, Mapping):
                reason = (
                    "malformed_report" if not isinstance(report, Mapping) else "incomplete_report"
                )
                malformed = _indeterminate_report(
                    reason,
                    missing_tables=missing_tables,
                    report_state=report.get("state") if isinstance(report, Mapping) else None,
                )
                result_metadata = _log_disparity_result_metadata(
                    malformed,
                    role_hashes=role_hashes,
                    evaluation_role=evaluation_role,
                    state="indeterminate",
                )
                observations = []
            else:
                result_metadata = _log_disparity_result_metadata(
                    report,
                    role_hashes=role_hashes,
                    evaluation_role=evaluation_role,
                    state="succeeded",
                )
                values = {
                    "log_disparity_mean_abs": summary_stats.get("mean_abs_log_disparity"),
                    "log_disparity_median_abs": summary_stats.get("median_abs_log_disparity"),
                    "log_disparity_share_significant": summary_stats.get("share_significant_bh"),
                }
                observations = [
                    MetricObservation(
                        model_name=model_name,
                        framework="custom",
                        emitted_key=key,
                        raw_value=value,
                        role_hashes=role_hashes,
                        source_metadata={
                            "report_state": "succeeded",
                            "result_metadata": result_metadata,
                        },
                        result_metadata=result_metadata,
                    )
                    for key, value in values.items()
                ]
        validations[model_name] = resolve_metric_observations(
            registry=DEFAULT_METRIC_CONTRACT_REGISTRY,
            model_name=model_name,
            framework="custom",
            expected_keys=expected_keys,
            observations=observations,
            context=context,
            requested_use=requested_use,
        )
    return validations


#: True => lower is "better" (orient as -value for ranking); mirrors LOG_DISPARITY_METRICS.
LOG_DISPARITY_MINIMIZE = dict(LOG_DISPARITY_METRICS)
