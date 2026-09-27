"""Combines evaluation evidence into one 3-level MultiIndex audit table.

The compatibility ``rank`` columns are absolute fixed-transform evidence;
candidate-relative min-max values are not used for policy selection.
``(framework, type, metric)`` where ``framework in {synthcity, syntheval, custom}``
and metric types include ``utility``, ``privacy``, ``fairness``, and explicit
``audit`` evidence columns for model-level failures.

Ranking (see module docstring of :func:`build_combined_table` for details) is
appended as extra columns in the same table, both per ``(framework, type)`` group
and rolled up across frameworks per ``type``, plus one overall rank.
"""

import math
import re
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any

import pandas as pd

from synthdata.evaluation.catalog import (
    LOG_DISPARITY_METRICS,
    SYNTHCITY_CATEGORY_TO_TYPE,
    classify_syntheval_metric,
    is_custom_syntheval_metric,
    is_redundant_synthcity_submetric,
)
from synthdata.evaluation.custom_eval import build_log_disparity_summary_table
from synthdata.evaluation.metric_contracts import (
    DEFAULT_METRIC_CONTRACT_REGISTRY,
    MetricValidationResult,
    UnknownMetricContractError,
)
from synthdata.evaluation.release_score import normalize_component
from synthdata.evaluation.syntheval_eval import extract_oriented_values, extract_raw_values
from synthdata.utils import get_logger

logger = get_logger(__name__)

_ALL = "__all__"
_RANK = "rank"
_SAFE_MODEL_ERROR_REASONS = {
    "metric_evaluation_failed",
    "synthcity_report_empty",
    "SynthCity emitted an empty failure report",
    "SynthCity metric evaluation failed.",
}
_SAFE_EXCEPTION_TYPE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _safe_synthcity_failure(error: object, error_type: object) -> tuple[str, str]:
    """Return audit-safe SynthCity failure reason and exception type."""
    reason = str(error) if isinstance(error, str) else ""
    if reason not in _SAFE_MODEL_ERROR_REASONS:
        reason = "metric_evaluation_failed"
    exception_type = str(error_type) if isinstance(error_type, str) else ""
    if not _SAFE_EXCEPTION_TYPE_PATTERN.fullmatch(exception_type):
        exception_type = "UnknownError"
    return reason, exception_type


def _is_missing_value(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _finite_real(value: object) -> float | None:
    """Return finite real values without coercing malformed evidence."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    numeric_value = float(value)
    return numeric_value if math.isfinite(numeric_value) else None


def validate_combined_table(combined: pd.DataFrame) -> pd.DataFrame:
    """Validate the persisted combined-table schema and numeric evidence.

    A blocked legacy evaluation is intentionally persisted as a zero-row,
    zero-column table because no candidate metrics were evaluated. That exact
    shape remains readable for historical outputs; every non-empty table must
    carry the normal overall-rank column.
    """
    if not isinstance(combined, pd.DataFrame):
        raise ValueError("Combined evaluation table must be a pandas DataFrame")
    if combined.shape == (0, 0):
        return combined
    if not isinstance(combined.columns, pd.MultiIndex) or combined.columns.nlevels != 3:
        raise ValueError("Combined evaluation table must have three-level metric columns")
    if combined.columns.has_duplicates:
        raise ValueError("Combined evaluation table contains duplicate metric columns")
    if combined.index.has_duplicates:
        raise ValueError("Combined evaluation table contains duplicate model names")
    if any(not isinstance(model, str) or not model.strip() for model in combined.index):
        raise ValueError("Combined evaluation table model index must contain non-empty names")

    required_rank = (_ALL, "overall", _RANK)
    if required_rank not in combined.columns:
        raise ValueError("Combined evaluation table is missing the overall rank column")

    frameworks = {"synthcity", "syntheval", "custom", _ALL}
    types = {"utility", "privacy", "fairness", "audit", "overall", "privacy_gate"}
    numeric_columns = []
    for framework, type_, metric in combined.columns:
        if not all(isinstance(value, str) and value for value in (framework, type_, metric)):
            raise ValueError(
                "Combined evaluation table column identities must be non-empty strings"
            )
        if framework not in frameworks or type_ not in types:
            raise ValueError(
                f"Unknown combined evaluation column identity: {(framework, type_, metric)!r}"
            )
        if (
            metric == _RANK
            or (
                framework in {"synthcity", "syntheval", "custom"}
                and type_ in {"utility", "privacy", "fairness"}
            )
            or (framework == _ALL and type_ in {"utility", "privacy", "fairness", "overall"})
        ):
            numeric_columns.append((framework, type_, metric))

    for column in numeric_columns:
        for value in combined[column]:
            if _is_missing_value(value):
                continue
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"Combined metric column {column!r} contains a non-numeric value")
            if not math.isfinite(float(value)):
                raise ValueError(f"Combined metric column {column!r} contains a non-finite value")

    gate_pass = (_ALL, "privacy_gate", "pass")
    if gate_pass in combined.columns:
        for value in combined[gate_pass]:
            if _is_missing_value(value):
                continue
            if not isinstance(value, bool):
                raise ValueError("Combined privacy-gate pass column must contain boolean values")
    return combined


def _materialize_validation_records(
    raw: pd.DataFrame,
    model_names: list,
    records_by_model: Mapping[str, tuple[Any, ...] | list[Any]],
) -> pd.DataFrame:
    """Add every validated identity to the raw table, including missing rows.

    Native framework tables are observed-output views and therefore omit an
    expected identity when an evaluator failed before emitting that row. The
    validation records are the authoritative identity set; their raw values
    fill observed-output gaps without replacing values already present in the
    native table.
    """
    raw = raw.reindex(model_names).copy()
    record_keys = []
    for records in records_by_model.values():
        for record in records:
            key = str(record.expected_key)
            if key not in record_keys:
                record_keys.append(key)

    missing_keys = [key for key in record_keys if key not in raw.columns]
    if missing_keys:
        missing = pd.DataFrame(pd.NA, index=raw.index, columns=missing_keys, dtype="object")
        raw = pd.concat([raw, missing], axis=1)

    for model_name, records in records_by_model.items():
        if model_name not in raw.index:
            continue
        for record in records:
            key = str(record.expected_key)
            value = record.raw_value
            if _is_missing_value(value) or not _is_missing_value(raw.at[model_name, key]):
                continue
            raw.at[model_name, key] = value
    return raw


def _append_validation_status_columns(
    raw: pd.DataFrame,
    model_names: list,
    validations: Mapping[str, MetricValidationResult],
    *,
    prefix: str,
) -> pd.DataFrame:
    """Persist model-level audit and decision state beside raw metric values."""
    if not validations:
        return raw

    status = pd.DataFrame(index=pd.Index(model_names))
    status_values = {
        f"{prefix}_audit_status": {},
        f"{prefix}_succeeded": {},
        f"{prefix}_decision_eligible": {},
        f"{prefix}_decision_status": {},
        f"{prefix}_expected_count": {},
        f"{prefix}_completed_count": {},
        f"{prefix}_indeterminate_count": {},
    }
    for model_name in model_names:
        validation = validations.get(model_name)
        if validation is None:
            values = {
                f"{prefix}_audit_status": "unvalidated",
                f"{prefix}_succeeded": False,
                f"{prefix}_decision_eligible": False,
                f"{prefix}_decision_status": "unvalidated",
                f"{prefix}_expected_count": 0,
                f"{prefix}_completed_count": 0,
                f"{prefix}_indeterminate_count": 0,
            }
        else:
            succeeded = bool(getattr(validation, "succeeded", validation.complete))
            decision_eligible = bool(validation.decision_eligible)
            values = {
                f"{prefix}_audit_status": "succeeded" if succeeded else "indeterminate",
                f"{prefix}_succeeded": succeeded,
                f"{prefix}_decision_eligible": decision_eligible,
                f"{prefix}_decision_status": validation.decision_status,
                f"{prefix}_expected_count": len(validation.expected_records),
                f"{prefix}_completed_count": len(validation.completed_keys),
                f"{prefix}_indeterminate_count": len(validation.indeterminate_keys),
            }
        for key, value in values.items():
            status_values[key][model_name] = value

    for key, values in status_values.items():
        status[key] = pd.Series(values, index=model_names)
    return pd.concat([raw, status], axis=1)


def _synthcity_frames(
    synthcity_results: dict[str, pd.DataFrame],
    model_names: list,
    synthcity_validations: Mapping[str, MetricValidationResult] | None = None,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Build (raw, oriented) models x metric-key tables from synthcity results.

    Entries with an "error" column (a model whose synthcity evaluation failed,
    see ``synthcity_eval.run_synthcity_evaluation``) contribute explicit audit
    evidence columns. They never contribute oriented ranking values.
    """
    if not synthcity_results and not synthcity_validations:
        empty = pd.DataFrame(index=pd.Index(model_names))
        return empty, empty

    failed = {name for name, res in synthcity_results.items() if "error" in res.columns}
    if failed:
        logger.warning("[synthcity] retaining failed models as raw audit rows: %s", sorted(failed))
    ok_results = {name: res for name, res in synthcity_results.items() if name not in failed}
    raw = (
        pd.DataFrame({name: res["mean"] for name, res in ok_results.items()}).T
        if ok_results
        else pd.DataFrame(index=pd.Index(model_names))
    )

    redundant_cols = [c for c in raw.columns if is_redundant_synthcity_submetric(c)]
    if redundant_cols:
        logger.info(
            "[synthcity] excluding known-redundant duplicate sub-metric(s) from combined "
            "table: %s (see catalog.SYNTHCITY_REDUNDANT_SUBMETRIC_SUFFIXES)",
            redundant_cols,
        )
        raw = raw.drop(columns=redundant_cols)

    if synthcity_validations:
        raw = _materialize_validation_records(
            raw,
            model_names,
            {
                model_name: tuple(validation.records)
                for model_name, validation in synthcity_validations.items()
            },
        )

    directions = {}
    for validation in (synthcity_validations or {}).values():
        for record in validation.records:
            if record.direction is not None:
                directions.setdefault(record.expected_key, record.direction)
    for res in ok_results.values():
        for metric_key, direction in res["direction"].items():
            directions.setdefault(metric_key, direction)
    sign = pd.Series(directions).map({"maximize": 1.0, "minimize": -1.0})

    common = raw.columns.intersection(sign.index)
    oriented = raw[common].multiply(sign[common], axis=1)

    if synthcity_validations:
        for model_name in oriented.index:
            validation = synthcity_validations.get(model_name)
            if validation is None or not getattr(
                validation, "policy_rank_eligible", validation.decision_eligible
            ):
                oriented.loc[model_name, :] = pd.NA
                logger.info(
                    "[synthcity] excluding all metrics from policy ranking for %s: "
                    "validation is incomplete or not decision-eligible",
                    model_name,
                )
                continue
            eligible_keys = set()
            for record in validation.expected_records:
                if record.status != "succeeded" or record.contract_id is None:
                    continue
                contract = DEFAULT_METRIC_CONTRACT_REGISTRY.get(record.contract_id)
                if (
                    contract.lifecycle_state == "operational"
                    and contract.value_role == "policy_scalar"
                    and "policy_rank" in contract.allowed_uses
                    and "candidate_independent" not in contract.qualifiers
                    and "legacy_aggregate" not in contract.qualifiers
                ):
                    eligible_keys.add(record.expected_key)
            excluded = [key for key in common if key not in eligible_keys]
            if excluded:
                oriented.loc[model_name, excluded] = pd.NA
                logger.info(
                    "[synthcity] excluding %d non-operational/diagnostic metric(s) from "
                    "policy ranking for %s",
                    len(excluded),
                    model_name,
                )
        oriented = oriented.dropna(axis=1, how="all")

    if failed:
        failure_frame = pd.DataFrame(
            {
                "__model_error": pd.Series(pd.NA, index=model_names, dtype="object"),
                "__model_error_type": pd.Series(pd.NA, index=model_names, dtype="object"),
            }
        )
        for model_name, result in synthcity_results.items():
            if model_name not in failed:
                continue
            if result.empty:
                error = "SynthCity emitted an empty failure report"
                error_type = "UnknownError"
            else:
                row = result.iloc[0]
                error = row.get("error", "SynthCity evaluation failed")
                error_type = row.get("error_type", "UnknownError")
            safe_error, safe_error_type = _safe_synthcity_failure(error, error_type)
            failure_frame.loc[model_name, "__model_error"] = safe_error
            failure_frame.loc[model_name, "__model_error_type"] = safe_error_type
        raw = pd.concat([raw, failure_frame], axis=1)

    if synthcity_validations:
        raw = _append_validation_status_columns(
            raw,
            model_names,
            synthcity_validations,
            prefix="__model_synthcity",
        )

    raw = raw.reindex(model_names)
    oriented = oriented.reindex(model_names)

    def _type_for_metric(metric_key: str) -> str:
        if metric_key.startswith("__model_"):
            return "audit"
        return SYNTHCITY_CATEGORY_TO_TYPE.get(metric_key.split(".")[0], "utility")

    raw_columns = [("synthcity", _type_for_metric(col), col) for col in raw.columns]
    oriented_columns = [("synthcity", _type_for_metric(col), col) for col in oriented.columns]
    raw.columns = pd.MultiIndex.from_tuples(raw_columns, names=["framework", "type", "metric"])
    oriented.columns = (
        pd.MultiIndex.from_tuples(oriented_columns, names=["framework", "type", "metric"])
        if oriented_columns
        else pd.MultiIndex.from_arrays([[], [], []], names=["framework", "type", "metric"])
    )
    return raw, oriented


def _syntheval_frames(
    benchmark_results: pd.DataFrame | None,
    benchmark_ranks: pd.DataFrame | None,
    model_names: list,
    syntheval_validations: Mapping[tuple[str, str], Mapping[str, MetricValidationResult]]
    | None = None,
    metric_execution_passes: Mapping[tuple[str, str, str], str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (raw, oriented) models x metric-name tables from SynthEval results.

    Metrics matching is_custom_syntheval_metric() (fork-only additions, plus
    their full_output=True per-(target_var, protected_attribute) sub-columns)
    are tagged framework="custom" instead of "syntheval".
    """
    if benchmark_results is None and not syntheval_validations:
        empty = pd.DataFrame(index=pd.Index(model_names))
        return empty, empty

    raw = (
        extract_raw_values(benchmark_results).reindex(model_names)
        if benchmark_results is not None
        else pd.DataFrame(index=pd.Index(model_names))
    )
    oriented = (
        extract_oriented_values(benchmark_ranks).reindex(model_names)
        if benchmark_ranks is not None
        else pd.DataFrame(index=pd.Index(model_names))
    )
    oriented = oriented.reindex(columns=raw.columns.intersection(oriented.columns))

    metric_execution_passes = metric_execution_passes or {}

    records_by_model: dict[str, list] = {model_name: [] for model_name in model_names}
    seen_records: dict[str, set[str]] = {model_name: set() for model_name in model_names}
    validation_items = sorted(
        (syntheval_validations or {}).items(),
        key=lambda item: (item[0][1] != "main", item[0][0], item[0][1]),
    )
    for (framework, execution_pass), model_validations in validation_items:
        for model_name, validation in model_validations.items():
            if model_name not in records_by_model:
                continue
            for record in validation.records:
                emitted_key = str(record.expected_key)
                owner = metric_execution_passes.get((framework, emitted_key, model_name))
                if owner is not None and owner != execution_pass:
                    continue
                if emitted_key in seen_records[model_name]:
                    continue
                records_by_model[model_name].append(record)
                seen_records[model_name].add(emitted_key)
    if any(records_by_model.values()):
        raw = _materialize_validation_records(raw, model_names, records_by_model)

    for (framework, execution_pass), model_validations in validation_items:
        raw = _append_validation_status_columns(
            raw,
            model_names,
            model_validations,
            prefix=f"__model_{framework}_{execution_pass}",
        )

    def _framework(metric: str) -> str:
        return "custom" if is_custom_syntheval_metric(metric) else "syntheval"

    def _type(metric: str) -> str:
        if metric.startswith("__model_"):
            return "audit"
        try:
            return classify_syntheval_metric(metric)
        except UnknownMetricContractError:
            logger.warning(
                "[syntheval] preserving unknown emitted key %s as audit evidence", metric
            )
            return "audit"

    if oriented.columns.size:
        for metric in oriented.columns:
            metric = str(metric)
            framework = _framework(metric)
            for model_name in oriented.index:
                execution_pass = metric_execution_passes.get(
                    (framework, metric, model_name), "main"
                )
                validations = (syntheval_validations or {}).get((framework, execution_pass))
                validation = validations.get(model_name) if validations else None
                if validation is None or not getattr(
                    validation, "policy_rank_eligible", validation.decision_eligible
                ):
                    oriented.loc[model_name, metric] = pd.NA
                    continue
                eligible_keys = {
                    record.expected_key
                    for record in validation.expected_records
                    if record.status == "succeeded"
                    and record.contract_id is not None
                    and record.value_role == "policy_scalar"
                    and record.lifecycle_state == "operational"
                    and "policy_rank" in record.allowed_uses
                }
                if metric not in eligible_keys:
                    oriented.loc[model_name, metric] = pd.NA

    raw.columns = pd.MultiIndex.from_tuples(
        [(_framework(str(col)), _type(str(col)), str(col)) for col in raw.columns],
        names=["framework", "type", "metric"],
    )
    oriented.columns = pd.MultiIndex.from_tuples(
        [(_framework(str(col)), _type(str(col)), str(col)) for col in oriented.columns],
        names=["framework", "type", "metric"],
    )
    return raw, oriented


def _log_disparity_frames(
    reports: dict[str, dict],
    model_names: list,
    custom_validations: Mapping[str, MetricValidationResult] | None = None,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    if not reports and not custom_validations:
        empty = pd.DataFrame(index=pd.Index(model_names))
        return empty, empty

    raw = (
        build_log_disparity_summary_table(reports).reindex(model_names)
        if reports
        else pd.DataFrame(index=pd.Index(model_names))
    )
    # Keep legacy/unvalidated report payloads available as audit evidence. The
    # summary builder intentionally withholds values when report state is not
    # explicitly successful, but raw evidence must not be rewritten or used
    # for policy ranking on that account.
    for model_name, report in reports.items():
        summary_stats = report.get("summary_stats") if isinstance(report, Mapping) else None
        if not isinstance(summary_stats, Mapping) or model_name not in raw.index:
            continue
        raw_values = {
            "log_disparity_mean_abs": summary_stats.get("mean_abs_log_disparity"),
            "log_disparity_median_abs": summary_stats.get("median_abs_log_disparity"),
            "log_disparity_share_significant": summary_stats.get("share_significant_bh"),
            "log_disparity_representation_safety": summary_stats.get(
                "representation_safety", summary_stats.get("share_significant_bh")
            ),
            "log_disparity_worst_abs": summary_stats.get(
                "worst_abs_log_disparity", summary_stats.get("mean_abs_log_disparity")
            ),
        }
        for metric, value in raw_values.items():
            if metric in raw.columns and value is not None:
                raw.at[model_name, metric] = value
    if custom_validations:
        raw = _materialize_validation_records(
            raw,
            model_names,
            {
                model_name: tuple(validation.records)
                for model_name, validation in custom_validations.items()
            },
        )
    sign = pd.Series(
        {m: (-1.0 if minimize else 1.0) for m, minimize in LOG_DISPARITY_METRICS.items()}
    )
    common = raw.columns.intersection(sign.index)
    oriented = raw[common].multiply(sign[common], axis=1)

    if custom_validations is None:
        oriented = oriented.iloc[:, 0:0]
    else:
        for model_name in oriented.index:
            validation = custom_validations.get(model_name)
            if validation is None or not getattr(
                validation, "policy_rank_eligible", validation.decision_eligible
            ):
                oriented.loc[model_name, :] = pd.NA
                continue
            eligible_keys = {
                record.expected_key
                for record in validation.expected_records
                if record.status == "succeeded"
                and record.value_role == "policy_scalar"
                and record.lifecycle_state == "operational"
                and "policy_rank" in record.allowed_uses
            }
            oriented.loc[model_name, [key for key in common if key not in eligible_keys]] = pd.NA

    if custom_validations:
        raw = _append_validation_status_columns(
            raw,
            model_names,
            custom_validations,
            prefix="__model_custom",
        )

    columns = [
        ("custom", "audit" if str(col).startswith("__model_") else "fairness", col)
        for col in raw.columns
    ]
    raw.columns = pd.MultiIndex.from_tuples(columns)
    oriented.columns = pd.MultiIndex.from_tuples(
        [("custom", "fairness", col) for col in oriented.columns],
        names=["framework", "type", "metric"],
    )
    return raw, oriented


def _release_evidence_frames(
    validations: Mapping[str, MetricValidationResult] | None,
    model_names: list,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialize canonical custom evidence; blocked records stay audit-visible."""
    if not validations:
        empty = pd.DataFrame(index=pd.Index(model_names))
        return empty, empty
    keys = sorted({record.expected_key for item in validations.values() for record in item.records})
    raw = pd.DataFrame(index=pd.Index(model_names), columns=pd.Index(keys), dtype=float)
    for model, validation in validations.items():
        for record in validation.records:
            if record.status == "succeeded" and record.raw_value is not None:
                raw.loc[model, record.expected_key] = record.raw_value
    oriented = raw.copy()
    invalid_policy_models: set[str] = set()
    for key in keys:
        identities = {
            (
                record.framework,
                record.expected_key,
                record.execution_pass,
                record.contract_id,
                validation.contract_digest,
            )
            for validation in validations.values()
            for record in validation.records
            if record.expected_key == key and record.is_expected
        }
        contract = None
        if len(identities) == 1:
            framework, emitted_key, execution_pass, contract_id, digest = next(iter(identities))
            try:
                resolved = DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
                    framework=framework,
                    emitted_key=emitted_key,
                    execution_pass=execution_pass,
                )
                if (
                    contract_id == resolved.contract_id
                    and digest == DEFAULT_METRIC_CONTRACT_REGISTRY.digest()
                ):
                    contract = resolved
            except (UnknownMetricContractError, ValueError):
                contract = None
        if (
            key == "equalized_odds.final.v1"
            or contract is None
            or contract.lifecycle_state != "operational"
            or "policy_rank" not in contract.allowed_uses
        ):
            oriented[key] = pd.NA
        elif contract.direction == "minimize":
            oriented[key] = -oriented[key]
        if key in {"tstr_macro_f1.v1", "mixed_mmd.v1", "elastic_net_jsd.v1"}:
            for model, validation in validations.items():
                if any(
                    record.expected_key == key
                    and record.status == "succeeded"
                    and _finite_real(record.policy_value) is None
                    for record in validation.records
                ):
                    invalid_policy_models.add(model)
                    oriented.loc[model, key] = pd.NA
    raw = _append_validation_status_columns(
        raw, model_names, validations, prefix="__model_custom_release_evidence"
    )
    for model in invalid_policy_models:
        raw.loc[model, "__model_custom_release_evidence_succeeded"] = False
        raw.loc[model, "__model_custom_release_evidence_decision_eligible"] = False
        raw.loc[model, "__model_custom_release_evidence_decision_status"] = "indeterminate"
        raw.loc[model, "__model_custom_release_evidence_audit_status"] = "indeterminate"
    raw.columns = pd.MultiIndex.from_tuples(
        [
            (
                "custom",
                "audit"
                if str(key).startswith("__model_")
                else "privacy"
                if key == "release_privacy.v1"
                else "fairness",
                key,
            )
            for key in raw.columns
        ]
    )
    oriented.columns = pd.MultiIndex.from_tuples(
        [
            (
                "custom",
                "privacy" if key == "release_privacy.v1" else "fairness",
                key,
            )
            for key in oriented.columns
        ]
    )
    return raw, oriented


def _minmax_scale(col: pd.Series) -> pd.Series:
    """Scale values for audit/compatibility diagnostic ranks, never policy selection."""
    valid = col.dropna()
    if valid.empty:
        return col
    lo, hi = valid.min(), valid.max()
    if hi == lo:
        return col.where(col.isna(), 0.5)
    return (col - lo) / (hi - lo)


#: Types rolled up in the combined table. Order matters for iteration below
#: but not for correctness (weighted sum is order-independent).
_TYPES = ("utility", "privacy", "fairness")

#: Equal weighting used when the caller doesn't pass ``rank_weights`` (e.g.
#: existing direct callers/tests predating evaluation.rank_weights) --
#: mirrors the pre-existing implicit behavior before per-type weights existed.
DEFAULT_RANK_WEIGHTS = {"utility": 1.0, "privacy": 1.0, "fairness": 1.0}


def build_combined_table(
    synthcity_results: dict[str, pd.DataFrame],
    syntheval_benchmark_results: pd.DataFrame | None,
    syntheval_benchmark_ranks: pd.DataFrame | None,
    log_disparity_reports: dict[str, dict],
    model_names: list,
    rank_weights: dict | None = None,
    synthcity_validations: Mapping[str, MetricValidationResult] | None = None,
    syntheval_validations: Mapping[tuple[str, str], Mapping[str, MetricValidationResult]]
    | None = None,
    metric_execution_passes: Mapping[tuple[str, str, str], str] | None = None,
    custom_validations: Mapping[str, MetricValidationResult] | None = None,
    release_evidence_validations: Mapping[str, MetricValidationResult] | None = None,
    task12_validations: Mapping[str, MetricValidationResult] | None = None,
) -> pd.DataFrame:
    """Build combined audit evidence and fixed-transform tuning utility.

    ``(__all__, overall, rank)`` remains for backward-compatible consumers,
    but equals complete canonical ``U_tuning`` only. Incomplete evidence stays
    indeterminate; no candidate-relative ranking or reweighting occurs.
    ``rank_weights`` is accepted for API compatibility and ignored.
    """

    legacy_validation_argument = task12_validations is not None
    if release_evidence_validations is not None and task12_validations is not None:
        raise ValueError(
            "Pass only one of release_evidence_validations or legacy task12_validations"
        )
    if release_evidence_validations is None:
        release_evidence_validations = task12_validations

    sc_raw, sc_oriented = _synthcity_frames(
        synthcity_results,
        model_names,
        synthcity_validations=synthcity_validations,
    )
    se_raw, se_oriented = _syntheval_frames(
        syntheval_benchmark_results,
        syntheval_benchmark_ranks,
        model_names,
        syntheval_validations=syntheval_validations,
        metric_execution_passes=metric_execution_passes,
    )
    ld_raw, ld_oriented = _log_disparity_frames(
        log_disparity_reports,
        model_names,
        custom_validations=custom_validations,
    )
    release_evidence_raw, release_evidence_oriented = _release_evidence_frames(
        release_evidence_validations, model_names
    )
    if legacy_validation_argument:
        legacy_prefix = "__model_custom_task12"
        semantic_prefix = "__model_custom_release_evidence"
        release_evidence_raw = release_evidence_raw.rename(
            columns={
                column: (*column[:2], column[2].replace(semantic_prefix, legacy_prefix))
                for column in release_evidence_raw
            }
        )

    raw_parts = [df for df in (sc_raw, se_raw, ld_raw, release_evidence_raw) if not df.empty]
    oriented_parts = [
        df
        for df in (sc_oriented, se_oriented, ld_oriented, release_evidence_oriented)
        if not df.empty
    ]

    if not raw_parts:
        raise ValueError("No evaluation results to combine: check evaluation config selection")

    raw_df = pd.concat(raw_parts, axis=1)
    oriented_df = (
        pd.concat(oriented_parts, axis=1)
        if oriented_parts
        else pd.DataFrame(
            index=pd.Index(model_names),
            columns=pd.MultiIndex.from_arrays([[], [], []], names=["framework", "type", "metric"]),
        )
    )

    combined = raw_df.copy()

    # Per-framework diagnostics retain legacy rank columns for report/table
    # compatibility. They are not consumed by policy selection.
    scaled_df = oriented_df.apply(_minmax_scale, axis=0)

    # Step 2: per (framework, type) sub-rank -- MEAN of that group's scaled metrics.
    groups = sorted(
        set(
            zip(
                oriented_df.columns.get_level_values(0),
                oriented_df.columns.get_level_values(1),
                strict=True,
            )
        )
    )
    for framework, type_ in groups:
        cols = [c for c in oriented_df.columns if c[0] == framework and c[1] == type_]
        combined[(framework, type_, _RANK)] = oriented_df[cols].mean(axis=1, skipna=True)
        scaled_cols = [c for c in scaled_df.columns if c[0] == framework and c[1] == type_]
        combined[(framework, type_, _RANK)] = scaled_df[scaled_cols].mean(axis=1, skipna=True)

    for type_ in _TYPES:
        group_cols = [
            (framework, type_, _RANK) for framework, current_type in groups if current_type == type_
        ]
        if group_cols:
            combined[(_ALL, type_, _RANK)] = combined[group_cols].mean(axis=1, skipna=True)

    # Fixed canonical utility transform. These metric identities are emitted
    # by canonical HPO objectives; missing any one component makes U_tuning indeterminate.
    utility_keys = {
        "tstr": "tstr_macro_f1.v1",
        "mmd": "mixed_mmd.v1",
        "jsd": "elastic_net_jsd.v1",
    }
    utility = pd.DataFrame(index=pd.Index(model_names), dtype=float)
    validations = release_evidence_validations or {}
    for component, metric in utility_keys.items():
        values = pd.Series(float("nan"), index=model_names, dtype=float)
        for column in combined.columns:
            if column[2] == metric:
                for model in model_names:
                    numeric_value = _finite_real(combined.loc[model, column])
                    if numeric_value is not None:
                        values.loc[model] = numeric_value
                break
        # Validation policy_value is already the producer's fixed transform;
        # consume it directly to avoid double normalization.
        for model in model_names:
            validation = validations.get(model)
            if validation is None:
                continue
            for record in validation.records:
                if str(record.expected_key) == metric and record.status == "succeeded":
                    values.loc[model] = _finite_real(record.policy_value)
                    break
        # Canonical HPO values are fixed policy scores: TSTR is a score, while
        # MMD/JSD are distances with configured fixed anchors/transforms.
        for model in model_names:
            if pd.isna(values.loc[model]):
                continue
            validation = validations.get(model)
            has_policy_record = validation is not None and any(
                str(record.expected_key) == metric and record.status == "succeeded"
                for record in validation.records
            )
            if validation is None and not has_policy_record:
                raw_value = values.loc[model]
                values.loc[model] = normalize_component(
                    raw_value,
                    kind="risk"
                    if component == "mmd"
                    else "jsd"
                    if component == "jsd"
                    else "direct",
                    anchor=1.0 if component == "mmd" else None,
                )
        utility[component] = values
    u_tuning = utility.mean(axis=1).where(utility.notna().all(axis=1))
    combined[(_ALL, "utility", "U_tuning")] = u_tuning
    combined[(_ALL, "overall", _RANK)] = u_tuning

    combined.columns = pd.MultiIndex.from_tuples(
        combined.columns, names=["framework", "type", "metric"]
    )
    combined = combined.sort_values((_ALL, "overall", _RANK), ascending=False)
    combined.index.name = "model"
    return validate_combined_table(combined)


def load_combined_table(path: str | Path, *, validate_artifact: bool = True) -> pd.DataFrame:
    """Load a ``combined_evaluation.csv`` written by :func:`build_combined_table`,
    reconstructing its 3-level ``(framework, type, metric)`` column MultiIndex.
    """
    path = Path(path)
    if validate_artifact:
        from synthdata.evaluation.artifacts import artifact_bundle_dir, validate_evaluation_bundle

        if artifact_bundle_dir(path.parent).is_dir():
            validate_evaluation_bundle(path.parent)
    return validate_combined_table(pd.read_csv(path, header=[0, 1, 2], index_col=0))


def simple_rank_summary(combined: pd.DataFrame) -> pd.DataFrame:
    """Flatten ``combined`` down to a plain model x {overall,utility,privacy,fairness}
    rank table (one row per model, sorted best-to-worst) for readable printing.
    """
    columns = {}
    if (_ALL, "overall", _RANK) in combined.columns:
        columns["overall"] = combined[(_ALL, "overall", _RANK)]
    for type_ in ("utility", "privacy", "fairness"):
        key = (_ALL, type_, _RANK)
        if key in combined.columns:
            columns[type_] = combined[key]

    summary = pd.DataFrame(columns)
    summary.index.name = "model"
    if "overall" in summary.columns:
        summary = summary.sort_values("overall", ascending=False)
    return summary.round(3)
