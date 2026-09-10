"""SynthEval-based evaluation: runs SynthEval's benchmark() across all cached
synthetic datasets using a custom preset (built from
:mod:`synthdata.evaluation.catalog`, filtered by the configured selection).
"""

import copy
import dataclasses
import hashlib
import json
import math
import multiprocessing
import os
import socket
import time
import traceback
import uuid
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from synthdata.data import Dataset, role_context_payload, semantic_context_digest
from synthdata.evaluation.catalog import (
    FAIRNESS_METRICS_WITH_POSITIVE_CLASS,
    SYNTHEVAL_METRIC_TYPE,
    SYNTHEVAL_PRESET,
    resolve_selection,
    syntheval_execution_manifest,
    syntheval_framework_for_emitted_key,
)
from synthdata.evaluation.metric_contracts import (
    DEFAULT_METRIC_CONTRACT_REGISTRY,
    AmbiguousMetricContractError,
    MetricEvaluationContext,
    MetricObservation,
    MetricValidationResult,
    UnknownMetricContractError,
    resolve_metric_observations,
)
from synthdata.utils import ensure_dir, get_logger, save_json

logger = get_logger(__name__)

_RANK_COLUMNS = {"rank", "u_rank", "p_rank", "f_rank"}
_CHECKPOINT_SCHEMA_VERSION = 1


def _atomic_json(path: Path, payload: dict) -> None:
    """Atomically replace a JSON sidecar in its destination directory."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    """Atomically replace a Parquet result checkpoint."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_parquet(temporary)
    os.replace(temporary, path)


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    """Content hash including column order and dtypes for cache validity."""
    digest = hashlib.sha256()
    digest.update(
        repr([(str(column), str(dtype)) for column, dtype in frame.dtypes.items()]).encode()
    )
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def _candidate_role_frames(dataset: Dataset) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Resolve the fit, tuning, and final-holdout model frames by name."""
    fit_frame = dataset.role_frame("train", imputed=True)
    tuning_frame = dataset.role_frame("tuning", imputed=True)
    final_holdout_frame = dataset.role_frame("final_holdout", imputed=True)
    if tuning_frame is None and dataset.legacy_two_role:
        tuning_frame = final_holdout_frame
    missing = [
        role
        for role, frame in (
            ("train", fit_frame),
            ("tuning", tuning_frame),
            ("final_holdout", final_holdout_frame),
        )
        if frame is None
    ]
    if missing:
        raise RuntimeError(
            "SynthEval requires populated imputed role frame(s): " + ", ".join(missing)
        )
    return fit_frame, tuning_frame, final_holdout_frame


def _evaluation_role_frames(
    dataset: Dataset, evaluation_role: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the train fit frame and one explicitly named evidence frame."""
    fit_frame, tuning_frame, final_holdout_frame = _candidate_role_frames(dataset)
    if evaluation_role == "tuning":
        return fit_frame, tuning_frame
    if evaluation_role == "final_holdout":
        dataset.require_canonical_roles("final-holdout evaluation")
        return fit_frame, final_holdout_frame
    raise ValueError(
        f"Unsupported SynthEval evaluation_role {evaluation_role!r}; "
        "expected 'tuning' or 'final_holdout'"
    )


def build_group_context(
    dataset: Dataset,
    synthetic_datasets: dict[str, pd.DataFrame],
    *,
    group_mode: str = "row",
    group_column: str | None = None,
) -> dict:
    """Resolve and validate the population context used by evaluation.

    Canonical datasets keep population identity in ``Dataset.role_groups`` and
    never put the raw identifier in a model frame. Historical datasets may use
    raw frame identifiers, but only through the explicit legacy compatibility
    path. The returned context contains counts and fingerprints, never raw IDs.
    """
    if group_mode == "row":
        return {
            "schema_version": "evaluation-group-v1",
            "group_mode": "row",
            "population_unit": "row",
            "group_column": None,
            "roles": {},
            "models": {},
        }
    if group_mode != "patient_group":
        raise ValueError(f"Unknown evaluation group mode: {group_mode!r}")
    if not isinstance(group_column, str) or not group_column.strip():
        raise ValueError(
            "evaluation.group_column must be a non-empty string for patient_group evaluation"
        )

    def sidecar_role_metadata(role: str, frame: pd.DataFrame | None) -> tuple[dict, set]:
        if frame is None:
            raise ValueError(
                f"Patient-group evaluation requires a populated {role} frame; got None"
            )
        groups = dataset.role_groups.get(role)
        if groups is None:
            raise ValueError(
                f"Canonical patient-group evaluation requires Dataset.role_groups[{role!r}]; "
                "raw identifiers are not accepted in canonical model frames"
            )
        values = pd.Series(groups).reset_index(drop=True)
        if len(values) != len(frame):
            raise ValueError(
                f"Dataset.role_groups[{role!r}] has {len(values)} values for a {len(frame)}-row "
                "role frame"
            )
        if values.isna().any():
            raise ValueError(
                f"Dataset.role_groups[{role!r}] contains missing values in {role} frame"
            )
        try:
            normalized = values.map(lambda value: json.dumps(value, sort_keys=True, default=str))
            identifiers = set(normalized.tolist())
            fingerprint = hashlib.sha256(
                json.dumps(normalized.tolist(), separators=(",", ":")).encode()
            ).hexdigest()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Dataset.role_groups[{role!r}] must contain hashable scalar values"
            ) from exc
        return (
            {
                "rows": int(len(frame)),
                "groups": int(len(identifiers)),
                "fingerprint": fingerprint,
                "group_source": "dataset_role_groups",
            },
            identifiers,
        )

    roles = {}
    if dataset.legacy_two_role:

        def legacy_role_metadata(role: str, frame: pd.DataFrame | None) -> tuple[dict, set]:
            if frame is None:
                raise ValueError(
                    f"Patient-group evaluation requires a populated {role} frame; got None"
                )
            if group_column not in frame.columns:
                raise ValueError(
                    f"Patient-group identifier column {group_column!r} is missing from {role} "
                    f"frame; available columns: {list(frame.columns)}"
                )
            values = frame[group_column]
            if values.isna().any():
                raise ValueError(
                    f"Patient-group identifier column {group_column!r} contains missing values in "
                    f"{role} frame"
                )
            try:
                identifiers = set(values.tolist())
                fingerprint = _frame_fingerprint(frame[[group_column]])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Patient-group identifier column {group_column!r} in {role} frame must contain "
                    "hashable scalar values"
                ) from exc
            return (
                {
                    "rows": int(len(frame)),
                    "groups": int(len(identifiers)),
                    "fingerprint": fingerprint,
                    "group_source": "legacy_frame_column",
                },
                identifiers,
            )

        train_metadata, train_identifiers = legacy_role_metadata(
            "train", dataset.role_frame("train", imputed=True)
        )
        holdout_metadata, holdout_identifiers = legacy_role_metadata(
            "holdout", dataset.role_frame("final_holdout", imputed=True)
        )
        overlap = train_identifiers & holdout_identifiers
        if overlap:
            raise ValueError(
                f"Patient-group identifiers overlap between train and holdout for column "
                f"{group_column!r} ({len(overlap)} group(s)); grouped evaluation requires disjoint "
                "real-data roles"
            )
        roles["train"] = train_metadata
        roles["holdout"] = holdout_metadata
    else:
        fit_frame, tuning_frame, final_holdout_frame = _candidate_role_frames(dataset)
        role_frames = {
            "train": fit_frame,
            "tuning": tuning_frame,
            "final_holdout": final_holdout_frame,
        }
        role_identifiers = {}
        for role, frame in role_frames.items():
            roles[role], role_identifiers[role] = sidecar_role_metadata(role, frame)
        for left_index, left_role in enumerate(role_frames):
            for right_role in list(role_frames)[left_index + 1 :]:
                overlap = role_identifiers[left_role] & role_identifiers[right_role]
                if overlap:
                    raise ValueError(
                        f"Patient-group identifiers overlap between {left_role} and {right_role} "
                        f"for column {group_column!r} ({len(overlap)} group(s)); grouped "
                        "evaluation requires disjoint real-data roles"
                    )

    models = {}
    for model_name, frame in synthetic_datasets.items():
        if not dataset.legacy_two_role and group_column in frame.columns:
            raise ValueError(
                f"Canonical synthetic model {model_name!r} contains the raw patient-group "
                f"column {group_column!r}; identity fields must be excluded from model frames"
            )
        models[model_name] = {
            "rows": int(len(frame)),
            "groups": None,
            "fingerprint": _frame_fingerprint(frame),
            "group_source": "unavailable_for_synthetic_output"
            if not dataset.legacy_two_role
            else "legacy_frame_column_required",
        }
        if dataset.legacy_two_role:
            metadata, _identifiers = legacy_role_metadata(f"synthetic model {model_name!r}", frame)
            models[model_name] = metadata

    return {
        "schema_version": "evaluation-group-v1",
        "group_mode": "patient_group",
        "population_unit": "patient_group",
        "group_column": group_column,
        "roles": roles,
        "models": models,
    }


def _checkpoint_model_id(model_name: str) -> str:
    """Stable path-safe model id while retaining the original name in metadata."""
    return hashlib.sha256(model_name.encode()).hexdigest()[:16]


def _available_memory_gib() -> float | None:
    """Return Linux MemAvailable in GiB, or None where it cannot be observed."""
    try:
        lines = Path("/proc/meminfo").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    return None


def resolve_model_workers(execution_cfg, *, n_models: int, n_columns: int) -> int:
    """Resolve a memory- and CPU-bounded number of concurrent model processes.

    The Linux ``MemAvailable`` value is already the kernel's estimate of
    immediately allocatable memory. It is therefore the only host-memory
    value used for auto-sizing: ``MemTotal`` can describe a smaller container
    or runner limit than the available-memory probe supplied by callers/tests,
    making worker selection depend on an unrelated second system read.
    """
    if not n_models:
        return 0
    requested = execution_cfg.model_workers
    if requested != "auto":
        return min(requested, execution_cfg.max_model_workers, n_models)

    cpu_count = os.cpu_count() or 1
    cpu_bound = max(1, cpu_count // execution_cfg.cores_per_model)
    per_model_gib = execution_cfg.memory_per_model_gib or max(6.0, 0.0135 * n_columns)
    available_gib = _available_memory_gib()
    if available_gib is None:
        memory_bound = 1
    else:
        budget_gib = max(0.0, available_gib - execution_cfg.memory_reserve_gib)
        memory_bound = max(1, int(budget_gib // per_model_gib))
    return max(1, min(n_models, execution_cfg.max_model_workers, cpu_bound, memory_bound))


def _checkpoint_paths(
    checkpoint_root: Path, pass_name: str, model_name: str
) -> tuple[Path, Path, Path]:
    model_dir = (
        checkpoint_root
        / f"checkpoints-v{_CHECKPOINT_SCHEMA_VERSION}"
        / pass_name
        / _checkpoint_model_id(model_name)
    )
    return model_dir, model_dir / "status.json", model_dir / "result.parquet"


def _execution_checkpoint_path(model_dir: Path) -> Path:
    return model_dir / "execution.json"


def _finite_number(value, *, allow_none: bool = False) -> bool:
    if value is None:
        return allow_none
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    return math.isfinite(float(value))


def _string_sequence(value) -> bool:
    return isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value)


def _execution_payload_succeeded(
    payload: dict,
    *,
    expected_manifest: dict | None = None,
    expected_pass_id: str | None = None,
    expected_target_view: str | None = None,
) -> bool:
    """Validate persisted execution status and structured rows before reuse."""
    if not isinstance(payload, dict):
        return False
    if payload.get("schema_version") != "syntheval-execution-v1":
        return False
    if expected_pass_id is not None and payload.get("pass_id") != expected_pass_id:
        return False
    if expected_target_view is not None and payload.get("target_view") != expected_target_view:
        return False
    semantic_context = payload.get("semantic_context")
    semantic_fingerprint = payload.get("semantic_context_digest")
    if semantic_context is None:
        if semantic_fingerprint is not None:
            return False
    else:
        if not isinstance(semantic_context, Mapping) or not isinstance(semantic_fingerprint, str):
            return False
        try:
            if semantic_fingerprint != semantic_context_digest(semantic_context):
                return False
        except (TypeError, ValueError):
            return False
    executions = payload.get("metric_executions")
    if not isinstance(executions, list) or not executions:
        return False
    if payload.get("execution_complete") is not True:
        return False
    if payload.get("execution_succeeded") is not True:
        return False

    methods = []
    for item in executions:
        if not isinstance(item, dict) or not isinstance(item.get("method"), str):
            return False
        method = item["method"]
        if not method or method in methods:
            return False
        methods.append(method)

        status = item.get("status")
        if not isinstance(status, dict) or status.get("method") != method:
            return False
        if status.get("state") != "succeeded":
            return False
        status_keys = {}
        for field in (
            "expected_keys",
            "observed_keys",
            "completed_keys",
            "failed_keys",
            "missing_keys",
            "duplicate_keys",
            "non_finite_keys",
            "unexpected_keys",
        ):
            values = status.get(field)
            if not _string_sequence(values) or len(values) != len(set(values)):
                return False
            status_keys[field] = tuple(values)
        expected = status_keys["expected_keys"]
        if not expected:
            return False
        if set(status_keys["completed_keys"]) != set(expected):
            return False
        if any(status_keys[field] for field in ("failed_keys", "missing_keys")):
            return False
        if any(
            status_keys[field] for field in ("duplicate_keys", "non_finite_keys", "unexpected_keys")
        ):
            return False
        if status.get("exception_type") or status.get("exception_message"):
            return False

        versioned = any("_v2" in key for key in expected)
        field = "normalized_rows_v2" if versioned else "normalized_rows"
        rows = item.get(field)
        if not isinstance(rows, (list, tuple)) or not rows:
            return False
        row_keys = []
        for row in rows:
            if not isinstance(row, dict):
                return False
            result_metadata = row.get("result_metadata", row.get("metadata", {}))
            if result_metadata is not None and not isinstance(result_metadata, Mapping):
                return False
            emitted_key = row.get("metric")
            if not isinstance(emitted_key, str) or not emitted_key or emitted_key in row_keys:
                return False
            if not isinstance(row.get("dim"), str) or not row["dim"]:
                return False
            if any(
                field_name not in row or not _finite_number(row[field_name])
                for field_name in ("val", "n_val")
            ):
                return False
            if any(
                field_name in row and not _finite_number(row[field_name], allow_none=True)
                for field_name in ("err", "n_err", "raw_value", "normalized_value")
            ):
                return False
            if versioned and (
                row.get("metric_version") != "v2"
                or not _finite_number(row.get("raw_value"))
                or not _finite_number(row.get("normalized_value"))
            ):
                return False
            row_keys.append(emitted_key)
        if set(row_keys) != set(status_keys["observed_keys"]):
            return False
        if not set(expected) <= set(row_keys):
            return False

        if expected_manifest is not None:
            manifest_keys = expected_manifest.get(method)
            if manifest_keys is None or tuple(str(key) for key in manifest_keys) != expected:
                return False

    return expected_manifest is None or set(methods) == set(expected_manifest)


def _valid_worker_exit_record(value: object) -> bool:
    if not isinstance(value, dict) or value.get("state") != "failed":
        return False
    for field in ("exception_type", "exception_message"):
        if not isinstance(value.get(field), str) or not value[field].strip():
            return False
    traceback_value = value.get("traceback")
    if traceback_value is not None and not isinstance(traceback_value, str):
        return False
    exit_code = value.get("exit_code")
    if exit_code is not None and (isinstance(exit_code, bool) or not isinstance(exit_code, int)):
        return False
    failed_at = value.get("failed_at")
    return failed_at is None or _finite_number(failed_at)


def _execution_payload_failed(
    payload: dict,
    *,
    expected_manifest: dict | None = None,
    expected_pass_id: str | None = None,
    expected_target_view: str | None = None,
) -> bool:
    """Validate failed audit evidence without making it reusable as a checkpoint."""
    if not isinstance(payload, dict):
        return False
    if payload.get("schema_version") != "syntheval-execution-v1":
        return False
    if expected_pass_id is not None and payload.get("pass_id") != expected_pass_id:
        return False
    if expected_target_view is not None and payload.get("target_view") != expected_target_view:
        return False
    if not isinstance(payload.get("execution_complete"), bool):
        return False
    if payload.get("execution_succeeded") is not False:
        return False
    if payload.get("policy_eligible") is not False:
        return False
    failure_reason = payload.get("failure_reason")
    if not isinstance(failure_reason, str) or not failure_reason.strip():
        return False
    worker_exit = payload.get("worker_exit")
    if worker_exit is not None and not _valid_worker_exit_record(worker_exit):
        return False

    executions = payload.get("metric_executions")
    if not isinstance(executions, list) or not executions:
        return False
    methods = []
    has_failed_method = False
    for item in executions:
        if not isinstance(item, dict) or not isinstance(item.get("method"), str):
            return False
        method = item["method"]
        if not method or method in methods:
            return False
        methods.append(method)
        status = item.get("status")
        if not isinstance(status, dict) or status.get("method") != method:
            return False
        status_keys = {}
        for field in (
            "expected_keys",
            "observed_keys",
            "completed_keys",
            "failed_keys",
            "missing_keys",
            "duplicate_keys",
            "non_finite_keys",
            "unexpected_keys",
        ):
            values = status.get(field)
            if not _string_sequence(values) or len(values) != len(set(values)):
                return False
            status_keys[field] = tuple(values)
        expected = status_keys["expected_keys"]
        if not expected:
            return False

        if expected_manifest is not None:
            manifest_keys = expected_manifest.get(method)
            if manifest_keys is None or tuple(str(key) for key in manifest_keys) != expected:
                return False

        if status.get("state") == "succeeded":
            if worker_exit is None:
                return False
            successful_payload = {
                "schema_version": "syntheval-execution-v1",
                "pass_id": payload.get("pass_id"),
                "target_view": payload.get("target_view"),
                "execution_complete": True,
                "execution_succeeded": True,
                "metric_executions": [item],
            }
            if not _execution_payload_succeeded(
                successful_payload,
                expected_manifest={method: expected},
                expected_pass_id=expected_pass_id,
                expected_target_view=expected_target_view,
            ):
                return False
            continue

        if status.get("state") not in {"failed", "timed_out", "blocked"}:
            return False
        has_failed_method = True
        expected_set = set(expected)
        observed_set = set(status_keys["observed_keys"])
        completed_set = set(status_keys["completed_keys"])
        failed_set = set(status_keys["failed_keys"])
        missing_set = set(status_keys["missing_keys"])
        if completed_set & failed_set:
            return False
        if completed_set | failed_set != expected_set:
            return False
        if completed_set != ((expected_set & observed_set) - failed_set):
            return False
        if missing_set != expected_set - observed_set:
            return False
        if any(
            status_keys[field] for field in ("duplicate_keys", "non_finite_keys", "unexpected_keys")
        ):
            return False
        for field in ("exception_type", "exception_message"):
            value = status.get(field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                return False
        traceback_value = status.get("exception_traceback")
        if traceback_value is not None and not isinstance(traceback_value, str):
            return False
        versioned = any("_v2" in key for key in expected)
        field = "normalized_rows_v2" if versioned else "normalized_rows"
        rows = item.get(field)
        if not isinstance(rows, (list, tuple)):
            return False
        row_keys = []
        for row in rows:
            if not isinstance(row, dict):
                return False
            result_metadata = row.get("result_metadata", row.get("metadata", {}))
            if result_metadata is not None and not isinstance(result_metadata, Mapping):
                return False
            emitted_key = row.get("metric")
            if not isinstance(emitted_key, str) or not emitted_key or emitted_key in row_keys:
                return False
            if not isinstance(row.get("dim"), str) or not row["dim"]:
                return False
            if any(
                field_name not in row or not _finite_number(row[field_name])
                for field_name in ("val", "n_val")
            ):
                return False
            if any(
                field_name in row and not _finite_number(row[field_name], allow_none=True)
                for field_name in ("err", "n_err", "raw_value", "normalized_value")
            ):
                return False
            if versioned and (
                row.get("metric_version") != "v2"
                or not _finite_number(row.get("raw_value"))
                or not _finite_number(row.get("normalized_value"))
            ):
                return False
            row_keys.append(emitted_key)
        if set(row_keys) != observed_set:
            return False

    if not has_failed_method and worker_exit is None:
        return False
    return expected_manifest is None or set(methods) == set(expected_manifest)


def _execution_sidecar_payload(
    execution,
    model_name: str | None = None,
    group_context: dict | None = None,
    context_fingerprint: str | None = None,
    role_context: dict | None = None,
    semantic_context: Mapping[str, Any] | None = None,
) -> dict:
    """Serialize structured execution evidence without embedding large raw objects."""
    return {
        "model_name": model_name,
        "group_context": group_context,
        "context_fingerprint": context_fingerprint,
        "role_context": role_context,
        "semantic_context": (dict(semantic_context) if semantic_context is not None else None),
        "semantic_context_digest": (
            semantic_context_digest(semantic_context) if semantic_context is not None else None
        ),
        "schema_version": execution.schema_version,
        "pass_id": execution.pass_id,
        "target_view": execution.target_view,
        "expected_manifest_digest": execution.expected_manifest_digest,
        "execution_complete": execution.execution_complete,
        "execution_succeeded": getattr(execution, "succeeded", False),
        "policy_eligible": execution.policy_eligible,
        "preprocessing_fingerprint": getattr(execution, "preprocessing_fingerprint", None),
        "preprocessing_metadata": getattr(execution, "preprocessing_metadata", None),
        "metric_executions": [
            {
                "method": item.method,
                "status": item.status.to_dict(),
                "normalized_rows": [
                    _execution_metric_row_payload(row) for row in item.normalized_rows
                ],
                "normalized_rows_v2": [
                    _execution_metric_row_payload(row) for row in item.normalized_rows_v2
                ],
            }
            for item in execution.metric_executions
        ],
    }


def _failed_execution_payload(
    *,
    model_name: str,
    pass_name: str,
    target_view: str,
    expected_manifest_digest: str,
    expected_output_manifest: dict,
    context_fingerprint: str,
    role_context: dict,
    group_context: dict | None,
    semantic_context: Mapping[str, Any] | None = None,
    failure_status: dict,
    existing_execution: dict | None = None,
) -> dict:
    """Serialize explicit failed execution evidence for a worker exit."""
    failure_reason = failure_status.get("exception") or failure_status.get("failure_reason")
    if not isinstance(failure_reason, str) or not failure_reason:
        failure_reason = f"SynthEval worker exited with code {failure_status.get('exit_code')}"
    exception_type = failure_status.get("exception_type") or "WorkerExit"
    exception_traceback = failure_status.get("traceback")
    worker_exit = {
        "state": "failed",
        "exit_code": failure_status.get("exit_code"),
        "exception_type": str(exception_type),
        "exception_message": failure_reason,
        "traceback": exception_traceback,
        "failed_at": failure_status.get("failed_at"),
    }

    if isinstance(existing_execution, dict):
        retained = copy.deepcopy(existing_execution)
        retained.update(
            {
                "execution_succeeded": False,
                "policy_eligible": False,
                "failure_reason": failure_reason,
                "worker_exit": worker_exit,
            }
        )
        if (
            retained.get("model_name") == model_name
            and retained.get("schema_version") == "syntheval-execution-v1"
            and retained.get("pass_id") == pass_name
            and retained.get("target_view") == target_view
            and retained.get("expected_manifest_digest") == expected_manifest_digest
            and retained.get("context_fingerprint") == context_fingerprint
            and _execution_payload_failed(
                retained,
                expected_manifest=expected_output_manifest,
                expected_pass_id=pass_name,
                expected_target_view=target_view,
            )
        ):
            logger.info(
                "[syntheval] retaining completed child metric evidence for failed model=%s; "
                "adding parent worker-exit record",
                model_name,
            )
            return retained
        logger.warning(
            "[syntheval] ignoring invalid or stale child execution evidence for failed model=%s; "
            "writing explicit worker-failure rows",
            model_name,
        )
    metric_executions = []
    for method, keys in expected_output_manifest.items():
        expected_keys = [str(key) for key in keys]
        if not expected_keys:
            continue
        metric_executions.append(
            {
                "method": str(method),
                "status": {
                    "method": str(method),
                    "state": "failed",
                    "expected_keys": expected_keys,
                    "observed_keys": [],
                    "completed_keys": [],
                    "failed_keys": expected_keys,
                    "missing_keys": expected_keys,
                    "duplicate_keys": [],
                    "non_finite_keys": [],
                    "unexpected_keys": [],
                    "warnings": [],
                    "exception_type": str(exception_type),
                    "exception_message": failure_reason,
                    "exception_traceback": exception_traceback,
                    "started_at": failure_status.get("started_at"),
                    "completed_at": None,
                    "elapsed_seconds": failure_status.get("elapsed_seconds"),
                },
                "normalized_rows": [],
                "normalized_rows_v2": [],
            }
        )
    return {
        "model_name": model_name,
        "group_context": group_context,
        "context_fingerprint": context_fingerprint,
        "role_context": role_context,
        "schema_version": "syntheval-execution-v1",
        "pass_id": pass_name,
        "target_view": target_view,
        "expected_manifest_digest": expected_manifest_digest,
        "execution_complete": False,
        "execution_succeeded": False,
        "policy_eligible": False,
        "preprocessing_fingerprint": None,
        "preprocessing_metadata": None,
        "failure_reason": failure_reason,
        "exit_code": failure_status.get("exit_code"),
        "worker_exit": worker_exit,
        "metric_executions": metric_executions,
    }


def _execution_manifest_digest(manifest: dict) -> str:
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _result_metadata_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return one normalized metadata object from a native SynthEval row."""
    metadata = row.get("result_metadata", row.get("metadata", {}))
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise ValueError("SynthEval metric result_metadata must be an object")
    return dict(metadata)


def _execution_metric_row_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    payload["result_metadata"] = _result_metadata_payload(row)
    return payload


def _valid_checkpoint(
    checkpoint_root: Path,
    pass_name: str,
    model_name: str,
    context_fingerprint: str,
    model_fingerprint: str,
    require_plots: bool,
    expected_manifest_digest: str | None = None,
    return_execution: bool = False,
    expected_manifest: dict | None = None,
    expected_target_view: str | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, dict] | None:
    model_dir, status_path, result_path = _checkpoint_paths(checkpoint_root, pass_name, model_name)
    execution_path = _execution_checkpoint_path(model_dir)
    if not (status_path.exists() and result_path.exists() and execution_path.exists()):
        return None
    try:
        status = json.loads(status_path.read_text())
        if (
            status.get("state") != "succeeded"
            or status.get("schema_version") != _CHECKPOINT_SCHEMA_VERSION
            or status.get("model_name") != model_name
            or status.get("context_fingerprint") != context_fingerprint
            or status.get("model_fingerprint") != model_fingerprint
            or (
                expected_manifest_digest is not None
                and status.get("expected_manifest_digest") != expected_manifest_digest
            )
            or (require_plots and not status.get("plots_completed"))
        ):
            return None
        execution = json.loads(execution_path.read_text())
        if (
            execution.get("schema_version") != "syntheval-execution-v1"
            or execution.get("context_fingerprint") != context_fingerprint
            or (
                expected_manifest_digest is not None
                and execution.get("expected_manifest_digest") != expected_manifest_digest
            )
            or not _execution_payload_succeeded(
                execution,
                expected_manifest=expected_manifest,
                expected_pass_id=pass_name,
                expected_target_view=expected_target_view,
            )
        ):
            logger.info(
                "[syntheval] checkpoint for %s has incomplete or semantically invalid metric execution; "
                "recomputing",
                model_name,
            )
            return None
        result = pd.read_parquet(result_path)
        return (result, execution) if return_execution else result
    except (OSError, ValueError, json.JSONDecodeError):
        logger.warning(
            "[syntheval] invalid checkpoint for %s at %s; recomputing", model_name, model_dir
        )
        return None


def _shutdown_nested_joblib_executor() -> None:
    """Stop SynthEval's reusable loky pool before a disposable worker exits.

    SynthEval's metric-level joblib calls intentionally keep the reusable pool
    alive. In a one-model subprocess that leaves interpreter shutdown waiting
    for loky's 300-second idle timeout, so terminate the pool explicitly here.
    """
    from joblib.externals.loky import reusable_executor

    executor = getattr(reusable_executor, "_executor", None)
    if executor is not None:
        executor.shutdown(wait=True, kill_workers=True)


def _model_worker(
    model_name: str,
    synthetic_frame: pd.DataFrame,
    real_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame | None,
    cat_cols: list,
    target_column: str,
    sensitive_columns: list,
    preset_path: str,
    checkpoint_root: str,
    pass_name: str,
    expected_output_manifest: dict,
    target_view: str,
    expected_manifest_digest: str,
    context_fingerprint: str,
    model_fingerprint: str,
    plots_output_dir: str | None,
    cores_per_model: int,
    group_context: dict | None = None,
    role_context: dict | None = None,
    semantic_context: Mapping[str, Any] | None = None,
) -> None:
    """Run exactly one model in a disposable child process and checkpoint it."""
    os.environ["LOKY_MAX_CPU_COUNT"] = str(cores_per_model)
    os.environ["OMP_NUM_THREADS"] = str(cores_per_model)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cores_per_model)
    # Workers change into their model-specific native-plot directory before
    # checkpointing. Keep checkpoint paths absolute so a successful evaluation
    # cannot fail merely because its current working directory changed.
    checkpoint_root_path = Path(checkpoint_root).resolve()
    model_dir, status_path, result_path = _checkpoint_paths(
        checkpoint_root_path, pass_name, model_name
    )
    execution_path = _execution_checkpoint_path(model_dir)
    ensure_dir(model_dir)
    start = time.time()
    _atomic_json(
        status_path,
        {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "state": "running",
            "model_name": model_name,
            "target_view": target_view,
            "expected_manifest_digest": expected_manifest_digest,
            "context_fingerprint": context_fingerprint,
            "role_context": role_context,
            "semantic_context": (dict(semantic_context) if semantic_context is not None else None),
            "model_fingerprint": model_fingerprint,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at": start,
            "shape": list(synthetic_frame.shape),
            "plots_completed": False,
        },
    )
    original_dir = Path.cwd()
    try:
        from syntheval import AnalysisConfig, SynthEval

        analysis_config = AnalysisConfig(
            dataset=real_frame,
            target_vars=target_column,
            confounder_vars=None,
            sensitive_vars=sensitive_columns,
        )
        plot_dir = None
        if plots_output_dir is not None:
            # Resolve before changing the worker's directory. SynthEval writes
            # native diagnostics relative to CWD; retaining an absolute path
            # lets us accurately inventory the files after evaluation.
            plot_dir = ensure_dir(Path(plots_output_dir).resolve() / model_name)
            os.chdir(plot_dir)
        se = SynthEval(
            real_frame,
            holdout_dataframe=holdout_frame,
            cat_cols=cat_cols,
            verbose=False,
            enable_plots=plot_dir is not None,
            console="off",
            show_warnings=False,
        )
        execution = se.evaluate(
            synthetic_frame,
            analysis_target=analysis_config,
            presets_file=preset_path,
            _dataset_name=model_name,
            return_execution=True,
            expected_output_manifest=expected_output_manifest,
            pass_id=pass_name,
            target_view=target_view,
            expected_manifest_digest=expected_manifest_digest,
            group_context=group_context,
        )
        if execution.normalized_table is None:
            raise RuntimeError("SynthEval returned no legacy normalized metric results")
        _atomic_parquet(result_path, execution.normalized_table)
        _atomic_json(
            execution_path,
            _execution_sidecar_payload(
                execution,
                model_name,
                group_context=group_context,
                context_fingerprint=context_fingerprint,
                role_context=role_context,
                semantic_context=semantic_context,
            ),
        )
        if not execution.succeeded:
            failed_methods = [
                item.method for item in execution.metric_executions if not item.status.succeeded
            ]
            raise RuntimeError(
                "SynthEval returned incomplete or failed required metrics for "
                f"model {model_name!r}: {', '.join(failed_methods) or 'unknown'}"
            )
        plot_files = (
            sorted(
                str(path.relative_to(plot_dir)) for path in plot_dir.rglob("*") if path.is_file()
            )
            if plot_dir
            else []
        )
        _atomic_json(
            status_path,
            {
                "schema_version": _CHECKPOINT_SCHEMA_VERSION,
                "state": "succeeded",
                "model_name": model_name,
                "target_view": target_view,
                "expected_manifest_digest": expected_manifest_digest,
                "context_fingerprint": context_fingerprint,
                "semantic_context": (
                    dict(semantic_context) if semantic_context is not None else None
                ),
                "semantic_context_digest": (
                    semantic_context_digest(semantic_context)
                    if semantic_context is not None
                    else None
                ),
                "role_context": role_context,
                "model_fingerprint": model_fingerprint,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "started_at": start,
                "completed_at": time.time(),
                "elapsed_seconds": time.time() - start,
                "shape": list(synthetic_frame.shape),
                "plots_completed": plot_dir is not None,
                "plot_files": plot_files,
                "execution_complete": execution.execution_complete,
                "execution_succeeded": execution.succeeded,
                "policy_eligible": execution.policy_eligible,
            },
        )
    except Exception as exc:  # noqa: BLE001 -- process boundary must persist any worker failure before re-raising
        logger.error(
            "[syntheval] worker failed for model=%s pass=%s; persisting failure at %s: %s: %s",
            model_name,
            pass_name,
            status_path,
            type(exc).__name__,
            exc,
        )
        _atomic_json(
            status_path,
            {
                "schema_version": _CHECKPOINT_SCHEMA_VERSION,
                "state": "failed",
                "model_name": model_name,
                "target_view": target_view,
                "expected_manifest_digest": expected_manifest_digest,
                "context_fingerprint": context_fingerprint,
                "role_context": role_context,
                "semantic_context": (
                    dict(semantic_context) if semantic_context is not None else None
                ),
                "semantic_context_digest": (
                    semantic_context_digest(semantic_context)
                    if semantic_context is not None
                    else None
                ),
                "model_fingerprint": model_fingerprint,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "started_at": start,
                "failed_at": time.time(),
                "elapsed_seconds": time.time() - start,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
                "shape": list(synthetic_frame.shape),
                "plots_completed": False,
            },
        )
        raise
    finally:
        try:
            _shutdown_nested_joblib_executor()
        finally:
            os.chdir(original_dir)


# ---------------------------------------------------------------------------
# Benchmark result caching
# ---------------------------------------------------------------------------
# SynthEval's benchmark() is expensive (tens of minutes for large datasets).
# After a successful run we persist the results and ranks as Parquet files
# alongside a small metadata sidecar that captures the cache key (SHA-256 of
# the preset JSON + sorted model names).  On the next run we load from cache
# if the key matches, skipping the full benchmark pass.
#
# Parquet is used (not CSV) because benchmark_results has a MultiIndex column
# level that CSV cannot round-trip without bespoke reconstruction logic.
# ---------------------------------------------------------------------------


def _compute_cache_key(
    preset: dict,
    model_names: list[str],
    ranking_strategy: str,
    evaluation_fingerprint: str | None = None,
) -> str:
    """Stable SHA-256 digest of the benchmark configuration and input fingerprint.

    ``ranking_strategy`` is included because it affects the ranks DataFrame
    returned by ``se.benchmark()`` (not just the metric values).
    """
    payload = json.dumps(
        {
            "preset": preset,
            "models": sorted(model_names),
            "ranking_strategy": ranking_strategy,
            "evaluation_fingerprint": evaluation_fingerprint,
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _save_syntheval_cache(
    results: pd.DataFrame,
    ranks: pd.DataFrame,
    cache_dir: Path,
    prefix: str,
    cache_key: str,
    *,
    context_fingerprint: str | None = None,
    role_context: dict | None = None,
    expected_manifest_digest: str | None = None,
    target_view: str | None = None,
    registry_digest: str | None = None,
) -> None:
    """Persist benchmark results + ranks to Parquet and write the cache-key sidecar.

    Files written:
    - ``<cache_dir>/<prefix>_results.parquet``  -- MultiIndex-column results
    - ``<cache_dir>/<prefix>_ranks.parquet``    -- ranks DataFrame
    - ``<cache_dir>/<prefix>_cache_meta.json``  -- cache key and role context
    """
    ensure_dir(cache_dir)
    _atomic_parquet(cache_dir / f"{prefix}_results.parquet", results)
    _atomic_parquet(cache_dir / f"{prefix}_ranks.parquet", ranks)
    meta_path = cache_dir / f"{prefix}_cache_meta.json"
    _atomic_json(
        meta_path,
        {
            "schema_version": "syntheval-cache-v1",
            "cache_key": cache_key,
            "context_fingerprint": context_fingerprint,
            "role_context": role_context,
            "expected_manifest_digest": expected_manifest_digest,
            "target_view": target_view,
            "registry_digest": registry_digest or DEFAULT_METRIC_CONTRACT_REGISTRY.digest(),
        },
    )
    logger.info(
        "[syntheval] cached %s benchmark results to %s (key=%s…)",
        prefix,
        cache_dir,
        cache_key[:12],
    )


def _load_syntheval_cache(
    cache_dir: Path,
    prefix: str,
    cache_key: str,
    *,
    context_fingerprint: str | None = None,
    expected_manifest_digest: str | None = None,
    expected_target_view: str | None = None,
    registry_digest: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Return (results, ranks) from cache if the key matches, else None.

    Returns None (cache miss) if any of the three expected files are absent or
    if the stored cache key doesn't match the current one.
    """
    results_path = cache_dir / f"{prefix}_results.parquet"
    ranks_path = cache_dir / f"{prefix}_ranks.parquet"
    meta_path = cache_dir / f"{prefix}_cache_meta.json"

    if not (results_path.exists() and ranks_path.exists() and meta_path.exists()):
        return None

    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("[syntheval] %s cache meta unreadable (%s); treating as miss", prefix, exc)
        return None

    if meta.get("cache_key") != cache_key:
        logger.info(
            "[syntheval] %s cache key mismatch (stored=%s…, current=%s…); recomputing",
            prefix,
            str(meta.get("cache_key", ""))[:12],
            cache_key[:12],
        )
        return None
    if context_fingerprint is not None and (
        meta.get("schema_version") != "syntheval-cache-v1"
        or meta.get("context_fingerprint") != context_fingerprint
    ):
        logger.info(
            "[syntheval] %s cache role context mismatch or missing metadata; recomputing",
            prefix,
        )
        return None
    if (
        expected_manifest_digest is not None
        and meta.get("expected_manifest_digest") != expected_manifest_digest
    ):
        logger.info("[syntheval] %s cache expected-manifest mismatch; recomputing", prefix)
        return None
    if expected_target_view is not None and meta.get("target_view") != expected_target_view:
        logger.info("[syntheval] %s cache target-view mismatch; recomputing", prefix)
        return None
    if registry_digest is not None and meta.get("registry_digest") != registry_digest:
        logger.info("[syntheval] %s cache registry mismatch; recomputing", prefix)
        return None

    try:
        results = pd.read_parquet(results_path)
        ranks = pd.read_parquet(ranks_path)
    except Exception as exc:  # noqa: BLE001 -- any parquet read error → cache miss
        logger.warning("[syntheval] %s cache files unreadable (%s); recomputing", prefix, exc)
        return None

    logger.info(
        "[syntheval] loaded %s benchmark results from cache (key=%s…, %d models, %d metrics)",
        prefix,
        cache_key[:12],
        len(results),
        len([c for c in results.columns.get_level_values(0).unique() if c not in _RANK_COLUMNS]),
    )
    return results, ranks


def _load_syntheval_execution_sidecars(
    cache_dir: Path,
    pass_name: str,
    model_names: list[str],
    expected_manifest_digest: str,
    context_fingerprint: str | None = None,
    *,
    expected_manifest: dict | None = None,
    expected_target_view: str | None = None,
) -> dict[str, dict] | None:
    """Load manifest-bound structured execution evidence for a cached pass."""
    executions = {}
    for model_name in model_names:
        model_dir, _status_path, _result_path = _checkpoint_paths(cache_dir, pass_name, model_name)
        path = _execution_checkpoint_path(model_dir)
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "[syntheval] %s execution sidecar unreadable for model %s at %s (%s); "
                "treating cache as incomplete",
                pass_name,
                model_name,
                path,
                exc,
            )
            return None
        if (
            payload.get("model_name") != model_name
            or payload.get("schema_version") != "syntheval-execution-v1"
            or payload.get("expected_manifest_digest") != expected_manifest_digest
            or (
                context_fingerprint is not None
                and payload.get("context_fingerprint") != context_fingerprint
            )
            or not _execution_payload_succeeded(
                payload,
                expected_manifest=expected_manifest,
                expected_pass_id=pass_name,
                expected_target_view=expected_target_view,
            )
        ):
            logger.warning(
                "[syntheval] %s execution sidecar identity or semantic validation failed for model %s at %s; "
                "treating cache as incomplete",
                pass_name,
                model_name,
                path,
            )
            return None
        executions[model_name] = payload
    return executions


#: Metrics that syntheval refuses to run unless the target has EXACTLY 2
#: classes (see e.g. metric_auroc_difference.py / metric_statistical_parity.py
#: / metric_equalized_odds.py / metric_equal_opportunity.py's own
#: `target_types.items() if value == 2` filtering) -- these are the only ones
#: a binary-target evaluation pass (see run_binary_target_syntheval_evaluation)
#: can newly enable; every other metric already runs fine against a 3+ class
#: target and must NOT be re-run against the collapsed binary one (which
#: would silently double-count/duplicate their results).
BINARY_ONLY_METRICS = frozenset(
    {"auroc_diff", "statistical_parity", "equalized_odds", "equal_opportunity"}
)


def build_preset(
    selection_cfg,
    positive_class=1,
    *,
    target_is_binary: bool | None = None,
) -> dict:
    """Filter the full SynthEval preset down to the configured selection.

    ``positive_class`` overrides the "positive_class" preset param (default
    ``1`` in SYNTHEVAL_PRESET) for the 3 fairness metrics in
    FAIRNESS_METRICS_WITH_POSITIVE_CLASS -- wired from
    ``cfg.evaluation.positive_class``, so it only makes sense when the real
    target column is already exactly 2 classes (e.g. hepatitis). It does NOT
    apply to the separate binary_target pass (see build_binary_preset), whose
    collapsed target is always 1=positive/0=negative by construction.
    """
    all_names = list(SYNTHEVAL_PRESET.keys())
    selected = resolve_selection(
        selection_cfg.enabled,
        selection_cfg.categories,
        selection_cfg.metrics,
        all_names,
        SYNTHEVAL_METRIC_TYPE,
    )
    preset = {k: v for k, v in SYNTHEVAL_PRESET.items() if k in selected}
    if target_is_binary is False:
        preset = {
            name: values for name, values in preset.items() if name not in BINARY_ONLY_METRICS
        }
    for name in FAIRNESS_METRICS_WITH_POSITIVE_CLASS & preset.keys():
        # Shallow-copy before overriding -- SYNTHEVAL_PRESET's nested dicts are
        # shared module-level objects reused on every call; mutating in place
        # would corrupt the global constant for subsequent calls in-process.
        preset[name] = {**preset[name], "positive_class": positive_class}
    return preset


def _evaluation_context_fingerprint(
    dataset: Dataset,
    preset: dict,
    pass_name: str,
    plots_enabled: bool,
    expected_output_manifest: dict | None = None,
    group_context: dict | None = None,
    fit_frame: pd.DataFrame | None = None,
    tuning_frame: pd.DataFrame | None = None,
    evaluation_role: str = "tuning",
    fit_roles: tuple[str, ...] = ("train",),
    target_view_context: Mapping[str, Any] | None = None,
    semantic_context: Mapping[str, Any] | None = None,
) -> str:
    """Fingerprint inputs shared by every model in one evaluation pass."""
    if fit_frame is None or tuning_frame is None:
        fit_frame, tuning_frame, _final_holdout_frame = _candidate_role_frames(dataset)
    role_context = _evaluation_role_context(
        dataset,
        fit_frame,
        tuning_frame,
        evaluation_role=evaluation_role,
        fit_roles=fit_roles,
        target_view_context=target_view_context,
        semantic_context=semantic_context,
    )
    payload = {
        "schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "pass_name": pass_name,
        "preset": preset,
        "dataset_name": dataset.name,
        "dataset_version": dataset.version,
        "target_column": dataset.target_column,
        "sensitive_columns": dataset.sensitive_columns,
        "categorical_columns": dataset.all_categorical_columns,
        "role_context": role_context,
        "plots_enabled": plots_enabled,
        "expected_output_manifest": expected_output_manifest,
        "group_context": group_context,
        "registry_digest": DEFAULT_METRIC_CONTRACT_REGISTRY.digest(),
        "preprocessing_contract": "syntheval-fit-role-v2",
        "target_view_context": dict(target_view_context or {}),
        "semantic_context": dict(semantic_context) if semantic_context is not None else None,
        "semantic_context_digest": (
            semantic_context_digest(semantic_context) if semantic_context is not None else None
        ),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _candidate_role_context(
    dataset: Dataset,
    fit_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
) -> dict:
    """Return the candidate-only role provenance used by one evaluation pass."""
    return _evaluation_role_context(dataset, fit_frame, tuning_frame, evaluation_role="tuning")


def _evaluation_role_context(
    dataset: Dataset,
    fit_frame: pd.DataFrame,
    evidence_frame: pd.DataFrame,
    *,
    evaluation_role: str,
    fit_roles: tuple[str, ...] = ("train",),
    target_view_context: Mapping[str, Any] | None = None,
    semantic_context: Mapping[str, Any] | None = None,
) -> dict:
    """Return role provenance for candidate or post-selection evidence."""
    if evaluation_role not in {"tuning", "final_holdout"}:
        raise ValueError(f"Unsupported evaluation role: {evaluation_role!r}")
    if not fit_roles:
        raise ValueError("Evaluation role context requires at least one fit role")
    context_roles = tuple(dict.fromkeys((*fit_roles, evaluation_role)))
    scoped_context = role_context_payload(dataset, context_roles)
    return {
        "schema_version": "evaluation-role-context-v1",
        "fit_role": fit_roles[0] if len(fit_roles) == 1 else None,
        "fit_roles": list(fit_roles),
        "evidence_role": evaluation_role,
        "fit_frame": _frame_fingerprint(fit_frame),
        "evidence_frame": _frame_fingerprint(evidence_frame),
        "assignment_fingerprint": scoped_context["assignment_fingerprint"],
        "assignment_policy_fingerprint": scoped_context["assignment_policy_fingerprint"],
        "semantic_fingerprint": dataset.semantic_fingerprint,
        "variable_schema_fingerprint": dataset.variable_schema_fingerprint,
        "compatibility_mode": "legacy_two_role" if dataset.legacy_two_role else None,
        "target_view_context": dict(target_view_context or {}),
        "semantic_context": (dict(semantic_context) if semantic_context is not None else None),
        "semantic_context_digest": (
            semantic_context_digest(semantic_context) if semantic_context is not None else None
        ),
    }


def _run_resumable_syntheval(
    synthetic_datasets: dict[str, pd.DataFrame],
    dataset: Dataset,
    preset: dict,
    preset_path: Path,
    output_folder: str | Path,
    ranking_strategy: str,
    execution_cfg,
    pass_name: str,
    plots_output_dir: str | Path | None = None,
    expected_output_manifest: dict | None = None,
    target_view: str = "native",
    expected_manifest_digest: str | None = None,
    group_context: dict | None = None,
    fit_frame: pd.DataFrame | None = None,
    tuning_frame: pd.DataFrame | None = None,
    evaluation_role: str = "tuning",
    fit_roles: tuple[str, ...] = ("train",),
    target_view_context: Mapping[str, Any] | None = None,
    semantic_context: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict]]:
    """Evaluate models in disposable bounded processes and resume checkpoints.

    Each child handles exactly one model. Its process exit releases all native
    and allocator high-water memory before the parent admits another model.
    Failed models remain in the returned structured execution map as explicit
    failure evidence; only fully successful payloads are reusable checkpoints.
    """
    checkpoint_root = ensure_dir(output_folder).resolve()
    plots_enabled = plots_output_dir is not None
    expected_output_manifest = expected_output_manifest or {}
    expected_manifest_digest = expected_manifest_digest or _execution_manifest_digest(
        expected_output_manifest
    )
    fit_frame, tuning_frame, _final_holdout_frame = (
        (fit_frame, tuning_frame, None)
        if fit_frame is not None and tuning_frame is not None
        else _candidate_role_frames(dataset)
    )
    context_fingerprint = _evaluation_context_fingerprint(
        dataset,
        preset,
        pass_name,
        plots_enabled,
        expected_output_manifest=expected_output_manifest,
        group_context=group_context,
        fit_frame=fit_frame,
        tuning_frame=tuning_frame,
        evaluation_role=evaluation_role,
        fit_roles=fit_roles,
        target_view_context=target_view_context,
        semantic_context=semantic_context,
    )
    role_context = _evaluation_role_context(
        dataset,
        fit_frame,
        tuning_frame,
        evaluation_role=evaluation_role,
        fit_roles=fit_roles,
        target_view_context=target_view_context,
        semantic_context=semantic_context,
    )
    results: dict[str, pd.DataFrame] = {}
    executions: dict[str, dict] = {}
    pending: list[tuple[str, pd.DataFrame, str]] = []
    for model_name, frame in synthetic_datasets.items():
        model_fingerprint = _frame_fingerprint(frame)
        cached = _valid_checkpoint(
            checkpoint_root,
            pass_name,
            model_name,
            context_fingerprint,
            model_fingerprint,
            plots_enabled,
            expected_manifest_digest=expected_manifest_digest,
            return_execution=True,
            expected_manifest=expected_output_manifest,
            expected_target_view=target_view,
        )
        if cached is None:
            pending.append((model_name, frame, model_fingerprint))
        else:
            logger.info("[syntheval] %s checkpoint hit for model %s", pass_name, model_name)
            results[model_name], executions[model_name] = cached

    if pending:
        workers = resolve_model_workers(
            execution_cfg,
            n_models=len(pending),
            n_columns=fit_frame.shape[1],
        )
        logger.info(
            "[syntheval] %s scheduling %d missing model(s) with %d disposable worker(s) "
            "(fit_role=train, tuning_role=tuning, fit=%s, tuning=%s, features=%d, plots=%s)",
            pass_name,
            len(pending),
            workers,
            fit_frame.shape,
            tuning_frame.shape,
            fit_frame.shape[1],
            plots_enabled,
        )
        context = multiprocessing.get_context("spawn")
        active: dict[str, multiprocessing.Process] = {}
        pending_iter = iter(pending)

        def start_next() -> bool:
            try:
                model_name, frame, model_fingerprint = next(pending_iter)
            except StopIteration:
                return False
            process = context.Process(
                target=_model_worker,
                args=(
                    model_name,
                    frame,
                    fit_frame,
                    tuning_frame,
                    dataset.all_categorical_columns,
                    dataset.target_column,
                    dataset.sensitive_columns,
                    str(preset_path.resolve()),
                    str(checkpoint_root),
                    pass_name,
                    expected_output_manifest,
                    target_view,
                    expected_manifest_digest,
                    context_fingerprint,
                    model_fingerprint,
                    str(plots_output_dir) if plots_output_dir else None,
                    execution_cfg.cores_per_model,
                    group_context,
                    role_context,
                    semantic_context,
                ),
                name=f"syntheval-{pass_name}-{model_name}",
            )
            process.start()
            active[model_name] = process
            logger.info(
                "[syntheval] %s started model=%s pid=%s", pass_name, model_name, process.pid
            )
            return True

        for _ in range(workers):
            if not start_next():
                break

        failures = []
        while active:
            completed = []
            for model_name, process in active.items():
                if process.is_alive():
                    continue
                process.join()
                completed.append((model_name, process.exitcode))
            if not completed:
                time.sleep(0.1)
                continue
            for model_name, exitcode in completed:
                del active[model_name]
                model_fingerprint = _frame_fingerprint(synthetic_datasets[model_name])
                cached = _valid_checkpoint(
                    checkpoint_root,
                    pass_name,
                    model_name,
                    context_fingerprint,
                    model_fingerprint,
                    plots_enabled,
                    expected_manifest_digest=expected_manifest_digest,
                    return_execution=True,
                    expected_manifest=expected_output_manifest,
                    expected_target_view=target_view,
                )
                if exitcode == 0 and cached is not None:
                    results[model_name], executions[model_name] = cached
                    logger.info("[syntheval] %s completed model=%s", pass_name, model_name)
                else:
                    model_dir, status_path, _ = _checkpoint_paths(
                        checkpoint_root, pass_name, model_name
                    )
                    ensure_dir(model_dir)
                    worker_status = {}
                    try:
                        loaded_status = json.loads(status_path.read_text())
                        if isinstance(loaded_status, dict):
                            worker_status = loaded_status
                    except (OSError, json.JSONDecodeError) as exc:
                        logger.warning(
                            "[syntheval] failed to read worker status for model=%s at %s: %s",
                            model_name,
                            status_path,
                            exc,
                        )
                    failure_reason = worker_status.get("exception") or worker_status.get(
                        "failure_reason"
                    )
                    if not isinstance(failure_reason, str) or not failure_reason:
                        failure_reason = (
                            f"worker exited without a valid succeeded checkpoint (exit={exitcode})"
                        )
                    failure_status = {
                        "schema_version": _CHECKPOINT_SCHEMA_VERSION,
                        "state": "failed",
                        "model_name": model_name,
                        "target_view": target_view,
                        "expected_manifest_digest": expected_manifest_digest,
                        "context_fingerprint": context_fingerprint,
                        "role_context": role_context,
                        "model_fingerprint": model_fingerprint,
                        "exit_code": exitcode,
                        "failed_at": time.time(),
                        "failure_reason": failure_reason,
                        "exception_type": worker_status.get("exception_type") or "WorkerExit",
                        "exception": failure_reason,
                        "traceback": worker_status.get("traceback"),
                    }
                    execution_path = _execution_checkpoint_path(model_dir)
                    existing_execution = None
                    try:
                        loaded_execution = json.loads(execution_path.read_text())
                        if isinstance(loaded_execution, dict):
                            existing_execution = loaded_execution
                    except (OSError, json.JSONDecodeError) as exc:
                        if execution_path.exists():
                            logger.warning(
                                "[syntheval] failed to read child execution evidence for model=%s "
                                "at %s: %s",
                                model_name,
                                execution_path,
                                exc,
                            )
                    _atomic_json(status_path, failure_status)
                    _atomic_json(
                        execution_path,
                        _failed_execution_payload(
                            model_name=model_name,
                            pass_name=pass_name,
                            target_view=target_view,
                            expected_manifest_digest=expected_manifest_digest,
                            expected_output_manifest=expected_output_manifest,
                            context_fingerprint=context_fingerprint,
                            role_context=role_context,
                            group_context=group_context,
                            semantic_context=semantic_context,
                            failure_status=failure_status,
                            existing_execution=existing_execution,
                        ),
                    )
                    executions[model_name] = json.loads(execution_path.read_text())
                    failures.append(f"{model_name} (exit={exitcode}, status={status_path})")
                    logger.error(
                        "[syntheval] %s model=%s exited %s; failure evidence persisted at %s",
                        pass_name,
                        model_name,
                        exitcode,
                        _execution_checkpoint_path(model_dir),
                    )
                start_next()
        if failures:
            logger.error(
                "[syntheval] %s returned partial results after %d failed model(s): %s. "
                "Completed model checkpoints remain resumable under %s",
                pass_name,
                len(failures),
                ", ".join(failures),
                checkpoint_root,
            )

    ordered_executions = {name: executions[name] for name in synthetic_datasets}
    benchmark_results, benchmark_ranks = build_syntheval_tables_from_executions(
        ordered_executions,
        list(synthetic_datasets),
        ranking_strategy,
    )
    return benchmark_results, benchmark_ranks, ordered_executions


def run_syntheval_evaluation(
    synthetic_datasets: dict[str, pd.DataFrame],
    dataset: Dataset,
    selection_cfg,
    preset_dir: str | Path,
    ranking_strategy: str = "linear",
    output_folder: str | Path | None = None,
    plots_output_dir: str | Path | None = None,
    positive_class=1,
    execution_cfg=None,
    return_execution: bool = False,
    group_context: dict | None = None,
    evaluation_role: str = "tuning",
    fit_frame: pd.DataFrame | None = None,
    fit_roles: tuple[str, ...] = ("train",),
    semantic_context: Mapping[str, Any] | None = None,
) -> (
    tuple[pd.DataFrame | None, pd.DataFrame | None]
    | tuple[pd.DataFrame | None, pd.DataFrame | None, dict[str, dict]]
):
    """Run SynthEval's benchmark() across all datasets.

    Returns ``(benchmark_results, benchmark_ranks)`` by default. When
    ``return_execution`` is true, appends the manifest-bound per-model
    structured execution payloads.

    Both are None if the selection resolves to zero metrics.

    If ``plots_output_dir`` is given, SynthEval's native per-metric plots (``SE_*.png``)
    are produced as a side effect of this same benchmark pass (one subfolder per model
    under ``plots_output_dir``), instead of requiring a separate, fully redundant
    benchmark/evaluate pass just to regenerate them.

    ``positive_class`` (from ``cfg.evaluation.positive_class``) is forwarded to
    :func:`build_preset` for the 3 fairness metrics -- see its docstring.
    """
    target_frame = dataset.role_frame("train", imputed=True)
    target_is_binary = (
        dataset.target_is_categorical
        and target_frame is not None
        and target_frame[dataset.target_column].nunique(dropna=False) == 2
    )
    preset = build_preset(
        selection_cfg,
        positive_class,
        target_is_binary=target_is_binary,
    )
    if not preset:
        logger.info("[syntheval] no metrics selected; skipping")
        return (None, None, {}) if return_execution else (None, None)

    default_fit_frame, tuning_frame = _evaluation_role_frames(dataset, evaluation_role)
    fit_frame = default_fit_frame if fit_frame is None else fit_frame
    preset_dir = ensure_dir(preset_dir)
    preset_filename = (
        "syntheval_preset.json"
        if evaluation_role == "tuning"
        else f"syntheval_{evaluation_role}_preset.json"
    )
    preset_path = preset_dir / preset_filename
    save_json(preset_path, preset)
    expected_output_manifest = syntheval_execution_manifest(
        preset,
        include_holdout_outputs=tuning_frame is not None,
        target_columns=[dataset.target_column],
        protected_columns=dataset.sensitive_columns,
    )
    expected_manifest_digest = _execution_manifest_digest(expected_output_manifest)

    cache_dir = (
        Path(output_folder)
        if output_folder
        else preset_dir
        / ("syntheval_benchmark" if evaluation_role == "tuning" else "syntheval_final_holdout")
    )
    context_fingerprint = _evaluation_context_fingerprint(
        dataset,
        preset,
        "main",
        plots_output_dir is not None,
        expected_output_manifest=expected_output_manifest,
        group_context=group_context,
        fit_frame=fit_frame,
        tuning_frame=tuning_frame,
        evaluation_role=evaluation_role,
        fit_roles=fit_roles,
        semantic_context=semantic_context,
    )
    role_context = _evaluation_role_context(
        dataset,
        fit_frame,
        tuning_frame,
        evaluation_role=evaluation_role,
        fit_roles=fit_roles,
        semantic_context=semantic_context,
    )
    model_fingerprints = {
        name: _frame_fingerprint(frame) for name, frame in synthetic_datasets.items()
    }
    cache_key = _compute_cache_key(
        preset,
        list(synthetic_datasets.keys()),
        ranking_strategy,
        hashlib.sha256(
            json.dumps(
                {"context": context_fingerprint, "models": model_fingerprints}, sort_keys=True
            ).encode()
        ).hexdigest(),
    )

    cached = _load_syntheval_cache(
        cache_dir,
        "main",
        cache_key,
        context_fingerprint=context_fingerprint,
        expected_manifest_digest=expected_manifest_digest,
        expected_target_view="native",
        registry_digest=DEFAULT_METRIC_CONTRACT_REGISTRY.digest(),
    )
    if cached is not None:
        executions = _load_syntheval_execution_sidecars(
            cache_dir,
            "main",
            list(synthetic_datasets),
            expected_manifest_digest,
            context_fingerprint,
            expected_manifest=expected_output_manifest,
            expected_target_view="native",
        )
        if executions is not None:
            validated_cached = _validated_cached_syntheval_tables(
                cached[0],
                cached[1],
                executions,
                list(synthetic_datasets),
                ranking_strategy,
                "main",
            )
            if validated_cached is not None:
                return (*validated_cached, executions) if return_execution else validated_cached
        logger.info(
            "[syntheval] main aggregate cache is present but structured execution evidence or "
            "tables are incomplete; recomputing model checkpoints"
        )

    if execution_cfg is None:
        from synthdata.config import SynthEvalExecutionConfig

        execution_cfg = SynthEvalExecutionConfig()
    benchmark_results, benchmark_ranks, executions = _run_resumable_syntheval(
        synthetic_datasets,
        dataset,
        preset,
        preset_path,
        cache_dir,
        ranking_strategy,
        execution_cfg,
        "main",
        plots_output_dir,
        expected_output_manifest=expected_output_manifest,
        target_view="native",
        expected_manifest_digest=expected_manifest_digest,
        group_context=group_context,
        fit_frame=fit_frame,
        tuning_frame=tuning_frame,
        evaluation_role=evaluation_role,
        fit_roles=fit_roles,
        semantic_context=semantic_context,
    )

    _save_syntheval_cache(
        benchmark_results,
        benchmark_ranks,
        cache_dir,
        "main",
        cache_key,
        context_fingerprint=context_fingerprint,
        role_context=role_context,
        expected_manifest_digest=expected_manifest_digest,
        target_view="native",
        registry_digest=DEFAULT_METRIC_CONTRACT_REGISTRY.digest(),
    )

    return (
        (benchmark_results, benchmark_ranks, executions)
        if return_execution
        else (benchmark_results, benchmark_ranks)
    )


def build_binary_target_series(
    series: pd.Series, positive_classes: list, negative_classes: list
) -> pd.Series:
    """Collapse a categorical Series to binary (1=positive_classes, 0=negative_classes).

    Fails loudly (rather than silently coercing to NaN) if any observed
    non-null value isn't covered by either list -- an uncovered value would
    otherwise silently vanish as an unexplained NaN in a supposedly
    fully-observed target column. Also fails loudly if the input has any
    missing values at all: the result is returned as ``int64`` (not float),
    which can't represent NaN -- and that's not merely a dtype nicety: only
    ``object``/``int`` dtypes get treated as *categorical* by SynthEval's
    ``AnalysisConfig`` (anything else, e.g. float, is classified as "num"
    i.e. continuous -- see ``syntheval.utils.configuration.AnalysisConfig``),
    so a float output here would silently make auroc_diff/statistical_parity/
    equalized_odds/equal_opportunity go right back to refusing to run, in a
    much harder-to-diagnose way than an explicit error at build time.
    """
    positive_set, negative_set = set(positive_classes), set(negative_classes)
    observed = set(series.dropna().unique().tolist())
    unmapped = observed - positive_set - negative_set
    if unmapped:
        raise ValueError(
            f"Observed value(s) {sorted(unmapped, key=str)} in column {series.name!r} are not "
            "covered by evaluation.binary_target.positive_classes/negative_classes "
            f"(positive={sorted(positive_classes, key=str)}, negative={sorted(negative_classes, key=str)})"
            " -- every observed value must be listed in one of the two."
        )
    if series.isna().any():
        raise ValueError(
            f"Column {series.name!r} has missing value(s) -- evaluation.binary_target requires a "
            "fully-observed target column (the main evaluation pipeline already drops rows with a "
            "missing target before this point; if this column has genuine missingness, it isn't a "
            "valid evaluation.binary_target.column)."
        )
    return pd.Series(
        np.where(series.isin(positive_set), 1, 0),
        index=series.index,
        name=series.name,
        dtype="int64",
    )


def build_binary_preset(selection_cfg) -> dict:
    """Filter SYNTHEVAL_PRESET down to just the exactly-2-classes-only metrics
    (BINARY_ONLY_METRICS), further filtered by the same selection_cfg used for
    the main preset -- e.g. disabling the 'fairness' category or explicitly
    excluding 'auroc_diff' via evaluation.syntheval.metrics also excludes it
    from this binary-target pass.

    Deliberately does NOT take a ``positive_class`` override (unlike
    build_preset): build_binary_target_series always normalizes the collapsed
    target to 1=positive_classes/0=negative_classes, so "positive_class" must
    stay SYNTHEVAL_PRESET's default of 1 here regardless of
    ``cfg.evaluation.positive_class`` -- threading it through would
    double-remap an already-fixed convention.
    """
    all_names = list(SYNTHEVAL_PRESET.keys())
    selected = resolve_selection(
        selection_cfg.enabled,
        selection_cfg.categories,
        selection_cfg.metrics,
        all_names,
        SYNTHEVAL_METRIC_TYPE,
    )
    return {k: v for k, v in SYNTHEVAL_PRESET.items() if k in selected and k in BINARY_ONLY_METRICS}


def run_binary_target_syntheval_evaluation(
    synthetic_datasets: dict[str, pd.DataFrame],
    dataset: Dataset,
    selection_cfg,
    binary_target_cfg,
    preset_dir: str | Path,
    ranking_strategy: str = "linear",
    output_folder: str | Path | None = None,
    execution_cfg=None,
    return_execution: bool = False,
    group_context: dict | None = None,
    evaluation_role: str = "tuning",
    fit_frame: pd.DataFrame | None = None,
    fit_roles: tuple[str, ...] = ("train",),
    semantic_context: Mapping[str, Any] | None = None,
) -> (
    tuple[pd.DataFrame | None, pd.DataFrame | None]
    | tuple[pd.DataFrame | None, pd.DataFrame | None, dict[str, dict]]
):
    """Run a second, separate SynthEval benchmark() pass against a binary-
    collapsed copy of the target column, for the metrics that require exactly
    2 target classes (BINARY_ONLY_METRICS) and would otherwise be unable to
    run at all against a 3+ class target.

    This never touches ``dataset.train_imputed_df``/``test_imputed_df``/the
    caller's ``synthetic_datasets`` -- only disposable copies, with the target
    column's values (not its name -- see build_binary_target_series) replaced
    in place, so it's still correctly excluded from the model's own feature
    set exactly like the original target (no leakage from the original,
    finer-grained labels lingering as a feature).

    Returns (benchmark_results, benchmark_ranks), both None if no
    BINARY_ONLY_METRICS are selected. When ``return_execution`` is true, the
    structured per-model execution payloads are appended.
    """
    preset = build_binary_preset(selection_cfg)
    if not preset:
        logger.info("[syntheval] binary-target pass: no eligible metrics selected; skipping")
        return (None, None, {}) if return_execution else (None, None)

    column = binary_target_cfg.column or dataset.target_column
    positive_classes = binary_target_cfg.positive_classes
    negative_classes = binary_target_cfg.negative_classes

    def _binarize(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[column] = build_binary_target_series(out[column], positive_classes, negative_classes)
        return out

    canonical_fit_frame, tuning_frame, final_holdout_frame = _candidate_role_frames(dataset)
    if evaluation_role == "tuning":
        evidence_frame = tuning_frame
    elif evaluation_role == "final_holdout":
        dataset.require_canonical_roles("final-holdout binary evaluation")
        evidence_frame = final_holdout_frame
    else:
        raise ValueError(
            f"Unsupported SynthEval evaluation_role {evaluation_role!r}; "
            "expected 'tuning' or 'final_holdout'"
        )
    fit_frame = canonical_fit_frame if fit_frame is None else fit_frame
    binary_fit_frame = _binarize(fit_frame)
    binary_tuning_frame = _binarize(tuning_frame)
    binary_final_holdout_frame = _binarize(final_holdout_frame)
    binary_evidence_frame = _binarize(evidence_frame)
    binary_synthetic_datasets = {name: _binarize(df) for name, df in synthetic_datasets.items()}
    binary_target_context = {
        "column": column,
        "positive_classes": list(positive_classes),
        "negative_classes": list(negative_classes),
        "encoding": {"positive": 1, "negative": 0},
    }
    binary_semantic_context = (
        {
            **dict(semantic_context),
            "target_view": "binary_collapsed",
            "binary_target_context": binary_target_context,
        }
        if semantic_context is not None
        else None
    )

    preset_dir = ensure_dir(preset_dir)
    preset_filename = (
        "syntheval_binary_target_preset.json"
        if evaluation_role == "tuning"
        else f"syntheval_{evaluation_role}_binary_target_preset.json"
    )
    preset_path = preset_dir / preset_filename
    save_json(preset_path, preset)
    binary_schema = {
        **dataset.variable_schema,
        column: {**dataset.variable_schema.get(column, {}), "kind": "categorical"},
    }
    binary_imputed_roles = {
        "train": binary_fit_frame,
        "tuning": binary_tuning_frame,
        "final_holdout": binary_final_holdout_frame,
    }
    binary_dataset = dataclasses.replace(
        dataset,
        target_column=column,
        variable_schema=binary_schema,
        imputed_roles=(
            {"train": binary_fit_frame, "final_holdout": binary_final_holdout_frame}
            if dataset.legacy_two_role
            else binary_imputed_roles
        ),
        train_imputed_df=binary_fit_frame if dataset.legacy_two_role else None,
        test_imputed_df=binary_final_holdout_frame if dataset.legacy_two_role else None,
    )
    expected_output_manifest = syntheval_execution_manifest(
        preset,
        include_holdout_outputs=evidence_frame is not None,
        target_columns=[binary_dataset.target_column],
        protected_columns=binary_dataset.sensitive_columns,
    )
    expected_manifest_digest = _execution_manifest_digest(expected_output_manifest)
    cache_dir = (
        Path(output_folder)
        if output_folder
        else preset_dir
        / ("syntheval_benchmark" if evaluation_role == "tuning" else "syntheval_final_holdout")
    )
    context_fingerprint = _evaluation_context_fingerprint(
        binary_dataset,
        preset,
        "binary_target",
        False,
        expected_output_manifest=expected_output_manifest,
        group_context=group_context,
        fit_frame=binary_fit_frame,
        tuning_frame=binary_evidence_frame,
        evaluation_role=evaluation_role,
        fit_roles=fit_roles,
        target_view_context=binary_target_context,
        semantic_context=binary_semantic_context,
    )
    role_context = _evaluation_role_context(
        binary_dataset,
        binary_fit_frame,
        binary_evidence_frame,
        evaluation_role=evaluation_role,
        fit_roles=fit_roles,
        target_view_context=binary_target_context,
        semantic_context=binary_semantic_context,
    )
    model_fingerprints = {
        name: _frame_fingerprint(frame) for name, frame in binary_synthetic_datasets.items()
    }
    cache_key = _compute_cache_key(
        preset,
        list(binary_synthetic_datasets.keys()),
        ranking_strategy,
        hashlib.sha256(
            json.dumps(
                {"context": context_fingerprint, "models": model_fingerprints}, sort_keys=True
            ).encode()
        ).hexdigest(),
    )
    cached = _load_syntheval_cache(
        cache_dir,
        "binary_target",
        cache_key,
        context_fingerprint=context_fingerprint,
        expected_manifest_digest=expected_manifest_digest,
        expected_target_view="binary_collapsed",
        registry_digest=DEFAULT_METRIC_CONTRACT_REGISTRY.digest(),
    )
    if cached is not None:
        executions = _load_syntheval_execution_sidecars(
            cache_dir,
            "binary_target",
            list(binary_synthetic_datasets),
            expected_manifest_digest,
            context_fingerprint,
            expected_manifest=expected_output_manifest,
            expected_target_view="binary_collapsed",
        )
        if executions is not None:
            validated_cached = _validated_cached_syntheval_tables(
                cached[0],
                cached[1],
                executions,
                list(binary_synthetic_datasets),
                ranking_strategy,
                "binary-target",
            )
            if validated_cached is not None:
                return (*validated_cached, executions) if return_execution else validated_cached
        logger.info(
            "[syntheval] binary-target aggregate cache is present but structured execution "
            "evidence or tables are incomplete; recomputing model checkpoints"
        )

    if execution_cfg is None:
        from synthdata.config import SynthEvalExecutionConfig

        execution_cfg = SynthEvalExecutionConfig()
    logger.info(
        "[syntheval] binary-target pass: scheduling %d datasets across %d metric(s) "
        "(column %r collapsed to binary: positive=%s, negative=%s)",
        len(binary_synthetic_datasets),
        len(preset),
        column,
        positive_classes,
        negative_classes,
    )
    benchmark_results, benchmark_ranks, executions = _run_resumable_syntheval(
        binary_synthetic_datasets,
        binary_dataset,
        preset,
        preset_path,
        cache_dir,
        ranking_strategy,
        execution_cfg,
        "binary_target",
        expected_output_manifest=expected_output_manifest,
        target_view="binary_collapsed",
        expected_manifest_digest=expected_manifest_digest,
        group_context=group_context,
        fit_frame=binary_fit_frame,
        tuning_frame=binary_evidence_frame,
        evaluation_role=evaluation_role,
        fit_roles=fit_roles,
        semantic_context=binary_semantic_context,
    )
    _save_syntheval_cache(
        benchmark_results,
        benchmark_ranks,
        cache_dir,
        "binary_target",
        cache_key,
        context_fingerprint=context_fingerprint,
        role_context=role_context,
        expected_manifest_digest=expected_manifest_digest,
        target_view="binary_collapsed",
        registry_digest=DEFAULT_METRIC_CONTRACT_REGISTRY.digest(),
    )

    return (
        (benchmark_results, benchmark_ranks, executions)
        if return_execution
        else (benchmark_results, benchmark_ranks)
    )


def merge_binary_target_results(
    benchmark_results: pd.DataFrame | None,
    benchmark_ranks: pd.DataFrame | None,
    binary_results: pd.DataFrame | None,
    binary_ranks: pd.DataFrame | None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Merge a binary-target-only pass's per-metric columns into the main
    pass's results.

    Only metric-value columns are merged, never the aggregate rank/u_rank/
    p_rank/f_rank columns from either pass -- those are always recomputed
    from scratch downstream in combine.build_combined_table purely from the
    per-metric oriented values (see extract_oriented_values), so a mini
    (BINARY_ONLY_METRICS-sized) benchmark's own internal aggregate ranks
    would be meaningless to propagate.
    """
    if binary_results is None:
        return benchmark_results, benchmark_ranks
    if benchmark_results is None:
        return binary_results, binary_ranks

    main_metrics = set(benchmark_results.columns.get_level_values(0)) - _RANK_COLUMNS
    new_metrics = [
        metric
        for metric in binary_results.columns.get_level_values(0).unique()
        if metric not in _RANK_COLUMNS and metric not in main_metrics
    ]
    skipped_metrics = [
        metric
        for metric in binary_results.columns.get_level_values(0).unique()
        if metric in main_metrics
    ]
    if skipped_metrics:
        logger.info(
            "[syntheval] preserving native values for binary-pass metric collision(s): %s",
            skipped_metrics,
        )
    for metric in new_metrics:
        benchmark_results[(metric, "value")] = binary_results[(metric, "value")]
        benchmark_results[(metric, "error")] = binary_results[(metric, "error")]
        benchmark_ranks[metric] = binary_ranks[metric]
    return benchmark_results, benchmark_ranks


def extract_raw_values(benchmark_results: pd.DataFrame) -> pd.DataFrame:
    """Models x metrics table of raw metric values from SynthEval's benchmark_results."""
    return benchmark_results.xs("value", axis=1, level=1)


def extract_oriented_values(benchmark_ranks: pd.DataFrame) -> pd.DataFrame:
    """Models x metrics table of SynthEval's pre-oriented (higher=better) n_val scores."""
    cols = [c for c in benchmark_ranks.columns if c not in _RANK_COLUMNS]
    return benchmark_ranks[cols]


def _safe_metric_table(
    benchmark_results: pd.DataFrame | None,
    level: str,
) -> pd.DataFrame:
    """Extract one benchmark-results level, returning an empty table if absent."""
    if benchmark_results is None or not isinstance(benchmark_results.columns, pd.MultiIndex):
        return pd.DataFrame()
    try:
        return benchmark_results.xs(level, axis=1, level=1)
    except (KeyError, IndexError):
        return pd.DataFrame(index=benchmark_results.index)


def _scalar_value(frame: pd.DataFrame, model_name: str, metric: str):
    if model_name not in frame.index or metric not in frame.columns:
        return None
    value = frame.loc[model_name, metric]
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        return value
    return value


def _structured_rows_for_item(item: dict) -> list[dict]:
    status = item.get("status") or {}
    expected_keys = [str(key) for key in status.get("expected_keys", ())]
    versioned = any("_v2" in key for key in expected_keys)
    field = "normalized_rows_v2" if versioned else "normalized_rows"
    return list(item.get(field) or ())


def _structured_rows_for_payload(payload: dict) -> list[dict]:
    rows = []
    for item in payload.get("metric_executions", ()):
        rows.extend(_structured_rows_for_item(item))
    return rows


def _structured_row_field(row: Mapping, field: str | None):
    if field is None:
        return None
    value = row
    for component in field.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(component)
    return value


def _structured_uncertainty(row: Mapping, field: str | None) -> float | None:
    value = _structured_row_field(row, field)
    if value is None or isinstance(value, bool) or not isinstance(value, Real):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _structured_sample_size(row: Mapping, field: str | None) -> int | None:
    value = _structured_row_field(row, field)
    if value is None or isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        return None
    converted = int(value)
    return converted if converted >= 0 else None


def _structured_metric_contract(emitted_key: str, execution_pass: str):
    framework = syntheval_framework_for_emitted_key(emitted_key)
    try:
        return DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
            framework=framework,
            emitted_key=emitted_key,
            execution_pass=execution_pass,
        )
    except (UnknownMetricContractError, AmbiguousMetricContractError):
        return None


def build_syntheval_tables_from_executions(
    executions: dict[str, dict],
    model_names: list[str],
    ranking_strategy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build benchmark tables from manifest-bound structured execution rows.

    Corrected versioned rows are selected by ``_structured_rows_for_item``;
    legacy rows remain available for methods that do not yet expose a v2
    normalizer. The legacy parquet checkpoint is therefore never the source of
    new root evaluation values.
    """
    from syntheval.syntheval import aggregate_benchmark_results

    if not executions:
        raise ValueError("Cannot build SynthEval tables without execution payloads")
    missing_models = [model_name for model_name in model_names if model_name not in executions]
    if missing_models:
        raise ValueError(f"Missing SynthEval execution payloads for models: {missing_models}")

    rows_by_model = {}
    metric_keys = []
    metric_dimensions = {}
    expected_metric_keys = []
    for model_name in model_names:
        rows = _structured_rows_for_payload(executions[model_name])
        by_metric = {}
        for row in rows:
            emitted_key = row.get("metric")
            if emitted_key is None:
                raise ValueError(
                    f"SynthEval model {model_name!r} emitted a row without a metric key"
                )
            emitted_key = str(emitted_key)
            if emitted_key in by_metric:
                raise ValueError(
                    f"SynthEval model {model_name!r} emitted duplicate structured key "
                    f"{emitted_key!r}"
                )
            by_metric[emitted_key] = row
            if emitted_key not in metric_dimensions:
                metric_dimensions[emitted_key] = row.get("dim", "u")
                metric_keys.append(emitted_key)
        for item in executions[model_name].get("metric_executions", ()):
            status = item.get("status") or {}
            for expected_key in status.get("expected_keys", ()):
                expected_key = str(expected_key)
                if expected_key not in expected_metric_keys:
                    expected_metric_keys.append(expected_key)
        rows_by_model[model_name] = by_metric

    for expected_key in expected_metric_keys:
        if expected_key not in metric_dimensions:
            metric_dimensions[expected_key] = "u"
        if expected_key not in metric_keys:
            metric_keys.append(expected_key)
    if not metric_keys:
        raise ValueError("SynthEval execution payloads contain no structured metric rows")

    benchmark_frames = {}
    for model_name in model_names:
        records = []
        for emitted_key in metric_keys:
            row = rows_by_model[model_name].get(emitted_key)
            if row is None:
                records.append(
                    {
                        "metric": emitted_key,
                        "dim": metric_dimensions[emitted_key],
                        "val": np.nan,
                        "err": np.nan,
                        "n_val": np.nan,
                        "n_err": np.nan,
                    }
                )
                continue
            raw_value = row.get("raw_value")
            normalized_value = row.get("normalized_value")
            records.append(
                {
                    "metric": emitted_key,
                    "dim": row.get("dim", metric_dimensions[emitted_key]),
                    "val": row.get("val") if raw_value is None else raw_value,
                    "err": row.get("err"),
                    "n_val": row.get("n_val") if normalized_value is None else normalized_value,
                    "n_err": row.get("n_err"),
                }
            )
        benchmark_frames[model_name] = pd.DataFrame.from_records(
            records,
            columns=["metric", "dim", "val", "err", "n_val", "n_err"],
        )
    return aggregate_benchmark_results(benchmark_frames, ranking_strategy)


def _validated_cached_syntheval_tables(
    cached_results: pd.DataFrame,
    cached_ranks: pd.DataFrame,
    executions: dict[str, dict],
    model_names: list[str],
    ranking_strategy: str,
    cache_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Rebuild cached tables and accept Parquet only when it agrees exactly."""
    try:
        rebuilt_results, rebuilt_ranks = build_syntheval_tables_from_executions(
            executions,
            model_names,
            ranking_strategy,
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning(
            "[syntheval] %s aggregate cache could not be rebuilt from structured executions "
            "(%s); treating cache as incomplete",
            cache_label,
            exc,
        )
        return None

    try:
        pd.testing.assert_frame_equal(cached_results, rebuilt_results)
        pd.testing.assert_frame_equal(cached_ranks, rebuilt_ranks)
    except AssertionError as exc:
        logger.warning(
            "[syntheval] %s aggregate cache differs from structured execution tables (%s); "
            "treating cache as incomplete",
            cache_label,
            exc,
        )
        return None
    return rebuilt_results, rebuilt_ranks


def _structured_observations(
    executions: dict[str, dict],
    *,
    role_hashes: dict[str, str],
) -> dict[str, list[MetricObservation]]:
    """Project durable fork execution payloads into root contract observations."""
    observations = {model_name: [] for model_name in executions}
    for model_name, payload in executions.items():
        execution_pass = payload.get("pass_id", "main")
        target_view = payload.get("target_view", "native")
        for item in payload.get("metric_executions", ()):
            status = item.get("status") or {}
            failed_keys = {str(key) for key in status.get("failed_keys", ())}
            observed_keys = set()
            rows = _structured_rows_for_item(item)
            for row in rows:
                emitted_key = str(row.get("metric"))
                if emitted_key == "None":
                    continue
                observed_keys.add(emitted_key)
                contract = _structured_metric_contract(emitted_key, execution_pass)
                normalized_value = row.get("normalized_value", row.get("n_val"))
                metadata = _result_metadata_payload(row)
                uncertainty_field = contract.uncertainty_field if contract else None
                sample_size_field = contract.sample_size_field if contract else None
                uncertainty = _structured_uncertainty(row, uncertainty_field)
                sample_size = _structured_sample_size(row, sample_size_field)
                error = None
                if emitted_key in failed_keys:
                    error = status.get("exception_message") or (
                        f"SynthEval method {item.get('method', '<unknown>')} reported "
                        f"terminal state {status.get('state', 'failed')}"
                    )
                observations[model_name].append(
                    MetricObservation(
                        model_name=model_name,
                        framework=syntheval_framework_for_emitted_key(emitted_key),
                        emitted_key=emitted_key,
                        raw_value=row.get("raw_value", row.get("val")),
                        execution_pass=execution_pass,
                        target_view=target_view,
                        uncertainty=uncertainty,
                        sample_size=sample_size,
                        error=error,
                        role_hashes=role_hashes,
                        source_metadata={
                            "method": item.get("method"),
                            "normalized_value": normalized_value,
                            "normalized_value_present": normalized_value is not None,
                            "metric_version": row.get("metric_version", "legacy"),
                            "metadata": metadata,
                            "result_metadata": metadata,
                            "uncertainty_field": uncertainty_field,
                            "sample_size_field": sample_size_field,
                            "execution_state": status.get("state"),
                        },
                        result_metadata=metadata,
                    )
                )
            for failed_key in failed_keys - observed_keys:
                observations[model_name].append(
                    MetricObservation(
                        model_name=model_name,
                        framework=syntheval_framework_for_emitted_key(failed_key),
                        emitted_key=failed_key,
                        raw_value=None,
                        execution_pass=execution_pass,
                        target_view=target_view,
                        error=status.get("exception_message")
                        or f"SynthEval method {item.get('method', '<unknown>')} reported "
                        f"terminal state {status.get('state', 'failed')}",
                        role_hashes=role_hashes,
                        source_metadata={
                            "method": item.get("method"),
                            "execution_state": status.get("state"),
                            "result_metadata": {},
                        },
                        result_metadata={},
                    )
                )
    return observations


def extend_syntheval_expected_diagnostics(
    expected_keys_by_framework: dict[str, list[str]],
    benchmark_results: pd.DataFrame | None,
    structured_executions: dict[str, dict] | None = None,
) -> dict[str, list[str]]:
    """Return static expectations while retaining observed diagnostics as audit extras.

    Qualified target, subgroup, and per-class rows are data-dependent and cannot
    become required merely because one run emitted them. The validation layer
    still records those observations as explicit non-expected audit records.
    """
    return {
        framework: list(dict.fromkeys(keys))
        for framework, keys in expected_keys_by_framework.items()
    }


def validate_syntheval_results(
    benchmark_results: pd.DataFrame | None,
    benchmark_ranks: pd.DataFrame | None,
    expected_keys_by_framework: dict[str, list[str]],
    *,
    role_hashes: dict[str, str],
    model_names: list[str],
    execution_pass: str = "main",
    target_view: str = "native",
    evaluation_role: str = "tuning",
    requested_use: str = "policy_rank",
    population_unit: str = "row",
    group_mode: str = "row",
    structured_executions: dict[str, dict] | None = None,
    resolved_configuration: dict | None = None,
) -> dict[tuple[str, str], dict[str, MetricValidationResult]]:
    """Validate normalized SynthEval rows and retain raw/rank provenance.

    The returned mapping is keyed by ``(framework, execution_pass)`` because
    the binary-target pass intentionally reuses emitted names such as
    ``auroc`` with a different target view.
    """
    raw = _safe_metric_table(benchmark_results, "value")
    errors = _safe_metric_table(benchmark_results, "error")
    oriented = (
        extract_oriented_values(benchmark_ranks) if benchmark_ranks is not None else pd.DataFrame()
    )
    observations_by_framework: dict[str, dict[str, list[MetricObservation]]] = {
        framework: {model_name: [] for model_name in model_names}
        for framework in expected_keys_by_framework
    }
    structured_by_model = (
        _structured_observations(structured_executions, role_hashes=role_hashes)
        if structured_executions is not None
        else None
    )

    for model_name in model_names:
        if structured_by_model is not None:
            for observation in structured_by_model.get(model_name, ()):
                observations_by_framework.setdefault(observation.framework, {}).setdefault(
                    model_name, []
                ).append(observation)
            continue
        for emitted_key in map(str, raw.columns):
            framework = syntheval_framework_for_emitted_key(emitted_key)
            if framework not in observations_by_framework:
                observations_by_framework[framework] = {name: [] for name in model_names}
            error_value = _scalar_value(errors, model_name, emitted_key)
            error = error_value if isinstance(error_value, str) else None
            uncertainty = error_value if isinstance(error_value, (int, float)) else None
            rank_value = _scalar_value(oriented, model_name, emitted_key)
            observations_by_framework[framework][model_name].append(
                MetricObservation(
                    model_name=model_name,
                    framework=framework,
                    emitted_key=emitted_key,
                    raw_value=_scalar_value(raw, model_name, emitted_key),
                    execution_pass=execution_pass,
                    target_view=target_view,
                    uncertainty=float(uncertainty) if uncertainty is not None else None,
                    error=error,
                    role_hashes=role_hashes,
                    source_metadata={
                        "normalized_value": rank_value,
                        "normalized_value_present": rank_value is not None,
                        "execution_pass": execution_pass,
                        "result_metadata": {},
                    },
                    result_metadata={},
                )
            )

    context = MetricEvaluationContext(
        execution_pass=execution_pass,
        target_view=target_view,
        evaluation_role=evaluation_role,
        population_unit=population_unit,
        group_mode=group_mode,
        role_hashes=role_hashes,
        resolved_configuration=resolved_configuration or {},
    )
    validations: dict[tuple[str, str], dict[str, MetricValidationResult]] = {}
    for framework, expected_keys in expected_keys_by_framework.items():
        if not expected_keys:
            continue
        validations[(framework, execution_pass)] = {
            model_name: resolve_metric_observations(
                registry=DEFAULT_METRIC_CONTRACT_REGISTRY,
                model_name=model_name,
                framework=framework,
                expected_keys=expected_keys,
                observations=observations_by_framework.get(framework, {}).get(model_name, []),
                context=context,
                requested_use=requested_use,
            )
            for model_name in model_names
        }
    return validations


def build_metric_execution_passes(
    validations: dict[tuple[str, str], dict[str, MetricValidationResult]],
) -> dict[tuple[str, str, str], str]:
    """Resolve each model/emitted-key pair to its first completed pass.

    The main pass is inserted before the binary-target pass. Main therefore
    wins when it emitted a repeated key such as ``auroc``; when it declared
    the key but that model's observations are missing or failed, a successful
    binary-target pass owns the merged value instead. Ownership is recorded
    per model because different models can complete different passes.
    """
    declarations: dict[tuple[str, str, str], list[tuple[str, bool]]] = {}
    for (framework, execution_pass), model_validations in validations.items():
        for model_name, validation in model_validations.items():
            completed_keys = set(validation.completed_keys)
            for emitted_key in validation.expected_keys:
                declarations.setdefault((framework, emitted_key, model_name), []).append(
                    (execution_pass, emitted_key in completed_keys)
                )

    execution_passes: dict[tuple[str, str, str], str] = {}
    for emitted_key, candidates in declarations.items():
        completed_passes = [execution_pass for execution_pass, completed in candidates if completed]
        execution_passes[emitted_key] = (
            completed_passes[0] if completed_passes else candidates[0][0]
        )
    return execution_passes
