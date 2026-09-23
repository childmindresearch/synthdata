"""Generic Optuna study management shared by all generation backends.

Provides:
- ``hpo_score``: the direction-aware composite objective used throughout the
  hepatitis notebooks (orient every metric so higher = better, then average).
- ``build_synthetic_eval_fn``: scores an arbitrary candidate synthetic DataFrame
  via synthcity's ``Metrics.evaluate`` (used as the HPO objective for generators,
  like TabPFGen, that don't go through synthcity's ``Benchmarks``).
- ``create_study``/``run_study``: Optuna study creation with SQLite-backed
  persistence (resumable across runs, inspectable with optuna-dashboard).
- ``BestParamsCache``: JSON-backed cache of best hyperparameters per model,
  keyed by generator family (``synthcity`` / ``tabpfgen``), mirroring
  ``output/hepatitis/hpo_best_params.json`` from the notebooks.
"""

import dataclasses
import enum
import errno
import hashlib
import json
import math
import os
import re
import stat
import unicodedata
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from numbers import Real
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised only on non-POSIX systems
    fcntl = None  # type: ignore[assignment]

import numpy as np
import optuna
import pandas as pd

from synthdata.config import HPOConfig
from synthdata.data import dataframe_fingerprint
from synthdata.evaluation.catalog import CANONICAL_HPO_ALLOWLIST
from synthdata.evaluation.metric_contracts import (
    DEFAULT_METRIC_CONTRACT_REGISTRY,
    MetricContractError,
)
from synthdata.utils import ensure_dir, get_logger, load_json, save_json

logger = get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


STAGE_A_SCREEN_SCHEMA_VERSION = "hpo-stage-a-v1"
HPO_CONTEXT_SCHEMA_VERSION = "hpo-context-v2"
HPO_TRIAL_CHECKPOINT_SCHEMA_VERSION = "hpo-trial-checkpoint-v1"
LEGACY_GENERATOR_METADATA_SCHEMA_VERSION = "generator-metadata-v1"
HPO_GENERATOR_METADATA_SCHEMA_VERSION = "generator-metadata-v2"
# Production datasets can be substantially wider than the historical unit-test
# fixtures, but checkpoint metadata must still have a bounded schema surface.
_MAX_STAGE_A_RESULT_COLUMNS = 4096
_MAX_STAGE_A_CHECK_BYTES = 131072
_MAX_STAGE_A_REASON_LENGTH = 4096
_STAGE_A_REASON_TRUNCATION_SUFFIX = " ... [truncated]"
HPO_METADATA_SERIALIZATION_REASON_CODE = "hpo_metadata_serialization_failure"
STAGE_A_SCREEN_IDS = (
    "shape_schema",
    "bounds_categories",
    "support",
    "dependencies",
    "exact_reuse",
    "subgroup_collapse",
)

_SAFE_EXCEPTION_MESSAGES = {
    "stage_a_screen_exception": "Stage A screen failed; exception details suppressed.",
    "metric_evaluation_exception": "Metric evaluation failed; exception details suppressed.",
    "hpo_trial_exception": "HPO trial failed; exception details suppressed.",
    "hpo_metric_not_eligible": "HPO metric report was not eligible; exception details suppressed.",
    HPO_METADATA_SERIALIZATION_REASON_CODE: "HPO metadata serialization failed; exception details suppressed.",
    "group_unsafe": "Grouped evaluation was unsafe; exception details suppressed.",
}
_SAFE_ERROR_LOCATION = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")
_SAFE_ERROR_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")


def _safe_exception_message(reason_code: str) -> str:
    return _SAFE_EXCEPTION_MESSAGES.get(
        reason_code, _SAFE_EXCEPTION_MESSAGES["hpo_trial_exception"]
    )


def _bound_stage_a_reason(reason: str) -> str:
    """Bound human-readable Stage-A evidence without changing structured evidence."""
    if len(reason) <= _MAX_STAGE_A_REASON_LENGTH:
        return reason
    prefix_length = _MAX_STAGE_A_REASON_LENGTH - len(_STAGE_A_REASON_TRUNCATION_SUFFIX)
    return reason[:prefix_length] + _STAGE_A_REASON_TRUNCATION_SUFFIX


class HPOMetricNotEligibleError(ValueError):
    """Raised when canonical metric evidence cannot produce an HPO score."""

    reason_code = "hpo_metric_not_eligible"


class HPOMetricEvaluationError(ValueError):
    """Raised when canonical metric computation emits failed evidence."""

    reason_code = "metric_evaluation_exception"


class HPOGroupUnsafeError(ValueError):
    """Raised when grouped evaluation cannot safely produce HPO evidence."""

    reason_code = "group_unsafe"


_CANONICAL_HPO_METRIC_KEYS = frozenset({"tstr_macro_f1.v1", "mixed_mmd.v1", "elastic_net_jsd.v1"})
_SAFE_METRIC_METADATA_FIELDS = frozenset(
    {
        "metric_name",
        "status",
        "direction",
        "finite",
        "eligible",
        "error_reason_code",
        "fit_roles",
        "evaluation_role",
    }
)
_REQUIRED_CANONICAL_RAW_METRIC_FIELDS = frozenset(
    {
        "metric_name",
        "status",
        "direction",
        "mean",
        "errors",
        "error_reason_code",
        "fit_roles",
        "evaluation_role",
    }
)
_CANONICAL_METRIC_ERROR_CODES = frozenset(
    {None, "metric_evaluation_exception", "hpo_metric_not_eligible", "hpo_trial_exception"}
)


def is_canonical_hpo_context(
    metric_config: Mapping[str, Sequence[str]] | None,
    *,
    expected_keys: Sequence[str] | None = None,
    utility_policy: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether resolved HPO identities exactly match canonical utility."""
    if not isinstance(metric_config, Mapping):
        return False
    configured_keys = [
        str(metric_name)
        for metric_names in metric_config.values()
        if isinstance(metric_names, (list, tuple))
        for metric_name in metric_names
    ]
    try:
        policy = _resolve_utility_policy(utility_policy)
    except (TypeError, ValueError):
        return False
    resolved_expected = (
        list(expected_keys) if expected_keys is not None else list(policy["metrics"])
    )
    return (
        len(configured_keys) == len(set(configured_keys))
        and set(configured_keys) == _CANONICAL_HPO_METRIC_KEYS
        and resolved_expected == list(policy["metrics"])
        and list(policy["metrics"])
        == [
            "tstr_macro_f1.v1",
            "mixed_mmd.v1",
            "elastic_net_jsd.v1",
        ]
    )


def sanitize_hpo_metric_metadata(
    value: Any, *, allowed_keys: Sequence[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Keep bounded per-metric status and role provenance for HPO artifacts."""
    expected = frozenset(allowed_keys or _CANONICAL_HPO_METRIC_KEYS)
    if not isinstance(value, Mapping):
        raise HPOMetricNotEligibleError("canonical metric metadata is not a mapping")
    if set(value) != expected:
        raise HPOMetricNotEligibleError("canonical metric metadata has incomplete identities")
    safe: dict[str, dict[str, Any]] = {}
    for metric_name, raw in value.items():
        if not isinstance(raw, Mapping):
            raise HPOMetricNotEligibleError("canonical metric metadata has an invalid value")
        if not _REQUIRED_CANONICAL_RAW_METRIC_FIELDS.issubset(raw):
            raise HPOMetricNotEligibleError("canonical metric metadata has missing fields")
        if raw["metric_name"] != metric_name:
            raise HPOMetricNotEligibleError("canonical metric metadata identity mismatch")
        if raw["fit_roles"] != ["train"] or raw["evaluation_role"] != "tuning":
            raise HPOMetricNotEligibleError("canonical metric metadata roles are invalid")
        expected_direction = "maximize" if metric_name == "tstr_macro_f1.v1" else "minimize"
        direction = raw.get("direction")
        if direction != expected_direction:
            raise HPOMetricNotEligibleError("canonical metric metadata direction is invalid")
        error_reason = raw["error_reason_code"]
        if error_reason not in _CANONICAL_METRIC_ERROR_CODES:
            raise HPOMetricNotEligibleError("canonical metric metadata error code is invalid")
        finite = (
            not isinstance(raw["mean"], bool)
            and isinstance(raw["mean"], (int, float, np.number))
            and math.isfinite(float(raw["mean"]))
        )
        errors = raw["errors"]
        if (
            isinstance(errors, bool)
            or not isinstance(errors, (int, float, np.number))
            or not math.isfinite(float(errors))
            or float(errors) < 0
            or float(errors) != int(float(errors))
        ):
            raise HPOMetricNotEligibleError("canonical metric metadata errors are invalid")
        status = raw["status"]
        if status not in {"complete", "failed"}:
            raise HPOMetricNotEligibleError("canonical metric metadata status is invalid")
        if status == "complete" and (not finite or errors != 0 or error_reason is not None):
            raise HPOMetricNotEligibleError("complete canonical metric metadata is inconsistent")
        if status == "failed" and (finite or error_reason is None):
            raise HPOMetricNotEligibleError("failed canonical metric metadata is inconsistent")
        safe[metric_name] = {
            "metric_name": metric_name,
            "status": status,
            "direction": direction,
            "finite": finite,
            "eligible": status == "complete" and finite,
            "error_reason_code": error_reason,
            "fit_roles": ["train"],
            "evaluation_role": "tuning",
        }
    return safe


def _validate_bounded_hpo_metric_metadata(
    value: Any, *, expected_keys: Sequence[str] | None = None
) -> dict[str, dict[str, Any]] | None:
    """Validate normalized canonical metric metadata at the durable boundary."""
    if value is None:
        return None
    expected = frozenset(expected_keys or _CANONICAL_HPO_METRIC_KEYS)
    if not isinstance(value, Mapping) or set(value) != expected:
        raise RuntimeError("HPO trial checkpoint metric metadata is unbounded or empty")
    validated: dict[str, dict[str, Any]] = {}
    for metric_name, raw in value.items():
        if metric_name not in expected or not isinstance(raw, Mapping):
            raise RuntimeError("HPO trial checkpoint metric metadata has unsafe identity")
        if set(raw) != _SAFE_METRIC_METADATA_FIELDS:
            raise RuntimeError("HPO trial checkpoint metric metadata has unsafe fields")
        if raw["metric_name"] != metric_name:
            raise RuntimeError("HPO trial checkpoint metric metadata identity mismatch")
        if raw["status"] not in {"complete", "failed"}:
            raise RuntimeError("HPO trial checkpoint metric metadata has unsafe status")
        expected_direction = "maximize" if metric_name == "tstr_macro_f1.v1" else "minimize"
        if raw["direction"] != expected_direction:
            raise RuntimeError("HPO trial checkpoint metric metadata has unsafe direction")
        if not isinstance(raw["finite"], bool) or not isinstance(raw["eligible"], bool):
            raise RuntimeError("HPO trial checkpoint metric metadata has unsafe flags")
        if raw["fit_roles"] != ["train"] or raw["evaluation_role"] != "tuning":
            raise RuntimeError("HPO trial checkpoint metric metadata has unsafe roles")
        error_code = raw["error_reason_code"]
        if error_code not in _CANONICAL_METRIC_ERROR_CODES:
            raise RuntimeError("HPO trial checkpoint metric metadata has unsafe error code")
        if raw["status"] == "complete":
            if error_code is not None or not raw["finite"] or not raw["eligible"]:
                raise RuntimeError("Complete HPO metric metadata is not eligible")
        elif raw["eligible"] or raw["finite"] or error_code is None:
            raise RuntimeError("Failed HPO metric metadata is inconsistent")
        validated[metric_name] = dict(raw)
    return validated


def _reject_unsafe_metadata_path(value: str | Path, location: str) -> str:
    """Return safe relative path identifiers without retaining raw filesystem paths."""
    if isinstance(value, Path) or not isinstance(value, str) or _looks_like_unsafe_path(value):
        raise HPOMetadataSerializationError(value, location)
    return value.replace("\\", "/")


def _looks_like_unsafe_path(value: str) -> bool:
    """Recognize path strings that cannot be durable metadata identifiers."""
    path = Path(value)
    components = re.split(r"[\\/]", value)
    return (
        path.is_absolute()
        or value.startswith(("/", "\\", "~/", "~\\"))
        or bool(re.match(r"^[A-Za-z]:", value))
        or any(component in {".", ".."} for component in components)
    )


class HPOMetadataSerializationError(TypeError):
    """Raised when durable HPO metadata contains an unsupported value."""

    reason_code = HPO_METADATA_SERIALIZATION_REASON_CODE

    def __init__(self, value: Any, location: str = "metadata") -> None:
        self.location = location
        super().__init__(f"unsupported HPO metadata value at {location}: {type(value).__name__}")


def normalize_hpo_metadata(value: Any, *, location: str = "metadata") -> Any:
    """Convert supported runtime values to deterministic JSON-compatible values.

    Mapping keys and fields ending in ``result_path`` remain protected.  Ordinary
    slash-bearing strings are valid values; unsafe absolute and traversal paths
    remain rejected.
    """
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        if _looks_like_unsafe_path(value) and not location.endswith("result_path"):
            raise HPOMetadataSerializationError(value, location)
        if location.endswith("result_path"):
            return _reject_unsafe_metadata_path(value, location)
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise HPOMetadataSerializationError(value, location)
        return value
    if isinstance(value, enum.Enum):
        return normalize_hpo_metadata(value.value, location=location)
    if isinstance(value, Path):
        raise HPOMetadataSerializationError(value, location)
    if isinstance(value, np.generic):
        return normalize_hpo_metadata(value.item(), location=location)
    if isinstance(value, np.ndarray):
        return [
            normalize_hpo_metadata(
                item,
                location=f"{location}[{index}]",
            )
            for index, item in enumerate(value.tolist())
        ]
    if type(value).__module__.startswith("torch") and type(value).__name__ == "device":
        return str(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        normalized_items: list[tuple[str, Any]] = []
        for key, item in value.items():
            if isinstance(key, os.PathLike):
                raise HPOMetadataSerializationError(key, location)
            normalized_key = str(key)
            if _looks_like_unsafe_path(normalized_key):
                raise HPOMetadataSerializationError(key, location)
            if isinstance(key, str):
                normalized_key = normalized_key.replace("\\", "/")
            normalized_items.append((normalized_key, item))
        for normalized_key, item in sorted(normalized_items, key=lambda entry: entry[0]):
            if normalized_key in result:
                raise HPOMetadataSerializationError(value, location)
            result[normalized_key] = normalize_hpo_metadata(
                item,
                location=f"{location}.{normalized_key}",
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            normalize_hpo_metadata(
                item,
                location=f"{location}[{index}]",
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, (set, frozenset)):
        normalized = [normalize_hpo_metadata(item, location=location) for item in value]
        return sorted(
            normalized, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))
        )
    raise HPOMetadataSerializationError(value, location)


def _safe_exception_type(error: BaseException) -> str:
    """Return an exception type identifier accepted by checkpoint validation."""
    raw_type = type(error).__name__
    if not isinstance(raw_type, str):
        return "Exception"
    safe_type = re.sub(r"[^A-Za-z0-9_.-]", "_", raw_type)
    if not safe_type or not re.match(r"^[A-Za-z_]", safe_type):
        safe_type = f"Exception_{safe_type}"
    return safe_type[:128] or "Exception"


def _safe_exception_location(location: str) -> str:
    """Return a bounded location identifier accepted by checkpoint validation."""
    if (
        isinstance(location, str)
        and len(location) <= 128
        and _SAFE_ERROR_LOCATION.fullmatch(location)
    ):
        return location
    return "unknown"


def hpo_exception_provenance(error: BaseException, *, location: str) -> dict[str, str]:
    """Return safe, stable diagnostics without preserving exception text."""
    reason_code = getattr(error, "reason_code", "hpo_trial_exception")
    if reason_code not in _SAFE_EXCEPTION_MESSAGES:
        reason_code = "hpo_trial_exception"
    error_type = _safe_exception_type(error)
    safe_location = _safe_exception_location(location)
    fingerprint = hashlib.sha256(f"{reason_code}:{error_type}:{safe_location}".encode()).hexdigest()
    return {
        "error_type": error_type,
        "error_message": _safe_exception_message(reason_code),
        "error_reason_code": reason_code,
        "error_location": safe_location,
        "error_fingerprint": fingerprint,
    }


def _validate_hpo_error_provenance(value: Any) -> dict[str, str]:
    """Validate bounded, sanitized exception provenance from a checkpoint."""
    if not isinstance(value, Mapping):
        raise RuntimeError("HPO error provenance must be an object")
    required = {
        "error_type",
        "error_message",
        "error_reason_code",
        "error_location",
        "error_fingerprint",
    }
    if set(value) != required:
        raise RuntimeError("HPO error provenance has an invalid shape")
    reason_code = value["error_reason_code"]
    if not isinstance(reason_code, str) or reason_code not in _SAFE_EXCEPTION_MESSAGES:
        raise RuntimeError("HPO error provenance has an unsafe reason code")
    error_type = value["error_type"]
    if not isinstance(error_type, str) or not error_type or len(error_type) > 128:
        raise RuntimeError("HPO error provenance has an invalid error type")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,127}", error_type):
        raise RuntimeError("HPO error provenance has an invalid error type")
    if value["error_message"] != _safe_exception_message(reason_code):
        raise RuntimeError("HPO error provenance has an unsafe error message")
    location = value["error_location"]
    if (
        not isinstance(location, str)
        or not location
        or len(location) > 128
        or not _SAFE_ERROR_LOCATION.fullmatch(location)
    ):
        raise RuntimeError("HPO error provenance has an unsafe error location")
    fingerprint = value["error_fingerprint"]
    if not isinstance(fingerprint, str) or not _SAFE_ERROR_FINGERPRINT.fullmatch(fingerprint):
        raise RuntimeError("HPO error provenance has an invalid fingerprint")
    expected = hpo_exception_provenance(
        type(error_type, (Exception,), {"reason_code": reason_code})(),
        location=location,
    )["error_fingerprint"]
    if fingerprint != expected:
        raise RuntimeError("HPO error provenance has an invalid fingerprint")
    return {key: value[key] for key in required}


def _validate_study_name(study_name: Any) -> str:
    """Validate study name as one portable, filesystem-safe identifier."""
    if not isinstance(study_name, str) or not study_name or not study_name.strip():
        raise ValueError("study_name must be a safe relative identifier")
    if study_name in {".", ".."}:
        raise ValueError("study_name must be a safe relative identifier")
    if any(character in study_name for character in ("/", "\\", ":")) or any(
        unicodedata.category(character).startswith("C") for character in study_name
    ):
        raise ValueError("study_name must be a safe relative identifier")
    path = Path(study_name)
    if path.is_absolute() or len(path.parts) != 1 or path.name != study_name:
        raise ValueError("study_name must be a safe relative identifier")
    return study_name


def _safe_stage_a_result_identifier(root: str | Path, path: str | Path) -> str:
    """Return Stage A path as a workspace-relative, traversal-free identifier."""
    root_path = Path(root).resolve()
    result_path = Path(path).resolve()
    try:
        relative = result_path.relative_to(root_path)
    except ValueError as exc:
        raise ValueError("Stage A result path must be inside its workspace") from exc
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Stage A result path must be relative to its workspace")
    return relative.as_posix()


def _checkpoint_stage_a_result_identifier(value: Any) -> str | None:
    """Keep only safe relative Stage A identifiers in durable checkpoints."""
    if value is None:
        return None
    if not isinstance(value, str) or not value or _looks_like_unsafe_path(value):
        raise RuntimeError("HPO trial checkpoint stage_a result_path is unsafe")
    return value.replace("\\", "/")


def _validate_stage_a_result_artifact(
    stage_a: Mapping[str, Any],
    *,
    root: str | Path | None,
    context: Mapping[str, Any],
    expected_study_name: str | None = None,
    expected_trial_number: int | None = None,
) -> None:
    """Validate referenced Stage A evidence at both checkpoint boundaries."""
    result_path = _checkpoint_stage_a_result_identifier(stage_a.get("result_path"))
    if result_path is None:
        return
    if root is None:
        raise RuntimeError("HPO checkpoint Stage A result artifact root is unavailable")
    root_path = Path(root).resolve()
    result_parts = Path(result_path).parts
    # Keep the workspace and every path component pinned by descriptors.  In
    # particular, do not validate with Path.resolve() and then reopen by name:
    # either the parent or the file could be replaced between those operations.
    try:
        workspace_fd = os.open(root_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise RuntimeError("HPO checkpoint Stage A result artifact root is unavailable") from exc
    result_fd = -1
    try:
        directory_fd = workspace_fd
        for part in result_parts[:-1]:
            next_fd = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            if directory_fd != workspace_fd:
                os.close(directory_fd)
            directory_fd = next_fd
        result_fd = os.open(
            result_parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        result_stat = os.fstat(result_fd)
        if not stat.S_ISREG(result_stat.st_mode):
            raise RuntimeError("HPO checkpoint Stage A result artifact is not a regular file")
        with os.fdopen(result_fd, "r", encoding="utf-8") as result_stream:
            result_fd = -1
            result = json.load(result_stream)
    except FileNotFoundError as exc:
        raise RuntimeError("HPO checkpoint Stage A result artifact is missing") from exc
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise RuntimeError(
                "HPO checkpoint Stage A result artifact uses a symlink path"
            ) from exc
        raise RuntimeError("HPO checkpoint Stage A result is unreadable") from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("HPO checkpoint Stage A result is unreadable") from exc
    finally:
        if result_fd != -1:
            os.close(result_fd)
        if directory_fd != workspace_fd:
            os.close(directory_fd)
        os.close(workspace_fd)
    if not isinstance(result, Mapping):
        raise RuntimeError("HPO checkpoint Stage A result is invalid")
    try:
        normalized = _json_document(result)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("HPO checkpoint Stage A result is not normalized JSON") from exc
    if normalized != result:
        raise RuntimeError("HPO checkpoint Stage A result is not normalized JSON")
    required = {
        "schema_version",
        "study_name",
        "trial_number",
        "contract_digest",
        "candidate_shape",
        "candidate_columns",
        "candidate_frame_fingerprint",
        "state",
        "passed",
        "pruned",
        "checks",
        "prune_reasons",
    }
    if set(result) != required:
        raise RuntimeError("HPO checkpoint Stage A result has an invalid schema")
    if result["schema_version"] != STAGE_A_SCREEN_SCHEMA_VERSION:
        raise RuntimeError("HPO checkpoint Stage A result has an unsupported schema")
    if expected_study_name is not None and result["study_name"] != expected_study_name:
        raise RuntimeError("HPO checkpoint Stage A study_name does not match checkpoint")
    if expected_trial_number is not None and result["trial_number"] != expected_trial_number:
        raise RuntimeError("HPO checkpoint Stage A trial_number does not match checkpoint")
    if not isinstance(result["study_name"], str) or not result["study_name"]:
        raise RuntimeError("HPO checkpoint Stage A study_name is invalid")
    if (
        isinstance(result["trial_number"], bool)
        or not isinstance(result["trial_number"], int)
        or result["trial_number"] < 0
    ):
        raise RuntimeError("HPO checkpoint Stage A trial_number is invalid")
    checkpoint_state = stage_a.get("state")
    checkpoint_digest = stage_a.get("contract_digest")
    if checkpoint_state not in {"passed", "pruned"} or not isinstance(checkpoint_digest, str):
        raise RuntimeError("HPO checkpoint Stage A evidence is incomplete")
    if result["state"] != checkpoint_state:
        raise RuntimeError("HPO checkpoint Stage A state does not match result")
    if result["contract_digest"] != checkpoint_digest:
        raise RuntimeError("HPO checkpoint Stage A contract digest does not match result")
    expected_digest = context.get("stage_a_contract_digest")
    if expected_digest is not None and checkpoint_digest != expected_digest:
        raise RuntimeError("HPO checkpoint Stage A contract digest does not match context")
    exception_result = (
        result["state"] == "pruned"
        and len(result["checks"]) == 1
        and isinstance(result["checks"][0], Mapping)
        and result["checks"][0].get("screen") == "stage_a_exception"
    )
    fingerprint = result["candidate_frame_fingerprint"]
    if fingerprint is None:
        if not exception_result:
            raise RuntimeError("HPO checkpoint Stage A candidate fingerprint is invalid")
    elif not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise RuntimeError("HPO checkpoint Stage A candidate fingerprint is invalid")
    if result["passed"] != (checkpoint_state == "passed") or result["pruned"] != (
        checkpoint_state == "pruned"
    ):
        raise RuntimeError("HPO checkpoint Stage A state flags are inconsistent")
    shape = result["candidate_shape"]
    columns = result["candidate_columns"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape
        )
    ):
        raise RuntimeError("HPO checkpoint Stage A result has invalid bounded fields")
    if (
        not isinstance(columns, list)
        or len(columns) > _MAX_STAGE_A_RESULT_COLUMNS
        or not all(
            isinstance(value, str)
            and 0 < len(value) <= 128
            and not _looks_like_unsafe_path(value)
            and not any(unicodedata.category(char).startswith("C") for char in value)
            for value in columns
        )
        or len(columns) != len(set(columns))
    ):
        raise RuntimeError("HPO checkpoint Stage A result has invalid bounded fields")
    if shape[1] != len(columns) or (not exception_result and (shape[0] < 1 or shape[1] < 1)):
        raise RuntimeError("HPO checkpoint Stage A candidate shape is inconsistent")
    checks = result["checks"]
    reasons = result["prune_reasons"]
    if (
        not isinstance(checks, list)
        or len(checks) > len(STAGE_A_SCREEN_IDS)
        or not isinstance(reasons, list)
        or len(reasons) > len(STAGE_A_SCREEN_IDS)
    ):
        raise RuntimeError("HPO checkpoint Stage A result has invalid bounded fields")
    allowed_check_keys = {"screen", "passed", "expected", "observed", "reason"}
    for check in checks:
        if not isinstance(check, Mapping) or not set(check) <= allowed_check_keys:
            raise RuntimeError("HPO checkpoint Stage A check has an invalid schema")
        screen = check.get("screen")
        if screen not in (*STAGE_A_SCREEN_IDS, "stage_a_exception"):
            raise RuntimeError("HPO checkpoint Stage A check has an unknown screen")
        if not isinstance(check.get("passed"), bool):
            raise RuntimeError("HPO checkpoint Stage A check has an invalid status")
        if "reason" in check and (
            not isinstance(check["reason"], str) or len(check["reason"]) > 4096
        ):
            raise RuntimeError("HPO checkpoint Stage A check reason is unbounded")
        try:
            normalized_check = _stage_a_check_json_document(check)
        except (TypeError, ValueError, OverflowError, HPOMetadataSerializationError) as exc:
            raise RuntimeError("HPO checkpoint Stage A check is not bounded JSON") from exc
        if (
            normalized_check != check
            or len(json.dumps(check, separators=(",", ":"))) > _MAX_STAGE_A_CHECK_BYTES
        ):
            raise RuntimeError("HPO checkpoint Stage A check is not bounded JSON")
    screens = [check.get("screen") for check in checks]
    if len(screens) != len(set(screens)):
        raise RuntimeError("HPO checkpoint Stage A checks contain duplicate screens")
    if "stage_a_exception" in screens and (len(screens) != 1 or result["state"] != "pruned"):
        raise RuntimeError("HPO checkpoint Stage A exception screen is not exclusive")
    if not all(isinstance(reason, str) and 0 < len(reason) <= 4096 for reason in reasons):
        raise RuntimeError("HPO checkpoint Stage A prune reasons are unbounded")
    if any(
        reason not in {check.get("reason") for check in checks if "reason" in check}
        for reason in reasons
    ):
        raise RuntimeError("HPO checkpoint Stage A prune reasons do not match checks")


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _scalar_equal(left: Any, right: Any) -> bool:
    if _is_missing_scalar(left) or _is_missing_scalar(right):
        return _is_missing_scalar(left) and _is_missing_scalar(right)
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return repr(left) == repr(right)


def _unique_non_missing(values) -> tuple[Any, ...]:
    unique = []
    for value in values:
        if _is_missing_scalar(value):
            continue
        if not any(_scalar_equal(value, existing) for existing in unique):
            unique.append(value)
    return tuple(unique)


def _json_safe(value: Any) -> Any:
    if _is_missing_scalar(value):
        return None
    if isinstance(value, Real) and not isinstance(value, bool):
        return float(value)
    return value


def _canonical_cell(value: Any) -> tuple:
    if _is_missing_scalar(value):
        return ("missing",)
    if isinstance(value, Real) and not isinstance(value, bool):
        return ("real", float(value))
    try:
        hash(value)
    except TypeError:
        return (type(value).__name__, repr(value))
    return (type(value).__name__, value)


def _canonical_rows(frame: pd.DataFrame, columns: tuple[str, ...]) -> set[tuple]:
    return {
        tuple(_canonical_cell(value) for value in row)
        for row in frame.loc[:, list(columns)].itertuples(index=False, name=None)
    }


def _dependency_key(values) -> str:
    return json.dumps(
        [_canonical_cell(value) for value in values],
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )


def _normalise_dependency_rules(
    source_df: pd.DataFrame,
    dependency_rules: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    normalised = []
    columns = set(source_df.columns)
    for rule in dependency_rules:
        if not isinstance(rule, dict):
            raise ValueError("Stage A dependency rules must be mappings")
        child = rule.get("child")
        parents = tuple(rule.get("parents", ()))
        if not isinstance(child, str) or not child:
            raise ValueError("Stage A dependency rules require a non-empty string child")
        if not parents or any(not isinstance(parent, str) or not parent for parent in parents):
            raise ValueError(
                f"Stage A dependency rule for {child!r} requires non-empty string parents"
            )
        if len(parents) != len(set(parents)):
            raise ValueError(f"Stage A dependency rule for {child!r} repeats a parent column")
        missing = sorted({child, *parents} - columns)
        if missing:
            raise ValueError(
                f"Stage A dependency rule for {child!r} references unknown columns {missing!r}"
            )

        mapping: dict[str, Any] = {}
        for row in source_df[[*parents, child]].itertuples(index=False, name=None):
            parent_key = _dependency_key(row[:-1])
            child_value = row[-1]
            if parent_key in mapping and not _scalar_equal(mapping[parent_key], child_value):
                raise ValueError(
                    f"Stage A dependency rule {child!r} <- {parents!r} is not deterministic "
                    "in the source frame"
                )
            mapping[parent_key] = _json_safe(child_value)
        normalised.append(
            {
                "child": child,
                "parents": list(parents),
                "mapping": mapping,
            }
        )
    return tuple(normalised)


@dataclasses.dataclass(frozen=True)
class StageAScreenContract:
    """Resolved, immutable contract for deterministic HPO candidate screens."""

    expected_n_samples: int
    columns: tuple[str, ...]
    target_column: str
    categorical_values: dict[str, tuple[Any, ...]]
    numeric_bounds: dict[str, tuple[float, float]]
    target_is_categorical: bool = True
    protected_columns: tuple[str, ...] = ()
    minimum_target_count: int = 1
    minimum_protected_group_count: int = 1
    minimum_target_by_protected_group_count: int = 1
    exact_reuse_max_rate: float = 0.0
    source_role: str = "train"
    source_frame_fingerprint: str | None = None
    dependency_rules: tuple[dict[str, Any], ...] = ()
    screen_ids: tuple[str, ...] = STAGE_A_SCREEN_IDS
    registry_digest: str | None = None
    role_context_fingerprint: str | None = None
    role_context: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    group_context: Mapping[str, Any] | None = None
    hpo_context: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(
            self,
            "categorical_values",
            {column: tuple(values) for column, values in self.categorical_values.items()},
        )
        object.__setattr__(
            self,
            "numeric_bounds",
            {column: tuple(bounds) for column, bounds in self.numeric_bounds.items()},
        )
        object.__setattr__(
            self,
            "protected_columns",
            tuple(self.protected_columns),
        )
        object.__setattr__(
            self,
            "dependency_rules",
            tuple(dict(rule) for rule in self.dependency_rules),
        )
        object.__setattr__(self, "screen_ids", tuple(self.screen_ids))
        if not isinstance(self.role_context, Mapping):
            raise TypeError("Stage A role_context must be a mapping")
        if self.group_context is not None and not isinstance(self.group_context, Mapping):
            raise TypeError("Stage A group_context must be a mapping or None")
        if not isinstance(self.hpo_context, Mapping):
            raise TypeError("Stage A hpo_context must be a mapping")
        object.__setattr__(self, "role_context", dict(self.role_context))
        object.__setattr__(
            self,
            "group_context",
            dict(self.group_context) if self.group_context is not None else None,
        )
        object.__setattr__(self, "hpo_context", dict(self.hpo_context))
        for name, value in (
            ("registry_digest", self.registry_digest),
            ("role_context_fingerprint", self.role_context_fingerprint),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"Stage A {name} must be a non-empty string or None")
        if isinstance(self.expected_n_samples, bool) or self.expected_n_samples < 1:
            raise ValueError("Stage A expected_n_samples must be a positive integer")
        if len(self.columns) != len(set(self.columns)):
            raise ValueError("Stage A contract columns must be unique")
        if self.target_column not in self.columns:
            raise ValueError("Stage A target_column must be present in contract columns")
        if not isinstance(self.target_is_categorical, bool):
            raise ValueError("Stage A target_is_categorical must be a boolean")
        if not set(self.categorical_values) <= set(self.columns):
            raise ValueError("Stage A categorical columns must be present in contract columns")
        if not set(self.numeric_bounds) <= set(self.columns):
            raise ValueError("Stage A numeric columns must be present in contract columns")
        if not set(self.protected_columns) <= set(self.columns):
            raise ValueError("Stage A protected columns must be present in contract columns")
        for name, value in (
            ("minimum_target_count", self.minimum_target_count),
            ("minimum_protected_group_count", self.minimum_protected_group_count),
            (
                "minimum_target_by_protected_group_count",
                self.minimum_target_by_protected_group_count,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Stage A {name} must be a non-negative integer")
        if (
            not isinstance(self.exact_reuse_max_rate, Real)
            or not 0 <= self.exact_reuse_max_rate <= 1
        ):
            raise ValueError("Stage A exact_reuse_max_rate must be between 0 and 1")
        if set(self.screen_ids) != set(STAGE_A_SCREEN_IDS):
            raise ValueError("Stage A contract must declare the complete screen id set")
        for column, bounds in self.numeric_bounds.items():
            if len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ValueError(f"Stage A numeric bounds for {column!r} must be increasing")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STAGE_A_SCREEN_SCHEMA_VERSION,
            "expected_n_samples": self.expected_n_samples,
            "columns": list(self.columns),
            "target_column": self.target_column,
            "target_is_categorical": self.target_is_categorical,
            "categorical_values": {
                column: [_json_safe(value) for value in values]
                for column, values in self.categorical_values.items()
            },
            "numeric_bounds": {
                column: [float(bounds[0]), float(bounds[1])]
                for column, bounds in self.numeric_bounds.items()
            },
            "protected_columns": list(self.protected_columns),
            "minimum_target_count": self.minimum_target_count,
            "minimum_protected_group_count": self.minimum_protected_group_count,
            "minimum_target_by_protected_group_count": self.minimum_target_by_protected_group_count,
            "exact_reuse_max_rate": float(self.exact_reuse_max_rate),
            "source_role": self.source_role,
            "source_frame_fingerprint": self.source_frame_fingerprint,
            "dependency_rules": [dict(rule) for rule in self.dependency_rules],
            "screen_ids": list(self.screen_ids),
            "registry_digest": self.registry_digest,
            "role_context_fingerprint": self.role_context_fingerprint,
            "role_context": dict(self.role_context),
            "group_context": dict(self.group_context) if self.group_context is not None else None,
            "hpo_context": dict(self.hpo_context),
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, default=str, separators=(",", ":")).encode()
        ).hexdigest()


@dataclasses.dataclass(frozen=True)
class StageAScreenResult:
    """Durable outcome of all required Stage A checks for one HPO trial."""

    contract_digest: str
    candidate_shape: tuple[int, int]
    candidate_columns: tuple[str, ...]
    candidate_frame_fingerprint: str | None
    state: str
    checks: tuple[dict[str, Any], ...]
    prune_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.state not in {"passed", "pruned"}:
            raise ValueError(f"Unknown Stage A result state: {self.state!r}")
        object.__setattr__(self, "candidate_shape", tuple(self.candidate_shape))
        object.__setattr__(self, "candidate_columns", tuple(self.candidate_columns))
        object.__setattr__(self, "checks", tuple(dict(check) for check in self.checks))
        object.__setattr__(self, "prune_reasons", tuple(self.prune_reasons))

    @property
    def passed(self) -> bool:
        return self.state == "passed"

    @property
    def pruned(self) -> bool:
        return self.state == "pruned"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STAGE_A_SCREEN_SCHEMA_VERSION,
            "contract_digest": self.contract_digest,
            "candidate_shape": list(self.candidate_shape),
            "candidate_columns": list(self.candidate_columns),
            "candidate_frame_fingerprint": self.candidate_frame_fingerprint,
            "state": self.state,
            "passed": self.passed,
            "pruned": self.pruned,
            "checks": [dict(check) for check in self.checks],
            "prune_reasons": list(self.prune_reasons),
        }


def build_stage_a_contract(
    source_df: pd.DataFrame,
    *,
    expected_n_samples: int,
    target_column: str,
    target_is_categorical: bool = True,
    categorical_columns: list[str] | tuple[str, ...] = (),
    protected_columns: list[str] | tuple[str, ...] = (),
    minimum_target_count: int = 1,
    minimum_protected_group_count: int = 1,
    minimum_target_by_protected_group_count: int = 1,
    exact_reuse_max_rate: float = 0.0,
    source_role: str = "train",
    source_frame_fingerprint: str | None = None,
    dependency_rules: tuple[dict[str, Any], ...] = (),
    registry_digest: str | None = None,
    role_context_fingerprint: str | None = None,
    role_context: Mapping[str, Any] | None = None,
    group_context: Mapping[str, Any] | None = None,
    hpo_context: Mapping[str, Any] | None = None,
) -> StageAScreenContract:
    """Resolve candidate shape, schema, vocabulary, bounds, and support rules."""
    if not isinstance(source_df, pd.DataFrame):
        raise TypeError("Stage A source_df must be a pandas DataFrame")
    if not isinstance(target_is_categorical, bool):
        raise TypeError("Stage A target_is_categorical must be a boolean")
    columns = tuple(str(column) for column in source_df.columns)
    if len(columns) != len(set(columns)):
        raise ValueError("Stage A source columns must be unique")
    if target_column not in columns:
        raise ValueError(f"Stage A target column {target_column!r} is not present in source_df")
    categorical = tuple(dict.fromkeys(str(column) for column in categorical_columns))
    protected = tuple(dict.fromkeys(str(column) for column in protected_columns))
    for column in (*categorical, *protected):
        if column not in columns:
            raise ValueError(f"Stage A declared column {column!r} is not present in source_df")
    categorical = tuple(
        dict.fromkeys((*categorical, *([target_column] if target_is_categorical else [])))
    )

    categorical_values = {
        column: _unique_non_missing(source_df[column].tolist()) for column in categorical
    }
    numeric_bounds = {}
    for column in columns:
        if column in categorical:
            continue
        numeric = pd.to_numeric(source_df[column], errors="coerce")
        finite = numeric[pd.notna(numeric) & numeric.map(math.isfinite)]
        if finite.empty:
            raise ValueError(f"Stage A numeric column {column!r} has no finite source values")
        numeric_bounds[column] = (float(finite.min()), float(finite.max()))

    return StageAScreenContract(
        expected_n_samples=expected_n_samples,
        columns=columns,
        target_column=target_column,
        target_is_categorical=target_is_categorical,
        categorical_values=categorical_values,
        numeric_bounds=numeric_bounds,
        protected_columns=protected,
        minimum_target_count=minimum_target_count,
        minimum_protected_group_count=minimum_protected_group_count,
        minimum_target_by_protected_group_count=minimum_target_by_protected_group_count,
        exact_reuse_max_rate=exact_reuse_max_rate,
        source_role=source_role,
        source_frame_fingerprint=source_frame_fingerprint or dataframe_fingerprint(source_df),
        dependency_rules=_normalise_dependency_rules(source_df, dependency_rules),
        registry_digest=registry_digest,
        role_context_fingerprint=role_context_fingerprint,
        role_context=role_context or {},
        group_context=group_context,
        hpo_context=hpo_context or {},
    )


def screen_stage_a(
    candidate_df: pd.DataFrame,
    contract: StageAScreenContract,
    source_df: pd.DataFrame,
) -> StageAScreenResult:
    """Run deterministic pre-evaluation screens and return a prune decision."""
    if not isinstance(candidate_df, pd.DataFrame):
        raise TypeError("Stage A candidate_df must be a pandas DataFrame")
    if not isinstance(source_df, pd.DataFrame):
        raise TypeError("Stage A source_df must be a pandas DataFrame")
    if tuple(str(column) for column in source_df.columns) != contract.columns:
        raise ValueError("Stage A source columns do not match the resolved contract")
    if (
        contract.source_frame_fingerprint is not None
        and dataframe_fingerprint(source_df) != contract.source_frame_fingerprint
    ):
        raise ValueError("Stage A source frame does not match the resolved contract")

    checks: list[dict[str, Any]] = []
    prune_reasons: list[str] = []

    def add_check(screen: str, passed: bool, expected: Any, observed: Any, reason: str) -> None:
        reason = _bound_stage_a_reason(reason)
        check = {
            "screen": screen,
            "passed": bool(passed),
            "expected": expected,
            "observed": observed,
        }
        if reason:
            check["reason"] = reason
            prune_reasons.append(reason)
        checks.append(check)

    candidate_columns = tuple(str(column) for column in candidate_df.columns)
    shape_ok = (
        len(candidate_df) == contract.expected_n_samples
        and candidate_columns == contract.columns
        and len(candidate_columns) == len(set(candidate_columns))
    )
    shape_reason = (
        "candidate shape/schema does not match the Stage A contract" if not shape_ok else ""
    )
    add_check(
        "shape_schema",
        shape_ok,
        {"rows": contract.expected_n_samples, "columns": list(contract.columns)},
        {"rows": len(candidate_df), "columns": list(candidate_columns)},
        shape_reason,
    )

    bounds_observed: dict[str, Any] = {}
    bounds_reasons = []
    for column, values in contract.categorical_values.items():
        if column not in candidate_df.columns:
            bounds_reasons.append(f"categorical column {column!r} is missing")
            continue
        unseen = [
            _json_safe(value)
            for value in _unique_non_missing(candidate_df[column].tolist())
            if not any(_scalar_equal(value, allowed) for allowed in values)
        ]
        missing_count = int(candidate_df[column].isna().sum())
        bounds_observed[column] = {"unseen": unseen, "missing": missing_count}
        if unseen or missing_count:
            bounds_reasons.append(
                f"categorical column {column!r} has unseen={unseen!r} or missing={missing_count}"
            )
    for column, (lower, upper) in contract.numeric_bounds.items():
        if column not in candidate_df.columns:
            bounds_reasons.append(f"numeric column {column!r} is missing")
            continue
        numeric = pd.to_numeric(candidate_df[column], errors="coerce")
        finite = numeric.notna() & numeric.map(math.isfinite)
        out_of_bounds = finite & ((numeric < lower) | (numeric > upper))
        invalid_count = int((~finite).sum())
        out_count = int(out_of_bounds.sum())
        bounds_observed[column] = {
            "min": float(numeric[finite].min()) if finite.any() else None,
            "max": float(numeric[finite].max()) if finite.any() else None,
            "invalid": invalid_count,
            "out_of_bounds": out_count,
        }
        if invalid_count or out_count:
            bounds_reasons.append(
                f"numeric column {column!r} has invalid={invalid_count} or "
                f"out_of_bounds={out_count}"
            )
    add_check(
        "bounds_categories",
        not bounds_reasons,
        {
            "categorical_values": contract.to_dict()["categorical_values"],
            "bounds": contract.to_dict()["numeric_bounds"],
        },
        bounds_observed,
        "; ".join(bounds_reasons),
    )

    support_observed: dict[str, Any] = {}
    support_reasons = []
    target_values = (
        contract.categorical_values.get(contract.target_column, ())
        if contract.target_is_categorical
        else ()
    )
    if contract.target_column not in candidate_df.columns:
        support_reasons.append(f"target column {contract.target_column!r} is missing")
    else:
        target_counts = candidate_df[contract.target_column].value_counts(dropna=False)
        target_support = {
            str(_json_safe(value)): int(
                sum(
                    int(count)
                    for observed, count in target_counts.items()
                    if _scalar_equal(observed, value)
                )
            )
            for value in target_values
        }
        support_observed["target"] = target_support
        insufficient = {
            str(_json_safe(value)): count
            for value, count in zip(target_values, target_support.values(), strict=True)
            if count < contract.minimum_target_count
        }
        if insufficient:
            support_reasons.append(f"target support below minimum: {insufficient!r}")
    add_check(
        "support",
        not support_reasons,
        {"minimum_target_count": contract.minimum_target_count},
        support_observed,
        "; ".join(support_reasons),
    )

    dependency_observed = []
    dependency_reasons = []
    for rule in contract.dependency_rules:
        child = rule.get("child")
        parents = tuple(rule.get("parents", ()))
        mapping = rule.get("mapping", {})
        missing = [column for column in (child, *parents) if column not in candidate_df.columns]
        violations = []
        if not missing:
            for row_number, row in enumerate(
                candidate_df[[*parents, child]].itertuples(index=False, name=None)
            ):
                parent_key = _dependency_key(row[:-1])
                expected = mapping.get(parent_key)
                observed = row[-1]
                if expected is None and parent_key not in mapping:
                    violations.append({"row": row_number, "reason": "unknown parent combination"})
                elif not _scalar_equal(observed, expected):
                    violations.append(
                        {
                            "row": row_number,
                            "reason": "child value does not match source dependency",
                        }
                    )
        dependency_observed.append(
            {
                "child": child,
                "parents": list(parents),
                "missing": missing,
                "violations": violations,
            }
        )
        if missing:
            dependency_reasons.append(
                f"dependency {child!r} is missing required column(s) {missing!r}"
            )
        if violations:
            dependency_reasons.append(
                f"dependency {child!r} has {len(violations)} violating candidate row(s)"
            )
    add_check(
        "dependencies",
        not dependency_reasons,
        {"rules": [dict(rule) for rule in contract.dependency_rules]},
        dependency_observed,
        "; ".join(dependency_reasons),
    )

    exact_reuse_observed = {"hit_count": None, "hit_rate": None}
    exact_reuse_reason = ""
    if shape_ok:
        source_rows = _canonical_rows(source_df, contract.columns)
        candidate_rows = [
            tuple(_canonical_cell(value) for value in row)
            for row in candidate_df.itertuples(index=False, name=None)
        ]
        hit_count = sum(row in source_rows for row in candidate_rows)
        hit_rate = hit_count / len(candidate_rows) if candidate_rows else 0.0
        exact_reuse_observed = {"hit_count": hit_count, "hit_rate": hit_rate}
        if hit_rate > contract.exact_reuse_max_rate:
            exact_reuse_reason = (
                f"exact reuse hit_rate={hit_rate:.6g} exceeds "
                f"maximum={contract.exact_reuse_max_rate:.6g}"
            )
    else:
        exact_reuse_reason = "exact reuse screen requires a valid candidate shape/schema"
    add_check(
        "exact_reuse",
        not exact_reuse_reason,
        {
            "source_role": contract.source_role,
            "max_hit_rate": float(contract.exact_reuse_max_rate),
        },
        exact_reuse_observed,
        exact_reuse_reason,
    )

    subgroup_observed: dict[str, Any] = {}
    subgroup_reasons = []
    for column in contract.protected_columns:
        if column not in candidate_df.columns:
            subgroup_reasons.append(f"protected column {column!r} is missing")
            continue
        if column in contract.numeric_bounds:
            subgroup_observed[column] = {
                "status": "not_applicable",
                "discrete": False,
                "reason": "continuous protected column is non-discrete",
            }
            continue
        values = contract.categorical_values.get(column, ())
        counts = candidate_df[column].value_counts(dropna=False)
        group_counts = {
            str(_json_safe(value)): int(
                sum(
                    int(count)
                    for observed, count in counts.items()
                    if _scalar_equal(observed, value)
                )
            )
            for value in values
        }
        subgroup_observed[column] = {"groups": group_counts}
        insufficient_groups = {
            group: count
            for group, count in group_counts.items()
            if count < contract.minimum_protected_group_count
        }
        if insufficient_groups:
            subgroup_reasons.append(
                f"protected-group support below minimum for {column!r}: {insufficient_groups!r}"
            )
        if contract.target_is_categorical and contract.target_column in candidate_df.columns:
            cells = {}
            source_cells = source_df[[column, contract.target_column]].drop_duplicates()
            for group, target in source_cells.itertuples(index=False, name=None):
                cell_count = int(
                    (
                        (candidate_df[column] == group)
                        & (candidate_df[contract.target_column] == target)
                    ).sum()
                )
                cell_key = f"{_json_safe(group)!r}|{_json_safe(target)!r}"
                cells[cell_key] = cell_count
                if cell_count < contract.minimum_target_by_protected_group_count:
                    subgroup_reasons.append(
                        f"target/protected cell {column!r}={cell_key} has count={cell_count}, "
                        f"minimum={contract.minimum_target_by_protected_group_count}"
                    )
            subgroup_observed[column]["target_cells"] = cells
    add_check(
        "subgroup_collapse",
        not subgroup_reasons,
        {
            "minimum_protected_group_count": contract.minimum_protected_group_count,
            "minimum_target_by_protected_group_count": contract.minimum_target_by_protected_group_count,
        },
        subgroup_observed,
        "; ".join(subgroup_reasons),
    )

    return StageAScreenResult(
        contract_digest=contract.digest,
        candidate_shape=(len(candidate_df), len(candidate_df.columns)),
        candidate_columns=candidate_columns,
        candidate_frame_fingerprint=dataframe_fingerprint(candidate_df),
        state="passed" if not prune_reasons else "pruned",
        checks=tuple(checks),
        prune_reasons=tuple(prune_reasons),
    )


def _stage_a_exception_result(
    candidate_df: object,
    contract: StageAScreenContract,
    error: TypeError | ValueError | RuntimeError,
) -> StageAScreenResult:
    if isinstance(candidate_df, pd.DataFrame):
        candidate_shape = (len(candidate_df), len(candidate_df.columns))
        candidate_columns = tuple(str(column) for column in candidate_df.columns)
    else:
        candidate_shape = (0, 0)
        candidate_columns = ()
    reason_code = "stage_a_screen_exception"
    message = _safe_exception_message(reason_code)
    reason = _bound_stage_a_reason(f"{reason_code}: {message}")
    return StageAScreenResult(
        contract_digest=contract.digest,
        candidate_shape=candidate_shape,
        candidate_columns=candidate_columns,
        candidate_frame_fingerprint=None,
        state="pruned",
        checks=(
            {
                "screen": "stage_a_exception",
                "passed": False,
                "expected": {"screen": "screen_stage_a"},
                "observed": {
                    "candidate_type": type(candidate_df).__name__,
                    "exception_type": type(error).__name__,
                    "exception_message": message,
                    "reason_code": reason_code,
                },
                "reason": reason,
            },
        ),
        prune_reasons=(reason,),
    )


def _record_stage_a_trial_result(
    trial: optuna.Trial,
    root: str | Path,
    study_name: str,
    result: StageAScreenResult,
) -> Path:
    result_path = persist_stage_a_result(root, study_name, trial.number, result)
    trial.set_user_attr("stage_a_state", result.state)
    trial.set_user_attr("stage_a_contract_digest", result.contract_digest)
    trial.set_user_attr("stage_a_result_path", _safe_stage_a_result_identifier(root, result_path))
    trial.set_user_attr("stage_a_prune_reasons", list(result.prune_reasons))
    return result_path


def persist_stage_a_exception(
    root: str | Path,
    study_name: str,
    contract: StageAScreenContract,
    error: TypeError | ValueError | RuntimeError,
) -> Path:
    """Persist a pre-trial Stage A construction failure for study diagnostics."""
    _require_stage_a_locking()
    study_name = _validate_study_name(study_name)
    result = _stage_a_exception_result(None, contract, error)
    path = Path(root) / study_name / "construction-failure.json"
    payload = {"study_name": study_name, "trial_number": None, **result.to_dict()}
    _locked_stage_a_json(
        Path(root),
        (study_name,),
        "construction-failure.json",
        payload,
        path,
        artifact_label="construction failure",
    )
    return path


def persist_stage_a_trial_exception(
    trial: optuna.Trial,
    root: str | Path,
    study_name: str,
    contract: StageAScreenContract,
    error: TypeError | ValueError | RuntimeError,
) -> StageAScreenResult:
    """Persist a candidate-construction failure against a concrete trial."""
    result = _stage_a_exception_result(None, contract, error)
    _record_stage_a_trial_result(trial, root, study_name, result)
    return result


def screen_stage_a_trial(
    trial: optuna.Trial,
    candidate_df: pd.DataFrame,
    contract: StageAScreenContract,
    source_df: pd.DataFrame,
    root: str | Path,
    study_name: str,
) -> StageAScreenResult:
    """Persist a trial screen and prune it when any required check fails."""
    study_name = _validate_study_name(study_name)
    try:
        result = screen_stage_a(candidate_df, contract, source_df)
    except (TypeError, ValueError, RuntimeError) as exc:
        result = _stage_a_exception_result(candidate_df, contract, exc)
        logger.warning(
            "[hpo][stage_a] screen failed for study=%s trial=%s candidate_shape=%s: %s",
            study_name,
            trial.number,
            result.candidate_shape,
            result.prune_reasons[0],
        )
    _record_stage_a_trial_result(trial, root, study_name, result)
    if result.pruned:
        raise optuna.TrialPruned("Stage A screen failed: " + "; ".join(result.prune_reasons))
    return result


def _require_stage_a_locking() -> Any:
    """Require POSIX advisory locking; unsupported platforms fail explicitly."""
    missing: list[str] = []
    if fcntl is None:
        missing.append("fcntl")
    else:
        if not callable(getattr(fcntl, "flock", None)):
            missing.append("fcntl.flock")
        if getattr(fcntl, "LOCK_EX", None) is None:
            missing.append("fcntl.LOCK_EX")
    for capability in ("O_DIRECTORY", "O_NOFOLLOW"):
        if getattr(os, capability, None) is None:
            missing.append(f"os.{capability}")
    if missing:
        raise RuntimeError(
            "Stage A persistence requires POSIX descriptor locking capabilities; "
            f"missing: {', '.join(missing)}"
        )
    return fcntl


def _open_stage_a_directory(root: Path, parts: Sequence[str], *, create: bool) -> int:
    """Open directory chain with no-follow descriptors, optionally creating it."""
    root = Path(root)
    # Do not resolve or create root by path.  Walk every existing component from
    # a pinned descriptor, so neither root nor an ancestor can be substituted by
    # a symlink during setup.  Callers must create the workspace root first.
    if root.is_absolute():
        directory_fd = os.open(os.sep, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        root_parts = root.parts[1:]
    else:
        directory_fd = os.open(".", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        root_parts = root.parts
    if any(part in {".", ".."} for part in root_parts):
        os.close(directory_fd)
        raise ValueError("Stage A workspace path contains traversal")
    try:
        for part in root_parts:
            try:
                next_fd = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, 0o700, dir_fd=directory_fd)
                next_fd = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
            os.close(directory_fd)
            directory_fd = next_fd
        for part in parts:
            try:
                next_fd = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, 0o700, dir_fd=directory_fd)
                next_fd = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
            os.close(directory_fd)
            directory_fd = next_fd
        return directory_fd
    except BaseException:
        os.close(directory_fd)
        raise


def _read_stage_a_json_fd(
    directory_fd: int, name: str, path: Path, *, artifact_label: str = "result"
) -> dict[str, Any]:
    _, value = _read_stage_a_json_bytes_fd(directory_fd, name, path, artifact_label=artifact_label)
    return value


def _read_stage_a_json_bytes_fd(
    directory_fd: int, name: str, path: Path, *, artifact_label: str = "result"
) -> tuple[bytes, dict[str, Any]]:
    file_fd = -1
    try:
        file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            raise RuntimeError(f"Stage A {artifact_label} at {path} is not a regular file")
        with os.fdopen(file_fd, "rb") as stream:
            file_fd = -1
            encoded = stream.read()
            value = json.loads(encoded)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Stage A {artifact_label} at {path} is missing") from exc
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise RuntimeError(f"Stage A {artifact_label} at {path} uses a symlink path") from exc
        raise RuntimeError(f"Stage A {artifact_label} at {path} is unreadable") from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Stage A {artifact_label} at {path} is unreadable") from exc
    finally:
        if file_fd != -1:
            os.close(file_fd)
    if not isinstance(value, dict):
        raise RuntimeError(f"Stage A {artifact_label} at {path} is invalid")
    return encoded, value


def _stage_a_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True, default=str).encode()


def _atomic_stage_a_json_fd(directory_fd: int, name: str, payload: dict[str, Any]) -> None:
    """Publish JSON using only pinned directory descriptors."""
    encoded = _stage_a_json_bytes(payload)
    temporary_name = f".{name}.{uuid.uuid4().hex}.tmp"
    file_fd = os.open(
        temporary_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory_fd,
    )
    try:
        view = memoryview(encoded)
        written = 0
        while written < len(encoded):
            count = os.write(file_fd, view[written:])
            if count <= 0:
                raise OSError("short write while persisting Stage A JSON")
            written += count
        if written != len(encoded):
            raise OSError("incomplete write while persisting Stage A JSON")
        os.fsync(file_fd)
    except BaseException:
        os.close(file_fd)
        with suppress(FileNotFoundError):
            os.unlink(temporary_name, dir_fd=directory_fd)
        raise
    else:
        os.close(file_fd)
    try:
        os.replace(temporary_name, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        os.fsync(directory_fd)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name, dir_fd=directory_fd)
        raise


def _locked_stage_a_json(
    root: Path,
    parts: Sequence[str],
    name: str,
    payload: dict[str, Any],
    path: Path,
    *,
    artifact_label: str = "result",
) -> None:
    """Create/read/publish one append-only JSON evidence file under one lock."""
    lock = _require_stage_a_locking()
    try:
        directory_fd = _open_stage_a_directory(root, parts, create=True)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise RuntimeError(f"Stage A {artifact_label} at {path} uses a symlink path") from exc
        raise RuntimeError(f"Stage A {artifact_label} at {path} is unreadable") from exc
    lock_fd = -1
    try:
        lock_fd = os.open(
            ".stage-a.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd
        )
        lock.flock(lock_fd, lock.LOCK_EX)
        try:
            cached_bytes, cached = _read_stage_a_json_bytes_fd(
                directory_fd, name, path, artifact_label=artifact_label
            )
        except RuntimeError as exc:
            if " is missing" not in str(exc):
                raise
            cached = None
            cached_bytes = None
        if cached is not None and (
            cached_bytes != _stage_a_json_bytes(payload) or cached != payload
        ):
            raise RuntimeError(
                f"Stage A {artifact_label} at {path} does not match the current {artifact_label}"
            )
        if cached is None:
            _atomic_stage_a_json_fd(directory_fd, name, payload)
        persisted_bytes, persisted = _read_stage_a_json_bytes_fd(
            directory_fd, name, path, artifact_label=artifact_label
        )
        if persisted_bytes != _stage_a_json_bytes(payload) or persisted != payload:
            raise RuntimeError(
                f"Stage A {artifact_label} at {path} does not match the current {artifact_label}"
            )
    finally:
        if lock_fd != -1:
            os.close(lock_fd)
        os.close(directory_fd)


def _locked_read_stage_a_json(
    root: Path,
    parts: Sequence[str],
    name: str,
    path: Path,
    *,
    artifact_label: str,
) -> dict[str, Any]:
    lock = _require_stage_a_locking()
    directory_fd = _open_stage_a_directory(root, parts, create=False)
    lock_fd = -1
    try:
        lock_fd = os.open(
            ".stage-a.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd
        )
        lock.flock(lock_fd, lock.LOCK_EX)
        _, payload = _read_stage_a_json_bytes_fd(
            directory_fd, name, path, artifact_label=artifact_label
        )
        return payload
    finally:
        if lock_fd != -1:
            os.close(lock_fd)
        os.close(directory_fd)


def _read_stage_a_json_pinned(root: str | Path, path: Path) -> dict[str, Any]:
    """Read Stage A JSON through descriptors pinned to its workspace."""
    # Normalize both operands lexically.  ``resolve`` must not be used here:
    # the descriptor walk below is deliberately responsible for rejecting
    # symlink components rather than following them during path validation.
    root_path = Path(os.path.abspath(root))
    result_path = Path(os.path.abspath(path))
    try:
        relative = result_path.relative_to(root_path)
    except ValueError as exc:
        raise RuntimeError(f"Stage A result at {path} is outside its workspace") from exc
    parts = relative.parts
    if not parts:
        raise RuntimeError(f"Stage A result at {path} is unreadable")
    workspace_fd = -1
    directory_fd = -1
    result_fd = -1
    try:
        workspace_fd = os.open(root_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        directory_fd = workspace_fd
        for part in parts[:-1]:
            next_fd = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            if directory_fd != workspace_fd:
                os.close(directory_fd)
            directory_fd = next_fd
        result_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        result_stat = os.fstat(result_fd)
        if not stat.S_ISREG(result_stat.st_mode):
            raise RuntimeError(f"Stage A result at {path} is not a regular file")
        with os.fdopen(result_fd, "r", encoding="utf-8") as result_stream:
            result_fd = -1
            result = json.load(result_stream)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Stage A result at {path} is missing") from exc
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise RuntimeError(f"Stage A result at {path} uses a symlink path") from exc
        raise RuntimeError(f"Stage A result at {path} is unreadable") from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Stage A result at {path} is unreadable") from exc
    finally:
        if result_fd != -1:
            os.close(result_fd)
        if directory_fd != -1 and directory_fd != workspace_fd:
            os.close(directory_fd)
        if workspace_fd != -1:
            os.close(workspace_fd)
    if not isinstance(result, dict):
        raise RuntimeError(f"Stage A result at {path} is invalid")
    return result


def persist_stage_a_contract(root: str | Path, contract: StageAScreenContract) -> Path:
    """Persist the resolved Stage A contract atomically for a study."""
    _require_stage_a_locking()
    root_path = Path(root)
    path = root_path / "contract.json"
    payload = {**contract.to_dict(), "digest": contract.digest}
    _locked_stage_a_json(root_path, (), "contract.json", payload, path, artifact_label="contract")
    return path


def persist_stage_a_result(
    root: str | Path,
    study_name: str,
    trial_number: int,
    result: StageAScreenResult,
) -> Path:
    """Persist one Stage A outcome per trial so prunes are resumable and auditable."""
    _require_stage_a_locking()
    study_name = _validate_study_name(study_name)
    if isinstance(trial_number, bool) or not isinstance(trial_number, int) or trial_number < 0:
        raise ValueError("Stage A trial_number must be a non-negative integer")
    path = Path(root) / study_name / f"trial-{trial_number}" / "result.json"
    payload = {"study_name": study_name, "trial_number": trial_number, **result.to_dict()}
    artifact_reference = {
        "state": result.state,
        "contract_digest": result.contract_digest,
        "result_path": _safe_stage_a_result_identifier(root, path),
    }
    _locked_stage_a_json(
        Path(root), (study_name, f"trial-{trial_number}"), "result.json", payload, path
    )
    persisted = _read_stage_a_json_pinned(root, path)
    if persisted != payload:
        raise RuntimeError(f"Stage A result at {path} does not match the current result")
    _validate_stage_a_result_artifact(
        artifact_reference,
        root=root,
        context={"stage_a_contract_digest": result.contract_digest},
        expected_study_name=study_name,
        expected_trial_number=trial_number,
    )
    return path


def validate_existing_stage_a_trial_result(
    root: str | Path,
    study_name: str,
    trial_number: int,
    contract: StageAScreenContract,
) -> bool:
    """Validate an already-persisted trial result without replacing it.

    Return ``False`` only when the expected result does not exist.  Any
    present artifact is read through pinned descriptors and validated against
    the trial identity and Stage A contract, so malformed or divergent
    evidence remains an explicit failure rather than being overwritten.
    """
    if isinstance(trial_number, bool) or not isinstance(trial_number, int) or trial_number < 0:
        raise ValueError("Stage A trial_number must be a non-negative integer")
    study_name = _validate_study_name(study_name)
    path = Path(root) / study_name / f"trial-{trial_number}" / "result.json"
    try:
        payload = _read_stage_a_json_pinned(root, path)
    except RuntimeError as exc:
        if str(exc).endswith(" is missing"):
            return False
        raise
    _validate_stage_a_result_artifact(
        {
            "state": payload.get("state"),
            "contract_digest": payload.get("contract_digest"),
            "result_path": _safe_stage_a_result_identifier(root, path),
        },
        root=root,
        context={"stage_a_contract_digest": contract.digest},
        expected_study_name=study_name,
        expected_trial_number=trial_number,
    )
    return True


def prepare_stage_a_screen(
    contract: StageAScreenContract | None,
    source_df: pd.DataFrame | None,
    root: str | Path | None,
    study_name: str | None,
) -> None:
    """Validate and persist the screen contract before an HPO study starts."""
    configured = (contract, source_df, root, study_name)
    if contract is None:
        if any(value is not None for value in configured[1:]):
            raise ValueError("Stage A source_df, root, and study_name require a Stage A contract")
        return
    if source_df is None or root is None or study_name is None:
        raise ValueError("Stage A contract requires source_df, root, and non-empty study_name")
    _require_stage_a_locking()
    _validate_study_name(study_name)
    persist_stage_a_contract(Path(root) / study_name, contract)


HPO_OBJECTIVE_METRICS = frozenset(CANONICAL_HPO_ALLOWLIST)
TUNING_UTILITY_METRICS = (
    "tstr_macro_f1.v1",
    "mixed_mmd.v1",
    "elastic_net_jsd.v1",
)
TUNING_UTILITY_WEIGHTS = (1 / 3, 1 / 3, 1 / 3)
TUNING_OBJECTIVE_VERSION = "release-utility-v1"


def _evaluate_train_frozen_mmd(
    train: pd.DataFrame,
    tuning: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    continuous_columns: Sequence[str],
    ordinal_columns: Sequence[str],
    nominal_columns: Sequence[str],
) -> dict[str, Any]:
    """Evaluate MMD with preprocessing and bandwidth frozen on train only."""
    from syntheval.metrics.utility.metric_max_mean_discrepancy import _mixed_kernel
    from syntheval.utils.preprocessing import MixedSchemaPreprocessor

    preprocessor = MixedSchemaPreprocessor.fit(
        train, list(continuous_columns), list(ordinal_columns), list(nominal_columns), None
    )
    train_values = preprocessor.transform(train, role="train")
    tuning_values = preprocessor.transform(tuning, role="tuning")
    candidate_values = preprocessor.transform(candidate, role="candidate")
    active = [role for role, values in train_values.items() if values.shape[1]]
    weights = {
        role: (1.0 / len(active) if role in active else 0.0)
        for role in ("continuous", "ordinal", "nominal")
    }
    fit_kernel = _mixed_kernel(train_values, train_values, 1.0, weights)
    fit_distances = -2.0 * np.log(
        np.maximum(fit_kernel[np.triu_indices(len(fit_kernel), 1)], 1e-300)
    )
    positive = fit_distances[fit_distances > 0]
    bandwidth = float(np.sqrt(np.median(positive))) if positive.size else 1.0
    bandwidth = max(bandwidth, np.finfo(float).eps)
    kxx = _mixed_kernel(tuning_values, tuning_values, bandwidth, weights)
    kyy = _mixed_kernel(candidate_values, candidate_values, bandwidth, weights)
    kxy = _mixed_kernel(tuning_values, candidate_values, bandwidth, weights)
    raw_biased = float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())
    biased = max(raw_biased, 0.0)
    return {
        "b_mmd_clip": biased,
        "bandwidth": bandwidth,
        "weights": weights,
        "preprocessing": preprocessor.metadata(),
        "fit_role": "train",
        "comparison_role": "tuning",
    }


def _evaluate_train_frozen_jsd(
    train: pd.DataFrame,
    tuning: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    feature_types: Mapping[str, str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Compare candidate with tuning while freezing JSD support on train."""
    from synthcity.metrics.eval_statistical import FrozenSupportJSD
    from synthcity.plugins.core.dataloader import GenericDataLoader

    class _TrainFrozenCandidateJSD(FrozenSupportJSD):
        def _support(self, frame, column, feature_type):
            return super()._support(train, column, feature_type)

        def _continuous_edges(self, frame, column):
            return super()._continuous_edges(train, column)

    train_loader = GenericDataLoader(train, feature_types=dict(feature_types or {}))
    evaluator = _TrainFrozenCandidateJSD(feature_types=dict(feature_types or {}))
    value, metadata = evaluator._score_frame(tuning, candidate, train_loader)
    metadata.update({"fit_role": "train", "comparison_role": "tuning"})
    return value, metadata


def _effective_metric_feature_types(
    feature_types: Mapping[str, str] | None,
    release_meta: Mapping[str, Any],
) -> tuple[dict[str, str], list[str]]:
    """Return schema types matching values emitted by release generalization."""
    effective = dict(feature_types or {})
    generalization = release_meta.get("synthetic", {}).get("generalization", {})
    generalized_columns: list[str] = []
    if isinstance(generalization, Mapping):
        for column, spec in generalization.items():
            if isinstance(spec, Mapping) and isinstance(spec.get("intervals"), Sequence):
                generalized_columns.append(str(column))
                effective[str(column)] = "categorical"
    return effective, generalized_columns


def _resolve_utility_policy(policy: Mapping[str, Any] | None = None) -> dict[str, list]:
    """Resolve fixed release utility policy, rejecting unsafe substitutions."""
    expected = list(TUNING_UTILITY_METRICS)
    expected_weights = list(TUNING_UTILITY_WEIGHTS)
    if policy is None:
        return {"metrics": expected, "weights": expected_weights}
    if not isinstance(policy, Mapping):
        raise ValueError("HPO utility_policy must be a mapping")
    metrics = policy.get("metrics")
    weights = policy.get("weights")
    if not isinstance(metrics, Sequence) or isinstance(metrics, (str, bytes)):
        raise ValueError("HPO utility_policy.metrics must be a sequence")
    if not isinstance(weights, Sequence) or isinstance(weights, (str, bytes)):
        raise ValueError("HPO utility_policy.weights must be a sequence")
    metrics = [str(metric) for metric in metrics]
    weights = list(weights)
    if metrics != expected or weights != expected_weights:
        raise ValueError("HPO utility_policy is fixed to equal thirds of TSTR, MMD, and JSD")
    if len(metrics) != len(set(metrics)):
        raise ValueError("HPO utility_policy.metrics must not contain duplicates")
    if any(
        isinstance(weight, bool) or not isinstance(weight, Real) or not math.isfinite(float(weight))
        for weight in weights
    ):
        raise ValueError("HPO utility_policy.weights must be finite real numbers")
    return {"metrics": metrics, "weights": [float(weight) for weight in weights]}


def _canonical_metric_framework(metric_key: str) -> str:
    """Return owning framework for one approved HPO identity."""
    if metric_key == "tstr_macro_f1.v1":
        return "syntheval"
    if metric_key in {"elastic_net_jsd.v1", "mixed_mmd.v1"}:
        return "synthcity"
    raise ValueError(f"Unsupported canonical HPO objective {metric_key!r}")


def evaluate_canonical_hpo_metrics(
    train_df: pd.DataFrame,
    tuning_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    *,
    metric_config: Mapping[str, Sequence[str]],
    target_column: str,
    feature_types: Mapping[str, str] | None = None,
    sensitive_features: Sequence[str] = (),
    seed: int = 0,
    release_generalization: Mapping[str, Any] | None = None,
    utility_policy: Mapping[str, Any] | None = None,
    group_context: Mapping[str, Any] | None = None,
    train_group_ids: Any | None = None,
    tuning_group_ids: Any | None = None,
) -> pd.DataFrame:
    """Evaluate approved HPO identities without native metric aliases.

    Every row carries its producer, fit roles, and support/bandwidth
    provenance. Unsupported release semantics are represented as an explicit
    failed row rather than being replaced by a row-level approximation.
    """
    validate_hpo_metric_config(dict(metric_config))
    policy = _resolve_utility_policy(utility_policy)

    group_mode = group_context.get("group_mode", "row") if group_context else "row"
    group_safety: dict[str, Any] | None = None
    if group_mode == "patient_group":
        if not isinstance(group_context, Mapping) or not isinstance(
            group_context.get("roles"), Mapping
        ):
            raise HPOGroupUnsafeError("patient-group contract is missing")
        if train_group_ids is None or tuning_group_ids is None:
            raise HPOGroupUnsafeError("patient-group IDs are missing")
        try:
            train_ids = _validate_aligned_group_ids(train_df, train_group_ids, "train")
            tuning_ids = _validate_aligned_group_ids(tuning_df, tuning_group_ids, "tuning")
            role_values = {"train": train_ids, "tuning": tuning_ids}
            role_contract: dict[str, Any] = {}
            for role, values in role_values.items():
                declared = group_context["roles"].get(role)
                if not isinstance(declared, Mapping):
                    raise HPOGroupUnsafeError("patient-group role contract is missing")
                fingerprint = dataframe_fingerprint(pd.DataFrame({"group_id": values}))
                if (
                    declared.get("rows") != len(values)
                    or declared.get("groups") != len(set(values))
                    or declared.get("fingerprint") != fingerprint
                    or declared.get("source") != "dataset_role_groups"
                ):
                    raise HPOGroupUnsafeError("patient-group role contract is contradictory")
                role_contract[role] = {
                    "rows": len(values),
                    "groups": len(set(values)),
                    "fingerprint": fingerprint,
                    "source": "dataset_role_groups",
                }
            group_safety = {
                "schema_version": "group-safety-v1",
                "status": "group_safe",
                "group_mode": "patient_group",
                "roles": role_contract,
            }
        except (TypeError, ValueError, KeyError) as exc:
            raise HPOGroupUnsafeError("patient-group contract is malformed") from exc

    def _validate_release_generalization(value: Mapping[str, Any] | None) -> None:
        """Reject release metadata that can alter HPO population semantics."""
        if value is None:
            return
        if not isinstance(value, Mapping):
            raise TypeError("release_generalization must be a mapping or None")
        forbidden_key_tokens = (
            "final_holdout",
            "hidden",
            "privacy",
            "fairness",
            "split",
            "population",
            "evaluation",
        )
        forbidden_roles = {"final_holdout", "hidden", "privacy", "fairness", "split"}

        def walk(current: Any, key: str | None = None) -> None:
            key_lower = key.lower() if key is not None else ""
            if any(token in key_lower for token in forbidden_key_tokens):
                raise ValueError(
                    f"Canonical HPO release_generalization contains forbidden metadata key: {key!r}"
                )
            if isinstance(current, str):
                normalized = current.strip().lower().replace("-", "_")
                if normalized in forbidden_roles:
                    raise ValueError(
                        f"Canonical HPO release_generalization contains forbidden role: {current!r}"
                    )
                return
            if isinstance(current, Mapping):
                for child_key, child_value in current.items():
                    walk(child_value, str(child_key))
            elif isinstance(current, (list, tuple, set)):
                for child_value in current:
                    walk(child_value, key)

        walk(value)

    _validate_release_generalization(release_generalization)

    def _validate_input_provenance(role: str, frame: pd.DataFrame) -> None:
        """Reject release metadata that can smuggle non-HPO populations in."""
        forbidden_roles = {"final_holdout", "hidden", "privacy", "fairness", "split"}
        allowed_roles = {
            "train": {"train"},
            "tuning": {"tuning"},
            "synthetic": {"synthetic", "release"},
        }[role]

        def walk(value: Any, key: str | None = None) -> None:
            key_lower = key.lower() if key is not None else ""
            if any(
                token in key_lower
                for token in ("final_holdout", "hidden", "privacy", "fairness", "split")
            ):
                raise ValueError(
                    f"Canonical HPO {role} input contains forbidden objective metadata: {key!r}"
                )
            if isinstance(value, str):
                normalized = value.strip().lower().replace("-", "_")
                if normalized in forbidden_roles:
                    raise ValueError(
                        f"Canonical HPO {role} input contains forbidden provenance role: {value!r}"
                    )
                if (
                    key_lower.endswith("role") or key_lower == "roles"
                ) and normalized not in allowed_roles:
                    raise ValueError(
                        f"Canonical HPO {role} input contains unsupported provenance role: {value!r}"
                    )
                return
            if isinstance(value, Mapping):
                for child_key, child_value in value.items():
                    walk(child_value, str(child_key))
            elif isinstance(value, (list, tuple, set)):
                for child_value in value:
                    walk(child_value, key)

        for attr_key, attr_value in frame.attrs.items():
            walk(attr_value, str(attr_key))

    for role, frame in (
        ("train", train_df),
        ("tuning", tuning_df),
        ("synthetic", synthetic_df),
    ):
        _validate_input_provenance(role, frame)

    keys = list(policy["metrics"])
    rows: dict[str, dict[str, Any]] = {}

    def _release_provenance(release_meta: Mapping[str, Any]) -> dict[str, Any]:
        """Return complete, role-addressable provenance for one evaluation."""
        role_hashes = {
            "synthetic": release_meta["synthetic"]["role_hash"],
            **{role: value["role_hash"] for role, value in release_meta["roles"].items()},
        }
        return {
            "release_transform_digest": release_meta["synthetic"].get(
                "release_transform_digest",
                release_meta["common_protocol_digest"],
            ),
            "common_protocol_digest": release_meta["common_protocol_digest"],
            "role_hashes": role_hashes,
        }

    for key in keys:
        framework = _canonical_metric_framework(key)
        metadata: dict[str, Any] = {
            "producer": key,
            "framework": framework,
            "error_reason_code": None,
            "fit_roles": ["train"],
            "evaluation_role": "tuning",
            "objective_version": TUNING_OBJECTIVE_VERSION,
            "orientation": ("maximize_score" if key == "tstr_macro_f1.v1" else "minimize_distance"),
            "contracts": {
                "fit_roles": ["train"],
                "comparison_role": "tuning",
                "excluded_roles": ["final_holdout"],
                "privacy": False,
                "fairness": False,
            },
            "provenance": {
                "fit_roles": ["train"],
                "evaluation_role": "tuning",
                "comparison_role": "tuning",
                "excluded_roles": ["final_holdout"],
                "privacy": False,
                "fairness": False,
            },
        }
        try:
            if key == "elastic_net_jsd.v1":
                from synthdata.evaluation.release import transform_release_roles

                released_synthetic, released_roles, release_meta = transform_release_roles(
                    synthetic_df,
                    {"train": train_df, "tuning": tuning_df},
                    release_generalization or {},
                )
                effective_feature_types, generalized_columns = _effective_metric_feature_types(
                    feature_types, release_meta
                )
                value, candidate_metadata = _evaluate_train_frozen_jsd(
                    released_roles["train"],
                    released_roles["tuning"],
                    released_synthetic,
                    feature_types=effective_feature_types,
                )
                metadata.update(
                    {
                        "support": candidate_metadata,
                        "metric_feature_types": effective_feature_types,
                        "original_feature_types": dict(feature_types or {}),
                        "generalized_categorical_columns": generalized_columns,
                        "release_transform_digest": release_meta["synthetic"].get(
                            "release_transform_digest", release_meta["common_protocol_digest"]
                        ),
                        "fit_roles": ["train"],
                        **_release_provenance(release_meta),
                        "orientation": "minimize_distance",
                        "provenance": {
                            "fit_roles": ["train"],
                            "support_fit_roles": ["train"],
                            "metric_feature_types": effective_feature_types,
                            "original_feature_types": dict(feature_types or {}),
                            "generalized_categorical_columns": generalized_columns,
                            "comparison_role": "tuning",
                            **_release_provenance(release_meta),
                        },
                    }
                )
                if value is None:
                    raise HPOMetricNotEligibleError(
                        "FrozenSupportJSD did not produce an eligible aggregate"
                    )
            elif key == "mixed_mmd.v1":
                from synthdata.evaluation.release import transform_release_roles

                released_synthetic, released_roles, release_meta = transform_release_roles(
                    synthetic_df,
                    {"train": train_df, "tuning": tuning_df},
                    release_generalization or {},
                )
                effective_feature_types, generalized_columns = _effective_metric_feature_types(
                    feature_types, release_meta
                )
                continuous = [
                    column
                    for column, kind in effective_feature_types.items()
                    if kind == "continuous"
                ]
                ordinal = [
                    column for column, kind in effective_feature_types.items() if kind == "ordinal"
                ]
                nominal = [
                    column
                    for column, kind in effective_feature_types.items()
                    if kind == "categorical"
                ]
                result = _evaluate_train_frozen_mmd(
                    released_roles["train"],
                    released_roles["tuning"],
                    released_synthetic,
                    continuous_columns=continuous,
                    ordinal_columns=ordinal,
                    nominal_columns=nominal,
                )
                value = result["b_mmd_clip"]
                metadata.update(
                    {
                        "bandwidth": result["bandwidth"],
                        "metric_feature_types": effective_feature_types,
                        "original_feature_types": dict(feature_types or {}),
                        "generalized_categorical_columns": generalized_columns,
                        "fit_roles": ["train"],
                        "evaluation_role": "tuning",
                        "provenance": {
                            "fit_roles": ["train"],
                            "bandwidth_fit_roles": ["train"],
                            "metric_feature_types": effective_feature_types,
                            "original_feature_types": dict(feature_types or {}),
                            "generalized_categorical_columns": generalized_columns,
                            "comparison_role": "tuning",
                            **_release_provenance(release_meta),
                        },
                        "orientation": "minimize_distance",
                    }
                )
                metadata["provenance"].update(_release_provenance(release_meta))
            else:
                from synthdata.evaluation.release import transform_release_roles
                from synthdata.evaluation.tstr import run_tstr_evaluation

                released_synthetic, released_roles, release_meta = transform_release_roles(
                    synthetic_df,
                    {"train": train_df, "tuning": tuning_df},
                    release_generalization or {},
                )
                tstr = run_tstr_evaluation(
                    released_synthetic,
                    released_roles["tuning"],
                    target_column=target_column,
                    evaluation_role="tuning",
                    seed=seed,
                )
                value = tstr.report["macro_f1"]
                metadata.update(
                    {
                        **_release_provenance(release_meta),
                        **tstr.report,
                    }
                )
                metadata["provenance"].update(_release_provenance(release_meta))
                metadata["orientation"] = "maximize_score"
            rows[key] = {
                "mean": float(value),
                "direction": "maximize" if key == "tstr_macro_f1.v1" else "minimize",
                "errors": 0,
                **metadata,
            }
        except HPOMetricNotEligibleError as exc:
            reason_code = "hpo_metric_not_eligible"
            rows[key] = {
                **metadata,
                "mean": float("nan"),
                "direction": "maximize" if key == "tstr_macro_f1.v1" else "minimize",
                "errors": 1,
                "error_type": type(exc).__name__,
                "error_messages": _safe_exception_message(reason_code),
                "error_reason_code": reason_code,
            }
        except (ImportError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            reason_code = "metric_evaluation_exception"
            rows[key] = {
                **metadata,
                "mean": float("nan"),
                "direction": "maximize" if key == "tstr_macro_f1.v1" else "minimize",
                "errors": 1,
                "error_type": type(exc).__name__,
                "error_messages": _safe_exception_message(reason_code),
                "error_reason_code": reason_code,
            }
    report = pd.DataFrame.from_dict(rows, orient="index")
    report.attrs["canonical_hpo"] = True
    report.attrs["canonical_hpo_keys"] = tuple(keys)
    if group_safety is not None:
        report.attrs["group_safety"] = group_safety
    successful_provenance = [
        row.get("provenance")
        for row in rows.values()
        if row.get("errors", 0) == 0 and isinstance(row.get("provenance"), Mapping)
    ]
    if successful_provenance:

        def provenance_part(item: Any) -> dict[str, Any]:
            return {
                field: item.get(field)
                for field in ("release_transform_digest", "common_protocol_digest", "role_hashes")
            }

        provenance_signature = json.dumps(
            provenance_part(successful_provenance[0]),
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        if any(
            json.dumps(
                provenance_part(item),
                sort_keys=True,
                default=str,
                separators=(",", ":"),
            )
            != provenance_signature
            for item in successful_provenance[1:]
        ):
            raise HPOMetricNotEligibleError(
                "Canonical HPO metrics have inconsistent report provenance"
            )
    first_provenance = next(
        (
            row.get("provenance")
            for row in rows.values()
            if isinstance(row.get("provenance"), Mapping)
            and isinstance(row.get("provenance", {}).get("role_hashes"), Mapping)
        ),
        None,
    )
    if isinstance(first_provenance, Mapping):
        report.attrs["hpo_provenance"] = {
            "fit_roles": ["train"],
            "comparison_role": "tuning",
            "release_transform_digest": first_provenance.get("release_transform_digest"),
            "role_hashes": first_provenance.get("role_hashes"),
            "contracts": dict(rows[keys[0]].get("contracts", {})),
            "support_provenance": {
                "fit_roles": ["train"],
                "support_contract": "train_frozen_v1",
                "support": rows.get("elastic_net_jsd.v1", {}).get("support"),
            },
            "bandwidth_provenance": {
                "fit_roles": ["train"],
                "comparison_role": "tuning",
                "contract": "train_frozen_v1",
                "bandwidth": rows.get("mixed_mmd.v1", {}).get("bandwidth"),
            },
            "objective_version": TUNING_OBJECTIVE_VERSION,
        }
    report.attrs["metric_metadata"] = {
        key: {
            **dict(row),
            "metric_name": key,
            "status": "failed" if row.get("errors") else "complete",
        }
        for key, row in rows.items()
    }
    report.attrs["result_metadata"] = dict(report.attrs["metric_metadata"])
    return report


def validate_hpo_metric_config(
    metric_config: dict,
    *,
    group_context: Mapping[str, Any] | None = None,
) -> None:
    """Reject HPO selections without an operational policy contract.

    Patient-group searches may use only metrics whose contract declares that
    the implementation preserves group-safe evaluation semantics.
    """
    if not metric_config:
        raise ValueError(
            "HPO requires an explicit non-empty metric_config with operational objective metrics"
        )
    if group_context is not None and not isinstance(group_context, Mapping):
        raise TypeError("HPO group_context must be a mapping or None")
    group_mode = group_context.get("group_mode", "row") if group_context else "row"
    if group_mode not in {"row", "patient_group"}:
        raise ValueError(f"Unknown HPO evaluation group mode: {group_mode!r}")

    invalid: list[str] = []
    configured_keys = []
    for category, metric_names in metric_config.items():
        if not isinstance(metric_names, (list, tuple)):
            raise ValueError(
                f"HPO metric_config[{category!r}] must be a list of canonical metric identities"
            )
        configured_keys.extend(str(metric_name) for metric_name in metric_names)

    for emitted_key in configured_keys:
        if emitted_key not in HPO_OBJECTIVE_METRICS:
            invalid.append(f"{emitted_key}: not in canonical HPO allowlist")
            continue
        framework = "syntheval" if emitted_key == "tstr_macro_f1.v1" else "synthcity"
        try:
            contract = DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
                framework=framework, emitted_key=emitted_key
            )
        except MetricContractError as exc:
            invalid.append(f"{emitted_key}: {exc}")
            continue
        if (
            contract.lifecycle_state != "operational"
            or "hpo_objective" not in contract.allowed_uses
        ):
            invalid.append(
                f"{emitted_key}: state={contract.lifecycle_state}, allowed={sorted(contract.allowed_uses)}"
            )
        elif group_mode == "patient_group" and contract.group_safety != "group_safe":
            invalid.append(
                f"{emitted_key}: group_safety={contract.group_safety}; patient_group HPO requires group-safe metrics"
            )

    if len(configured_keys) != len(set(configured_keys)):
        raise ValueError("HPO metric_config must not select duplicate metric keys")
    if invalid:
        raise ValueError(
            "HPO metric_config contains metrics that are not approved operational objectives: "
            + "; ".join(invalid)
        )


def hpo_context_digest(context: Mapping[str, Any]) -> str:
    """Hash the complete context that determines an HPO objective."""

    _require_hpo_provenance(context, label="HPO context")

    def _objective_context(value: Any, key: str | None = None) -> Any:
        if key is not None and ("final_holdout" in key.lower() or key.lower() == "holdout"):
            return None
        if isinstance(value, Mapping):
            return {
                str(child_key): _objective_context(child_value, str(child_key))
                for child_key, child_value in value.items()
                if not (
                    "final_holdout" in str(child_key).lower() or str(child_key).lower() == "holdout"
                )
            }
        if isinstance(value, (list, tuple)):
            normalised = [_objective_context(item) for item in value]
            return sorted(
                normalised,
                key=lambda item: json.dumps(
                    item, sort_keys=True, default=str, separators=(",", ":")
                ),
            )
        return value

    return hashlib.sha256(
        json.dumps(
            _objective_context(context), sort_keys=True, default=str, separators=(",", ":")
        ).encode()
    ).hexdigest()


_HPO_PROVENANCE_FIELDS = (
    "release_transform_digest",
    "role_hashes",
    "contracts",
    "support_provenance",
    "bandwidth_provenance",
    "objective_version",
)


def _provenance_value(context: Mapping[str, Any], field: str) -> Any:
    """Resolve provenance fields from hardened canonical contexts."""
    return context.get(field)


def _has_provenance_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (Mapping, Sequence)):
        return bool(value)
    return True


def _contains_blank_string(value: Any) -> bool:
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, Mapping):
        return any(_contains_blank_string(child) for item in value.items() for child in item)
    if isinstance(value, (Sequence, set)):
        return any(_contains_blank_string(child) for child in value)
    return False


def _require_hpo_provenance(context: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """Return required objective provenance or fail closed."""
    if not isinstance(context, Mapping):
        raise TypeError(f"{label} must be a mapping")
    missing = [
        field
        for field in _HPO_PROVENANCE_FIELDS
        if not _has_provenance_value(_provenance_value(context, field))
    ]
    if missing:
        raise ValueError(f"{label} is missing required provenance: {missing}")
    role_hashes = _provenance_value(context, "role_hashes")
    if (
        not isinstance(role_hashes, Mapping)
        or any(
            not isinstance(role_hash, str) or not role_hash.strip()
            for role_hash in role_hashes.values()
        )
        or any(role not in role_hashes for role in ("train", "tuning"))
    ):
        raise ValueError(f"{label}.role_hashes must contain non-empty train and tuning hashes")
    for field in _HPO_PROVENANCE_FIELDS:
        if _contains_blank_string(_provenance_value(context, field)):
            raise ValueError(f"{label}.{field} must not contain blank strings")
    contracts = _provenance_value(context, "contracts")
    if not isinstance(contracts, Mapping):
        raise ValueError(f"{label}.contracts must be an object")
    if (
        contracts.get("fit_roles") != ["train"]
        or contracts.get("comparison_role") != "tuning"
        or contracts.get("excluded_roles") != ["final_holdout"]
        or contracts.get("privacy") is not False
        or contracts.get("fairness") is not False
    ):
        raise ValueError(
            f"{label}.contracts must declare train fit, tuning comparison, exactly "
            "final_holdout exclusion, and disable privacy/fairness"
        )
    support = _provenance_value(context, "support_provenance")
    bandwidth = _provenance_value(context, "bandwidth_provenance")
    if not isinstance(support, Mapping):
        raise ValueError(f"{label}.support_provenance must be an object")
    if (
        support.get("fit_roles") != ["train"]
        or support.get("support_contract") != "train_frozen_v1"
    ):
        raise ValueError(
            f"{label}.support_provenance must declare train-frozen support fit on train"
        )
    if not isinstance(bandwidth, Mapping):
        raise ValueError(f"{label}.bandwidth_provenance must be an object")
    if (
        bandwidth.get("fit_roles") != ["train"]
        or bandwidth.get("comparison_role") != "tuning"
        or bandwidth.get("contract") != "train_frozen_v1"
    ):
        raise ValueError(
            f"{label}.bandwidth_provenance must declare train-frozen fit on train and tuning comparison"
        )
    if (
        not isinstance(_provenance_value(context, "objective_version"), str)
        or not _provenance_value(context, "objective_version").strip()
    ):
        raise ValueError(f"{label}.objective_version must be a non-empty string")
    return {field: _provenance_value(context, field) for field in _HPO_PROVENANCE_FIELDS}


def _require_validated_hpo_context(context: Any, *, label: str) -> dict[str, Any]:
    """Require canonical context before reading or writing durable HPO state."""
    if not isinstance(context, Mapping):
        raise ValueError(f"{label} must be a canonical hpo_context")
    if context.get("schema_version") != HPO_CONTEXT_SCHEMA_VERSION:
        raise ValueError(f"{label} has an unsupported schema")
    fingerprint = context.get("role_context_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint.strip():
        raise ValueError(f"{label}.role_context_fingerprint must be a non-empty string")
    _require_hpo_provenance(context, label=label)
    return dict(context)


def build_hpo_context(
    *,
    task_type: str,
    metric_config: Mapping[str, Sequence[str]],
    registry_digest: str | None,
    stage_a_contract_digest: str | None,
    group_context: Mapping[str, Any] | None,
    role_context_fingerprint: str,
    role_context: Mapping[str, Any],
    variable_columns: Sequence[str] | None = None,
    attack_target_types: Mapping[str, str] | None = None,
    utility_policy: Mapping[str, Any] | None = None,
    release_transform_digest: str | None = None,
    role_hashes: Mapping[str, str] | None = None,
    contracts: Mapping[str, Any] | None = None,
    support_provenance: Mapping[str, Any] | None = None,
    bandwidth_provenance: Mapping[str, Any] | None = None,
    objective_version: str | None = None,
) -> dict[str, Any]:
    """Build the durable identity for one generation HPO objective."""
    if task_type not in {"classification", "regression"}:
        raise ValueError(f"Unsupported HPO task_type {task_type!r}")
    if not isinstance(role_context_fingerprint, str) or not role_context_fingerprint.strip():
        raise ValueError("HPO role_context_fingerprint must be a non-empty string")
    if not isinstance(role_context, Mapping):
        raise TypeError("HPO role_context must be a mapping")
    if group_context is not None and not isinstance(group_context, Mapping):
        raise TypeError("HPO group_context must be a mapping or None")

    resolved_metric_config = {
        str(category): [str(metric_name) for metric_name in metric_names]
        for category, metric_names in metric_config.items()
    }
    validate_hpo_metric_config(resolved_metric_config, group_context=group_context)
    policy = _resolve_utility_policy(utility_policy)
    context = {
        "schema_version": HPO_CONTEXT_SCHEMA_VERSION,
        "task_type": task_type,
        "registry_digest": registry_digest or DEFAULT_METRIC_CONTRACT_REGISTRY.digest(),
        "stage_a_contract_digest": stage_a_contract_digest,
        "metric_config": resolved_metric_config,
        "expected_emitted_keys": list(policy["metrics"]),
        "utility_expected_emitted_keys": list(policy["metrics"]),
        "group_context": dict(group_context) if group_context is not None else None,
        "role_context_fingerprint": role_context_fingerprint,
        "role_context": dict(role_context),
        "objective_version": objective_version,
        "utility_policy": policy,
    }
    context["canonical_hpo"] = is_canonical_hpo_context(
        resolved_metric_config,
        expected_keys=policy["metrics"],
        utility_policy=policy,
    )
    context["canonical_expected_keys"] = list(policy["metrics"]) if context["canonical_hpo"] else []
    context.update(
        {
            "release_transform_digest": release_transform_digest,
            "role_hashes": dict(role_hashes) if role_hashes is not None else None,
            "contracts": dict(contracts) if contracts is not None else None,
            "support_provenance": dict(support_provenance)
            if support_provenance is not None
            else None,
            "bandwidth_provenance": dict(bandwidth_provenance)
            if bandwidth_provenance is not None
            else None,
            "objective_version": objective_version,
        }
    )
    _require_hpo_provenance(context, label="HPO context")
    return context


def _hpo_row_error(row: pd.Series) -> str | None:
    errors = row.get("errors")
    if (
        isinstance(errors, Real)
        and not isinstance(errors, bool)
        and math.isfinite(float(errors))
        and float(errors) > 0
    ):
        return f"error_count={errors}"
    for column in ("error_types", "error_messages"):
        value = row.get(column)
        if value is None:
            continue
        try:
            missing = bool(pd.isna(value))
        except (TypeError, ValueError):
            missing = False
        if not missing and str(value):
            return f"{column}={value}"
    return None


def hpo_score(
    report_df: pd.DataFrame,
    *,
    expected_keys: Sequence[str] | None = None,
    utility_policy: Mapping[str, Any] | None = None,
) -> float:
    """Direction-aware composite score: orient metrics so higher=better, negate mean.

    ``report_df`` must have ``mean`` and ``direction`` columns (as returned by
    synthcity's ``Metrics.evaluate``/``Benchmarks.evaluate``). When
    ``expected_keys`` is supplied, the report must contain exactly one row for
    every statically declared emitted identity before any score is calculated.
    The result is suitable as an Optuna objective under ``direction="minimize"``.
    """
    if not isinstance(report_df, pd.DataFrame):
        raise HPOMetricNotEligibleError("HPO evaluation report has an invalid shape")
    provenance = report_df.attrs.get("hpo_provenance")
    if provenance is None:
        if report_df.attrs.get("canonical_hpo") is True or expected_keys is not None:
            raise HPOMetricNotEligibleError(
                "HPO evaluation report is missing required hpo_provenance"
            )
        raise ValueError("HPO evaluation report is missing required hpo_provenance")
    try:
        _require_hpo_provenance(provenance, label="HPO evaluation provenance")
    except (TypeError, ValueError) as exc:
        if report_df.attrs.get("canonical_hpo") is True or expected_keys is not None:
            raise HPOMetricNotEligibleError(
                "HPO evaluation provenance is not decision-eligible"
            ) from exc
        raise
    policy = _resolve_utility_policy(utility_policy)
    if report_df.empty or report_df.columns.duplicated().any():
        raise HPOMetricNotEligibleError("HPO evaluation emitted no metric rows")
    if "mean" not in report_df.columns or "direction" not in report_df.columns:
        raise HPOMetricNotEligibleError("HPO evaluation must emit mean and direction columns")

    canonical_keys = set(report_df.attrs.get("canonical_hpo_keys", ()))
    if not canonical_keys and expected_keys is not None:
        canonical_keys = set(expected_keys) & HPO_OBJECTIVE_METRICS
    if not canonical_keys and report_df.attrs.get("canonical_hpo") is True:
        canonical_keys = set(HPO_OBJECTIVE_METRICS)
    observed_canonical = {str(key) for key in report_df.index} & canonical_keys
    if report_df.attrs.get("canonical_hpo") is True and (
        not canonical_keys or observed_canonical != canonical_keys
    ):
        raise HPOMetricNotEligibleError(
            "HPO evaluation is not decision-eligible: incomplete metric set; "
            "canonical utility objective requires the complete metric set: "
            f"missing={sorted(canonical_keys - observed_canonical)}"
        )

    if expected_keys is not None:
        required_keys = tuple(expected_keys)
        if len(required_keys) != len(set(required_keys)):
            raise HPOMetricNotEligibleError("HPO expected emitted metric keys must be unique")
        observed_keys = [str(key) for key in report_df.index]
        missing_keys = [key for key in required_keys if key not in observed_keys]
        duplicate_keys = sorted({key for key in observed_keys if observed_keys.count(key) > 1})
        unexpected_keys = sorted(set(observed_keys) - set(required_keys))
        if missing_keys or duplicate_keys or unexpected_keys:
            details = []
            if missing_keys:
                details.append(f"missing={missing_keys}")
            if duplicate_keys:
                details.append(f"duplicate={duplicate_keys}")
            if unexpected_keys:
                details.append(f"unexpected={unexpected_keys}")
            raise HPOMetricNotEligibleError(
                "HPO evaluation emitted an incomplete metric set: " + ", ".join(details)
            )

    scores = []
    legacy_aggregate_scores: list[tuple[str, float]] = []
    candidate_dependent_keys: set[str] = set()
    invalid: list[str] = []
    for emitted_key, row in report_df.iterrows():
        emitted_key = str(emitted_key)
        try:
            framework = "syntheval" if emitted_key == "tstr_macro_f1.v1" else "synthcity"
            contract = DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
                framework=framework, emitted_key=emitted_key
            )
        except MetricContractError as exc:
            invalid.append(f"{emitted_key}: {exc}")
            continue

        error = _hpo_row_error(row)
        raw_value = row.get("mean")
        direction = row.get("direction")
        if error:
            invalid.append(f"{emitted_key}: {error}")
            continue
        if (
            isinstance(raw_value, bool)
            or not isinstance(raw_value, Real)
            or not math.isfinite(float(raw_value))
        ):
            invalid.append(f"{emitted_key}: mean is not a finite real number")
            continue
        if contract.value_role == "diagnostic" and "candidate_independent" in contract.qualifiers:
            continue
        if (
            contract.lifecycle_state != "operational"
            or contract.value_role != "policy_scalar"
            or "hpo_objective" not in contract.allowed_uses
        ):
            invalid.append(
                f"{emitted_key}: state={contract.lifecycle_state}, "
                f"role={contract.value_role}, allowed={sorted(contract.allowed_uses)}"
            )
        elif direction != contract.direction:
            invalid.append(
                f"{emitted_key}: direction={direction!r}, expected={contract.direction!r}"
            )
        else:
            sign = 1.0 if direction == "maximize" else -1.0
            oriented_value = sign * float(raw_value)
            if "legacy_aggregate" in contract.qualifiers:
                legacy_aggregate_scores.append((emitted_key, oriented_value))
            else:
                scores.append(oriented_value)
                if "candidate_dependent" in contract.qualifiers:
                    candidate_dependent_keys.add(emitted_key)

    if invalid:
        raise HPOMetricNotEligibleError(
            "HPO evaluation is not decision-eligible: " + "; ".join(invalid)
        )
    for emitted_key, oriented_value in legacy_aggregate_scores:
        base_key = emitted_key.rsplit(".", 1)[0]
        if any(
            candidate_key.startswith(f"{base_key}.") for candidate_key in candidate_dependent_keys
        ):
            continue
        scores.append(oriented_value)
    if not scores:
        raise HPOMetricNotEligibleError("HPO evaluation emitted no eligible objective rows")
    required_utility_keys = tuple(policy["metrics"])
    observed_keys = [str(key) for key in report_df.index]
    if len(observed_keys) != len(set(observed_keys)):
        raise HPOMetricNotEligibleError("HPO evaluation emitted duplicate utility evidence")
    missing_utility_keys = set(required_utility_keys) - set(observed_keys)
    unexpected_utility_keys = set(observed_keys) - set(required_utility_keys)
    if missing_utility_keys or unexpected_utility_keys:
        raise HPOMetricNotEligibleError(
            "HPO evaluation must emit exactly fixed release utility metrics: "
            f"missing={sorted(missing_utility_keys)}, "
            f"unexpected={sorted(unexpected_utility_keys)}"
        )
    utility = []
    for key in required_utility_keys:
        row = report_df.loc[key]
        value = float(row["mean"])
        if key in {"elastic_net_jsd.v1", "mixed_mmd.v1"}:
            value = 1.0 - value
        utility.append(value)
    if not all(math.isfinite(value) for value in utility):
        raise HPOMetricNotEligibleError("HPO evaluation emitted non-finite utility evidence")
    return -sum(weight * value for weight, value in zip(policy["weights"], utility, strict=True))


def _validate_aligned_group_ids(
    frame: pd.DataFrame,
    group_ids: Any,
    role: str,
) -> list[Any]:
    """Resolve group IDs by stable row identity and reject positional ambiguity."""
    if not frame.index.is_unique:
        raise ValueError(f"Patient-group HPO requires unique {role} row identities")
    if isinstance(group_ids, Mapping):
        values = pd.Series(group_ids, dtype=object)
        if not values.index.is_unique:
            raise ValueError(
                f"Patient-group HPO group IDs for {role} contain duplicate row identities"
            )
        missing = frame.index.difference(values.index)
        extra = values.index.difference(frame.index)
        if len(missing) or len(extra):
            raise ValueError(
                f"Patient-group HPO group IDs for {role} do not align with DataFrame rows"
            )
        values = values.reindex(frame.index)
    elif isinstance(group_ids, pd.Series):
        values = pd.Series(group_ids.to_numpy(copy=False), index=group_ids.index, dtype=object)
    elif isinstance(group_ids, pd.Index):
        values = pd.Series(group_ids.to_numpy(copy=False), dtype=object)
    else:
        try:
            values = pd.Series(group_ids, dtype=object)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Patient-group HPO group IDs for {role} are not row-aligned") from exc
        if not isinstance(frame.index, pd.RangeIndex) or frame.index != pd.RangeIndex(len(frame)):
            raise ValueError(
                f"Patient-group HPO group IDs for {role} require stable row identities, "
                "not positional values"
            )
    if (
        len(values) != len(frame)
        or not values.index.is_unique
        or not values.index.equals(frame.index)
    ):
        raise ValueError(f"Patient-group HPO group IDs for {role} do not align with DataFrame rows")
    if values.isna().any():
        raise ValueError(f"Patient-group HPO group IDs for {role} contain missing values")
    try:
        values.map(hash)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Patient-group HPO group IDs for {role} are not hashable") from exc
    return values.tolist()


def build_synthetic_eval_fn(
    train_reference_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    target_column: str,
    sensitive_features: list,
    metric_config: dict,
    seed: int,
    workspace: str | Path | None = None,
    task_type: str = "classification",
    classification_score: str = "balanced_accuracy",
    feature_types: dict[str, str] | None = None,
    source_table: dict[str, str] | None = None,
    group_context: Mapping[str, Any] | None = None,
    quasi_identifier_columns: list[str] | None = None,
    sensitive_target_types: dict[str, str] | None = None,
    train_group_ids: Any | None = None,
    holdout_group_ids: Any | None = None,
    semantic_context: Mapping[str, Any] | None = None,
    expected_emitted_keys: Sequence[str] | None = None,
    release_generalization: Mapping[str, Any] | None = None,
    utility_policy: Mapping[str, Any] | None = None,
) -> Callable[[pd.DataFrame], float]:
    """Build a ``syn_df -> score`` function for one HPO candidate.

    ``train_reference_df`` is the fit role and ``holdout_df`` is the tuning
    role. Builds a second independent synthetic draw (bootstrap resample) for
    DomiasMIA's reference set, and an augmented fit-role+synthetic set for
    augmentation metrics. No evaluator-internal split is used.

    Canonical identities are evaluated by their canonical producers;
    they must never be translated to native SynthCity aliases.
    """
    validate_hpo_metric_config(metric_config, group_context=group_context)
    configured_keys = {
        str(metric_name) for values in metric_config.values() for metric_name in values
    }
    policy = _resolve_utility_policy(utility_policy)
    expected_keys = list(policy["metrics"])
    if expected_emitted_keys is not None:
        supplied_keys = [str(key) for key in expected_emitted_keys]
        if supplied_keys != expected_keys:
            raise ValueError(
                "HPO expected emitted metric keys are fixed to the release utility policy"
            )
    if len(expected_keys) != len(set(expected_keys)):
        raise ValueError("HPO expected emitted metric keys must be unique")
    group_mode = group_context.get("group_mode", "row") if group_context else "row"
    if group_mode not in {"row", "patient_group"}:
        raise ValueError(f"Invalid group mode {group_mode!r}. Supported: ['row', 'patient_group']")
    if group_mode == "patient_group" and (train_group_ids is None or holdout_group_ids is None):
        raise ValueError(
            "Patient-group HPO objectives require group IDs for both train and tuning loaders"
        )
    if group_mode == "patient_group":
        train_group_ids = _validate_aligned_group_ids(train_reference_df, train_group_ids, "train")
        holdout_group_ids = _validate_aligned_group_ids(holdout_df, holdout_group_ids, "tuning")

    canonical_keys = {
        "elastic_net_jsd.v1",
        "mixed_mmd.v1",
        "tstr_macro_f1.v1",
    }
    configured_keys = {
        str(metric_name) for metric_names in metric_config.values() for metric_name in metric_names
    }
    if configured_keys == canonical_keys:

        def canonical_eval_fn(syn_df: pd.DataFrame) -> float:
            report = evaluate_canonical_hpo_metrics(
                train_reference_df,
                holdout_df,
                syn_df,
                metric_config=metric_config,
                target_column=target_column,
                feature_types=feature_types,
                sensitive_features=sensitive_features,
                seed=seed,
                release_generalization=release_generalization,
                utility_policy=policy,
                group_context=group_context,
                train_group_ids=train_group_ids,
                tuning_group_ids=holdout_group_ids,
            )
            return hpo_score(report, expected_keys=expected_keys, utility_policy=policy)

        return canonical_eval_fn

    from synthcity.metrics import Metrics
    from synthcity.plugins.core.dataloader import GenericDataLoader

    from synthdata.evaluation.synthcity_eval import _semantic_metric_workspace

    def _generated_group_ids(count: int, namespace: str) -> list[tuple[str, int]] | None:
        if group_mode != "patient_group":
            return None
        return [(f"hpo.{namespace}", index) for index in range(count)]

    workspace_path = _semantic_metric_workspace(
        str(workspace) if workspace else None,
        target_column=target_column,
        sensitive_features=sensitive_features,
        metrics=metric_config,
        task_type=task_type,
        evaluation_role="tuning",
        group_mode=group_mode,
        feature_types=feature_types,
        source_table=source_table,
        sensitive_target_types=sensitive_target_types,
        quasi_identifier_columns=quasi_identifier_columns,
        classification_score=classification_score,
        semantic_context=semantic_context,
        structural_n_clusters=None,
        structural_min_rows_per_cluster=10,
        group_ids={"train": train_group_ids, "tuning": holdout_group_ids},
    )

    def _loader(df: pd.DataFrame, group_ids=None) -> GenericDataLoader:
        return GenericDataLoader(
            df,
            target_column=target_column,
            sensitive_features=sensitive_features,
            group_ids=group_ids if group_mode == "patient_group" else None,
            important_features=list(quasi_identifier_columns or []),
            feature_types=dict(feature_types or {}),
            source_table=dict(source_table or {}),
        )

    def eval_fn(syn_df: pd.DataFrame) -> float:
        ref_df = syn_df.sample(n=len(syn_df), replace=True, random_state=seed + 1).reset_index(
            drop=True
        )
        x_aug = pd.concat([train_reference_df, syn_df], ignore_index=True)
        synthetic_group_ids = _generated_group_ids(len(syn_df), "synthetic")
        reference_synthetic_group_ids = _generated_group_ids(len(ref_df), "reference_synthetic")
        augmented_group_ids = (
            list(train_group_ids) + list(synthetic_group_ids)
            if train_group_ids is not None and synthetic_group_ids is not None
            else None
        )
        report = Metrics.evaluate(
            _loader(holdout_df, holdout_group_ids),
            _loader(syn_df, synthetic_group_ids),
            _loader(train_reference_df, train_group_ids),
            _loader(ref_df, reference_synthetic_group_ids),
            _loader(x_aug, augmented_group_ids),
            metrics=metric_config,
            task_type=task_type,
            group_mode=group_mode,
            random_state=seed,
            workspace=workspace_path,
            quasi_identifier_columns=quasi_identifier_columns,
            sensitive_target_types=dict(sensitive_target_types or {}),
            classification_score=classification_score,
            semantic_context=dict(semantic_context) if semantic_context is not None else None,
            X_gt_group_ids=holdout_group_ids,
            X_syn_group_ids=synthetic_group_ids,
            X_train_group_ids=train_group_ids,
            X_ref_syn_group_ids=reference_synthetic_group_ids,
            X_augmented_group_ids=augmented_group_ids,
        )
        if group_mode == "patient_group":
            group_safety = getattr(report, "attrs", {}).get("group_safety")
            if (
                not isinstance(group_safety, Mapping)
                or group_safety.get("schema_version") != "group-safety-v1"
                or group_safety.get("status") != "group_safe"
                or group_safety.get("group_mode") != "patient_group"
            ):
                raise HPOGroupUnsafeError()
        return hpo_score(report, expected_keys=expected_keys, utility_policy=policy)

    return eval_fn


def default_storage_url(output_dir: str | Path) -> str:
    db_path = Path(output_dir) / "optuna_studies.db"
    ensure_dir(db_path.parent)
    return f"sqlite:///{db_path}"


def contextual_study_name(study_name: str, hpo_context: Mapping[str, Any] | None = None) -> str:
    """Return a stable, context-specific Optuna study name."""
    study_name = _validate_study_name(study_name)
    context = _require_validated_hpo_context(hpo_context, label="HPO context")
    return f"{study_name}-{hpo_context_digest(context)[:16]}"


def default_best_params_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "hpo_best_params.json"


def cleanup_hpo_generator_checkpoints(
    workspace: str | Path,
    study: optuna.Study,
    plugin_name: str,
) -> int:
    """Compact completed synthcity HPO generator caches.

    Synthcity stores a fully serialized generator for every benchmark trial.
    Keep the best trial and the highest-numbered trial with a saved generator
    as conservative recovery artifacts once no trial is running; remove the
    other generator caches even when a time-limited study finished below its
    configured target. Metric and synthetic-data caches are intentionally left
    untouched.
    """
    if not plugin_name:
        raise ValueError("plugin_name must not be empty")

    running = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.RUNNING]
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if running:
        logger.info(
            "[%s] HPO still has %d running trial(s); retaining generator checkpoints in %s",
            study.study_name,
            len(running),
            workspace,
        )
        return 0
    if not completed:
        logger.info(
            "[%s] HPO has no completed trials; retaining generator checkpoints in %s",
            study.study_name,
            workspace,
        )
        return 0

    workspace_path = Path(workspace)
    if not workspace_path.is_dir():
        logger.info(
            "[%s] no synthcity checkpoint workspace found at %s",
            study.study_name,
            workspace_path,
        )
        return 0

    trial_numbers = {trial.number for trial in study.trials}
    pattern = re.compile(rf"_trial_(?P<trial>\d+)_{re.escape(plugin_name)}_.*_generator_\d+\.bkp$")
    checkpoints_by_trial: dict[int, list[Path]] = {}
    for path in workspace_path.iterdir():
        if not path.is_file():
            continue
        match = pattern.search(path.name)
        if match is None:
            continue
        trial_number = int(match.group("trial"))
        if trial_number not in trial_numbers:
            continue
        checkpoints_by_trial.setdefault(trial_number, []).append(path)

    if not checkpoints_by_trial:
        logger.info(
            "[%s] no generator checkpoints found for plugin=%s in %s",
            study.study_name,
            plugin_name,
            workspace_path,
        )
        return 0

    keep_trial_numbers = {study.best_trial.number, max(checkpoints_by_trial)}
    to_delete = [
        path
        for trial_number, paths in checkpoints_by_trial.items()
        if trial_number not in keep_trial_numbers
        for path in paths
    ]

    failures: list[tuple[Path, OSError]] = []
    for path in to_delete:
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError as exc:
            failures.append((path, exc))

    if failures:
        details = "; ".join(f"{path}: {exc}" for path, exc in failures)
        logger.error(
            "[%s] failed to delete %d HPO generator checkpoint(s): %s",
            study.study_name,
            len(failures),
            details,
        )
        raise RuntimeError(
            f"Failed to delete {len(failures)} HPO generator checkpoint(s)"
        ) from failures[0][1]

    kept_count = sum(
        len(paths)
        for trial_number, paths in checkpoints_by_trial.items()
        if trial_number in keep_trial_numbers
    )
    logger.info(
        "[%s] HPO checkpoint cleanup complete for plugin=%s: kept %d checkpoint(s) "
        "for trial(s) %s; deleted %d from %s",
        study.study_name,
        plugin_name,
        kept_count,
        sorted(keep_trial_numbers),
        len(to_delete),
        workspace_path,
    )
    return len(to_delete)


def create_study(
    study_name: str,
    hpo_cfg: HPOConfig,
    output_dir: str | Path,
    seed: int,
    *,
    hpo_context: Mapping[str, Any] | None = None,
) -> optuna.Study:
    study_name = _validate_study_name(study_name)
    context_payload = _require_validated_hpo_context(hpo_context, label="HPO study context")
    storage = hpo_cfg.storage or default_storage_url(output_dir)
    contextual_name = contextual_study_name(study_name, context_payload)
    study = optuna.create_study(
        study_name=contextual_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=storage,
        load_if_exists=True,
    )
    context_digest = hpo_context_digest(context_payload)
    stored_digest = study.user_attrs.get("hpo_context_digest")
    stored_context = study.user_attrs.get("hpo_context")
    if stored_digest is None:
        if study.trials:
            raise RuntimeError(
                f"HPO study {study.study_name!r} has trials but no {HPO_CONTEXT_SCHEMA_VERSION} "
                "context metadata; refusing to reuse unverified evidence"
            )
        study.set_user_attr("hpo_context_schema_version", HPO_CONTEXT_SCHEMA_VERSION)
        study.set_user_attr("hpo_context_digest", context_digest)
        study.set_user_attr("hpo_context", context_payload)
    elif stored_digest != context_digest or stored_context != context_payload:
        raise RuntimeError(
            f"HPO study {study.study_name!r} context metadata does not match the current "
            f"{HPO_CONTEXT_SCHEMA_VERSION} payload"
        )
    return study


_HPO_CHECKPOINT_STATE_NAMES = {
    "COMPLETE": "complete",
    "PRUNED": "pruned",
    "FAIL": "failed",
    "RUNNING": "running",
}
_HPO_GENERATOR_EVIDENCE_STATES = frozenset(
    {"not_recorded", "not_attempted", "pending", "missing", "present"}
)
_HPO_PATE_ACCOUNTING_FIELDS = (
    "accountant",
    "requested_epsilon",
    "requested_delta",
    "requested_alpha",
    "requested_lamda",
    "resolved_epsilon",
    "resolved_delta",
    "resolved_alpha",
    "resolved_lamda",
    "effective_epsilon",
    "effective_delta",
    "effective_alpha",
    "effective_lamda",
    "iterations",
    "max_iter",
    "stopping_state",
)


def _validate_hpo_generator_metadata(payload: Any, label: str) -> None:
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{label} must be an object")
    required = (
        "schema_version",
        "generator_context",
        "plugin_name",
        "plugin_fqdn",
        "requested_parameters",
        "n_samples",
        "random_state",
        "privacy_accounting",
    )
    missing = [field for field in required if field not in payload]
    if missing:
        raise RuntimeError(f"{label} is incomplete; missing {missing}")
    schema_version = payload["schema_version"]
    if schema_version != HPO_GENERATOR_METADATA_SCHEMA_VERSION:
        raise RuntimeError(f"{label} has an unsupported schema")
    implementation_fingerprint = payload.get("implementation_fingerprint")
    if schema_version == HPO_GENERATOR_METADATA_SCHEMA_VERSION and (
        not isinstance(implementation_fingerprint, str) or not implementation_fingerprint.strip()
    ):
        raise RuntimeError(f"{label}.implementation_fingerprint must be a non-empty string")
    for field in ("plugin_name", "plugin_fqdn"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise RuntimeError(f"{label}.{field} must be a non-empty string")
    if not isinstance(payload["generator_context"], Mapping):
        raise RuntimeError(f"{label}.generator_context must be an object")
    if not isinstance(payload["requested_parameters"], Mapping):
        raise RuntimeError(f"{label}.requested_parameters must be an object")
    try:
        normalized = normalize_hpo_metadata(payload)
    except HPOMetadataSerializationError as exc:
        raise RuntimeError(f"{label} contains unserializable metadata") from exc
    if normalized != payload:
        raise RuntimeError(f"{label} is not normalized JSON metadata")
    n_samples = payload["n_samples"]
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples <= 0:
        raise RuntimeError(f"{label}.n_samples must be a positive integer")
    random_state = payload["random_state"]
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise RuntimeError(f"{label}.random_state must be an integer")

    context = payload["generator_context"]
    claim_type = context.get("privacy_claim_type")
    if claim_type == "none":
        if payload["privacy_accounting"] is not None:
            raise RuntimeError(f"{label}.privacy_accounting must be null for claim 'none'")
        return
    if claim_type != "formal_dp":
        raise RuntimeError(f"{label}.generator_context has unknown privacy claim {claim_type!r}")
    requested_accounting = context.get("requested_accounting")
    if not isinstance(requested_accounting, Mapping):
        raise RuntimeError(f"{label}.generator_context.requested_accounting must be an object")
    accounting = payload["privacy_accounting"]
    if not isinstance(accounting, Mapping):
        raise RuntimeError(f"{label}.privacy_accounting must be an object for formal_dp")
    missing = [field for field in _HPO_PATE_ACCOUNTING_FIELDS if field not in accounting]
    if missing:
        raise RuntimeError(f"{label}.privacy_accounting is incomplete; missing {missing}")
    for parameter in ("epsilon", "delta", "alpha", "lamda"):
        if accounting.get(f"requested_{parameter}") != requested_accounting.get(parameter):
            raise RuntimeError(
                f"{label}.privacy_accounting.requested_{parameter} does not match "
                "generator_context.requested_accounting"
            )
    for field in (
        "resolved_epsilon",
        "resolved_delta",
        "resolved_alpha",
        "resolved_lamda",
        "effective_epsilon",
        "effective_delta",
        "effective_alpha",
        "effective_lamda",
    ):
        if accounting.get(field) is None:
            raise RuntimeError(f"{label}.privacy_accounting.{field} must be populated")
    if accounting.get("stopping_state") == "not_fitted":
        raise RuntimeError(f"{label}.privacy_accounting.stopping_state cannot be not_fitted")


def _validate_hpo_generator_evidence(
    payload: Any,
    *,
    label: str,
    trial_state: str,
    expected_implementation_fingerprint: str | None = None,
) -> None:
    if payload is None:
        return
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{label} must be an object")
    state = payload.get("state", "not_recorded")
    if state not in _HPO_GENERATOR_EVIDENCE_STATES:
        raise RuntimeError(f"{label}.state has an unknown value {state!r}")
    plugin_name = payload.get("plugin_name")
    if plugin_name is not None and (not isinstance(plugin_name, str) or not plugin_name.strip()):
        raise RuntimeError(f"{label}.plugin_name must be a non-empty string or null")
    claim_type = payload.get("privacy_claim_type")
    if claim_type is not None and claim_type not in {"none", "formal_dp"}:
        raise RuntimeError(f"{label}.privacy_claim_type has an unknown value {claim_type!r}")
    implementation_fingerprint = payload.get("implementation_fingerprint")
    if implementation_fingerprint is not None and (
        not isinstance(implementation_fingerprint, str) or not implementation_fingerprint.strip()
    ):
        raise RuntimeError(f"{label}.implementation_fingerprint must be a non-empty string or null")
    if (
        expected_implementation_fingerprint is not None
        and implementation_fingerprint != expected_implementation_fingerprint
    ):
        raise RuntimeError(
            f"{label}.implementation_fingerprint does not match the current implementation"
        )
    metadata = payload.get("metadata")
    if state == "present":
        _validate_hpo_generator_metadata(metadata, f"{label}.metadata")
        metadata_context = metadata["generator_context"]
        if plugin_name != metadata["plugin_name"]:
            raise RuntimeError(f"{label}.plugin_name does not match its metadata")
        if claim_type != metadata_context["privacy_claim_type"]:
            raise RuntimeError(f"{label}.privacy_claim_type does not match its metadata")
        metadata_fingerprint = metadata.get("implementation_fingerprint")
        if metadata_fingerprint != implementation_fingerprint:
            raise RuntimeError(f"{label}.implementation_fingerprint does not match its metadata")
        if (
            expected_implementation_fingerprint is not None
            and metadata_fingerprint != expected_implementation_fingerprint
        ):
            raise RuntimeError(
                f"{label}.metadata.implementation_fingerprint does not match the current implementation"
            )
    elif metadata is not None:
        raise RuntimeError(f"{label}.metadata must be null unless state is 'present'")
    if trial_state == "complete" and claim_type == "formal_dp" and state != "present":
        raise RuntimeError(
            f"{label} must contain complete formal-DP generator metadata for a completed trial"
        )


def _json_document(payload: Any) -> Any:
    return normalize_hpo_metadata(payload)


def _stage_a_check_json_document(payload: Any) -> Any:
    """Normalize Stage-A check data with standard bounded JSON protections."""
    return _json_document(payload)


def _validate_hpo_trial_checkpoint(
    payload: Any,
    *,
    checkpoint_root: str | Path | None = None,
    stage_a_root: str | Path | None = None,
    expected_study_name: str | None = None,
    expected_trial_number: int | None = None,
    expected_context_digest: str | None = None,
    expected_implementation_fingerprint: str | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise RuntimeError("HPO trial checkpoint must be a JSON object")
    if payload.get("schema_version") != HPO_TRIAL_CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("HPO trial checkpoint has an unsupported schema")
    study_name = payload.get("study_name")
    if not isinstance(study_name, str) or not study_name:
        raise RuntimeError("HPO trial checkpoint study_name must be non-empty")
    try:
        _validate_study_name(study_name)
    except ValueError as exc:
        raise RuntimeError("HPO trial checkpoint study_name is unsafe") from exc
    if expected_study_name is not None and study_name != expected_study_name:
        raise RuntimeError("HPO trial checkpoint study_name does not match the study")
    trial_number = payload.get("trial_number")
    if isinstance(trial_number, bool) or not isinstance(trial_number, int) or trial_number < 0:
        raise RuntimeError("HPO trial checkpoint trial_number must be non-negative")
    if expected_trial_number is not None and trial_number != expected_trial_number:
        raise RuntimeError("HPO trial checkpoint trial_number does not match the path")
    state = payload.get("state")
    if state not in set(_HPO_CHECKPOINT_STATE_NAMES.values()):
        raise RuntimeError(f"HPO trial checkpoint has an invalid state {state!r}")
    objective_value = payload.get("objective_value")
    if state == "complete":
        if (
            isinstance(objective_value, bool)
            or not isinstance(objective_value, Real)
            or not math.isfinite(float(objective_value))
        ):
            raise RuntimeError("Completed HPO trial checkpoint must contain a finite objective")
    elif objective_value is not None:
        raise RuntimeError("Non-completed HPO trial checkpoint must not contain an objective")
    if not isinstance(payload.get("params"), Mapping):
        raise RuntimeError("HPO trial checkpoint params must be an object")

    context = payload.get("hpo_context")
    context_digest = payload.get("hpo_context_digest")
    if context is None:
        raise RuntimeError("HPO trial checkpoint requires canonical hpo_context")
    try:
        validated_context = _require_validated_hpo_context(
            context, label="HPO trial checkpoint hpo_context"
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(str(exc)) from exc
    if context_digest != hpo_context_digest(validated_context):
        raise RuntimeError("HPO trial checkpoint context digest does not match its context")
    if expected_context_digest is None:
        raise RuntimeError("HPO trial checkpoint requires expected context digest")
    if context_digest != expected_context_digest:
        raise RuntimeError("HPO trial checkpoint context does not match the current study")

    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("HPO trial checkpoint metadata must be an object")
    _validate_hpo_generator_evidence(
        metadata.get("generator"),
        label="HPO trial checkpoint metadata.generator",
        trial_state=state,
        expected_implementation_fingerprint=expected_implementation_fingerprint,
    )
    stage_a = metadata.get("stage_a")
    if not isinstance(stage_a, Mapping):
        raise RuntimeError("HPO trial checkpoint stage_a metadata must be an object")
    _validate_stage_a_result_artifact(
        stage_a,
        root=stage_a_root if stage_a_root is not None else checkpoint_root,
        context=validated_context,
        expected_study_name=study_name,
        expected_trial_number=trial_number,
    )
    metric_metadata = metadata.get("metric_metadata")
    expected_metric_keys = validated_context.get("expected_emitted_keys")
    historical_identity_fields = "expected_emitted_keys" not in validated_context
    if not historical_identity_fields and (
        not isinstance(expected_metric_keys, list) or not expected_metric_keys
    ):
        raise RuntimeError("HPO trial checkpoint context has no expected metric identities")
    canonical_checkpoint = bool(expected_metric_keys) and is_canonical_hpo_context(
        validated_context.get("metric_config"),
        expected_keys=expected_metric_keys,
        utility_policy=validated_context.get("utility_policy"),
    )
    # ``hpo-context-v2`` predates explicit canonical identity fields.  Keep
    # those historical contexts readable; enforce new fields when present.
    if (
        "canonical_hpo" in validated_context
        and validated_context.get("canonical_hpo") != canonical_checkpoint
    ):
        raise RuntimeError("HPO trial checkpoint canonical context flag is inconsistent")
    if "canonical_expected_keys" in validated_context and validated_context.get(
        "canonical_expected_keys"
    ) != (list(expected_metric_keys) if canonical_checkpoint else []):
        raise RuntimeError("HPO trial checkpoint canonical metric identities are inconsistent")
    validated_metric_metadata = (
        _validate_bounded_hpo_metric_metadata(metric_metadata, expected_keys=expected_metric_keys)
        if canonical_checkpoint and expected_metric_keys
        else metric_metadata
    )
    result_metadata = metadata.get("result_metadata")
    validated_result_metadata = (
        _validate_bounded_hpo_metric_metadata(result_metadata, expected_keys=expected_metric_keys)
        if canonical_checkpoint and expected_metric_keys
        else result_metadata
    )
    if not canonical_checkpoint and result_metadata is None and metric_metadata is not None:
        validated_result_metadata = metric_metadata
    if (
        not canonical_checkpoint
        and (metric_metadata is not None or result_metadata is not None)
        and metric_metadata != validated_result_metadata
    ):
        raise RuntimeError("HPO trial checkpoint result_metadata does not match metric_metadata")
    if (
        canonical_checkpoint
        and state == "complete"
        and (validated_metric_metadata is None or validated_result_metadata is None)
    ):
        raise RuntimeError("Completed HPO trial checkpoint requires canonical metric metadata")
    if (
        canonical_checkpoint
        and (validated_metric_metadata is not None or validated_result_metadata is not None)
        and validated_metric_metadata != validated_result_metadata
    ):
        raise RuntimeError("HPO trial checkpoint result_metadata does not match metric_metadata")
    canonical_metric_items = (
        validated_metric_metadata if isinstance(validated_metric_metadata, Mapping) else {}
    )
    if canonical_checkpoint and state == "complete":
        if any(
            item["status"] != "complete"
            or not item["finite"]
            or not item["eligible"]
            or item["error_reason_code"] is not None
            for item in canonical_metric_items.values()
        ):
            raise RuntimeError("Completed HPO trial checkpoint lacks complete eligible metrics")
    elif (
        canonical_checkpoint
        and canonical_metric_items
        and any(item["status"] != "failed" for item in canonical_metric_items.values())
    ):
        raise RuntimeError("Non-completed HPO trial checkpoint has completed metric evidence")
    outcome = metadata.get("outcome")
    error_provenance = metadata.get("hpo_error_provenance")
    _validate_hpo_recovery_evidence(
        error_provenance,
        outcome,
        expected_state=state,
        require_provenance=state != "complete",
    )
    return dict(payload)


def persist_hpo_trial_checkpoint(
    root: str | Path,
    study_name: str,
    trial: Any,
    *,
    hpo_context: Mapping[str, Any] | None = None,
    expected_implementation_fingerprint: str | None = None,
    stage_a_root: str | Path | None = None,
) -> Path:
    """Persist one immutable, schema-versioned HPO trial outcome."""
    study_name = _validate_study_name(study_name)
    trial_number = getattr(trial, "number", None)
    if isinstance(trial_number, bool) or not isinstance(trial_number, int) or trial_number < 0:
        raise ValueError("HPO checkpoint trial number must be a non-negative integer")
    state_name = getattr(getattr(trial, "state", None), "name", None)
    state = _HPO_CHECKPOINT_STATE_NAMES.get(state_name)
    if state is None:
        raise RuntimeError(
            f"Cannot persist HPO trial {trial_number}: unsupported Optuna state {state_name!r}"
        )

    context = _json_document(
        _require_validated_hpo_context(hpo_context, label="HPO checkpoint context")
    )
    attrs = getattr(trial, "user_attrs", {}) or {}
    stage_a_state = attrs.get("stage_a_state")
    stage_a_contract_digest = attrs.get("stage_a_contract_digest")
    raw_stage_a_path = attrs.get("stage_a_result_path")
    stage_a_result_path = _checkpoint_stage_a_result_identifier(raw_stage_a_path)
    stage_a_fields_present = any(
        value is not None for value in (stage_a_state, stage_a_contract_digest, raw_stage_a_path)
    )
    if stage_a_state is not None and stage_a_state not in {"passed", "pruned"}:
        raise RuntimeError("HPO checkpoint Stage A state is invalid")
    if stage_a_fields_present and (
        stage_a_result_path is None
        or not isinstance(stage_a_state, str)
        or not isinstance(stage_a_contract_digest, str)
        or not stage_a_contract_digest
    ):
        raise RuntimeError("HPO checkpoint Stage A evidence is incomplete")
    _validate_stage_a_result_artifact(
        {
            "state": stage_a_state,
            "contract_digest": stage_a_contract_digest,
            "result_path": stage_a_result_path,
        },
        root=stage_a_root if stage_a_root is not None else root,
        context=context,
        expected_study_name=study_name,
        expected_trial_number=trial_number,
    )
    metric_metadata = attrs.get("metric_metadata")
    result_metadata = attrs.get("result_metadata")
    if (
        not is_canonical_hpo_context(
            context.get("metric_config"),
            expected_keys=context.get("expected_emitted_keys"),
            utility_policy=context.get("utility_policy"),
        )
        and result_metadata is None
        and metric_metadata is not None
    ):
        result_metadata = metric_metadata
    error_provenance = attrs.get("hpo_error_provenance")
    payload = _json_document(
        {
            "schema_version": HPO_TRIAL_CHECKPOINT_SCHEMA_VERSION,
            "study_name": study_name,
            "trial_number": trial_number,
            "state": state,
            "objective_value": (
                float(trial.value) if state == "complete" and trial.value is not None else None
            ),
            "params": dict(getattr(trial, "params", {}) or {}),
            "hpo_context_digest": hpo_context_digest(context) if context is not None else None,
            "hpo_context": context,
            "metadata": {
                "stage_a": {
                    "state": stage_a_state,
                    "contract_digest": stage_a_contract_digest,
                    "result_path": stage_a_result_path,
                    "prune_reasons": attrs.get("stage_a_prune_reasons", []),
                },
                "metric_metadata": metric_metadata,
                "result_metadata": result_metadata,
                "outcome": attrs.get("hpo_outcome"),
                "hpo_error_provenance": error_provenance,
                "hpo_error_provenance_state": None,
                "generator": {
                    "state": attrs.get("generator_metadata_state", "not_recorded"),
                    "plugin_name": attrs.get("generator_plugin_name"),
                    "privacy_claim_type": attrs.get("generator_privacy_claim_type"),
                    "implementation_fingerprint": attrs.get("generator_implementation_fingerprint"),
                    "metadata": attrs.get("generator_metadata"),
                },
            },
        }
    )
    _validate_hpo_trial_checkpoint(
        payload,
        checkpoint_root=Path(root).resolve(),
        stage_a_root=Path(stage_a_root).resolve() if stage_a_root is not None else None,
        expected_study_name=study_name,
        expected_context_digest=hpo_context_digest(context),
        expected_implementation_fingerprint=expected_implementation_fingerprint,
    )
    path = Path(root) / study_name / f"trial-{trial_number}" / "checkpoint.json"
    _locked_stage_a_json(
        Path(root),
        (study_name, f"trial-{trial_number}"),
        "checkpoint.json",
        payload,
        path,
        artifact_label="trial checkpoint",
    )
    return path


def load_hpo_trial_checkpoint(
    path: str | Path,
    *,
    hpo_context: Mapping[str, Any] | None = None,
    expected_implementation_fingerprint: str | None = None,
    stage_a_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load and validate a durable HPO trial checkpoint."""
    checkpoint_path = Path(path)
    context = _require_validated_hpo_context(hpo_context, label="HPO checkpoint context")
    try:
        expected_study_name = _validate_study_name(checkpoint_path.parents[1].name)
        trial_directory = checkpoint_path.parents[0].name
        expected_trial_number = int(trial_directory.removeprefix("trial-"))
        if not trial_directory.startswith("trial-") or expected_trial_number < 0:
            raise ValueError
        if checkpoint_path.name != "checkpoint.json":
            raise ValueError
    except (IndexError, ValueError) as exc:
        raise RuntimeError("HPO trial checkpoint path has an invalid identity") from exc
    payload = _locked_read_stage_a_json(
        checkpoint_path.parents[2],
        (expected_study_name, trial_directory),
        checkpoint_path.name,
        checkpoint_path,
        artifact_label="trial checkpoint",
    )
    return _validate_hpo_trial_checkpoint(
        payload,
        checkpoint_root=checkpoint_path.parents[2] if len(checkpoint_path.parents) >= 3 else None,
        stage_a_root=stage_a_root,
        expected_study_name=expected_study_name,
        expected_trial_number=expected_trial_number,
        expected_context_digest=hpo_context_digest(context),
        expected_implementation_fingerprint=expected_implementation_fingerprint,
    )


def _persist_hpo_trial_checkpoints(
    output_dir: str | Path,
    study: optuna.Study,
    *,
    hpo_context: Mapping[str, Any] | None,
    expected_implementation_fingerprint: str | None = None,
    stage_a_root: str | Path | None = None,
) -> None:
    for trial in study.trials:
        if getattr(getattr(trial, "state", None), "name", None) in _HPO_CHECKPOINT_STATE_NAMES:
            persist_hpo_trial_checkpoint(
                Path(output_dir) / "hpo_checkpoints",
                study.study_name,
                trial,
                hpo_context=hpo_context,
                expected_implementation_fingerprint=expected_implementation_fingerprint,
                stage_a_root=stage_a_root,
            )


_HPO_RUNNING_RECOVERY_SCHEMA_VERSION = "hpo-running-recovery-v1"


def _hpo_running_recovery_metadata(
    study: optuna.Study,
    trial: Any,
    *,
    context_digest: str,
) -> dict[str, Any]:
    """Return expected metadata identifying one verified stale-trial recovery."""
    return {
        "schema_version": _HPO_RUNNING_RECOVERY_SCHEMA_VERSION,
        "study_name": study.study_name,
        "context_digest": context_digest,
        "trial_number": trial.number,
        "original_state": "RUNNING",
        "reason_code": "stale_running_trial_recovery",
        "terminal_state": "FAIL",
    }


def _repair_recovered_trial_provenance(
    study: optuna.Study,
    trial: Any,
    *,
    validated_provenance: Mapping[str, str],
    validated_outcome: Mapping[str, Any],
) -> None:
    """Add bounded failure evidence for a verified stale-running recovery."""
    attrs = getattr(trial, "user_attrs", {}) or {}
    provenance = attrs.get("hpo_error_provenance")
    outcome = attrs.get("hpo_outcome")
    repaired_outcome = dict(validated_outcome)
    repaired_outcome.update(
        {key: value for key, value in validated_provenance.items() if key not in repaired_outcome}
    )
    repaired_outcome.setdefault("state", "failed")
    if provenance is None:
        study._storage.set_trial_user_attr(  # noqa: SLF001 - durable recovery repair
            trial._trial_id,
            "hpo_error_provenance",
            validated_provenance,
        )
    if repaired_outcome != outcome:
        study._storage.set_trial_user_attr(  # noqa: SLF001 - durable recovery repair
            trial._trial_id,
            "hpo_outcome",
            repaired_outcome,
        )


def _validate_hpo_recovery_evidence(
    provenance: Any,
    value: Any,
    *,
    expected_state: str = "failed",
    require_provenance: bool = True,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Validate all bounded evidence used to recover a stale HPO trial."""
    if provenance is None:
        if require_provenance:
            raise RuntimeError("Non-completed HPO trial checkpoint requires error provenance")
        validated_provenance: dict[str, str] = {}
    else:
        try:
            normalized_provenance = normalize_hpo_metadata(provenance)
        except HPOMetadataSerializationError as exc:
            raise RuntimeError(
                "HPO trial checkpoint hpo_error_provenance is not normalized JSON metadata"
            ) from exc
        if normalized_provenance != provenance:
            raise RuntimeError(
                "HPO trial checkpoint hpo_error_provenance is not normalized JSON metadata"
            )
        validated_provenance = _validate_hpo_error_provenance(provenance)
    if value is None:
        return validated_provenance, {}
    if not isinstance(value, Mapping):
        raise RuntimeError("Recovered HPO trial outcome must be an object")
    allowed = set(validated_provenance) | {"state", "status", "group_safety"}
    if set(value) - allowed:
        raise RuntimeError("Recovered HPO trial outcome has unknown fields")
    provenance_keys = set(validated_provenance)
    present_provenance = provenance_keys.intersection(value)
    if present_provenance and present_provenance != provenance_keys:
        raise RuntimeError("Recovered HPO trial outcome has incomplete provenance")
    if present_provenance and any(
        value[key] != validated_provenance[key] for key in provenance_keys
    ):
        raise RuntimeError("Recovered HPO trial outcome does not match error provenance")
    if "state" in value and value["state"] != expected_state:
        raise RuntimeError("Recovered HPO trial outcome has an invalid state")
    if "status" in value and value["status"] != "group_unsafe":
        raise RuntimeError("Recovered HPO trial outcome has an invalid status")
    if "group_safety" in value:
        group_safety = value["group_safety"]
        if group_safety != {"status": "group_unsafe", "reason_code": "group_unsafe"}:
            raise RuntimeError("Recovered HPO trial outcome has malformed group safety")
        if value.get("status") != "group_unsafe":
            raise RuntimeError("Recovered HPO trial outcome has inconsistent group safety")
    if value.get("status") == "group_unsafe" and "group_safety" not in value:
        raise RuntimeError("Recovered HPO trial outcome is missing group safety")
    if value.get("status") == "group_unsafe" and expected_state == "complete":
        raise RuntimeError("Recovered HPO trial outcome cannot mark a completed trial group unsafe")
    return validated_provenance, dict(value)


def _validated_trial_recovery_evidence(
    trial: Any,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Return validated recovery evidence from one persisted trial."""
    attrs = getattr(trial, "user_attrs", {}) or {}
    existing_provenance = attrs.get("hpo_error_provenance")
    provenance = (
        existing_provenance
        if existing_provenance is not None
        else hpo_exception_provenance(Exception(), location="stale_running_trial_recovery")
    )
    return _validate_hpo_recovery_evidence(provenance, attrs.get("hpo_outcome"))


def _repair_terminal_recovered_trial(
    study: optuna.Study,
    trial: Any,
    *,
    validated_provenance: Mapping[str, str],
    validated_outcome: Mapping[str, Any],
) -> None:
    """Repair terminal RDB trials while retaining their terminal state."""
    storage = study._storage  # noqa: SLF001 - Optuna has no terminal-attr repair API
    backend = getattr(storage, "_backend", None)
    if backend is None or not hasattr(backend, "scoped_session"):
        raise RuntimeError("HPO stale-recovery evidence requires an RDB-backed Optuna study")
    attrs = getattr(trial, "user_attrs", {}) or {}
    provenance = attrs.get("hpo_error_provenance")
    outcome = attrs.get("hpo_outcome")
    repaired_outcome = dict(validated_outcome)
    repaired_outcome.update(
        {key: value for key, value in validated_provenance.items() if key not in repaired_outcome}
    )
    repaired_outcome.setdefault("state", "failed")
    needs_provenance = provenance is None
    needs_outcome = repaired_outcome != outcome
    if not needs_provenance and not needs_outcome:
        return
    from optuna.storages._rdb import models
    from optuna.storages._rdb.storage import _create_scoped_session

    with _create_scoped_session(backend.scoped_session, True) as session:
        stored_trial = models.TrialModel.find_or_raise_by_id(trial._trial_id, session)
        stored_trial.state = optuna.trial.TrialState.RUNNING
        if needs_provenance:
            backend._set_trial_attr_without_commit(  # noqa: SLF001
                session,
                models.TrialUserAttributeModel,
                trial._trial_id,
                "hpo_error_provenance",
                validated_provenance,
            )
        if needs_outcome:
            backend._set_trial_attr_without_commit(  # noqa: SLF001
                session,
                models.TrialUserAttributeModel,
                trial._trial_id,
                "hpo_outcome",
                repaired_outcome,
            )
        stored_trial.state = optuna.trial.TrialState.FAIL
    refreshed = backend.get_trial(trial._trial_id)
    storage._add_trials_to_cache(  # noqa: SLF001 - refresh CachedStorage after direct repair
        trial._study_id if hasattr(trial, "_study_id") else study._study_id,  # noqa: SLF001
        [refreshed],
    )


def _recover_running_trials(
    study: optuna.Study,
    *,
    context: Mapping[str, Any],
) -> int:
    """Terminalize persisted interrupted trials before allocating more work.

    Optuna does not expose a public ``Study`` method for changing a persisted
    trial by number.  Its storage contract does expose the transition by the
    trial's stable storage id, which preserves all existing trial evidence.
    """
    context_digest = hpo_context_digest(context)
    recovered = 0
    for trial in study.trials:
        if trial.state not in {
            optuna.trial.TrialState.RUNNING,
            optuna.trial.TrialState.FAIL,
        }:
            continue
        recovery = _hpo_running_recovery_metadata(study, trial, context_digest=context_digest)
        existing = (trial.user_attrs or {}).get("hpo_running_recovery")
        if existing is not None and existing != recovery:
            raise RuntimeError(
                f"HPO study {study.study_name!r} trial {trial.number} has conflicting "
                "running-trial recovery metadata"
            )
        if trial.state == optuna.trial.TrialState.FAIL:
            if existing == recovery:
                validated_provenance, validated_outcome = _validated_trial_recovery_evidence(trial)
                _repair_terminal_recovered_trial(
                    study,
                    trial,
                    validated_provenance=validated_provenance,
                    validated_outcome=validated_outcome,
                )
            continue

        # Validate every piece of evidence before adding the recovery marker.
        validated_provenance, validated_outcome = _validated_trial_recovery_evidence(trial)
        if existing is None:
            study._storage.set_trial_user_attr(  # noqa: SLF001 - storage transition needs trial id
                trial._trial_id,
                "hpo_running_recovery",
                recovery,  # noqa: SLF001
            )

        _repair_recovered_trial_provenance(
            study,
            trial,
            validated_provenance=validated_provenance,
            validated_outcome=validated_outcome,
        )

        transitioned = study._storage.set_trial_state_values(  # noqa: SLF001
            trial._trial_id,  # noqa: SLF001
            optuna.trial.TrialState.FAIL,
        )
        if not transitioned:
            raise RuntimeError(
                f"HPO study {study.study_name!r} trial {trial.number} remains RUNNING; "
                "refusing to allocate replacement trials"
            )
        recovered += 1
        logger.warning(
            "[%s] recovered stale RUNNING trial=%d as FAIL reason=%s",
            study.study_name,
            trial.number,
            recovery["reason_code"],
        )
    return recovered


def run_study(
    study_name: str,
    objective_fn: Callable[[optuna.Trial], float],
    hpo_cfg: HPOConfig,
    output_dir: str | Path,
    seed: int,
    drop_keys: tuple = ("n_iter",),
    checkpoint_workspace: str | Path | None = None,
    checkpoint_plugin: str | None = None,
    hpo_context: Mapping[str, Any] | None = None,
    checkpoint_implementation_fingerprint: str | None = None,
    stage_a_root: str | Path | None = None,
) -> dict:
    """Run (or resume, via SQLite storage) an Optuna study; return best params.

    ``drop_keys`` are removed from the returned best-params dict: e.g. ``n_iter``
    is capped during search for speed, so the searched value is unreliable and
    generation should fall back to the plugin's own default instead.

    When both checkpoint arguments are provided, synthcity HPO studies with no
    running trial retain only their best and latest generator caches. Studies
    with a running trial or no completed trial retain every cache so an active
    or unsuccessful run remains recoverable.
    """
    study_name = _validate_study_name(study_name)
    context_payload = _require_validated_hpo_context(hpo_context, label="HPO study context")
    if (checkpoint_workspace is None) != (checkpoint_plugin is None):
        raise ValueError("checkpoint_workspace and checkpoint_plugin must be provided together")
    if checkpoint_implementation_fingerprint is not None and (
        not isinstance(checkpoint_implementation_fingerprint, str)
        or not checkpoint_implementation_fingerprint.strip()
    ):
        raise ValueError("checkpoint_implementation_fingerprint must be a non-empty string")

    study = create_study(
        study_name,
        hpo_cfg,
        output_dir,
        seed,
        hpo_context=context_payload,
    )
    _recover_running_trials(study, context=context_payload)

    if checkpoint_implementation_fingerprint is not None:
        checkpoint_dir = Path(output_dir) / "hpo_checkpoints" / study.study_name
        existing_checkpoints = sorted(checkpoint_dir.glob("trial-*/checkpoint.json"))
        for checkpoint_path in existing_checkpoints:
            load_hpo_trial_checkpoint(
                checkpoint_path,
                hpo_context=context_payload,
                expected_implementation_fingerprint=checkpoint_implementation_fingerprint,
                stage_a_root=stage_a_root,
            )
        if existing_checkpoints:
            logger.info(
                "[%s] validated %d resumable HPO checkpoint(s) for implementation=%s",
                study.study_name,
                len(existing_checkpoints),
                checkpoint_implementation_fingerprint[:16],
            )

    def _set_outcome(trial: optuna.Trial, state: str, error: BaseException | None = None) -> None:
        attrs = getattr(trial, "user_attrs", {}) or {}
        existing_outcome = attrs.get("hpo_outcome")
        existing_provenance = attrs.get("hpo_error_provenance")
        provenance = existing_provenance
        if error is not None and provenance is None:
            provenance = hpo_exception_provenance(error, location="tracked_objective")

        require_provenance = state != "complete"
        validated_provenance, validated_outcome = _validate_hpo_recovery_evidence(
            provenance,
            existing_outcome,
            expected_state=state,
            require_provenance=require_provenance,
        )
        outcome = dict(validated_outcome)
        if error is not None:
            outcome.update(validated_provenance)
        outcome["state"] = state
        _validate_hpo_recovery_evidence(
            validated_provenance if provenance is not None else None,
            outcome,
            expected_state=state,
            require_provenance=require_provenance,
        )

        if error is not None and existing_provenance is None:
            trial.set_user_attr("hpo_error_provenance", validated_provenance)
        trial.set_user_attr("hpo_outcome", outcome)

    def _tracked_objective(trial: optuna.Trial) -> float:
        try:
            value = objective_fn(trial)
        except optuna.TrialPruned as exc:
            _set_outcome(trial, "pruned", exc)
            raise
        except (TypeError, ValueError, RuntimeError) as exc:
            _set_outcome(trial, "failed", exc)
            raise
        except Exception as exc:
            _set_outcome(trial, "failed", exc)
            raise
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            error = ValueError(f"HPO objective returned a non-finite or non-real value: {value!r}")
            _set_outcome(trial, "failed", error)
            raise error
        _set_outcome(trial, "complete")
        return float(value)

    def _checkpoint_callback(current_study: optuna.Study, trial: Any) -> None:
        persist_hpo_trial_checkpoint(
            Path(output_dir) / "hpo_checkpoints",
            current_study.study_name,
            trial,
            hpo_context=context_payload,
            expected_implementation_fingerprint=checkpoint_implementation_fingerprint,
            stage_a_root=stage_a_root,
        )

    terminal_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
    }
    n_finished = sum(trial.state in terminal_states for trial in study.trials)
    n_remaining = max(hpo_cfg.n_trials - n_finished, 0)
    try:
        if n_remaining > 0:
            logger.info(
                "[%s] starting hyperparameter optimization: %d trial(s) remaining "
                "(%d already terminal, target=%d)",
                study.study_name,
                n_remaining,
                n_finished,
                hpo_cfg.n_trials,
            )
            study.optimize(
                _tracked_objective,
                n_trials=n_remaining,
                timeout=hpo_cfg.timeout_seconds,
                show_progress_bar=False,
                callbacks=[_checkpoint_callback],
            )
    finally:
        _persist_hpo_trial_checkpoints(
            output_dir,
            study,
            hpo_context=context_payload,
            expected_implementation_fingerprint=checkpoint_implementation_fingerprint,
            stage_a_root=stage_a_root,
        )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        terminal_states = {
            state.name: sum(1 for trial in study.trials if trial.state == state)
            for state in optuna.trial.TrialState
            if any(trial.state == state for trial in study.trials)
        }
        logger.error(
            "[%s] no completed HPO trials; refusing default fallback (states=%s)",
            study.study_name,
            terminal_states,
        )
        raise RuntimeError(
            f"HPO study {study.study_name!r} produced no completed trials; "
            f"terminal states={terminal_states}"
        )

    best = {k: v for k, v in study.best_params.items() if k not in drop_keys}
    logger.info(
        "[%s] best score=%.4f (n_trials=%d) params=%s",
        study.study_name,
        study.best_value,
        len(completed),
        best,
    )
    if checkpoint_workspace is not None and checkpoint_plugin is not None:
        cleanup_hpo_generator_checkpoints(
            checkpoint_workspace,
            study,
            checkpoint_plugin,
        )
    return best


class BestParamsCache:
    """JSON-backed cache of best hyperparameters scoped to verified HPO context."""

    def __init__(
        self,
        path: str | Path,
        *,
        role_context_fingerprint: str | None = None,
        role_context: dict | None = None,
        allow_unverified_legacy: bool = False,
        hpo_context: Mapping[str, Any] | None = None,
    ):
        self.path = Path(path)
        cached_data = load_json(self.path, default={})
        if hpo_context is None:
            raise ValueError("HPO cache requires a verified hpo_context")
        if role_context_fingerprint is not None or role_context is not None:
            raise ValueError(
                "hpo_context cannot be combined with legacy role-context cache arguments"
            )
        context_payload = _require_validated_hpo_context(hpo_context, label="HPO cache context")
        self._context = {
            "schema_version": HPO_CONTEXT_SCHEMA_VERSION,
            "hpo_context_digest": hpo_context_digest(context_payload),
            "hpo_context": context_payload,
        }
        cached_context = cached_data.get("_metadata")
        self._data = cached_data if cached_context == self._context else {}

    def get(self, family: str, model_name: str) -> dict:
        return self._data.get(family, {}).get(model_name, {})

    def has(self, family: str, model_name: str) -> bool:
        return model_name in self._data.get(family, {})

    def set(self, family: str, model_name: str, params: dict) -> None:
        self._data.setdefault(family, {})[model_name] = params
        if self._context is not None:
            self._data["_metadata"] = self._context
        save_json(self.path, self._data)
