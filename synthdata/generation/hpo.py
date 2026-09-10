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
import hashlib
import json
import math
import os
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

import optuna
import pandas as pd

from synthdata.config import HPOConfig
from synthdata.data import dataframe_fingerprint
from synthdata.evaluation.catalog import TASK12_HPO_ALLOWLIST
from synthdata.evaluation.metric_contracts import (
    DEFAULT_METRIC_CONTRACT_REGISTRY,
    MetricContractError,
)
from synthdata.utils import ensure_dir, get_logger, load_json, save_json

logger = get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


STAGE_A_SCREEN_SCHEMA_VERSION = "hpo-stage-a-v1"
HPO_CONTEXT_SCHEMA_VERSION = "hpo-context-v1"
HPO_TRIAL_CHECKPOINT_SCHEMA_VERSION = "hpo-trial-checkpoint-v1"
LEGACY_GENERATOR_METADATA_SCHEMA_VERSION = "generator-metadata-v1"
HPO_GENERATOR_METADATA_SCHEMA_VERSION = "generator-metadata-v2"
STAGE_A_SCREEN_IDS = (
    "shape_schema",
    "bounds_categories",
    "support",
    "dependencies",
    "exact_reuse",
    "subgroup_collapse",
)


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
        dict.fromkeys(
            (*categorical, *protected, *([target_column] if target_is_categorical else []))
        )
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
        values = contract.categorical_values.get(column, ())
        if column not in candidate_df.columns:
            subgroup_reasons.append(f"protected column {column!r} is missing")
            continue
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
    message = str(error) or type(error).__name__
    reason = f"Stage A screen raised {type(error).__name__}: {message}"
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
    trial.set_user_attr("stage_a_result_path", str(result_path))
    trial.set_user_attr("stage_a_prune_reasons", list(result.prune_reasons))
    return result_path


def persist_stage_a_exception(
    root: str | Path,
    study_name: str,
    contract: StageAScreenContract,
    error: TypeError | ValueError | RuntimeError,
) -> Path:
    """Persist a pre-trial Stage A construction failure for study diagnostics."""
    result = _stage_a_exception_result(None, contract, error)
    path = Path(root) / study_name / "construction-failure.json"
    payload = {"study_name": study_name, "trial_number": None, **result.to_dict()}
    if path.exists():
        try:
            cached = load_json(path)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Stage A construction failure at {path} is unreadable") from exc
        if cached != payload:
            raise RuntimeError(
                f"Stage A construction failure at {path} does not match the current failure"
            )
        return path
    _atomic_stage_a_json(path, payload)
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


def _atomic_stage_a_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    os.replace(temporary, path)


def persist_stage_a_contract(root: str | Path, contract: StageAScreenContract) -> Path:
    """Persist the resolved Stage A contract atomically for a study."""
    path = Path(root) / "contract.json"
    payload = {**contract.to_dict(), "digest": contract.digest}
    if path.exists():
        try:
            cached = load_json(path)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Stage A contract at {path} is unreadable") from exc
        if cached != payload:
            raise RuntimeError(f"Stage A contract at {path} does not match the current contract")
        return path
    _atomic_stage_a_json(path, payload)
    return path


def persist_stage_a_result(
    root: str | Path,
    study_name: str,
    trial_number: int,
    result: StageAScreenResult,
) -> Path:
    """Persist one Stage A outcome per trial so prunes are resumable and auditable."""
    if isinstance(trial_number, bool) or trial_number < 0:
        raise ValueError("Stage A trial_number must be a non-negative integer")
    path = Path(root) / study_name / f"trial-{trial_number}" / "result.json"
    payload = {"study_name": study_name, "trial_number": trial_number, **result.to_dict()}
    if path.exists():
        try:
            cached = load_json(path)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Stage A result at {path} is unreadable") from exc
        if cached != payload:
            raise RuntimeError(f"Stage A result at {path} does not match the current result")
        return path
    _atomic_stage_a_json(path, payload)
    return path


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
    if source_df is None or root is None or not study_name:
        raise ValueError("Stage A contract requires source_df, root, and non-empty study_name")
    persist_stage_a_contract(Path(root) / study_name, contract)


HPO_OBJECTIVE_METRICS = frozenset(TASK12_HPO_ALLOWLIST)


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
) -> pd.DataFrame:
    """Evaluate approved HPO identities without native metric aliases.

    Every row carries its producer, fit roles, and support/bandwidth
    provenance. Unsupported release semantics are represented as an explicit
    failed row rather than being replaced by a row-level approximation.
    """
    validate_hpo_metric_config(dict(metric_config))
    keys = [str(key) for values in metric_config.values() for key in values]
    rows: dict[str, dict[str, Any]] = {}
    train_loader: Any = None
    tuning_loader: Any = None
    synthetic_loader: Any = None

    for key in keys:
        framework = _canonical_metric_framework(key)
        metadata: dict[str, Any] = {
            "producer": key,
            "framework": framework,
            "fit_roles": ["train"],
            "evaluation_role": "tuning",
            "provenance": {"fit_roles": ["train"], "evaluation_role": "tuning"},
        }
        try:
            if key == "elastic_net_jsd.v1":
                from synthcity.metrics.eval_statistical import FrozenSupportJSD
                from synthcity.plugins.core.dataloader import GenericDataLoader

                if train_loader is None:
                    train_loader = GenericDataLoader(train_df)
                    tuning_loader = GenericDataLoader(tuning_df)
                    synthetic_loader = GenericDataLoader(synthetic_df)
                evaluator = FrozenSupportJSD(
                    feature_types=dict(feature_types or {}),
                )
                result = evaluator.evaluate_frozen_support(
                    train_loader, tuning_loader, synthetic_loader
                )
                value = result["candidate"]
                metadata.update(
                    {
                        "support": result["metadata"]["candidate"],
                        "fit_roles": ["train"],
                        "provenance": {
                            "fit_roles": ["train"],
                            "support_fit_roles": ["train"],
                        },
                    }
                )
            elif key == "mixed_mmd.v1":
                from syntheval.metrics.utility.metric_max_mean_discrepancy import (
                    mixed_rbf_mmd_v2,
                )

                continuous = [
                    column for column, kind in (feature_types or {}).items() if kind == "continuous"
                ]
                ordinal = [
                    column for column, kind in (feature_types or {}).items() if kind == "ordinal"
                ]
                nominal = [
                    column
                    for column, kind in (feature_types or {}).items()
                    if kind == "categorical"
                ]
                result = mixed_rbf_mmd_v2(
                    train_df,
                    synthetic_df,
                    continuous_columns=continuous,
                    ordinal_columns=ordinal,
                    nominal_columns=nominal,
                )
                value = result["b_mmd_clip"]
                metadata.update(
                    {
                        "bandwidth": result["bandwidth"],
                        "fit_roles": ["train"],
                        "provenance": {
                            "fit_roles": ["train"],
                            "bandwidth_fit_roles": ["train"],
                        },
                    }
                )
            else:
                # TSTR requires a verified release-form candidate. Never pass
                # an ordinary HPO candidate through as a synthetic release.
                raise ValueError(
                    "tstr_macro_f1.v1 requires a verified release-form candidate; "
                    "ordinary HPO candidates cannot use row-level fallback semantics"
                )
            rows[key] = {
                "mean": float(value),
                "direction": "maximize" if key == "tstr_macro_f1.v1" else "minimize",
                **metadata,
            }
        except (ImportError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            rows[key] = {
                "mean": float("nan"),
                "direction": "maximize" if key == "tstr_macro_f1.v1" else "minimize",
                "errors": 1,
                "error_messages": str(exc),
                **metadata,
            }
    report = pd.DataFrame.from_dict(rows, orient="index")
    report.attrs["canonical_hpo"] = True
    report.attrs["canonical_hpo_keys"] = tuple(keys)
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
    return hashlib.sha256(
        json.dumps(dict(context), sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()


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
) -> dict[str, Any]:
    """Build the durable identity for one generation HPO objective."""
    if task_type not in {"classification", "regression"}:
        raise ValueError(f"Unsupported HPO task_type {task_type!r}")
    if not isinstance(role_context_fingerprint, str) or not role_context_fingerprint:
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
    return {
        "schema_version": HPO_CONTEXT_SCHEMA_VERSION,
        "task_type": task_type,
        "registry_digest": registry_digest or DEFAULT_METRIC_CONTRACT_REGISTRY.digest(),
        "stage_a_contract_digest": stage_a_contract_digest,
        "metric_config": resolved_metric_config,
        "expected_emitted_keys": [
            str(metric_name)
            for metric_names in resolved_metric_config.values()
            for metric_name in metric_names
        ],
        "group_context": dict(group_context) if group_context is not None else None,
        "role_context_fingerprint": role_context_fingerprint,
        "role_context": dict(role_context),
    }


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
        if value is not None and not pd.isna(value) and str(value):
            return f"{column}={value}"
    return None


def hpo_score(
    report_df: pd.DataFrame,
    *,
    expected_keys: Sequence[str] | None = None,
) -> float:
    """Direction-aware composite score: orient metrics so higher=better, negate mean.

    ``report_df`` must have ``mean`` and ``direction`` columns (as returned by
    synthcity's ``Metrics.evaluate``/``Benchmarks.evaluate``). When
    ``expected_keys`` is supplied, the report must contain exactly one row for
    every statically declared emitted identity before any score is calculated.
    The result is suitable as an Optuna objective under ``direction="minimize"``.
    """
    if report_df.empty:
        raise ValueError("HPO evaluation emitted no metric rows")
    if "mean" not in report_df.columns or "direction" not in report_df.columns:
        raise ValueError("HPO evaluation must emit mean and direction columns")

    canonical_keys = set(report_df.attrs.get("canonical_hpo_keys", ()))
    if not canonical_keys and expected_keys is not None:
        canonical_keys = set(expected_keys) & HPO_OBJECTIVE_METRICS
    if not canonical_keys and report_df.attrs.get("canonical_hpo") is True:
        canonical_keys = set(HPO_OBJECTIVE_METRICS)
    observed_canonical = {str(key) for key in report_df.index} & canonical_keys
    if report_df.attrs.get("canonical_hpo") is True and (
        not canonical_keys or observed_canonical != canonical_keys
    ):
        raise ValueError(
            "HPO evaluation is not decision-eligible: incomplete metric set; "
            "canonical utility objective requires the complete metric set: "
            f"missing={sorted(canonical_keys - observed_canonical)}"
        )

    if expected_keys is not None:
        required_keys = tuple(expected_keys)
        if len(required_keys) != len(set(required_keys)):
            raise ValueError("HPO expected emitted metric keys must be unique")
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
            raise ValueError(
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
        raise ValueError("HPO evaluation is not decision-eligible: " + "; ".join(invalid))
    for emitted_key, oriented_value in legacy_aggregate_scores:
        base_key = emitted_key.rsplit(".", 1)[0]
        if any(
            candidate_key.startswith(f"{base_key}.") for candidate_key in candidate_dependent_keys
        ):
            continue
        scores.append(oriented_value)
    if not scores:
        raise ValueError("HPO evaluation emitted no eligible objective rows")
    return -sum(scores) / len(scores)


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
) -> Callable[[pd.DataFrame], float]:
    """Build a ``syn_df -> score`` function for one HPO candidate.

    ``train_reference_df`` is the fit role and ``holdout_df`` is the tuning
    role. Builds a second independent synthetic draw (bootstrap resample) for
    DomiasMIA's reference set, and an augmented fit-role+synthetic set for
    augmentation metrics. No evaluator-internal split is used.

    Canonical Task 12 identities are evaluated by their canonical producers;
    they must never be translated to native SynthCity aliases.
    """
    validate_hpo_metric_config(metric_config, group_context=group_context)
    expected_keys = (
        list(expected_emitted_keys)
        if expected_emitted_keys is not None
        else [
            str(metric_name)
            for metric_names in metric_config.values()
            for metric_name in metric_names
        ]
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

    canonical_keys = {
        "elastic_net_jsd.v1",
        "mixed_mmd.v1",
        "tstr_macro_f1.v1",
    }
    configured_keys = {
        str(metric_name) for metric_names in metric_config.values() for metric_name in metric_names
    }
    if configured_keys.issubset(canonical_keys):

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
            )
            return hpo_score(report, expected_keys=expected_keys)

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
        return hpo_score(report, expected_keys=expected_keys)

    return eval_fn


def default_storage_url(output_dir: str | Path) -> str:
    db_path = Path(output_dir) / "optuna_studies.db"
    ensure_dir(db_path.parent)
    return f"sqlite:///{db_path}"


def contextual_study_name(study_name: str, hpo_context: Mapping[str, Any] | None = None) -> str:
    """Return a stable, context-specific Optuna study name."""
    if not isinstance(study_name, str) or not study_name:
        raise ValueError("study_name must be a non-empty string")
    if hpo_context is None:
        return study_name
    return f"{study_name}-{hpo_context_digest(hpo_context)[:16]}"


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
    storage = hpo_cfg.storage or default_storage_url(output_dir)
    contextual_name = contextual_study_name(study_name, hpo_context)
    study = optuna.create_study(
        study_name=contextual_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=storage,
        load_if_exists=True,
    )
    if hpo_context is not None:
        context_payload = dict(hpo_context)
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
    if schema_version not in {
        LEGACY_GENERATOR_METADATA_SCHEMA_VERSION,
        HPO_GENERATOR_METADATA_SCHEMA_VERSION,
    }:
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
    return json.loads(json.dumps(payload, sort_keys=True, default=str, allow_nan=False))


def _validate_hpo_trial_checkpoint(
    payload: Any,
    *,
    expected_study_name: str | None = None,
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
    if expected_study_name is not None and study_name != expected_study_name:
        raise RuntimeError("HPO trial checkpoint study_name does not match the study")
    trial_number = payload.get("trial_number")
    if isinstance(trial_number, bool) or not isinstance(trial_number, int) or trial_number < 0:
        raise RuntimeError("HPO trial checkpoint trial_number must be non-negative")
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
        if context_digest is not None:
            raise RuntimeError("HPO trial checkpoint has a context digest without context")
    elif not isinstance(context, Mapping):
        raise RuntimeError("HPO trial checkpoint hpo_context must be an object or null")
    elif context_digest != hpo_context_digest(context):
        raise RuntimeError("HPO trial checkpoint context digest does not match its context")
    if expected_context_digest is not None and context_digest != expected_context_digest:
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
    metric_metadata = metadata.get("metric_metadata")
    if metric_metadata is not None and not isinstance(metric_metadata, Mapping):
        raise RuntimeError("HPO trial checkpoint metric_metadata must be an object or null")
    result_metadata = metadata.get("result_metadata")
    if result_metadata is not None and not isinstance(result_metadata, Mapping):
        raise RuntimeError("HPO trial checkpoint result_metadata must be an object or null")
    if (
        metric_metadata is not None
        and result_metadata is not None
        and dict(metric_metadata) != dict(result_metadata)
    ):
        raise RuntimeError("HPO trial checkpoint result_metadata does not match metric_metadata")
    outcome = metadata.get("outcome")
    if outcome is not None and not isinstance(outcome, Mapping):
        raise RuntimeError("HPO trial checkpoint outcome must be an object or null")
    return dict(payload)


def persist_hpo_trial_checkpoint(
    root: str | Path,
    study_name: str,
    trial: Any,
    *,
    hpo_context: Mapping[str, Any] | None = None,
    expected_implementation_fingerprint: str | None = None,
) -> Path:
    """Persist one immutable, schema-versioned HPO trial outcome."""
    if not isinstance(study_name, str) or not study_name:
        raise ValueError("HPO checkpoint study_name must be non-empty")
    trial_number = getattr(trial, "number", None)
    if isinstance(trial_number, bool) or not isinstance(trial_number, int) or trial_number < 0:
        raise ValueError("HPO checkpoint trial number must be a non-negative integer")
    state_name = getattr(getattr(trial, "state", None), "name", None)
    state = _HPO_CHECKPOINT_STATE_NAMES.get(state_name)
    if state is None:
        raise RuntimeError(
            f"Cannot persist HPO trial {trial_number}: unsupported Optuna state {state_name!r}"
        )

    context = _json_document(dict(hpo_context)) if hpo_context is not None else None
    attrs = getattr(trial, "user_attrs", {}) or {}
    metric_metadata = attrs.get("metric_metadata")
    result_metadata = attrs.get("result_metadata", metric_metadata)
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
                    "state": attrs.get("stage_a_state"),
                    "contract_digest": attrs.get("stage_a_contract_digest"),
                    "result_path": attrs.get("stage_a_result_path"),
                    "prune_reasons": attrs.get("stage_a_prune_reasons", []),
                },
                "metric_metadata": metric_metadata,
                "result_metadata": result_metadata,
                "outcome": attrs.get("hpo_outcome"),
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
        expected_study_name=study_name,
        expected_context_digest=(hpo_context_digest(context) if context is not None else None),
        expected_implementation_fingerprint=expected_implementation_fingerprint,
    )
    path = Path(root) / study_name / f"trial-{trial_number}" / "checkpoint.json"
    if path.exists():
        try:
            cached = load_json(path)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"HPO trial checkpoint at {path} is unreadable") from exc
        _validate_hpo_trial_checkpoint(
            cached,
            expected_study_name=study_name,
            expected_context_digest=(hpo_context_digest(context) if context is not None else None),
            expected_implementation_fingerprint=expected_implementation_fingerprint,
        )
        if cached != payload:
            raise RuntimeError(f"HPO trial checkpoint at {path} does not match the current trial")
        return path
    _atomic_stage_a_json(path, payload)
    return path


def load_hpo_trial_checkpoint(
    path: str | Path,
    *,
    expected_implementation_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Load and validate a durable HPO trial checkpoint."""
    checkpoint_path = Path(path)
    try:
        payload = load_json(checkpoint_path)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"HPO trial checkpoint at {checkpoint_path} is unreadable") from exc
    return _validate_hpo_trial_checkpoint(
        payload,
        expected_implementation_fingerprint=expected_implementation_fingerprint,
    )


def _persist_hpo_trial_checkpoints(
    output_dir: str | Path,
    study: optuna.Study,
    *,
    hpo_context: Mapping[str, Any] | None,
    expected_implementation_fingerprint: str | None = None,
) -> None:
    for trial in study.trials:
        if getattr(getattr(trial, "state", None), "name", None) in _HPO_CHECKPOINT_STATE_NAMES:
            persist_hpo_trial_checkpoint(
                Path(output_dir) / "hpo_checkpoints",
                study.study_name,
                trial,
                hpo_context=hpo_context,
                expected_implementation_fingerprint=expected_implementation_fingerprint,
            )


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
        hpo_context=hpo_context,
    )
    if checkpoint_implementation_fingerprint is not None:
        checkpoint_dir = Path(output_dir) / "hpo_checkpoints" / study.study_name
        existing_checkpoints = sorted(checkpoint_dir.glob("trial-*/checkpoint.json"))
        for checkpoint_path in existing_checkpoints:
            load_hpo_trial_checkpoint(
                checkpoint_path,
                expected_implementation_fingerprint=checkpoint_implementation_fingerprint,
            )
        if existing_checkpoints:
            logger.info(
                "[%s] validated %d resumable HPO checkpoint(s) for implementation=%s",
                study.study_name,
                len(existing_checkpoints),
                checkpoint_implementation_fingerprint[:16],
            )

    def _set_outcome(trial: optuna.Trial, state: str, error: BaseException | None = None) -> None:
        existing_outcome = getattr(trial, "user_attrs", {}).get("hpo_outcome")
        outcome = dict(existing_outcome) if isinstance(existing_outcome, Mapping) else {}
        outcome["state"] = state
        if error is not None:
            outcome.update(
                {
                    "error_type": type(error).__name__,
                    "error_message": str(error) or type(error).__name__,
                }
            )
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
            hpo_context=hpo_context,
            expected_implementation_fingerprint=checkpoint_implementation_fingerprint,
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
            hpo_context=hpo_context,
            expected_implementation_fingerprint=checkpoint_implementation_fingerprint,
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
    """JSON-backed cache of best hyperparameters with optional role provenance."""

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
        self._context = None
        if hpo_context is not None:
            if role_context_fingerprint is not None or role_context is not None:
                raise ValueError(
                    "hpo_context cannot be combined with legacy role-context cache arguments"
                )
            context_payload = dict(hpo_context)
            self._context = {
                "schema_version": HPO_CONTEXT_SCHEMA_VERSION,
                "hpo_context_digest": hpo_context_digest(context_payload),
                "hpo_context": context_payload,
            }
            cached_context = cached_data.get("_metadata")
            if cached_context == self._context:
                self._data = cached_data
            else:
                logger.info(
                    "[hpo cache] ignoring stale or unverified cache at %s for hpo_context=%s",
                    self.path,
                    self._context["hpo_context_digest"][:16],
                )
                self._data = {}
        elif role_context_fingerprint is None:
            self._data = cached_data
        else:
            self._context = {
                "schema_version": "hpo-cache-v1",
                "role_context_fingerprint": role_context_fingerprint,
                "role_context": role_context,
            }
            cached_context = cached_data.get("_metadata")
            context_matches = (
                isinstance(cached_context, dict)
                and cached_context.get("schema_version") == "hpo-cache-v1"
                and cached_context.get("role_context_fingerprint") == role_context_fingerprint
            )
            if context_matches or allow_unverified_legacy and "_metadata" not in cached_data:
                self._data = cached_data
            else:
                logger.info(
                    "[hpo cache] ignoring stale or unverified cache at %s for role_context=%s",
                    self.path,
                    role_context_fingerprint[:16],
                )
                self._data = {}

    def get(self, family: str, model_name: str) -> dict:
        return self._data.get(family, {}).get(model_name, {})

    def has(self, family: str, model_name: str) -> bool:
        return model_name in self._data.get(family, {})

    def set(self, family: str, model_name: str, params: dict) -> None:
        self._data.setdefault(family, {})[model_name] = params
        if self._context is not None:
            self._data["_metadata"] = self._context
        save_json(self.path, self._data)
