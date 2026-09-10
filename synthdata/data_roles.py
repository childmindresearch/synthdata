"""Deterministic role allocation and population-identity validation.

This module keeps population assignment separate from model-frame construction.
Patient identifiers are accepted only long enough to assign complete groups;
returned role frames never contain those identifiers.
"""

import dataclasses
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd

from synthdata.config import DataSplitConfig
from synthdata.utils import get_logger

logger = get_logger(__name__)

ROLE_NAMES = ("train", "tuning", "final_holdout")


@dataclasses.dataclass
class PopulationIdentity:
    """Resolved non-modeling population identity aligned to source rows."""

    groups: pd.Series | None
    model_frame: pd.DataFrame
    row_keys: pd.Series
    source: str
    metadata: dict[str, Any]
    identity_sidecar: pd.Series | None = None


@dataclasses.dataclass
class RoleAssignment:
    """Three-role frames and auditable assignment metadata."""

    frames: dict[str, pd.DataFrame]
    groups: dict[str, pd.Series | None]
    assignment: pd.DataFrame
    metadata: dict[str, Any]
    assignment_fingerprint: str
    assignment_policy_fingerprint: str


class _CandidateRecord(TypedDict):
    candidate_number: int
    positions_by_role: dict[str, np.ndarray]
    frames: dict[str, pd.DataFrame]
    violations: list[str]
    ratio_errors: dict[str, float]
    target_balance_error: float
    encounter_metrics: dict[str, float]
    encounter_balance_error: float


def _stable_value(value: Any) -> str:
    """Serialize a scalar consistently for validation and fingerprints."""
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


TOKENIZATION_ALGORITHM = "hmac-sha256"
TOKENIZATION_VERSION = "population-group-token-v1"
PATIENT_ID_HMAC_KEY_ENV = "SYNTHDATA_PATIENT_ID_HMAC_KEY"


def _opaque_group_token(value: Any, scope: str, *, token_secret: str | bytes) -> str:
    """Return deterministic, opaque token for one population identifier."""
    key = token_secret.encode() if isinstance(token_secret, str) else token_secret
    message = f"{TOKENIZATION_VERSION}:{scope}:{_stable_value(value)}".encode()
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def _required_token_secret() -> str:
    secret = os.environ.get(PATIENT_ID_HMAC_KEY_ENV, "")
    if not secret.strip():
        raise ValueError(
            f"Canonical patient_group identity requires non-empty {PATIENT_ID_HMAC_KEY_ENV}; "
            "set this external secret before loading patient-group data"
        )
    return secret


def _payload_fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _series_fingerprint(series: pd.Series) -> str:
    payload = [_stable_value(value) for value in series.tolist()]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_identity_series(series: pd.Series, description: str) -> None:
    if series.isna().any():
        raise ValueError(f"{description} contains missing values")
    non_scalar = [value for value in series.tolist() if not pd.api.types.is_scalar(value)]
    if non_scalar:
        raise ValueError(
            f"{description} must contain scalar values; found non-scalar value(s): {non_scalar[:3]}"
        )
    empty = [value for value in series.tolist() if isinstance(value, str) and not value.strip()]
    if empty:
        raise ValueError(f"{description} contains empty or whitespace-only values")


def _read_mapping(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Patient identity mapping file not found: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Failed to read patient identity mapping {path}: {exc}") from exc
    if path.suffix.lower() == ".csv":
        try:
            return pd.read_csv(path, low_memory=False)
        except (OSError, pd.errors.ParserError) as exc:
            raise ValueError(f"Failed to read patient identity mapping {path}: {exc}") from exc
    raise ValueError(
        f"Unsupported patient identity mapping extension {path.suffix!r}; "
        "expected .csv, .parquet, or .pq"
    )


def resolve_population_identity(
    df: pd.DataFrame,
    split: DataSplitConfig | None,
    *,
    token_scope: str = "default-population-scope",
) -> PopulationIdentity:
    """Resolve and remove the configured population identifier from ``df``."""
    row_keys = pd.Series(np.arange(len(df), dtype=np.int64), index=df.index, name="row_key")
    if split is None or split.mode == "row":
        return PopulationIdentity(
            groups=None,
            model_frame=df.copy(),
            row_keys=row_keys,
            source="row",
            metadata={"mode": "row", "population_unit": "row"},
            identity_sidecar=None,
        )

    if split.mode != "patient_group":
        raise ValueError(f"Unsupported data.split.mode: {split.mode!r}")

    if split.one_row_per_patient:
        raise ValueError(
            "one_row_per_patient is not a leakage-safe identity source; configure "
            "patient_id_column or an approved identity mapping sidecar"
        )

    token_secret = _required_token_secret()

    if split.patient_id_column is not None:
        column = split.patient_id_column
        if column not in df.columns:
            raise KeyError(
                f"data.split.patient_id_column {column!r} is missing from source data; "
                f"available columns: {list(df.columns)}"
            )
        groups = df[column].copy()
        _validate_identity_series(groups, f"Patient identifier column {column!r}")
        normalized = groups.map(
            lambda value: _opaque_group_token(value, token_scope, token_secret=token_secret)
        )
        model_frame = df.drop(columns=[column]).copy()
        metadata = {
            "mode": "patient_group",
            "population_unit": "patient_group",
            "identity_source": "source_column",
            "identity_column": column,
            "identity_fingerprint": _series_fingerprint(normalized),
            "n_population_groups": int(normalized.nunique()),
            "raw_identifier_persisted": False,
            "tokenization_algorithm": TOKENIZATION_ALGORITHM,
            "tokenization_version": TOKENIZATION_VERSION,
            "tokenization_scope_fingerprint": hashlib.sha256(token_scope.encode()).hexdigest(),
        }
        return PopulationIdentity(
            groups=normalized.rename("population_group"),
            model_frame=model_frame,
            row_keys=row_keys,
            source="source_column",
            metadata=metadata,
            identity_sidecar=groups.rename("patient_id"),
        )

    if split.identity_mapping_path is None:
        raise ValueError(
            "patient_group mode requires patient_id_column, identity_mapping_path, "
            "or an approved identity mapping sidecar"
        )
    row_column = split.mapping_row_key_column
    patient_column = split.mapping_patient_key_column
    if row_column is None or patient_column is None:
        raise ValueError(
            "mapping_row_key_column and mapping_patient_key_column are required for "
            "identity_mapping_path"
        )
    if row_column not in df.columns:
        raise KeyError(
            f"Mapping row-key column {row_column!r} is missing from source data; "
            f"available columns: {list(df.columns)}"
        )

    mapping_path = Path(split.identity_mapping_path).expanduser()
    mapping = _read_mapping(mapping_path)
    missing_mapping_columns = [
        column for column in (row_column, patient_column) if column not in mapping.columns
    ]
    if missing_mapping_columns:
        raise KeyError(
            f"Patient identity mapping {mapping_path} is missing column(s) "
            f"{missing_mapping_columns}; available columns: {list(mapping.columns)}"
        )
    _validate_identity_series(
        df[row_column],
        f"Source mapping row-key column {row_column!r}",
    )
    _validate_identity_series(
        mapping[row_column],
        f"Patient identity mapping row-key column {row_column!r}",
    )
    _validate_identity_series(
        mapping[patient_column],
        f"Patient identity mapping column {patient_column!r}",
    )
    source_keys = df[row_column].map(_stable_value)
    mapping_keys = mapping[row_column].map(_stable_value)
    if source_keys.duplicated().any():
        raise ValueError(f"Source mapping row-key column {row_column!r} contains duplicate values")
    if mapping_keys.duplicated().any():
        raise ValueError(
            f"Patient identity mapping row-key column {row_column!r} contains duplicate values"
        )
    source_key_set = set(source_keys.tolist())
    mapping_key_set = set(mapping_keys.tolist())
    if source_key_set != mapping_key_set:
        raise ValueError(
            "Patient identity mapping does not provide complete one-to-one source coverage: "
            f"missing={len(source_key_set - mapping_key_set)}, "
            f"extra={len(mapping_key_set - source_key_set)}"
        )
    mapped_patients = mapping[patient_column]
    key_to_patient = dict(zip(mapping_keys, mapped_patients, strict=True))
    groups = source_keys.map(key_to_patient)
    _validate_identity_series(groups, "Patient identity mapping assignments")
    safe_groups = groups.map(
        lambda value: _opaque_group_token(value, token_scope, token_secret=token_secret)
    )
    if patient_column in df.columns and patient_column != row_column:
        source_patients = df[patient_column].copy()
        _validate_identity_series(
            source_patients,
            f"Source mapping patient-key column {patient_column!r}",
        )
        source_patient_values = source_patients.map(_stable_value)
        mapped_patient_values = groups.map(_stable_value)
        conflicts = source_patient_values.ne(mapped_patient_values).to_numpy()
        if conflicts.any():
            conflict_row_keys = df[row_column].to_numpy()[conflicts].tolist()
            raise ValueError(
                f"Source mapping patient-key column {patient_column!r} conflicts with the "
                f"identity mapping for row key value(s): {conflict_row_keys[:5]}"
            )
    identity_columns = [row_column]
    if patient_column in df.columns and patient_column != row_column:
        identity_columns.append(patient_column)
    model_frame = df.drop(columns=identity_columns).copy()
    metadata = {
        "mode": "patient_group",
        "population_unit": "patient_group",
        "identity_source": "mapping_file",
        "mapping_path": str(mapping_path),
        "mapping_fingerprint": _file_fingerprint(mapping_path),
        "mapping_row_key_column": row_column,
        "mapping_patient_key_column": patient_column,
        "identity_columns_removed": identity_columns,
        "identity_fingerprint": _series_fingerprint(safe_groups),
        "n_population_groups": int(safe_groups.nunique()),
        "raw_identifier_persisted": False,
        "tokenization_algorithm": TOKENIZATION_ALGORITHM,
        "tokenization_version": TOKENIZATION_VERSION,
        "tokenization_scope_fingerprint": hashlib.sha256(token_scope.encode()).hexdigest(),
    }
    return PopulationIdentity(
        groups=safe_groups.rename("population_group"),
        model_frame=model_frame,
        row_keys=row_keys,
        source="mapping_file",
        metadata=metadata,
        identity_sidecar=groups.rename("patient_id"),
    )


def _match_configured_value(values: list[Any], configured: Any) -> Any:
    exact = [value for value in values if value == configured]
    if len(exact) == 1:
        return exact[0]
    string_matches = [value for value in values if str(value) == str(configured)]
    if len(string_matches) == 1:
        return string_matches[0]
    return None


def resolve_support_constraints(
    frame: pd.DataFrame,
    target_column: str,
    protected_columns: list[str],
    split: DataSplitConfig,
) -> dict[str, Any]:
    """Resolve configured support floors against observed source vocabularies."""
    target_values = frame[target_column].dropna().unique().tolist()
    if not target_values:
        raise ValueError(f"target column {target_column!r} has no observed values")

    class_minimums: dict[Any, int] = {}
    for value in target_values:
        override = _match_configured_value(list(split.class_count_overrides), value)
        if override is not None:
            class_minimums[value] = max(
                split.minimum_class_count,
                split.class_count_overrides[override],
            )
        else:
            class_minimums[value] = split.minimum_class_count
    unknown_class_keys = [
        key
        for key in split.class_count_overrides
        if _match_configured_value(target_values, key) is None
    ]
    if unknown_class_keys:
        raise ValueError(
            "data.split.class_count_overrides contains unknown target value(s): "
            f"{unknown_class_keys}; observed values={target_values}"
        )

    protected_minimums: dict[str, dict[Any, int]] = {}
    unexpected_protected_columns = sorted(
        set(split.protected_group_count_overrides) - set(protected_columns)
    )
    if unexpected_protected_columns:
        raise ValueError(
            "data.split.protected_group_count_overrides contains unexpected protected "
            f"column(s): {unexpected_protected_columns}; expected={protected_columns}"
        )
    for column in protected_columns:
        if column not in frame.columns:
            raise KeyError(f"Protected attribute {column!r} is missing from modeling data")
        values = frame[column].dropna().unique().tolist()
        configured_overrides = split.protected_group_count_overrides.get(column, {})
        if (
            column not in split.protected_group_count_overrides
            and split.protected_group_count_overrides
        ):
            raise ValueError(
                f"data.split.protected_group_count_overrides contains no entry for {column!r}; "
                f"configured columns={list(split.protected_group_count_overrides)}"
            )
        protected_minimums[column] = {}
        for value in values:
            override = _match_configured_value(list(configured_overrides), value)
            protected_minimums[column][value] = max(
                split.minimum_protected_group_count,
                configured_overrides[override] if override is not None else 0,
            )
        unknown_keys = [
            key for key in configured_overrides if _match_configured_value(values, key) is None
        ]
        if unknown_keys:
            raise ValueError(
                f"data.split.protected_group_count_overrides[{column!r}] contains unknown "
                f"group value(s): {unknown_keys}; observed values={values}"
            )

    cell_minimums: dict[str, dict[Any, dict[Any, int]]] = {}
    unexpected_cell_columns = sorted(
        set(split.target_by_protected_group_count_overrides) - set(protected_columns)
    )
    if unexpected_cell_columns:
        raise ValueError(
            "data.split.target_by_protected_group_count_overrides contains unexpected "
            f"protected column(s): {unexpected_cell_columns}; expected={protected_columns}"
        )
    for column in protected_columns:
        target_values_for_column = target_values
        configured_cells = split.target_by_protected_group_count_overrides.get(column, {})
        if (
            column not in split.target_by_protected_group_count_overrides
            and split.target_by_protected_group_count_overrides
        ):
            raise ValueError(
                f"data.split.target_by_protected_group_count_overrides contains no entry for "
                f"{column!r}"
            )
        protected_values = frame[column].dropna().unique().tolist()
        unknown_protected_keys = [
            key
            for key in configured_cells
            if _match_configured_value(protected_values, key) is None
        ]
        if unknown_protected_keys:
            raise ValueError(
                f"data.split.target_by_protected_group_count_overrides[{column!r}] contains "
                f"unknown group value(s): {unknown_protected_keys}; observed values="
                f"{protected_values}"
            )
        cell_minimums[column] = {}
        for protected_value in protected_values:
            configured_targets = configured_cells.get(protected_value)
            if configured_targets is None:
                configured_targets = next(
                    (
                        configured_cells[key]
                        for key in configured_cells
                        if str(key) == str(protected_value)
                    ),
                    {},
                )
            cell_minimums[column][protected_value] = {}
            for target_value in target_values_for_column:
                override = _match_configured_value(list(configured_targets), target_value)
                cell_minimums[column][protected_value][target_value] = max(
                    split.minimum_target_by_protected_group_count,
                    configured_targets[override] if override is not None else 0,
                )
            unknown_targets = [
                key
                for key in configured_targets
                if _match_configured_value(target_values_for_column, key) is None
            ]
            if unknown_targets:
                raise ValueError(
                    f"data.split.target_by_protected_group_count_overrides[{column!r}]"
                    f"[{protected_value!r}] contains unknown target value(s): {unknown_targets}"
                )

    return {
        "minimum_class_count": {str(value): minimum for value, minimum in class_minimums.items()},
        "minimum_protected_group_count": {
            column: {str(value): minimum for value, minimum in values.items()}
            for column, values in protected_minimums.items()
        },
        "minimum_target_by_protected_group_count": {
            column: {
                str(protected_value): {
                    str(target_value): minimum
                    for target_value, minimum in target_values_for_group.items()
                }
                for protected_value, target_values_for_group in values.items()
            }
            for column, values in cell_minimums.items()
        },
        "_class_minimums": class_minimums,
        "_protected_minimums": protected_minimums,
        "_cell_minimums": cell_minimums,
    }


def _role_counts(total_rows: int, split: DataSplitConfig) -> dict[str, int]:
    fractions = np.array(
        [split.train_fraction, split.tuning_fraction, split.final_holdout_fraction], dtype=float
    )
    raw_counts = total_rows * fractions
    counts = np.floor(raw_counts).astype(int)
    remainder = total_rows - int(counts.sum())
    order = np.argsort(-(raw_counts - counts), kind="stable")
    for index in order[:remainder]:
        counts[index] += 1
    if any(count == 0 for count in counts):
        raise ValueError(
            "Three-role split requires at least one row in every role; "
            f"rows={total_rows}, requested_counts={counts.tolist()}"
        )
    return dict(zip(ROLE_NAMES, counts, strict=True))


def _summary_for_role(
    frame: pd.DataFrame,
    target_column: str,
    groups: pd.Series | None,
    protected_columns: list[str],
) -> dict[str, Any]:
    target_counts = {
        str(value): int(count)
        for value, count in frame[target_column].value_counts(dropna=False).items()
    }
    protected_counts = {
        column: {
            str(value): int(count)
            for value, count in frame[column].value_counts(dropna=False).items()
        }
        for column in protected_columns
    }
    target_by_protected = {}
    for column in protected_columns:
        column_counts = {
            str(protected_value): {} for protected_value in frame[column].dropna().unique().tolist()
        }
        for group_key, count in (
            frame.groupby([column, target_column], dropna=False, observed=False).size().items()
        ):
            if not isinstance(group_key, tuple) or len(group_key) != 2:
                raise TypeError("groupby key must contain protected and target values")
            protected_value, target_value = group_key
            column_counts.setdefault(str(protected_value), {})[str(target_value)] = int(count)
        target_by_protected[column] = column_counts
    return {
        "rows": int(len(frame)),
        "groups": int(groups.nunique()) if groups is not None else None,
        "target_counts": target_counts,
        "protected_group_counts": protected_counts,
        "target_by_protected_group_counts": target_by_protected,
    }


def _constraint_violations(
    frame: pd.DataFrame,
    target_column: str,
    protected_columns: list[str],
    constraints: dict[str, Any],
) -> list[str]:
    violations = []
    target_counts = frame[target_column].value_counts(dropna=False).to_dict()
    for value, minimum in constraints["_class_minimums"].items():
        actual = int(target_counts.get(value, 0))
        if actual < minimum:
            violations.append(f"class {value!r}: {actual} < {minimum}")
    for column in protected_columns:
        counts = frame[column].value_counts(dropna=False).to_dict()
        for value, minimum in constraints["_protected_minimums"][column].items():
            actual = int(counts.get(value, 0))
            if actual < minimum:
                violations.append(f"{column} group {value!r}: {actual} < {minimum}")
        cells = (
            frame.groupby([column, target_column], dropna=False, observed=False).size().to_dict()
        )
        for protected_value, target_minimums in constraints["_cell_minimums"][column].items():
            for target_value, minimum in target_minimums.items():
                actual = int(cells.get((protected_value, target_value), 0))
                if actual < minimum:
                    violations.append(
                        f"{column}={protected_value!r}, target={target_value!r}: "
                        f"{actual} < {minimum}"
                    )
    return violations


def _target_balance_error(
    role_frames: dict[str, pd.DataFrame],
    frame: pd.DataFrame,
    target_column: str,
    split: DataSplitConfig,
) -> float:
    expected_fractions = dict(
        zip(
            ROLE_NAMES,
            [split.train_fraction, split.tuning_fraction, split.final_holdout_fraction],
            strict=True,
        )
    )
    full_distribution = frame[target_column].value_counts(normalize=True, dropna=False)
    errors = []
    for role, role_frame in role_frames.items():
        role_distribution = role_frame[target_column].value_counts(normalize=True, dropna=False)
        aligned = role_distribution.reindex(full_distribution.index, fill_value=0.0)
        errors.append(float(np.abs(aligned - full_distribution).mean()))
        errors.append(abs(len(role_frame) / len(frame) - expected_fractions[role]))
    return float(np.mean(errors)) if errors else 0.0


def _build_row_candidate(
    frame: pd.DataFrame,
    target_column: str,
    counts: dict[str, int],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    fractions = np.array(
        [counts[role] / len(frame) for role in ROLE_NAMES],
        dtype=float,
    )
    target_values = frame[target_column].unique().tolist()
    assignments: dict[str, list[int]] = {role: [] for role in ROLE_NAMES}
    target_counts = {role: {value: 0 for value in target_values} for role in ROLE_NAMES}
    desired_target_counts = {
        role: {
            value: len(frame[frame[target_column] == value]) * fractions[index]
            for value in target_values
        }
        for index, role in enumerate(ROLE_NAMES)
    }

    for value in target_values:
        positions = np.flatnonzero(frame[target_column].to_numpy() == value)
        positions = rng.permutation(positions)
        raw_counts = len(positions) * fractions
        stratum_counts = np.floor(raw_counts).astype(int)
        remainder = len(positions) - int(stratum_counts.sum())
        order = np.argsort(-(raw_counts - stratum_counts), kind="stable")
        for index in order[:remainder]:
            stratum_counts[index] += 1
        cursor = 0
        for index, role in enumerate(ROLE_NAMES):
            selected = positions[cursor : cursor + stratum_counts[index]]
            assignments[role].extend(selected.tolist())
            target_counts[role][value] += len(selected)
            cursor += stratum_counts[index]

    while True:
        role_surplus = {role: len(assignments[role]) - counts[role] for role in ROLE_NAMES}
        overfull = [role for role in ROLE_NAMES if role_surplus[role] > 0]
        underfull = [role for role in ROLE_NAMES if role_surplus[role] < 0]
        if not overfull and not underfull:
            break
        if not overfull or not underfull:
            raise RuntimeError("Stratified row allocation could not satisfy role counts")
        source_role = max(
            overfull,
            key=lambda role: (role_surplus[role], -ROLE_NAMES.index(role)),
        )
        destination_role = min(
            underfull,
            key=lambda role: (role_surplus[role], ROLE_NAMES.index(role)),
        )
        value = max(
            target_values,
            key=lambda candidate: (
                desired_target_counts[destination_role][candidate]
                - target_counts[destination_role][candidate],
                str(candidate),
            ),
        )
        source_positions = assignments[source_role]
        move_index = next(
            index
            for index, position in enumerate(source_positions)
            if frame.iloc[position][target_column] == value
        )
        position = source_positions.pop(move_index)
        assignments[destination_role].append(position)
        target_counts[source_role][value] -= 1
        target_counts[destination_role][value] += 1

    return {role: np.array(sorted(positions), dtype=int) for role, positions in assignments.items()}


def _encounter_balance_metrics(
    role_frames: dict[str, pd.DataFrame],
    frame: pd.DataFrame,
    target_column: str,
    split: DataSplitConfig,
) -> dict[str, float]:
    encounter_column = split.encounter_label_column
    if encounter_column is None:
        return {"encounter_count_error": 0.0, "encounter_target_balance_error": 0.0}
    if encounter_column not in frame.columns:
        raise KeyError(
            f"data.split.encounter_label_column {encounter_column!r} is missing from the split frame"
        )
    if frame[encounter_column].isna().any():
        raise ValueError(
            f"data.split.encounter_label_column {encounter_column!r} contains missing values"
        )
    encounter_values = frame[encounter_column].unique().tolist()
    target_values = frame[target_column].unique().tolist()
    full_counts = frame[encounter_column].value_counts().to_dict()
    count_errors = []
    target_errors = []
    fractions = dict(
        zip(
            ROLE_NAMES,
            [split.train_fraction, split.tuning_fraction, split.final_holdout_fraction],
            strict=True,
        )
    )
    for role, role_frame in role_frames.items():
        role_counts = role_frame[encounter_column].value_counts().to_dict()
        for encounter_value in encounter_values:
            total = full_counts[encounter_value]
            count_errors.append(abs(role_counts.get(encounter_value, 0) / total - fractions[role]))
            full_distribution = (
                frame.loc[frame[encounter_column] == encounter_value, target_column]
                .value_counts(normalize=True)
                .reindex(target_values, fill_value=0.0)
            )
            role_distribution = (
                role_frame.loc[role_frame[encounter_column] == encounter_value, target_column]
                .value_counts(normalize=True)
                .reindex(target_values, fill_value=0.0)
            )
            target_errors.append(float(np.abs(role_distribution - full_distribution).mean()))
    return {
        "encounter_count_error": float(np.mean(count_errors)) if count_errors else 0.0,
        "encounter_target_balance_error": (float(np.mean(target_errors)) if target_errors else 0.0),
    }


def _build_group_candidate(
    frame: pd.DataFrame,
    target_column: str,
    group_values: pd.Series,
    split: DataSplitConfig,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    normalized_groups = group_values
    group_positions = {
        group: np.flatnonzero(normalized_groups.to_numpy() == group)
        for group in normalized_groups.unique().tolist()
    }
    if len(group_positions) < len(ROLE_NAMES):
        raise ValueError(
            "patient_group split requires at least one complete population group per role; "
            f"found {len(group_positions)} group(s)"
        )
    target_row_counts = _role_counts(len(frame), split)
    target_values = frame[target_column].value_counts(dropna=False)
    full_target_distribution = frame[target_column].value_counts(normalize=True, dropna=False)
    target_fractions = np.array(
        [split.train_fraction, split.tuning_fraction, split.final_holdout_fraction],
        dtype=float,
    )
    group_target_counts = {
        group: frame.iloc[positions][target_column].value_counts(dropna=False)
        for group, positions in group_positions.items()
    }
    encounter_column = split.encounter_label_column
    encounter_values = (
        frame[encounter_column].unique().tolist() if encounter_column is not None else []
    )
    full_encounter_counts = (
        frame[encounter_column].value_counts().to_dict() if encounter_column is not None else {}
    )
    full_encounter_target_distributions = (
        {
            encounter: frame.loc[frame[encounter_column] == encounter, target_column]
            .value_counts(normalize=True)
            .reindex(target_values.index, fill_value=0.0)
            for encounter in encounter_values
        }
        if encounter_column is not None
        else {}
    )
    group_encounter_counts = (
        {
            group: frame.iloc[positions][encounter_column].value_counts().to_dict()
            for group, positions in group_positions.items()
        }
        if encounter_column is not None
        else {}
    )
    group_encounter_target_counts = {}
    if encounter_column is not None:
        for group, positions in group_positions.items():
            group_frame = frame.iloc[positions]
            group_encounter_target_counts[group] = {
                encounter: group_frame.loc[
                    group_frame[encounter_column] == encounter, target_column
                ].value_counts(dropna=False)
                for encounter in encounter_values
            }
    group_order = rng.permutation(list(group_positions))
    assignments: dict[str, list[int]] = {role: [] for role in ROLE_NAMES}
    current_counts = {role: 0 for role in ROLE_NAMES}
    current_target_counts = {
        role: {value: 0 for value in target_values.index.tolist()} for role in ROLE_NAMES
    }
    current_encounter_counts = {
        role: {encounter: 0 for encounter in encounter_values} for role in ROLE_NAMES
    }
    current_encounter_target_counts = {
        role: {
            encounter: {value: 0 for value in target_values.index.tolist()}
            for encounter in encounter_values
        }
        for role in ROLE_NAMES
    }
    for group_number, group in enumerate(group_order):
        positions = group_positions[group].tolist()
        if group_number < len(ROLE_NAMES):
            role = ROLE_NAMES[group_number]
        else:
            group_targets = group_target_counts[group]

            def assignment_cost(
                candidate: str,
                *,
                positions: list[int] = positions,
                group_targets: pd.Series = group_targets,
                group_name: str = group,
            ) -> tuple[float, float, float, float, int]:
                role_index = ROLE_NAMES.index(candidate)
                proposed_rows = current_counts[candidate] + len(positions)
                row_error = abs(proposed_rows / len(frame) - target_fractions[role_index])
                proposed_distribution = []
                for value, _total in target_values.items():
                    proposed = current_target_counts[candidate][value] + int(
                        group_targets.get(value, 0)
                    )
                    proposed_distribution.append(
                        abs(proposed / max(proposed_rows, 1) - full_target_distribution[value])
                    )
                target_error = float(np.mean(proposed_distribution))
                encounter_error = 0.0
                if encounter_column is not None:
                    encounter_count_errors = []
                    encounter_target_errors = []
                    for encounter in encounter_values:
                        proposed_encounter_count = current_encounter_counts[candidate][
                            encounter
                        ] + int(group_encounter_counts[group_name].get(encounter, 0))
                        encounter_count_errors.append(
                            abs(
                                proposed_encounter_count / max(full_encounter_counts[encounter], 1)
                                - target_fractions[role_index]
                            )
                        )
                        proposed_encounter_targets = [
                            current_encounter_target_counts[candidate][encounter][value]
                            + int(
                                group_encounter_target_counts[group_name][encounter].get(value, 0)
                            )
                            for value in target_values.index
                        ]
                        encounter_target_errors.append(
                            float(
                                np.mean(
                                    np.abs(
                                        np.asarray(proposed_encounter_targets)
                                        / max(proposed_encounter_count, 1)
                                        - full_encounter_target_distributions[encounter].to_numpy()
                                    )
                                )
                            )
                        )
                    encounter_error = float(
                        np.mean(encounter_count_errors) + np.mean(encounter_target_errors)
                    )
                current_fill_ratio = current_counts[candidate] / max(
                    target_row_counts[candidate], 1
                )
                return current_fill_ratio, encounter_error, target_error, row_error, role_index

            role = min(ROLE_NAMES, key=assignment_cost)
        assignments[role].extend(positions)
        current_counts[role] += len(positions)
        for value in target_values.index:
            current_target_counts[role][value] += int(group_target_counts[group].get(value, 0))
        if encounter_column is not None:
            for encounter in encounter_values:
                current_encounter_counts[role][encounter] += int(
                    group_encounter_counts[group].get(encounter, 0)
                )
                for value in target_values.index:
                    current_encounter_target_counts[role][encounter][value] += int(
                        group_encounter_target_counts[group][encounter].get(value, 0)
                    )
    return {role: np.array(sorted(positions), dtype=int) for role, positions in assignments.items()}


def allocate_roles(
    frame: pd.DataFrame,
    target_column: str,
    split: DataSplitConfig,
    protected_columns: list[str] | None = None,
    groups: pd.Series | None = None,
    seed: int = 0,
) -> RoleAssignment:
    """Allocate deterministic train/tuning/final-holdout roles."""
    if target_column not in frame.columns:
        raise KeyError(f"Target column {target_column!r} is missing from split frame")
    if frame[target_column].isna().any():
        raise ValueError(
            f"Target column {target_column!r} contains missing values; "
            "three-role allocation requires an observed target"
        )
    protected_columns = list(protected_columns or [])
    if split.encounter_label_column is not None:
        if split.encounter_label_column not in frame.columns:
            raise KeyError(
                f"Encounter label column {split.encounter_label_column!r} is missing from split frame"
            )
        if frame[split.encounter_label_column].isna().any():
            raise ValueError(
                f"Encounter label column {split.encounter_label_column!r} contains missing values"
            )
        if split.encounter_label_column == target_column:
            raise ValueError("Encounter label column must be distinct from the target column")
    constraints = resolve_support_constraints(frame, target_column, protected_columns, split)
    counts = _role_counts(len(frame), split)
    rng_seed = split.seed if split.seed is not None else seed
    candidate_records: list[_CandidateRecord] = []
    n_candidates = split.candidate_count
    for candidate_number in range(n_candidates):
        rng = np.random.default_rng(rng_seed + candidate_number)
        if groups is None:
            positions_by_role = _build_row_candidate(frame, target_column, counts, rng)
        else:
            positions_by_role = _build_group_candidate(frame, target_column, groups, split, rng)
        candidate_frames = {
            role: frame.iloc[positions].reset_index(drop=True)
            for role, positions in positions_by_role.items()
        }
        violations = [
            f"{role}: {violation}"
            for role, candidate_frame in candidate_frames.items()
            for violation in _constraint_violations(
                candidate_frame, target_column, protected_columns, constraints
            )
        ]
        ratio_errors = {
            role: abs(len(candidate_frames[role]) / len(frame) - fraction)
            for role, fraction in zip(
                ROLE_NAMES,
                [split.train_fraction, split.tuning_fraction, split.final_holdout_fraction],
                strict=True,
            )
        }
        target_error = _target_balance_error(candidate_frames, frame, target_column, split)
        encounter_metrics = _encounter_balance_metrics(
            candidate_frames,
            frame,
            target_column,
            split,
        )
        encounter_error = (
            encounter_metrics["encounter_count_error"]
            + encounter_metrics["encounter_target_balance_error"]
        )
        candidate_records.append(
            {
                "candidate_number": candidate_number,
                "positions_by_role": positions_by_role,
                "frames": candidate_frames,
                "violations": violations,
                "ratio_errors": ratio_errors,
                "target_balance_error": target_error,
                "encounter_metrics": encounter_metrics,
                "encounter_balance_error": encounter_error,
            }
        )

    feasible = [
        record
        for record in candidate_records
        if not record["violations"]
        and max(record["ratio_errors"].values()) <= split.ratio_tolerance
        and record["target_balance_error"] <= split.target_balance_tolerance
        and record["encounter_balance_error"] <= split.encounter_balance_tolerance
    ]
    if not feasible:
        best = min(
            candidate_records,
            key=lambda record: (
                len(record["violations"]),
                max(record["ratio_errors"].values()),
                record["target_balance_error"],
                record["encounter_balance_error"],
                record["candidate_number"],
            ),
        )
        raise ValueError(
            "No deterministic three-role assignment satisfies the configured support/balance "
            f"constraints after {n_candidates} candidate(s). Best candidate="
            f"{best['candidate_number']}, violations={best['violations'][:8]}, "
            f"ratio_errors={best['ratio_errors']}, "
            f"target_balance_error={best['target_balance_error']:.6f}, "
            f"encounter_balance_error={best['encounter_balance_error']:.6f}, "
            f"tolerances=(ratio={split.ratio_tolerance}, "
            f"target_balance={split.target_balance_tolerance}, "
            f"encounter_balance={split.encounter_balance_tolerance})"
        )
    selected = min(
        feasible,
        key=lambda record: (
            max(record["ratio_errors"].values()),
            record["target_balance_error"],
            record["encounter_balance_error"],
            record["candidate_number"],
        ),
    )

    selected_positions = selected["positions_by_role"]
    role_frames = selected["frames"]
    role_groups = {
        role: groups.iloc[positions].reset_index(drop=True) if groups is not None else None
        for role, positions in selected_positions.items()
    }
    role_labels = np.full(len(frame), "", dtype=object)
    for role, positions in selected_positions.items():
        role_labels[positions] = role
    assignment = pd.DataFrame(
        {
            "row_key": np.arange(len(frame), dtype=np.int64),
            "role": role_labels,
        }
    )
    if groups is not None:
        assignment["population_group_hash"] = groups.to_numpy(copy=True)
    assignment_payload = assignment.to_json(orient="records", date_format="iso")
    if not isinstance(assignment_payload, str):
        raise TypeError("role assignment JSON payload must be a string")
    assignment_fingerprint = hashlib.sha256(assignment_payload.encode()).hexdigest()
    assignment_policy = {
        "schema_version": "role-assignment-policy-v2",
        "algorithm": "candidate_shuffle_v2_stratified_encounter",
        "mode": split.mode,
        "target_column": target_column,
        "protected_columns": protected_columns,
        "groups_provided": groups is not None,
        "effective_seed": rng_seed,
        "split": dataclasses.asdict(split),
    }
    assignment_policy_fingerprint = _payload_fingerprint(assignment_policy)
    role_metadata = {
        "roles": {
            role: _summary_for_role(
                role_frames[role], target_column, role_groups[role], protected_columns
            )
            for role in ROLE_NAMES
        },
        "requested_fractions": {
            "train": split.train_fraction,
            "tuning": split.tuning_fraction,
            "final_holdout": split.final_holdout_fraction,
        },
        "achieved_fractions": {role: len(role_frames[role]) / len(frame) for role in ROLE_NAMES},
        "assignment_algorithm": assignment_policy["algorithm"],
        "assignment_policy": assignment_policy,
        "assignment_policy_fingerprint": assignment_policy_fingerprint,
        "assignment_seed": rng_seed,
        "candidate_count": n_candidates,
        "selected_candidate": selected["candidate_number"],
        "ratio_tolerance": split.ratio_tolerance,
        "target_balance_tolerance": split.target_balance_tolerance,
        "encounter_balance_tolerance": split.encounter_balance_tolerance,
        "encounter_policy": {
            "label_column": split.encounter_label_column,
            "objective": (
                "encounter_counts_and_target_distribution"
                if split.encounter_label_column is not None
                else "disabled"
            ),
        },
        "resolved_constraints": {
            key: value for key, value in constraints.items() if not key.startswith("_")
        },
        "preflight": {
            "selected_candidate_violations": selected["violations"],
            "selected_candidate_ratio_errors": selected["ratio_errors"],
            "selected_candidate_target_balance_error": selected["target_balance_error"],
            "selected_candidate_encounter_metrics": selected["encounter_metrics"],
            "selected_candidate_encounter_balance_error": selected["encounter_balance_error"],
        },
    }
    logger.info(
        "Allocated roles using candidate=%d seed=%d: train=%d tuning=%d final_holdout=%d",
        selected["candidate_number"],
        rng_seed,
        len(role_frames["train"]),
        len(role_frames["tuning"]),
        len(role_frames["final_holdout"]),
    )
    return RoleAssignment(
        frames=role_frames,
        groups=role_groups,
        assignment=assignment,
        metadata=role_metadata,
        assignment_fingerprint=assignment_fingerprint,
        assignment_policy_fingerprint=assignment_policy_fingerprint,
    )
