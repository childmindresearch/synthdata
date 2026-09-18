"""Generic dataset loading, typing, and splitting.

Supports two data sources so a collaborator can point the pipeline at their own data:

- ``source: uci``: fetch (and locally cache) a dataset from the UCI ML repository by id.
- ``source: csv``/``source: parquet``: load a local CSV or Parquet file directly
  (the reader used is auto-detected from ``data.path``'s file extension --
  ``.csv`` vs. ``.parquet``/``.pq`` -- rather than from ``source`` itself).

The same :class:`Dataset` object is produced either way and consumed by every
downstream stage (imputation, generation, evaluation, plotting).
"""

import dataclasses
import hashlib
import json
import os
import tempfile
import types
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from synthdata.config import Config
from synthdata.data_roles import (
    ROLE_NAMES,
    RoleAssignment,
    allocate_roles,
    resolve_population_identity,
)
from synthdata.utils import ensure_dir, get_logger, git_commit

logger = get_logger(__name__)

IMPUTATION_CACHE_KEY_FILENAME = ".imputation_cache_key.json"


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """Return a stable fingerprint for a DataFrame's values and structure."""
    metadata = {
        "columns": [str(column) for column in df.columns],
        "dtypes": [str(dtype) for dtype in df.dtypes],
        "index_dtype": str(df.index.dtype),
        "index_name": df.index.name,
        "shape": list(df.shape),
    }
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, default=str).encode())
    values = pd.util.hash_pandas_object(df, index=True).to_numpy(dtype=np.uint64, copy=False)
    digest.update(values.tobytes())
    return digest.hexdigest()


def file_fingerprint(path: str | Path) -> str:
    """Return the SHA-256 fingerprint of a source file's exact bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def role_context_payload(
    dataset,
    roles: tuple[str, ...] = ROLE_NAMES,
    *,
    candidate_phase: bool = False,
) -> dict:
    """Describe named role inputs and provenance used by one operation.

    Candidate contexts deliberately record ``final_holdout`` as raw-equivalent:
    final-phase imputation has not yet been validated and must not become part
    of generation experiment lineage.
    """
    role_set = set(roles)
    assignment_fingerprint = None
    assignment = dataset.assignment
    if assignment is not None:
        assignment = assignment[assignment["role"].isin(role_set)].copy()
        assignment = assignment.sort_values("row_key").reset_index(drop=True)
        assignment_fingerprint = dataframe_fingerprint(assignment)
    role_payload = {}
    for role in roles:
        raw_frame = dataset.role_frame(role, imputed=False)
        imputed_frame = dataset.role_frame(role, imputed=True)
        frame = imputed_frame if imputed_frame is not None else raw_frame
        if frame is None:
            raise ValueError(f"Role context requires a populated {role!r} role")
        role_payload[role] = {
            "raw_fingerprint": (
                dataframe_fingerprint(raw_frame) if raw_frame is not None else None
            ),
            "imputed_fingerprint": (
                dataframe_fingerprint(raw_frame)
                if candidate_phase and role == "final_holdout" and raw_frame is not None
                else (dataframe_fingerprint(imputed_frame) if imputed_frame is not None else None)
            ),
            "rows": int(len(frame)),
        }
    identity_fingerprint = dataset.identity_fingerprint
    if assignment is not None and "population_group_hash" in assignment:
        selected = assignment[assignment["role"].isin(role_set)]
        identity_fingerprint = dataframe_fingerprint(
            selected[["row_key", "population_group_hash"]].sort_values("row_key")
        )
    return {
        "schema_version": "role-context-v1",
        "dataset_name": dataset.name,
        "dataset_version": dataset.version,
        "roles": role_payload,
        "assignment_fingerprint": assignment_fingerprint,
        "assignment_policy_fingerprint": dataset.assignment_policy_fingerprint,
        "identity_fingerprint": identity_fingerprint,
        "semantic_fingerprint": dataset.semantic_fingerprint,
        "variable_schema_fingerprint": dataset.variable_schema_fingerprint,
        "compatibility_mode": "legacy_two_role" if dataset.legacy_two_role else None,
    }


def role_context_fingerprint(
    dataset,
    roles: tuple[str, ...] = ROLE_NAMES,
    *,
    candidate_phase: bool = False,
) -> str:
    """Hash exact role inputs and split/schema provenance for an operation."""
    payload = role_context_payload(dataset, roles, candidate_phase=candidate_phase)
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def validate_imputation_cache_lineage(
    dataset,
    cache_record: Mapping[str, object] | None,
    *,
    required: bool = False,
) -> None:
    """Fail closed when candidate imputation metadata is stale or incomplete.

    This validation is intentionally separate from :func:`load_imputed_splits`,
    which historically treats stale CSVs as a cache miss. Generation must not
    proceed on that permissive path because its experiment lineage would then
    be ambiguous.
    """
    if cache_record is None:
        if required:
            raise RuntimeError(
                "Candidate imputation lineage is unavailable; rerun imputation before generation."
            )
        return
    if not dataset.has_canonical_roles:
        return
    persisted_path = dataset.data_dir / IMPUTATION_CACHE_KEY_FILENAME
    try:
        with persisted_path.open() as cache_file:
            persisted_record = json.load(cache_file)
    except (OSError, json.JSONDecodeError) as exc:
        if required:
            raise RuntimeError(
                "Candidate imputation lineage metadata is unreadable; "
                "rerun imputation before generation."
            ) from exc
        return
    if not isinstance(persisted_record, dict):
        raise RuntimeError("Candidate imputation lineage metadata is not an object")
    expected_cache_key = cache_record.get("cache_key")
    if persisted_record.get("cache_key") != expected_cache_key:
        raise RuntimeError(
            "Candidate imputation cache key does not match current dataset lineage; "
            "rerun imputation before generation."
        )
    expected = {
        "dataset_version": dataset.version,
        "variable_schema_fingerprint": dataset.variable_schema_fingerprint,
        "assignment_fingerprint": dataset.assignment_fingerprint,
        "assignment_policy_fingerprint": dataset.assignment_policy_fingerprint,
        "identity_fingerprint": dataset.role_metadata.get("identity", {}).get(
            "identity_fingerprint"
        ),
        "semantic_fingerprint": dataset.semantic_fingerprint,
        "role_fingerprints": dataset.role_fingerprints,
    }
    mismatches = {
        field: (persisted_record.get(field), value)
        for field, value in expected.items()
        if persisted_record.get(field) != value
    }
    if mismatches:
        raise RuntimeError(
            "Candidate imputation cache lineage does not match current dataset; "
            f"rerun imputation before generation (mismatches={mismatches})."
        )
    if (
        persisted_record.get("cache_contract") == "canonical_roles_v1"
        and persisted_record.get("imputation_method") == "hyperimpute"
        and not _canonical_candidate_fit_state_is_valid(dataset, persisted_record)
    ):
        raise RuntimeError(
            "Candidate HyperImpute fit state is missing or invalid; "
            "rerun imputation before generation."
        )


def _canonical_candidate_fit_state_is_valid(dataset, cache_record: Mapping[str, object]) -> bool:
    """Validate persisted candidate HyperImpute state before generation."""
    state = cache_record.get("fit_state")
    if not isinstance(state, dict):
        return False
    if state.get("status") not in {"fitted", "not_required", "disabled"}:
        return False
    if state.get("backend") != "hyperimpute":
        return False
    if state.get("fit_frame_fingerprint_version") != "dataframe_fingerprint_v1":
        return False
    if state.get("fit_roles") != ["train"] or state.get("transform_roles") != ["train", "tuning"]:
        return False
    if state.get("continuous_plugin") != cache_record.get("continuous_plugin"):
        return False
    if state.get("categorical_plugin") != "most_frequent":
        return False
    if state.get("fit_frame_fingerprint") != dataframe_fingerprint(dataset.roles["train"]):
        return False
    if state.get("feature_columns") != list(dataset.feature_columns):
        return False
    if state.get("categorical_columns") != list(dataset.categorical_columns):
        return False
    from synthdata.imputation.hyperimpute_backend import metadata_fingerprint

    return state.get("state_fingerprint") == metadata_fingerprint(state)


SEMANTIC_CONTEXT_SCHEMA_VERSION = "semantic-context-v1"


def semantic_declaration_fingerprint(value) -> str:
    """Return deterministic fingerprint for one semantic declaration."""
    encoded = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def semantic_context_payload(
    dataset,
    *,
    classification_score: str | None = None,
    roles: tuple[str, ...] | None = None,
) -> dict:
    """Return the resolved semantic declarations used by downstream stages."""
    if classification_score is not None and classification_score not in {
        "balanced_accuracy",
        "macro_f1",
    }:
        raise ValueError(
            "classification_score must be 'balanced_accuracy' or 'macro_f1', "
            f"got {classification_score!r}"
        )

    modeling_columns = list(dataset.feature_columns) + [dataset.target_column]
    feature_types = {
        column: dataset.variable_schema[column]["kind"]
        for column in modeling_columns
        if column in dataset.variable_schema
    }
    variable_schema = {
        column: dict(dataset.variable_schema[column])
        for column in modeling_columns
        if column in dataset.variable_schema
    }
    # Candidate semantic declarations must not inherit final-holdout provenance.
    # Full role provenance remains available through an explicit roles=ROLE_NAMES.
    semantic_roles = roles
    if semantic_roles is None:
        semantic_roles = ("train", "tuning") if dataset.has_canonical_roles else ROLE_NAMES
    role_context = role_context_payload(dataset, semantic_roles)
    identity_metadata = dataset.role_metadata.get("identity", {})
    identity_dimension = {
        "fingerprint": role_context["identity_fingerprint"],
        "source": identity_metadata.get("identity_source"),
        "mode": identity_metadata.get("mode"),
        "tokenization_algorithm": identity_metadata.get("tokenization_algorithm"),
        "tokenization_version": identity_metadata.get("tokenization_version"),
        "tokenization_scope_fingerprint": identity_metadata.get("tokenization_scope_fingerprint"),
        "hmac_key_fingerprint": identity_metadata.get("hmac_key_fingerprint"),
    }
    payload = {
        "schema_version": SEMANTIC_CONTEXT_SCHEMA_VERSION,
        "dataset_name": dataset.name,
        "dataset_version": dataset.version,
        "target_column": dataset.target_column,
        "task_type": "classification" if dataset.target_is_categorical else "regression",
        "feature_columns": modeling_columns[:-1],
        "nominal_columns": list(dataset.nominal_columns),
        "ordinal_columns": list(dataset.ordinal_columns),
        "categorical_columns": list(dataset.categorical_columns),
        "protected_columns": list(dataset.protected_columns),
        "sensitive_columns": list(dataset.sensitive_columns),
        "quasi_identifier_columns": list(dataset.quasi_identifier_columns),
        "release_generalization": dict(dataset.release_generalization),
        "quasi_identifier_fingerprint": semantic_declaration_fingerprint(
            dataset.quasi_identifier_columns
        ),
        "sensitive_fingerprint": semantic_declaration_fingerprint(dataset.sensitive_columns),
        "protected_fingerprint": semantic_declaration_fingerprint(dataset.protected_columns),
        "schema_fingerprint": semantic_declaration_fingerprint(variable_schema),
        "release_generalization_fingerprint": semantic_declaration_fingerprint(
            dataset.release_generalization
        ),
        "role_fingerprint": role_context_fingerprint(dataset, semantic_roles),
        "identity_fingerprint": role_context["identity_fingerprint"],
        "identity_dimension": identity_dimension,
        "feature_types": feature_types,
        "sensitive_target_types": {
            column: feature_types[column]
            for column in dataset.sensitive_columns
            if column in feature_types
        },
        "source_table": {
            column: entry["source_table"]
            for column, entry in variable_schema.items()
            if entry.get("source_table") is not None
        },
        "variable_schema": variable_schema,
        "variable_schema_fingerprint": dataset.variable_schema_fingerprint,
        "semantic_fingerprint": dataset.semantic_fingerprint,
        "compatibility_mode": "legacy_two_role" if dataset.legacy_two_role else None,
        "classification_score": classification_score,
    }
    return json.loads(json.dumps(payload, sort_keys=True, default=str))


def semantic_context_digest(payload: Mapping) -> str:
    """Hash a versioned semantic context payload without including raw data."""
    if not isinstance(payload, Mapping):
        raise TypeError("semantic context must be a mapping")
    if payload.get("schema_version") != SEMANTIC_CONTEXT_SCHEMA_VERSION:
        raise ValueError(
            f"semantic context has an unsupported schema: {payload.get('schema_version')!r}"
        )
    encoded = json.dumps(dict(payload), sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def validate_semantic_context(
    payload: Mapping | None,
    *,
    target_column: str,
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str],
    target_is_categorical: bool,
    variable_schema_fingerprint: str | None,
    frame_columns: Sequence[str] | None = None,
) -> dict | None:
    """Validate the semantic envelope consumed by an external generator."""
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise TypeError("semantic_context must be a mapping or None")
    context = json.loads(json.dumps(dict(payload), sort_keys=True, default=str))
    required_fields = (
        "schema_version",
        "dataset_name",
        "dataset_version",
        "target_column",
        "task_type",
        "feature_columns",
        "nominal_columns",
        "ordinal_columns",
        "categorical_columns",
        "protected_columns",
        "sensitive_columns",
        "quasi_identifier_columns",
        "release_generalization",
        "quasi_identifier_fingerprint",
        "sensitive_fingerprint",
        "protected_fingerprint",
        "schema_fingerprint",
        "release_generalization_fingerprint",
        "role_fingerprint",
        "identity_fingerprint",
        "identity_dimension",
        "sensitive_target_types",
        "feature_types",
        "source_table",
        "variable_schema",
        "variable_schema_fingerprint",
        "semantic_fingerprint",
        "compatibility_mode",
        "classification_score",
    )
    missing = [field for field in required_fields if field not in context]
    if missing:
        raise ValueError(f"semantic_context is incomplete; missing {missing}")
    semantic_context_digest(context)

    expected_features = list(feature_columns)
    model_columns = [*expected_features, target_column]
    if target_column in expected_features or len(model_columns) != len(set(model_columns)):
        raise ValueError("semantic_context generator columns must contain a unique target")
    if context["target_column"] != target_column:
        raise ValueError(
            "semantic_context target_column does not match the external generator request"
        )
    if context["feature_columns"] != expected_features:
        raise ValueError(
            "semantic_context feature_columns do not match the external generator request"
        )
    if context["categorical_columns"] != list(categorical_columns):
        raise ValueError(
            "semantic_context categorical_columns do not match the external generator request"
        )
    expected_task_type = "classification" if target_is_categorical else "regression"
    if context["task_type"] != expected_task_type:
        raise ValueError("semantic_context task_type does not match the external generator request")
    if context["variable_schema_fingerprint"] != variable_schema_fingerprint:
        raise ValueError(
            "semantic_context variable_schema_fingerprint does not match the external generator request"
        )

    if frame_columns is not None:
        actual_columns = list(frame_columns)
        if len(actual_columns) != len(set(actual_columns)) or set(actual_columns) != set(
            model_columns
        ):
            raise ValueError(
                "semantic_context modeling columns do not match the external generator frame"
            )

    list_fields = (
        "feature_columns",
        "nominal_columns",
        "ordinal_columns",
        "categorical_columns",
        "protected_columns",
        "sensitive_columns",
        "quasi_identifier_columns",
    )
    for field in list_fields:
        if not isinstance(context[field], list):
            raise ValueError(f"semantic_context.{field} must be a list")
    mapping_fields = (
        "feature_types",
        "sensitive_target_types",
        "source_table",
        "variable_schema",
    )
    for field in mapping_fields:
        if not isinstance(context.get(field), Mapping):
            raise ValueError(f"semantic_context.{field} must be an object")
    fingerprint_fields = (
        "quasi_identifier_fingerprint",
        "sensitive_fingerprint",
        "protected_fingerprint",
        "schema_fingerprint",
        "release_generalization_fingerprint",
        "role_fingerprint",
        "identity_fingerprint",
    )
    for field in fingerprint_fields:
        if not isinstance(context[field], str) or not context[field]:
            raise ValueError(f"semantic_context.{field} must be a non-empty string")
    for field in ("variable_schema_fingerprint", "semantic_fingerprint"):
        if context[field] is not None and (
            not isinstance(context[field], str) or not context[field]
        ):
            raise ValueError(f"semantic_context.{field} must be a string or None")
    if not isinstance(context["identity_dimension"], Mapping):
        raise ValueError("semantic_context.identity_dimension must be an object")
    expected_declaration_fingerprints = {
        "quasi_identifier_fingerprint": semantic_declaration_fingerprint(
            context["quasi_identifier_columns"]
        ),
        "sensitive_fingerprint": semantic_declaration_fingerprint(context["sensitive_columns"]),
        "protected_fingerprint": semantic_declaration_fingerprint(context["protected_columns"]),
        "schema_fingerprint": semantic_declaration_fingerprint(context["variable_schema"]),
        "release_generalization_fingerprint": semantic_declaration_fingerprint(
            context["release_generalization"]
        ),
    }
    for field, expected in expected_declaration_fingerprints.items():
        if context[field] != expected:
            raise ValueError(f"semantic_context.{field} does not match its declaration")
    if context["identity_dimension"].get("fingerprint") != context["identity_fingerprint"]:
        raise ValueError(
            "semantic_context.identity_dimension fingerprint disagrees with identity_fingerprint"
        )

    if not set(context["nominal_columns"]) | set(context["ordinal_columns"]) <= set(
        expected_features
    ):
        raise ValueError("semantic_context categorical role columns must be model features")
    if set(context["categorical_columns"]) != set(context["nominal_columns"]) | set(
        context["ordinal_columns"]
    ):
        raise ValueError("semantic_context categorical_columns disagree with nominal/ordinal roles")
    if not set(context["protected_columns"]) <= set(model_columns):
        raise ValueError("semantic_context protected_columns contain unknown modeling columns")
    if not set(context["sensitive_columns"]) <= set(model_columns):
        raise ValueError("semantic_context sensitive_columns contain unknown modeling columns")
    if not isinstance(context["release_generalization"], Mapping):
        raise ValueError("semantic_context.release_generalization must be an object")
    sensitive_qi_overlap = sorted(
        set(context["sensitive_columns"]) & set(context["quasi_identifier_columns"])
    )
    if sensitive_qi_overlap:
        raise ValueError(
            "semantic_context sensitive_columns must not overlap quasi_identifier_columns: "
            f"{sensitive_qi_overlap}"
        )
    sensitive_target_overlap = sorted(set(context["sensitive_columns"]) & {target_column})
    if sensitive_target_overlap:
        raise ValueError(
            "semantic_context sensitive_columns must not overlap target_column: "
            f"{sensitive_target_overlap}"
        )
    qi_target_overlap = sorted(set(context["quasi_identifier_columns"]) & {target_column})
    if qi_target_overlap:
        raise ValueError(
            "semantic_context quasi_identifier_columns must not overlap target_column: "
            f"{qi_target_overlap}"
        )
    if not set(context["quasi_identifier_columns"]) <= set(expected_features):
        raise ValueError("semantic_context quasi_identifier_columns contain non-feature columns")

    feature_types = dict(context["feature_types"])
    if set(feature_types) != set(model_columns):
        raise ValueError("semantic_context feature_types must cover every modeling column")
    if any(value not in {"categorical", "continuous"} for value in feature_types.values()):
        raise ValueError("semantic_context feature_types contain an unsupported kind")
    expected_target_kind = "categorical" if target_is_categorical else "continuous"
    if feature_types[target_column] != expected_target_kind:
        raise ValueError(
            "semantic_context target type does not match the external generator request"
        )

    variable_schema = dict(context["variable_schema"])
    if set(variable_schema) != set(model_columns):
        raise ValueError("semantic_context variable_schema must cover every modeling column")
    for column in model_columns:
        entry = variable_schema[column]
        if not isinstance(entry, Mapping) or entry.get("kind") != feature_types[column]:
            raise ValueError(
                f"semantic_context variable_schema kind does not match feature_types for {column!r}"
            )
    expected_sensitive_types = {
        column: feature_types[column] for column in context["sensitive_columns"]
    }
    if dict(context["sensitive_target_types"]) != expected_sensitive_types:
        raise ValueError("semantic_context sensitive_target_types disagree with sensitive_columns")
    if not set(context["source_table"]) <= set(model_columns):
        raise ValueError("semantic_context source_table contains unknown modeling columns")
    if context["classification_score"] not in {None, "balanced_accuracy", "macro_f1"}:
        raise ValueError("semantic_context classification_score is unsupported")
    return context


def semantic_context_fingerprint(
    dataset,
    *,
    classification_score: str | None = None,
    roles: tuple[str, ...] | None = None,
) -> str:
    """Return the digest of the resolved semantic context for a dataset."""
    return semantic_context_digest(
        semantic_context_payload(dataset, classification_score=classification_score, roles=roles)
    )


@dataclasses.dataclass
class Dataset:
    """Container for a loaded dataset plus derived metadata used by every stage."""

    name: str
    target_column: str
    feature_columns: list
    nominal_columns: list
    ordinal_columns: list
    sensitive_columns: list
    data_dir: Path

    #: Full dataset, possibly containing missing values (pre-imputation).
    full_df: pd.DataFrame
    #: Historical two-role frames. New datasets use ``roles`` instead.
    train_df: pd.DataFrame | None
    test_df: pd.DataFrame | None

    #: Canonical raw role frames for new datasets. Keys are exactly
    #: ``train``, ``tuning``, and ``final_holdout``.
    roles: dict[str, pd.DataFrame] = dataclasses.field(default_factory=dict)
    #: Non-modeling population groups aligned with each canonical role.
    role_groups: dict[str, pd.Series | None] = dataclasses.field(default_factory=dict)
    #: Immutable row-to-role assignment table with hashed population groups only.
    assignment: pd.DataFrame | None = None
    #: Resolved split, identity, support, and semantic metadata.
    role_metadata: dict = dataclasses.field(default_factory=dict)
    #: Historical data is readable but cannot provide a tuning role.
    legacy_two_role: bool = False

    #: Freeform dataset version label (see DataConfig.version), recorded in
    #: experiment manifests for traceability. None if not set by the user.
    version: str | None = None

    #: Explicit protected attributes and quasi-identifiers. Sensitive columns
    #: are retained as a separate declaration for attack semantics.
    protected_columns: list = dataclasses.field(default_factory=list)
    quasi_identifier_columns: list = dataclasses.field(default_factory=list)

    #: Explicit, resolved variable schema used to derive the categorical roles.
    #: Each entry is ``{"kind": "categorical"|"continuous",
    #: "ordinal_order": list|None}`` and includes the target column.
    variable_schema: dict = dataclasses.field(default_factory=dict)
    #: SHA-256 fingerprint of the exact source schema CSV, if one was used.
    variable_schema_fingerprint: str | None = None
    #: SHA-256 fingerprint of the exact local source file, or of the raw loaded
    #: source frame for sources without a local file.
    source_fingerprint: str | None = None
    #: Fingerprints of the cleaned source frame and deterministic split frames.
    full_fingerprint: str | None = None
    train_split_fingerprint: str | None = None
    test_split_fingerprint: str | None = None
    role_fingerprints: dict[str, str] = dataclasses.field(default_factory=dict)
    assignment_fingerprint: str | None = None
    assignment_policy_fingerprint: str | None = None
    semantic_fingerprint: str | None = None
    identity_sidecar: pd.Series | None = None
    identity_fingerprint: str | None = None
    release_generalization: dict = dataclasses.field(default_factory=dict)
    role_context_fingerprint: str | None = None

    #: Numeric model-space frames populated once imputation has run
    #: (see synthdata.imputation).
    full_imputed_df: pd.DataFrame | None = None
    train_imputed_df: pd.DataFrame | None = None
    test_imputed_df: pd.DataFrame | None = None
    imputed_roles: dict[str, pd.DataFrame] = dataclasses.field(default_factory=dict)
    #: User-facing copies with configured ordinal labels restored.
    full_imputed_decoded_df: pd.DataFrame | None = None
    train_imputed_decoded_df: pd.DataFrame | None = None
    test_imputed_decoded_df: pd.DataFrame | None = None
    decoded_roles: dict[str, pd.DataFrame] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize role state and capture fingerprints for held frames."""
        self.protected_columns = list(self.protected_columns)
        self.sensitive_columns = list(self.sensitive_columns)

        if self.roles:
            if self.legacy_two_role:
                raise ValueError(
                    "legacy_two_role datasets must be constructed from train_df/test_df"
                )
            if set(self.roles) != set(ROLE_NAMES):
                raise ValueError(
                    f"New Dataset.roles must contain exactly {list(ROLE_NAMES)}, got {list(self.roles)}"
                )
        elif self.train_df is not None and self.test_df is not None:
            self.roles = {
                "train": self.train_df,
                "final_holdout": self.test_df,
            }
            self.role_groups = {"train": None, "final_holdout": None}
            self.legacy_two_role = True
        elif self.train_df is not None or self.test_df is not None:
            raise ValueError("Dataset requires both historical train_df and test_df")
        elif not self.legacy_two_role:
            raise ValueError("New Dataset requires canonical train/tuning/final_holdout roles")

        if self.role_groups and set(self.role_groups) - set(self.roles):
            raise ValueError("Dataset.role_groups contains a role not present in Dataset.roles")
        if not self.role_groups:
            self.role_groups = {role: None for role in self.roles}
        for role in self.roles:
            if role not in self.role_fingerprints:
                self.role_fingerprints[role] = dataframe_fingerprint(self.roles[role])

        if self.full_fingerprint is None:
            self.full_fingerprint = dataframe_fingerprint(self.full_df)
        if self.train_split_fingerprint is None and self.train_df is not None:
            self.train_split_fingerprint = dataframe_fingerprint(self.train_df)
        if self.test_split_fingerprint is None and self.test_df is not None:
            self.test_split_fingerprint = dataframe_fingerprint(self.test_df)
        if self.assignment is not None and self.assignment_fingerprint is None:
            self.assignment_fingerprint = dataframe_fingerprint(self.assignment)
        if self.assignment_policy_fingerprint is None:
            split_metadata = self.role_metadata.get("split", {})
            self.assignment_policy_fingerprint = split_metadata.get("assignment_policy_fingerprint")
        if self.identity_fingerprint is None:
            self.identity_fingerprint = self.role_metadata.get("identity", {}).get(
                "identity_fingerprint"
            )
        if self.role_context_fingerprint is None:
            self.role_context_fingerprint = self.assignment_fingerprint

    @property
    def has_canonical_roles(self) -> bool:
        """Whether this dataset has the complete three-role contract."""
        return not self.legacy_two_role and set(self.roles) == set(ROLE_NAMES)

    def require_canonical_roles(self, operation: str) -> None:
        """Reject historical two-role data at new selection/policy boundaries."""
        if not self.has_canonical_roles:
            raise RuntimeError(
                f"{operation} requires canonical train/tuning/final_holdout roles; "
                "this dataset is labeled legacy_two_role and cannot be used for new "
                "HPO, final-policy, or release claims"
            )

    def role_frame(self, role: str, *, imputed: bool = False) -> pd.DataFrame | None:
        """Return a named role, with an explicit legacy-only compatibility view."""
        frames = self.imputed_roles if imputed else self.roles
        frame = frames.get(role)
        if frame is not None:
            return frame
        if self.legacy_two_role:
            if role == "train":
                return self.train_imputed_df if imputed else self.train_df
            if role == "final_holdout":
                return self.test_imputed_df if imputed else self.test_df
        return None

    def set_imputed_roles(self, frames: dict[str, pd.DataFrame]) -> None:
        """Attach role-specific imputed frames and derived compatibility views."""
        if set(frames) != set(self.roles):
            raise ValueError(
                f"Imputed role keys {list(frames)} do not match raw role keys {list(self.roles)}"
            )
        for role, frame in frames.items():
            if frame.columns.tolist() != self.full_df.columns.tolist():
                raise ValueError(f"Imputed role {role!r} columns do not match the raw model frame")
            if len(frame) != len(self.roles[role]):
                raise ValueError(
                    f"Imputed role {role!r} has {len(frame)} rows; expected {len(self.roles[role])}"
                )
        self.imputed_roles = {role: frame.copy() for role, frame in frames.items()}
        if self.has_canonical_roles:
            if self.assignment is None:
                raise ValueError(
                    "Canonical Dataset requires an assignment table to reconstruct full row order"
                )
            ordered_frames = [self.imputed_roles[role] for role in ROLE_NAMES]
            row_keys = pd.concat(
                [
                    self.assignment.loc[self.assignment["role"] == role, "row_key"]
                    for role in ROLE_NAMES
                ],
                ignore_index=True,
            )
            if len(row_keys) != len(pd.concat(ordered_frames, ignore_index=True)):
                raise ValueError(
                    "Canonical Dataset assignment row counts do not match imputed role rows"
                )
            full_imputed = pd.concat(ordered_frames, ignore_index=True)
            full_imputed.index = row_keys.to_numpy()
            self.full_imputed_df = full_imputed.sort_index().reset_index(drop=True)
            self.train_imputed_df = None
            self.test_imputed_df = None
        else:
            self.full_imputed_df = self.imputed_roles.get("train")
            self.train_imputed_df = self.imputed_roles.get("train")
            self.test_imputed_df = self.imputed_roles.get("final_holdout")
        self.attach_decoded_imputed_splits()

    @property
    def categorical_columns(self) -> list:
        """All categorical-encoded feature columns in feature-column order.

        Every backend that discretely encodes categorical columns (bit-encoding
        in refidiff, one-hot in tabimpute, etc.) doesn't itself need to
        distinguish nominal from ordinal -- both are encoded/decoded to exact
        observed category values the same way, the only difference is whether
        an order is preserved for the ordinal ones (see
        synthdata.imputation.refidiff_backend._fit_categorical_binary_encoders).
        So most call sites want this combined list; use ``nominal_columns``/
        ``ordinal_columns`` directly only when the distinction actually matters.
        """
        categorical = set(self.nominal_columns) | set(self.ordinal_columns)
        return [column for column in self.feature_columns if column in categorical]

    @property
    def ordinal_category_orders(self) -> dict:
        """Return configured ordinal labels in their low-to-high order."""
        return {
            column: list(self.variable_schema[column]["ordinal_order"])
            for column in self.ordinal_columns
            if column in self.variable_schema
            and self.variable_schema[column].get("ordinal_order") is not None
        }

    def decode_ordinal_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Restore configured ordinal labels in a model-space DataFrame."""
        return decode_ordinal_columns(df, self.ordinal_category_orders)

    @property
    def target_is_categorical(self) -> bool:
        """Whether the resolved schema declares the target as categorical.

        Datasets created through the legacy explicit-list configuration have no
        reliable target-kind declaration, so retain the historical assumption
        that their target is categorical.
        """
        target_entry = self.variable_schema.get(self.target_column)
        if target_entry is None:
            return True
        return target_entry.get("kind") == "categorical"

    @property
    def all_categorical_columns(self) -> list:
        """Categorical feature columns plus the target when declared categorical."""
        if self.target_is_categorical:
            return self.categorical_columns + [self.target_column]
        return self.categorical_columns

    def paths(self) -> dict:
        d = self.data_dir
        paths = {
            "full": d / "full.csv",
            "full_imputed": d / "full_imputed.csv",
            "full_imputed_decoded": d / "full_imputed_decoded.csv",
        }
        for role in self.roles:
            paths[role] = d / f"{role}.csv"
            paths[f"{role}_imputed"] = d / f"{role}_imputed.csv"
            paths[f"{role}_imputed_decoded"] = d / f"{role}_imputed_decoded.csv"
        if self.legacy_two_role:
            paths.update(
                {
                    "test": d / "test.csv",
                    "test_imputed": d / "test_imputed.csv",
                    "test_imputed_decoded": d / "test_imputed_decoded.csv",
                }
            )
        return paths

    def attach_decoded_imputed_splits(self) -> None:
        """Attach decoded user-facing views for any loaded imputed roles."""
        self.decoded_roles = {
            role: self.decode_ordinal_frame(frame) for role, frame in self.imputed_roles.items()
        }
        self.full_imputed_decoded_df = (
            self.decode_ordinal_frame(self.full_imputed_df)
            if self.full_imputed_df is not None
            else None
        )
        self.train_imputed_decoded_df = self.decoded_roles.get("train")
        self.test_imputed_decoded_df = self.decoded_roles.get("final_holdout")
        if self.legacy_two_role:
            self.train_imputed_decoded_df = (
                self.decode_ordinal_frame(self.train_imputed_df)
                if self.train_imputed_df is not None
                else None
            )
            self.test_imputed_decoded_df = (
                self.decode_ordinal_frame(self.test_imputed_df)
                if self.test_imputed_df is not None
                else None
            )


# ---------------------------------------------------------------------------
# UCI loading (with local caching, mirrors the notebook's fetch_ucirepo pattern)
# ---------------------------------------------------------------------------


def _fetch_uci_dataset(uci_id: int, cache_dir: Path):
    """Fetch a UCI dataset, caching features/targets/variables/metadata locally."""
    cache_files = {
        "features": cache_dir / "features.csv",
        "targets": cache_dir / "targets.csv",
        "variables": cache_dir / "variables.csv",
        "metadata": cache_dir / "metadata.json",
    }

    if all(p.exists() for p in cache_files.values()):
        logger.info("Loading cached UCI dataset id=%s from %s", uci_id, cache_dir)
        features = pd.read_csv(cache_files["features"])
        targets = pd.read_csv(cache_files["targets"])
        variables = pd.read_csv(cache_files["variables"])
        with open(cache_files["metadata"]) as f:
            metadata = json.load(f)
        data_ns = types.SimpleNamespace(features=features, targets=targets)
        return types.SimpleNamespace(data=data_ns, variables=variables, metadata=metadata)

    from ucimlrepo import fetch_ucirepo

    logger.info("Fetching UCI dataset id=%s (not cached)", uci_id)
    repo = fetch_ucirepo(id=uci_id)

    ensure_dir(cache_dir)
    repo.data.features.to_csv(cache_files["features"], index=False)
    repo.data.targets.to_csv(cache_files["targets"], index=False)
    repo.variables.to_csv(cache_files["variables"], index=False)
    metadata = repo.metadata if isinstance(repo.metadata, dict) else dict(repo.metadata)
    with open(cache_files["metadata"], "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return repo


def _load_uci(cfg: Config, data_dir: Path) -> tuple:
    assert cfg.data.uci_id is not None
    repo = _fetch_uci_dataset(cfg.data.uci_id, data_dir / cfg.data.raw_cache_subdir)
    df = pd.concat([repo.data.features, repo.data.targets], axis=1)
    variable_types = None
    if hasattr(repo, "variables") and repo.variables is not None:
        variable_types = dict(zip(repo.variables["name"], repo.variables["type"], strict=True))
    return df, variable_types


#: File extensions read via :func:`pandas.read_parquet` in :func:`_load_local_file`;
#: anything else falls back to :func:`pandas.read_csv`.
_PARQUET_SUFFIXES = (".parquet", ".pq")


def _load_local_file(cfg: Config) -> tuple:
    """Load a local CSV or Parquet file, auto-detected from ``cfg.data.path``'s extension.

    Dispatches on the file extension rather than ``cfg.data.source`` so a
    mismatched ``source`` value (e.g. ``source: csv`` pointing at a ``.parquet``
    file) still loads correctly instead of silently mis-parsing the file.
    """
    assert cfg.data.path is not None
    path = Path(cfg.data.path)
    suffix = path.suffix.lower()
    if suffix in _PARQUET_SUFFIXES:
        logger.info("Loading local Parquet file: %s", path)
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        logger.info("Loading local CSV file: %s", path)
        # low_memory=False: read the whole file in one pass rather than pandas'
        # default chunked read, which can infer a different dtype per chunk for
        # a column that's almost entirely NaN except for a handful of string
        # values far down the file (e.g. loris_combined.csv's >99%-missing
        # DailyMeds__med_type_0X columns) -- emits `DtypeWarning: Columns (...)
        # have mixed types` and silently mixes float/object dtype for the same
        # column across chunks. Same eventual per-column dtype either way, just
        # inferred consistently in one pass instead of reconciled after the fact.
        df = pd.read_csv(path, low_memory=False)
    else:
        raise ValueError(
            f"Unsupported file extension {suffix!r} for data.path={cfg.data.path!r}; "
            "expected one of .csv, .parquet, .pq"
        )
    return df, None


# ---------------------------------------------------------------------------
# Column typing helpers
# ---------------------------------------------------------------------------


def _schema_fingerprint(path: Path) -> str:
    """Return a SHA-256 fingerprint of a variable-schema source file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


_SCHEMA_OPTIONAL_FIELDS = {
    "ordinal_order",
    "source_table",
    "lower_bound",
    "upper_bound",
    "causal_parents",
    "categorical_values",
}


def _parse_schema_json_list(
    raw_value: str,
    schema_path: Path,
    row_number: int,
    column: str,
    field_name: str,
) -> list:
    if not raw_value.strip():
        return []
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Variable schema {schema_path} row {row_number} column {column!r} has invalid "
            f"{field_name}; expected a JSON list: {exc.msg}."
        ) from exc
    if not isinstance(value, list):
        raise ValueError(
            f"Variable schema {schema_path} row {row_number} column {column!r} must use a "
            f"JSON list for {field_name}."
        )
    encoded_values = [json.dumps(item, sort_keys=True, default=str) for item in value]
    if len(encoded_values) != len(set(encoded_values)):
        raise ValueError(
            f"Variable schema {schema_path} row {row_number} column {column!r} has duplicate "
            f"{field_name} values."
        )
    return value


def _parse_schema_bound(
    raw_value: str,
    schema_path: Path,
    row_number: int,
    column: str,
    field_name: str,
) -> float | int | None:
    if not raw_value.strip():
        return None
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Variable schema {schema_path} row {row_number} column {column!r} has a non-numeric "
            f"{field_name}: {raw_value!r}."
        ) from exc
    if not np.isfinite(value):
        raise ValueError(
            f"Variable schema {schema_path} row {row_number} column {column!r} has a non-finite "
            f"{field_name}: {raw_value!r}."
        )
    return int(value) if value.is_integer() else value


def validate_schema_dag(schema: dict, schema_path: str | Path | None = None) -> None:
    """Validate advisory causal-parent metadata without executing dependencies."""
    path_label = f" in {schema_path}" if schema_path is not None else ""
    declared_columns = set(schema)
    graph = {}
    for column, entry in schema.items():
        parents = entry.get("causal_parents") or []
        unknown = sorted(set(parents) - declared_columns)
        if unknown:
            raise ValueError(
                f"Variable schema{path_label} column {column!r} references unknown causal parent(s): "
                f"{unknown}"
            )
        if column in parents:
            raise ValueError(
                f"Variable schema{path_label} column {column!r} cannot list itself as a causal parent"
            )
        graph[column] = list(parents)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(column: str) -> None:
        if column in visiting:
            raise ValueError(
                f"Variable schema{path_label} causal_parents contains a cycle involving {column!r}"
            )
        if column in visited:
            return
        visiting.add(column)
        for parent in graph[column]:
            visit(parent)
        visiting.remove(column)
        visited.add(column)

    for column in graph:
        visit(column)


def schema_semantic_fingerprint(metadata: dict) -> str:
    """Return a stable digest for resolved schema semantics, not just CSV bytes."""
    encoded = json.dumps(metadata, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def resolve_schema_metadata(schema: dict, train_frame: pd.DataFrame) -> tuple[dict, str]:
    """Resolve categorical vocabularies and numeric bounds from the fit role only."""
    resolved = {}
    for column, entry in schema.items():
        if column not in train_frame.columns:
            raise KeyError(f"Schema column {column!r} is missing from the train role")
        observed = train_frame[column].dropna().unique().tolist()
        if not observed:
            raise ValueError(
                f"Schema column {column!r} has no observed values in the train role; "
                "cannot resolve its modeling semantics"
            )
        result = dict(entry)
        if entry["kind"] == "categorical":
            configured = entry.get("categorical_values")
            vocabulary = list(configured) if configured else observed
            unknown = [value for value in observed if value not in vocabulary]
            if unknown:
                raise ValueError(
                    f"Schema column {column!r} has train values not present in its declared "
                    f"categorical_values: {unknown}"
                )
            result["resolved_vocabulary"] = vocabulary
            result["resolved_lower_bound"] = None
            result["resolved_upper_bound"] = None
        else:
            if entry.get("categorical_values"):
                raise ValueError(
                    f"Continuous schema column {column!r} cannot declare categorical_values"
                )
            numeric = pd.to_numeric(train_frame[column], errors="coerce")
            if numeric.isna().all():
                raise ValueError(
                    f"Continuous schema column {column!r} has no numeric values in the train role"
                )
            observed_min = float(numeric.min())
            observed_max = float(numeric.max())
            lower_bound = entry.get("lower_bound")
            upper_bound = entry.get("upper_bound")
            if lower_bound is not None and observed_min < lower_bound:
                raise ValueError(
                    f"Train values for {column!r} fall below declared lower_bound "
                    f"{lower_bound}: observed minimum={observed_min}"
                )
            if upper_bound is not None and observed_max > upper_bound:
                raise ValueError(
                    f"Train values for {column!r} exceed declared upper_bound "
                    f"{upper_bound}: observed maximum={observed_max}"
                )
            result["resolved_vocabulary"] = None
            result["resolved_lower_bound"] = (
                lower_bound if lower_bound is not None else observed_min
            )
            result["resolved_upper_bound"] = (
                upper_bound if upper_bound is not None else observed_max
            )
        resolved[column] = result
    metadata = {
        "columns": resolved,
        "source_table": {
            column: entry.get("source_table")
            for column, entry in resolved.items()
            if entry.get("source_table") is not None
        },
        "causal_parents": {
            column: entry.get("causal_parents", []) for column, entry in resolved.items()
        },
        "fit_role": "train",
    }
    return metadata, schema_semantic_fingerprint(metadata)


def load_variable_schema(path: str | Path, modeling_columns: list) -> tuple[dict, str]:
    """Read and strictly validate the explicit variable schema CSV.

    The CSV requires ``column`` and ``kind`` fields. ``kind`` is exactly
    ``categorical`` or ``continuous``. ``ordinal_order`` is optional; when it
    is present it must be a non-empty square-bracketed list on a categorical row, ordered
    from low to high. A blank order denotes a nominal categorical. Every
    retained feature *and* the target must appear exactly once, and no stale
    declarations are accepted.
    """
    schema_path = Path(path)
    if not schema_path.exists():
        raise FileNotFoundError(f"Variable schema file not found: {schema_path}")

    try:
        schema_df = pd.read_csv(schema_path, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError) as exc:
        raise ValueError(f"Failed to read variable schema CSV {schema_path}: {exc}") from exc

    required_columns = {"column", "kind"}
    missing_columns = required_columns - set(schema_df.columns)
    if missing_columns:
        raise ValueError(
            f"Variable schema {schema_path} is missing required column(s) "
            f"{sorted(missing_columns)}; required columns are 'column' and 'kind'."
        )

    if schema_df["column"].str.strip().eq("").any():
        bad_rows = (schema_df.index[schema_df["column"].str.strip().eq("")] + 2).tolist()
        raise ValueError(
            f"Variable schema {schema_path} has blank column name(s) at CSV row(s) {bad_rows}."
        )
    schema_df["column"] = schema_df["column"].str.strip()
    duplicate_columns = schema_df.loc[schema_df["column"].duplicated(), "column"].tolist()
    if duplicate_columns:
        raise ValueError(
            f"Variable schema {schema_path} declares column(s) more than once: "
            f"{sorted(set(duplicate_columns))}."
        )

    expected_columns = set(modeling_columns)
    declared_columns = set(schema_df["column"])
    missing_declarations = sorted(expected_columns - declared_columns)
    stale_declarations = sorted(declared_columns - expected_columns)
    if missing_declarations or stale_declarations:
        details = []
        if missing_declarations:
            details.append(f"missing declaration(s): {missing_declarations}")
        if stale_declarations:
            details.append(f"stale/non-modeling declaration(s): {stale_declarations}")
        raise ValueError(
            f"Variable schema {schema_path} must declare every retained feature and target exactly "
            f"once after source cleanup; {'; '.join(details)}."
        )

    unknown_schema_fields = set(schema_df.columns) - {"column", "kind"} - _SCHEMA_OPTIONAL_FIELDS
    if unknown_schema_fields:
        raise ValueError(
            f"Variable schema {schema_path} contains unsupported field(s): "
            f"{sorted(unknown_schema_fields)}"
        )
    schema = {}
    for row_number, row in enumerate(schema_df.to_dict(orient="records"), start=2):
        column = row["column"]
        raw_kind = row["kind"]
        kind = raw_kind.strip().lower()
        if kind not in {"categorical", "continuous"}:
            raise ValueError(
                f"Variable schema {schema_path} row {row_number} column {column!r} has invalid "
                f"kind {kind!r}; expected 'categorical' or 'continuous'."
            )

        raw_order = row.get("ordinal_order", "")
        raw_order = raw_order.strip()
        ordinal_order = None
        if raw_order:
            if kind != "categorical":
                raise ValueError(
                    f"Variable schema {schema_path} row {row_number} column {column!r} supplies "
                    "ordinal_order but is declared continuous."
                )
            try:
                ordinal_order = json.loads(raw_order)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Variable schema {schema_path} row {row_number} column {column!r} has invalid "
                    f"ordinal_order value; expected a square-bracketed list: {exc.msg}."
                ) from exc
            if not isinstance(ordinal_order, list) or not ordinal_order:
                raise ValueError(
                    f"Variable schema {schema_path} row {row_number} column {column!r} must use a "
                    "non-empty square-bracketed list for ordinal_order."
                )
            try:
                unique_order_values = {
                    json.dumps(value, sort_keys=True, default=str) for value in ordinal_order
                }
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Variable schema {schema_path} row {row_number} column {column!r} has an "
                    f"ordinal_order value that cannot be represented in JSON: {exc}."
                ) from exc
            if len(unique_order_values) != len(ordinal_order):
                raise ValueError(
                    f"Variable schema {schema_path} row {row_number} column {column!r} has duplicate "
                    "ordinal_order values."
                )
        source_table = row.get("source_table", "").strip() or None
        lower_bound = _parse_schema_bound(
            row.get("lower_bound", ""), schema_path, row_number, column, "lower_bound"
        )
        upper_bound = _parse_schema_bound(
            row.get("upper_bound", ""), schema_path, row_number, column, "upper_bound"
        )
        if lower_bound is not None and upper_bound is not None and lower_bound > upper_bound:
            raise ValueError(
                f"Variable schema {schema_path} row {row_number} column {column!r} has "
                f"lower_bound {lower_bound} greater than upper_bound {upper_bound}."
            )
        if (lower_bound is not None or upper_bound is not None) and kind != "continuous":
            raise ValueError(
                f"Variable schema {schema_path} row {row_number} column {column!r} supplies "
                "numeric bounds but is not declared continuous."
            )
        causal_parents = _parse_schema_json_list(
            row.get("causal_parents", ""),
            schema_path,
            row_number,
            column,
            "causal_parents",
        )
        if any(not isinstance(parent, str) or not parent.strip() for parent in causal_parents):
            raise ValueError(
                f"Variable schema {schema_path} row {row_number} column {column!r} must use "
                "non-empty string causal parent names"
            )
        categorical_values = _parse_schema_json_list(
            row.get("categorical_values", ""),
            schema_path,
            row_number,
            column,
            "categorical_values",
        )
        if categorical_values and kind != "categorical":
            raise ValueError(
                f"Variable schema {schema_path} row {row_number} column {column!r} supplies "
                "categorical_values but is not declared categorical."
            )
        schema[column] = {
            "kind": kind,
            "ordinal_order": ordinal_order,
            "source_table": source_table,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "causal_parents": causal_parents,
            "categorical_values": categorical_values or None,
        }

    validate_schema_dag(schema, schema_path)

    logger.info(
        "Loaded strict variable schema from %s: %d columns (%d categorical, %d ordinal)",
        schema_path,
        len(schema),
        sum(entry["kind"] == "categorical" for entry in schema.values()),
        sum(entry["ordinal_order"] is not None for entry in schema.values()),
    )
    return schema, _schema_fingerprint(schema_path)


def schema_column_roles(schema: dict, target_column: str) -> tuple[list, list, dict]:
    """Derive nominal/ordinal compatibility lists and ordinal orders from a schema."""
    nominal_columns = [
        column
        for column, entry in schema.items()
        if column != target_column
        and entry["kind"] == "categorical"
        and entry["ordinal_order"] is None
    ]
    ordinal_columns = [
        column
        for column, entry in schema.items()
        if column != target_column and entry["ordinal_order"] is not None
    ]
    ordinal_orders = {
        column: entry["ordinal_order"]
        for column, entry in schema.items()
        if column != target_column and entry["ordinal_order"] is not None
    }
    return nominal_columns, ordinal_columns, ordinal_orders


def infer_nominal_columns(
    df: pd.DataFrame,
    feature_columns: list,
    explicit: str | list,
    ordinal_columns: list | None = None,
    unique_threshold: int = 10,
    uci_variable_types: dict | None = None,
) -> list:
    """Determine which feature columns should be treated as nominal (unordered categorical).

    Resolution order:
        1. An explicit list of column names in the config always wins.
        2. If UCI variable metadata is available, use its "Categorical" tag.
        3. Otherwise fall back to a dtype/cardinality heuristic
           (object/category/bool dtype, or nunique <= unique_threshold).

    ``ordinal_columns`` (a dataset's separately-configured ordered-categorical
    columns) are excluded from every resolution path above, regardless of
    source -- they're a distinct first-class role handled by the caller, never
    folded back into "nominal" just because they'd otherwise match the heuristic.
    """
    ordinal_set = set(ordinal_columns or [])

    if isinstance(explicit, list):
        return [c for c in explicit if c in feature_columns and c not in ordinal_set]

    if uci_variable_types is not None:
        cats = [
            c
            for c in feature_columns
            if uci_variable_types.get(c) == "Categorical" and c not in ordinal_set
        ]
        if cats:
            return cats

    cats = []
    for c in feature_columns:
        if c in ordinal_set:
            continue
        dtype = df[c].dtype
        if (
            dtype in (object, bool)
            or str(dtype) == "category"
            or df[c].nunique(dropna=True) <= unique_threshold
        ):
            cats.append(c)
    return cats


def encode_ordinal_columns(df: pd.DataFrame, ordinal_categories: dict) -> pd.DataFrame:
    """Map declared-ordinal columns to integers in their configured natural order.

    ``ordinal_categories`` maps a column name to its category values ordered
    from lowest to highest (e.g. ``{"activity_level": ["Very Light", "Light",
    "Moderate", "Heavy", "Exceptional"]}``, see ``data.ordinal_column_categories``
    in :class:`~synthdata.config.DataConfig`). Unlike
    :func:`label_encode_non_numeric_columns`'s alphabetical fallback (used for
    columns nobody declared an order for), this preserves the column's true
    real-world ordering -- which matters for every backend that treats a
    "numeric" column as continuous, since their natural numeric order is
    exactly what a plain-numeric imputation/generation pass relies on to
    model it correctly. Missing values are preserved as NaN.

    Raises if a configured column is missing from ``df``, or if the column
    contains an observed value not present in its configured category list
    (fail loudly rather than silently coercing an unrecognized category to
    NaN or an arbitrary position).
    """
    out = df.copy()
    for col, categories in ordinal_categories.items():
        if col not in out.columns:
            raise KeyError(
                f"data.ordinal_column_categories references column {col!r}, which is not "
                f"present in the loaded data. Available columns: {list(out.columns)}"
            )
        cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
        observed = set(out[col].dropna().unique().tolist())
        unknown = observed - set(cat_to_idx)
        if unknown:
            raise ValueError(
                f"data.ordinal_column_categories[{col!r}] does not include observed value(s) "
                f"{sorted(unknown, key=str)}; every value present in the data must be listed "
                f"in its configured order (configured categories={categories})."
            )
        out[col] = out[col].map(cat_to_idx).astype(float)
    return out


def decode_ordinal_columns(df: pd.DataFrame, ordinal_categories: dict) -> pd.DataFrame:
    """Restore ordinal labels from their zero-based model-space codes.

    ``ordinal_categories`` maps each column to the labels used by
    :func:`encode_ordinal_columns`, ordered from low to high. Missing values
    remain missing. Non-integral or out-of-range codes are rejected instead of
    silently producing an invalid category label.
    """
    out = df.copy()
    for col, categories in ordinal_categories.items():
        if col not in out.columns:
            raise KeyError(
                f"Ordinal decoder references column {col!r}, which is not present in the "
                f"DataFrame. Available columns: {list(out.columns)}"
            )
        if not categories:
            raise ValueError(f"Ordinal decoder has no categories configured for column {col!r}")

        try:
            codes = pd.to_numeric(out[col], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Ordinal column {col!r} contains non-numeric model-space values and cannot "
                "be decoded"
            ) from exc

        non_integral = codes.notna() & codes.mod(1).ne(0)
        out_of_range = codes.notna() & ((codes < 0) | (codes >= len(categories)))
        if non_integral.any() or out_of_range.any():
            invalid_codes = codes[non_integral | out_of_range].dropna().tolist()
            raise ValueError(
                f"Ordinal column {col!r} contains invalid model-space code(s) "
                f"{invalid_codes}; expected integral values in [0, {len(categories) - 1}]"
            )

        out[col] = codes.map(dict(enumerate(categories)))
    return out


def warn_non_numeric_feature_columns(
    df: pd.DataFrame, feature_columns: list, categorical_columns: list
) -> list:
    """Log a loud warning for any feature column declared numeric but not actually numeric.

    ``categorical_columns`` here is the caller's already-combined
    ``nominal_columns + ordinal_columns`` list (see ``Dataset.categorical_columns``).
    A column not listed in it is assumed to already be numeric (e.g. an ordinal
    column pre-encoded to integers), but a plain CSV source can still surprise
    us with a string-valued column that was correctly excluded (e.g. an ordinal
    band stored as text like "Light"/"Heavy") yet was never actually
    numeric-encoded at the source. Every downstream imputation/generation
    backend that builds a numeric matrix falls back to label-encoding such
    columns (see :func:`label_encode_non_numeric_columns`), so this check
    doesn't change behavior -- it exists purely to surface the issue
    immediately at dataset-load time (this function is the single source of
    truth for this check; call it here rather than re-deriving the same
    "numeric_columns" filter independently in each backend), rather than it
    being discovered (or, worse, silently missed) deep inside whichever
    backend happens to run first.

    Returns the list of offending column names (empty if none).
    """
    numeric_columns = [c for c in feature_columns if c not in categorical_columns]
    offending = [c for c in numeric_columns if not pd.api.types.is_numeric_dtype(df[c])]
    if offending:
        logger.warning(
            "%d feature column(s) are not listed in data.nominal_columns/data.ordinal_columns "
            "(so are assumed numeric) but actually contain non-numeric values: %s. Every backend "
            "falls back to label-encoding these columns (preserves missingness, but does not "
            "guarantee categories are numbered in their true order -- alphabetical by default). "
            "Add them to data.nominal_columns (if unordered) or data.ordinal_columns (with an "
            "entry in data.ordinal_column_categories if not already numeric-coded) for correct "
            "treatment.",
            len(offending),
            offending,
        )
    return offending


def remap_binary_one_two(df: pd.DataFrame) -> pd.DataFrame:
    """Remap any column whose only non-null unique values are {1, 2} to {0, 1}."""
    out = df.copy()
    binary_cols = [c for c in out.columns if set(out[c].dropna().unique().tolist()) == {1, 2}]
    if binary_cols:
        out[binary_cols] = out[binary_cols] - 1
    return out


def cast_integer_like_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Cast fully-observed, whole-numbered columns to int dtype (no-op otherwise).

    Some libraries (e.g. SynthEval's ``AnalysisConfig``) infer "categorical" from
    dtype rather than cardinality, so a numerically-binary column stored as
    float (e.g. {0.0, 1.0}) would silently be treated as continuous downstream.
    """
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            continue
        series = out[c]
        if series.isna().any() or not pd.api.types.is_numeric_dtype(series):
            continue
        if np.all(np.mod(series, 1) == 0):
            out[c] = series.astype(int)
    return out


def label_encode_non_numeric_columns(
    df: pd.DataFrame, columns: list, categorical_columns: list | None = None
) -> tuple[pd.DataFrame, dict]:
    """Factorize non-numeric columns (and any declared categorical ones) to integer codes.

    Some backends (``TabImputeCategorical``, TabPFN) require a fully numeric
    matrix, but a plain CSV source (unlike the pre-encoded UCI hepatitis
    example) commonly has string-valued categorical columns (e.g.
    "Light"/"Heavy"/...) -- those are always factorized regardless of
    ``categorical_columns``. Missing values are preserved as NaN so they're
    still treated as missing rather than a category. Returns the encoded
    frame plus ``{column: categories}``, needed to decode output back to the
    original labels via :func:`decode_label_encoded_columns`.

    ``categorical_columns`` (if given) additionally forces factorization for
    columns that are *already numeric* but represent a category, not a
    continuous quantity -- e.g. a 5-level ordinal stored as raw values
    ``{1..5}``, or a binary variable stored as ``{0, 2}``. Without this, such
    a column would pass through unencoded, and some backends' internal
    categorical handling returns *compact 0-indexed class predictions*
    regardless of the input's actual value domain (confirmed for TabPFN's
    unsupervised experiment API): a 5-class column's synthetic output would
    come back as ``{0..4}`` instead of the true ``{1..5}``, and a ``{0, 2}``
    binary column's as ``{0, 1}`` instead of ``{0, 2}`` -- silently shifting/
    relabeling the column's entire domain in the synthetic output. Routing
    every declared categorical column through the same factorize/decode
    round-trip as string columns (regardless of dtype) guarantees the model
    only ever sees/produces compact 0-indexed codes internally, and that
    :func:`decode_label_encoded_columns` always maps back to the true
    observed domain afterward. Already-numeric columns *not* listed in
    ``categorical_columns`` (i.e. genuinely continuous ones) still pass
    through unchanged.
    """
    encoded = df[columns].copy()
    category_maps = {}
    force_factorize = set(categorical_columns or [])
    for col in columns:
        if pd.api.types.is_numeric_dtype(encoded[col]) and col not in force_factorize:
            continue
        codes, categories = pd.factorize(encoded[col], sort=True)
        codes = codes.astype(float)
        codes[codes == -1] = np.nan  # factorize maps NaN -> -1
        encoded[col] = codes
        category_maps[col] = categories
    return encoded, category_maps


def decode_label_encoded_columns(df: pd.DataFrame, category_maps: dict) -> pd.DataFrame:
    """Invert :func:`label_encode_non_numeric_columns`, mapping codes back to labels."""
    decoded = df.copy()
    for col, categories in category_maps.items():
        if col not in decoded.columns:
            continue
        codes = decoded[col].round().clip(0, len(categories) - 1).astype(int)
        decoded[col] = categories.take(codes)
    return decoded


def mask_outliers_as_missing(df: pd.DataFrame, columns: list, threshold: float) -> pd.DataFrame:
    """Set numeric values beyond ``threshold`` std-devs of their column mean to NaN.

    Plain (non-robust) mean/std, not a robust median/MAD-based z-score: many of
    this kind of column are zero-/mode-inflated ordinal-ish measures (e.g. a
    day-count column with median=MAD=1 but a legitimate long tail out to 30),
    for which MAD-based z-scores false-positive heavily on real boundary values
    (confirmed empirically) while under-flagging true outliers whenever the
    "bulk" of the column has zero MAD (all-too-common for zero-inflated
    columns). Plain std is itself inflated by genuine outliers, but for a
    single (or few) extreme value(s) among ``n`` otherwise-plausible ones its
    z-score stays roughly ``sqrt(n)`` regardless of how extreme the value is,
    which is more than enough separation at this dataset's scale. Catches both
    "not administered" sentinel codes (e.g. a lone 999 among otherwise 0-30
    values) and corrupt outlier rows (e.g. a derived metric blown up by a
    division artifact), either of which can otherwise cause float32 overflow
    inside TabPFN/TabImpute. Non-numeric and constant (zero-std) columns are
    left untouched.

    Deliberately a single pass (not iterative): re-fitting mean/std after each
    removal and repeating would catch smaller residual outliers, but also
    cascades into masking legitimate boundary values (confirmed empirically --
    e.g. removing a 999 sentinel from a 0-31 day-count column shrinks std
    enough that a legitimate, repeated 30 then looks like an "outlier" too).
    A single pass only removes the most egregious values -- exactly what's
    needed to avoid literal float32 infinity/overflow -- and leaves smaller
    (still-plausible) residual outliers alone.
    """
    out = df.copy()
    for col in columns:
        if col not in out.columns or not pd.api.types.is_numeric_dtype(out[col]):
            continue
        series = out[col]
        mean = series.mean()
        std = series.std()
        if not std or np.isnan(std):
            continue
        z = (series - mean).abs() / std
        outliers = z > threshold
        if outliers.any():
            logger.info(
                "Masking %d outlier value(s) in %r as missing (|z| > %.1f, e.g. %s)",
                int(outliers.sum()),
                col,
                threshold,
                series[outliers].tolist()[:5],
            )
            out.loc[outliers, col] = np.nan
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _persist_role_assignment(dataset: Dataset, assignment: RoleAssignment) -> None:
    """Persist an immutable assignment under its digest and a small mutable pointer."""
    assignment_root = ensure_dir(dataset.data_dir / "assignments")
    assignment_id = assignment.assignment_fingerprint
    if assignment.assignment_policy_fingerprint:
        assignment_id = f"{assignment_id}-{assignment.assignment_policy_fingerprint}"
    assignment_dir = assignment_root / assignment_id
    ensure_dir(assignment_dir)
    assignment_path = assignment_dir / "assignment.csv"
    manifest_path = assignment_dir / "manifest.json"
    if assignment_path.exists():
        existing = pd.read_csv(assignment_path)
        if existing.to_dict(orient="records") != assignment.assignment.to_dict(orient="records"):
            raise ValueError(
                f"Assignment digest collision or corruption at {assignment_path}; "
                "refusing to overwrite an immutable split assignment"
            )
    else:
        assignment.assignment.to_csv(assignment_path, index=False)
    if manifest_path.exists():
        with manifest_path.open() as manifest_file:
            existing_manifest = json.load(manifest_file)
        if existing_manifest.get("assignment_fingerprint") != assignment.assignment_fingerprint:
            raise ValueError(
                f"Assignment manifest {manifest_path} does not match its directory digest"
            )
    else:
        with manifest_path.open("w") as manifest_file:
            json.dump(
                {
                    "assignment_fingerprint": assignment.assignment_fingerprint,
                    "assignment_policy_fingerprint": assignment.assignment_policy_fingerprint,
                    "assignment_path": str(assignment_path.relative_to(dataset.data_dir)),
                    "metadata": assignment.metadata,
                    "created_at": datetime.now(UTC).isoformat(),
                    "git_commit": git_commit(),
                },
                manifest_file,
                indent=2,
                default=str,
            )
    pointer_path = dataset.data_dir / "assignment_index.json"
    with pointer_path.open("w") as pointer_file:
        json.dump(
            {
                "assignment_fingerprint": assignment.assignment_fingerprint,
                "assignment_policy_fingerprint": assignment.assignment_policy_fingerprint,
                "assignment_id": assignment_id,
                "assignment_manifest": str(manifest_path.relative_to(dataset.data_dir)),
            },
            pointer_file,
            indent=2,
        )


def _configured_identity_columns(split_cfg) -> set[str]:
    if split_cfg is None:
        return set()
    return {
        column
        for column in (
            split_cfg.patient_id_column,
            split_cfg.mapping_row_key_column,
            split_cfg.mapping_patient_key_column,
        )
        if column is not None
    }


def _validate_loader_column_declarations(
    cfg: Config,
    target_column: str,
    split_cfg,
) -> dict[str, list[str]]:
    drop_columns = [] if cfg.data.drop_columns is None else cfg.data.drop_columns
    declarations = {
        "protected_columns": list(cfg.data.protected_columns),
        "sensitive_columns": list(cfg.data.sensitive_columns),
        "quasi_identifier_columns": list(cfg.data.quasi_identifier_columns),
    }
    identity_columns = _configured_identity_columns(split_cfg)
    conflicts = {
        "target/identity": sorted({target_column} & identity_columns),
        "target/drop": sorted({target_column} & set(drop_columns)),
    }
    for declaration_name, columns in declarations.items():
        conflicts[f"{declaration_name}/identity"] = sorted(set(columns) & identity_columns)
        conflicts[f"{declaration_name}/drop"] = sorted(set(columns) & set(drop_columns))

    encounter_column = split_cfg.encounter_label_column if split_cfg is not None else None
    if encounter_column is not None:
        conflicts["encounter/target"] = sorted({encounter_column} & {target_column})
        conflicts["encounter/identity"] = sorted({encounter_column} & identity_columns)
        conflicts["encounter/drop"] = sorted({encounter_column} & set(drop_columns))
        for declaration_name, columns in declarations.items():
            conflicts[f"encounter/{declaration_name}"] = sorted({encounter_column} & set(columns))

    nonempty_conflicts = {name: values for name, values in conflicts.items() if values}
    if nonempty_conflicts:
        raise ValueError(
            "Conflicting data column declarations: "
            + "; ".join(f"{name}={values}" for name, values in nonempty_conflicts.items())
        )
    return declarations


def write_dataset_manifest(cfg: Config, dataset: Dataset) -> None:
    """Record which dataset version/source produced ``dataset.data_dir``.

    Unlike experiments (see :mod:`synthdata.experiment`), which version each
    generation/evaluation/plot *run*, this manifest versions the *dataset
    itself*: it is written once per `data_dir` (i.e. once per `data.version`)
    and updated (timestamp/commit refreshed) on every subsequent load, so a
    collaborator can tell exactly which source config produced any cached
    `data/<name>/<version>/` directory.
    """
    manifest_path = dataset.data_dir / "dataset_manifest.json"
    manifest = {
        "dataset_name": dataset.name,
        "dataset_version": dataset.version,
        "source": cfg.data.source,
        "uci_id": cfg.data.uci_id,
        "path": cfg.data.path,
        "target_column": dataset.target_column,
        "feature_columns": dataset.feature_columns,
        "nominal_columns": dataset.nominal_columns,
        "ordinal_columns": dataset.ordinal_columns,
        "variable_schema": dataset.variable_schema,
        "variable_schema_fingerprint": dataset.variable_schema_fingerprint,
        "source_fingerprint": dataset.source_fingerprint,
        "full_fingerprint": dataframe_fingerprint(dataset.full_df),
        "train_split_fingerprint": dataset.train_split_fingerprint,
        "test_split_fingerprint": dataset.test_split_fingerprint,
        "sensitive_columns": dataset.sensitive_columns,
        "protected_columns": dataset.protected_columns,
        "release_generalization": dataset.release_generalization,
        "n_rows": int(len(dataset.full_df)),
        "seed": cfg.seed,
        "last_loaded_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
    }
    if dataset.has_canonical_roles:
        manifest.update(
            {
                "compatibility_mode": None,
                "protected_columns": dataset.protected_columns,
                "quasi_identifier_columns": dataset.quasi_identifier_columns,
                "role_names": list(ROLE_NAMES),
                "role_fingerprints": dataset.role_fingerprints,
                "role_metadata": dataset.role_metadata,
                "assignment_fingerprint": dataset.assignment_fingerprint,
                "assignment_policy_fingerprint": dataset.assignment_policy_fingerprint,
                "semantic_fingerprint": dataset.semantic_fingerprint,
                "n_roles": {role: int(len(dataset.roles[role])) for role in ROLE_NAMES},
                "role_paths": {
                    role: str(dataset.paths()[role].relative_to(dataset.data_dir))
                    for role in ROLE_NAMES
                },
            }
        )
    else:
        manifest.update(
            {
                "compatibility_mode": "legacy_two_role",
                "role_names": ["train", "final_holdout"],
                "n_train": int(len(dataset.train_df)) if dataset.train_df is not None else 0,
                "n_test": int(len(dataset.test_df)) if dataset.test_df is not None else 0,
                "legacy_restrictions": [
                    "no_tuning_role",
                    "ineligible_for_new_hpo",
                    "ineligible_for_final_policy",
                    "ineligible_for_release_claims",
                ],
            }
        )
    key_fingerprint = dataset.role_metadata.get("identity", {}).get("hmac_key_fingerprint")
    if key_fingerprint is not None:
        manifest["hmac_key_fingerprint"] = key_fingerprint
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{manifest_path.name}.", dir=manifest_path.parent
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2, default=str)
            manifest_file.flush()
            os.fsync(manifest_file.fileno())
        os.replace(temporary_path, manifest_path)
        temporary_path = None
        if os.name == "posix":
            directory_fd = os.open(manifest_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary_path is not None:
            with suppress(FileNotFoundError):
                temporary_path.unlink()


def load_dataset(cfg: Config) -> Dataset:
    """Load, type, and split the dataset described by ``cfg.data``.

    A configured ``data.split`` produces canonical ``train``, ``tuning``, and
    ``final_holdout`` roles before any learned transformation. Historical
    two-role loading requires the explicit ``data.legacy_two_role: true`` flag.
    """
    data_dir_base = Path(cfg.data.data_dir)
    data_version_scope = f"data_v_{cfg.data.version}" if cfg.data.version else "data_v_unversioned"
    data_dir = ensure_dir(data_dir_base / data_version_scope)
    prior_manifest_path = data_dir / "dataset_manifest.json"
    expected_key_fingerprint = None
    if prior_manifest_path.exists():
        try:
            prior_manifest = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError("Dataset manifest read failed: io_error") from exc
        except json.JSONDecodeError as exc:
            raise ValueError("Dataset manifest read failed: invalid_json") from exc
        if not isinstance(prior_manifest, dict):
            raise ValueError("Dataset manifest validation failed: invalid_structure")
        if set(prior_manifest.get("role_names", ())) == set(ROLE_NAMES) and (
            "hmac_key_fingerprint" not in prior_manifest
        ):
            raise ValueError(
                "Existing canonical dataset manifest has no patient identity key fingerprint; "
                "refusing to bootstrap or rewrite identity. Perform an explicit identity "
                "migration or rotation before loading this dataset."
            )
        expected_key_fingerprint = prior_manifest.get("hmac_key_fingerprint")

    if cfg.data.source == "uci":
        df, variable_types = _load_uci(cfg, data_dir)
        source_fingerprint = dataframe_fingerprint(df)
    elif cfg.data.source in ("csv", "parquet"):
        df, variable_types = _load_local_file(cfg)
        assert cfg.data.path is not None
        source_fingerprint = file_fingerprint(cfg.data.path)
    else:
        raise ValueError(f"Unknown data.source: {cfg.data.source!r}")

    if cfg.data.uppercase_columns:
        df.columns = df.columns.str.upper()
        if variable_types is not None:
            variable_types = {k.upper(): v for k, v in variable_types.items()}

    if cfg.data.raw_target_column and cfg.data.raw_target_column in df.columns:
        df = df.rename(columns={cfg.data.raw_target_column: cfg.data.target_column})

    if cfg.data.remap_binary_one_two:
        df = remap_binary_one_two(df)

    target_column = cfg.data.target_column
    if target_column not in df.columns:
        raise KeyError(
            f"target_column '{target_column}' not found in loaded data columns: {list(df.columns)}"
        )

    if cfg.data.drop_rows_missing_target:
        n_before = len(df)
        df = df[df[target_column].notna()].reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped:
            logger.info(
                "Dropped %d/%d rows with missing target_column %r",
                n_dropped,
                n_before,
                target_column,
            )

    split_cfg = cfg.data.split
    if split_cfg is None and not cfg.data.legacy_two_role:
        raise ValueError(
            "data.split is required for canonical roles; set data.legacy_two_role=true "
            "only when intentionally reading historical train/test data"
        )
    if split_cfg is not None and cfg.data.legacy_two_role:
        raise ValueError("data.split and data.legacy_two_role are mutually exclusive")
    if split_cfg is not None and cfg.data.patient_id_column is not None:
        if split_cfg.mode != "patient_group":
            raise ValueError("data.patient_id_column requires data.split.mode='patient_group'")
        split_cfg = dataclasses.replace(
            split_cfg,
            patient_id_column=cfg.data.patient_id_column,
            identity_mapping_path=None,
            mapping_row_key_column=None,
            mapping_patient_key_column=None,
            one_row_per_patient=False,
        )
    semantic_declarations = _validate_loader_column_declarations(
        cfg,
        target_column,
        split_cfg,
    )
    token_scope = hashlib.sha256(
        json.dumps(
            {
                "dataset_name": cfg.name,
                "dataset_version": cfg.data.version,
                "source_fingerprint": source_fingerprint,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    identity = resolve_population_identity(
        df,
        split_cfg,
        token_scope=token_scope,
        token_key_path=data_dir_base / ".patient_id_hmac_key",
        expected_key_fingerprint=expected_key_fingerprint,
    )
    df = identity.model_frame
    groups = identity.groups
    if identity.identity_sidecar is not None:
        identity.identity_sidecar = identity.identity_sidecar.reset_index(drop=True)

    configured_identity_columns = _configured_identity_columns(split_cfg)

    if cfg.data.drop_columns:
        if split_cfg is not None and set(cfg.data.drop_columns) & configured_identity_columns:
            raise ValueError(
                "data.drop_columns cannot contain population identity columns; identity is "
                "removed from model frames by the split resolver"
            )
        df = df.drop(columns=[c for c in cfg.data.drop_columns if c in df.columns])

    feature_columns = [c for c in df.columns if c != target_column]
    modeling_columns = feature_columns + [target_column]
    variable_schema = {}
    variable_schema_fingerprint = None

    if cfg.data.variable_schema_path:
        variable_schema, variable_schema_fingerprint = load_variable_schema(
            cfg.data.variable_schema_path, modeling_columns
        )
        nominal_columns, ordinal_columns, ordinal_orders = schema_column_roles(
            variable_schema, target_column
        )
        if ordinal_orders:
            df = encode_ordinal_columns(df, ordinal_orders)
    else:
        # Legacy explicit lists remain readable so existing recorded experiments
        # can be reproduced. New datasets must use variable_schema_path; no
        # dtype/cardinality inference is permitted here.
        if cfg.data.nominal_columns is None:
            raise ValueError(
                "data.variable_schema_path is required for new datasets. Existing configurations "
                "may temporarily provide an explicit data.nominal_columns list together with "
                "data.ordinal_columns, but automatic type inference is not supported."
            )
        if cfg.data.ordinal_column_categories:
            df = encode_ordinal_columns(df, cfg.data.ordinal_column_categories)
        ordinal_columns = [c for c in cfg.data.ordinal_columns if c in feature_columns]
        nominal_columns = [
            c
            for c in cfg.data.nominal_columns
            if c in feature_columns and c not in set(ordinal_columns)
        ]
        variable_schema = {
            column: {
                "kind": "categorical"
                if column == target_column or column in nominal_columns + ordinal_columns
                else "continuous",
                "ordinal_order": cfg.data.ordinal_column_categories.get(column),
                "source_table": None,
                "lower_bound": None,
                "upper_bound": None,
                "causal_parents": [],
                "categorical_values": None,
            }
            for column in modeling_columns
        }
    categorical_columns = nominal_columns + ordinal_columns
    warn_non_numeric_feature_columns(df, feature_columns, categorical_columns)

    target_is_categorical = (
        variable_schema.get(target_column, {}).get("kind", "categorical") == "categorical"
    )

    # Cast whole-numbered categorical columns to a proper int dtype.
    # Some downstream tooling (e.g. SynthEval's AnalysisConfig) infers "categorical"
    # from dtype (object/int) rather than cardinality, so a float-typed {0.0, 1.0}
    # categorical column would otherwise silently be treated as continuous.
    columns_to_cast = categorical_columns + ([target_column] if target_is_categorical else [])
    df = cast_integer_like_columns(df, columns_to_cast)

    if cfg.data.outlier_zscore_threshold is not None and cfg.data.outlier_columns:
        outlier_columns = [
            c
            for c in cfg.data.outlier_columns
            if c in feature_columns and c not in categorical_columns
        ]
        df = mask_outliers_as_missing(df, outlier_columns, cfg.data.outlier_zscore_threshold)

    protected_columns = semantic_declarations["protected_columns"]
    sensitive_columns = semantic_declarations["sensitive_columns"]
    quasi_identifier_columns = semantic_declarations["quasi_identifier_columns"]
    for declaration_name, columns in semantic_declarations.items():
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise KeyError(
                f"data.{declaration_name} references column(s) not present in the modeling "
                f"frame: {missing}; available columns: {list(df.columns)}"
            )
        if target_column in columns:
            raise ValueError(
                f"data.{declaration_name} must not contain target column {target_column!r}"
            )

    release_generalization = json.loads(
        json.dumps(cfg.evaluation.release_generalization.columns, sort_keys=True, default=str)
    )
    invalid_release_columns = sorted(set(release_generalization) - set(feature_columns))
    if invalid_release_columns:
        raise ValueError(
            "evaluation.release_generalization.columns must reference model features only; "
            f"invalid column(s): {invalid_release_columns}"
        )
    identity_columns = _configured_identity_columns(split_cfg)
    identity_release_columns = sorted(set(release_generalization) & identity_columns)
    if identity_release_columns:
        raise ValueError(
            "evaluation.release_generalization.columns must not reference identity columns: "
            f"{identity_release_columns}"
        )

    if split_cfg is None:
        train_df, test_df = train_test_split(
            df,
            train_size=cfg.data.train_size,
            random_state=cfg.seed,
            stratify=df[target_column] if cfg.data.stratify else None,
        )
        dataset = Dataset(
            name=cfg.name,
            target_column=target_column,
            feature_columns=feature_columns,
            nominal_columns=nominal_columns,
            ordinal_columns=ordinal_columns,
            sensitive_columns=list(sensitive_columns),
            data_dir=data_dir,
            full_df=df,
            train_df=train_df,
            test_df=test_df,
            version=cfg.data.version,
            protected_columns=list(protected_columns),
            quasi_identifier_columns=quasi_identifier_columns,
            variable_schema=variable_schema,
            variable_schema_fingerprint=variable_schema_fingerprint,
            source_fingerprint=source_fingerprint,
            role_metadata={
                "compatibility_mode": "legacy_two_role",
                "identity": identity.metadata,
            },
            identity_sidecar=identity.identity_sidecar,
            identity_fingerprint=identity.metadata.get("identity_fingerprint"),
            release_generalization=release_generalization,
        )
        paths = dataset.paths()
        df.to_csv(paths["full"], index=False)
        train_df.to_csv(paths["train"], index=False)
        test_df.to_csv(paths["test"], index=False)
    else:
        if split_cfg.mode == "patient_group" and groups is None:
            raise ValueError(
                "patient_group mode resolved no population groups; refusing to fall back to row mode"
            )
        df = df.reset_index(drop=True)
        if groups is not None:
            groups = groups.reset_index(drop=True)
        role_assignment = allocate_roles(
            df,
            target_column,
            split_cfg,
            protected_columns=protected_columns,
            groups=groups,
            seed=cfg.seed,
        )
        resolved_semantics, semantic_fingerprint = resolve_schema_metadata(
            variable_schema, role_assignment.frames["train"]
        )
        role_metadata = {
            "compatibility_mode": None,
            "identity": identity.metadata,
            "split": role_assignment.metadata,
            "semantic": resolved_semantics,
        }
        dataset = Dataset(
            name=cfg.name,
            target_column=target_column,
            feature_columns=feature_columns,
            nominal_columns=nominal_columns,
            ordinal_columns=ordinal_columns,
            sensitive_columns=list(sensitive_columns),
            data_dir=data_dir,
            full_df=df,
            train_df=None,
            test_df=None,
            roles=role_assignment.frames,
            role_groups=role_assignment.groups,
            assignment=role_assignment.assignment,
            role_metadata=role_metadata,
            version=cfg.data.version,
            protected_columns=list(protected_columns),
            quasi_identifier_columns=quasi_identifier_columns,
            variable_schema=variable_schema,
            variable_schema_fingerprint=variable_schema_fingerprint,
            source_fingerprint=source_fingerprint,
            assignment_fingerprint=role_assignment.assignment_fingerprint,
            assignment_policy_fingerprint=role_assignment.assignment_policy_fingerprint,
            semantic_fingerprint=semantic_fingerprint,
            identity_sidecar=identity.identity_sidecar,
            identity_fingerprint=identity.metadata.get("identity_fingerprint"),
            release_generalization=release_generalization,
        )
        paths = dataset.paths()
        df.to_csv(paths["full"], index=False)
        for role in ROLE_NAMES:
            dataset.roles[role].to_csv(paths[role], index=False)
        _persist_role_assignment(dataset, role_assignment)

    write_dataset_manifest(cfg, dataset)

    role_log = (
        ", ".join(f"{role}={len(dataset.roles[role])}" for role in dataset.roles)
        if dataset.roles
        else "none"
    )
    logger.info(
        "Loaded dataset '%s' (version=%s): %d rows, %d features (%d categorical: %d nominal + "
        "%d ordinal), target=%r, protected=%s, roles=%s, compatibility=%s",
        cfg.name,
        cfg.data.version or "unversioned",
        len(df),
        len(feature_columns),
        len(categorical_columns),
        len(nominal_columns),
        len(ordinal_columns),
        target_column,
        protected_columns,
        role_log,
        "legacy_two_role" if dataset.legacy_two_role else "canonical",
    )
    return dataset


def load_imputed_splits(
    dataset: Dataset,
    expected_cache_key: str | None = None,
    phase: str = "candidate",
) -> Dataset:
    """Attach imputed role CSVs only when their provenance still matches.

    Canonical caches carry phase-aware fit/transform metadata.  ``candidate``
    is the default because generation and downstream candidate evaluation use
    the train-fitted cache; final refits can request ``phase="final"``.
    """
    if phase not in {"candidate", "final"}:
        raise ValueError(f"Unsupported imputation cache phase: {phase!r}")
    paths = dataset.paths()
    if dataset.has_canonical_roles and phase == "final":
        # Final refit artifacts must coexist with candidate artifacts. Keeping
        # them phase-specific prevents final evaluation from invalidating the
        # train-fitted cache used by generation.
        final_dir = dataset.data_dir / "imputation_final"
        imputed_paths = {role: final_dir / f"{role}_imputed.csv" for role in ROLE_NAMES}
        provenance_path = final_dir / IMPUTATION_CACHE_KEY_FILENAME
    else:
        provenance_path = dataset.data_dir / IMPUTATION_CACHE_KEY_FILENAME
        if dataset.has_canonical_roles:
            imputed_paths = {role: paths[f"{role}_imputed"] for role in ROLE_NAMES}
        else:
            imputed_paths = {
                "full": paths["full_imputed"],
                "train": paths["train_imputed"],
                "test": paths["test_imputed"],
            }
    if not all(path.exists() for path in imputed_paths.values()):
        return dataset

    if not provenance_path.exists():
        logger.warning(
            "Ignoring imputed CSVs under %s because provenance file %s is missing; "
            "rerun imputation to create a validated cache",
            dataset.data_dir,
            provenance_path,
        )
        return dataset
    try:
        with provenance_path.open() as provenance_file:
            provenance = json.load(provenance_file)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Ignoring imputed CSVs under %s because provenance file %s is invalid (%s); "
            "rerun imputation",
            dataset.data_dir,
            provenance_path,
            exc,
        )
        return dataset

    if expected_cache_key is not None and provenance.get("cache_key") != expected_cache_key:
        logger.warning(
            "Ignoring imputed CSVs under %s because cache key differs: stored=%s, expected=%s; "
            "rerun imputation",
            dataset.data_dir,
            provenance.get("cache_key"),
            expected_cache_key,
        )
        return dataset

    if dataset.has_canonical_roles:
        fit_roles = ["train"] if phase == "candidate" else ["train", "tuning"]
        transform_roles = ["train", "tuning"] if phase == "candidate" else ["final_holdout"]
        fit_frame = (
            dataset.roles["train"]
            if phase == "candidate"
            else pd.concat([dataset.roles["train"], dataset.roles["tuning"]], axis=0)
        )
        expected_provenance = {
            "cache_contract": "canonical_roles_v1",
            "source_fingerprint": dataset.source_fingerprint,
            "full_fingerprint": dataframe_fingerprint(dataset.full_df),
            "role_names": list(ROLE_NAMES),
            "role_fingerprints": dataset.role_fingerprints,
            "assignment_fingerprint": dataset.assignment_fingerprint,
            "semantic_fingerprint": dataset.semantic_fingerprint,
            "phase": phase,
            "fit_roles": fit_roles,
            "transform_roles": transform_roles,
            "fit_frame_fingerprint": dataframe_fingerprint(fit_frame),
        }
    else:
        assert dataset.train_df is not None
        assert dataset.test_df is not None
        expected_provenance = {
            "source_fingerprint": dataset.source_fingerprint,
            "full_fingerprint": dataframe_fingerprint(dataset.full_df),
            "train_split_fingerprint": dataframe_fingerprint(dataset.train_df),
            "test_split_fingerprint": dataframe_fingerprint(dataset.test_df),
        }
    mismatches = {
        field: (provenance.get(field), expected)
        for field, expected in expected_provenance.items()
        if provenance.get(field) != expected
    }
    if mismatches:
        logger.warning(
            "Ignoring stale imputed CSVs under %s; source/split fingerprints differ: %s",
            dataset.data_dir,
            mismatches,
        )
        return dataset

    try:
        frames = {name: pd.read_csv(path) for name, path in imputed_paths.items()}
    except (OSError, pd.errors.ParserError) as exc:
        logger.warning(
            "Ignoring imputed CSVs under %s because a role cache could not be read: %s; "
            "rerun imputation",
            dataset.data_dir,
            exc,
        )
        return dataset
    expected_columns = dataset.full_df.columns.tolist()
    recorded_row_counts = provenance.get("imputed_row_counts")
    if isinstance(recorded_row_counts, dict) and all(
        isinstance(recorded_row_counts.get(name), int) for name in imputed_paths
    ):
        expected_rows = {name: recorded_row_counts[name] for name in imputed_paths}
    else:
        if dataset.has_canonical_roles:
            expected_rows = {role: len(dataset.roles[role]) for role in ROLE_NAMES}
        else:
            assert dataset.train_df is not None
            assert dataset.test_df is not None
            expected_rows = {
                "full": len(dataset.full_df),
                "train": len(dataset.train_df),
                "test": len(dataset.test_df),
            }
    invalid_frames = {
        name: {
            "rows": len(frame),
            "expected_rows": expected_rows[name],
            "columns_match": frame.columns.tolist() == expected_columns,
        }
        for name, frame in frames.items()
        if len(frame) != expected_rows[name] or frame.columns.tolist() != expected_columns
    }
    if invalid_frames:
        logger.warning(
            "Ignoring imputed CSVs under %s because cached frame shapes/columns are stale: %s",
            dataset.data_dir,
            invalid_frames,
        )
        return dataset

    if dataset.has_canonical_roles:
        dataset.set_imputed_roles(frames)
    else:
        dataset.full_imputed_df = frames["full"]
        dataset.train_imputed_df = frames["train"]
        dataset.test_imputed_df = frames["test"]
        dataset.imputed_roles = {
            "train": frames["train"],
            "final_holdout": frames["test"],
        }
        dataset.attach_decoded_imputed_splits()
    return dataset
