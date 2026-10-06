"""Durable artifacts needed to render evaluation plots without recomputing metrics.

The evaluation stage persists the full log-disparity report tables because
``combined_evaluation.csv`` deliberately contains only their summary metrics.
This lets ``synthdata-plot`` reconstruct Plotly reports from recorded results
rather than rerunning any evaluator.
"""

import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import subprocess
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from numbers import Real
from pathlib import Path
from typing import Any, cast

import pandas as pd

from synthdata.data import semantic_context_digest
from synthdata.evaluation.catalog import CANONICAL_EXPECTED_MANIFEST, CUSTOM_CANONICAL_MANIFEST
from synthdata.evaluation.metric_contracts import (
    CONTRACT_REGISTRY_VERSION,
    CONTRACT_SCHEMA_VERSION,
    CONTRACT_STATES,
    DEFAULT_METRIC_CONTRACT_REGISTRY,
    DIRECTIONS,
    METRIC_USES,
    RESULT_STATUSES,
    VALUE_ROLES,
    AmbiguousMetricContractError,
    MetricAnchors,
    MetricContract,
    MetricContractRegistry,
    MetricEvaluationContext,
    MetricStatusRecord,
    MetricValidationResult,
    UnknownMetricContractError,
    is_verified_authoritative_tstr,
    safe_metric_metadata,
    safe_metric_status_error,
)
from synthdata.evaluation.syntheval_eval import (
    _execution_payload_failed,
    _execution_payload_succeeded,
)
from synthdata.utils import ensure_dir, get_logger

logger = get_logger(__name__)

_ARTIFACT_SCHEMA_VERSION = 1
_BUNDLE_NAME = "evaluation_artifacts-v1"
_ATTEMPTS_DIRNAME = "attempts"
_LOG_REPORT_TABLES = (
    "leaf_results",
    "hierarchy_results",
    "subgroup_table",
    "leaf_equity_table",
    "legend_table",
    "label_counts",
)
_SAFE_LOG_DISPARITY_REASONS = frozenset(
    {
        "evaluator_exception",
        "incomplete_report",
        "log_disparity_evaluation_failed",
        "log_disparity_evaluation_indeterminate",
        "malformed_report",
        "metric_evaluation_failed",
        "missing_declared_protected_fields",
        "missing_or_invalid_release_provenance",
        "missing release provenance",
        "missing_real_evidence_provenance",
        "missing_requested_evaluation_role",
        "report_state_missing_or_unknown",
        "synthetic_real_population_alias",
        "undeclared_protected_fields",
    }
)
_SAFE_EXCEPTION_TYPE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_GENERATOR_METADATA_SCHEMA_VERSION = "generator-metadata-v1"
_GENERATOR_METADATA_V2_SCHEMA_VERSION = "generator-metadata-v2"
_PATE_ACCOUNTING_FIELDS = (
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
_GENERATOR_METADATA_REQUIRED_FIELDS = (
    "schema_version",
    "generator_context",
    "plugin_name",
    "plugin_fqdn",
    "requested_parameters",
    "n_samples",
    "random_state",
    "privacy_accounting",
)
_SOURCE_PROVENANCE_SCHEMA_VERSION = "source-provenance-v4"
_LEGACY_SOURCE_PROVENANCE_SCHEMA_VERSIONS = frozenset(
    {"source-provenance-v2", "source-provenance-v3"}
)
#: SynthCity commit the editable fork's repairs are applied on top of.
_SYNTHCITY_BASELINE_REVISION = "0ef2950c8b9991c2742c90bed849a3c3b647f61c"
_SYNTHCITY_BASELINE_SOURCE = "pre_repair_fork_commit"
#: Why each editable fork diverges from its baseline. Recorded in evaluation
#: manifests so a bundle states which patched behavior produced its metrics.
_SYNTHCITY_FORK_REPAIRS = (
    {
        "id": "SC-GROUPS",
        "scope": "group-aware tabular loaders, generated namespaces, and grouped internal boundaries",
        "rationale": (
            "Patient-group evaluation requires aligned, disjoint group metadata and explicit "
            "blocking where SynthCity internals lack validated group-aware behavior."
        ),
    },
    {
        "id": "SC-DETECTION",
        "scope": "detector raw/effective identities and GMM operational exclusion",
        "rationale": (
            "Raw detector AUC must remain auditable while policy-facing distinguishability is "
            "inversion-aware; the GMM component probability is not a labelled binary detector."
        ),
    },
    {
        "id": "SC-FEATURE-RANK",
        "scope": "feature-importance rank direction and typed failure handling",
        "rationale": (
            "Agreement correlation is better when larger, and model or SHAP failures must not "
            "become finite random-looking policy values."
        ),
    },
    {
        "id": "SC-SCHEMA-JSD",
        "scope": "pre-coercion schema mismatch and shared-support statistical outputs",
        "rationale": (
            "Schema compatibility must be measured before dtype alignment, and JSD variable rows "
            "need common support and source-table metadata for aggregation."
        ),
    },
    {
        "id": "SC-ATTRIBUTE-INFERENCE",
        "scope": "explicit QIs, target typing, score policy, uncertainty, and worst-target aggregation",
        "rationale": (
            "Attribute inference requires a declared threat protocol, target-appropriate scoring, "
            "and per-target evidence rather than an all-feature or exact-equality shortcut."
        ),
    },
    {
        "id": "SC-PRIVACY-DIAGNOSTICS",
        "scope": "identifiability variants, structural proxy labels, DOMIAS raw/effective values",
        "rationale": (
            "Legacy formulas must remain readable, new weighted or proxy meanings must be named, "
            "and raw DOMIAS values must not be treated as calibrated release risk."
        ),
    },
    {
        "id": "SC-PATE",
        "scope": "PATE-GAN parameter forwarding, accounting, and generator metadata",
        "rationale": (
            "Requested and effective privacy accounting must be visible at the generator boundary "
            "so caches, HPO checkpoints, and evaluation artifacts can reconstruct the claim."
        ),
    },
    {
        "id": "SC-FAILURE-OBSERVABILITY",
        "scope": "metric and benchmark failure preservation",
        "rationale": (
            "An empty or failed evaluator result must remain a typed, testcase-specific failure "
            "rather than disappearing from a partial aggregate."
        ),
    },
)
_SYNTHEVAL_FORK_REPAIRS = (
    {
        "id": "SE-STRUCTURED-EXECUTION",
        "scope": "structured execution and normalized v2 outputs",
        "rationale": (
            "The root adapter needs per-method status, expected identities, and failure evidence."
        ),
    },
    {
        "id": "SE-METRIC-SEMANTICS",
        "scope": "metric semantics and metadata propagation",
        "rationale": (
            "Root metric contracts require stable emitted keys, uncertainty, and support metadata."
        ),
    },
    {
        "id": "SE-FIT-ROLE-PREPROCESSING",
        "scope": "fit-role preprocessing behavior",
        "rationale": (
            "Evaluation artifacts must identify the role used to fit preprocessing state."
        ),
    },
    {
        "id": "SE-HOLDOUT-UNKNOWN-CATEGORIES",
        "scope": "real holdout categories absent from train",
        "rationale": (
            "Classifier-based holdout metrics map unseen holdout categories to the train mode "
            "below a materiality cap instead of blocking on a few rare rows."
        ),
    },
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_GENERATION_CACHE_SCHEMA_VERSION = "generation-cache-v3"
_FINAL_REFIT_CACHE_SCHEMA_VERSION = "final-refit-v2"
_HPO_CONTEXT_SCHEMA_VERSIONS = frozenset({"hpo-context-v1", "hpo-context-v2"})
_LEGACY_CACHE_SCHEMA_VERSIONS = frozenset({"generation-cache-v2", "final-refit-v1"})
_CACHE_ENVELOPE_DYNAMIC_FIELDS = frozenset(
    {"cache_key", "row_count", "synthetic_data_sha256", "generator_metadata"}
)

# These dimensions are intentionally independent.  ``state`` remains the
# historical coarse result for consumers which have not migrated yet.
EVIDENCE_EXECUTION_STATES = frozenset({"not_started", "succeeded", "failed", "blocked"})
METRIC_COMPLETENESS_STATES = frozenset({"complete", "incomplete", "not_applicable"})
SCORE_COMPLETENESS_STATES = frozenset({"complete", "indeterminate", "not_applicable"})
AUDIT_OUTCOME_STATES = frozenset({"complete", "failed", "blocked", "indeterminate"})
_PROVENANCE_INVENTORY_FIELDS = (
    "raw_values",
    "normalized_values",
    "formulas",
    "anchors",
    "role_hashes",
    "release_transform_digest",
    "attack_protocol",
    "seeds",
    "supports",
    "intervals",
    "invalid_reasons",
    "selected_model_provenance",
)
_ROLE_HASH_MAP_KEYS = frozenset({"imputed_evaluation", "custom_raw_evaluation"})
_ROLE_NAMES = frozenset({"train", "tuning", "final_holdout", "refit_fit", "synthetic", "reference"})
_FIT_FRAME_FINGERPRINT_KEYS = frozenset({"raw", "imputed"})
_FINAL_REFIT_IDENTITY_KEYS = frozenset(
    {
        "model_name",
        "cache_key",
        "data_sha256",
        "metadata_sha256",
        "fit_frame_fingerprint",
        "fit_frame_fingerprints",
    }
)
_LEGACY_FINAL_EVIDENCE_SCHEMA = "final-holdout-evidence-legacy-v1"


@dataclass(frozen=True)
class GenerationInventory:
    """Validated generation outputs declared by an experiment manifest."""

    expected_outputs: tuple[str, ...]
    produced_outputs: tuple[str, ...]
    failed_outputs: tuple[str, ...]


def _final_evidence_state_defaults(evidence: Mapping[str, Any]) -> dict[str, str]:
    """Return explicit state dimensions while retaining legacy ``state`` semantics."""
    state = evidence.get("state")
    if state == "succeeded":
        return {
            "evidence_execution_state": "succeeded",
            "metric_completeness_state": "complete",
            "score_completeness_state": "complete",
            "audit_outcome_state": "complete",
        }
    if state == "blocked":
        return {
            "evidence_execution_state": "blocked",
            "metric_completeness_state": "not_applicable",
            "score_completeness_state": "not_applicable",
            "audit_outcome_state": "blocked",
        }
    if state == "failed":
        return {
            "evidence_execution_state": "failed",
            "metric_completeness_state": "incomplete",
            "score_completeness_state": "indeterminate",
            "audit_outcome_state": "failed",
        }
    return {
        "evidence_execution_state": "not_started",
        "metric_completeness_state": "not_applicable",
        "score_completeness_state": "not_applicable",
        "audit_outcome_state": "blocked",
    }


def _normalize_final_provenance_inventory(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Create complete, JSON-safe provenance inventory for explicit migration."""
    supplied = evidence.get("provenance_inventory")
    if supplied is not None and not isinstance(supplied, Mapping):
        raise ValueError("Final-holdout provenance_inventory must be an object")
    inventory = (
        {
            field: (
                dict(supplied[field])
                if isinstance(supplied.get(field), Mapping)
                else supplied.get(field, {})
            )
            for field in _PROVENANCE_INVENTORY_FIELDS
        }
        if isinstance(supplied, Mapping)
        else {field: {} for field in _PROVENANCE_INVENTORY_FIELDS}
    )
    if inventory["role_hashes"] == {} and isinstance(evidence.get("role_hashes"), Mapping):
        inventory["role_hashes"] = dict(evidence["role_hashes"])
    selected_model_provenance = inventory["selected_model_provenance"]
    if isinstance(selected_model_provenance, Mapping) and selected_model_provenance:
        inventory["selected_model_provenance"] = dict(selected_model_provenance)
    else:
        inventory["selected_model_provenance"] = {"model": evidence.get("selected_model")}
    return inventory


def artifact_bundle_dir(evaluation_dir: str | Path) -> Path:
    """Return the versioned artifact bundle directory for an evaluation."""
    return Path(evaluation_dir) / _BUNDLE_NAME


def select_evaluation_attempt(
    evaluation_dir: str | Path, *, experiment_id: str | None = None
) -> tuple[Path, dict[str, Any]]:
    """Select append-only evaluation destination and its lineage metadata.

    The canonical directory is retained for the first evaluation. Once any
    evidence exists there, subsequent evaluations receive exclusive attempt
    directories and never touch historical files.
    """
    root = Path(evaluation_dir)
    evidence_names = ("combined_evaluation.csv", "report.md", _BUNDLE_NAME)
    has_history = any((root / name).exists() for name in evidence_names)
    if not has_history and root.is_dir():
        has_history = any(path.name != _ATTEMPTS_DIRNAME for path in root.iterdir())
    if not has_history and (root / _ATTEMPTS_DIRNAME).is_dir():
        has_history = any((root / _ATTEMPTS_DIRNAME).iterdir())
    if not has_history:
        return root, {
            "attempt_id": "canonical",
            "attempt_root": str(root),
            "source_experiment_id": experiment_id,
            "prior_attempt": None,
        }
    attempts = root / _ATTEMPTS_DIRNAME
    ensure_dir(attempts)
    while True:
        attempt_id = (
            f"eval_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}_{uuid.uuid4().hex[:12]}"
        )
        attempt_root = attempts / attempt_id
        try:
            attempt_root.mkdir()
        except FileExistsError:
            continue
        return attempt_root, {
            "attempt_id": attempt_id,
            "attempt_root": str(attempt_root),
            "source_experiment_id": experiment_id,
            "prior_attempt": str(root),
        }


def expected_evaluation_context(
    dataset,
    *,
    classification_score: str | None = None,
) -> dict[str, Any]:
    """Return current role identities expected by candidate evaluation artifacts."""
    from synthdata.data import (
        dataframe_fingerprint,
        role_context_fingerprint,
        semantic_context_fingerprint,
    )

    context_roles = ("train", "final_holdout") if dataset.legacy_two_role else ("train", "tuning")
    full_context_roles = (
        ("train", "final_holdout")
        if dataset.legacy_two_role
        else ("train", "tuning", "final_holdout")
    )
    role_hashes = {}
    raw_role_hashes = {}
    for role in context_roles:
        frame = dataset.role_frame(role, imputed=True)
        if frame is None:
            raise ValueError(f"Evaluation artifact context requires a populated {role!r} role")
        role_hashes[role] = dataframe_fingerprint(frame)
        raw_frame = dataset.role_frame(role, imputed=False)
        if raw_frame is None:
            raise ValueError(f"Evaluation artifact context requires a populated raw {role!r} role")
        raw_role_hashes[role] = dataframe_fingerprint(raw_frame)
    return {
        "role_context_fingerprints": {
            "candidate": role_context_fingerprint(dataset, context_roles),
            "full": role_context_fingerprint(dataset, full_context_roles),
        },
        "role_hashes": role_hashes,
        "role_hashes_by_framework": {
            "synthcity": role_hashes,
            "syntheval": role_hashes,
            "custom": raw_role_hashes,
        },
        "semantic_context_fingerprint": semantic_context_fingerprint(
            dataset,
            classification_score=classification_score,
        ),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_parquet(temporary)
    os.replace(temporary, path)


def _model_artifact_id(model_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_name).strip("-") or "model"
    return f"{slug}-{hashlib.sha256(model_name.encode()).hexdigest()[:12]}"


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage_a_base_study_name(model_name: str) -> str:
    """Return generation's base study name for one supported failed HPO output."""
    if model_name.startswith("tabpfgen_"):
        supported_models = {"tabpfgen_standard_hpo", "tabpfgen_custom_hpo"}
        if model_name not in supported_models:
            raise ValueError(f"Unsupported Stage A failed output model {model_name!r}")
    elif not model_name.endswith("_hpo") or not model_name[: -len("_hpo")]:
        raise ValueError(f"Unsupported Stage A failed output model {model_name!r}")
    return f"hpo_{model_name[: -len('_hpo')]}"


def _load_stage_a_bound_hpo_context(
    generation_root: Path,
    failure: Mapping[str, Any],
    *,
    model_name: str,
) -> tuple[dict[str, Any], str]:
    """Load current digest-versioned context and require its failed-output bindings."""
    from synthdata.generation import hpo

    label = f"Stage A failure {model_name!r} HPO context"
    expected_digest = _sha256_digest(failure.get("hpo_context_digest"), f"{label} digest")
    expected_filename = f"hpo_context-{expected_digest}.json"
    context_path_value = _non_empty_string(failure.get("hpo_context_path"), f"{label} path")
    if context_path_value != expected_filename:
        raise ValueError(f"{label} path must reference its digest-versioned artifact")
    context_path = _relative_artifact_path(generation_root, context_path_value, label)
    if context_path.is_symlink() or not context_path.is_file():
        raise ValueError(f"{label} artifact must be a regular file")
    try:
        payload = json.loads(context_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} artifact is unreadable") from exc
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "context_digest",
        "context",
        "context_file",
    }:
        raise ValueError(f"{label} artifact has an invalid shape")
    if payload.get("schema_version") != hpo.HPO_CONTEXT_SCHEMA_VERSION:
        raise ValueError(f"{label} artifact has an unsupported schema")
    if payload.get("context_file") != expected_filename:
        raise ValueError(f"{label} artifact filename binding is invalid")
    if _sha256_digest(payload.get("context_digest"), f"{label} artifact digest") != expected_digest:
        raise ValueError(f"{label} artifact digest does not match generation binding")
    context = payload.get("context")
    try:
        validated_context = hpo._require_validated_hpo_context(context, label=label)
        recomputed_digest = hpo.hpo_context_digest(validated_context)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} context is invalid") from exc
    if recomputed_digest != expected_digest:
        raise ValueError(f"{label} context digest does not match generation binding")

    # The mutable pointer is the context persisted as current by this generation
    # run. A valid but stale versioned artifact alone cannot authorize omission.
    pointer_path = _relative_artifact_path(generation_root, "hpo_context.json", label)
    if pointer_path.is_symlink() or not pointer_path.is_file():
        raise ValueError(f"{label} current context pointer is missing or unsafe")
    try:
        current_payload = json.loads(pointer_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} current context pointer is unreadable") from exc
    if current_payload != payload:
        raise ValueError(f"{label} artifact is stale or differs from current generation context")

    expected_contract_digest = _sha256_digest(
        validated_context.get("stage_a_contract_digest"),
        f"{label}.stage_a_contract_digest",
    )
    if (
        _sha256_digest(
            failure.get("stage_a_contract_digest"),
            f"Stage A failure {model_name!r} contract digest",
        )
        != expected_contract_digest
    ):
        raise ValueError(f"Stage A failure {model_name!r} contract binding differs from context")
    study_name = _non_empty_string(
        failure.get("hpo_study"), f"Stage A failure {model_name!r}.hpo_study"
    )
    legacy_study = hpo.contextual_study_name(
        _stage_a_base_study_name(model_name), validated_context
    )
    study_matches = study_name == legacy_study
    if not study_matches:
        scoped_base_prefix = f"{_stage_a_base_study_name(model_name)}-"
        scoped_match = re.fullmatch(
            rf"{re.escape(scoped_base_prefix)}([0-9a-f]{{64}})-([0-9a-f]{{16}})",
            study_name,
        )
        if scoped_match is not None:
            scoped_base = f"{scoped_base_prefix}{scoped_match.group(1)}"
            study_matches = study_name == hpo.contextual_study_name(scoped_base, validated_context)
    if not study_matches:
        raise ValueError(
            f"Stage A failure {model_name!r} study does not match its bound HPO context"
        )
    return validated_context, expected_contract_digest


def load_generation_inventory(
    manifest_path: str | Path,
    generation_dir: str | Path,
) -> GenerationInventory:
    """Resolve and validate latest generation completeness from experiment evidence."""
    manifest_file = Path(manifest_path)
    if manifest_file.is_symlink() or not manifest_file.is_file():
        raise ValueError("Experiment manifest must be a regular file")
    try:
        manifest = json.loads(manifest_file.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Experiment manifest is unreadable") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("Experiment manifest must be an object")
    runs = manifest.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Experiment manifest runs must be a list")
    if any(not isinstance(run, Mapping) for run in runs):
        raise ValueError("Experiment manifest runs must contain objects")
    generation_runs = [run for run in runs if run.get("stage") == "generation"]
    if not generation_runs:
        raise ValueError("Experiment manifest has no generation record")
    generation = generation_runs[-1]
    artifacts_payload = generation.get("artifacts")
    if not isinstance(artifacts_payload, Mapping):
        raise ValueError("Generation manifest artifacts must be an object")

    inventory_fields = {"expected_outputs", "produced_outputs", "failed_outputs", "failed_models"}
    status = generation.get("status")
    if "status" not in generation:
        if inventory_fields & set(generation):
            raise ValueError("Legacy generation manifest cannot contain partial inventory fields")
        legacy_models = _string_list(artifacts_payload.get("models"), "Legacy generation models")
        if len(legacy_models) != len(set(legacy_models)):
            raise ValueError("Legacy generation models must not contain duplicates")
        return GenerationInventory(
            expected_outputs=tuple(legacy_models),
            produced_outputs=tuple(legacy_models),
            failed_outputs=(),
        )

    if not isinstance(status, str) or status not in {"complete", "partial"}:
        raise ValueError(f"Generation manifest has invalid status {status!r}")
    expected_outputs = _string_list(
        generation.get("expected_outputs"), "Generation expected_outputs"
    )
    produced_outputs = _string_list(
        generation.get("produced_outputs"), "Generation produced_outputs"
    )
    failed_outputs = _string_list(generation.get("failed_outputs"), "Generation failed_outputs")
    for label, values in (
        ("expected_outputs", expected_outputs),
        ("produced_outputs", produced_outputs),
        ("failed_outputs", failed_outputs),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"Generation {label} must not contain duplicates")
    if set(expected_outputs) != set(produced_outputs) | set(failed_outputs):
        raise ValueError("Generation expected_outputs must equal produced plus failed outputs")
    if set(produced_outputs) & set(failed_outputs):
        raise ValueError("Generation produced_outputs and failed_outputs must be disjoint")

    failed_models = generation.get("failed_models")
    if not isinstance(failed_models, list):
        raise ValueError("Generation failed_models must be a list")
    failed_by_model: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(failed_models):
        label = f"Generation failed_models[{index}]"
        if not isinstance(item, Mapping) or set(item) != {
            "model",
            "hpo_study",
            "hpo_context_path",
            "hpo_context_digest",
            "stage_a_contract_digest",
            "evidence_references",
        }:
            raise ValueError(f"{label} has an invalid shape")
        model_name = _non_empty_string(item.get("model"), f"{label}.model")
        if model_name in failed_by_model:
            raise ValueError("Generation failed_models must not contain duplicate models")
        failed_by_model[model_name] = item
    if set(failed_by_model) != set(failed_outputs):
        raise ValueError("Generation failed_models do not match failed_outputs")

    models = _string_list(artifacts_payload.get("models"), "Generation artifact models")
    if len(models) != len(set(models)) or set(models) != set(produced_outputs):
        raise ValueError("Generation artifact models do not match produced_outputs")
    n_models = generation.get("n_models")
    if (
        isinstance(n_models, bool)
        or not isinstance(n_models, int)
        or n_models != len(produced_outputs)
    ):
        raise ValueError("Generation n_models does not match produced_outputs")

    if status == "complete":
        if failed_outputs or failed_models or set(expected_outputs) != set(produced_outputs):
            raise ValueError("Complete generation manifest contains failed or missing outputs")
    elif not failed_outputs:
        raise ValueError("Partial generation manifest must declare failed outputs")

    generation_root = Path(generation_dir)
    if generation_root.is_symlink() or not generation_root.is_dir():
        raise ValueError("Generated-data root must be a regular directory")
    generation_root = generation_root.resolve()
    stage_a_root = generation_root / "hpo_stage_a"
    for model_name, failure in failed_by_model.items():
        context, expected_contract_digest = _load_stage_a_bound_hpo_context(
            generation_root, failure, model_name=model_name
        )
        study_name = cast(str, failure["hpo_study"])
        references = failure.get("evidence_references")
        if not isinstance(references, list) or not references:
            raise ValueError(f"Stage A failure {model_name!r} requires evidence references")
        trial_numbers: list[int] = []
        result_paths: set[str] = set()
        contract_digest: str | None = None
        for index, reference in enumerate(references):
            label = f"Stage A failure {model_name!r}.evidence_references[{index}]"
            if not isinstance(reference, Mapping) or set(reference) != {
                "trial_number",
                "result_path",
                "contract_digest",
            }:
                raise ValueError(f"{label} has an invalid shape")
            trial_number = reference.get("trial_number")
            if (
                isinstance(trial_number, bool)
                or not isinstance(trial_number, int)
                or trial_number < 0
            ):
                raise ValueError(f"{label}.trial_number must be a non-negative integer")
            result_path = _non_empty_string(reference.get("result_path"), f"{label}.result_path")
            digest = _sha256_digest(reference.get("contract_digest"), f"{label}.contract_digest")
            if (
                digest != expected_contract_digest
                or result_path in result_paths
                or (contract_digest is not None and digest != contract_digest)
            ):
                raise ValueError(f"{label} duplicates or conflicts with prior evidence")
            trial_numbers.append(trial_number)
            result_paths.add(result_path)
            contract_digest = digest

            from synthdata.generation import hpo

            evidence_reference = {
                "state": "pruned",
                "contract_digest": digest,
                "result_path": result_path,
            }
            try:
                hpo._validate_stage_a_result_artifact(
                    evidence_reference,
                    root=stage_a_root,
                    context=context,
                    expected_study_name=study_name,
                    expected_trial_number=trial_number,
                )
                result = hpo._read_stage_a_json_pinned(stage_a_root, stage_a_root / result_path)
            except (RuntimeError, ValueError) as exc:
                raise ValueError(f"{label} failed Stage A evidence validation") from exc
            if not any(
                check.get("screen") in hpo.STAGE_A_SCREEN_IDS and check.get("passed") is False
                for check in result["checks"]
            ):
                raise ValueError(f"{label} does not prove Stage A rejection")
        if sorted(trial_numbers) != list(range(len(trial_numbers))):
            raise ValueError(f"Stage A failure {model_name!r} evidence trials are incomplete")

    return GenerationInventory(
        expected_outputs=tuple(expected_outputs),
        produced_outputs=tuple(produced_outputs),
        failed_outputs=tuple(failed_outputs),
    )


def load_validated_generated_datasets(
    generation_dir: str | Path,
    dataset,
    *,
    model_names: list[str] | None = None,
    classification_score: str | None = None,
    generation_hpo_enabled: bool | None = None,
    generation_inventory: GenerationInventory | None = None,
) -> dict[str, pd.DataFrame]:
    """Load generated CSVs only when their current cache envelopes verify.

    Canonical evaluation treats generated data and its sidecar as one
    integrity-bound artifact.  This preflight deliberately rejects legacy or
    unlisted files rather than allowing them to reach any evaluator.
    """
    root = Path(generation_dir)
    if root.is_symlink() or not root.is_dir() or not root.resolve().is_dir():
        raise ValueError("Generated-data root must be a regular directory")
    root = root.resolve()
    csv_paths = sorted(root.glob("*.csv"))
    if any(
        path.is_symlink() or not path.is_file() or not path.resolve().is_file()
        for path in csv_paths
    ):
        raise ValueError("Generated CSVs must be contained regular files, not symlinks")
    discovered = {path.stem for path in csv_paths}
    sidecar_paths = sorted(root.glob("*.cache.json"))
    if any(
        path.is_symlink() or not path.is_file() or not path.resolve().is_file()
        for path in sidecar_paths
    ):
        raise ValueError("Generated cache sidecars must be contained regular files, not symlinks")
    sidecar_models = {path.name[: -len(".cache.json")] for path in sidecar_paths}
    if sidecar_models != discovered:
        missing_sidecars = sorted(discovered - sidecar_models)
        orphaned_sidecars = sorted(sidecar_models - discovered)
        raise ValueError(
            "Generated CSV/cache inventory mismatch; "
            f"missing_sidecars={missing_sidecars}, orphaned_sidecars={orphaned_sidecars}"
        )

    requested = set(model_names) if model_names else None
    if generation_inventory is not None:
        declared_produced = set(generation_inventory.produced_outputs)
        if discovered != declared_produced:
            missing = sorted(declared_produced - discovered)
            unexpected = sorted(discovered - declared_produced)
            raise ValueError(
                "Generated model inventory does not match experiment manifest; "
                f"missing={missing}, unexpected={unexpected}"
            )
        expected = set(generation_inventory.expected_outputs)
        allowed_missing = set(generation_inventory.failed_outputs)
        selected = expected if requested is None else requested
        unexplained_requested = selected - expected
        if unexplained_requested:
            raise ValueError(
                "Requested evaluation models are not declared by the generation manifest: "
                f"{sorted(unexplained_requested)}"
            )
        missing_requested = (selected - declared_produced) - allowed_missing
        if missing_requested:
            raise ValueError(
                "Generated model inventory mismatch; "
                f"missing={sorted(missing_requested)}, unexpected=[]"
            )
        validation_models = declared_produced
    else:
        expected = requested if requested is not None else discovered
        allowed_missing = set()
        unexplained_missing = expected - discovered
        unexpected = discovered - expected
        if unexplained_missing or unexpected:
            raise ValueError(
                "Generated model inventory mismatch; "
                f"missing={sorted(unexplained_missing)}, unexpected={sorted(unexpected)}"
            )
        validation_models = expected - allowed_missing
    if not validation_models:
        return {}

    from synthdata.data import role_context_payload, semantic_context_payload

    # Scope is an identity of persisted generated data, not a mutable setting
    # in the evaluation config.  Read every envelope before constructing the
    # expected role context so one stale config flag cannot reinterpret caches.
    cache_payloads: dict[str, Mapping[str, Any]] = {}
    persisted_scopes: dict[str, tuple[str, ...]] = {}
    for model_name in sorted(validation_models):
        metadata_path = _relative_artifact_path(root, f"{model_name}.cache.json", "Generated cache")
        if (
            metadata_path.is_symlink()
            or not metadata_path.is_file()
            or not metadata_path.resolve().is_file()
        ):
            raise ValueError(f"Generated cache sidecar is missing or unsafe for {model_name!r}")
        try:
            payload = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Generated cache sidecar is unreadable for {model_name!r}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"Generated cache for {model_name!r} must be an object")
        if payload.get("schema_version") != _GENERATION_CACHE_SCHEMA_VERSION:
            raise ValueError(f"Generated cache for {model_name!r} is not generation-cache-v3")
        cache_payloads[model_name] = payload
        persisted_scopes[model_name] = _persisted_generation_scope(
            payload, f"Generated cache for {model_name!r}"
        )

    scopes = set(persisted_scopes.values())
    if len(scopes) != 1:
        raise ValueError(
            "Selected generation caches have mixed persisted generation scopes: "
            f"{sorted((model, scope) for model, scope in persisted_scopes.items())}"
        )
    candidate_roles = next(iter(scopes))

    # Canonical candidate evaluation always includes tuning, even for a
    # non-HPO generator. HPO metadata still validates its own train+tuning
    # scope above; it must not redefine candidate provenance.
    candidate_roles = ("train", "final_holdout") if dataset.legacy_two_role else ("train", "tuning")
    expected_fit_context = role_context_payload(dataset, ("train",))
    # ``generation_hpo_enabled`` remains accepted for callers compiled against
    # this API, but is intentionally not consulted: persisted HPO metadata is
    # authoritative (including when config is stale or disagrees).
    expected_role_context = role_context_payload(dataset, candidate_roles)
    semantic_context = semantic_context_payload(
        dataset,
        classification_score=classification_score,
        roles=candidate_roles,
    )
    expected_columns = list(dataset.full_df.columns)
    expected_schema = dataset.variable_schema_fingerprint
    expected_registry = DEFAULT_METRIC_CONTRACT_REGISTRY.digest()
    loaded: dict[str, pd.DataFrame] = {}
    for model_name in sorted(validation_models):
        csv_path = _relative_artifact_path(root, f"{model_name}.csv", "Generated data")
        metadata_path = _relative_artifact_path(root, f"{model_name}.cache.json", "Generated cache")
        if (
            metadata_path.is_symlink()
            or not metadata_path.is_file()
            or not metadata_path.resolve().is_file()
        ):
            raise ValueError(f"Generated cache sidecar is missing or unsafe for {model_name!r}")
        payload = cache_payloads[model_name]
        _validate_cache_envelope(
            payload,
            f"Generated cache for {model_name!r}",
            expected_model_name=model_name,
            expected_role_context=expected_role_context,
            expected_role_context_fingerprint=_mapping_digest(expected_role_context),
            expected_fit_context=expected_fit_context,
            expected_fit_context_fingerprint=_mapping_digest(expected_fit_context),
            expected_semantic_context_fingerprint=semantic_context_digest(semantic_context),
            expected_registry_digest=expected_registry,
        )
        if payload.get("variable_schema_fingerprint") != expected_schema:
            raise ValueError(f"Generated cache for {model_name!r} has a stale variable schema")
        if payload["columns"] != expected_columns:
            raise ValueError(f"Generated cache for {model_name!r} has unexpected columns")
        data_digest = _file_digest(csv_path)
        if data_digest != payload["synthetic_data_sha256"]:
            raise ValueError(f"Generated data digest does not match cache for {model_name!r}")
        try:
            frame = pd.read_csv(csv_path)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            raise ValueError(f"Generated data is unreadable for {model_name!r}") from exc
        if list(frame.columns) != expected_columns or list(frame.columns) != payload["columns"]:
            raise ValueError(f"Generated data columns do not match cache for {model_name!r}")
        if len(frame) != payload["row_count"]:
            raise ValueError(f"Generated data row count does not match cache for {model_name!r}")
        loaded[model_name] = frame
    if generation_inventory is not None and requested is not None:
        return {model_name: loaded[model_name] for model_name in sorted(requested & set(loaded))}
    return loaded


def _non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _string_list(value: Any, label: str, *, unique: bool = False) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{label} must be a list of non-empty strings")
    if unique and len(value) != len(set(value)):
        raise ValueError(f"{label} must not contain duplicates")
    return value


def _string_mapping(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not key.strip() or not isinstance(item, str) or not item.strip()
        for key, item in value.items()
    ):
        raise ValueError(f"{label} must map non-empty strings to non-empty strings")
    return value


def _validate_source_provenance(
    payload: Any,
    *,
    path: Path | None = None,
    allow_legacy: bool = True,
) -> None:
    if not isinstance(payload, Mapping):
        label = f" at {path}" if path is not None else ""
        raise ValueError(f"Source provenance must be an object{label}")
    schema_version = payload.get("schema_version")
    if schema_version in _LEGACY_SOURCE_PROVENANCE_SCHEMA_VERSIONS:
        if not allow_legacy:
            raise ValueError("Legacy source provenance is not allowed")
        return
    if schema_version is None:
        return
    if schema_version != _SOURCE_PROVENANCE_SCHEMA_VERSION:
        label = f" at {path}" if path is not None else ""
        raise ValueError(f"Unsupported source provenance schema {schema_version!r}{label}")

    synthcity = payload.get("synthcity")
    if not isinstance(synthcity, Mapping):
        raise ValueError("Source provenance requires SynthCity provenance")
    baseline_revision = _non_empty_string(
        synthcity.get("baseline_revision"), "Source provenance SynthCity baseline revision"
    )
    if _GIT_REVISION_PATTERN.fullmatch(baseline_revision) is None:
        raise ValueError("Source provenance SynthCity baseline revision must be a full Git SHA")
    _non_empty_string(
        synthcity.get("baseline_source"), "Source provenance SynthCity baseline source"
    )


def _validate_generator_metadata_payload(payload: Any, label: str) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be an object")
    if payload.get("schema_version") not in {
        _GENERATOR_METADATA_SCHEMA_VERSION,
        _GENERATOR_METADATA_V2_SCHEMA_VERSION,
    }:
        raise ValueError(f"{label} has an unsupported schema")
    missing = [field for field in _GENERATOR_METADATA_REQUIRED_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"{label} is incomplete; missing {missing}")
    for field in ("plugin_name", "plugin_fqdn"):
        _non_empty_string(payload.get(field), f"{label}.{field}")
    if not isinstance(payload.get("requested_parameters"), Mapping):
        raise ValueError(f"{label}.requested_parameters must be an object")
    _non_negative_int(payload.get("n_samples"), f"{label}.n_samples")
    if payload.get("n_samples") == 0:
        raise ValueError(f"{label}.n_samples must be positive")
    if isinstance(payload.get("random_state"), bool) or not isinstance(
        payload.get("random_state"), int
    ):
        raise ValueError(f"{label}.random_state must be an integer")
    if payload.get("schema_version") == _GENERATOR_METADATA_V2_SCHEMA_VERSION:
        implementation_fingerprint = payload.get("implementation_fingerprint")
        if (
            not isinstance(implementation_fingerprint, str)
            or not implementation_fingerprint.strip()
        ):
            raise ValueError(f"{label}.implementation_fingerprint must be non-empty")
        if _SHA256_PATTERN.fullmatch(implementation_fingerprint) is None:
            raise ValueError(f"{label}.implementation_fingerprint must be a SHA-256 digest")
    context = payload.get("generator_context")
    if not isinstance(context, Mapping):
        raise ValueError(f"{label}.generator_context must be an object")
    claim_type = _non_empty_string(
        context.get("privacy_claim_type"), f"{label}.generator_context.privacy_claim_type"
    )
    if claim_type not in {"none", "formal_dp"}:
        raise ValueError(f"{label} has unknown privacy claim type {claim_type!r}")
    if claim_type != "formal_dp":
        if payload.get("privacy_accounting") is not None:
            raise ValueError(f"{label}.privacy_accounting must be null for privacy claim 'none'")
        return
    requested_accounting = context.get("requested_accounting")
    if not isinstance(requested_accounting, Mapping):
        raise ValueError(f"{label}.generator_context.requested_accounting must be an object")
    accounting = payload.get("privacy_accounting")
    if not isinstance(accounting, Mapping):
        raise ValueError(f"{label}.privacy_accounting must be an object for formal_dp")
    missing = [field for field in _PATE_ACCOUNTING_FIELDS if field not in accounting]
    if missing:
        raise ValueError(f"{label}.privacy_accounting is incomplete; missing {missing}")
    for parameter in ("epsilon", "delta", "alpha", "lamda"):
        requested = requested_accounting.get(parameter)
        if accounting.get(f"requested_{parameter}") != requested:
            raise ValueError(
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
            raise ValueError(f"{label}.privacy_accounting.{field} must be populated")
    if accounting.get("stopping_state") == "not_fitted":
        raise ValueError(f"{label}.privacy_accounting.stopping_state cannot be not_fitted")


def _mapping_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(payload), sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256_digest(value: Any, label: str) -> str:
    digest = _non_empty_string(value, label)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _persisted_generation_scope(payload: Mapping[str, Any], label: str) -> tuple[str, ...]:
    """Derive generation roles from immutable cache HPO metadata.

    All-null HPO fields identify ordinary train-only generation.  HPO caches
    must carry one complete canonical context, its schema marker, and its
    canonical digest; partial metadata is never interpreted heuristically.
    """
    context = payload.get("hpo_context")
    schema = payload.get("hpo_context_schema_version")
    digest = payload.get("hpo_context_digest")
    if context is None and schema is None and digest is None:
        role_context = payload.get("role_context")
        if isinstance(role_context, Mapping):
            roles = role_context.get("roles")
            if isinstance(roles, Mapping) and "tuning" in roles:
                return ("train", "tuning")
        return ("train",)
    if not isinstance(context, Mapping):
        raise ValueError(f"{label} has incomplete HPO metadata")
    if not isinstance(schema, str) or schema not in _HPO_CONTEXT_SCHEMA_VERSIONS:
        raise ValueError(f"{label}.hpo_context_schema_version is unsupported")
    _validate_persisted_hpo_context(context, schema, digest, label)
    return ("train", "tuning")


def _validate_persisted_hpo_context(
    context: Mapping[str, Any], schema: Any, digest: Any, label: str
) -> None:
    """Validate the HPO context according to its persisted schema version."""
    if schema == "hpo-context-v1":
        required_fields = {
            "schema_version",
            "task_type",
            "registry_digest",
            "stage_a_contract_digest",
            "metric_config",
            "expected_emitted_keys",
            "group_context",
            "role_context_fingerprint",
            "role_context",
        }
        if set(context) != required_fields or context.get("schema_version") != schema:
            raise ValueError(f"{label}.hpo_context has an invalid historical v1 shape")
        task_type = context.get("task_type")
        if not isinstance(task_type, str) or task_type not in {"classification", "regression"}:
            raise ValueError(f"{label}.hpo_context.task_type is invalid")
        _non_empty_string(context.get("registry_digest"), f"{label}.hpo_context.registry_digest")
        contract_digest = context.get("stage_a_contract_digest")
        if contract_digest is not None:
            _sha256_digest(contract_digest, f"{label}.hpo_context.stage_a_contract_digest")
        metric_config = context.get("metric_config")
        if not isinstance(metric_config, Mapping) or any(
            not isinstance(category, str)
            or not category.strip()
            or not isinstance(metrics, list)
            or any(not isinstance(metric, str) or not metric.strip() for metric in metrics)
            for category, metrics in metric_config.items()
        ):
            raise ValueError(f"{label}.hpo_context.metric_config is invalid")
        _string_list(
            context.get("expected_emitted_keys"),
            f"{label}.hpo_context.expected_emitted_keys",
            unique=True,
        )
        group_context = context.get("group_context")
        if group_context is not None and not isinstance(group_context, Mapping):
            raise ValueError(f"{label}.hpo_context.group_context must be an object or None")
        _non_empty_string(
            context.get("role_context_fingerprint"),
            f"{label}.hpo_context.role_context_fingerprint",
        )
        if not isinstance(context.get("role_context"), Mapping):
            raise ValueError(f"{label}.hpo_context.role_context must be an object")
        expected_digest = _mapping_digest(context)
    elif schema == "hpo-context-v2":
        from synthdata.generation.hpo import _require_validated_hpo_context, hpo_context_digest

        try:
            validated_context = _require_validated_hpo_context(
                context, label=f"{label}.hpo_context"
            )
            expected_digest = hpo_context_digest(validated_context)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}.hpo_context is invalid") from exc
    else:
        raise ValueError(f"{label}.hpo_context_schema_version is unsupported")

    supplied_digest = _sha256_digest(digest, f"{label}.hpo_context_digest")
    if supplied_digest != expected_digest:
        raise ValueError(f"{label}.hpo_context_digest does not match hpo_context")


def _validate_cache_envelope(
    payload: Any,
    label: str,
    *,
    expected_model_name: str | None = None,
    expected_role_context: Mapping[str, Any] | None = None,
    expected_role_context_fingerprint: str | None = None,
    expected_semantic_context_fingerprint: str | None = None,
    expected_registry_digest: str | None = None,
    expected_fit_context: Mapping[str, Any] | None = None,
    expected_fit_context_fingerprint: str | None = None,
) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be an object")
    schema_version = _non_empty_string(payload.get("schema_version"), f"{label}.schema_version")
    required_by_schema = {
        _GENERATION_CACHE_SCHEMA_VERSION: (
            "model_name",
            "role_context_fingerprint",
            "role_context",
            "fit_context_fingerprint",
            "fit_context",
            "variable_schema_fingerprint",
            "semantic_context",
            "semantic_context_digest",
            "columns",
            "task_type",
            "target_view",
            "n_samples",
            "seed",
            "device",
            "registry_digest",
            "hpo_context_schema_version",
            "hpo_context_digest",
            "hpo_context",
            "resolved_parameters",
            "generator_metadata_schema_version",
            "generator_context",
            "implementation_fingerprint",
            "cache_key",
            "row_count",
            "synthetic_data_sha256",
            "generator_metadata",
        ),
        _FINAL_REFIT_CACHE_SCHEMA_VERSION: (
            "model_name",
            "backend",
            "columns",
            "fit_roles",
            "fit_frame_fingerprint",
            "fit_frame_fingerprints",
            "input_role_hashes",
            "role_context_fingerprint",
            "role_context",
            "variable_schema_fingerprint",
            "semantic_context",
            "semantic_context_digest",
            "task_type",
            "target_view",
            "n_samples",
            "seed",
            "device",
            "parameters",
            "generator_metadata_schema_version",
            "generator_context",
            "implementation_fingerprint",
            "cache_key",
            "synthetic_data_sha256",
            "generator_metadata",
        ),
    }
    try:
        required_fields = required_by_schema[schema_version]
    except KeyError as exc:
        raise ValueError(f"{label} has an unsupported schema {schema_version!r}") from exc
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise ValueError(f"{label} is incomplete; missing {missing}")

    model_name = _non_empty_string(payload["model_name"], f"{label}.model_name")
    if expected_model_name is not None and model_name != expected_model_name:
        raise ValueError(f"{label}.model_name does not match the expected model")
    _string_list(payload["columns"], f"{label}.columns", unique=True)
    _non_empty_string(payload["task_type"], f"{label}.task_type")
    target_view = _non_empty_string(payload["target_view"], f"{label}.target_view")
    if target_view != "native":
        raise ValueError(f"{label}.target_view must be 'native'")
    _non_negative_int(payload["n_samples"], f"{label}.n_samples")
    if payload["n_samples"] == 0:
        raise ValueError(f"{label}.n_samples must be positive")
    if isinstance(payload["seed"], bool) or not isinstance(payload["seed"], int):
        raise ValueError(f"{label}.seed must be an integer")
    _non_empty_string(payload["device"], f"{label}.device")
    registry_digest = _sha256_digest(payload["registry_digest"], f"{label}.registry_digest")
    if expected_registry_digest is not None and registry_digest != expected_registry_digest:
        raise ValueError(f"{label}.registry_digest does not match the evaluation contract")

    role_context = payload["role_context"]
    if not isinstance(role_context, Mapping):
        raise ValueError(f"{label}.role_context must be an object")
    _validate_role_context_payload(role_context, f"{label}.role_context")
    role_context_fingerprint = _sha256_digest(
        payload["role_context_fingerprint"], f"{label}.role_context_fingerprint"
    )
    if _mapping_digest(role_context) != role_context_fingerprint:
        raise ValueError(f"{label}.role_context_fingerprint does not match role_context")
    if (
        expected_role_context_fingerprint is not None
        and role_context_fingerprint != expected_role_context_fingerprint
    ):
        raise ValueError(f"{label}.role_context_fingerprint does not match the evaluation manifest")
    if expected_role_context is not None and dict(role_context) != dict(expected_role_context):
        raise ValueError(f"{label}.role_context does not match the evaluation manifest")

    if schema_version == _GENERATION_CACHE_SCHEMA_VERSION:
        fit_context = payload["fit_context"]
        if not isinstance(fit_context, Mapping):
            raise ValueError(f"{label}.fit_context must be an object")
        _validate_role_context_payload(fit_context, f"{label}.fit_context")
        fit_context_fingerprint = _sha256_digest(
            payload["fit_context_fingerprint"], f"{label}.fit_context_fingerprint"
        )
        if _mapping_digest(fit_context) != fit_context_fingerprint:
            raise ValueError(f"{label}.fit_context_fingerprint does not match fit_context")
        if (
            expected_fit_context_fingerprint is not None
            and fit_context_fingerprint != expected_fit_context_fingerprint
        ):
            raise ValueError(
                f"{label}.fit_context_fingerprint does not match the generation contract"
            )
        if expected_fit_context is not None and dict(fit_context) != dict(expected_fit_context):
            raise ValueError(f"{label}.fit_context does not match the generation contract")

    semantic_manifest = {
        "semantic_context": payload["semantic_context"],
        "semantic_context_fingerprint": payload["semantic_context_digest"],
    }
    _validate_semantic_context_manifest(semantic_manifest)
    semantic_fingerprint = semantic_manifest["semantic_context_fingerprint"]
    if (
        expected_semantic_context_fingerprint is not None
        and semantic_fingerprint != expected_semantic_context_fingerprint
    ):
        raise ValueError(f"{label}.semantic_context_digest does not match the evaluation manifest")
    variable_schema_fingerprint = payload["variable_schema_fingerprint"]
    if variable_schema_fingerprint is not None:
        _sha256_digest(variable_schema_fingerprint, f"{label}.variable_schema_fingerprint")

    implementation_fingerprint = _sha256_digest(
        payload["implementation_fingerprint"], f"{label}.implementation_fingerprint"
    )
    cache_key = _sha256_digest(payload["cache_key"], f"{label}.cache_key")
    identity_payload = {
        key: value for key, value in payload.items() if key not in _CACHE_ENVELOPE_DYNAMIC_FIELDS
    }
    if cache_key != _mapping_digest(identity_payload):
        raise ValueError(f"{label}.cache_key does not match the cache identity")
    _sha256_digest(payload["synthetic_data_sha256"], f"{label}.synthetic_data_sha256")

    generator_context = payload["generator_context"]
    if not isinstance(generator_context, Mapping):
        raise ValueError(f"{label}.generator_context must be an object")
    _non_empty_string(
        payload["generator_metadata_schema_version"],
        f"{label}.generator_metadata_schema_version",
    )
    generator_metadata = payload["generator_metadata"]
    _validate_generator_metadata_payload(generator_metadata, f"{label}.generator_metadata")
    if generator_metadata["generator_context"] != generator_context:
        raise ValueError(f"{label}.generator_context does not match generator_metadata")
    if generator_metadata["n_samples"] != payload["n_samples"]:
        raise ValueError(f"{label}.n_samples does not match generator_metadata")
    if generator_metadata["random_state"] != payload["seed"]:
        raise ValueError(f"{label}.seed does not match generator_metadata")
    if (
        generator_metadata.get("schema_version") == _GENERATOR_METADATA_V2_SCHEMA_VERSION
        and generator_metadata.get("implementation_fingerprint") != implementation_fingerprint
    ):
        raise ValueError(f"{label}.implementation_fingerprint does not match generator_metadata")

    parameter_field = (
        "resolved_parameters"
        if schema_version == _GENERATION_CACHE_SCHEMA_VERSION
        else "parameters"
    )
    parameters = payload[parameter_field]
    if not isinstance(parameters, Mapping):
        raise ValueError(f"{label}.{parameter_field} must be an object")
    if generator_metadata["requested_parameters"] != dict(parameters):
        raise ValueError(f"{label}.{parameter_field} does not match generator_metadata")

    if schema_version == _GENERATION_CACHE_SCHEMA_VERSION:
        row_count = payload["row_count"]
        _non_negative_int(row_count, f"{label}.row_count")
        if row_count != payload["n_samples"]:
            raise ValueError(f"{label}.row_count does not match n_samples")
        hpo_context = payload["hpo_context"]
        hpo_schema = payload["hpo_context_schema_version"]
        hpo_digest = payload["hpo_context_digest"]
        if hpo_context is None:
            if hpo_schema is not None or hpo_digest is not None:
                raise ValueError(f"{label} has HPO metadata without an HPO context")
        else:
            if not isinstance(hpo_context, Mapping):
                raise ValueError(f"{label}.hpo_context must be an object or None")
            if not isinstance(hpo_schema, str) or hpo_schema not in _HPO_CONTEXT_SCHEMA_VERSIONS:
                raise ValueError(f"{label}.hpo_context_schema_version is unsupported")
            _validate_persisted_hpo_context(hpo_context, hpo_schema, hpo_digest, label)
    else:
        _non_empty_string(payload["backend"], f"{label}.backend")
        fit_roles = _string_list(payload["fit_roles"], f"{label}.fit_roles", unique=True)
        if fit_roles != ["train", "tuning"]:
            raise ValueError(f"{label}.fit_roles must be ['train', 'tuning']")
        _sha256_digest(payload["fit_frame_fingerprint"], f"{label}.fit_frame_fingerprint")
        fit_fingerprints = payload["fit_frame_fingerprints"]
        if not isinstance(fit_fingerprints, Mapping):
            raise ValueError(f"{label}.fit_frame_fingerprints must be an object")
        for representation in ("raw", "imputed"):
            _sha256_digest(
                fit_fingerprints.get(representation),
                f"{label}.fit_frame_fingerprints.{representation}",
            )
        role_hashes = payload["input_role_hashes"]
        if not isinstance(role_hashes, Mapping):
            raise ValueError(f"{label}.input_role_hashes must be an object")
        for representation in ("raw", "imputed"):
            representation_hashes = role_hashes.get(representation)
            if not isinstance(representation_hashes, Mapping):
                raise ValueError(f"{label}.input_role_hashes.{representation} must be an object")
            for role, digest in representation_hashes.items():
                _sha256_digest(
                    digest,
                    f"{label}.input_role_hashes.{representation}.{role}",
                )


def _validate_generator_metadata_manifest(
    payload: Any,
    *,
    manifest: Mapping[str, Any] | None = None,
    expected_model_names: list[str] | None = None,
    artifact_root: Path | None = None,
) -> None:
    if payload is None:
        return
    if not isinstance(payload, Mapping):
        raise ValueError("Evaluation artifact generator_metadata must be an object")
    if expected_model_names is not None and set(payload) != set(expected_model_names):
        raise ValueError("Evaluation artifact generator metadata model inventory does not match")
    manifest_role_context = manifest.get("role_context") if manifest is not None else None
    expected_role_context = (
        manifest_role_context.get("candidate")
        if isinstance(manifest_role_context, Mapping)
        else None
    )
    manifest_role_fingerprints = manifest.get("role_context_fingerprint") if manifest else None
    expected_role_context_fingerprint = (
        manifest_role_fingerprints.get("candidate")
        if isinstance(manifest_role_fingerprints, Mapping)
        else None
    )
    expected_semantic_fingerprint = (
        manifest.get("semantic_context_fingerprint") if manifest is not None else None
    )
    contract_entry = manifest.get("metric_contract_manifest") if manifest is not None else None
    expected_registry_digest = (
        contract_entry.get("digest") if isinstance(contract_entry, Mapping) else None
    )
    for model_name, entry in payload.items():
        _non_empty_string(model_name, "Evaluation artifact generator model name")
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"Evaluation artifact generator_metadata[{model_name!r}] must be an object"
            )
        state = _non_empty_string(
            entry.get("state"), f"Evaluation artifact generator_metadata[{model_name!r}].state"
        )
        if state == "present":
            cache_metadata = entry.get("cache_metadata")
            if (
                not isinstance(cache_metadata, Mapping)
                or cache_metadata.get("schema_version") in _LEGACY_CACHE_SCHEMA_VERSIONS
            ):
                _validate_generator_metadata_payload(
                    entry.get("metadata"),
                    f"Evaluation artifact generator_metadata[{model_name!r}].metadata",
                )
                continue
            _validate_cache_envelope(
                cache_metadata,
                f"Evaluation artifact generator_metadata[{model_name!r}].cache_metadata",
                expected_model_name=model_name,
                expected_role_context=expected_role_context,
                expected_role_context_fingerprint=expected_role_context_fingerprint,
                expected_semantic_context_fingerprint=expected_semantic_fingerprint,
                expected_registry_digest=expected_registry_digest,
            )
            metadata_path = Path(
                _non_empty_string(
                    entry.get("metadata_path"),
                    f"Evaluation artifact generator_metadata[{model_name!r}].metadata_path",
                )
            )
            data_path = Path(
                _non_empty_string(
                    entry.get("data_path"),
                    f"Evaluation artifact generator_metadata[{model_name!r}].data_path",
                )
            )
            if artifact_root is not None:
                metadata_path = _relative_artifact_path(
                    artifact_root, metadata_path, "Generator metadata"
                )
                data_path = _relative_artifact_path(artifact_root, data_path, "Generated data")
            if metadata_path.is_symlink() or data_path.is_symlink():
                raise ValueError("Generator cache artifacts must not be symlinks")
            if not metadata_path.is_file() or not metadata_path.resolve().is_file():
                raise FileNotFoundError("Generator metadata artifact is missing")
            if not data_path.is_file() or not data_path.resolve().is_file():
                raise FileNotFoundError("Generated data artifact is missing")
            try:
                sidecar_payload = json.loads(metadata_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("Generator metadata artifact is unreadable") from exc
            if sidecar_payload != dict(cache_metadata):
                raise ValueError("Generator cache envelope does not match its metadata sidecar")
            if _file_digest(metadata_path) != _sha256_digest(
                entry.get("metadata_sha256"),
                f"Evaluation artifact generator_metadata[{model_name!r}].metadata_sha256",
            ):
                raise ValueError(
                    "Generator metadata failed integrity verification; cache envelope does not "
                    "match its metadata sidecar"
                )
            data_digest = _file_digest(data_path)
            if data_digest != _sha256_digest(
                entry.get("data_sha256"),
                f"Evaluation artifact generator_metadata[{model_name!r}].data_sha256",
            ):
                raise ValueError("Generated data failed integrity verification")
            if data_digest != cache_metadata["synthetic_data_sha256"]:
                raise ValueError(
                    "Generator cache synthetic_data_sha256 does not match generated data"
                )
            try:
                frame = pd.read_csv(data_path)
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                raise ValueError("Generated data artifact is unreadable") from exc
            if list(frame.columns) != cache_metadata["columns"]:
                raise ValueError("Generated data columns do not match cache metadata")
            if len(frame) != cache_metadata["n_samples"]:
                raise ValueError("Generated data row count does not match cache metadata")
            if entry.get("metadata") != cache_metadata["generator_metadata"]:
                raise ValueError("Generator metadata entry does not match its cache envelope")
        elif state == "legacy":
            _validate_generator_metadata_payload(
                entry.get("metadata"),
                f"Evaluation artifact generator_metadata[{model_name!r}].metadata",
            )
        elif state == "invalid":
            _non_empty_string(
                entry.get("error"),
                f"Evaluation artifact generator_metadata[{model_name!r}].error",
            )
        elif state != "missing":
            raise ValueError(
                f"Evaluation artifact generator_metadata[{model_name!r}] has unknown state {state!r}"
            )


def _validate_role_context_payload(payload: Any, label: str) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be an object")
    if payload.get("schema_version") != "role-context-v1":
        raise ValueError(f"{label} has an unsupported schema")
    _non_empty_string(payload.get("dataset_name"), f"{label}.dataset_name")
    dataset_version = payload.get("dataset_version")
    if dataset_version is not None:
        _non_empty_string(dataset_version, f"{label}.dataset_version")
    roles = payload.get("roles")
    if not isinstance(roles, dict) or not roles:
        raise ValueError(f"{label}.roles must be a non-empty object")
    for role, role_payload in roles.items():
        _non_empty_string(role, f"{label} role name")
        if not isinstance(role_payload, Mapping):
            raise ValueError(f"{label}.roles[{role!r}] must be an object")
        for field in ("raw_fingerprint", "imputed_fingerprint"):
            fingerprint = role_payload.get(field)
            if fingerprint is not None:
                _non_empty_string(fingerprint, f"{label}.roles[{role!r}].{field}")
        _non_negative_int(role_payload.get("rows"), f"{label}.roles[{role!r}].rows")
    for field in (
        "assignment_fingerprint",
        "assignment_policy_fingerprint",
        "semantic_fingerprint",
        "variable_schema_fingerprint",
        "compatibility_mode",
    ):
        value = payload.get(field)
        if value is not None:
            _non_empty_string(value, f"{label}.{field}")


def _validate_evaluation_role_context_payload(payload: Any, label: str) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be an object")
    if payload.get("schema_version") != "evaluation-role-context-v1":
        raise ValueError(f"{label} has an unsupported schema")
    fit_roles = _string_list(payload.get("fit_roles"), f"{label}.fit_roles", unique=True)
    if not fit_roles:
        raise ValueError(f"{label}.fit_roles must not be empty")
    evidence_role = _non_empty_string(payload.get("evidence_role"), f"{label}.evidence_role")
    if evidence_role not in {"tuning", "final_holdout"}:
        raise ValueError(f"{label}.evidence_role has an unknown value {evidence_role!r}")
    fit_role = payload.get("fit_role")
    if fit_role is not None:
        _non_empty_string(fit_role, f"{label}.fit_role")
    for field in ("fit_frame", "evidence_frame"):
        _non_empty_string(payload.get(field), f"{label}.{field}")
    for field in (
        "assignment_fingerprint",
        "assignment_policy_fingerprint",
        "semantic_fingerprint",
        "variable_schema_fingerprint",
        "compatibility_mode",
    ):
        value = payload.get(field)
        if value is not None:
            _non_empty_string(value, f"{label}.{field}")


def _validate_role_context_manifest(manifest: Mapping[str, Any]) -> None:
    role_context = manifest.get("role_context")
    fingerprints = manifest.get("role_context_fingerprint")
    if role_context is None and fingerprints is None:
        return
    if not isinstance(role_context, dict):
        raise ValueError("Evaluation artifact role_context must be an object")
    for scope, payload in role_context.items():
        _non_empty_string(scope, "Evaluation artifact role-context scope")
        _validate_role_context_payload(payload, f"Evaluation artifact role_context[{scope!r}]")
    if not isinstance(fingerprints, dict):
        raise ValueError("Evaluation artifact role_context_fingerprint must be an object")
    if set(fingerprints) != set(role_context):
        raise ValueError(
            "Evaluation artifact role_context_fingerprint keys must match role_context scopes"
        )
    for scope, fingerprint in fingerprints.items():
        _non_empty_string(
            fingerprint,
            f"Evaluation artifact role_context_fingerprint[{scope!r}]",
        )


def _validate_semantic_context_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the versioned semantic context and its recorded digest."""
    context = manifest.get("semantic_context")
    fingerprint = manifest.get("semantic_context_fingerprint")
    if context is None and fingerprint is None:
        return
    if not isinstance(context, Mapping):
        raise ValueError("Evaluation artifact semantic_context must be an object")
    required = (
        "schema_version",
        "target_column",
        "task_type",
        "feature_columns",
        "protected_columns",
        "quasi_identifier_columns",
        "feature_types",
        "source_table",
    )
    missing = [field for field in required if field not in context]
    if missing:
        raise ValueError(f"Evaluation artifact semantic_context is incomplete; missing {missing}")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("Evaluation artifact semantic_context_fingerprint must be non-empty")
    try:
        expected = semantic_context_digest(context)
    except (TypeError, ValueError) as exc:
        raise ValueError("Evaluation artifact semantic_context is invalid") from exc
    if fingerprint != expected:
        raise ValueError(
            "Evaluation artifact semantic_context_fingerprint does not match semantic_context"
        )


def _finite_or_none(value: Any, label: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite real number or None")


def _non_negative_int(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    """Reject omitted or unknown keys in persisted closed-schema mappings."""
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"{label} has invalid fields; missing={missing}, unknown={unknown}")


def _validate_role_hash_map(value: Any, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    for role, digest in value.items():
        _non_empty_string(role, f"{label} role")
        if role not in _ROLE_NAMES:
            raise ValueError(f"{label} contains unknown role {role!r}")
        _sha256_digest(digest, f"{label}.{role}")


def _validate_role_hashes(value: Any, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    if set(value) == {"evidence"}:
        _validate_role_hash_map(value["evidence"], f"{label}.evidence")
        return
    _require_exact_keys(value, set(_ROLE_HASH_MAP_KEYS), label)
    for key in value:
        _validate_role_hash_map(value[key], f"{label}.{key}")


def _validate_fit_frame_fingerprints(value: Any, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    _require_exact_keys(value, set(_FIT_FRAME_FINGERPRINT_KEYS), label)
    for key in _FIT_FRAME_FINGERPRINT_KEYS:
        _non_empty_string(value[key], f"{label}.{key}")


def _validate_final_refit_identity(
    value: Any, label: str, *, require_complete: bool = True
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    _require_exact_keys(value, set(_FINAL_REFIT_IDENTITY_KEYS), label)
    legacy = value.get("fit_frame_fingerprints") is None
    for key in (
        "model_name",
        "cache_key",
        "data_sha256",
        "metadata_sha256",
        "fit_frame_fingerprint",
    ):
        _non_empty_string(value[key], f"{label}.{key}")
    _sha256_digest(value["data_sha256"], f"{label}.data_sha256")
    _sha256_digest(value["metadata_sha256"], f"{label}.metadata_sha256")
    if legacy:
        return
    _non_empty_string(value["fit_frame_fingerprint"], f"{label}.fit_frame_fingerprint")
    _validate_fit_frame_fingerprints(
        value["fit_frame_fingerprints"], f"{label}.fit_frame_fingerprints"
    )


def _validate_selected_model_provenance(value: Any, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    _require_exact_keys(value, {"model", "selection", "fit_roles", "refit"}, label)
    _non_empty_string(value["model"], f"{label}.model")
    if not isinstance(value["selection"], Mapping):
        raise ValueError(f"{label}.selection must be an object")
    _non_empty_string(value["selection"].get("source"), f"{label}.selection.source")
    _non_empty_string(value["selection"].get("model"), f"{label}.selection.model")
    _string_list(value["fit_roles"], f"{label}.fit_roles", unique=True)
    _validate_final_refit_identity(value["refit"], f"{label}.refit")


def _relative_artifact_path(root: Path, value: Any, label: str) -> Path:
    relative = Path(
        _non_empty_string(str(value) if isinstance(value, Path) else value, f"{label} path")
    )
    if relative.is_absolute():
        raise ValueError(f"{label} path must be relative to its recorded root")
    root = root.resolve()
    # Check lexical components before resolving: resolving first would hide an
    # intermediate or final symlink and make an apparently contained path ambiguous.
    current = root
    for component in relative.parts:
        if component in {"", "."}:
            continue
        if component == "..":
            current = current.parent
            continue
        current /= component
        if current.is_symlink():
            raise ValueError(f"{label} path contains a symlink component")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} path escapes its recorded root") from exc
    return path


def _safe_source_path(source: Path, label: str, *, contained_in: Path | None = None) -> Path:
    """Return source only when it is a contained, canonical regular file."""
    if source.is_symlink():
        raise ValueError(f"{label} source must not be a symlink")
    if not source.exists() or not source.is_file():
        raise ValueError(f"{label} source must be a regular file")
    resolved = source.resolve()
    if contained_in is not None:
        root = contained_in.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{label} source escapes its declared root") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"{label} source must resolve to a regular file")
    return resolved


def _safe_bundle_destination(bundle_dir: Path, relative: str, label: str) -> Path:
    """Return destination after rejecting traversal and symlinked components."""
    destination = _relative_artifact_path(bundle_dir, relative, label)
    bundle_root = bundle_dir.resolve()
    try:
        destination.relative_to(bundle_root)
    except ValueError as exc:
        raise ValueError(f"{label} destination escapes the artifact bundle") from exc
    current = bundle_root
    for component in destination.relative_to(bundle_root).parts[:-1]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{label} destination contains a symlink")
    if destination.is_symlink():
        raise ValueError(f"{label} destination must not be a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _ensure_safe_directory(path: Path, label: str) -> Path:
    """Create directory only when existing path components are not symlinks."""
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink component")
    return ensure_dir(path)


def _copy_regular_source(
    source: Path, destination: Path, label: str, *, contained_in: Path | None = None
) -> None:
    """Copy a regular source without following source or destination symlinks."""
    canonical_source = _safe_source_path(source, label, contained_in=contained_in)
    if destination.is_symlink() or not destination.parent.is_dir():
        raise ValueError(f"{label} destination is unsafe")
    shutil.copyfile(canonical_source, destination, follow_symlinks=False)


def _bundle_source_path(
    value: Any,
    *,
    bundle_dir: Path,
    label: str,
    source_root: Path | None = None,
) -> Path:
    """Resolve transient or already-bundled source paths without using CWD."""
    raw = Path(_non_empty_string(value, f"{label} path"))
    source = (
        raw
        if raw.is_absolute()
        else _relative_artifact_path(source_root or bundle_dir.parent, raw, label)
    )
    return _safe_source_path(source, label)


def _verified_artifact_path(root: Path, entry: Mapping[str, Any], label: str) -> Path:
    if not isinstance(entry, Mapping):
        raise ValueError(f"{label} manifest entry must be an object")
    path = _relative_artifact_path(root, entry.get("path"), label)
    digest = _non_empty_string(entry.get("sha256"), f"{label} sha256")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} sha256 must be a lowercase SHA-256 digest")
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} artifact is missing")
    if _file_digest(path) != digest:
        raise ValueError(f"{label} failed integrity verification")
    return path


def _clear_native_plot_destination(path: Path) -> None:
    """Remove native plot output without ever following a destination symlink."""
    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _native_plot_source_files(root: Path) -> list[tuple[Path, Path]]:
    """Return regular, contained native plot files and their relative IDs."""
    if root.is_symlink() or not root.is_dir():
        return []
    resolved_root = root.resolve()
    files: list[tuple[Path, Path]] = []
    for source in sorted(root.rglob("*")):
        if source.is_symlink() or not source.is_file():
            continue
        resolved_source = source.resolve()
        try:
            resolved_source.relative_to(resolved_root)
        except ValueError:
            continue
        files.append((source, source.relative_to(root)))
    return files


def _native_bundle_root(bundle_dir: Path, root_value: Any) -> Path:
    """Resolve native plot root and require it to remain inside its bundle."""
    root = _relative_artifact_path(bundle_dir.parent, root_value, "Native SynthEval plot root")
    if root.is_symlink():
        raise ValueError("Native SynthEval plot root must not be a symlink")
    try:
        root.relative_to(bundle_dir.resolve())
    except ValueError as exc:
        raise ValueError("Native SynthEval plot root escapes the artifact bundle") from exc
    return root


def _read_json_artifact(
    root: Path, entry: Mapping[str, Any], label: str
) -> tuple[dict[str, Any], Path]:
    path = _verified_artifact_path(root, entry, label)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} artifact is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} artifact must contain a JSON object")
    return payload, path


def _contract_registry_from_payload(
    payload: Mapping[str, Any], path: Path
) -> MetricContractRegistry:
    required = {
        "schema_version",
        "registry_version",
        "digest",
        "contracts",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Metric contract manifest is incomplete at {path}: {missing}")
    if payload["schema_version"] != CONTRACT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported metric contract schema at {path}")
    if payload["registry_version"] != CONTRACT_REGISTRY_VERSION:
        raise ValueError(
            f"Unsupported metric contract registry at {path}: {payload['registry_version']!r}"
        )
    digest = _non_empty_string(payload["digest"], "Metric contract manifest digest")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(
            f"Metric contract manifest digest must be a lowercase SHA-256 digest at {path}"
        )
    contracts_payload = payload["contracts"]
    if not isinstance(contracts_payload, list):
        raise ValueError(f"Metric contract manifest contracts must be a list at {path}")

    contracts = []
    contract_fields = (
        "contract_id",
        "framework",
        "emitted_key_pattern",
        "semantic_family",
        "direction",
        "value_role",
        "lifecycle_state",
        "allowed_uses",
        "execution_pass",
        "target_view",
        "population_unit",
        "group_safety",
        "required_roles",
        "raw_range",
        "anchors",
        "uncertainty_field",
        "sample_size_field",
        "preprocessing_contract",
        "classification_score_policy",
        "schema_prerequisites",
        "qualifiers",
        "status_reason",
        "framework_identity",
        "metric_version",
        "policy_transform",
        "preprocessing_fit_role",
        "uncertainty_semantics",
        "sample_size_unit",
        "normalization_method",
        "required_support",
        "release_transform_digest",
        "seed",
        "protocol_version",
    )
    for index, raw_contract in enumerate(contracts_payload):
        if not isinstance(raw_contract, Mapping):
            raise ValueError(f"Metric contract {index} is not an object at {path}")
        missing = [field for field in contract_fields if field not in raw_contract]
        if missing:
            raise ValueError(f"Metric contract {index} is incomplete at {path}: {missing}")
        anchors = raw_contract["anchors"]
        if not isinstance(anchors, Mapping) or set(anchors) != {"ideal", "chance", "bad"}:
            raise ValueError(f"Metric contract {index} has invalid anchors at {path}")
        for anchor_name, anchor_value in anchors.items():
            _finite_or_none(anchor_value, f"Metric contract {index} anchor {anchor_name!r}")
        raw_range = raw_contract["raw_range"]
        if raw_range is not None:
            if not isinstance(raw_range, list) or len(raw_range) != 2:
                raise ValueError(f"Metric contract {index} has invalid raw_range at {path}")
            for bound in raw_range:
                _finite_or_none(bound, f"Metric contract {index} raw_range bound")
            raw_range = tuple(raw_range)
        for field in ("allowed_uses", "required_roles", "schema_prerequisites", "qualifiers"):
            _string_list(raw_contract[field], f"Metric contract {index} {field}", unique=True)
        for field in (
            "contract_id",
            "framework",
            "emitted_key_pattern",
            "semantic_family",
            "execution_pass",
            "target_view",
            "population_unit",
            "group_safety",
            "framework_identity",
            "metric_version",
            "policy_transform",
        ):
            _non_empty_string(raw_contract[field], f"Metric contract {index} {field}")
        for field in (
            "preprocessing_contract",
            "preprocessing_fit_role",
            "uncertainty_semantics",
            "sample_size_unit",
            "normalization_method",
            "required_support",
            "release_transform_digest",
            "protocol_version",
        ):
            if raw_contract[field] is not None:
                _non_empty_string(raw_contract[field], f"Metric contract {index} {field}")
        try:
            contracts.append(
                MetricContract(
                    contract_id=raw_contract["contract_id"],
                    framework=raw_contract["framework"],
                    emitted_key_pattern=raw_contract["emitted_key_pattern"],
                    semantic_family=raw_contract["semantic_family"],
                    direction=raw_contract["direction"],
                    value_role=raw_contract["value_role"],
                    lifecycle_state=raw_contract["lifecycle_state"],
                    allowed_uses=frozenset(raw_contract["allowed_uses"]),
                    execution_pass=raw_contract["execution_pass"],
                    target_view=raw_contract["target_view"],
                    population_unit=raw_contract["population_unit"],
                    group_safety=raw_contract["group_safety"],
                    required_roles=tuple(raw_contract["required_roles"]),
                    raw_range=raw_range,
                    anchors=MetricAnchors(**dict(anchors)),
                    uncertainty_field=raw_contract["uncertainty_field"],
                    sample_size_field=raw_contract["sample_size_field"],
                    preprocessing_contract=raw_contract["preprocessing_contract"],
                    classification_score_policy=raw_contract["classification_score_policy"],
                    schema_prerequisites=tuple(raw_contract["schema_prerequisites"]),
                    qualifiers=tuple(raw_contract["qualifiers"]),
                    status_reason=raw_contract["status_reason"],
                    framework_identity=raw_contract["framework_identity"],
                    metric_version=raw_contract["metric_version"],
                    policy_transform=raw_contract["policy_transform"],
                    preprocessing_fit_role=raw_contract["preprocessing_fit_role"],
                    uncertainty_semantics=raw_contract["uncertainty_semantics"],
                    sample_size_unit=raw_contract["sample_size_unit"],
                    normalization_method=raw_contract["normalization_method"],
                    required_support=raw_contract["required_support"],
                    release_transform_digest=raw_contract["release_transform_digest"],
                    seed=raw_contract["seed"],
                    protocol_version=raw_contract["protocol_version"],
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Metric contract {index} is invalid") from exc
    try:
        registry = MetricContractRegistry(contracts)
    except ValueError as exc:
        raise ValueError("Metric contract manifest is invalid") from exc
    if registry.digest() != digest:
        raise ValueError(f"Metric contract manifest digest does not match its contracts at {path}")
    return registry


def _metric_context_from_payload(payload: Any, label: str) -> MetricEvaluationContext | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be an object or None")
    required = (
        "execution_pass",
        "target_view",
        "evaluation_role",
        "population_unit",
        "group_mode",
        "role_hashes",
        "resolved_configuration",
    )
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"{label} is incomplete; missing {missing}")
    for field in required[:5]:
        _non_empty_string(payload[field], f"{label}.{field}")
    role_hashes = _string_mapping(payload["role_hashes"], f"{label}.role_hashes")
    resolved_configuration = payload["resolved_configuration"]
    if not isinstance(resolved_configuration, dict):
        raise ValueError(f"{label}.resolved_configuration must be an object")
    try:
        return MetricEvaluationContext(
            execution_pass=payload["execution_pass"],
            target_view=payload["target_view"],
            evaluation_role=payload["evaluation_role"],
            population_unit=payload["population_unit"],
            group_mode=payload["group_mode"],
            role_hashes=role_hashes,
            resolved_configuration=resolved_configuration,
        )
    except ValueError as exc:
        raise ValueError(f"{label} is invalid") from exc


def _metric_status_record_from_payload(
    payload: Any,
    *,
    label: str,
    model_name: str,
    framework: str,
    context: MetricEvaluationContext | None,
    registry: MetricContractRegistry | None,
    allow_mixed_execution_pass: bool = False,
) -> MetricStatusRecord:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be an object")
    required = (
        "model_name",
        "expected_key",
        "framework",
        "status",
        "contract_id",
        "is_expected",
        "raw_value",
        "policy_value",
        "uncertainty",
        "sample_size",
        "observed_count",
        "execution_pass",
        "target_view",
        "population_unit",
        "group_mode",
        "required_roles",
        "role_hashes",
        "allowed_uses",
        "value_role",
        "lifecycle_state",
        "direction",
        "policy_transform",
        "qualifiers",
        "error",
        "source_metadata",
    )
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"{label} is incomplete; missing {missing}")
    if payload["model_name"] != model_name:
        raise ValueError(
            f"{label} has model_name {payload['model_name']!r}, expected {model_name!r}"
        )
    if payload["framework"] != framework:
        raise ValueError(f"{label} has framework {payload['framework']!r}, expected {framework!r}")
    expected_key = _non_empty_string(payload["expected_key"], f"{label}.expected_key")
    status = _non_empty_string(payload["status"], f"{label}.status")
    if status not in RESULT_STATUSES:
        raise ValueError(f"{label} has unknown status {status!r}")
    if not isinstance(payload["is_expected"], bool):
        raise ValueError(f"{label}.is_expected must be boolean")
    contract_id = payload["contract_id"]
    if contract_id is not None:
        _non_empty_string(contract_id, f"{label}.contract_id")
    execution_pass = _non_empty_string(payload["execution_pass"], f"{label}.execution_pass")
    target_view = _non_empty_string(payload["target_view"], f"{label}.target_view")
    population_unit = _non_empty_string(payload["population_unit"], f"{label}.population_unit")
    group_mode = _non_empty_string(payload["group_mode"], f"{label}.group_mode")
    if population_unit not in {"row", "patient_group"}:
        raise ValueError(f"{label}.population_unit has unknown value {population_unit!r}")
    if group_mode not in {"row", "patient_group"}:
        raise ValueError(f"{label}.group_mode has unknown value {group_mode!r}")
    required_roles = _string_list(payload["required_roles"], f"{label}.required_roles", unique=True)
    role_hashes = _string_mapping(payload["role_hashes"], f"{label}.role_hashes")
    allowed_uses = _string_list(payload["allowed_uses"], f"{label}.allowed_uses", unique=True)
    if not set(allowed_uses) <= METRIC_USES:
        raise ValueError(f"{label}.allowed_uses contains an unknown metric use")
    value_role = _non_empty_string(payload["value_role"], f"{label}.value_role")
    if value_role not in VALUE_ROLES:
        raise ValueError(f"{label}.value_role has unknown value {value_role!r}")
    lifecycle_state = _non_empty_string(payload["lifecycle_state"], f"{label}.lifecycle_state")
    if lifecycle_state not in CONTRACT_STATES:
        raise ValueError(f"{label}.lifecycle_state has unknown value {lifecycle_state!r}")
    direction = payload["direction"]
    if direction is not None and direction not in DIRECTIONS:
        raise ValueError(f"{label}.direction has unknown value {direction!r}")
    policy_transform = _non_empty_string(payload["policy_transform"], f"{label}.policy_transform")
    qualifiers = _string_list(payload["qualifiers"], f"{label}.qualifiers", unique=True)
    error = payload["error"]
    if error is not None and not isinstance(error, str):
        raise ValueError(f"{label}.error must be a string or None")
    expected_error = safe_metric_status_error(status)
    safe_error_pattern = (
        rf"^{re.escape(expected_error)}; exception_type=[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)$"
        if expected_error
        else r"$^"
    )
    if (
        error is not None
        and error != expected_error
        and re.fullmatch(safe_error_pattern, error) is None
    ):
        # Durable status errors are intentionally a closed schema.  In
        # particular, evaluator exception bodies must never be persisted.
        raise ValueError(f"{label}.error is not a safe metric status reason")
    source_metadata = payload["source_metadata"]
    if not isinstance(source_metadata, dict):
        raise ValueError(f"{label}.source_metadata must be an object")
    source_metadata = safe_metric_metadata(
        source_metadata, label=f"{label}.source_metadata", strict=True
    )
    result_metadata = payload.get(
        "result_metadata",
        source_metadata.get("result_metadata", {}),
    )
    if not isinstance(result_metadata, dict):
        raise ValueError(f"{label}.result_metadata must be an object")
    result_metadata = safe_metric_metadata(
        result_metadata, label=f"{label}.result_metadata", strict=True
    )
    if (
        "result_metadata" in source_metadata
        and source_metadata["result_metadata"] != result_metadata
    ):
        raise ValueError(f"{label}.result_metadata does not match source_metadata")
    fit_roles = payload.get("fit_roles", ())
    if not isinstance(fit_roles, list) or any(not isinstance(role, str) for role in fit_roles):
        raise ValueError(f"{label}.fit_roles must be a list of strings")
    provenance = payload.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError(f"{label}.provenance must be an object")
    provenance = safe_metric_metadata(provenance, label=f"{label}.provenance", strict=True)
    support = safe_metric_metadata(payload.get("support"), label=f"{label}.support", strict=True)
    bandwidth = safe_metric_metadata(
        payload.get("bandwidth"), label=f"{label}.bandwidth", strict=True
    )
    for field in ("raw_value", "policy_value", "uncertainty"):
        _finite_or_none(payload[field], f"{label}.{field}")
    sample_size = payload["sample_size"]
    if sample_size is not None:
        _non_negative_int(sample_size, f"{label}.sample_size")
    _non_negative_int(payload["observed_count"], f"{label}.observed_count")
    try:
        record = MetricStatusRecord(
            model_name=model_name,
            expected_key=expected_key,
            framework=framework,
            status=status,
            contract_id=contract_id,
            is_expected=payload["is_expected"],
            raw_value=payload["raw_value"],
            policy_value=payload["policy_value"],
            uncertainty=payload["uncertainty"],
            sample_size=sample_size,
            observed_count=payload["observed_count"],
            execution_pass=execution_pass,
            target_view=target_view,
            population_unit=population_unit,
            group_mode=group_mode,
            required_roles=tuple(required_roles),
            role_hashes=role_hashes,
            allowed_uses=frozenset(allowed_uses),
            value_role=value_role,
            lifecycle_state=lifecycle_state,
            direction=direction,
            policy_transform=policy_transform,
            qualifiers=tuple(qualifiers),
            error=error,
            source_metadata=source_metadata,
            result_metadata=result_metadata,
            fit_roles=tuple(fit_roles),
            support=support,
            bandwidth=bandwidth,
            provenance=provenance,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is invalid") from exc

    if context is not None:
        if not allow_mixed_execution_pass and record.execution_pass != context.execution_pass:
            raise ValueError(f"{label}.execution_pass does not match its evaluation context")
        if record.population_unit != context.population_unit:
            raise ValueError(f"{label}.population_unit does not match its evaluation context")
        if record.group_mode != context.group_mode:
            raise ValueError(f"{label}.group_mode does not match its evaluation context")

    contract = None
    resolved_contract = None
    if registry is not None:
        try:
            resolved_contract = registry.resolve(
                framework=record.framework,
                emitted_key=record.expected_key,
                execution_pass=record.execution_pass,
            )
        except (AmbiguousMetricContractError, UnknownMetricContractError):
            resolved_contract = None
        if contract_id is None:
            if resolved_contract is not None:
                raise ValueError(f"{label} is missing the contract_id for a known metric")
        else:
            try:
                contract = registry.get(contract_id)
            except UnknownMetricContractError as exc:
                raise ValueError(f"{label} references an unknown contract {contract_id!r}") from exc
            if resolved_contract is None or resolved_contract.contract_id != contract.contract_id:
                raise ValueError(f"{label} contract_id does not own its emitted metric identity")
            if context is not None and context.evaluation_role == "final_holdout":
                expected_roles = tuple(
                    "final_holdout" if role == "tuning" else role
                    for role in contract.required_roles
                )
            else:
                expected_roles = contract.required_roles
            expected_metadata = {
                "required_roles": tuple(expected_roles),
                "allowed_uses": contract.allowed_uses,
                "value_role": contract.value_role,
                "lifecycle_state": contract.lifecycle_state,
                "direction": contract.direction,
                "policy_transform": contract.policy_transform,
                "qualifiers": contract.qualifiers,
            }
            actual_metadata = {
                "required_roles": record.required_roles,
                "allowed_uses": record.allowed_uses,
                "value_role": record.value_role,
                "lifecycle_state": record.lifecycle_state,
                "direction": record.direction,
                "policy_transform": record.policy_transform,
                "qualifiers": record.qualifiers,
            }
            if actual_metadata != expected_metadata:
                raise ValueError(f"{label} contract metadata does not match its contract")
            if record.status == "succeeded":
                if record.raw_value is None:
                    raise ValueError(f"{label} succeeded without a raw value")
                if contract.value_role == "policy_scalar" and record.policy_value is None:
                    raise ValueError(f"{label} policy scalar succeeded without a policy value")
                if contract.value_role == "diagnostic" and record.policy_value is not None:
                    raise ValueError(f"{label} diagnostic unexpectedly has a policy value")
            for metadata_name, expected in (
                ("protocol_version", contract.protocol_version),
                ("seed", contract.seed),
                ("release_transform_digest", contract.release_transform_digest),
            ):
                requires_observed = (
                    record.status == "succeeded"
                    and expected is not None
                    and (metadata_name == "seed" or expected == "release-evidence-v2")
                )
                if requires_observed and (
                    metadata_name not in record.provenance
                    or record.provenance.get(metadata_name) != expected
                ):
                    raise ValueError(f"{label} {metadata_name} does not match its contract")
            if contract.required_support is not None:
                support = record.support
                if isinstance(support, Mapping):
                    support_contract = support.get("support_contract", support.get("contract"))
                    if support_contract != contract.required_support:
                        raise ValueError(f"{label} support does not match its contract")
                    support_state = support.get("state", support.get("status"))
                    if support_state in {
                        "missing",
                        "insufficient",
                        "invalid",
                        "blocked",
                        "unsupported",
                        "indeterminate",
                    }:
                        raise ValueError(f"{label} support is not valid")
                    if record.status == "succeeded" and len(support) <= 1:
                        raise ValueError(f"{label} support is empty or incomplete")
                    if (
                        record.status == "succeeded"
                        and contract.required_support == "all_target_protected_cells"
                    ):
                        slices = support.get("slices")
                        if not isinstance(slices, (list, tuple)) or not slices:
                            raise ValueError(f"{label} equalized-odds support is incomplete")
                        if any(
                            not isinstance(cell, Mapping)
                            or not {"protected_column", "target_class", "state"}.issubset(cell)
                            or cell["state"] != "valid"
                            for cell in slices
                        ):
                            raise ValueError(f"{label} equalized-odds support is incomplete")
                    elif record.status == "succeeded" and not any(
                        key not in {"support_contract", "contract", "state", "status"}
                        for key in support
                    ):
                        raise ValueError(f"{label} support is incomplete")
                elif support != contract.required_support:
                    raise ValueError(f"{label} support does not match its contract")
            _finite_or_none(record.bandwidth, f"{label}.bandwidth")
    return record


def _metric_validation_result_from_payload(
    payload: Any,
    *,
    label: str,
    model_name: str,
    framework: str,
    registry: MetricContractRegistry | None,
    allow_mixed_execution_pass: bool = False,
) -> MetricValidationResult:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be an object")
    required = (
        "model_name",
        "requested_use",
        "contract_digest",
        "audit_complete",
        "complete",
        "succeeded",
        "decision_eligible",
        "policy_rank_eligible",
        "decision_status",
        "expected_keys",
        "completed_keys",
        "failed_keys",
        "indeterminate_keys",
        "status_counts",
        "evaluation_context",
        "records",
    )
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"{label} is incomplete; missing {missing}")
    if payload["model_name"] != model_name:
        raise ValueError(
            f"{label} has model_name {payload['model_name']!r}, expected {model_name!r}"
        )
    requested_use = _non_empty_string(payload["requested_use"], f"{label}.requested_use")
    if requested_use not in METRIC_USES:
        raise ValueError(f"{label} has unknown requested use {requested_use!r}")
    contract_digest = _non_empty_string(payload["contract_digest"], f"{label}.contract_digest")
    if registry is not None and contract_digest != registry.digest():
        raise ValueError(f"{label} contract digest does not match the persisted contract manifest")
    context = _metric_context_from_payload(
        payload["evaluation_context"], f"{label}.evaluation_context"
    )
    record_payloads = payload["records"]
    if not isinstance(record_payloads, list):
        raise ValueError(f"{label}.records must be a list")
    if record_payloads and context is None:
        raise ValueError(f"{label} has records but no evaluation context")
    records = tuple(
        _metric_status_record_from_payload(
            record,
            label=f"{label}.records[{index}]",
            model_name=model_name,
            framework=framework,
            context=context,
            registry=registry,
            allow_mixed_execution_pass=allow_mixed_execution_pass,
        )
        for index, record in enumerate(record_payloads)
    )
    expected_record_keys = [record.expected_key for record in records if record.is_expected]
    if len(expected_record_keys) != len(set(expected_record_keys)):
        raise ValueError(f"{label} contains duplicate expected metric identities")
    try:
        result = MetricValidationResult(
            model_name=model_name,
            requested_use=requested_use,
            contract_digest=contract_digest,
            records=records,
            evaluation_context=context,
        )
    except ValueError as exc:
        raise ValueError(f"{label} is invalid") from exc

    sequences = {
        "expected_keys": result.expected_keys,
        "completed_keys": result.completed_keys,
        "failed_keys": result.failed_keys,
        "indeterminate_keys": result.indeterminate_keys,
    }
    for field, expected in sequences.items():
        actual = _string_list(payload[field], f"{label}.{field}", unique=True)
        if actual != list(expected):
            raise ValueError(f"{label}.{field} does not match its records")
    status_counts = payload["status_counts"]
    if not isinstance(status_counts, dict):
        raise ValueError(f"{label}.status_counts must be an object")
    for status, count in status_counts.items():
        if status not in RESULT_STATUSES:
            raise ValueError(f"{label}.status_counts contains unknown status {status!r}")
        _non_negative_int(count, f"{label}.status_counts[{status!r}]")
    if status_counts != result.status_counts:
        raise ValueError(f"{label}.status_counts does not match its records")
    for field, expected in (
        ("audit_complete", result.audit_complete),
        ("complete", result.complete),
        ("succeeded", result.succeeded),
        ("decision_eligible", result.decision_eligible),
        ("policy_rank_eligible", result.policy_rank_eligible),
        ("decision_status", result.decision_status),
    ):
        if payload[field] != expected:
            raise ValueError(f"{label}.{field} does not match its records")
    if not isinstance(payload["decision_status"], str):
        raise ValueError(f"{label}.decision_status must be a string")
    return result


def _validate_metric_status_payload(
    payload: Mapping[str, Any],
    *,
    framework: str,
    container_key: str,
    registry: MetricContractRegistry | None,
    manifest_source_provenance: Mapping[str, Any] | None,
    expected_semantic_context_fingerprint: str | None = None,
) -> None:
    if payload.get("schema_version") != _ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"{framework} metric status sidecar has an unsupported schema")
    if payload.get("framework") != framework:
        raise ValueError(f"Unexpected framework in {framework} metric status sidecar")
    source_provenance = payload.get("source_provenance")
    if not isinstance(source_provenance, dict):
        raise ValueError(f"{framework} metric status sidecar source_provenance must be an object")
    if manifest_source_provenance is not None and source_provenance != manifest_source_provenance:
        raise ValueError(
            f"{framework} metric status sidecar source provenance does not match the artifact manifest"
        )
    semantic_context = payload.get("semantic_context")
    semantic_fingerprint = payload.get("semantic_context_fingerprint")
    if semantic_context is None:
        if semantic_fingerprint is not None:
            raise ValueError(
                f"{framework} metric status sidecar semantic_context_fingerprint requires semantic_context"
            )
    else:
        if not isinstance(semantic_context, Mapping):
            raise ValueError(
                f"{framework} metric status sidecar semantic_context must be an object or None"
            )
        if not isinstance(semantic_fingerprint, str) or not semantic_fingerprint:
            raise ValueError(
                f"{framework} metric status sidecar semantic_context_fingerprint must be non-empty"
            )
        try:
            expected_fingerprint = semantic_context_digest(semantic_context)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{framework} metric status sidecar semantic_context is invalid"
            ) from exc
        if semantic_fingerprint != expected_fingerprint:
            raise ValueError(
                f"{framework} metric status sidecar semantic_context_fingerprint does not match semantic_context"
            )
    if (
        expected_semantic_context_fingerprint is not None
        and semantic_fingerprint != expected_semantic_context_fingerprint
    ):
        raise ValueError(
            f"{framework} metric status sidecar semantic_context_fingerprint does not match "
            "the evaluation artifact manifest"
        )
    actual_container_key = container_key
    container = payload.get(actual_container_key)
    legacy_custom_models = framework == "custom" and container_key == "passes" and container is None
    if legacy_custom_models:
        actual_container_key = "models"
        container = payload.get(actual_container_key)
    if not isinstance(container, dict):
        raise ValueError(
            f"{framework} metric status sidecar is missing its {container_key!r} mapping"
        )
    if actual_container_key == "models":
        for model_name, result_payload in container.items():
            model_name = _non_empty_string(model_name, f"{framework} model name")
            result = _metric_validation_result_from_payload(
                result_payload,
                label=f"{framework} model {model_name!r}",
                model_name=model_name,
                framework=framework,
                registry=registry,
            )
            if framework == "custom" and (
                result.evaluation_context is None
                or result.evaluation_context.execution_pass != "main"
            ):
                raise ValueError(
                    f"{framework} legacy model {model_name!r} must be a main-pass result"
                )
        return
    for pass_identity, model_results in container.items():
        pass_identity = _non_empty_string(pass_identity, f"{framework} execution pass")
        pass_framework, separator, execution_pass = pass_identity.partition(":")
        if not separator or pass_framework != framework:
            raise ValueError(f"Invalid {framework} execution pass identity {pass_identity!r}")
        _non_empty_string(execution_pass, f"{framework} execution pass name")
        if not isinstance(model_results, dict):
            raise ValueError(f"{framework} pass {pass_identity!r} must map models to results")
        for model_name, result_payload in model_results.items():
            model_name = _non_empty_string(model_name, f"{framework} model name")
            result = _metric_validation_result_from_payload(
                result_payload,
                label=f"{framework} pass {pass_identity!r}, model {model_name!r}",
                model_name=model_name,
                framework=pass_framework,
                registry=registry,
            )
            if result.evaluation_context is None:
                raise ValueError(
                    f"{framework} pass {pass_identity!r}, model {model_name!r} has no evaluation context"
                )
            if result.evaluation_context.execution_pass != execution_pass:
                raise ValueError(
                    f"{framework} pass {pass_identity!r}, model {model_name!r} has a mismatched execution pass"
                )


def _validate_syntheval_execution_payload(
    payload: Any,
    *,
    label: str,
    model_name: str,
    execution_pass: str,
    expected_semantic_context_fingerprint: str | None = None,
    expected_semantic_context: Mapping[str, Any] | None = None,
) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be an object")
    if payload.get("model_name") != model_name:
        raise ValueError(f"{label} has a mismatched model_name")
    if payload.get("schema_version") != "syntheval-execution-v1":
        raise ValueError(f"{label} has an unsupported execution schema")
    if payload.get("pass_id") != execution_pass:
        raise ValueError(f"{label} has a mismatched pass_id")
    expected_target_view = {
        "main": "native",
        "binary_target": "binary_collapsed",
    }.get(execution_pass)
    if expected_target_view is None:
        raise ValueError(f"{label} has an unknown execution pass")
    if expected_target_view is not None and payload.get("target_view") != expected_target_view:
        raise ValueError(f"{label} has a mismatched target_view")
    for field in ("expected_manifest_digest", "context_fingerprint"):
        _non_empty_string(payload.get(field), f"{label}.{field}")
    for field in ("execution_complete", "execution_succeeded", "policy_eligible"):
        if not isinstance(payload.get(field), bool):
            raise ValueError(f"{label}.{field} must be boolean")
    for field in ("group_context", "role_context", "preprocessing_metadata"):
        value = payload.get(field)
        if value is not None and not isinstance(value, dict):
            raise ValueError(f"{label}.{field} must be an object or None")
    semantic_context = payload.get("semantic_context")
    semantic_fingerprint = payload.get("semantic_context_digest")
    if semantic_context is None:
        if semantic_fingerprint is not None:
            raise ValueError(f"{label}.semantic_context_digest requires semantic_context")
    else:
        if not isinstance(semantic_context, Mapping):
            raise ValueError(f"{label}.semantic_context must be an object or None")
        if not isinstance(semantic_fingerprint, str) or not semantic_fingerprint:
            raise ValueError(f"{label}.semantic_context_digest must be non-empty")
        try:
            expected_semantic_fingerprint = semantic_context_digest(semantic_context)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}.semantic_context is invalid") from exc
        if semantic_fingerprint != expected_semantic_fingerprint:
            raise ValueError(f"{label}.semantic_context_digest does not match semantic_context")
    if (
        expected_semantic_context_fingerprint is not None
        and semantic_fingerprint != expected_semantic_context_fingerprint
    ):
        raise ValueError(
            f"{label}.semantic_context_digest does not match the evaluation artifact manifest"
        )
    if expected_semantic_context is not None and dict(semantic_context or {}) != dict(
        expected_semantic_context
    ):
        raise ValueError(f"{label}.semantic_context does not match its pass manifest")
    if payload.get("role_context") is not None:
        _validate_evaluation_role_context_payload(payload["role_context"], f"{label}.role_context")
    preprocessing_fingerprint = payload.get("preprocessing_fingerprint")
    if preprocessing_fingerprint is not None:
        _non_empty_string(preprocessing_fingerprint, f"{label}.preprocessing_fingerprint")
    execution_payload = dict(payload)
    if _execution_payload_succeeded(
        execution_payload,
        expected_pass_id=execution_pass,
        expected_target_view=expected_target_view,
    ):
        return
    if _execution_payload_failed(
        execution_payload,
        expected_pass_id=execution_pass,
        expected_target_view=expected_target_view,
    ):
        return
    raise ValueError(f"{label} failed structured execution validation")


def _load_contract_registry_if_present(
    bundle_dir: Path,
    manifest: Mapping[str, Any],
    *,
    allow_legacy: bool = False,
) -> MetricContractRegistry | None:
    entry = manifest.get("metric_contract_manifest")
    if not entry:
        return None
    payload, path = _read_json_artifact(
        bundle_dir.parent,
        entry,
        "Metric contract manifest sidecar",
    )
    try:
        registry = _contract_registry_from_payload(payload, path)
    except ValueError:
        if not allow_legacy:
            raise
        logger.warning(
            "[evaluation artifacts] using explicit legacy compatibility for contract "
            "manifest at %s",
            path,
        )
        return None
    if entry.get("digest") != payload["digest"]:
        raise ValueError("Metric contract manifest entry digest does not match its payload")
    if entry.get("registry_version") != payload["registry_version"]:
        raise ValueError(
            "Metric contract manifest entry registry version does not match its payload"
        )
    source_provenance = manifest.get("source_provenance")
    if isinstance(source_provenance, Mapping):
        recorded_digest = source_provenance.get("metric_contract_digest")
        if recorded_digest is not None and recorded_digest != payload["digest"]:
            raise ValueError("Metric contract manifest digest does not match source provenance")
    return registry


def _enrich_final_refit_evidence(
    evidence: Mapping[str, Any], *, artifact_root: Path | None = None
) -> dict[str, Any]:
    """Validate and digest refit files referenced by final-holdout evidence."""
    enriched = dict(evidence)
    legacy_marker = evidence.get("legacy_schema_version")
    if legacy_marker == _LEGACY_FINAL_EVIDENCE_SCHEMA:
        inventory_supplied = isinstance(evidence.get("provenance_inventory"), Mapping)
        for name, value in _final_evidence_state_defaults(enriched).items():
            enriched.setdefault(name, value)
        enriched.setdefault("provenance_inventory", _normalize_final_provenance_inventory(enriched))
        enriched.setdefault("provenance_inventory_supplied", inventory_supplied)
        enriched["provenance_inventory_legacy_migrated"] = True
    elif legacy_marker is not None:
        raise ValueError("Final-holdout evidence has an unsupported legacy_schema_version")
    state = enriched.get("state")
    if state != "succeeded" and state != "failed":
        return enriched
    refit = enriched.get("final_refit")
    if refit is None and state == "failed":
        raise ValueError(
            "Non-blocked final-holdout evidence requires final_refit generator metadata"
        )
    if not isinstance(refit, Mapping):
        raise ValueError("Final-holdout evidence final_refit metadata must be an object")
    data_path = Path(_non_empty_string(refit.get("path"), "Final refit data path"))
    metadata_path = Path(_non_empty_string(refit.get("metadata_path"), "Final refit metadata path"))
    if artifact_root is not None:
        data_path = _relative_artifact_path(artifact_root, data_path, "Final refit data")
        metadata_path = _relative_artifact_path(
            artifact_root, metadata_path, "Final refit metadata"
        )
    cache_key = _non_empty_string(refit.get("cache_key"), "Final refit cache key")
    fit_roles = _string_list(refit.get("fit_roles"), "Final refit fit_roles", unique=True)
    if fit_roles != ["train", "tuning"]:
        raise ValueError(
            f"Final refit must declare exactly fit_roles=['train', 'tuning']; got {fit_roles!r}"
        )
    if data_path.is_symlink() or metadata_path.is_symlink():
        raise ValueError("Final refit artifacts must not be symlinks")
    if not data_path.is_file() or not data_path.resolve().is_file():
        raise FileNotFoundError("Final refit synthetic data artifact is missing")
    if not metadata_path.is_file() or not metadata_path.resolve().is_file():
        raise FileNotFoundError("Final refit metadata artifact is missing")
    for field, path, artifact_name in (
        ("data_sha256", data_path, "synthetic data"),
        ("metadata_sha256", metadata_path, "metadata"),
    ):
        if field in refit:
            recorded_digest = _non_empty_string(refit[field], f"Final refit {field}")
            if recorded_digest != _file_digest(path):
                raise ValueError(f"Final refit {artifact_name} failed integrity verification")
    try:
        cache_metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Final refit metadata artifact is unreadable") from exc
    if not isinstance(cache_metadata, dict):
        raise ValueError("Final refit metadata artifact must be an object")
    cache_schema = cache_metadata.get("schema_version")
    generator_metadata = refit.get("generator_metadata", cache_metadata.get("generator_metadata"))
    if cache_schema in _LEGACY_CACHE_SCHEMA_VERSIONS:
        if cache_schema != "final-refit-v1":
            raise ValueError("Final refit metadata artifact has an invalid schema")
        if cache_metadata.get("cache_key") != cache_key:
            raise ValueError("Final refit cache-key mismatch")
        if generator_metadata is None:
            raise ValueError(
                "Non-blocked final-holdout evidence requires complete generator_metadata"
            )
        _validate_generator_metadata_payload(generator_metadata, "Final refit generator_metadata")
        enriched["final_refit"] = {
            **dict(refit),
            "provenance_state": "legacy",
            "data_sha256": _file_digest(data_path),
            "metadata_sha256": _file_digest(metadata_path),
            "generator_metadata": generator_metadata,
        }
        return enriched

    _validate_cache_envelope(
        cache_metadata,
        "Final refit cache_metadata",
        expected_model_name=refit.get("model_name"),
    )
    if cache_metadata.get("cache_key") != cache_key:
        raise ValueError("Final refit cache-key mismatch")
    if generator_metadata is None:
        raise ValueError("Non-blocked final-holdout evidence requires complete generator_metadata")
    _validate_generator_metadata_payload(generator_metadata, "Final refit generator_metadata")
    for field in (
        "model_name",
        "backend",
        "columns",
        "fit_roles",
        "fit_frame_fingerprint",
        "fit_frame_fingerprints",
        "input_role_hashes",
        "role_context_fingerprint",
        "role_context",
        "variable_schema_fingerprint",
        "semantic_context",
        "semantic_context_digest",
        "task_type",
        "target_view",
        "n_samples",
        "seed",
        "device",
        "parameters",
        "generator_metadata_schema_version",
        "generator_context",
        "implementation_fingerprint",
        "cache_key",
        "synthetic_data_sha256",
        "generator_metadata",
    ):
        if refit.get(field) != cache_metadata.get(field):
            raise ValueError(f"Final refit {field} does not match its metadata sidecar")
    try:
        frame = pd.read_csv(data_path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError("Final refit data artifact is unreadable") from exc
    if list(frame.columns) != cache_metadata["columns"]:
        raise ValueError("Final refit data columns do not match metadata")
    if len(frame) != cache_metadata["n_samples"]:
        raise ValueError("Final refit data row count does not match metadata")
    data_digest = _file_digest(data_path)
    if data_digest != cache_metadata["synthetic_data_sha256"]:
        raise ValueError("Final refit synthetic data failed its cache digest")
    enriched["final_refit"] = {
        **dict(refit),
        "provenance_state": "verified",
        "model_name": refit.get("model_name", cache_metadata.get("model_name")),
        "fit_frame_fingerprint": refit.get(
            "fit_frame_fingerprint", cache_metadata.get("fit_frame_fingerprint")
        ),
        "fit_frame_fingerprints": refit.get(
            "fit_frame_fingerprints", cache_metadata.get("fit_frame_fingerprints")
        ),
        "data_sha256": data_digest,
        "metadata_sha256": _file_digest(metadata_path),
        "cache_metadata": dict(cache_metadata),
        "generator_metadata": generator_metadata,
    }
    return enriched


def _bundle_refit_evidence(evidence: Mapping[str, Any], *, bundle_dir: Path) -> dict[str, Any]:
    """Copy transient refit files into bundle and return portable evidence."""
    if bundle_dir.is_symlink():
        raise ValueError("Evaluation artifact bundle must not be a symlink")
    bundle_root = bundle_dir.resolve()
    attempt_root = bundle_root.parent
    payload = dict(evidence)
    refit = payload.get("final_refit")
    if not isinstance(refit, Mapping):
        return payload
    copied = dict(refit)
    for source_key, artifact_name in (
        ("path", "final_refit/data.csv"),
        ("metadata_path", "final_refit/metadata.json"),
    ):
        source = _bundle_source_path(
            refit.get(source_key), bundle_dir=bundle_root, label=f"Final refit {source_key}"
        )
        destination = _safe_bundle_destination(
            bundle_root, artifact_name, f"Final refit {source_key}"
        )
        _copy_regular_source(source, destination, f"Final refit {source_key}")
        copied[source_key] = str(destination.relative_to(attempt_root))
        _relative_artifact_path(attempt_root, copied[source_key], f"Final refit {source_key}")
        copied["data_sha256" if source_key == "path" else "metadata_sha256"] = _file_digest(
            destination
        )
    payload["final_refit"] = copied
    return payload


def _bundle_generator_metadata(
    metadata: Mapping[str, Any],
    *,
    bundle_dir: Path,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Copy candidate cache files and replace transient paths with bundle IDs."""
    if bundle_dir.is_symlink():
        raise ValueError("Evaluation artifact bundle must not be a symlink")
    bundle_root = bundle_dir.resolve()
    attempt_root = bundle_root.parent
    if source_root is not None and any(
        isinstance(entry, Mapping) and entry.get("state") == "present"
        for entry in metadata.values()
    ):
        if (
            source_root.is_symlink()
            or not source_root.is_dir()
            or not source_root.resolve().is_dir()
        ):
            raise ValueError("Generator source root must be a regular directory")
        source_root = source_root.resolve()
    result: dict[str, Any] = {}
    for model_name, entry in metadata.items():
        if not isinstance(entry, Mapping) or entry.get("state") != "present":
            result[model_name] = dict(entry) if isinstance(entry, Mapping) else entry
            continue
        copied = dict(entry)
        safe_model_id = _model_artifact_id(str(model_name))
        for source_key, artifact_name in (
            ("data_path", f"generation/{safe_model_id}.csv"),
            ("metadata_path", f"generation/{safe_model_id}.cache.json"),
        ):
            source = _bundle_source_path(
                entry.get(source_key),
                bundle_dir=bundle_root,
                source_root=source_root,
                label=f"Generator {source_key}",
            )
            destination = _safe_bundle_destination(
                bundle_root, artifact_name, f"Generator {source_key}"
            )
            _copy_regular_source(source, destination, f"Generator {source_key}")
            copied[source_key] = str(destination.relative_to(attempt_root))
            _relative_artifact_path(attempt_root, copied[source_key], f"Generator {source_key}")
            copied["data_sha256" if source_key == "data_path" else "metadata_sha256"] = (
                _file_digest(destination)
            )
        result[model_name] = copied
    return result


def _git_provenance(
    path: Path,
    *,
    baseline_revision: str | None = None,
    baseline_source: str = "not_recorded",
) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
        diff = subprocess.run(
            ["git", "-C", str(path), "diff", "--binary", "HEAD", "--"],
            capture_output=True,
            check=False,
            timeout=5,
        )
        untracked = subprocess.run(
            ["git", "-C", str(path), "ls-files", "--others", "--exclude-standard", "-z"],
            capture_output=True,
            check=False,
            timeout=2,
        )
        status = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"],
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return {
            "revision": None,
            "dirty": None,
            "error": "git_provenance_unavailable",
        }
    revision_value = revision.stdout.strip() if revision.returncode == 0 else None
    if revision_value is not None and _GIT_REVISION_PATTERN.fullmatch(revision_value) is None:
        revision_value = None
    status_entries = status.stdout.splitlines() if status.returncode == 0 else []
    untracked_entries = (
        [entry.decode(errors="surrogateescape") for entry in untracked.stdout.split(b"\0") if entry]
        if untracked.returncode == 0
        else [entry[3:] for entry in status_entries if entry.startswith("?? ")]
    )
    command_errors = []
    for name, result in (
        ("revision", revision),
        ("diff", diff),
        ("untracked", untracked),
        ("status", status),
    ):
        if result.returncode != 0:
            command_errors.append(f"{name}_command_failed")

    tracked_diff_digest = hashlib.sha256(diff.stdout).hexdigest() if diff.returncode == 0 else None
    untracked_content_digest = None
    content_error = None
    if untracked.returncode == 0:
        try:
            content_hasher = hashlib.sha256()
            for relative_name in sorted(untracked_entries):
                file_path = path / relative_name
                content_hasher.update(relative_name.encode("utf-8", errors="surrogateescape"))
                content_hasher.update(b"\0")
                if file_path.is_symlink():
                    content_hasher.update(b"symlink\0")
                    content_hasher.update(
                        os.readlink(file_path).encode("utf-8", errors="surrogateescape")
                    )
                elif file_path.is_file():
                    with file_path.open("rb") as file_handle:
                        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                            content_hasher.update(chunk)
                else:
                    raise OSError(f"untracked path is not a regular file or symlink: {file_path}")
            untracked_content_digest = content_hasher.hexdigest()
        except OSError:
            content_error = "untracked_content_digest_failed"

    worktree_content_digest = None
    if revision_value and tracked_diff_digest and untracked_content_digest:
        worktree_hasher = hashlib.sha256()
        worktree_hasher.update(revision_value.encode())
        worktree_hasher.update(b"\0")
        worktree_hasher.update(tracked_diff_digest.encode())
        worktree_hasher.update(b"\0")
        worktree_hasher.update(untracked_content_digest.encode())
        worktree_content_digest = worktree_hasher.hexdigest()
    errors = list(command_errors)
    if content_error:
        errors.append(content_error)
    return {
        "revision": revision_value,
        "baseline_revision": baseline_revision,
        "baseline_source": baseline_source,
        "dirty": bool(status_entries) if status.returncode == 0 else None,
        "status_entries": ["worktree_dirty"] if status_entries else [],
        "untracked_entries": ["untracked_files_present"] if untracked_entries else [],
        "tracked_diff_digest": tracked_diff_digest,
        "untracked_content_digest": untracked_content_digest,
        "worktree_content_digest": worktree_content_digest,
        "error": errors or None,
    }


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _fork_attribution(path: Path, distribution: str) -> dict[str, Any]:
    """Record a lightweight attribution/license check for an editable fork."""
    license_files = [
        name for name in ("LICENSE", "LICENSE.txt", "COPYING") if (path / name).is_file()
    ]
    license_digests = {name: _file_digest(path / name) for name in license_files}
    try:
        metadata = cast(Mapping[str, str], importlib.metadata.metadata(distribution))
        package_name = metadata.get("Name")
        package_license = metadata.get("License")
    except importlib.metadata.PackageNotFoundError:
        package_name = None
        package_license = None
    return {
        "status": "passed" if license_files else "failed",
        "license_files": license_files,
        "license_digests": license_digests,
        "verification": "license_file_presence_and_digest",
        "package_name": package_name,
        "package_license": package_license,
    }


def _fork_provenance(
    path: Path,
    distribution: str,
    fork_repairs: tuple[Mapping[str, str], ...],
    *,
    baseline_revision: str | None = None,
    baseline_source: str = "not_recorded",
) -> dict[str, Any]:
    provenance = _git_provenance(
        path,
        baseline_revision=baseline_revision,
        baseline_source=baseline_source,
    )
    provenance.update(
        {
            "package_name": distribution,
            "package_version": _package_version(distribution),
            "attribution": _fork_attribution(path, distribution),
            "fork_repairs": [dict(repair) for repair in fork_repairs],
        }
    )
    return provenance


def collect_source_provenance(
    *,
    config_path: str | Path | None = None,
    metric_contract_digest: str | None = None,
) -> dict[str, Any]:
    """Collect root and editable-fork identities for an evaluation manifest."""
    repository_root = Path(__file__).resolve().parents[2]
    config_digest = None
    if config_path is not None:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configured evaluation config is missing at {config_path}")
        config_digest = _file_digest(config_path)
    return {
        "schema_version": _SOURCE_PROVENANCE_SCHEMA_VERSION,
        "root": _git_provenance(repository_root),
        "package": {
            "name": "synthdata",
            "version": _package_version("synthdata"),
        },
        "synthcity": _fork_provenance(
            repository_root / "submodules" / "synthcity",
            "synthcity",
            _SYNTHCITY_FORK_REPAIRS,
            baseline_revision=_SYNTHCITY_BASELINE_REVISION,
            baseline_source=_SYNTHCITY_BASELINE_SOURCE,
        ),
        "syntheval": _fork_provenance(
            repository_root / "submodules" / "syntheval",
            "syntheval",
            _SYNTHEVAL_FORK_REPAIRS,
        ),
        "config_digest": config_digest,
        "metric_contract_digest": metric_contract_digest,
    }


def _release_score_payload(
    evidence: Mapping[str, Any],
    model_index: Any,
    selected_model: str,
) -> dict[str, Any]:
    """Normalize release-score evidence into an audit-only sidecar payload."""
    models = evidence.get("models") if isinstance(evidence.get("models"), Mapping) else evidence
    if not isinstance(models, Mapping):
        raise ValueError("Release-score evidence must be an object keyed by model")
    normalized = {str(name): value for name, value in models.items()}
    expected = {selected_model}
    if set(normalized) != expected:
        raise ValueError(
            "Release-score evidence model inventory must contain selected model only: "
            f"recorded={sorted(normalized)!r}, expected={sorted(expected)!r}"
        )
    return {
        "schema_version": "release-score-evidence-v1",
        "audit_only": True,
        "inventory_scope": "selected_model",
        "selected_model": selected_model,
        "candidate_audit_models": [str(name) for name in model_index],
        "models": normalized,
    }


def _safe_log_disparity_reason(value: Any, *, fallback: str) -> str:
    """Return an allowlisted log-disparity reason code."""
    if isinstance(value, str) and value in _SAFE_LOG_DISPARITY_REASONS:
        return value
    return fallback


def _safe_log_disparity_error_type(value: Any) -> str:
    """Return a validated exception type without retaining exception details."""
    if isinstance(value, str) and _SAFE_EXCEPTION_TYPE_PATTERN.fullmatch(value):
        return value
    return "UnknownError"


def _validate_release_score_manifest_entry(entry: Mapping[str, Any]) -> None:
    """Validate manifest metadata for release-score evidence."""
    if not isinstance(entry, Mapping):
        raise ValueError("Release-score evidence manifest entry must be an object")
    if entry.get("audit_only") is not True:
        raise ValueError("Release-score evidence must be marked audit_only")
    if entry.get("inventory_scope") != "selected_model":
        raise ValueError("Release-score evidence inventory scope must be selected_model")
    _non_empty_string(entry.get("selected_model"), "Release-score evidence selected model")
    _non_empty_string(entry.get("path"), "Release-score evidence")
    digest = _non_empty_string(entry.get("sha256"), "Release-score evidence sha256")
    if not _SHA256_PATTERN.fullmatch(digest):
        raise ValueError("Release-score evidence sha256 must be a lowercase SHA-256 digest")
    _string_list(entry.get("models"), "Release-score evidence models", unique=True)
    _string_list(
        entry.get("candidate_audit_models"),
        "Release-score evidence candidate audit models",
        unique=True,
    )


def _validate_release_score_payload(payload: Mapping[str, Any], *, path: Path) -> None:
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "audit_only",
            "inventory_scope",
            "selected_model",
            "candidate_audit_models",
            "models",
        },
        "Release-score evidence",
    )
    if payload.get("schema_version") != "release-score-evidence-v1":
        raise ValueError(f"Unsupported release-score evidence schema at {path}")
    if payload.get("audit_only") is not True:
        raise ValueError(f"Release-score evidence must be audit-only at {path}")
    if payload.get("inventory_scope") != "selected_model":
        raise ValueError(f"Release-score evidence inventory scope is invalid at {path}")
    selected_model = _non_empty_string(
        payload.get("selected_model"), "Release-score evidence selected model"
    )
    candidate_audit_models = payload.get("candidate_audit_models")
    _string_list(
        candidate_audit_models, "Release-score evidence candidate audit models", unique=True
    )
    if selected_model not in (candidate_audit_models or []):
        raise ValueError(
            f"Release-score selected model {selected_model!r} is absent from candidate audit models"
        )
    models = payload.get("models")
    if not isinstance(models, Mapping):
        raise ValueError(f"Release-score evidence models must be an object at {path}")
    for model_name, score in models.items():
        if not isinstance(model_name, str) or not model_name:
            raise ValueError(f"Release-score evidence contains invalid model name at {path}")
        if not isinstance(score, Mapping):
            raise ValueError(f"Release-score evidence for {model_name!r} must be an object")
        if score.get("audit_only") is not True:
            raise ValueError(f"Release-score evidence for {model_name!r} is not audit-only")
        _validate_release_score_record(score, model_name=model_name, path=path)


def _validate_release_score_record(
    score: Mapping[str, Any], *, model_name: str, path: Path
) -> None:
    """Validate score state, finite values, and complete decomposition."""
    if score.get("status") == "succeeded" and (
        not isinstance(score.get("score"), (int, float))
        or isinstance(score.get("score"), bool)
        or not math.isfinite(float(cast(Real, score.get("score"))))
    ):
        raise ValueError(f"Succeeded release score for {model_name!r} is not finite")
    allowed_score_keys = {
        "status",
        "score",
        "audit_only",
        "dimensions",
        "provenance",
        "R_final",
        "formula",
        "weights",
        "anchors",
        "indeterminate_dimensions",
    }
    required_score_keys = {"status", "score", "audit_only", "dimensions", "provenance"}
    _require_exact_keys(
        score,
        required_score_keys | (set(score) & (allowed_score_keys - required_score_keys)),
        f"Release-score record for {model_name!r}",
    )
    provenance = score["provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError(f"Release-score provenance for {model_name!r} must be an object")
    status = score.get("status")
    if status not in {"succeeded", "indeterminate"}:
        raise ValueError(f"Release-score state for {model_name!r} is invalid at {path}")
    if status == "indeterminate" and "final_holdout_binding" not in provenance:
        return
    _require_exact_keys(
        provenance,
        {"final_holdout_binding", "final_holdout_binding_digest"},
        f"Release-score provenance for {model_name!r}",
    )
    binding = provenance["final_holdout_binding"]
    if not isinstance(binding, Mapping):
        raise ValueError(f"Release-score binding for {model_name!r} must be an object")
    _require_exact_keys(
        binding,
        {
            "schema_version",
            "selected_model",
            "role_context_fingerprint",
            "role_hashes",
            "raw_imputed_role_hashes",
            "release_transform_digest",
            "common_protocol_digest",
            "final_refit_identity",
            "selected_model_provenance",
        },
        f"Release-score binding for {model_name!r}",
    )
    # Persist preflight uses a sentinel binding before final evidence is copied.
    # Full recursive validation runs once canonical binding is attached.
    if provenance.get("final_holdout_binding_digest") == "placeholder":
        return
    _non_empty_string(binding["schema_version"], "Release-score binding schema_version")
    _non_empty_string(binding["selected_model"], "Release-score binding selected_model")
    _non_empty_string(binding["role_context_fingerprint"], "Release-score binding role context")
    _validate_role_hashes(
        binding["role_hashes"], f"Release-score binding for {model_name!r}.role_hashes"
    )
    raw_hashes = binding["raw_imputed_role_hashes"]
    if not isinstance(raw_hashes, Mapping):
        raise ValueError("Release-score raw/imputed role hashes must be an object")
    _require_exact_keys(
        raw_hashes, {"evidence", "custom_raw"}, "Release-score raw/imputed role hashes"
    )
    _validate_role_hash_map(
        raw_hashes["evidence"], "Release-score raw/imputed role hashes.evidence"
    )
    _validate_role_hash_map(
        raw_hashes["custom_raw"], "Release-score raw/imputed role hashes.custom_raw"
    )
    _sha256_digest(binding["release_transform_digest"], "Release-score release transform digest")
    _sha256_digest(binding["common_protocol_digest"], "Release-score common protocol digest")
    identity = binding["final_refit_identity"]
    _validate_final_refit_identity(
        identity, f"Release-score final-refit identity for {model_name!r}"
    )
    _validate_selected_model_provenance(
        binding["selected_model_provenance"],
        f"Release-score selected-model provenance for {model_name!r}",
    )
    value = score.get("score")
    if status == "succeeded" and (
        not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
    ):
        raise ValueError(f"Succeeded release score for {model_name!r} is not finite")
    if status not in {"succeeded", "indeterminate"}:
        raise ValueError(f"Release-score state for {model_name!r} is invalid at {path}")

    def finite(value: Any) -> bool:
        return (
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        )

    if status == "succeeded" and not finite(value):
        raise ValueError(f"Succeeded release score for {model_name!r} is not finite")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and not 0 <= value <= 1:
        raise ValueError(f"Release score for {model_name!r} is outside [0, 1]")
    if status == "indeterminate" and value is not None:
        raise ValueError(f"Indeterminate release score for {model_name!r} must be null")
    for field in ("formula",):
        if field in score:
            _non_empty_string(score[field], f"Release-score {field}")
    if "R_final" in score and not finite(score["R_final"]):
        raise ValueError(f"Release-score R_final for {model_name!r} is not finite")
    if "weights" in score:
        weights = score["weights"]
        if not isinstance(weights, Mapping):
            raise ValueError("Release-score weights must be an object")
        _require_exact_keys(weights, {"utility", "privacy", "fairness"}, "Release-score weights")
        for key, weight in weights.items():
            if not finite(weight) or not 0 <= weight <= 1:
                raise ValueError(f"Release-score weights.{key} is invalid")
    if "anchors" in score:
        anchors = score["anchors"]
        if not isinstance(anchors, Mapping):
            raise ValueError("Release-score anchors must be an object")
        _require_exact_keys(
            anchors,
            {
                "mmd",
                "epsilon_excess",
                "mia_advantage",
                "attribute_disclosure",
                "equalized_odds_gap",
                "worst_absolute_log_disparity",
            },
            "Release-score anchors",
        )
        for key, anchor in anchors.items():
            if not finite(anchor) or anchor <= 0:
                raise ValueError(f"Release-score anchors.{key} is invalid")
    if "indeterminate_dimensions" in score:
        _string_list(
            score["indeterminate_dimensions"], "Release-score indeterminate_dimensions", unique=True
        )
    dimensions = score.get("dimensions")
    required = {
        "utility": {"tstr", "mmd", "jsd"},
        "privacy": {"k", "l", "dcr", "epsilon", "mia", "attribute"},
        "fairness": {"representation", "eo", "worst_log_disparity"},
    }
    if not isinstance(dimensions, Mapping) or set(dimensions) != set(required):
        raise ValueError(f"Release-score dimensions are incomplete for {model_name!r}")
    all_succeeded = True
    for dimension, components in required.items():
        item = dimensions.get(dimension)
        if not isinstance(item, Mapping):
            raise ValueError(f"Release-score {dimension} is not an object")
        if set(item) - {"score", "components", "identity"} or not {"score", "components"}.issubset(
            item
        ):
            raise ValueError(f"Release-score {dimension} has invalid fields")
        dimension_score = item.get("score")
        if dimension_score is not None and not finite(dimension_score):
            raise ValueError(f"Release-score {dimension} score is not finite")
        if (
            isinstance(dimension_score, (int, float))
            and not isinstance(dimension_score, bool)
            and not 0 <= dimension_score <= 1
        ):
            raise ValueError(f"Release-score {dimension} score is outside [0, 1]")
        component_payload = item.get("components")
        if not isinstance(component_payload, Mapping) or set(component_payload) != components:
            raise ValueError(f"Release-score {dimension} components are incomplete")
        for component in components:
            record = component_payload[component]
            if not isinstance(record, Mapping) or record.get("status") not in {
                "succeeded",
                "indeterminate",
            }:
                raise ValueError(
                    f"Release-score component {dimension}.{component} has invalid state"
                )
            if set(record) - {"score", "status", "evidence"} or not {"score", "status"}.issubset(
                record
            ):
                raise ValueError(
                    f"Release-score component {dimension}.{component} has invalid fields"
                )
            component_score = record.get("score")
            if record["status"] == "succeeded" and not finite(component_score):
                raise ValueError(f"Release-score component {dimension}.{component} is not finite")
            if (
                isinstance(component_score, (int, float))
                and not isinstance(component_score, bool)
                and not 0 <= component_score <= 1
            ):
                raise ValueError(
                    f"Release-score component {dimension}.{component} is outside [0, 1]"
                )
            if record["status"] == "indeterminate" and component_score is not None:
                raise ValueError(f"Indeterminate component {dimension}.{component} must be null")
            all_succeeded &= record["status"] == "succeeded"
        all_succeeded &= dimension_score is not None
    if (status == "succeeded") != all_succeeded:
        raise ValueError(f"Release-score state is inconsistent for {model_name!r}")


def _release_score_binding(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Return canonical identity of exact final-holdout score evidence."""
    inventory = evidence.get("provenance_inventory")
    refit = evidence.get("final_refit")
    if not isinstance(inventory, Mapping):
        raise ValueError("Release-score provenance requires provenance_inventory")
    if not isinstance(refit, Mapping) and evidence.get("state") != "blocked":
        raise ValueError("Release-score provenance requires final_refit identity")
    refit_mapping: Mapping[str, Any] = refit if isinstance(refit, Mapping) else {}
    inventory_role_hashes = inventory.get("role_hashes")
    if not isinstance(inventory_role_hashes, Mapping):
        raise ValueError("Release-score provenance requires role hash inventory")
    imputed_role_hashes = inventory_role_hashes.get("imputed_evaluation")
    custom_raw_role_hashes = inventory_role_hashes.get("custom_raw_evaluation")
    binding = {
        "schema_version": "final-holdout-release-binding-v1",
        "selected_model": evidence.get("selected_model"),
        "role_context_fingerprint": evidence.get("role_context_fingerprint"),
        "role_hashes": evidence.get("role_hashes", inventory.get("role_hashes")),
        "raw_imputed_role_hashes": {
            "evidence": imputed_role_hashes,
            "custom_raw": custom_raw_role_hashes,
        },
        "release_transform_digest": inventory.get("release_transform_digest"),
        "common_protocol_digest": evidence.get("common_protocol_digest")
        or inventory.get("common_protocol_digest")
        or (
            inventory.get("supports", {}).get("release_transform", {}).get("common_protocol_digest")
            if isinstance(inventory.get("supports"), Mapping)
            and isinstance(inventory.get("supports", {}).get("release_transform"), Mapping)
            else None
        ),
        "final_refit_identity": {
            key: refit_mapping.get(key)
            for key in (
                "model_name",
                "cache_key",
                "data_sha256",
                "metadata_sha256",
                "fit_frame_fingerprint",
                "fit_frame_fingerprints",
            )
        },
        "selected_model_provenance": {
            "model": inventory.get("selected_model_provenance", {}).get("model")
            if isinstance(inventory.get("selected_model_provenance"), Mapping)
            else None,
            "selection": evidence.get("candidate_selection", {}),
            "fit_roles": refit_mapping.get("fit_roles"),
            "refit": {
                key: (
                    inventory.get("selected_model_provenance", {}).get("refit", {}).get(key)
                    if isinstance(inventory.get("selected_model_provenance"), Mapping)
                    and isinstance(
                        inventory.get("selected_model_provenance", {}).get("refit"), Mapping
                    )
                    else refit_mapping.get(key)
                )
                or refit_mapping.get(key)
                for key in _FINAL_REFIT_IDENTITY_KEYS
            },
        },
    }
    if evidence.get("state") != "blocked":
        binding["final_refit_identity"]["model_name"] = (
            binding["final_refit_identity"]["model_name"] or binding["selected_model"]
        )
        if not isinstance(binding["selected_model_provenance"].get("selection"), Mapping):
            binding["selected_model_provenance"]["selection"] = {
                "source": "final_holdout_evidence",
                "model": binding["selected_model"],
            }
        if not binding["selected_model_provenance"].get("model"):
            binding["selected_model_provenance"]["model"] = binding["selected_model"]
        identity = binding["final_refit_identity"]
        required = {
            "role_context_fingerprint": binding["role_context_fingerprint"],
            "release_transform_digest": binding["release_transform_digest"],
            "common_protocol_digest": binding["common_protocol_digest"],
            "selected_model": binding["selected_model"],
            **{
                field: identity.get(field)
                for field in ("model_name", "cache_key", "data_sha256", "metadata_sha256")
            },
        }
        missing = [field for field, value in required.items() if value in (None, "", {}, [])]
        if missing:
            raise ValueError(
                "Release-score provenance binding identity is incomplete: " + ", ".join(missing)
            )
    return binding


def _bind_release_score_provenance(score: dict[str, Any], evidence: Mapping[str, Any]) -> None:
    if (
        score.get("status") == "indeterminate"
        and evidence.get("state") in {"failed", "blocked"}
        and not isinstance(evidence.get("final_refit"), Mapping)
        and not evidence.get("role_context_fingerprint")
        and not evidence.get("common_protocol_digest")
    ):
        # Failed/blocked indeterminate producer states may intentionally carry
        # no release binding. They remain indeterminate and are not migrated
        # into a synthetic current binding.
        return
    binding = _release_score_binding(evidence)
    provenance = score.setdefault("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("Release-score provenance must be an object")
    provenance["final_holdout_binding"] = binding
    provenance["final_holdout_binding_digest"] = _mapping_digest(binding)


def _validate_release_score_binding(
    score: Mapping[str, Any], evidence: Mapping[str, Any], *, label: str
) -> None:
    provenance = score.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{label} is missing release-score provenance")
    binding = provenance.get("final_holdout_binding")
    digest = provenance.get("final_holdout_binding_digest")
    if (
        score.get("status") == "indeterminate"
        and evidence.get("state") in {"failed", "blocked"}
        and binding is None
        and digest is None
    ):
        return
    expected = _release_score_binding(evidence)
    if binding != expected or digest != _mapping_digest(expected):
        raise ValueError(f"{label} final-holdout provenance binding is invalid")


def _partition_framework_validation_results(
    validation_results: Mapping[Any, Any] | None,
    *,
    custom_validation_results: Mapping[Any, Any] | None,
) -> tuple[
    dict[tuple[str, str], Mapping[str, Any]] | None,
    dict[tuple[str, str], Mapping[str, Any]] | None,
]:
    """Partition mixed validation results before writing framework sidecars."""
    syntheval_results: dict[tuple[str, str], Mapping[str, Any]] = {}
    custom_results: dict[tuple[str, str], Mapping[str, Any]] = {}
    ordinary_custom_results = {
        str(model_name): result
        for model_name, result in (custom_validation_results or {}).items()
        if not isinstance(model_name, tuple)
    }
    if ordinary_custom_results:
        custom_results[("custom", "main")] = ordinary_custom_results
    for mapping in (validation_results, custom_validation_results):
        if mapping is None:
            continue
        for identity, results in mapping.items():
            if not isinstance(identity, tuple) or len(identity) != 2:
                continue
            framework, execution_pass = identity
            if not isinstance(framework, str) or not isinstance(execution_pass, str):
                raise ValueError("Validation framework/pass identity must contain strings")
            if framework == "syntheval":
                if not isinstance(results, Mapping):
                    raise ValueError(f"SynthEval pass {identity!r} must map models to results")
                syntheval_results[(framework, execution_pass)] = results
            elif framework == "custom":
                if not isinstance(results, Mapping):
                    raise ValueError(f"Custom pass {identity!r} must map models to results")
                custom_results[(framework, execution_pass)] = results
    return (
        syntheval_results or None,
        custom_results or None,
    )


def _partition_syntheval_passes(
    pass_results: Mapping[tuple[str, str], Mapping[str, Any]] | None,
) -> dict[tuple[str, str], Mapping[str, Any]] | None:
    """Keep only SynthEval passes in SynthEval execution artifacts."""
    if pass_results is None:
        return None
    return {
        identity: results
        for identity, results in pass_results.items()
        if identity[0] == "syntheval"
    }


def _validate_custom_raw_role_hashes(
    validation_results: Mapping[tuple[str, str], Mapping[str, Any]] | None,
    role_context: Mapping[str, Any] | None,
) -> None:
    """Reject custom contexts whose hashes are not the declared raw roles."""
    if validation_results is None or role_context is None:
        return
    candidate = role_context.get("candidate")
    if not isinstance(candidate, Mapping) or not isinstance(candidate.get("roles"), Mapping):
        raise ValueError("Custom validation requires a declared candidate raw role map")
    raw_hashes = {
        role: details.get("raw_fingerprint")
        for role, details in candidate["roles"].items()
        if isinstance(details, Mapping)
    }
    if not raw_hashes or any(
        not isinstance(value, str) or not value for value in raw_hashes.values()
    ):
        raise ValueError("Custom validation requires non-empty declared raw role hashes")
    for pass_identity, results in validation_results.items():
        for model_name, result in results.items():
            context = getattr(result, "evaluation_context", None)
            if context is None and isinstance(result, Mapping):
                context = result.get("evaluation_context")
            observed = getattr(context, "role_hashes", None)
            if observed is None and isinstance(context, Mapping):
                observed = context.get("role_hashes")
            if dict(observed or {}) != raw_hashes:
                raise ValueError(
                    f"Custom model {model_name!r} pass {pass_identity!r} role hashes "
                    "do not match the declared raw role map"
                )


def persist_evaluation_artifacts(
    evaluation_dir: str | Path,
    combined: pd.DataFrame,
    log_disparity_reports: dict[str, dict],
    *,
    native_syntheval_plot_dir: str | Path | None,
    synthcity_validation_results: Mapping[str, Any] | None = None,
    syntheval_validation_results: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    syntheval_execution_results: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    custom_validation_results: Mapping[Any, Any] | None = None,
    metric_contract_manifest: dict[str, Any] | None = None,
    source_provenance: Mapping[str, Any] | None = None,
    role_context: Mapping[str, Any] | None = None,
    role_context_fingerprint: Mapping[str, str] | None = None,
    semantic_context: Mapping[str, Any] | None = None,
    semantic_context_fingerprint: str | None = None,
    generator_metadata: Mapping[str, Any] | None = None,
    final_holdout_evidence: Mapping[str, Any] | None = None,
    release_score_evidence: Mapping[str, Any] | None = None,
    release_score: Mapping[str, Any] | None = None,
    attempt_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Persist plot-ready evaluation outputs and return the bundle manifest path.

    Failed log-disparity model reports are recorded with their diagnostic
    context rather than omitted.  Every file is written atomically so a plot
    command never mistakes a partial report for a successful evaluation.
    """
    evaluation_dir = Path(evaluation_dir)
    syntheval_validation_results, custom_validation_results = (
        _partition_framework_validation_results(
            syntheval_validation_results,
            custom_validation_results=custom_validation_results,
        )
    )
    _validate_custom_raw_role_hashes(custom_validation_results, role_context)
    syntheval_execution_results = _partition_syntheval_passes(syntheval_execution_results)
    semantic_contexts: dict[str, dict[str, Any]] = {}
    for pass_key, model_payloads in (syntheval_execution_results or {}).items():
        pass_identity = ":".join(pass_key) if isinstance(pass_key, tuple) else str(pass_key)
        pass_context: Mapping[str, Any] | None = None
        for model_name, payload in model_payloads.items():
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"SynthEval pass {pass_identity!r}, model {model_name!r} payload must be an object"
                )
            context = payload.get("semantic_context")
            digest = payload.get("semantic_context_digest")
            if not isinstance(context, Mapping) or not isinstance(digest, str):
                if pass_identity == "syntheval:main":
                    # Explicit legacy compatibility: historical main-only
                    # checkpoints predate persisted semantic payloads.
                    continue
                raise ValueError(
                    f"SynthEval pass {pass_identity!r}, model {model_name!r} requires semantic context"
                )
            if digest != semantic_context_digest(context):
                raise ValueError(
                    f"SynthEval pass {pass_identity!r}, model {model_name!r} has invalid semantic context digest"
                )
            if pass_context is None:
                pass_context = context
            elif dict(pass_context) != dict(context):
                raise ValueError(
                    f"SynthEval pass {pass_identity!r} has inconsistent semantic contexts"
                )
        if pass_context is not None:
            semantic_contexts[pass_identity] = {
                "semantic_context": dict(pass_context),
                "semantic_context_fingerprint": semantic_context_digest(pass_context),
            }
    bundle_dir = artifact_bundle_dir(evaluation_dir)
    if bundle_dir.is_symlink():
        raise ValueError(f"Evaluation artifact bundle must not be a symlink: {bundle_dir}")
    bundle_dir = _ensure_safe_directory(bundle_dir, "Evaluation artifact bundle")
    combined_path = evaluation_dir / "combined_evaluation.csv"
    if not combined_path.exists():
        raise FileNotFoundError(
            f"Cannot persist evaluation artifacts: combined table is missing at {combined_path}"
        )

    semantic_manifest = {
        "semantic_context": semantic_context,
        "semantic_context_fingerprint": semantic_context_fingerprint,
    }
    if semantic_context is not None and semantic_context_fingerprint is None:
        semantic_manifest["semantic_context_fingerprint"] = semantic_context_digest(
            semantic_context
        )
    _validate_semantic_context_manifest(semantic_manifest)
    semantic_sidecar_fields = (
        {
            "semantic_context": dict(semantic_context),
            "semantic_context_fingerprint": semantic_manifest["semantic_context_fingerprint"],
        }
        if semantic_context is not None
        else {}
    )

    # Preflight score and final-evidence linkage before writing either score
    # sidecar.  This is intentionally before all artifact production below so
    # malformed release evidence cannot leave a plausible success marker.
    score_input = release_score_evidence
    if score_input is not None and release_score is not None:
        raise ValueError("Provide only one of release_score_evidence and release_score")
    if score_input is None:
        score_input = release_score
    score_payload = None
    preflight_evidence_payload = None
    if score_input is not None:
        if final_holdout_evidence is None:
            raise ValueError(
                "Release-score evidence requires final-holdout selected-model metadata"
            )
        selected_model = _non_empty_string(
            final_holdout_evidence.get("selected_model"), "Final-holdout selected model"
        )
        score_payload = _release_score_payload(score_input, combined.index, selected_model)
        score_record = score_payload["models"][selected_model]
        if not isinstance(score_record, dict):
            raise ValueError("Release-score record must be mutable for provenance binding")
        if score_record.get("status") not in {"succeeded", "indeterminate"}:
            raise ValueError("Release-score record has invalid state")
        if score_record.get("status") == "indeterminate":
            _validate_release_score_record(score_record, model_name=selected_model, path=bundle_dir)
        preflight_evidence_payload = _enrich_final_refit_evidence(
            _bundle_refit_evidence(final_holdout_evidence, bundle_dir=bundle_dir),
            artifact_root=evaluation_dir,
        )
        _bind_release_score_provenance(score_record, preflight_evidence_payload)
        if score_record.get("status") == "succeeded":
            _validate_release_score_record(score_record, model_name=selected_model, path=bundle_dir)
        _validate_release_score_payload(score_payload, path=bundle_dir)

    if final_holdout_evidence is not None:
        preflight_evidence_payload = _enrich_final_refit_evidence(
            _bundle_refit_evidence(final_holdout_evidence, bundle_dir=bundle_dir),
            artifact_root=evaluation_dir,
        )
        if score_payload is not None:
            selected = score_payload["selected_model"]
            preflight_evidence_payload["release_score"] = score_payload["models"][selected]
        preflight_evidence_payload["schema_version"] = "final-holdout-evidence-v1"
        _validate_final_holdout_evidence_payload(
            preflight_evidence_payload,
            entry={
                "state": preflight_evidence_payload.get("state"),
                "selected_model": preflight_evidence_payload.get("selected_model"),
            },
            manifest={
                "combined_evaluation": {"models": list(combined.index)},
                "semantic_context_fingerprint": semantic_manifest.get(
                    "semantic_context_fingerprint"
                ),
            },
            path=bundle_dir / "final_holdout_evidence.json",
        )

    log_manifest: dict[str, dict[str, Any]] = {}
    log_root = _ensure_safe_directory(bundle_dir / "log_disparity", "Log-disparity artifact")
    for model_name, report in sorted(log_disparity_reports.items()):
        report = cast(dict[str, Any], report)
        model_dir = _ensure_safe_directory(
            log_root / _model_artifact_id(model_name), "Log-disparity artifact"
        )
        metadata_path = model_dir / "metadata.json"
        declared_state = report.get("state") if isinstance(report, Mapping) else None
        if declared_state not in {"succeeded", "failed", "indeterminate"}:
            report = {
                "state": "indeterminate",
                "reason": "report_state_missing_or_unknown",
                "missing_tables": list(_LOG_REPORT_TABLES),
            }
            declared_state = "indeterminate"
        if "error" in report or declared_state == "failed":
            reason = _safe_log_disparity_reason(
                report.get("reason"), fallback="log_disparity_evaluation_failed"
            )
            metadata = {
                "state": "failed",
                "model_name": model_name,
                "error_type": _safe_log_disparity_error_type(report.get("error_type")),
                "reason": reason,
            }
            _atomic_json(metadata_path, metadata)
            log_manifest[model_name] = {
                "path": str(model_dir.relative_to(bundle_dir)),
                "state": "failed",
                "metadata_sha256": _file_digest(metadata_path),
            }
            continue

        if declared_state == "indeterminate":
            result_metadata = report.get("result_metadata")
            if not isinstance(result_metadata, Mapping):
                result_metadata = {}
            reason = report.get("reason") or result_metadata.get("release_evidence_reason")
            if not isinstance(reason, str) or not reason:
                raise ValueError(
                    f"Cannot persist indeterminate log-disparity report for model {model_name!r}: "
                    "missing reason."
                )
            missing_tables = report.get("missing_tables", list(_LOG_REPORT_TABLES))
            if missing_tables != list(_LOG_REPORT_TABLES):
                raise ValueError(
                    f"Cannot persist indeterminate log-disparity report for model {model_name!r}: "
                    "missing_tables must list all required tables."
                )
            safe_reason = _safe_log_disparity_reason(
                reason, fallback="log_disparity_evaluation_indeterminate"
            )
            metadata = {
                "state": "indeterminate",
                "model_name": model_name,
                "reason": safe_reason,
                "missing_tables": list(missing_tables),
                "result_metadata": safe_metric_metadata(
                    result_metadata, label="log-disparity result_metadata"
                ),
            }
            _atomic_json(metadata_path, metadata)
            log_manifest[model_name] = {
                "path": str(model_dir.relative_to(bundle_dir)),
                "state": "indeterminate",
                "reason": safe_reason,
                "missing_tables": list(missing_tables),
                "metadata_sha256": _file_digest(metadata_path),
            }
            continue

        missing = [name for name in _LOG_REPORT_TABLES if name not in report]
        if missing:
            raise ValueError(
                f"Cannot persist log-disparity report for model {model_name!r}: "
                f"missing required table(s) {missing}."
            )
        metadata = {
            "state": "succeeded",
            "model_name": model_name,
            "summary_stats": report["summary_stats"],
            "protected_group_cols": report["protected_group_cols"],
            "protected_order_map": report["protected_order_map"],
            "target_order": report["target_order"],
        }
        if "result_metadata" in report:
            if not isinstance(report["result_metadata"], Mapping):
                raise ValueError("Log-disparity result_metadata must be an object")
            metadata["result_metadata"] = safe_metric_metadata(
                report["result_metadata"], label="log-disparity result_metadata"
            )
        _atomic_json(metadata_path, metadata)
        table_manifest = {}
        for name in _LOG_REPORT_TABLES:
            path = model_dir / f"{name}.parquet"
            table = report[name]
            if not isinstance(table, pd.DataFrame):
                raise ValueError(
                    f"Cannot persist log-disparity table {name!r} for model {model_name!r}: "
                    "expected a DataFrame."
                )
            _atomic_parquet(path, table)
            table_manifest[name] = {
                "filename": path.name,
                "sha256": _file_digest(path),
            }
        log_manifest[model_name] = {
            "path": str(model_dir.relative_to(bundle_dir)),
            "state": "succeeded",
            "metadata_sha256": _file_digest(metadata_path),
            "tables": table_manifest,
        }

    native_files = []
    native_root_manifest = None
    bundle_native_root = bundle_dir / "native_syntheval_plots"
    if bundle_native_root.is_symlink():
        raise ValueError("Native SynthEval plot root must not be a symlink")
    _clear_native_plot_destination(bundle_native_root)
    if native_syntheval_plot_dir is not None:
        native_root = Path(native_syntheval_plot_dir)
        source_files = _native_plot_source_files(native_root)
        if source_files:
            _ensure_safe_directory(bundle_native_root, "Native SynthEval plot destination")
            for path, relative_path in source_files:
                destination = _safe_bundle_destination(
                    bundle_dir,
                    str(Path("native_syntheval_plots") / relative_path),
                    "Native SynthEval plot",
                )
                _copy_regular_source(
                    path, destination, "Native SynthEval plot", contained_in=native_root
                )
            native_files = [
                {
                    "path": str(relative_path),
                    "sha256": _file_digest(bundle_native_root / relative_path),
                }
                for _path, relative_path in source_files
            ]
            if native_files:
                native_root_manifest = str(bundle_native_root.relative_to(evaluation_dir))

    contract_artifacts = {}
    if metric_contract_manifest is not None:
        contract_path = bundle_dir / "metric_contract_manifest.json"
        _atomic_json(contract_path, metric_contract_manifest)
        contract_artifacts = {
            "path": str(contract_path.relative_to(evaluation_dir)),
            "sha256": _file_digest(contract_path),
            "registry_version": metric_contract_manifest.get("registry_version"),
            "digest": metric_contract_manifest.get("digest"),
        }

    status_artifacts: dict[str, Any] = {}
    if synthcity_validation_results is not None:
        status_payload = {
            **semantic_sidecar_fields,
            "schema_version": _ARTIFACT_SCHEMA_VERSION,
            "framework": "synthcity",
            "source_provenance": dict(source_provenance or {}),
            "models": {
                model_name: result.to_dict() if hasattr(result, "to_dict") else result
                for model_name, result in sorted(synthcity_validation_results.items())
            },
        }
        status_path = bundle_dir / "synthcity_metric_status.json"
        _atomic_json(status_path, status_payload)
        status_artifacts = {
            "path": str(status_path.relative_to(evaluation_dir)),
            "sha256": _file_digest(status_path),
            "models": sorted(status_payload["models"]),
        }

    if syntheval_validation_results is not None:
        status_payload = {
            **semantic_sidecar_fields,
            "schema_version": _ARTIFACT_SCHEMA_VERSION,
            "framework": "syntheval",
            "source_provenance": dict(source_provenance or {}),
            "passes": {
                f"{framework}:{execution_pass}": {
                    model_name: result.to_dict() if hasattr(result, "to_dict") else result
                    for model_name, result in sorted(results.items())
                }
                for (framework, execution_pass), results in sorted(
                    syntheval_validation_results.items()
                )
            },
        }
        status_path = bundle_dir / "syntheval_metric_status.json"
        _atomic_json(status_path, status_payload)
        status_artifacts["syntheval"] = {
            "path": str(status_path.relative_to(evaluation_dir)),
            "sha256": _file_digest(status_path),
            "passes": sorted(status_payload["passes"]),
        }

    if syntheval_execution_results is not None:
        execution_payload = {
            **semantic_sidecar_fields,
            "schema_version": _ARTIFACT_SCHEMA_VERSION,
            "framework": "syntheval",
            "source_provenance": dict(source_provenance or {}),
            "passes": {
                f"{framework}:{execution_pass}": {
                    model_name: result.to_dict() if hasattr(result, "to_dict") else result
                    for model_name, result in sorted(results.items())
                }
                for (framework, execution_pass), results in sorted(
                    syntheval_execution_results.items()
                )
            },
        }
        execution_path = bundle_dir / "syntheval_execution.json"
        _atomic_json(execution_path, execution_payload)
        status_artifacts["syntheval_execution"] = {
            "path": str(execution_path.relative_to(evaluation_dir)),
            "sha256": _file_digest(execution_path),
            "passes": sorted(execution_payload["passes"]),
        }

    if custom_validation_results is not None:
        status_payload = {
            **semantic_sidecar_fields,
            "schema_version": _ARTIFACT_SCHEMA_VERSION,
            "framework": "custom",
            "source_provenance": dict(source_provenance or {}),
            "passes": {
                f"{framework}:{execution_pass}": {
                    model_name: result.to_dict() if hasattr(result, "to_dict") else result
                    for model_name, result in sorted(results.items())
                }
                for (framework, execution_pass), results in sorted(
                    custom_validation_results.items()
                )
            },
        }
        status_path = bundle_dir / "custom_metric_status.json"
        _atomic_json(status_path, status_payload)
        status_artifacts["custom"] = {
            "path": str(status_path.relative_to(evaluation_dir)),
            "sha256": _file_digest(status_path),
            "passes": sorted(status_payload["passes"]),
        }

    final_holdout_artifact = None
    if final_holdout_evidence is not None:
        if final_holdout_evidence.get("evaluation_role") != "final_holdout":
            raise ValueError("Final-holdout evidence must declare evaluation_role='final_holdout'")
        evidence_path = bundle_dir / "final_holdout_evidence.json"
        evidence_payload = _bundle_refit_evidence(final_holdout_evidence, bundle_dir=bundle_dir)
        evidence_payload = _enrich_final_refit_evidence(
            evidence_payload, artifact_root=evaluation_dir
        )
        if score_payload is not None:
            evidence_payload["release_score"] = score_payload["models"][
                score_payload["selected_model"]
            ]
            _validate_release_score_binding(
                evidence_payload["release_score"],
                evidence_payload,
                label="Final-holdout release score",
            )
        evidence_payload["schema_version"] = "final-holdout-evidence-v1"
        _atomic_json(evidence_path, evidence_payload)
        final_holdout_artifact = {
            "path": str(evidence_path.relative_to(evaluation_dir)),
            "sha256": _file_digest(evidence_path),
            "state": evidence_payload.get("state"),
            "selected_model": evidence_payload.get("selected_model"),
        }

    # Release scores are durable audit evidence only.  Keep this separate from
    # combined evaluation so loading it can never affect candidate selection.
    release_score_artifact = None
    if score_payload is not None:
        score_path = bundle_dir / "release_score_evidence.json"
        _atomic_json(score_path, score_payload)
        release_score_artifact = {
            "path": str(score_path.relative_to(evaluation_dir)),
            "sha256": _file_digest(score_path),
            "models": sorted(score_payload["models"]),
            "candidate_audit_models": score_payload["candidate_audit_models"],
            "selected_model": score_payload["selected_model"],
            "inventory_scope": score_payload["inventory_scope"],
            "audit_only": True,
        }

    manifest = {
        "schema_version": _ARTIFACT_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "combined_evaluation": {
            "path": str(combined_path.relative_to(evaluation_dir)),
            "sha256": _file_digest(combined_path),
            "models": list(combined.index),
        },
        "log_disparity": log_manifest,
        "native_syntheval_plots": {
            "root": native_root_manifest,
            "files": native_files,
        },
        "source_provenance": dict(source_provenance or {}),
        "canonical_metric_manifest": list(CANONICAL_EXPECTED_MANIFEST),
    }
    if attempt_metadata is not None:
        manifest["evaluation_attempt"] = dict(attempt_metadata)
    if role_context is not None:
        manifest["role_context"] = dict(role_context)
    if role_context_fingerprint is not None:
        manifest["role_context_fingerprint"] = dict(role_context_fingerprint)
    if semantic_context is not None:
        manifest["semantic_context"] = dict(semantic_context)
        manifest["semantic_context_fingerprint"] = semantic_manifest["semantic_context_fingerprint"]
    if semantic_contexts:
        manifest["semantic_contexts"] = semantic_contexts
    if generator_metadata is not None:
        source_generation_root = None
        if (
            attempt_metadata is not None
            and attempt_metadata.get("source_generation_root") is not None
        ):
            source_generation_root = Path(
                _non_empty_string(
                    attempt_metadata["source_generation_root"],
                    "Evaluation attempt source_generation_root",
                )
            )
        manifest["generator_metadata"] = {
            model_name: dict(metadata) if isinstance(metadata, Mapping) else metadata
            for model_name, metadata in sorted(
                _bundle_generator_metadata(
                    generator_metadata,
                    bundle_dir=bundle_dir,
                    source_root=source_generation_root,
                ).items()
            )
        }
        _validate_generator_metadata_manifest(
            manifest["generator_metadata"],
            manifest=manifest,
            artifact_root=evaluation_dir,
            expected_model_names=(
                [str(model_name) for model_name in combined.index]
                if len(combined.index) > 0
                else None
            ),
        )
    if contract_artifacts:
        manifest["metric_contract_manifest"] = contract_artifacts
    if status_artifacts:
        if "path" in status_artifacts:
            manifest["synthcity_metric_status"] = {
                key: value
                for key, value in status_artifacts.items()
                if key in {"path", "sha256", "models"}
            }
        for framework in ("syntheval", "custom"):
            if framework in status_artifacts:
                manifest[f"{framework}_metric_status"] = status_artifacts[framework]
        if "syntheval_execution" in status_artifacts:
            manifest["syntheval_execution"] = status_artifacts["syntheval_execution"]
    if final_holdout_artifact is not None:
        manifest["final_holdout_evidence"] = final_holdout_artifact
    if release_score_artifact is not None:
        manifest["release_score_evidence"] = release_score_artifact
    manifest_path = bundle_dir / "manifest.json"
    _atomic_json(manifest_path, manifest)
    logger.info(
        "[evaluation artifacts] persisted %d log-disparity report(s) and %d native SynthEval file(s) under %s",
        len(log_manifest),
        len(native_files),
        bundle_dir,
    )
    return manifest_path


def _load_metric_status_sidecar(
    evaluation_dir: str | Path,
    *,
    manifest_key: str,
    framework: str,
    container_key: str,
) -> dict:
    bundle_dir, manifest = _load_manifest(evaluation_dir)
    entry = manifest.get(manifest_key)
    if not entry:
        raise FileNotFoundError(
            f"{framework} metric status sidecar is not recorded in the evaluation manifest"
        )
    payload, _path = _read_json_artifact(
        bundle_dir.parent,
        entry,
        f"{framework} metric status sidecar",
    )
    registry = _load_contract_registry_if_present(bundle_dir, manifest)
    _validate_metric_status_payload(
        payload,
        framework=framework,
        container_key=container_key,
        registry=registry,
        manifest_source_provenance=manifest.get("source_provenance"),
        expected_semantic_context_fingerprint=manifest.get("semantic_context_fingerprint"),
    )
    return payload


def load_synthcity_metric_status(evaluation_dir: str | Path) -> dict:
    """Load and integrity-check the additive SynthCity metric status sidecar."""
    return _load_metric_status_sidecar(
        evaluation_dir,
        manifest_key="synthcity_metric_status",
        framework="synthcity",
        container_key="models",
    )


def load_syntheval_metric_status(evaluation_dir: str | Path) -> dict:
    """Load and integrity-check the additive SynthEval metric status sidecar."""
    return _load_metric_status_sidecar(
        evaluation_dir,
        manifest_key="syntheval_metric_status",
        framework="syntheval",
        container_key="passes",
    )


def load_syntheval_execution(evaluation_dir: str | Path) -> dict:
    """Load and integrity-check structured SynthEval execution evidence."""
    bundle_dir, manifest = _load_manifest(evaluation_dir)
    entry = manifest.get("syntheval_execution")
    if not entry:
        raise FileNotFoundError(
            "SynthEval execution sidecar is not recorded in the evaluation manifest"
        )
    payload, _path = _read_json_artifact(
        bundle_dir.parent,
        entry,
        "SynthEval execution sidecar",
    )
    if payload.get("schema_version") != _ARTIFACT_SCHEMA_VERSION:
        raise ValueError("SynthEval execution sidecar has an unsupported artifact schema")
    if payload.get("framework") != "syntheval":
        raise ValueError("Unexpected framework in SynthEval execution sidecar")
    source_provenance = payload.get("source_provenance")
    if not isinstance(source_provenance, dict):
        raise ValueError("SynthEval execution sidecar source_provenance must be an object")
    if source_provenance != manifest.get("source_provenance", {}):
        raise ValueError(
            "SynthEval execution sidecar source provenance does not match the artifact manifest"
        )
    _load_contract_registry_if_present(bundle_dir, manifest)
    passes = payload.get("passes")
    if not isinstance(passes, dict):
        raise ValueError("SynthEval execution sidecar is missing its 'passes' mapping")
    for pass_identity, model_payloads in passes.items():
        pass_identity = _non_empty_string(pass_identity, "SynthEval execution pass")
        pass_framework, separator, execution_pass = pass_identity.partition(":")
        if not separator or pass_framework != "syntheval":
            raise ValueError(f"Invalid SynthEval execution pass identity {pass_identity!r}")
        _non_empty_string(execution_pass, "SynthEval execution pass name")
        if not isinstance(model_payloads, dict):
            raise ValueError(
                f"SynthEval execution pass {pass_identity!r} must map models to payloads"
            )
        context_entry = manifest.get("semantic_contexts", {}).get(pass_identity)
        if not isinstance(context_entry, Mapping):
            if execution_pass != "main" or "semantic_contexts" in manifest:
                raise ValueError(
                    f"SynthEval pass {pass_identity!r} has no manifest semantic context"
                )
            context_entry = {
                "semantic_context": manifest.get("semantic_context"),
                "semantic_context_fingerprint": manifest.get("semantic_context_fingerprint"),
            }
        pass_context = context_entry.get("semantic_context")
        pass_digest = context_entry.get("semantic_context_fingerprint")
        if not isinstance(pass_context, Mapping) or not isinstance(pass_digest, str):
            if "semantic_contexts" not in manifest and execution_pass == "main":
                pass_context = None
                pass_digest = None
            else:
                raise ValueError(
                    f"SynthEval pass {pass_identity!r} has an invalid manifest semantic context"
                )
        if pass_context is not None and pass_digest != semantic_context_digest(pass_context):
            raise ValueError(
                f"SynthEval pass {pass_identity!r} manifest semantic context digest is invalid"
            )
        for model_name, execution_payload in model_payloads.items():
            model_name = _non_empty_string(model_name, "SynthEval execution model name")
            _validate_syntheval_execution_payload(
                execution_payload,
                label=f"SynthEval pass {pass_identity!r}, model {model_name!r}",
                model_name=model_name,
                execution_pass=execution_pass,
                expected_semantic_context_fingerprint=pass_digest,
                expected_semantic_context=pass_context,
            )
    return payload


def load_custom_metric_status(evaluation_dir: str | Path) -> dict:
    """Load and integrity-check the additive custom metric status sidecar."""
    return _load_metric_status_sidecar(
        evaluation_dir,
        manifest_key="custom_metric_status",
        framework="custom",
        container_key="passes",
    )


def validate_evaluation_bundle(
    evaluation_dir: str | Path,
    *,
    expected_config_path: str | Path | None = None,
    expected_role_context_fingerprints: Mapping[str, str] | None = None,
    expected_role_hashes: Mapping[str, str] | None = None,
    expected_role_hashes_by_framework: Mapping[str, Mapping[str, str]] | None = None,
    expected_semantic_context_fingerprint: str | None = None,
    expected_population_unit: str | None = None,
    expected_group_mode: str | None = None,
    expected_evaluation_role: str = "tuning",
    allow_legacy: bool = False,
) -> dict:
    """Validate cross-artifact semantic identity before a consumer renders it.

    Low-level loaders verify each file independently. This stricter entry point
    additionally proves that the recorded contract registry, configuration,
    role context, model inventory, and validation contexts describe the same
    evaluation. ``allow_legacy`` explicitly permits an embedded, internally
    consistent prior contract registry or bundles predating validation sidecars.
    """
    bundle_dir, manifest = _load_manifest(evaluation_dir)
    combined_models = _string_list(
        manifest["combined_evaluation"]["models"],
        "Combined evaluation models",
        unique=True,
    )
    combined_path = _verified_artifact_path(
        bundle_dir.parent,
        manifest["combined_evaluation"],
        "Combined evaluation",
    )
    from synthdata.evaluation.combine import load_combined_table

    combined = load_combined_table(combined_path, validate_artifact=False)

    recorded_manifest = manifest.get("canonical_metric_manifest")
    if tuple(recorded_manifest or ()) != CANONICAL_EXPECTED_MANIFEST:
        raise ValueError(
            "Evaluation bundle canonical metric manifest must exactly match canonical identities"
        )

    if expected_semantic_context_fingerprint is not None:
        recorded_semantic_fingerprint = manifest.get("semantic_context_fingerprint")
        if recorded_semantic_fingerprint != expected_semantic_context_fingerprint:
            raise ValueError(
                "Evaluation bundle semantic-context fingerprint does not match the current dataset"
            )

    # Validate recorded semantic evidence before any legacy compatibility
    # return.  Manifest hashes alone do not validate payload meaning.
    try:
        if "final_holdout_evidence" in manifest:
            load_final_holdout_evidence(evaluation_dir)
        if "release_score_evidence" in manifest:
            load_release_score_evidence(evaluation_dir)
    except (KeyError, OSError, TypeError, ValueError):
        raise ValueError("Evaluation bundle semantic sidecar validation failed") from None

    registry = _load_contract_registry_if_present(
        bundle_dir,
        manifest,
        allow_legacy=allow_legacy,
    )
    if registry is None:
        semantic_sidecar_keys = {
            "final_holdout_evidence",
            "release_score_evidence",
            "synthcity_metric_status",
            "syntheval_metric_status",
            "syntheval_execution",
            "custom_metric_status",
        }
        if semantic_sidecar_keys & set(manifest) and not allow_legacy:
            raise ValueError(
                "Evaluation bundle without a metric contract manifest cannot bypass "
                "semantic validation sidecars"
            )
        if not allow_legacy:
            raise FileNotFoundError(
                "Metric contract manifest is required for strict evaluation bundle validation"
            )
        logger.warning(
            "[evaluation artifacts] using explicit legacy compatibility for bundle without "
            "a metric contract manifest: %s",
            bundle_dir,
        )
        return manifest
    if registry.digest() != DEFAULT_METRIC_CONTRACT_REGISTRY.digest() and not allow_legacy:
        raise ValueError(
            "Evaluation bundle metric contract registry does not match the current registry"
        )

    source_provenance = manifest.get("source_provenance", {})
    if expected_config_path is not None:
        expected_config_digest = _file_digest(Path(expected_config_path))
        if (
            not isinstance(source_provenance, Mapping)
            or source_provenance.get("config_digest") != expected_config_digest
        ):
            raise ValueError(
                "Evaluation bundle configuration digest does not match the requested config"
            )

    if expected_role_context_fingerprints is not None:
        recorded_fingerprints = manifest.get("role_context_fingerprint")
        if not isinstance(recorded_fingerprints, Mapping) or dict(recorded_fingerprints) != dict(
            expected_role_context_fingerprints
        ):
            raise ValueError(
                "Evaluation bundle role-context fingerprints do not match the current dataset"
            )

    status_keys_by_model: dict[str, dict[str, set[str]]] = {}

    def validate_status_contexts(
        payload: Mapping[str, Any],
        *,
        framework: str,
        container_key: str,
    ) -> set[str]:
        actual_container_key = container_key
        if framework == "custom" and container_key == "passes" and "passes" not in payload:
            actual_container_key = "models"
        container = payload[actual_container_key]
        pass_identities = set()
        if actual_container_key == "models":
            observed_models = set(container)
            if observed_models != set(combined_models):
                raise ValueError(
                    f"{framework} metric status model inventory does not match combined evaluation: "
                    f"recorded={sorted(observed_models)!r}, combined={sorted(combined_models)!r}"
                )
            entries = (
                (model_name, "main", result_payload, framework)
                for model_name, result_payload in container.items()
            )
        else:
            entries = []
            for pass_identity, model_results in container.items():
                execution_pass = pass_identity.split(":", 1)[1]
                pass_identities.add(execution_pass)
                observed_models = set(model_results)
                if observed_models != set(combined_models):
                    raise ValueError(
                        f"{framework} pass {pass_identity!r} model inventory does not match "
                        f"combined evaluation: recorded={sorted(observed_models)!r}, "
                        f"combined={sorted(combined_models)!r}"
                    )
                entries.extend(
                    (model_name, execution_pass, result_payload, pass_identity.split(":", 1)[0])
                    for model_name, result_payload in model_results.items()
                )

        for model_name, execution_pass, result_payload, pass_framework in entries:
            pass_identities.add(execution_pass)
            status_keys_by_model.setdefault(model_name, {}).setdefault(pass_framework, set())
            status_keys_by_model[model_name][pass_framework].update(
                str(record["expected_key"])
                for record in result_payload.get("records", [])
                if isinstance(record, Mapping) and record.get("expected_key") is not None
            )
            context = result_payload.get("evaluation_context")
            if not isinstance(context, Mapping):
                raise ValueError(
                    f"{framework} model {model_name!r} has no persisted evaluation context"
                )
            if result_payload.get("contract_digest") != registry.digest():
                raise ValueError(
                    f"{framework} model {model_name!r} contract digest does not match the current registry"
                )
            expected_target_view = (
                "binary_collapsed" if execution_pass == "binary_target" else "native"
            )
            if context.get("execution_pass") != execution_pass:
                raise ValueError(
                    f"{framework} model {model_name!r} validation pass does not match its sidecar"
                )
            if context.get("target_view") != expected_target_view:
                raise ValueError(
                    f"{framework} model {model_name!r} validation target view does not match its pass"
                )
            if context.get("evaluation_role") != expected_evaluation_role:
                raise ValueError(
                    f"{framework} model {model_name!r} validation role does not match the bundle"
                )
            framework_role_hashes = expected_role_hashes
            if expected_role_hashes_by_framework is not None:
                framework_role_hashes = expected_role_hashes_by_framework.get(
                    framework, expected_role_hashes
                )
            if framework_role_hashes is not None and dict(context.get("role_hashes", {})) != dict(
                framework_role_hashes
            ):
                raise ValueError(
                    f"{framework} model {model_name!r} validation role hashes do not match the current dataset"
                )
            if (
                expected_population_unit is not None
                and context.get("population_unit") != expected_population_unit
            ):
                raise ValueError(
                    f"{framework} model {model_name!r} validation population unit does not match the bundle"
                )
            if expected_group_mode is not None and context.get("group_mode") != expected_group_mode:
                raise ValueError(
                    f"{framework} model {model_name!r} validation group mode does not match the bundle"
                )
        return pass_identities

    status_passes = set()
    status_entries = (
        ("synthcity", "synthcity_metric_status", "models"),
        ("syntheval", "syntheval_metric_status", "passes"),
        ("custom", "custom_metric_status", "passes"),
    )
    present_statuses = []
    for framework, manifest_key, container_key in status_entries:
        if manifest_key not in manifest:
            continue
        if framework == "synthcity":
            payload = load_synthcity_metric_status(evaluation_dir)
        elif framework == "syntheval":
            payload = load_syntheval_metric_status(evaluation_dir)
        else:
            payload = load_custom_metric_status(evaluation_dir)
        present_statuses.append(framework)
        status_passes.update(
            validate_status_contexts(
                payload,
                framework=framework,
                container_key=container_key,
            )
        )

    if not present_statuses:
        if allow_legacy:
            logger.warning(
                "[evaluation artifacts] using explicit legacy compatibility for bundle "
                "without validation status sidecars: %s",
                bundle_dir,
            )
            return manifest
        raise FileNotFoundError(
            "Strict evaluation bundle validation requires at least one metric status sidecar"
        )

    for model_name, framework_keys in status_keys_by_model.items():
        for framework, keys in framework_keys.items():
            for emitted_key in sorted(keys):
                matching_columns = [
                    column
                    for column in combined.columns
                    if isinstance(column, tuple)
                    and len(column) == 3
                    and column[0] == framework
                    and column[2] == emitted_key
                    and column[1] in {"utility", "privacy", "fairness", "audit"}
                ]
                if not matching_columns:
                    raise ValueError(
                        f"{framework} status identity {emitted_key!r} for model {model_name!r} "
                        "is absent from the combined evaluation table"
                    )

    execution_entry = manifest.get("syntheval_execution")
    if execution_entry is not None:
        execution_payload = load_syntheval_execution(evaluation_dir)
        execution_passes = set(execution_payload.get("passes", {}))
        status_execution_passes = {
            f"syntheval:{execution_pass}" for execution_pass in status_passes
        }
        if not execution_passes <= status_execution_passes:
            raise ValueError(
                "SynthEval execution sidecar contains a pass without a matching validation sidecar"
            )
        for pass_identity, model_payloads in execution_payload["passes"].items():
            observed_models = set(model_payloads)
            if observed_models != set(combined_models):
                raise ValueError(
                    f"SynthEval execution pass {pass_identity!r} model inventory does not match "
                    "combined evaluation"
                )

    return manifest


def _validate_final_holdout_evidence_payload(
    payload: Mapping[str, Any],
    *,
    entry: Mapping[str, Any],
    manifest: Mapping[str, Any],
    path: Path,
    registry: MetricContractRegistry | None = None,
) -> None:
    if payload.get("schema_version") != "final-holdout-evidence-v1":
        raise ValueError(f"Unsupported final-holdout evidence schema at {path}")
    if payload.get("evaluation_role") != "final_holdout":
        raise ValueError(f"Final-holdout evidence has an invalid evaluation role at {path}")
    state = _non_empty_string(payload.get("state"), "Final-holdout evidence state")
    if state not in {"blocked", "failed", "succeeded"}:
        raise ValueError(f"Final-holdout evidence has an invalid state {state!r} at {path}")
    state_sets = {
        "evidence_execution_state": EVIDENCE_EXECUTION_STATES,
        "metric_completeness_state": METRIC_COMPLETENESS_STATES,
        "score_completeness_state": SCORE_COMPLETENESS_STATES,
        "audit_outcome_state": AUDIT_OUTCOME_STATES,
    }
    for field, allowed in state_sets.items():
        value = _non_empty_string(payload.get(field), f"Final-holdout {field}")
        if value not in allowed:
            raise ValueError(f"Final-holdout {field} has invalid value {value!r} at {path}")
    execution_state = payload["evidence_execution_state"]
    metric_state = payload["metric_completeness_state"]
    score_state = payload["score_completeness_state"]
    audit_state = payload["audit_outcome_state"]
    state_tuple = (execution_state, metric_state, score_state, audit_state)
    valid_state_tuples = {
        "blocked": {
            ("blocked", "not_applicable", "not_applicable", "blocked"),
            ("not_started", "not_applicable", "not_applicable", "blocked"),
        },
        "succeeded": {("succeeded", "complete", "complete", "complete")},
        "failed": {
            ("failed", "incomplete", "indeterminate", "failed"),
            ("succeeded", "incomplete", "indeterminate", "failed"),
            ("succeeded", "complete", "indeterminate", "indeterminate"),
        },
    }
    if state_tuple not in valid_state_tuples[state]:
        raise ValueError(
            f"Final-holdout state {state!r} has contradictory state dimensions: {state_tuple!r}"
        )
    if execution_state == "not_started" and audit_state != "blocked":
        raise ValueError("Not-started final-holdout evidence must have blocked audit outcome")
    if execution_state == "blocked" and audit_state != "blocked":
        raise ValueError("Blocked final-holdout evidence must have blocked audit outcome")
    if metric_state == "complete" and execution_state in {"not_started", "blocked"}:
        raise ValueError("Blocked or not-started evidence cannot have complete metrics")
    if score_state == "complete" and metric_state != "complete":
        raise ValueError("Complete final-holdout score requires complete metrics")
    if audit_state == "complete" and score_state != "complete":
        raise ValueError("Complete final-holdout audit requires complete score")
    inventory = payload.get("provenance_inventory")
    if not isinstance(inventory, Mapping):
        raise ValueError("Final-holdout provenance_inventory must be an object")
    missing_inventory = [field for field in _PROVENANCE_INVENTORY_FIELDS if field not in inventory]
    if missing_inventory:
        raise ValueError(
            f"Final-holdout provenance_inventory is incomplete; missing {missing_inventory}"
        )
    if state != "blocked" and not payload.get("provenance_inventory_legacy_migrated", False):
        empty_inventory = [
            field
            for field in _PROVENANCE_INVENTORY_FIELDS
            if field not in {"intervals", "invalid_reasons"}
            and inventory[field] in ({}, [], None, "")
        ]
        if empty_inventory:
            raise ValueError(
                f"Final-holdout provenance_inventory contains empty evidence: {empty_inventory}"
            )
    score_payload = payload.get("release_score")
    if isinstance(score_payload, Mapping):
        selected_for_score = payload.get("selected_model")
        if isinstance(selected_for_score, str):
            _validate_release_score_record(score_payload, model_name=selected_for_score, path=path)
            if score_payload.get("status") == "succeeded" or (
                score_payload.get("status") == "indeterminate"
                and isinstance(
                    score_payload.get("provenance", {}).get("final_holdout_binding")
                    if isinstance(score_payload.get("provenance"), Mapping)
                    else None,
                    Mapping,
                )
            ):
                _validate_release_score_binding(
                    score_payload, payload, label="Final-holdout release score"
                )
    if (
        payload["audit_outcome_state"] == "complete"
        and score_payload is not None
        and (not isinstance(score_payload, Mapping) or score_payload.get("audit_only") is not True)
    ):
        raise ValueError("Completed final-holdout evidence must carry audit-only release score")
    semantic_context = payload.get("semantic_context")
    semantic_fingerprint = payload.get("semantic_context_fingerprint")
    manifest_semantic_fingerprint = manifest.get("semantic_context_fingerprint")
    if semantic_context is None:
        if semantic_fingerprint is not None:
            raise ValueError("Final-holdout semantic_context_fingerprint requires semantic_context")
        if state != "blocked" and manifest_semantic_fingerprint is not None:
            raise ValueError("Non-blocked final-holdout evidence requires semantic_context")
    else:
        if not isinstance(semantic_context, Mapping):
            raise ValueError("Final-holdout semantic_context must be an object or None")
        if not isinstance(semantic_fingerprint, str) or not semantic_fingerprint:
            raise ValueError("Final-holdout semantic_context_fingerprint must be non-empty")
        try:
            expected_semantic_fingerprint = semantic_context_digest(semantic_context)
        except (TypeError, ValueError) as exc:
            raise ValueError("Final-holdout semantic_context is invalid") from exc
        if semantic_fingerprint != expected_semantic_fingerprint:
            raise ValueError(
                "Final-holdout semantic_context_fingerprint does not match semantic_context"
            )
        if (
            manifest_semantic_fingerprint is not None
            and semantic_fingerprint != manifest_semantic_fingerprint
        ):
            raise ValueError(
                "Final-holdout semantic_context_fingerprint does not match "
                "the evaluation artifact manifest"
            )
    if entry.get("state") != state:
        raise ValueError(
            f"Final-holdout evidence state does not match its manifest entry at {path}"
        )
    selected_model = payload.get("selected_model")
    if selected_model is not None:
        _non_empty_string(selected_model, "Final-holdout selected model")
    if entry.get("selected_model") != selected_model:
        raise ValueError(
            f"Final-holdout selected model does not match its manifest entry at {path}"
        )
    model_names = manifest["combined_evaluation"]["models"]
    if state == "blocked":
        if selected_model is not None:
            raise ValueError("Blocked final-holdout evidence must not select a model")
    elif selected_model is None:
        raise ValueError(f"{state.capitalize()} final-holdout evidence must select a model")
    elif selected_model not in model_names:
        raise ValueError(
            f"Final-holdout selected model {selected_model!r} is absent from combined evaluation"
        )

    evidence_role = payload.get("evidence_role")
    if evidence_role is not None and evidence_role != "final_holdout":
        raise ValueError(f"Final-holdout evidence has an invalid evidence_role at {path}")
    fit_roles = payload.get("fit_roles")
    if fit_roles is not None and _string_list(fit_roles, "Final-holdout fit_roles") != [
        "train",
        "tuning",
    ]:
        raise ValueError("Final-holdout evidence must declare fit_roles=['train', 'tuning']")
    if state == "blocked" and payload.get("final_refit") is not None:
        raise ValueError("Blocked final-holdout evidence must not include final_refit metadata")
    if state != "blocked":
        refit = payload.get("final_refit")
        if not isinstance(refit, Mapping):
            raise ValueError(
                "Non-blocked final-holdout evidence requires final_refit generator metadata"
            )
        generator_metadata = refit.get("generator_metadata")
        if generator_metadata is None:
            raise ValueError(
                "Non-blocked final-holdout evidence requires complete generator_metadata"
            )
        _validate_generator_metadata_payload(
            generator_metadata,
            "Final-holdout final_refit generator_metadata",
        )
        if refit.get("provenance_state") != "legacy":
            if refit.get("model_name") != selected_model:
                raise ValueError("Final-refit model does not match the selected model")
            manifest_role_context = manifest.get("role_context")
            manifest_role_fingerprints = manifest.get("role_context_fingerprint")
            if isinstance(manifest_role_context, Mapping) and "candidate" in manifest_role_context:
                if refit.get("role_context") != manifest_role_context["candidate"]:
                    raise ValueError(
                        "Final-refit role context does not match the artifact manifest"
                    )
                if not isinstance(manifest_role_fingerprints, Mapping):
                    raise ValueError("Final-refit role-context fingerprint is missing")
                if refit.get("role_context_fingerprint") != manifest_role_fingerprints.get(
                    "candidate"
                ):
                    raise ValueError(
                        "Final-refit role-context fingerprint does not match the artifact manifest"
                    )
            if (
                manifest.get("semantic_context_fingerprint") is not None
                and refit.get("semantic_context_digest") != manifest["semantic_context_fingerprint"]
            ):
                raise ValueError(
                    "Final-refit semantic context does not match the artifact manifest"
                )
            contract_entry = manifest.get("metric_contract_manifest")
            if isinstance(contract_entry, Mapping) and refit.get(
                "registry_digest"
            ) != contract_entry.get("digest"):
                raise ValueError("Final-refit registry digest does not match the artifact manifest")

    manifest_role_context = manifest.get("role_context")
    if isinstance(manifest_role_context, Mapping) and "full" in manifest_role_context:
        if payload.get("role_context") != manifest_role_context["full"]:
            raise ValueError("Final-holdout role context does not match the artifact manifest")
        manifest_fingerprints = manifest.get("role_context_fingerprint")
        if not isinstance(manifest_fingerprints, Mapping):
            raise ValueError("Final-holdout role-context fingerprint is missing from the manifest")
        if payload.get("role_context_fingerprint") != manifest_fingerprints.get("full"):
            raise ValueError(
                "Final-holdout role-context fingerprint does not match the artifact manifest"
            )
    elif "role_context" in payload:
        _validate_role_context_payload(payload["role_context"], "Final-holdout role_context")
        if "role_context_fingerprint" in payload:
            _non_empty_string(
                payload["role_context_fingerprint"],
                "Final-holdout role_context_fingerprint",
            )

    candidate_selection = payload.get("candidate_selection")
    if candidate_selection is not None:
        if not isinstance(candidate_selection, Mapping):
            raise ValueError("Final-holdout candidate_selection must be an object")
        if candidate_selection.get("source") != "combined_evaluation.csv":
            raise ValueError("Final-holdout candidate selection has an invalid source")
        if candidate_selection.get("model") != selected_model:
            raise ValueError(
                "Final-holdout candidate selection model does not match selected_model"
            )
        _finite_or_none(
            candidate_selection.get("overall_rank"),
            "Final-holdout candidate selection overall_rank",
        )
        gate_pass = candidate_selection.get("privacy_gate_pass")
        if gate_pass is not None and not isinstance(gate_pass, bool):
            raise ValueError("Final-holdout candidate selection privacy_gate_pass must be boolean")

    frameworks = payload.get("frameworks")
    if frameworks is not None and not isinstance(frameworks, Mapping):
        raise ValueError("Final-holdout frameworks must be an object")
    if not isinstance(frameworks, Mapping) or registry is None:
        return

    def validate_result(
        result_payload: Any,
        *,
        model_name: str,
        framework: str,
        label: str,
        execution_pass: str,
    ) -> None:
        result = _metric_validation_result_from_payload(
            result_payload,
            label=label,
            model_name=model_name,
            framework=framework,
            registry=registry,
        )
        context = result.evaluation_context
        if context is None or context.evaluation_role != "final_holdout":
            raise ValueError(f"{label} must carry final-holdout evaluation context")
        if context.execution_pass != execution_pass:
            raise ValueError(f"{label} execution pass does not match its final-holdout section")

    synthcity = frameworks.get("synthcity")
    if isinstance(synthcity, Mapping):
        validation = synthcity.get("validation")
        if isinstance(validation, Mapping):
            for model_name, result_payload in validation.items():
                validate_result(
                    result_payload,
                    model_name=model_name,
                    framework="synthcity",
                    label=f"Final-holdout SynthCity model {model_name!r}",
                    execution_pass="main",
                )

    custom = frameworks.get("custom")
    if isinstance(custom, Mapping):
        validation = custom.get("validation")
        if isinstance(validation, Mapping):
            for model_name, result_payload in validation.items():
                validate_result(
                    result_payload,
                    model_name=model_name,
                    framework="custom",
                    label=f"Final-holdout custom model {model_name!r}",
                    execution_pass="main",
                )
        release_evidence_validation = custom.get("release_evidence_validation")
        if state != "blocked" and not isinstance(release_evidence_validation, Mapping):
            raise ValueError(
                "Non-blocked final-holdout evidence requires release-evidence validation"
            )
        if isinstance(release_evidence_validation, Mapping):
            for model_name, result_payload in release_evidence_validation.items():
                result = _metric_validation_result_from_payload(
                    result_payload,
                    label=f"Final-holdout release-evidence model {model_name!r}",
                    model_name=model_name,
                    framework="custom",
                    registry=registry,
                    allow_mixed_execution_pass=True,
                )
                if result.evaluation_context is None:
                    raise ValueError(
                        f"Final-holdout release-evidence model {model_name!r} must carry evaluation context"
                    )
                context = result.evaluation_context
                if context.evaluation_role != "final_holdout":
                    raise ValueError(
                        f"Final-holdout release-evidence model {model_name!r} has the wrong evaluation role"
                    )
                expected_keys = tuple(CUSTOM_CANONICAL_MANIFEST)
                observed_keys = tuple(
                    record.expected_key
                    for record in sorted(
                        result.expected_records,
                        key=lambda record: expected_keys.index(record.expected_key),
                    )
                )
                if observed_keys != expected_keys:
                    raise ValueError(
                        f"Final-holdout release-evidence model {model_name!r} must contain canonical identities"
                    )
                if any(
                    record.execution_pass
                    != (
                        "final_audit"
                        if record.expected_key == "equalized_odds.final.v1"
                        else "main"
                    )
                    for record in result.expected_records
                ):
                    raise ValueError(
                        f"Final-holdout release-evidence model {model_name!r} has an invalid execution pass"
                    )
                recorded_hashes = payload.get("role_hashes", {}).get("custom_raw_evaluation", {})
                if isinstance(recorded_hashes, Mapping) and dict(context.role_hashes) != dict(
                    recorded_hashes
                ):
                    raise ValueError(
                        f"Final-holdout release-evidence model {model_name!r} role hashes do not match evidence"
                    )
                for record in result.expected_records:
                    if record.status == "succeeded":
                        if record.fit_roles != ("train", "tuning"):
                            raise ValueError(
                                f"Final-holdout release-evidence {record.expected_key} has invalid fit_roles"
                            )
                        if not isinstance(record.support, Mapping):
                            raise ValueError(
                                f"Final-holdout release-evidence {record.expected_key} is missing support"
                            )
                        metadata = {
                            **dict(record.source_metadata),
                            **dict(record.result_metadata),
                            **dict(record.provenance),
                        }
                        if metadata.get("protocol_version") != "release-evidence-v2":
                            raise ValueError(
                                f"Final-holdout release-evidence {record.expected_key} has invalid protocol_version"
                            )
                        if metadata.get("release_transform_digest") is None:
                            raise ValueError(
                                f"Final-holdout release-evidence {record.expected_key} is missing release transform digest"
                            )
                if state == "succeeded" and not result.complete:
                    raise ValueError(
                        "Succeeded final-holdout evidence has incomplete release-evidence validation"
                    )

    syntheval = frameworks.get("syntheval")
    if isinstance(syntheval, Mapping):
        validation = syntheval.get("validation")
        if isinstance(validation, Mapping):
            for pass_identity, model_results in validation.items():
                if not isinstance(pass_identity, str):
                    raise ValueError("Final-holdout SynthEval validation pass must be a string")
                pass_framework, separator, execution_pass = pass_identity.partition(":")
                if not separator or pass_framework != "syntheval":
                    raise ValueError(
                        f"Invalid final-holdout SynthEval validation pass {pass_identity!r}"
                    )
                if not isinstance(model_results, Mapping):
                    raise ValueError(
                        f"Final-holdout SynthEval pass {pass_identity!r} must map models to results"
                    )
                for model_name, result_payload in model_results.items():
                    if pass_identity == "syntheval:main":
                        records = result_payload.get("records", [])
                        for record in records:
                            if record.get("expected_key") != "tstr_macro_f1.v1":
                                continue
                            verification_payload = {
                                "result_metadata": {
                                    **dict(record.get("source_metadata") or {}),
                                    **dict(record.get("result_metadata") or {}),
                                    **dict(record.get("provenance") or {}),
                                }
                            }
                            metadata = verification_payload["result_metadata"]
                            if isinstance(metadata.get("prediction_artifact"), Mapping):
                                verification_payload["prediction_artifact"] = metadata[
                                    "prediction_artifact"
                                ]
                            if record.get(
                                "status"
                            ) == "succeeded" and not is_verified_authoritative_tstr(
                                verification_payload
                            ):
                                raise ValueError(
                                    f"Final-holdout SynthEval model {model_name!r} has an "
                                    "unverified authoritative TSTR artifact"
                                )
                    validate_result(
                        result_payload,
                        model_name=model_name,
                        framework="syntheval",
                        label=(
                            f"Final-holdout SynthEval pass {pass_identity!r}, model {model_name!r}"
                        ),
                        execution_pass=execution_pass,
                    )

        execution = syntheval.get("execution")
        if isinstance(execution, Mapping):
            for execution_pass, model_payloads in execution.items():
                if execution_pass not in {"main", "binary_target"}:
                    raise ValueError(
                        f"Unknown final-holdout SynthEval execution pass {execution_pass!r}"
                    )
                if not isinstance(model_payloads, Mapping):
                    raise ValueError(
                        f"Final-holdout SynthEval execution pass {execution_pass!r} must map models"
                    )
                for model_name, execution_payload in model_payloads.items():
                    _validate_syntheval_execution_payload(
                        execution_payload,
                        label=(
                            f"Final-holdout SynthEval execution pass {execution_pass!r}, "
                            f"model {model_name!r}"
                        ),
                        model_name=model_name,
                        execution_pass=execution_pass,
                        expected_semantic_context_fingerprint=manifest.get(
                            "semantic_context_fingerprint"
                        ),
                    )


def load_final_holdout_evidence(evaluation_dir: str | Path) -> dict:
    """Load and integrity-check the post-selection final-holdout sidecar."""
    bundle_dir, manifest = _load_manifest(evaluation_dir)
    entry = manifest.get("final_holdout_evidence")
    if not entry:
        raise FileNotFoundError(
            "Final-holdout evidence sidecar is not recorded in the evaluation manifest"
        )
    payload, path = _read_json_artifact(
        bundle_dir.parent,
        entry,
        "Final-holdout evidence sidecar",
    )
    registry = _load_contract_registry_if_present(bundle_dir, manifest)
    original_refit = payload.get("final_refit")
    enriched_payload = _enrich_final_refit_evidence(payload, artifact_root=bundle_dir.parent)
    if isinstance(original_refit, Mapping) and isinstance(
        enriched_payload.get("final_refit"), Mapping
    ):
        enriched_refit = enriched_payload["final_refit"]
        for field in ("data_sha256", "metadata_sha256"):
            if field in original_refit:
                recorded_digest = _non_empty_string(
                    original_refit[field],
                    f"Final-holdout refit {field}",
                )
                if recorded_digest != enriched_refit[field]:
                    artifact_name = "synthetic data" if field == "data_sha256" else "metadata"
                    raise ValueError(
                        f"Final refit {artifact_name} failed integrity verification at {path}"
                    )
    _validate_final_holdout_evidence_payload(
        enriched_payload,
        entry=entry,
        manifest=manifest,
        path=path,
        registry=registry,
    )
    return enriched_payload


def load_release_score_evidence(evaluation_dir: str | Path) -> dict[str, Any]:
    """Load and integrity-check audit-only release-score decomposition."""
    bundle_dir, manifest = _load_manifest(evaluation_dir)
    entry = manifest.get("release_score_evidence")
    if not entry:
        raise FileNotFoundError(
            "Release-score evidence sidecar is not recorded in the evaluation manifest"
        )
    payload, path = _read_json_artifact(bundle_dir.parent, entry, "Release-score evidence sidecar")
    _validate_release_score_payload(payload, path=path)
    combined_models = set(manifest["combined_evaluation"]["models"])
    if set(payload["candidate_audit_models"]) != combined_models:
        raise ValueError(
            "Release-score candidate audit inventory does not match combined evaluation"
        )
    final_payload = load_final_holdout_evidence(evaluation_dir)
    if final_payload.get("state") != "succeeded":
        raise ValueError("Release-score evidence requires succeeded final-holdout evidence")
    if final_payload.get("audit_outcome_state") != "complete":
        raise ValueError("Release-score evidence requires complete final-holdout audit")
    final_entry = manifest.get("final_holdout_evidence")
    if not isinstance(final_entry, Mapping):
        raise ValueError("Release-score evidence requires final-holdout selected-model metadata")
    selected_model = _non_empty_string(
        final_payload.get("selected_model"), "Final-holdout selected model"
    )
    if payload["selected_model"] != selected_model:
        raise ValueError("Release-score selected model does not match final-holdout evidence")
    if set(payload["models"]) != {selected_model}:
        raise ValueError("Release-score evidence inventory must contain selected model only")
    final_score = final_payload.get("release_score")
    if not isinstance(final_score, Mapping):
        raise ValueError("Final-holdout evidence is missing its release score")
    _validate_release_score_record(final_score, model_name=selected_model, path=path)
    _validate_release_score_binding(final_score, final_payload, label="Final-holdout release score")
    if dict(final_score) != dict(payload["models"][selected_model]):
        raise ValueError("Release-score evidence does not match final-holdout release score")
    if set(entry["models"]) != set(payload["models"]):
        raise ValueError("Release-score manifest model inventory does not match its payload")
    if entry["selected_model"] != selected_model:
        raise ValueError("Release-score manifest selected model does not match its payload")
    if set(entry["candidate_audit_models"]) != combined_models:
        raise ValueError(
            "Release-score manifest candidate inventory does not match combined evaluation"
        )
    return payload


def load_metric_contract_manifest(evaluation_dir: str | Path) -> dict:
    """Load and integrity-check the metric contract manifest sidecar."""
    bundle_dir, manifest = _load_manifest(evaluation_dir)
    entry = manifest.get("metric_contract_manifest")
    if not entry:
        raise FileNotFoundError(
            "Metric contract manifest sidecar is not recorded in the evaluation manifest"
        )
    payload, _path = _read_json_artifact(
        bundle_dir.parent,
        entry,
        "Metric contract manifest sidecar",
    )
    _contract_registry_from_payload(payload, _path)
    if entry.get("digest") != payload["digest"]:
        raise ValueError("Metric contract manifest entry digest does not match its payload")
    if entry.get("registry_version") != payload["registry_version"]:
        raise ValueError(
            "Metric contract manifest entry registry version does not match its payload"
        )
    source_provenance = manifest.get("source_provenance")
    if isinstance(source_provenance, Mapping):
        recorded_digest = source_provenance.get("metric_contract_digest")
        if recorded_digest is not None and recorded_digest != payload["digest"]:
            raise ValueError("Metric contract manifest digest does not match source provenance")
    return payload


def _load_manifest(evaluation_dir: str | Path) -> tuple[Path, dict]:
    bundle_dir = artifact_bundle_dir(evaluation_dir)
    path = bundle_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation artifact manifest is missing at {path}. Run `synthdata-evaluate` "
            "once with this version to persist plot-ready evaluation artifacts."
        )
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Evaluation artifact manifest is unreadable") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Evaluation artifact manifest must contain a JSON object")
    if manifest.get("schema_version") != _ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported evaluation artifact schema at {path}: "
            f"{manifest.get('schema_version')!r}; expected {_ARTIFACT_SCHEMA_VERSION}."
        )
    source_provenance = manifest.get("source_provenance")
    if source_provenance is not None and not isinstance(source_provenance, dict):
        raise ValueError(f"Evaluation artifact source_provenance must be an object at {path}")
    if source_provenance is not None:
        _validate_source_provenance(source_provenance, path=path)
    _validate_role_context_manifest(manifest)
    _validate_semantic_context_manifest(manifest)
    _validate_generator_metadata_manifest(
        manifest.get("generator_metadata"),
        manifest=manifest,
        artifact_root=bundle_dir.parent,
    )
    combined = manifest.get("combined_evaluation")
    if not isinstance(combined, Mapping):
        raise ValueError(f"Evaluation artifact manifest is missing combined_evaluation at {path}")
    combined_path = _verified_artifact_path(bundle_dir.parent, combined, "Combined evaluation")
    recorded_models = _string_list(
        combined.get("models"),
        "Combined evaluation models",
        unique=True,
    )
    try:
        from synthdata.evaluation.combine import load_combined_table

        combined_frame = load_combined_table(combined_path, validate_artifact=False)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError(
            f"Combined evaluation artifact is semantically invalid at {combined_path}"
        ) from exc
    actual_models = [str(model_name) for model_name in combined_frame.index]
    if actual_models != recorded_models:
        raise ValueError(
            "Combined evaluation model inventory does not match its manifest: "
            f"recorded={recorded_models!r}, actual={actual_models!r}"
        )
    for key in ("log_disparity", "native_syntheval_plots"):
        value = manifest.get(key)
        if value is not None and not isinstance(value, dict):
            raise ValueError(f"Evaluation artifact manifest field {key!r} must be an object")
    native = manifest.get("native_syntheval_plots", {})
    if native:
        root_value = native.get("root")
        files = native.get("files", [])
        if root_value is not None:
            _native_bundle_root(bundle_dir, root_value)
        if not isinstance(files, list):
            raise ValueError("Native SynthEval plot manifest files must be a list")
        if root_value is None and files:
            raise ValueError("Native SynthEval plot files require a recorded root")
        seen_paths = set()
        for entry in files:
            if not isinstance(entry, Mapping):
                raise ValueError("Native SynthEval plot manifest entries must be objects")
            relative_path = _non_empty_string(entry.get("path"), "Native SynthEval plot")
            if relative_path in seen_paths:
                raise ValueError(
                    f"Native SynthEval plot manifest contains duplicate path {relative_path!r}"
                )
            seen_paths.add(relative_path)
            _relative_artifact_path(
                _native_bundle_root(bundle_dir, root_value),
                relative_path,
                "Native SynthEval plot",
            )
            digest = _non_empty_string(entry.get("sha256"), "Native SynthEval plot sha256")
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError("Native SynthEval plot sha256 must be a lowercase SHA-256 digest")
            _verified_artifact_path(
                _native_bundle_root(bundle_dir, root_value), entry, "Native SynthEval plot"
            )
    release_entry = manifest.get("release_score_evidence")
    if release_entry is not None:
        _validate_release_score_manifest_entry(release_entry)
    return bundle_dir, manifest


def _summary_number(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a real number or NaN")
    if math.isinf(float(value)):
        raise ValueError(f"{label} must not be infinite")


def _validate_log_disparity_table(
    table: Any,
    *,
    table_name: str,
    model_name: str,
    protected_group_cols: list[str],
) -> None:
    if not isinstance(table, pd.DataFrame):
        raise ValueError(
            f"Log-disparity table {table_name!r} for model {model_name!r} is not a DataFrame"
        )
    if table.columns.has_duplicates:
        raise ValueError(
            f"Log-disparity table {table_name!r} for model {model_name!r} has duplicate columns"
        )
    required_columns = {
        "leaf_results": {
            "Model",
            "EquityValue",
            "EquityLabel",
            "EquityColor",
            "BH_p",
            "background_n",
            "user_n",
        },
        "hierarchy_results": {
            "Model",
            "level",
            "TARGET_LABEL",
            "EquityValue",
            "EquityLabel",
            "EquityColor",
            "BH_p",
            "background_n",
            "user_n",
            "Background_Rate",
            "Observed_Rate",
        },
        "subgroup_table": {
            "Characteristic",
            "Protected Subgroup",
            "Equity Value",
            "BH-adjusted p-value",
            "EquityLabel",
            "EquityColor",
        },
        "leaf_equity_table": {
            "Protected Subgroup",
            "Equity Value",
            "BH-adjusted p-value",
            "EquityLabel",
            "EquityColor",
        },
        "legend_table": {"Description", "Metric Value Rule", "Color"},
        "label_counts": {"Model", "EquityLabel", "count"},
    }[table_name]
    missing = sorted(required_columns - set(table.columns))
    if missing:
        raise ValueError(
            f"Log-disparity table {table_name!r} for model {model_name!r} is missing {missing}"
        )
    for column in ("Model",):
        if column in table:
            values = table[column].dropna().tolist()
            if any(not isinstance(value, str) or value != model_name for value in values):
                raise ValueError(
                    f"Log-disparity table {table_name!r} has an invalid {column} identity"
                )
    string_columns = {
        "subgroup_table": (
            "Characteristic",
            "Protected Subgroup",
            "Equity Value",
            "BH-adjusted p-value",
            "EquityLabel",
            "EquityColor",
        ),
        "leaf_equity_table": (
            "Protected Subgroup",
            "Equity Value",
            "BH-adjusted p-value",
            "EquityLabel",
            "EquityColor",
        ),
        "legend_table": ("Description", "Metric Value Rule", "Color"),
        "label_counts": ("Model", "EquityLabel"),
    }.get(table_name, ("EquityLabel", "EquityColor"))
    for column in string_columns:
        values = table[column].dropna().tolist()
        if any(not isinstance(value, str) for value in values):
            raise ValueError(
                f"Log-disparity table {table_name!r} column {column!r} must contain strings"
            )
    numeric_columns = {
        "leaf_results": ("EquityValue", "BH_p", "background_n", "user_n"),
        "hierarchy_results": (
            "EquityValue",
            "BH_p",
            "background_n",
            "user_n",
            "Background_Rate",
            "Observed_Rate",
        ),
        "label_counts": ("count",),
    }.get(table_name, ())
    for column in numeric_columns:
        for value in table[column].dropna().tolist():
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(
                    f"Log-disparity table {table_name!r} column {column!r} must be numeric"
                )
    if table_name == "hierarchy_results":
        for column in protected_group_cols:
            if column not in table:
                raise ValueError(
                    f"Log-disparity hierarchy table for model {model_name!r} is missing protected column {column!r}"
                )


def _validate_log_disparity_metadata(
    metadata: Mapping[str, Any], *, model_name: str, path: Path
) -> tuple[list[str], dict[str, list[str]]]:
    if metadata.get("model_name") != model_name:
        raise ValueError(f"Log-disparity metadata has a mismatched model name at {path}")
    summary_stats = metadata.get("summary_stats")
    if not isinstance(summary_stats, dict):
        raise ValueError(f"Log-disparity metadata summary_stats must be an object at {path}")
    required_summary = (
        "model",
        "n_subgroups",
        "mean_abs_log_disparity",
        "median_abs_log_disparity",
        "share_significant_bh",
    )
    missing = [field for field in required_summary if field not in summary_stats]
    if missing:
        raise ValueError(f"Log-disparity metadata is missing {missing} at {path}")
    if summary_stats["model"] != model_name:
        raise ValueError(f"Log-disparity summary has a mismatched model name at {path}")
    _non_negative_int(summary_stats["n_subgroups"], "Log-disparity n_subgroups")
    for field in (
        "mean_abs_log_disparity",
        "median_abs_log_disparity",
        "share_significant_bh",
    ):
        _summary_number(summary_stats[field], f"Log-disparity summary {field}")
    share = summary_stats["share_significant_bh"]
    if not math.isnan(float(share)) and not 0.0 <= float(share) <= 1.0:
        raise ValueError("Log-disparity share_significant_bh must be between 0 and 1")
    protected_group_cols = _string_list(
        metadata.get("protected_group_cols"),
        "Log-disparity protected_group_cols",
        unique=True,
    )
    protected_order_map = metadata.get("protected_order_map")
    if not isinstance(protected_order_map, dict):
        raise ValueError(f"Log-disparity protected_order_map must be an object at {path}")
    if set(protected_order_map) != set(protected_group_cols):
        raise ValueError(f"Log-disparity protected_order_map keys do not match metadata at {path}")
    normalized_order_map = {}
    for column, values in protected_order_map.items():
        normalized_order_map[column] = _string_list(
            values,
            f"Log-disparity protected_order_map[{column!r}]",
            unique=True,
        )
    _string_list(metadata.get("target_order"), "Log-disparity target_order", unique=True)
    return protected_group_cols, normalized_order_map


def load_log_disparity_reports(evaluation_dir: str | Path) -> dict[str, dict]:
    """Load persisted log-disparity reports for offline Plotly rendering."""
    bundle_dir, manifest = _load_manifest(evaluation_dir)
    reports = {}
    for model_name, entry in manifest.get("log_disparity", {}).items():
        model_name = _non_empty_string(model_name, "Log-disparity model name")
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"Log-disparity manifest entry for model {model_name!r} must be an object"
            )
        model_dir = _relative_artifact_path(bundle_dir, entry.get("path"), "Log-disparity model")
        metadata_digest = _non_empty_string(
            entry.get("metadata_sha256"), "Log-disparity metadata sha256"
        )
        metadata_path = _verified_artifact_path(
            model_dir,
            {"path": "metadata.json", "sha256": metadata_digest},
            f"Log-disparity metadata for model {model_name!r}",
        )
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Log-disparity metadata for model {model_name!r} is unreadable"
            ) from exc
        if not isinstance(metadata, dict):
            raise ValueError(
                f"Log-disparity metadata for model {model_name!r} must be an object at {metadata_path}"
            )
        if entry.get("state") not in {"failed", "indeterminate", "succeeded"}:
            raise ValueError(
                f"Log-disparity manifest entry for model {model_name!r} has invalid state "
                f"{entry.get('state')!r}"
            )
        if metadata.get("state") != entry["state"]:
            raise ValueError(
                f"Log-disparity metadata state does not match its manifest entry at {metadata_path}"
            )
        if metadata.get("state") == "failed":
            if "tables" in entry:
                raise ValueError(
                    f"Log-disparity failed artifact for model {model_name!r} must not include tables"
                )
            reason = _safe_log_disparity_reason(
                metadata.get("reason"), fallback="log_disparity_evaluation_failed"
            )
            reports[model_name] = {
                "state": "failed",
                "reason": reason,
                "error_type": _safe_log_disparity_error_type(metadata.get("error_type")),
            }
            continue
        if metadata.get("state") == "indeterminate":
            if "tables" in entry:
                raise ValueError(
                    f"Log-disparity indeterminate artifact for model {model_name!r} must not include tables"
                )
            reason = _non_empty_string(
                metadata.get("reason"),
                f"Log-disparity indeterminate reason for model {model_name!r}",
            )
            if reason not in _SAFE_LOG_DISPARITY_REASONS:
                raise ValueError(
                    f"Log-disparity indeterminate reason for model {model_name!r} is not allowlisted"
                )
            missing_tables = metadata.get("missing_tables")
            if missing_tables != list(_LOG_REPORT_TABLES):
                raise ValueError(
                    f"Log-disparity indeterminate artifact for model {model_name!r} has invalid missing_tables"
                )
            reports[model_name] = {
                "state": "indeterminate",
                "reason": reason,
                "missing_tables": list(missing_tables),
            }
            if "result_metadata" in metadata:
                result_metadata = metadata["result_metadata"]
                if not isinstance(result_metadata, dict):
                    raise ValueError(
                        f"Log-disparity indeterminate metadata for model {model_name!r} must include result_metadata"
                    )
                reports[model_name]["result_metadata"] = safe_metric_metadata(
                    result_metadata,
                    label=f"Log-disparity indeterminate result_metadata for model {model_name!r}",
                    strict=True,
                )
            continue
        protected_group_cols, protected_order_map = _validate_log_disparity_metadata(
            metadata,
            model_name=model_name,
            path=metadata_path,
        )
        report = {
            "state": "succeeded",
            "summary_stats": metadata["summary_stats"],
            "protected_group_cols": metadata["protected_group_cols"],
            "protected_order_map": metadata["protected_order_map"],
            "target_order": metadata["target_order"],
        }
        if "result_metadata" in metadata:
            if not isinstance(metadata["result_metadata"], dict):
                raise ValueError(
                    f"Log-disparity result_metadata must be an object at {metadata_path}"
                )
            report["result_metadata"] = safe_metric_metadata(
                metadata["result_metadata"],
                label=f"Log-disparity result_metadata for model {model_name!r}",
                strict=True,
            )
        tables = entry.get("tables")
        if not isinstance(tables, dict) or set(tables) != set(_LOG_REPORT_TABLES):
            raise ValueError(
                f"Log-disparity artifact for model {model_name!r} has an invalid table inventory"
            )
        for table_name in _LOG_REPORT_TABLES:
            table_entry = tables[table_name]
            if not isinstance(table_entry, Mapping):
                raise ValueError(
                    f"Log-disparity table manifest for {table_name!r}, model {model_name!r} must be an object"
                )
            table_path = _relative_artifact_path(
                model_dir,
                table_entry.get("filename"),
                f"Log-disparity table {table_name!r} for model {model_name!r}",
            )
            path = _verified_artifact_path(
                model_dir,
                {
                    "path": str(table_path.relative_to(model_dir)),
                    "sha256": table_entry.get("sha256"),
                },
                f"Log-disparity table {table_name!r} for model {model_name!r}",
            )
            try:
                table = pd.read_parquet(path)
            except (OSError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Log-disparity table {table_name!r} for model {model_name!r} is unreadable"
                ) from exc
            _validate_log_disparity_table(
                table,
                table_name=table_name,
                model_name=model_name,
                protected_group_cols=protected_group_cols,
            )
            report[table_name] = table
        missing = [name for name in _LOG_REPORT_TABLES if name not in report]
        if missing:
            raise ValueError(
                f"Log-disparity artifact for model {model_name!r} is incomplete; missing {missing}."
            )
        reports[model_name] = report
    return reports


def verify_native_syntheval_artifacts(evaluation_dir: str | Path) -> None:
    """Fail loudly if a recorded native SynthEval plot was deleted or changed."""
    _bundle_dir, manifest = _load_manifest(evaluation_dir)
    native = manifest.get("native_syntheval_plots", {})
    root_value = native.get("root")
    files = native.get("files", [])
    if root_value is not None:
        _non_empty_string(root_value, "Native SynthEval plot root")
    if not isinstance(files, list):
        raise ValueError("Native SynthEval plot manifest files must be a list")
    if root_value is None:
        logger.info(
            "[evaluation artifacts] native SynthEval plots were disabled for this evaluation"
        )
        return
    if not files:
        raise FileNotFoundError(
            f"No native SynthEval plot files were recorded under {root_value}. "
            "Re-run `synthdata-evaluate` to regenerate its native diagnostics."
        )
    root = _native_bundle_root(_bundle_dir, root_value)
    missing_or_changed = []
    seen_paths = set()
    for entry in files:
        if not isinstance(entry, Mapping):
            raise ValueError("Native SynthEval plot manifest entries must be objects")
        relative_path = _non_empty_string(entry.get("path"), "Native SynthEval plot")
        if relative_path in seen_paths:
            raise ValueError(
                f"Native SynthEval plot manifest contains duplicate path {relative_path!r}"
            )
        seen_paths.add(relative_path)
        path = _relative_artifact_path(root, relative_path, "Native SynthEval plot")
        digest = _non_empty_string(entry.get("sha256"), "Native SynthEval plot sha256")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(
                f"Native SynthEval plot sha256 must be a lowercase SHA-256 digest for {path}"
            )
        if path.is_symlink() or not path.is_file() or _file_digest(path) != digest:
            missing_or_changed.append(str(path))
    if missing_or_changed:
        raise FileNotFoundError(
            "Native SynthEval plot artifact(s) are missing or changed: "
            + ", ".join(missing_or_changed[:10])
            + (" ..." if len(missing_or_changed) > 10 else "")
            + ". Re-run `synthdata-evaluate` to regenerate them."
        )
    logger.info("[evaluation artifacts] verified %d native SynthEval plot file(s)", len(files))
