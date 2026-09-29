"""Orchestrates the evaluation stage: synthcity + SynthEval + custom (log
disparity / fork-only fairness) evaluators, combined into one ranked table.
"""

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pandas as pd

from synthdata.config import Config
from synthdata.data import (
    Dataset,
    dataframe_fingerprint,
    role_context_fingerprint,
    role_context_payload,
    semantic_context_digest,
    semantic_context_payload,
)
from synthdata.evaluation import (
    artifacts,
    combine,
    custom_eval,
    privacy_gate,
    release_evidence_eval,
    report,
    synthcity_eval,
    syntheval_eval,
)
from synthdata.evaluation.catalog import (
    syntheval_execution_keys_by_framework,
    syntheval_execution_manifest,
)
from synthdata.evaluation.metric_contracts import (
    DEFAULT_METRIC_CONTRACT_REGISTRY,
    MetricEvaluationContext,
    is_verified_authoritative_tstr,
)
from synthdata.evaluation.release import transform_release_roles
from synthdata.evaluation.release_score import compute_release_score
from synthdata.evaluation.tstr import run_tstr_evaluation
from synthdata.utils import ensure_dir, get_logger

logger = get_logger(__name__)


def _authoritative_tstr_results(
    executions: dict,
    *,
    trusted_role_hashes: dict[str, str] | None = None,
) -> dict[str, dict]:
    """Extract authoritative TSTR records without calculating TSTR here."""
    results = {}

    def verified(report: object) -> bool:
        if not isinstance(report, dict):
            return False
        return is_verified_authoritative_tstr(report, trusted_role_hashes=trusted_role_hashes)

    def blocked(reason: str) -> dict:
        return {
            "state": "blocked",
            "reason": reason,
            "result_metadata": {"tstr_producer_available": False},
        }

    for model_name, payload in (executions or {}).items():
        if not isinstance(payload, dict):
            payload = getattr(payload, "report", {})
        if not isinstance(payload, dict):
            continue
        for candidate in (
            payload.get("authoritative_tstr"),
            payload.get("tstr_result"),
            payload.get("task10_tstr"),
        ):
            if isinstance(candidate, dict):
                if verified(candidate):
                    results[model_name] = candidate
                else:
                    results[model_name] = blocked(
                        "authoritative TSTR producer metadata or artifact is unverified"
                    )
                break
        if model_name in results:
            continue
        for item in payload.get("metric_executions", ()):
            rows = item.get("results", item.get("rows"))
            if rows is None:
                rows = item.get("normalized_rows_v2", item.get("normalized_rows", ()))
            if isinstance(rows, dict):
                rows = rows.values()
            for row in rows:
                metadata = row.get("result_metadata", row.get("metadata", {}))
                if not isinstance(metadata, dict):
                    metadata = {}
                # Authoritative TSTR stores its durable fairness artifact either in row
                # metadata or directly beside normalized metric values.
                if "prediction_artifact" not in metadata and isinstance(
                    row.get("prediction_artifact"), dict
                ):
                    metadata = {**metadata, "prediction_artifact": row["prediction_artifact"]}
                if not isinstance(metadata, dict) or not verified(
                    {**row, "result_metadata": metadata}
                ):
                    continue
                if row.get("metric") not in {"macro_f1", "tstr_macro_f1"}:
                    continue
                results[model_name] = {
                    "state": "complete"
                    if row.get("raw_value", row.get("val")) is not None
                    else "blocked",
                    "macro_f1": row.get("raw_value", row.get("val")),
                    "equalized_odds": metadata.get("equalized_odds"),
                    "prediction_artifact": metadata.get("prediction_artifact"),
                    "class_supports": metadata.get("class_supports", metadata.get("support", {})),
                    "target_order": metadata.get("target_order", []),
                    "result_metadata": metadata,
                }
        if model_name not in results:
            results[model_name] = blocked(
                "authoritative TSTR producer result is unavailable or untagged"
            )
    return results


def _candidate_role_frames(
    dataset: Dataset,
    *,
    audit_only_legacy_adapter: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return fit, tuning, and final-holdout frames for evaluation context.

    Legacy two-role data has no tuning role. Substituting its final holdout is
    unsafe for candidate evaluation and is allowed only for an explicitly
    requested audit-only adapter.
    """
    train_frame = dataset.role_frame("train", imputed=True)
    tuning_frame = dataset.role_frame("tuning", imputed=True)
    final_holdout_frame = dataset.role_frame("final_holdout", imputed=True)
    if tuning_frame is None and dataset.legacy_two_role and audit_only_legacy_adapter:
        tuning_frame = final_holdout_frame
    elif tuning_frame is None and dataset.legacy_two_role:
        raise RuntimeError(
            "Legacy two-role datasets cannot substitute final_holdout for tuning; "
            "request the explicit audit-only legacy adapter"
        )
    missing = [
        role
        for role, frame in (
            ("train", train_frame),
            ("tuning", tuning_frame),
            ("final_holdout", final_holdout_frame),
        )
        if frame is None
    ]
    if missing:
        raise RuntimeError(
            "Evaluation requires populated imputed role frame(s): " + ", ".join(missing)
        )
    assert train_frame is not None
    assert tuning_frame is not None
    assert final_holdout_frame is not None
    return train_frame, tuning_frame, final_holdout_frame


def _syntheval_target_type_flags(dataset: Dataset, train_frame: pd.DataFrame) -> tuple[bool, bool]:
    """Resolve SynthEval binary/multiclass routing from the training target."""
    class_count = train_frame[dataset.target_column].nunique(dropna=False)
    target_is_binary = dataset.target_is_categorical and class_count == 2
    target_is_multiclass = dataset.target_is_categorical and class_count > 2
    return target_is_binary, target_is_multiclass


def _generation_metadata(cfg: Config, model_names: list[str]) -> dict[str, dict]:
    """Load complete generator cache envelopes for evaluation provenance."""
    metadata_by_model = {}
    generation_dir = Path(cfg.generation.output_dir)
    for model_name in model_names:
        metadata_path = generation_dir / f"{model_name}.cache.json"
        data_path = generation_dir / f"{model_name}.csv"
        # ``source_generation_root`` is persisted separately and is the root
        # used by the artifact bundler for relative source paths.  Persist
        # only names here; storing ``generation_dir / name`` would make the
        # bundler resolve the generation root twice when output_dir is
        # relative.
        metadata_source = f"{model_name}.cache.json"
        data_source = f"{model_name}.csv"
        if not metadata_path.exists():
            metadata_by_model[model_name] = {
                "state": "missing",
                "metadata_path": metadata_source,
                "data_path": data_source,
            }
            continue
        try:
            payload = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Generator metadata for {model_name!r} is unreadable at {metadata_path}"
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Generator metadata for {model_name!r} must be an object at {metadata_path}"
            )
        generator_metadata = payload.get("generator_metadata")
        is_current_envelope = payload.get("schema_version") == "generation-cache-v3"
        state = (
            "present"
            if is_current_envelope and generator_metadata is not None
            else "invalid"
            if is_current_envelope
            else "legacy"
            if generator_metadata is not None
            else "missing"
        )
        entry = {
            "state": state,
            "metadata_path": metadata_source,
            "data_path": data_source,
            "metadata_sha256": artifacts._file_digest(metadata_path),
            "data_sha256": artifacts._file_digest(data_path) if data_path.is_file() else None,
            "metadata": generator_metadata,
            "cache_metadata": payload,
        }
        if state == "invalid":
            entry["error"] = "Current generation cache is missing generator_metadata"
        metadata_by_model[model_name] = entry
    return metadata_by_model


def select_models(
    cfg: Config,
    synthetic_datasets: dict[str, pd.DataFrame],
    generation_inventory: artifacts.GenerationInventory | None = None,
) -> dict[str, pd.DataFrame]:
    """Restrict evaluation to requested, manifest-declared generated models."""
    requested = list(cfg.evaluation.models or [])
    if not requested and generation_inventory is not None:
        requested = list(generation_inventory.expected_outputs)
    if not requested:
        return synthetic_datasets
    missing = [model for model in requested if model not in synthetic_datasets]
    allowed_missing = (
        set(generation_inventory.failed_outputs) & set(requested)
        if generation_inventory is not None
        else set()
    )
    unexplained_missing = [model for model in missing if model not in allowed_missing]
    if unexplained_missing:
        raise ValueError(
            "Requested evaluation models are missing from generated datasets: "
            f"{unexplained_missing}. Regenerate or remove missing models before evaluation."
        )
    return {name: synthetic_datasets[name] for name in requested if name in synthetic_datasets}


def _evaluation_coverage(
    requested_models: list[str],
    selected_datasets: dict[str, pd.DataFrame],
    generation_inventory: artifacts.GenerationInventory | None,
) -> dict | None:
    """Describe validated model coverage without creating rows for failed outputs."""
    if generation_inventory is None:
        return None
    failed_names = set(generation_inventory.failed_outputs)
    failed_outputs = [name for name in requested_models if name in failed_names]
    if not failed_outputs:
        return None
    return {
        "status": "partial",
        "requested_models": list(requested_models),
        "evaluated_models": sorted(selected_datasets),
        "failed_outputs": failed_outputs,
    }


def _load_generation_inventory(experiment) -> artifacts.GenerationInventory | None:
    """Reload the experiment's validated generation inventory for evaluation."""
    if experiment is None:
        return None
    manifest_path = getattr(experiment, "manifest_path", None)
    generation_dir = getattr(experiment, "generation_dir", None)
    if manifest_path is None or generation_dir is None:
        raise ValueError(
            "Evaluation experiment must provide manifest_path and generation_dir "
            "to validate generation completeness"
        )
    return artifacts.load_generation_inventory(manifest_path, generation_dir)


def _synthcity_attack_target_types(dataset: Dataset, selection_cfg) -> dict[str, str]:
    """Resolve SynthCity attack targets from the dataset schema and config."""
    configured = dict(selection_cfg.sensitive_target_types)
    unknown = sorted(set(configured) - set(dataset.sensitive_columns))
    if unknown:
        raise ValueError(
            f"SynthCity sensitive_target_types contains non-sensitive columns: {unknown}"
        )
    target_types = {}
    for column in dataset.sensitive_columns:
        schema_entry = dataset.variable_schema.get(column)
        if schema_entry is None:
            raise ValueError(
                f"SynthCity attack target {column!r} is missing from the dataset variable schema"
            )
        schema_kind = schema_entry["kind"]
        configured_kind = configured.get(column)
        if configured_kind is not None and configured_kind != schema_kind:
            raise ValueError(
                f"SynthCity sensitive target {column!r} overrides dataset schema kind "
                f"{schema_kind!r} with {configured_kind!r}"
            )
        target_types[column] = schema_kind
    return target_types


def _synthcity_semantic_context(dataset: Dataset, selection_cfg) -> dict:
    """Resolve one authoritative semantic payload for SynthCity evaluation."""
    dataset_qis = list(dataset.quasi_identifier_columns)
    configured_qis = list(selection_cfg.quasi_identifier_columns)
    if configured_qis and configured_qis != dataset_qis:
        raise ValueError(
            "SynthCity quasi_identifier_columns must match the Dataset declaration: "
            f"dataset={dataset_qis!r}, evaluation={configured_qis!r}"
        )
    if len(dataset_qis) != len(set(dataset_qis)):
        raise ValueError(f"Dataset quasi_identifier_columns must be unique: {dataset_qis!r}")
    sensitive_overlap = sorted(set(dataset_qis) & set(dataset.sensitive_columns))
    if sensitive_overlap:
        raise ValueError(
            "Dataset sensitive_columns must not overlap quasi_identifier_columns: "
            f"{sensitive_overlap}"
        )
    target_overlap = sorted(set(dataset_qis) & {dataset.target_column})
    if target_overlap:
        raise ValueError(
            f"Dataset quasi_identifier_columns must not overlap target_column: {target_overlap}"
        )
    sensitive_target_overlap = sorted(set(dataset.sensitive_columns) & {dataset.target_column})
    if sensitive_target_overlap:
        raise ValueError(
            f"Dataset sensitive_columns must not overlap target_column: {sensitive_target_overlap}"
        )
    missing_qis = sorted(set(dataset_qis) - set(dataset.feature_columns))
    if missing_qis:
        raise ValueError(f"Dataset quasi_identifier_columns are not model features: {missing_qis}")
    semantic_context = semantic_context_payload(
        dataset,
        classification_score=selection_cfg.classification_score,
    )
    semantic_context["sensitive_target_types"] = _synthcity_attack_target_types(
        dataset, selection_cfg
    )
    return semantic_context


def _select_policy_model(combined: pd.DataFrame) -> tuple[str | None, str | None]:
    """Select candidate using complete fixed-transform tuning utility only."""
    utility_column = ("__all__", "utility", "U_tuning")
    if utility_column not in combined.columns:
        return None, "candidate table did not produce U_tuning"

    eligible = combined.loc[
        combined[utility_column].map(
            lambda value: (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            )
        )
    ]
    if eligible.empty:
        return None, "no candidate has complete finite U_tuning"
    selected = eligible.sort_values(utility_column, ascending=False).index[0]
    return str(selected), None


def _release_score_inputs(validations: dict) -> tuple[dict, dict, dict]:
    """Adapt validated final evidence into release-score component inputs.

    Release evidence deliberately emits aggregate records. Keep those records as the
    evidence attached to each adapter; never derive a component from a
    candidate-relative value or from an unsuccessful record.
    """
    utility: dict[str, object] = {}
    privacy: dict[str, object] = {}
    fairness: dict[str, object] = {}

    def add(
        destination: dict,
        name: str,
        value: object,
        evidence: object,
        adapter: str,
        *,
        normalized: bool = False,
    ) -> None:
        if value is not None and name not in destination:
            component = {
                "value": value,
                "evidence": evidence,
                "adapter": adapter,
            }
            if normalized:
                component["score"] = value
            destination[name] = component

    def iter_validations(value):
        if isinstance(value, dict):
            for nested in value.values():
                yield from iter_validations(nested)
        elif hasattr(value, "records"):
            yield value

    for validation in iter_validations(validations):
        records = getattr(validation, "records", ())
        for record in records:
            if getattr(record, "status", None) != "succeeded":
                continue
            key = str(getattr(record, "expected_key", "")).lower()
            raw = getattr(record, "raw_value", None)
            metadata = {}
            metadata.update(getattr(record, "source_metadata", {}) or {})
            metadata.update(getattr(record, "result_metadata", {}) or {})
            evidence = {"record": record.to_dict(), "metadata": metadata}
            if "release_privacy.v1" in key:
                for component in ("k", "l"):
                    aggregate = metadata.get(component)
                    score = aggregate.get("safety_score") if isinstance(aggregate, dict) else None
                    add(
                        privacy,
                        f"S_{component}",
                        score,
                        evidence,
                        f"{key}.{component}.safety_score",
                        normalized=True,
                    )
                for component in ("dcr", "epsilon", "mia", "attribute"):
                    alias = {
                        "dcr": "S_DCR",
                        "epsilon": "S_epsilon",
                        "mia": "S_MIA",
                        "attribute": "S_attribute",
                    }[component]
                    add(
                        privacy,
                        alias,
                        metadata.get(component),
                        evidence,
                        f"{key}.{component}",
                        normalized=True,
                    )
            elif "representation_evidence.v1" in key:
                add(fairness, "S_representation", raw, evidence, key, normalized=True)
                stats = metadata.get("summary_stats", {})
                if isinstance(stats, dict):
                    add(
                        fairness,
                        "S_worst_log_disparity",
                        stats.get("worst_abs_log_disparity"),
                        evidence,
                        f"{key}.summary_stats.worst_abs_log_disparity",
                        normalized=True,
                    )
            elif "equalized_odds.final.v1" in key:
                add(fairness, "S_EO", raw, evidence, key, normalized=True)
            elif any(name in key for name in ("tstr", "macro_f1")):
                add(utility, "tstr", raw, evidence, key)
            elif "mmd" in key:
                add(utility, "mmd", raw, evidence, key)
            elif "jsd" in key:
                add(utility, "jsd", raw, evidence, key)
            elif "dcr" in key or "distance_to_closest" in key:
                add(privacy, "dcr", raw, evidence, key)
            elif "epsilon" in key:
                add(privacy, "epsilon", raw, evidence, key)
            elif "mia" in key:
                add(privacy, "mia", raw, evidence, key)
            elif "attribute" in key:
                add(privacy, "attribute", raw, evidence, key)
            elif "log_disparity" in key:
                add(fairness, "worst_log_disparity", raw, evidence, key)
    return utility, privacy, fairness


def _validation_payloads(
    validations: dict,
) -> dict[str, dict]:
    """Serialize validation results while preserving framework/pass identity."""
    payloads = {}
    for key, model_validations in sorted(validations.items()):
        pass_key = ":".join(str(part) for part in key) if isinstance(key, tuple) else str(key)
        payloads[pass_key] = {
            model_name: result.to_dict() if hasattr(result, "to_dict") else result
            for model_name, result in sorted(model_validations.items())
        }
    return payloads


def _single_framework_validation_payload(validations: dict) -> dict:
    """Serialize model-keyed validation results for one framework."""
    return {
        model_name: result.to_dict() if hasattr(result, "to_dict") else result
        for model_name, result in sorted(validations.items())
    }


def _incomplete_validation_records(
    framework: str,
    validations: dict,
    execution_pass: str | None = None,
) -> list[dict]:
    """Serialize incomplete final-evidence validation outcomes."""
    failures = []
    for model_name, validation in sorted(validations.items()):
        if validation.complete:
            continue
        failures.append(
            {
                "framework": framework,
                "execution_pass": execution_pass,
                "model": model_name,
                "status_counts": validation.status_counts,
                "failed_keys": list(validation.failed_keys),
                "indeterminate_keys": list(validation.indeterminate_keys),
            }
        )
    return failures


def _final_holdout_state_dimensions(
    validation_failures: list[dict], release_score: dict
) -> tuple[str, str, str, str]:
    """Derive final-holdout state dimensions without overstating evidence."""
    metric_state = "incomplete" if validation_failures else "complete"
    score_succeeded = release_score.get("status") == "succeeded"
    score_state = "complete" if score_succeeded and not validation_failures else "indeterminate"
    audit_state = (
        "failed" if validation_failures else "complete" if score_succeeded else "indeterminate"
    )
    evidence_state = "succeeded" if metric_state == "complete" and score_succeeded else "failed"
    return evidence_state, metric_state, score_state, audit_state


def _run_final_holdout_evidence(
    cfg: Config,
    dataset: Dataset,
    selected_datasets: dict[str, pd.DataFrame],
    combined: pd.DataFrame,
    group_context: dict,
    population_unit: str,
    group_mode: str,
    role_context: dict,
    role_context_fingerprints: dict[str, str],
) -> dict:
    """Evaluate only the selected candidate against the untouched final role."""
    base_evidence = {
        "evaluation_role": "final_holdout",
        "role_context": role_context["full"],
        "role_context_fingerprint": role_context_fingerprints["full"],
        "fit_roles": ["train", "tuning"],
        "evidence_role": "final_holdout",
    }

    def blocked_evidence(reason: str) -> dict:
        """Build auditable evidence when final-holdout evaluation cannot start."""
        return {
            **base_evidence,
            "state": "blocked",
            "evidence_execution_state": "blocked",
            "metric_completeness_state": "not_applicable",
            "score_completeness_state": "not_applicable",
            "audit_outcome_state": "blocked",
            "provenance_inventory": {
                field: {}
                for field in (
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
                )
            }
            | {
                "invalid_reasons": {"final_holdout": reason},
                "selected_model_provenance": {
                    "model": None,
                    "status": "not_selected",
                    "reason": reason,
                },
            },
            "selected_model": None,
            "reason": reason,
        }

    if not dataset.has_canonical_roles:
        return blocked_evidence(
            "legacy_two_role dataset cannot provide distinct final-holdout evidence"
        )

    selected_model, selection_error = _select_policy_model(combined)
    if selection_error is not None:
        return blocked_evidence(selection_error)
    if selected_model not in selected_datasets:
        raise RuntimeError(
            f"Candidate ranking selected unknown model {selected_model!r}; "
            f"available models: {sorted(selected_datasets)}"
        )

    eval_cfg = cfg.evaluation
    output_dir = ensure_dir(eval_cfg.output_dir)
    synthcity_semantics = _synthcity_semantic_context(dataset, eval_cfg.synthcity)
    base_evidence["semantic_context"] = synthcity_semantics
    base_evidence["semantic_context_fingerprint"] = semantic_context_digest(synthcity_semantics)
    train_frame = dataset.role_frame("train", imputed=True)
    tuning_frame = dataset.role_frame("tuning", imputed=True)
    final_holdout_frame = dataset.role_frame("final_holdout", imputed=True)
    if train_frame is None or tuning_frame is None or final_holdout_frame is None:
        raise RuntimeError(
            "Final-holdout evidence requires populated imputed train, tuning, and final_holdout roles"
        )
    real_fit_frame = pd.concat([train_frame, tuning_frame], ignore_index=True)
    real_fit_frame_fingerprint = dataframe_fingerprint(real_fit_frame)
    from synthdata.generation.pipeline import refit_selected_model

    refit_frame, refit_metadata = refit_selected_model(
        cfg,
        dataset,
        selected_model,
        output_dir=output_dir / "final_refit",
    )
    selected_dataset = {selected_model: refit_frame}
    final_group_context = syntheval_eval.build_group_context(
        dataset,
        selected_dataset,
        group_mode=group_mode,
        group_column=group_context["group_column"],
    )
    final_role_hashes = {
        "train": dataframe_fingerprint(train_frame),
        "tuning": dataframe_fingerprint(tuning_frame),
        "final_holdout": dataframe_fingerprint(final_holdout_frame),
        "refit_fit": refit_metadata["fit_frame_fingerprint"],
    }
    raw_train_frame = dataset.role_frame("train", imputed=False)
    raw_tuning_frame = dataset.role_frame("tuning", imputed=False)
    raw_final_holdout_frame = dataset.role_frame("final_holdout", imputed=False)
    if raw_train_frame is None or raw_tuning_frame is None or raw_final_holdout_frame is None:
        raise RuntimeError(
            "Final-holdout evidence requires populated raw train, tuning, and final_holdout roles"
        )
    final_custom_role_hashes = {
        "train": dataframe_fingerprint(raw_train_frame),
        "tuning": dataframe_fingerprint(raw_tuning_frame),
        "final_holdout": dataframe_fingerprint(raw_final_holdout_frame),
        "refit_fit": refit_metadata["fit_frame_fingerprints"]["raw"],
    }
    released_final_synthetic, released_final_roles, release_transform_metadata = (
        transform_release_roles(
            refit_frame,
            {"final_holdout": raw_final_holdout_frame},
            dataset.release_generalization,
        )
    )
    released_final_dataset = {selected_model: released_final_synthetic}
    # All final evidence consumers share this exact release-form object.  Do not
    # let individual metric adapters transform or normalize independent copies.
    selected_dataset = released_final_dataset
    authoritative_tstr_results = {}
    if eval_cfg.custom.enabled:
        for model_name in selected_dataset:
            authoritative_tstr_results[model_name] = run_tstr_evaluation(
                released_final_dataset[model_name],
                released_final_roles["final_holdout"],
                target_column=dataset.target_column,
                evaluation_role="final_holdout",
                seed=cfg.seed,
                protected_columns=list(dataset.protected_columns),
                role_hashes=final_custom_role_hashes,
            ).envelope
    group_configuration = {
        "group_context": final_group_context,
        "fit_roles": ["train", "tuning"],
        "fit_frame_fingerprint": real_fit_frame_fingerprint,
        "refit_fit_frame_fingerprint": refit_metadata["fit_frame_fingerprint"],
    }

    train_group_ids = dataset.role_groups.get("train")
    tuning_group_ids = dataset.role_groups.get("tuning")
    final_synthcity_results = synthcity_eval.run_synthcity_evaluation(
        selected_dataset,
        real_fit_frame,
        final_holdout_frame,
        dataset.target_column,
        synthcity_semantics["protected_columns"],
        eval_cfg.synthcity,
        n_samples=cfg.generation.n_samples,
        seed=cfg.seed,
        workspace=str(output_dir / "synthcity_final_holdout_workspace"),
        real_reference_group_ids=dataset.role_groups.get("final_holdout"),
        real_train_group_ids=(list(train_group_ids) if train_group_ids is not None else [])
        + (list(tuning_group_ids) if tuning_group_ids is not None else []),
        sensitive_target_types=synthcity_semantics["sensitive_target_types"],
        feature_types=synthcity_semantics["feature_types"],
        source_table=synthcity_semantics["source_table"],
        semantic_context=synthcity_semantics,
        quasi_identifier_columns=synthcity_semantics["quasi_identifier_columns"],
        task_type="classification" if dataset.target_is_categorical else "regression",
        classification_score=synthcity_semantics["classification_score"],
        group_mode=group_mode,
        evaluation_role="final_holdout",
        released_synthetic_datasets=released_final_dataset,
        released_reference_frame=released_final_roles["final_holdout"],
    )
    final_synthcity_metric_config = synthcity_eval.resolve_metric_config(eval_cfg.synthcity)
    final_synthcity_validations = synthcity_eval.validate_synthcity_results(
        final_synthcity_results,
        final_synthcity_metric_config,
        role_hashes=final_role_hashes,
        requested_use="audit",
        variable_columns=list(real_fit_frame.columns),
        attack_target_types=_synthcity_attack_target_types(dataset, eval_cfg.synthcity),
        context=MetricEvaluationContext(
            evaluation_role="final_holdout",
            role_hashes=final_role_hashes,
            population_unit=population_unit,
            group_mode=group_mode,
            resolved_configuration=group_configuration,
        ),
    )

    final_execution_failures = []
    final_syntheval_output = (None, None, {})
    if eval_cfg.syntheval.enabled:
        try:
            final_syntheval_output = syntheval_eval.run_syntheval_evaluation(
                selected_dataset,
                dataset,
                eval_cfg.syntheval,
                preset_dir=output_dir,
                ranking_strategy=eval_cfg.ranking_strategy,
                output_folder=output_dir / "syntheval_final_holdout",
                plots_output_dir=None,
                positive_class=eval_cfg.positive_class,
                execution_cfg=eval_cfg.syntheval_execution,
                return_execution=True,
                group_context=final_group_context,
                semantic_context=synthcity_semantics,
                evaluation_role="final_holdout",
                fit_frame=real_fit_frame,
                fit_roles=("train", "tuning"),
                released_final_holdout_frame=released_final_roles["final_holdout"],
                released_synthetic_datasets=released_final_dataset,
            )
        except Exception as exc:  # noqa: BLE001 - preserve process-control exceptions
            exception_type = type(exc).__name__
            logger.error(
                "[final holdout] SynthEval main pass failed for selected model=%s; "
                "reason_code=%s; exception_type=%s",
                selected_model,
                "final_holdout_execution_failed",
                exception_type,
            )
            final_execution_failures.append(
                {
                    "stage": "final_holdout",
                    "framework": "syntheval",
                    "execution_pass": "main",
                    "model": selected_model,
                    "status": "failed",
                    "policy_eligible": False,
                    "error_type": "SynthEvalExecutionError",
                    "exception_type": exception_type,
                    "reason_code": "final_holdout_execution_failed",
                    "failure_reason": "Final-holdout SynthEval execution failed.",
                }
            )
            final_syntheval_output = (None, None, {})

    if len(final_syntheval_output) == 3:
        final_benchmark_results, final_benchmark_ranks, final_syntheval_executions = (
            final_syntheval_output
        )
    else:
        final_benchmark_results, final_benchmark_ranks = final_syntheval_output
        final_syntheval_executions = {}
    final_syntheval_validations = {}
    if eval_cfg.syntheval.enabled:
        target_is_binary, target_is_multiclass = _syntheval_target_type_flags(dataset, train_frame)
        final_syntheval_preset = syntheval_eval.build_preset(
            eval_cfg.syntheval,
            positive_class=eval_cfg.positive_class,
            target_is_binary=target_is_binary,
            target_is_multiclass=target_is_multiclass,
        )
        final_syntheval_manifest = syntheval_execution_manifest(
            final_syntheval_preset,
            include_holdout_outputs=True,
            target_columns=[dataset.target_column],
            protected_columns=dataset.protected_columns,
            target_is_binary=False if target_is_multiclass else target_is_binary,
        )
        final_syntheval_validations = syntheval_eval.validate_syntheval_results(
            final_benchmark_results,
            final_benchmark_ranks,
            syntheval_eval.extend_syntheval_expected_diagnostics(
                syntheval_execution_keys_by_framework(final_syntheval_manifest),
                final_benchmark_results,
                structured_executions=final_syntheval_executions,
            ),
            role_hashes=final_role_hashes,
            model_names=[selected_model],
            execution_pass="main",
            target_view="native",
            evaluation_role="final_holdout",
            requested_use="audit",
            structured_executions=final_syntheval_executions,
            population_unit=population_unit,
            group_mode=group_mode,
            resolved_configuration=group_configuration,
        )

    final_syntheval_validations = {
        key: value for key, value in final_syntheval_validations.items() if key[0] == "syntheval"
    }

    final_custom_reports = custom_eval.run_log_disparity_evaluation(
        selected_dataset,
        dataset,
        eval_cfg.log_disparity,
        eval_cfg.custom,
        evaluation_role="final_holdout",
        reference_frame=released_final_roles.get("final_holdout"),
    )
    final_custom_validations = (
        custom_eval.validate_log_disparity_results(
            final_custom_reports,
            [selected_model],
            role_hashes=final_custom_role_hashes,
            requested_use="audit",
            population_unit=population_unit,
            group_mode=group_mode,
            evaluation_role="final_holdout",
            resolved_configuration=group_configuration,
        )
        if final_custom_reports
        else {}
    )
    final_release_evidence_observations = release_evidence_eval.run_release_evidence_evaluation(
        selected_dataset,
        dataset,
        evaluation_role="final_holdout",
        generalization=dataset.release_generalization,
        quasi_identifiers=list(dataset.quasi_identifier_columns),
        sensitive_fields=list(dataset.sensitive_columns),
        protected_columns=list(dataset.protected_columns),
        role_hashes=final_custom_role_hashes,
        tstr_results=authoritative_tstr_results,
        seed=cfg.seed,
        release_form_inputs=(
            released_final_synthetic,
            released_final_roles,
            release_transform_metadata,
        ),
    )
    final_release_evidence_validations = release_evidence_eval.validate_release_evidence_results(
        final_release_evidence_observations,
        role_hashes=final_custom_role_hashes,
        evaluation_role="final_holdout",
        population_unit=population_unit,
        group_mode=group_mode,
        requested_use="audit",
    )

    final_validation_failures = list(final_execution_failures)
    final_validation_failures.extend(
        _incomplete_validation_records("synthcity", final_synthcity_validations)
    )
    for (framework, execution_pass), validations in final_syntheval_validations.items():
        final_validation_failures.extend(
            _incomplete_validation_records(framework, validations, execution_pass)
        )
    final_validation_failures.extend(
        _incomplete_validation_records("custom", final_custom_validations)
    )
    if eval_cfg.custom.enabled:
        final_validation_failures.extend(
            _incomplete_validation_records("custom", final_release_evidence_validations)
        )

    rank_value = combined.loc[selected_model, ("__all__", "overall", "rank")]
    tuning_utility = combined.loc[selected_model, ("__all__", "utility", "U_tuning")]
    selection = {
        "source": "combined_evaluation.csv",
        "model": selected_model,
        "U_tuning": float(tuning_utility),
        # Retained as audit evidence; this value is not used for selection.
        "overall_rank": float(rank_value) if pd.notna(rank_value) else None,
    }
    gate_column = ("__all__", "privacy_gate", "pass")
    if gate_column in combined.columns:
        gate_value = combined.loc[selected_model, gate_column]
        selection["privacy_gate_pass"] = bool(gate_value) if pd.notna(gate_value) else False
    final_score_validations = {
        ("synthcity", "main"): final_synthcity_validations,
        **final_syntheval_validations,
        ("custom", "log_disparity"): final_custom_validations,
        ("custom", "release_evidence"): final_release_evidence_validations,
    }
    utility_evidence, privacy_evidence, fairness_evidence = _release_score_inputs(
        final_score_validations
    )
    release_score = compute_release_score(
        utility=utility_evidence,
        privacy=privacy_evidence,
        fairness=fairness_evidence,
        provenance={"model": selected_model, "evaluation_role": "final_holdout"},
    )
    (
        evidence_state,
        metric_completeness_state,
        score_completeness_state,
        audit_outcome_state,
    ) = _final_holdout_state_dimensions(final_validation_failures, release_score)
    evidence = {
        **base_evidence,
        "state": evidence_state,
        "selected_model": selected_model,
        "candidate_selection": selection,
        "release_score": release_score,
        "evidence_execution_state": "failed" if final_execution_failures else "succeeded",
        "metric_completeness_state": metric_completeness_state,
        "score_completeness_state": score_completeness_state,
        "audit_outcome_state": audit_outcome_state,
        "provenance_inventory": {
            "raw_values": {
                "frameworks": {
                    str(key): {model: validation.to_dict() for model, validation in value.items()}
                    for key, value in final_score_validations.items()
                },
            },
            "normalized_values": {
                "release_score": release_score,
            },
            "formulas": {
                "U_tuning": "(S_TSTR + S_MMD + S_JSD) / 3",
                "R_final": "0.45*U + 0.30*P + 0.25*F",
            },
            "anchors": {
                "source": "metric contract and release-score configuration",
                "release_score_status": release_score.get("status"),
                "required_components": release_score.get("required_components", []),
            },
            "role_hashes": {
                "imputed_evaluation": final_role_hashes,
                "custom_raw_evaluation": final_custom_role_hashes,
            },
            "release_transform_digest": release_transform_metadata["common_protocol_digest"],
            "attack_protocol": {
                "evaluation_role": "final_holdout",
                "privacy": "release-form final audit",
            },
            "seeds": {"evaluation": cfg.seed},
            "supports": {
                "synthetic_rows": len(released_final_synthetic),
                "final_holdout_rows": len(released_final_roles["final_holdout"]),
                "release_transform": release_transform_metadata,
            },
            "intervals": dict(dataset.release_generalization),
            "invalid_reasons": final_validation_failures,
            "selected_model_provenance": {
                "model": selected_model,
                "selection": selection,
                "fit_roles": ["train", "tuning"],
                "refit": refit_metadata,
            },
        },
        "final_refit": refit_metadata,
        "role_hashes": {
            "imputed_evaluation": final_role_hashes,
            "custom_raw_evaluation": final_custom_role_hashes,
        },
        "frameworks": {
            "synthcity": {
                "validation": _single_framework_validation_payload(final_synthcity_validations),
            },
            "syntheval": {
                "validation": _validation_payloads(final_syntheval_validations),
                "execution": {"main": final_syntheval_executions},
            },
            "custom": {
                "validation": _single_framework_validation_payload(final_custom_validations),
                "release_evidence_validation": _single_framework_validation_payload(
                    final_release_evidence_validations
                ),
                "summary": {
                    name: report.get("summary_stats", report)
                    for name, report in final_custom_reports.items()
                },
            },
        },
    }
    if final_validation_failures:
        evidence["failure_reasons"] = final_validation_failures
    return evidence


def _run_blocked_legacy_evaluation(
    cfg: Config,
    dataset: Dataset,
    synthetic_datasets: dict[str, pd.DataFrame],
    experiment=None,
) -> tuple[pd.DataFrame, dict]:
    """Persist an auditable blocked result without evaluating legacy roles."""
    eval_cfg = cfg.evaluation
    output_dir, attempt_metadata = artifacts.select_evaluation_attempt(
        eval_cfg.output_dir, experiment_id=getattr(experiment, "id", None)
    )
    output_dir = ensure_dir(output_dir)
    eval_cfg.output_dir = str(output_dir)
    attempt_metadata["source_generation_root"] = str(cfg.generation.output_dir)
    if experiment is not None:
        experiment.evaluation_dir = output_dir
        experiment.plots_dir = output_dir / "plots"
        cfg.plots.output_dir = str(experiment.plots_dir)
    requested_datasets = select_models(cfg, synthetic_datasets)
    legacy_semantic_context = semantic_context_payload(
        dataset,
        classification_score=eval_cfg.synthcity.classification_score,
        roles=("train", "final_holdout"),
    )
    logger.warning(
        "[evaluation] legacy_two_role dataset=%s cannot support candidate evaluation or "
        "ranking; writing blocked evidence for requested models=%s",
        dataset.name,
        sorted(requested_datasets),
    )
    group_context = syntheval_eval.build_group_context(
        dataset,
        {},
        group_mode=eval_cfg.group_mode,
        group_column=eval_cfg.group_column,
    )
    population_unit = group_context["population_unit"]
    group_mode = group_context["group_mode"]
    train_frame = dataset.role_frame("train", imputed=True)
    final_holdout_frame = dataset.role_frame("final_holdout", imputed=True)
    if train_frame is None or final_holdout_frame is None:
        raise RuntimeError(
            "Legacy evaluation requires populated imputed train and final_holdout roles"
        )
    role_hashes = {
        "train": dataframe_fingerprint(train_frame),
        "final_holdout": dataframe_fingerprint(final_holdout_frame),
    }
    role_context_roles = ("train", "final_holdout")
    role_context = {
        "candidate": role_context_payload(dataset, role_context_roles),
        "full": role_context_payload(dataset, role_context_roles),
    }
    role_context_fingerprints = {
        "candidate": role_context_fingerprint(dataset, role_context_roles),
        "full": role_context_fingerprint(dataset, role_context_roles),
    }
    empty_columns = pd.MultiIndex.from_arrays([[], [], []], names=["framework", "type", "metric"])
    combined = pd.DataFrame(index=pd.Index([], name="model"), columns=empty_columns)
    combined.to_csv(output_dir / "combined_evaluation.csv")
    metric_contract_manifest = DEFAULT_METRIC_CONTRACT_REGISTRY.manifest()
    source_provenance = artifacts.collect_source_provenance(
        config_path=cfg.config_path,
        metric_contract_digest=metric_contract_manifest["digest"],
    )
    final_holdout_evidence = _run_final_holdout_evidence(
        cfg,
        dataset,
        {},
        combined,
        group_context,
        population_unit,
        group_mode,
        role_context,
        role_context_fingerprints,
    )
    generator_metadata = _generation_metadata(cfg, list(requested_datasets))
    artifact_manifest = artifacts.persist_evaluation_artifacts(
        output_dir,
        combined,
        {},
        native_syntheval_plot_dir=None,
        synthcity_validation_results={},
        syntheval_validation_results={},
        custom_validation_results={},
        metric_contract_manifest=metric_contract_manifest,
        source_provenance=source_provenance,
        role_context=role_context,
        role_context_fingerprint=role_context_fingerprints,
        generator_metadata=generator_metadata,
        semantic_context=legacy_semantic_context,
        final_holdout_evidence=final_holdout_evidence,
        attempt_metadata=attempt_metadata,
    )
    extras = {
        "selected_datasets": {},
        "group_context": group_context,
        "population_unit": population_unit,
        "group_mode": group_mode,
        "role_hashes": role_hashes,
        "candidate_role_hashes": {"train": role_hashes["train"]},
        "synthcity_results": {},
        "synthcity_validation": {},
        "syntheval_validation": {},
        "custom_validation": {},
        "syntheval_benchmark_results": None,
        "syntheval_benchmark_ranks": None,
        "syntheval_execution": {},
        "log_disparity_reports": {},
        "privacy_gate_result": None,
        "final_holdout_evidence": final_holdout_evidence,
        "artifact_manifest": str(artifact_manifest),
    }
    if eval_cfg.generate_report:
        try:
            report.save_evaluation_report(cfg, dataset, combined, extras, experiment)
        except ValueError as exc:
            # Legacy bundles intentionally lack canonical tuning provenance;
            # retain durable blocked artifacts instead of inventing report context.
            logger.warning(
                "[evaluation] blocked legacy report omitted; "
                "reason_code=legacy_report_context_missing exception_type=%s",
                type(exc).__name__,
            )
            (output_dir / "report.md").write_text(
                "# Evaluation report\n\nStatus: blocked\n\n"
                "Legacy two-role dataset has no canonical tuning evidence.\n"
            )
        else:
            extras["report_path"] = "report.md"
    return combined, extras


def run_evaluation(
    cfg: Config,
    dataset: Dataset,
    synthetic_datasets: dict[str, pd.DataFrame],
    experiment=None,
) -> tuple[pd.DataFrame, dict]:
    """Run the full evaluation stage.

    Returns ``(combined_table, extras)`` where ``combined_table`` is the single
    ranked, multi-index DataFrame (see :mod:`synthdata.evaluation.combine`) and
    ``extras`` holds the raw per-framework results (useful for plotting).

    When ``cfg.evaluation.save_per_model_syntheval_plots`` is enabled,
    SynthEval's native per-metric plots are always produced as a side effect
    of the single benchmark pass below. Callers must NOT run a second
    evaluation merely to obtain those diagnostics.

    ``experiment`` (optional :class:`synthdata.experiment.Experiment`) is only used to
    include the experiment id in the generated Markdown report (see
    ``cfg.evaluation.generate_report``); it has no effect on any other stage output.
    """
    if dataset.legacy_two_role:
        return _run_blocked_legacy_evaluation(cfg, dataset, synthetic_datasets, experiment)

    eval_cfg = cfg.evaluation
    output_dir, attempt_metadata = artifacts.select_evaluation_attempt(
        eval_cfg.output_dir, experiment_id=getattr(experiment, "id", None)
    )
    output_dir = ensure_dir(output_dir)
    eval_cfg.output_dir = str(output_dir)
    attempt_metadata["source_generation_root"] = str(cfg.generation.output_dir)
    if experiment is not None:
        experiment.evaluation_dir = output_dir
        experiment.plots_dir = output_dir / "plots"
        cfg.plots.output_dir = str(experiment.plots_dir)
    synthcity_semantics = _synthcity_semantic_context(dataset, eval_cfg.synthcity)

    generation_inventory = _load_generation_inventory(experiment)
    selected_datasets = select_models(cfg, synthetic_datasets, generation_inventory)
    model_names = sorted(selected_datasets)
    requested_models = (
        list(eval_cfg.models)
        if eval_cfg.models
        else list(generation_inventory.expected_outputs)
        if generation_inventory is not None
        else list(model_names)
    )
    evaluation_coverage = _evaluation_coverage(
        requested_models, selected_datasets, generation_inventory
    )
    if evaluation_coverage is not None:
        attempt_metadata["evaluation_coverage"] = evaluation_coverage
        logger.warning(
            "[evaluation] partial model coverage experiment=%s dataset=%s@%s "
            "evaluated=%s failed_outputs=%s",
            getattr(experiment, "id", None),
            dataset.name,
            dataset.version,
            model_names,
            evaluation_coverage["failed_outputs"],
        )
    logger.info("Evaluating %d models: %s", len(model_names), model_names)
    train_frame, tuning_frame, final_holdout_frame = _candidate_role_frames(dataset)
    group_context = syntheval_eval.build_group_context(
        dataset,
        selected_datasets,
        group_mode=eval_cfg.group_mode,
        group_column=eval_cfg.group_column,
    )
    population_unit = group_context["population_unit"]
    group_mode = group_context["group_mode"]
    group_configuration = {"group_context": group_context}

    synthcity_results = synthcity_eval.run_synthcity_evaluation(
        selected_datasets,
        train_frame,
        tuning_frame,
        dataset.target_column,
        synthcity_semantics["protected_columns"],
        eval_cfg.synthcity,
        n_samples=cfg.generation.n_samples,
        seed=cfg.seed,
        workspace=str(output_dir / "synthcity_workspace"),
        real_reference_group_ids=dataset.role_groups.get("tuning"),
        real_train_group_ids=dataset.role_groups.get("train"),
        sensitive_target_types=synthcity_semantics["sensitive_target_types"],
        feature_types=synthcity_semantics["feature_types"],
        source_table=synthcity_semantics["source_table"],
        semantic_context=synthcity_semantics,
        quasi_identifier_columns=synthcity_semantics["quasi_identifier_columns"],
        task_type="classification" if dataset.target_is_categorical else "regression",
        classification_score=synthcity_semantics["classification_score"],
        group_mode=group_mode,
    )
    synthcity_metric_config = synthcity_eval.resolve_metric_config(eval_cfg.synthcity)
    candidate_role_hashes = {
        "train": dataframe_fingerprint(train_frame),
        "tuning": dataframe_fingerprint(tuning_frame),
    }
    raw_train_frame = dataset.role_frame("train", imputed=False)
    raw_tuning_frame = dataset.role_frame("tuning", imputed=False)
    if raw_train_frame is None or raw_tuning_frame is None:
        raise RuntimeError("Evaluation requires populated raw train and tuning role frames")
    raw_candidate_role_hashes = {
        "train": dataframe_fingerprint(raw_train_frame),
        "tuning": dataframe_fingerprint(raw_tuning_frame),
    }
    role_hashes = {
        **candidate_role_hashes,
        "final_holdout": dataframe_fingerprint(final_holdout_frame),
    }
    if dataset.legacy_two_role:
        role_hashes["test"] = role_hashes["tuning"]
    synthcity_validation_results = synthcity_eval.validate_synthcity_results(
        synthcity_results,
        synthcity_metric_config,
        role_hashes=candidate_role_hashes,
        requested_use="audit",
        variable_columns=list(tuning_frame.columns),
        attack_target_types=_synthcity_attack_target_types(dataset, eval_cfg.synthcity),
        context=MetricEvaluationContext(
            role_hashes=candidate_role_hashes,
            population_unit=population_unit,
            group_mode=group_mode,
            resolved_configuration={
                "metric_config": synthcity_metric_config,
                "fit_roles": ["train"],
                "quasi_identifier_columns": synthcity_semantics["quasi_identifier_columns"],
                "sensitive_target_types": synthcity_semantics["sensitive_target_types"],
                "feature_types": synthcity_semantics["feature_types"],
                "source_table": synthcity_semantics["source_table"],
                "structural_n_clusters": list(eval_cfg.synthcity.structural_n_clusters),
                "structural_min_rows_per_cluster": eval_cfg.synthcity.structural_min_rows_per_cluster,
                **group_configuration,
            },
        ),
    )
    logger.info(
        "[synthcity contracts] validated %d model result set(s): %s",
        len(synthcity_validation_results),
        {
            name: validation.status_counts
            for name, validation in synthcity_validation_results.items()
        },
    )

    # Native SynthEval diagnostics can only be created during SynthEval's
    # metric pass. Always produce them when this evaluation config enables
    # them, independent of the CLI's broader --plot switch, so a later
    # synthdata-plot run never needs to rerun evaluation merely to obtain
    # these diagnostics.
    want_syntheval_plots = eval_cfg.syntheval.enabled and eval_cfg.save_per_model_syntheval_plots
    plots_output_dir = (
        Path(cfg.plots.output_dir) / "evaluation" / "syntheval_plots"
        if want_syntheval_plots
        else None
    )
    syntheval_output = (None, None, {})
    if eval_cfg.syntheval.enabled:
        syntheval_output = syntheval_eval.run_syntheval_evaluation(
            selected_datasets,
            dataset,
            eval_cfg.syntheval,
            preset_dir=output_dir,
            ranking_strategy=eval_cfg.ranking_strategy,
            output_folder=output_dir / "syntheval_benchmark",
            plots_output_dir=plots_output_dir,
            positive_class=eval_cfg.positive_class,
            execution_cfg=eval_cfg.syntheval_execution,
            return_execution=True,
            group_context=group_context,
            semantic_context=synthcity_semantics,
        )
    if len(syntheval_output) == 3:
        benchmark_results, benchmark_ranks, syntheval_executions = syntheval_output
    else:
        benchmark_results, benchmark_ranks = syntheval_output
        syntheval_executions = {}
    syntheval_validations = {}
    if eval_cfg.syntheval.enabled:
        target_is_binary, target_is_multiclass = _syntheval_target_type_flags(dataset, train_frame)
        syntheval_preset = syntheval_eval.build_preset(
            eval_cfg.syntheval,
            positive_class=eval_cfg.positive_class,
            target_is_binary=target_is_binary,
            target_is_multiclass=target_is_multiclass,
        )
        syntheval_manifest = syntheval_execution_manifest(
            syntheval_preset,
            include_holdout_outputs=tuning_frame is not None,
            target_columns=[dataset.target_column],
            protected_columns=dataset.protected_columns,
            target_is_binary=False if target_is_multiclass else target_is_binary,
        )
        syntheval_expected_keys = syntheval_eval.extend_syntheval_expected_diagnostics(
            syntheval_execution_keys_by_framework(syntheval_manifest),
            benchmark_results,
            structured_executions=syntheval_executions,
        )
        syntheval_validations = syntheval_eval.validate_syntheval_results(
            benchmark_results,
            benchmark_ranks,
            syntheval_expected_keys,
            role_hashes=candidate_role_hashes,
            role_hashes_by_framework={
                "syntheval": candidate_role_hashes,
                "custom": raw_candidate_role_hashes,
            },
            model_names=model_names,
            requested_use="audit",
            structured_executions=syntheval_executions,
            population_unit=population_unit,
            group_mode=group_mode,
            resolved_configuration=group_configuration,
        )

    log_disparity_reports = (
        custom_eval.run_log_disparity_evaluation(
            selected_datasets,
            dataset,
            eval_cfg.log_disparity,
            eval_cfg.custom,
            evaluation_role="tuning",
        )
        if eval_cfg.custom.enabled
        else {}
    )
    custom_validations = (
        custom_eval.validate_log_disparity_results(
            log_disparity_reports,
            model_names,
            role_hashes=dict(raw_candidate_role_hashes),
            expected_role_hashes=dict(raw_candidate_role_hashes),
            requested_use="audit",
            population_unit=population_unit,
            group_mode=group_mode,
            resolved_configuration={
                **group_configuration,
                "input_representation": "raw",
            },
        )
        if log_disparity_reports
        else None
    )
    release_evidence_observations = {}
    if eval_cfg.custom.enabled:
        release_evidence_observations = release_evidence_eval.run_release_evidence_evaluation(
            selected_datasets,
            dataset,
            evaluation_role="tuning",
            generalization=dataset.release_generalization,
            quasi_identifiers=list(dataset.quasi_identifier_columns),
            sensitive_fields=list(dataset.sensitive_columns),
            protected_columns=list(dataset.protected_columns),
            role_hashes=dict(raw_candidate_role_hashes),
            seed=cfg.seed,
        )
    candidate_release_digest = None
    if release_evidence_observations:
        first_observations = next(iter(release_evidence_observations.values()), ())
        for observation in first_observations:
            if observation.emitted_key == "release_privacy.v1":
                candidate_release_digest = observation.result_metadata.get(
                    "release_transform_digest"
                )
                break
    release_evidence_validations = {}
    if eval_cfg.custom.enabled:
        release_evidence_validations = release_evidence_eval.validate_release_evidence_results(
            release_evidence_observations,
            role_hashes=dict(raw_candidate_role_hashes),
            evaluation_role="tuning",
            population_unit=population_unit,
            group_mode=group_mode,
            requested_use="audit",
        )
    if candidate_release_digest is not None:
        release_role_hashes = dict(raw_candidate_role_hashes)
        release_role_hashes["__release_transform_digest__"] = candidate_release_digest
        release_evidence_validations = release_evidence_eval.validate_release_evidence_results(
            release_evidence_observations,
            role_hashes=release_role_hashes,
            evaluation_role="tuning",
            population_unit=population_unit,
            group_mode=group_mode,
            requested_use="audit",
        )
    metric_execution_passes = syntheval_eval.build_metric_execution_passes(syntheval_validations)

    combined = combine.build_combined_table(
        synthcity_results,
        benchmark_results,
        benchmark_ranks,
        log_disparity_reports,
        model_names,
        rank_weights=eval_cfg.rank_weights,
        synthcity_validations=synthcity_validation_results,
        syntheval_validations=syntheval_validations,
        metric_execution_passes=metric_execution_passes,
        custom_validations=custom_validations,
        release_evidence_validations=release_evidence_validations,
    )

    gate_validation_results = {
        ("synthcity", "main"): synthcity_validation_results,
        **syntheval_validations,
    }
    if custom_validations is not None:
        gate_validation_results[("custom", "main")] = custom_validations
    if release_evidence_validations:
        gate_validation_results[("custom", "main")] = release_evidence_validations
    gate_result = privacy_gate.evaluate_privacy_gate(
        combined,
        eval_cfg.privacy_gate,
        registry=DEFAULT_METRIC_CONTRACT_REGISTRY,
        validation_results=gate_validation_results,
        execution_passes=cast(Mapping[tuple[str, ...], str], metric_execution_passes),
    )
    combined = privacy_gate.merge_privacy_gate_results(combined, gate_result)

    combined.to_csv(output_dir / "combined_evaluation.csv")

    metric_contract_manifest = DEFAULT_METRIC_CONTRACT_REGISTRY.manifest()
    context_roles = (
        ("train", "tuning") if dataset.has_canonical_roles else ("train", "final_holdout")
    )
    full_context_roles = (
        ("train", "tuning", "final_holdout")
        if dataset.has_canonical_roles
        else ("train", "final_holdout")
    )
    role_context = {
        "candidate": role_context_payload(dataset, context_roles),
        "full": role_context_payload(dataset, full_context_roles),
    }
    role_context_fingerprints = {
        "candidate": role_context_fingerprint(dataset, context_roles),
        "full": role_context_fingerprint(dataset, full_context_roles),
    }
    source_provenance = artifacts.collect_source_provenance(
        config_path=cfg.config_path,
        metric_contract_digest=metric_contract_manifest["digest"],
    )
    final_holdout_evidence = _run_final_holdout_evidence(
        cfg,
        dataset,
        selected_datasets,
        combined,
        group_context,
        population_unit,
        group_mode,
        role_context,
        role_context_fingerprints,
    )
    generator_metadata = _generation_metadata(cfg, model_names)
    release_score_evidence = (
        {final_holdout_evidence["selected_model"]: final_holdout_evidence["release_score"]}
        if final_holdout_evidence.get("release_score") is not None
        else None
    )
    syntheval_execution_artifacts = {}
    if syntheval_executions:
        syntheval_execution_artifacts[("syntheval", "main")] = syntheval_executions
    artifact_manifest = artifacts.persist_evaluation_artifacts(
        output_dir,
        combined,
        log_disparity_reports,
        native_syntheval_plot_dir=plots_output_dir,
        synthcity_validation_results=synthcity_validation_results,
        syntheval_validation_results=syntheval_validations,
        syntheval_execution_results=syntheval_execution_artifacts or None,
        custom_validation_results=custom_validations,
        metric_contract_manifest=metric_contract_manifest,
        source_provenance=source_provenance,
        role_context=role_context,
        role_context_fingerprint=role_context_fingerprints,
        generator_metadata=generator_metadata,
        semantic_context=synthcity_semantics,
        final_holdout_evidence=final_holdout_evidence,
        release_score_evidence=release_score_evidence,
        attempt_metadata=attempt_metadata,
    )

    extras = {
        "selected_datasets": selected_datasets,
        "group_context": group_context,
        "semantic_context": synthcity_semantics,
        "population_unit": population_unit,
        "group_mode": group_mode,
        "role_hashes": role_hashes,
        "candidate_role_hashes": candidate_role_hashes,
        "role_context": role_context,
        "role_context_fingerprint": role_context_fingerprints,
        "generator_metadata": generator_metadata,
        "synthcity_results": synthcity_results,
        "synthcity_validation": {
            name: validation.to_dict() for name, validation in synthcity_validation_results.items()
        },
        "syntheval_validation": {
            f"{framework}:{execution_pass}": {
                name: validation.to_dict() for name, validation in validations.items()
            }
            for (framework, execution_pass), validations in syntheval_validations.items()
        },
        "custom_validation": {
            name: validation.to_dict() for name, validation in (custom_validations or {}).items()
        },
        "release_evidence_validation": {
            name: validation.to_dict() for name, validation in release_evidence_validations.items()
        },
        "syntheval_benchmark_results": benchmark_results,
        "syntheval_benchmark_ranks": benchmark_ranks,
        "syntheval_execution": syntheval_execution_artifacts,
        "log_disparity_reports": log_disparity_reports,
        "privacy_gate_result": gate_result,
        "final_holdout_evidence": final_holdout_evidence,
        "artifact_manifest": str(artifact_manifest),
    }
    if evaluation_coverage is not None:
        extras["evaluation_coverage"] = evaluation_coverage

    if eval_cfg.generate_report:
        report_path = report.save_evaluation_report(cfg, dataset, combined, extras, experiment)
        extras["report_path"] = str(report_path)

    return combined, extras
