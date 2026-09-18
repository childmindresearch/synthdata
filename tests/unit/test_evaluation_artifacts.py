"""Tests for persisted evaluation artifacts used by artifact-only plotting."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
import pytest

from synthdata.data import role_context_payload, semantic_context_digest, semantic_context_payload
from synthdata.evaluation import artifacts
from synthdata.evaluation.artifacts import (
    artifact_bundle_dir,
    collect_source_provenance,
    load_custom_metric_status,
    load_final_holdout_evidence,
    load_log_disparity_reports,
    load_metric_contract_manifest,
    load_release_score_evidence,
    load_synthcity_metric_status,
    load_syntheval_execution,
    load_syntheval_metric_status,
    persist_evaluation_artifacts,
    select_evaluation_attempt,
    validate_evaluation_bundle,
    verify_native_syntheval_artifacts,
)
from synthdata.evaluation.metric_contracts import (
    DEFAULT_METRIC_CONTRACT_REGISTRY,
    MetricContractRegistry,
    MetricEvaluationContext,
    MetricObservation,
    MetricValidationResult,
    resolve_metric_observations,
)
from synthdata.evaluation.synthcity_eval import validate_synthcity_results
from synthdata.evaluation.syntheval_eval import (
    _failed_execution_payload,
    validate_syntheval_results,
)
from synthdata.log_disparity.metric_log_disparity import build_log_disparity_report_figure

pytestmark = pytest.mark.unit


def test_evaluation_attempt_preserves_existing_evidence(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    historical = evaluation_dir / "combined_evaluation.csv"
    historical.write_bytes(b"historical failed evidence\n")

    attempt, metadata = select_evaluation_attempt(evaluation_dir, experiment_id="exp-1")

    assert attempt.parent == evaluation_dir / "attempts"
    assert attempt != evaluation_dir
    assert not historical.is_symlink()
    assert historical.read_bytes() == b"historical failed evidence\n"
    assert metadata["attempt_id"].startswith("eval_")
    assert metadata["source_experiment_id"] == "exp-1"
    assert metadata["prior_attempt"] == str(evaluation_dir)


def test_first_evaluation_attempt_keeps_canonical_layout(tmp_path):
    attempt, metadata = select_evaluation_attempt(tmp_path / "evaluation")

    assert attempt == tmp_path / "evaluation"
    assert metadata["attempt_id"] == "canonical"
    assert metadata["prior_attempt"] is None


def _dataframe(
    data: Mapping[object, Sequence[object]] | Sequence[Sequence[object]] | None = None,
    *,
    index: Sequence[object] | pd.Index | None = None,
    columns: Sequence[object] | pd.Index | None = None,
) -> pd.DataFrame:
    """Construct test frames at pandas' dynamically typed boundary."""
    normalized_index = index if isinstance(index, pd.Index) else pd.Index(index) if index else None
    normalized_columns = (
        columns if isinstance(columns, pd.Index) else pd.Index(columns) if columns else None
    )
    return pd.DataFrame(data, index=normalized_index, columns=normalized_columns)


def _combined() -> pd.DataFrame:
    frame = _dataframe(index=pd.Index(["model_a"]))
    frame[("__all__", "utility", "rank")] = [0.5]
    frame[("__all__", "privacy", "rank")] = [0.5]
    frame[("__all__", "fairness", "rank")] = [0.5]
    frame[("__all__", "overall", "rank")] = [0.5]
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    return frame


def _complete_final_evidence_fields() -> dict:
    return {
        "evidence_execution_state": "succeeded",
        "metric_completeness_state": "complete",
        "score_completeness_state": "complete",
        "audit_outcome_state": "complete",
        "selected_model": "model_a",
        "role_context_fingerprint": "3" * 64,
        "common_protocol_digest": "4" * 64,
        "candidate_selection": {"source": "combined_evaluation.csv", "model": "model_a"},
        "provenance_inventory": {
            field: (
                {
                    "imputed_evaluation": {
                        "train": "a" * 64,
                        "tuning": "b" * 64,
                        "final_holdout": "c" * 64,
                        "refit_fit": "d" * 64,
                    },
                    "custom_raw_evaluation": {
                        "train": "e" * 64,
                        "tuning": "f" * 64,
                        "final_holdout": "0" * 64,
                        "refit_fit": "1" * 64,
                    },
                }
                if field == "role_hashes"
                else "2" * 64
                if field == "release_transform_digest"
                else {"model": "model_a"}
                if field == "selected_model_provenance"
                else {"evidence": field}
            )
            for field in artifacts._PROVENANCE_INVENTORY_FIELDS
        },
    }


def _contract_manifest() -> dict:
    return MetricContractRegistry(()).manifest()


def _role_context(role_names: list[str]) -> dict:
    return {
        "schema_version": "role-context-v1",
        "dataset_name": "testds",
        "dataset_version": "v1",
        "roles": {
            role: {
                "raw_fingerprint": f"{role}-raw",
                "imputed_fingerprint": f"{role}-imputed",
                "rows": 1,
            }
            for role in role_names
        },
        "assignment_fingerprint": None,
        "assignment_policy_fingerprint": None,
        "semantic_fingerprint": "semantic-fingerprint",
        "variable_schema_fingerprint": "schema-fingerprint",
        "compatibility_mode": None,
    }


def _semantic_context() -> dict:
    return {
        "schema_version": "semantic-context-v1",
        "dataset_name": "testds",
        "dataset_version": "v1",
        "target_column": "target",
        "task_type": "classification",
        "feature_columns": ["feature"],
        "protected_columns": ["protected"],
        "quasi_identifier_columns": ["feature"],
        "feature_types": {
            "feature": "continuous",
            "protected": "categorical",
            "target": "categorical",
        },
        "source_table": {"feature": "measurements"},
    }


def _validation_payload(model_name: str, contract_digest: str = "digest-1") -> dict:
    return MetricValidationResult(
        model_name=model_name,
        requested_use="policy_rank",
        contract_digest=contract_digest,
        records=(),
        evaluation_context=MetricEvaluationContext(),
    ).to_dict()


def _context_validation_payload(
    model_name: str,
    contract_digest: str,
    role_hashes: dict,
    *,
    execution_pass: str = "main",
    target_view: str = "native",
) -> dict:
    return MetricValidationResult(
        model_name=model_name,
        requested_use="policy_rank",
        contract_digest=contract_digest,
        records=(),
        evaluation_context=MetricEvaluationContext(
            execution_pass=execution_pass,
            target_view=target_view,
            role_hashes=role_hashes,
            evaluation_role="tuning",
            population_unit="row",
            group_mode="row",
        ),
    ).to_dict()


def test_mixed_framework_statuses_are_partitioned_and_bundle_validated(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    combined = _combined()
    combined.to_csv(evaluation_dir / "combined_evaluation.csv")
    contract_manifest = DEFAULT_METRIC_CONTRACT_REGISTRY.manifest()
    imputed_role_hashes = {"train": "train-imputed", "tuning": "tuning-imputed"}
    raw_role_hashes = {"train": "train-raw", "tuning": "tuning-raw"}
    syntheval = _context_validation_payload(
        "model_a", contract_manifest["digest"], imputed_role_hashes
    )
    binary = _context_validation_payload(
        "model_a",
        contract_manifest["digest"],
        imputed_role_hashes,
        execution_pass="binary_target",
        target_view="binary_collapsed",
    )
    custom = _context_validation_payload("model_a", contract_manifest["digest"], raw_role_hashes)
    custom_binary = _context_validation_payload(
        "model_a",
        contract_manifest["digest"],
        raw_role_hashes,
        execution_pass="binary_target",
        target_view="binary_collapsed",
    )

    persist_evaluation_artifacts(
        evaluation_dir,
        combined,
        {},
        native_syntheval_plot_dir=None,
        syntheval_validation_results={
            ("syntheval", "main"): {"model_a": syntheval},
            ("syntheval", "binary_target"): {"model_a": binary},
            ("custom", "main"): {"model_a": custom},
            ("custom", "binary_target"): {"model_a": custom_binary},
        },
        custom_validation_results={
            ("custom", "main"): {"model_a": custom},
            ("custom", "binary_target"): {"model_a": custom_binary},
        },
        metric_contract_manifest=contract_manifest,
        role_context={
            "candidate": {
                **_role_context(["train", "tuning"]),
                "roles": {
                    role: {
                        **details,
                        "raw_fingerprint": raw_role_hashes[role],
                    }
                    for role, details in _role_context(["train", "tuning"])["roles"].items()
                },
            }
        },
        role_context_fingerprint={"candidate": "context"},
    )

    syntheval_status = load_syntheval_metric_status(evaluation_dir)
    assert set(syntheval_status["passes"]) == {"syntheval:main", "syntheval:binary_target"}
    assert load_custom_metric_status(evaluation_dir)["passes"] == {
        "custom:main": {"model_a": custom},
        "custom:binary_target": {"model_a": custom_binary},
    }
    validate_evaluation_bundle(evaluation_dir)

    status_path = artifact_bundle_dir(evaluation_dir) / "syntheval_metric_status.json"
    payload = json.loads(status_path.read_text())
    payload["passes"]["custom:main"] = payload["passes"]["syntheval:main"]
    status_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _refresh_manifest_digest(evaluation_dir, "syntheval_metric_status")
    with pytest.raises(ValueError, match="Invalid syntheval execution pass identity"):
        load_syntheval_metric_status(evaluation_dir)


def test_custom_role_hash_mismatch_fails_before_bundle_sidecar_write(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    combined = _combined()
    combined.to_csv(evaluation_dir / "combined_evaluation.csv")
    manifest = DEFAULT_METRIC_CONTRACT_REGISTRY.manifest()
    mismatched = _context_validation_payload(
        "model_a", manifest["digest"], {"train": "imputed", "tuning": "imputed"}
    )
    with pytest.raises(ValueError, match="declared raw role map"):
        persist_evaluation_artifacts(
            evaluation_dir,
            combined,
            {},
            native_syntheval_plot_dir=None,
            custom_validation_results={("custom", "main"): {"model_a": mismatched}},
            metric_contract_manifest=manifest,
            role_context={"candidate": _role_context(["train", "tuning"])},
        )
    assert not (artifact_bundle_dir(evaluation_dir) / "custom_metric_status.json").exists()


def test_custom_extra_role_hash_fails_before_bundle_sidecar_write(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    combined = _combined()
    combined.to_csv(evaluation_dir / "combined_evaluation.csv")
    manifest = DEFAULT_METRIC_CONTRACT_REGISTRY.manifest()
    extra_hash = _context_validation_payload(
        "model_a",
        manifest["digest"],
        {"train": "train-raw", "tuning": "tuning-raw", "__release_transform_digest__": "digest"},
    )

    with pytest.raises(ValueError, match="declared raw role map"):
        persist_evaluation_artifacts(
            evaluation_dir,
            combined,
            {},
            native_syntheval_plot_dir=None,
            custom_validation_results={("custom", "main"): {"model_a": extra_hash}},
            metric_contract_manifest=manifest,
            role_context={"candidate": _role_context(["train", "tuning"])},
        )
    assert not (artifact_bundle_dir(evaluation_dir) / "custom_metric_status.json").exists()


def _execution_payload(model_name: str) -> dict:
    return {
        "model_name": model_name,
        "context_fingerprint": "context-digest",
        "schema_version": "syntheval-execution-v1",
        "pass_id": "main",
        "target_view": "native",
        "expected_manifest_digest": "execution-digest",
        "execution_complete": True,
        "execution_succeeded": True,
        "policy_eligible": False,
        "metric_executions": [
            {
                "method": "statistics",
                "status": {
                    "method": "statistics",
                    "state": "succeeded",
                    "expected_keys": ["corr_mat_diff_v2"],
                    "observed_keys": ["corr_mat_diff_v2"],
                    "completed_keys": ["corr_mat_diff_v2"],
                    "failed_keys": [],
                    "missing_keys": [],
                    "duplicate_keys": [],
                    "non_finite_keys": [],
                    "unexpected_keys": [],
                    "warnings": [],
                    "exception_type": None,
                    "exception_message": None,
                },
                "normalized_rows": [],
                "normalized_rows_v2": [
                    {
                        "metric": "corr_mat_diff_v2",
                        "dim": "u",
                        "val": 0.1,
                        "err": 0.0,
                        "n_val": 0.2,
                        "n_err": 0.0,
                        "metric_version": "v2",
                        "raw_value": 0.1,
                        "normalized_value": 0.2,
                        "metadata": {"valid_pairs": 1, "source": "execution"},
                        "result_metadata": {"valid_pairs": 1, "source": "execution"},
                    }
                ],
            }
        ],
    }


def test_failed_binary_unknown_category_execution_round_trips_strictly(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    semantic_context = {
        **_semantic_context(),
        "target_view": "binary_collapsed",
        "binary_target_context": {
            "column": "target",
            "positive_classes": [1],
            "negative_classes": [0],
            "encoding": {"positive": 1, "negative": 0},
        },
    }
    payload = _failed_execution_payload(
        model_name="model_a",
        pass_name="binary_target",
        target_view="binary_collapsed",
        expected_manifest_digest="manifest-hash",
        expected_output_manifest={"auroc_diff": ("auroc_diff",)},
        context_fingerprint="context-hash",
        role_context=None,
        group_context=None,
        semantic_context=semantic_context,
        failure_status={
            "exception_type": "ValueError",
            "reason_code": "synthetic_binary_target_invalid",
        },
    )

    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {},
        native_syntheval_plot_dir=None,
        syntheval_validation_results={
            ("syntheval", "binary_target"): {
                "model_a": _context_validation_payload(
                    "model_a",
                    DEFAULT_METRIC_CONTRACT_REGISTRY.digest(),
                    {"train": "train-imputed", "tuning": "tuning-imputed"},
                    execution_pass="binary_target",
                    target_view="binary_collapsed",
                )
            }
        },
        syntheval_execution_results={("syntheval", "binary_target"): {"model_a": payload}},
        metric_contract_manifest=DEFAULT_METRIC_CONTRACT_REGISTRY.manifest(),
        role_context={"candidate": _role_context(["train", "tuning"])},
        role_context_fingerprint={
            "candidate": artifacts._mapping_digest(_role_context(["train", "tuning"]))
        },
        semantic_context=_semantic_context(),
    )

    loaded = load_syntheval_execution(evaluation_dir)
    loaded_payload = loaded["passes"]["syntheval:binary_target"]["model_a"]
    assert loaded_payload["execution_succeeded"] is False
    assert loaded_payload["worker_exit"]["reason_code"] == "synthetic_binary_target_invalid"
    assert loaded_payload["semantic_context"] == semantic_context
    assert loaded_payload["semantic_context_digest"] == semantic_context_digest(semantic_context)
    validate_evaluation_bundle(evaluation_dir)


def _refresh_manifest_digest(evaluation_dir, manifest_key: str) -> None:
    manifest_path = artifact_bundle_dir(evaluation_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest[manifest_key]
    artifact_path = evaluation_dir / entry["path"]
    entry["sha256"] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _final_refit_files(tmp_path) -> dict:
    refit_dir = tmp_path / "final_refit"
    refit_dir.mkdir()
    data_path = refit_dir / "model_a.csv"
    metadata_path = refit_dir / "model_a.cache.json"
    generator_metadata = {
        "schema_version": "generator-metadata-v1",
        "generator_context": {"privacy_claim_type": "none"},
        "plugin_name": "test_generator",
        "plugin_fqdn": "test.generator",
        "requested_parameters": {},
        "n_samples": 1,
        "random_state": 0,
        "privacy_accounting": None,
    }
    data_path.write_text("feature,target\n1,0\n")
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": "final-refit-v1",
                "cache_key": "refit-key",
                "generator_metadata": generator_metadata,
            }
        )
    )
    return {
        "path": str(data_path),
        "metadata_path": str(metadata_path),
        "cache_key": "refit-key",
        "fit_roles": ["train", "tuning"],
        "generator_metadata": generator_metadata,
        "fit_frame_fingerprint": "2" * 64,
        "fit_frame_fingerprints": {"imputed": "2" * 64, "raw": "3" * 64},
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "model_name": "model_a",
    }


def _succeeded_release_score(score: float = 0.8) -> dict:
    dimensions = {
        "utility": ("tstr", "mmd", "jsd"),
        "privacy": ("k", "l", "dcr", "epsilon", "mia", "attribute"),
        "fairness": ("representation", "eo", "worst_log_disparity"),
    }
    return {
        "model_a": {
            "status": "succeeded",
            "score": score,
            "audit_only": True,
            "dimensions": {
                name: {
                    "score": score,
                    "components": {
                        component: {"score": score, "status": "succeeded"}
                        for component in components
                    },
                }
                for name, components in dimensions.items()
            },
        }
    }


def _current_generation_cache_files(tmp_path) -> tuple[dict, dict]:
    cache_dir = tmp_path / "generation"
    cache_dir.mkdir()
    data_path = cache_dir / "model_a.csv"
    metadata_path = cache_dir / "model_a.cache.json"
    data_path.write_text("feature,target\n1,0\n")
    role_context = _role_context(["train", "tuning"])
    fit_context = _role_context(["train"])
    generator_metadata = {
        "schema_version": "generator-metadata-v2",
        "generator_context": {"privacy_claim_type": "none"},
        "plugin_name": "model_a",
        "plugin_fqdn": "test.generator",
        "requested_parameters": {"temperature": 0.1},
        "n_samples": 1,
        "random_state": 0,
        "privacy_accounting": None,
        "implementation_fingerprint": "b" * 64,
    }
    cache_identity = {
        "schema_version": "generation-cache-v3",
        "model_name": "model_a",
        "role_context_fingerprint": artifacts._mapping_digest(role_context),
        "role_context": role_context,
        "fit_context_fingerprint": artifacts._mapping_digest(fit_context),
        "fit_context": fit_context,
        "variable_schema_fingerprint": "c" * 64,
        "semantic_context": _semantic_context(),
        "semantic_context_digest": semantic_context_digest(_semantic_context()),
        "columns": ["feature", "target"],
        "task_type": "classification",
        "target_view": "native",
        "n_samples": 1,
        "seed": 0,
        "device": "cpu",
        "registry_digest": DEFAULT_METRIC_CONTRACT_REGISTRY.digest(),
        "hpo_context_schema_version": None,
        "hpo_context_digest": None,
        "hpo_context": None,
        "resolved_parameters": {"temperature": 0.1},
        "generator_metadata_schema_version": "generator-metadata-v2",
        "generator_context": {"privacy_claim_type": "none"},
        "implementation_fingerprint": "b" * 64,
    }
    cache_metadata = {
        **cache_identity,
        "cache_key": artifacts._mapping_digest(cache_identity),
        "row_count": 1,
        "synthetic_data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "generator_metadata": generator_metadata,
    }
    metadata_path.write_text(json.dumps(cache_metadata, indent=2, sort_keys=True))
    entry = {
        "state": "present",
        "metadata_path": str(metadata_path),
        "data_path": str(data_path),
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "metadata": generator_metadata,
        "cache_metadata": cache_metadata,
    }
    return entry, role_context


def _write_valid_generated_artifact(
    tmp_path, dataset, model_name="ctgan", roles=("train", "tuning")
):
    generation_dir = tmp_path / "generated"
    generation_dir.mkdir()
    frame = dataset.full_df.iloc[:1].copy()
    data_path = generation_dir / f"{model_name}.csv"
    data_path.write_text(frame.to_csv(index=False))
    role_context = role_context_payload(dataset, roles)
    fit_context = role_context_payload(dataset, ("train",))
    semantic_context = semantic_context_payload(
        dataset,
        classification_score="balanced_accuracy",
        roles=roles,
    )
    generator_metadata = {
        "schema_version": "generator-metadata-v2",
        "generator_context": {"privacy_claim_type": "none"},
        "plugin_name": model_name,
        "plugin_fqdn": f"synthcity.{model_name}",
        "requested_parameters": {},
        "n_samples": len(frame),
        "random_state": 0,
        "privacy_accounting": None,
        "implementation_fingerprint": "a" * 64,
    }
    identity = {
        "schema_version": "generation-cache-v3",
        "model_name": model_name,
        "role_context_fingerprint": artifacts._mapping_digest(role_context),
        "role_context": role_context,
        "fit_context_fingerprint": artifacts._mapping_digest(fit_context),
        "fit_context": fit_context,
        "variable_schema_fingerprint": dataset.variable_schema_fingerprint,
        "semantic_context": semantic_context,
        "semantic_context_digest": semantic_context_digest(semantic_context),
        "columns": list(frame.columns),
        "task_type": "classification",
        "target_view": "native",
        "n_samples": len(frame),
        "seed": 0,
        "device": "cpu",
        "registry_digest": DEFAULT_METRIC_CONTRACT_REGISTRY.digest(),
        "hpo_context_schema_version": None,
        "hpo_context_digest": None,
        "hpo_context": None,
        "resolved_parameters": {},
        "generator_metadata_schema_version": "generator-metadata-v2",
        "generator_context": generator_metadata["generator_context"],
        "implementation_fingerprint": "a" * 64,
    }
    cache = {
        **identity,
        "cache_key": artifacts._mapping_digest(identity),
        "row_count": len(frame),
        "synthetic_data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "generator_metadata": generator_metadata,
    }
    (generation_dir / f"{model_name}.cache.json").write_text(json.dumps(cache))
    return generation_dir, cache


def test_valid_current_generated_cache_is_loadable(tmp_path, make_canonical_dataset):
    dataset = make_canonical_dataset()
    generation_dir, _cache = _write_valid_generated_artifact(
        tmp_path, dataset, roles=("train", "tuning")
    )

    loaded = artifacts.load_validated_generated_datasets(
        generation_dir,
        dataset,
        model_names=["ctgan"],
        classification_score="balanced_accuracy",
        generation_hpo_enabled=True,
    )

    assert list(loaded) == ["ctgan"]
    assert list(loaded["ctgan"].columns) == list(dataset.full_df.columns)


def test_legacy_non_hpo_train_only_generated_cache_is_rejected(tmp_path, make_canonical_dataset):
    dataset = make_canonical_dataset()
    generation_dir, _cache = _write_valid_generated_artifact(tmp_path, dataset, roles=("train",))

    with pytest.raises(ValueError, match="role_context|evaluation manifest"):
        artifacts.load_validated_generated_datasets(
            generation_dir,
            dataset,
            model_names=["ctgan"],
            classification_score="balanced_accuracy",
            generation_hpo_enabled=False,
        )


def test_non_hpo_scope_comes_from_cache_when_evaluation_hpo_flag_is_stale(
    tmp_path, make_canonical_dataset
):
    dataset = make_canonical_dataset()
    generation_dir, _cache = _write_valid_generated_artifact(
        tmp_path, dataset, roles=("train", "tuning")
    )

    loaded = artifacts.load_validated_generated_datasets(
        generation_dir,
        dataset,
        model_names=["ctgan"],
        classification_score="balanced_accuracy",
        generation_hpo_enabled=True,
    )

    assert list(loaded) == ["ctgan"]


@pytest.mark.parametrize(
    "field, value",
    [
        ("hpo_context_schema_version", "hpo-context-v1"),
        ("hpo_context_digest", "a" * 64),
        ("hpo_context", {}),
    ],
)
def test_partial_hpo_metadata_is_rejected_before_evaluator(
    tmp_path, make_canonical_dataset, monkeypatch, field, value
):
    dataset = make_canonical_dataset()
    generation_dir, cache = _write_valid_generated_artifact(tmp_path, dataset, roles=("train",))
    cache[field] = value
    (generation_dir / "ctgan.cache.json").write_text(json.dumps(cache))
    backend_calls = []
    monkeypatch.setattr(
        "synthdata.evaluation.synthcity_eval.run_synthcity_evaluation",
        lambda *args, **kwargs: backend_calls.append(True),
    )

    with pytest.raises(ValueError, match="HPO metadata|hpo_context_schema_version"):
        artifacts.load_validated_generated_datasets(generation_dir, dataset, model_names=["ctgan"])
    assert backend_calls == []


def test_stale_hpo_config_does_not_reject_non_hpo_generated_cache(
    tmp_path, make_canonical_dataset, monkeypatch
):
    dataset = make_canonical_dataset()
    generation_dir, _cache = _write_valid_generated_artifact(
        tmp_path, dataset, roles=("train", "tuning")
    )
    backend_calls = []
    monkeypatch.setattr(
        "synthdata.evaluation.synthcity_eval.run_synthcity_evaluation",
        lambda *args, **kwargs: backend_calls.append(True),
    )

    loaded = artifacts.load_validated_generated_datasets(
        generation_dir,
        dataset,
        model_names=["ctgan"],
        classification_score="balanced_accuracy",
        generation_hpo_enabled=True,
    )
    assert list(loaded) == ["ctgan"]
    assert backend_calls == []


@pytest.mark.parametrize("failure", ["missing_sidecar", "role_context", "digest", "unexpected"])
def test_generated_input_preflight_fails_before_evaluator(
    tmp_path, make_canonical_dataset, monkeypatch, failure
):
    dataset = make_canonical_dataset()
    generation_dir, cache = _write_valid_generated_artifact(tmp_path, dataset)
    if failure == "missing_sidecar":
        (generation_dir / "ctgan.cache.json").unlink()
    elif failure == "role_context":
        cache["role_context"]["roles"]["train"]["rows"] += 1
        cache["role_context_fingerprint"] = artifacts._mapping_digest(cache["role_context"])
        (generation_dir / "ctgan.cache.json").write_text(json.dumps(cache))
    elif failure == "digest":
        (generation_dir / "ctgan.csv").write_text("feature,target\n999,0\n")
    else:
        (generation_dir / "manual.csv").write_text("feature,target\n1,0\n")

    backend_calls = []
    monkeypatch.setattr(
        "synthdata.evaluation.synthcity_eval.run_synthcity_evaluation",
        lambda *args, **kwargs: backend_calls.append(True),
    )

    with pytest.raises(ValueError):
        artifacts.load_validated_generated_datasets(
            generation_dir,
            dataset,
            model_names=["ctgan"],
            classification_score="balanced_accuracy",
        )
    assert backend_calls == []


def test_non_hpo_train_only_cache_rejects_tampering_before_evaluator(
    tmp_path, make_canonical_dataset, monkeypatch
):
    dataset = make_canonical_dataset()
    generation_dir, cache = _write_valid_generated_artifact(tmp_path, dataset, roles=("train",))
    cache["role_context"]["roles"]["train"]["rows"] += 1
    cache["role_context_fingerprint"] = artifacts._mapping_digest(cache["role_context"])
    (generation_dir / "ctgan.cache.json").write_text(json.dumps(cache))

    backend_calls = []
    monkeypatch.setattr(
        "synthdata.evaluation.synthcity_eval.run_synthcity_evaluation",
        lambda *args, **kwargs: backend_calls.append(True),
    )

    with pytest.raises(ValueError):
        artifacts.load_validated_generated_datasets(
            generation_dir,
            dataset,
            model_names=["ctgan"],
            classification_score="balanced_accuracy",
            generation_hpo_enabled=False,
        )
    assert backend_calls == []


def _rewrite_generation_cache_entry(evaluation_dir, entry, mutate) -> None:
    metadata_path = Path(entry["metadata_path"])
    manifest_path = artifact_bundle_dir(evaluation_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cache_metadata = json.loads(metadata_path.read_text())
    mutate(cache_metadata)
    identity = {
        key: value
        for key, value in cache_metadata.items()
        if key not in artifacts._CACHE_ENVELOPE_DYNAMIC_FIELDS
    }
    cache_metadata["cache_key"] = artifacts._mapping_digest(identity)
    metadata_path.write_text(json.dumps(cache_metadata, indent=2, sort_keys=True))
    manifest_entry = manifest["generator_metadata"]["model_a"]
    manifest_entry["cache_metadata"] = cache_metadata
    manifest_entry["metadata"] = cache_metadata["generator_metadata"]
    manifest_entry["metadata_sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _persist_current_generation_bundle(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    entry, role_context = _current_generation_cache_files(tmp_path)
    semantic_context = _semantic_context()
    role_context_fingerprint = artifacts._mapping_digest(role_context)
    contract_manifest = DEFAULT_METRIC_CONTRACT_REGISTRY.manifest()
    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {},
        native_syntheval_plot_dir=None,
        synthcity_validation_results={
            "model_a": _validation_payload("model_a", contract_manifest["digest"]),
        },
        metric_contract_manifest=contract_manifest,
        role_context={"candidate": role_context},
        role_context_fingerprint={"candidate": role_context_fingerprint},
        semantic_context=semantic_context,
        generator_metadata={"model_a": entry},
    )
    return evaluation_dir, entry, role_context, semantic_context


def _current_final_refit_files(tmp_path) -> tuple[dict, dict, dict]:
    refit_dir = tmp_path / "final_refit_current"
    refit_dir.mkdir()
    data_path = refit_dir / "model_a.csv"
    metadata_path = refit_dir / "model_a.cache.json"
    data_path.write_text("feature,target\n1,0\n")
    role_context = _role_context(["train", "tuning"])
    semantic_context = _semantic_context()
    generator_context = {"privacy_claim_type": "none"}
    generator_metadata = {
        "schema_version": "generator-metadata-v2",
        "generator_context": generator_context,
        "plugin_name": "model_a",
        "plugin_fqdn": "test.generator",
        "requested_parameters": {"temperature": 0.1},
        "n_samples": 1,
        "random_state": 0,
        "privacy_accounting": None,
        "implementation_fingerprint": "f" * 64,
    }
    cache_identity = {
        "schema_version": "final-refit-v2",
        "model_name": "model_a",
        "backend": "test",
        "columns": ["feature", "target"],
        "fit_roles": ["train", "tuning"],
        "fit_frame_fingerprint": "a" * 64,
        "fit_frame_fingerprints": {"raw": "b" * 64, "imputed": "c" * 64},
        "input_role_hashes": {
            "raw": {"train": "d" * 64, "tuning": "e" * 64},
            "imputed": {"train": "2" * 64, "tuning": "3" * 64},
        },
        "role_context_fingerprint": artifacts._mapping_digest(role_context),
        "role_context": role_context,
        "variable_schema_fingerprint": "1" * 64,
        "semantic_context": semantic_context,
        "semantic_context_digest": semantic_context_digest(semantic_context),
        "task_type": "classification",
        "target_view": "native",
        "n_samples": 1,
        "seed": 0,
        "device": "cpu",
        "registry_digest": DEFAULT_METRIC_CONTRACT_REGISTRY.digest(),
        "parameters": {"temperature": 0.1},
        "generator_metadata_schema_version": "generator-metadata-v2",
        "generator_context": generator_context,
        "implementation_fingerprint": "f" * 64,
    }
    cache_metadata = {
        **cache_identity,
        "cache_key": artifacts._mapping_digest(cache_identity),
        "synthetic_data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "generator_metadata": generator_metadata,
    }
    metadata_path.write_text(json.dumps(cache_metadata, indent=2, sort_keys=True))
    refit = {
        **cache_metadata,
        "path": str(data_path),
        "metadata_path": str(metadata_path),
        "cache_state": "generated",
    }
    return refit, role_context, semantic_context


def _persist_current_final_refit_bundle(tmp_path):
    evaluation_dir = tmp_path / "final_evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    refit, candidate_role_context, semantic_context = _current_final_refit_files(tmp_path)
    full_role_context = _role_context(["train", "tuning", "final_holdout"])
    contract_manifest = DEFAULT_METRIC_CONTRACT_REGISTRY.manifest()
    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {},
        native_syntheval_plot_dir=None,
        metric_contract_manifest=contract_manifest,
        role_context={
            "candidate": candidate_role_context,
            "full": full_role_context,
        },
        role_context_fingerprint={
            "candidate": artifacts._mapping_digest(candidate_role_context),
            "full": artifacts._mapping_digest(full_role_context),
        },
        semantic_context=semantic_context,
        final_holdout_evidence={
            **_complete_final_evidence_fields(),
            "evaluation_role": "final_holdout",
            "state": "succeeded",
            "selected_model": "model_a",
            "evidence_role": "final_holdout",
            "fit_roles": ["train", "tuning"],
            "semantic_context": semantic_context,
            "semantic_context_fingerprint": semantic_context_digest(semantic_context),
            "role_context": full_role_context,
            "role_context_fingerprint": artifacts._mapping_digest(full_role_context),
            "candidate_selection": {
                "source": "combined_evaluation.csv",
                "model": "model_a",
                "overall_rank": 1.0,
            },
            "final_refit": refit,
        },
    )
    return evaluation_dir, refit


def test_current_generation_cache_envelope_round_trips_strictly(tmp_path):
    evaluation_dir, _entry, role_context, semantic_context = _persist_current_generation_bundle(
        tmp_path
    )

    manifest = validate_evaluation_bundle(
        evaluation_dir,
        expected_role_context_fingerprints={
            "candidate": artifacts._mapping_digest(role_context),
        },
        expected_semantic_context_fingerprint=semantic_context_digest(semantic_context),
    )

    generator_entry = manifest["generator_metadata"]["model_a"]
    assert generator_entry["cache_metadata"]["schema_version"] == "generation-cache-v3"
    assert generator_entry["cache_metadata"]["target_view"] == "native"
    assert generator_entry["cache_metadata"]["resolved_parameters"] == {"temperature": 0.1}
    assert generator_entry["metadata_sha256"]
    assert generator_entry["data_sha256"]


def test_absolute_transient_sources_are_bundled_as_portable_paths(tmp_path):
    evaluation_dir, entry, _role_context, _semantic_context = _persist_current_generation_bundle(
        tmp_path
    )
    manifest = json.loads((artifact_bundle_dir(evaluation_dir) / "manifest.json").read_text())
    generator_entry = manifest["generator_metadata"]["model_a"]

    assert not Path(generator_entry["data_path"]).is_absolute()
    assert not Path(generator_entry["metadata_path"]).is_absolute()
    assert (artifact_bundle_dir(evaluation_dir) / "generation").is_dir()
    validate_evaluation_bundle(evaluation_dir)

    final_evaluation, refit = _persist_current_final_refit_bundle(tmp_path)
    final_evidence = load_final_holdout_evidence(final_evaluation)
    assert not Path(final_evidence["final_refit"]["path"]).is_absolute()
    assert not Path(final_evidence["final_refit"]["metadata_path"]).is_absolute()
    assert refit["path"].startswith("/")
    assert (artifact_bundle_dir(final_evaluation) / "final_refit" / "data.csv").is_file()


def test_relative_generator_sources_use_recorded_generation_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    entry, role_context = _current_generation_cache_files(tmp_path)
    entry = {**entry, "data_path": "model_a.csv", "metadata_path": "model_a.cache.json"}
    semantic_context = _semantic_context()

    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {},
        native_syntheval_plot_dir=None,
        role_context={"candidate": role_context},
        role_context_fingerprint={"candidate": artifacts._mapping_digest(role_context)},
        semantic_context=semantic_context,
        metric_contract_manifest=DEFAULT_METRIC_CONTRACT_REGISTRY.manifest(),
        synthcity_validation_results={
            "model_a": _validation_payload("model_a", DEFAULT_METRIC_CONTRACT_REGISTRY.digest())
        },
        generator_metadata={"model_a": entry},
        attempt_metadata={"source_generation_root": "generation"},
    )

    manifest = validate_evaluation_bundle(evaluation_dir)
    generator_entry = manifest["generator_metadata"]["model_a"]
    assert generator_entry["data_path"].startswith("evaluation_artifacts-v1/generation/model_a-")
    assert not Path(generator_entry["data_path"]).is_absolute()
    assert (artifact_bundle_dir(evaluation_dir) / "generation").is_dir()


def test_relative_evaluation_root_bundles_and_validates_generator_and_refit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output_dir = Path("output")
    evaluation_dir = output_dir / "evaluation"
    final_evaluation_dir = output_dir / "final_evaluation"
    evaluation_dir.mkdir(parents=True)
    final_evaluation_dir.mkdir(parents=True)
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    _combined().to_csv(final_evaluation_dir / "combined_evaluation.csv")

    entry, role_context = _current_generation_cache_files(tmp_path)
    semantic_context = _semantic_context()
    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {},
        native_syntheval_plot_dir=None,
        role_context={"candidate": role_context},
        role_context_fingerprint={"candidate": artifacts._mapping_digest(role_context)},
        semantic_context=semantic_context,
        metric_contract_manifest=DEFAULT_METRIC_CONTRACT_REGISTRY.manifest(),
        synthcity_validation_results={
            "model_a": _validation_payload("model_a", DEFAULT_METRIC_CONTRACT_REGISTRY.digest())
        },
        generator_metadata={"model_a": entry},
    )
    refit, candidate_role_context, semantic_context = _current_final_refit_files(tmp_path)
    full_role_context = _role_context(["train", "tuning", "final_holdout"])
    persist_evaluation_artifacts(
        final_evaluation_dir,
        _combined(),
        {},
        native_syntheval_plot_dir=None,
        metric_contract_manifest=DEFAULT_METRIC_CONTRACT_REGISTRY.manifest(),
        role_context={"candidate": candidate_role_context, "full": full_role_context},
        role_context_fingerprint={
            "candidate": artifacts._mapping_digest(candidate_role_context),
            "full": artifacts._mapping_digest(full_role_context),
        },
        semantic_context=semantic_context,
        final_holdout_evidence={
            **_complete_final_evidence_fields(),
            "evaluation_role": "final_holdout",
            "state": "succeeded",
            "selected_model": "model_a",
            "evidence_role": "final_holdout",
            "fit_roles": ["train", "tuning"],
            "semantic_context": semantic_context,
            "semantic_context_fingerprint": semantic_context_digest(semantic_context),
            "role_context": full_role_context,
            "role_context_fingerprint": artifacts._mapping_digest(full_role_context),
            "final_refit": refit,
        },
    )

    generator_manifest = validate_evaluation_bundle(evaluation_dir)
    final_evidence = load_final_holdout_evidence(final_evaluation_dir)

    generator_paths = generator_manifest["generator_metadata"]["model_a"]
    generator_data_path = Path(generator_paths["data_path"])
    generator_metadata_path = Path(generator_paths["metadata_path"])
    assert not generator_data_path.is_absolute()
    assert not generator_metadata_path.is_absolute()
    assert generator_data_path.parts[:1] == ("evaluation_artifacts-v1",)
    assert generator_metadata_path.parts[:1] == ("evaluation_artifacts-v1",)
    assert (evaluation_dir / generator_data_path).is_file()
    assert (evaluation_dir / generator_metadata_path).is_file()
    assert final_evidence["final_refit"]["provenance_state"] == "verified"
    refit_data_path = Path(final_evidence["final_refit"]["path"])
    refit_metadata_path = Path(final_evidence["final_refit"]["metadata_path"])
    assert not refit_data_path.is_absolute()
    assert not refit_metadata_path.is_absolute()
    assert refit_data_path.parts[:1] == ("evaluation_artifacts-v1",)
    assert refit_metadata_path.parts[:1] == ("evaluation_artifacts-v1",)
    assert (final_evaluation_dir / refit_data_path).is_file()
    assert (final_evaluation_dir / refit_metadata_path).is_file()


@pytest.mark.parametrize("source_key", ["data_path", "metadata_path"])
def test_relative_generator_source_rejects_missing_escape_and_symlink(tmp_path, source_key):
    generation_root = tmp_path / "generation"
    generation_root.mkdir()
    source = generation_root / (
        "model_a.csv" if source_key == "data_path" else "model_a.cache.json"
    )
    source.write_text("source")
    (generation_root / "model_a.csv").write_text("source")
    (generation_root / "model_a.cache.json").write_text("source")
    entry = {"state": "present", "data_path": "model_a.csv", "metadata_path": "model_a.cache.json"}
    entry[source_key] = "missing" if source_key == "data_path" else "../outside"
    if source_key == "metadata_path":
        (tmp_path / "outside").write_text("outside")
    with pytest.raises(ValueError, match="(regular file|escapes)"):
        artifacts._bundle_generator_metadata(
            {"model_a": entry},
            bundle_dir=tmp_path / "evaluation" / "evaluation_artifacts-v1",
            source_root=generation_root,
        )

    entry = {"state": "present", "data_path": "model_a.csv", "metadata_path": "model_a.cache.json"}
    symlink = generation_root / "symlink"
    symlink.symlink_to(source)
    entry["data_path"] = "model_a.csv"
    entry["metadata_path"] = "symlink"
    entry[source_key] = "symlink"
    with pytest.raises(ValueError, match="symlink"):
        artifacts._bundle_generator_metadata(
            {"model_a": entry},
            bundle_dir=tmp_path / "evaluation" / "evaluation_artifacts-v1",
            source_root=generation_root,
        )


@pytest.mark.parametrize("relative", ["/absolute/file.csv", "../escape/file.csv"])
def test_persisted_artifact_paths_reject_absolute_or_traversal(relative, tmp_path):
    with pytest.raises(ValueError, match="relative|escapes"):
        artifacts._relative_artifact_path(tmp_path, relative, "Persisted artifact")


def test_persisted_artifact_paths_reject_symlink_components(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "link").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        artifacts._relative_artifact_path(tmp_path, "link/file.csv", "Persisted artifact")


def test_generation_cache_sidecar_cannot_diverge_from_embedded_envelope(tmp_path):
    evaluation_dir, entry, _role_context, _semantic_context = _persist_current_generation_bundle(
        tmp_path
    )
    metadata_path = Path(entry["metadata_path"])
    cache_metadata = json.loads(metadata_path.read_text())
    cache_metadata["resolved_parameters"] = {"temperature": 0.9}
    identity = {
        key: value
        for key, value in cache_metadata.items()
        if key not in artifacts._CACHE_ENVELOPE_DYNAMIC_FIELDS
    }
    cache_metadata["cache_key"] = artifacts._mapping_digest(identity)
    metadata_path.write_text(json.dumps(cache_metadata, indent=2, sort_keys=True))

    manifest_path = artifact_bundle_dir(evaluation_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["generator_metadata"]["model_a"]["metadata_sha256"] = hashlib.sha256(
        metadata_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    with pytest.raises(ValueError, match="does not match its metadata sidecar"):
        validate_evaluation_bundle(evaluation_dir)


def test_current_final_refit_envelope_round_trips_strictly(tmp_path):
    evaluation_dir, _refit = _persist_current_final_refit_bundle(tmp_path)

    evidence = load_final_holdout_evidence(evaluation_dir)

    assert evidence["final_refit"]["provenance_state"] == "verified"
    assert evidence["final_refit"]["cache_metadata"]["schema_version"] == "final-refit-v2"
    assert evidence["final_refit"]["cache_metadata"]["target_view"] == "native"
    assert evidence["final_refit"]["cache_metadata"]["fit_roles"] == ["train", "tuning"]


def test_legacy_final_holdout_marker_migrates_to_complete_shape(tmp_path):
    evaluation_dir, _refit = _persist_current_final_refit_bundle(tmp_path)
    evidence_path = artifact_bundle_dir(evaluation_dir) / "final_holdout_evidence.json"
    payload = json.loads(evidence_path.read_text())
    for field in (
        "evidence_execution_state",
        "metric_completeness_state",
        "score_completeness_state",
        "audit_outcome_state",
        "provenance_inventory",
    ):
        payload.pop(field)
    payload["legacy_schema_version"] = "final-holdout-evidence-legacy-v1"
    evidence_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _refresh_manifest_digest(evaluation_dir, "final_holdout_evidence")

    migrated = load_final_holdout_evidence(evaluation_dir)

    assert migrated["schema_version"] == "final-holdout-evidence-v1"
    assert migrated["provenance_inventory_legacy_migrated"] is True
    assert migrated["evidence_execution_state"] == "succeeded"
    assert set(migrated["provenance_inventory"]) == set(artifacts._PROVENANCE_INVENTORY_FIELDS)
    assert migrated["provenance_inventory"]["selected_model_provenance"]["model"] == "model_a"


@pytest.mark.parametrize("marker", ["final-holdout-evidence-v2", "", "legacy-v1", None])
def test_final_holdout_loader_rejects_unknown_or_unmarked_legacy_marker(tmp_path, marker):
    evaluation_dir, _refit = _persist_current_final_refit_bundle(tmp_path)
    evidence_path = artifact_bundle_dir(evaluation_dir) / "final_holdout_evidence.json"
    payload = json.loads(evidence_path.read_text())
    if marker is None:
        for field in (
            "evidence_execution_state",
            "metric_completeness_state",
            "score_completeness_state",
            "audit_outcome_state",
            "provenance_inventory",
        ):
            payload.pop(field)
    else:
        payload["legacy_schema_version"] = marker
    evidence_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _refresh_manifest_digest(evaluation_dir, "final_holdout_evidence")

    with pytest.raises(
        ValueError, match="(unsupported legacy_schema_version|state|provenance_inventory)"
    ):
        load_final_holdout_evidence(evaluation_dir)


def test_current_final_refit_sidecar_tampering_fails_after_hash_refresh(tmp_path):
    evaluation_dir, refit = _persist_current_final_refit_bundle(tmp_path)
    metadata_path = artifact_bundle_dir(evaluation_dir) / "final_refit" / "metadata.json"
    cache_metadata = json.loads(metadata_path.read_text())
    cache_metadata["target_view"] = "binary_collapsed"
    identity = {
        key: value
        for key, value in cache_metadata.items()
        if key not in artifacts._CACHE_ENVELOPE_DYNAMIC_FIELDS
    }
    cache_metadata["cache_key"] = artifacts._mapping_digest(identity)
    metadata_path.write_text(json.dumps(cache_metadata, indent=2, sort_keys=True))

    evidence_path = artifact_bundle_dir(evaluation_dir) / "final_holdout_evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["final_refit"].update(
        {
            "target_view": "binary_collapsed",
            "cache_key": cache_metadata["cache_key"],
            "synthetic_data_sha256": cache_metadata["synthetic_data_sha256"],
            "generator_metadata": cache_metadata["generator_metadata"],
            "cache_metadata": cache_metadata,
            "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        }
    )
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True))
    _refresh_manifest_digest(evaluation_dir, "final_holdout_evidence")

    with pytest.raises(ValueError, match="target_view must be 'native'"):
        load_final_holdout_evidence(evaluation_dir)


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("role_context", "role_context_fingerprint does not match the evaluation manifest"),
        ("semantic_context", "semantic_context_digest does not match the evaluation manifest"),
        ("parameters", "resolved_parameters does not match generator_metadata"),
        ("target_view", "target_view must be 'native'"),
        (
            "implementation_fingerprint",
            "implementation_fingerprint does not match generator_metadata",
        ),
    ],
)
def test_current_generation_cache_tampering_fails_after_hash_refresh(tmp_path, tamper, message):
    evaluation_dir, entry, _role_context, _semantic_context = _persist_current_generation_bundle(
        tmp_path
    )

    def mutate(cache_metadata):
        if tamper == "role_context":
            cache_metadata["role_context"]["roles"]["train"]["rows"] = 2
            cache_metadata["role_context_fingerprint"] = artifacts._mapping_digest(
                cache_metadata["role_context"]
            )
        elif tamper == "semantic_context":
            cache_metadata["semantic_context"]["quasi_identifier_columns"] = ["protected"]
            cache_metadata["semantic_context_digest"] = semantic_context_digest(
                cache_metadata["semantic_context"]
            )
        elif tamper == "parameters":
            cache_metadata["resolved_parameters"] = {"temperature": 0.9}
        elif tamper == "target_view":
            cache_metadata["target_view"] = "binary_collapsed"
        else:
            cache_metadata["implementation_fingerprint"] = "d" * 64

    _rewrite_generation_cache_entry(evaluation_dir, entry, mutate)

    with pytest.raises(ValueError, match=message):
        validate_evaluation_bundle(evaluation_dir)


def _report() -> dict:
    hierarchy = _dataframe(
        {
            "Model": ["model_a"],
            "level": ["target"],
            "TARGET_LABEL": ["positive"],
            "user_n": [10],
            "EquityColor": ["#ffffff"],
            "EquityLabel": ["Equal"],
            "EquityValue": [0.0],
            "Background_Rate": [1.0],
            "Observed_Rate": [1.0],
            "background_n": [10],
            "BH_p": [1.0],
        }
    )
    return {
        "state": "succeeded",
        "summary_stats": {
            "model": "model_a",
            "n_subgroups": 1,
            "mean_abs_log_disparity": 0.0,
            "median_abs_log_disparity": 0.0,
            "share_significant_bh": 0.0,
        },
        "leaf_results": hierarchy.copy(),
        "hierarchy_results": hierarchy,
        "subgroup_table": _dataframe(
            {
                "Characteristic": ["Target"],
                "Protected Subgroup": ["positive"],
                "Equity Value": ["0.000"],
                "BH-adjusted p-value": ["1.000"],
                "EquityColor": ["#ffffff"],
                "EquityLabel": ["Equal"],
            }
        ),
        "leaf_equity_table": _dataframe(
            {
                "Protected Subgroup": ["positive"],
                "Equity Value": ["0.000"],
                "BH-adjusted p-value": ["1.000"],
                "EquityColor": ["#ffffff"],
                "EquityLabel": ["Equal"],
            }
        ),
        "legend_table": _dataframe(
            {
                "Description": ["Equal"],
                "Metric Value Rule": ["0"],
                "Color": ["#ffffff"],
            }
        ),
        "label_counts": _dataframe({"Model": ["model_a"], "EquityLabel": ["Equal"], "count": [1]}),
        "protected_group_cols": [],
        "protected_order_map": {},
        "target_order": ["positive"],
    }


def test_log_disparity_artifacts_round_trip_and_render(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {"model_a": _report()},
        native_syntheval_plot_dir=None,
    )

    loaded = load_log_disparity_reports(evaluation_dir)
    assert loaded["model_a"]["state"] == "succeeded"
    assert loaded["model_a"]["summary_stats"]["model"] == "model_a"
    assert artifact_bundle_dir(evaluation_dir).joinpath("manifest.json").exists()
    assert build_log_disparity_report_figure(loaded["model_a"]).data


def test_successful_log_disparity_result_metadata_redacts_sentinels(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    sentinel = "SECRET /tmp/private patient_id=42 traceback HMAC key"
    report = _report()
    report["result_metadata"] = {
        "valid_pairs": 2,
        "bounded_enum": "succeeded",
        "digest": "b" * 64,
        "nested": {"safe": True, "unsafe": sentinel},
        "items": [1.5, sentinel],
    }
    persist_evaluation_artifacts(
        evaluation_dir, _combined(), {"model_a": report}, native_syntheval_plot_dir=None
    )

    loaded = load_log_disparity_reports(evaluation_dir)
    metadata = loaded["model_a"]["result_metadata"]
    metadata_path = (
        artifact_bundle_dir(evaluation_dir)
        / "log_disparity"
        / artifacts._model_artifact_id("model_a")
        / "metadata.json"
    )
    assert sentinel not in metadata_path.read_text()
    assert sentinel not in json.dumps(loaded["model_a"]["result_metadata"])
    assert metadata == {
        "bounded_enum": "succeeded",
        "digest": "b" * 64,
        "items": [1.5],
        "nested": {"safe": True},
        "valid_pairs": 2,
    }


def test_indeterminate_log_disparity_tampered_metadata_is_rejected(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {
            "model_a": {
                "state": "indeterminate",
                "reason": "missing release provenance",
                "missing_tables": [
                    "leaf_results",
                    "hierarchy_results",
                    "subgroup_table",
                    "leaf_equity_table",
                    "legend_table",
                    "label_counts",
                ],
                "result_metadata": {"valid": 1},
            }
        },
        native_syntheval_plot_dir=None,
    )
    metadata_path = (
        artifact_bundle_dir(evaluation_dir)
        / "log_disparity"
        / artifacts._model_artifact_id("model_a")
        / "metadata.json"
    )
    metadata = json.loads(metadata_path.read_text())
    metadata["result_metadata"]["unsafe"] = "SECRET /tmp/patient traceback HMAC"
    metadata_path.write_text(json.dumps(metadata))
    manifest_path = artifact_bundle_dir(evaluation_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["log_disparity"]["model_a"]["metadata_sha256"] = hashlib.sha256(
        metadata_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    with pytest.raises(ValueError, match="unsafe metadata"):
        load_log_disparity_reports(evaluation_dir)


def test_log_disparity_indeterminate_artifact_round_trips_metadata_only(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    report = {
        "state": "indeterminate",
        "reason": "missing release provenance",
        "missing_tables": [
            "leaf_results",
            "hierarchy_results",
            "subgroup_table",
            "leaf_equity_table",
            "legend_table",
            "label_counts",
        ],
        "result_metadata": {"release_evidence_state": "indeterminate"},
    }
    persist_evaluation_artifacts(
        evaluation_dir, _combined(), {"model_a": report}, native_syntheval_plot_dir=None
    )

    loaded = load_log_disparity_reports(evaluation_dir)

    assert loaded["model_a"] == {
        "state": "indeterminate",
        "reason": "missing release provenance",
        "missing_tables": report["missing_tables"],
        "result_metadata": report["result_metadata"],
    }
    assert "summary_stats" not in loaded["model_a"]


@pytest.mark.parametrize("state", ["unknown", None])
def test_log_disparity_unknown_or_absent_state_with_tables_is_indeterminate(tmp_path, state):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    report = _report()
    if state is not None:
        report["state"] = state
    else:
        report.pop("state")

    persist_evaluation_artifacts(
        evaluation_dir, _combined(), {"model_a": report}, native_syntheval_plot_dir=None
    )

    loaded = load_log_disparity_reports(evaluation_dir)

    assert loaded["model_a"]["state"] == "indeterminate"
    assert loaded["model_a"]["reason"] == "report_state_missing_or_unknown"
    assert "summary_stats" not in loaded["model_a"]


def test_log_disparity_failed_artifact_redacts_error_body(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    report = {
        "state": "failed",
        "error": "secret=/tmp/private/id=123 compute exploded",
        "error_type": "ValueError",
    }
    persist_evaluation_artifacts(
        evaluation_dir, _combined(), {"model_a": report}, native_syntheval_plot_dir=None
    )

    loaded = load_log_disparity_reports(evaluation_dir)

    assert loaded["model_a"] == {
        "state": "failed",
        "reason": "log_disparity_evaluation_failed",
        "error_type": "ValueError",
    }
    assert "secret" not in json.dumps(loaded)


def test_release_score_evidence_round_trip_and_manifest_integrity(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    combined = pd.concat([_combined(), _combined().rename(index={"model_a": "model_b"})])
    combined.to_csv(evaluation_dir / "combined_evaluation.csv")
    score = {
        "model_a": {
            "status": "succeeded",
            "score": 0.8125,
            "dimensions": {
                "utility": {
                    "score": 0.8,
                    "components": {
                        key: {"score": 0.8, "status": "succeeded"} for key in ("tstr", "mmd", "jsd")
                    },
                },
                "privacy": {
                    "score": 0.9,
                    "components": {
                        key: {"score": 0.8, "status": "succeeded"}
                        for key in ("k", "l", "dcr", "epsilon", "mia", "attribute")
                    },
                },
                "fairness": {
                    "score": 0.7,
                    "components": {
                        key: {"score": 0.8, "status": "succeeded"}
                        for key in ("representation", "eo", "worst_log_disparity")
                    },
                },
            },
            "audit_only": True,
        }
    }
    persist_evaluation_artifacts(
        evaluation_dir,
        combined,
        {},
        native_syntheval_plot_dir=None,
        final_holdout_evidence={
            **_complete_final_evidence_fields(),
            "evaluation_role": "final_holdout",
            "state": "succeeded",
            "selected_model": "model_a",
            "final_refit": _final_refit_files(tmp_path),
        },
        release_score=score,
    )
    loaded = load_release_score_evidence(evaluation_dir)
    assert loaded["models"]["model_a"] == score["model_a"]
    binding = loaded["models"]["model_a"]["provenance"]["final_holdout_binding"]
    assert binding["raw_imputed_role_hashes"]["evidence"] == {
        "train": "a" * 64,
        "tuning": "b" * 64,
        "final_holdout": "c" * 64,
        "refit_fit": "d" * 64,
    }
    assert binding["raw_imputed_role_hashes"]["custom_raw"] == {
        "train": "e" * 64,
        "tuning": "f" * 64,
        "final_holdout": "0" * 64,
        "refit_fit": "1" * 64,
    }
    assert loaded["selected_model"] == "model_a"
    assert loaded["candidate_audit_models"] == ["model_a", "model_b"]
    final_path = artifact_bundle_dir(evaluation_dir) / "final_holdout_evidence.json"
    final_payload = json.loads(final_path.read_text())
    final_payload["release_score"]["score"] = 0.8126
    final_path.write_text(json.dumps(final_payload, indent=2, sort_keys=True))
    _refresh_manifest_digest(evaluation_dir, "final_holdout_evidence")
    with pytest.raises(ValueError, match="does not match final-holdout release score"):
        load_release_score_evidence(evaluation_dir)
    final_payload["release_score"]["score"] = 0.8125
    final_path.write_text(json.dumps(final_payload, indent=2, sort_keys=True))
    _refresh_manifest_digest(evaluation_dir, "final_holdout_evidence")
    manifest_path = artifact_bundle_dir(evaluation_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    sidecar = artifact_bundle_dir(evaluation_dir) / "release_score_evidence.json"
    sidecar.write_text(sidecar.read_text().replace("0.8125", "0.8126"))
    with pytest.raises(ValueError, match="integrity"):
        load_release_score_evidence(evaluation_dir)
    assert manifest["release_score_evidence"]["audit_only"] is True
    assert manifest["release_score_evidence"]["inventory_scope"] == "selected_model"
    assert manifest["release_score_evidence"]["selected_model"] == "model_a"


def test_release_score_malformed_write_does_not_create_score_sidecars(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    evidence = {
        **_complete_final_evidence_fields(),
        "evaluation_role": "final_holdout",
        "state": "succeeded",
        "selected_model": "model_a",
        "final_refit": _final_refit_files(tmp_path),
    }
    with pytest.raises(ValueError, match="not finite"):
        persist_evaluation_artifacts(
            evaluation_dir,
            _combined(),
            {},
            native_syntheval_plot_dir=None,
            final_holdout_evidence=evidence,
            release_score={"model_a": {"status": "succeeded", "score": float("nan")}},
        )
    bundle = artifact_bundle_dir(evaluation_dir)
    assert not (bundle / "release_score_evidence.json").exists()
    assert not (bundle / "final_holdout_evidence.json").exists()


@pytest.mark.parametrize(
    "field",
    [
        "role_context_fingerprint",
        "common_protocol_digest",
        "release_transform_digest",
        "role_hashes",
        "final_refit",
        "selected_model",
    ],
)
def test_current_release_binding_missing_identity_does_not_create_artifacts(tmp_path, field):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    evidence = {
        **_complete_final_evidence_fields(),
        "evaluation_role": "final_holdout",
        "state": "succeeded",
        "final_refit": _final_refit_files(tmp_path),
    }
    if field == "release_transform_digest":
        evidence["provenance_inventory"] = {
            **evidence["provenance_inventory"],
            "release_transform_digest": "",
        }
    elif field == "role_hashes":
        evidence["provenance_inventory"] = {
            **evidence["provenance_inventory"],
            "role_hashes": {},
        }
    elif field == "final_refit":
        evidence[field] = None
    else:
        evidence[field] = ""

    with pytest.raises(ValueError, match="binding identity|final_refit|selected model|role_hashes"):
        persist_evaluation_artifacts(
            evaluation_dir,
            _combined(),
            {},
            native_syntheval_plot_dir=None,
            final_holdout_evidence=evidence,
            release_score=_succeeded_release_score(),
        )

    bundle = artifact_bundle_dir(evaluation_dir)
    assert not (bundle / "release_score_evidence.json").exists()
    assert not (bundle / "final_holdout_evidence.json").exists()


@pytest.mark.parametrize("location", ["score", "dimension", "component"])
@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_release_score_out_of_range_values_are_rejected_on_write(tmp_path, location, value):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    score = _succeeded_release_score()
    record = score["model_a"]
    if location == "score":
        record["score"] = value
    elif location == "dimension":
        record["dimensions"]["utility"]["score"] = value
    else:
        record["dimensions"]["utility"]["components"]["tstr"]["score"] = value
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        persist_evaluation_artifacts(
            evaluation_dir,
            _combined(),
            {},
            native_syntheval_plot_dir=None,
            final_holdout_evidence={
                **_complete_final_evidence_fields(),
                "evaluation_role": "final_holdout",
                "state": "succeeded",
                "selected_model": "model_a",
                "final_refit": _final_refit_files(tmp_path),
            },
            release_score=score,
        )


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_release_score_out_of_range_sidecar_is_rejected_after_hash_refresh(tmp_path, value):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {},
        native_syntheval_plot_dir=None,
        final_holdout_evidence={
            **_complete_final_evidence_fields(),
            "evaluation_role": "final_holdout",
            "state": "succeeded",
            "selected_model": "model_a",
            "final_refit": _final_refit_files(tmp_path),
        },
        release_score=_succeeded_release_score(),
    )
    sidecar = artifact_bundle_dir(evaluation_dir) / "release_score_evidence.json"
    payload = json.loads(sidecar.read_text())
    payload["models"]["model_a"]["score"] = value
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _refresh_manifest_digest(evaluation_dir, "release_score_evidence")
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        load_release_score_evidence(evaluation_dir)


def test_indeterminate_log_disparity_unknown_reason_rejected_after_hash_refresh(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {
            "model_a": {
                "state": "indeterminate",
                "reason": "missing release provenance",
                "missing_tables": list(artifacts._LOG_REPORT_TABLES),
            }
        },
        native_syntheval_plot_dir=None,
    )
    metadata_path = (
        artifact_bundle_dir(evaluation_dir)
        / "log_disparity"
        / artifacts._model_artifact_id("model_a")
        / "metadata.json"
    )
    metadata = json.loads(metadata_path.read_text())
    metadata["reason"] = "SECRET /tmp/private traceback HMAC"
    metadata_path.write_text(json.dumps(metadata))
    manifest_path = artifact_bundle_dir(evaluation_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["log_disparity"]["model_a"]["metadata_sha256"] = hashlib.sha256(
        metadata_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    with pytest.raises(ValueError, match="not allowlisted"):
        load_log_disparity_reports(evaluation_dir)


def test_release_score_binding_tamper_fails_after_hash_refresh(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    dimensions = {
        "utility": ("tstr", "mmd", "jsd"),
        "privacy": ("k", "l", "dcr", "epsilon", "mia", "attribute"),
        "fairness": ("representation", "eo", "worst_log_disparity"),
    }
    score = {
        "model_a": {
            "status": "succeeded",
            "score": 0.8,
            "audit_only": True,
            "dimensions": {
                name: {
                    "score": 0.8,
                    "components": {
                        component: {"score": 0.8, "status": "succeeded"} for component in components
                    },
                }
                for name, components in dimensions.items()
            },
        }
    }
    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {},
        native_syntheval_plot_dir=None,
        final_holdout_evidence={
            **_complete_final_evidence_fields(),
            "evaluation_role": "final_holdout",
            "state": "succeeded",
            "selected_model": "model_a",
            "final_refit": _final_refit_files(tmp_path),
        },
        release_score=score,
    )
    evidence_path = artifact_bundle_dir(evaluation_dir) / "final_holdout_evidence.json"
    payload = json.loads(evidence_path.read_text())
    payload["release_score"]["provenance"]["final_holdout_binding"]["selected_model"] = "model_b"
    payload["release_score"]["provenance"]["final_holdout_binding_digest"] = (
        artifacts._mapping_digest(payload["release_score"]["provenance"]["final_holdout_binding"])
    )
    evidence_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _refresh_manifest_digest(evaluation_dir, "final_holdout_evidence")
    with pytest.raises(ValueError, match="provenance binding"):
        load_release_score_evidence(evaluation_dir)

    payload = json.loads(evidence_path.read_text())
    binding = payload["release_score"]["provenance"]["final_holdout_binding"]
    binding["raw_imputed_role_hashes"]["evidence"] = {
        "imputed_evaluation": dict(binding["raw_imputed_role_hashes"]["evidence"])
    }
    payload["release_score"]["provenance"]["final_holdout_binding_digest"] = (
        artifacts._mapping_digest(binding)
    )
    evidence_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _refresh_manifest_digest(evaluation_dir, "final_holdout_evidence")
    with pytest.raises(ValueError, match="role hashes.evidence"):
        load_release_score_evidence(evaluation_dir)


def test_indeterminate_release_score_binding_tamper_fails_after_hash_refresh(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    score = _succeeded_release_score()
    record = score["model_a"]
    record["status"] = "indeterminate"
    record["score"] = None
    record["provenance"] = {}
    for dimension in record["dimensions"].values():
        dimension["score"] = None
        for component in dimension["components"].values():
            component["status"] = "indeterminate"
            component["score"] = None
    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {},
        native_syntheval_plot_dir=None,
        final_holdout_evidence={
            **_complete_final_evidence_fields(),
            "evaluation_role": "final_holdout",
            "state": "failed",
            "evidence_execution_state": "succeeded",
            "metric_completeness_state": "complete",
            "score_completeness_state": "indeterminate",
            "audit_outcome_state": "indeterminate",
            "selected_model": "model_a",
            "final_refit": _final_refit_files(tmp_path),
        },
        release_score=score,
    )
    evidence_path = artifact_bundle_dir(evaluation_dir) / "final_holdout_evidence.json"
    valid = load_final_holdout_evidence(evaluation_dir)
    assert valid["release_score"]["status"] == "indeterminate"
    payload = json.loads(evidence_path.read_text())
    binding = payload["release_score"]["provenance"]["final_holdout_binding"]
    binding["selected_model"] = "model_b"
    payload["release_score"]["provenance"]["final_holdout_binding_digest"] = (
        artifacts._mapping_digest(binding)
    )
    evidence_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _refresh_manifest_digest(evaluation_dir, "final_holdout_evidence")

    with pytest.raises(ValueError, match="provenance binding"):
        load_final_holdout_evidence(evaluation_dir)


def test_indeterminate_release_score_without_binding_round_trips(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    score = _succeeded_release_score()
    record = score["model_a"]
    record.update(status="indeterminate", score=None, provenance={})
    for dimension in record["dimensions"].values():
        dimension["score"] = None
        for component in dimension["components"].values():
            component.update(status="indeterminate", score=None)
    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {},
        native_syntheval_plot_dir=None,
        final_holdout_evidence={
            **_complete_final_evidence_fields(),
            "evaluation_role": "final_holdout",
            "state": "failed",
            "evidence_execution_state": "succeeded",
            "metric_completeness_state": "complete",
            "score_completeness_state": "indeterminate",
            "audit_outcome_state": "indeterminate",
            "selected_model": "model_a",
            "final_refit": _final_refit_files(tmp_path),
        },
        release_score=score,
    )

    evidence_path = artifact_bundle_dir(evaluation_dir) / "final_holdout_evidence.json"
    payload = json.loads(evidence_path.read_text())
    payload["release_score"]["provenance"] = {}
    evidence_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _refresh_manifest_digest(evaluation_dir, "final_holdout_evidence")

    loaded = load_final_holdout_evidence(evaluation_dir)

    assert loaded["release_score"]["status"] == "indeterminate"
    assert loaded["release_score"]["provenance"] == {}


def test_manifest_intermediate_symlink_path_is_rejected(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    persist_evaluation_artifacts(
        evaluation_dir, _combined(), {"model_a": _report()}, native_syntheval_plot_dir=None
    )
    bundle = artifact_bundle_dir(evaluation_dir)
    (bundle / "log_link").symlink_to(bundle / "log_disparity", target_is_directory=True)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["log_disparity"]["model_a"]["path"] = "log_link/model_a"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    with pytest.raises(ValueError, match="symlink component"):
        load_log_disparity_reports(evaluation_dir)


def test_bundled_native_plot_is_portable_and_integrity_checked(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    native_dir = tmp_path / "native"
    evaluation_dir.mkdir()
    native_dir.mkdir()
    (native_dir / "metric.png").write_bytes(b"plot")
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {"model_a": _report()},
        native_syntheval_plot_dir=native_dir,
    )
    (native_dir / "metric.png").unlink()

    verify_native_syntheval_artifacts(evaluation_dir)
    manifest = json.loads((artifact_bundle_dir(evaluation_dir) / "manifest.json").read_text())
    native_manifest = manifest["native_syntheval_plots"]
    assert not Path(native_manifest["root"]).is_absolute()
    bundled_plot = evaluation_dir / native_manifest["root"] / "metric.png"
    assert bundled_plot.is_file()
    bundled_plot.unlink()
    with pytest.raises(FileNotFoundError, match="Native SynthEval"):
        verify_native_syntheval_artifacts(evaluation_dir)


def test_native_plot_bundle_rebuild_removes_stale_files(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    native_dir = tmp_path / "native"
    evaluation_dir.mkdir()
    native_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    (native_dir / "old.png").write_bytes(b"old")
    persist_evaluation_artifacts(
        evaluation_dir, _combined(), {}, native_syntheval_plot_dir=native_dir
    )

    (native_dir / "old.png").unlink()
    (native_dir / "new.png").write_bytes(b"new")
    persist_evaluation_artifacts(
        evaluation_dir, _combined(), {}, native_syntheval_plot_dir=native_dir
    )

    bundle_root = artifact_bundle_dir(evaluation_dir) / "native_syntheval_plots"
    assert not (bundle_root / "old.png").exists()
    assert (bundle_root / "new.png").read_bytes() == b"new"
    verify_native_syntheval_artifacts(evaluation_dir)


def test_native_plot_bundle_skips_symlink_escape(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    native_dir = tmp_path / "native"
    external_dir = tmp_path / "external"
    evaluation_dir.mkdir()
    native_dir.mkdir()
    external_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    (external_dir / "secret.png").write_bytes(b"secret")
    (native_dir / "local.png").write_bytes(b"local")
    (native_dir / "escape.png").symlink_to(external_dir / "secret.png")
    (native_dir / "external_dir").symlink_to(external_dir, target_is_directory=True)

    persist_evaluation_artifacts(
        evaluation_dir, _combined(), {}, native_syntheval_plot_dir=native_dir
    )

    bundle_root = artifact_bundle_dir(evaluation_dir) / "native_syntheval_plots"
    assert (bundle_root / "local.png").is_file()
    assert not (bundle_root / "escape.png").exists()
    assert not (bundle_root / "external_dir").exists()
    manifest = json.loads((artifact_bundle_dir(evaluation_dir) / "manifest.json").read_text())
    assert [entry["path"] for entry in manifest["native_syntheval_plots"]["files"]] == ["local.png"]


def test_contract_and_status_sidecars_round_trip(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {},
        native_syntheval_plot_dir=None,
        synthcity_validation_results={
            "model_a": _validation_payload("model_a", _contract_manifest()["digest"])
        },
        metric_contract_manifest=_contract_manifest(),
        source_provenance={"synthcity": {"revision": "fork-sha"}},
        role_context={
            "candidate": _role_context(["train", "tuning"]),
            "full": _role_context(["train", "tuning", "final_holdout"]),
        },
        role_context_fingerprint={
            "candidate": "candidate-context",
            "full": "full-context",
        },
    )

    status = load_synthcity_metric_status(evaluation_dir)
    contract = load_metric_contract_manifest(evaluation_dir)
    assert status["models"]["model_a"]["contract_digest"] == contract["digest"]
    assert status["source_provenance"]["synthcity"]["revision"] == "fork-sha"
    assert contract["digest"] == _contract_manifest()["digest"]
    manifest = json.loads((artifact_bundle_dir(evaluation_dir) / "manifest.json").read_text())
    assert "metric_contract_manifest" in manifest
    assert "synthcity_metric_status" in manifest
    assert manifest["source_provenance"]["synthcity"]["revision"] == "fork-sha"
    assert set(manifest["role_context"]["candidate"]["roles"]) == {"train", "tuning"}
    assert set(manifest["role_context"]["full"]["roles"]) == {
        "train",
        "tuning",
        "final_holdout",
    }
    assert manifest["role_context_fingerprint"] == {
        "candidate": "candidate-context",
        "full": "full-context",
    }


def test_metric_status_artifact_round_trips_provenance_fields(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    contract = DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
        framework="synthcity", emitted_key="mixed_mmd.v1"
    )
    context = MetricEvaluationContext(role_hashes={"train": "train-hash", "tuning": "tuning-hash"})
    validation = resolve_metric_observations(
        registry=DEFAULT_METRIC_CONTRACT_REGISTRY,
        model_name="model_a",
        framework="synthcity",
        expected_keys=["mixed_mmd.v1"],
        observations=[
            MetricObservation(
                "model_a",
                "synthcity",
                "mixed_mmd.v1",
                0.2,
                direction=contract.direction,
                role_hashes=dict(context.role_hashes),
                fit_roles=("train",),
                support={"support_contract": contract.required_support},
                bandwidth=1.5,
                provenance={
                    "protocol_version": contract.protocol_version,
                    "seed": contract.seed,
                    "release_transform_digest": contract.release_transform_digest,
                },
            )
        ],
        context=context,
        requested_use="hpo_objective",
    )
    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {},
        native_syntheval_plot_dir=None,
        synthcity_validation_results={"model_a": validation},
        metric_contract_manifest=DEFAULT_METRIC_CONTRACT_REGISTRY.manifest(),
    )
    record = load_synthcity_metric_status(evaluation_dir)["models"]["model_a"]["records"][0]
    assert record["fit_roles"] == ["train"]
    assert record["support"] == {"support_contract": contract.required_support}
    assert record["bandwidth"] == 1.5
    assert record["provenance"]["protocol_version"] == contract.protocol_version


def test_semantic_context_round_trips_and_tampering_is_rejected(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    semantic_context = _semantic_context()
    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {},
        native_syntheval_plot_dir=None,
        semantic_context=semantic_context,
    )

    manifest_path = artifact_bundle_dir(evaluation_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["semantic_context"] == semantic_context
    assert manifest["semantic_context_fingerprint"] == semantic_context_digest(semantic_context)
    validate_evaluation_bundle(
        evaluation_dir,
        expected_semantic_context_fingerprint=semantic_context_digest(semantic_context),
        allow_legacy=True,
    )

    manifest["semantic_context"]["quasi_identifier_columns"] = ["protected"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="semantic_context_fingerprint"):
        validate_evaluation_bundle(evaluation_dir, allow_legacy=True)


def test_metric_status_sidecar_rejects_manifest_semantic_context_mismatch(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    semantic_context = _semantic_context()
    contract_manifest = DEFAULT_METRIC_CONTRACT_REGISTRY.manifest()
    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {},
        native_syntheval_plot_dir=None,
        semantic_context=semantic_context,
        synthcity_validation_results={
            "model_a": _validation_payload("model_a", contract_manifest["digest"]),
        },
        metric_contract_manifest=contract_manifest,
    )

    status_path = artifact_bundle_dir(evaluation_dir) / "synthcity_metric_status.json"
    status_payload = json.loads(status_path.read_text())
    assert status_payload["semantic_context"] == semantic_context
    assert status_payload["semantic_context_fingerprint"] == semantic_context_digest(
        semantic_context
    )
    tampered_context = {
        **semantic_context,
        "quasi_identifier_columns": ["protected"],
    }
    status_payload["semantic_context"] = tampered_context
    status_payload["semantic_context_fingerprint"] = semantic_context_digest(tampered_context)
    status_path.write_text(json.dumps(status_payload, indent=2, sort_keys=True))
    _refresh_manifest_digest(evaluation_dir, "synthcity_metric_status")

    with pytest.raises(ValueError, match="does not match the evaluation artifact manifest"):
        load_synthcity_metric_status(evaluation_dir)


def test_cross_framework_status_sidecars_round_trip(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {},
        native_syntheval_plot_dir=None,
        syntheval_validation_results={
            ("syntheval", "main"): {
                "model_a": _validation_payload("model_a"),
            },
        },
        syntheval_execution_results={
            ("syntheval", "main"): {
                "model_a": _execution_payload("model_a"),
            }
        },
        custom_validation_results={"model_a": _validation_payload("model_a")},
    )

    syntheval_status = load_syntheval_metric_status(evaluation_dir)
    syntheval_execution = load_syntheval_execution(evaluation_dir)
    custom_status = load_custom_metric_status(evaluation_dir)
    assert syntheval_status["passes"]["syntheval:main"]["model_a"]["records"] == []
    execution_row = syntheval_execution["passes"]["syntheval:main"]["model_a"]["metric_executions"][
        0
    ]["normalized_rows_v2"][0]
    assert (
        execution_row["result_metadata"]
        == execution_row["metadata"]
        == {
            "valid_pairs": 1,
            "source": "execution",
        }
    )
    assert custom_status["passes"]["custom:main"]["model_a"]["records"] == []


def test_custom_legacy_model_sidecar_remains_readable(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    contract_manifest = DEFAULT_METRIC_CONTRACT_REGISTRY.manifest()
    custom = _context_validation_payload(
        "model_a", contract_manifest["digest"], {"train": "train", "tuning": "tuning"}
    )
    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {},
        native_syntheval_plot_dir=None,
        custom_validation_results={"model_a": custom},
        metric_contract_manifest=contract_manifest,
    )
    path = artifact_bundle_dir(evaluation_dir) / "custom_metric_status.json"
    payload = json.loads(path.read_text())
    payload["models"] = payload.pop("passes")["custom:main"]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _refresh_manifest_digest(evaluation_dir, "custom_metric_status")

    loaded = load_custom_metric_status(evaluation_dir)
    assert loaded["models"]["model_a"] == custom


def test_custom_pass_sidecar_rejects_mismatched_context(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    contract_manifest = DEFAULT_METRIC_CONTRACT_REGISTRY.manifest()
    custom = _context_validation_payload(
        "model_a", contract_manifest["digest"], {"train": "train", "tuning": "tuning"}
    )
    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {},
        native_syntheval_plot_dir=None,
        custom_validation_results={("custom", "binary_target"): {"model_a": custom}},
        metric_contract_manifest=contract_manifest,
    )

    with pytest.raises(ValueError, match="mismatched execution pass"):
        load_custom_metric_status(evaluation_dir)


def test_metric_status_sidecar_rejects_cross_framework_pass_identity(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {},
        native_syntheval_plot_dir=None,
        syntheval_validation_results={
            ("syntheval", "main"): {"model_a": _validation_payload("model_a")}
        },
    )
    status_path = artifact_bundle_dir(evaluation_dir) / "syntheval_metric_status.json"
    payload = json.loads(status_path.read_text())
    payload["passes"]["custom:main"] = payload["passes"].pop("syntheval:main")
    status_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _refresh_manifest_digest(evaluation_dir, "syntheval_metric_status")

    with pytest.raises(ValueError, match="Invalid syntheval execution pass identity"):
        load_syntheval_metric_status(evaluation_dir)


def test_result_metadata_round_trips_across_framework_status_sidecars(tmp_path):
    role_hashes = {"train": "train-hash", "tuning": "tuning-hash"}
    context = MetricEvaluationContext(role_hashes=role_hashes)
    metadata_by_framework = {
        "synthcity": ("sanity.data_mismatch.score", {"source": "native-synthcity"}),
        "syntheval": ("corr_mat_diff_v2", {"source": "native-syntheval"}),
        "custom": ("log_disparity_mean_abs", {"source": "root-custom"}),
    }
    validations = {}
    for framework, (emitted_key, result_metadata) in metadata_by_framework.items():
        contract = DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
            framework=framework,
            emitted_key=emitted_key,
        )
        observation = MetricObservation(
            model_name="model_a",
            framework=framework,
            emitted_key=emitted_key,
            raw_value=0.25,
            direction=contract.direction,
            role_hashes=role_hashes,
            source_metadata={"result_metadata": result_metadata},
            result_metadata=result_metadata,
        )
        validation = resolve_metric_observations(
            registry=DEFAULT_METRIC_CONTRACT_REGISTRY,
            model_name="model_a",
            framework=framework,
            expected_keys=[emitted_key],
            observations=[observation],
            context=context,
            requested_use="audit",
        )
        validations[framework] = validation

    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {},
        native_syntheval_plot_dir=None,
        synthcity_validation_results={"model_a": validations["synthcity"]},
        syntheval_validation_results={("syntheval", "main"): {"model_a": validations["syntheval"]}},
        custom_validation_results={"model_a": validations["custom"]},
        metric_contract_manifest=DEFAULT_METRIC_CONTRACT_REGISTRY.manifest(),
    )

    loaded_by_framework = {
        "synthcity": load_synthcity_metric_status(evaluation_dir)["models"]["model_a"]["records"],
        "syntheval": load_syntheval_metric_status(evaluation_dir)["passes"]["syntheval:main"][
            "model_a"
        ]["records"],
        "custom": load_custom_metric_status(evaluation_dir)["passes"]["custom:main"]["model_a"][
            "records"
        ],
    }
    for framework, (emitted_key, result_metadata) in metadata_by_framework.items():
        record = next(
            item for item in loaded_by_framework[framework] if item["expected_key"] == emitted_key
        )
        assert record["result_metadata"] == result_metadata
        assert record["source_metadata"]["result_metadata"] == result_metadata


def test_failed_syntheval_execution_evidence_round_trips_without_becoming_cacheable(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    failed_execution = _failed_execution_payload(
        model_name="model_a",
        pass_name="main",
        target_view="native",
        expected_manifest_digest="execution-digest",
        expected_output_manifest={"statistics": ("corr_mat_diff_v2",)},
        context_fingerprint="context-digest",
        role_context={
            "schema_version": "evaluation-role-context-v1",
            "fit_roles": ["train"],
            "evidence_role": "tuning",
            "fit_frame": "fit-frame",
            "evidence_frame": "evidence-frame",
        },
        group_context=None,
        failure_status={"exit_code": 1, "failure_reason": "worker failed"},
    )

    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {},
        native_syntheval_plot_dir=None,
        syntheval_execution_results={
            ("syntheval", "main"): {"model_a": failed_execution},
        },
    )

    loaded = load_syntheval_execution(evaluation_dir)
    payload = loaded["passes"]["syntheval:main"]["model_a"]
    assert payload["execution_succeeded"] is False
    assert payload["metric_executions"][0]["status"]["state"] == "failed"


def test_partial_syntheval_execution_evidence_round_trips_with_sibling_rows(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    partial_execution = _execution_payload("model_a")
    partial_execution.update(
        {
            "execution_succeeded": False,
            "failure_reason": "worker failed after one metric completed",
            "worker_exit": {
                "state": "failed",
                "exit_code": 1,
                "error_type": "RuntimeError",
                "reason_code": "unknown_exception",
                "failure_reason": "SynthEval worker failed with an unknown error.",
                "exception_type": "RuntimeError",
                "failed_at": 1.0,
            },
        }
    )
    partial_execution["metric_executions"].append(
        {
            "method": "ks_test",
            "status": {
                "method": "ks_test",
                "state": "failed",
                "expected_keys": ["ks_tvd_stat_v2"],
                "observed_keys": [],
                "completed_keys": [],
                "failed_keys": ["ks_tvd_stat_v2"],
                "missing_keys": ["ks_tvd_stat_v2"],
                "duplicate_keys": [],
                "non_finite_keys": [],
                "unexpected_keys": [],
                "warnings": [],
                "error_type": "ValueError",
                "reason_code": "unknown_exception",
                "failure_reason": "SynthEval worker failed with an unknown error.",
            },
            "normalized_rows": [],
            "normalized_rows_v2": [],
        }
    )

    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {},
        native_syntheval_plot_dir=None,
        syntheval_execution_results={
            ("syntheval", "main"): {"model_a": partial_execution},
        },
    )

    loaded = load_syntheval_execution(evaluation_dir)
    payload = loaded["passes"]["syntheval:main"]["model_a"]
    methods = {item["method"]: item for item in payload["metric_executions"]}
    assert payload["execution_succeeded"] is False
    assert methods["statistics"]["normalized_rows_v2"][0]["metric"] == "corr_mat_diff_v2"
    assert methods["ks_test"]["status"]["failed_keys"] == ["ks_tvd_stat_v2"]


def test_group_unsafe_syntheval_status_round_trips_as_audit_evidence(tmp_path):
    results = _dataframe({("avg_dwm_diff", "value"): [0.2]}, index=pd.Index(["model_a"]))
    results.columns = pd.MultiIndex.from_tuples(results.columns)
    ranks = _dataframe({"avg_dwm_diff": [0.2]}, index=pd.Index(["model_a"]))
    validations = validate_syntheval_results(
        results,
        ranks,
        {"syntheval": ["avg_dwm_diff"]},
        role_hashes={"train": "train-hash", "tuning": "tuning-hash"},
        model_names=["model_a"],
        requested_use="audit",
        population_unit="patient_group",
        group_mode="patient_group",
        resolved_configuration={"group_column": "patient_id"},
    )

    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {},
        native_syntheval_plot_dir=None,
        syntheval_validation_results=validations,
    )

    loaded = load_syntheval_metric_status(evaluation_dir)
    records = loaded["passes"]["syntheval:main"]["model_a"]["records"]
    record = next(item for item in records if item["expected_key"] == "avg_dwm_diff")
    assert record["status"] == "group_unsafe"
    assert record["is_expected"] is True


def test_group_unsafe_synthcity_status_round_trips_as_audit_evidence(tmp_path):
    report = _dataframe(
        {
            "mean": [float("nan")],
            "errors": [1],
            "direction": ["maximize"],
        },
        index=["sanity.common_rows_proportion.score"],
    )
    report.attrs["group_safety"] = {
        "schema_version": "group-safety-v1",
        "status": "group_unsafe",
        "group_mode": "patient_group",
        "task_type": "classification",
        "reason": "unsupported grouped modality",
        "loader_types": {"X_gt": "syn_seq", "X_syn": "syn_seq"},
        "metrics": ["sanity.common_rows_proportion"],
    }
    validations = validate_synthcity_results(
        {"model_a": report},
        {"sanity": ["common_rows_proportion"]},
        role_hashes={"train": "train-hash", "tuning": "tuning-hash"},
        context=MetricEvaluationContext(
            role_hashes={"train": "train-hash", "tuning": "tuning-hash"},
            population_unit="patient_group",
            group_mode="patient_group",
        ),
    )

    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {},
        native_syntheval_plot_dir=None,
        synthcity_validation_results=validations,
        metric_contract_manifest=DEFAULT_METRIC_CONTRACT_REGISTRY.manifest(),
    )

    loaded = load_synthcity_metric_status(evaluation_dir)
    records = loaded["models"]["model_a"]["records"]
    record = next(
        item for item in records if item["expected_key"] == "sanity.common_rows_proportion.score"
    )
    assert record["status"] == "group_unsafe"
    assert record["is_expected"] is True
    assert record["source_metadata"]["group_safety"]["reason"] == ("unsupported grouped modality")


def _persist_strict_bundle(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    combined = _combined()
    combined.to_csv(evaluation_dir / "combined_evaluation.csv")
    contract_manifest = DEFAULT_METRIC_CONTRACT_REGISTRY.manifest()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("evaluation:\n  group_mode: row\n")
    role_hashes = {"train": "train-hash", "tuning": "tuning-hash"}
    persist_evaluation_artifacts(
        evaluation_dir,
        combined,
        {},
        native_syntheval_plot_dir=None,
        synthcity_validation_results={
            "model_a": _context_validation_payload(
                "model_a",
                contract_manifest["digest"],
                role_hashes,
            )
        },
        metric_contract_manifest=contract_manifest,
        source_provenance={
            "config_digest": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "metric_contract_digest": contract_manifest["digest"],
        },
        role_context={"candidate": _role_context(["train", "tuning"])},
        role_context_fingerprint={"candidate": "candidate-context"},
    )
    return evaluation_dir, config_path, role_hashes


def test_strict_bundle_validation_accepts_matching_context(tmp_path):
    evaluation_dir, config_path, role_hashes = _persist_strict_bundle(tmp_path)

    manifest = validate_evaluation_bundle(
        evaluation_dir,
        expected_config_path=config_path,
        expected_role_context_fingerprints={"candidate": "candidate-context"},
        expected_role_hashes=role_hashes,
        expected_population_unit="row",
        expected_group_mode="row",
    )

    assert manifest["combined_evaluation"]["models"] == ["model_a"]


def test_strict_bundle_validation_rejects_status_identity_missing_from_table(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    combined = _combined()
    metric_column = ("synthcity", "utility", "stats.ks_test.marginal")
    combined[metric_column] = [0.5]
    combined.columns = pd.MultiIndex.from_tuples(combined.columns)
    combined.to_csv(evaluation_dir / "combined_evaluation.csv")

    registry_manifest = DEFAULT_METRIC_CONTRACT_REGISTRY.manifest()
    validation = resolve_metric_observations(
        registry=DEFAULT_METRIC_CONTRACT_REGISTRY,
        model_name="model_a",
        framework="synthcity",
        expected_keys=["stats.ks_test.marginal"],
        observations=[
            MetricObservation(
                model_name="model_a",
                framework="synthcity",
                emitted_key="stats.ks_test.marginal",
                raw_value=0.5,
                direction="maximize",
                role_hashes={"train": "train-hash", "tuning": "tuning-hash"},
            )
        ],
        context=MetricEvaluationContext(
            role_hashes={"train": "train-hash", "tuning": "tuning-hash"},
            resolved_configuration={"protocol": "test-v1"},
        ),
        requested_use="audit",
    )
    persist_evaluation_artifacts(
        evaluation_dir,
        combined,
        {},
        native_syntheval_plot_dir=None,
        synthcity_validation_results={"model_a": validation},
        metric_contract_manifest=registry_manifest,
    )

    tampered = combined.drop(columns=[metric_column])
    tampered.to_csv(evaluation_dir / "combined_evaluation.csv")
    manifest_path = artifact_bundle_dir(evaluation_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    combined_entry = manifest["combined_evaluation"]
    combined_entry["sha256"] = hashlib.sha256(
        (evaluation_dir / combined_entry["path"]).read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    with pytest.raises(ValueError, match="absent from the combined evaluation table"):
        validate_evaluation_bundle(evaluation_dir)


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("registry", "current registry"),
        ("config", "configuration digest"),
        ("role", "role hashes"),
        ("pass", "validation pass"),
    ),
)
def test_strict_bundle_validation_rejects_hash_consistent_semantic_tampering(
    tmp_path, tamper, message
):
    evaluation_dir, config_path, role_hashes = _persist_strict_bundle(tmp_path)
    manifest_path = artifact_bundle_dir(evaluation_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    if tamper == "registry":
        empty_manifest = MetricContractRegistry(()).manifest()
        contract_path = artifact_bundle_dir(evaluation_dir) / "metric_contract_manifest.json"
        contract_path.write_text(json.dumps(empty_manifest, indent=2, sort_keys=True))
        contract_entry = manifest["metric_contract_manifest"]
        contract_entry["sha256"] = hashlib.sha256(contract_path.read_bytes()).hexdigest()
        contract_entry["digest"] = empty_manifest["digest"]
        manifest["source_provenance"]["metric_contract_digest"] = empty_manifest["digest"]
        status_path = artifact_bundle_dir(evaluation_dir) / "synthcity_metric_status.json"
        status_payload = json.loads(status_path.read_text())
        status_payload["models"]["model_a"]["contract_digest"] = empty_manifest["digest"]
        status_path.write_text(json.dumps(status_payload, indent=2, sort_keys=True))
    elif tamper == "config":
        manifest["source_provenance"]["config_digest"] = "0" * 64
    else:
        status_path = artifact_bundle_dir(evaluation_dir) / "synthcity_metric_status.json"
        status_payload = json.loads(status_path.read_text())
        context = status_payload["models"]["model_a"]["evaluation_context"]
        if tamper == "role":
            context["role_hashes"]["train"] = "tampered-train-hash"
        else:
            context["execution_pass"] = "binary_target"
        status_path.write_text(json.dumps(status_payload, indent=2, sort_keys=True))

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    if tamper in {"registry", "role", "pass"}:
        _refresh_manifest_digest(evaluation_dir, "synthcity_metric_status")

    with pytest.raises(ValueError, match=message):
        validate_evaluation_bundle(
            evaluation_dir,
            expected_config_path=config_path,
            expected_role_context_fingerprints={"candidate": "candidate-context"},
            expected_role_hashes=role_hashes,
            expected_population_unit="row",
            expected_group_mode="row",
        )


def test_final_holdout_evidence_round_trip(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    refit = _final_refit_files(tmp_path)
    full_role_context = _role_context(["train", "tuning", "final_holdout"])
    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {},
        native_syntheval_plot_dir=None,
        final_holdout_evidence={
            **_complete_final_evidence_fields(),
            "evaluation_role": "final_holdout",
            "state": "succeeded",
            "selected_model": "model_a",
            "evidence_role": "final_holdout",
            "fit_roles": ["train", "tuning"],
            "role_context": full_role_context,
            "role_context_fingerprint": "full-context",
            "candidate_selection": {
                "source": "combined_evaluation.csv",
                "model": "model_a",
                "overall_rank": 1.0,
            },
            "role_hashes": {"train": "train-hash", "final_holdout": "holdout-hash"},
            "final_refit": refit,
        },
        role_context={"full": full_role_context},
        role_context_fingerprint={"full": "full-context"},
    )

    evidence = load_final_holdout_evidence(evaluation_dir)

    assert evidence["schema_version"] == "final-holdout-evidence-v1"
    assert evidence["evaluation_role"] == "final_holdout"
    assert evidence["selected_model"] == "model_a"
    assert evidence["final_refit"]["fit_roles"] == ["train", "tuning"]
    serialized = (artifact_bundle_dir(evaluation_dir) / "final_holdout_evidence.json").read_text()
    assert str(tmp_path) not in serialized
    assert evidence["final_refit"]["path"] == "evaluation_artifacts-v1/final_refit/data.csv"
    assert len(evidence["final_refit"]["data_sha256"]) == 64

    bundle_data_path = artifact_bundle_dir(evaluation_dir) / "final_refit" / "data.csv"
    bundle_data_path.write_text("feature,target\n2,1\n")
    with pytest.raises(ValueError, match="Final refit synthetic data failed integrity"):
        load_final_holdout_evidence(evaluation_dir)

    bundle_data_path.write_text("feature,target\n1,0\n")
    (artifact_bundle_dir(evaluation_dir) / "final_refit" / "metadata.json").write_text(
        '{"schema_version": "final-refit-v1", "cache_key": "tampered-key"}'
    )
    with pytest.raises(ValueError, match="Final refit metadata failed integrity"):
        load_final_holdout_evidence(evaluation_dir)


def test_final_holdout_persistence_rejects_missing_generator_metadata(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    refit = _final_refit_files(tmp_path)
    refit.pop("generator_metadata")
    metadata_path = Path(refit["metadata_path"])
    metadata_payload = json.loads(metadata_path.read_text())
    metadata_payload.pop("generator_metadata")
    metadata_path.write_text(json.dumps(metadata_payload))
    full_role_context = _role_context(["train", "tuning", "final_holdout"])

    with pytest.raises(ValueError, match="requires complete generator_metadata"):
        persist_evaluation_artifacts(
            evaluation_dir,
            _combined(),
            {},
            native_syntheval_plot_dir=None,
            final_holdout_evidence={
                "evaluation_role": "final_holdout",
                "state": "succeeded",
                "selected_model": "model_a",
                "evidence_role": "final_holdout",
                "fit_roles": ["train", "tuning"],
                "role_context": full_role_context,
                "role_context_fingerprint": "full-context",
                "candidate_selection": {
                    "source": "combined_evaluation.csv",
                    "model": "model_a",
                    "overall_rank": 1.0,
                },
                "final_refit": refit,
            },
            role_context={"full": full_role_context},
            role_context_fingerprint={"full": "full-context"},
        )


def test_final_holdout_loader_rejects_hash_consistent_semantic_tampering(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    combined = _combined()
    combined.to_csv(evaluation_dir / "combined_evaluation.csv")
    full_role_context = _role_context(["train", "tuning", "final_holdout"])
    persist_evaluation_artifacts(
        evaluation_dir,
        combined,
        {},
        native_syntheval_plot_dir=None,
        final_holdout_evidence={
            **_complete_final_evidence_fields(),
            "evaluation_role": "final_holdout",
            "state": "succeeded",
            "selected_model": "model_a",
            "evidence_role": "final_holdout",
            "fit_roles": ["train", "tuning"],
            "role_context": full_role_context,
            "role_context_fingerprint": "full-context",
            "candidate_selection": {
                "source": "combined_evaluation.csv",
                "model": "model_a",
                "overall_rank": 1.0,
            },
            "final_refit": _final_refit_files(tmp_path),
        },
        role_context={"full": full_role_context},
        role_context_fingerprint={"full": "full-context"},
    )

    evidence_path = artifact_bundle_dir(evaluation_dir) / "final_holdout_evidence.json"
    payload = json.loads(evidence_path.read_text())
    payload["state"] = "blocked"
    evidence_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _refresh_manifest_digest(evaluation_dir, "final_holdout_evidence")

    with pytest.raises(ValueError, match="contradictory state dimensions"):
        load_final_holdout_evidence(evaluation_dir)


@pytest.mark.parametrize(
    "field",
    [
        "evidence_execution_state",
        "metric_completeness_state",
        "score_completeness_state",
        "audit_outcome_state",
        "provenance_inventory",
    ],
)
def test_final_holdout_loader_rejects_omitted_task15_fields(tmp_path, field):
    evaluation_dir, _refit = _persist_current_final_refit_bundle(tmp_path)
    evidence_path = artifact_bundle_dir(evaluation_dir) / "final_holdout_evidence.json"
    payload = json.loads(evidence_path.read_text())
    payload.pop(field)
    evidence_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _refresh_manifest_digest(evaluation_dir, "final_holdout_evidence")

    with pytest.raises(ValueError, match="(state|provenance_inventory)"):
        load_final_holdout_evidence(evaluation_dir)


def test_final_holdout_loader_rejects_manifest_semantic_context_mismatch(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    semantic_context = _semantic_context()
    full_role_context = _role_context(["train", "tuning", "final_holdout"])
    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {},
        native_syntheval_plot_dir=None,
        semantic_context=semantic_context,
        final_holdout_evidence={
            **_complete_final_evidence_fields(),
            "evaluation_role": "final_holdout",
            "state": "succeeded",
            "selected_model": "model_a",
            "evidence_role": "final_holdout",
            "fit_roles": ["train", "tuning"],
            "semantic_context": semantic_context,
            "semantic_context_fingerprint": semantic_context_digest(semantic_context),
            "role_context": full_role_context,
            "role_context_fingerprint": "full-context",
            "candidate_selection": {
                "source": "combined_evaluation.csv",
                "model": "model_a",
                "overall_rank": 1.0,
            },
            "final_refit": _final_refit_files(tmp_path),
        },
        role_context={"full": full_role_context},
        role_context_fingerprint={"full": "full-context"},
    )

    evidence_path = artifact_bundle_dir(evaluation_dir) / "final_holdout_evidence.json"
    payload = json.loads(evidence_path.read_text())
    tampered_context = {
        **semantic_context,
        "quasi_identifier_columns": ["protected"],
    }
    payload["semantic_context"] = tampered_context
    payload["semantic_context_fingerprint"] = semantic_context_digest(tampered_context)
    evidence_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _refresh_manifest_digest(evaluation_dir, "final_holdout_evidence")

    with pytest.raises(ValueError, match="does not match the evaluation artifact manifest"):
        load_final_holdout_evidence(evaluation_dir)


def test_bundle_validation_rejects_tampered_final_holdout_evidence(tmp_path):
    evaluation_dir, _refit = _persist_current_final_refit_bundle(tmp_path)

    validate_evaluation_bundle(evaluation_dir, allow_legacy=True)
    evidence_path = artifact_bundle_dir(evaluation_dir) / "final_holdout_evidence.json"
    payload = json.loads(evidence_path.read_text())
    payload["state"] = "blocked"
    evidence_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _refresh_manifest_digest(evaluation_dir, "final_holdout_evidence")

    with pytest.raises(ValueError, match="semantic sidecar validation failed"):
        validate_evaluation_bundle(evaluation_dir, allow_legacy=True)


def test_bundle_validation_rejects_tampered_release_score_evidence(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    combined = _combined()
    combined.to_csv(evaluation_dir / "combined_evaluation.csv")
    refit, candidate_role_context, semantic_context = _current_final_refit_files(tmp_path)
    full_role_context = _role_context(["train", "tuning", "final_holdout"])
    persist_evaluation_artifacts(
        evaluation_dir,
        combined,
        {},
        native_syntheval_plot_dir=None,
        final_holdout_evidence={
            **_complete_final_evidence_fields(),
            "evaluation_role": "final_holdout",
            "state": "succeeded",
            "selected_model": "model_a",
            "evidence_role": "final_holdout",
            "fit_roles": ["train", "tuning"],
            "semantic_context": semantic_context,
            "semantic_context_fingerprint": semantic_context_digest(semantic_context),
            "role_context": full_role_context,
            "role_context_fingerprint": artifacts._mapping_digest(full_role_context),
            "candidate_selection": {
                "source": "combined_evaluation.csv",
                "model": "model_a",
                "overall_rank": 1.0,
            },
            "final_refit": refit,
        },
        release_score={
            "model_a": {
                "status": "succeeded",
                "score": 0.8125,
                "dimensions": {
                    "utility": {
                        "score": 0.8,
                        "components": {
                            key: {"score": 0.8, "status": "succeeded"}
                            for key in ("tstr", "mmd", "jsd")
                        },
                    },
                    "privacy": {
                        "score": 0.9,
                        "components": {
                            key: {"score": 0.8, "status": "succeeded"}
                            for key in ("k", "l", "dcr", "epsilon", "mia", "attribute")
                        },
                    },
                    "fairness": {
                        "score": 0.7,
                        "components": {
                            key: {"score": 0.8, "status": "succeeded"}
                            for key in ("representation", "eo", "worst_log_disparity")
                        },
                    },
                },
                "audit_only": True,
            }
        },
        role_context={"candidate": candidate_role_context, "full": full_role_context},
        role_context_fingerprint={
            "candidate": artifacts._mapping_digest(candidate_role_context),
            "full": artifacts._mapping_digest(full_role_context),
        },
        semantic_context=semantic_context,
    )

    validate_evaluation_bundle(evaluation_dir, allow_legacy=True)
    score_path = artifact_bundle_dir(evaluation_dir) / "release_score_evidence.json"
    payload = json.loads(score_path.read_text())
    payload["selected_model"] = "unknown-model"
    score_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _refresh_manifest_digest(evaluation_dir, "release_score_evidence")

    with pytest.raises(ValueError, match="semantic sidecar validation failed"):
        validate_evaluation_bundle(evaluation_dir, allow_legacy=True)


def test_sidecar_integrity_checks_reject_tampering(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {},
        native_syntheval_plot_dir=None,
        synthcity_validation_results={"model_a": {"records": []}},
        syntheval_validation_results={
            ("syntheval", "main"): {"model_a": {"records": []}},
        },
        custom_validation_results={"model_a": {"records": []}},
        metric_contract_manifest=_contract_manifest(),
    )

    status_path = artifact_bundle_dir(evaluation_dir) / "synthcity_metric_status.json"
    status_path.write_text(
        status_path.read_text().replace('"records": []', '"records": ["tampered"]')
    )
    with pytest.raises(ValueError, match="status sidecar failed integrity"):
        load_synthcity_metric_status(evaluation_dir)

    syntheval_path = artifact_bundle_dir(evaluation_dir) / "syntheval_metric_status.json"
    syntheval_path.write_text(
        syntheval_path.read_text().replace('"records": []', '"records": ["tampered"]')
    )
    with pytest.raises(ValueError, match="status sidecar failed integrity"):
        load_syntheval_metric_status(evaluation_dir)

    custom_path = artifact_bundle_dir(evaluation_dir) / "custom_metric_status.json"
    custom_path.write_text(
        custom_path.read_text().replace('"records": []', '"records": ["tampered"]')
    )
    with pytest.raises(ValueError, match="status sidecar failed integrity"):
        load_custom_metric_status(evaluation_dir)

    contract_path = artifact_bundle_dir(evaluation_dir) / "metric_contract_manifest.json"
    contract_path.write_text(
        contract_path.read_text().replace(
            f'"digest": "{_contract_manifest()["digest"]}"', '"digest": "tampered"'
        )
    )
    with pytest.raises(ValueError, match="contract manifest sidecar failed integrity"):
        load_metric_contract_manifest(evaluation_dir)


def test_manifest_loader_rejects_hash_consistent_combined_schema_tampering(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    combined = _combined()
    combined.to_csv(evaluation_dir / "combined_evaluation.csv")
    persist_evaluation_artifacts(
        evaluation_dir,
        combined,
        {},
        native_syntheval_plot_dir=None,
    )

    combined.drop(columns=[("__all__", "overall", "rank")]).to_csv(
        evaluation_dir / "combined_evaluation.csv"
    )
    _refresh_manifest_digest(evaluation_dir, "combined_evaluation")

    with pytest.raises(ValueError, match="semantically invalid"):
        load_log_disparity_reports(evaluation_dir)


def test_status_loader_rejects_hash_consistent_derived_field_tampering(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {},
        native_syntheval_plot_dir=None,
        synthcity_validation_results={"model_a": _validation_payload("model_a")},
    )

    status_path = artifact_bundle_dir(evaluation_dir) / "synthcity_metric_status.json"
    payload = json.loads(status_path.read_text())
    payload["models"]["model_a"]["decision_status"] = "eligible"
    status_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _refresh_manifest_digest(evaluation_dir, "synthcity_metric_status")

    with pytest.raises(ValueError, match="decision_status does not match"):
        load_synthcity_metric_status(evaluation_dir)


def test_contract_loader_rejects_hash_consistent_digest_tampering(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {},
        native_syntheval_plot_dir=None,
        metric_contract_manifest=_contract_manifest(),
    )

    contract_path = artifact_bundle_dir(evaluation_dir) / "metric_contract_manifest.json"
    payload = json.loads(contract_path.read_text())
    payload["digest"] = "0" * 64
    contract_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _refresh_manifest_digest(evaluation_dir, "metric_contract_manifest")

    with pytest.raises(ValueError, match="digest does not match its contracts"):
        load_metric_contract_manifest(evaluation_dir)


def test_execution_loader_rejects_hash_consistent_structural_tampering(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {},
        native_syntheval_plot_dir=None,
        syntheval_execution_results={
            ("syntheval", "main"): {"model_a": _execution_payload("model_a")}
        },
    )

    execution_path = artifact_bundle_dir(evaluation_dir) / "syntheval_execution.json"
    payload = json.loads(execution_path.read_text())
    payload["passes"]["syntheval:main"]["model_a"]["execution_succeeded"] = False
    execution_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _refresh_manifest_digest(evaluation_dir, "syntheval_execution")

    with pytest.raises(ValueError, match="structured execution validation"):
        load_syntheval_execution(evaluation_dir)


def test_execution_loader_rejects_manifest_semantic_context_mismatch(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    semantic_context = _semantic_context()
    execution = _execution_payload("model_a")
    execution["semantic_context"] = semantic_context
    execution["semantic_context_digest"] = semantic_context_digest(semantic_context)
    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {},
        native_syntheval_plot_dir=None,
        semantic_context=semantic_context,
        syntheval_execution_results={
            ("syntheval", "main"): {"model_a": execution},
        },
    )

    execution_path = artifact_bundle_dir(evaluation_dir) / "syntheval_execution.json"
    payload = json.loads(execution_path.read_text())
    tampered_context = {
        **semantic_context,
        "quasi_identifier_columns": ["protected"],
    }
    tampered_execution = payload["passes"]["syntheval:main"]["model_a"]
    tampered_execution["semantic_context"] = tampered_context
    tampered_execution["semantic_context_digest"] = semantic_context_digest(tampered_context)
    execution_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _refresh_manifest_digest(evaluation_dir, "syntheval_execution")

    with pytest.raises(ValueError, match="does not match the evaluation artifact manifest"):
        load_syntheval_execution(evaluation_dir)


def test_log_disparity_loader_rejects_hash_consistent_schema_tampering(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {"model_a": _report()},
        native_syntheval_plot_dir=None,
    )

    manifest_path = artifact_bundle_dir(evaluation_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["log_disparity"]["model_a"]
    table_path = artifact_bundle_dir(evaluation_dir) / entry["path"] / "subgroup_table.parquet"
    table = pd.read_parquet(table_path).drop(columns=["EquityColor"])
    table.to_parquet(table_path)
    entry["tables"]["subgroup_table"]["sha256"] = hashlib.sha256(
        table_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    with pytest.raises(ValueError, match="EquityColor"):
        load_log_disparity_reports(evaluation_dir)


def test_source_provenance_records_editable_fork_identity():
    provenance = collect_source_provenance()

    assert provenance["schema_version"] == "source-provenance-v3"
    assert provenance["plan02_governance"]["schema_version"] == (
        "evaluation-modernization-02-governance-v1"
    )
    assert provenance["plan02_governance"]["baseline_revision"] == (
        "0ef2950c8b9991c2742c90bed849a3c3b647f61c"
    )
    assert provenance["package"]["name"] == "synthdata"
    assert provenance["synthcity"]["revision"]
    assert provenance["synthcity"]["package_version"]
    assert provenance["synthcity"]["baseline_revision"] == (
        "0ef2950c8b9991c2742c90bed849a3c3b647f61c"
    )
    assert provenance["synthcity"]["baseline_source"] == (
        "user_confirmed_committed_pre_refactor_baseline"
    )
    assert "starting_revision" not in provenance["synthcity"]
    assert "dirty" in provenance["synthcity"]
    assert len(provenance["synthcity"]["worktree_content_digest"]) == 64
    assert provenance["synthcity"]["attribution"]["status"] == "passed"
    assert len(provenance["synthcity"]["attribution"]["license_digests"]["LICENSE"]) == 64
    assert provenance["synthcity"]["fork_repairs"]
    assert all(repair["tracking_reference"] for repair in provenance["synthcity"]["fork_repairs"])
    assert provenance["syntheval"]["revision"]
    assert provenance["syntheval"]["baseline_revision"] is None
    assert provenance["syntheval"]["baseline_source"] == "not_recorded_for_plan_02"
    assert len(provenance["syntheval"]["worktree_content_digest"]) == 64
    assert provenance["syntheval"]["attribution"]["status"] == "passed"
    assert all(repair["tracking_reference"] for repair in provenance["syntheval"]["fork_repairs"])


def test_source_provenance_rejects_invalid_governance_ledger(tmp_path):
    governance_path = tmp_path / "governance.json"
    governance_path.write_text(
        json.dumps(
            {
                "schema_version": "evaluation-modernization-02-governance-v1",
            }
        )
    )

    with pytest.raises(ValueError, match="governance ledger plan"):
        collect_source_provenance(governance_path=governance_path)


def test_legacy_source_provenance_schema_remains_readable(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _combined().to_csv(evaluation_dir / "combined_evaluation.csv")
    persist_evaluation_artifacts(
        evaluation_dir,
        _combined(),
        {},
        native_syntheval_plot_dir=None,
        source_provenance={
            "schema_version": "source-provenance-v2",
            "synthcity": {"revision": "legacy-fork-sha"},
        },
    )

    manifest = validate_evaluation_bundle(evaluation_dir, allow_legacy=True)

    assert manifest["source_provenance"]["schema_version"] == "source-provenance-v2"
