"""Focused orchestration tests for contract-aware evaluation outputs."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from synthdata.data import dataframe_fingerprint
from synthdata.evaluation import (
    _final_holdout_state_dimensions,
    _generation_metadata,
    _release_score_inputs,
    _select_policy_model,
    _synthcity_semantic_context,
    artifacts,
    custom_eval,
    privacy_gate,
    run_evaluation,
    synthcity_eval,
    syntheval_eval,
    task12_eval,
)
from synthdata.evaluation import tstr as tstr_module
from synthdata.evaluation.release import transform_release_roles
from synthdata.evaluation.release_score import compute_release_score
from synthdata.generation import pipeline as generation_pipeline

pytestmark = pytest.mark.unit


def _dataframe(
    data: dict[object, list[object]], index: list[str] | pd.Index | None = None
) -> pd.DataFrame:
    """Construct test frames while keeping pandas' heterogeneous columns typed."""
    return pd.DataFrame(
        cast("dict[str, list[object]]", data),
        index=pd.Index(index) if index is not None else None,
    )


def test_metric_validation_failure_keeps_finite_release_score_indeterminate():
    state = _final_holdout_state_dimensions(
        [{"framework": "syntheval", "model": "model_a"}],
        {"status": "succeeded", "R_final": 0.42},
    )

    assert state == ("failed", "incomplete", "indeterminate", "failed")


def _fake_refit_metadata(synthetic, output_dir):
    output_dir = output_dir if isinstance(output_dir, Path) else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "model_a.csv"
    metadata_path = output_dir / "model_a.cache.json"
    generator_metadata = {
        "schema_version": "generator-metadata-v1",
        "generator_context": {"privacy_claim_type": "none"},
        "plugin_name": "test_generator",
        "plugin_fqdn": "test.generator",
        "requested_parameters": {},
        "n_samples": len(synthetic),
        "random_state": 0,
        "privacy_accounting": None,
    }
    synthetic.to_csv(data_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": "final-refit-v1",
                "cache_key": "refit-key",
                "generator_metadata": generator_metadata,
            }
        )
    )
    return synthetic.copy(), {
        "fit_frame_fingerprint": "refit-imputed",
        "fit_frame_fingerprints": {"raw": "refit-raw", "imputed": "refit-imputed"},
        "input_role_hashes": {"raw": {}, "imputed": {}},
        "path": str(data_path),
        "metadata_path": str(metadata_path),
        "cache_key": "refit-key",
        "fit_roles": ["train", "tuning"],
        "generator_metadata": generator_metadata,
    }


def _fake_refit_metadata_with_hash(synthetic, output_dir, refit_hash):
    refit_frame, metadata = _fake_refit_metadata(synthetic, output_dir)
    metadata["fit_frame_fingerprint"] = refit_hash
    metadata["fit_frame_fingerprints"] = {"raw": refit_hash, "imputed": refit_hash}
    return refit_frame, metadata


def test_run_evaluation_validates_ranks_and_persists_status(
    make_config, make_canonical_dataset, monkeypatch
):
    cfg = make_config()
    cfg.evaluation.syntheval.enabled = True
    cfg.evaluation.custom.enabled = False
    cfg.evaluation.save_per_model_syntheval_plots = False
    cfg.evaluation.generate_report = False
    cfg.evaluation.privacy_gate.enabled = False

    dataset = make_canonical_dataset()
    dataset.quasi_identifier_columns = ["feature"]
    dataset.variable_schema = {
        column: {
            "kind": "categorical" if column == "target" else "continuous",
            "source_table": "measurements" if column == "feature" else None,
        }
        for column in dataset.full_df.columns
    }
    synthetic = dataset.role_frame("train", imputed=True).copy()
    synthcity_report = _dataframe(
        {"mean": [0.25], "direction": ["minimize"]},
        index=["privacy.identifiability_score.score_OC"],
    )

    received_feature_types = {}
    received_semantics = {}
    received_semantic_context = None

    def fake_run_synthcity_evaluation(*args, **kwargs):
        nonlocal received_semantic_context
        received_feature_types.update(kwargs["feature_types"])
        received_semantics.update(
            {
                "quasi_identifier_columns": kwargs["quasi_identifier_columns"],
                "source_table": kwargs["source_table"],
                "sensitive_target_types": kwargs["sensitive_target_types"],
                "classification_score": kwargs["classification_score"],
            }
        )
        received_semantic_context = kwargs["semantic_context"]
        return {"model_a": synthcity_report}

    monkeypatch.setattr(synthcity_eval, "run_synthcity_evaluation", fake_run_synthcity_evaluation)
    monkeypatch.setattr(
        syntheval_eval,
        "run_syntheval_evaluation",
        lambda *args, **kwargs: (None, None),
    )
    monkeypatch.setattr(
        custom_eval,
        "run_log_disparity_evaluation",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        privacy_gate,
        "evaluate_privacy_gate",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        generation_pipeline,
        "refit_selected_model",
        lambda *args, **kwargs: _fake_refit_metadata(synthetic, kwargs["output_dir"]),
    )

    combined, extras = run_evaluation(cfg, dataset, {"model_a": synthetic})

    assert received_feature_types == {
        column: "categorical" if column == "target" else "continuous"
        for column in dataset.full_df.columns
    }
    assert received_semantics == {
        "quasi_identifier_columns": ["feature"],
        "source_table": {"feature": "measurements"},
        "sensitive_target_types": {"protected": "continuous"},
        "classification_score": "balanced_accuracy",
    }
    assert received_semantic_context == {
        **_synthcity_semantic_context(dataset, cfg.evaluation.synthcity),
    }
    metric_column = (
        "synthcity",
        "privacy",
        "privacy.identifiability_score.score_OC",
    )
    assert combined.loc["model_a", metric_column] == pytest.approx(0.25)
    assert pd.isna(combined.loc["model_a", ("__all__", "overall", "rank")])
    status_counts = extras["synthcity_validation"]["model_a"]["status_counts"]
    assert status_counts["succeeded"] == 1
    assert status_counts["missing"] > 0
    assert extras["synthcity_validation"]["model_a"]["decision_eligible"] is False

    status = artifacts.load_synthcity_metric_status(cfg.evaluation.output_dir)
    assert status["models"]["model_a"]["decision_eligible"] is False
    manifest = extras["artifact_manifest"]
    assert manifest.endswith("evaluation_artifacts-v1/manifest.json")


def test_run_evaluation_persists_generator_metadata_sidecars(
    make_config, make_canonical_dataset, monkeypatch
):
    cfg = make_config()
    cfg.evaluation.syntheval.enabled = False
    cfg.evaluation.custom.enabled = True
    cfg.evaluation.save_per_model_syntheval_plots = False
    cfg.evaluation.generate_report = False
    cfg.evaluation.privacy_gate.enabled = False

    dataset = make_canonical_dataset()
    synthetic = dataset.role_frame("train", imputed=True).copy()
    generation_dir = Path(cfg.generation.output_dir)
    generation_dir.mkdir(parents=True, exist_ok=True)
    generator_metadata = {
        "schema_version": "generator-metadata-v1",
        "generator_context": {
            "privacy_claim_type": "formal_dp",
            "requested_accounting": {
                "epsilon": 1.0,
                "delta": 1e-6,
                "alpha": 100,
                "lamda": 0.1,
            },
        },
        "plugin_name": "pategan",
        "plugin_fqdn": "privacy.pategan",
        "requested_parameters": {
            "epsilon": 1.0,
            "delta": 1e-6,
            "alpha": 100,
            "lamda": 0.1,
        },
        "n_samples": cfg.generation.n_samples,
        "random_state": cfg.seed,
        "privacy_accounting": {
            "schema_version": "pate-accounting-v1",
            "privacy_claim_type": "formal_dp",
            "accountant": "pate_moments_v1",
            "requested_epsilon": 1.0,
            "requested_delta": 1e-6,
            "requested_alpha": 100,
            "requested_lamda": 0.1,
            "resolved_epsilon": 1.0,
            "resolved_delta": 1e-6,
            "resolved_alpha": 100,
            "resolved_lamda": 0.1,
            "effective_epsilon": 1.2,
            "effective_delta": 1e-6,
            "effective_alpha": 100,
            "effective_lamda": 0.1,
            "iterations": 2,
            "max_iter": 10,
            "stopping_state": "epsilon_reached",
        },
    }
    (generation_dir / "model_a.cache.json").write_text(
        json.dumps({"generator_metadata": generator_metadata})
    )
    report = _dataframe(
        {"mean": [0.25], "direction": ["minimize"]},
        index=["privacy.identifiability_score.score_OC"],
    )

    monkeypatch.setattr(
        synthcity_eval, "run_synthcity_evaluation", lambda *args, **kwargs: {"model_a": report}
    )
    monkeypatch.setattr(
        syntheval_eval, "run_syntheval_evaluation", lambda *args, **kwargs: (None, None)
    )
    monkeypatch.setattr(custom_eval, "run_log_disparity_evaluation", lambda *args, **kwargs: {})
    monkeypatch.setattr(privacy_gate, "evaluate_privacy_gate", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        generation_pipeline,
        "refit_selected_model",
        lambda *args, **kwargs: _fake_refit_metadata(synthetic, kwargs["output_dir"]),
    )

    _combined, extras = run_evaluation(cfg, dataset, {"model_a": synthetic})

    assert extras["generator_metadata"]["model_a"]["state"] == "legacy"
    assert extras["generator_metadata"]["model_a"]["metadata"] == generator_metadata
    manifest = json.loads(
        (Path(cfg.evaluation.output_dir) / "evaluation_artifacts-v1" / "manifest.json").read_text()
    )
    assert manifest["generator_metadata"]["model_a"]["metadata"] == generator_metadata


def test_generation_metadata_preserves_current_cache_envelope(make_config):
    cfg = make_config()
    generation_dir = Path(cfg.generation.output_dir)
    generation_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = generation_dir / "model_a.cache.json"
    data_path = generation_dir / "model_a.csv"
    generator_metadata = {
        "schema_version": "generator-metadata-v1",
        "generator_context": {"privacy_claim_type": "none"},
        "plugin_name": "model_a",
        "plugin_fqdn": "test.generator",
        "requested_parameters": {},
        "n_samples": 1,
        "random_state": 0,
        "privacy_accounting": None,
    }
    cache_metadata = {
        "schema_version": "generation-cache-v3",
        "model_name": "model_a",
        "generator_metadata": generator_metadata,
        "semantic_context_digest": "semantic-digest",
        "resolved_parameters": {},
    }
    metadata_path.write_text(json.dumps(cache_metadata))
    data_path.write_text("feature,target\n1,0\n")

    result = _generation_metadata(cfg, ["model_a"])["model_a"]

    assert result["state"] == "present"
    assert result["cache_metadata"] == cache_metadata
    assert result["metadata"] == generator_metadata
    assert result["metadata_path"] == str(metadata_path)
    assert result["data_path"] == str(data_path)
    assert result["metadata_sha256"]
    assert result["data_sha256"]


def test_selection_uses_complete_tuning_utility_and_ignores_gate_and_legacy_rank():
    combined = _dataframe({}, index=["model_a"])
    combined[("__all__", "overall", "rank")] = [1.0]
    combined[("__all__", "privacy_gate", "pass")] = [False]
    combined[("__all__", "utility", "U_tuning")] = [0.75]
    combined.columns = pd.MultiIndex.from_tuples(combined.columns)

    selected, error = _select_policy_model(combined)

    assert selected == "model_a"
    assert error is None


def test_selection_requires_complete_finite_tuning_utility():
    combined = _dataframe(
        {
            ("__all__", "overall", "rank"): [1.0],
            ("__all__", "utility", "U_tuning"): [float("nan")],
        },
        index=["model_a"],
    )
    combined.columns = pd.MultiIndex.from_tuples(combined.columns)

    selected, error = _select_policy_model(combined)

    assert selected is None
    assert error == "no candidate has complete finite U_tuning"


def test_selection_failure_persists_auditable_blocked_final_evidence(
    make_config, make_canonical_dataset, monkeypatch
):
    cfg = make_config()
    cfg.evaluation.generate_report = False
    dataset = make_canonical_dataset()
    synthetic = dataset.role_frame("train", imputed=True).copy()
    combined = _dataframe(
        {
            ("__all__", "overall", "rank"): [1.0],
            ("__all__", "utility", "U_tuning"): [float("nan")],
        },
        index=["model_a"],
    )
    combined.columns = pd.MultiIndex.from_tuples(combined.columns)

    monkeypatch.setattr(
        "synthdata.evaluation.combine.build_combined_table", lambda *args, **kwargs: combined
    )
    monkeypatch.setattr(custom_eval, "run_log_disparity_evaluation", lambda *args, **kwargs: {})
    _result, extras = run_evaluation(cfg, dataset, {"model_a": synthetic})

    evidence = artifacts.load_final_holdout_evidence(cfg.evaluation.output_dir)
    assert extras["final_holdout_evidence"]["state"] == "blocked"
    assert (
        evidence["evidence_execution_state"],
        evidence["metric_completeness_state"],
        evidence["score_completeness_state"],
        evidence["audit_outcome_state"],
    ) == ("blocked", "not_applicable", "not_applicable", "blocked")
    assert evidence["selected_model"] is None
    assert evidence["provenance_inventory"]["invalid_reasons"]
    assert evidence["provenance_inventory"]["selected_model_provenance"]["status"] == "not_selected"


def test_multi_model_selection_does_not_rerank_on_final_holdout_evidence():
    combined = _dataframe(
        {
            ("__all__", "utility", "U_tuning"): [0.90, 0.80],
            ("__all__", "overall", "rank"): [0.90, 0.80],
        },
        index=["model_a", "model_b"],
    )
    combined.columns = pd.MultiIndex.from_tuples(combined.columns)

    selected, error = _select_policy_model(combined)

    assert error is None
    assert selected == "model_a"
    final_holdout_scores = {"model_a": 0.20, "model_b": 0.95}
    assert final_holdout_scores["model_b"] > final_holdout_scores["model_a"]
    assert selected == "model_a"


def test_release_score_adapter_maps_successful_task12_aggregate_records():
    role_hashes = {"train": "train", "tuning": "tuning", "final_holdout": "holdout"}
    common = {
        "producer": "task12-test",
        "protocol_version": task12_eval.TASK12_PROTOCOL_VERSION,
        "seed": 7,
        "release_transform_digest": "release-transform",
        "common_protocol_digest": "common-protocol",
        "role_hashes": role_hashes,
        "fit_roles": ["train", "tuning"],
    }
    records = [
        task12_eval._task12_record(
            "model_a",
            "release_privacy.v1",
            0.6,
            role_hashes=role_hashes,
            evaluation_role="final_holdout",
            metadata={
                **common,
                "k": {"safety_score": 0.91},
                "l": {"safety_score": 0.82},
                "dcr": 0.73,
                "epsilon": 0.64,
                "mia": 0.55,
                "attribute": 0.46,
            },
        ),
        task12_eval._task12_record(
            "model_a",
            "representation_evidence.v1",
            0.87,
            role_hashes=role_hashes,
            evaluation_role="final_holdout",
            metadata={**common, "summary_stats": {"worst_abs_log_disparity": 0.12}},
        ),
        task12_eval._task12_record(
            "model_a",
            "equalized_odds.final.v1",
            0.78,
            role_hashes=role_hashes,
            evaluation_role="final_holdout",
            metadata=common,
        ),
    ]
    blocked_validations = task12_eval.validate_task12_custom_results(
        {"model_a": records},
        role_hashes=role_hashes,
        evaluation_role="final_holdout",
    )
    validation = blocked_validations["model_a"]
    successful_records = tuple(
        replace(
            record,
            status="succeeded",
            contract_id=f"custom.{record.expected_key}",
            raw_value=next(
                item.raw_value for item in records if item.emitted_key == record.expected_key
            ),
            policy_value=next(
                item.raw_value for item in records if item.emitted_key == record.expected_key
            ),
        )
        for record in validation.records
    )
    validations = replace(validation, records=successful_records)

    _utility, privacy, fairness = _release_score_inputs(
        {("custom", "task12"): {"model_a": validations}}
    )

    assert {name: item["value"] for name, item in privacy.items()} == {
        "S_k": 0.91,
        "S_l": 0.82,
        "S_DCR": 0.73,
        "S_epsilon": 0.64,
        "S_MIA": 0.55,
        "S_attribute": 0.46,
    }
    assert {name: item["value"] for name, item in fairness.items()} == {
        "S_representation": 0.87,
        "S_worst_log_disparity": 0.12,
        "S_EO": 0.78,
    }
    assert privacy["S_k"]["evidence"]["metadata"]["common_protocol_digest"] == "common-protocol"
    assert privacy["S_k"]["evidence"]["record"]["model_name"] == "model_a"

    release_score = compute_release_score(
        utility={"S_TSTR": 0.8, "S_MMD": 0.7, "S_JSD": 0.6},
        privacy=privacy,
        fairness=fairness,
    )

    assert release_score["status"] == "succeeded"
    assert release_score["dimensions"]["privacy"]["score"] == pytest.approx(
        (0.64 * 0.55 * 0.46) ** (1 / 3)
    )
    assert release_score["dimensions"]["fairness"]["score"] == pytest.approx(
        0.4 * 0.87 + 0.4 * 0.78 + 0.2 * 0.12
    )
    assert release_score["score"] is not None


def test_run_evaluation_rejects_conflicting_synthcity_qi_override(
    make_config, make_canonical_dataset
):
    cfg = make_config()
    dataset = make_canonical_dataset()
    dataset.quasi_identifier_columns = ["feature"]
    cfg.evaluation.synthcity.quasi_identifier_columns = ["protected"]

    with pytest.raises(ValueError, match="must match the Dataset declaration"):
        run_evaluation(cfg, dataset, {})


def test_run_evaluation_propagates_patient_group_context(
    make_config, make_canonical_dataset, monkeypatch
):
    cfg = make_config()
    cfg.evaluation.group_mode = "patient_group"
    cfg.evaluation.group_column = "group"
    cfg.evaluation.syntheval.enabled = True
    cfg.evaluation.custom.enabled = False
    cfg.evaluation.save_per_model_syntheval_plots = False
    cfg.evaluation.generate_report = False
    cfg.evaluation.privacy_gate.enabled = False

    dataset = make_canonical_dataset()
    dataset.variable_schema = {
        column: {"kind": "categorical" if column == "target" else "continuous"}
        for column in dataset.full_df.columns
    }
    synthetic = dataset.role_frame("train", imputed=True).copy()
    synthcity_report = _dataframe(
        {"mean": [0.25], "direction": ["minimize"]},
        index=["privacy.identifiability_score.score_OC"],
    )
    received_context = {}

    monkeypatch.setattr(
        synthcity_eval,
        "run_synthcity_evaluation",
        lambda *args, **kwargs: {"model_a": synthcity_report},
    )
    real_validate = synthcity_eval.validate_synthcity_results

    def capture_validate(*args, **kwargs):
        received_context["value"] = kwargs["context"]
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(synthcity_eval, "validate_synthcity_results", capture_validate)
    monkeypatch.setattr(
        syntheval_eval,
        "run_syntheval_evaluation",
        lambda *args, **kwargs: (None, None),
    )
    monkeypatch.setattr(
        custom_eval,
        "run_log_disparity_evaluation",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        generation_pipeline,
        "refit_selected_model",
        lambda *args, **kwargs: _fake_refit_metadata(synthetic, kwargs["output_dir"]),
    )
    monkeypatch.setattr(
        privacy_gate,
        "evaluate_privacy_gate",
        lambda *args, **kwargs: None,
    )

    _combined, extras = run_evaluation(cfg, dataset, {"model_a": synthetic})

    context = received_context["value"]
    assert context.group_mode == "patient_group"
    assert context.population_unit == "patient_group"
    assert context.resolved_configuration["group_context"]["group_column"] == "group"
    assert extras["group_context"]["group_column"] == "group"
    assert extras["population_unit"] == "patient_group"


def test_run_evaluation_records_post_selection_final_holdout_evidence(
    make_config, make_canonical_dataset, monkeypatch
):
    cfg = make_config()
    cfg.evaluation.synthcity.metrics = ["identifiability_score"]
    cfg.evaluation.syntheval.enabled = True
    cfg.evaluation.custom.enabled = False
    cfg.evaluation.binary_target.enabled = False
    cfg.evaluation.save_per_model_syntheval_plots = False
    cfg.evaluation.generate_report = False
    cfg.evaluation.privacy_gate.enabled = False

    dataset = make_canonical_dataset()
    synthetic = dataset.role_frame("train", imputed=True).copy()
    synthcity_calls = []
    synthcity_fit_frames = []
    synthcity_evidence_frames = []
    syntheval_calls = []
    syntheval_fit_roles = []
    syntheval_fit_frames = []
    syntheval_released_datasets = []
    syntheval_released_references = []
    synthcity_report = _dataframe(
        {"mean": [0.25] * 4, "direction": ["minimize"] * 4},
        index=[
            "privacy.identifiability_score.score",
            "privacy.identifiability_score.score_OC",
            "privacy.identifiability_score.score_entropy_weighted",
            "privacy.identifiability_score.score_OC_entropy_weighted",
        ],
    )

    def fake_run_synthcity_evaluation(*args, **kwargs):
        synthcity_calls.append(kwargs.get("evaluation_role", "tuning"))
        synthcity_fit_frames.append(args[1])
        synthcity_evidence_frames.append(args[2])
        return {"model_a": synthcity_report}

    def fake_run_syntheval_evaluation(*args, **kwargs):
        syntheval_calls.append(kwargs.get("evaluation_role", "tuning"))
        syntheval_fit_roles.append(kwargs.get("fit_roles"))
        syntheval_fit_frames.append(kwargs.get("fit_frame"))
        syntheval_released_datasets.append(kwargs.get("released_synthetic_datasets"))
        syntheval_released_references.append(kwargs.get("released_final_holdout_frame"))
        return None, None, {}

    def fake_refit_selected_model(*args, **kwargs):
        return _fake_refit_metadata(synthetic, kwargs["output_dir"])

    monkeypatch.setattr(generation_pipeline, "refit_selected_model", fake_refit_selected_model)

    monkeypatch.setattr(synthcity_eval, "run_synthcity_evaluation", fake_run_synthcity_evaluation)
    monkeypatch.setattr(syntheval_eval, "run_syntheval_evaluation", fake_run_syntheval_evaluation)
    monkeypatch.setattr(
        "synthdata.evaluation._select_policy_model", lambda _combined: ("model_a", None)
    )

    combined, extras = run_evaluation(cfg, dataset, {"model_a": synthetic})

    assert synthcity_calls == ["tuning", "final_holdout"]
    expected_real_fit = pd.concat(
        [dataset.role_frame("train", imputed=True), dataset.role_frame("tuning", imputed=True)],
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(synthcity_fit_frames[1], expected_real_fit)
    pd.testing.assert_frame_equal(
        synthcity_evidence_frames[1], dataset.role_frame("final_holdout", imputed=True)
    )
    assert syntheval_calls == ["tuning", "final_holdout"]
    assert syntheval_fit_roles == [None, ("train", "tuning")]
    pd.testing.assert_frame_equal(syntheval_fit_frames[1], expected_real_fit)
    assert syntheval_released_datasets[0] is None
    assert syntheval_released_datasets[1] is not None
    assert syntheval_released_references[1] is not None
    assert "final_holdout_evidence" in extras
    evidence = artifacts.load_final_holdout_evidence(cfg.evaluation.output_dir)
    assert evidence["state"] == "failed"
    assert evidence["selected_model"] == "model_a"
    assert evidence["role_context"]["roles"].keys() == {
        "train",
        "tuning",
        "final_holdout",
    }
    final_validation = evidence["frameworks"]["synthcity"]["validation"]
    assert final_validation["model_a"]["evaluation_context"]["evaluation_role"] == "final_holdout"
    final_configuration = final_validation["model_a"]["evaluation_context"][
        "resolved_configuration"
    ]
    assert final_configuration["fit_frame_fingerprint"] != "refit-imputed"
    assert final_configuration["refit_fit_frame_fingerprint"] == "refit-imputed"
    assert list(combined.index) == ["model_a"]


def test_run_evaluation_keeps_multi_model_selection_outside_final_holdout(
    make_config, make_canonical_dataset, monkeypatch
):
    cfg = make_config()
    cfg.evaluation.synthcity.metrics = ["identifiability_score"]
    cfg.evaluation.syntheval.enabled = False
    cfg.evaluation.custom.enabled = False
    cfg.evaluation.binary_target.enabled = False
    cfg.evaluation.save_per_model_syntheval_plots = False
    cfg.evaluation.generate_report = False
    cfg.evaluation.privacy_gate.enabled = False

    dataset = make_canonical_dataset()
    synthetic = dataset.role_frame("train", imputed=True).copy()
    selected_models = []
    final_task12_models = []
    final_custom_models = []
    synthcity_report = _dataframe(
        {"mean": [0.25] * 4, "direction": ["minimize"] * 4},
        index=[
            "privacy.identifiability_score.score",
            "privacy.identifiability_score.score_OC",
            "privacy.identifiability_score.score_entropy_weighted",
            "privacy.identifiability_score.score_OC_entropy_weighted",
        ],
    )

    def fake_run_synthcity(selected, *args, **kwargs):
        selected_models.append((kwargs.get("evaluation_role", "tuning"), set(selected)))
        # Deliberately make final evidence look better for model_b if it were run.
        report = synthcity_report.copy()
        if kwargs.get("evaluation_role") == "final_holdout":
            report.loc[:, "mean"] = 0.99
        return {model: report for model in selected}

    def fake_run_task12(selected, *args, **kwargs):
        if kwargs.get("evaluation_role") == "final_holdout":
            final_task12_models.append(set(selected))
        return {}

    def fake_run_log_disparity(selected, *args, **kwargs):
        if kwargs.get("evaluation_role") == "final_holdout":
            final_custom_models.append(set(selected))
        return {}

    combined = _dataframe(
        {
            ("__all__", "utility", "U_tuning"): [0.90, 0.80],
            ("__all__", "overall", "rank"): [0.90, 0.80],
        },
        index=["model_a", "model_b"],
    )
    combined.columns = pd.MultiIndex.from_tuples(combined.columns)

    monkeypatch.setattr(synthcity_eval, "run_synthcity_evaluation", fake_run_synthcity)
    monkeypatch.setattr(task12_eval, "run_task12_custom_evaluation", fake_run_task12)
    monkeypatch.setattr(custom_eval, "run_log_disparity_evaluation", fake_run_log_disparity)
    monkeypatch.setattr(
        "synthdata.evaluation.combine.build_combined_table", lambda *args, **kwargs: combined
    )
    monkeypatch.setattr(
        generation_pipeline,
        "refit_selected_model",
        lambda *args, **kwargs: _fake_refit_metadata(synthetic, kwargs["output_dir"]),
    )

    result, extras = run_evaluation(
        cfg,
        dataset,
        {"model_a": synthetic.copy(), "model_b": synthetic.copy()},
    )

    assert selected_models == [("tuning", {"model_a", "model_b"}), ("final_holdout", {"model_a"})]
    assert final_task12_models == [{"model_a"}]
    assert final_custom_models == [{"model_a"}]
    assert set(result.index) == {"model_a", "model_b"}
    assert extras["final_holdout_evidence"]["selected_model"] == "model_a"

    evidence = artifacts.load_final_holdout_evidence(cfg.evaluation.output_dir)
    release_score = artifacts.load_release_score_evidence(cfg.evaluation.output_dir)
    assert evidence["selected_model"] == "model_a"
    assert "model_b" not in evidence["frameworks"]["synthcity"]["validation"]
    assert release_score["candidate_audit_models"] == ["model_a", "model_b"]
    assert set(release_score["models"]) == {"model_a"}
    assert release_score["selected_model"] == "model_a"


@pytest.mark.parametrize("finite_final_score", [False, True])
def test_run_evaluation_records_authoritative_final_task10_evidence(
    make_config, make_canonical_dataset, monkeypatch, finite_final_score
):
    cfg = make_config()
    cfg.evaluation.synthcity.metrics = ["identifiability_score"]
    cfg.evaluation.syntheval.enabled = False
    cfg.evaluation.custom.enabled = True
    cfg.evaluation.binary_target.enabled = False
    cfg.evaluation.save_per_model_syntheval_plots = False
    cfg.evaluation.generate_report = False
    cfg.evaluation.privacy_gate.enabled = False

    dataset = make_canonical_dataset()
    synthetic = dataset.role_frame("train", imputed=True).copy()
    final_role_hashes = {
        role: dataframe_fingerprint(dataset.role_frame(role, imputed=False))
        for role in ("train", "tuning", "final_holdout")
    }
    refit_hash = hashlib.sha256(b"refit-raw").hexdigest()
    final_role_hashes["refit_fit"] = refit_hash
    synthcity_report = _dataframe(
        {"mean": [0.25] * 4, "direction": ["minimize"] * 4},
        index=[
            "privacy.identifiability_score.score",
            "privacy.identifiability_score.score_OC",
            "privacy.identifiability_score.score_entropy_weighted",
            "privacy.identifiability_score.score_OC_entropy_weighted",
        ],
    )

    monkeypatch.setattr(
        synthcity_eval,
        "run_synthcity_evaluation",
        lambda *args, **kwargs: {"model_a": synthcity_report},
    )
    monkeypatch.setattr(
        custom_eval,
        "run_log_disparity_evaluation",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        generation_pipeline,
        "refit_selected_model",
        lambda *args, **kwargs: _fake_refit_metadata_with_hash(
            synthetic, kwargs["output_dir"], refit_hash
        ),
    )
    monkeypatch.setattr(
        "synthdata.evaluation._select_policy_model", lambda _combined: ("model_a", None)
    )

    tstr_calls = []
    real_run_tstr = tstr_module.run_tstr_evaluation

    def run_authoritative_task10(*args, **kwargs):
        tstr_calls.append(kwargs["evaluation_role"])
        result = real_run_tstr(*args, **kwargs)
        assert result.report is not None
        assert result.envelope is not None
        metadata = result.report["result_metadata"]
        metadata["role_hashes"] = final_role_hashes
        result.report["prediction_artifact"]["role_hashes"] = final_role_hashes
        result.report["prediction_artifact"]["artifact_digest"] = tstr_module._artifact_digest(
            result.report["prediction_artifact"]
        )
        result.envelope["result_metadata"] = metadata
        result.envelope["report"] = result.report
        return result

    monkeypatch.setattr("synthdata.evaluation.run_tstr_evaluation", run_authoritative_task10)

    def valid_release(*args, **kwargs):
        synthetic_release, reference = args[:2]
        provenance = synthetic_release.attrs["release_provenance"]
        return {
            "status": "succeeded",
            "value": 0.0,
            "producer": "task12_release_privacy",
            "protocol_version": "task12-evaluation-v1",
            "seed": kwargs["seed"],
            "release_transform_digest": provenance["release_transform_digest"],
            "common_protocol_digest": provenance["common_protocol_digest"],
            "role_hashes": final_role_hashes,
            "fit_roles": ["train", "tuning"],
            "support": {
                "state": "valid",
                "support_contract": "declared_support_v1",
                "roles": {
                    "synthetic": {
                        "population": 12,
                        "population_floor": 1,
                        "role_hash": final_role_hashes["refit_fit"],
                    },
                    "reference": {
                        "population": 6,
                        "population_floor": 1,
                        "role_hash": final_role_hashes["final_holdout"],
                    },
                },
                "role_population_floor": 1,
            },
            "population_identity": {"synthetic": "synthetic", "final_holdout": "holdout"},
        }

    def valid_representation(*args, **kwargs):
        if kwargs.get("evaluation_role") != "final_holdout" or args[3].categories != []:
            return {}
        released, _roles, _metadata = transform_release_roles(
            args[0]["model_a"],
            {"final_holdout": args[1].role_frame("final_holdout", imputed=False)},
            None,
        )
        provenance = released.attrs["release_provenance"]
        return {
            "model_a": {
                "summary_stats": {"representation_safety": 0.0},
                "result_metadata": {
                    "producer": "task12_representation",
                    "protocol_version": "task12-evaluation-v1",
                    "seed": cfg.seed,
                    "release_transform_digest": provenance["release_transform_digest"],
                    "common_protocol_digest": provenance["common_protocol_digest"],
                    "role_hashes": final_role_hashes,
                    "support": {
                        "support_contract": "declared_support_v1",
                        "roles": {
                            "synthetic": {
                                "population": 12,
                                "population_floor": 1,
                                "role_hash": final_role_hashes["refit_fit"],
                            },
                            "reference": {
                                "population": 6,
                                "population_floor": 1,
                                "role_hash": final_role_hashes["final_holdout"],
                            },
                        },
                        "role_population_floor": 1,
                        "protected_slices": {"state": "valid", "floor": 1},
                    },
                    "fit_roles": ["train", "tuning"],
                },
            }
        }

    monkeypatch.setattr(task12_eval, "release_privacy_evidence", valid_release)
    monkeypatch.setattr(
        task12_eval.custom_eval, "run_log_disparity_evaluation", valid_representation
    )
    real_task12_record = task12_eval._task12_record

    def valid_task12_record(*args, **kwargs):
        if args[1] == "equalized_odds.final.v1":
            kwargs["metadata"] = {
                **kwargs.get("metadata", {}),
                "role_hashes": final_role_hashes,
            }
        record = real_task12_record(*args, **kwargs)
        if record.emitted_key != "equalized_odds.final.v1":
            support = dict(record.support or {})
            support.update(
                {
                    "roles": {
                        "synthetic": {
                            "population": 12,
                            "population_floor": 1,
                            "role_hash": final_role_hashes["refit_fit"],
                        },
                        "reference": {
                            "population": 6,
                            "population_floor": 1,
                            "role_hash": final_role_hashes["final_holdout"],
                        },
                    },
                    "role_population_floor": 1,
                    "protected_slices": {"state": "valid", "floor": 1},
                }
            )
            record = replace(
                record,
                support=support,
                source_metadata={**record.source_metadata, "support": support},
                result_metadata={**record.result_metadata, "support": support},
                provenance={**record.provenance, "support": support},
            )
        return record

    monkeypatch.setattr(task12_eval, "_task12_record", valid_task12_record)
    if finite_final_score:
        monkeypatch.setattr(
            "synthdata.evaluation.compute_release_score",
            lambda **kwargs: {
                "status": "succeeded",
                "R_final": 0.42,
                "audit_only": True,
                "dimensions": {"utility": 0.4, "privacy": 0.4, "fairness": 0.5},
            },
        )

    combined, extras = run_evaluation(cfg, dataset, {"model_a": synthetic})

    evidence = artifacts.load_final_holdout_evidence(cfg.evaluation.output_dir)
    expected_state = "succeeded" if finite_final_score else "failed"
    assert evidence["state"] == expected_state
    assert extras["final_holdout_evidence"]["state"] == expected_state
    assert evidence["evidence_execution_state"] == "succeeded"
    assert evidence["metric_completeness_state"] == "complete"
    expected_score_state = "complete" if finite_final_score else "indeterminate"
    expected_audit_state = "complete" if finite_final_score else "indeterminate"
    assert evidence["score_completeness_state"] == expected_score_state
    assert evidence["audit_outcome_state"] == expected_audit_state
    assert evidence["release_score"]["status"] == (
        "succeeded" if finite_final_score else "indeterminate"
    )
    if finite_final_score:
        assert evidence["release_score"]["R_final"] == 0.42
        assert evidence["release_score"]["audit_only"] is True
    assert "dimensions" in evidence["release_score"]
    release_score_evidence = artifacts.load_release_score_evidence(cfg.evaluation.output_dir)
    assert release_score_evidence["models"]["model_a"] == evidence["release_score"]
    assert list(combined.index) == ["model_a"]
    assert tstr_calls == ["final_holdout"]
    task12_records = evidence["frameworks"]["custom"]["task12_validation"]["model_a"]["records"]
    assert [record["expected_key"] for record in task12_records] == [
        "release_privacy.v1",
        "representation_evidence.v1",
        "equalized_odds.final.v1",
    ]
    assert all(record["status"] == "succeeded" for record in task12_records)


def test_run_evaluation_blocks_legacy_before_candidate_ranking(
    make_config, make_dataset, monkeypatch
):
    cfg = make_config()
    cfg.evaluation.synthcity.metrics = ["identifiability_score"]
    cfg.evaluation.syntheval.enabled = False
    cfg.evaluation.custom.enabled = False
    cfg.evaluation.binary_target.enabled = False
    cfg.evaluation.save_per_model_syntheval_plots = False
    cfg.evaluation.generate_report = True
    cfg.evaluation.privacy_gate.enabled = False

    dataset = make_dataset()
    dataset.train_imputed_df = dataset.train_df.copy()
    dataset.test_imputed_df = dataset.test_df.copy()
    synthetic = dataset.test_imputed_df.copy()
    synthcity_calls = []
    synthcity_report = _dataframe(
        {"mean": [0.25], "direction": ["minimize"]},
        index=["privacy.identifiability_score.score_OC"],
    )

    def fake_run_synthcity_evaluation(*args, **kwargs):
        synthcity_calls.append(kwargs.get("evaluation_role", "tuning"))
        return {"model_a": synthcity_report}

    monkeypatch.setattr(synthcity_eval, "run_synthcity_evaluation", fake_run_synthcity_evaluation)

    combined, extras = run_evaluation(cfg, dataset, {"model_a": synthetic})

    assert synthcity_calls == []
    assert combined.empty
    assert ("__all__", "overall", "rank") not in combined.columns
    evidence = extras["final_holdout_evidence"]
    assert evidence["state"] == "blocked"
    assert evidence["selected_model"] is None
    assert "legacy_two_role" in evidence["reason"]
    assert artifacts.load_final_holdout_evidence(cfg.evaluation.output_dir)["state"] == "blocked"
    assert (Path(cfg.evaluation.output_dir) / "report.md").exists()


def test_run_evaluation_persists_failed_final_framework_evidence(
    make_config, make_canonical_dataset, monkeypatch
):
    cfg = make_config()
    cfg.evaluation.synthcity.metrics = ["identifiability_score"]
    cfg.evaluation.syntheval.enabled = False
    cfg.evaluation.custom.enabled = False
    cfg.evaluation.binary_target.enabled = True
    cfg.evaluation.save_per_model_syntheval_plots = False
    cfg.evaluation.generate_report = False
    cfg.evaluation.privacy_gate.enabled = False

    dataset = make_canonical_dataset()
    synthetic = dataset.role_frame("train", imputed=True).copy()
    synthcity_report = _dataframe(
        {"mean": [0.25], "direction": ["minimize"]},
        index=["privacy.identifiability_score.score_OC"],
    )
    failed_report = _dataframe({"error": ["framework failed"], "error_type": ["RuntimeError"]})
    disabled_syntheval_calls = []

    def fake_run_synthcity_evaluation(*args, **kwargs):
        if kwargs.get("evaluation_role") == "final_holdout":
            return {"model_a": failed_report}
        return {"model_a": synthcity_report}

    def fake_run_syntheval_evaluation(*args, **kwargs):
        disabled_syntheval_calls.append("main")
        return None, None, {}

    def fail_if_binary_syntheval_runs(*args, **kwargs):
        raise AssertionError("disabled SynthEval must skip binary execution")

    def fake_refit_selected_model(*args, **kwargs):
        return _fake_refit_metadata(synthetic, kwargs["output_dir"])

    monkeypatch.setattr(synthcity_eval, "run_synthcity_evaluation", fake_run_synthcity_evaluation)
    monkeypatch.setattr(syntheval_eval, "run_syntheval_evaluation", fake_run_syntheval_evaluation)
    monkeypatch.setattr(
        syntheval_eval,
        "run_binary_target_syntheval_evaluation",
        fail_if_binary_syntheval_runs,
    )
    monkeypatch.setattr(generation_pipeline, "refit_selected_model", fake_refit_selected_model)
    monkeypatch.setattr(
        "synthdata.evaluation._select_policy_model", lambda _combined: ("model_a", None)
    )

    _combined, extras = run_evaluation(cfg, dataset, {"model_a": synthetic})

    evidence = extras["final_holdout_evidence"]
    assert evidence["state"] == "failed"
    assert evidence["failure_reasons"] == [
        {
            "framework": "synthcity",
            "execution_pass": None,
            "model": "model_a",
            "status_counts": {"failed": 4},
            "failed_keys": [
                "privacy.identifiability_score.score",
                "privacy.identifiability_score.score_OC",
                "privacy.identifiability_score.score_entropy_weighted",
                "privacy.identifiability_score.score_OC_entropy_weighted",
            ],
            "indeterminate_keys": [
                "privacy.identifiability_score.score",
                "privacy.identifiability_score.score_OC",
                "privacy.identifiability_score.score_entropy_weighted",
                "privacy.identifiability_score.score_OC_entropy_weighted",
            ],
        },
    ]
    assert artifacts.load_final_holdout_evidence(cfg.evaluation.output_dir)["state"] == "failed"
    assert disabled_syntheval_calls == []
    assert evidence["frameworks"]["syntheval"] == {
        "validation": {},
        "execution": {"main": {}, "binary_target": {}},
    }


def test_run_evaluation_persists_failed_final_syntheval_worker(
    make_config, make_canonical_dataset, monkeypatch
):
    cfg = make_config()
    cfg.evaluation.synthcity.metrics = ["identifiability_score"]
    cfg.evaluation.syntheval.enabled = True
    cfg.evaluation.custom.enabled = False
    cfg.evaluation.binary_target.enabled = False
    cfg.evaluation.save_per_model_syntheval_plots = False
    cfg.evaluation.generate_report = False
    cfg.evaluation.privacy_gate.enabled = False

    dataset = make_canonical_dataset()
    synthetic = dataset.role_frame("train", imputed=True).copy()
    synthcity_report = _dataframe(
        {"mean": [0.25] * 4, "direction": ["minimize"] * 4},
        index=[
            "privacy.identifiability_score.score",
            "privacy.identifiability_score.score_OC",
            "privacy.identifiability_score.score_entropy_weighted",
            "privacy.identifiability_score.score_OC_entropy_weighted",
        ],
    )

    def fake_run_synthcity_evaluation(*args, **kwargs):
        return {"model_a": synthcity_report}

    def fake_run_syntheval_evaluation(*args, **kwargs):
        if kwargs.get("evaluation_role") == "final_holdout":
            raise RuntimeError("worker failed; checkpoint=/candidate/model_a/status.json")
        return None, None, {}

    monkeypatch.setattr(synthcity_eval, "run_synthcity_evaluation", fake_run_synthcity_evaluation)
    monkeypatch.setattr(syntheval_eval, "run_syntheval_evaluation", fake_run_syntheval_evaluation)
    monkeypatch.setattr(
        generation_pipeline,
        "refit_selected_model",
        lambda *args, **kwargs: _fake_refit_metadata(synthetic, kwargs["output_dir"]),
    )
    monkeypatch.setattr(
        "synthdata.evaluation._select_policy_model", lambda _combined: ("model_a", None)
    )

    _combined, extras = run_evaluation(cfg, dataset, {"model_a": synthetic})

    evidence = artifacts.load_final_holdout_evidence(cfg.evaluation.output_dir)
    assert evidence["state"] == "failed"
    assert evidence["frameworks"]["syntheval"]["validation"]
    assert any(
        reason["framework"] == "syntheval"
        and reason["execution_pass"] == "main"
        and reason["model"] == "model_a"
        and reason["exception_type"] == "RuntimeError"
        and "worker failed" in reason["error"]
        for reason in evidence["failure_reasons"]
    )
    assert extras["final_holdout_evidence"]["state"] == "failed"


def test_run_evaluation_passes_real_fit_to_final_binary_evidence(
    make_config, make_canonical_dataset, monkeypatch
):
    cfg = make_config()
    cfg.evaluation.synthcity.metrics = ["identifiability_score"]
    cfg.evaluation.syntheval.enabled = True
    cfg.evaluation.custom.enabled = False
    cfg.evaluation.binary_target.enabled = True
    cfg.evaluation.binary_target.positive_classes = [1]
    cfg.evaluation.binary_target.negative_classes = [0]
    cfg.evaluation.save_per_model_syntheval_plots = False
    cfg.evaluation.generate_report = False
    cfg.evaluation.privacy_gate.enabled = False

    dataset = make_canonical_dataset()
    synthetic = dataset.role_frame("train", imputed=True).copy()
    synthcity_report = _dataframe(
        {"mean": [0.25] * 4, "direction": ["minimize"] * 4},
        index=[
            "privacy.identifiability_score.score",
            "privacy.identifiability_score.score_OC",
            "privacy.identifiability_score.score_entropy_weighted",
            "privacy.identifiability_score.score_OC_entropy_weighted",
        ],
    )
    binary_calls = []
    main_calls = []

    def fake_run_synthcity_evaluation(*args, **kwargs):
        return {"model_a": synthcity_report}

    def fake_run_syntheval_evaluation(*args, **kwargs):
        if kwargs.get("evaluation_role") == "final_holdout":
            main_calls.append(kwargs)
        return None, None, {}

    def fake_run_binary_target_syntheval_evaluation(*args, **kwargs):
        fit_frame = kwargs.get("fit_frame")
        binary_calls.append(
            {
                "evaluation_role": kwargs.get("evaluation_role", "tuning"),
                "synthetic": args[0]["model_a"].copy(),
                "fit_frame": fit_frame.copy() if fit_frame is not None else None,
                "fit_roles": kwargs.get("fit_roles"),
                "released_datasets": kwargs.get("released_synthetic_datasets"),
                "released_reference": kwargs.get("released_final_holdout_frame"),
            }
        )
        return None, None, {}

    def fake_refit_selected_model(*args, **kwargs):
        return _fake_refit_metadata(synthetic, kwargs["output_dir"])

    monkeypatch.setattr(synthcity_eval, "run_synthcity_evaluation", fake_run_synthcity_evaluation)
    monkeypatch.setattr(syntheval_eval, "run_syntheval_evaluation", fake_run_syntheval_evaluation)
    monkeypatch.setattr(
        syntheval_eval,
        "run_binary_target_syntheval_evaluation",
        fake_run_binary_target_syntheval_evaluation,
    )
    monkeypatch.setattr(generation_pipeline, "refit_selected_model", fake_refit_selected_model)
    monkeypatch.setattr(
        "synthdata.evaluation._select_policy_model", lambda _combined: ("model_a", None)
    )

    _combined, extras = run_evaluation(cfg, dataset, {"model_a": synthetic})

    expected_real_fit = pd.concat(
        [dataset.role_frame("train", imputed=True), dataset.role_frame("tuning", imputed=True)],
        ignore_index=True,
    )
    assert [call["evaluation_role"] for call in binary_calls] == ["tuning", "final_holdout"]
    pd.testing.assert_frame_equal(binary_calls[1]["fit_frame"], expected_real_fit)
    pd.testing.assert_frame_equal(binary_calls[1]["synthetic"], synthetic)
    assert binary_calls[1]["fit_roles"] == ("train", "tuning")
    assert main_calls[0]["released_synthetic_datasets"] is binary_calls[1]["released_datasets"]
    assert main_calls[0]["released_final_holdout_frame"] is binary_calls[1]["released_reference"]
    assert extras["final_holdout_evidence"]["state"] == "failed"
    assert artifacts.load_final_holdout_evidence(cfg.evaluation.output_dir)[
        "binary_target_mapping"
    ] == {
        "column": "target",
        "positive_classes": [1],
        "negative_classes": [0],
        "encoding": {"positive": 1, "negative": 0},
    }
