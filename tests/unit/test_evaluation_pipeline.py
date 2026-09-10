"""Focused orchestration tests for contract-aware evaluation outputs."""

import json
from pathlib import Path

import pandas as pd
import pytest

from synthdata.evaluation import (
    _generation_metadata,
    _select_policy_model,
    _synthcity_semantic_context,
    artifacts,
    custom_eval,
    privacy_gate,
    run_evaluation,
    synthcity_eval,
    syntheval_eval,
)
from synthdata.generation import pipeline as generation_pipeline

pytestmark = pytest.mark.unit


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


def test_run_evaluation_validates_ranks_and_persists_status(
    make_config, make_canonical_dataset, monkeypatch
):
    cfg = make_config()
    cfg.evaluation.syntheval.enabled = False
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
    synthcity_report = pd.DataFrame(
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
    cfg.evaluation.custom.enabled = False
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
    report = pd.DataFrame(
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


def test_selection_rejects_non_finite_overall_rank_when_gate_is_disabled():
    combined = pd.DataFrame(index=["model_a"])
    combined[("__all__", "overall", "rank")] = [float("nan")]
    combined.columns = pd.MultiIndex.from_tuples(combined.columns)

    selected, error = _select_policy_model(combined)

    assert selected is None
    assert error == "no candidate has a decision-eligible overall rank"


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
    cfg.evaluation.syntheval.enabled = False
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
    synthcity_report = pd.DataFrame(
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
    cfg.evaluation.syntheval.enabled = False
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
    synthcity_report = pd.DataFrame(
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
    assert "final_holdout_evidence" in extras
    evidence = artifacts.load_final_holdout_evidence(cfg.evaluation.output_dir)
    assert evidence["state"] == "succeeded"
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
    synthcity_report = pd.DataFrame(
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
    cfg.evaluation.binary_target.enabled = False
    cfg.evaluation.save_per_model_syntheval_plots = False
    cfg.evaluation.generate_report = False
    cfg.evaluation.privacy_gate.enabled = False

    dataset = make_canonical_dataset()
    synthetic = dataset.role_frame("train", imputed=True).copy()
    synthcity_report = pd.DataFrame(
        {"mean": [0.25], "direction": ["minimize"]},
        index=["privacy.identifiability_score.score_OC"],
    )
    failed_report = pd.DataFrame({"error": ["framework failed"], "error_type": ["RuntimeError"]})

    def fake_run_synthcity_evaluation(*args, **kwargs):
        if kwargs.get("evaluation_role") == "final_holdout":
            return {"model_a": failed_report}
        return {"model_a": synthcity_report}

    def fake_run_syntheval_evaluation(*args, **kwargs):
        return None, None, {}

    def fake_refit_selected_model(*args, **kwargs):
        return _fake_refit_metadata(synthetic, kwargs["output_dir"])

    monkeypatch.setattr(synthcity_eval, "run_synthcity_evaluation", fake_run_synthcity_evaluation)
    monkeypatch.setattr(syntheval_eval, "run_syntheval_evaluation", fake_run_syntheval_evaluation)
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
        }
    ]
    assert artifacts.load_final_holdout_evidence(cfg.evaluation.output_dir)["state"] == "failed"


def test_run_evaluation_persists_failed_final_syntheval_worker(
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
    synthcity_report = pd.DataFrame(
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
    cfg.evaluation.syntheval.enabled = False
    cfg.evaluation.custom.enabled = False
    cfg.evaluation.binary_target.enabled = True
    cfg.evaluation.binary_target.positive_classes = [1]
    cfg.evaluation.binary_target.negative_classes = [0]
    cfg.evaluation.save_per_model_syntheval_plots = False
    cfg.evaluation.generate_report = False
    cfg.evaluation.privacy_gate.enabled = False

    dataset = make_canonical_dataset()
    synthetic = dataset.role_frame("train", imputed=True).copy()
    synthcity_report = pd.DataFrame(
        {"mean": [0.25] * 4, "direction": ["minimize"] * 4},
        index=[
            "privacy.identifiability_score.score",
            "privacy.identifiability_score.score_OC",
            "privacy.identifiability_score.score_entropy_weighted",
            "privacy.identifiability_score.score_OC_entropy_weighted",
        ],
    )
    binary_calls = []

    def fake_run_synthcity_evaluation(*args, **kwargs):
        return {"model_a": synthcity_report}

    def fake_run_syntheval_evaluation(*args, **kwargs):
        return None, None, {}

    def fake_run_binary_target_syntheval_evaluation(*args, **kwargs):
        fit_frame = kwargs.get("fit_frame")
        binary_calls.append(
            {
                "evaluation_role": kwargs.get("evaluation_role", "tuning"),
                "synthetic": args[0]["model_a"].copy(),
                "fit_frame": fit_frame.copy() if fit_frame is not None else None,
                "fit_roles": kwargs.get("fit_roles"),
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
    assert extras["final_holdout_evidence"]["state"] == "succeeded"
    assert artifacts.load_final_holdout_evidence(cfg.evaluation.output_dir)[
        "binary_target_mapping"
    ] == {
        "column": "target",
        "positive_classes": [1],
        "negative_classes": [0],
        "encoding": {"positive": 1, "negative": 0},
    }
