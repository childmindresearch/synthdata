"""Tests for plotting orchestration over persisted evaluation artifacts."""

import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.run_plots as run_plots
from synthdata.data import dataframe_fingerprint, role_context_payload
from synthdata.evaluation import artifacts, combine, report
from synthdata.evaluation.artifacts import expected_evaluation_context
from synthdata.imputation.pipeline import run_imputation
from synthdata.plotting import evaluation_plots

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("generate_report", [True, False])
@pytest.mark.parametrize("context_change", [None, "unrelated"])
def test_regenerated_report_respects_flag_and_receives_validated_coverage(
    monkeypatch, tmp_path, generate_report, context_change
):
    recorded_coverage = {
        "status": "partial",
        "expected_outputs": ["model-a", "model-b"],
        "succeeded_outputs": ["model-a"],
        "failed_outputs": ["model-b"],
    }
    dataset = SimpleNamespace(
        legacy_two_role=False,
        role_frame=lambda role, imputed=False: (
            pd.DataFrame({"feature": [1]}) if role == "final_holdout" and not imputed else None
        ),
    )
    experiment = SimpleNamespace(
        generation_dir=tmp_path / "generation",
        evaluation_dir=tmp_path / "evaluation",
        plots_dir=tmp_path / "plots",
        manifest_path=tmp_path / "experiment.json",
        record=lambda *args, **kwargs: None,
    )
    cfg = SimpleNamespace(
        seed=7,
        config_path=tmp_path / "config.yaml",
        generation=SimpleNamespace(output_dir=tmp_path / "generation"),
        evaluation=SimpleNamespace(
            output_dir=tmp_path / "evaluation", generate_report=generate_report, group_mode="row"
        ),
        plots=SimpleNamespace(sections=["evaluation"], output_dir=tmp_path / "plots"),
        experiment=SimpleNamespace(id=None),
    )
    validated_manifest = {"evaluation_attempt": {"evaluation_coverage": recorded_coverage}}
    report_calls = []
    rank_plot_calls = []
    validation_calls = []
    cache_loads = []
    experiment.evaluation_dir.mkdir(parents=True)
    (experiment.evaluation_dir / "combined_evaluation.csv").touch()
    candidate_context = {
        "schema_version": "role-context-v1",
        "dataset_name": "testds",
        "dataset_version": "v1",
        "roles": {
            role: {
                "raw_fingerprint": f"{role}-raw",
                "imputed_fingerprint": f"{role}-imp",
                "rows": 1,
            }
            for role in ("train", "tuning")
        },
        "assignment_fingerprint": None,
        "assignment_policy_fingerprint": None,
        "identity_fingerprint": "identity",
        "semantic_fingerprint": "semantic",
        "variable_schema_fingerprint": "schema",
        "compatibility_mode": None,
    }
    candidate_full_context = {
        **candidate_context,
        "roles": {
            **candidate_context["roles"],
            "final_holdout": {
                "raw_fingerprint": "final-raw",
                "imputed_fingerprint": "final-raw",
                "rows": 1,
            },
        },
    }
    evaluation_full_context = {
        **candidate_full_context,
        "roles": {
            **candidate_full_context["roles"],
            "final_holdout": {
                **candidate_full_context["roles"]["final_holdout"],
                "imputed_fingerprint": "final-imputed",
            },
        },
    }
    if context_change == "unrelated":
        evaluation_full_context["semantic_fingerprint"] = "foreign-semantic"
    recorded_fingerprints = {
        "candidate": run_plots._mapping_digest(candidate_context),
        "full": run_plots._mapping_digest(evaluation_full_context),
    }
    expected_fingerprints = {
        **recorded_fingerprints,
        "full": run_plots._mapping_digest(candidate_full_context),
    }
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "manifest.json").write_text(
        json.dumps(
            {
                "role_context": {"candidate": candidate_context, "full": evaluation_full_context},
                "role_context_fingerprint": recorded_fingerprints,
                "final_holdout_evidence": {"state": "succeeded"},
            }
        )
    )

    monkeypatch.setattr(sys, "argv", ["run_plots", "--config", str(tmp_path / "config.yaml")])
    monkeypatch.setattr(run_plots, "load_config", lambda _: cfg)
    monkeypatch.setattr(run_plots, "set_global_seed", lambda _: None)
    monkeypatch.setattr(run_plots, "load_dataset", lambda _: dataset)
    monkeypatch.setattr(
        run_plots,
        "load_imputed_splits",
        lambda loaded, **kwargs: cache_loads.append(kwargs) or loaded,
    )
    monkeypatch.setattr(run_plots, "_cache_key_record", lambda *_: {"cache_key": "recorded"})
    monkeypatch.setattr(run_plots, "load_experiment", lambda *_args, **_kwargs: experiment)
    monkeypatch.setattr(run_plots, "_load_synthetic_datasets", lambda _: {})
    monkeypatch.setattr(
        artifacts,
        "expected_evaluation_context",
        lambda _: {
            "role_context_fingerprints": expected_fingerprints,
            "role_hashes": {"train": "train-hash", "tuning": "tuning-hash"},
            "role_hashes_by_framework": {
                "synthcity": {"train": "train-hash", "tuning": "tuning-hash"},
                "syntheval": {"train": "train-hash", "tuning": "tuning-hash"},
                "custom": {"train": "train-raw-hash", "tuning": "tuning-raw-hash"},
            },
        },
    )

    def payload_for_roles(_dataset, roles, *, candidate_phase=False):
        if tuple(roles) == ("train", "tuning"):
            return candidate_context
        assert tuple(roles) == ("train", "tuning", "final_holdout")
        assert candidate_phase
        return candidate_full_context

    monkeypatch.setattr("synthdata.data.role_context_payload", payload_for_roles)

    def validate_bundle(_evaluation_dir, **kwargs):
        validation_calls.append(kwargs)
        if kwargs["expected_role_context_fingerprints"] != recorded_fingerprints:
            raise ValueError("Evaluation bundle role-context fingerprints do not match")
        return validated_manifest

    monkeypatch.setattr(
        artifacts,
        "validate_evaluation_bundle",
        validate_bundle,
    )
    monkeypatch.setattr(combine, "load_combined_table", lambda _: pd.DataFrame())
    monkeypatch.setattr(artifacts, "load_log_disparity_reports", lambda _: {})
    monkeypatch.setattr(artifacts, "artifact_bundle_dir", lambda _: bundle_dir)
    monkeypatch.setattr(
        artifacts,
        "load_generation_inventory",
        lambda *_args: SimpleNamespace(produced_outputs=("model-a",), failed_outputs=("model-b",)),
    )
    monkeypatch.setattr(
        evaluation_plots,
        "save_rank_tradeoff_plots",
        lambda *args, **kwargs: rank_plot_calls.append(kwargs),
    )
    monkeypatch.setattr(evaluation_plots, "save_log_disparity_plots", lambda *_args: None)
    monkeypatch.setattr(artifacts, "verify_native_syntheval_artifacts", lambda _: None)
    monkeypatch.setattr(
        report,
        "save_evaluation_report",
        lambda _cfg, _dataset, _combined, extras, _experiment: report_calls.append(extras),
    )

    if context_change is not None:
        with pytest.raises(ValueError, match="role-context fingerprints"):
            run_plots.main()
        assert not rank_plot_calls
        assert cache_loads[0]["expected_cache_key"] == "recorded"
        return

    run_plots.main()

    if generate_report:
        assert report_calls[0]["evaluation_coverage"] == recorded_coverage
    else:
        assert report_calls == []
    assert rank_plot_calls[0]["missing_stage_a_outputs"] == ("model-b",)
    assert validation_calls[0]["expected_role_context_fingerprints"] == recorded_fingerprints
    assert validation_calls[0]["expected_role_hashes"] == {
        "train": "train-hash",
        "tuning": "tuning-hash",
    }
    assert validation_calls[0]["expected_role_hashes_by_framework"]["custom"] == {
        "train": "train-raw-hash",
        "tuning": "tuning-raw-hash",
    }
    assert cache_loads[0]["expected_cache_key"] == "recorded"


def test_final_holdout_handoff_uses_reloaded_imputation_cache(
    monkeypatch, tmp_path, make_config, make_canonical_dataset
):
    cfg = make_config()
    cfg.imputation.enabled = False
    cfg.imputation.method = "hyperimpute"
    cfg.plots.sections = ["evaluation"]
    cfg.evaluation.group_mode = "row"
    cfg.evaluation.generate_report = False

    persisted_dataset = make_canonical_dataset()
    run_imputation(cfg, persisted_dataset)
    fresh_dataset = make_canonical_dataset()
    fresh_dataset.data_dir = persisted_dataset.data_dir
    fresh_dataset.full_imputed_df = None
    fresh_dataset.imputed_roles = {}
    cache_key = run_plots._cache_key_record(cfg, fresh_dataset)["cache_key"]
    run_plots.load_imputed_splits(fresh_dataset, expected_cache_key=cache_key)
    assert fresh_dataset.full_imputed_df is not None

    evaluation_context = expected_evaluation_context(fresh_dataset)
    candidate_context = role_context_payload(fresh_dataset, ("train", "tuning"))
    recorded_full_context = role_context_payload(
        fresh_dataset,
        ("train", "tuning", "final_holdout"),
        candidate_phase=True,
    )
    final_holdout = fresh_dataset.role_frame("final_holdout", imputed=False).copy()
    final_holdout["feature"] += 0.5
    recorded_full_context["roles"]["final_holdout"]["imputed_fingerprint"] = dataframe_fingerprint(
        final_holdout
    )
    recorded_fingerprints = {
        "candidate": run_plots._mapping_digest(candidate_context),
        "full": run_plots._mapping_digest(recorded_full_context),
    }

    fresh_dataset.full_imputed_df = None
    fresh_dataset.imputed_roles = {}

    experiment = SimpleNamespace(
        generation_dir=tmp_path / "generation",
        evaluation_dir=tmp_path / "evaluation",
        plots_dir=tmp_path / "plots",
        manifest_path=tmp_path / "experiment.json",
        record=lambda *args, **kwargs: None,
    )
    experiment.evaluation_dir.mkdir()
    (experiment.evaluation_dir / "combined_evaluation.csv").touch()
    bundle_dir = artifacts.artifact_bundle_dir(experiment.evaluation_dir)
    bundle_dir.mkdir()
    (bundle_dir / "manifest.json").write_text(
        json.dumps(
            {
                "role_context": {
                    "candidate": candidate_context,
                    "full": recorded_full_context,
                },
                "role_context_fingerprint": recorded_fingerprints,
                "final_holdout_evidence": {"state": "succeeded"},
            }
        )
    )

    rank_plot_calls = []
    validation_calls = []
    monkeypatch.setattr(sys, "argv", ["run_plots", "--config", str(tmp_path / "config.yaml")])
    monkeypatch.setattr(run_plots, "load_config", lambda _: cfg)
    monkeypatch.setattr(run_plots, "set_global_seed", lambda _: None)
    monkeypatch.setattr(run_plots, "load_dataset", lambda _: fresh_dataset)

    def load_experiment(_cfg, *, dataset, allow_final_holdout_handoff):
        assert allow_final_holdout_handoff is True
        assert dataset.full_imputed_df is not None
        assert all(dataset.role_frame(role, imputed=True) is not None for role in dataset.roles)
        return experiment

    monkeypatch.setattr(run_plots, "load_experiment", load_experiment)
    monkeypatch.setattr(run_plots, "_load_synthetic_datasets", lambda _: {})

    def validate_bundle(_evaluation_dir, **kwargs):
        validation_calls.append(kwargs)
        assert kwargs["expected_role_context_fingerprints"] == recorded_fingerprints
        assert kwargs["expected_role_hashes"] == evaluation_context["role_hashes"]
        assert (
            kwargs["expected_role_hashes_by_framework"]
            == evaluation_context["role_hashes_by_framework"]
        )
        return {}

    monkeypatch.setattr(artifacts, "validate_evaluation_bundle", validate_bundle)
    monkeypatch.setattr(combine, "load_combined_table", lambda _: pd.DataFrame())
    monkeypatch.setattr(artifacts, "load_log_disparity_reports", lambda _: {})
    monkeypatch.setattr(
        artifacts,
        "load_generation_inventory",
        lambda *_args: SimpleNamespace(produced_outputs=(), failed_outputs=()),
    )
    monkeypatch.setattr(
        evaluation_plots,
        "save_rank_tradeoff_plots",
        lambda *args, **kwargs: rank_plot_calls.append(kwargs),
    )
    monkeypatch.setattr(evaluation_plots, "save_log_disparity_plots", lambda *_args: None)
    monkeypatch.setattr(artifacts, "verify_native_syntheval_artifacts", lambda _: None)

    run_plots.main()

    assert validation_calls
    assert rank_plot_calls
    assert fresh_dataset.full_imputed_df is not None
    assert (
        dataframe_fingerprint(fresh_dataset.role_frame("train", imputed=True))
        == evaluation_context["role_hashes"]["train"]
    )
