"""Tests for plotting orchestration over persisted evaluation artifacts."""

import sys
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.run_plots as run_plots
from synthdata.evaluation import artifacts, combine, report
from synthdata.plotting import evaluation_plots

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("generate_report", [True, False])
def test_regenerated_report_respects_flag_and_receives_validated_coverage(
    monkeypatch, tmp_path, generate_report
):
    recorded_coverage = {
        "status": "partial",
        "expected_outputs": ["model-a", "model-b"],
        "succeeded_outputs": ["model-a"],
        "failed_outputs": ["model-b"],
    }
    dataset = SimpleNamespace(legacy_two_role=False)
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
    experiment.evaluation_dir.mkdir(parents=True)
    (experiment.evaluation_dir / "combined_evaluation.csv").touch()

    monkeypatch.setattr(sys, "argv", ["run_plots", "--config", str(tmp_path / "config.yaml")])
    monkeypatch.setattr(run_plots, "load_config", lambda _: cfg)
    monkeypatch.setattr(run_plots, "set_global_seed", lambda _: None)
    monkeypatch.setattr(run_plots, "load_dataset", lambda _: dataset)
    monkeypatch.setattr(run_plots, "load_imputed_splits", lambda loaded, **kwargs: loaded)
    monkeypatch.setattr(run_plots, "_cache_key_record", lambda *_: {"cache_key": "recorded"})
    monkeypatch.setattr(run_plots, "load_experiment", lambda *_args, **_kwargs: experiment)
    monkeypatch.setattr(run_plots, "_load_synthetic_datasets", lambda _: {})
    monkeypatch.setattr(
        artifacts,
        "expected_evaluation_context",
        lambda _: {
            "role_context_fingerprints": {},
            "role_hashes": {},
            "role_hashes_by_framework": {},
        },
    )
    monkeypatch.setattr(
        artifacts, "validate_evaluation_bundle", lambda *_args, **_kwargs: validated_manifest
    )
    monkeypatch.setattr(combine, "load_combined_table", lambda _: pd.DataFrame())
    monkeypatch.setattr(artifacts, "load_log_disparity_reports", lambda _: {})
    monkeypatch.setattr(artifacts, "artifact_bundle_dir", lambda _: tmp_path / "bundle")
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

    run_plots.main()

    if generate_report:
        assert report_calls[0]["evaluation_coverage"] == recorded_coverage
    else:
        assert report_calls == []
    assert rank_plot_calls[0]["missing_stage_a_outputs"] == ("model-b",)
