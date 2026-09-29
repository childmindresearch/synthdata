"""Symlink safety tests for generation plot output paths."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from synthdata.evaluation.syntheval_eval import _safe_output_directory
from synthdata.generation.hpo import HPO_CONTEXT_SCHEMA_VERSION
from synthdata.plotting import generation_plots


def test_generation_output_parent_symlink_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "generation"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        _safe_output_directory(tmp_path, "generation/model", "Generation plot output")


def test_generation_output_final_symlink_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "generation"
    output.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    (output / "model.png").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        target = output / "model.png"
        if target.is_symlink():
            raise ValueError("Generation plot output must not be a symlink")


@pytest.mark.parametrize("all_rejected", [False, True])
def test_hpo_plots_include_persisted_trial_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, all_rejected: bool
) -> None:
    rejected = SimpleNamespace(
        number=0,
        state=SimpleNamespace(name="PRUNED"),
        value=None,
        user_attrs={
            "stage_a_state": "pruned",
            "stage_a_prune_reasons": ["minimum support screen failed"],
        },
    )
    if all_rejected:
        trials = [rejected, SimpleNamespace(**{**rejected.__dict__, "number": 1})]
    else:
        trials = [
            SimpleNamespace(
                number=0,
                state=SimpleNamespace(name="COMPLETE"),
                value=0.75,
                user_attrs={},
            ),
            SimpleNamespace(
                number=1,
                state=SimpleNamespace(name="PRUNED"),
                value=None,
                user_attrs={},
            ),
            rejected,
            SimpleNamespace(
                number=3,
                state=SimpleNamespace(name="PRUNED"),
                value=None,
                user_attrs={
                    "stage_a_state": "pruned",
                    "stage_a_prune_reasons": ["stage_a_screen_exception: screen failed"],
                },
            ),
        ]
    study = SimpleNamespace(
        trials=trials,
        user_attrs={
            "hpo_context_schema_version": HPO_CONTEXT_SCHEMA_VERSION,
            "hpo_context_digest": "context-digest",
            "hpo_context": {},
        },
    )

    monkeypatch.setattr(generation_plots, "contextual_study_name", lambda name, context: name)
    monkeypatch.setattr(generation_plots, "hpo_context_digest", lambda context: "context-digest")
    monkeypatch.setattr(generation_plots, "_load_study", lambda name, storage: study)

    import optuna.visualization as visualization

    for name in ("plot_optimization_history", "plot_param_importances", "plot_slice"):
        monkeypatch.setattr(visualization, name, lambda study: object())
    if all_rejected:

        def no_completed_trials(study):
            raise ValueError("Study must contain completed trials")

        monkeypatch.setattr(visualization, "plot_param_importances", no_completed_trials)

    gen_cfg = SimpleNamespace(
        hpo=SimpleNamespace(enabled=True, storage="sqlite:///unused.db"),
        output_dir=str(tmp_path / "generation"),
        synthcity=SimpleNamespace(enabled=True, names=["model"]),
        tabpfgen=SimpleNamespace(enabled=False, variants=[]),
    )
    cfg = SimpleNamespace(generation=gen_cfg)
    saved = []
    monkeypatch.setattr(
        generation_plots,
        "save_plotly_figure",
        lambda figure, path, formats: saved.append((figure, path, formats)),
    )

    generation_plots.save_hpo_plots(cfg, tmp_path / "plots", hpo_context={})

    outcome_figure, outcome_path, formats = next(
        (figure, path, formats)
        for figure, path, formats in saved
        if str(path).endswith("_trial_outcomes")
    )
    assert formats == ("html",)
    assert outcome_figure.layout.title.text.startswith("HPO trial outcomes (")
    table = outcome_figure.data[0]
    classifications = table.cells.values[2]
    reasons = table.cells.values[4]
    objectives = table.cells.values[3]
    assert "minimum support screen failed" in reasons
    assert str(outcome_path).endswith("_trial_outcomes")

    if all_rejected:
        assert classifications == ["Rejected", "Rejected"]
        assert objectives == ["—", "—"]
        assert "rejected: 2" in outcome_figure.layout.title.text
    else:
        assert classifications == ["Completed", "Pruned", "Rejected", "Pruned"]
        assert reasons[3] == "stage_a_screen_exception: screen failed"
        assert objectives == [0.75, "—", "—", "—"]


def test_hpo_plots_preserve_available_studies_when_configured_study_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    completed_study = SimpleNamespace(
        trials=[
            SimpleNamespace(
                number=4,
                state=SimpleNamespace(name="COMPLETE"),
                value=0.82,
                user_attrs={},
            )
        ],
        user_attrs={
            "hpo_context_schema_version": HPO_CONTEXT_SCHEMA_VERSION,
            "hpo_context_digest": "context-digest",
            "hpo_context": {},
        },
    )
    rejected_study = SimpleNamespace(
        trials=[
            SimpleNamespace(
                number=7,
                state=SimpleNamespace(name="PRUNED"),
                value=None,
                user_attrs={
                    "stage_a_state": "pruned",
                    "stage_a_prune_reasons": ["minimum support screen failed"],
                },
            )
        ],
        user_attrs={
            "hpo_context_schema_version": HPO_CONTEXT_SCHEMA_VERSION,
            "hpo_context_digest": "context-digest",
            "hpo_context": {},
        },
    )
    studies = {
        "hpo_available_before_gap": completed_study,
        "hpo_missing": None,
        "hpo_available_after_gap": rejected_study,
    }
    loaded_names = []

    def load_study(name: str, storage: str):
        loaded_names.append(name)
        return studies[name]

    monkeypatch.setattr(generation_plots, "contextual_study_name", lambda name, context: name)
    monkeypatch.setattr(generation_plots, "hpo_context_digest", lambda context: "context-digest")
    monkeypatch.setattr(generation_plots, "_load_study", load_study)

    import optuna.visualization as visualization

    for name in ("plot_optimization_history", "plot_param_importances", "plot_slice"):
        monkeypatch.setattr(visualization, name, lambda study: object())

    cfg = SimpleNamespace(
        generation=SimpleNamespace(
            hpo=SimpleNamespace(enabled=True, storage="sqlite:///unused.db"),
            output_dir=str(tmp_path / "generation"),
            synthcity=SimpleNamespace(
                enabled=True,
                names=["available_before_gap", "missing", "available_after_gap"],
            ),
            tabpfgen=SimpleNamespace(enabled=False, variants=[]),
        )
    )
    saved = []
    monkeypatch.setattr(
        generation_plots,
        "save_plotly_figure",
        lambda figure, path, formats: saved.append((figure, path, formats)),
    )

    generation_plots.save_hpo_plots(cfg, tmp_path / "plots", hpo_context={})

    assert loaded_names == [
        "hpo_available_before_gap",
        "hpo_missing",
        "hpo_available_after_gap",
    ]
    available_artifact_ids = {
        generation_plots._model_artifact_id(name)
        for name in ("hpo_available_before_gap", "hpo_available_after_gap")
    }
    missing_artifact_id = generation_plots._model_artifact_id("hpo_missing")
    saved_paths = [path for _figure, path, _formats in saved]
    assert saved_paths
    assert all(
        any(path.name.startswith(f"{artifact_id}_") for artifact_id in available_artifact_ids)
        for path in saved_paths
    )
    assert all(not path.name.startswith(f"{missing_artifact_id}_") for path in saved_paths)

    outcome_figures = {
        path.name: figure
        for figure, path, _formats in saved
        if str(path).endswith("_trial_outcomes")
    }
    assert set(outcome_figures) == {
        f"{generation_plots._model_artifact_id(name)}_trial_outcomes"
        for name in ("hpo_available_before_gap", "hpo_available_after_gap")
    }

    completed_cells = (
        outcome_figures[
            f"{generation_plots._model_artifact_id('hpo_available_before_gap')}_trial_outcomes"
        ]
        .data[0]
        .cells.values
    )
    assert completed_cells[0] == [4]
    assert completed_cells[2] == ["Completed"]
    assert completed_cells[3] == [0.82]
    assert completed_cells[4] == ["—"]

    rejected_cells = (
        outcome_figures[
            f"{generation_plots._model_artifact_id('hpo_available_after_gap')}_trial_outcomes"
        ]
        .data[0]
        .cells.values
    )
    assert rejected_cells[0] == [7]
    assert rejected_cells[2] == ["Rejected"]
    assert rejected_cells[3] == ["—"]
    assert rejected_cells[4] == ["minimum support screen failed"]
