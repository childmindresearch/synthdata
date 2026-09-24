"""Tests for evaluation rank and log-disparity plotting filters."""

import json
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from synthdata.config import Config
from synthdata.evaluation.artifacts import _model_artifact_id
from synthdata.plotting.evaluation_plots import (
    plot_rank_tradeoff,
    plot_rank_tradeoff_3d,
    save_log_disparity_plots,
    save_rank_tradeoff_plots,
)

pytestmark = pytest.mark.unit


def _combined(rows: dict[str, tuple[float, float, float, float]]) -> pd.DataFrame:
    columns = pd.MultiIndex.from_tuples(
        [
            ("__all__", "utility", "rank"),
            ("__all__", "privacy", "rank"),
            ("__all__", "fairness", "rank"),
            ("__all__", "overall", "rank"),
            ("__all__", "overall", "decision_eligible"),
            ("__all__", "overall", "audit_status"),
        ]
    )
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=columns[:4])
    frame[("__all__", "overall", "decision_eligible")] = True
    frame[("__all__", "overall", "audit_status")] = "succeeded"
    return frame


def test_rank_tradeoff_filters_nonfinite_rows_and_annotations():
    combined = _combined(
        {
            "complete": (1.0, 2.0, 3.0, 2.0),
            "missing": (np.nan, 2.0, 3.0, 2.0),
            "infinite": (1.0, np.inf, 3.0, 2.0),
        }
    )
    figure = plot_rank_tradeoff(
        combined,
        ("__all__", "utility", "rank"),
        ("__all__", "privacy", "rank"),
        "Utility",
        "Privacy",
        "Trade-off",
    )
    assert len(figure.axes[0].collections) == 1
    assert [text.get_text() for text in figure.axes[0].texts] == ["complete"]


def test_rank_tradeoff_3d_omits_incomplete_rows_without_raising():
    figure = plot_rank_tradeoff_3d(
        _combined(
            {
                "complete": (1.0, 2.0, 3.0, 2.0),
                "incomplete": (1.0, np.nan, 3.0, np.nan),
            }
        )
    )
    model_traces = [trace for trace in figure.data if trace.mode == "markers+text"]
    assert [trace.name for trace in model_traces] == ["complete"]


def test_partial_static_rank_plot_labels_missing_outputs_and_filters_absent_models():
    missing_outputs = ("not-generated-a", "not-generated-b")
    combined = _combined(
        {
            "complete": (1.0, 2.0, 3.0, 2.0),
            "not-generated-a": (4.0, 5.0, 6.0, 5.0),
        }
    )

    figure = plot_rank_tradeoff(
        combined,
        ("__all__", "utility", "rank"),
        ("__all__", "privacy", "rank"),
        "Utility rank",
        "Privacy rank",
        "Trade-off",
        produced_outputs=("complete",),
        missing_stage_a_outputs=missing_outputs,
    )

    assert len(figure.axes[0].collections) == 1
    assert [text.get_text() for text in figure.axes[0].texts] == ["complete"]
    assert "PARTIAL COHORT" in figure.axes[0].get_title()
    assert all(output in figure.axes[0].get_title() for output in missing_outputs)
    assert figure._synthdata_status == {
        "status": "partial",
        "reason": "Partial cohort; missing Stage A outputs: not-generated-a, not-generated-b",
        "generation_status": "partial",
        "missing_stage_a_outputs": list(missing_outputs),
    }


def test_partial_interactive_rank_plot_metadata_lists_missing_outputs_and_real_models():
    missing_outputs = ("not-generated",)
    combined = _combined(
        {
            "complete": (1.0, 2.0, 3.0, 2.0),
            "not-generated": (4.0, 5.0, 6.0, 5.0),
        }
    )

    figure = plot_rank_tradeoff_3d(
        combined,
        produced_outputs=("complete",),
        missing_stage_a_outputs=missing_outputs,
    )

    model_traces = [trace for trace in figure.data if trace.mode == "markers+text"]
    assert [trace.name for trace in model_traces] == ["complete"]
    assert "PARTIAL COHORT" in figure.layout.title.text
    assert "not-generated" in figure.layout.title.text
    assert figure.layout.meta == {
        "status": "partial",
        "reason": "Partial cohort; missing Stage A outputs: not-generated",
        "generation_status": "partial",
        "missing_stage_a_outputs": ["not-generated"],
    }


def test_complete_rank_plots_keep_existing_status_and_title():
    combined = _combined({"complete": (1.0, 2.0, 3.0, 2.0)})

    static_figure = plot_rank_tradeoff(
        combined,
        ("__all__", "utility", "rank"),
        ("__all__", "privacy", "rank"),
        "Utility rank",
        "Privacy rank",
        "Trade-off",
    )
    interactive_figure = plot_rank_tradeoff_3d(combined)

    assert static_figure.axes[0].get_title() == "Trade-off"
    assert static_figure._synthdata_status == {"status": "succeeded", "reason": ""}
    assert interactive_figure.layout.title.text == "Utility, Privacy, and Fairness Rank Trade-off"
    assert interactive_figure.layout.meta == {"status": "succeeded", "reason": ""}


@pytest.mark.parametrize(
    ("missing_outputs", "expected_status"),
    [(("not-generated",), "partial"), ((), "succeeded")],
)
def test_saved_static_rank_plot_embeds_status_metadata(
    monkeypatch, tmp_path, missing_outputs, expected_status
):
    combined = _combined(
        {
            "complete": (1.0, 2.0, 3.0, 2.0),
            "not-generated": (4.0, 5.0, 6.0, 5.0),
        }
    )
    monkeypatch.setattr(
        "synthdata.plotting.evaluation_plots.save_plotly_figure",
        lambda figure, path, formats: None,
    )
    cfg = cast(Config, SimpleNamespace(plots=SimpleNamespace(dpi=40, formats=["png"])))

    save_rank_tradeoff_plots(
        cfg,
        combined,
        tmp_path,
        produced_outputs=("complete",),
        missing_stage_a_outputs=missing_outputs,
    )

    plot_path = tmp_path / "evaluation" / "utility_vs_privacy.png"
    with Image.open(plot_path) as artifact:
        metadata = json.loads(artifact.info["Description"])
    assert metadata["status"] == expected_status
    assert metadata.get("missing_stage_a_outputs", []) == list(missing_outputs)
    if missing_outputs:
        assert metadata["generation_status"] == "partial"
        assert "not-generated" in metadata["reason"]
    else:
        assert metadata == {"status": "succeeded", "reason": ""}


def test_rank_tradeoff_supports_more_than_20_base_models():
    models = {f"model_{index:02d}": (1.0, 2.0, 3.0, 2.0) for index in range(21)}
    figure = plot_rank_tradeoff(
        _combined(models),
        ("__all__", "utility", "rank"),
        ("__all__", "privacy", "rank"),
        "Utility",
        "Privacy",
        "Trade-off",
    )

    assert len(figure.axes[0].collections) == 21
    assert [text.get_text() for text in figure.axes[0].texts] == sorted(models)
    colors = [tuple(collection.get_facecolors()[0]) for collection in figure.axes[0].collections]
    assert len(set(colors)) == len(models)


def test_rank_tradeoff_3d_supports_more_than_20_base_models():
    models = {f"model_{index:02d}": (1.0, 2.0, 3.0, 2.0) for index in range(21)}
    figure = plot_rank_tradeoff_3d(_combined(models))
    model_traces = [trace for trace in figure.data if trace.mode == "markers+text"]

    assert [trace.name for trace in model_traces] == sorted(models)
    assert len({trace.marker.color for trace in model_traces}) == len(models)


def test_rank_plots_with_no_eligible_rows_are_valid_empty_plots():
    combined = _combined({"incomplete": (np.nan, np.nan, np.nan, np.nan)})
    figure = plot_rank_tradeoff_3d(combined)
    assert figure.data == ()
    assert figure.layout.meta["status"] == "indeterminate"


def test_rank_plot_with_absent_status_columns_is_withheld():
    combined = _combined({"complete": (1.0, 2.0, 3.0, 2.0)}).drop(
        columns=[
            ("__all__", "overall", "decision_eligible"),
            ("__all__", "overall", "audit_status"),
        ]
    )
    figure = plot_rank_tradeoff_3d(combined)
    assert figure.layout.meta["status"] == "withheld"


def test_rank_plot_filters_failed_and_indeterminate_records():
    combined = _combined({"complete": (1.0, 2.0, 3.0, 2.0), "failed": (2, 2, 2, 2)})
    combined[("__all__", "overall", "decision_eligible")] = [True, True]
    combined[("__all__", "overall", "audit_status")] = ["succeeded", "failed"]
    figure = plot_rank_tradeoff_3d(combined)
    assert [trace.name for trace in figure.data if trace.mode == "markers+text"] == ["complete"]


def test_save_log_disparity_plots_only_saves_succeeded_reports(monkeypatch, tmp_path):
    saved = []
    monkeypatch.setattr(
        "synthdata.plotting.evaluation_plots.save_plotly_figure",
        lambda figure, path, formats: saved.append(path.name),
    )
    save_log_disparity_plots(
        {
            "failed": {"state": "failed", "report_figure": object()},
            "indeterminate": {"state": "indeterminate", "report_figure": object()},
            "complete": {"state": "succeeded", "report_figure": object()},
        },
        tmp_path,
    )
    assert saved == [_model_artifact_id("complete")]


def test_save_log_disparity_plots_uses_artifact_id_for_path_like_model_names(monkeypatch, tmp_path):
    saved = []
    monkeypatch.setattr(
        "synthdata.plotting.evaluation_plots.save_plotly_figure",
        lambda figure, path, formats: saved.append(path),
    )
    model_name = "team/model: candidate"

    save_log_disparity_plots(
        {model_name: {"state": "succeeded", "report_figure": object()}},
        tmp_path,
    )

    assert saved[0].parent == tmp_path / "evaluation" / "log_disparity"
    assert saved[0].name != model_name
    assert "/" not in saved[0].name
