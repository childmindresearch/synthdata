"""Tests for persisted evaluation artifacts used by artifact-only plotting."""

import pandas as pd
import pytest

from synthdata.evaluation.artifacts import (
    artifact_bundle_dir,
    load_log_disparity_reports,
    persist_evaluation_artifacts,
    verify_native_syntheval_artifacts,
)
from synthdata.log_disparity.metric_log_disparity import build_log_disparity_report_figure

pytestmark = pytest.mark.unit


def _combined() -> pd.DataFrame:
    frame = pd.DataFrame(index=["model_a"])
    frame[("__all__", "utility", "rank")] = [0.5]
    frame[("__all__", "privacy", "rank")] = [0.5]
    frame[("__all__", "fairness", "rank")] = [0.5]
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    return frame


def _report() -> dict:
    hierarchy = pd.DataFrame(
        {
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
        "summary_stats": {
            "model": "model_a",
            "n_subgroups": 1,
            "mean_abs_log_disparity": 0.0,
            "median_abs_log_disparity": 0.0,
            "share_significant_bh": 0.0,
        },
        "leaf_results": hierarchy.copy(),
        "hierarchy_results": hierarchy,
        "subgroup_table": pd.DataFrame(
            {
                "Characteristic": ["Target"],
                "Protected Subgroup": ["positive"],
                "Equity Value": [0.0],
                "BH-adjusted p-value": [1.0],
                "EquityColor": ["#ffffff"],
                "EquityLabel": ["Equal"],
            }
        ),
        "leaf_equity_table": pd.DataFrame(
            {
                "Protected Subgroup": ["positive"],
                "Equity Value": [0.0],
                "BH-adjusted p-value": [1.0],
                "EquityColor": ["#ffffff"],
                "EquityLabel": ["Equal"],
            }
        ),
        "legend_table": pd.DataFrame(
            {
                "Description": ["Equal"],
                "Metric Value Rule": ["0"],
                "Color": ["#ffffff"],
            }
        ),
        "label_counts": pd.DataFrame(
            {"Model": ["model_a"], "EquityLabel": ["Equal"], "count": [1]}
        ),
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
    assert loaded["model_a"]["summary_stats"]["model"] == "model_a"
    assert artifact_bundle_dir(evaluation_dir).joinpath("manifest.json").exists()
    assert build_log_disparity_report_figure(loaded["model_a"]).data


def test_missing_native_plot_is_reported(tmp_path):
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

    with pytest.raises(FileNotFoundError, match="Native SynthEval"):
        verify_native_syntheval_artifacts(evaluation_dir)
