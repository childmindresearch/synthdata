"""Unit tests for the isolated RefiDiff masked-cell benchmark helpers."""

import numpy as np
import pandas as pd
import pytest

from synthdata.imputation.benchmark import (
    apply_artificial_mask,
    benchmark_study_dir,
    create_artificial_mask,
    resolve_score_columns,
    score_masked_cells,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def benchmark_frame():
    return pd.DataFrame(
        {
            "numeric": [1.0, 2.0, 3.0, 4.0, np.nan, 6.0],
            "category": ["a", "b", "a", "b", "a", "b"],
            "sensitive": [0, 1, 0, 1, 0, 1],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )


@pytest.mark.parametrize("mechanism", ["mcar", "mar", "mnar"])
def test_artificial_mask_is_deterministic_and_only_hides_observed_non_sensitive_cells(
    benchmark_frame, mechanism
):
    kwargs = {
        "df": benchmark_frame,
        "feature_columns": ["numeric", "category", "sensitive"],
        "sensitive_columns": ["sensitive"],
        "fraction": 0.3,
        "mechanism": mechanism,
        "seed": 17,
    }

    mask = create_artificial_mask(**kwargs)

    assert mask == create_artificial_mask(**kwargs)
    assert "sensitive" not in mask
    assert "target" not in mask
    for column, positions in mask.items():
        assert positions
        assert benchmark_frame.iloc[positions][column].notna().all()

    masked = apply_artificial_mask(benchmark_frame, mask)
    for column, positions in mask.items():
        assert masked.iloc[positions][column].isna().all()
    assert masked["target"].equals(benchmark_frame["target"])
    assert masked["sensitive"].equals(benchmark_frame["sensitive"])


def test_score_masked_cells_uses_only_held_out_cells_and_observed_scale(benchmark_frame):
    mask = {"numeric": [0, 1], "category": [0, 1]}
    imputed = benchmark_frame.copy()
    imputed.loc[0, "numeric"] = 2.0
    imputed.loc[1, "numeric"] = 2.0
    imputed.loc[0, "category"] = "a"
    imputed.loc[1, "category"] = "a"

    metrics, summary = score_masked_cells(benchmark_frame, imputed, mask, ["category"])

    assert set(metrics["column"]) == {"numeric", "category"}
    assert summary["n_masked"] == 4
    expected_rmse = np.sqrt(0.5) / np.std(np.array([3.0, 4.0, 6.0]))
    assert summary["numeric_macro_standardized_rmse"] == pytest.approx(expected_rmse)
    assert summary["categorical_macro_accuracy"] == pytest.approx(0.5)
    assert summary["categorical_macro_balanced_accuracy"] == pytest.approx(0.5)
    assert summary["objective"] == pytest.approx((expected_rmse + 0.5) / 2)


def test_artificial_mask_rejects_unknown_mechanism(benchmark_frame):
    with pytest.raises(ValueError, match="Unknown artificial masking mechanism"):
        create_artificial_mask(
            benchmark_frame,
            ["numeric"],
            [],
            fraction=0.3,
            mechanism="unsupported",
            seed=0,
        )


def test_mar_mask_does_not_use_target_as_a_missingness_predictor(benchmark_frame):
    frame = benchmark_frame.copy()
    feature_columns = ["numeric", "category"]
    baseline = create_artificial_mask(
        frame, feature_columns, [], fraction=0.3, mechanism="mar", seed=23
    )
    frame["target"] = 1 - frame["target"]

    assert (
        create_artificial_mask(frame, feature_columns, [], fraction=0.3, mechanism="mar", seed=23)
        == baseline
    )


def test_benchmark_study_directory_rejects_path_traversal(make_config):
    cfg = make_config()
    with pytest.raises(ValueError, match="path-safe"):
        benchmark_study_dir(cfg, "../outside")


def test_score_panel_excludes_sensitive_columns(make_dataset):
    dataset = make_dataset()
    excluded = dataset.feature_columns[0]
    dataset.sensitive_columns = [excluded]

    assert resolve_score_columns(dataset, None) == [
        column for column in dataset.feature_columns if column != excluded
    ]
    with pytest.raises(ValueError, match="non-sensitive feature"):
        resolve_score_columns(dataset, [excluded])
