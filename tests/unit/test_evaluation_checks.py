"""Output sanity checks (synthdata.evaluation.checks)."""

import types

import numpy as np
import pandas as pd
import pytest

from synthdata.evaluation.checks import run_output_checks


@pytest.fixture
def dataset():
    rng = np.random.default_rng(0)
    n = 200

    def frame():
        return pd.DataFrame(
            {
                "age": rng.uniform(20, 80, n).round(1),
                "sex": rng.integers(0, 2, n),
                "target": np.repeat([0, 1], n // 2),
            }
        )

    return types.SimpleNamespace(
        train_imputed_df=frame(),
        test_imputed_df=frame(),
        target_column="target",
        target_is_categorical=True,
        all_categorical_columns=["sex", "target"],
    )


def _status(table, check, model):
    row = table[(table["check"] == check) & (table["model"] == model)]
    assert len(row) == 1, (check, model)
    return row["status"].iloc[0]


def _combined(scores: dict) -> pd.DataFrame:
    """Combined-table stand-in: model -> (utility, privacy, tstr_macro_f1)."""
    frame = pd.DataFrame(
        {
            ("__all__", "utility", "rank"): {m: v[0] for m, v in scores.items()},
            ("__all__", "privacy", "rank"): {m: v[1] for m, v in scores.items()},
            ("custom", "utility", "tstr_macro_f1"): {m: v[2] for m, v in scores.items()},
        }
    )
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    return frame


def test_clean_synthetic_data_passes(dataset):
    rng = np.random.default_rng(1)
    synthetic = dataset.test_imputed_df.copy()
    synthetic["age"] = (synthetic["age"] + rng.normal(0, 0.01, len(synthetic))).clip(21, 79)
    table = run_output_checks(dataset, {"good": synthetic}, pd.DataFrame(), True)
    assert (table["status"] == "pass").all(), table


def test_copied_invented_and_out_of_range_rows_warn(dataset):
    bad = dataset.train_imputed_df.copy()  # every row is a training row
    bad.loc[:9, "sex"] = 7  # 5% of categorical cells hold an unseen code
    bad.loc[:9, "age"] = 500.0
    bad["target"] = 0  # one class only
    table = run_output_checks(dataset, {"bad": bad}, pd.DataFrame(), True)
    for check in (
        "exact_copies_of_train_rows",
        "unseen_categories",
        "out_of_train_range",
        "class_shares_match_train",
    ):
        assert _status(table, check, "bad") == "warn", check


def test_missing_column_is_reported_and_stops_the_other_data_checks(dataset):
    synthetic = dataset.test_imputed_df.drop(columns="age")
    table = run_output_checks(dataset, {"m": synthetic}, pd.DataFrame(), True)
    assert table["check"].tolist() == ["columns_match"]
    assert _status(table, "columns_match", "m") == "warn"


def test_baselines_are_skipped_by_the_data_checks(dataset):
    table = run_output_checks(
        dataset, {"baseline_train_copy": dataset.train_imputed_df}, pd.DataFrame(), True
    )
    assert table.empty


def test_baselines_in_their_expected_places_pass():
    combined = _combined(
        {
            "baseline_train_copy": (1.0, 0.0, 0.8),
            "baseline_marginals": (0.0, 1.0, 0.5),
            "ctgan": (0.6, 0.6, 0.7),
            "ctgan__rep1": (0.5, 0.5, 0.6),
        }
    )
    dataset = types.SimpleNamespace(train_imputed_df=None, test_imputed_df=None)
    table = run_output_checks(dataset, {}, combined, True)
    assert len(table) == 3
    assert (table["status"] == "pass").all(), table


def test_baselines_out_of_place_warn():
    combined = _combined(
        {
            "baseline_train_copy": (0.2, 0.9, 0.4),
            "baseline_marginals": (0.3, 1.0, 0.5),
            "ctgan": (0.6, 0.6, 0.7),
        }
    )
    dataset = types.SimpleNamespace(train_imputed_df=None, test_imputed_df=None)
    table = run_output_checks(dataset, {}, combined, True)
    assert (table["status"] == "warn").all(), table
    assert len(table) == 3
