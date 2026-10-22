"""Unit tests for missing-block detection, labels and block indicators."""

import numpy as np
import pandas as pd
import pytest

from synthdata.data import add_block_indicators, add_missing_indicators, remask_synthetic
from synthdata.missing_blocks import (
    FORM_NOT_GIVEN,
    ITEM_MISSING,
    SKIP_LOGIC,
    describe_patterns,
    find_blocks,
)

pytestmark = pytest.mark.unit


def _clinic(n=600, seed=0):
    """A form skipped by rule, a form not given at random, and scattered gaps."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "age": rng.integers(18, 80, n).astype(float),
            "uses_substances": rng.choice(["no", "yes"], n),
            "freq_a": rng.integers(0, 5, n).astype(float),
            "freq_b": rng.integers(0, 5, n).astype(float),
            "form_x": rng.normal(size=n),
            "form_y": rng.normal(size=n),
            "form_z": rng.normal(size=n),
            "lab": rng.normal(size=n),
            "target": rng.integers(0, 2, n),
        }
    )
    frame.loc[frame["uses_substances"] == "no", ["freq_a", "freq_b"]] = np.nan
    not_given = rng.random(n) < 0.3
    frame.loc[not_given, ["form_x", "form_y", "form_z"]] = np.nan
    frame.loc[rng.random(n) < 0.03, "form_y"] = np.nan  # a few extra item gaps
    frame.loc[rng.random(n) < 0.2, "lab"] = np.nan
    return frame


FEATURES = ["age", "uses_substances", "freq_a", "freq_b", "form_x", "form_y", "form_z", "lab"]


def test_finds_columns_missing_together():
    blocks = find_blocks(_clinic()[FEATURES].isna())
    assert blocks == [["form_x", "form_y", "form_z"], ["freq_a", "freq_b"]]


def test_labels_skip_logic_forms_and_scattered_gaps():
    table = describe_patterns(_clinic(), FEATURES, exclude_from_rules=["target"]).set_index(
        "columns"
    )
    assert table.loc["freq_a; freq_b", "label"] == SKIP_LOGIC
    assert "uses_substances" in table.loc["freq_a; freq_b", "rule"]
    assert table.loc["form_x; form_y; form_z", "label"] == FORM_NOT_GIVEN
    assert table.loc["form_x; form_y; form_z", "whole_share"] > 0.85
    assert table.loc["lab", "label"] == ITEM_MISSING
    assert table.loc["lab", "kind"] == "column"
    assert table["rule_auc"].loc["lab"] < 0.7


def test_block_flag_covers_whole_rows_and_column_flags_only_the_rest():
    frame = _clinic()
    widened, blocks, covered = add_block_indicators(frame, {"form": ["form_x", "form_y", "form_z"]})
    assert blocks == {"form__missing": ["form_x", "form_y", "form_z"]}
    whole = frame[["form_x", "form_y", "form_z"]].isna().all(axis=1)
    assert widened["form__missing"].eq(whole.astype(int)).all()
    # form_y's extra gaps (~3%) fall below 5% once whole-block rows are left out.
    widened, flags = add_missing_indicators(widened, FEATURES, frame.index, 0.05, covered)
    assert "form_y__missing" not in flags and "form_x__missing" not in flags
    assert "lab__missing" in flags
    # With a lower threshold the leftover gaps get their own flag, without the block rows.
    _, flags = add_missing_indicators(frame, ["form_y"], frame.index, 0.01, covered)
    assert flags == {"form_y__missing": "form_y"}


def test_remask_blanks_whole_blocks_and_drops_every_flag():
    synthetic = pd.DataFrame(
        {
            "a": [1.0, 2.0],
            "b": [3.0, 4.0],
            "c": [5.0, 6.0],
            "blk__missing": [1, 0],
            "c__missing": [0, 1],
        }
    )
    released = remask_synthetic(synthetic, {"c__missing": "c"}, {"blk__missing": ["a", "b"]})
    assert released.columns.tolist() == ["a", "b", "c"]
    assert released.isna().to_numpy().tolist() == [[True, True, False], [False, False, True]]


def test_load_dataset_adds_auto_block_flags(tmp_path):
    from synthdata.config import Config, DataConfig
    from synthdata.data import load_dataset

    path = tmp_path / "raw.csv"
    _clinic().to_csv(path, index=False)
    cfg = Config(
        name="blocks",
        data=DataConfig(
            source="csv",
            path=str(path),
            target_column="target",
            nominal_columns=["uses_substances"],
            data_dir=str(tmp_path / "derived"),
        ),
    )
    cfg.imputation.missing_indicators.blocks = "auto"
    dataset = load_dataset(cfg)
    assert dataset.missing_block_columns == {
        "block_1__missing": ["form_x", "form_y", "form_z"],
        "block_2__missing": ["freq_a", "freq_b"],
    }
    assert {"block_1__missing", "block_2__missing", "lab__missing"} <= set(dataset.feature_columns)
    assert "freq_a__missing" not in dataset.feature_columns
    assert "block_1__missing" in dataset.nominal_columns

    cfg.imputation.missing_indicators.blocks = {"form": ["form_x", "nope"]}
    with pytest.raises(KeyError, match="nope"):
        load_dataset(cfg)


def test_blocks_config_is_validated(make_config):
    from synthdata.config import _validate

    cfg = make_config()
    cfg.imputation.missing_indicators.blocks = "sometimes"
    with pytest.raises(ValueError, match="'auto'"):
        _validate(cfg)
    cfg.imputation.missing_indicators.blocks = {"one": ["a"]}
    with pytest.raises(ValueError, match="at least two"):
        _validate(cfg)


def test_indicator_only_columns_stay_out_of_blocks_and_other_forms_are_not_rules():
    frame = _clinic()
    # A second form missing in exactly the same rows as form_x/y/z.
    frame["form2_a"] = frame["form_x"].where(frame["form_x"].notna(), np.nan) * 0 + 1
    frame.loc[frame["form_x"].isna() & frame["form_z"].isna(), "form2_a"] = np.nan
    features = [*FEATURES, "form2_a"]
    table = describe_patterns(
        frame, features, exclude_from_rules=["target"], exclude_from_blocks=["form_z"]
    ).set_index("columns")
    assert "form_x; form_y; form2_a" in table.index
    assert bool(table.loc["form_z", "indicator_only"])
    # The other co-missing columns are not usable predictors, so no "rule" is found.
    assert table.loc["form_x; form_y; form2_a", "label"] == FORM_NOT_GIVEN
