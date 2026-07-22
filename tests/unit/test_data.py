"""Unit tests for the pure column-typing/transform helpers in synthdata.data."""

import numpy as np
import pandas as pd
import pytest

from synthdata.data import (
    cast_integer_like_columns,
    decode_label_encoded_columns,
    infer_categorical_columns,
    label_encode_non_numeric_columns,
    mask_outliers_as_missing,
    remap_binary_one_two,
)

pytestmark = pytest.mark.unit


class TestInferCategoricalColumns:
    def test_explicit_list_wins_and_is_filtered_to_features(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
        result = infer_categorical_columns(
            df, feature_columns=["a", "b"], explicit=["a", "not_a_feature"]
        )
        assert result == ["a"]

    def test_uci_metadata_used_when_no_explicit_list(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1.5, 2.5, 3.5]})
        result = infer_categorical_columns(
            df,
            feature_columns=["a", "b"],
            explicit="auto",
            uci_variable_types={"a": "Categorical", "b": "Continuous"},
        )
        assert result == ["a"]

    def test_falls_back_to_heuristic_when_no_uci_categorical_tags(self):
        df = pd.DataFrame({"a": np.linspace(0, 100, 20), "b": ["x", "y"] * 10})
        result = infer_categorical_columns(
            df,
            feature_columns=["a", "b"],
            explicit="auto",
            uci_variable_types={"a": "Continuous", "b": "Continuous"},
        )
        # No UCI column tagged "Categorical" -> heuristic: b is object dtype,
        # a is high-cardinality numeric (stays continuous).
        assert result == ["b"]

    def test_heuristic_dtype_and_cardinality(self):
        df = pd.DataFrame(
            {
                "obj_col": ["x", "y", "z"],
                "bool_col": [True, False, True],
                "low_card_numeric": [1, 1, 2],
                "high_card_numeric": np.linspace(0, 1, 3),
            }
        )
        result = infer_categorical_columns(
            df,
            feature_columns=list(df.columns),
            explicit="auto",
            unique_threshold=2,
        )
        assert set(result) == {"obj_col", "bool_col", "low_card_numeric"}

    def test_no_categorical_columns_inferred(self):
        df = pd.DataFrame({"x": np.linspace(0, 100, 20)})
        result = infer_categorical_columns(
            df, feature_columns=["x"], explicit="auto", unique_threshold=2
        )
        assert result == []


class TestRemapBinaryOneTwo:
    def test_remaps_only_pure_one_two_columns(self):
        df = pd.DataFrame({"binary": [1, 2, 1, 2], "not_binary": [0, 1, 2, 1]})
        out = remap_binary_one_two(df)
        assert out["binary"].tolist() == [0, 1, 0, 1]
        assert out["not_binary"].tolist() == [0, 1, 2, 1]

    def test_no_binary_columns_unchanged(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        out = remap_binary_one_two(df)
        pd.testing.assert_frame_equal(out, df)

    def test_preserves_nans(self):
        df = pd.DataFrame({"binary": [1, 2, np.nan, 2]})
        out = remap_binary_one_two(df)
        assert out["binary"].tolist()[:2] == [0, 1]
        assert np.isnan(out["binary"].iloc[2])
        assert out["binary"].iloc[3] == 1

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"binary": [1, 2, 1]})
        remap_binary_one_two(df)
        assert df["binary"].tolist() == [1, 2, 1]


class TestCastIntegerLikeColumns:
    def test_whole_float_column_cast_to_int(self):
        df = pd.DataFrame({"a": [0.0, 1.0, 2.0]})
        out = cast_integer_like_columns(df, ["a"])
        assert out["a"].dtype == np.int64 or pd.api.types.is_integer_dtype(out["a"])

    def test_non_whole_float_column_stays_float(self):
        df = pd.DataFrame({"a": [0.0, 1.5, 2.0]})
        out = cast_integer_like_columns(df, ["a"])
        assert pd.api.types.is_float_dtype(out["a"])

    def test_column_with_nan_stays_float(self):
        df = pd.DataFrame({"a": [0.0, np.nan, 2.0]})
        out = cast_integer_like_columns(df, ["a"])
        assert pd.api.types.is_float_dtype(out["a"])

    def test_missing_column_skipped_silently(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        out = cast_integer_like_columns(df, ["a", "does_not_exist"])
        assert list(out.columns) == ["a"]

    def test_non_numeric_column_skipped(self):
        df = pd.DataFrame({"a": ["x", "y"]})
        out = cast_integer_like_columns(df, ["a"])
        assert out["a"].tolist() == ["x", "y"]


class TestLabelEncodeRoundtrip:
    def test_string_column_encoded_and_decoded(self):
        df = pd.DataFrame({"color": ["red", "blue", "red", "green"]})
        encoded, maps = label_encode_non_numeric_columns(df, ["color"])
        assert pd.api.types.is_numeric_dtype(encoded["color"])
        decoded = decode_label_encoded_columns(encoded, maps)
        assert decoded["color"].tolist() == df["color"].tolist()

    def test_nan_preserved_through_encoding(self):
        df = pd.DataFrame({"color": ["red", None, "blue"]})
        encoded, maps = label_encode_non_numeric_columns(df, ["color"])
        assert np.isnan(encoded["color"].iloc[1])
        assert "color" in maps

    def test_already_numeric_column_passes_through(self):
        df = pd.DataFrame({"num": [1, 2, 3]})
        encoded, maps = label_encode_non_numeric_columns(df, ["num"])
        assert encoded["num"].tolist() == [1, 2, 3]
        assert "num" not in maps

    def test_decode_clips_out_of_range_codes(self):
        df = pd.DataFrame({"color": ["red", "blue"]})
        _, maps = label_encode_non_numeric_columns(df, ["color"])
        out_of_range = pd.DataFrame({"color": [99.0]})
        decoded = decode_label_encoded_columns(out_of_range, maps)
        # Clipped to the last valid category rather than raising.
        assert decoded["color"].iloc[0] in maps["color"]

    def test_decode_skips_missing_column(self):
        df = pd.DataFrame({"other": [1, 2]})
        decoded = decode_label_encoded_columns(df, {"color": pd.Index(["red", "blue"])})
        assert list(decoded.columns) == ["other"]


class TestMaskOutliersAsMissing:
    def test_single_outlier_masked(self):
        df = pd.DataFrame({"x": [1, 2, 3, 2, 1, 999]})
        out = mask_outliers_as_missing(df, ["x"], threshold=2.0)
        assert np.isnan(out["x"].iloc[-1])
        assert out["x"].iloc[:-1].notna().all()

    def test_no_outliers_untouched(self):
        df = pd.DataFrame({"x": [1, 2, 3, 2, 1]})
        out = mask_outliers_as_missing(df, ["x"], threshold=3.0)
        pd.testing.assert_series_equal(out["x"], df["x"])

    def test_constant_column_skipped(self):
        df = pd.DataFrame({"x": [5, 5, 5, 5]})
        out = mask_outliers_as_missing(df, ["x"], threshold=1.0)
        assert out["x"].tolist() == [5, 5, 5, 5]

    def test_non_numeric_column_skipped(self):
        df = pd.DataFrame({"x": ["a", "b", "c"]})
        out = mask_outliers_as_missing(df, ["x"], threshold=1.0)
        assert out["x"].tolist() == ["a", "b", "c"]

    def test_single_pass_does_not_cascade(self):
        # A single 999 sentinel among mostly-0/1 values; after masking it,
        # the remaining legitimate boundary value (30) should NOT also get
        # flagged in a second implicit pass (function is single-pass only).
        values = [0] * 20 + [1] * 5 + [30, 999]
        df = pd.DataFrame({"x": values})
        out = mask_outliers_as_missing(df, ["x"], threshold=3.0)
        assert np.isnan(out["x"].iloc[-1])  # 999 masked
        assert out["x"].iloc[-2] == 30  # legitimate boundary value kept
