"""Unit tests for the pure column-typing/transform helpers in synthdata.data."""

import json
import logging

import numpy as np
import pandas as pd
import pytest

from synthdata.config import Config, DataConfig
from synthdata.data import (
    _load_local_file,
    cast_integer_like_columns,
    decode_label_encoded_columns,
    decode_ordinal_columns,
    encode_ordinal_columns,
    label_encode_non_numeric_columns,
    load_dataset,
    load_variable_schema,
    mask_outliers_as_missing,
    remap_binary_one_two,
    schema_column_roles,
    split_by_patient,
    stratification_key,
    warn_non_numeric_feature_columns,
)

pytestmark = pytest.mark.unit


class TestVariableSchema:
    def _write_schema(self, tmp_path, text: str):
        path = tmp_path / "variable_schema.csv"
        path.write_text(text)
        return path

    def test_valid_schema_derives_nominal_and_ordinal_roles(self, tmp_path):
        path = self._write_schema(
            tmp_path,
            "column,kind,ordinal_order\n"
            "age,continuous,\n"
            'severity,categorical,"[0, 1, 2]"\n'
            "site,categorical,\n"
            "target,categorical,\n",
        )
        schema, fingerprint = load_variable_schema(path, ["age", "severity", "site", "target"])
        nominal, ordinal, orders = schema_column_roles(schema, "target")
        assert nominal == ["site"]
        assert ordinal == ["severity"]
        assert orders == {"severity": [0, 1, 2]}
        assert len(fingerprint) == 64

    def test_schema_requires_exact_modeling_column_coverage(self, tmp_path):
        path = self._write_schema(
            tmp_path,
            "column,kind\nage,continuous\nstale,categorical\n",
        )
        with pytest.raises(ValueError, match="missing declaration.*target.*stale"):
            load_variable_schema(path, ["age", "target"])

    def test_schema_rejects_duplicate_columns(self, tmp_path):
        path = self._write_schema(
            tmp_path,
            "column,kind\nage,continuous\nage,categorical\ntarget,categorical\n",
        )
        with pytest.raises(ValueError, match="more than once"):
            load_variable_schema(path, ["age", "target"])

    def test_schema_rejects_invalid_kind(self, tmp_path):
        path = self._write_schema(
            tmp_path,
            "column,kind\nage,ratio\ntarget,categorical\n",
        )
        with pytest.raises(ValueError, match="invalid kind"):
            load_variable_schema(path, ["age", "target"])

    def test_schema_rejects_order_for_continuous_column(self, tmp_path):
        path = self._write_schema(
            tmp_path,
            'column,kind,ordinal_order\nage,continuous,"[1, 2]"\ntarget,categorical,\n',
        )
        with pytest.raises(ValueError, match="declared continuous"):
            load_variable_schema(path, ["age", "target"])

    def test_schema_rejects_duplicate_ordinal_order_values(self, tmp_path):
        path = self._write_schema(
            tmp_path,
            'column,kind,ordinal_order\nseverity,categorical,"[0, 1, 1]"\ntarget,categorical,\n',
        )
        with pytest.raises(ValueError, match="duplicate ordinal_order"):
            load_variable_schema(path, ["severity", "target"])

    def test_load_dataset_uses_schema_roles_and_ordinal_encoding(self, tmp_path):
        raw_path = tmp_path / "raw.csv"
        pd.DataFrame(
            {
                "age": [10, 11, 12, 13],
                "severity": ["Low", "High", "Medium", "Low"],
                "site": ["A", "B", "A", "B"],
                "target": [0, 1, 0, 1],
            }
        ).to_csv(raw_path, index=False)
        schema_path = self._write_schema(
            tmp_path,
            "column,kind,ordinal_order\n"
            "age,continuous,\n"
            'severity,categorical,"[""Low"", ""Medium"", ""High""]"\n'
            "site,categorical,\n"
            "target,categorical,\n",
        )
        cfg = Config(
            name="schema_test",
            data=DataConfig(
                source="csv",
                path=str(raw_path),
                target_column="target",
                variable_schema_path=str(schema_path),
                data_dir=str(tmp_path / "derived"),
                train_fraction=0.5,
                tuning_fraction=0.0,
                holdout_fraction=0.5,
            ),
        )

        dataset = load_dataset(cfg)

        assert dataset.nominal_columns == ["site"]
        assert dataset.ordinal_columns == ["severity"]
        assert dataset.target_is_categorical is True
        assert dataset.all_categorical_columns == ["site", "severity", "target"]
        assert dataset.full_df["severity"].tolist() == [0.0, 2.0, 1.0, 0.0]
        assert dataset.variable_schema_fingerprint
        manifest = json.loads((dataset.data_dir / "dataset_manifest.json").read_text())
        assert manifest["variable_schema"]["severity"]["ordinal_order"] == [
            "Low",
            "Medium",
            "High",
        ]
        assert manifest["source_fingerprint"] == dataset.source_fingerprint
        assert manifest["full_fingerprint"] == dataset.full_fingerprint
        assert manifest["train_split_fingerprint"] == dataset.train_split_fingerprint
        assert manifest["test_split_fingerprint"] == dataset.test_split_fingerprint

    def test_continuous_target_remains_continuous_in_dataset_metadata(self, tmp_path):
        raw_path = tmp_path / "raw.csv"
        pd.DataFrame(
            {
                "score": [0, 1, 0, 1],
                "target": [0.0, 1.5, 2.0, 3.5],
            }
        ).to_csv(raw_path, index=False)
        schema_path = self._write_schema(
            tmp_path,
            "column,kind\nscore,continuous\ntarget,continuous\n",
        )
        cfg = Config(
            name="continuous_target_schema_test",
            data=DataConfig(
                source="csv",
                path=str(raw_path),
                target_column="target",
                variable_schema_path=str(schema_path),
                data_dir=str(tmp_path / "derived"),
                train_fraction=0.5,
                tuning_fraction=0.0,
                holdout_fraction=0.5,
            ),
        )

        dataset = load_dataset(cfg)

        assert dataset.target_is_categorical is False
        assert dataset.all_categorical_columns == []
        assert pd.api.types.is_float_dtype(dataset.full_df["target"])


class TestSplitByPatient:
    @staticmethod
    def _visits(n_patients=60, seed=0):
        rng = np.random.default_rng(seed)
        visits = rng.integers(1, 4, size=n_patients)
        patient = np.repeat([f"p{i:03d}" for i in range(n_patients)], visits)
        label = np.repeat(np.arange(n_patients) % 3 == 0, visits).astype(int)
        df = pd.DataFrame({"x": rng.normal(size=len(patient)), "target": label})
        return df, pd.Series(patient, index=df.index)

    FRACTIONS = (0.6, 0.2, 0.2)

    def test_no_patient_spans_two_splits(self):
        df, patients = self._visits()
        splits = split_by_patient(df, patients, self.FRACTIONS, seed=1, strata=None)
        sets = [set(patients[index]) for index in splits]
        assert sets[0].isdisjoint(sets[1]) and sets[0].isdisjoint(sets[2])
        assert sets[1].isdisjoint(sets[2])
        assert sorted(i for index in splits for i in index) == list(df.index)
        # Folds balance rows (encounters), with whole patients in each.
        shares = [len(index) / len(df) for index in splits]
        assert shares == pytest.approx(self.FRACTIONS, abs=0.03)

    def test_patients_are_stratified_by_their_rows_target(self):
        df, patients = self._visits()
        strata = stratification_key(df, ["target"], None, patients, min_groups=5)
        splits = split_by_patient(df, patients, self.FRACTIONS, seed=3, strata=strata)
        overall = df["target"].mean()
        for index in splits:
            assert abs(df.loc[index, "target"].mean() - overall) <= 0.05

    def test_same_seed_gives_the_same_split(self):
        df, patients = self._visits()
        first = split_by_patient(df, patients, self.FRACTIONS, seed=5, strata=None)
        second = split_by_patient(df, patients, self.FRACTIONS, seed=5, strata=None)
        assert all(a.equals(b) for a, b in zip(first, second, strict=True))

    def test_without_patients_every_row_is_its_own_group(self):
        df, _ = self._visits()
        train, tuning, holdout = split_by_patient(df, None, (0.5, 0.2, 0.3), seed=9, strata=None)
        assert (len(train), len(tuning), len(holdout)) == pytest.approx(
            (0.5 * len(df), 0.2 * len(df), 0.3 * len(df)), abs=1
        )

    def test_missing_patient_id_is_rejected(self):
        df, patients = self._visits()
        patients.iloc[0] = None
        with pytest.raises(ValueError, match="missing value"):
            split_by_patient(df, patients, self.FRACTIONS, seed=1, strata=None)

    def test_sparse_joint_strata_fall_back_to_the_first_column(self):
        df = pd.DataFrame({"target": [0] * 10 + [1] * 10, "age": [10] * 9 + [70] + [10] * 10})
        groups = pd.Series(range(20))
        key = stratification_key(df, ["target", "age"], [None, [30, 60]], groups, min_groups=5)
        # The lone 70-year-old is too rare a stratum; it joins plain target 0.
        assert key.iloc[9] == "0"
        assert key.iloc[0] == "0|[-inf, 30.0)"
        assert key.nunique() == 3

    def test_missing_values_form_their_own_stratum(self):
        df = pd.DataFrame({"sex": [0.0, None, 1.0]})
        key = stratification_key(df, ["sex"], None, pd.Series(range(3)), min_groups=0)
        assert key.tolist() == ["0.0", "missing", "1.0"]

    def test_load_dataset_groups_by_patient_and_drops_the_identifier(self, tmp_path):
        df, patients = self._visits()
        df.insert(0, "subject", patients)
        raw_path = tmp_path / "raw.csv"
        df.to_csv(raw_path, index=False)
        cfg = Config(
            name="patients",
            data=DataConfig(
                source="csv",
                path=str(raw_path),
                target_column="target",
                patient_id_column="subject",
                nominal_columns=[],
                data_dir=str(tmp_path / "derived"),
            ),
        )

        dataset = load_dataset(cfg)

        assert "subject" not in dataset.full_df.columns
        assert "subject" not in dataset.feature_columns
        train_patients = set(df.loc[dataset.train_df.index, "subject"])
        test_patients = set(df.loc[dataset.test_df.index, "subject"])
        assert train_patients.isdisjoint(test_patients)
        tuning_patients = set(df.loc[dataset.tuning_index, "subject"])
        assert tuning_patients <= train_patients
        assert tuning_patients.isdisjoint(df.loc[dataset.search_train_df.index, "subject"])
        # train counts tuning too.
        n = dataset.n_patients
        assert n["train"] + n["test"] == 60 and 0 < n["tuning"] < n["train"]
        manifest = json.loads((dataset.data_dir / "dataset_manifest.json").read_text())
        assert manifest["patient_id_column"] == "subject"
        assert manifest["n_patients"] == n
        report = pd.read_csv(dataset.data_dir / "split_report.csv")
        assert set(report["split"]) == {"train", "tuning", "holdout"}
        assert set(report["column"]) == {"target", "(all)"}

    def test_unknown_patient_column_is_rejected(self, tmp_path):
        df, _ = self._visits()
        raw_path = tmp_path / "raw.csv"
        df.to_csv(raw_path, index=False)
        cfg = Config(
            name="patients",
            data=DataConfig(
                source="csv",
                path=str(raw_path),
                target_column="target",
                patient_id_column="subject",
                nominal_columns=[],
                data_dir=str(tmp_path / "derived"),
            ),
        )
        with pytest.raises(KeyError, match="patient_id_column"):
            load_dataset(cfg)


def test_legacy_dataset_assumes_categorical_target(make_dataset):
    dataset = make_dataset()

    assert dataset.target_is_categorical is True
    assert dataset.all_categorical_columns == ["target"]


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

    def test_categorical_columns_forces_factorize_of_numeric_column(self):
        # A numeric-but-categorical column with a non-zero-indexed/gapped
        # domain (e.g. a real 5-level ordinal stored as {1..5}, or a nominal
        # stored as {0, 2}) must be factorized like a string column when
        # declared in categorical_columns, not passed through raw -- some
        # backends' categorical handling returns compact 0-indexed codes
        # regardless of the input's actual values, silently shifting the
        # column's domain in their output unless it goes through this same
        # factorize/decode round-trip.
        df = pd.DataFrame({"marital_status": [1, 3, 5, 1]})
        encoded, maps = label_encode_non_numeric_columns(
            df, ["marital_status"], categorical_columns=["marital_status"]
        )
        assert "marital_status" in maps
        assert encoded["marital_status"].tolist() == [0.0, 1.0, 2.0, 0.0]
        decoded = decode_label_encoded_columns(encoded, maps)
        assert decoded["marital_status"].tolist() == df["marital_status"].tolist()

    def test_categorical_columns_does_not_affect_uncategorical_numeric_columns(self):
        df = pd.DataFrame({"num": [1, 2, 3], "cat": [1, 3, 5]})
        encoded, maps = label_encode_non_numeric_columns(
            df, ["num", "cat"], categorical_columns=["cat"]
        )
        assert "num" not in maps
        assert encoded["num"].tolist() == [1, 2, 3]
        assert "cat" in maps


class TestEncodeOrdinalColumns:
    def test_encodes_in_configured_order_not_alphabetical(self):
        df = pd.DataFrame({"activity": ["Heavy", "Light", "Very Light", None]})
        out = encode_ordinal_columns(
            df, {"activity": ["Very Light", "Light", "Moderate", "Heavy", "Exceptional"]}
        )
        assert out["activity"].tolist()[:3] == [3.0, 1.0, 0.0]
        assert np.isnan(out["activity"].iloc[3])

    def test_decodes_back_to_configured_string_labels(self):
        df = pd.DataFrame({"activity": ["Heavy", "Light", "Very Light", None]})
        encoded = encode_ordinal_columns(
            df, {"activity": ["Very Light", "Light", "Moderate", "Heavy"]}
        )

        decoded = decode_ordinal_columns(
            encoded, {"activity": ["Very Light", "Light", "Moderate", "Heavy"]}
        )

        assert decoded["activity"].tolist()[:3] == ["Heavy", "Light", "Very Light"]
        assert pd.isna(decoded["activity"].iloc[3])

    def test_rejects_invalid_model_space_code(self):
        with pytest.raises(ValueError, match="activity"):
            decode_ordinal_columns(
                pd.DataFrame({"activity": [0.5]}),
                {"activity": ["Low", "High"]},
            )

    def test_raises_on_unknown_observed_category(self):
        df = pd.DataFrame({"activity": ["Light", "Unheard Of"]})
        with pytest.raises(ValueError, match="Unheard Of"):
            encode_ordinal_columns(df, {"activity": ["Light", "Heavy"]})

    def test_raises_on_missing_column(self):
        df = pd.DataFrame({"other": [1, 2]})
        with pytest.raises(KeyError, match="activity"):
            encode_ordinal_columns(df, {"activity": ["Light", "Heavy"]})


class TestWarnNonNumericFeatureColumns:
    def test_no_warning_when_all_declared_numeric_columns_are_numeric(self, caplog):
        df = pd.DataFrame({"num": [1.0, 2.0], "cat": ["a", "b"]})
        backend_logger = logging.getLogger("synthdata.data")
        backend_logger.addHandler(caplog.handler)
        try:
            with caplog.at_level("WARNING"):
                offending = warn_non_numeric_feature_columns(df, ["num", "cat"], ["cat"])
        finally:
            backend_logger.removeHandler(caplog.handler)
        assert offending == []
        assert caplog.text == ""

    def test_warns_on_string_valued_column_not_declared_categorical(self, caplog):
        # "activity" is not in nominal_columns/ordinal_columns, so it's assumed
        # numeric, but actually holds text values (e.g. an ordinal band never
        # encoded).
        df = pd.DataFrame({"num": [1.0, 2.0], "activity": ["Light", "Heavy"]})
        backend_logger = logging.getLogger("synthdata.data")
        backend_logger.addHandler(caplog.handler)
        try:
            with caplog.at_level("WARNING"):
                offending = warn_non_numeric_feature_columns(df, ["num", "activity"], [])
        finally:
            backend_logger.removeHandler(caplog.handler)
        assert offending == ["activity"]
        assert "activity" in caplog.text


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


class TestLoadLocalFile:
    """``_load_local_file`` dispatches on the file extension (CSV vs. Parquet),
    regardless of the configured ``data.source`` string -- see its docstring.
    """

    def _cfg(self, path) -> Config:
        return Config(data=DataConfig(source="csv", path=str(path), target_column="target"))

    def test_reads_csv_file(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "target": [0, 1]})
        path = tmp_path / "data.csv"
        df.to_csv(path, index=False)
        out, variable_types = _load_local_file(self._cfg(path))
        pd.testing.assert_frame_equal(out, df)
        assert variable_types is None

    def test_reads_parquet_file(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "target": [0, 1]})
        path = tmp_path / "data.parquet"
        df.to_parquet(path, index=False)
        out, variable_types = _load_local_file(self._cfg(path))
        pd.testing.assert_frame_equal(out, df)
        assert variable_types is None

    def test_reads_pq_extension_as_parquet(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "target": [0, 1]})
        path = tmp_path / "data.pq"
        df.to_parquet(path, index=False)
        out, _ = _load_local_file(self._cfg(path))
        pd.testing.assert_frame_equal(out, df)

    def test_unsupported_extension_raises(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text("{}")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            _load_local_file(self._cfg(path))
