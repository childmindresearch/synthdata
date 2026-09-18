"""Unit tests for the pure column-typing/transform helpers in synthdata.data."""

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from synthdata.config import Config, DataConfig, DataSplitConfig
from synthdata.data import (
    _load_local_file,
    cast_integer_like_columns,
    decode_label_encoded_columns,
    decode_ordinal_columns,
    encode_ordinal_columns,
    infer_nominal_columns,
    label_encode_non_numeric_columns,
    load_dataset,
    load_variable_schema,
    mask_outliers_as_missing,
    remap_binary_one_two,
    schema_column_roles,
    semantic_context_fingerprint,
    semantic_context_payload,
    validate_imputation_cache_lineage,
    warn_non_numeric_feature_columns,
    write_dataset_manifest,
)

pytestmark = pytest.mark.unit


def test_candidate_imputation_lineage_rejects_stale_assignment_and_identity(
    make_config, make_canonical_dataset
):
    from synthdata.imputation.pipeline import _cache_key_record

    dataset = make_canonical_dataset()
    cfg = make_config()
    record = _cache_key_record(cfg, dataset)
    record["assignment_fingerprint"] = "stale-assignment"
    record["identity_fingerprint"] = "stale-identity"
    (dataset.data_dir / ".imputation_cache_key.json").write_text(json.dumps(record))

    with pytest.raises(RuntimeError, match="rerun imputation before generation"):
        validate_imputation_cache_lineage(dataset, record, required=True)


def test_candidate_imputation_lineage_rejects_persisted_cache_key_mismatch(
    make_config, make_canonical_dataset
):
    from synthdata.imputation.pipeline import _cache_key_record

    dataset = make_canonical_dataset()
    record = _cache_key_record(make_config(), dataset)
    persisted = dict(record, cache_key="old-lineage")
    (dataset.data_dir / ".imputation_cache_key.json").write_text(json.dumps(persisted))

    with pytest.raises(RuntimeError, match="cache key does not match"):
        validate_imputation_cache_lineage(dataset, record, required=True)


def test_candidate_imputation_lineage_accepts_current_cache(make_config, make_canonical_dataset):
    from synthdata.imputation.pipeline import _cache_key_record, run_imputation

    dataset = make_canonical_dataset()
    record = _cache_key_record(make_config(), dataset)
    run_imputation(make_config(), dataset)
    record = json.loads((dataset.data_dir / ".imputation_cache_key.json").read_text())
    (dataset.data_dir / ".imputation_cache_key.json").write_text(json.dumps(record))
    validate_imputation_cache_lineage(dataset, record, required=True)


def test_candidate_imputation_lineage_rejects_absent_fit_state(make_config, make_canonical_dataset):
    from synthdata.imputation.pipeline import run_imputation

    dataset = make_canonical_dataset()
    run_imputation(make_config(), dataset)
    path = dataset.data_dir / ".imputation_cache_key.json"
    record = json.loads(path.read_text())
    del record["fit_state"]
    path.write_text(json.dumps(record))

    with pytest.raises(RuntimeError, match="fit state is missing or invalid"):
        validate_imputation_cache_lineage(dataset, record, required=True)


def test_candidate_imputation_lineage_rejects_mismatched_fit_state(
    make_config, make_canonical_dataset
):
    from synthdata.imputation.pipeline import run_imputation

    dataset = make_canonical_dataset()
    run_imputation(make_config(), dataset)
    path = dataset.data_dir / ".imputation_cache_key.json"
    record = json.loads(path.read_text())
    record["fit_state"]["fit_frame_fingerprint"] = "stale"
    path.write_text(json.dumps(record))

    with pytest.raises(RuntimeError, match="fit state is missing or invalid"):
        validate_imputation_cache_lineage(dataset, record, required=True)


def test_candidate_imputation_lineage_rejects_old_fingerprint_contract(
    make_config, make_canonical_dataset
):
    from synthdata.imputation.pipeline import run_imputation

    dataset = make_canonical_dataset()
    run_imputation(make_config(), dataset)
    path = dataset.data_dir / ".imputation_cache_key.json"
    record = json.loads(path.read_text())
    del record["fit_state"]["fit_frame_fingerprint_version"]
    record["fit_state"]["state_fingerprint"] = "stale"
    path.write_text(json.dumps(record))

    with pytest.raises(RuntimeError, match="fit state is missing or invalid"):
        validate_imputation_cache_lineage(dataset, record, required=True)


class TestInferNominalColumns:
    def test_explicit_list_wins_and_is_filtered_to_features(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
        result = infer_nominal_columns(
            df, feature_columns=["a", "b"], explicit=["a", "not_a_feature"]
        )
        assert result == ["a"]

    def test_explicit_list_excludes_ordinal_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
        result = infer_nominal_columns(
            df, feature_columns=["a", "b"], explicit=["a", "b"], ordinal_columns=["b"]
        )
        assert result == ["a"]

    def test_uci_metadata_used_when_no_explicit_list(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1.5, 2.5, 3.5]})
        result = infer_nominal_columns(
            df,
            feature_columns=["a", "b"],
            explicit="auto",
            uci_variable_types={"a": "Categorical", "b": "Continuous"},
        )
        assert result == ["a"]

    def test_falls_back_to_heuristic_when_no_uci_categorical_tags(self):
        df = pd.DataFrame({"a": np.linspace(0, 100, 20), "b": ["x", "y"] * 10})
        result = infer_nominal_columns(
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
        result = infer_nominal_columns(
            df,
            feature_columns=list(df.columns),
            explicit="auto",
            unique_threshold=2,
        )
        assert set(result) == {"obj_col", "bool_col", "low_card_numeric"}

    def test_heuristic_excludes_ordinal_columns(self):
        df = pd.DataFrame({"obj_col": ["x", "y", "z"], "low_card_numeric": [1, 1, 2]})
        result = infer_nominal_columns(
            df,
            feature_columns=list(df.columns),
            explicit="auto",
            ordinal_columns=["low_card_numeric"],
            unique_threshold=2,
        )
        assert result == ["obj_col"]

    def test_no_categorical_columns_inferred(self):
        df = pd.DataFrame({"x": np.linspace(0, 100, 20)})
        result = infer_nominal_columns(
            df, feature_columns=["x"], explicit="auto", unique_threshold=2
        )
        assert result == []


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
                train_size=0.5,
                stratify=True,
                legacy_two_role=True,
            ),
        )

        dataset = load_dataset(cfg)

        assert dataset.nominal_columns == ["site"]
        assert dataset.ordinal_columns == ["severity"]
        assert dataset.target_is_categorical is True
        assert dataset.all_categorical_columns == ["severity", "site", "target"]
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

    def test_legacy_two_role_loader_does_not_require_patient_hmac_secret(
        self, tmp_path, monkeypatch
    ):
        raw_path = tmp_path / "raw.csv"
        pd.DataFrame({"feature": [10, 11, 12, 13], "target": [0, 1, 0, 1]}).to_csv(
            raw_path, index=False
        )
        schema_path = tmp_path / "schema.csv"
        schema_path.write_text("column,kind\nfeature,continuous\ntarget,categorical\n")
        cfg = Config(
            name="legacy_no_secret",
            data=DataConfig(
                source="csv",
                path=str(raw_path),
                target_column="target",
                variable_schema_path=str(schema_path),
                data_dir=str(tmp_path / "derived"),
                train_size=0.5,
                stratify=True,
                legacy_two_role=True,
            ),
        )
        monkeypatch.delenv("SYNTHDATA_PATIENT_ID_HMAC_KEY", raising=False)

        dataset = load_dataset(cfg)

        assert dataset.legacy_two_role is True
        assert set(dataset.roles) == {"train", "final_holdout"}

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
                train_size=0.5,
                stratify=False,
                legacy_two_role=True,
            ),
        )

        dataset = load_dataset(cfg)

        assert dataset.target_is_categorical is False
        assert dataset.all_categorical_columns == []
        assert pd.api.types.is_float_dtype(dataset.full_df["target"])

    @staticmethod
    def _write_canonical_inputs(tmp_path, *, release_generalization=None):
        tmp_path.mkdir(parents=True, exist_ok=True)
        patients = np.repeat(np.arange(1, 13), 2)
        raw_path = tmp_path / "canonical.csv"
        pd.DataFrame(
            {
                "patient_id": patients,
                "feature": np.arange(24, dtype=float),
                "sensitive": np.tile(["low", "high"], 12),
                "protected": np.repeat(["A", "B"], 12),
                "target": np.tile([0, 1], 12),
            }
        ).to_csv(raw_path, index=False)
        schema_path = tmp_path / "canonical_schema.csv"
        schema_path.write_text(
            "column,kind\nfeature,continuous\nsensitive,categorical\n"
            "protected,categorical\ntarget,categorical\n"
        )
        cfg = Config(
            name="canonical_loader",
            seed=7,
            data=DataConfig(
                source="csv",
                path=str(raw_path),
                target_column="target",
                patient_id_column="patient_id",
                canonical=True,
                sensitive_columns=["sensitive"],
                protected_columns=["protected"],
                variable_schema_path=str(schema_path),
                data_dir=str(tmp_path / "derived"),
                split=DataSplitConfig(
                    mode="patient_group",
                    patient_id_column="patient_id",
                    train_fraction=0.5,
                    tuning_fraction=0.25,
                    final_holdout_fraction=0.25,
                    candidate_count=64,
                ),
            ),
        )
        if release_generalization is not None:
            cfg.evaluation.release_generalization.columns = release_generalization
        return cfg

    def test_canonical_loader_keeps_identity_sidecar_out_of_all_artifacts(self, tmp_path):
        dataset = load_dataset(self._write_canonical_inputs(tmp_path))
        assert dataset.identity_sidecar is not None
        assert dataset.full_df is not None
        assert dataset.assignment is not None
        payload = semantic_context_payload(dataset)
        manifest = json.loads((dataset.data_dir / "dataset_manifest.json").read_text())
        assignment_files = list((dataset.data_dir / "assignments").glob("*/assignment.csv"))

        assert dataset.identity_sidecar.tolist() == list(np.repeat(np.arange(1, 13), 2))
        assert len(dataset.identity_sidecar) == len(dataset.full_df)
        assert "patient_id" not in dataset.full_df.columns
        assert all("patient_id" not in frame.columns for frame in dataset.roles.values())
        assert "patient_id" not in dataset.feature_columns
        for groups in dataset.role_groups.values():
            groups = cast(pd.Series, groups)
            assert set(groups.astype(str)).isdisjoint({str(value) for value in range(1, 13)})
        assert "patient_id" not in str(dataset.assignment.to_dict())
        assert "identity_sidecar" not in manifest
        assert "patient_id" not in json.dumps(payload)
        for path in [
            *assignment_files,
            *(dataset.data_dir / f"{role}.csv" for role in dataset.roles),
        ]:
            assert "patient_id" not in path.read_text().splitlines()[0]
            assert "unit-test-only-patient-id-secret" not in path.read_text()
        assert "unit-test-only-patient-id-secret" not in json.dumps(manifest)
        assert "unit-test-only-patient-id-secret" not in json.dumps(payload)
        assert all(
            not Path(role_path).is_absolute() for role_path in manifest["role_paths"].values()
        )
        assert all(
            (dataset.data_dir / role_path).exists() for role_path in manifest["role_paths"].values()
        )
        assert str(dataset.data_dir) not in json.dumps(manifest)

    def test_canonical_loader_sanitizes_manifest_read_errors(self, tmp_path):
        cfg = self._write_canonical_inputs(tmp_path)
        dataset_dir = tmp_path / "derived" / "data_v_unversioned"
        dataset_dir.mkdir(parents=True)
        sentinel = "manifest-error-sentinel"
        (dataset_dir / "dataset_manifest.json").write_text('{"broken": "' + sentinel + '"')

        with pytest.raises(ValueError, match="Dataset manifest read failed: invalid_json") as error:
            load_dataset(cfg)

        assert sentinel not in str(error.value)
        assert str(dataset_dir) not in str(error.value)

    def test_canonical_loader_bootstraps_and_reuses_identity_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SYNTHDATA_PATIENT_ID_HMAC_KEY", raising=False)
        cfg = self._write_canonical_inputs(tmp_path)
        first = load_dataset(cfg)
        key_path = tmp_path / "derived" / ".patient_id_hmac_key"
        assert key_path.exists()
        assert key_path.stat().st_mode & 0o777 == 0o600
        key = key_path.read_text()
        first_tokens = first.assignment["population_group_hash"].tolist()
        first_manifest = json.loads((first.data_dir / "dataset_manifest.json").read_text())

        second = load_dataset(cfg)
        assert key_path.read_text() == key
        assert second.assignment["population_group_hash"].tolist() == first_tokens
        assert (
            second.role_metadata["identity"]["hmac_key_fingerprint"]
            == first_manifest["hmac_key_fingerprint"]
        )
        assert key not in json.dumps(first_manifest)

    def test_canonical_loader_env_key_overrides_bootstrap(self, tmp_path, monkeypatch):
        key_path = tmp_path / "derived" / ".patient_id_hmac_key"
        key_path.parent.mkdir()
        key_path.write_text("file-secret")
        monkeypatch.setenv("SYNTHDATA_PATIENT_ID_HMAC_KEY", "environment-secret")
        dataset = load_dataset(self._write_canonical_inputs(tmp_path))
        expected = hashlib.sha256(b"environment-secret").hexdigest()
        assert dataset.role_metadata["identity"]["hmac_key_fingerprint"] == expected
        assert key_path.read_text() == "file-secret"

    def test_concurrent_manifest_publication_leaves_valid_manifest_and_no_temps(self, tmp_path):
        dataset = load_dataset(self._write_canonical_inputs(tmp_path))
        cfg = self._write_canonical_inputs(tmp_path)

        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(lambda _: write_dataset_manifest(cfg, dataset), range(2)))

        manifest_path = dataset.data_dir / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert (
            manifest["hmac_key_fingerprint"]
            == dataset.role_metadata["identity"]["hmac_key_fingerprint"]
        )
        assert list(dataset.data_dir.glob(f".{manifest_path.name}.*")) == []

    def test_canonical_loader_rejects_key_lineage_mismatch(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SYNTHDATA_PATIENT_ID_HMAC_KEY", raising=False)
        cfg = self._write_canonical_inputs(tmp_path)
        load_dataset(cfg)
        (tmp_path / "derived" / ".patient_id_hmac_key").write_text("rotated")
        with pytest.raises(ValueError, match="fingerprint does not match"):
            load_dataset(cfg)

    def test_canonical_loader_rejects_different_environment_key_and_preserves_local_key(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("SYNTHDATA_PATIENT_ID_HMAC_KEY", raising=False)
        cfg = self._write_canonical_inputs(tmp_path)
        load_dataset(cfg)
        key_path = tmp_path / "derived" / ".patient_id_hmac_key"
        original_key = key_path.read_text()
        monkeypatch.setenv("SYNTHDATA_PATIENT_ID_HMAC_KEY", "different-environment-secret")

        with pytest.raises(ValueError, match="fingerprint does not match"):
            load_dataset(cfg)
        assert key_path.read_text() == original_key

    def test_canonical_loader_rejects_old_manifest_without_key_fingerprint(self, tmp_path):
        cfg = self._write_canonical_inputs(tmp_path)
        dataset = load_dataset(cfg)
        manifest_path = dataset.data_dir / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.pop("hmac_key_fingerprint")
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(ValueError, match="no patient identity key fingerprint"):
            load_dataset(cfg)

    def test_canonical_loader_rejects_missing_key_with_prior_lineage(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SYNTHDATA_PATIENT_ID_HMAC_KEY", raising=False)
        cfg = self._write_canonical_inputs(tmp_path)
        load_dataset(cfg)
        (tmp_path / "derived" / ".patient_id_hmac_key").unlink()
        with pytest.raises(ValueError, match="refusing to generate a replacement"):
            load_dataset(cfg)

    def test_canonical_loader_propagates_distinct_sensitive_and_protected_roles(self, tmp_path):
        dataset = load_dataset(self._write_canonical_inputs(tmp_path))
        payload = semantic_context_payload(dataset)
        manifest = json.loads((dataset.data_dir / "dataset_manifest.json").read_text())

        assert dataset.sensitive_columns == ["sensitive"]
        assert dataset.protected_columns == ["protected"]
        assert manifest["sensitive_columns"] == ["sensitive"]
        assert manifest["protected_columns"] == ["protected"]
        assert payload["sensitive_columns"] == ["sensitive"]
        assert payload["protected_columns"] == ["protected"]
        assert payload["sensitive_target_types"] == {"sensitive": "categorical"}

    def test_release_generalization_is_copied_and_changes_semantic_context(self, tmp_path):
        mapping = {"feature": {"kind": "bin", "bins": [0, 10]}}
        first = load_dataset(
            self._write_canonical_inputs(tmp_path / "first", release_generalization=mapping)
        )
        second = load_dataset(
            self._write_canonical_inputs(
                tmp_path / "second",
                release_generalization={"feature": {"kind": "bin", "bins": [0, 5]}},
            )
        )

        assert first.release_generalization == mapping
        assert semantic_context_payload(first)["release_generalization"] == mapping
        assert semantic_context_fingerprint(first) != semantic_context_fingerprint(second)

    @pytest.mark.parametrize(
        "mapping", [{"target": {"kind": "bin"}}, {"patient_id": {"kind": "bin"}}]
    )
    def test_release_generalization_rejects_target_and_identity(self, tmp_path, mapping):
        with pytest.raises(ValueError, match="model features only|identity columns"):
            load_dataset(self._write_canonical_inputs(tmp_path, release_generalization=mapping))

    @staticmethod
    def _canonical_config(raw_path, schema_path, data_dir, split):
        return Config(
            name="canonical_loader_test",
            data=DataConfig(
                source="csv",
                path=str(raw_path),
                target_column="target",
                variable_schema_path=str(schema_path),
                data_dir=str(data_dir),
                split=split,
            ),
        )

    def test_loader_rejects_encounter_label_that_would_be_dropped(self, tmp_path):
        raw_path = tmp_path / "raw.csv"
        pd.DataFrame(
            {
                "patient_id": [1, 1, 2, 2, 3, 3],
                "encounter": ["early", "late"] * 3,
                "feature": [1, 2, 3, 4, 5, 6],
                "target": [0, 1, 0, 1, 0, 1],
            }
        ).to_csv(raw_path, index=False)
        schema_path = tmp_path / "schema.csv"
        schema_path.write_text(
            "column,kind\nencounter,categorical\nfeature,continuous\ntarget,categorical\n"
        )
        split = DataSplitConfig(
            mode="patient_group",
            patient_id_column="patient_id",
            encounter_label_column="encounter",
        )
        cfg = self._canonical_config(raw_path, schema_path, tmp_path / "derived", split)
        cfg.data.drop_columns = ["encounter"]

        with pytest.raises(ValueError, match="encounter/drop"):
            load_dataset(cfg)

    def test_loader_accepts_none_drop_columns(self, tmp_path):
        raw_path = tmp_path / "raw.csv"
        pd.DataFrame(
            {
                "feature": [1, 2, 3, 4],
                "target": [0, 1, 0, 1],
            }
        ).to_csv(raw_path, index=False)
        schema_path = tmp_path / "schema.csv"
        schema_path.write_text("column,kind\nfeature,continuous\ntarget,categorical\n")
        cfg = Config(
            name="none_drop_columns",
            data=DataConfig(
                source="csv",
                path=str(raw_path),
                target_column="target",
                variable_schema_path=str(schema_path),
                data_dir=str(tmp_path / "derived"),
                legacy_two_role=True,
            ),
        )
        cfg.data.drop_columns = None

        dataset = load_dataset(cfg)

        assert "feature" in dataset.full_df.columns
        assert "target" in dataset.full_df.columns

    def test_loader_rejects_mapping_patient_key_overlap_with_protected_column(self, tmp_path):
        raw_path = tmp_path / "raw.csv"
        pd.DataFrame(
            {
                "row_id": [10, 11, 12, 13, 14, 15],
                "feature": [1, 2, 3, 4, 5, 6],
                "target": [0, 1, 0, 1, 0, 1],
            }
        ).to_csv(raw_path, index=False)
        mapping_path = tmp_path / "identity.csv"
        pd.DataFrame({"row_id": [10, 11, 12, 13, 14, 15], "patient_id": [1, 1, 2, 2, 3, 3]}).to_csv(
            mapping_path,
            index=False,
        )
        schema_path = tmp_path / "schema.csv"
        schema_path.write_text("column,kind\nfeature,continuous\ntarget,categorical\n")
        split = DataSplitConfig(
            mode="patient_group",
            identity_mapping_path=str(mapping_path),
            mapping_row_key_column="row_id",
            mapping_patient_key_column="patient_id",
        )
        cfg = self._canonical_config(raw_path, schema_path, tmp_path / "derived", split)
        cfg.data.protected_columns = ["patient_id"]

        with pytest.raises(ValueError, match="protected_columns/identity"):
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
