"""Unit tests for synthdata.imputation.pipeline's config-aware caching."""

import json
import logging

import numpy as np
import pandas as pd
import pytest

from synthdata.data import dataframe_fingerprint, load_imputed_splits
from synthdata.imputation import pipeline as imputation_pipeline
from synthdata.imputation.pipeline import (
    _CACHE_KEY_FILENAME,
    _cache_key_payload,
    _cache_key_record,
    _load_cached_key,
    run_imputation,
)
from synthdata.imputation.tabimpute_backend import (
    TabImputeState,
    state_metadata,
    state_metadata_fingerprint,
)

pytestmark = pytest.mark.unit


class TestCacheKeyPayload:
    def test_same_config_and_dataset_hash_identically(self, make_config, make_dataset):
        cfg = make_config()
        dataset = make_dataset()
        assert (
            _cache_key_record(cfg, dataset)["cache_key"]
            == _cache_key_record(cfg, dataset)["cache_key"]
        )

    def test_nominal_columns_change_changes_hash(self, make_config, make_dataset):
        cfg = make_config()
        dataset_a = make_dataset(nominal_columns=["smoker"])
        dataset_b = make_dataset(nominal_columns=["smoker", "group"])
        assert (
            _cache_key_record(cfg, dataset_a)["cache_key"]
            != _cache_key_record(cfg, dataset_b)["cache_key"]
        )

    def test_ordinal_columns_change_changes_hash(self, make_config, make_dataset):
        cfg = make_config()
        dataset_a = make_dataset(ordinal_columns=["smoker"])
        dataset_b = make_dataset(ordinal_columns=["smoker", "group"])
        assert (
            _cache_key_record(cfg, dataset_a)["cache_key"]
            != _cache_key_record(cfg, dataset_b)["cache_key"]
        )

    def test_moving_column_between_nominal_and_ordinal_changes_hash(
        self, make_config, make_dataset
    ):
        """Same combined categorical set, but a different nominal/ordinal split
        must still hash differently -- ordinal columns get order-preserving
        encoding (see refidiff_backend._fit_categorical_binary_encoders), so
        this is a real behavior-affecting distinction, not just a relabeling.
        """
        cfg = make_config()
        dataset_a = make_dataset(nominal_columns=["smoker"], ordinal_columns=["group"])
        dataset_b = make_dataset(nominal_columns=["group"], ordinal_columns=["smoker"])
        assert (
            _cache_key_record(cfg, dataset_a)["cache_key"]
            != _cache_key_record(cfg, dataset_b)["cache_key"]
        )

    def test_method_change_changes_hash(self, make_config, make_dataset):
        dataset = make_dataset()
        cfg_a = make_config()
        cfg_a.imputation.method = "tabimpute"
        cfg_b = make_config()
        cfg_b.imputation.method = "refidiff"
        assert (
            _cache_key_record(cfg_a, dataset)["cache_key"]
            != _cache_key_record(cfg_b, dataset)["cache_key"]
        )

    def test_source_data_change_changes_hash(self, make_config, make_dataset):
        cfg = make_config()
        dataset_a = make_dataset()
        dataset_b = make_dataset()
        dataset_b.full_df = dataset_b.full_df.copy()
        dataset_b.full_df.loc[dataset_b.full_df.index[0], "age"] = 99

        assert (
            _cache_key_record(cfg, dataset_a)["cache_key"]
            != _cache_key_record(cfg, dataset_b)["cache_key"]
        )

    def test_split_membership_change_changes_hash(self, make_config, make_dataset):
        cfg = make_config()
        dataset_a = make_dataset()
        dataset_b = make_dataset()
        dataset_b.train_df = dataset_b.test_df.copy()
        dataset_b.test_df = dataset_a.train_df.copy()

        assert (
            _cache_key_record(cfg, dataset_a)["cache_key"]
            != _cache_key_record(cfg, dataset_b)["cache_key"]
        )

    def test_refidiff_params_included_only_for_refidiff_method(self, make_config, make_dataset):
        cfg = make_config()
        cfg.imputation.method = "tabimpute"
        dataset = make_dataset()
        assert "refidiff" not in _cache_key_payload(cfg, dataset)
        cfg.imputation.method = "refidiff"
        assert "refidiff" in _cache_key_payload(cfg, dataset)

    def test_unrelated_field_does_not_change_hash(self, make_config, make_dataset):
        """validation_margin only affects the post-hoc report, not imputed values."""
        dataset = make_dataset()
        cfg_a = make_config()
        cfg_a.imputation.validation_margin = 0.2
        cfg_b = make_config()
        cfg_b.imputation.validation_margin = 0.9
        assert (
            _cache_key_record(cfg_a, dataset)["cache_key"]
            == _cache_key_record(cfg_b, dataset)["cache_key"]
        )


class TestLoadCachedKey:
    def test_missing_file_returns_none(self, tmp_path):
        assert _load_cached_key(tmp_path / "nope.json") is None

    def test_corrupt_file_returns_none_and_warns(self, tmp_path, caplog):
        path = tmp_path / _CACHE_KEY_FILENAME
        path.write_text("{not valid json")
        pipeline_logger = logging.getLogger("synthdata.imputation.pipeline")
        pipeline_logger.addHandler(caplog.handler)
        try:
            with caplog.at_level("WARNING"):
                result = _load_cached_key(path)
        finally:
            pipeline_logger.removeHandler(caplog.handler)
        assert result is None
        assert "Failed to parse imputation cache-key file" in caplog.text

    def test_valid_file_returns_cache_key(self, tmp_path):
        path = tmp_path / _CACHE_KEY_FILENAME
        path.write_text(json.dumps({"cache_key": "abc123"}))
        assert _load_cached_key(path) == "abc123"


class TestRunImputationCaching:
    def test_canonical_tabimpute_cache_records_train_fit_state(
        self, make_config, make_canonical_dataset
    ):
        cfg = make_config()
        dataset = make_canonical_dataset()

        run_imputation(cfg, dataset)

        record = json.loads((dataset.data_dir / _CACHE_KEY_FILENAME).read_text())
        fit_state = record["fit_state"]
        assert fit_state["backend"] == "tabimpute"
        assert fit_state["fit_role"] == "train"
        assert fit_state["status"] == "not_required"
        assert fit_state["fit_frame_fingerprint"] == dataframe_fingerprint(dataset.roles["train"])
        assert fit_state["state_fingerprint"] == state_metadata_fingerprint(fit_state)

    def test_canonical_cache_without_fit_state_retrains(
        self, make_config, make_canonical_dataset, mocker
    ):
        cfg = make_config()
        dataset = make_canonical_dataset()
        run_imputation(cfg, dataset)

        cache_path = dataset.data_dir / _CACHE_KEY_FILENAME
        record = json.loads(cache_path.read_text())
        del record["fit_state"]
        cache_path.write_text(json.dumps(record))

        rerun_dataset = make_canonical_dataset()
        rerun = mocker.patch(
            "synthdata.imputation.pipeline._impute_canonical_roles",
            wraps=imputation_pipeline._impute_canonical_roles,
        )

        run_imputation(cfg, rerun_dataset)

        rerun.assert_called_once()

    def test_persists_and_reloads_decoded_ordinal_splits(self, make_config, make_dataset, mocker):
        cfg = make_config()
        encoded = pd.DataFrame(
            {
                "activity": [0.0, 1.0, None, 0.0],
                "target": [0, 1, 0, 1],
            }
        )
        dataset = make_dataset(
            df=encoded,
            feature_columns=["activity"],
            ordinal_columns=["activity"],
        )
        dataset.variable_schema = {
            "activity": {
                "kind": "categorical",
                "ordinal_order": ["Low", "High"],
            },
            "target": {"kind": "categorical", "ordinal_order": None},
        }
        mocker.patch(
            "synthdata.imputation.tabimpute_backend.impute_dataframe",
            return_value=encoded.fillna({"activity": 1.0}),
        )

        run_imputation(cfg, dataset)

        assert dataset.full_imputed_decoded_df["activity"].tolist() == [
            "Low",
            "High",
            "High",
            "Low",
        ]
        decoded_path = dataset.paths()["full_imputed_decoded"]
        assert pd.read_csv(decoded_path)["activity"].tolist() == [
            "Low",
            "High",
            "High",
            "Low",
        ]

        reloaded = make_dataset(
            df=encoded,
            feature_columns=["activity"],
            ordinal_columns=["activity"],
            name=dataset.name,
        )
        reloaded.data_dir = dataset.data_dir
        reloaded.variable_schema = dataset.variable_schema
        load_imputed_splits(reloaded)

        assert reloaded.full_imputed_decoded_df["activity"].tolist() == [
            "Low",
            "High",
            "High",
            "Low",
        ]

    def test_first_run_calls_backend_and_writes_cache_key(self, make_config, make_dataset, mocker):
        cfg = make_config()
        dataset = make_dataset()
        mock_impute = mocker.patch(
            "synthdata.imputation.tabimpute_backend.impute_dataframe",
            return_value=dataset.full_df.fillna(0),
        )
        run_imputation(cfg, dataset)
        mock_impute.assert_called_once()
        assert (dataset.data_dir / _CACHE_KEY_FILENAME).exists()

    def test_second_run_with_unchanged_config_reuses_cache(self, make_config, make_dataset, mocker):
        cfg = make_config()
        dataset = make_dataset()
        mock_impute = mocker.patch(
            "synthdata.imputation.tabimpute_backend.impute_dataframe",
            return_value=dataset.full_df.fillna(0),
        )
        run_imputation(cfg, dataset)
        run_imputation(cfg, dataset)
        mock_impute.assert_called_once()  # not called a second time -- cache hit

    def test_nominal_columns_change_forces_retrain(self, make_config, make_dataset, mocker):
        cfg = make_config()
        dataset = make_dataset(nominal_columns=["smoker"])
        mock_impute = mocker.patch(
            "synthdata.imputation.tabimpute_backend.impute_dataframe",
            return_value=dataset.full_df.fillna(0),
        )
        run_imputation(cfg, dataset)

        # Simulate rerunning after editing data.nominal_columns in the config:
        # a fresh Dataset with the resolved column list changed, same data_dir.
        dataset_b = make_dataset(nominal_columns=["smoker", "group"], name=dataset.name)
        dataset_b.data_dir = dataset.data_dir
        run_imputation(cfg, dataset_b)

        assert mock_impute.call_count == 2  # retrained instead of reusing stale cache

    def test_source_data_change_forces_retrain(self, make_config, make_dataset, mocker):
        cfg = make_config()
        dataset = make_dataset()
        mock_impute = mocker.patch(
            "synthdata.imputation.tabimpute_backend.impute_dataframe",
            return_value=dataset.full_df.fillna(0),
        )
        run_imputation(cfg, dataset)

        dataset_b = make_dataset(name=dataset.name)
        dataset_b.data_dir = dataset.data_dir
        dataset_b.full_df = dataset_b.full_df.copy()
        dataset_b.full_df.loc[dataset_b.full_df.index[0], "age"] = 99
        run_imputation(cfg, dataset_b)

        assert mock_impute.call_count == 2


class TestTabImputeStateMetadata:
    @staticmethod
    def _state() -> TabImputeState:
        return TabImputeState(
            feature_columns=("feature", "category"),
            categorical_columns=("category",),
            category_maps={"category": pd.Index(["A", "B"])},
            means=np.array([1.0, 0.5, 0.5]),
            stds=np.array([2.0, 0.5, 0.5]),
            block_slices={"feature": (0, 1), "category": (1, 3)},
            device="cpu",
            imputer=object(),
        )

    def test_state_metadata_is_stable_for_same_train_fit(self):
        first = state_metadata(self._state(), "train-fingerprint")
        second = state_metadata(self._state(), "train-fingerprint")

        assert first == second
        assert first["state_fingerprint"] == state_metadata_fingerprint(first)
        assert "category_map_fingerprints" in first
        assert "scaling_state_fingerprint" in first

    def test_state_metadata_changes_when_train_fit_changes(self):
        first = state_metadata(self._state(), "train-fingerprint-a")
        changed = state_metadata(self._state(), "train-fingerprint-b")

        assert first["state_fingerprint"] != changed["state_fingerprint"]

    def test_canonical_fit_metadata_ignores_non_train_role_values(
        self, make_config, make_canonical_dataset, mocker
    ):
        cfg = make_config()
        first = make_canonical_dataset()
        second = make_canonical_dataset()
        for dataset in (first, second):
            for role in ("tuning", "final_holdout"):
                role_frame = dataset.roles[role].copy()
                role_frame.loc[role_frame.index[0], "feature"] = np.nan
                dataset.roles[role] = role_frame
        second.roles["tuning"].loc[second.roles["tuning"].index[1], "feature"] = 999.0
        second.roles["final_holdout"].loc[second.roles["final_holdout"].index[1], "feature"] = 999.0

        fit_state = TabImputeState(
            feature_columns=("feature", "protected"),
            categorical_columns=("protected",),
            category_maps={"protected": pd.Index(["A", "B"])},
            means=np.array([1.0, 0.5, 0.5]),
            stds=np.array([2.0, 0.5, 0.5]),
            block_slices={"feature": (0, 1), "protected": (1, 3)},
            device="cpu",
            imputer=object(),
        )
        fit = mocker.patch(
            "synthdata.imputation.tabimpute_backend.fit_dataframe", return_value=fit_state
        )

        def transform(*args):
            transformed = args[1].copy()
            transformed["feature"] = transformed["feature"].fillna(0.0)
            return transformed

        mocker.patch(
            "synthdata.imputation.tabimpute_backend.transform_dataframe",
            side_effect=transform,
        )

        _, first_metadata = imputation_pipeline._impute_canonical_roles(cfg, first, "cpu")
        _, second_metadata = imputation_pipeline._impute_canonical_roles(cfg, second, "cpu")

        assert first_metadata == second_metadata
        assert fit.call_args_list[0].args[0].equals(first.roles["train"])
        assert fit.call_args_list[1].args[0].equals(second.roles["train"])

    def test_changed_split_membership_rejects_cached_imputation(
        self, make_config, make_dataset, mocker
    ):
        cfg = make_config()
        dataset = make_dataset()
        mock_impute = mocker.patch(
            "synthdata.imputation.tabimpute_backend.impute_dataframe",
            return_value=dataset.full_df.fillna(0),
        )
        run_imputation(cfg, dataset)

        changed = make_dataset(name=dataset.name)
        changed.data_dir = dataset.data_dir
        changed.train_df = changed.test_df.copy()
        changed.test_df = dataset.train_df.copy()
        load_imputed_splits(changed)

        assert changed.train_imputed_df is None
        assert changed.test_imputed_df is None
        mock_impute.assert_called_once()

    def test_expected_cache_key_rejects_stale_imputed_frames(
        self, make_config, make_dataset, mocker
    ):
        cfg = make_config()
        dataset = make_dataset()
        mock_impute = mocker.patch(
            "synthdata.imputation.tabimpute_backend.impute_dataframe",
            return_value=dataset.full_df.fillna(0),
        )
        run_imputation(cfg, dataset)

        stale = make_dataset(name=dataset.name)
        stale.data_dir = dataset.data_dir
        load_imputed_splits(stale, expected_cache_key="different-cache-key")

        assert stale.full_imputed_df is None
        mock_impute.assert_called_once()

    def test_cache_disabled_always_retrains(self, make_config, make_dataset, mocker):
        cfg = make_config()
        cfg.imputation.cache = False
        dataset = make_dataset()
        mock_impute = mocker.patch(
            "synthdata.imputation.tabimpute_backend.impute_dataframe",
            return_value=dataset.full_df.fillna(0),
        )
        run_imputation(cfg, dataset)
        run_imputation(cfg, dataset)
        assert mock_impute.call_count == 2

    def test_corrupt_cache_key_file_forces_retrain(self, make_config, make_dataset, mocker):
        cfg = make_config()
        dataset = make_dataset()
        mock_impute = mocker.patch(
            "synthdata.imputation.tabimpute_backend.impute_dataframe",
            return_value=dataset.full_df.fillna(0),
        )
        run_imputation(cfg, dataset)
        (dataset.data_dir / _CACHE_KEY_FILENAME).write_text("{not valid json")
        run_imputation(cfg, dataset)
        assert mock_impute.call_count == 2
