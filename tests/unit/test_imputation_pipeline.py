"""Unit tests for synthdata.imputation.pipeline's config-aware caching."""

import json
import logging

import pytest

from synthdata.imputation.pipeline import (
    _CACHE_KEY_FILENAME,
    _cache_key_payload,
    _cache_key_record,
    _load_cached_key,
    run_imputation,
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
