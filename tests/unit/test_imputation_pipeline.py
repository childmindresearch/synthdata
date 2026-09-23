"""Unit tests for synthdata.imputation.pipeline's config-aware caching."""

import json
import logging
from typing import cast

import numpy as np
import pandas as pd
import pytest

from synthdata.data import dataframe_fingerprint, load_imputed_splits
from synthdata.imputation import pipeline as imputation_pipeline
from synthdata.imputation.hyperimpute_backend import (
    FIT_FRAME_FINGERPRINT_VERSION,
    HyperImputeError,
    HyperImputeState,
    fit_dataframe,
    metadata_fingerprint,
    transform_dataframe,
)
from synthdata.imputation.hyperimpute_backend import (
    state_metadata as hyper_state_metadata,
)
from synthdata.imputation.pipeline import (
    _CACHE_KEY_FILENAME,
    _cache_key_payload,
    _cache_key_record,
    _load_cached_key,
    build_validation_report,
    run_imputation,
)
from synthdata.imputation.tabimpute_backend import (
    TabImputeState,
    _Imputer,
    state_metadata,
    state_metadata_fingerprint,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("plugin", ["median", "mean"])
def test_fixed_hyperimpute_plugins_fill_values_and_preserve_identity(plugin):
    frame = pd.DataFrame(
        {
            "value": [1.0, 2.0, 100.0, None],
            "category": ["a", "b", "b", None],
            "patient_id": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        }
    )
    state = fit_dataframe(
        frame, ["value", "category"], ["category"], fit_roles=("train",), continuous_plugin=plugin
    )
    result = transform_dataframe(state, frame)
    assert result.loc[3, "value"] == (2.0 if plugin == "median" else 34.333333333333336)
    assert result.loc[3, "category"] == "b"
    assert result["patient_id"].equals(frame["patient_id"])
    assert result["target"].equals(frame["target"])


def test_fixed_hyperimpute_rejects_all_missing_training_feature():
    frame = pd.DataFrame({"value": [None, None], "target": [0, 1]})
    with pytest.raises(HyperImputeError, match="no observed training value"):
        fit_dataframe(frame, ["value"], [], fit_roles=("train",))


def test_fitted_hyperimpute_state_uses_canonical_frame_fingerprint():
    frame = pd.DataFrame({"value": [1.0, None], "target": [0, 1]})
    state = fit_dataframe(frame, ["value"], [], fit_roles=("train",))

    assert state.fit_fingerprint == dataframe_fingerprint(frame)
    metadata = hyper_state_metadata(state)
    assert metadata["fit_frame_fingerprint_version"] == FIT_FRAME_FINGERPRINT_VERSION


def test_final_phase_fits_train_and_tuning_and_transforms_holdout_only(
    make_config, make_canonical_dataset, mocker
):
    cfg = make_config()
    cfg.imputation.method = "hyperimpute"
    dataset = make_canonical_dataset()
    dataset.roles["final_holdout"].loc[dataset.roles["final_holdout"].index[0], "feature"] = np.nan
    state = HyperImputeState(
        ("feature", "protected"),
        ("protected",),
        "median",
        "most_frequent",
        object(),
        object(),
        ("train", "tuning"),
        "fit",
    )
    fit = mocker.patch("synthdata.imputation.hyperimpute_backend.fit_dataframe", return_value=state)
    transform = mocker.patch(
        "synthdata.imputation.hyperimpute_backend.transform_dataframe",
        side_effect=lambda _, frame: frame.copy(),
    )
    imputation_pipeline._impute_canonical_roles(cfg, dataset, "cpu", phase="final")
    pd.testing.assert_frame_equal(
        fit.call_args.args[0], pd.concat([dataset.roles["train"], dataset.roles["tuning"]])
    )
    assert transform.call_count == 1
    pd.testing.assert_frame_equal(transform.call_args.args[1], dataset.roles["final_holdout"])


def test_final_phase_noop_metadata_uses_concat_fit_fingerprint(make_config, make_canonical_dataset):
    cfg = make_config()
    cfg.imputation.method = "hyperimpute"
    dataset = make_canonical_dataset()
    _, metadata = imputation_pipeline._impute_canonical_roles(cfg, dataset, "cpu", phase="final")
    fit_frame = pd.concat([dataset.roles["train"], dataset.roles["tuning"]])
    assert metadata is not None
    assert metadata["fit_roles"] == ["train", "tuning"]
    assert metadata["transform_roles"] == ["final_holdout"]
    assert metadata["fit_frame_fingerprint"] == dataframe_fingerprint(fit_frame)
    from synthdata.imputation.hyperimpute_backend import metadata_fingerprint

    assert metadata["state_fingerprint"] == metadata_fingerprint(metadata)


def test_candidate_validation_report_uses_only_train_and_tuning(
    make_config, make_canonical_dataset
):
    cfg = make_config()
    dataset = make_canonical_dataset()
    raw_roles = {role: frame.copy() for role, frame in dataset.roles.items()}
    imputed_roles = {role: frame.copy() for role, frame in dataset.roles.items()}

    raw_roles["train"].iloc[0, raw_roles["train"].columns.get_loc("feature")] = np.nan
    imputed_roles["train"].iloc[0, imputed_roles["train"].columns.get_loc("feature")] = 1000.0
    raw_roles["tuning"].iloc[0, raw_roles["tuning"].columns.get_loc("protected")] = np.nan
    imputed_roles["tuning"].iloc[0, imputed_roles["tuning"].columns.get_loc("protected")] = "A"
    raw_roles["final_holdout"].iloc[0, raw_roles["final_holdout"].columns.get_loc("feature")] = (
        np.nan
    )
    imputed_roles["final_holdout"].iloc[
        0, imputed_roles["final_holdout"].columns.get_loc("feature")
    ] = 2000.0
    dataset.roles = raw_roles
    dataset.set_imputed_roles(imputed_roles)

    report = build_validation_report(cfg, dataset)

    assert len(report) == 2
    assert report["column"].is_unique
    assert report["column"].tolist() == ["feature", "protected"]
    report_by_column = report.set_index("column")
    assert report_by_column["n_missing"].to_dict() == {"feature": 1, "protected": 1}
    assert report_by_column["n_imputed"].to_dict() == {"feature": 1, "protected": 1}
    assert not report.set_index("column").loc["feature", "all_valid"]
    assert report.set_index("column").loc["protected", "all_valid"]
    assert "target" not in report["column"].tolist()


def test_validation_report_uses_schema_datatypes_and_type_specific_statistics(
    make_config, make_canonical_dataset
):
    cfg = make_config()
    dataset = make_canonical_dataset()
    raw_roles = {role: frame.copy() for role, frame in dataset.roles.items()}
    imputed_roles = {role: frame.copy() for role, frame in dataset.roles.items()}
    raw_roles["train"].loc[raw_roles["train"].index[0], "feature"] = np.nan
    imputed_roles["train"].loc[imputed_roles["train"].index[0], "feature"] = 5.0
    raw_roles["tuning"].loc[raw_roles["tuning"].index[0], "protected"] = np.nan
    imputed_roles["tuning"].loc[imputed_roles["tuning"].index[0], "protected"] = "A"
    dataset.roles = raw_roles
    dataset.set_imputed_roles(imputed_roles)

    report = build_validation_report(cfg, dataset).set_index("column")

    assert report.loc["feature", "datatype"] == "continuous"
    feature_observed = pd.concat(
        [raw_roles["train"]["feature"], raw_roles["tuning"]["feature"]]
    ).dropna()
    feature_imputed = pd.concat(
        [
            imputed_roles["train"].loc[raw_roles["train"]["feature"].isna(), "feature"],
            imputed_roles["tuning"].loc[raw_roles["tuning"]["feature"].isna(), "feature"],
        ]
    ).dropna()
    assert report.loc["feature", "obs_cardinality"] == feature_observed.nunique()
    assert report.loc["feature", "imp_cardinality"] == feature_imputed.nunique()
    assert report.loc["feature", "obs_mean"] == pytest.approx(feature_observed.mean())
    assert pd.isna(report.loc["feature", "obs_mode"])
    assert report.loc["protected", "datatype"] == "categorical"
    assert report.loc["protected", "obs_cardinality"] == 2
    assert report.loc["protected", "imp_cardinality"] == 1
    protected_observed = pd.concat(
        [raw_roles["train"]["protected"], raw_roles["tuning"]["protected"]]
    ).dropna()
    assert report.loc["protected", "obs_mode"] == protected_observed.mode().iloc[0]
    assert report.loc["protected", "imp_mode"] == "A"
    assert pd.isna(report.loc["protected", "obs_mean"])
    assert pd.isna(report.loc["protected", "imp_std"])


def test_empty_validation_report_has_complete_schema(make_config, make_canonical_dataset):
    cfg = make_config()
    dataset = make_canonical_dataset()

    report = build_validation_report(cfg, dataset)

    assert report.empty
    assert report.columns.tolist() == [
        "column",
        "datatype",
        "categorical",
        "n_missing",
        "obs_cardinality",
        "imp_cardinality",
        "obs_mean",
        "obs_std",
        "imp_mean",
        "imp_std",
        "obs_mode",
        "imp_mode",
        "n_imputed",
        "n_valid",
        "all_valid",
    ]


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

    def test_categorical_order_follows_feature_order(self, make_config, make_dataset):
        cfg = make_config()
        dataset = make_dataset(
            feature_columns=["group", "smoker", "age"],
            nominal_columns=["smoker", "group"],
        )

        assert dataset.categorical_columns == ["group", "smoker"]
        assert _cache_key_payload(cfg, dataset)["categorical_columns"] == ["group", "smoker"]

    def test_categorical_order_change_changes_hash(self, make_config, make_dataset):
        cfg = make_config()
        first = make_dataset(
            feature_columns=["group", "smoker", "age"],
            nominal_columns=["group", "smoker"],
            name="first-order",
        )
        second = make_dataset(
            feature_columns=["smoker", "group", "age"],
            nominal_columns=["group", "smoker"],
            name="second-order",
        )

        assert (
            _cache_key_record(cfg, first)["cache_key"]
            != _cache_key_record(cfg, second)["cache_key"]
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

    def test_phase_and_plugin_change_cache_key(self, make_config, make_canonical_dataset):
        cfg = make_config()
        cfg.imputation.method = "hyperimpute"
        dataset = make_canonical_dataset()
        candidate = _cache_key_record(cfg, dataset, phase="candidate")
        final = _cache_key_record(cfg, dataset, phase="final")
        cfg.imputation.continuous_plugin = "mean"
        mean = _cache_key_record(cfg, dataset, phase="candidate")
        assert candidate["cache_key"] != final["cache_key"]
        assert candidate["cache_key"] != mean["cache_key"]
        assert final["fit_roles"] == ["train", "tuning"]
        assert final["transform_roles"] == ["final_holdout"]

    @pytest.mark.parametrize("builder", [_cache_key_payload, _cache_key_record])
    def test_cache_key_builders_reject_invalid_phase(self, make_config, make_dataset, builder):
        with pytest.raises(ValueError, match="expected 'candidate' or 'final'"):
            builder(make_config(), make_dataset(), phase="invalid")


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
    def test_invalid_phase_rejected_before_writing_artifacts(self, make_config, make_dataset):
        dataset = make_dataset()

        with pytest.raises(ValueError, match="expected 'candidate' or 'final'"):
            run_imputation(make_config(), dataset, phase="invalid")

        assert list(dataset.data_dir.iterdir()) == []

    def test_canonical_candidate_cache_reload_matches_generation_key(
        self, make_config, make_canonical_dataset
    ):
        cfg = make_config()
        cfg.imputation.method = "hyperimpute"
        dataset = make_canonical_dataset()

        run_imputation(cfg, dataset)
        record = json.loads((dataset.data_dir / _CACHE_KEY_FILENAME).read_text())
        fresh = make_canonical_dataset()
        fresh.data_dir = dataset.data_dir

        assert record["cache_key"] == _cache_key_record(cfg, fresh)["cache_key"]
        load_imputed_splits(fresh, expected_cache_key=record["cache_key"])
        assert fresh.full_imputed_df is not None

    def test_canonical_candidate_cache_records_phase_aware_metadata(
        self, make_config, make_canonical_dataset
    ):
        cfg = make_config()
        cfg.imputation.method = "hyperimpute"
        dataset = make_canonical_dataset()

        run_imputation(cfg, dataset)
        record = json.loads((dataset.data_dir / _CACHE_KEY_FILENAME).read_text())

        assert record["phase"] == "candidate"
        assert record["fit_roles"] == ["train"]
        assert record["transform_roles"] == ["train", "tuning"]
        assert record["fit_frame_fingerprint"] == dataframe_fingerprint(dataset.roles["train"])
        assert "fit_role" not in record
        assert "fit_role_fingerprint" not in record

    def test_final_phase_preserves_candidate_cache(self, make_config, make_canonical_dataset):
        cfg = make_config()
        cfg.imputation.method = "hyperimpute"
        dataset = make_canonical_dataset()

        run_imputation(cfg, dataset)
        candidate_record = json.loads((dataset.data_dir / _CACHE_KEY_FILENAME).read_text())
        run_imputation(cfg, dataset, phase="final")

        assert json.loads((dataset.data_dir / _CACHE_KEY_FILENAME).read_text()) == candidate_record
        final_record = json.loads(
            (dataset.data_dir / "imputation_final" / _CACHE_KEY_FILENAME).read_text()
        )
        assert final_record["phase"] == "final"

    def test_canonical_final_cache_writes_and_reloads_phase_metadata(
        self, make_config, make_canonical_dataset
    ):
        cfg = make_config()
        cfg.imputation.method = "hyperimpute"
        dataset = make_canonical_dataset()

        run_imputation(cfg, dataset, phase="final")
        record = json.loads(
            (dataset.data_dir / "imputation_final" / _CACHE_KEY_FILENAME).read_text()
        )
        fresh = make_canonical_dataset()
        fresh.data_dir = dataset.data_dir

        load_imputed_splits(fresh, expected_cache_key=record["cache_key"], phase="final")
        fit_frame = pd.concat([fresh.roles["train"], fresh.roles["tuning"]], axis=0)
        assert record["phase"] == "final"
        assert record["fit_roles"] == ["train", "tuning"]
        assert record["transform_roles"] == ["final_holdout"]
        assert record["fit_frame_fingerprint"] == dataframe_fingerprint(fit_frame)
        assert fresh.full_imputed_df is not None

    @pytest.mark.parametrize(
        "change",
        [
            lambda record: record.update(phase="final"),
            lambda record: record.update(fit_frame_fingerprint="changed"),
            lambda record: record.pop("phase"),
            lambda record: record.pop("fit_roles"),
            lambda record: record.pop("transform_roles"),
            lambda record: record.pop("fit_frame_fingerprint"),
        ],
        ids=[
            "phase-mismatch",
            "fit-frame-mismatch",
            "missing-phase",
            "missing-fit-roles",
            "missing-transform-roles",
            "missing-fit-frame",
        ],
    )
    def test_canonical_candidate_loader_rejects_invalid_phase_metadata(
        self, make_config, make_canonical_dataset, change
    ):
        cfg = make_config()
        cfg.imputation.method = "hyperimpute"
        dataset = make_canonical_dataset()
        run_imputation(cfg, dataset)
        cache_path = dataset.data_dir / _CACHE_KEY_FILENAME
        record = json.loads(cache_path.read_text())
        change(record)
        cache_path.write_text(json.dumps(record))

        fresh = make_canonical_dataset()
        fresh.data_dir = dataset.data_dir
        fresh.full_imputed_df = None
        fresh.imputed_roles = {}
        load_imputed_splits(fresh)

        assert fresh.full_imputed_df is None

    def test_canonical_hyperimpute_cache_records_train_fit_state(
        self, make_config, make_canonical_dataset
    ):
        cfg = make_config()
        cfg.imputation.method = "hyperimpute"
        dataset = make_canonical_dataset()

        run_imputation(cfg, dataset)

        record = json.loads((dataset.data_dir / _CACHE_KEY_FILENAME).read_text())
        fit_state = record["fit_state"]
        assert fit_state["backend"] == "hyperimpute"
        assert fit_state["fit_roles"] == ["train"]
        assert fit_state["status"] == "not_required"
        assert fit_state["fit_frame_fingerprint"] == dataframe_fingerprint(dataset.roles["train"])
        assert fit_state["transform_roles"] == ["train", "tuning"]
        assert fit_state["categorical_columns"] == dataset.categorical_columns
        assert record["categorical_columns"] == dataset.categorical_columns

    def test_reordered_hyperimpute_state_retrains(
        self, make_config, make_canonical_dataset, mocker
    ):
        cfg = make_config()
        cfg.imputation.method = "hyperimpute"
        dataset = make_canonical_dataset()
        run_imputation(cfg, dataset)

        cache_path = dataset.data_dir / _CACHE_KEY_FILENAME
        record = json.loads(cache_path.read_text())
        record["fit_state"]["categorical_columns"] = ["feature", "protected"]
        record["fit_state"]["state_fingerprint"] = metadata_fingerprint(record["fit_state"])
        cache_path.write_text(json.dumps(record))

        rerun = mocker.patch(
            "synthdata.imputation.pipeline._impute_canonical_roles",
            wraps=imputation_pipeline._impute_canonical_roles,
        )
        run_imputation(cfg, make_canonical_dataset())

        rerun.assert_called_once()

    def test_canonical_cache_without_fit_state_retrains(
        self, make_config, make_canonical_dataset, mocker
    ):
        cfg = make_config()
        cfg.imputation.method = "hyperimpute"
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

    @pytest.mark.parametrize("version", [None, "dataframe_fingerprint_v0"])
    def test_canonical_cache_with_stale_fingerprint_version_retrains(
        self, make_config, make_canonical_dataset, mocker, version
    ):
        cfg = make_config()
        cfg.imputation.method = "hyperimpute"
        dataset = make_canonical_dataset()
        run_imputation(cfg, dataset)

        cache_path = dataset.data_dir / _CACHE_KEY_FILENAME
        record = json.loads(cache_path.read_text())
        if version is None:
            record["fit_state"].pop("fit_frame_fingerprint_version")
        else:
            record["fit_state"]["fit_frame_fingerprint_version"] = version
        record["fit_state"]["state_fingerprint"] = metadata_fingerprint(record["fit_state"])
        cache_path.write_text(json.dumps(record))

        rerun_dataset = make_canonical_dataset()
        rerun = mocker.patch(
            "synthdata.imputation.pipeline._impute_canonical_roles",
            wraps=imputation_pipeline._impute_canonical_roles,
        )

        run_imputation(cfg, rerun_dataset)

        rerun.assert_called_once()

    def test_canonical_cache_with_current_fingerprint_version_is_reused(
        self, make_config, make_canonical_dataset, mocker
    ):
        cfg = make_config()
        cfg.imputation.method = "hyperimpute"
        dataset = make_canonical_dataset()
        run_imputation(cfg, dataset)

        rerun_dataset = make_canonical_dataset()
        rerun = mocker.patch(
            "synthdata.imputation.pipeline._impute_canonical_roles",
            wraps=imputation_pipeline._impute_canonical_roles,
        )

        run_imputation(cfg, rerun_dataset)

        rerun.assert_not_called()

    def test_persists_and_reloads_decoded_ordinal_splits(self, make_config, make_dataset, mocker):
        cfg = make_config()
        cfg.imputation.method = "tabimpute"
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
        cfg.imputation.method = "tabimpute"
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
        cfg.imputation.method = "tabimpute"
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
        cfg.imputation.method = "tabimpute"
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
        cfg.imputation.method = "tabimpute"
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
            imputer=cast(_Imputer, object()),
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
        cfg.imputation.method = "hyperimpute"
        first = make_canonical_dataset()
        second = make_canonical_dataset()
        for dataset in (first, second):
            for role in ("tuning", "final_holdout"):
                role_frame = dataset.roles[role].copy()
                role_frame.loc[role_frame.index[0], "feature"] = np.nan
                dataset.roles[role] = role_frame
        second.roles["tuning"].loc[second.roles["tuning"].index[1], "feature"] = 999.0
        second.roles["final_holdout"].loc[second.roles["final_holdout"].index[1], "feature"] = 999.0

        fit_state = HyperImputeState(
            ("feature", "protected"),
            ("protected",),
            "median",
            "most_frequent",
            object(),
            object(),
            ("train",),
            "train-fingerprint",
        )
        fit = mocker.patch(
            "synthdata.imputation.hyperimpute_backend.fit_dataframe", return_value=fit_state
        )

        def transform(*args):
            transformed = args[1].copy()
            transformed["feature"] = transformed["feature"].fillna(0.0)
            return transformed

        transform_mock = mocker.patch(
            "synthdata.imputation.hyperimpute_backend.transform_dataframe", side_effect=transform
        )

        _, first_metadata = imputation_pipeline._impute_canonical_roles(cfg, first, "cpu")
        _, second_metadata = imputation_pipeline._impute_canonical_roles(cfg, second, "cpu")

        assert first_metadata == second_metadata
        assert fit.call_args_list[0].args[0].equals(first.roles["train"])
        assert fit.call_args_list[1].args[0].equals(second.roles["train"])
        for call, expected in zip(
            transform_mock.call_args_list,
            [first.roles[role] for role in ("train", "tuning")]
            + [second.roles[role] for role in ("train", "tuning")],
            strict=True,
        ):
            pd.testing.assert_frame_equal(call.args[1], expected)

    @pytest.mark.parametrize("method", ["tabimpute", "refidiff"])
    def test_canonical_legacy_methods_are_deferred(
        self, make_config, make_canonical_dataset, method
    ):
        cfg = make_config()
        cfg.imputation.method = method
        with pytest.raises(imputation_pipeline.RoleIsolationError, match="deferred"):
            run_imputation(cfg, make_canonical_dataset())

    @pytest.mark.parametrize("method", ["tabimpute", "refidiff"])
    def test_disabled_canonical_legacy_methods_are_deferred_before_cache(
        self, make_config, make_canonical_dataset, method
    ):
        cfg = make_config()
        cfg.imputation.method = method
        cfg.imputation.enabled = False
        dataset = make_canonical_dataset()
        dataset.full_imputed_df = None
        with pytest.raises(imputation_pipeline.RoleIsolationError, match="deferred"):
            run_imputation(cfg, dataset)
        assert dataset.full_imputed_df is None

    def test_changed_split_membership_rejects_cached_imputation(
        self, make_config, make_dataset, mocker
    ):
        cfg = make_config()
        cfg.imputation.method = "tabimpute"
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
        cfg.imputation.method = "tabimpute"
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
        cfg.imputation.method = "tabimpute"
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
        cfg.imputation.method = "tabimpute"
        dataset = make_dataset()
        mock_impute = mocker.patch(
            "synthdata.imputation.tabimpute_backend.impute_dataframe",
            return_value=dataset.full_df.fillna(0),
        )
        run_imputation(cfg, dataset)
        (dataset.data_dir / _CACHE_KEY_FILENAME).write_text("{not valid json")
        run_imputation(cfg, dataset)
        assert mock_impute.call_count == 2
