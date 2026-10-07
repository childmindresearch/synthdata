"""Unit tests for the RefiDiff imputation backend and its pipeline dispatch.

Fast tests (binary categorical encode/decode, ``_mean_std``, dispatch logic)
run by default. One ``@pytest.mark.slow`` test exercises a real, tiny,
CPU-only end-to-end ``impute_dataframe`` call with the MLP fallback denoiser
(no mamba-ssm dependency needed) to confirm the full pipeline produces no
NaNs and never overwrites observed values.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from synthdata.config import RefiDiffConfig
from synthdata.imputation import refidiff_backend
from synthdata.imputation.pipeline import _impute_splits
from synthdata.imputation.refidiff_backend import (
    _apply_refiners,
    _checkpoint_dir,
    _checkpoint_identity,
    _decode_bits_to_categorical,
    _encode_categorical_to_bits,
    _fit_categorical_binary_encoders,
    _fit_refiners,
    _mean_std,
    impute_dataframe,
)

pytestmark = pytest.mark.unit


class TestBinaryCategoricalEncoding:
    def test_round_trip_preserves_categories(self):
        series = pd.Series(["a", "b", "c", "a", None, "c"])
        encoders = _fit_categorical_binary_encoders(pd.DataFrame({"col": series}), ["col"])
        encoder = encoders["col"]
        assert encoder["n_bits"] == 2  # ceil(log2(3)) == 2

        bits, missing = _encode_categorical_to_bits(series, encoder)
        assert missing.tolist() == [False, False, False, False, True, False]

        decoded = _decode_bits_to_categorical(bits, encoder, "col")
        # Missing rows had an all-zero placeholder encoding; every non-missing
        # row must round-trip exactly.
        observed = ~missing
        assert (decoded[observed] == series[observed].to_numpy()).all()

    def test_single_category_uses_one_bit(self):
        series = pd.Series(["only", "only", "only"])
        encoders = _fit_categorical_binary_encoders(pd.DataFrame({"col": series}), ["col"])
        assert encoders["col"]["n_bits"] == 1

    def test_out_of_range_decode_is_clipped_and_logged(self, caplog):
        # 3 categories -> 2 bits -> bit pattern "11" (index 3) is out of range.
        encoder = {"idx_to_cat": {0: "a", 1: "b", 2: "c"}, "n_categories": 3, "n_bits": 2}
        bits = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        # synthdata.utils.get_logger() sets propagate=False, so caplog's
        # root-attached handler needs to be added directly to this module's
        # logger to see its records.
        backend_logger = logging.getLogger("synthdata.imputation.refidiff_backend")
        backend_logger.addHandler(caplog.handler)
        try:
            with caplog.at_level("WARNING"):
                decoded = _decode_bits_to_categorical(bits, encoder, "mycol")
        finally:
            backend_logger.removeHandler(caplog.handler)
        assert decoded.tolist() == ["a", "b", "c", "c"]  # index 3 clipped to 2 ("c")
        assert "out-of-range" in caplog.text
        assert "mycol" in caplog.text

    def test_out_of_range_decode_projects_to_nearest_valid_binary_code(self):
        # 3 categories -> 2 bits. Invalid 11 has Hamming distance 1 to 01 and
        # 10; the deterministic tie rule selects the lower category index (1).
        encoder = {"idx_to_cat": {0: "a", 1: "b", 2: "c"}, "n_categories": 3, "n_bits": 2}
        bits = np.array([[1, 1]], dtype=float)
        diagnostics = []

        decoded = _decode_bits_to_categorical(
            bits, encoder, "mycol", policy="nearest_valid", diagnostics=diagnostics
        )

        assert decoded.tolist() == ["b"]
        assert diagnostics == [
            {
                "column": "mycol",
                "n_rows": 1,
                "n_categories": 3,
                "n_invalid": 1,
                "invalid_rate": 1.0,
                "policy": "nearest_valid",
            }
        ]

    def test_out_of_range_decode_can_fail_loudly(self):
        encoder = {"idx_to_cat": {0: "a", 1: "b", 2: "c"}, "n_categories": 3, "n_bits": 2}
        with pytest.raises(ValueError, match="categorical_decode_policy='error'"):
            _decode_bits_to_categorical(
                np.array([[1, 1]], dtype=float), encoder, "mycol", policy="error"
            )


class TestMeanStd:
    def test_uses_observed_entries_only(self):
        data = np.array([[1.0, 100.0], [2.0, 999.0], [3.0, 100.0]])
        # column 1's second row (999.0) is missing -- must not affect mean/std.
        missing_mask = np.array([[False, False], [False, True], [False, False]])
        mean, std = _mean_std(data, missing_mask)
        assert mean[0] == pytest.approx(2.0)
        assert mean[1] == pytest.approx(100.0)
        assert std[1] == pytest.approx(0.0, abs=1e-9) or std[1] == pytest.approx(1.0)

    def test_constant_column_gets_std_one_not_nan(self):
        data = np.array([[5.0], [5.0], [5.0]])
        missing_mask = np.zeros((3, 1), dtype=bool)
        mean, std = _mean_std(data, missing_mask)
        assert mean[0] == pytest.approx(5.0)
        assert std[0] == pytest.approx(1.0)  # guarded against divide-by-zero


class TestPipelineDispatch:
    def test_tabimpute_imputes_train_alone_then_test_with_train_context(
        self, make_config, make_dataset, mocker
    ):
        cfg = make_config()
        dataset = make_dataset()
        mock_impute = mocker.patch(
            "synthdata.imputation.tabimpute_backend.impute_dataframe",
            side_effect=lambda frame, *args, **kwargs: frame.fillna(0),
        )
        train_imputed, test_imputed = _impute_splits(cfg, dataset, device="cpu")
        first, second = (call.args[0] for call in mock_impute.call_args_list)
        assert first.index.equals(dataset.train_df.index)
        assert second.index.equals(dataset.train_df.index.append(dataset.test_df.index))
        assert train_imputed.index.equals(dataset.train_df.index)
        assert test_imputed.index.equals(dataset.test_df.index)

    def test_refidiff_fits_on_train_and_transforms_test(self, make_config, make_dataset, mocker):
        cfg = make_config()
        cfg.imputation.method = "refidiff"
        dataset = make_dataset()
        state = object()
        mock_fit = mocker.patch.object(
            refidiff_backend, "fit", return_value=(state, dataset.train_df.fillna(0))
        )
        mock_transform = mocker.patch.object(
            refidiff_backend, "transform", side_effect=lambda _, frame: frame.fillna(0)
        )
        _impute_splits(cfg, dataset, device="cpu")
        args, kwargs = mock_fit.call_args
        assert args[0].index.equals(dataset.train_df.index)
        assert kwargs["data_dir"] == dataset.data_dir
        assert kwargs["refidiff_cfg"] is cfg.imputation.refidiff
        assert kwargs["seed"] == cfg.seed
        assert mock_transform.call_args.args[0] is state
        assert mock_transform.call_args.args[1].index.equals(dataset.test_df.index)


class TestRefiners:
    def test_replaying_a_fitted_pass_matches_the_fit_pass(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 3))
        X[:, 2] = (X[:, 0] > 0).astype(float)
        missing = rng.random(X.shape) < 0.15
        start = np.where(missing, 0.0, X)
        filled, refiners = _fit_refiners(start, missing, 2, catboost_warmup_iterations=5)
        np.testing.assert_allclose(_apply_refiners(start, missing, refiners), filled)

    def test_columns_complete_in_fit_rows_still_get_a_model_when_requested(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(40, 2))
        no_missing = np.zeros_like(X, dtype=bool)
        _, refiners = _fit_refiners(
            X, no_missing, 2, catboost_warmup_iterations=5, model_columns=np.array([False, True])
        )
        assert [refiner[0] for refiner in refiners] == [1]


class TestCheckpointIdentity:
    def test_checkpoint_identity_changes_with_training_parameters_and_data(self, tmp_path):
        cfg = RefiDiffConfig(epochs=10, catboost_warmup_iterations=100)
        frame = pd.DataFrame({"value": [1.0, np.nan], "category": ["a", "b"]})
        identity = _checkpoint_identity(
            frame,
            ["value", "category"],
            ["category"],
            2,
            cfg,
            "mlp",
            seed=42,
            refinement_columns=None,
        )
        changed_cfg = RefiDiffConfig(epochs=11, catboost_warmup_iterations=100)
        changed_identity = _checkpoint_identity(
            frame,
            ["value", "category"],
            ["category"],
            2,
            changed_cfg,
            "mlp",
            seed=42,
            refinement_columns=None,
        )

        assert identity != changed_identity
        assert _checkpoint_dir(tmp_path, identity) != _checkpoint_dir(tmp_path, changed_identity)


@pytest.mark.slow
class TestRefiDiffEndToEnd:
    def test_impute_dataframe_fills_missing_without_overwriting_observed(self, tmp_path):
        rng = np.random.default_rng(0)
        n = 50
        df = pd.DataFrame(
            {
                "num1": rng.normal(size=n),
                "num2": rng.normal(loc=5, scale=2, size=n),
                "cat1": rng.choice(["a", "b", "c"], size=n),
                "target": rng.integers(0, 2, size=n),
            }
        )
        num1_missing = rng.random(n) < 0.2
        cat1_missing = rng.random(n) < 0.2
        df.loc[num1_missing, "num1"] = np.nan
        df.loc[cat1_missing, "cat1"] = np.nan

        cfg = RefiDiffConfig(
            hidden_dim=8,
            epochs=15,
            early_stopping_patience=15,
            batch_size=32,
            num_steps=4,
            num_trials=2,
            denoiser="mlp",
            checkpoint_every=5,
        )

        result = impute_dataframe(
            df,
            ["num1", "num2", "cat1"],
            ["cat1"],
            "target",
            device="cpu",
            refidiff_cfg=cfg,
            data_dir=tmp_path,
        )

        assert result.isna().sum().sum() == 0
        assert np.allclose(result.loc[~num1_missing, "num1"], df.loc[~num1_missing, "num1"])
        assert (result.loc[~cat1_missing, "cat1"] == df.loc[~cat1_missing, "cat1"]).all()
        assert set(result["cat1"].unique()) <= {"a", "b", "c"}

    def test_handles_string_valued_column_not_declared_categorical(self, tmp_path, caplog):
        """Regression test: a plain CSV can have a string-valued column (e.g. an
        ordinal band stored as text) that isn't listed in categorical_columns
        -- confirmed on the real Loris dataset's BIA__BIA_Activity_Level
        ("Very Light"/"Light"/"Moderate"/"Heavy"/"Exceptional"), which crashed
        with `ValueError: could not convert string to float` before
        refidiff_backend started reusing synthdata.data.label_encode_non_numeric_columns
        as a fallback (mirroring tabimpute_backend's existing behavior).
        """
        rng = np.random.default_rng(0)
        n = 40
        df = pd.DataFrame(
            {
                "num1": rng.normal(size=n),
                "activity_level": rng.choice(
                    ["Very Light", "Light", "Moderate", "Heavy", "Exceptional"], size=n
                ),
                "target": rng.integers(0, 2, size=n),
            }
        )
        missing = rng.random(n) < 0.2
        df.loc[missing, "activity_level"] = np.nan

        cfg = RefiDiffConfig(
            hidden_dim=8,
            epochs=10,
            early_stopping_patience=10,
            batch_size=32,
            num_steps=3,
            num_trials=1,
            denoiser="mlp",
            checkpoint_every=5,
        )

        backend_logger = logging.getLogger("synthdata.imputation.refidiff_backend")
        backend_logger.addHandler(caplog.handler)
        try:
            with caplog.at_level("WARNING"):
                # activity_level deliberately NOT in categorical_columns.
                result = impute_dataframe(
                    df,
                    ["num1", "activity_level"],
                    [],
                    "target",
                    device="cpu",
                    refidiff_cfg=cfg,
                    data_dir=tmp_path,
                )
        finally:
            backend_logger.removeHandler(caplog.handler)

        assert result.isna().sum().sum() == 0
        assert set(result["activity_level"].unique()) <= {
            "Very Light",
            "Light",
            "Moderate",
            "Heavy",
            "Exceptional",
        }
        assert (result.loc[~missing, "activity_level"] == df.loc[~missing, "activity_level"]).all()
        assert "label-encoded as a fallback" in caplog.text
        assert "activity_level" in caplog.text

    def test_fit_is_repeatable_and_transform_fills_held_out_rows(self, tmp_path):
        rng = np.random.default_rng(3)
        n = 60
        df = pd.DataFrame(
            {
                "num1": rng.normal(size=n),
                "num2": rng.normal(size=n),
                "cat1": rng.choice(["a", "b", "c"], size=n),
                "target": rng.integers(0, 2, size=n),
            }
        )
        df.loc[rng.random(n) < 0.2, "num1"] = np.nan
        df.loc[rng.random(n) < 0.2, "cat1"] = np.nan
        train, test = df.iloc[:40], df.iloc[40:].copy()
        cfg = RefiDiffConfig(
            hidden_dim=8,
            epochs=5,
            early_stopping_patience=5,
            batch_size=32,
            num_steps=3,
            num_trials=1,
            denoiser="mlp",
            checkpoint_every=5,
            catboost_warmup_iterations=5,
        )
        kwargs = dict(device="cpu", refidiff_cfg=cfg, data_dir=tmp_path, seed=0)
        features = ["num1", "num2", "cat1"]

        state, train_imputed = refidiff_backend.fit(
            train, features, ["cat1"], "target", model_columns=["num2"], **kwargs
        )
        # A label never seen in train is passed through, not decoded to another.
        test.loc[test.index[0], "cat1"] = "unseen"
        test.loc[test.index[1], "num2"] = np.nan
        test_imputed = refidiff_backend.transform(state, test)
        _, train_again = refidiff_backend.fit(
            train, features, ["cat1"], "target", model_columns=["num2"], **kwargs
        )

        pd.testing.assert_frame_equal(train_imputed, train_again)
        assert not test_imputed[features].isna().any().any()
        assert test_imputed.loc[test.index[0], "cat1"] == "unseen"
        observed = test["num1"].notna()
        assert (test_imputed.loc[observed, "num1"] == test.loc[observed, "num1"]).all()
