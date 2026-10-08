"""Unit tests for the scikit-learn imputers, the two-phase fit and missing indicators."""

import numpy as np
import pandas as pd
import pytest

from synthdata.data import add_missing_indicators, remask_synthetic
from synthdata.imputation import sklearn_backend
from synthdata.imputation.pipeline import _impute_splits, imputation_drift

pytestmark = pytest.mark.unit


def _frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "x": x,
            "y": 3 * x + rng.normal(scale=0.1, size=n),
            "colour": rng.choice(["red", "green", "blue"], size=n),
            "flag": (x > 0).astype(int),
            "target": rng.integers(0, 2, size=n),
        }
    )
    return frame


FEATURES = ["x", "y", "colour", "flag"]


def _blank(frame, column, rows):
    out = frame.copy()
    out.loc[rows, column] = np.nan
    return out


class TestSimple:
    def test_fills_with_fit_rows_median_and_mode_only(self):
        fit_df = _blank(_frame(), "y", [0, 1, 2])
        fit_df = _blank(fit_df, "colour", [3])
        state, fit_imputed = sklearn_backend.fit(
            fit_df, FEATURES, ["colour", "flag"], ["colour", "flag"], "simple", seed=0
        )
        assert fit_imputed.loc[0, "y"] == pytest.approx(fit_df["y"].median())
        assert fit_imputed.loc[3, "colour"] == fit_df["colour"].mode()[0]

        # Rows passed to transform never move the fitted median.
        other = _blank(_frame(seed=1), "y", [0]).assign(x=lambda d: d["x"] + 100)
        other["y"] = other["y"] + 1000
        other.loc[0, "y"] = np.nan
        assert sklearn_backend.transform(state, other).loc[0, "y"] == pytest.approx(
            fit_df["y"].median()
        )

    def test_observed_values_and_target_are_untouched(self):
        fit_df = _blank(_frame(), "x", [5, 6])
        _, imputed = sklearn_backend.fit(fit_df, FEATURES, ["colour", "flag"], [], "simple", 0)
        observed = fit_df["x"].notna()
        pd.testing.assert_series_equal(imputed.loc[observed, "x"], fit_df.loc[observed, "x"])
        pd.testing.assert_series_equal(imputed["target"], fit_df["target"])


class TestMissForest:
    def test_uses_other_columns_and_returns_valid_categories(self):
        frame = _frame(300)
        rows = list(range(0, 300, 10))
        blanked = _blank(_blank(_blank(frame, "y", rows), "colour", rows[:5]), "flag", rows[5:])
        state, imputed = sklearn_backend.fit(
            blanked, FEATURES, ["colour", "flag"], ["colour", "flag"], "missforest", seed=0
        )
        # y is almost 3x, so forests predict it far better than its median does.
        forest_error = (imputed.loc[rows, "y"] - frame.loc[rows, "y"]).abs().mean()
        median_error = (frame["y"].median() - frame.loc[rows, "y"]).abs().mean()
        assert forest_error < median_error / 3
        assert set(imputed["colour"]) <= {"red", "green", "blue"}
        assert set(imputed["flag"]) <= {0, 1}
        assert state.one_hot == ["colour"]  # three categories; binary flag stays a code

    def test_is_deterministic_for_a_seed(self):
        blanked = _blank(_frame(), "y", list(range(0, 200, 7)))
        args = (blanked, FEATURES, ["colour", "flag"], ["colour", "flag"], "missforest")
        pd.testing.assert_frame_equal(
            sklearn_backend.fit(*args, seed=3)[1], sklearn_backend.fit(*args, seed=3)[1]
        )


class TestTwoPhases:
    @pytest.fixture
    def dataset(self, make_dataset):
        frame = _blank(_frame(100), "y", list(range(0, 100, 5)))
        dataset = make_dataset(
            df=frame, feature_columns=FEATURES, nominal_columns=["colour", "flag"]
        )
        dataset.tuning_index = dataset.train_df.index[:20]
        return dataset

    def test_hpo_fit_never_sees_tuning_rows_and_final_fit_sees_all_train(
        self, make_config, dataset
    ):
        cfg = make_config()
        cfg.imputation.method = "simple"
        cfg.generation.hpo.enabled = True
        # Make tuning rows extreme so a median that saw them would move.
        dataset.train_df.loc[dataset.tuning_index, "y"] = 1e6
        train_imputed, test_imputed, search_imputed = _impute_splits(cfg, dataset, "cpu")

        missing = dataset.search_train_df["y"].isna()
        rows = dataset.search_train_df.index[missing]
        assert (search_imputed.loc[rows, "y"] == dataset.search_train_df["y"].median()).all()
        assert (train_imputed.loc[rows, "y"] == dataset.train_df["y"].median()).all()
        assert test_imputed.index.equals(dataset.test_df.index)

        drift = imputation_drift(dataset, search_imputed, train_imputed)
        assert drift.set_index("column").loc["y", "drift"] > 0

    def test_no_hpo_fit_when_hpo_is_off(self, make_config, dataset):
        cfg = make_config()
        cfg.imputation.method = "simple"
        cfg.generation.hpo.enabled = False
        assert _impute_splits(cfg, dataset, "cpu")[2] is None

    def test_drift_is_zero_when_both_fits_agree(self, dataset):
        filled = dataset.train_df.fillna(0)
        drift = imputation_drift(dataset, filled, filled.copy())
        assert (drift["drift"] == 0).all()


class TestMissingIndicators:
    def test_indicators_follow_raw_missingness_above_the_threshold(self):
        frame = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": [1.0, 2.0, 3.0, np.nan]})
        # Threshold judged on rows 0-2 only: a is 1/3 missing there, b never.
        widened, indicators = add_missing_indicators(frame, ["a", "b"], frame.index[:3], 0.3)
        assert indicators == {"a__missing": "a"}
        assert widened["a__missing"].tolist() == [0, 1, 0, 0]

    def test_remask_blanks_flagged_synthetic_values_and_drops_indicators(self):
        synthetic = pd.DataFrame({"a": [1.0, 2.0, 3.0], "a__missing": [0, 1, 0]})
        released = remask_synthetic(synthetic, {"a__missing": "a"})
        assert released.columns.tolist() == ["a"]
        assert released["a"].isna().tolist() == [False, True, False]

    def test_name_clash_is_rejected(self):
        frame = pd.DataFrame({"a": [np.nan, 1.0], "a__missing": [0, 0]})
        with pytest.raises(ValueError, match="already exist"):
            add_missing_indicators(frame, ["a"], frame.index, 0.1)
