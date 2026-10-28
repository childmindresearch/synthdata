"""Unit tests for the miceforest (MICE + predictive mean matching) imputer."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("miceforest")

from synthdata.imputation import miceforest_backend, sklearn_backend  # noqa: E402

pytestmark = pytest.mark.unit

FEATURES = ["x", "y", "colour", "flag", "grade", "const"]
CATEGORICAL = ["colour", "flag", "grade", "const"]
NOMINAL = ["colour", "flag", "const"]


def _frame(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    # A two-humped ordinal code that the other columns barely predict: averaging
    # fills the empty middle, mean matching keeps the humps.
    grade = np.where(rng.random(n) < 0.5, rng.choice([0, 1], n), rng.choice([4, 5], n))
    return pd.DataFrame(
        {
            "x": x,
            "y": 3 * x + rng.normal(scale=0.1, size=n),
            "colour": rng.choice(["red", "green", "blue"], size=n, p=[0.6, 0.3, 0.1]),
            "flag": (rng.random(n) < 0.6).astype(int),
            "grade": grade.astype(float),
            "const": "a",
            "target": rng.integers(0, 2, size=n),
        }
    )


def _blanked(frame, rows, columns=("y", "colour", "flag", "grade", "const")):
    out = frame.copy()
    out.loc[rows, list(columns)] = np.nan
    return out


@pytest.fixture(scope="module")
def fitted():
    frame = _frame()
    rows = frame.index[::4]
    blanked = _blanked(frame, rows)
    state, imputed = miceforest_backend.fit(blanked, FEATURES, CATEGORICAL, NOMINAL, seed=1)
    return frame, blanked, rows, state, imputed


def test_fills_every_gap_and_keeps_observed_values_and_target(fitted):
    frame, blanked, _, _, imputed = fitted
    assert not imputed[FEATURES].isna().any().any()
    observed = blanked.notna()
    for column in FEATURES + ["target"]:
        rows = observed[column]
        pd.testing.assert_series_equal(imputed.loc[rows, column], blanked.loc[rows, column])
    assert set(imputed["const"]) == {"a"}


def test_fills_are_observed_values_that_keep_the_distribution(fitted):
    frame, blanked, rows, _, imputed = fitted
    # Mean matching copies observed donors: only real codes, both humps kept.
    assert set(imputed.loc[rows, "grade"]) <= set(blanked["grade"].dropna())
    middle = imputed.loc[rows, "grade"].isin([2.0, 3.0]).mean()
    assert middle == 0
    # Categories are drawn, not argmaxed: the minority classes survive.
    shares = imputed.loc[rows, "colour"].value_counts(normalize=True)
    assert shares.get("blue", 0) > 0.03 and shares.get("green", 0) > 0.15
    assert 0.4 < imputed.loc[rows, "flag"].mean() < 0.8
    # The forests still use the other columns: y is almost 3x.
    error = (imputed.loc[rows, "y"] - frame.loc[rows, "y"]).abs().mean()
    assert error < (frame["y"] - frame["y"].median()).abs().mean() / 3

    # Mean matching keeps the observed spread; MissForest's rounded conditional
    # means pull the fills toward the middle and the commonest class.
    observed_std = blanked["grade"].std()
    assert imputed.loc[rows, "grade"].std() > 0.8 * observed_std
    _, forest = sklearn_backend.fit(blanked, FEATURES, CATEGORICAL, NOMINAL, "missforest", seed=1)
    assert forest.loc[rows, "grade"].std() < 0.8 * observed_std
    assert forest.loc[rows, "colour"].eq("red").mean() > shares["red"]


def test_is_deterministic_and_transform_never_refits(fitted):
    _, blanked, _, state, imputed = fitted
    _, again = miceforest_backend.fit(blanked, FEATURES, CATEGORICAL, NOMINAL, seed=1)
    pd.testing.assert_frame_equal(imputed, again)

    other = _blanked(_frame(200, seed=5), list(range(0, 200, 3)))
    first = miceforest_backend.transform(state, other)
    pd.testing.assert_frame_equal(first, miceforest_backend.transform(state, other))
    # Fills for new rows come from the fit rows' observed values.
    filled = other["grade"].isna()
    assert set(first.loc[filled, "grade"]) <= set(blanked["grade"].dropna())


def test_column_names_lightgbm_rejects_round_trip():
    # LightGBM refuses feature names with JSON characters; the kernel must
    # never see them, and the output keeps the original names.
    names = {"x": "x [a,b]", "y": 'y: "b"', "colour": "colour {c}"}
    frame = _frame(n=300).rename(columns=names)
    rows = frame.index[::4]
    blanked = _blanked(frame, rows, columns=(names["y"], names["colour"]))
    features = [names.get(c, c) for c in FEATURES]
    categorical = [names.get(c, c) for c in CATEGORICAL]
    nominal = [names.get(c, c) for c in NOMINAL]
    state, imputed = miceforest_backend.fit(blanked, features, categorical, nominal, seed=1)
    assert list(imputed.columns) == list(frame.columns)
    assert not imputed[features].isna().any().any()
    again = miceforest_backend.transform(state, blanked)
    assert list(again.columns) == list(frame.columns)
    assert not again[features].isna().any().any()


def test_a_rare_binary_level_is_imputed():
    # With 3 positives in 2000 rows every leaf predicts exactly 0, and
    # miceforest's log-odds would be -inf without the clip.
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.normal(size=n)
    flag = np.zeros(n, dtype=int)
    flag[rng.choice(n, 3, replace=False)] = 1
    frame = pd.DataFrame({"x": x, "y": x + rng.normal(size=n), "rare": flag})
    blanked = frame.copy()
    blanked.loc[blanked.index[::5], "rare"] = np.nan
    blanked.loc[blanked.index[1::7], "x"] = np.nan
    state, imputed = miceforest_backend.fit(blanked, ["x", "y", "rare"], ["rare"], ["rare"], seed=1)
    assert not imputed[["x", "rare"]].isna().any().any()
    assert set(imputed["rare"]) <= {0, 1}
    again = miceforest_backend.transform(state, blanked)
    assert not again[["x", "rare"]].isna().any().any()


def test_columns_without_a_model_are_rejected():
    frame = _frame(300)
    state, _ = miceforest_backend.fit(
        _blanked(frame, frame.index[::5], ["y"]),
        FEATURES,
        CATEGORICAL,
        NOMINAL,
        seed=0,
        model_columns=["y"],
    )
    with pytest.raises(ValueError, match="no model"):
        miceforest_backend.transform(state, _blanked(frame, [0], ["x"]))


def test_pipeline_dispatch_fills_train_and_holdout(make_config, make_dataset):
    from synthdata.imputation.pipeline import _impute_splits

    frame = _frame(300)
    frame.loc[::7, "y"] = np.nan
    frame.loc[::5, "colour"] = np.nan
    dataset = make_dataset(df=frame, feature_columns=FEATURES, nominal_columns=NOMINAL)
    cfg = make_config()
    assert cfg.imputation.method == "miceforest"  # the default
    cfg.generation.hpo.enabled = False
    train_imputed, test_imputed, _ = _impute_splits(cfg, dataset, "cpu")
    assert not train_imputed[FEATURES].isna().any().any()
    assert not test_imputed[FEATURES].isna().any().any()
    assert test_imputed.index.equals(dataset.test_df.index)
