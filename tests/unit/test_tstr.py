"""Unit tests for the fixed-XGBoost TSTR scores and class-prior matching."""

import numpy as np
import pandas as pd
import pytest

from synthdata.evaluation.tstr import match_class_prior, tstr_scores

pytestmark = pytest.mark.unit


def _frame(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame(
        {
            "x": x,
            "site": rng.choice(["a", "b"], size=n),
            "y": np.where(x > 1.0, "rare", "common"),
        }
    )


def test_tstr_learns_a_real_signal_and_is_deterministic():
    fit, score = _frame(400, 0), _frame(300, 1)
    first = tstr_scores(fit, score, "y", ["site"], ["common", "rare"], [0, 1])
    second = tstr_scores(fit, score, "y", ["site"], ["common", "rare"], [0, 1])
    assert first == second
    assert first.macro_f1 > 0.9
    assert first.macro_auprc > 0.9
    assert set(first.per_class_f1) == {"common", "rare"}


def test_a_class_the_fit_set_lacks_scores_zero_f1():
    fit = pd.concat([_frame(400, 0)] * 1).assign(y=lambda d: np.where(d.x > 0, "common", "middle"))
    score = _frame(300, 1)
    scores = tstr_scores(fit, score, "y", ["site"], ["common", "middle", "rare"], [0])
    assert scores.per_class_f1["rare"] == 0.0


def test_unseen_test_categories_do_not_break_encoding():
    fit, score = _frame(200, 0), _frame(100, 1)
    score.loc[:5, "site"] = "new_site"
    tstr_scores(fit, score, "y", ["site"], ["common", "rare"], [0])


def test_match_class_prior_restores_train_shares_and_size():
    balanced = pd.DataFrame({"y": ["a"] * 50 + ["b"] * 50, "v": range(100)})
    prior = pd.Series({"a": 0.9, "b": 0.1})
    out = match_class_prior(balanced, "y", prior, seed=0)
    assert len(out) == 100
    assert out["y"].value_counts().to_dict() == {"a": 90, "b": 10}
    # "b" is drawn without replacement; "a" needs replacement.
    assert out.loc[out.y == "b", "v"].is_unique


def test_match_class_prior_renormalizes_over_present_classes():
    df = pd.DataFrame({"y": ["a"] * 10 + ["b"] * 10})
    out = match_class_prior(df, "y", pd.Series({"a": 0.6, "b": 0.2, "c": 0.2}), seed=0)
    assert out["y"].value_counts().to_dict() == {"a": 15, "b": 5}


def test_float_coded_synthetic_labels_match_integer_classes():
    fit = _frame(300, 0).assign(y=lambda d: (d.x > 1.0).astype(float))
    score = _frame(200, 1).assign(y=lambda d: (d.x > 1.0).astype(int))
    assert tstr_scores(fit, score, "y", ["site"], [0, 1], [0]).macro_f1 > 0.9


def test_a_collapsed_fit_set_predicts_its_one_class():
    fit = _frame(100, 0).assign(y="common")
    scores = tstr_scores(fit, _frame(200, 1), "y", ["site"], ["common", "rare"], [0])
    assert scores.per_class_f1["rare"] == 0.0
    assert scores.per_class_f1["common"] > 0.5
