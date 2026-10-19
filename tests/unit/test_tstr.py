"""Unit tests for the fixed-XGBoost TSTR scores and class-prior matching."""

import numpy as np
import pandas as pd
import pytest

from synthdata.evaluation.tstr import XGB_PARAMS, tstr_scores, tstr_xgb_params

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


def test_user_xgb_params_override_only_their_keys_and_reach_the_fit(monkeypatch):
    params = tstr_xgb_params({"max_depth": 5, "subsample": 0.8})
    assert params == {**XGB_PARAMS, "max_depth": 5, "subsample": 0.8}
    assert tstr_xgb_params(None) == XGB_PARAMS

    import xgboost

    seen = []
    real = xgboost.XGBClassifier

    class Spy(real):
        def __init__(self, **kwargs):
            seen.append(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(xgboost, "XGBClassifier", Spy)
    fit, score = _frame(400, 0), _frame(300, 1)
    tuned = tstr_scores(
        fit, score, "y", ["site"], ["common", "rare"], [0, 1], xgb_params={"subsample": 0.5}
    )
    assert [k["subsample"] for k in seen] == [0.5, 0.5]
    assert [k["random_state"] for k in seen] == [0, 1]
    assert tuned.macro_f1 > 0.9
    # Row subsampling makes the fit depend on its seed; seeded, it still repeats.
    again = tstr_scores(
        fit, score, "y", ["site"], ["common", "rare"], [0, 1], xgb_params={"subsample": 0.5}
    )
    assert tuned == again


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


def test_float_coded_synthetic_labels_match_integer_classes():
    fit = _frame(300, 0).assign(y=lambda d: (d.x > 1.0).astype(float))
    score = _frame(200, 1).assign(y=lambda d: (d.x > 1.0).astype(int))
    assert tstr_scores(fit, score, "y", ["site"], [0, 1], [0]).macro_f1 > 0.9


def test_a_collapsed_fit_set_predicts_its_one_class():
    fit = _frame(100, 0).assign(y="common")
    scores = tstr_scores(fit, _frame(200, 1), "y", ["site"], ["common", "rare"], [0])
    assert scores.per_class_f1["rare"] == 0.0
    assert scores.per_class_f1["common"] > 0.5


def _weak_frame(n, seed):
    """A rare class (about 10%) only weakly tied to ``x``."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    rare = x + 2.0 * rng.normal(size=n) > 2.8
    return pd.DataFrame(
        {"x": x, "site": rng.choice(["a", "b"], size=n), "y": np.where(rare, "rare", "common")}
    )


def test_class_weights_make_the_fit_predict_the_minority_class():
    """Balanced class weights lift balanced accuracy where an unweighted fit
    predicts the majority (balanced accuracy about 0.5)."""
    fit, score = _weak_frame(2000, 0), _weak_frame(2000, 1)
    scores = tstr_scores(fit, score, "y", ["site"], ["common", "rare"], [0, 1])
    assert scores.balanced_accuracy > 0.6
    assert scores.per_class_f1["rare"] > 0.2
    assert set(scores.seed_sd) == {
        "macro_f1",
        "macro_auprc",
        "macro_auroc",
        "balanced_accuracy",
        "kappa",
    }


def test_kappa_matches_sklearn_and_weights_ordinal_errors_by_distance():
    """Kappa is scikit-learn's; quadratic weights forgive a near miss on an
    ordinal target more than a far one."""
    from sklearn.metrics import cohen_kappa_score

    from synthdata.evaluation.tstr import _kappa

    y_true = np.array([0, 0, 1, 1, 2, 2, 2, 0])
    near = np.array([0, 1, 1, 2, 2, 1, 2, 0])
    far = np.array([0, 2, 1, 0, 2, 0, 2, 0])
    assert _kappa(y_true, near, 3, False) == pytest.approx(cohen_kappa_score(y_true, near))
    assert _kappa(y_true, near, 3, True) == pytest.approx(
        cohen_kappa_score(y_true, near, weights="quadratic")
    )
    assert _kappa(y_true, near, 3, True) > _kappa(y_true, far, 3, True)
    assert _kappa(np.zeros(5, int), np.zeros(5, int), 2, False) == 0.0


def test_target_classes_orders_an_ordinal_target_numerically():
    from synthdata.evaluation.tstr import target_classes

    codes = pd.Series([10, 2, 1, 2, None])
    assert target_classes(codes, ordinal=True) == [1, 2, 10]
    assert target_classes(pd.Series(["b", "a"])) == ["a", "b"]
