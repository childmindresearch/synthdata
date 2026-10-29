"""TSTR XGBoost tuning script (scripts.tune_tstr_xgboost) on planted data."""

import numpy as np
import pandas as pd
import pytest

from scripts import tune_tstr_xgboost as tx

pytestmark = pytest.mark.unit


def _frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.where(x > 0.8, "a", np.where(x < -0.8, "b", "c"))
    patients = np.repeat(np.arange(n // 3), 3)
    frame = pd.DataFrame({"x": x, "noise": rng.normal(size=n), "y": y})
    return frame, pd.Series(patients, index=frame.index)


def test_folds_never_split_a_patient():
    frame, patients = _frame()
    for fit, score in tx.grouped_folds(frame, "y", patients, 3, seed=0):
        assert not set(patients.loc[fit]) & set(patients.loc[score])
        assert set(frame.loc[score, "y"]) == {"a", "b", "c"}


def test_tuning_reports_defaults_configured_and_best_on_the_same_folds():
    frame, patients = _frame()
    configured = {"n_estimators": 20, "max_depth": 2}
    summary, trials = tx.tune(
        frame,
        patients,
        "y",
        [],
        ["a", "b", "c"],
        n_trials=3,
        n_folds=3,
        seed=0,
        configured=configured,
    )
    assert len(trials) == 3
    # Trial 0 is the defaults, so the best trial is at least as good.
    assert summary["best"]["macro_auprc"] >= summary["defaults"]["macro_auprc"] - 1e-12
    assert summary["defaults"]["best_minus_this"] >= -1e-12
    assert summary["configured"]["params"] == configured
    # x separates the classes, so every fit is far above chance (1/3).
    assert summary["best"]["macro_auprc"] > 0.8
    assert set(summary["best_params"]) >= {"n_estimators", "max_depth", "learning_rate"}
