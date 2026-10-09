"""Train-on-synthetic, test-on-real (TSTR) with a fixed XGBoost classifier.

The classifier's hyperparameters never change, so a TSTR score measures the
training data rather than the classifier (ported from the core of
``feat/final-release-evaluation``'s ``tstr.py``, without its provenance layer).
Scores are averaged over a few XGBoost seeds. Fitting the same classifier on
real training rows (train-on-real, test-on-real, TRTR) gives the ceiling a
synthetic dataset can be compared against.
"""

import dataclasses

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score

from synthdata.utils import get_logger

logger = get_logger(__name__)

#: Fixed evaluator, as in the feat branch: small, shallow, deterministic.
XGB_PARAMS = {
    "n_estimators": 80,
    "max_depth": 4,
    "learning_rate": 0.08,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "tree_method": "hist",
}


@dataclasses.dataclass
class TSTRScores:
    """Seed-averaged scores of one fit-on-A, score-on-B comparison."""

    macro_f1: float
    macro_auprc: float
    balanced_accuracy: float
    per_class_f1: dict
    #: The same scores from a second fit with rows weighted by inverse class
    #: frequency (``weighted=True`` only), so the classifier predicts the
    #: minority classes; logged to compare objectives, not used to rank.
    weighted_macro_f1: float | None = None
    weighted_balanced_accuracy: float | None = None
    #: Standard deviation of each score across seeds: the evaluator's own
    #: noise, against which a difference between candidates is judged.
    seed_sd: dict = dataclasses.field(default_factory=dict)


def _encode(
    train: pd.DataFrame, test: pd.DataFrame, nominal_columns: list
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot nominal and non-numeric columns; align test to train's columns."""
    one_hot = [
        c
        for c in train.columns
        if c in nominal_columns or not pd.api.types.is_numeric_dtype(train[c])
    ]

    def encode(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[one_hot] = df[one_hot].astype(str)
        return pd.get_dummies(df, columns=one_hot, dtype=float)

    x_train = encode(train)
    # Categories seen only in the real test rows get no column, as for any
    # deployed model trained on the synthetic data.
    x_test = encode(test[train.columns]).reindex(columns=x_train.columns, fill_value=0.0)
    names = [f"f{i}" for i in range(x_train.shape[1])]  # XGBoost rejects [, ], <
    x_train.columns = names
    x_test.columns = names
    return x_train.astype(float), x_test.astype(float)


def _label_codes(values: pd.Series, classes: list) -> pd.Series:
    """Map target values to class indices; numeric labels match as numbers (1 == 1.0)."""
    if all(isinstance(c, (int, float, np.number)) for c in classes):
        lookup = {float(c): i for i, c in enumerate(classes)}
        return pd.to_numeric(values, errors="coerce").map(lookup)
    return values.astype(str).map({str(c): i for i, c in enumerate(classes)})


def tstr_scores(
    fit_df: pd.DataFrame,
    score_df: pd.DataFrame,
    target_column: str,
    nominal_columns: list,
    classes: list,
    seeds: list,
    weighted: bool = False,
) -> TSTRScores:
    """Fit the fixed XGBoost on ``fit_df`` once per seed and score on ``score_df``.

    ``classes`` fixes the label set (the real training classes), so a fit set
    that lacks a class still yields a model scored over every class: that
    class's F1 is then 0. Rows of ``fit_df`` with a label outside ``classes``
    are dropped. ``weighted`` adds a fit with ``sample_weight`` balanced by
    class (scikit-learn's ``compute_sample_weight("balanced")``).
    """
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier

    y_fit = _label_codes(fit_df[target_column], classes)
    fit_df = fit_df[y_fit.notna()]
    y_fit = y_fit[y_fit.notna()].astype(int).to_numpy()
    y_true = _label_codes(score_df[target_column], classes)
    if y_true.isna().any():
        raise ValueError(f"{target_column!r} has values outside the training classes {classes}")
    y_true = y_true.astype(int).to_numpy()

    features = [c for c in fit_df.columns if c != target_column]
    x_fit, x_score = _encode(fit_df[features], score_df[features], nominal_columns)

    # XGBoost needs labels 0..k-1 with every label present; remap to the
    # classes the fit set has and spread probabilities back over all classes.
    present = np.unique(y_fit)
    if len(present) == 0:
        raise ValueError(f"fit set has no rows with a target in {classes}")
    to_local = {c: i for i, c in enumerate(present)}

    def predict_proba(seed, sample_weight=None):
        proba = np.zeros((len(x_score), len(classes)))
        if len(present) == 1:
            # A collapsed fit set can only ever predict its one class.
            proba[:, present[0]] = 1.0
        else:
            model = XGBClassifier(**XGB_PARAMS, random_state=seed)
            model.fit(x_fit, np.vectorize(to_local.get)(y_fit), sample_weight=sample_weight)
            proba[:, present] = model.predict_proba(x_score)
        return proba

    f1s, auprcs, balanced, per_class = [], [], [], []
    weighted_f1s, weighted_balanced = [], []
    for seed in seeds:
        proba = predict_proba(seed)
        y_pred = proba.argmax(axis=1)
        f1s.append(
            f1_score(y_true, y_pred, labels=range(len(classes)), average="macro", zero_division=0)
        )
        per_class.append(
            f1_score(y_true, y_pred, labels=range(len(classes)), average=None, zero_division=0)
        )
        balanced.append(balanced_accuracy_score(y_true, y_pred))
        scored = [i for i in range(len(classes)) if (y_true == i).any()]
        auprcs.append(np.mean([average_precision_score(y_true == i, proba[:, i]) for i in scored]))
        if weighted:
            w_pred = predict_proba(seed, compute_sample_weight("balanced", y_fit)).argmax(axis=1)
            weighted_f1s.append(
                f1_score(
                    y_true, w_pred, labels=range(len(classes)), average="macro", zero_division=0
                )
            )
            weighted_balanced.append(balanced_accuracy_score(y_true, w_pred))

    return TSTRScores(
        macro_f1=float(np.mean(f1s)),
        macro_auprc=float(np.mean(auprcs)),
        balanced_accuracy=float(np.mean(balanced)),
        per_class_f1={
            str(c): float(v) for c, v in zip(classes, np.mean(per_class, axis=0), strict=True)
        },
        weighted_macro_f1=float(np.mean(weighted_f1s)) if weighted else None,
        weighted_balanced_accuracy=float(np.mean(weighted_balanced)) if weighted else None,
        seed_sd={
            name: float(np.std(values))
            for name, values in {
                "macro_f1": f1s,
                "macro_auprc": auprcs,
                "balanced_accuracy": balanced,
                "weighted_macro_f1": weighted_f1s,
                "weighted_balanced_accuracy": weighted_balanced,
            }.items()
            if values
        },
    )
