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
) -> TSTRScores:
    """Fit the fixed XGBoost on ``fit_df`` once per seed and score on ``score_df``.

    ``classes`` fixes the label set (the real training classes), so a fit set
    that lacks a class still yields a model scored over every class: that
    class's F1 is then 0. Rows of ``fit_df`` with a label outside ``classes``
    are dropped.
    """
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

    f1s, auprcs, balanced, per_class = [], [], [], []
    for seed in seeds:
        proba = np.zeros((len(x_score), len(classes)))
        if len(present) == 1:
            # A collapsed fit set can only ever predict its one class.
            proba[:, present[0]] = 1.0
        else:
            model = XGBClassifier(**XGB_PARAMS, random_state=seed)
            model.fit(x_fit, np.vectorize(to_local.get)(y_fit))
            proba[:, present] = model.predict_proba(x_score)
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

    return TSTRScores(
        macro_f1=float(np.mean(f1s)),
        macro_auprc=float(np.mean(auprcs)),
        balanced_accuracy=float(np.mean(balanced)),
        per_class_f1={
            str(c): float(v) for c, v in zip(classes, np.mean(per_class, axis=0), strict=True)
        },
    )


def match_class_prior(
    synthetic_df: pd.DataFrame, target_column: str, prior: pd.Series, seed: int
) -> pd.DataFrame:
    """Resample ``synthetic_df`` so its class shares match ``prior``, keeping its size.

    Classes are drawn without replacement while a class has enough rows and
    with replacement otherwise. Classes missing from ``synthetic_df`` cannot be
    created; the prior is renormalized over the classes it has. This stops a
    generator from raising macro-F1 by rebalancing the classes.
    """
    n = len(synthetic_df)
    counts = synthetic_df[target_column].value_counts()
    prior = prior[prior.index.isin(counts.index)]
    if prior.empty:
        return synthetic_df
    prior = prior / prior.sum()
    # Largest-remainder rounding, so the class sizes add up to n.
    exact = prior * n
    sizes = np.floor(exact).astype(int)
    shortfall = n - int(sizes.sum())
    sizes[(exact - sizes).sort_values(ascending=False).index[:shortfall]] += 1

    rng = np.random.default_rng(seed)
    parts = []
    for cls, size in sizes.items():
        rows = synthetic_df[synthetic_df[target_column] == cls]
        replace = size > len(rows)
        if replace:
            logger.info(
                "class %r: drawing %d rows from %d with replacement to match the train prior",
                cls,
                size,
                len(rows),
            )
        take = rng.choice(len(rows), size=size, replace=replace)
        parts.append(rows.iloc[take])
    order = rng.permutation(n)
    return pd.concat(parts).iloc[order].reset_index(drop=True)
