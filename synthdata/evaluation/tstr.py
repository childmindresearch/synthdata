"""Train-on-synthetic, test-on-real (TSTR) with a fixed XGBoost classifier.

The classifier's hyperparameters never change, so a TSTR score measures the
training data rather than the classifier (ported from the core of
``feat/final-release-evaluation``'s ``tstr.py``, without its provenance layer).
Scores are averaged over the given XGBoost seeds; with ``subsample`` and
``colsample_bytree`` at 1 the classifier is deterministic, so one seed is
enough and the seeds add no noise estimate. ``evaluation.xgboost_params``
can replace any of these settings (for example with parameters tuned for the
real data); with row or column subsampling the seeds then matter. Every fit weights its rows by
inverse class frequency (scikit-learn's ``compute_sample_weight("balanced")``),
so each class contributes equally to the loss and an imbalanced target is
handled the same way in HPO, the TRTR ceiling and holdout evaluation. The score's sampling noise comes
instead from a stratified bootstrap of the scored rows (``n_bootstrap``).
Fitting the same classifier on real training rows (train-on-real,
test-on-real, TRTR) gives the ceiling a synthetic dataset can be compared
against, and :func:`chance_scores` the floor.
"""

import dataclasses

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)

from synthdata.utils import get_logger

logger = get_logger(__name__)

#: Fixed evaluator, as in the feat branch: small, shallow, deterministic.
#: ``evaluation.xgboost_params`` overrides any of these keys.
XGB_PARAMS = {
    "n_estimators": 80,
    "max_depth": 4,
    "learning_rate": 0.08,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "tree_method": "hist",
}

#: Keys the pipeline sets itself, so ``evaluation.xgboost_params`` may
#: not: the seeds come from ``tstr_seeds`` and the multi-class objective
#: from the target's classes.
RESERVED_XGB_KEYS = frozenset({"random_state", "seed", "objective", "num_class"})


def tstr_xgb_params(overrides: dict | None = None) -> dict:
    """The TSTR classifier's parameters: :data:`XGB_PARAMS` updated by ``overrides``."""
    return {**XGB_PARAMS, **(overrides or {})}


@dataclasses.dataclass
class TSTRScores:
    """Seed-averaged scores of one fit-on-A, score-on-B comparison."""

    macro_f1: float
    macro_auprc: float
    balanced_accuracy: float
    per_class_f1: dict
    #: Standard deviation of each score across seeds (0 for the deterministic
    #: fixed classifier).
    seed_sd: dict = dataclasses.field(default_factory=dict)
    #: Macro one-vs-rest AUROC (threshold-free; chance 0.5 whatever the class shares).
    macro_auroc: float | None = None
    #: Cohen's kappa: agreement with the true labels beyond chance (0 = no
    #: better than chance, 1 = perfect). Quadratic-weighted when the target is
    #: ordinal, so a prediction one class off costs less than one far off.
    kappa: float | None = None
    #: Bootstrap standard error of each score over the scored rows (first
    #: seed's predictions; ``n_bootstrap`` > 0 only): the evaluation's
    #: sampling noise, against which a difference between candidates is judged.
    boot_se: dict = dataclasses.field(default_factory=dict)
    #: Macro AUPRC within each group of each ``groups`` column (groups with
    #: at least ``MIN_GROUP_ROWS`` rows): ``{column: {group: auprc}}``.
    group_auprc: dict = dataclasses.field(default_factory=dict)


#: Smallest group scored by ``group_auprc``; smaller groups are too noisy.
MIN_GROUP_ROWS = 30


def _macro_auprc(y_true: np.ndarray, proba: np.ndarray) -> float:
    scored = [i for i in range(proba.shape[1]) if (y_true == i).any()]
    return float(np.mean([average_precision_score(y_true == i, proba[:, i]) for i in scored]))


def _macro_auroc(y_true: np.ndarray, proba: np.ndarray) -> float:
    scored = [i for i in range(proba.shape[1]) if 0 < (y_true == i).sum() < len(y_true)]
    if not scored:
        return float("nan")
    return float(np.mean([roc_auc_score(y_true == i, proba[:, i]) for i in scored]))


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    return float(
        f1_score(y_true, y_pred, labels=range(n_classes), average="macro", zero_division=0)
    )


def _kappa(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int, ordinal: bool) -> float:
    if len(np.unique(np.r_[y_true, y_pred])) < 2:
        # Both sides constant and equal: kappa is 0/0; no agreement beyond chance.
        return 0.0
    return float(
        cohen_kappa_score(
            y_true, y_pred, labels=range(n_classes), weights="quadratic" if ordinal else None
        )
    )


def chance_scores(score_df: pd.DataFrame, target_column: str, classes: list) -> dict:
    """Each score of a classifier with no information, on ``score_df``.

    macro-F1: always predicting the most frequent class.
    Macro AUPRC: the mean class prevalence. Macro AUROC: 0.5. Balanced
    accuracy: 1 / number of classes present. Cohen's kappa: 0 by construction.
    """
    y_true = _label_codes(score_df[target_column], classes).astype(int).to_numpy()
    majority = np.full_like(y_true, np.bincount(y_true).argmax())
    present = np.unique(y_true)
    constant_f1 = _macro_f1(y_true, majority, len(classes))
    return {
        "macro_f1": constant_f1,
        "macro_auprc": float(np.mean([(y_true == i).mean() for i in present])),
        "macro_auroc": 0.5,
        "balanced_accuracy": 1.0 / len(present),
        "kappa": 0.0,
    }


def _bootstrap_se(y_true, proba, n_classes, n_bootstrap, seed, ordinal=False) -> dict:
    """Stratified bootstrap standard errors of the threshold and ranking scores."""
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(y_true == i) for i in np.unique(y_true)]
    draws = {
        "macro_f1": [],
        "macro_auprc": [],
        "macro_auroc": [],
        "balanced_accuracy": [],
        "kappa": [],
    }
    y_pred = proba.argmax(axis=1)
    for _ in range(n_bootstrap):
        idx = np.concatenate([rng.choice(rows, len(rows)) for rows in by_class])
        y, p = y_true[idx], proba[idx]
        draws["macro_f1"].append(_macro_f1(y, y_pred[idx], n_classes))
        draws["macro_auprc"].append(_macro_auprc(y, p))
        draws["macro_auroc"].append(_macro_auroc(y, p))
        draws["balanced_accuracy"].append(balanced_accuracy_score(y, y_pred[idx]))
        draws["kappa"].append(_kappa(y, y_pred[idx], n_classes, ordinal))
    return {name: float(np.nanstd(values, ddof=1)) for name, values in draws.items()}


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


def target_classes(values: pd.Series, ordinal: bool = False) -> list:
    """The target's classes in a fixed order: low to high for an ordinal
    target (stored as integer codes in its declared order), else by label."""
    present = values.dropna().unique().tolist()
    if ordinal:
        return sorted(present, key=float)
    return sorted(present, key=str)


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
    n_bootstrap: int = 0,
    groups: dict | None = None,
    ordinal: bool = False,
    xgb_params: dict | None = None,
) -> TSTRScores:
    """Fit the fixed XGBoost on ``fit_df`` once per seed and score on ``score_df``.

    ``classes`` fixes the label set (the real training classes), so a fit set
    that lacks a class still yields a model scored over every class: that
    class's F1 is then 0. Rows of ``fit_df`` with a label outside ``classes``
    are dropped. Rows are weighted by inverse class frequency
    (``compute_sample_weight("balanced")``), so every class counts equally.
    ``n_bootstrap`` > 0 adds bootstrap standard errors (``boot_se``).
    ``ordinal``: the target's classes are ordered and ``classes`` lists them
    low to high, so kappa uses quadratic weights.
    ``groups`` (``{column: labels aligned with score_df}``) adds the macro
    AUPRC within each group (``group_auprc``), from the first seed's fit.
    ``xgb_params`` overrides keys of :data:`XGB_PARAMS` (see :func:`tstr_xgb_params`).
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

    sample_weight = compute_sample_weight("balanced", y_fit)
    params = tstr_xgb_params(xgb_params)

    def predict_proba(seed):
        proba = np.zeros((len(x_score), len(classes)))
        if len(present) == 1:
            # A collapsed fit set can only ever predict its one class.
            proba[:, present[0]] = 1.0
        else:
            model = XGBClassifier(**params, random_state=seed)
            model.fit(x_fit, np.vectorize(to_local.get)(y_fit), sample_weight=sample_weight)
            proba[:, present] = model.predict_proba(x_score)
        return proba

    f1s, auprcs, aurocs, balanced, kappas, per_class = [], [], [], [], [], []
    first_proba = None
    for seed in seeds:
        proba = predict_proba(seed)
        y_pred = proba.argmax(axis=1)
        f1s.append(_macro_f1(y_true, y_pred, len(classes)))
        per_class.append(
            f1_score(y_true, y_pred, labels=range(len(classes)), average=None, zero_division=0)
        )
        balanced.append(balanced_accuracy_score(y_true, y_pred))
        kappas.append(_kappa(y_true, y_pred, len(classes), ordinal))
        auprcs.append(_macro_auprc(y_true, proba))
        aurocs.append(_macro_auroc(y_true, proba))
        if first_proba is None:
            first_proba = proba
    if first_proba is None:
        raise ValueError("TSTR needs at least one seed")

    boot_se = (
        _bootstrap_se(y_true, first_proba, len(classes), n_bootstrap, seeds[0], ordinal)
        if n_bootstrap > 0
        else {}
    )
    group_auprc = {}
    for column, labels in (groups or {}).items():
        labels = pd.Series(np.asarray(labels))
        group_auprc[column] = {
            str(group): _macro_auprc(y_true[rows], first_proba[rows])
            for group, rows in labels.groupby(labels).indices.items()
            if len(rows) >= MIN_GROUP_ROWS
        }

    return TSTRScores(
        macro_f1=float(np.mean(f1s)),
        macro_auprc=float(np.mean(auprcs)),
        macro_auroc=float(np.mean(aurocs)),
        boot_se=boot_se,
        group_auprc=group_auprc,
        balanced_accuracy=float(np.mean(balanced)),
        kappa=float(np.mean(kappas)),
        per_class_f1={
            str(c): float(v) for c, v in zip(classes, np.mean(per_class, axis=0), strict=True)
        },
        seed_sd={
            name: float(np.std(values))
            for name, values in {
                "macro_f1": f1s,
                "macro_auprc": auprcs,
                "macro_auroc": aurocs,
                "balanced_accuracy": balanced,
                "kappa": kappas,
            }.items()
            if values
        },
    )
