"""Subgroup fairness gaps of the holdout TSTR classifier, compared with real data.

SynthEval's statistical parity, equalized odds and equal opportunity fit
their classifier and score it inside the synthetic data. When the synthetic
data carries no class signal, the classifier predicts one class for every
row and every gap is exactly 0, the best possible score: on the simulated
clinical dataset ARF, TabularARGN, DDPM and the marginals baseline scored a
perfect 0 that way, and CDTD, the one generator with a learnable signal,
ranked last.

Here the gaps come from the classifier holdout TSTR already uses
(:func:`synthdata.evaluation.tstr.tstr_predictor`: the fixed, class-balanced
XGBoost), fitted on the synthetic data and predicting the real holdout, and
from the same classifier fitted on the real train split. For each class
(one against the rest) and each protected group (one against the rest, the
groups of :func:`synthdata.evaluation.privacy_attacks.protected_group_masks`)
fairlearn's ``MetricFrame`` gives the selection rate, true-positive rate and
false-positive rate of the group and of everyone else; their signed
differences are the group's demographic parity, equal opportunity and
equalized odds gaps. A synthetic dataset is fair in the sense that matters
for release when a model trained on it treats groups as one trained on the
real data does, so each metric is the mean absolute difference between the
synthetic and the real gap (equalized odds takes the larger of its TPR and
FPR differences, as fairlearn's ``equalized_odds_difference`` does). 0 means
the real disparities are kept; erasing or inverting them scores high.

A classifier that learned nothing from the synthetic data predicts almost
the same classes for every group, so its gaps are near 0 and each error is
about the size of the real gap: it scores like a dataset that erased the
disparities, which it has. No separate gate is needed for it, so every
model gets a score. ``fairness_signal`` (1 when the bootstrap 95% interval
of the classifier's holdout balanced accuracy is above chance) is reported
next to the errors, unranked, to say which models are in that position.
"""

import dataclasses

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from synthdata.evaluation.tstr import tstr_predictor
from synthdata.utils import get_logger

logger = get_logger(__name__)

#: Ranked metrics: mean |synthetic gap - real gap|, lower is better, ideal 0.
FAIRNESS_GAP_METRICS = (
    "demographic_parity_gap_error",
    "equalized_odds_gap_error",
    "equal_opportunity_gap_error",
)

#: Reported next to them, unranked: 1 when the classifier beats chance on the holdout.
FAIRNESS_SIGNAL_COLUMN = "fairness_signal"

#: Smallest group, and smallest rest, scored on the holdout; smaller give noisy rates.
MIN_GROUP_ROWS = 30

#: Bootstrap draws for the balanced-accuracy interval behind ``fairness_signal``.
N_BOOTSTRAP = 200

#: Two-sided level of that interval.
CONFIDENCE = 0.95


@dataclasses.dataclass
class FairnessGaps:
    """One model's gap errors, its holdout balanced accuracy and the gaps behind them."""

    errors: dict
    signal: bool
    balanced_accuracy: float
    balanced_accuracy_ci: tuple
    chance: float
    #: One row per (class, column, group): synthetic and real gaps.
    gaps: pd.DataFrame


def balanced_accuracy_interval(
    y_true: np.ndarray, y_pred: np.ndarray, n_bootstrap: int, seed: int, confidence: float
) -> tuple[float, float]:
    """Percentile bootstrap interval of balanced accuracy, resampling within each class."""
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(y_true == c) for c in np.unique(y_true)]
    draws = []
    for _ in range(n_bootstrap):
        idx = np.concatenate([rng.choice(rows, len(rows)) for rows in by_class])
        draws.append(balanced_accuracy_score(y_true[idx], y_pred[idx]))
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(draws, [tail, 100 - tail])
    return float(low), float(high)


def group_gaps(y_true: np.ndarray, y_pred: np.ndarray, masks: dict, n_classes: int) -> pd.DataFrame:
    """Signed gaps (group minus rest) per class one-vs-rest and protected group.

    ``masks``: ``{(column, group): boolean array over the scored rows}``.
    Columns ``class, column, group, dp, tpr, fpr``; a rate fairlearn cannot
    compute (no positives or negatives on one side) is NaN.
    """
    from fairlearn.metrics import (
        MetricFrame,
        false_positive_rate,
        selection_rate,
        true_positive_rate,
    )

    rows = []
    for (column, group), mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() < MIN_GROUP_ROWS or (~mask).sum() < MIN_GROUP_ROWS:
            continue
        for cls in range(n_classes):
            truth = (y_true == cls).astype(int)
            pred = (y_pred == cls).astype(int)
            rates = {}
            for name, metric in (
                ("dp", selection_rate),
                ("tpr", true_positive_rate),
                ("fpr", false_positive_rate),
            ):
                frame = MetricFrame(
                    metrics=metric, y_true=truth, y_pred=pred, sensitive_features=mask
                )
                by_group = frame.by_group
                # A side with no positives (TPR) or negatives (FPR) has no rate.
                defined = {
                    "tpr": truth[mask].any() and truth[~mask].any(),
                    "fpr": (1 - truth[mask]).any() and (1 - truth[~mask]).any(),
                }.get(name, True)
                rates[name] = float(by_group[True] - by_group[False]) if defined else np.nan
            rows.append({"class": cls, "column": column, "group": group, **rates})
    return pd.DataFrame(rows, columns=["class", "column", "group", "dp", "tpr", "fpr"])


def gap_errors(syn: pd.DataFrame, real: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Mean |synthetic gap - real gap| per metric, and the joined gap table."""
    keys = ["class", "column", "group"]
    joined = syn.merge(real, on=keys, suffixes=("_syn", "_real"))
    diff = {m: (joined[f"{m}_syn"] - joined[f"{m}_real"]).abs() for m in ("dp", "tpr", "fpr")}
    eo = pd.concat([diff["tpr"], diff["fpr"]], axis=1).max(axis=1, skipna=True)
    errors = {
        "demographic_parity_gap_error": float(diff["dp"].mean()),
        "equalized_odds_gap_error": float(eo.mean()),
        "equal_opportunity_gap_error": float(diff["tpr"].mean()),
    }
    return errors, joined


def fairness_gap_scores(
    synthetic_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    nominal_columns: list,
    classes: list,
    masks: dict,
    seed: int,
    xgb_params: dict | None = None,
    real_predictions: np.ndarray | None = None,
) -> FairnessGaps:
    """Gap errors of one synthetic dataset against the real-trained classifier.

    ``real_predictions`` (the real-trained classifier's holdout classes) can
    be passed so it is fitted once for all models.
    """
    y_true, predict = tstr_predictor(
        synthetic_df, test_df, target, nominal_columns, classes, xgb_params
    )
    y_syn = predict(seed).argmax(axis=1)
    if real_predictions is None:
        real_predictions = real_holdout_predictions(
            train_df, test_df, target, nominal_columns, classes, seed, xgb_params
        )
    chance = 1.0 / len(np.unique(y_true))
    accuracy = float(balanced_accuracy_score(y_true, y_syn))
    interval = balanced_accuracy_interval(y_true, y_syn, N_BOOTSTRAP, seed, CONFIDENCE)
    signal = interval[0] > chance
    errors, joined = gap_errors(
        group_gaps(y_true, y_syn, masks, len(classes)),
        group_gaps(y_true, real_predictions, masks, len(classes)),
    )
    joined["class"] = [str(classes[c]) for c in joined["class"]]
    return FairnessGaps(errors, bool(signal), accuracy, interval, chance, joined)


def real_holdout_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    nominal_columns: list,
    classes: list,
    seed: int,
    xgb_params: dict | None = None,
) -> np.ndarray:
    """Holdout classes predicted by the fixed classifier fitted on the real train split."""
    _, predict = tstr_predictor(train_df, test_df, target, nominal_columns, classes, xgb_params)
    return predict(seed).argmax(axis=1)


def build_fairness_gap_table(results: dict) -> pd.DataFrame:
    """Models x (holdout balanced accuracy, gap errors) summary for the report and the ranking."""
    rows = {}
    for name, r in results.items():
        rows[name] = {
            FAIRNESS_SIGNAL_COLUMN: int(r.signal),
            "holdout_balanced_accuracy": r.balanced_accuracy,
            "balanced_accuracy_ci_low": r.balanced_accuracy_ci[0],
            "balanced_accuracy_ci_high": r.balanced_accuracy_ci[1],
            "chance_balanced_accuracy": r.chance,
            **r.errors,
        }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "model"
    return table


def build_fairness_gap_detail(results: dict) -> pd.DataFrame:
    """Every model's synthetic and real gap per class and protected group."""
    frames = [r.gaps.assign(model=name) for name, r in results.items() if not r.gaps.empty]
    if not frames:
        return pd.DataFrame()
    detail = pd.concat(frames, ignore_index=True)
    return detail[["model", *[c for c in detail.columns if c != "model"]]]
