"""Redundant columns and hierarchies: an advisory report to help slim the dataset.

Three kinds of finding, each with a suggested action for a person to review:

* **Functional dependencies** X -> Y between categorical (or few-valued
  numeric) columns, with X of up to ``max_lhs`` columns: knowing X fixes Y,
  possibly up to a small error. Found by Desbordante's Pyro (Kruse & Naumann,
  *PVLDB* 2018) in its own environment (``envs/desbordante``; Desbordante is
  AGPL-3.0, so synthdata never imports it). Pyro's error measure counts
  violating row pairs, which lets near-unique X "determine" anything, so each
  candidate is re-scored here with the g3 error (share of rows to remove for
  X -> Y to hold exactly; Kivinen & Mannila 1995) on the indicator-fit rows and
  on the holdout, and kept only with enough rows per X value and when it also
  holds on the holdout.
* **Unique column combinations** among the role columns (quasi-identifiers,
  sensitive, protected): column sets that single out every row. A privacy
  finding, found by Desbordante's HyUCC.
* **Numeric redundancy**: continuous columns that a linear model on the other
  continuous columns predicts almost perfectly (cross-validated R²), such as a
  score that is the sum of its items. Formulas like BMI are not linear but
  still score high.

Blanks count as a value of their own, so skip patterns show up as
dependencies too. Nothing is changed automatically.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from synthdata.utils import get_logger

logger = get_logger(__name__)

DEPENDENCIES_FILENAME = "dependencies.csv"
#: Text that stands for a blank cell when the data goes to Desbordante.
BLANK = "<NA>"
#: Suggested actions.
MERGE_HIERARCHY = "merge_hierarchy"
DERIVE = "derive"
REVIEW = "review"
PRIVACY_REVIEW = "privacy_review"

COLUMNS = [
    "kind",
    "lhs",
    "rhs",
    "train_error",
    "holdout_error",
    "rows_per_lhs_value",
    "lhs_values",
    "rhs_values",
    "r2",
    "action",
]


def fd_candidate_columns(frame: pd.DataFrame, categorical: list, max_values: int = 50) -> list:
    """Categorical columns, plus numeric columns with at most ``max_values`` distinct values."""
    categorical = set(categorical)
    out = []
    for column in frame.columns:
        n = frame[column].nunique(dropna=False)
        if n < 2:
            continue
        if column in categorical or n <= max_values:
            out.append(column)
    return out


def _as_text(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.astype(object).where(frame.notna(), BLANK).astype(str)


def g3_error(frame: pd.DataFrame, lhs: list, rhs: str) -> float:
    """Share of rows to remove so that ``lhs`` -> ``rhs`` holds exactly (blanks are a value)."""
    if frame.empty:
        return float("nan")
    text = _as_text(frame[[*lhs, rhs]])
    kept = text.groupby([*lhs, rhs], dropna=False).size().groupby(level=list(range(len(lhs)))).max()
    return float(1 - kept.sum() / len(text))


def score_fds(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    candidates: list,
    categorical: list,
    max_error: float = 0.01,
    min_rows_per_value: float = 5,
) -> pd.DataFrame:
    """Re-score ``(lhs, rhs)`` candidates; keep those that hold on train and holdout.

    Kept: g3 error at most ``max_error`` on train and at most twice that on
    the holdout, and on average at least ``min_rows_per_value`` train rows
    per distinct ``lhs`` value. Action: ``merge_hierarchy`` for an exact
    dependency between two categorical columns where the left one has more
    values (sub_program -> program), ``derive`` for other exact ones, else
    ``review``.
    """
    categorical = set(categorical)
    rows = []
    for lhs, rhs in candidates:
        lhs_values = int(_as_text(train[lhs]).drop_duplicates().shape[0])
        per_value = len(train) / lhs_values
        if per_value < min_rows_per_value:
            continue
        train_error = g3_error(train, lhs, rhs)
        holdout_error = g3_error(holdout, lhs, rhs)
        if train_error > max_error or not holdout_error <= 2 * max_error:
            continue
        rhs_values = int(train[rhs].nunique(dropna=False))
        exact = train_error == 0 and holdout_error == 0
        if exact and len(lhs) == 1 and {lhs[0], rhs} <= categorical and lhs_values > rhs_values:
            action = MERGE_HIERARCHY
        elif exact:
            action = DERIVE
        else:
            action = REVIEW
        rows.append(
            {
                "kind": "fd",
                "lhs": "; ".join(map(str, lhs)),
                "rhs": str(rhs),
                "train_error": train_error,
                "holdout_error": holdout_error,
                "rows_per_lhs_value": per_value,
                "lhs_values": lhs_values,
                "rhs_values": rhs_values,
                "r2": np.nan,
                "action": action,
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def numeric_redundancy(
    train: pd.DataFrame, columns: list, min_r2: float = 0.95, seed: int = 0
) -> pd.DataFrame:
    """Continuous columns predicted from the other continuous ones with R² >= ``min_r2``.

    Linear regression on median-filled values, 5-fold cross-validated R²
    (``derive`` from 0.99, else ``review``); ``lhs`` lists up to three
    predictors with the largest standardized coefficients (at least a tenth of
    the largest).
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_score

    numeric = [c for c in columns if train[c].nunique() > 2]
    rows = []
    if len(numeric) < 2 or len(train) < 10:
        return pd.DataFrame(rows, columns=COLUMNS)
    filled = train[numeric].astype(float)
    filled = filled.fillna(filled.median())
    scaled = (filled - filled.mean()) / filled.std(ddof=0).replace(0, 1)
    folds = KFold(5, shuffle=True, random_state=seed)
    for column in numeric:
        x, y = scaled.drop(columns=[column]), scaled[column]
        r2 = float(np.mean(cross_val_score(LinearRegression(), x, y, cv=folds, scoring="r2")))
        if r2 < min_r2:
            continue
        weights = pd.Series(LinearRegression().fit(x, y).coef_, index=x.columns).abs()
        top = weights.nlargest(3)
        top = top[top >= 0.1 * top.iloc[0]]
        rows.append(
            {
                "kind": "numeric",
                "lhs": "; ".join(map(str, top.index)),
                "rhs": str(column),
                "train_error": np.nan,
                "holdout_error": np.nan,
                "rows_per_lhs_value": np.nan,
                "lhs_values": np.nan,
                "rhs_values": np.nan,
                "r2": r2,
                "action": DERIVE if r2 >= 0.99 else REVIEW,
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def ucc_rows(train: pd.DataFrame, uccs: list, max_columns: int = 3) -> pd.DataFrame:
    """Unique column combinations of up to ``max_columns`` columns (all rows singled out)."""
    rows = [
        {
            "kind": "ucc",
            "lhs": "; ".join(map(str, cols)),
            "rhs": "",
            "train_error": 0.0,
            "holdout_error": np.nan,
            "rows_per_lhs_value": 1.0,
            "lhs_values": len(train),
            "rhs_values": np.nan,
            "r2": np.nan,
            "action": PRIVACY_REVIEW,
        }
        for cols in uccs
        if len(cols) <= max_columns
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


def run_desbordante(
    train: pd.DataFrame,
    fd_columns: list,
    ucc_columns: list,
    env_dir: str | Path,
    max_lhs: int,
    error: float,
    seed: int,
) -> tuple[list, list]:
    """``([(lhs, rhs), ...], [ucc columns, ...])`` from Desbordante in its own environment."""
    from synthdata.generation.sidecar import run_in_env

    columns = list(dict.fromkeys([*fd_columns, *ucc_columns]))
    spec = {
        "fd_columns": list(map(str, fd_columns)),
        "ucc_columns": list(map(str, ucc_columns)),
        # Pyro's pair-based error is looser than g3; candidates are re-scored here.
        "error": error,
        "max_lhs": max_lhs,
        "seed": seed,
    }
    data = _as_text(train[columns])
    data.columns = data.columns.map(str)
    found = run_in_env(env_dir, "run_desbordante.py", data, spec, "desbordante")
    names = {str(c): c for c in columns}
    fds, uccs = [], []
    for row in found.itertuples():
        lhs = [names[c] for c in json.loads(row.lhs)]
        if row.kind == "afd" and lhs:
            fds.append((lhs, names[json.loads(row.rhs)[0]]))
        elif row.kind == "ucc":
            uccs.append(lhs)
    return fds, uccs


def save_dependencies_report(
    cfg,
    dataset,
    env_dir: str | Path = "envs/desbordante",
    max_lhs: int = 2,
    max_error: float = 0.01,
    min_rows_per_value: float = 5,
    min_r2: float = 0.95,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Write ``dependencies.csv`` (see the module docstring) and log a summary."""
    from synthdata.experiment import imputation_output_dir
    from synthdata.missing_blocks import detection_frame

    train = detection_frame(dataset).drop(columns=[dataset.target_column])
    holdout = dataset.test_df.reindex(columns=train.columns)
    if not dataset.indicator_only_values.empty:
        holdout[dataset.indicator_only_values.columns] = dataset.indicator_only_values.loc[
            dataset.test_df.index
        ]
    categorical = [*dataset.categorical_columns, *dataset.indicator_only_values.columns]
    categorical = [c for c in categorical if c in train.columns] + [
        c for c in train.columns if not pd.api.types.is_numeric_dtype(train[c])
    ]
    fd_columns = fd_candidate_columns(train, categorical)
    ucc_columns = [c for c in dataset.role_columns if c in train.columns]
    fds, uccs = run_desbordante(
        train, fd_columns, ucc_columns, env_dir, max_lhs, max_error, cfg.seed
    )
    continuous = [c for c in train.columns if c not in set(categorical)]
    table = pd.concat(
        [
            frame
            for frame in (
                score_fds(train, holdout, fds, categorical, max_error, min_rows_per_value),
                ucc_rows(train, uccs),
                numeric_redundancy(train, continuous, min_r2, cfg.seed),
            )
            if not frame.empty
        ]
        or [pd.DataFrame(columns=COLUMNS)],
        ignore_index=True,
    )
    path = Path(output_dir or imputation_output_dir(cfg)) / DEPENDENCIES_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    logger.info(
        "Dependencies: %d candidate FD(s) from Desbordante, %d kept; actions %s. Review %s.",
        len(fds),
        int((table["kind"] == "fd").sum()),
        table["action"].value_counts().to_dict(),
        path,
    )
    return table
