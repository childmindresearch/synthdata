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
SKIP_LOGIC = "skip_logic"

COLUMNS = [
    "kind",
    "lhs",
    "rhs",
    "support",
    "train_error",
    "holdout_error",
    "rows_per_lhs_value",
    "lhs_values",
    "rhs_values",
    "r2",
    "action",
    "check_rule",
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


def _label(value) -> str:
    """One text form per value: blanks are BLANK, 2 and 2.0 are both "2"."""
    if pd.isna(value):
        return BLANK
    if isinstance(value, (int, float, np.integer, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _as_text(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.apply(lambda column: column.map(_label)).astype(str)


def make_rule(lhs: list, rhs) -> str:
    """JSON rule: ``lhs`` and ``rhs`` as ``[column, value]``; a null value means "any".

    A functional dependency X -> Y has only nulls; a conditional one fixes some
    values, e.g. ``{"lhs": [["uses_substances", "no"]], "rhs": ["freq", "<NA>"]}``.
    """
    return json.dumps({"lhs": [list(item) for item in lhs], "rhs": list(rhs)})


def rule_error(frame: pd.DataFrame, rule: str | dict) -> tuple[int, float]:
    """Rows the rule applies to, and the share of them to remove for it to hold.

    The rule applies to rows matching every fixed left-side value. With a
    fixed right-side value the error is the share of those rows without it;
    otherwise it is the g3 error of "the "any" left-side columns fix the
    right-side column" within them (Kivinen & Mannila 1995). NaN when a
    column is absent or no row applies.
    """
    rule = json.loads(rule) if isinstance(rule, str) else rule
    lhs, (rhs, rhs_value) = rule["lhs"], rule["rhs"]
    columns = [c for c, _ in lhs] + [rhs]
    if frame.empty or any(c not in frame.columns for c in columns):
        return 0, float("nan")
    text = _as_text(frame[list(dict.fromkeys(columns))])
    applies = pd.Series(True, index=text.index)
    for column, value in lhs:
        if value is not None:
            applies &= text[column] == str(value)
    rows = text[applies]
    if rows.empty:
        return 0, float("nan")
    if rhs_value is not None:
        return len(rows), float((rows[rhs] != str(rhs_value)).mean())
    free = [c for c, value in lhs if value is None]
    if not free:
        return len(rows), float(1 - rows[rhs].value_counts().iloc[0] / len(rows))
    kept = rows.groupby([*free, rhs]).size().groupby(level=list(range(len(free)))).max()
    return len(rows), float(1 - kept.sum() / len(rows))


def g3_error(frame: pd.DataFrame, lhs: list, rhs: str) -> float:
    """Share of rows to remove so that ``lhs`` -> ``rhs`` holds exactly (blanks are a value)."""
    return rule_error(frame, make_rule([[c, None] for c in lhs], [rhs, None]))[1]


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
                "support": len(train),
                "train_error": train_error,
                "holdout_error": holdout_error,
                "rows_per_lhs_value": per_value,
                "lhs_values": lhs_values,
                "rhs_values": rhs_values,
                "r2": np.nan,
                "action": action,
                "check_rule": make_rule([[c, None] for c in lhs], [rhs, None]),
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def score_cfds(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    candidates: list,
    fds: list,
    max_error: float = 0.01,
    min_support: int = 20,
) -> pd.DataFrame:
    """Re-score conditional FDs ``(lhs pairs, rhs pair)``; keep those that hold on both splits.

    Dropped: rules without a fixed value (plain FDs, reported by
    :func:`score_fds`), rules implied by a kept FD ``fds`` (``sub = C1 ->
    major = C`` restates sub -> major), and rules applying to fewer than
    ``min_support`` train rows (5 holdout rows). Kept: error at most
    ``max_error`` on train and twice that on the holdout. Action
    ``skip_logic`` when the rule sets a blank (``uses_substances = no ->
    frequency blank``), else ``review``.
    """
    implied = [(set(lhs), rhs) for lhs, rhs in fds]
    rows, seen = [], set()
    for lhs, rhs in candidates:
        if all(value is None for _, value in [*lhs, rhs]):
            continue
        attributes = {c for c, _ in lhs}
        if any(rhs[0] == fd_rhs and fd_lhs <= attributes for fd_lhs, fd_rhs in implied):
            continue
        rule = make_rule(lhs, rhs)
        if rule in seen:
            continue
        seen.add(rule)
        support, train_error = rule_error(train, rule)
        holdout_support, holdout_error = rule_error(holdout, rule)
        if support < min_support or train_error > max_error:
            continue
        if holdout_support < 5 or not holdout_error <= 2 * max_error:
            continue
        rows.append(
            {
                "kind": "cfd",
                "lhs": _side(lhs),
                "rhs": _side([rhs]),
                "support": support,
                "train_error": train_error,
                "holdout_error": holdout_error,
                "rows_per_lhs_value": np.nan,
                "lhs_values": np.nan,
                "rhs_values": np.nan,
                "r2": np.nan,
                "action": SKIP_LOGIC if rhs[1] == BLANK else REVIEW,
                "check_rule": rule,
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
                "support": len(train),
                "check_rule": "",
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
            "support": len(train),
            "check_rule": "",
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
    cfd_minsup: int = 0,
) -> tuple[list, list, list]:
    """FDs ``[(lhs, rhs)]``, conditional FDs ``[(lhs pairs, rhs pair)]`` and UCCs from Desbordante.

    Runs in its own environment. Conditional FDs are searched on
    ``fd_columns`` when ``cfd_minsup`` (rows) is above 0.
    """
    from synthdata.generation.sidecar import run_in_env

    columns = list(dict.fromkeys([*fd_columns, *ucc_columns]))
    spec = {
        "fd_columns": list(map(str, fd_columns)),
        "ucc_columns": list(map(str, ucc_columns)),
        "cfd_columns": list(map(str, fd_columns)) if cfd_minsup > 0 else [],
        "cfd_minsup": int(cfd_minsup),
        "cfd_minconf": 1 - error,
        # Pyro's pair-based error is looser than g3; candidates are re-scored here.
        "error": error,
        "max_lhs": max_lhs,
        "seed": seed,
    }
    data = _as_text(train[columns])
    data.columns = data.columns.map(str)
    found = run_in_env(env_dir, "run_desbordante.py", data, spec, "desbordante")
    names = {str(c): c for c in columns}
    fds, cfds, uccs = [], [], []
    for row in found.itertuples():
        if row.kind == "cfd":
            lhs = [(names[c], value) for c, value in json.loads(row.lhs)]
            column, value = json.loads(row.rhs)
            if lhs:
                cfds.append((lhs, (names[column], value)))
            continue
        lhs = [names[c] for c in json.loads(row.lhs)]
        if row.kind == "afd" and lhs:
            fds.append((lhs, names[json.loads(row.rhs)[0]]))
        elif row.kind == "ucc":
            uccs.append(lhs)
    return fds, cfds, uccs


def dependencies_table(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    categorical: list,
    ucc_columns: list,
    env_dir: str | Path = "envs/desbordante",
    max_lhs: int = 2,
    max_error: float = 0.01,
    min_rows_per_value: float = 5,
    min_r2: float = 0.95,
    cfd_min_support: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Every finding on ``train`` (re-scored on ``holdout``), one row each (see the module docstring).

    ``cfd_min_support`` is the smallest share of train rows (at least 20 rows)
    a conditional FD must apply to; 0 skips the conditional search.
    """
    fd_columns = fd_candidate_columns(train, categorical)
    min_support = max(20, int(cfd_min_support * len(train))) if cfd_min_support > 0 else 0
    fds, cfds, uccs = run_desbordante(
        train, fd_columns, ucc_columns, env_dir, max_lhs, max_error, seed, min_support
    )
    fd_table = score_fds(train, holdout, fds, categorical, max_error, min_rows_per_value)
    kept = set(fd_table["check_rule"])
    kept_fds = [
        (lhs, rhs) for lhs, rhs in fds if make_rule([[c, None] for c in lhs], [rhs, None]) in kept
    ]
    continuous = [c for c in train.columns if c not in set(categorical)]
    frames = [
        fd_table,
        score_cfds(train, holdout, cfds, kept_fds, max_error, max(min_support, 20)),
        ucc_rows(train, uccs),
        numeric_redundancy(train, continuous, min_r2, seed),
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    logger.info(
        "Dependencies: Desbordante found %d FD(s) and %d conditional FD(s); %d and %d kept.",
        len(fds),
        len(cfds),
        len(fd_table),
        int(sum((f["kind"] == "cfd").sum() for f in frames)),
    )
    return pd.concat(frames, ignore_index=True)


def save_dependencies_report(
    cfg, dataset, output_dir: str | Path | None = None, **options
) -> pd.DataFrame:
    """Write ``dependencies.csv`` for ``dataset``; ``options`` go to :func:`dependencies_table`."""
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
    ucc_columns = [c for c in dataset.role_columns if c in train.columns]
    options.setdefault("seed", cfg.seed)
    table = dependencies_table(train, holdout, categorical, ucc_columns, **options)
    path = Path(output_dir or imputation_output_dir(cfg)) / DEPENDENCIES_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    logger.info(
        "Dependencies: actions %s. Review %s.", table["action"].value_counts().to_dict(), path
    )
    return table


#: Evaluation: a model fails the rules check when its mean rule error is more
#: than this above the real holdout's.
RULE_ERROR_MARGIN = 0.01


def _side(items) -> str:
    return "; ".join(f"{c}={'*' if v is None else v}" for c, v in items)


def load_rules(path: str | Path) -> pd.DataFrame:
    """Accepted rules from a ``dependencies.csv`` or ``missing_blocks.csv``-style file.

    Every row with a ``check_rule`` counts (a cell may hold one rule or a JSON list
    of them, as block rows do); delete the rows you don't accept, or add an
    ``accept`` column and set it to false for them. Returns one row per rule
    with ``kind``, ``lhs``, ``rhs`` (as text) and ``rule`` (JSON).
    """
    table = pd.read_csv(path, keep_default_na=False)
    if "check_rule" not in table.columns:
        raise ValueError(
            f"{path} has no 'check_rule' column; use a dependencies or missing-blocks file"
        )
    if "accept" in table.columns:
        accept = table["accept"].astype(str).str.strip().str.lower()
        table = table[accept.isin(["true", "1", "yes", "y"])]
    rows = []
    for kind, cell in zip(table["kind"], table["check_rule"], strict=True):
        if not str(cell).strip():
            continue
        parsed = json.loads(cell)
        for rule in parsed if isinstance(parsed, list) else [parsed]:
            rows.append(
                {
                    "kind": kind,
                    "lhs": _side(rule["lhs"]),
                    "rhs": _side([rule["rhs"]]),
                    "rule": json.dumps(rule),
                }
            )
    return pd.DataFrame(rows, columns=["kind", "lhs", "rhs", "rule"])


def rule_violations(rules: pd.DataFrame, frames: dict) -> pd.DataFrame:
    """Rule error (share of applicable rows breaking it) of each rule in each frame.

    One row per rule (``kind``, ``lhs``, ``rhs``), one column per frame; NaN
    where the frame lacks a column or no row applies.
    """
    out = rules[["kind", "lhs", "rhs"]].copy()
    for name, frame in frames.items():
        out[name] = [rule_error(frame, rule)[1] for rule in rules["rule"]]
    return out


def rule_check_rows(violations: pd.DataFrame, reference: str, models: list) -> list:
    """``checks.csv`` rows: each model's mean rule error against the reference's plus a margin."""
    from synthdata.evaluation.checks import _row

    limit = float(np.nanmean(violations[reference])) + RULE_ERROR_MARGIN
    rows = []
    for model in models:
        value = float(np.nanmean(violations[model])) if violations[model].notna().any() else np.nan
        worst = (
            violations.loc[violations[model].idxmax()] if violations[model].notna().any() else None
        )
        detail = (
            f"mean share of rows breaking {int(violations[model].notna().sum())} accepted rules; "
            f"worst: {worst['lhs']} -> {worst['rhs']} ({worst[model]:.1%})"
            if worst is not None
            else "no accepted rule applies"
        )
        rows.append(_row("dependency_rules", model, not value > limit, value, limit, detail))
    return rows


def evaluate_rules(cfg, dataset, synthetic_datasets: dict, output_dir: str | Path) -> list:
    """Score the accepted rules (``evaluation.dependency_rules``) on each synthetic dataset.

    Synthetic data is compared as released (blanked where its missing flags
    say so), with the real holdout as the reference. Writes
    ``dependency_violations.csv`` and returns ``checks.csv`` rows.
    """
    from synthdata.data import remask_synthetic
    from synthdata.evaluation.baselines import is_baseline

    rules = load_rules(cfg.evaluation.dependency_rules)
    if rules.empty:
        logger.warning("No accepted rules in %s", cfg.evaluation.dependency_rules)
        return []
    holdout = dataset.test_df
    if not dataset.indicator_only_values.empty:
        holdout = holdout.join(dataset.indicator_only_values.loc[holdout.index])
    frames = {"holdout": holdout}
    for name, synthetic in sorted(synthetic_datasets.items()):
        frames[name] = remask_synthetic(
            synthetic, dataset.missing_indicator_columns, dataset.missing_block_columns
        )
    violations = rule_violations(rules, frames)
    violations.to_csv(Path(output_dir) / "dependency_violations.csv", index=False)
    models = [name for name in sorted(synthetic_datasets) if not is_baseline(name)]
    return rule_check_rows(violations, "holdout", models)
