"""Redundant columns and hierarchies: an advisory report to help slim the dataset.

Every finding is computed exactly and kept in full; thresholds only decide
which rows the readable ``dependencies.csv`` lists and the suggested action.

* **Functional dependencies** X -> Y between any two columns (and, opt-in,
  two-column X): knowing X fixes Y, up to the g3 error (share of rows to
  remove for it to hold exactly; Kivinen & Mannila 1995), on the
  indicator-fit rows and on the holdout, with a chance-corrected score so
  that near-unique X does not pass as a rule. See
  :mod:`synthdata.dependency_search`.
* **One-condition rules** X = v -> Y = w, which include skip logic
  (``use=no -> freq=<NA>``), kept when significant after a Bonferroni
  correction.
* **Unique column combinations** among the role columns (quasi-identifiers,
  sensitive, protected): column sets that single out every row. A privacy
  finding, found by Desbordante's HyUCC in its own environment
  (``envs/desbordante``; Desbordante is AGPL-3.0, so synthdata never imports it).
* **Numeric redundancy**: every continuous column's cross-validated R² from a
  linear model on the other continuous columns, such as a score that is the
  sum of its items.

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
ID_LIKE = "id_like"
NO_ACTION = "none"

COLUMNS = [
    "kind",
    "lhs",
    "rhs",
    "support",
    "train_error",
    "holdout_error",
    "score",
    "p_value",
    "lhs_values",
    "rhs_values",
    "r2",
    "action",
    "check_rule",
]


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


def fd_rows(
    pairs: pd.DataFrame, categorical: list, max_error: float = 0.01, min_score: float = 0.5
) -> pd.DataFrame:
    """``fd`` report rows: every scored pair (see ``dependency_search``) within ``max_error``.

    ``max_error`` only decides what is listed here (the all-pairs file has the
    rest). Actions: ``id_like`` when the chance-corrected score is below
    ``min_score`` (X has so many values that it fixes Y by accident);
    otherwise, when exact on train and holdout, ``merge_hierarchy`` for two
    categorical columns where X has more values (sub_program -> program) and
    ``derive`` for the rest; else ``review``.
    """
    categorical = {str(c) for c in categorical}
    rows = []
    listed = pairs[pairs["train_error"] <= max_error].sort_values(
        ["score", "train_error"], ascending=[False, True]
    )
    for row in listed.itertuples():
        lhs = row.lhs.split("; ")
        exact = row.train_error == 0 and row.holdout_error == 0
        if not row.score >= min_score:
            action = ID_LIKE
        elif (
            exact
            and len(lhs) == 1
            and {lhs[0], row.rhs} <= categorical
            and (row.lhs_values > row.rhs_values)
        ):
            action = MERGE_HIERARCHY
        elif exact:
            action = DERIVE
        else:
            action = REVIEW
        rows.append(
            {
                "kind": "fd",
                "lhs": row.lhs,
                "rhs": row.rhs,
                "support": np.nan,
                "train_error": row.train_error,
                "holdout_error": row.holdout_error,
                "score": row.score,
                "p_value": np.nan,
                "lhs_values": row.lhs_values,
                "rhs_values": row.rhs_values,
                "r2": np.nan,
                "action": action,
                "check_rule": make_rule([[c, None] for c in lhs], [row.rhs, None]),
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def condition_rows(
    rules: pd.DataFrame, pairs: pd.DataFrame, max_error: float = 0.01
) -> pd.DataFrame:
    """``cfd`` report rows: one-condition rules with confidence at least 1 - ``max_error``.

    ``rules`` holds only rules that passed the significance test (see
    ``dependency_search``); ``max_error`` only decides what is listed here.
    Rules of an X -> Y pair that is itself listed as an ``fd`` row are left
    out (they restate it; the rules file has them). Action ``skip_logic``
    when the rule blanks Y on an answered X (``use=no -> freq=<NA>``), else
    ``review``. Blank -> blank rules (columns missing together) stay in the
    rules file only: they number in the tens of thousands on wide data, and
    the missing-blocks report groups them into blocks.
    """
    within = pairs[pairs["train_error"] <= max_error]
    listed_fds = set(zip(within["lhs"], within["rhs"], strict=True))
    listed = rules[rules["confidence"] >= 1 - max_error]
    keep = [pair not in listed_fds for pair in zip(listed["lhs"], listed["rhs"], strict=True)]
    listed = listed[keep]
    # Blank -> blank rules restate co-missingness, which the blocks report groups.
    listed = listed[~((listed["lhs_value"] == BLANK) & (listed["rhs_value"] == BLANK))]
    rows = []
    for row in listed.sort_values(["rhs", "support"], ascending=[True, False]).itertuples():
        action = SKIP_LOGIC if row.rhs_value == BLANK else REVIEW
        rows.append(
            {
                "kind": "cfd",
                "lhs": _side([(row.lhs, row.lhs_value)]),
                "rhs": _side([(row.rhs, row.rhs_value)]),
                "support": row.support,
                "train_error": 1 - row.confidence,
                "holdout_error": 1 - row.holdout_confidence,
                "score": np.nan,
                "p_value": row.p_value,
                "lhs_values": np.nan,
                "rhs_values": np.nan,
                "r2": np.nan,
                "action": action,
                "check_rule": make_rule([[row.lhs, row.lhs_value]], [row.rhs, row.rhs_value]),
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def numeric_redundancy(
    train: pd.DataFrame, columns: list, min_r2: float = 0.95, seed: int = 0
) -> pd.DataFrame:
    """Every continuous column's cross-validated R² from the other continuous ones.

    Linear regression on median-filled values, 5-fold cross-validated R².
    ``min_r2`` only sets the action: ``derive`` from 0.99, ``review`` from
    ``min_r2``, else ``none``. ``lhs`` lists up to three predictors with the
    largest standardized coefficients (at least a tenth of the largest).
    Linear only: a nonlinear formula can score lower than it should.
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
                "score": np.nan,
                "p_value": np.nan,
                "lhs_values": np.nan,
                "rhs_values": np.nan,
                "r2": r2,
                "action": DERIVE if r2 >= 0.99 else REVIEW if r2 >= min_r2 else NO_ACTION,
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS).sort_values("r2", ascending=False)


def ucc_rows(train: pd.DataFrame, uccs: list) -> pd.DataFrame:
    """Every minimal unique column combination (all rows singled out)."""
    rows = [
        {
            "kind": "ucc",
            "support": len(train),
            "check_rule": "",
            "lhs": "; ".join(map(str, cols)),
            "rhs": "",
            "train_error": 0.0,
            "holdout_error": np.nan,
            "score": np.nan,
            "p_value": np.nan,
            "lhs_values": len(train),
            "rhs_values": np.nan,
            "r2": np.nan,
            "action": PRIVACY_REVIEW,
        }
        for cols in uccs
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


def run_hyucc(train: pd.DataFrame, ucc_columns: list, env_dir: str | Path) -> list:
    """All minimal unique column combinations of ``ucc_columns``, from Desbordante's HyUCC.

    Runs in its own environment (Desbordante is AGPL-3.0). HyUCC is exact and
    has no size limit.
    """
    from synthdata.generation.sidecar import run_in_env

    if not ucc_columns:
        return []
    data = _as_text(train[ucc_columns])
    data.columns = data.columns.map(str)
    spec = {"ucc_columns": list(data.columns)}
    found = run_in_env(env_dir, "run_desbordante.py", data, spec, "desbordante")
    names = {str(c): c for c in ucc_columns}
    return [[names[c] for c in json.loads(lhs)] for lhs in found["lhs"]]


def dependencies_table(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    categorical: list,
    ucc_columns: list,
    env_dir: str | Path = "envs/desbordante",
    max_lhs: int = 1,
    max_error: float = 0.01,
    min_r2: float = 0.95,
    alpha: float = 0.05,
    permutations: int = 5,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """The report (see the module docstring) and the full tables behind it.

    Returns ``(report, tables)``, where ``tables`` has ``pairs`` (every
    ordered column pair), ``rules`` (every significant one-condition rule)
    and, with ``max_lhs`` 2, ``pairs2`` (two-column rules within
    ``max_error``).
    """
    from synthdata.dependency_search import single_column_search, two_column_search

    if max_lhs not in (1, 2):
        raise ValueError(
            f"max_lhs must be 1 or 2, not {max_lhs}: three or more determining columns "
            "cannot be searched exhaustively on wide data"
        )
    train_text, holdout_text = _as_text(train), _as_text(holdout)
    pairs, rules = single_column_search(train_text, holdout_text, permutations, alpha, seed)
    tables = {"pairs": pairs, "rules": rules}
    scored = pairs
    if max_lhs == 2:
        tables["pairs2"] = two_column_search(
            train_text, holdout_text, pairs, max_error, permutations, seed
        )
        scored = pd.concat([pairs, tables["pairs2"]], ignore_index=True)
    continuous = [c for c in train.columns if c not in set(categorical)]
    frames = [
        fd_rows(scored, categorical, max_error),
        condition_rows(rules, pairs, max_error),
        ucc_rows(train, run_hyucc(train, ucc_columns, env_dir)),
        numeric_redundancy(train, continuous, min_r2, seed),
    ]
    frames = [f for f in frames if not f.empty]
    report = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    return report, tables


def save_dependencies_report(
    cfg, dataset, output_dir: str | Path | None = None, **options
) -> pd.DataFrame:
    """Write ``dependencies.csv`` and its full tables for ``dataset``.

    ``options`` go to :func:`dependencies_table`. The full tables are
    ``dependencies_pairs.parquet``, ``dependencies_rules.parquet`` and, with
    two determining columns, ``dependencies_pairs2.parquet``.
    """
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
    table, tables = dependencies_table(train, holdout, categorical, ucc_columns, **options)
    folder = Path(output_dir or imputation_output_dir(cfg))
    folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(folder / DEPENDENCIES_FILENAME, index=False)
    for name, frame in tables.items():
        frame.to_parquet(folder / f"dependencies_{name}.parquet", index=False)
    logger.info(
        "Dependencies: %s. Review %s; every pair and rule is in dependencies_*.parquet.",
        table.groupby(["kind", "action"]).size().to_dict(),
        folder / DEPENDENCIES_FILENAME,
    )
    return table


def add_report_arguments(parser) -> None:
    """The dependency-report options shared by ``synthdata-dependencies`` and ``synthdata-profile``."""
    parser.add_argument("--env-dir", default="envs/desbordante", help="Desbordante environment.")
    parser.add_argument(
        "--max-lhs",
        type=int,
        choices=[1, 2],
        default=1,
        help="Determining columns per rule; 2 scores every column pair against every column "
        "(hours on hundreds of columns).",
    )
    parser.add_argument(
        "--max-error",
        type=float,
        default=0.01,
        help="Rules within this error are listed in dependencies.csv (all are in the parquet files).",
    )
    parser.add_argument(
        "--min-r2", type=float, default=0.95, help="R² from which a numeric column gets 'review'."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for one-condition rules (Bonferroni-corrected).",
    )
    parser.add_argument(
        "--permutations", type=int, default=5, help="Shuffles for the chance-corrected score."
    )


def report_options(args) -> dict:
    """Keyword options for :func:`save_dependencies_report` from parsed arguments."""
    names = ["env_dir", "max_lhs", "max_error", "min_r2", "alpha", "permutations"]
    return {name: getattr(args, name) for name in names}


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


COMPARISON_FILENAME = "skip_logic_comparison.csv"


def compare_skip_logic(blocks: pd.DataFrame, dependencies: pd.DataFrame) -> pd.DataFrame:
    """Skip logic as found by the missing-blocks tree and by conditional FDs, side by side.

    One row per missing-blocks pattern labelled ``skip_logic_candidate``, with
    the conditional FDs (``skip_logic`` rows) that blank any of its columns,
    then one row per such conditional FD that matches no candidate (it may
    sit in a pattern with another label, shown in ``blocks_label``, or in no
    pattern). ``found_by`` is "both", "blocks only" or "cfd only".
    """
    from synthdata.missing_blocks import SKIP_LOGIC as BLOCKS_SKIP_LOGIC

    cfds = dependencies[dependencies["action"] == SKIP_LOGIC].copy()
    cfds["column"] = cfds["rhs"].str.rsplit("=", n=1).str[0]
    cfds["text"] = cfds["lhs"] + " -> " + cfds["rhs"]
    label_of = {}
    for row in blocks.itertuples():
        for column in row.columns.split("; "):
            label_of[column] = (row.pattern, row.label)
    rows, matched = [], set()
    for row in blocks[blocks["label"] == BLOCKS_SKIP_LOGIC].itertuples():
        members = set(row.columns.split("; "))
        hits = cfds[cfds["column"].isin(members)]
        matched |= set(hits.index)
        rows.append(
            {
                "columns": row.columns,
                "blocks_pattern": row.pattern,
                "blocks_label": row.label,
                "tree_rule": row.rule,
                "rule_auc": row.rule_auc,
                "cfd_rules": " | ".join(hits["text"]),
                "found_by": "both" if len(hits) else "blocks only",
            }
        )
    for _, row in cfds.drop(index=list(matched)).iterrows():
        pattern, label = label_of.get(row["column"], ("", ""))
        rows.append(
            {
                "columns": row["column"],
                "blocks_pattern": pattern,
                "blocks_label": label,
                "tree_rule": "",
                "rule_auc": np.nan,
                "cfd_rules": row["text"],
                "found_by": "cfd only",
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "columns",
            "blocks_pattern",
            "blocks_label",
            "tree_rule",
            "rule_auc",
            "cfd_rules",
            "found_by",
        ],
    )
