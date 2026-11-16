"""Blocks of columns that go missing together, and what kind of missingness each looks like.

Clinical data often loses whole forms at once (an assessment never given), or
skips items on purpose (skip logic: "no substance use" skips the frequency
items). Such gaps are "not assessed" rather than unknown, and imputing them
invents values that were never there. This module finds the patterns and
labels each one as a *candidate* for a person to confirm against the codebook;
the data alone cannot prove why a value is missing.

* :func:`find_blocks` groups columns whose missing cells are nearly the same
  rows (Jaccard similarity of the missing masks, average-linkage clustering).
* :func:`describe_patterns` reports every block and every other frequently
  missing column: how often it is missing, how often a block is missing as a
  whole, and how well a depth-2 decision tree on the other columns predicts
  the missingness (a near-perfect tree suggests an eligibility or skip rule,
  and its splits name the rule).
* ``missing_indicators.blocks`` (see ``synthdata.data.add_missing_indicators``)
  gives each block one ``<block>__missing`` flag instead of one per column.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from synthdata.utils import get_logger

logger = get_logger(__name__)

#: File written next to ``data_readiness.csv`` in the imputation output folder.
BLOCKS_FILENAME = "missing_blocks.csv"
#: Labels, in the order they are checked.
SKIP_LOGIC = "skip_logic_candidate"
FORM_NOT_GIVEN = "form_not_given_candidate"
ITEM_MISSING = "item_missing"


def find_blocks(missing: pd.DataFrame, min_jaccard: float = 0.9) -> list[list[str]]:
    """Groups of two or more columns whose missing masks overlap by ``min_jaccard``.

    ``missing`` is a boolean frame (True = missing). Columns are clustered by
    average linkage on 1 - Jaccard similarity and cut at ``1 - min_jaccard``,
    so on average every pair in a block is missing in nearly the same rows.
    Blocks are returned largest first, each in the frame's column order.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    columns = [c for c in missing.columns if missing[c].any()]
    if len(columns) < 2:
        return []
    matrix = missing[columns].to_numpy(dtype=bool).T
    distances = pdist(matrix, metric="jaccard")
    labels = fcluster(linkage(distances, method="average"), 1 - min_jaccard, "distance")
    groups: dict[int, list[str]] = {}
    for column, label in zip(columns, labels, strict=True):
        groups.setdefault(int(label), []).append(column)
    blocks = [g for g in groups.values() if len(g) >= 2]
    order = {c: i for i, c in enumerate(missing.columns)}
    blocks = [sorted(b, key=order.__getitem__) for b in blocks]
    return sorted(blocks, key=lambda b: (-len(b), order[b[0]]))


def block_names(blocks: list[list[str]]) -> dict[str, list[str]]:
    """``{"block_1": [...], ...}`` for automatically found blocks."""
    return {f"block_{i}": block for i, block in enumerate(blocks, start=1)}


def _encode(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list]]:
    """Numeric copy for a decision tree, and each text column's values by code.

    Text becomes codes 0, 1, ... in order of appearance; blanks stay NaN.
    """
    out, levels = {}, {}
    for column in frame.columns:
        values = frame[column]
        if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            out[column] = values.astype(float)
        else:
            codes, uniques = pd.factorize(values)
            out[column] = pd.Series(codes, index=frame.index, dtype=float).replace(-1, np.nan)
            levels[str(column)] = list(uniques)
    return pd.DataFrame(out, index=frame.index), levels


def _number(value: float) -> str:
    """A threshold as a person would write it: whole numbers without decimals."""
    return str(int(value)) if float(value).is_integer() else f"{value:.4g}"


def _value_set(column: str, values: list, others: list, max_listed: int = 4) -> str:
    """``column = a``, ``column in {a, b}`` or, when shorter, ``column not in {c}``.

    ``others`` are the column's remaining values. Long lists are cut after
    ``max_listed`` values.
    """
    shown = [str(v) for v in values]
    if len(shown) == 1:
        return f"{column} = {shown[0]}"
    if len(shown) > max_listed and others and len(others) < len(shown):
        if len(others) == 1:
            return f"{column} is not {others[0]}"
        return _value_set(column, others, [], max_listed).replace(" in {", " not in {", 1)
    if len(shown) > max_listed:
        shown = [*shown[:max_listed], f"... ({len(values)} values)"]
    return f"{column} in {{{', '.join(shown)}}}"


def _condition(column: str, splits: list, x: pd.DataFrame, levels: dict, blank: bool) -> str:
    """Plain text for every split on ``column`` along one path of the tree.

    ``splits`` holds ``(threshold, left)`` pairs. Text columns are named by
    their values, 0/1 columns by their value, whole-number columns by whole
    bounds and a split on blanks alone (threshold infinity) as answered/blank.
    ``blank`` adds "(... or blank)" when blank rows of the column reach the leaf:
    the tree sends them down whichever branch fits best, and plain
    thresholds would hide where they went.
    """
    if any(np.isinf(t) for t, _ in splits):
        return f"{column} answered" if all(left for _, left in splits) else f"{column} blank"
    if blank:
        return f"({_condition(column, splits, x, levels, False)} or blank)"
    if column in levels:
        values = levels[column]
        keep = set(range(len(values)))
        for threshold, left in splits:
            cut = int(np.floor(threshold))
            keep &= {i for i in keep if (i <= cut) == left}
        others = [values[i] for i in range(len(values)) if i not in keep]
        return _value_set(column, [values[i] for i in sorted(keep)], others)
    observed = x[column].dropna()
    whole = bool(len(observed)) and bool(np.all(np.mod(observed, 1) == 0))
    upper = min((t for t, left in splits if left), default=np.inf)
    lower = max((t for t, left in splits if not left), default=-np.inf)
    if whole and set(observed.unique()) <= {0, 1} and (np.isinf(upper) or np.isinf(lower)):
        return f"{column} = {0 if np.isfinite(upper) else 1}"
    if whole:
        low = None if np.isinf(lower) else int(np.floor(lower)) + 1
        high = None if np.isinf(upper) else int(np.floor(upper))
        if low is not None and high is not None:
            text = f"{column} = {low}" if low == high else f"{column} {low} to {high}"
        elif high is not None:
            text = f"{column} <= {high}"
        else:
            text = f"{column} >= {low}"
        return text
    if np.isfinite(lower) and np.isfinite(upper):
        return f"{_number(lower)} < {column} <= {_number(upper)}"
    if np.isfinite(upper):
        return f"{column} <= {_number(upper)}"
    return f"{column} > {_number(lower)}"


def _tree_leaves(tree, x: pd.DataFrame, y: np.ndarray, levels: dict) -> list[dict]:
    """Each leaf of a fitted tree: its condition in words and its rows.

    Counts come from the rows themselves (``tree.apply``), so they are exact
    whatever the scikit-learn version stores in the tree.
    """
    inner = tree.tree_
    names = [str(c) for c in x.columns]
    rows_at = pd.Series(tree.apply(x.to_numpy()), index=x.index)
    leaves = []

    def walk(node: int, path: dict) -> None:
        left, right = inner.children_left[node], inner.children_right[node]
        if left == right:  # a leaf
            mask = (rows_at == node).to_numpy()
            parts = [
                _condition(c, s, x, levels, bool(x.loc[mask, c].isna().any()))
                for c, s in path.items()
            ]
            leaves.append(
                {
                    "condition": " and ".join(parts) or "all rows",
                    "rows": int(mask.sum()),
                    "missing": int(y[mask].sum()),
                }
            )
            return
        column, threshold = names[inner.feature[node]], float(inner.threshold[node])
        for child, goes_left in ((left, True), (right, False)):
            walk(child, {**path, column: [*path.get(column, []), (threshold, goes_left)]})

    walk(0, {})
    return leaves


def describe_leaves(leaves: list[dict]) -> str:
    """The leaves as text, most-missing first, separated by `` | ``.

    Each reads ``<condition>: <missing> of <rows> rows missing (<share>%)``,
    e.g. ``visit_year <= 2019 and stay_days <= 30: 412 of 420 rows
    missing (98%)``.
    """
    ordered = sorted(leaves, key=lambda leaf: (-leaf["missing"] / leaf["rows"], -leaf["rows"]))
    return " | ".join(
        f"{leaf['condition']}: {leaf['missing']} of {leaf['rows']} rows missing "
        f"({100 * leaf['missing'] / leaf['rows']:.0f}%)"
        for leaf in ordered
    )


def missingness_rule(
    predictors: pd.DataFrame, missing: pd.Series, seed: int = 0
) -> tuple[float, str, list[dict]]:
    """How well a depth-2 tree on ``predictors`` predicts ``missing``, and its rule.

    Returns the 5-fold cross-validated ROC AUC, the tree's leaves as text (see
    :func:`describe_leaves`) and the leaves themselves. The AUC is NaN, and
    the rule empty, when there are too few rows of either kind to
    cross-validate.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.tree import DecisionTreeClassifier

    y = missing.astype(int).to_numpy()
    if predictors.empty or min(y.sum(), len(y) - y.sum()) < 5:
        return float("nan"), "", []
    x, levels = _encode(predictors)
    tree = DecisionTreeClassifier(max_depth=2, min_samples_leaf=5, random_state=seed)
    folds = StratifiedKFold(5, shuffle=True, random_state=seed)
    auc = float(np.mean(cross_val_score(tree, x.to_numpy(), y, cv=folds, scoring="roc_auc")))
    tree.fit(x.to_numpy(), y)
    leaves = _tree_leaves(tree, x, y, levels)
    return auc, describe_leaves(leaves), leaves


def summarize(
    label: str, auc: float, whole: float, leaves: list[dict], rule_auc: float = 0.95
) -> str:
    """A plain-language reading of one pattern: what it looks like and where it sits."""
    total = sum(leaf["missing"] for leaf in leaves)
    where = ""
    if leaves and total:
        top = max(leaves, key=lambda leaf: (leaf["missing"], -leaf["rows"]))
        where = (
            f" Most gaps ({top['missing']} of {total}) are where {top['condition']}, "
            f"which is missing in {100 * top['missing'] / top['rows']:.0f}% of its rows."
        )
    if np.isnan(auc):
        return "Too few missing (or too few answered) rows to look for a rule."
    if label == SKIP_LOGIC:
        return (
            f"Other columns predict these gaps well (AUC {auc:.2f}): likely a skip or "
            f"eligibility rule.{where} Check it against the codebook."
        )
    if label == FORM_NOT_GIVEN:
        return (
            f"The columns go missing together ({100 * whole:.0f}% of gap rows miss all of "
            f"them) without a clear rule (AUC {auc:.2f}, below {rule_auc:.2f}): likely a "
            f"form or section not given.{where}"
        )
    return (
        f"Other columns predict these gaps with AUC {auc:.2f}, below the {rule_auc:.2f} "
        f"that suggests a rule: looks like ordinary missing answers, unless the split "
        f"below makes sense clinically.{where}"
    )


def block_rules(columns: list) -> str:
    """JSON list of rules "first column blank -> each other column blank".

    In the format of ``synthdata.dependencies.make_rule``, so
    ``evaluation.dependency_rules`` can check that synthetic data blanks the
    block together.
    """
    from synthdata.dependencies import BLANK

    first, *rest = columns
    return json.dumps([{"lhs": [[first, BLANK]], "rhs": [c, BLANK]} for c in rest])


def describe_patterns(
    frame: pd.DataFrame,
    columns: list,
    min_missing_fraction: float = 0.05,
    min_jaccard: float = 0.9,
    rule_auc: float = 0.95,
    whole_block_share: float = 0.8,
    exclude_from_rules: list | None = None,
    exclude_from_blocks: list | None = None,
    min_observed_share: float = 0.9,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per block and per other column missing in ``min_missing_fraction`` of rows.

    Columns: ``pattern``, ``kind`` (block or column), ``label`` (below),
    ``summary`` (see :func:`summarize`), ``columns``, ``n_columns``,
    ``missing_rows`` and ``missing_share`` (rows missing any of its columns),
    ``whole_share`` (of those, rows missing all of them), ``rule_auc`` and
    ``rule`` (see :func:`missingness_rule`), ``indicator_only`` (the column
    is in ``exclude_from_blocks``, so it stays out of blocks; the pipeline
    keeps only its flag) and ``check_rule``. The rule's predictors leave out the
    pattern's own columns, ``exclude_from_rules`` (e.g. the target) and every
    column observed in fewer than ``min_observed_share`` of the pattern's
    missing rows: a gate question is answered when its items are skipped,
    whereas another form missing in the same rows would only restate that the
    rows are missing together. Labels:

    * ``skip_logic_candidate``: the tree predicts the missingness with AUC at
      least ``rule_auc``, so a rule on other columns may decide it.
    * ``form_not_given_candidate``: a block missing as a whole in at least
      ``whole_block_share`` of its missing rows, with no clear rule.
    * ``item_missing``: neither; looks like ordinary missing answers.

    Block rows also carry ``check_rule`` (see :func:`block_rules`), for
    ``evaluation.dependency_rules``.
    """
    shares = frame[columns].isna().mean()
    candidates = [c for c in columns if shares[c] > 0 and shares[c] >= min_missing_fraction]
    no_blocks = set(exclude_from_blocks or [])
    blocks = find_blocks(frame[[c for c in candidates if c not in no_blocks]].isna(), min_jaccard)
    in_block = {c for block in blocks for c in block}
    patterns = [("block", b) for b in blocks]
    patterns += [("column", [c]) for c in candidates if c not in in_block]
    excluded = set(exclude_from_rules or [])
    rows = []
    for i, (kind, members) in enumerate(patterns, start=1):
        masks = frame[members].isna()
        any_missing, all_missing = masks.any(axis=1), masks.all(axis=1)
        whole = float(all_missing.sum() / any_missing.sum()) if any_missing.any() else 0.0
        predictors = frame.drop(columns=[c for c in [*members, *excluded] if c in frame])
        observed = predictors.loc[any_missing].notna().mean()
        predictors = predictors.loc[:, observed >= min_observed_share]
        auc, rule, leaves = missingness_rule(predictors, any_missing, seed)
        if auc >= rule_auc:
            label = SKIP_LOGIC
        elif kind == "block" and whole >= whole_block_share:
            label = FORM_NOT_GIVEN
        else:
            label = ITEM_MISSING
        rows.append(
            {
                "pattern": f"block_{i}" if kind == "block" else members[0],
                "kind": kind,
                "label": label,
                "summary": summarize(label, auc, whole, leaves, rule_auc),
                "columns": "; ".join(map(str, members)),
                "n_columns": len(members),
                "missing_rows": int(any_missing.sum()),
                "missing_share": round(float(any_missing.mean()), 3),
                "whole_share": round(whole, 3),
                "rule_auc": round(auc, 3),
                "rule": rule,
                "indicator_only": kind == "column" and members[0] in no_blocks,
                "check_rule": block_rules(members) if kind == "block" else "",
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "pattern",
            "kind",
            "label",
            "summary",
            "columns",
            "n_columns",
            "missing_rows",
            "missing_share",
            "whole_share",
            "rule_auc",
            "rule",
            "indicator_only",
            "check_rule",
        ],
    )


def detection_frame(dataset) -> pd.DataFrame:
    """Raw feature values (indicator-only columns included) on the indicator-fit rows.

    These are train minus tuning, the rows the missing indicators are chosen
    on, so no held-out row shapes the report.
    """
    indicators = set(dataset.missing_indicator_columns) | set(dataset.missing_block_columns)
    rows = dataset.search_train_df.index
    features = [c for c in dataset.feature_columns if c not in indicators]
    frame = dataset.full_df.loc[rows, [*features, dataset.target_column]]
    if not dataset.indicator_only_values.empty:
        frame = frame.join(dataset.indicator_only_values.loc[rows])
    return frame


def save_blocks_report(cfg, dataset, output_dir: str | Path | None = None) -> pd.DataFrame:
    """Write ``missing_blocks.csv`` and log a summary; returns the table."""
    from synthdata.experiment import imputation_output_dir

    frame = detection_frame(dataset)
    columns = [c for c in frame.columns if c != dataset.target_column]
    indicators = cfg.imputation.missing_indicators
    table = describe_patterns(
        frame,
        columns,
        # Every column with a gap: a rare pattern can still be skip logic.
        min_missing_fraction=0.0,
        min_jaccard=indicators.block_min_jaccard,
        exclude_from_rules=[dataset.target_column],
        # These keep only their flag in the pipeline and cannot be in a block.
        exclude_from_blocks=list(dataset.indicator_only_values.columns),
        seed=cfg.seed,
    )
    path = Path(output_dir or imputation_output_dir(cfg)) / BLOCKS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    blocks = table[table["kind"] == "block"]
    logger.info(
        "Missingness patterns: %d block(s) covering %d column(s), %d other column(s); "
        "labels %s. Review %s against the codebook.",
        len(blocks),
        int(blocks["n_columns"].sum()),
        int((table["kind"] == "column").sum()),
        table["label"].value_counts().to_dict(),
        path,
    )
    return table
