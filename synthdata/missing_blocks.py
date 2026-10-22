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


def _encode(frame: pd.DataFrame) -> pd.DataFrame:
    """Numeric copy for a decision tree; text becomes codes, blanks stay NaN."""
    out = {}
    for column in frame.columns:
        values = frame[column]
        if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            out[column] = values.astype(float)
        else:
            codes, _ = pd.factorize(values)
            out[column] = pd.Series(codes, index=frame.index, dtype=float).replace(-1, np.nan)
    return pd.DataFrame(out, index=frame.index)


def missingness_rule(
    predictors: pd.DataFrame, missing: pd.Series, seed: int = 0
) -> tuple[float, str]:
    """How well a depth-2 tree on ``predictors`` predicts ``missing``, and its rule.

    Returns the 5-fold cross-validated ROC AUC and the tree's splits as text.
    The AUC is NaN when there are too few rows of either kind to cross-validate.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.tree import DecisionTreeClassifier, export_text

    y = missing.astype(int).to_numpy()
    if predictors.empty or min(y.sum(), len(y) - y.sum()) < 5:
        return float("nan"), ""
    x = _encode(predictors)
    tree = DecisionTreeClassifier(max_depth=2, min_samples_leaf=5, random_state=seed)
    folds = StratifiedKFold(5, shuffle=True, random_state=seed)
    auc = float(np.mean(cross_val_score(tree, x, y, cv=folds, scoring="roc_auc")))
    tree.fit(x, y)
    rule = export_text(tree, feature_names=[str(c) for c in x.columns], show_weights=True)
    return auc, " | ".join(line.strip() for line in rule.strip().splitlines())


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

    Columns: ``pattern``, ``kind`` (block or column), ``columns``,
    ``n_columns``, ``missing_share`` (rows missing any of its columns),
    ``whole_share`` (of those, rows missing all of them), ``rule_auc`` and
    ``rule`` (see :func:`missingness_rule`), ``indicator_only`` (the column
    is in ``exclude_from_blocks``, so it stays out of blocks; the pipeline
    keeps only its flag), and ``label``. The rule's predictors leave out the
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
    """
    shares = frame[columns].isna().mean()
    candidates = [c for c in columns if shares[c] >= min_missing_fraction]
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
        auc, rule = missingness_rule(predictors, any_missing, seed)
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
                "columns": "; ".join(map(str, members)),
                "n_columns": len(members),
                "missing_share": float(any_missing.mean()),
                "whole_share": whole,
                "rule_auc": auc,
                "rule": rule,
                "indicator_only": kind == "column" and members[0] in no_blocks,
                "label": label,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "pattern",
            "kind",
            "columns",
            "n_columns",
            "missing_share",
            "whole_share",
            "rule_auc",
            "rule",
            "indicator_only",
            "label",
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
        min_missing_fraction=indicators.min_missing_fraction,
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
