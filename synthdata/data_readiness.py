"""Data-readiness check: columns likely to be hard to model, before any fitting.

Run at the start of ``synthdata-impute`` on the train split (so nothing about
the test split decides anything), it flags columns that tend to give weak
synthetic data, break metrics or leak identity, each with a suggested change
to the dataset or config. It only advises: nothing is dropped or changed,
because that would change the released schema, which is the user's call.

Results go to ``data_readiness.csv`` (one row per finding: check, column,
value, threshold, status, action) in the imputation output folder, a grouped
warning per check in the log, and a table in report.md (section 6).

Every check and default threshold, with its source, is listed in
docs/verification.md ("Data readiness"); thresholds are set in
``data.readiness`` (:class:`synthdata.config.DataReadinessConfig`).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from synthdata.config import Config, DataReadinessConfig
from synthdata.data import Dataset, identifier_like_columns
from synthdata.utils import get_logger

logger = get_logger(__name__)

READINESS_FILENAME = "data_readiness.csv"

COLUMNS = ["check", "column", "value", "threshold", "status", "action"]

#: check -> (what it means, what to do). The action is shown in the CSV and log.
CHECKS = {
    "constant": (
        "one value (or all missing) in the train split",
        "drop it with data.drop_columns: it carries no information and makes "
        "covariance-based metrics singular",
    ),
    "near_zero_variance": (
        "one value dominates and few values are distinct (caret nearZeroVar)",
        "drop it or merge its rare values; it may be constant within a split",
    ),
    "duplicate": (
        "same information as another column (identical up to relabelling)",
        "keep one of the two; duplicates make covariance-based metrics singular "
        "and double-weight distance metrics",
    ),
    "high_correlation": (
        "abs(Spearman) with another continuous column at or above the threshold",
        "keep one, or derive one from the other after generation",
    ),
    "heavy_missingness": (
        "missing in a large share of the train rows",
        "consider dropping it: most of its values are the imputer's, not data",
    ),
    "rare_categories": (
        "categories with very few train rows",
        "merge rare categories into a broader one; generators rarely learn them "
        "and they are the easiest to re-identify",
    ),
    "high_cardinality": (
        "nominal column with many levels",
        "group its levels; each level is a separate one-hot input to the generators",
    ),
    "unseen_category": (
        "categories in the tuning or test split that the rows they are compared with lack",
        "merge them into a category the training rows have; a generator cannot produce them",
    ),
    "split_constant": (
        "constant in the tuning or test split but not in train",
        "merge rare values; metrics fitted on that split see a constant column",
    ),
    "skewed": (
        "abs(skewness) above the threshold",
        "consider a log transform before generation",
    ),
    "zero_inflated": (
        "large share of exact zeros in a continuous column",
        "consider splitting into a zero/non-zero indicator and the non-zero amount",
    ),
    "identifier_like": (
        "looks like a direct identifier (name, or unique on almost every row)",
        "drop it with data.drop_columns, or move a patient id to data.patient_id_column",
    ),
    "rows_per_column": (
        "few train rows per modelled column",
        "drop or aggregate columns (e.g. keep scale totals, not items); expect "
        "deep generators to underfit",
    ),
    "small_group": (
        "target class or protected group with few rows in a split",
        "merge classes or groups; per-group metrics and HPO screens are unstable below this",
    ),
}


def _row(check, column, value, threshold, status="warn", detail=None) -> dict:
    action = CHECKS[check][1] + (f" ({detail})" if detail else "")
    return {
        "check": check,
        "column": column,
        "value": value,
        "threshold": threshold,
        "status": status,
        "action": action,
    }


def _constant(series: pd.Series) -> bool:
    return series.dropna().nunique() <= 1


def _near_zero_variance(series: pd.Series, freq_ratio: float, unique_percent: float):
    """caret ``nearZeroVar``: (frequency ratio, percent unique) if flagged, else None."""
    values = series.dropna()
    counts = values.value_counts()
    if len(counts) < 2:
        return None
    ratio = counts.iloc[0] / counts.iloc[1]
    percent_unique = 100 * len(counts) / len(values)
    # caret: freqRatio > freqCut & percentUnique <= uniqueCut
    if ratio > freq_ratio and percent_unique <= unique_percent:
        return ratio, percent_unique
    return None


def _duplicates(df: pd.DataFrame, columns: list, categorical: set) -> dict:
    """``{column: earlier column it duplicates}``.

    Categorical columns match when they split the rows the same way (identical
    up to relabelling, e.g. two missing indicators of an instrument that is
    always missing together); other columns when their values are identical.
    """
    seen, dupes = {}, {}
    for column in columns:
        if column in categorical:
            codes, _ = pd.factorize(df[column], use_na_sentinel=True)
            key = ("categorical", codes.tobytes())
        else:
            key = ("values", pd.util.hash_pandas_object(df[column], index=False).values.tobytes())
        if key in seen:
            dupes[column] = seen[key]
        else:
            seen[key] = column
    return dupes


def _split_rows(dataset: Dataset) -> dict:
    """The splits to compare: search-train (train minus tuning), tuning and test."""
    splits = {"test": dataset.test_df}
    if len(dataset.tuning_index):
        splits["tuning"] = dataset.tuning_df
    return splits


def assess(dataset: Dataset, cfg: DataReadinessConfig) -> pd.DataFrame:
    """Every finding of the readiness check on ``dataset``'s train split."""
    train = dataset.train_df
    features = [c for c in dataset.feature_columns if c in train.columns]
    categorical = set(dataset.categorical_columns)
    nominal = set(dataset.nominal_columns)
    continuous = [
        c for c in features if c not in categorical and pd.api.types.is_numeric_dtype(train[c])
    ]
    rows = []

    constant = [c for c in features if _constant(train[c])]
    rows += [_row("constant", c, train[c].dropna().nunique(), 1) for c in constant]
    usable = [c for c in features if c not in constant]

    if cfg.near_zero_freq_ratio is not None:
        for c in usable:
            hit = _near_zero_variance(
                train[c], cfg.near_zero_freq_ratio, cfg.near_zero_unique_percent
            )
            if hit:
                rows.append(
                    _row(
                        "near_zero_variance",
                        c,
                        round(hit[0], 1),
                        f"ratio > {cfg.near_zero_freq_ratio:g} and "
                        f"unique <= {cfg.near_zero_unique_percent:g}%",
                        detail=f"{hit[1]:.1f}% distinct",
                    )
                )

    duplicates = _duplicates(train, usable, categorical)
    rows += [
        _row("duplicate", c, other, "identical", detail=f"same as {other}")
        for c, other in duplicates.items()
    ]

    if cfg.max_abs_correlation is not None:
        candidates = [c for c in continuous if c in usable and c not in duplicates]
        if len(candidates) > 1:
            # Spearman as Pearson on each column's ranks, pairwise-complete:
            # within ~0.003 of pandas' exact pairwise Spearman and ~10x faster
            # on wide data with missing values.
            corr = train[candidates].rank().corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
            pairs = upper.stack()
            for (a, b), value in pairs[pairs >= cfg.max_abs_correlation].items():
                rows.append(
                    _row(
                        "high_correlation",
                        b,
                        round(float(value), 3),
                        cfg.max_abs_correlation,
                        detail=f"with {a}",
                    )
                )

    if cfg.max_missing_fraction is not None:
        missing = train[features].isna().mean()
        for c, value in missing[missing >= cfg.max_missing_fraction].items():
            rows.append(
                _row("heavy_missingness", c, round(float(value), 3), cfg.max_missing_fraction)
            )
    for c in dataset.indicator_only_values.columns:
        rows.append(
            _row(
                "heavy_missingness",
                c,
                round(float(dataset.indicator_only_values[c].loc[train.index].isna().mean()), 3),
                "indicator-only",
                status="info",
                detail="already kept as its missing indicator only",
            )
        )

    for c in usable:
        if c not in categorical:
            continue
        counts = train[c].dropna().value_counts()
        if cfg.min_category_rows is not None:
            rare = counts[counts < cfg.min_category_rows]
            if len(rare):
                rows.append(
                    _row(
                        "rare_categories",
                        c,
                        len(rare),
                        f"< {cfg.min_category_rows} rows",
                        detail="levels "
                        + ", ".join(map(str, rare.index[:5]))
                        + (", ..." if len(rare) > 5 else ""),
                    )
                )
        if cfg.max_levels is not None and c in nominal and len(counts) > cfg.max_levels:
            rows.append(_row("high_cardinality", c, len(counts), cfg.max_levels))

    splits = _split_rows(dataset)
    reference = {"tuning": dataset.search_train_df, "test": train}
    for c in usable:
        for name, split in splits.items():
            if c not in split.columns or split.empty:
                continue
            if c in categorical:
                unseen = set(split[c].dropna()) - set(reference[name][c].dropna())
                if unseen:
                    rows.append(
                        _row(
                            "unseen_category",
                            c,
                            len(unseen),
                            0,
                            detail=f"in {name}: "
                            + ", ".join(map(str, sorted(unseen, key=str)[:5])),
                        )
                    )
            if _constant(split[c]):
                rows.append(_row("split_constant", c, name, "not constant", status="info"))

    for c in continuous:
        if c not in usable:
            continue
        values = pd.to_numeric(train[c], errors="coerce").dropna()
        if len(values) < 3:
            continue
        if cfg.max_abs_skew is not None:
            skew = float(values.skew())
            if abs(skew) > cfg.max_abs_skew:
                rows.append(_row("skewed", c, round(skew, 2), cfg.max_abs_skew, status="info"))
        if cfg.max_zero_fraction is not None:
            zeros = float((values == 0).mean())
            if zeros >= cfg.max_zero_fraction:
                rows.append(
                    _row("zero_inflated", c, round(zeros, 3), cfg.max_zero_fraction, status="info")
                )

    for c in identifier_like_columns(train, features, dataset.categorical_columns):
        rows.append(_row("identifier_like", c, "", "name or ~unique"))

    if cfg.min_rows_per_column is not None and features:
        ratio = len(train) / len(features)
        if ratio < cfg.min_rows_per_column:
            rows.append(
                _row(
                    "rows_per_column",
                    "(dataset)",
                    round(ratio, 2),
                    cfg.min_rows_per_column,
                    detail=f"{len(train)} rows, {len(features)} columns",
                )
            )

    if cfg.min_group_rows is not None:
        group_columns = [dataset.target_column] if dataset.target_is_categorical else []
        group_columns += [c for c in dataset.protected_columns if c in categorical]
        for c in group_columns:
            levels = train[c].dropna().unique()
            for name, split in splits.items():
                counts = split[c].value_counts().reindex(levels, fill_value=0)
                small = counts[counts < cfg.min_group_rows]
                if len(small):
                    rows.append(
                        _row(
                            "small_group",
                            c,
                            int(small.min()),
                            cfg.min_group_rows,
                            detail=f"in {name}: "
                            + ", ".join(f"{k} ({v})" for k, v in small.items()),
                        )
                    )

    return pd.DataFrame(rows, columns=COLUMNS)


def log_findings(findings: pd.DataFrame, path: Path | None = None) -> None:
    """One log line per check: how many columns and the first few names."""
    where = f" (see {path})" if path is not None else ""
    if findings.empty:
        logger.info("Data readiness: no findings%s", where)
        return
    for check, group in findings.groupby("check", sort=False):
        columns = list(dict.fromkeys(group["column"].astype(str)))
        log = logger.warning if (group["status"] == "warn").any() else logger.info
        log(
            "Data readiness: %d column(s) %s [%s]: %s%s. Suggested: %s%s",
            len(columns),
            CHECKS[check][0],
            check,
            ", ".join(columns[:8]),
            ", ..." if len(columns) > 8 else "",
            CHECKS[check][1],
            where,
        )


def save_readiness_report(cfg: Config, dataset: Dataset) -> pd.DataFrame | None:
    """Run the check, write ``data_readiness.csv`` and log the findings.

    Returns the findings, or None when ``data.readiness.enabled`` is off.
    """
    from synthdata.experiment import imputation_output_dir

    if not cfg.data.readiness.enabled:
        return None
    findings = assess(dataset, cfg.data.readiness)
    path = imputation_output_dir(cfg) / READINESS_FILENAME
    findings.to_csv(path, index=False)
    log_findings(findings, path)
    return findings
