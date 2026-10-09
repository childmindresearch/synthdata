"""Sanity checks on every evaluation run: does the output look like it should?

The integration tests prove the pipeline on a fixture. These checks run on the
real data of every run and catch what a fixture cannot: a generator that
copies patients, invents categories or drifts out of range on this dataset,
and metric wiring that no longer separates the two fixed baselines (see
:mod:`synthdata.evaluation.baselines`). A baseline in the wrong place means a
metric is broken, not that a generator is good or bad.

Each check is a row of ``checks.csv`` with ``status`` "pass" or "warn". The
report lists the warnings under "Read with care", and ``synthdata-evaluate
--strict-checks`` exits with an error when there is any.
"""

import numpy as np
import pandas as pd

from synthdata.evaluation.baselines import is_baseline
from synthdata.utils import get_logger, split_replicate_name

logger = get_logger(__name__)

#: Same tolerances as the HPO screens (``generation.hpo.constraints``), so a
#: model chosen by the search passes them unless sampling the final data drifted.
COPY_MARGIN = 0.02
MAX_UNSEEN_CATEGORY_SHARE = 0.01
MAX_OUT_OF_RANGE_SHARE = 0.01
MAX_CLASS_SHARE_GAP = 0.02

COLUMNS = ["check", "model", "status", "value", "limit", "detail"]


def _row(check, model, ok, value, limit, detail) -> dict:
    return {
        "check": check,
        "model": model,
        "status": "pass" if ok else "warn",
        "value": value,
        "limit": limit,
        "detail": detail,
    }


def _row_hashes(frame: pd.DataFrame, columns: list) -> pd.Series:
    return pd.util.hash_pandas_object(frame[columns].reset_index(drop=True), index=False)


def _dataset_checks(name, synthetic, train, holdout, dataset, match_class_prior) -> list:
    rows = []
    columns = list(train.columns)
    missing = [c for c in columns if c not in synthetic.columns]
    extra = [c for c in synthetic.columns if c not in columns]
    rows.append(
        _row(
            "columns_match",
            name,
            not missing and not extra,
            len(missing) + len(extra),
            0,
            f"missing {missing}, unexpected {extra}" if missing or extra else "",
        )
    )
    if missing:
        return rows

    # Exact copies of training rows, against the rate at which real holdout
    # rows already repeat a training row (duplicates are legitimate on
    # low-cardinality data).
    train_hashes = set(_row_hashes(train, columns))
    copy_share = float(_row_hashes(synthetic, columns).isin(train_hashes).mean())
    holdout_share = float(_row_hashes(holdout, columns).isin(train_hashes).mean())
    limit = holdout_share + COPY_MARGIN
    rows.append(
        _row(
            "exact_copies_of_train_rows",
            name,
            copy_share <= limit,
            copy_share,
            limit,
            f"holdout rows that repeat a train row: {holdout_share:.3f}",
        )
    )

    categorical = [c for c in dataset.all_categorical_columns if c in columns]
    if dataset.target_is_categorical and dataset.target_column not in categorical:
        categorical.append(dataset.target_column)
    unseen = cells = 0
    for column in categorical:
        values = synthetic[column].dropna()
        unseen += int((~values.isin(set(train[column].dropna()))).sum())
        cells += len(values)
    share = unseen / cells if cells else 0.0
    rows.append(
        _row(
            "unseen_categories",
            name,
            share <= MAX_UNSEEN_CATEGORY_SHARE,
            share,
            MAX_UNSEEN_CATEGORY_SHARE,
            "share of categorical cells with a value absent from train",
        )
    )

    continuous = [
        c for c in columns if c not in categorical and pd.api.types.is_numeric_dtype(train[c])
    ]
    outside = cells = 0
    for column in continuous:
        values = pd.to_numeric(synthetic[column], errors="coerce").dropna()
        low, high = train[column].min(), train[column].max()
        outside += int(((values < low) | (values > high)).sum())
        cells += len(values)
    share = outside / cells if cells else 0.0
    rows.append(
        _row(
            "out_of_train_range",
            name,
            share <= MAX_OUT_OF_RANGE_SHARE,
            share,
            MAX_OUT_OF_RANGE_SHARE,
            "share of continuous cells outside the train min/max",
        )
    )

    if match_class_prior and dataset.target_is_categorical:
        target = dataset.target_column
        shares = pd.concat(
            [
                train[target].value_counts(normalize=True),
                synthetic[target].value_counts(normalize=True),
            ],
            axis=1,
        ).fillna(0.0)
        gap = float((shares.iloc[:, 0] - shares.iloc[:, 1]).abs().max())
        rows.append(
            _row(
                "class_shares_match_train",
                name,
                gap <= MAX_CLASS_SHARE_GAP,
                gap,
                MAX_CLASS_SHARE_GAP,
                "largest gap between a class's train and synthetic share",
            )
        )
    return rows


def _mean_by_model(series: pd.Series) -> pd.Series:
    return series.groupby([split_replicate_name(m)[0] for m in series.index]).mean()


def _baseline_checks(combined: pd.DataFrame) -> list:
    rows = []
    columns = {c[1:]: c for c in combined.columns if c[0] == "__all__"}
    scores = {
        dim: _mean_by_model(combined[columns[(dim, "rank")]].astype(float))
        for dim in ("utility", "privacy")
        if (dim, "rank") in columns
    }
    copy_, marginals = "baseline_train_copy", "baseline_marginals"

    privacy = scores.get("privacy")
    if privacy is not None and copy_ in privacy.index and len(privacy) > 1:
        others = privacy.drop(copy_)
        ok = bool(privacy[copy_] <= others.min() + 1e-9)
        rows.append(
            _row(
                "train_copy_has_the_lowest_privacy",
                copy_,
                ok,
                float(privacy[copy_]),
                float(others.min()),
                "" if ok else "privacy metrics do not penalise copying real rows",
            )
        )
    utility = scores.get("utility")
    if utility is not None and {copy_, marginals} <= set(utility.index):
        ok = bool(utility[copy_] > utility[marginals])
        rows.append(
            _row(
                "train_copy_beats_marginals_on_utility",
                copy_,
                ok,
                float(utility[copy_]),
                float(utility[marginals]),
                "" if ok else "utility metrics do not reward the real joint structure",
            )
        )
    tstr = [c for c in combined.columns if c[2] == "tstr_macro_f1"]
    if tstr:
        f1 = _mean_by_model(combined[tstr[0]].astype(float))
        if {copy_, marginals} <= set(f1.index):
            ok = bool(f1[copy_] > f1[marginals])
            rows.append(
                _row(
                    "train_copy_beats_marginals_on_tstr",
                    copy_,
                    ok,
                    float(f1[copy_]),
                    float(f1[marginals]),
                    ""
                    if ok
                    else "a classifier trained on real rows is no better than on shuffled "
                    "columns: the target may carry no signal, or TSTR is miswired",
                )
            )
    return rows


def run_output_checks(
    dataset,
    synthetic_datasets: dict[str, pd.DataFrame],
    combined: pd.DataFrame,
    match_class_prior: bool,
) -> pd.DataFrame:
    """Run every check; return one row per (check, model)."""
    train, holdout = dataset.train_imputed_df, dataset.test_imputed_df
    rows = []
    for name, synthetic in sorted(synthetic_datasets.items()):
        if is_baseline(name):
            continue
        try:
            rows += _dataset_checks(name, synthetic, train, holdout, dataset, match_class_prior)
        except Exception as exc:  # noqa: BLE001 -- a check must never stop the evaluation
            logger.warning("[checks] %s: could not run the data checks: %s", name, exc)
            rows.append(_row("data_checks_ran", name, False, np.nan, np.nan, str(exc)))
    if not combined.empty:
        rows += _baseline_checks(combined)
    table = pd.DataFrame(rows, columns=COLUMNS)
    warned = table[table["status"] == "warn"]
    for row in warned.itertuples():
        logger.warning(
            "[checks] %s on %s: %.4g (limit %.4g) %s",
            row.check,
            row.model,
            row.value,
            row.limit,
            row.detail,
        )
    logger.info("[checks] %d passed, %d warnings", len(table) - len(warned), len(warned))
    return table
