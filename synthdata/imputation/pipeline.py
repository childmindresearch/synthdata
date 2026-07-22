"""Method-agnostic imputation pipeline: caching, dispatch, rounding, validation.

:func:`run_imputation` dispatches to the configured backend's
``impute_dataframe`` (``synthdata.imputation.tabimpute_backend`` by default, or
``synthdata.imputation.refidiff_backend`` when ``imputation.method ==
"refidiff"``), then applies shared post-processing (rounding, caching to CSV,
validation reporting) identically regardless of which backend produced the
imputed values.
"""

import numpy as np
import pandas as pd

from synthdata.config import Config
from synthdata.data import Dataset
from synthdata.utils import ensure_dir, get_logger, resolve_device

logger = get_logger(__name__)


def _impute_dataframe(cfg: Config, df: pd.DataFrame, dataset: Dataset, device: str) -> pd.DataFrame:
    """Dispatch to the configured imputation backend's ``impute_dataframe``."""
    method = cfg.imputation.method
    if method == "tabimpute":
        from synthdata.imputation.tabimpute_backend import impute_dataframe

        return impute_dataframe(
            df,
            dataset.feature_columns,
            dataset.categorical_columns,
            dataset.target_column,
            device=device,
        )
    if method == "refidiff":
        from synthdata.imputation.refidiff_backend import impute_dataframe

        return impute_dataframe(
            df,
            dataset.feature_columns,
            dataset.categorical_columns,
            dataset.target_column,
            device=device,
            refidiff_cfg=cfg.imputation.refidiff,
            data_dir=dataset.data_dir,
        )
    # Unreachable in practice: Config._validate() already restricts
    # imputation.method to {"tabimpute", "refidiff"} before this runs.
    raise ValueError(f"Unknown imputation.method: {method!r}")


def apply_rounding(
    df: pd.DataFrame,
    feature_columns: list,
    round_rules: dict,
    round_to_int_default: bool = True,
) -> pd.DataFrame:
    """Apply post-imputation rounding: explicit per-column decimals, else nearest int.

    Non-numeric columns (e.g. string categories decoded by ``impute_dataframe``)
    are left untouched regardless of ``round_to_int_default``.
    """
    out = df.copy()
    for col in feature_columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            continue
        if col in round_rules:
            out[col] = out[col].round(round_rules[col])
        elif round_to_int_default:
            out[col] = out[col].round(0).astype(int)
    return out


def validate_imputed_column(
    observed: pd.Series,
    imputed: pd.Series,
    is_categorical: bool,
    margin: float = 0.2,
) -> dict:
    """Check that imputed values are plausible given the observed distribution.

    Categorical columns: imputed values must be within the observed category set
    (and, for numerically-coded categories, integral). Continuous columns:
    imputed values must fall within ``[obs_min - margin * range, obs_max + margin * range]``.
    """
    if is_categorical:
        observed_categories = set(observed.dropna().unique().tolist())
        if pd.api.types.is_numeric_dtype(observed):
            ok = imputed.apply(
                lambda v: (float(v).is_integer()) and (round(v) in observed_categories)
            )
        else:
            ok = imputed.isin(observed_categories)
    else:
        obs_min, obs_max = observed.min(), observed.max()
        span = obs_max - obs_min
        lo, hi = obs_min - margin * span, obs_max + margin * span
        ok = imputed.between(lo, hi)
    return {
        "n_imputed": int(len(imputed)),
        "n_valid": int(ok.sum()),
        "all_valid": bool(ok.all()),
    }


def run_imputation(cfg: Config, dataset: Dataset) -> Dataset:
    """Impute ``dataset.full_df`` and populate the ``*_imputed`` splits.

    Caches to ``full_imputed.csv``/``train_imputed.csv``/``test_imputed.csv`` under
    ``cfg.data.data_dir``; reused on subsequent runs unless ``cfg.imputation.cache``
    is False.
    """
    paths = dataset.paths()

    if (
        cfg.imputation.cache
        and paths["full_imputed"].exists()
        and paths["train_imputed"].exists()
        and paths["test_imputed"].exists()
    ):
        logger.info("Using cached imputed data at %s", dataset.data_dir)
        dataset.full_imputed_df = pd.read_csv(paths["full_imputed"])
        dataset.train_imputed_df = pd.read_csv(paths["train_imputed"])
        dataset.test_imputed_df = pd.read_csv(paths["test_imputed"])
        return dataset

    if not cfg.imputation.enabled:
        logger.info("Imputation disabled; using rows with complete cases only")
        full_imputed = dataset.full_df.dropna().copy()
        if full_imputed.empty:
            raise RuntimeError(
                "imputation.enabled=false requires complete-case rows, but every row has "
                "at least one missing feature value (0 complete cases out of "
                f"{len(dataset.full_df)}). Set imputation.enabled: true in the config."
            )
    else:
        device = resolve_device(cfg.imputation.device)
        n_missing = int(dataset.full_df[dataset.feature_columns].isna().sum().sum())
        logger.info(
            "Imputing %d missing values across %d feature columns via method=%s on device=%s",
            n_missing,
            len(dataset.feature_columns),
            cfg.imputation.method,
            device,
        )
        full_imputed = _impute_dataframe(cfg, dataset.full_df, dataset, device)
        full_imputed = apply_rounding(
            full_imputed,
            dataset.feature_columns,
            cfg.imputation.round_rules,
            cfg.imputation.round_to_int_default,
        )

    ensure_dir(dataset.data_dir)
    full_imputed.to_csv(paths["full_imputed"], index=False)

    # When imputation is disabled, full_imputed is a complete-case subset of
    # full_df (dropna()), so its index may no longer contain every train/test
    # row -- intersect rather than assume a full match (still a strict subset
    # when imputation ran, since full_imputed then shares full_df's index).
    train_imputed = full_imputed.loc[full_imputed.index.intersection(dataset.train_df.index)]
    test_imputed = full_imputed.loc[full_imputed.index.intersection(dataset.test_df.index)]
    if not cfg.imputation.enabled and (
        len(train_imputed) < len(dataset.train_df) or len(test_imputed) < len(dataset.test_df)
    ):
        logger.info(
            "Complete-case filtering dropped train %d->%d, test %d->%d rows",
            len(dataset.train_df),
            len(train_imputed),
            len(dataset.test_df),
            len(test_imputed),
        )
    train_imputed.to_csv(paths["train_imputed"], index=False)
    test_imputed.to_csv(paths["test_imputed"], index=False)

    dataset.full_imputed_df = full_imputed
    dataset.train_imputed_df = train_imputed
    dataset.test_imputed_df = test_imputed
    return dataset


def build_validation_report(cfg: Config, dataset: Dataset) -> pd.DataFrame:
    """Build a per-column validation table comparing observed vs. imputed values."""
    rows = []
    full_df = dataset.full_df
    full_imputed = dataset.full_imputed_df
    if full_imputed is None:
        raise RuntimeError("run_imputation() must be called before build_validation_report()")

    for col in dataset.feature_columns:
        missing_mask = full_df[col].isna()
        n_missing = int(missing_mask.sum())
        if n_missing == 0:
            continue
        observed = full_df.loc[~missing_mask, col]
        imputed = full_imputed.loc[missing_mask, col]
        is_categorical = col in dataset.categorical_columns
        is_numeric = pd.api.types.is_numeric_dtype(observed)
        result = validate_imputed_column(
            observed, imputed, is_categorical, cfg.imputation.validation_margin
        )
        rows.append(
            {
                "column": col,
                "categorical": is_categorical,
                "n_missing": n_missing,
                "obs_mean": float(observed.mean()) if is_numeric and len(observed) else np.nan,
                "obs_std": float(observed.std()) if is_numeric and len(observed) else np.nan,
                "imp_mean": float(imputed.mean()) if is_numeric and len(imputed) else np.nan,
                "imp_std": float(imputed.std()) if is_numeric and len(imputed) else np.nan,
                **result,
            }
        )
    return pd.DataFrame(rows)
