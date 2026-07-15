"""TabImpute-based missing data imputation.

Wraps ``tabimpute.interface.TabImputeCategorical`` (a TabPFN-based imputer that
one-hot encodes designated categorical columns before imputing, then recovers
category values via softmax + argmax). Includes a small compatibility shim for
version drift between the pinned ``tabpfn`` version and the one ``tabimpute`` was
built against (see the shim's docstring for details).
"""

import numpy as np
import pandas as pd

from synthdata.config import Config
from synthdata.data import Dataset
from synthdata.utils import ensure_dir, get_logger, resolve_device

logger = get_logger(__name__)

_SHIM_APPLIED = False


def _apply_tabpfn_compat_shim() -> None:
    """Patch missing tabpfn encoder classes used by an older tabimpute release.

    ``tabimpute`` was built against an older ``tabpfn`` release and imports a few
    encoder classes directly from ``tabpfn.model.encoders``. If the installed
    ``tabpfn`` version has moved/renamed those classes, we backfill them from
    ``tabimpute.model.encoders`` (which vendors compatible copies) so that
    ``TabImputeCategorical`` can be imported/instantiated without patching either
    library. This is a no-op if the classes already exist.
    """
    global _SHIM_APPLIED
    if _SHIM_APPLIED:
        return

    import tabpfn.model.encoders as _tabpfn_enc

    try:
        import tabimpute.model.encoders as _ti_enc
    except ImportError:
        _SHIM_APPLIED = True
        return

    for cls_name in (
        "SequentialEncoder",
        "VariableNumFeaturesEncoderStep",
        "InputNormalizationEncoderStep",
    ):
        if not hasattr(_tabpfn_enc, cls_name) and hasattr(_ti_enc, cls_name):
            setattr(_tabpfn_enc, cls_name, getattr(_ti_enc, cls_name))

    _SHIM_APPLIED = True


def impute_dataframe(
    df: pd.DataFrame,
    feature_columns: list,
    categorical_columns: list,
    target_column: str,
    device: str = "cpu",
) -> pd.DataFrame:
    """Impute missing values in ``feature_columns`` of ``df`` via TabImputeCategorical.

    The target column is assumed fully observed and is passed through unchanged.
    Returns a new DataFrame with the same column order as ``df``.
    """
    _apply_tabpfn_compat_shim()
    from tabimpute.interface import TabImputeCategorical

    imputer = TabImputeCategorical(device=device)

    x_full = df[feature_columns].values.astype(float)
    cat_indices = [
        feature_columns.index(c) for c in categorical_columns if c in feature_columns
    ]

    x_imputed = imputer.impute(x_full.copy(), categorical_columns=cat_indices)

    imputed_df = pd.DataFrame(x_imputed, columns=feature_columns, index=df.index)
    imputed_df[target_column] = df[target_column].values
    return imputed_df[list(df.columns)]


def apply_rounding(
    df: pd.DataFrame,
    feature_columns: list,
    round_rules: dict,
    round_to_int_default: bool = True,
) -> pd.DataFrame:
    """Apply post-imputation rounding: explicit per-column decimals, else nearest int."""
    out = df.copy()
    for col in feature_columns:
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

    Categorical columns: imputed values must be integral and within the observed
    category set. Continuous columns: imputed values must fall within
    ``[obs_min - margin * range, obs_max + margin * range]``.
    """
    if is_categorical:
        observed_categories = set(observed.dropna().unique().tolist())
        ok = imputed.apply(
            lambda v: (float(v).is_integer()) and (round(v) in observed_categories)
        )
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
    else:
        device = resolve_device(cfg.imputation.device)
        n_missing = int(dataset.full_df[dataset.feature_columns].isna().sum().sum())
        logger.info(
            "Imputing %d missing values across %d feature columns on device=%s",
            n_missing,
            len(dataset.feature_columns),
            device,
        )
        full_imputed = impute_dataframe(
            dataset.full_df,
            dataset.feature_columns,
            dataset.categorical_columns,
            dataset.target_column,
            device=device,
        )
        full_imputed = apply_rounding(
            full_imputed,
            dataset.feature_columns,
            cfg.imputation.round_rules,
            cfg.imputation.round_to_int_default,
        )

    ensure_dir(dataset.data_dir)
    full_imputed.to_csv(paths["full_imputed"], index=False)

    train_imputed = full_imputed.loc[dataset.train_df.index]
    test_imputed = full_imputed.loc[dataset.test_df.index]
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
        result = validate_imputed_column(
            observed, imputed, is_categorical, cfg.imputation.validation_margin
        )
        rows.append(
            {
                "column": col,
                "categorical": is_categorical,
                "n_missing": n_missing,
                "obs_mean": float(observed.mean()) if len(observed) else np.nan,
                "obs_std": float(observed.std()) if len(observed) else np.nan,
                "imp_mean": float(imputed.mean()) if len(imputed) else np.nan,
                "imp_std": float(imputed.std()) if len(imputed) else np.nan,
                **result,
            }
        )
    return pd.DataFrame(rows)
