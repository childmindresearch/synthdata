"""Method-agnostic imputation pipeline: caching, dispatch, rounding, validation.

:func:`run_imputation` dispatches to the configured backend's
``impute_dataframe`` (``synthdata.imputation.tabimpute_backend`` by default, or
``synthdata.imputation.refidiff_backend`` when ``imputation.method ==
"refidiff"``), then applies shared post-processing (rounding, caching to CSV,
validation reporting) identically regardless of which backend produced the
imputed values.
"""

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from synthdata.config import Config
from synthdata.data import (
    IMPUTATION_CACHE_KEY_FILENAME,
    Dataset,
    dataframe_fingerprint,
    load_imputed_splits,
)
from synthdata.utils import ensure_dir, get_logger, resolve_device

logger = get_logger(__name__)

#: Sidecar filename (under ``dataset.data_dir``) recording the config fields that
#: determined the currently-cached imputed CSVs -- see :func:`_cache_key_record`.
_CACHE_KEY_FILENAME = IMPUTATION_CACHE_KEY_FILENAME


def _persist_decoded_imputed_splits(dataset: Dataset) -> None:
    """Persist label-preserving views alongside numeric model-space caches."""
    dataset.attach_decoded_imputed_splits()
    decoded_frames = {
        "full_imputed_decoded": dataset.full_imputed_decoded_df,
        "train_imputed_decoded": dataset.train_imputed_decoded_df,
        "test_imputed_decoded": dataset.test_imputed_decoded_df,
    }
    missing = [name for name, frame in decoded_frames.items() if frame is None]
    if missing:
        raise RuntimeError(
            f"Cannot persist decoded imputed data because split(s) are missing: {missing}"
        )
    paths = dataset.paths()
    for name, frame in decoded_frames.items():
        frame.to_csv(paths[name], index=False)
    logger.info(
        "Wrote ordinal-decoded imputed splits under %s (model-space caches remain in "
        "full_imputed.csv/train_imputed.csv/test_imputed.csv)",
        dataset.data_dir,
    )


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
            seed=cfg.seed,
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


def _cache_key_payload(cfg: Config, dataset: Dataset) -> dict:
    """Build the dict of config/dataset fields that determine imputed values.

    Deliberately narrower than "the whole Config": only fields that actually
    change what :func:`_impute_dataframe`/:func:`apply_rounding` produce, so an
    unrelated config edit (e.g. ``evaluation.*``, ``imputation.validation_margin``,
    which only affects the post-hoc report, not the imputed values themselves)
    doesn't force an unnecessary retrain. Uses ``dataset.feature_columns``/
    ``dataset.nominal_columns``/``dataset.ordinal_columns`` (the already-resolved
    lists) rather than the written schema/config directly, so equivalent
    declarations correctly hash identically. The exact ordinal orders are
    included because RefiDiff's categorical encoding preserves that order. Exact
    fingerprints of the full source and deterministic train/test splits are also
    included so a refreshed source export cannot reuse an imputation cache merely
    because its columns and resolved schema happen to be unchanged.
    """
    imp_cfg = cfg.imputation
    payload = {
        "seed": cfg.seed,
        "target_column": dataset.target_column,
        "feature_columns": sorted(dataset.feature_columns),
        "nominal_columns": sorted(dataset.nominal_columns),
        "ordinal_columns": sorted(dataset.ordinal_columns),
        "ordinal_orders": {
            column: entry["ordinal_order"]
            for column, entry in dataset.variable_schema.items()
            if entry["ordinal_order"] is not None
        },
        "imputation_enabled": imp_cfg.enabled,
        "imputation_method": imp_cfg.method,
        "round_rules": imp_cfg.round_rules,
        "round_to_int_default": imp_cfg.round_to_int_default,
        "dataset_version": dataset.version,
        "variable_schema_fingerprint": dataset.variable_schema_fingerprint,
        "source_fingerprint": dataset.source_fingerprint,
        "full_fingerprint": dataframe_fingerprint(dataset.full_df),
        "train_split_fingerprint": dataframe_fingerprint(dataset.train_df),
        "test_split_fingerprint": dataframe_fingerprint(dataset.test_df),
    }
    if imp_cfg.method == "refidiff":
        payload["refidiff"] = dataclasses.asdict(imp_cfg.refidiff)
    return payload


def _cache_key_record(cfg: Config, dataset: Dataset) -> dict:
    """``_cache_key_payload`` plus its own sha256 digest under ``"cache_key"``."""
    payload = _cache_key_payload(cfg, dataset)
    encoded = json.dumps(payload, sort_keys=True, default=str)
    record = dict(payload)
    record["cache_key"] = hashlib.sha256(encoded.encode()).hexdigest()
    return record


def _load_cached_key(path: Path) -> str | None:
    """Read a previous run's ``cache_key`` from ``path``, or ``None`` if absent/unreadable.

    A missing file means either no cache exists yet or it predates this
    cache-key feature -- either way, treated as "no recorded key" (cache miss)
    rather than an error. A present-but-corrupt file (rare -- e.g. truncated by
    an interrupted write) is narrowly caught, logged, and also treated as a
    cache miss: safe to self-heal by retraining and rewriting the file, since
    this is cache metadata, not a scientific artifact.
    """
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f).get("cache_key")
    except json.JSONDecodeError as exc:
        logger.warning(
            "Failed to parse imputation cache-key file %s (%s); treating cached imputed data "
            "as stale and retraining",
            path,
            exc,
        )
        return None


def run_imputation(cfg: Config, dataset: Dataset) -> Dataset:
    """Impute ``dataset.full_df`` and populate the ``*_imputed`` splits.

    Caches to ``full_imputed.csv``/``train_imputed.csv``/``test_imputed.csv`` under
    ``cfg.data.data_dir``; reused on subsequent runs unless ``cfg.imputation.cache``
    is False. Reuse also requires the cache-key sidecar file
    ``.imputation_cache_key.json``, also under ``data_dir``) to match a fresh
    hash of the current config's imputation-relevant fields and exact
    source/split fingerprints (see :func:`_cache_key_payload`) -- so editing
    e.g. ``nominal_columns``/``ordinal_columns`` or refreshing the source and
    rerunning correctly retrains instead of silently reusing stale imputed
    CSVs from before the change.
    """
    paths = dataset.paths()
    cache_key_path = dataset.data_dir / _CACHE_KEY_FILENAME
    cache_record = _cache_key_record(cfg, dataset)
    current_key = cache_record["cache_key"]
    cached_key = _load_cached_key(cache_key_path)

    cached_csvs_exist = (
        paths["full_imputed"].exists()
        and paths["train_imputed"].exists()
        and paths["test_imputed"].exists()
    )

    if cfg.imputation.cache and cached_csvs_exist and cached_key == current_key:
        dataset = load_imputed_splits(dataset)
        if dataset.full_imputed_df is not None:
            _persist_decoded_imputed_splits(dataset)
            logger.info(
                "Using cached imputed data at %s (cache_key=%s)",
                dataset.data_dir,
                current_key[:16],
            )
            return dataset
        logger.warning(
            "Imputation cache key matched at %s, but cached frames failed provenance/shape "
            "validation; retraining",
            dataset.data_dir,
        )

    if cfg.imputation.cache and cached_csvs_exist and cached_key != current_key:
        logger.info(
            "Imputation-relevant config changed since the cached imputed data at %s was "
            "produced (cached cache_key=%s, current=%s) -- retraining instead of reusing "
            "the stale cache",
            dataset.data_dir,
            cached_key[:16] if cached_key else None,
            current_key[:16],
        )

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
    cache_record["imputed_row_counts"] = {
        "full": len(full_imputed),
        "train": len(train_imputed),
        "test": len(test_imputed),
    }
    with open(cache_key_path, "w") as f:
        json.dump(cache_record, f, indent=2, sort_keys=True, default=str)
    logger.info("Wrote imputation cache-key %s to %s", current_key[:16], cache_key_path)

    dataset.full_imputed_df = full_imputed
    dataset.train_imputed_df = train_imputed
    dataset.test_imputed_df = test_imputed
    _persist_decoded_imputed_splits(dataset)
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
