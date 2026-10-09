"""Method-agnostic imputation pipeline: caching, dispatch, rounding, validation.

:func:`run_imputation` dispatches to the configured backend
(``synthdata.imputation.sklearn_backend`` for ``missforest``/``simple``,
``tabimpute_backend`` or ``refidiff_backend``), fitting it twice (see
:func:`_impute_splits`), then applies shared post-processing (rounding,
caching to CSV, validation reporting) identically regardless of which backend
produced the imputed values.
"""

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from synthdata.config import Config
from synthdata.data import (
    Dataset,
    dataframe_fingerprint,
    load_imputed_splits,
    migrate_legacy_imputation_layout,
)
from synthdata.experiment import imputation_output_dir
from synthdata.utils import ensure_dir, get_logger, resolve_device

logger = get_logger(__name__)

#: File name of the imputation reports under :func:`imputation_output_dir`.
VALIDATION_REPORT_FILENAME = "imputation_validation_report.csv"
DRIFT_REPORT_FILENAME = "imputation_drift.csv"


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
        paths["full_imputed_decoded"].parent,
    )


def _fit_and_apply(
    cfg: Config,
    dataset: Dataset,
    fit_df: pd.DataFrame,
    apply_to: pd.DataFrame,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the configured imputer on ``fit_df``, then fill ``fit_df`` and ``apply_to``.

    ``apply_to`` never shapes the fitted imputer. RefiDiff and the
    scikit-learn methods fill it with the fitted model. TabImpute has no
    fitting step (it is a pretrained in-context model), so ``fit_df`` is
    imputed on its own and ``apply_to`` with ``fit_df`` as context; rows of
    ``apply_to`` can inform each other's imputations but never ``fit_df``'s.
    """
    method = cfg.imputation.method
    if method in ("missforest", "simple"):
        from synthdata.imputation import sklearn_backend

        state, fit_imputed = sklearn_backend.fit(
            fit_df,
            dataset.feature_columns,
            dataset.categorical_columns,
            dataset.nominal_columns,
            method,
            seed=cfg.seed,
            missforest_cfg=cfg.imputation.missforest,
        )

        def _transform(frame: pd.DataFrame) -> pd.DataFrame:
            return sklearn_backend.transform(state, frame)

    elif method == "tabimpute":
        from synthdata.imputation.tabimpute_backend import impute_dataframe

        def _impute(frame: pd.DataFrame) -> pd.DataFrame:
            return impute_dataframe(
                frame,
                dataset.feature_columns,
                dataset.categorical_columns,
                dataset.target_column,
                device=device,
            )

        fit_imputed = _impute(fit_df)

        def _transform(frame: pd.DataFrame) -> pd.DataFrame:
            return _impute(pd.concat([fit_df, frame])).iloc[len(fit_df) :]

    elif method == "refidiff":
        from synthdata.imputation import refidiff_backend

        state, fit_imputed = refidiff_backend.fit(
            fit_df,
            dataset.feature_columns,
            dataset.categorical_columns,
            dataset.target_column,
            device=device,
            refidiff_cfg=cfg.imputation.refidiff,
            data_dir=dataset.data_dir,
            seed=cfg.seed,
            model_columns=[c for c in dataset.feature_columns if apply_to[c].isna().any()],
        )

        def _transform(frame: pd.DataFrame) -> pd.DataFrame:
            return refidiff_backend.transform(state, frame)

    else:
        # Unreachable in practice: Config._validate() already restricts
        # imputation.method before this runs.
        raise ValueError(f"Unknown imputation.method: {method!r}")

    applied = _transform(apply_to) if not apply_to.empty else apply_to.copy()
    return fit_imputed, applied


def runs_search_phase(cfg: Config, dataset: Dataset) -> bool:
    """Whether HPO needs its own imputation fitted on train minus tuning."""
    return cfg.generation.hpo.enabled and not dataset.tuning_index.empty


def _impute_splits(
    cfg: Config, dataset: Dataset, device: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Impute in two phases so no fit ever sees the rows it is scored on.

    Phase 1 (only when HPO runs): fit on train minus tuning
    (:attr:`Dataset.search_train_df`) and fill the tuning rows, so
    hyperparameter scores are not shaped by imputations that saw the tuning
    rows. Phase 2: refit with the same settings and seed on all of train
    (tuning included), the data the final generators train on, and fill the
    holdout. This is the refit-after-selection pattern of scikit-learn's
    ``GridSearchCV(refit=True)``. Returns ``(train_imputed, test_imputed,
    search_imputed)`` where ``search_imputed`` holds phase 1's train rows
    (``None`` when phase 1 did not run), each in ``dataset.train_df`` order.
    """
    search_imputed = None
    if runs_search_phase(cfg, dataset):
        fit_imputed, tuning_imputed = _fit_and_apply(
            cfg, dataset, dataset.search_train_df, dataset.tuning_df, device
        )
        search_imputed = pd.concat([fit_imputed, tuning_imputed]).loc[dataset.train_df.index]
    train_imputed, test_imputed = _fit_and_apply(
        cfg, dataset, dataset.train_df, dataset.test_df, device
    )
    return train_imputed.loc[dataset.train_df.index], test_imputed, search_imputed


def imputation_drift(
    dataset: Dataset, search_imputed: pd.DataFrame, train_imputed: pd.DataFrame
) -> pd.DataFrame:
    """Compare the two phases' fills of the cells both of them imputed.

    Those are the missing cells of the train rows outside the tuning split.
    Continuous columns: Wasserstein distance divided by the column's observed
    standard deviation. Categorical columns: total variation distance between
    the two fills' category shares. Both are 0 when the fills agree.
    """
    from scipy.stats import wasserstein_distance

    rows = dataset.search_train_df
    records = []
    for column in dataset.feature_columns:
        missing = rows[column].isna()
        if not missing.any():
            continue
        index = rows.index[missing]
        phase1, phase2 = search_imputed.loc[index, column], train_imputed.loc[index, column]
        if column in dataset.categorical_columns or not pd.api.types.is_numeric_dtype(phase1):
            shares = pd.concat(
                [phase1.value_counts(normalize=True), phase2.value_counts(normalize=True)], axis=1
            ).fillna(0)
            metric, value = (
                "total_variation",
                0.5 * float((shares.iloc[:, 0] - shares.iloc[:, 1]).abs().sum()),
            )
        else:
            scale = float(rows[column].std()) or 1.0
            metric = "wasserstein_over_std"
            value = wasserstein_distance(phase1.astype(float), phase2.astype(float)) / scale
        records.append(
            {"column": column, "n_cells": int(missing.sum()), "metric": metric, "drift": value}
        )
    return pd.DataFrame(records, columns=["column", "n_cells", "metric", "drift"])


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
    change what :func:`_impute_splits`/:func:`apply_rounding` produce, so an
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
        # Caches written before the two-phase fit are stale.
        "imputer_fit_split": "search_then_train",
        "search_phase": runs_search_phase(cfg, dataset),
        "round_rules": imp_cfg.round_rules,
        "round_to_int_default": imp_cfg.round_to_int_default,
        "dataset_version": dataset.version,
        "variable_schema_fingerprint": dataset.variable_schema_fingerprint,
        "source_fingerprint": dataset.source_fingerprint,
        "full_fingerprint": dataframe_fingerprint(dataset.full_df),
        "train_split_fingerprint": dataframe_fingerprint(dataset.train_df),
        "test_split_fingerprint": dataframe_fingerprint(dataset.test_df),
        "tuning_split_fingerprint": dataframe_fingerprint(dataset.tuning_df),
    }
    if imp_cfg.method == "refidiff":
        payload["refidiff"] = dataclasses.asdict(imp_cfg.refidiff)
    if imp_cfg.method == "missforest":
        payload["missforest"] = dataclasses.asdict(imp_cfg.missforest)
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
    """Impute the train and test splits and populate the ``*_imputed`` frames.

    The imputer learns from the train split only and then fills test (see
    :func:`_impute_splits`), so held-out rows never influence train values.

    Caches the phase-2 fill to ``full_imputed.csv``/``train_imputed.csv``/
    ``test_imputed.csv`` under ``<data_dir>/imputation_final/`` and, when HPO runs,
    the phase-1 fill to ``<data_dir>/imputation_initial/train_imputed.csv`` (see
    :meth:`Dataset.paths`); reused on subsequent runs unless ``cfg.imputation.cache``
    is False. Reuse also requires the cache-key sidecar file
    (``imputation_final/.imputation_cache_key.json``) to match a fresh
    hash of the current config's imputation-relevant fields and exact
    source/split fingerprints (see :func:`_cache_key_payload`) -- so editing
    e.g. ``nominal_columns``/``ordinal_columns`` or refreshing the source and
    rerunning correctly retrains instead of silently reusing stale imputed
    CSVs from before the change.
    """
    migrate_legacy_imputation_layout(dataset)
    legacy_drift = dataset.data_dir / DRIFT_REPORT_FILENAME
    if legacy_drift.is_file():
        legacy_drift.replace(imputation_output_dir(cfg) / DRIFT_REPORT_FILENAME)
    paths = dataset.paths()
    cache_key_path = paths["imputation_cache_key"]
    cache_record = _cache_key_record(cfg, dataset)
    current_key = cache_record["cache_key"]
    cached_key = _load_cached_key(cache_key_path)

    cached_csvs_exist = (
        paths["full_imputed"].exists()
        and paths["train_imputed"].exists()
        and paths["test_imputed"].exists()
        and (paths["search_imputed"].exists() or not runs_search_phase(cfg, dataset))
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

    search_imputed = None
    if not cfg.imputation.enabled:
        logger.info("Imputation disabled; using rows with complete cases only")
        full_imputed = dataset.full_df.dropna().copy()
        if full_imputed.empty:
            raise RuntimeError(
                "imputation.enabled=false requires complete-case rows, but every row has "
                "at least one missing feature value (0 complete cases out of "
                f"{len(dataset.full_df)}). Set imputation.enabled: true in the config."
            )
        # full_imputed is a complete-case subset of full_df, so intersect
        # rather than assume every train/test row survived.
        train_imputed = full_imputed.loc[full_imputed.index.intersection(dataset.train_df.index)]
        test_imputed = full_imputed.loc[full_imputed.index.intersection(dataset.test_df.index)]
        if len(train_imputed) < len(dataset.train_df) or len(test_imputed) < len(dataset.test_df):
            logger.info(
                "Complete-case filtering dropped train %d->%d, test %d->%d rows",
                len(dataset.train_df),
                len(train_imputed),
                len(dataset.test_df),
                len(test_imputed),
            )
    else:
        device = resolve_device(cfg.imputation.device)
        n_missing = int(dataset.full_df[dataset.feature_columns].isna().sum().sum())
        logger.info(
            "Imputing %d missing values across %d feature columns via method=%s on device=%s "
            "(fitted on the %d train rows outside the tuning split for HPO: %s; then on all "
            "%d train rows for the final models and the holdout)",
            n_missing,
            len(dataset.feature_columns),
            cfg.imputation.method,
            device,
            len(dataset.search_train_df),
            "yes" if runs_search_phase(cfg, dataset) else "skipped, HPO is off",
            len(dataset.train_df),
        )
        train_imputed, test_imputed, search_imputed = (
            None
            if frame is None
            else apply_rounding(
                frame,
                dataset.feature_columns,
                cfg.imputation.round_rules,
                cfg.imputation.round_to_int_default,
            )
            for frame in _impute_splits(cfg, dataset, device)
        )
        full_imputed = pd.concat([train_imputed, test_imputed]).loc[dataset.full_df.index]
        if search_imputed is not None:
            _report_drift(cfg, dataset, search_imputed, train_imputed)

    ensure_dir(paths["full_imputed"].parent)
    full_imputed.to_csv(paths["full_imputed"], index=False)
    train_imputed.to_csv(paths["train_imputed"], index=False)
    test_imputed.to_csv(paths["test_imputed"], index=False)
    if search_imputed is not None:
        ensure_dir(paths["search_imputed"].parent)
        search_imputed.to_csv(paths["search_imputed"], index=False)
    else:
        paths["search_imputed"].unlink(missing_ok=True)
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
    dataset.search_imputed_df = search_imputed
    _persist_decoded_imputed_splits(dataset)
    return dataset


def _report_drift(
    cfg: Config, dataset: Dataset, search_imputed: pd.DataFrame, train_imputed: pd.DataFrame
) -> None:
    """Write ``imputation_drift.csv`` and warn about columns above the threshold."""
    drift = imputation_drift(dataset, search_imputed, train_imputed)
    drift_path = imputation_output_dir(cfg) / DRIFT_REPORT_FILENAME
    drift.to_csv(drift_path, index=False)
    high = drift[drift["drift"] > cfg.imputation.drift_warn_threshold]
    if not high.empty:
        logger.warning(
            "Imputed values differ between the HPO fit and the final fit by more than %.2f "
            "in %d column(s): %s. Hyperparameters were tuned on the first fit's data; see %s",
            cfg.imputation.drift_warn_threshold,
            len(high),
            high.sort_values("drift", ascending=False)["column"].head(10).tolist(),
            drift_path,
        )


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


def save_validation_report(cfg: Config, dataset: Dataset) -> pd.DataFrame:
    """Write :func:`build_validation_report` to a CSV and log a short summary.

    The full table goes to ``imputation_validation_report.csv`` under
    :func:`~synthdata.experiment.imputation_output_dir`; the log gets only the
    totals and the columns whose imputed values fall outside the observed
    range or category set.
    """
    report = build_validation_report(cfg, dataset)
    path = imputation_output_dir(cfg) / VALIDATION_REPORT_FILENAME
    report.to_csv(path, index=False)
    if report.empty:
        logger.info("Imputation validation: no feature had missing values. Report: %s", path)
        return report
    failing = report[~report["all_valid"]]
    logger.info(
        "Imputation validation: %d imputed cell(s) across %d column(s); %d of those cells "
        "(%.1f%%) are within the observed range or category set; %d column(s) have "
        "implausible values%s. Full report: %s",
        int(report["n_imputed"].sum()),
        len(report),
        int(report["n_valid"].sum()),
        100 * report["n_valid"].sum() / max(int(report["n_imputed"].sum()), 1),
        len(failing),
        f" ({', '.join(failing['column'].astype(str).head(10))}"
        f"{', ...' if len(failing) > 10 else ''})"
        if len(failing)
        else "",
        path,
    )
    return report
