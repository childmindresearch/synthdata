"""Role-isolated HyperImpute pipeline with legacy two-role compatibility.

Canonical roles use fixed HyperImpute plugins. Legacy TabImpute and RefiDiff
remain available only for explicit two-role datasets.
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
from synthdata.data_roles import ROLE_NAMES
from synthdata.utils import ensure_dir, get_logger, resolve_device

logger = get_logger(__name__)


class RoleIsolationError(RuntimeError):
    """Raised when an imputation backend cannot honor canonical role isolation."""


#: Sidecar filename (under ``dataset.data_dir``) recording the config fields that
#: determined the currently-cached imputed CSVs -- see :func:`_cache_key_record`.
_CACHE_KEY_FILENAME = IMPUTATION_CACHE_KEY_FILENAME


def _persist_decoded_imputed_splits(dataset: Dataset) -> None:
    """Persist label-preserving views alongside numeric model-space caches."""
    dataset.attach_decoded_imputed_splits()
    decoded_frames = {"full_imputed_decoded": dataset.full_imputed_decoded_df}
    decoded_frames.update(
        {f"{role}_imputed_decoded": dataset.decoded_roles.get(role) for role in dataset.roles}
    )
    if dataset.legacy_two_role:
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
        "role-specific imputed CSVs)",
        dataset.data_dir,
    )


def _impute_dataframe(cfg: Config, df: pd.DataFrame, dataset: Dataset, device: str) -> pd.DataFrame:
    """Dispatch to the configured imputation backend's ``impute_dataframe``."""
    method = cfg.imputation.method
    if method == "hyperimpute":
        from synthdata.imputation.hyperimpute_backend import fit_dataframe, transform_dataframe

        state = fit_dataframe(
            df,
            dataset.feature_columns,
            dataset.categorical_columns,
            fit_roles=("train", "final_holdout"),
            continuous_plugin=getattr(cfg.imputation, "continuous_plugin", "median"),
            random_state=cfg.seed,
        )
        return transform_dataframe(state, df)
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
    # Unreachable in practice: Config._validate() restricts method names.
    raise ValueError(f"Unknown imputation.method: {method!r}")


def _impute_canonical_roles(
    cfg: Config, dataset: Dataset, device: str, phase: str = "candidate"
) -> tuple[dict[str, pd.DataFrame], dict | None]:
    """Run candidate (train fit) or final (train+tuning fit) role isolation."""
    dataset.require_canonical_roles("canonical imputation")
    role_frames = dataset.roles
    if phase not in {"candidate", "final"}:
        raise ValueError("canonical imputation phase must be 'candidate' or 'final'")
    n_missing = {
        role: int(frame[dataset.feature_columns].isna().sum().sum())
        for role, frame in role_frames.items()
    }
    if cfg.imputation.method in {"tabimpute", "refidiff"}:
        raise RoleIsolationError(
            f"Canonical imputation method {cfg.imputation.method!r} is deferred: "
            "TabImpute/RefiDiff are not supported for role-isolated execution. "
            "Use fixed HyperImpute plugins with a non-legacy method."
        )

    if not any(n_missing.values()):
        logger.info("Canonical roles contain no missing feature values; imputation is a no-op")
        from synthdata.imputation.hyperimpute_backend import state_metadata as hyper_state_metadata

        fit_frame = (
            role_frames["train"]
            if phase == "candidate"
            else pd.concat([role_frames["train"], role_frames["tuning"]], axis=0)
        )
        state_metadata = hyper_state_metadata(
            None,
            status="not_required",
            transform_roles=(["train", "tuning"] if phase == "candidate" else ["final_holdout"]),
            fit_roles=["train"] if phase == "candidate" else ["train", "tuning"],
            fit_frame_fingerprint=dataframe_fingerprint(fit_frame),
            feature_columns=dataset.feature_columns,
            categorical_columns=dataset.categorical_columns,
            continuous_plugin=cfg.imputation.continuous_plugin,
        )
        return {role: frame.copy() for role, frame in role_frames.items()}, state_metadata

    from synthdata.imputation.hyperimpute_backend import fit_dataframe, transform_dataframe
    from synthdata.imputation.hyperimpute_backend import state_metadata as hyper_state_metadata

    fit_frame = (
        role_frames["train"]
        if phase == "candidate"
        else pd.concat([role_frames["train"], role_frames["tuning"]], axis=0)
    )
    state = fit_dataframe(
        fit_frame,
        dataset.feature_columns,
        dataset.categorical_columns,
        fit_roles=("train",) if phase == "candidate" else ("train", "tuning"),
        continuous_plugin=getattr(cfg.imputation, "continuous_plugin", "median"),
        random_state=cfg.seed,
    )
    if phase == "candidate":
        transformed = {
            role: transform_dataframe(state, role_frames[role]) for role in ("train", "tuning")
        }
        transformed["final_holdout"] = role_frames["final_holdout"].copy()
        roles = ["train", "tuning"]
    else:
        transformed = {role: role_frames[role].copy() for role in ("train", "tuning")}
        transformed["final_holdout"] = transform_dataframe(state, role_frames["final_holdout"])
        roles = ["final_holdout"]
    return transformed, hyper_state_metadata(state, transform_roles=roles)


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


def _cache_key_payload(cfg: Config, dataset: Dataset, phase: str = "candidate") -> dict:
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
        "continuous_plugin": imp_cfg.continuous_plugin,
        "categorical_plugin": "most_frequent",
        "dataset_version": dataset.version,
        "variable_schema_fingerprint": dataset.variable_schema_fingerprint,
        "source_fingerprint": dataset.source_fingerprint,
        "full_fingerprint": dataframe_fingerprint(dataset.full_df),
    }
    if dataset.has_canonical_roles:
        payload.update(
            {
                "cache_contract": "canonical_roles_v1",
                "role_names": list(ROLE_NAMES),
                "role_fingerprints": dataset.role_fingerprints,
                "assignment_fingerprint": dataset.assignment_fingerprint,
                "assignment_policy_fingerprint": dataset.assignment_policy_fingerprint,
                "identity_fingerprint": dataset.role_metadata.get("identity", {}).get(
                    "identity_fingerprint"
                ),
                "semantic_fingerprint": dataset.semantic_fingerprint,
                "phase": phase,
                "fit_roles": ["train"] if phase == "candidate" else ["train", "tuning"],
                "transform_roles": ["train", "tuning"]
                if phase == "candidate"
                else ["final_holdout"],
                "fit_frame_fingerprint": dataframe_fingerprint(
                    dataset.roles["train"]
                    if phase == "candidate"
                    else pd.concat([dataset.roles["train"], dataset.roles["tuning"]], axis=0)
                ),
                "role_row_counts": {role: len(dataset.roles[role]) for role in ROLE_NAMES},
            }
        )
    else:
        payload.update(
            {
                "cache_contract": "legacy_two_role_v1",
                "train_split_fingerprint": dataframe_fingerprint(dataset.train_df),
                "test_split_fingerprint": dataframe_fingerprint(dataset.test_df),
            }
        )
    if imp_cfg.method == "refidiff":
        payload["refidiff"] = dataclasses.asdict(imp_cfg.refidiff)
    if imp_cfg.method == "hyperimpute" and not dataset.has_canonical_roles:
        payload["fit_roles"] = ["train", "final_holdout"]
        payload["transform_roles"] = ["train", "final_holdout"]
    return payload


def _cache_key_record(cfg: Config, dataset: Dataset, phase: str = "candidate") -> dict:
    """``_cache_key_payload`` plus its own sha256 digest under ``"cache_key"``."""
    payload = _cache_key_payload(cfg, dataset, phase)
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
    record = _load_cache_record(path)
    return record.get("cache_key") if record is not None else None


def _load_cache_record(path: Path) -> dict | None:
    """Read a cache sidecar record, returning None for absent/corrupt metadata."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            record = json.load(f)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Failed to parse imputation cache-key file %s (%s); treating cached imputed data "
            "as stale and retraining",
            path,
            exc,
        )
        return None
    if not isinstance(record, dict):
        logger.warning(
            "Imputation cache-key file %s does not contain a JSON object; treating cached "
            "imputed data as stale and retraining",
            path,
        )
        return None
    return record


def _canonical_hyperimpute_state_is_valid(
    cfg: Config, dataset: Dataset, cached_record: dict | None, phase: str = "candidate"
) -> bool:
    """Require an intact train-fitted state record for canonical HyperImpute caches."""
    if not dataset.has_canonical_roles or cfg.imputation.method != "hyperimpute":
        return True
    state = cached_record.get("fit_state") if cached_record is not None else None
    if not isinstance(state, dict):
        return False
    if state.get("status") not in {"fitted", "not_required", "disabled"}:
        return False
    expected_fit_roles = ["train"] if phase == "candidate" else ["train", "tuning"]
    expected_transform_roles = ["train", "tuning"] if phase == "candidate" else ["final_holdout"]
    if state.get("backend") != "hyperimpute" or state.get("fit_roles") != expected_fit_roles:
        return False
    if state.get("transform_roles") != expected_transform_roles:
        return False
    if state.get("continuous_plugin") != cfg.imputation.continuous_plugin:
        return False
    if state.get("categorical_plugin") != "most_frequent":
        return False
    fit_frame = (
        dataset.roles["train"]
        if phase == "candidate"
        else pd.concat([dataset.roles["train"], dataset.roles["tuning"]], axis=0)
    )
    if state.get("fit_frame_fingerprint") != dataframe_fingerprint(fit_frame):
        return False
    if state.get("feature_columns") != list(dataset.feature_columns):
        return False
    if state.get("categorical_columns") != list(dataset.categorical_columns):
        return False
    from synthdata.imputation.hyperimpute_backend import metadata_fingerprint

    return state.get("state_fingerprint") == metadata_fingerprint(state)


def run_imputation(cfg: Config, dataset: Dataset, phase: str = "candidate") -> Dataset:
    """Impute dataset roles and populate the role-specific imputed frames.

    Canonical datasets cache ``train_imputed.csv``/``tuning_imputed.csv``/
    ``final_holdout_imputed.csv``; legacy datasets retain their historical
    ``full_imputed.csv``/``train_imputed.csv``/``test_imputed.csv`` cache. Caches are
    reused on subsequent runs unless ``cfg.imputation.cache``
    is False. Reuse also requires the cache-key sidecar file
    ``.imputation_cache_key.json``, also under ``data_dir``) to match a fresh
    hash of the current config's imputation-relevant fields and exact
    source/split fingerprints (see :func:`_cache_key_payload`) -- so editing
    e.g. ``nominal_columns``/``ordinal_columns`` or refreshing the source and
    rerunning correctly retrains instead of silently reusing stale imputed
    CSVs from before the change.
    """
    if dataset.has_canonical_roles and cfg.imputation.method in {"tabimpute", "refidiff"}:
        raise RoleIsolationError(
            f"Canonical imputation method {cfg.imputation.method!r} is deferred: "
            "TabImpute/RefiDiff are not supported for role-isolated execution. "
            "Use canonical method='hyperimpute'."
        )
    paths = dataset.paths()
    cache_key_path = dataset.data_dir / _CACHE_KEY_FILENAME
    cache_record = _cache_key_record(cfg, dataset, phase)
    current_key = cache_record["cache_key"]
    cached_record = _load_cache_record(cache_key_path)
    cached_key = cached_record.get("cache_key") if cached_record is not None else None

    cached_paths = (
        [paths[f"{role}_imputed"] for role in ROLE_NAMES]
        if dataset.has_canonical_roles
        else [paths["full_imputed"], paths["train_imputed"], paths["test_imputed"]]
    )
    cached_csvs_exist = all(path.exists() for path in cached_paths)

    if cfg.imputation.cache and cached_csvs_exist and cached_key == current_key:
        if _canonical_hyperimpute_state_is_valid(cfg, dataset, cached_record, phase):
            dataset = load_imputed_splits(dataset, expected_cache_key=current_key)
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
        else:
            logger.warning(
                "Canonical HyperImpute cache at %s lacks valid phase-fitted state provenance; "
                "retraining",
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

    fit_state_metadata = None
    if not cfg.imputation.enabled and dataset.has_canonical_roles:
        missing_by_role = {
            role: int(frame[dataset.feature_columns].isna().sum().sum())
            for role, frame in dataset.roles.items()
        }
        if any(missing_by_role.values()):
            raise RoleIsolationError(
                "imputation.enabled=false cannot produce canonical role frames with missing "
                f"features; enable imputation or provide complete roles. missing_values_by_role="
                f"{missing_by_role}"
            )
        role_imputed = {role: frame.copy() for role, frame in dataset.roles.items()}
        dataset.set_imputed_roles(role_imputed)
        full_imputed = dataset.full_imputed_df
        if cfg.imputation.method == "hyperimpute":
            from synthdata.imputation.hyperimpute_backend import state_metadata

            fit_frame = (
                dataset.roles["train"]
                if phase == "candidate"
                else pd.concat([dataset.roles["train"], dataset.roles["tuning"]])
            )
            fit_state_metadata = state_metadata(
                None,
                status="disabled",
                fit_roles=["train"] if phase == "candidate" else ["train", "tuning"],
                fit_frame_fingerprint=dataframe_fingerprint(fit_frame),
                feature_columns=dataset.feature_columns,
                categorical_columns=dataset.categorical_columns,
                continuous_plugin=cfg.imputation.continuous_plugin,
                transform_roles=["train", "tuning"] if phase == "candidate" else ["final_holdout"],
            )
    elif not cfg.imputation.enabled:
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
        if dataset.has_canonical_roles:
            role_imputed, fit_state_metadata = _impute_canonical_roles(cfg, dataset, device, phase)
            role_imputed = {
                role: apply_rounding(
                    frame,
                    dataset.feature_columns,
                    cfg.imputation.round_rules,
                    cfg.imputation.round_to_int_default,
                )
                for role, frame in role_imputed.items()
            }
            dataset.set_imputed_roles(role_imputed)
            full_imputed = dataset.full_imputed_df
        else:
            full_imputed = _impute_dataframe(cfg, dataset.full_df, dataset, device)
            full_imputed = apply_rounding(
                full_imputed,
                dataset.feature_columns,
                cfg.imputation.round_rules,
                cfg.imputation.round_to_int_default,
            )

    if full_imputed is None:
        raise RuntimeError("Imputation did not produce a full model-space frame")

    ensure_dir(dataset.data_dir)
    full_imputed.to_csv(paths["full_imputed"], index=False)

    if dataset.has_canonical_roles:
        for role in ROLE_NAMES:
            dataset.imputed_roles[role].to_csv(paths[f"{role}_imputed"], index=False)
        if cfg.imputation.method == "hyperimpute":
            if fit_state_metadata is None:
                raise RuntimeError(
                    "Canonical HyperImpute imputation completed without fitted-state provenance"
                )
            cache_record["fit_state"] = fit_state_metadata
        cache_record["imputed_row_counts"] = {
            "full": len(full_imputed),
            **{role: len(dataset.imputed_roles[role]) for role in ROLE_NAMES},
        }
    else:
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

    if dataset.has_canonical_roles:
        dataset.full_imputed_df = full_imputed
    else:
        dataset.full_imputed_df = full_imputed
        dataset.train_imputed_df = train_imputed
        dataset.test_imputed_df = test_imputed
        dataset.imputed_roles = {
            "train": train_imputed,
            "final_holdout": test_imputed,
        }
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
