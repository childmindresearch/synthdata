"""Append-only, train-only masked-cell validation helpers.

This module never writes ordinary ``*_imputed.csv`` cache files. It hides only
originally observed, non-sensitive feature cells in ``Dataset.train_df``, then
scores a candidate solely on those held-out cells. Every mask and candidate
result is persisted under a study directory so interrupted Optuna studies can
resume without regenerating masks or overwriting prior evidence.
"""

import dataclasses
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from synthdata.config import Config
from synthdata.data import Dataset
from synthdata.experiment import dataset_version_scope
from synthdata.utils import ensure_dir, get_logger

logger = get_logger(__name__)


def _study_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ_refidiff_benchmark")


def benchmark_study_dir(cfg: Config, study_id: str) -> Path:
    """Return a version-scoped, append-only RefiDiff benchmark directory."""
    if not study_id or Path(study_id).name != study_id or study_id in {".", ".."}:
        raise ValueError(
            f"Benchmark study id must be a non-empty path-safe label, got {study_id!r}."
        )
    return (
        Path(cfg.imputation.benchmark.output_dir)
        / dataset_version_scope(cfg)
        / f"benchmark_{study_id}"
    )


def _source_fingerprint(dataset: Dataset) -> str:
    frame_hash = np.asarray(
        pd.util.hash_pandas_object(dataset.train_df, index=True), dtype=np.uint64
    ).tobytes()
    return hashlib.sha256(frame_hash).hexdigest()


def _study_identity(cfg: Config, dataset: Dataset) -> dict:
    return {
        "algorithm": "refidiff-masked-benchmark-v1",
        "dataset_name": dataset.name,
        "dataset_version": dataset.version,
        "source_train_sha256": _source_fingerprint(dataset),
        "schema_sha256": dataset.variable_schema_fingerprint,
        "feature_columns": dataset.feature_columns,
        "categorical_columns": dataset.categorical_columns,
        "sensitive_columns_excluded": dataset.sensitive_columns,
        "seed": cfg.seed,
        "mask_fraction": cfg.imputation.benchmark.mask_fraction,
        "n_masks": cfg.imputation.benchmark.n_masks,
        "mechanisms": cfg.imputation.benchmark.mechanisms,
        "score_columns": resolve_score_columns(dataset, cfg.imputation.benchmark.score_columns),
    }


def _write_or_validate_identity(study_dir: Path, identity: dict, cfg: Config) -> None:
    ensure_dir(study_dir)
    identity_path = study_dir / "identity.json"
    if identity_path.exists():
        with open(identity_path) as f:
            recorded = json.load(f)
        if recorded != identity:
            raise RuntimeError(
                f"Refusing to resume benchmark at {study_dir}: data, schema, mask protocol, or "
                "seed differs from its immutable identity. Start a new --study-id instead."
            )
        return
    with open(identity_path, "w") as f:
        json.dump(identity, f, indent=2, sort_keys=True)
    with open(study_dir / "config_snapshot.json", "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2, sort_keys=True, default=str)


def _mask_count(n_eligible: int, fraction: float) -> int:
    if n_eligible < 2:
        return 0
    return min(max(1, round(n_eligible * fraction)), n_eligible - 1)


def _sampling_weights(
    df: pd.DataFrame,
    feature_columns: list,
    column: str,
    eligible_rows: np.ndarray,
    mechanism: str,
) -> np.ndarray | None:
    """Return deterministic MAR/MNAR sampling weights for one feature.

    MAR uses the first other numeric feature with sufficient observations;
    MNAR uses the held-out column itself. Both are only masking mechanisms:
    the imputer receives neither the held-out values nor the weights.
    """
    if mechanism == "mcar":
        return None
    source = df[column] if mechanism == "mnar" else None
    if source is None:
        for candidate in feature_columns:
            if candidate != column and pd.api.types.is_numeric_dtype(df[candidate]):
                source = df[candidate]
                break
    if source is None:
        return None
    values = source.iloc[eligible_rows]
    if not pd.api.types.is_numeric_dtype(values):
        values = pd.Series(pd.factorize(values.astype(str))[0], index=values.index)
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or np.nanstd(numeric) == 0:
        return None
    logits = np.clip((numeric - np.nanmean(numeric)) / np.nanstd(numeric), -4, 4)
    weights = 1 / (1 + np.exp(-logits))
    return weights / weights.sum()


def create_artificial_mask(
    df: pd.DataFrame,
    feature_columns: list,
    sensitive_columns: list,
    fraction: float,
    mechanism: str,
    seed: int,
) -> dict[str, list[int]]:
    """Select observed, non-sensitive feature rows to hide without replacement."""
    if mechanism not in {"mcar", "mar", "mnar"}:
        raise ValueError(f"Unknown artificial masking mechanism: {mechanism!r}")
    rng = np.random.default_rng(seed)
    mask: dict[str, list[int]] = {}
    for column in feature_columns:
        if column in sensitive_columns:
            continue
        eligible_rows = np.flatnonzero(df[column].notna().to_numpy())
        n_mask = _mask_count(len(eligible_rows), fraction)
        if n_mask == 0:
            continue
        weights = _sampling_weights(df, feature_columns, column, eligible_rows, mechanism)
        selected = rng.choice(eligible_rows, size=n_mask, replace=False, p=weights)
        mask[column] = sorted(int(row) for row in selected)
    return mask


def resolve_score_columns(dataset: Dataset, configured_columns: list | None) -> list:
    """Validate the score panel while retaining all features as imputer context."""
    allowed = [
        column for column in dataset.feature_columns if column not in dataset.sensitive_columns
    ]
    if configured_columns is None:
        return allowed
    unknown = sorted(set(configured_columns) - set(allowed))
    if unknown:
        raise ValueError(
            "imputation.benchmark.score_columns must contain non-sensitive feature columns only; "
            f"invalid entries: {unknown}"
        )
    return list(configured_columns)


def apply_artificial_mask(df: pd.DataFrame, mask: dict[str, list[int]]) -> pd.DataFrame:
    """Return a copy of ``df`` with exactly the persisted benchmark cells hidden."""
    masked = df.copy()
    for column, rows in mask.items():
        masked.loc[masked.index[rows], column] = np.nan
    return masked


def _column_metrics(
    truth: pd.Series, predicted: pd.Series, observed: pd.Series, categorical: bool
) -> dict[str, float]:
    if categorical:
        return {
            "accuracy": float(accuracy_score(truth, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
            "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        }
    truth_values = pd.to_numeric(truth, errors="raise").to_numpy(dtype=float)
    predicted_values = pd.to_numeric(predicted, errors="raise").to_numpy(dtype=float)
    observed_values = pd.to_numeric(observed, errors="raise").dropna().to_numpy(dtype=float)
    if observed_values.size == 0:
        raise ValueError(
            "Cannot standardize a numeric benchmark column with no unmasked observations."
        )
    scale = float(np.std(observed_values))
    if scale == 0:
        scale = 1.0
    error = predicted_values - truth_values
    return {
        "standardized_mae": float(np.mean(np.abs(error)) / scale),
        "standardized_rmse": float(np.sqrt(np.mean(error**2)) / scale),
    }


def score_masked_cells(
    truth: pd.DataFrame,
    imputed: pd.DataFrame,
    mask: dict[str, list[int]],
    categorical_columns: list,
) -> tuple[pd.DataFrame, dict]:
    """Score only persisted artificially hidden cells, macro-aggregated by column."""
    rows = []
    for column, positions in mask.items():
        row_labels = truth.index[positions]
        categorical = column in categorical_columns
        observed = truth.loc[truth.index.difference(row_labels), column]
        metrics = _column_metrics(
            truth.loc[row_labels, column], imputed.loc[row_labels, column], observed, categorical
        )
        rows.append(
            {"column": column, "categorical": categorical, "n_masked": len(positions), **metrics}
        )
    metrics_df = pd.DataFrame(rows)
    numeric = metrics_df.loc[~metrics_df["categorical"]] if len(metrics_df) else metrics_df
    categorical = metrics_df.loc[metrics_df["categorical"]] if len(metrics_df) else metrics_df
    components = []
    summary: dict[str, float | int] = {
        "n_columns": len(metrics_df),
        "n_masked": int(metrics_df["n_masked"].sum()) if len(metrics_df) else 0,
    }
    if len(numeric):
        summary["numeric_macro_standardized_mae"] = float(numeric["standardized_mae"].mean())
        summary["numeric_macro_standardized_rmse"] = float(numeric["standardized_rmse"].mean())
        components.append(summary["numeric_macro_standardized_rmse"])
    if len(categorical):
        summary["categorical_macro_accuracy"] = float(categorical["accuracy"].mean())
        summary["categorical_macro_balanced_accuracy"] = float(
            categorical["balanced_accuracy"].mean()
        )
        summary["categorical_macro_f1"] = float(categorical["macro_f1"].mean())
        components.append(1 - summary["categorical_macro_balanced_accuracy"])
    if not components:
        raise ValueError("Artificial mask contains no scoreable feature cells.")
    summary["objective"] = float(np.mean(components))
    return metrics_df, summary


def _persist_mask(
    study_dir: Path, mechanism: str, mask_index: int, mask: dict[str, list[int]]
) -> Path:
    path = ensure_dir(study_dir / "masks") / f"{mechanism}_{mask_index:02d}.json"
    if path.exists():
        with open(path) as f:
            recorded = json.load(f)
        if recorded != mask:
            raise RuntimeError(f"Persisted mask differs from deterministic regeneration: {path}")
    else:
        with open(path, "w") as f:
            json.dump(mask, f, indent=2, sort_keys=True)
    return path


def run_refidiff_benchmark(cfg: Config, dataset: Dataset, study_id: str | None = None) -> Path:
    """Reject deferred RefiDiff benchmark execution explicitly."""
    raise RuntimeError(
        "RefiDiff benchmark is deferred and blocked: legacy RefiDiff execution is not "
        "available in role-isolated Task 6. No study was started."
    )
