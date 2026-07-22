"""Generic dataset loading, typing, and splitting.

Supports two data sources so a collaborator can point the pipeline at their own data:

- ``source: uci``: fetch (and locally cache) a dataset from the UCI ML repository by id.
- ``source: csv``: load a local CSV file directly.

The same :class:`Dataset` object is produced either way and consumed by every
downstream stage (imputation, generation, evaluation, plotting).
"""

import dataclasses
import json
import types
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from synthdata.config import Config
from synthdata.utils import ensure_dir, get_logger, git_commit

logger = get_logger(__name__)


@dataclasses.dataclass
class Dataset:
    """Container for a loaded dataset plus derived metadata used by every stage."""

    name: str
    target_column: str
    feature_columns: list
    categorical_columns: list
    sensitive_columns: list
    data_dir: Path

    #: Full dataset, possibly containing missing values (pre-imputation).
    full_df: pd.DataFrame
    #: Train/test split of full_df (same rows as the imputed splits, pre-imputation).
    train_df: pd.DataFrame
    test_df: pd.DataFrame

    #: Freeform dataset version label (see DataConfig.version), recorded in
    #: experiment manifests for traceability. None if not set by the user.
    version: str | None = None

    #: Populated once imputation has run (see synthdata.imputation).
    full_imputed_df: pd.DataFrame | None = None
    train_imputed_df: pd.DataFrame | None = None
    test_imputed_df: pd.DataFrame | None = None

    @property
    def all_categorical_columns(self) -> list:
        """Categorical feature columns plus the target column."""
        return list(self.categorical_columns) + [self.target_column]

    def paths(self) -> dict:
        d = self.data_dir
        return {
            "full": d / "full.csv",
            "train": d / "train.csv",
            "test": d / "test.csv",
            "full_imputed": d / "full_imputed.csv",
            "train_imputed": d / "train_imputed.csv",
            "test_imputed": d / "test_imputed.csv",
        }


# ---------------------------------------------------------------------------
# UCI loading (with local caching, mirrors the notebook's fetch_ucirepo pattern)
# ---------------------------------------------------------------------------


def _fetch_uci_dataset(uci_id: int, cache_dir: Path):
    """Fetch a UCI dataset, caching features/targets/variables/metadata locally."""
    cache_files = {
        "features": cache_dir / "features.csv",
        "targets": cache_dir / "targets.csv",
        "variables": cache_dir / "variables.csv",
        "metadata": cache_dir / "metadata.json",
    }

    if all(p.exists() for p in cache_files.values()):
        logger.info("Loading cached UCI dataset id=%s from %s", uci_id, cache_dir)
        features = pd.read_csv(cache_files["features"])
        targets = pd.read_csv(cache_files["targets"])
        variables = pd.read_csv(cache_files["variables"])
        with open(cache_files["metadata"]) as f:
            metadata = json.load(f)
        data_ns = types.SimpleNamespace(features=features, targets=targets)
        return types.SimpleNamespace(data=data_ns, variables=variables, metadata=metadata)

    from ucimlrepo import fetch_ucirepo

    logger.info("Fetching UCI dataset id=%s (not cached)", uci_id)
    repo = fetch_ucirepo(id=uci_id)

    ensure_dir(cache_dir)
    repo.data.features.to_csv(cache_files["features"], index=False)
    repo.data.targets.to_csv(cache_files["targets"], index=False)
    repo.variables.to_csv(cache_files["variables"], index=False)
    metadata = repo.metadata if isinstance(repo.metadata, dict) else dict(repo.metadata)
    with open(cache_files["metadata"], "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return repo


def _load_uci(cfg: Config, data_dir: Path) -> tuple:
    repo = _fetch_uci_dataset(cfg.data.uci_id, data_dir / cfg.data.raw_cache_subdir)
    df = pd.concat([repo.data.features, repo.data.targets], axis=1)
    variable_types = None
    if hasattr(repo, "variables") and repo.variables is not None:
        variable_types = dict(zip(repo.variables["name"], repo.variables["type"], strict=True))
    return df, variable_types


def _load_csv(cfg: Config) -> tuple:
    df = pd.read_csv(cfg.data.path)
    return df, None


# ---------------------------------------------------------------------------
# Column typing helpers
# ---------------------------------------------------------------------------


def infer_categorical_columns(
    df: pd.DataFrame,
    feature_columns: list,
    explicit: str | list,
    unique_threshold: int = 10,
    uci_variable_types: dict | None = None,
) -> list:
    """Determine which feature columns should be treated as categorical.

    Resolution order:
        1. An explicit list of column names in the config always wins.
        2. If UCI variable metadata is available, use its "Categorical" tag.
        3. Otherwise fall back to a dtype/cardinality heuristic
           (object/category/bool dtype, or nunique <= unique_threshold).
    """
    if isinstance(explicit, list):
        return [c for c in explicit if c in feature_columns]

    if uci_variable_types is not None:
        cats = [c for c in feature_columns if uci_variable_types.get(c) == "Categorical"]
        if cats:
            return cats

    cats = []
    for c in feature_columns:
        dtype = df[c].dtype
        if (
            dtype in (object, bool)
            or str(dtype) == "category"
            or df[c].nunique(dropna=True) <= unique_threshold
        ):
            cats.append(c)
    return cats


def remap_binary_one_two(df: pd.DataFrame) -> pd.DataFrame:
    """Remap any column whose only non-null unique values are {1, 2} to {0, 1}."""
    out = df.copy()
    binary_cols = [c for c in out.columns if set(out[c].dropna().unique().tolist()) == {1, 2}]
    if binary_cols:
        out[binary_cols] = out[binary_cols] - 1
    return out


def cast_integer_like_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Cast fully-observed, whole-numbered columns to int dtype (no-op otherwise).

    Some libraries (e.g. SynthEval's ``AnalysisConfig``) infer "categorical" from
    dtype rather than cardinality, so a numerically-binary column stored as
    float (e.g. {0.0, 1.0}) would silently be treated as continuous downstream.
    """
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            continue
        series = out[c]
        if series.isna().any() or not pd.api.types.is_numeric_dtype(series):
            continue
        if np.all(np.mod(series, 1) == 0):
            out[c] = series.astype(int)
    return out


def label_encode_non_numeric_columns(df: pd.DataFrame, columns: list) -> tuple[pd.DataFrame, dict]:
    """Factorize any non-numeric (e.g. string-valued) columns to integer codes.

    Some backends (``TabImputeCategorical``, TabPFN) require a fully numeric
    matrix, but a plain CSV source (unlike the pre-encoded UCI hepatitis
    example) commonly has string-valued categorical columns (e.g.
    "Light"/"Heavy"/...). Missing values are preserved as NaN so they're still
    treated as missing rather than a category. Returns the encoded frame plus
    ``{column: categories}``, needed to decode output back to the original
    labels via :func:`decode_label_encoded_columns`. Already-numeric columns
    pass through unchanged.
    """
    encoded = df[columns].copy()
    category_maps = {}
    for col in columns:
        if pd.api.types.is_numeric_dtype(encoded[col]):
            continue
        codes, categories = pd.factorize(encoded[col], sort=True)
        codes = codes.astype(float)
        codes[codes == -1] = np.nan  # factorize maps NaN -> -1
        encoded[col] = codes
        category_maps[col] = categories
    return encoded, category_maps


def decode_label_encoded_columns(df: pd.DataFrame, category_maps: dict) -> pd.DataFrame:
    """Invert :func:`label_encode_non_numeric_columns`, mapping codes back to labels."""
    decoded = df.copy()
    for col, categories in category_maps.items():
        if col not in decoded.columns:
            continue
        codes = decoded[col].round().clip(0, len(categories) - 1).astype(int)
        decoded[col] = categories.take(codes)
    return decoded


def mask_outliers_as_missing(df: pd.DataFrame, columns: list, threshold: float) -> pd.DataFrame:
    """Set numeric values beyond ``threshold`` std-devs of their column mean to NaN.

    Plain (non-robust) mean/std, not a robust median/MAD-based z-score: many of
    this kind of column are zero-/mode-inflated ordinal-ish measures (e.g. a
    day-count column with median=MAD=1 but a legitimate long tail out to 30),
    for which MAD-based z-scores false-positive heavily on real boundary values
    (confirmed empirically) while under-flagging true outliers whenever the
    "bulk" of the column has zero MAD (all-too-common for zero-inflated
    columns). Plain std is itself inflated by genuine outliers, but for a
    single (or few) extreme value(s) among ``n`` otherwise-plausible ones its
    z-score stays roughly ``sqrt(n)`` regardless of how extreme the value is,
    which is more than enough separation at this dataset's scale. Catches both
    "not administered" sentinel codes (e.g. a lone 999 among otherwise 0-30
    values) and corrupt outlier rows (e.g. a derived metric blown up by a
    division artifact), either of which can otherwise cause float32 overflow
    inside TabPFN/TabImpute. Non-numeric and constant (zero-std) columns are
    left untouched.

    Deliberately a single pass (not iterative): re-fitting mean/std after each
    removal and repeating would catch smaller residual outliers, but also
    cascades into masking legitimate boundary values (confirmed empirically --
    e.g. removing a 999 sentinel from a 0-31 day-count column shrinks std
    enough that a legitimate, repeated 30 then looks like an "outlier" too).
    A single pass only removes the most egregious values -- exactly what's
    needed to avoid literal float32 infinity/overflow -- and leaves smaller
    (still-plausible) residual outliers alone.
    """
    out = df.copy()
    for col in columns:
        if col not in out.columns or not pd.api.types.is_numeric_dtype(out[col]):
            continue
        series = out[col]
        mean = series.mean()
        std = series.std()
        if not std or np.isnan(std):
            continue
        z = (series - mean).abs() / std
        outliers = z > threshold
        if outliers.any():
            logger.info(
                "Masking %d outlier value(s) in %r as missing (|z| > %.1f, e.g. %s)",
                int(outliers.sum()),
                col,
                threshold,
                series[outliers].tolist()[:5],
            )
            out.loc[outliers, col] = np.nan
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def write_dataset_manifest(cfg: Config, dataset: Dataset) -> None:
    """Record which dataset version/source produced ``dataset.data_dir``.

    Unlike experiments (see :mod:`synthdata.experiment`), which version each
    generation/evaluation/plot *run*, this manifest versions the *dataset
    itself*: it is written once per `data_dir` (i.e. once per `data.version`)
    and updated (timestamp/commit refreshed) on every subsequent load, so a
    collaborator can tell exactly which source config produced any cached
    `data/<name>/<version>/` directory.
    """
    manifest_path = dataset.data_dir / "dataset_manifest.json"
    manifest = {
        "dataset_name": dataset.name,
        "dataset_version": dataset.version,
        "source": cfg.data.source,
        "uci_id": cfg.data.uci_id,
        "path": cfg.data.path,
        "target_column": dataset.target_column,
        "feature_columns": dataset.feature_columns,
        "categorical_columns": dataset.categorical_columns,
        "sensitive_columns": dataset.sensitive_columns,
        "n_rows": int(len(dataset.full_df)),
        "n_train": int(len(dataset.train_df)),
        "n_test": int(len(dataset.test_df)),
        "seed": cfg.seed,
        "last_loaded_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def load_dataset(cfg: Config) -> Dataset:
    """Load, type, and split the dataset described by ``cfg.data``.

    Produces (and caches to ``cfg.data.data_dir`` [nested under ``cfg.data.version``
    if set]) ``full.csv``, ``train.csv``, and ``test.csv``. These are the
    pre-imputation splits; :mod:`synthdata.imputation` later fills in
    ``*_imputed.csv`` variants aligned to the same row indices.
    """
    data_dir_base = Path(cfg.data.data_dir)
    data_dir = ensure_dir(data_dir_base / cfg.data.version if cfg.data.version else data_dir_base)

    if cfg.data.source == "uci":
        df, variable_types = _load_uci(cfg, data_dir)
    elif cfg.data.source == "csv":
        df, variable_types = _load_csv(cfg)
    else:
        raise ValueError(f"Unknown data.source: {cfg.data.source!r}")

    if cfg.data.uppercase_columns:
        df.columns = df.columns.str.upper()
        if variable_types is not None:
            variable_types = {k.upper(): v for k, v in variable_types.items()}

    if cfg.data.drop_columns:
        df = df.drop(columns=[c for c in cfg.data.drop_columns if c in df.columns])

    if cfg.data.raw_target_column and cfg.data.raw_target_column in df.columns:
        df = df.rename(columns={cfg.data.raw_target_column: cfg.data.target_column})

    if cfg.data.remap_binary_one_two:
        df = remap_binary_one_two(df)

    target_column = cfg.data.target_column
    if target_column not in df.columns:
        raise KeyError(
            f"target_column '{target_column}' not found in loaded data columns: {list(df.columns)}"
        )

    if cfg.data.drop_rows_missing_target:
        n_before = len(df)
        df = df[df[target_column].notna()].reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped:
            logger.info(
                "Dropped %d/%d rows with missing target_column %r",
                n_dropped,
                n_before,
                target_column,
            )

    feature_columns = [c for c in df.columns if c != target_column]
    categorical_columns = infer_categorical_columns(
        df,
        feature_columns,
        cfg.data.categorical_columns,
        unique_threshold=cfg.data.auto_categorical_unique_threshold,
        uci_variable_types=variable_types,
    )

    # Cast whole-numbered categorical columns (incl. target) to a proper int dtype.
    # Some downstream tooling (e.g. SynthEval's AnalysisConfig) infers "categorical"
    # from dtype (object/int) rather than cardinality, so a float-typed {0.0, 1.0}
    # target/categorical column would otherwise silently be treated as continuous.
    df = cast_integer_like_columns(df, categorical_columns + [target_column])

    if cfg.data.outlier_zscore_threshold is not None and cfg.data.outlier_columns:
        outlier_columns = [
            c
            for c in cfg.data.outlier_columns
            if c in feature_columns and c not in categorical_columns
        ]
        df = mask_outliers_as_missing(df, outlier_columns, cfg.data.outlier_zscore_threshold)

    missing_sensitive = [c for c in cfg.data.sensitive_columns if c not in df.columns]
    if missing_sensitive:
        raise KeyError(f"sensitive_columns not found in data: {missing_sensitive}")

    train_df, test_df = train_test_split(
        df,
        train_size=cfg.data.train_size,
        random_state=cfg.seed,
        stratify=df[target_column] if cfg.data.stratify else None,
    )

    dataset = Dataset(
        name=cfg.name,
        target_column=target_column,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        sensitive_columns=list(cfg.data.sensitive_columns),
        data_dir=data_dir,
        full_df=df,
        train_df=train_df,
        test_df=test_df,
        version=cfg.data.version,
    )

    paths = dataset.paths()
    df.to_csv(paths["full"], index=False)
    train_df.to_csv(paths["train"], index=False)
    test_df.to_csv(paths["test"], index=False)

    write_dataset_manifest(cfg, dataset)

    logger.info(
        "Loaded dataset '%s' (version=%s): %d rows, %d features (%d categorical), "
        "target=%r, sensitive=%s, train=%d/test=%d",
        cfg.name,
        cfg.data.version or "unversioned",
        len(df),
        len(feature_columns),
        len(categorical_columns),
        target_column,
        dataset.sensitive_columns,
        len(train_df),
        len(test_df),
    )
    return dataset


def load_imputed_splits(dataset: Dataset) -> Dataset:
    """Attach already-imputed CSVs (produced by synthdata.imputation) to a Dataset."""
    paths = dataset.paths()
    if paths["full_imputed"].exists():
        dataset.full_imputed_df = pd.read_csv(paths["full_imputed"])
    if paths["train_imputed"].exists():
        dataset.train_imputed_df = pd.read_csv(paths["train_imputed"])
    if paths["test_imputed"].exists():
        dataset.test_imputed_df = pd.read_csv(paths["test_imputed"])
    return dataset
