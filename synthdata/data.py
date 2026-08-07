"""Generic dataset loading, typing, and splitting.

Supports two data sources so a collaborator can point the pipeline at their own data:

- ``source: uci``: fetch (and locally cache) a dataset from the UCI ML repository by id.
- ``source: csv``/``source: parquet``: load a local CSV or Parquet file directly
  (the reader used is auto-detected from ``data.path``'s file extension --
  ``.csv`` vs. ``.parquet``/``.pq`` -- rather than from ``source`` itself).

The same :class:`Dataset` object is produced either way and consumed by every
downstream stage (imputation, generation, evaluation, plotting).
"""

import dataclasses
import hashlib
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

IMPUTATION_CACHE_KEY_FILENAME = ".imputation_cache_key.json"


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """Return a stable fingerprint for a DataFrame's values and structure."""
    metadata = {
        "columns": [str(column) for column in df.columns],
        "dtypes": [str(dtype) for dtype in df.dtypes],
        "index_dtype": str(df.index.dtype),
        "index_name": df.index.name,
        "shape": list(df.shape),
    }
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, default=str).encode())
    values = pd.util.hash_pandas_object(df, index=True).to_numpy(dtype=np.uint64, copy=False)
    digest.update(values.tobytes())
    return digest.hexdigest()


def file_fingerprint(path: str | Path) -> str:
    """Return the SHA-256 fingerprint of a source file's exact bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclasses.dataclass
class Dataset:
    """Container for a loaded dataset plus derived metadata used by every stage."""

    name: str
    target_column: str
    feature_columns: list
    nominal_columns: list
    ordinal_columns: list
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

    #: Explicit, resolved variable schema used to derive the categorical roles.
    #: Each entry is ``{"kind": "categorical"|"continuous",
    #: "ordinal_order": list|None}`` and includes the target column.
    variable_schema: dict = dataclasses.field(default_factory=dict)
    #: SHA-256 fingerprint of the exact source schema CSV, if one was used.
    variable_schema_fingerprint: str | None = None
    #: SHA-256 fingerprint of the exact local source file, or of the raw loaded
    #: source frame for sources without a local file.
    source_fingerprint: str | None = None
    #: Fingerprints of the cleaned source frame and deterministic split frames.
    full_fingerprint: str | None = None
    train_split_fingerprint: str | None = None
    test_split_fingerprint: str | None = None

    #: Numeric model-space frames populated once imputation has run
    #: (see synthdata.imputation).
    full_imputed_df: pd.DataFrame | None = None
    train_imputed_df: pd.DataFrame | None = None
    test_imputed_df: pd.DataFrame | None = None
    #: User-facing copies with configured ordinal labels restored.
    full_imputed_decoded_df: pd.DataFrame | None = None
    train_imputed_decoded_df: pd.DataFrame | None = None
    test_imputed_decoded_df: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        """Capture fingerprints for the exact frames held by this dataset."""
        if self.full_fingerprint is None:
            self.full_fingerprint = dataframe_fingerprint(self.full_df)
        if self.train_split_fingerprint is None:
            self.train_split_fingerprint = dataframe_fingerprint(self.train_df)
        if self.test_split_fingerprint is None:
            self.test_split_fingerprint = dataframe_fingerprint(self.test_df)

    @property
    def categorical_columns(self) -> list:
        """All categorical-encoded feature columns: nominal + ordinal.

        Every backend that discretely encodes categorical columns (bit-encoding
        in refidiff, one-hot in tabimpute, etc.) doesn't itself need to
        distinguish nominal from ordinal -- both are encoded/decoded to exact
        observed category values the same way, the only difference is whether
        an order is preserved for the ordinal ones (see
        synthdata.imputation.refidiff_backend._fit_categorical_binary_encoders).
        So most call sites want this combined list; use ``nominal_columns``/
        ``ordinal_columns`` directly only when the distinction actually matters.
        """
        return list(self.nominal_columns) + list(self.ordinal_columns)

    @property
    def ordinal_category_orders(self) -> dict:
        """Return configured ordinal labels in their low-to-high order."""
        return {
            column: list(self.variable_schema[column]["ordinal_order"])
            for column in self.ordinal_columns
            if column in self.variable_schema
            and self.variable_schema[column].get("ordinal_order") is not None
        }

    def decode_ordinal_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Restore configured ordinal labels in a model-space DataFrame."""
        return decode_ordinal_columns(df, self.ordinal_category_orders)

    @property
    def target_is_categorical(self) -> bool:
        """Whether the resolved schema declares the target as categorical.

        Datasets created through the legacy explicit-list configuration have no
        reliable target-kind declaration, so retain the historical assumption
        that their target is categorical.
        """
        target_entry = self.variable_schema.get(self.target_column)
        if target_entry is None:
            return True
        return target_entry.get("kind") == "categorical"

    @property
    def all_categorical_columns(self) -> list:
        """Categorical feature columns plus the target when declared categorical."""
        if self.target_is_categorical:
            return self.categorical_columns + [self.target_column]
        return self.categorical_columns

    def paths(self) -> dict:
        d = self.data_dir
        return {
            "full": d / "full.csv",
            "train": d / "train.csv",
            "test": d / "test.csv",
            "full_imputed": d / "full_imputed.csv",
            "train_imputed": d / "train_imputed.csv",
            "test_imputed": d / "test_imputed.csv",
            "full_imputed_decoded": d / "full_imputed_decoded.csv",
            "train_imputed_decoded": d / "train_imputed_decoded.csv",
            "test_imputed_decoded": d / "test_imputed_decoded.csv",
        }

    def attach_decoded_imputed_splits(self) -> None:
        """Attach decoded user-facing views for any loaded imputed splits."""
        self.full_imputed_decoded_df = (
            self.decode_ordinal_frame(self.full_imputed_df)
            if self.full_imputed_df is not None
            else None
        )
        self.train_imputed_decoded_df = (
            self.decode_ordinal_frame(self.train_imputed_df)
            if self.train_imputed_df is not None
            else None
        )
        self.test_imputed_decoded_df = (
            self.decode_ordinal_frame(self.test_imputed_df)
            if self.test_imputed_df is not None
            else None
        )


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


#: File extensions read via :func:`pandas.read_parquet` in :func:`_load_local_file`;
#: anything else falls back to :func:`pandas.read_csv`.
_PARQUET_SUFFIXES = (".parquet", ".pq")


def _load_local_file(cfg: Config) -> tuple:
    """Load a local CSV or Parquet file, auto-detected from ``cfg.data.path``'s extension.

    Dispatches on the file extension rather than ``cfg.data.source`` so a
    mismatched ``source`` value (e.g. ``source: csv`` pointing at a ``.parquet``
    file) still loads correctly instead of silently mis-parsing the file.
    """
    path = Path(cfg.data.path)
    suffix = path.suffix.lower()
    if suffix in _PARQUET_SUFFIXES:
        logger.info("Loading local Parquet file: %s", path)
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        logger.info("Loading local CSV file: %s", path)
        # low_memory=False: read the whole file in one pass rather than pandas'
        # default chunked read, which can infer a different dtype per chunk for
        # a column that's almost entirely NaN except for a handful of string
        # values far down the file (e.g. loris_combined.csv's >99%-missing
        # DailyMeds__med_type_0X columns) -- emits `DtypeWarning: Columns (...)
        # have mixed types` and silently mixes float/object dtype for the same
        # column across chunks. Same eventual per-column dtype either way, just
        # inferred consistently in one pass instead of reconciled after the fact.
        df = pd.read_csv(path, low_memory=False)
    else:
        raise ValueError(
            f"Unsupported file extension {suffix!r} for data.path={cfg.data.path!r}; "
            "expected one of .csv, .parquet, .pq"
        )
    return df, None


# ---------------------------------------------------------------------------
# Column typing helpers
# ---------------------------------------------------------------------------


def _schema_fingerprint(path: Path) -> str:
    """Return a SHA-256 fingerprint of a variable-schema source file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_variable_schema(path: str | Path, modeling_columns: list) -> tuple[dict, str]:
    """Read and strictly validate the explicit variable schema CSV.

    The CSV requires ``column`` and ``kind`` fields. ``kind`` is exactly
    ``categorical`` or ``continuous``. ``ordinal_order`` is optional; when it
    is present it must be a non-empty square-bracketed list on a categorical row, ordered
    from low to high. A blank order denotes a nominal categorical. Every
    retained feature *and* the target must appear exactly once, and no stale
    declarations are accepted.
    """
    schema_path = Path(path)
    if not schema_path.exists():
        raise FileNotFoundError(f"Variable schema file not found: {schema_path}")

    try:
        schema_df = pd.read_csv(schema_path, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError) as exc:
        raise ValueError(f"Failed to read variable schema CSV {schema_path}: {exc}") from exc

    required_columns = {"column", "kind"}
    missing_columns = required_columns - set(schema_df.columns)
    if missing_columns:
        raise ValueError(
            f"Variable schema {schema_path} is missing required column(s) "
            f"{sorted(missing_columns)}; required columns are 'column' and 'kind'."
        )

    if schema_df["column"].str.strip().eq("").any():
        bad_rows = (schema_df.index[schema_df["column"].str.strip().eq("")] + 2).tolist()
        raise ValueError(
            f"Variable schema {schema_path} has blank column name(s) at CSV row(s) {bad_rows}."
        )
    schema_df["column"] = schema_df["column"].str.strip()
    duplicate_columns = schema_df.loc[schema_df["column"].duplicated(), "column"].tolist()
    if duplicate_columns:
        raise ValueError(
            f"Variable schema {schema_path} declares column(s) more than once: "
            f"{sorted(set(duplicate_columns))}."
        )

    expected_columns = set(modeling_columns)
    declared_columns = set(schema_df["column"])
    missing_declarations = sorted(expected_columns - declared_columns)
    stale_declarations = sorted(declared_columns - expected_columns)
    if missing_declarations or stale_declarations:
        details = []
        if missing_declarations:
            details.append(f"missing declaration(s): {missing_declarations}")
        if stale_declarations:
            details.append(f"stale/non-modeling declaration(s): {stale_declarations}")
        raise ValueError(
            f"Variable schema {schema_path} must declare every retained feature and target exactly "
            f"once after source cleanup; {'; '.join(details)}."
        )

    schema_columns = ["column", "kind"]
    if "ordinal_order" in schema_df.columns:
        schema_columns.append("ordinal_order")
    schema = {}
    for row_number, values in enumerate(
        schema_df[schema_columns].itertuples(index=False, name=None), start=2
    ):
        column, raw_kind, *order_values = values
        kind = raw_kind.strip().lower()
        if kind not in {"categorical", "continuous"}:
            raise ValueError(
                f"Variable schema {schema_path} row {row_number} column {column!r} has invalid "
                f"kind {kind!r}; expected 'categorical' or 'continuous'."
            )

        raw_order = order_values[0] if order_values else ""
        raw_order = raw_order.strip()
        ordinal_order = None
        if raw_order:
            if kind != "categorical":
                raise ValueError(
                    f"Variable schema {schema_path} row {row_number} column {column!r} supplies "
                    "ordinal_order but is declared continuous."
                )
            try:
                ordinal_order = json.loads(raw_order)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Variable schema {schema_path} row {row_number} column {column!r} has invalid "
                    f"ordinal_order value; expected a square-bracketed list: {exc.msg}."
                ) from exc
            if not isinstance(ordinal_order, list) or not ordinal_order:
                raise ValueError(
                    f"Variable schema {schema_path} row {row_number} column {column!r} must use a "
                    "non-empty square-bracketed list for ordinal_order."
                )
            try:
                unique_order_values = {json.dumps(value, sort_keys=True) for value in ordinal_order}
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Variable schema {schema_path} row {row_number} column {column!r} has an "
                    f"ordinal_order value that cannot be represented in JSON: {exc}."
                ) from exc
            if len(unique_order_values) != len(ordinal_order):
                raise ValueError(
                    f"Variable schema {schema_path} row {row_number} column {column!r} has duplicate "
                    "ordinal_order values."
                )
        schema[column] = {"kind": kind, "ordinal_order": ordinal_order}

    logger.info(
        "Loaded strict variable schema from %s: %d columns (%d categorical, %d ordinal)",
        schema_path,
        len(schema),
        sum(entry["kind"] == "categorical" for entry in schema.values()),
        sum(entry["ordinal_order"] is not None for entry in schema.values()),
    )
    return schema, _schema_fingerprint(schema_path)


def schema_column_roles(schema: dict, target_column: str) -> tuple[list, list, dict]:
    """Derive nominal/ordinal compatibility lists and ordinal orders from a schema."""
    nominal_columns = [
        column
        for column, entry in schema.items()
        if column != target_column
        and entry["kind"] == "categorical"
        and entry["ordinal_order"] is None
    ]
    ordinal_columns = [
        column
        for column, entry in schema.items()
        if column != target_column and entry["ordinal_order"] is not None
    ]
    ordinal_orders = {
        column: entry["ordinal_order"]
        for column, entry in schema.items()
        if column != target_column and entry["ordinal_order"] is not None
    }
    return nominal_columns, ordinal_columns, ordinal_orders


def infer_nominal_columns(
    df: pd.DataFrame,
    feature_columns: list,
    explicit: str | list,
    ordinal_columns: list | None = None,
    unique_threshold: int = 10,
    uci_variable_types: dict | None = None,
) -> list:
    """Determine which feature columns should be treated as nominal (unordered categorical).

    Resolution order:
        1. An explicit list of column names in the config always wins.
        2. If UCI variable metadata is available, use its "Categorical" tag.
        3. Otherwise fall back to a dtype/cardinality heuristic
           (object/category/bool dtype, or nunique <= unique_threshold).

    ``ordinal_columns`` (a dataset's separately-configured ordered-categorical
    columns) are excluded from every resolution path above, regardless of
    source -- they're a distinct first-class role handled by the caller, never
    folded back into "nominal" just because they'd otherwise match the heuristic.
    """
    ordinal_set = set(ordinal_columns or [])

    if isinstance(explicit, list):
        return [c for c in explicit if c in feature_columns and c not in ordinal_set]

    if uci_variable_types is not None:
        cats = [
            c
            for c in feature_columns
            if uci_variable_types.get(c) == "Categorical" and c not in ordinal_set
        ]
        if cats:
            return cats

    cats = []
    for c in feature_columns:
        if c in ordinal_set:
            continue
        dtype = df[c].dtype
        if (
            dtype in (object, bool)
            or str(dtype) == "category"
            or df[c].nunique(dropna=True) <= unique_threshold
        ):
            cats.append(c)
    return cats


def encode_ordinal_columns(df: pd.DataFrame, ordinal_categories: dict) -> pd.DataFrame:
    """Map declared-ordinal columns to integers in their configured natural order.

    ``ordinal_categories`` maps a column name to its category values ordered
    from lowest to highest (e.g. ``{"activity_level": ["Very Light", "Light",
    "Moderate", "Heavy", "Exceptional"]}``, see ``data.ordinal_column_categories``
    in :class:`~synthdata.config.DataConfig`). Unlike
    :func:`label_encode_non_numeric_columns`'s alphabetical fallback (used for
    columns nobody declared an order for), this preserves the column's true
    real-world ordering -- which matters for every backend that treats a
    "numeric" column as continuous, since their natural numeric order is
    exactly what a plain-numeric imputation/generation pass relies on to
    model it correctly. Missing values are preserved as NaN.

    Raises if a configured column is missing from ``df``, or if the column
    contains an observed value not present in its configured category list
    (fail loudly rather than silently coercing an unrecognized category to
    NaN or an arbitrary position).
    """
    out = df.copy()
    for col, categories in ordinal_categories.items():
        if col not in out.columns:
            raise KeyError(
                f"data.ordinal_column_categories references column {col!r}, which is not "
                f"present in the loaded data. Available columns: {list(out.columns)}"
            )
        cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
        observed = set(out[col].dropna().unique().tolist())
        unknown = observed - set(cat_to_idx)
        if unknown:
            raise ValueError(
                f"data.ordinal_column_categories[{col!r}] does not include observed value(s) "
                f"{sorted(unknown, key=str)}; every value present in the data must be listed "
                f"in its configured order (configured categories={categories})."
            )
        out[col] = out[col].map(cat_to_idx).astype(float)
    return out


def decode_ordinal_columns(df: pd.DataFrame, ordinal_categories: dict) -> pd.DataFrame:
    """Restore ordinal labels from their zero-based model-space codes.

    ``ordinal_categories`` maps each column to the labels used by
    :func:`encode_ordinal_columns`, ordered from low to high. Missing values
    remain missing. Non-integral or out-of-range codes are rejected instead of
    silently producing an invalid category label.
    """
    out = df.copy()
    for col, categories in ordinal_categories.items():
        if col not in out.columns:
            raise KeyError(
                f"Ordinal decoder references column {col!r}, which is not present in the "
                f"DataFrame. Available columns: {list(out.columns)}"
            )
        if not categories:
            raise ValueError(f"Ordinal decoder has no categories configured for column {col!r}")

        try:
            codes = pd.to_numeric(out[col], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Ordinal column {col!r} contains non-numeric model-space values and cannot "
                "be decoded"
            ) from exc

        non_integral = codes.notna() & codes.mod(1).ne(0)
        out_of_range = codes.notna() & ((codes < 0) | (codes >= len(categories)))
        if non_integral.any() or out_of_range.any():
            invalid_codes = codes[non_integral | out_of_range].dropna().tolist()
            raise ValueError(
                f"Ordinal column {col!r} contains invalid model-space code(s) "
                f"{invalid_codes}; expected integral values in [0, {len(categories) - 1}]"
            )

        out[col] = codes.map(dict(enumerate(categories)))
    return out


def warn_non_numeric_feature_columns(
    df: pd.DataFrame, feature_columns: list, categorical_columns: list
) -> list:
    """Log a loud warning for any feature column declared numeric but not actually numeric.

    ``categorical_columns`` here is the caller's already-combined
    ``nominal_columns + ordinal_columns`` list (see ``Dataset.categorical_columns``).
    A column not listed in it is assumed to already be numeric (e.g. an ordinal
    column pre-encoded to integers), but a plain CSV source can still surprise
    us with a string-valued column that was correctly excluded (e.g. an ordinal
    band stored as text like "Light"/"Heavy") yet was never actually
    numeric-encoded at the source. Every downstream imputation/generation
    backend that builds a numeric matrix falls back to label-encoding such
    columns (see :func:`label_encode_non_numeric_columns`), so this check
    doesn't change behavior -- it exists purely to surface the issue
    immediately at dataset-load time (this function is the single source of
    truth for this check; call it here rather than re-deriving the same
    "numeric_columns" filter independently in each backend), rather than it
    being discovered (or, worse, silently missed) deep inside whichever
    backend happens to run first.

    Returns the list of offending column names (empty if none).
    """
    numeric_columns = [c for c in feature_columns if c not in categorical_columns]
    offending = [c for c in numeric_columns if not pd.api.types.is_numeric_dtype(df[c])]
    if offending:
        logger.warning(
            "%d feature column(s) are not listed in data.nominal_columns/data.ordinal_columns "
            "(so are assumed numeric) but actually contain non-numeric values: %s. Every backend "
            "falls back to label-encoding these columns (preserves missingness, but does not "
            "guarantee categories are numbered in their true order -- alphabetical by default). "
            "Add them to data.nominal_columns (if unordered) or data.ordinal_columns (with an "
            "entry in data.ordinal_column_categories if not already numeric-coded) for correct "
            "treatment.",
            len(offending),
            offending,
        )
    return offending


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


def label_encode_non_numeric_columns(
    df: pd.DataFrame, columns: list, categorical_columns: list | None = None
) -> tuple[pd.DataFrame, dict]:
    """Factorize non-numeric columns (and any declared categorical ones) to integer codes.

    Some backends (``TabImputeCategorical``, TabPFN) require a fully numeric
    matrix, but a plain CSV source (unlike the pre-encoded UCI hepatitis
    example) commonly has string-valued categorical columns (e.g.
    "Light"/"Heavy"/...) -- those are always factorized regardless of
    ``categorical_columns``. Missing values are preserved as NaN so they're
    still treated as missing rather than a category. Returns the encoded
    frame plus ``{column: categories}``, needed to decode output back to the
    original labels via :func:`decode_label_encoded_columns`.

    ``categorical_columns`` (if given) additionally forces factorization for
    columns that are *already numeric* but represent a category, not a
    continuous quantity -- e.g. a 5-level ordinal stored as raw values
    ``{1..5}``, or a binary variable stored as ``{0, 2}``. Without this, such
    a column would pass through unencoded, and some backends' internal
    categorical handling returns *compact 0-indexed class predictions*
    regardless of the input's actual value domain (confirmed for TabPFN's
    unsupervised experiment API): a 5-class column's synthetic output would
    come back as ``{0..4}`` instead of the true ``{1..5}``, and a ``{0, 2}``
    binary column's as ``{0, 1}`` instead of ``{0, 2}`` -- silently shifting/
    relabeling the column's entire domain in the synthetic output. Routing
    every declared categorical column through the same factorize/decode
    round-trip as string columns (regardless of dtype) guarantees the model
    only ever sees/produces compact 0-indexed codes internally, and that
    :func:`decode_label_encoded_columns` always maps back to the true
    observed domain afterward. Already-numeric columns *not* listed in
    ``categorical_columns`` (i.e. genuinely continuous ones) still pass
    through unchanged.
    """
    encoded = df[columns].copy()
    category_maps = {}
    force_factorize = set(categorical_columns or [])
    for col in columns:
        if pd.api.types.is_numeric_dtype(encoded[col]) and col not in force_factorize:
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
        "nominal_columns": dataset.nominal_columns,
        "ordinal_columns": dataset.ordinal_columns,
        "variable_schema": dataset.variable_schema,
        "variable_schema_fingerprint": dataset.variable_schema_fingerprint,
        "source_fingerprint": dataset.source_fingerprint,
        "full_fingerprint": dataframe_fingerprint(dataset.full_df),
        "train_split_fingerprint": dataframe_fingerprint(dataset.train_df),
        "test_split_fingerprint": dataframe_fingerprint(dataset.test_df),
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

    Produces (and caches to ``cfg.data.data_dir/data_v_<cfg.data.version>``)
    ``full.csv``, ``train.csv``, and ``test.csv``. These are the
    pre-imputation splits; :mod:`synthdata.imputation` later fills in
    ``*_imputed.csv`` variants aligned to the same row indices.
    """
    data_dir_base = Path(cfg.data.data_dir)
    data_version_scope = f"data_v_{cfg.data.version}" if cfg.data.version else "data_v_unversioned"
    data_dir = ensure_dir(data_dir_base / data_version_scope)

    if cfg.data.source == "uci":
        df, variable_types = _load_uci(cfg, data_dir)
        source_fingerprint = dataframe_fingerprint(df)
    elif cfg.data.source in ("csv", "parquet"):
        df, variable_types = _load_local_file(cfg)
        source_fingerprint = file_fingerprint(cfg.data.path)
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
    modeling_columns = feature_columns + [target_column]
    variable_schema = {}
    variable_schema_fingerprint = None

    if cfg.data.variable_schema_path:
        variable_schema, variable_schema_fingerprint = load_variable_schema(
            cfg.data.variable_schema_path, modeling_columns
        )
        nominal_columns, ordinal_columns, ordinal_orders = schema_column_roles(
            variable_schema, target_column
        )
        if ordinal_orders:
            df = encode_ordinal_columns(df, ordinal_orders)
    else:
        # Legacy explicit lists remain readable so existing recorded experiments
        # can be reproduced. New datasets must use variable_schema_path; no
        # dtype/cardinality inference is permitted here.
        if cfg.data.nominal_columns is None:
            raise ValueError(
                "data.variable_schema_path is required for new datasets. Existing configurations "
                "may temporarily provide an explicit data.nominal_columns list together with "
                "data.ordinal_columns, but automatic type inference is not supported."
            )
        if cfg.data.ordinal_column_categories:
            df = encode_ordinal_columns(df, cfg.data.ordinal_column_categories)
        ordinal_columns = [c for c in cfg.data.ordinal_columns if c in feature_columns]
        nominal_columns = [
            c
            for c in cfg.data.nominal_columns
            if c in feature_columns and c not in set(ordinal_columns)
        ]
        variable_schema = {
            column: {
                "kind": "categorical"
                if column == target_column or column in nominal_columns + ordinal_columns
                else "continuous",
                "ordinal_order": cfg.data.ordinal_column_categories.get(column),
            }
            for column in modeling_columns
        }
    categorical_columns = nominal_columns + ordinal_columns
    warn_non_numeric_feature_columns(df, feature_columns, categorical_columns)

    target_is_categorical = (
        variable_schema.get(target_column, {}).get("kind", "categorical") == "categorical"
    )

    # Cast whole-numbered categorical columns to a proper int dtype.
    # Some downstream tooling (e.g. SynthEval's AnalysisConfig) infers "categorical"
    # from dtype (object/int) rather than cardinality, so a float-typed {0.0, 1.0}
    # categorical column would otherwise silently be treated as continuous.
    columns_to_cast = categorical_columns + ([target_column] if target_is_categorical else [])
    df = cast_integer_like_columns(df, columns_to_cast)

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
        nominal_columns=nominal_columns,
        ordinal_columns=ordinal_columns,
        sensitive_columns=list(cfg.data.sensitive_columns),
        data_dir=data_dir,
        full_df=df,
        train_df=train_df,
        test_df=test_df,
        version=cfg.data.version,
        variable_schema=variable_schema,
        variable_schema_fingerprint=variable_schema_fingerprint,
        source_fingerprint=source_fingerprint,
    )

    paths = dataset.paths()
    df.to_csv(paths["full"], index=False)
    train_df.to_csv(paths["train"], index=False)
    test_df.to_csv(paths["test"], index=False)

    write_dataset_manifest(cfg, dataset)

    logger.info(
        "Loaded dataset '%s' (version=%s): %d rows, %d features (%d categorical: %d nominal + "
        "%d ordinal), target=%r, sensitive=%s, train=%d/test=%d",
        cfg.name,
        cfg.data.version or "unversioned",
        len(df),
        len(feature_columns),
        len(categorical_columns),
        len(nominal_columns),
        len(ordinal_columns),
        target_column,
        dataset.sensitive_columns,
        len(train_df),
        len(test_df),
    )
    return dataset


def load_imputed_splits(dataset: Dataset) -> Dataset:
    """Attach imputed CSVs only when their source and split provenance still matches."""
    paths = dataset.paths()
    imputed_paths = {
        "full": paths["full_imputed"],
        "train": paths["train_imputed"],
        "test": paths["test_imputed"],
    }
    if not all(path.exists() for path in imputed_paths.values()):
        return dataset

    provenance_path = dataset.data_dir / IMPUTATION_CACHE_KEY_FILENAME
    if not provenance_path.exists():
        logger.warning(
            "Ignoring imputed CSVs under %s because provenance file %s is missing; "
            "rerun imputation to create a validated cache",
            dataset.data_dir,
            provenance_path,
        )
        return dataset
    try:
        with provenance_path.open() as provenance_file:
            provenance = json.load(provenance_file)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Ignoring imputed CSVs under %s because provenance file %s is invalid (%s); "
            "rerun imputation",
            dataset.data_dir,
            provenance_path,
            exc,
        )
        return dataset

    expected_provenance = {
        "source_fingerprint": dataset.source_fingerprint,
        "full_fingerprint": dataframe_fingerprint(dataset.full_df),
        "train_split_fingerprint": dataframe_fingerprint(dataset.train_df),
        "test_split_fingerprint": dataframe_fingerprint(dataset.test_df),
    }
    mismatches = {
        field: (provenance.get(field), expected)
        for field, expected in expected_provenance.items()
        if provenance.get(field) != expected
    }
    if mismatches:
        logger.warning(
            "Ignoring stale imputed CSVs under %s; source/split fingerprints differ: %s",
            dataset.data_dir,
            mismatches,
        )
        return dataset

    frames = {name: pd.read_csv(path) for name, path in imputed_paths.items()}
    expected_columns = dataset.full_df.columns.tolist()
    recorded_row_counts = provenance.get("imputed_row_counts")
    if isinstance(recorded_row_counts, dict) and all(
        isinstance(recorded_row_counts.get(name), int) for name in imputed_paths
    ):
        expected_rows = {name: recorded_row_counts[name] for name in imputed_paths}
    else:
        expected_rows = {
            "full": len(dataset.full_df),
            "train": len(dataset.train_df),
            "test": len(dataset.test_df),
        }
    invalid_frames = {
        name: {
            "rows": len(frame),
            "expected_rows": expected_rows[name],
            "columns_match": frame.columns.tolist() == expected_columns,
        }
        for name, frame in frames.items()
        if len(frame) != expected_rows[name] or frame.columns.tolist() != expected_columns
    }
    if invalid_frames:
        logger.warning(
            "Ignoring imputed CSVs under %s because cached frame shapes/columns are stale: %s",
            dataset.data_dir,
            invalid_frames,
        )
        return dataset

    dataset.full_imputed_df = frames["full"]
    dataset.train_imputed_df = frames["train"]
    dataset.test_imputed_df = frames["test"]
    dataset.attach_decoded_imputed_splits()
    return dataset
