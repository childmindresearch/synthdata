"""TabImpute-based missing data imputation backend.

Wraps ``tabimpute.interface.TabImputeCategorical`` (a TabPFN-based imputer that
one-hot encodes designated categorical columns before imputing, then recovers
category values via softmax + argmax). Includes a small compatibility shim for
version drift between the pinned ``tabpfn`` version and the one ``tabimpute`` was
built against (see the shim's docstring for details).

This is the default imputation backend (``imputation.method: tabimpute``). For
wide datasets where one-hot encoding many categorical columns causes
out-of-memory errors, see :mod:`synthdata.imputation.refidiff_backend`.
"""

import dataclasses
import hashlib
import json

import numpy as np
import pandas as pd

from synthdata.data import decode_label_encoded_columns, label_encode_non_numeric_columns
from synthdata.utils import get_logger

_SHIM_APPLIED = False
TABIMPUTE_STATE_SCHEMA_VERSION = "tabimpute-state-v1"
logger = get_logger(__name__)


@dataclasses.dataclass
class TabImputeState:
    """Train-fitted encoding and scaling state for canonical role transforms."""

    feature_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    category_maps: dict
    means: np.ndarray
    stds: np.ndarray
    block_slices: dict[str, tuple[int, int]]
    imputer: object
    device: str = "cpu"


def state_metadata_fingerprint(metadata: dict) -> str:
    """Fingerprint serializable TabImpute metadata without its stored digest."""
    payload = {key: value for key, value in metadata.items() if key != "state_fingerprint"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()


def state_metadata(state: TabImputeState, fit_frame_fingerprint: str) -> dict:
    """Return durable provenance for a train-fitted TabImpute transform state."""
    payload = {
        "schema_version": TABIMPUTE_STATE_SCHEMA_VERSION,
        "status": "fitted",
        "backend": "tabimpute",
        "fit_role": "train",
        "fit_frame_fingerprint": fit_frame_fingerprint,
        "feature_columns": list(state.feature_columns),
        "categorical_columns": list(state.categorical_columns),
        "category_map_fingerprints": {
            column: state_metadata_fingerprint({"categories": categories.tolist()})
            for column, categories in state.category_maps.items()
        },
        "category_map_sizes": {
            column: len(categories) for column, categories in state.category_maps.items()
        },
        "block_slices": {column: list(bounds) for column, bounds in state.block_slices.items()},
        "scaling_state_fingerprint": state_metadata_fingerprint(
            {"means": state.means.tolist(), "stds": state.stds.tolist()}
        ),
        "device": state.device,
    }
    return {**payload, "state_fingerprint": state_metadata_fingerprint(payload)}


def no_fit_state_metadata(
    feature_columns: list,
    categorical_columns: list,
    fit_frame_fingerprint: str,
    status: str,
) -> dict:
    """Describe a canonical TabImpute run where no fitted state was required."""
    payload = {
        "schema_version": TABIMPUTE_STATE_SCHEMA_VERSION,
        "status": status,
        "backend": "tabimpute",
        "fit_role": "train",
        "fit_frame_fingerprint": fit_frame_fingerprint,
        "feature_columns": list(feature_columns),
        "categorical_columns": list(categorical_columns),
    }
    return {**payload, "state_fingerprint": state_metadata_fingerprint(payload)}


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


def _encode_with_fixed_maps(
    df: pd.DataFrame,
    feature_columns: list,
    category_maps: dict,
) -> pd.DataFrame:
    """Encode a role with category maps learned from the train role only."""
    encoded = df[feature_columns].copy()
    for column, categories in category_maps.items():
        known_values = set(categories.tolist())
        observed = encoded[column].dropna()
        unknown = [value for value in observed.unique().tolist() if value not in known_values]
        if unknown:
            raise ValueError(
                f"tabimpute: role contains value(s) in categorical column {column!r} "
                f"that are absent from the train-role vocabulary: {unknown}"
            )
        category_to_code = {value: index for index, value in enumerate(categories)}
        encoded[column] = encoded[column].map(category_to_code).astype(float)
    for column in feature_columns:
        if column in category_maps:
            continue
        try:
            encoded[column] = pd.to_numeric(encoded[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"tabimpute: feature column {column!r} is not numeric and has no train-fitted "
                "category map"
            ) from exc
    return encoded


def _build_matrix(
    encoded: pd.DataFrame,
    feature_columns: list,
    categorical_columns: list,
    category_maps: dict,
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """Build a fixed-width numeric matrix and remember each feature's block."""
    categorical_set = set(categorical_columns)
    blocks = []
    block_slices = {}
    offset = 0
    for column in feature_columns:
        if column in categorical_set:
            categories = category_maps.get(column)
            if categories is None or len(categories) == 0:
                raise ValueError(
                    f"tabimpute: categorical column {column!r} has no train-role vocabulary"
                )
            values = encoded[column].to_numpy(dtype=float)
            missing = np.isnan(values)
            block = np.zeros((len(encoded), len(categories)), dtype=np.float64)
            observed_rows = np.flatnonzero(~missing)
            if len(observed_rows):
                codes = values[observed_rows].astype(int)
                if np.any(values[observed_rows] != codes):
                    raise ValueError(
                        f"tabimpute: categorical column {column!r} has non-integral encoded values"
                    )
                if np.any(codes < 0) or np.any(codes >= len(categories)):
                    raise ValueError(
                        f"tabimpute: categorical column {column!r} has an invalid encoded value"
                    )
                block[observed_rows, codes] = 1.0
            block[missing, :] = np.nan
        else:
            values = encoded[column].to_numpy(dtype=float)
            block = values[:, None]
        blocks.append(block)
        block_slices[column] = (offset, offset + block.shape[1])
        offset += block.shape[1]
    return np.hstack(blocks).astype(np.float64), block_slices


def _fit_scaling(matrix: np.ndarray, feature_columns: list, block_slices: dict) -> tuple:
    observed_counts = np.sum(~np.isnan(matrix), axis=0)
    if np.any(observed_counts == 0):
        empty_blocks = [
            column
            for column, (start, end) in block_slices.items()
            if np.any(observed_counts[start:end] == 0)
        ]
        raise ValueError(
            "tabimpute: train role has no observed values for encoded feature block(s): "
            f"{empty_blocks}"
        )
    means = np.nansum(matrix, axis=0) / observed_counts
    centered = np.where(np.isnan(matrix), 0.0, matrix - means)
    variances = np.sum(centered * centered, axis=0) / observed_counts
    stds = np.sqrt(variances)
    stds = np.where(np.isfinite(stds) & (stds > 0), stds, 1.0)
    if not np.isfinite(means).all():
        raise ValueError(
            "tabimpute: train-role scaling contains non-finite means for feature columns "
            f"{feature_columns}"
        )
    return means, stds


def fit_dataframe(
    train_df: pd.DataFrame,
    feature_columns: list,
    categorical_columns: list,
    target_column: str,
    device: str = "cpu",
) -> TabImputeState:
    """Fit all reusable TabImpute state on the canonical train role only."""
    del target_column
    _apply_tabpfn_compat_shim()
    from tabimpute.interface import ImputePFN

    encoded, category_maps = label_encode_non_numeric_columns(
        train_df,
        feature_columns,
        categorical_columns=categorical_columns,
    )
    matrix, block_slices = _build_matrix(
        encoded, feature_columns, categorical_columns, category_maps
    )
    means, stds = _fit_scaling(matrix, feature_columns, block_slices)
    logger.info(
        "tabimpute: fitting canonical state on train role (%d rows, %d encoded columns, "
        "%d categorical maps)",
        len(train_df),
        matrix.shape[1],
        len(category_maps),
    )
    return TabImputeState(
        feature_columns=tuple(feature_columns),
        categorical_columns=tuple(categorical_columns),
        category_maps=category_maps,
        means=means,
        stds=stds,
        block_slices=block_slices,
        device=device,
        imputer=ImputePFN(device=device, preprocessors=None),
    )


def transform_dataframe(
    state: TabImputeState,
    df: pd.DataFrame,
    feature_columns: list,
    categorical_columns: list,
    target_column: str,
    role_name: str,
) -> pd.DataFrame:
    """Transform one role with state fitted on ``train`` without refitting."""
    if tuple(feature_columns) != state.feature_columns:
        raise ValueError("tabimpute: transform feature columns differ from fitted train state")
    if tuple(categorical_columns) != state.categorical_columns:
        raise ValueError("tabimpute: transform categorical columns differ from fitted train state")
    encoded = _encode_with_fixed_maps(df, feature_columns, state.category_maps)
    matrix, block_slices = _build_matrix(
        encoded, feature_columns, categorical_columns, state.category_maps
    )
    missing_matrix = np.isnan(matrix)
    if not missing_matrix.any():
        result = df.copy()
        logger.info(
            "tabimpute: role=%s has no missing feature values; transform is a no-op", role_name
        )
        return result[list(df.columns)]

    normalized = (matrix - state.means) / (state.stds + 1e-16)
    logger.info(
        "tabimpute: transforming role=%s (%d rows, %d missing encoded values) with train-fitted state",
        role_name,
        len(df),
        int(missing_matrix.sum()),
    )
    imputed_normalized, _ = state.imputer.get_imputation(normalized.copy())
    imputed_matrix = imputed_normalized * (state.stds + 1e-16) + state.means
    encoded_result = encoded.copy()
    categorical_set = set(categorical_columns)
    for column in feature_columns:
        start, end = block_slices[column]
        missing = encoded_result[column].isna().to_numpy()
        if not missing.any():
            continue
        if column in categorical_set:
            encoded_result.loc[missing, column] = np.argmax(
                imputed_matrix[missing, start:end], axis=1
            )
        else:
            encoded_result.loc[missing, column] = imputed_matrix[missing, start]
    result = decode_label_encoded_columns(encoded_result, state.category_maps)
    result[target_column] = df[target_column].to_numpy()
    return result[list(df.columns)]


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

    encoded, category_maps = label_encode_non_numeric_columns(df, feature_columns)
    x_full = encoded.values.astype(float)
    cat_indices = [feature_columns.index(c) for c in categorical_columns if c in feature_columns]

    x_imputed = imputer.impute(x_full.copy(), categorical_columns=cat_indices)

    imputed_df = pd.DataFrame(x_imputed, columns=feature_columns, index=df.index)
    imputed_df = decode_label_encoded_columns(imputed_df, category_maps)
    imputed_df[target_column] = df[target_column].values
    return imputed_df[list(df.columns)]
