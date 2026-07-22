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

import pandas as pd

from synthdata.data import decode_label_encoded_columns, label_encode_non_numeric_columns

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

    encoded, category_maps = label_encode_non_numeric_columns(df, feature_columns)
    x_full = encoded.values.astype(float)
    cat_indices = [feature_columns.index(c) for c in categorical_columns if c in feature_columns]

    x_imputed = imputer.impute(x_full.copy(), categorical_columns=cat_indices)

    imputed_df = pd.DataFrame(x_imputed, columns=feature_columns, index=df.index)
    imputed_df = decode_label_encoded_columns(imputed_df, category_maps)
    imputed_df[target_column] = df[target_column].values
    return imputed_df[list(df.columns)]
