"""MICE with LightGBM forests and predictive mean matching (``miceforest``).

``imputation.method: miceforest`` fills each incomplete column from the others
round after round, like MissForest, but every filled value is drawn rather
than averaged: a forest predicts the cell, and the value is copied from one of
the ``mean_match_candidates`` observed rows whose predictions are closest
(predictive mean matching, van Buuren, *Flexible Imputation of Missing Data*,
2018, section 3.4). Imputations therefore keep the observed distribution:
only values that occur in the data, the observed spread and skew, and
categories in proportion to their predicted probabilities instead of always
the most likely one. MissForest's conditional means shrink toward the middle
and collapse categorical columns onto their mode.

Nominal columns are LightGBM classification targets. Continuous and ordinal
columns (ordinal codes keep their order) are regression targets whose
predictions are matched to observed donors, so ordinal fills are always valid
codes.

The kernel is fitted once on ``fit_df`` and only applied afterwards, with
donors from ``fit_df``; rows passed to :func:`transform` never shape it.
Observed values are returned unchanged, and the target column is neither an
input nor an output. The kernel sees the columns as ``c0``, ``c1``, ...:
LightGBM rejects feature names with JSON characters (``"``, ``,``, ``:``,
brackets and braces), so the original names never reach it.

Binary columns are matched on LightGBM's log-odds. When a forest is certain
(a predicted probability of exactly 0 or 1, common for a rare level or a
column the others determine) miceforest's ``logodds`` returns an infinity and
its nearest-neighbour search stops on it; :func:`_finite_logodds` clips the
probabilities just inside (0, 1) first, which keeps their order.
"""

import contextlib
import dataclasses
from typing import Any

import numpy as np
import pandas as pd

from synthdata.config import MiceForestConfig


@dataclasses.dataclass
class FittedMiceForest:
    """A fitted ``miceforest`` kernel plus the encoding it was fitted with."""

    #: Columns the kernel sees (features with at least one observed fit value).
    columns: list
    #: Observed categories per classification column, in code order.
    categories: dict
    #: Columns with one observed value: filled with it, never modelled.
    constants: dict
    #: Columns the kernel has a model for.
    modeled: list
    seed: int
    kernel: Any


#: Probabilities are clipped to [_EPS, 1 - _EPS] before taking log-odds.
_EPS = 1e-12


@contextlib.contextmanager
def _finite_logodds():
    """Make miceforest's log-odds finite while the block runs.

    ``miceforest.imputation_kernel`` (6.0.5) maps binary predictions through
    ``utils.logodds``, which divides by ``1 - p`` without a guard; a ``p`` of
    exactly 0 or 1 gives an infinity that scipy's KDTree rejects ("data must
    be finite"). Clipping in float64 keeps every other value unchanged.
    """
    from miceforest import imputation_kernel

    original = imputation_kernel.logodds

    def clipped(probability):
        p = np.clip(np.asarray(probability, dtype=np.float64), _EPS, 1.0 - _EPS)
        return np.log(p / (1.0 - p))

    imputation_kernel.logodds = clipped  # ty: ignore[invalid-assignment]
    try:
        yield
    finally:
        imputation_kernel.logodds = original


def _kernel_names(state: FittedMiceForest) -> dict:
    """``{column: name the kernel sees}``, positional so LightGBM accepts every name."""
    return {column: f"c{i}" for i, column in enumerate(state.columns)}


def _to_kernel(encoded: pd.DataFrame, state: FittedMiceForest) -> pd.DataFrame:
    return encoded.rename(columns=_kernel_names(state))


def _from_kernel(filled: pd.DataFrame, state: FittedMiceForest) -> pd.DataFrame:
    return filled.rename(columns={v: k for k, v in _kernel_names(state).items()})


def _encode(frame: pd.DataFrame, state: FittedMiceForest) -> pd.DataFrame:
    """Frame in the kernel's dtypes; unseen categories count as missing."""
    parts = {}
    for column in state.columns:
        values = frame[column]
        if column in state.constants:
            values = values.fillna(state.constants[column])
        if column in state.categories:
            parts[column] = pd.Categorical(values, categories=state.categories[column])
        else:
            parts[column] = pd.to_numeric(values).astype(float)
    return pd.DataFrame(parts, index=frame.index)


def _decode(filled: pd.DataFrame, frame: pd.DataFrame, state: FittedMiceForest) -> pd.DataFrame:
    """Map filled cells back to the original labels; observed values stay as they were."""
    out = frame.copy()
    for column in state.columns:
        values = filled[column]
        if column in state.categories:
            values = pd.Series(
                state.categories[column].take(values.cat.codes.to_numpy()), index=frame.index
            )
        out[column] = frame[column].where(frame[column].notna(), values)
    return out


def fit(
    fit_df: pd.DataFrame,
    feature_columns: list,
    categorical_columns: list,
    nominal_columns: list,
    seed: int,
    miceforest_cfg: MiceForestConfig | None = None,
    model_columns: list | None = None,
) -> tuple[FittedMiceForest, pd.DataFrame]:
    """Fit the kernel on ``fit_df`` and return it with ``fit_df`` imputed.

    ``model_columns`` are the columns that need a model: those missing in
    ``fit_df`` or in the rows :func:`transform` will fill later. Defaults to
    every feature column.
    """
    import miceforest as mf

    cfg = miceforest_cfg or MiceForestConfig()
    columns = [c for c in feature_columns if fit_df[c].notna().any()]
    nominal = set(nominal_columns)
    categories = {
        c: pd.Index(sorted(fit_df[c].dropna().unique()))
        for c in columns
        if c in nominal
        or (c in categorical_columns and not pd.api.types.is_numeric_dtype(fit_df[c]))
    }
    constants = {c: categories[c][0] for c in categories if len(categories[c]) < 2}
    wanted = set(feature_columns if model_columns is None else model_columns)
    modeled = [c for c in columns if c in wanted and c not in constants]
    state = FittedMiceForest(columns, categories, constants, modeled, seed, kernel=None)
    encoded = _encode(fit_df, state).reset_index(drop=True)
    if not modeled:
        return state, transform(state, fit_df)
    names = _kernel_names(state)
    state.kernel = mf.ImputationKernel(
        _to_kernel(encoded, state),
        variable_schema=[names[c] for c in modeled],
        mean_match_candidates=cfg.mean_match_candidates,
        random_state=seed,
    )
    with _finite_logodds():
        state.kernel.mice(cfg.iterations, num_threads=cfg.num_threads, deterministic=True)
    filled = _from_kernel(state.kernel.complete_data(dataset=0), state).set_axis(fit_df.index)
    return state, _decode(filled, fit_df, state)


def transform(state: FittedMiceForest, frame: pd.DataFrame) -> pd.DataFrame:
    """Fill ``frame``'s missing values with the already-fitted kernel."""
    if frame.empty:
        return frame.copy()
    encoded = _encode(frame, state)
    missing = [c for c in state.columns if encoded[c].isna().any()]
    unmodeled = sorted(set(missing) - set(state.modeled))
    if unmodeled:
        raise ValueError(
            f"miceforest has no model for these columns with missing values: {unmodeled}"
        )
    if not missing:
        return _decode(encoded, frame, state)
    with _finite_logodds():
        imputed = state.kernel.impute_new_data(
            _to_kernel(encoded, state).reset_index(drop=True),
            random_state=np.random.RandomState(state.seed),
        )
    filled = _from_kernel(imputed.complete_data(dataset=0), state).set_axis(frame.index)
    return _decode(filled, frame, state)
