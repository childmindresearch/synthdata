"""Median/mode and MissForest imputation with scikit-learn.

``imputation.method: simple`` fills continuous columns with their median and
categorical ones with their most frequent value (``SimpleImputer``).
``imputation.method: missforest`` predicts each incomplete column from the
others with random forests, round after round (``IterativeImputer`` with a
``RandomForestRegressor``), which scikit-learn documents as its MissForest
equivalent (Stekhoven & Buehlmann, *Bioinformatics* 2012). Nominal columns
with more than two categories are one-hot encoded so the forests predict
category probabilities rather than averaging arbitrary codes; the most likely
category is kept.

Both imputers are fitted once and then only applied, so rows passed to
:func:`transform` never shape the fitted model. (hyperimpute's iterative
plugins refit on whatever they transform, which is why they are not used.)
Observed values are always returned unchanged, and the target column is
neither an input nor an output.
"""

import dataclasses

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer

from synthdata.config import MissForestConfig


@dataclasses.dataclass
class FittedImputer:
    """A fitted imputer plus the encoding it was fitted with."""

    method: str
    columns: list
    #: Observed categories per categorical column, in code order.
    categories: dict
    #: Nominal columns encoded as one indicator per category (missforest only).
    one_hot: list
    model: object


def _encode(frame: pd.DataFrame, state: FittedImputer) -> pd.DataFrame:
    """Numeric matrix the imputer sees; unseen categories count as missing."""
    parts = []
    for column in state.columns:
        values = frame[column]
        if column not in state.categories:
            parts.append(values.astype(float).rename(column))
            continue
        codes = pd.Series(
            pd.Categorical(values, categories=state.categories[column]).codes,
            index=frame.index,
            dtype=float,
        ).replace(-1, np.nan)
        if column in state.one_hot:
            for code in range(len(state.categories[column])):
                parts.append(
                    codes.eq(code).astype(float).where(codes.notna()).rename(f"{column}={code}")
                )
        else:
            parts.append(codes.rename(column))
    return pd.concat(parts, axis=1)


def _decode(matrix: pd.DataFrame, frame: pd.DataFrame, state: FittedImputer) -> pd.DataFrame:
    """Map imputed codes back to values and keep every observed value as it was."""
    out = frame.copy()
    for column in state.columns:
        if column in state.one_hot:
            n = len(state.categories[column])
            codes = matrix[[f"{column}={code}" for code in range(n)]].to_numpy().argmax(axis=1)
        elif column in state.categories:
            n = len(state.categories[column])
            codes = matrix[column].round().clip(0, n - 1).astype(int).to_numpy()
        else:
            filled = matrix[column]
            out[column] = frame[column].where(frame[column].notna(), filled)
            continue
        filled = pd.Series(state.categories[column].take(codes), index=frame.index)
        out[column] = frame[column].where(frame[column].notna(), filled)
    return out


def fit(
    fit_df: pd.DataFrame,
    feature_columns: list,
    categorical_columns: list,
    nominal_columns: list,
    method: str,
    seed: int,
    missforest_cfg: MissForestConfig | None = None,
) -> tuple[FittedImputer, pd.DataFrame]:
    """Fit the imputer on ``fit_df`` and return it with ``fit_df`` imputed."""
    categorical = set(categorical_columns)
    categories = {
        column: pd.Index(sorted(fit_df[column].dropna().unique()))
        for column in feature_columns
        if column in categorical or not pd.api.types.is_numeric_dtype(fit_df[column])
    }
    # Columns with no observed train value cannot be learned; leave them as they are.
    columns = [c for c in feature_columns if fit_df[c].notna().any()]
    one_hot = (
        [c for c in nominal_columns if c in categories and len(categories[c]) > 2]
        if method == "missforest"
        else []
    )
    state = FittedImputer(method, columns, categories, one_hot, model=None)
    encoded = _encode(fit_df, state)
    if method == "simple":
        state.model = {
            "median": SimpleImputer(strategy="median"),
            "most_frequent": SimpleImputer(strategy="most_frequent"),
        }
        groups = _simple_groups(state, encoded.columns)
        for strategy, cols in groups.items():
            if cols:
                state.model[strategy].fit(encoded[cols])
    elif method == "missforest":
        cfg = missforest_cfg or MissForestConfig()
        state.model = IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=cfg.n_estimators, n_jobs=-1, random_state=seed
            ),
            max_iter=cfg.max_iter,
            n_nearest_features=cfg.n_nearest_features,
            initial_strategy="median",
            skip_complete=True,
            keep_empty_features=True,
            random_state=seed,
        ).fit(encoded)
    else:
        raise ValueError(f"Unknown scikit-learn imputation method: {method!r}")
    return state, transform(state, fit_df)


def _simple_groups(state: FittedImputer, encoded_columns) -> dict:
    """Split encoded columns into median (continuous) and mode (categorical) groups."""
    return {
        "median": [c for c in encoded_columns if c not in state.categories],
        "most_frequent": [c for c in encoded_columns if c in state.categories],
    }


def transform(state: FittedImputer, frame: pd.DataFrame) -> pd.DataFrame:
    """Fill ``frame``'s missing values with the already-fitted imputer."""
    if frame.empty:
        return frame.copy()
    encoded = _encode(frame, state)
    if state.method == "simple":
        filled = encoded.copy()
        for strategy, cols in _simple_groups(state, encoded.columns).items():
            if cols:
                filled[cols] = state.model[strategy].transform(encoded[cols])
    else:
        filled = pd.DataFrame(
            state.model.transform(encoded), columns=encoded.columns, index=frame.index
        )
    return _decode(filled, frame, state)
