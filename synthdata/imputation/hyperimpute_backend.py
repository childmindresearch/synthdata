"""Deterministic, stateful wrappers around HyperImpute's simple plugins."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import pandas as pd


class HyperImputeError(RuntimeError):
    """Raised when fixed-plugin imputation cannot satisfy its contract."""


@dataclass
class HyperImputeState:
    """Fitted plugin state and provenance for one canonical imputation run."""

    feature_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    continuous_plugin: str
    categorical_plugin: str
    numeric_imputer: object | None
    categorical_imputer: object | None
    fit_roles: tuple[str, ...]
    fit_fingerprint: str


def _fingerprint(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


def metadata_fingerprint(metadata: dict) -> str:
    """Return fingerprint over metadata fields excluding its stored digest."""
    payload = {key: value for key, value in metadata.items() if key != "state_fingerprint"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _plugin_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Normalize missing sentinels accepted by sklearn-backed plugins."""
    values = frame[columns].copy()
    return values.replace({None: float("nan")})


def _plugin(name: str, random_state: int) -> object:
    if name == "median":
        from hyperimpute.plugins.imputers.plugin_median import MedianPlugin

        return MedianPlugin(random_state=random_state)
    if name == "mean":
        from hyperimpute.plugins.imputers.plugin_mean import MeanPlugin

        return MeanPlugin(random_state=random_state)
    if name == "most_frequent":
        from hyperimpute.plugins.imputers.plugin_most_frequent import MostFrequentPlugin

        return MostFrequentPlugin(random_state=random_state)
    raise ValueError(f"Unsupported fixed HyperImpute plugin: {name!r}")


def fit_dataframe(
    frame: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    *,
    fit_roles: tuple[str, ...],
    continuous_plugin: str = "median",
    random_state: int = 0,
) -> HyperImputeState:
    """Fit fixed plugins on one frame; never performs model selection."""
    categorical = set(categorical_columns)
    categorical.update(
        column for column in feature_columns if not pd.api.types.is_numeric_dtype(frame[column])
    )
    for column in feature_columns:
        if frame[column].dropna().empty:
            raise HyperImputeError(f"Cannot impute feature {column!r}: no observed training value.")
    numeric_columns = [column for column in feature_columns if column not in categorical]
    categorical_columns = [column for column in feature_columns if column in categorical]
    numeric_imputer: Any = _plugin(continuous_plugin, random_state) if numeric_columns else None
    categorical_imputer: Any = (
        _plugin("most_frequent", random_state) if categorical_columns else None
    )
    if numeric_imputer is not None:
        numeric_imputer.fit(_plugin_frame(frame, numeric_columns))
    if categorical_imputer is not None:
        categorical_imputer.fit(_plugin_frame(frame, categorical_columns))
    return HyperImputeState(
        tuple(feature_columns),
        tuple(categorical_columns),
        continuous_plugin,
        "most_frequent",
        numeric_imputer,
        categorical_imputer,
        fit_roles,
        _fingerprint(frame),
    )


def transform_dataframe(state: HyperImputeState, frame: pd.DataFrame) -> pd.DataFrame:
    """Transform frame with fitted plugins, preserving identity and target columns."""
    out = frame.copy()
    original_index = out.index
    numeric = [c for c in state.feature_columns if c not in state.categorical_columns]
    categorical = list(state.categorical_columns)
    if state.numeric_imputer is not None:
        numeric_imputer: Any = state.numeric_imputer
        assert numeric_imputer is not None
        values = numeric_imputer.transform(_plugin_frame(out, numeric))
        out.loc[:, numeric] = pd.DataFrame(
            values.to_numpy(), index=original_index, columns=pd.Index(numeric)
        ).to_numpy()
    if state.categorical_imputer is not None:
        categorical_imputer: Any = state.categorical_imputer
        assert categorical_imputer is not None
        values = categorical_imputer.transform(_plugin_frame(out, categorical))
        out.loc[:, categorical] = pd.DataFrame(
            values.to_numpy(), index=original_index, columns=pd.Index(categorical)
        ).to_numpy()
    return out


def state_metadata(
    state: HyperImputeState | None,
    *,
    status: str = "fitted",
    transform_roles: list[str] | None = None,
    fit_roles: list[str] | None = None,
    fit_frame_fingerprint: str | None = None,
    feature_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    continuous_plugin: str = "median",
) -> dict:
    """Serialize cache-relevant plugin and fit provenance."""
    if state is None:
        payload = {
            "backend": "hyperimpute",
            "status": status,
            "fit_roles": fit_roles or ["train"],
            "fit_frame_fingerprint": fit_frame_fingerprint,
            "transform_roles": transform_roles or ["train", "tuning", "final_holdout"],
        }
        payload["feature_columns"] = feature_columns or []
        payload["categorical_columns"] = categorical_columns or []
        payload["continuous_plugin"] = continuous_plugin
        payload["categorical_plugin"] = "most_frequent"
        payload["state_fingerprint"] = metadata_fingerprint(payload)
        return payload
    payload = {
        "backend": "hyperimpute",
        "status": status,
        "fit_roles": list(state.fit_roles),
        "fit_frame_fingerprint": state.fit_fingerprint,
        "feature_columns": list(state.feature_columns),
        "categorical_columns": list(state.categorical_columns),
        "continuous_plugin": state.continuous_plugin,
        "categorical_plugin": state.categorical_plugin,
        "transform_roles": transform_roles or ["train", "tuning", "final_holdout"],
    }
    payload["state_fingerprint"] = metadata_fingerprint(payload)
    return payload
