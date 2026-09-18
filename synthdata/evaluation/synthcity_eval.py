"""synthcity-based evaluation.

Rather than re-fitting every generator (which ``Benchmarks.evaluate`` does
internally and can be very expensive for GAN/diffusion-style plugins), this
module evaluates the *already-generated* synthetic CSVs from
:mod:`synthdata.generation` uniformly -- synthcity-native and TabPFN/TabPFGen
datasets alike -- by wrapping each cached DataFrame in a bootstrap-resampling
adapter (``PregeneratedSyntheticModel``). This mirrors the notebook's approach
for external (non-synthcity) generators and generalizes it to every model, so
evaluation is fast, reproducible from cached artifacts, and framework-agnostic.
"""

import hashlib
import json
import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any

import pandas as pd

from synthdata.data import semantic_context_digest
from synthdata.evaluation.catalog import (
    SYNTHCITY_CANONICAL_MANIFEST,
    SYNTHCITY_CATEGORY_TO_TYPE,
    SYNTHCITY_EMITTED_KEY_SUFFIXES,
    SYNTHCITY_METRIC_CONFIG,
    emitted_keys_for_synthcity_metrics,
    is_known_synthcity_emitted_key,
    resolve_selection,
)
from synthdata.evaluation.metric_contracts import (
    DEFAULT_METRIC_CONTRACT_REGISTRY,
    MetricEvaluationContext,
    MetricObservation,
    MetricValidationResult,
    resolve_metric_observations,
)
from synthdata.utils import get_logger

logger = get_logger(__name__)


def _semantic_metric_workspace(
    workspace: str | None,
    *,
    target_column: str,
    sensitive_features: list,
    metrics: dict,
    task_type: str,
    evaluation_role: str,
    group_mode: str,
    feature_types: Mapping[str, str] | None,
    source_table: Mapping[str, str] | None,
    sensitive_target_types: Mapping[str, str] | None,
    quasi_identifier_columns: list[str] | None,
    classification_score: str = "balanced_accuracy",
    structural_n_clusters: list[int] | None,
    structural_min_rows_per_cluster: int,
    group_ids: Mapping[str, Any] | None,
    semantic_context: Mapping[str, Any] | None = None,
) -> Path:
    def digest_values(values: Any) -> str | None:
        if values is None:
            return None
        return hashlib.sha256(
            json.dumps(
                [{"type": type(value).__qualname__, "value": repr(value)} for value in values],
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    payload = {
        "schema_version": "synthcity-metric-context-v1",
        "registry_digest": DEFAULT_METRIC_CONTRACT_REGISTRY.digest(),
        "target_column": target_column,
        "sensitive_features": list(sensitive_features),
        "metrics": metrics,
        "task_type": task_type,
        "evaluation_role": evaluation_role,
        "group_mode": group_mode,
        "feature_types": dict(feature_types or {}),
        "source_table": dict(source_table or {}),
        "sensitive_target_types": dict(sensitive_target_types or {}),
        "quasi_identifier_columns": list(quasi_identifier_columns or []),
        "classification_score": classification_score,
        "structural_n_clusters": list(structural_n_clusters or []),
        "structural_min_rows_per_cluster": structural_min_rows_per_cluster,
        "group_ids": {label: digest_values(values) for label, values in (group_ids or {}).items()},
        "semantic_context": dict(semantic_context) if semantic_context is not None else None,
        "semantic_context_digest": (
            semantic_context_digest(semantic_context) if semantic_context is not None else None
        ),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()
    return (Path(workspace) if workspace else Path("workspace")) / f"semantic-{digest[:16]}"


def _base_emitted_keys(
    metric_config: Mapping[str, list[str]],
    *,
    variable_columns: list[str] | None = None,
    attack_target_types: Mapping[str, str] | None = None,
) -> list[str]:
    return emitted_keys_for_synthcity_metrics(
        metric_config,
        variable_columns=variable_columns,
        attack_target_types=attack_target_types,
    )


def _legacy_expected_keys_from_base_names(declared_base_keys: list[str]) -> list[str]:
    """Expand legacy direct-adapter base names for backward compatibility.

    Production callers must pass the fully qualified manifest returned by
    ``emitted_keys_for_synthcity_metrics``. This compatibility path exists only
    for older direct ``validate_synthcity_report`` callers that supplied a
    framework base name instead of a resolved manifest.
    """
    expected_keys: list[str] = []
    for base_key in declared_base_keys:
        suffixes = SYNTHCITY_EMITTED_KEY_SUFFIXES.get(base_key)
        if suffixes is not None:
            expected_keys.extend(f"{base_key}.{suffix}" for suffix in suffixes)
        elif is_known_synthcity_emitted_key(base_key):
            expected_keys.append(base_key)
        else:
            raise ValueError(
                f"SynthCity legacy declared metric {base_key!r} has no static emitted-key contract"
            )
    return list(dict.fromkeys(expected_keys))


def _declared_expected_keys(
    expected_keys: list[str] | None,
    expected_base_keys: list[str] | None,
) -> list[str]:
    if expected_keys is not None and expected_base_keys is not None:
        raise ValueError("Provide expected_keys or expected_base_keys, not both")
    if expected_keys is not None:
        resolved = list(expected_keys)
    elif expected_base_keys:
        resolved = _legacy_expected_keys_from_base_names(expected_base_keys)
    else:
        resolved = []
    if len(resolved) != len(set(resolved)):
        raise ValueError("SynthCity expected emitted metric keys must be unique")
    canonical_keys = set(SYNTHCITY_CANONICAL_MANIFEST)
    unsafe_canonical = [key for key in resolved if key in canonical_keys]
    if unsafe_canonical:
        raise ValueError(
            "SynthCity canonical identities require canonical metric producers, not native report aliases: "
            + ", ".join(unsafe_canonical)
        )
    return resolved


def _row_value(row: pd.Series, column: str) -> Any:
    if column not in row:
        return None
    value = row[column]
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        return value
    return value


def _row_error(row: pd.Series) -> str | None:
    error_messages = _row_value(row, "error_messages")
    error_types = _row_value(row, "error_types")
    errors = _row_value(row, "errors")
    if not error_messages and not error_types and not errors:
        return None

    if error_types:
        return f"metric_evaluation_failed; exception_type={error_types}"
    if errors:
        return "metric_evaluation_failed"
    return "metric_evaluation_failed"


def _row_sample_size(row: pd.Series, field: str | None = "rounds") -> int | None:
    if field is None:
        return None
    value = _row_value(row, field)
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    converted = float(value)
    if not math.isfinite(converted) or converted < 0 or not converted.is_integer():
        return None
    return int(converted)


def _metric_result_metadata(report_metadata: Mapping[str, Any], emitted_key: str) -> dict[str, Any]:
    """Select the native metadata belonging to one qualified metric identity."""
    candidates = [
        (str(key), value)
        for key, value in report_metadata.items()
        if isinstance(value, Mapping)
        and (str(key) == emitted_key or emitted_key.startswith(f"{key}."))
    ]
    if not candidates:
        return {}
    _, metadata = max(candidates, key=lambda item: len(item[0]))
    return dict(metadata)


def build_synthcity_observations(
    model_name: str,
    report: pd.DataFrame,
    *,
    expected_keys: list[str] | None = None,
    expected_base_keys: list[str] | None = None,
    role_hashes: Mapping[str, str] | None = None,
    source_metadata: Mapping[str, Any] | None = None,
) -> tuple[list[str], tuple[MetricObservation, ...]]:
    """Convert one native SynthCity report into contract observations.

    Successful native rows are expected to contain ``mean`` and ``direction``.
    A model-level error frame or an empty report is expanded to the declared
    fully qualified manifest so the failure remains visible to the contract
    resolver. ``expected_base_keys`` is retained only for legacy direct callers.
    """
    role_hashes = dict(role_hashes or {})
    source_metadata = dict(source_metadata or {})
    raw_report_metadata = report.attrs.get(
        "result_metadata",
        report.attrs.get("metric_metadata", {}),
    )
    if not isinstance(raw_report_metadata, Mapping):
        raise ValueError("SynthCity report metric_metadata must be an object")
    report_metadata = dict(raw_report_metadata)
    report_semantic_context = report.attrs.get("semantic_context")
    if isinstance(report_semantic_context, Mapping):
        source_metadata["semantic_context"] = dict(report_semantic_context)
        source_metadata["semantic_context_digest"] = semantic_context_digest(
            report_semantic_context
        )
    declared_keys = _declared_expected_keys(expected_keys, expected_base_keys)

    group_safety = report.attrs.get("group_safety")
    if isinstance(group_safety, Mapping) and group_safety.get("status") == "group_unsafe":
        expected_keys = declared_keys or list(dict.fromkeys(str(key) for key in report.index))
        observations = tuple(
            MetricObservation(
                model_name=model_name,
                framework="synthcity",
                emitted_key=key,
                raw_value=None,
                role_hashes=role_hashes,
                source_metadata={
                    **source_metadata,
                    "report_state": "group_unsafe",
                    "group_safety": dict(group_safety),
                    "metric_metadata": report_metadata,
                    "result_metadata": _metric_result_metadata(report_metadata, key),
                },
                result_metadata=_metric_result_metadata(report_metadata, key),
                fit_roles=tuple(source_metadata.get("fit_roles", ())),
                support=_metric_result_metadata(report_metadata, key).get("support"),
                bandwidth=_metric_result_metadata(report_metadata, key).get("bandwidth"),
                provenance=source_metadata,
            )
            for key in expected_keys
        )
        return expected_keys, observations

    has_native_values = "mean" in report.columns or "direction" in report.columns
    if report.empty or not has_native_values:
        error_type = _row_value(report.iloc[0], "error_type") if not report.empty else None
        error = (
            f"metric_evaluation_failed; exception_type={error_type}"
            if error_type
            else "synthcity_report_empty"
        )
        observations = tuple(
            MetricObservation(
                model_name=model_name,
                framework="synthcity",
                emitted_key=key,
                raw_value=None,
                direction=None,
                error=str(error),
                role_hashes=role_hashes,
                source_metadata={
                    **source_metadata,
                    "report_state": "failed",
                    "metric_metadata": report_metadata,
                    "result_metadata": _metric_result_metadata(report_metadata, key),
                },
                result_metadata=_metric_result_metadata(report_metadata, key),
                fit_roles=tuple(source_metadata.get("fit_roles", ())),
                support=_metric_result_metadata(report_metadata, key).get("support"),
                bandwidth=_metric_result_metadata(report_metadata, key).get("bandwidth"),
                provenance=source_metadata,
            )
            for key in declared_keys
        )
        return declared_keys, observations

    emitted_keys = [str(key) for key in report.index]
    expected_keys = declared_keys or list(dict.fromkeys(emitted_keys))
    observations = []
    for emitted_key, (_, row) in zip(emitted_keys, report.iterrows(), strict=True):
        try:
            contract = DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
                framework="synthcity", emitted_key=emitted_key
            )
        except (ValueError, KeyError):
            contract = None
        uncertainty_field = contract.uncertainty_field if contract else "stddev"
        sample_size_field = contract.sample_size_field if contract else "rounds"
        direction = _row_value(row, "direction")
        if not isinstance(direction, str):
            direction = None
        uncertainty = _row_value(row, uncertainty_field) if uncertainty_field else None
        if not isinstance(uncertainty, Real):
            uncertainty = None
        sample_size = _row_sample_size(row, sample_size_field)
        result_metadata = _metric_result_metadata(report_metadata, emitted_key)
        observations.append(
            MetricObservation(
                model_name=model_name,
                framework="synthcity",
                emitted_key=emitted_key,
                raw_value=_row_value(row, "mean"),
                direction=direction,
                uncertainty=float(uncertainty) if uncertainty is not None else None,
                sample_size=sample_size,
                error=_row_error(row),
                role_hashes=role_hashes,
                source_metadata={
                    **source_metadata,
                    "report_state": "succeeded" if _row_error(row) is None else "failed",
                    "metric_metadata": report_metadata,
                    "uncertainty_field": uncertainty_field,
                    "sample_size_field": sample_size_field,
                    "rounds": sample_size,
                    "errors": _row_value(row, "errors"),
                    "durations": _row_value(row, "durations"),
                    "error_types": _row_value(row, "error_types"),
                    "error_messages": None,
                    "result_metadata": result_metadata,
                },
                result_metadata=result_metadata,
                fit_roles=tuple(
                    result_metadata.get("fit_roles", source_metadata.get("fit_roles", ()))
                ),
                support=result_metadata.get("support"),
                bandwidth=result_metadata.get("bandwidth"),
                provenance={**source_metadata, **result_metadata},
            )
        )
    return expected_keys, tuple(observations)


def build_canonical_synthcity_observations(
    model_name: str,
    values: Mapping[str, Any],
    *,
    fit_roles: tuple[str, ...],
    support: Any,
    bandwidth: Any = None,
    provenance: Mapping[str, Any] | None = None,
    role_hashes: Mapping[str, str] | None = None,
) -> tuple[str, tuple[MetricObservation, ...]]:
    """Build one canonical SynthCity observation per declared value.

    This narrow adapter accepts only canonical HPO identities; native report
    rows cannot expand or rename this manifest.
    """
    expected = tuple(values)
    if set(expected) != set(SYNTHCITY_CANONICAL_MANIFEST):
        raise ValueError("Canonical SynthCity values must contain exactly approved HPO identities")
    metadata = dict(provenance or {})
    metadata.update({"fit_roles": list(fit_roles), "support": support, "bandwidth": bandwidth})
    observations = tuple(
        MetricObservation(
            model_name=model_name,
            framework="synthcity",
            emitted_key=key,
            raw_value=values[key],
            role_hashes=dict(role_hashes or {}),
            fit_roles=fit_roles,
            support=support,
            bandwidth=bandwidth,
            provenance=metadata,
            source_metadata=metadata,
            result_metadata=metadata,
        )
        for key in SYNTHCITY_CANONICAL_MANIFEST
    )
    return model_name, observations


# Deprecated compatibility alias for historical callers.
build_task12_synthcity_observations = build_canonical_synthcity_observations


def validate_synthcity_report(
    model_name: str,
    report: pd.DataFrame,
    *,
    expected_keys: list[str] | None = None,
    expected_base_keys: list[str] | None = None,
    context: MetricEvaluationContext | None = None,
    requested_use: str = "audit",
) -> MetricValidationResult:
    """Validate native SynthCity rows without dropping raw failure evidence."""
    context = context or MetricEvaluationContext()
    expected_keys, observations = build_synthcity_observations(
        model_name,
        report,
        expected_keys=expected_keys,
        expected_base_keys=expected_base_keys,
        role_hashes=context.role_hashes,
        source_metadata={"requested_use": requested_use},
    )
    return resolve_metric_observations(
        registry=DEFAULT_METRIC_CONTRACT_REGISTRY,
        model_name=model_name,
        framework="synthcity",
        expected_keys=expected_keys,
        observations=observations,
        context=context,
        requested_use=requested_use,
    )


def validate_synthcity_results(
    synthcity_results: Mapping[str, pd.DataFrame],
    metric_config: Mapping[str, list[str]],
    *,
    role_hashes: Mapping[str, str] | None = None,
    requested_use: str = "audit",
    context: MetricEvaluationContext | None = None,
    variable_columns: list[str] | None = None,
    attack_target_types: Mapping[str, str] | None = None,
) -> dict[str, MetricValidationResult]:
    """Validate all model reports with one resolved selection and context."""
    expected_keys = _base_emitted_keys(
        metric_config,
        variable_columns=variable_columns,
        attack_target_types=attack_target_types,
    )
    context = context or MetricEvaluationContext(role_hashes=dict(role_hashes or {}))
    return {
        model_name: validate_synthcity_report(
            model_name,
            report,
            expected_keys=expected_keys,
            context=context,
            requested_use=requested_use,
        )
        for model_name, report in synthcity_results.items()
    }


class PregeneratedSyntheticModel:
    """Wraps an already-generated synthetic DataFrame in a fit/sample API.

    Bootstrap-resamples the cached data with a caller-supplied seed, so that
    independent-looking draws (e.g. for DomiasMIA's reference set) can be
    produced without re-running the original (possibly expensive) generator.
    """

    def __init__(self, synthetic_df: pd.DataFrame):
        self._df = synthetic_df.reset_index(drop=True)

    def fit(self, X):
        return self

    def sample(self, count: int, random_state: int = 0) -> pd.DataFrame:
        return self._df.sample(n=count, replace=True, random_state=random_state).reset_index(
            drop=True
        )


class _ExternalGeneratorAdapter:
    """Minimal adapter for external models exposing fit(X) and sample(count)."""

    def __init__(self, model, random_state: int = 0):
        self.model = model
        self.random_state = random_state

    def fit(self, X):
        self.model.fit(X)
        return self

    def generate(self, count: int, random_state: int | None = None) -> pd.DataFrame:
        seed = random_state if random_state is not None else self.random_state
        x_syn = self.model.sample(count, random_state=seed)
        if not isinstance(x_syn, pd.DataFrame):
            x_syn = pd.DataFrame(x_syn)
        return x_syn


def _align_dtypes(x_syn: pd.DataFrame, x_ref: pd.DataFrame) -> pd.DataFrame:
    for col in x_ref.columns:
        if x_ref[col].dtype != x_syn[col].dtype:
            try:
                x_syn[col] = x_syn[col].astype(x_ref[col].dtype)
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "[_align_dtypes] cast failed for column %s (ref dtype=%s, syn dtype=%s): %s",
                    col,
                    x_ref[col].dtype,
                    x_syn[col].dtype,
                    exc,
                )
    return x_syn


def _schema_mismatch_score(x_ref: pd.DataFrame, x_syn: pd.DataFrame) -> float:
    """Measure dtype incompatibility before any root-side coercion."""
    if list(x_ref.columns) != list(x_syn.columns):
        raise ValueError(
            "Schema comparison requires identical column order; "
            f"reference={list(x_ref.columns)!r}, synthetic={list(x_syn.columns)!r}"
        )
    mismatch_count = sum(
        reference_dtype != synthetic_dtype
        for reference_dtype, synthetic_dtype in zip(x_ref.dtypes, x_syn.dtypes, strict=True)
    )
    return mismatch_count / (len(x_ref.columns) + 1)


def _validated_group_ids(group_ids: Any, expected_length: int, label: str) -> list[Any] | None:
    if group_ids is None:
        return None
    values = list(group_ids)
    if len(values) != expected_length:
        raise ValueError(
            f"{label} group_ids length {len(values)} does not match frame length {expected_length}"
        )
    return values


def _synthetic_group_ids(count: int, namespace: str) -> list[tuple[str, int]]:
    return [(namespace, index) for index in range(count)]


def run_synthcity_metrics(
    synthetic_df: pd.DataFrame,
    x_real_reference: pd.DataFrame,
    x_real_train: pd.DataFrame,
    n_samples: int,
    target_column: str,
    sensitive_features: list,
    metrics: dict,
    task_type: str = "classification",
    random_state: int = 42,
    workspace: str | None = None,
    quasi_identifier_columns: list[str] | None = None,
    classification_score: str = "balanced_accuracy",
    sensitive_target_types: Mapping[str, str] | None = None,
    feature_types: Mapping[str, str] | None = None,
    source_table: Mapping[str, str] | None = None,
    real_reference_group_ids: Any = None,
    real_train_group_ids: Any = None,
    structural_n_clusters: list[int] | None = None,
    structural_min_rows_per_cluster: int = 10,
    evaluation_role: str = "tuning",
    group_mode: str = "row",
    semantic_context: Mapping[str, Any] | None = None,
    released_synthetic_df: pd.DataFrame | None = None,
    released_reference_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Evaluate a cached synthetic DataFrame with synthcity's Metrics.evaluate.

    Mirrors how synthcity's Benchmarks calls Metrics.evaluate internally:
      X_gt        = held-out real data
      X_syn       = synthetic (bootstrap draw, seed=random_state)
      X_train     = real training data (for DomiasMIA)
      X_ref_syn   = second independent synthetic draw (seed=random_state + 1)
      X_augmented = X_real_train concatenated with X_syn (for augmentation metrics)
    """
    from synthcity.metrics import Metrics
    from synthcity.plugins.core.dataloader import GenericDataLoader

    if classification_score not in {"balanced_accuracy", "macro_f1"}:
        raise ValueError(
            "classification_score must be 'balanced_accuracy' or 'macro_f1', "
            f"got {classification_score!r}"
        )
    if group_mode not in {"row", "patient_group"}:
        raise ValueError(f"Invalid group mode {group_mode!r}. Supported: ['row', 'patient_group']")
    if group_mode == "patient_group" and (
        real_reference_group_ids is None or real_train_group_ids is None
    ):
        raise ValueError(
            "Patient-group SynthCity evaluation requires group IDs for real reference and train roles"
        )

    released_synthetic_df = synthetic_df if released_synthetic_df is None else released_synthetic_df
    released_reference_df = (
        x_real_reference if released_reference_df is None else released_reference_df
    )
    adapter = _ExternalGeneratorAdapter(
        PregeneratedSyntheticModel(released_synthetic_df), random_state=random_state
    )
    adapter.fit(x_real_train)

    x_real_reference = released_reference_df
    x_syn_generated = released_synthetic_df.copy()[x_real_reference.columns]
    reference_group_ids = _validated_group_ids(
        real_reference_group_ids, len(x_real_reference), "real reference"
    )
    train_group_ids = _validated_group_ids(real_train_group_ids, len(x_real_train), "real train")
    if (reference_group_ids is None) != (train_group_ids is None):
        raise ValueError(
            "real_reference_group_ids and real_train_group_ids must be provided together"
        )
    synthetic_group_ids = (
        _synthetic_group_ids(len(x_syn_generated), "synthetic")
        if reference_group_ids is not None
        else None
    )
    schema_mismatch_score = _schema_mismatch_score(x_real_reference, x_syn_generated)
    x_syn_raw = _align_dtypes(x_syn_generated, x_real_reference)
    x_ref_syn_raw = _align_dtypes(
        released_synthetic_df.copy()[x_real_reference.columns], x_real_reference
    )
    x_augmented_raw = pd.concat([x_real_train, x_syn_raw], ignore_index=True)
    reference_synthetic_group_ids = (
        _synthetic_group_ids(len(x_ref_syn_raw), "reference_synthetic")
        if reference_group_ids is not None
        else None
    )
    augmented_group_ids = (
        train_group_ids + synthetic_group_ids
        if train_group_ids is not None and synthetic_group_ids is not None
        else None
    )

    def _loader(df: pd.DataFrame, group_ids: Any = None):
        return GenericDataLoader(
            df,
            target_column=target_column,
            sensitive_features=sensitive_features,
            important_features=quasi_identifier_columns or [],
            group_ids=group_ids,
            feature_types=dict(feature_types or {}),
            source_table=dict(source_table or {}),
        )

    results = Metrics.evaluate(
        _loader(x_real_reference, reference_group_ids),
        _loader(x_syn_raw, synthetic_group_ids),
        _loader(x_real_train, train_group_ids),
        _loader(x_ref_syn_raw, reference_synthetic_group_ids),
        _loader(x_augmented_raw, augmented_group_ids),
        metrics=metrics,
        task_type=task_type,
        group_mode=group_mode,
        random_state=random_state,
        workspace=_semantic_metric_workspace(
            workspace,
            target_column=target_column,
            sensitive_features=sensitive_features,
            metrics=metrics,
            task_type=task_type,
            evaluation_role=evaluation_role,
            group_mode=group_mode,
            feature_types=feature_types,
            source_table=source_table,
            sensitive_target_types=sensitive_target_types,
            quasi_identifier_columns=quasi_identifier_columns,
            classification_score=classification_score,
            semantic_context=semantic_context,
            structural_n_clusters=structural_n_clusters,
            structural_min_rows_per_cluster=structural_min_rows_per_cluster,
            group_ids={
                "reference": reference_group_ids,
                "train": train_group_ids,
            },
        ),
        quasi_identifier_columns=quasi_identifier_columns,
        classification_score=classification_score,
        sensitive_target_types=dict(sensitive_target_types or {}),
        semantic_context=dict(semantic_context) if semantic_context is not None else None,
        feature_types=dict(feature_types or {}),
        source_table=dict(source_table or {}),
        X_gt_group_ids=reference_group_ids,
        X_syn_group_ids=synthetic_group_ids,
        X_train_group_ids=train_group_ids,
        X_ref_syn_group_ids=reference_synthetic_group_ids,
        X_augmented_group_ids=augmented_group_ids,
        structural_n_clusters=structural_n_clusters,
        structural_min_rows_per_cluster=structural_min_rows_per_cluster,
    )
    data_mismatch_key = "sanity.data_mismatch.score"
    if "data_mismatch" in metrics.get("sanity", []) and data_mismatch_key in results.index:
        for column in ("min", "max", "mean", "median"):
            results.loc[data_mismatch_key, column] = schema_mismatch_score
        for column in ("stddev", "iqr"):
            results.loc[data_mismatch_key, column] = 0.0
        results.loc[data_mismatch_key, "rounds"] = 1
        results.loc[data_mismatch_key, "schema_source"] = "pre_dtype_alignment"
        results.loc[data_mismatch_key, "schema_reference_dtypes"] = ",".join(
            str(dtype) for dtype in x_real_reference.dtypes
        )
        results.loc[data_mismatch_key, "schema_synthetic_dtypes"] = ",".join(
            str(dtype) for dtype in x_syn_generated.dtypes
        )
    return results


def resolve_metric_config(selection_cfg) -> dict:
    """Filter SYNTHCITY_METRIC_CONFIG down to the configured selection."""
    all_names = [n for names in SYNTHCITY_METRIC_CONFIG.values() for n in names]
    name_to_type = {
        n: SYNTHCITY_CATEGORY_TO_TYPE[cat]
        for cat, names in SYNTHCITY_METRIC_CONFIG.items()
        for n in names
    }
    selected = resolve_selection(
        selection_cfg.enabled,
        selection_cfg.categories,
        selection_cfg.metrics,
        all_names,
        name_to_type,
    )
    metric_config = {
        cat: [n for n in names if n in selected] for cat, names in SYNTHCITY_METRIC_CONFIG.items()
    }
    return {cat: names for cat, names in metric_config.items() if names}


def run_synthcity_evaluation(
    synthetic_datasets: dict[str, pd.DataFrame],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    sensitive_features: list,
    selection_cfg,
    n_samples: int | None = None,
    seed: int = 42,
    workspace: str | None = None,
    sensitive_target_types: Mapping[str, str] | None = None,
    feature_types: Mapping[str, str] | None = None,
    source_table: Mapping[str, str] | None = None,
    quasi_identifier_columns: list[str] | None = None,
    classification_score: str | None = None,
    real_reference_group_ids: Any = None,
    real_train_group_ids: Any = None,
    task_type: str = "classification",
    evaluation_role: str = "tuning",
    group_mode: str = "row",
    semantic_context: Mapping[str, Any] | None = None,
    released_synthetic_datasets: Mapping[str, pd.DataFrame] | None = None,
    released_reference_frame: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Run synthcity Metrics on every cached synthetic dataset.

    Returns ``{model_name: DataFrame}`` where each DataFrame is indexed by
    metric key (e.g. ``"stats.wasserstein_dist.joint"``) with at least
    ``mean``/``direction`` columns, as returned by ``Metrics.evaluate``.
    """
    metric_config = resolve_metric_config(selection_cfg)
    if not metric_config:
        logger.info("[synthcity] no metrics selected; skipping")
        return {}
    if evaluation_role not in {"tuning", "final_holdout"}:
        raise ValueError(f"Unsupported SynthCity evaluation role: {evaluation_role!r}")
    if group_mode not in {"row", "patient_group"}:
        raise ValueError(f"Invalid group mode {group_mode!r}. Supported: ['row', 'patient_group']")

    results = {}
    resolved_sensitive_target_types = dict(
        sensitive_target_types or selection_cfg.sensitive_target_types
    )
    resolved_quasi_identifier_columns = (
        list(quasi_identifier_columns)
        if quasi_identifier_columns is not None
        else list(selection_cfg.quasi_identifier_columns)
    )
    resolved_classification_score = classification_score or selection_cfg.classification_score
    for name, syn_df in synthetic_datasets.items():
        n = n_samples or len(syn_df)
        logger.info("[synthcity] evaluating %s against %s evidence", name, evaluation_role)
        try:
            results[name] = run_synthcity_metrics(
                syn_df,
                test_df,
                train_df,
                n,
                target_column,
                sensitive_features,
                metric_config,
                task_type=task_type,
                group_mode=group_mode,
                random_state=seed,
                workspace=workspace,
                quasi_identifier_columns=(resolved_quasi_identifier_columns or None),
                classification_score=resolved_classification_score,
                sensitive_target_types=resolved_sensitive_target_types,
                feature_types=feature_types,
                source_table=source_table,
                semantic_context=semantic_context,
                real_reference_group_ids=real_reference_group_ids,
                real_train_group_ids=real_train_group_ids,
                structural_n_clusters=selection_cfg.structural_n_clusters,
                structural_min_rows_per_cluster=selection_cfg.structural_min_rows_per_cluster,
                evaluation_role=evaluation_role,
                released_synthetic_df=(
                    released_synthetic_datasets.get(name)
                    if released_synthetic_datasets is not None
                    else None
                ),
                released_reference_df=released_reference_frame,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            logger.warning(
                "[synthcity] evaluation failed for %s; reason_code=metric_evaluation_failed "
                "exception_type=%s",
                name,
                type(exc).__name__,
            )
            results[name] = pd.DataFrame(
                {
                    "error": ["SynthCity metric evaluation failed."],
                    "error_type": [type(exc).__name__],
                }
            )
    return results
