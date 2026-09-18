"""synthcity plugin fit/generate/hyperparameter-search glue."""

import hashlib
import importlib
import importlib.metadata
import inspect
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, cast

import optuna
import pandas as pd
import torch

from synthdata.config import HPOConfig
from synthdata.data import semantic_context_digest
from synthdata.evaluation.catalog import emitted_keys_for_synthcity_metrics
from synthdata.generation.hpo import (
    HPO_GENERATOR_METADATA_SCHEMA_VERSION,
    StageAScreenContract,
    _resolve_utility_policy,
    _safe_exception_message,
    evaluate_canonical_hpo_metrics,
    hpo_score,
    persist_stage_a_trial_exception,
    prepare_stage_a_screen,
    screen_stage_a_trial,
    validate_hpo_metric_config,
)
from synthdata.utils import get_logger

logger = get_logger(__name__)


class _BenchmarksAPI(Protocol):
    @staticmethod
    def evaluate(
        generators: list[tuple[str, str, dict]], train_loader: object, **kwargs: object
    ) -> dict[str, pd.DataFrame]: ...


GENERATOR_METADATA_SCHEMA_VERSION = "generator-metadata-v1"
_PATE_ACCOUNTING_SCHEMA_VERSION = "pate-accounting-v1"
_PATE_PRIVACY_CLAIM_TYPE = "formal_dp"
_PATE_ACCOUNTING_PARAMETERS = ("epsilon", "delta", "alpha", "lamda")
_PATE_ACCOUNTING_METADATA_KEYS = {
    "schema_version",
    "privacy_claim_type",
    "accountant",
    "requested_epsilon",
    "requested_delta",
    "requested_alpha",
    "requested_lamda",
    "resolved_epsilon",
    "resolved_delta",
    "resolved_alpha",
    "resolved_lamda",
    "effective_epsilon",
    "effective_delta",
    "effective_alpha",
    "effective_lamda",
    "iterations",
    "max_iter",
    "stopping_state",
}


def make_loader(
    df: pd.DataFrame,
    target_column: str,
    sensitive_features: list,
    random_state: int = 0,
    fairness_column: str | None = None,
    group_ids=None,
    important_features: list | None = None,
    feature_types: dict[str, str] | None = None,
    release_generalization: dict | None = None,
    source_table: dict[str, str] | None = None,
):
    from synthcity.plugins.core.dataloader import GenericDataLoader

    kwargs = dict(
        target_column=target_column,
        sensitive_features=sensitive_features,
        random_state=random_state,
        important_features=list(important_features or []),
        group_ids=group_ids,
        feature_types=dict(feature_types or {}),
        source_table=dict(source_table or {}),
    )
    if fairness_column:
        kwargs["fairness_column"] = fairness_column
    return GenericDataLoader(df, **kwargs)


def get_plugin_class(name: str):
    from synthcity.plugins import Plugins

    return Plugins().get_type(name)


def plugin_accepts(name: str, param_name: str) -> bool:
    cls = get_plugin_class(name)
    sig = inspect.signature(cls.__init__)
    return param_name in sig.parameters


def generator_implementation_fingerprint(name: str) -> str:
    base_name = name.removesuffix("_hpo")
    source = []
    if base_name.startswith("tabpfn_"):
        module = importlib.import_module("synthdata.generation.tabpfn_backend")
        source_path = Path(module.__file__ or "")
        source.append(source_path.read_text())
    elif base_name.startswith("tabpfgen_"):
        module = importlib.import_module("synthdata.generation.tabpfgen_backend")
        source_path = Path(module.__file__ or "")
        source.append(source_path.read_text())
        try:
            source.append(inspect.getsource(module.TabPFGen))
        except (OSError, TypeError):
            source.append(f"{module.TabPFGen.__module__}.{module.TabPFGen.__qualname__}")
    else:
        plugin_cls = get_plugin_class(base_name)
        source.append(inspect.getsource(plugin_cls))
        base_cls = plugin_cls.__mro__[1] if len(plugin_cls.__mro__) > 1 else None
        if base_cls is not None and base_cls.__module__.startswith("synthcity"):
            source.append(inspect.getsource(base_cls))

    distributions = ["synthdata", "numpy", "pandas", "torch"]
    if base_name.startswith("tabpfn_"):
        distributions.extend(["tabpfn", "tabpfn-extensions"])
    elif base_name.startswith("tabpfgen_"):
        distributions.extend(["tabpfgen", "tabpfn", "tabpfn-extensions"])
    else:
        distributions.extend(["synthcity", "xgboost"])
    versions = {}
    for distribution in sorted(set(distributions)):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None

    payload = {
        "schema_version": "generator-implementation-v1",
        "generator": base_name,
        "source": source,
        "dependencies": versions,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def generator_metadata_context(name: str, params: dict) -> dict:
    base_name = name.removesuffix("_hpo")
    if base_name != "pategan":
        return {"privacy_claim_type": "none"}

    plugin_signature = inspect.signature(get_plugin_class(base_name).__init__)
    requested_accounting = {}
    for parameter_name in _PATE_ACCOUNTING_PARAMETERS:
        parameter = plugin_signature.parameters.get(parameter_name)
        if parameter is None or parameter.default is inspect.Parameter.empty:
            raise RuntimeError(
                f"PATE-GAN plugin signature does not define a default for {parameter_name!r}"
            )
        requested_accounting[parameter_name] = params.get(parameter_name, parameter.default)
    return {
        "privacy_claim_type": "formal_dp",
        "accountant": "pate_moments_v1",
        "requested_accounting": requested_accounting,
    }


def generator_metadata_is_valid(metadata: object, expected_context: dict) -> bool:
    if not isinstance(metadata, dict):
        return False
    required_fields = (
        "schema_version",
        "generator_context",
        "plugin_name",
        "plugin_fqdn",
        "requested_parameters",
        "n_samples",
        "random_state",
        "privacy_accounting",
    )
    if any(field not in metadata for field in required_fields):
        return False
    schema_version = metadata.get("schema_version")
    if not (
        schema_version
        in {
            GENERATOR_METADATA_SCHEMA_VERSION,
            HPO_GENERATOR_METADATA_SCHEMA_VERSION,
        }
        and metadata.get("generator_context") == expected_context
        and all(
            isinstance(metadata.get(field), str) and metadata[field].strip()
            for field in ("plugin_name", "plugin_fqdn")
        )
        and isinstance(metadata.get("requested_parameters"), dict)
        and isinstance(metadata.get("n_samples"), int)
        and not isinstance(metadata.get("n_samples"), bool)
        and metadata["n_samples"] > 0
        and isinstance(metadata.get("random_state"), int)
        and not isinstance(metadata.get("random_state"), bool)
    ):
        return False
    if schema_version == HPO_GENERATOR_METADATA_SCHEMA_VERSION and (
        not isinstance(metadata.get("implementation_fingerprint"), str)
        or not metadata["implementation_fingerprint"].strip()
    ):
        return False
    if expected_context.get("privacy_claim_type") == "none":
        return metadata["privacy_accounting"] is None
    accounting = metadata.get("privacy_accounting")
    if not isinstance(accounting, dict) or not _PATE_ACCOUNTING_METADATA_KEYS.issubset(accounting):
        return False
    if (
        accounting.get("schema_version") != _PATE_ACCOUNTING_SCHEMA_VERSION
        or accounting.get("privacy_claim_type") != _PATE_PRIVACY_CLAIM_TYPE
        or accounting.get("accountant") != expected_context.get("accountant")
    ):
        return False
    expected_accounting = expected_context.get("requested_accounting", {})
    for parameter_name, expected_value in expected_accounting.items():
        actual_value = accounting.get(f"requested_{parameter_name}")
        if actual_value != expected_value:
            return False
    return (
        all(
            accounting.get(field) is not None
            for field in (
                "resolved_epsilon",
                "resolved_delta",
                "resolved_alpha",
                "resolved_lamda",
                "effective_epsilon",
                "effective_delta",
                "effective_alpha",
                "effective_lamda",
            )
        )
        and accounting.get("stopping_state") != "not_fitted"
    )


def generator_metadata_matches_request(
    metadata: object,
    name: str,
    params: dict,
    n_samples: int,
    random_state: int,
) -> bool:
    expected_context = generator_metadata_context(name, params)
    if not generator_metadata_is_valid(metadata, expected_context):
        return False
    if not isinstance(metadata, dict):
        return False
    return (
        metadata.get("plugin_name") == name.removesuffix("_hpo")
        and metadata.get("requested_parameters") == dict(params)
        and metadata.get("n_samples") == n_samples
        and metadata.get("random_state") == random_state
    )


def build_generator_metadata(
    name: str,
    params: dict,
    n_samples: int,
    random_state: int,
    *,
    plugin_fqdn: str,
    privacy_accounting: dict | None = None,
    implementation_fingerprint: str | None = None,
) -> dict:
    """Build and validate the common metadata record for one generation call."""
    generator_context = generator_metadata_context(name, params)
    metadata = {
        "schema_version": (
            HPO_GENERATOR_METADATA_SCHEMA_VERSION
            if implementation_fingerprint is not None
            else GENERATOR_METADATA_SCHEMA_VERSION
        ),
        "generator_context": generator_context,
        "plugin_name": name,
        "plugin_fqdn": plugin_fqdn,
        "requested_parameters": dict(params),
        "n_samples": int(n_samples),
        "random_state": int(random_state),
        "privacy_accounting": privacy_accounting,
    }
    if implementation_fingerprint is not None:
        metadata["implementation_fingerprint"] = implementation_fingerprint
    if not generator_metadata_is_valid(metadata, generator_context):
        raise RuntimeError(
            f"Plugin {name!r} did not return complete generator metadata for "
            f"privacy claim {generator_context['privacy_claim_type']!r}"
        )
    return metadata


def build_hpo_generator_metadata(
    name: str,
    params: dict,
    synthetic_size: int | None,
    random_state: int,
    metric_metadata: dict | None,
) -> dict:
    """Normalize benchmark runtime metadata for durable HPO checkpoints."""
    base_name = name.removesuffix("_hpo")
    expected_context = generator_metadata_context(base_name, params)
    runtime = None
    if isinstance(metric_metadata, dict):
        candidate = metric_metadata.get(f"generator.{base_name}")
        if isinstance(candidate, dict):
            runtime = candidate

    accounting = None
    implementation_fingerprint = generator_implementation_fingerprint(base_name)
    if expected_context["privacy_claim_type"] == "formal_dp":
        required_runtime_fields = (
            "schema_version",
            "plugin_name",
            "plugin_fqdn",
            "requested_parameters",
            "n_samples",
            "random_state",
            "privacy_claim_type",
            "accounting",
        )
        if runtime is None:
            raise RuntimeError(
                f"HPO benchmark did not return PATE generator metadata for {base_name!r}"
            )
        missing_runtime = [field for field in required_runtime_fields if field not in runtime]
        if missing_runtime:
            raise RuntimeError(
                f"HPO benchmark returned incomplete PATE metadata for {base_name!r}; "
                f"missing {missing_runtime}"
            )
        if runtime["schema_version"] != "generator-runtime-metadata-v2":
            raise RuntimeError(
                f"HPO benchmark returned unsupported PATE metadata schema "
                f"{runtime['schema_version']!r} for {base_name!r}"
            )
        if runtime["plugin_name"] != base_name:
            raise RuntimeError(
                f"HPO benchmark PATE metadata names plugin {runtime['plugin_name']!r}, "
                f"expected {base_name!r}"
            )
        if runtime["privacy_claim_type"] != "formal_dp":
            raise RuntimeError(
                f"HPO benchmark PATE metadata has claim type {runtime['privacy_claim_type']!r}"
            )
        if not isinstance(runtime["requested_parameters"], dict):
            raise RuntimeError(
                f"HPO benchmark PATE requested_parameters must be an object for {base_name!r}"
            )
        if (
            isinstance(runtime["n_samples"], bool)
            or not isinstance(runtime["n_samples"], int)
            or runtime["n_samples"] <= 0
        ):
            raise RuntimeError(
                f"HPO benchmark PATE n_samples must be a positive integer for {base_name!r}"
            )
        if isinstance(runtime["random_state"], bool) or not isinstance(
            runtime["random_state"], int
        ):
            raise RuntimeError(
                f"HPO benchmark PATE random_state must be an integer for {base_name!r}"
            )
        if (
            generator_metadata_context(base_name, runtime["requested_parameters"])
            != expected_context
        ):
            raise RuntimeError(
                f"HPO benchmark PATE requested accounting does not match HPO parameters for "
                f"{base_name!r}"
            )
        if not isinstance(runtime["accounting"], dict):
            raise RuntimeError(
                f"HPO benchmark did not return PATE accounting metadata for {base_name!r}"
            )
        accounting = runtime["accounting"]

    requested_parameters = (
        runtime.get("requested_parameters")
        if runtime is not None and isinstance(runtime.get("requested_parameters"), dict)
        else params
    )
    plugin_fqdn = (
        runtime.get("plugin_fqdn")
        if runtime is not None and isinstance(runtime.get("plugin_fqdn"), str)
        else f"synthcity.{base_name}"
    )
    n_samples = runtime.get("n_samples") if runtime is not None else synthetic_size
    if n_samples is None:
        n_samples = 1
    trial_random_state = runtime.get("random_state") if runtime is not None else random_state
    metadata = build_generator_metadata(
        base_name,
        requested_parameters,
        int(n_samples),
        int(trial_random_state),
        plugin_fqdn=plugin_fqdn,
        privacy_accounting=accounting,
        implementation_fingerprint=implementation_fingerprint,
    )
    return json.loads(json.dumps(metadata, sort_keys=True, default=str, allow_nan=False))


def fit_generate(
    name: str,
    params: dict,
    train_loader,
    n_samples: int,
    random_state: int = 42,
    workspace: str | None = None,
    device: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    from pathlib import Path

    from synthcity.plugins import Plugins

    plugin_kwargs = dict(params)
    if workspace is not None and plugin_accepts(name, "workspace"):
        plugin_kwargs["workspace"] = Path(workspace)
    if device is not None and "device" not in plugin_kwargs and plugin_accepts(name, "device"):
        plugin_kwargs["device"] = torch.device(device)

    model = Plugins().get(name, **plugin_kwargs)
    model.fit(train_loader)
    synthetic_loader = model.generate(
        count=n_samples,
        random_state=random_state,
        _group_namespace="synthetic",
    )
    accounting_metadata = None
    accounting_getter = getattr(model, "get_accounting_metadata", None)
    if accounting_getter is not None:
        if not callable(accounting_getter):
            raise TypeError(f"Plugin {name!r} exposes a non-callable accounting accessor")
        accounting_metadata = accounting_getter()
        if not isinstance(accounting_metadata, dict):
            raise TypeError(
                f"Plugin {name!r} accounting metadata must be a dict, got "
                f"{type(accounting_metadata).__name__}"
            )

    metadata = build_generator_metadata(
        name,
        params,
        n_samples,
        random_state,
        plugin_fqdn=model.fqdn(),
        privacy_accounting=accounting_metadata,
    )
    return synthetic_loader.dataframe(), metadata


def build_synthcity_objective(
    name: str,
    train_loader,
    hpo_cfg: HPOConfig,
    seed: int,
    workspace: str | None = None,
    device: str = "cpu",
    *,
    tuning_loader=None,
    task_type: str = "classification",
    classification_score: str = "balanced_accuracy",
    semantic_context: dict | None = None,
    synthetic_size: int | None = None,
    stage_a_contract: StageAScreenContract | None = None,
    stage_a_source_df: pd.DataFrame | None = None,
    stage_a_root: str | None = None,
    study_name: str | None = None,
    group_context: dict | None = None,
    expected_emitted_keys: list[str] | None = None,
    train_df: pd.DataFrame | None = None,
    tuning_df: pd.DataFrame | None = None,
    target_column: str | None = None,
    feature_types: dict[str, str] | None = None,
    release_generalization: dict | None = None,
):
    """Build an Optuna objective for a synthcity plugin's native hyperparameter space.

    Mirrors the hepatitis notebook's HPO cell: samples from the plugin's own
    ``sample_hyperparameters_optuna``, caps ``n_iter`` for speed (only if the
    plugin exposes it), forces CPU for MPS (which lacks the float64 support
    synthcity's metrics need internally) but otherwise uses ``device``, and
    scores each trial via a single-model, single-repeat ``Benchmarks.evaluate``
    call.
    """
    validate_hpo_metric_config(hpo_cfg.metric_config, group_context=group_context)
    configured_keys = {
        str(metric_name)
        for metric_names in hpo_cfg.metric_config.values()
        for metric_name in metric_names
    }
    canonical = configured_keys.issubset({"elastic_net_jsd.v1", "mixed_mmd.v1", "tstr_macro_f1.v1"})
    if canonical and (train_df is None or tuning_df is None or target_column is None):
        raise ValueError("Canonical SynthCity HPO requires train_df, tuning_df, and target_column")
    utility_policy = _resolve_utility_policy(hpo_cfg.utility_policy) if canonical else None
    expected_keys = (
        list(expected_emitted_keys)
        if expected_emitted_keys is not None
        else list(utility_policy["metrics"] if utility_policy is not None else ())
        if canonical
        else emitted_keys_for_synthcity_metrics(hpo_cfg.metric_config)
    )
    if len(expected_keys) != len(set(expected_keys)):
        raise ValueError("HPO expected emitted metric keys must be unique")
    group_mode = group_context.get("group_mode", "row") if group_context else "row"
    if group_mode not in {"row", "patient_group"}:
        raise ValueError(f"Invalid group mode {group_mode!r}. Supported: ['row', 'patient_group']")
    if train_loader is None or tuning_loader is None:
        raise ValueError("SynthCity HPO requires explicit train and tuning loaders")
    if group_context and group_context.get("group_mode") == "patient_group":
        missing_group_loaders = [
            label
            for label, loader in (("train", train_loader), ("tuning", tuning_loader))
            if getattr(loader, "group_ids", None) is None
        ]
        if missing_group_loaders:
            raise ValueError(
                "Patient-group HPO objectives require group IDs for loaders: "
                + ", ".join(missing_group_loaders)
            )

    from pathlib import Path

    plugin_cls = get_plugin_class(name)
    base_name = name.removesuffix("_hpo")
    implementation_fingerprint = generator_implementation_fingerprint(base_name)
    accepts_device = plugin_accepts(name, "device")
    accepts_iter = plugin_accepts(name, "n_iter")
    iter_cap = hpo_cfg.model_iter_caps.get(name, hpo_cfg.n_iter_cap)
    workspace_path = Path(workspace) if workspace else Path("workspace")
    semantic_digest = (
        semantic_context_digest(semantic_context) if semantic_context is not None else None
    )
    if semantic_digest is not None:
        workspace_path = workspace_path / f"semantic-{semantic_digest[:16]}"
    prepare_stage_a_screen(stage_a_contract, stage_a_source_df, stage_a_root, study_name)
    trial_device = "cpu" if device == "mps" else device

    def set_trial_attr(trial: optuna.Trial, key: str, value) -> None:
        setter = getattr(trial, "set_user_attr", None)
        if setter is None:
            return
        if not callable(setter):
            raise TypeError(f"Optuna trial attribute setter for {key!r} is not callable")
        setter(key, value)

    def objective(trial: optuna.Trial) -> float:
        set_trial_attr(trial, "generator_plugin_name", base_name)
        set_trial_attr(
            trial,
            "generator_privacy_claim_type",
            "formal_dp" if base_name == "pategan" else "none",
        )
        set_trial_attr(trial, "generator_metadata_state", "not_attempted")
        set_trial_attr(
            trial,
            "generator_implementation_fingerprint",
            implementation_fingerprint,
        )
        try:
            params = plugin_cls.sample_hyperparameters_optuna(trial)
            if accepts_iter:
                params["n_iter"] = min(params.get("n_iter", iter_cap), iter_cap)
            params["random_state"] = seed
            if accepts_device:
                params["device"] = torch.device(trial_device)
            set_trial_attr(trial, "generator_metadata_state", "pending")
        except (TypeError, ValueError, RuntimeError) as exc:
            if stage_a_contract is not None:
                if stage_a_root is None or study_name is None:
                    raise RuntimeError("Stage A screening context is incomplete") from exc
                persist_stage_a_trial_exception(
                    trial,
                    stage_a_root,
                    study_name,
                    stage_a_contract,
                    exc,
                )
            logger.warning(
                "[%s] Stage A candidate construction failed for trial %d: %s (%s)",
                name,
                trial.number,
                "stage_a_screen_exception",
                type(exc).__name__,
            )
            raise optuna.TrialPruned(
                f"stage_a_screen_exception: {_safe_exception_message('stage_a_screen_exception')} "
                f"[{type(exc).__name__}]"
            ) from exc

        trial_id = f"trial_{trial.number}"
        candidate_screen = None
        if stage_a_contract is not None:
            if stage_a_root is None or study_name is None or stage_a_source_df is None:
                raise RuntimeError("Stage A screening context is incomplete")

            def candidate_screen(candidate_df: pd.DataFrame):
                return screen_stage_a_trial(
                    trial,
                    candidate_df,
                    stage_a_contract,
                    stage_a_source_df,
                    stage_a_root,
                    study_name,
                )

        try:
            set_trial_attr(
                trial, "semantic_context", dict(semantic_context)
            ) if semantic_context is not None else None
            set_trial_attr(
                trial, "semantic_context_digest", semantic_digest
            ) if semantic_digest is not None else None
            if canonical:
                if train_df is None or tuning_df is None or target_column is None:
                    raise RuntimeError("Canonical SynthCity HPO inputs were not resolved")
                candidate_df, generator_metadata = fit_generate(
                    name,
                    params,
                    train_loader,
                    synthetic_size or len(train_loader),
                    seed,
                    workspace=str(workspace_path),
                    device=trial_device,
                )
                if candidate_screen is not None:
                    candidate_screen(candidate_df)
                metric_report = evaluate_canonical_hpo_metrics(
                    train_df,
                    tuning_df,
                    candidate_df,
                    metric_config=hpo_cfg.metric_config,
                    target_column=target_column,
                    feature_types=feature_types,
                    sensitive_features=(),
                    seed=seed,
                    release_generalization=release_generalization,
                    utility_policy=utility_policy,
                )
                metric_report.attrs["metric_metadata"] = {
                    "producer": "synthdata.generation.hpo.evaluate_canonical_hpo_metrics",
                    "fit_roles": ["train"],
                    "evaluation_role": "tuning",
                    "candidate_train_only": True,
                }
            else:
                from synthcity.benchmark import Benchmarks

                evaluate_kwargs = {
                    "X_test": tuning_loader,
                    "repeats": 1,
                    "metrics": hpo_cfg.metric_config,
                    "task_type": task_type,
                    "group_mode": group_mode,
                    "classification_score": classification_score,
                    "semantic_context": semantic_context,
                    "workspace": str(workspace_path),
                    "fit_on_X": True,
                }
                if synthetic_size is not None:
                    evaluate_kwargs["synthetic_size"] = synthetic_size
                if candidate_screen is not None:
                    evaluate_kwargs["candidate_screen"] = candidate_screen
                report = cast(_BenchmarksAPI, Benchmarks).evaluate(
                    [(trial_id, name, params)], train_loader, **evaluate_kwargs
                )
                if not isinstance(report, dict) or trial_id not in report:
                    raise RuntimeError(
                        f"SynthCity benchmark did not return report for {trial_id!r}"
                    )
                metric_report = report[trial_id]
            group_safety = getattr(metric_report, "attrs", {}).get("group_safety")
            if isinstance(group_safety, Mapping) and group_safety.get("status") == "group_unsafe":
                reason_code = "group_unsafe"
                set_trial_attr(
                    trial,
                    "hpo_outcome",
                    {
                        "status": "group_unsafe",
                        "group_safety": {"status": "group_unsafe", "reason_code": reason_code},
                    },
                )
                logger.warning(
                    "[%s] pruning trial %d because grouped evaluation is unsafe: %s",
                    name,
                    trial.number,
                    reason_code,
                )
                raise optuna.TrialPruned(
                    "group_unsafe: Grouped evaluation unsafe; details suppressed."
                )
            metric_metadata = metric_report.attrs.get("metric_metadata", {})
            if metric_metadata:
                set_trial_attr(trial, "metric_metadata", metric_metadata)
                set_trial_attr(trial, "result_metadata", metric_metadata)
            if not canonical:
                generator_metadata = build_hpo_generator_metadata(
                    base_name, params, synthetic_size, seed, metric_metadata
                )
            set_trial_attr(trial, "generator_metadata", generator_metadata)
            set_trial_attr(trial, "generator_metadata_state", "present")
            return hpo_score(
                metric_report,
                expected_keys=expected_keys,
                utility_policy=utility_policy,
            )
        except optuna.TrialPruned:
            if getattr(trial, "user_attrs", {}).get("generator_metadata_state") != "present":
                set_trial_attr(trial, "generator_metadata_state", "missing")
            raise
        except (TypeError, ValueError, RuntimeError) as exc:
            if getattr(trial, "user_attrs", {}).get("generator_metadata_state") != "present":
                set_trial_attr(trial, "generator_metadata_state", "missing")
            if stage_a_contract is not None and not getattr(trial, "user_attrs", {}).get(
                "stage_a_state"
            ):
                if stage_a_root is None or study_name is None:
                    raise RuntimeError("Stage A screening context is incomplete") from exc
                persist_stage_a_trial_exception(
                    trial,
                    stage_a_root,
                    study_name,
                    stage_a_contract,
                    exc,
                )
            reason_code = "hpo_trial_exception"
            logger.warning(
                "[%s] trial %d failed: %s (%s)",
                name,
                trial.number,
                reason_code,
                type(exc).__name__,
            )
            raise optuna.TrialPruned(
                f"{reason_code}: {_safe_exception_message(reason_code)} [{type(exc).__name__}]"
            ) from exc

    return objective
