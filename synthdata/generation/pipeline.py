"""Orchestrates synthetic data generation across synthcity, TabPFN, and TabPFGen.

For each enabled model family, generates a default synthetic dataset and,
if ``generation.hpo.enabled``, an Optuna-tuned ``*_hpo`` variant. Everything is
cached to CSV under ``generation.output_dir`` (skip regeneration unless
``generation.force_retrain``), and best hyperparameters are cached to a shared
JSON file (see :class:`synthdata.generation.hpo.BestParamsCache`).
"""

import hashlib
import json
import os
import re
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import pandas as pd

from synthdata.config import Config
from synthdata.data import (
    Dataset,
    dataframe_fingerprint,
    role_context_fingerprint,
    role_context_payload,
    semantic_context_digest,
    semantic_context_payload,
    validate_final_imputation_lineage,
    validate_imputation_cache_lineage,
)
from synthdata.evaluation.metric_contracts import DEFAULT_METRIC_CONTRACT_REGISTRY
from synthdata.evaluation.release import PROTOCOL_VERSION, _release_transform_digest
from synthdata.generation import hpo as hpo_mod
from synthdata.generation import synthcity_backend as sc
from synthdata.generation import tabpfn_backend as tpfn
from synthdata.imputation.pipeline import _cache_key_record
from synthdata.utils import (
    ensure_dir,
    get_logger,
    load_json,
    resolve_device,
    save_json,
    set_global_seed,
)

# tabpfgen_backend is imported lazily (see the `gen_cfg.tabpfgen.enabled` branch
# below) because it does `from tabpfgen import TabPFGen` at module scope (to
# subclass TabPFGenSGLDLabels), which requires the optional `tabpfn` extra.
# Keeping it out of this module's top-level imports lets `synthdata.generation`
# (and anything testing config/caching/other-backend logic) import cleanly with
# only the base dependencies installed.

logger = get_logger(__name__)

GENERATION_CACHE_SCHEMA_VERSION = "generation-cache-v3"
FINAL_REFIT_CACHE_SCHEMA_VERSION = "final-refit-v3"


def _structured_generator_metadata(value) -> dict | None:
    if not isinstance(value, dict):
        return None
    if value.get("schema_version") != sc.GENERATOR_METADATA_SCHEMA_VERSION:
        return None
    return value


def _resolve_generator_metadata(
    value,
    *,
    name: str,
    params: dict,
    n_samples: int,
    random_state: int,
    generator_identity: str,
) -> dict:
    expected_context = sc.generator_metadata_context(name, params)
    metadata_name = name.removesuffix("_hpo")
    if isinstance(value, dict):
        metadata = _structured_generator_metadata(value)
        if metadata is None:
            raise RuntimeError(f"Generator {name!r} returned metadata with an unsupported schema")
    elif expected_context.get("privacy_claim_type") == "none":
        metadata = sc.build_generator_metadata(
            metadata_name,
            params,
            n_samples,
            random_state,
            plugin_fqdn=generator_identity,
        )
    else:
        raise RuntimeError(
            f"Generator {name!r} did not return metadata for privacy claim "
            f"{expected_context['privacy_claim_type']!r}"
        )
    if not sc.generator_metadata_is_valid(metadata, expected_context):
        raise RuntimeError(
            f"Generator {name!r} returned incomplete metadata for privacy claim "
            f"{expected_context['privacy_claim_type']!r}"
        )
    return metadata


def needs_imputed_data(gen_cfg) -> bool:
    """Whether any configured model actually requires the imputed train split.

    synthcity and TabPFGen always fit on ``train_imputed_df``. TabPFN can fit on
    either split (see ``TabPFNConfig.data_variants``); it only needs imputed data
    if "imputed" is one of the requested variants.
    """
    return (
        (gen_cfg.synthcity.enabled and bool(gen_cfg.synthcity.names))
        or gen_cfg.tabpfgen.enabled
        or (gen_cfg.tabpfn.enabled and "imputed" in gen_cfg.tabpfn.data_variants)
    )


def _role_frame(dataset: Dataset, role: str, *, imputed: bool) -> pd.DataFrame | None:
    """Return a named role, retaining only train compatibility fallback."""
    frame = dataset.role_frame(role, imputed=imputed)
    if frame is not None:
        return frame
    if dataset.legacy_two_role:
        if role == "train":
            return dataset.train_imputed_df if imputed else dataset.train_df
        if role == "final_holdout":
            raise RuntimeError(
                "Legacy final_holdout access is blocked in generation helpers; "
                "final_holdout is never a tuning or fit role"
            )
    return None


def _required_role_frame(dataset: Dataset, role: str, *, imputed: bool) -> pd.DataFrame:
    """Return populated canonical role required for HPO provenance."""
    frame = _role_frame(dataset, role, imputed=imputed)
    if frame is None:
        representation = "imputed" if imputed else "raw"
        raise RuntimeError(f"HPO provenance requires a populated {representation} {role} role")
    return frame


def _combine_canonical_roles(dataset: Dataset, *, imputed: bool) -> pd.DataFrame:
    """Build the post-selection fit frame from train plus tuning only."""
    dataset.require_canonical_roles("final model refit")
    train_frame = dataset.role_frame("train", imputed=imputed)
    tuning_frame = dataset.role_frame("tuning", imputed=imputed)
    if train_frame is None or tuning_frame is None:
        representation = "imputed" if imputed else "raw"
        raise RuntimeError(
            f"Final model refit requires populated raw/imputed train and tuning roles "
            f"({representation})"
        )
    return pd.concat([train_frame, tuning_frame], ignore_index=True)


def _build_stage_a_contract(
    cfg: Config,
    dataset: Dataset,
    train_imputed_frame: pd.DataFrame | None,
) -> hpo_mod.StageAScreenContract | None:
    """Build the same Stage A contract used by generation HPO searches."""
    gen_cfg = cfg.generation
    if not gen_cfg.hpo.enabled or not (
        (gen_cfg.synthcity.enabled and bool(gen_cfg.synthcity.names)) or gen_cfg.tabpfgen.enabled
    ):
        return None
    if train_imputed_frame is None:
        raise RuntimeError("Stage A HPO screening requires an imputed train role")
    stage_a_cfg = gen_cfg.hpo.stage_a
    role_context_roles = ("train", "tuning")
    return hpo_mod.build_stage_a_contract(
        train_imputed_frame,
        expected_n_samples=gen_cfg.n_samples,
        target_column=dataset.target_column,
        target_is_categorical=dataset.target_is_categorical,
        categorical_columns=dataset.categorical_columns,
        protected_columns=dataset.protected_columns,
        minimum_target_count=stage_a_cfg.minimum_class_count,
        minimum_protected_group_count=stage_a_cfg.minimum_protected_group_count,
        minimum_target_by_protected_group_count=(
            stage_a_cfg.minimum_target_by_protected_group_count
        ),
        source_role="train",
        source_frame_fingerprint=dataframe_fingerprint(train_imputed_frame),
        dependency_rules=tuple(stage_a_cfg.dependency_rules),
        registry_digest=DEFAULT_METRIC_CONTRACT_REGISTRY.digest(),
        role_context_fingerprint=role_context_fingerprint(dataset, role_context_roles),
        role_context=role_context_payload(dataset, role_context_roles),
        group_context=_build_hpo_group_context(cfg, dataset, role_context_roles),
        hpo_context={
            "metric_config": {
                str(category): list(metric_names)
                for category, metric_names in gen_cfg.hpo.metric_config.items()
            },
            "n_samples": gen_cfg.n_samples,
            "n_iter_cap": gen_cfg.hpo.n_iter_cap,
            "model_iter_caps": dict(gen_cfg.hpo.model_iter_caps),
            "sgld_step_cap": gen_cfg.hpo.sgld_step_cap,
            "seed": cfg.seed,
            "device": str(cfg.device),
        },
    )


def _build_hpo_group_context(cfg: Config, dataset: Dataset, roles: tuple[str, ...]) -> dict:
    """Describe HPO group inputs without persisting raw population identifiers."""
    context = {
        "schema_version": "generation-hpo-group-v1",
        "group_mode": cfg.evaluation.group_mode,
        "group_column": cfg.evaluation.group_column,
        "roles": {},
    }
    for role in roles:
        frame = dataset.role_frame(role, imputed=True)
        if frame is None:
            raise ValueError(f"HPO group context requires a populated {role!r} role")
        groups = dataset.role_groups.get(role)
        if cfg.evaluation.group_mode == "patient_group" and groups is None:
            raise ValueError(
                f"Patient-group HPO requires Dataset.role_groups[{role!r}] for grouped evaluation"
            )
        if groups is None:
            context["roles"][role] = {
                "rows": int(len(frame)),
                "groups": None,
                "fingerprint": None,
                "source": "unavailable",
            }
            continue
        values = pd.Series(groups).reset_index(drop=True)
        if len(values) != len(frame):
            raise ValueError(
                f"HPO group context has {len(values)} values for {len(frame)} rows in {role!r}"
            )
        try:
            group_count = int(values.nunique(dropna=False))
            fingerprint = dataframe_fingerprint(pd.DataFrame({"group_id": values}))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"HPO group identifiers in {role!r} are not hashable") from exc
        context["roles"][role] = {
            "rows": int(len(frame)),
            "groups": group_count,
            "fingerprint": fingerprint,
            "source": "dataset_role_groups",
        }
    return context


def _build_hpo_context(
    cfg: Config,
    dataset: Dataset,
    *,
    task_type: str,
    role_context: dict,
    role_context_digest: str,
    stage_a_contract: hpo_mod.StageAScreenContract | None,
    device: str,
    semantic_context: Mapping[str, object] | None = None,
) -> dict | None:
    """Build the cache/study identity for the configured generation objective."""
    gen_cfg = cfg.generation
    if not gen_cfg.hpo.enabled:
        return None
    resolved_semantic_context = (
        dict(semantic_context)
        if semantic_context is not None
        else semantic_context_payload(
            dataset,
            classification_score=cfg.evaluation.synthcity.classification_score,
        )
    )
    release_transform_digest = _release_transform_digest(
        PROTOCOL_VERSION, dataset.release_generalization
    )
    role_hashes = {
        role: dataframe_fingerprint(_required_role_frame(dataset, role, imputed=False))
        for role in ("train", "tuning")
    }
    contracts = {
        "fit_roles": ["train"],
        "comparison_role": "tuning",
        "excluded_roles": ["final_holdout"],
        "privacy": False,
        "fairness": False,
    }
    support_provenance = {
        "fit_roles": ["train"],
        "support_contract": "train_frozen_v1",
    }
    bandwidth_provenance = {
        "fit_roles": ["train"],
        "comparison_role": "tuning",
        "contract": "train_frozen_v1",
    }
    objective_version = hpo_mod.TUNING_OBJECTIVE_VERSION
    context = hpo_mod.build_hpo_context(
        task_type=task_type,
        metric_config=gen_cfg.hpo.metric_config,
        registry_digest=DEFAULT_METRIC_CONTRACT_REGISTRY.digest(),
        stage_a_contract_digest=stage_a_contract.digest if stage_a_contract else None,
        group_context=_build_hpo_group_context(cfg, dataset, ("train", "tuning")),
        role_context_fingerprint=role_context_digest,
        role_context=role_context,
        variable_columns=list(dataset.full_df.columns),
        attack_target_types=cast(
            Mapping[str, str], resolved_semantic_context.get("sensitive_target_types", {})
        ),
        utility_policy=gen_cfg.hpo.utility_policy,
        release_transform_digest=release_transform_digest,
        role_hashes=role_hashes,
        contracts=contracts,
        support_provenance=support_provenance,
        bandwidth_provenance=bandwidth_provenance,
        objective_version=objective_version,
    )
    resolved_policy = hpo_mod._resolve_utility_policy(gen_cfg.hpo.utility_policy)
    context["objective_context"] = {
        "n_samples": gen_cfg.n_samples,
        "n_iter_cap": gen_cfg.hpo.n_iter_cap,
        "model_iter_caps": dict(gen_cfg.hpo.model_iter_caps),
        "sgld_step_cap": gen_cfg.hpo.sgld_step_cap,
        "seed": cfg.seed,
        "device": device,
        "utility_policy": resolved_policy,
        "objective_version": objective_version,
        "release_transform_digest": release_transform_digest,
        "role_hashes": role_hashes,
        "contracts": contracts,
        "support_provenance": support_provenance,
        "bandwidth_provenance": bandwidth_provenance,
        "metric_contract_manifest": DEFAULT_METRIC_CONTRACT_REGISTRY.manifest(),
    }
    context["metric_contract_manifest"] = context["objective_context"]["metric_contract_manifest"]
    context["semantic_context"] = resolved_semantic_context
    context["semantic_context_digest"] = semantic_context_digest(resolved_semantic_context)
    return context


def _refit_cache_stem(model_name: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_name).strip("-") or "model"
    return f"{readable}-{hashlib.sha256(model_name.encode()).hexdigest()[:12]}"


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    os.replace(temporary, path)


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _final_imputation_lineage(cfg: Config, dataset: Dataset, fit_frame: pd.DataFrame) -> dict:
    """Validate and describe final-phase imputation inputs for refit provenance."""
    if not dataset.has_canonical_roles:
        return {"phase": "legacy_two_role", "cache_key": None}

    paths = dataset.imputation_paths("final")
    cache_path = paths["cache_key"]
    artifact_path = paths["train_tuning_imputed"]
    try:
        cache_record = load_json(cache_path)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Final refit requires readable final-imputation lineage at {cache_path}"
        ) from exc
    expected_cache_key = _cache_key_record(cfg, dataset, phase="final")["cache_key"]
    validated_lineage = validate_final_imputation_lineage(dataset, expected_cache_key)
    if not isinstance(cache_record, dict) or cache_record.get("cache_key") != expected_cache_key:
        raise RuntimeError(
            "Final refit imputation cache key does not match current dataset/config lineage"
        )
    if (
        cache_record.get("phase") != "final"
        or cache_record.get("fit_roles")
        != [
            "train",
            "tuning",
        ]
        or cache_record.get("transform_roles") != ["train", "tuning", "final_holdout"]
    ):
        raise RuntimeError(
            "Final refit imputation cache does not describe a final train+tuning fit"
        )
    if cache_record.get("fit_frame_fingerprint") != dataframe_fingerprint(
        pd.concat([dataset.roles["train"], dataset.roles["tuning"]], axis=0)
    ):
        raise RuntimeError("Final refit imputation cache fit-frame fingerprint is stale")
    try:
        persisted_fit = pd.read_csv(artifact_path, low_memory=False)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise RuntimeError(
            f"Final refit train+tuning imputation artifact is unreadable at {artifact_path}"
        ) from exc
    persisted_fingerprint = dataframe_fingerprint(persisted_fit)
    recorded_fingerprints = cache_record.get("imputed_frame_fingerprints")
    if not isinstance(recorded_fingerprints, dict):
        raise RuntimeError("Final refit imputation cache lacks artifact fingerprints")
    if recorded_fingerprints.get("train_tuning") != persisted_fingerprint:
        raise RuntimeError("Final refit train+tuning imputation artifact fingerprint is stale")
    if validated_lineage["output_fingerprints"].get("train_tuning") != persisted_fingerprint:
        raise RuntimeError("Final refit fit artifact differs from validated cache lineage")
    try:
        pd.testing.assert_frame_equal(
            persisted_fit.reset_index(drop=True),
            fit_frame.reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError as exc:
        raise RuntimeError(
            "Final refit frame differs from the persisted final train+tuning imputation artifact"
        ) from exc
    return {
        "phase": "final",
        "cache_key": expected_cache_key,
        "cache_path": str(cache_path),
        "train_tuning_artifact": str(artifact_path),
        "train_tuning_fingerprint": persisted_fingerprint,
        "final_holdout_fingerprint": validated_lineage["output_fingerprints"]["final_holdout"],
    }


def _validate_candidate_final_refit_pair(
    cfg: Config, candidate_dataset: Dataset, final_dataset: Dataset
) -> tuple[dict, dict]:
    """Validate candidate HPO context and final refit Dataset share source lineage."""
    if candidate_dataset is final_dataset:
        raise ValueError("Candidate and final imputation phases require separate Dataset objects")
    candidate_dataset.require_canonical_roles("final model refit candidate context")
    final_dataset.require_canonical_roles("final model refit")
    if candidate_dataset.final_imputation_lineage is not None:
        raise ValueError("Candidate Dataset cannot carry final-phase imputation outputs")
    if final_dataset.candidate_imputation_lineage is not None:
        raise ValueError("Final Dataset cannot carry candidate-phase imputation outputs")
    identity_fields = (
        "name",
        "version",
        "source_fingerprint",
        "full_fingerprint",
        "assignment_fingerprint",
        "assignment_policy_fingerprint",
        "identity_fingerprint",
        "variable_schema_fingerprint",
        "semantic_fingerprint",
        "role_fingerprints",
    )
    mismatches = {
        field: (getattr(candidate_dataset, field), getattr(final_dataset, field))
        for field in identity_fields
        if getattr(candidate_dataset, field) != getattr(final_dataset, field)
    }
    if mismatches:
        raise ValueError(f"Candidate and final Dataset lineage differs: {mismatches}")

    candidate_record = _cache_key_record(cfg, candidate_dataset, phase="candidate")
    validate_imputation_cache_lineage(
        candidate_dataset,
        candidate_record,
        required=True,
        require_attached_proof=True,
    )
    final_record = _cache_key_record(cfg, final_dataset, phase="final")
    final_lineage = validate_final_imputation_lineage(final_dataset, final_record["cache_key"])
    candidate_lineage = candidate_dataset.candidate_imputation_lineage
    if not isinstance(candidate_lineage, dict):
        raise RuntimeError("Selected-model refit requires validated candidate imputation proof")
    return candidate_lineage, final_lineage


def refit_selected_model(
    cfg: Config,
    dataset: Dataset,
    model_name: str,
    output_dir: str | Path | None = None,
    *,
    imputation_phase: str = "final",
    candidate_dataset: Dataset | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Refit the selected model using validated final-imputed train+tuning roles."""
    if imputation_phase != "final":
        raise ValueError(
            "refit_selected_model requires imputation_phase='final'; "
            "use refit_candidate_model for explicit candidate-phase refits"
        )
    candidate = candidate_dataset
    if candidate is None:
        raise ValueError("Selected-model refit requires its candidate_dataset HPO context")
    candidate_lineage, _ = _validate_candidate_final_refit_pair(cfg, candidate, dataset)
    return _refit_model(
        cfg,
        dataset,
        model_name,
        output_dir,
        imputation_phase="final",
        candidate_dataset=candidate,
        candidate_lineage=candidate_lineage,
    )


def refit_candidate_model(
    cfg: Config,
    dataset: Dataset,
    model_name: str,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Explicitly refit a candidate-phase model for non-release workflows."""
    dataset.require_canonical_roles("candidate model refit")
    validate_imputation_cache_lineage(
        dataset,
        _cache_key_record(cfg, dataset, phase="candidate"),
        required=True,
        require_attached_proof=True,
    )
    candidate_lineage = dataset.candidate_imputation_lineage
    if not isinstance(candidate_lineage, dict):
        raise RuntimeError("Candidate model refit requires validated candidate imputation proof")
    return _refit_model(
        cfg,
        dataset,
        model_name,
        output_dir,
        imputation_phase="candidate",
        candidate_dataset=dataset,
        candidate_lineage=candidate_lineage,
    )


def _refit_model(
    cfg: Config,
    dataset: Dataset,
    model_name: str,
    output_dir: str | Path | None = None,
    *,
    imputation_phase: str,
    candidate_dataset: Dataset,
    candidate_lineage: dict,
) -> tuple[pd.DataFrame, dict]:
    """Refit one selected candidate using only canonical train and tuning roles.

    The returned metadata is persisted beside the synthetic CSV and is suitable
    for inclusion in post-selection evidence. Final-holdout lineage is validated,
    but the holdout frame is never included in the generator fit.
    """
    dataset.require_canonical_roles("final model refit")
    if imputation_phase not in {"candidate", "final"}:
        raise ValueError(f"Unsupported refit imputation phase: {imputation_phase!r}")
    candidate_dataset.require_canonical_roles("final model refit candidate context")
    gen_cfg = cfg.generation
    output_root = ensure_dir(output_dir or Path(gen_cfg.output_dir) / "final_refit")
    candidate_context = role_context_payload(candidate_dataset, ("train", "tuning"))
    candidate_context_digest = role_context_fingerprint(candidate_dataset, ("train", "tuning"))
    candidate_semantic_context = semantic_context_payload(
        candidate_dataset,
        classification_score=cfg.evaluation.synthcity.classification_score,
    )
    refit_semantic_context = semantic_context_payload(
        dataset,
        classification_score=cfg.evaluation.synthcity.classification_score,
    )
    refit_semantic_context_digest = semantic_context_digest(refit_semantic_context)
    task_type = "classification" if dataset.target_is_categorical else "regression"
    device = resolve_device(cfg.device)
    expected_columns = list(dataset.full_df.columns)

    raw_fit_frame = _combine_canonical_roles(dataset, imputed=False)
    imputed_fit_frame = _combine_canonical_roles(dataset, imputed=True)
    stage_a_contract = _build_stage_a_contract(
        cfg,
        candidate_dataset,
        _role_frame(candidate_dataset, "train", imputed=True),
    )
    hpo_context = _build_hpo_context(
        cfg,
        candidate_dataset,
        task_type=task_type,
        role_context=candidate_context,
        role_context_digest=candidate_context_digest,
        stage_a_contract=stage_a_contract,
        device=device,
        semantic_context=candidate_semantic_context,
    )
    raw_role_hashes = {
        role: dataframe_fingerprint(_required_role_frame(dataset, role, imputed=False))
        for role in ("train", "tuning")
    }
    imputed_role_hashes = {
        role: dataframe_fingerprint(_required_role_frame(dataset, role, imputed=True))
        for role in ("train", "tuning")
    }
    final_imputation_lineage = (
        _final_imputation_lineage(cfg, dataset, imputed_fit_frame)
        if imputation_phase == "final"
        else {"phase": "candidate", "cache_key": None}
    )

    backend = None
    fit_frame = imputed_fit_frame
    params: dict = {}
    build_fn = None

    tabpfn_specs = {
        "tabpfn_standard": ("standard", False),
        "tabpfn_standard_imputed": ("standard", True),
        "tabpfn_custom": ("custom", False),
        "tabpfn_custom_imputed": ("custom", True),
    }
    tabpfgen_specs = {
        "tabpfgen_standard": ("standard", False),
        "tabpfgen_standard_hpo": ("standard", True),
        "tabpfgen_custom": ("custom", False),
        "tabpfgen_custom_hpo": ("custom", True),
    }

    def load_hpo_params(family: str, cache_name: str) -> dict:
        best_params_path = gen_cfg.hpo.best_params_path or hpo_mod.default_best_params_path(
            gen_cfg.output_dir
        )
        if hpo_context is None:
            raise RuntimeError(
                "Cannot load selected HPO parameters without a complete canonical hpo_context"
            )
        cache = hpo_mod.BestParamsCache(best_params_path, hpo_context=hpo_context)
        if not cache.has(family, cache_name):
            raise RuntimeError(
                f"Cannot refit selected HPO candidate {model_name!r}: best parameters for "
                f"{family}/{cache_name} are missing or stale at {best_params_path}"
            )
        return dict(cache.get(family, cache_name))

    if model_name in tabpfn_specs:
        variant, use_imputed = tabpfn_specs[model_name]
        if not gen_cfg.tabpfn.enabled:
            raise RuntimeError(f"Selected model {model_name!r} is disabled in generation config")
        if variant not in gen_cfg.tabpfn.variants:
            raise RuntimeError(
                f"Selected model {model_name!r} is not enabled in generation.tabpfn.variants"
            )
        expected_variant = "imputed" if use_imputed else "raw"
        if expected_variant not in gen_cfg.tabpfn.data_variants:
            raise RuntimeError(
                f"Selected model {model_name!r} requires TabPFN data variant "
                f"{expected_variant!r}, which is not configured"
            )
        tpfn.validate_tabpfn_target(dataset.target_column, dataset.target_is_categorical)
        backend = "tabpfn"
        # Final refit always uses the post-selection imputed train+tuning frame,
        # including candidates whose search variant used raw training values.
        fit_frame = imputed_fit_frame
        if variant == "standard":

            def build_final_model():
                return tpfn.generate_tabpfn_standard(
                    fit_frame,
                    dataset.feature_columns,
                    dataset.categorical_columns,
                    dataset.target_column,
                    gen_cfg.n_samples,
                    target_is_categorical=dataset.target_is_categorical,
                    variable_schema_fingerprint=dataset.variable_schema_fingerprint,
                    semantic_context=refit_semantic_context,
                )
        else:

            def build_final_model():
                return tpfn.generate_tabpfn_custom(
                    fit_frame,
                    dataset.categorical_columns,
                    dataset.target_column,
                    gen_cfg.n_samples,
                    target_is_categorical=dataset.target_is_categorical,
                    variable_schema_fingerprint=dataset.variable_schema_fingerprint,
                    semantic_context=refit_semantic_context,
                )

        build_fn = build_final_model
    elif model_name in tabpfgen_specs:
        variant, is_hpo = tabpfgen_specs[model_name]
        if not gen_cfg.tabpfgen.enabled:
            raise RuntimeError(f"Selected model {model_name!r} is disabled in generation config")
        if variant not in gen_cfg.tabpfgen.variants:
            raise RuntimeError(
                f"Selected model {model_name!r} is not enabled in generation.tabpfgen.variants"
            )
        if is_hpo and not gen_cfg.hpo.enabled:
            raise RuntimeError(
                f"Selected HPO model {model_name!r} but generation.hpo.enabled is false"
            )
        from synthdata.generation import tabpfgen_backend as tpfgen

        tpfgen.validate_tabpfgen_target(dataset.target_column, dataset.target_is_categorical)
        backend = "tabpfgen"
        if is_hpo:
            implementation_fingerprint = sc.generator_implementation_fingerprint(
                f"tabpfgen_{variant}"
            )
            cache_name = f"tabpfgen_{variant}-{implementation_fingerprint}"
            params = load_hpo_params("tabpfgen", cache_name)
        elif variant == "standard":
            params = dict(gen_cfg.tabpfgen.standard_params)
        else:
            params = dict(gen_cfg.tabpfgen.custom_params)
        if variant == "standard":

            def build_final_model():
                return tpfgen.generate_tabpfgen_standard(
                    imputed_fit_frame,
                    dataset.feature_columns,
                    dataset.categorical_columns,
                    dataset.target_column,
                    gen_cfg.n_samples,
                    tabpfgen_params=params,
                    relabel_with_classifier=is_hpo,
                    target_is_categorical=dataset.target_is_categorical,
                    variable_schema_fingerprint=dataset.variable_schema_fingerprint,
                    semantic_context=refit_semantic_context,
                )
        else:

            def build_final_model():
                return tpfgen.generate_tabpfgen_custom(
                    imputed_fit_frame,
                    dataset.feature_columns,
                    dataset.categorical_columns,
                    dataset.target_column,
                    gen_cfg.n_samples,
                    seed=cfg.seed,
                    sgld_params=params,
                    target_is_categorical=dataset.target_is_categorical,
                    variable_schema_fingerprint=dataset.variable_schema_fingerprint,
                    semantic_context=refit_semantic_context,
                )

        build_fn = build_final_model
    else:
        is_hpo = model_name.endswith("_hpo")
        plugin_name = model_name[:-4] if is_hpo else model_name
        if not gen_cfg.synthcity.enabled or plugin_name not in gen_cfg.synthcity.names:
            raise RuntimeError(
                f"Selected model {model_name!r} has no configured final-refit backend"
            )
        backend = "synthcity"
        if is_hpo:
            if not gen_cfg.hpo.enabled:
                raise RuntimeError(
                    f"Selected HPO model {model_name!r} but generation.hpo.enabled is false"
                )
            implementation_fingerprint = sc.generator_implementation_fingerprint(plugin_name)
            cache_name = f"{plugin_name}-{implementation_fingerprint}"
            params = load_hpo_params("synthcity", cache_name)
            if gen_cfg.hpo.final_n_iter_override and sc.plugin_accepts(plugin_name, "n_iter"):
                params["n_iter"] = gen_cfg.hpo.final_n_iter_override
        else:
            params = dict(gen_cfg.synthcity.params.get(plugin_name, {}))
        feature_types = {column: entry["kind"] for column, entry in dataset.variable_schema.items()}
        source_table = {
            column: entry["source_table"]
            for column, entry in dataset.variable_schema.items()
            if entry.get("source_table") is not None
        }
        fairness_column = dataset.protected_columns[0] if dataset.protected_columns else None
        fit_loader = sc.make_loader(
            imputed_fit_frame,
            dataset.target_column,
            dataset.sensitive_columns,
            random_state=cfg.seed,
            fairness_column=fairness_column,
            important_features=dataset.quasi_identifier_columns,
            group_ids=(
                pd.concat(
                    cast(
                        list[pd.Series],
                        [dataset.role_groups["train"], dataset.role_groups["tuning"]],
                    ),
                    ignore_index=True,
                )
                if dataset.role_groups.get("train") is not None
                and dataset.role_groups.get("tuning") is not None
                else None
            ),
            feature_types=feature_types,
            source_table=source_table,
        )

        def build_final_model():
            return sc.fit_generate(
                plugin_name,
                params,
                fit_loader,
                gen_cfg.n_samples,
                cfg.seed,
                workspace=str(output_root / "synthcity_workspace"),
                device=device,
            )

        build_fn = build_final_model

    if build_fn is None or backend is None:
        raise RuntimeError(f"Unable to resolve final refit backend for {model_name!r}")

    generator_context = sc.generator_metadata_context(model_name, params)
    implementation_fingerprint = sc.generator_implementation_fingerprint(model_name)
    fit_frame_fingerprint = dataframe_fingerprint(fit_frame)
    cache_metadata = {
        "schema_version": FINAL_REFIT_CACHE_SCHEMA_VERSION,
        "model_name": model_name,
        "backend": backend,
        "columns": expected_columns,
        "fit_roles": ["train", "tuning"],
        "imputation_phase": imputation_phase,
        "imputation_lineage": final_imputation_lineage,
        "candidate_imputation_lineage": candidate_lineage,
        "fit_frame_fingerprint": fit_frame_fingerprint,
        "fit_frame_fingerprints": {
            "raw": dataframe_fingerprint(raw_fit_frame),
            "imputed": dataframe_fingerprint(imputed_fit_frame),
        },
        "input_role_hashes": {
            "raw": raw_role_hashes,
            "imputed": imputed_role_hashes,
        },
        "role_context_fingerprint": candidate_context_digest,
        "role_context": candidate_context,
        "variable_schema_fingerprint": dataset.variable_schema_fingerprint,
        "semantic_context": refit_semantic_context,
        "semantic_context_digest": refit_semantic_context_digest,
        "task_type": task_type,
        "target_view": "native",
        "n_samples": gen_cfg.n_samples,
        "seed": cfg.seed,
        "device": device,
        "registry_digest": DEFAULT_METRIC_CONTRACT_REGISTRY.digest(),
        "parameters": params,
        "generator_metadata_schema_version": sc.GENERATOR_METADATA_SCHEMA_VERSION,
        "generator_context": generator_context,
        "implementation_fingerprint": implementation_fingerprint,
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_metadata, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()
    cache_stem = _refit_cache_stem(model_name)
    synthetic_path = output_root / f"{cache_stem}.csv"
    metadata_path = output_root / f"{cache_stem}.cache.json"
    cached_metadata = None
    if synthetic_path.exists() and metadata_path.exists() and not gen_cfg.force_retrain:
        try:
            cached_metadata = load_json(metadata_path)
        except ValueError as exc:
            logger.warning(
                "[final refit] cache metadata at %s is invalid: %s; regenerating %s",
                metadata_path,
                exc,
                model_name,
            )
        if cached_metadata is not None and not isinstance(cached_metadata, dict):
            logger.warning(
                "[final refit] cache metadata at %s is not a JSON object; regenerating %s",
                metadata_path,
                model_name,
            )
            cached_metadata = None
        cache_metadata_valid = (
            cached_metadata is not None
            and cached_metadata.get("schema_version") == FINAL_REFIT_CACHE_SCHEMA_VERSION
            and cached_metadata.get("cache_key") == cache_key
            and all(cached_metadata.get(key) == value for key, value in cache_metadata.items())
            and sc.generator_metadata_matches_request(
                cached_metadata.get("generator_metadata"),
                model_name,
                params,
                gen_cfg.n_samples,
                cfg.seed,
            )
        )
        if cache_metadata_valid:
            try:
                cached_frame = pd.read_csv(synthetic_path)
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                logger.warning(
                    "[final refit] cached data at %s is unreadable: %s; regenerating %s",
                    synthetic_path,
                    exc,
                    model_name,
                )
            else:
                if (
                    list(cached_frame.columns) == expected_columns
                    and len(cached_frame) == gen_cfg.n_samples
                ):
                    try:
                        cached_data_digest = _file_digest(synthetic_path)
                    except OSError as exc:
                        logger.warning(
                            "[final refit] cached data at %s could not be digested: %s; "
                            "regenerating %s",
                            synthetic_path,
                            exc,
                            model_name,
                        )
                    else:
                        if cached_metadata.get("synthetic_data_sha256") == cached_data_digest:
                            logger.info(
                                "[final refit] cache hit for %s at %s (role_context=%s)",
                                model_name,
                                synthetic_path,
                                candidate_context_digest[:16],
                            )
                            return cached_frame, {
                                **cache_metadata,
                                "cache_key": cache_key,
                                "synthetic_data_sha256": cached_metadata.get(
                                    "synthetic_data_sha256"
                                ),
                                "generator_metadata": cached_metadata.get("generator_metadata"),
                                "path": str(synthetic_path),
                                "metadata_path": str(metadata_path),
                                "cache_state": "hit",
                            }
                        logger.warning(
                            "[final refit] cached data at %s failed its digest check; "
                            "regenerating %s",
                            synthetic_path,
                            model_name,
                        )
                else:
                    logger.warning(
                        "[final refit] cached data at %s has columns %s and %d rows; "
                        "expected columns %s and %d rows; regenerating %s",
                        synthetic_path,
                        list(cached_frame.columns),
                        len(cached_frame),
                        expected_columns,
                        gen_cfg.n_samples,
                        model_name,
                    )

    logger.info(
        "[final refit] generating selected model=%s backend=%s from fit_roles=%s fit_shape=%s",
        model_name,
        backend,
        ["train", "tuning"],
        fit_frame.shape,
    )
    set_global_seed(cfg.seed)
    result = build_fn()
    synthetic_frame = result[0] if isinstance(result, tuple) else result
    generator_metadata = _resolve_generator_metadata(
        result[1] if isinstance(result, tuple) and len(result) > 1 else None,
        name=model_name,
        params=params,
        n_samples=gen_cfg.n_samples,
        random_state=cfg.seed,
        generator_identity=f"synthdata.{backend}.{model_name}",
    )
    if list(synthetic_frame.columns) != expected_columns:
        raise RuntimeError(
            f"Final refit for {model_name!r} returned columns {list(synthetic_frame.columns)!r}; "
            f"expected {expected_columns!r}"
        )
    if len(synthetic_frame) != gen_cfg.n_samples:
        raise RuntimeError(
            f"Final refit for {model_name!r} returned {len(synthetic_frame)} rows; "
            f"expected {gen_cfg.n_samples}"
        )
    _atomic_csv(synthetic_path, synthetic_frame)
    cache_metadata["synthetic_data_sha256"] = _file_digest(synthetic_path)
    save_json(
        metadata_path,
        {
            **cache_metadata,
            "cache_key": cache_key,
            "generator_metadata": generator_metadata,
        },
    )
    return synthetic_frame, {
        **cache_metadata,
        "cache_key": cache_key,
        "generator_metadata": generator_metadata,
        "path": str(synthetic_path),
        "metadata_path": str(metadata_path),
        "cache_state": "generated",
    }


def _expected_generation_outputs(gen_cfg) -> list[str]:
    """List configured output names before attempting any model generation."""
    expected = []
    if gen_cfg.synthcity.enabled:
        for name in gen_cfg.synthcity.names:
            expected.append(name)
            if gen_cfg.hpo.enabled:
                expected.append(f"{name}_hpo")
    if gen_cfg.tabpfn.enabled:
        for variant in gen_cfg.tabpfn.data_variants:
            suffix = "_imputed" if variant == "imputed" else ""
            for model in gen_cfg.tabpfn.variants:
                expected.append(f"tabpfn_{model}{suffix}")
    if gen_cfg.tabpfgen.enabled:
        for variant in gen_cfg.tabpfgen.variants:
            expected.append(f"tabpfgen_{variant}")
            if gen_cfg.hpo.enabled:
                expected.append(f"tabpfgen_{variant}_hpo")
    return list(dict.fromkeys(expected))


def run_generation(
    cfg: Config,
    dataset: Dataset,
    plot_callback: Callable | None = None,
    experiment=None,
) -> dict[str, pd.DataFrame]:
    """Generate (or load cached) synthetic datasets for every configured model.

    ``plot_callback(name, synthetic_df, extra)`` is invoked right after each
    dataset is (re)generated (not when loaded from cache), so callers can save
    real-vs-synthetic figures inline; see :mod:`synthdata.plotting.generation_plots`.
    If ``experiment`` (a :class:`synthdata.experiment.Experiment`) is given, a
    failed plot callback is recorded to its manifest (``stage="generation_plot_failed"``)
    in addition to the console warning, so the skip is visible from the
    output directory alone.
    """
    gen_cfg = cfg.generation
    if gen_cfg.hpo.enabled:
        dataset.require_canonical_roles("generation HPO")
        hpo_mod.validate_hpo_metric_config(
            gen_cfg.hpo.metric_config,
            group_context={"group_mode": cfg.evaluation.group_mode},
        )

    fit_imputed_df = _role_frame(dataset, "train", imputed=True)
    if fit_imputed_df is None and needs_imputed_data(gen_cfg):
        raise RuntimeError(
            "Dataset must be imputed before generation (run synthdata.imputation.run_imputation first)"
        )
    if needs_imputed_data(gen_cfg):
        validate_imputation_cache_lineage(
            dataset,
            _cache_key_record(cfg, dataset),
            required=dataset.has_canonical_roles,
        )

    output_dir = ensure_dir(gen_cfg.output_dir)
    n_samples = gen_cfg.n_samples
    seed = cfg.seed
    device = resolve_device(cfg.device)
    task_type = "classification" if dataset.target_is_categorical else "regression"
    # Candidate artifacts are evaluated against train+tuning regardless of
    # whether HPO selected parameters.  This is provenance only: every
    # generator below still receives train_loader exclusively when HPO is off.
    context_roles = ("train", "tuning") if dataset.has_canonical_roles else ("train",)
    fit_context_roles = ("train",)
    generation_role_context = role_context_payload(dataset, context_roles)
    generation_role_context_fingerprint = role_context_fingerprint(dataset, context_roles)
    fit_context = role_context_payload(dataset, fit_context_roles)
    fit_context_fingerprint = role_context_fingerprint(dataset, fit_context_roles)
    generation_semantic_context = semantic_context_payload(
        dataset,
        classification_score=cfg.evaluation.synthcity.classification_score,
        roles=context_roles,
    )
    generation_semantic_context_digest = semantic_context_digest(generation_semantic_context)
    if experiment is not None and hasattr(experiment, "validate_generation_context"):
        full_roles = (
            ("train", "tuning", "final_holdout")
            if dataset.has_canonical_roles
            else (
                "train",
                "final_holdout",
            )
        )
        experiment.validate_generation_context(
            generation_role_context,
            generation_role_context_fingerprint,
            full_context=role_context_payload(dataset, full_roles, candidate_phase=True),
            full_fingerprint=role_context_fingerprint(dataset, full_roles, candidate_phase=True),
        )

    stage_a_contract = None
    stage_a_root = None
    if gen_cfg.hpo.enabled and (
        (gen_cfg.synthcity.enabled and bool(gen_cfg.synthcity.names)) or gen_cfg.tabpfgen.enabled
    ):
        if fit_imputed_df is None:
            raise RuntimeError("Stage A HPO screening requires an imputed train role")
        stage_a_contract = _build_stage_a_contract(cfg, dataset, fit_imputed_df)
        if stage_a_contract is None:
            raise RuntimeError("Stage A HPO screening did not produce a contract")
        stage_a_root = output_dir / "hpo_stage_a"
        logger.info(
            "[stage_a] resolved HPO screen contract digest=%s source_role=train shape=%s",
            stage_a_contract.digest[:16],
            fit_imputed_df.shape,
        )

    hpo_context = _build_hpo_context(
        cfg,
        dataset,
        task_type=task_type,
        role_context=generation_role_context,
        role_context_digest=generation_role_context_fingerprint,
        stage_a_contract=stage_a_contract,
        device=device,
        semantic_context=generation_semantic_context,
    )
    hpo_context_artifact_path = None
    hpo_context_digest = None
    if hpo_context is not None:
        hpo_context_path = output_dir / "hpo_context.json"
        hpo_context_digest = hpo_mod.hpo_context_digest(hpo_context)
        versioned_hpo_context_path = output_dir / f"hpo_context-{hpo_context_digest}.json"
        hpo_context_payload = {
            "schema_version": hpo_mod.HPO_CONTEXT_SCHEMA_VERSION,
            "context_digest": hpo_context_digest,
            "context": hpo_context,
            "context_file": versioned_hpo_context_path.name,
        }
        if versioned_hpo_context_path.exists():
            try:
                cached_hpo_context = load_json(versioned_hpo_context_path)
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    f"Persisted HPO context at {versioned_hpo_context_path} is unreadable"
                ) from exc
            if cached_hpo_context != hpo_context_payload:
                raise RuntimeError(
                    f"Persisted HPO context at {versioned_hpo_context_path} does not match the current context"
                )
        else:
            _atomic_json(versioned_hpo_context_path, hpo_context_payload)
        try:
            persisted_hpo_context = load_json(versioned_hpo_context_path)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Persisted HPO context at {versioned_hpo_context_path} is unreadable"
            ) from exc
        if (
            persisted_hpo_context != hpo_context_payload
            or hpo_mod.hpo_context_digest(persisted_hpo_context.get("context", {}))
            != hpo_context_digest
        ):
            raise RuntimeError(
                f"Persisted HPO context at {versioned_hpo_context_path} failed digest validation"
            )
        _atomic_json(hpo_context_path, hpo_context_payload)
        hpo_context_artifact_path = versioned_hpo_context_path.name
    hpo_group_context = hpo_context.get("group_context") if hpo_context else None
    best_params_path = gen_cfg.hpo.best_params_path or hpo_mod.default_best_params_path(output_dir)
    best_params = (
        hpo_mod.BestParamsCache(best_params_path, hpo_context=hpo_context)
        if hpo_context is not None
        else None
    )

    def _require_best_params_cache() -> hpo_mod.BestParamsCache:
        """Return validated HPO cache, failing closed if HPO context is absent."""
        if best_params is None:
            raise RuntimeError("HPO generation requires a complete canonical hpo_context")
        return best_params

    synthetic_datasets: dict[str, pd.DataFrame] = {}
    expected_outputs = _expected_generation_outputs(gen_cfg)
    failed_model_outcomes: list[dict] = []

    def _stage_a_failure_outcome(
        model_name: str,
        error: hpo_mod.StageAExhaustionError,
        *,
        study_name: str | None = None,
    ) -> dict:
        if hpo_context is None or hpo_context_digest is None or hpo_context_artifact_path is None:
            raise RuntimeError("Stage A exhaustion requires a persisted HPO context artifact")
        expected_study_name = hpo_mod.contextual_study_name(
            study_name or f"hpo_{model_name.removesuffix('_hpo')}", hpo_context
        )
        if error.study_name != expected_study_name:
            raise RuntimeError(
                f"Stage A exhaustion study {error.study_name!r} does not match current "
                f"HPO context study {expected_study_name!r}"
            )
        stage_a_contract_digest = hpo_context.get("stage_a_contract_digest")
        if not isinstance(stage_a_contract_digest, str) or not stage_a_contract_digest:
            raise RuntimeError("Stage A exhaustion HPO context has no contract digest")
        return {
            "model": model_name,
            "hpo_study": error.study_name,
            "hpo_context_path": hpo_context_artifact_path,
            "hpo_context_digest": hpo_context_digest,
            "stage_a_contract_digest": stage_a_contract_digest,
            "evidence_references": error.evidence_references,
        }

    def _cached_or_build(name, build_fn, *, hpo_context=None, resolved_parameters=None):
        path = output_dir / f"{name}.csv"
        metadata_path = output_dir / f"{name}.cache.json"
        expected_hpo_context_digest = (
            hpo_mod.hpo_context_digest(hpo_context) if hpo_context is not None else None
        )
        expected_parameters = dict(resolved_parameters or {})
        generator_context = sc.generator_metadata_context(name, expected_parameters)
        implementation_fingerprint = sc.generator_implementation_fingerprint(name)
        cache_metadata = {
            "schema_version": GENERATION_CACHE_SCHEMA_VERSION,
            "model_name": name,
            "role_context_fingerprint": generation_role_context_fingerprint,
            "role_context": generation_role_context,
            "fit_context_fingerprint": fit_context_fingerprint,
            "fit_context": fit_context,
            "variable_schema_fingerprint": dataset.variable_schema_fingerprint,
            "semantic_context": generation_semantic_context,
            "semantic_context_digest": generation_semantic_context_digest,
            "columns": list(dataset.full_df.columns),
            "task_type": task_type,
            "target_view": "native",
            "n_samples": n_samples,
            "seed": seed,
            "device": device,
            "registry_digest": DEFAULT_METRIC_CONTRACT_REGISTRY.digest(),
            "hpo_context_schema_version": (
                hpo_mod.HPO_CONTEXT_SCHEMA_VERSION if hpo_context is not None else None
            ),
            "hpo_context_digest": expected_hpo_context_digest,
            "hpo_context": hpo_context,
            "resolved_parameters": expected_parameters,
            "generator_metadata_schema_version": sc.GENERATOR_METADATA_SCHEMA_VERSION,
            "generator_context": generator_context,
            "implementation_fingerprint": implementation_fingerprint,
        }
        expected_cache_key = hashlib.sha256(
            json.dumps(cache_metadata, sort_keys=True, default=str, separators=(",", ":")).encode()
        ).hexdigest()
        if path.exists() and not gen_cfg.force_retrain:
            metadata = None
            if metadata_path.exists():
                try:
                    metadata = load_json(metadata_path)
                except ValueError as exc:
                    logger.warning(
                        "[%s] synthetic cache metadata at %s is invalid: %s",
                        name,
                        metadata_path,
                        exc,
                    )
            if metadata is not None and not isinstance(metadata, dict):
                logger.warning(
                    "[%s] synthetic cache metadata at %s is not a JSON object; regenerating",
                    name,
                    metadata_path,
                )
                metadata = None
            cache_matches = (
                metadata is not None
                and metadata.get("schema_version") == GENERATION_CACHE_SCHEMA_VERSION
                and metadata.get("cache_key") == expected_cache_key
                and all(metadata.get(key) == value for key, value in cache_metadata.items())
                and sc.generator_metadata_matches_request(
                    metadata.get("generator_metadata"),
                    name,
                    expected_parameters,
                    n_samples,
                    seed,
                )
            )
            if cache_matches:
                try:
                    df = pd.read_csv(path)
                    data_digest = _file_digest(path)
                except (OSError, ValueError, pd.errors.ParserError) as exc:
                    logger.warning(
                        "[%s] synthetic cache at %s is unreadable: %s; regenerating",
                        name,
                        path,
                        exc,
                    )
                else:
                    if (
                        list(df.columns) == cache_metadata["columns"]
                        and len(df) == n_samples
                        and metadata.get("row_count") == len(df)
                        and metadata.get("synthetic_data_sha256") == data_digest
                    ):
                        logger.info(
                            "[%s] using cached synthetic data at %s (cache=%s)",
                            name,
                            path,
                            expected_cache_key[:16],
                        )
                        synthetic_datasets[name] = df
                        return df
                    logger.warning(
                        "[%s] synthetic cache at %s failed schema/size/digest validation; regenerating",
                        name,
                        path,
                    )
            logger.info(
                "[%s] synthetic cache at %s is stale for role_context=%s; regenerating",
                name,
                path,
                generation_role_context_fingerprint[:16],
            )

        logger.info("[%s] generating synthetic data (n_samples=%d)", name, n_samples)
        result = build_fn()
        df, extra = result if isinstance(result, tuple) else (result, None)
        generator_metadata = _resolve_generator_metadata(
            extra,
            name=name,
            params=expected_parameters,
            n_samples=n_samples,
            random_state=seed,
            generator_identity=f"synthdata.generation.{name}",
        )
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{name} generator returned {type(df).__name__}, expected DataFrame")
        if list(df.columns) != cache_metadata["columns"]:
            raise RuntimeError(
                f"{name} generator returned columns {list(df.columns)!r}; "
                f"expected {cache_metadata['columns']!r}"
            )
        if len(df) != n_samples:
            raise RuntimeError(f"{name} generator returned {len(df)} rows; expected {n_samples}")
        _atomic_csv(path, df)
        data_digest = _file_digest(path)
        _atomic_json(
            metadata_path,
            {
                **cache_metadata,
                "cache_key": expected_cache_key,
                "row_count": len(df),
                "synthetic_data_sha256": data_digest,
                "generator_metadata": generator_metadata,
            },
        )
        persisted_metadata = load_json(metadata_path)
        expected_persisted_context = {
            key: cache_metadata[key]
            for key in (
                "role_context_fingerprint",
                "role_context",
                "fit_context_fingerprint",
                "fit_context",
                "semantic_context_digest",
            )
        }
        if any(
            persisted_metadata.get(key) != value
            for key, value in expected_persisted_context.items()
        ):
            raise RuntimeError(
                f"{name} generation cache persisted context differs from validated experiment context"
            )
        synthetic_datasets[name] = df
        if plot_callback is not None:
            try:
                plot_callback(name, df, extra)
            except Exception as exc:  # noqa: BLE001 - plotting must not fail generation
                # Plotting must never break generation: skip just this
                # figure, but persist the skip to the experiment manifest so
                # it's visible from the output directory, not just the console.
                logger.warning(
                    "[%s] plot callback failed; reason_code=plot_callback_failed exception_type=%s",
                    name,
                    type(exc).__name__,
                )
                if experiment is not None:
                    try:
                        experiment.record(
                            "generation_plot_failed",
                            model=name,
                            reason_code="plot_callback_failed",
                            message="Generation plot callback failed.",
                            error_type=type(exc).__name__,
                        )
                    except Exception:  # noqa: BLE001 - manifest persistence is best effort
                        logger.warning(
                            "[%s] could not persist generation plot failure; "
                            "reason_code=plot_failure_manifest_write_failed",
                            name,
                        )
        return df

    # ------------------------------------------------------------------
    # synthcity models
    # ------------------------------------------------------------------
    if gen_cfg.synthcity.enabled and gen_cfg.synthcity.names:
        if fit_imputed_df is None:
            raise RuntimeError("SynthCity generation requires an imputed train role")
        fairness_column = dataset.protected_columns[0] if dataset.protected_columns else None
        train_loader = sc.make_loader(
            fit_imputed_df,
            dataset.target_column,
            dataset.sensitive_columns,
            random_state=seed,
            fairness_column=fairness_column,
            important_features=dataset.quasi_identifier_columns,
            group_ids=dataset.role_groups.get("train"),
            feature_types={
                column: entry["kind"] for column, entry in dataset.variable_schema.items()
            },
            source_table={
                column: entry["source_table"]
                for column, entry in dataset.variable_schema.items()
                if entry.get("source_table") is not None
            },
        )
        tuning_loader = None
        if gen_cfg.hpo.enabled:
            tuning_df = _role_frame(dataset, "tuning", imputed=True)
            if tuning_df is None:
                raise RuntimeError("Canonical HPO generation requires an imputed tuning role")
            tuning_loader = sc.make_loader(
                tuning_df,
                dataset.target_column,
                dataset.sensitive_columns,
                random_state=seed,
                fairness_column=fairness_column,
                important_features=dataset.quasi_identifier_columns,
                group_ids=dataset.role_groups.get("tuning"),
                feature_types={
                    column: entry["kind"] for column, entry in dataset.variable_schema.items()
                },
                source_table={
                    column: entry["source_table"]
                    for column, entry in dataset.variable_schema.items()
                    if entry.get("source_table") is not None
                },
            )

        for name in gen_cfg.synthcity.names:
            model_params = dict(gen_cfg.synthcity.params.get(name, {}))
            _cached_or_build(
                name,
                lambda name=name, model_params=model_params: sc.fit_generate(
                    name,
                    model_params,
                    train_loader,
                    n_samples,
                    seed,
                    workspace=str(output_dir / "synthcity_workspace"),
                    device=device,
                ),
                resolved_parameters=model_params,
            )

            if gen_cfg.hpo.enabled:
                cache = _require_best_params_cache()
                implementation_fingerprint = sc.generator_implementation_fingerprint(name)
                cache_name = f"{name}-{implementation_fingerprint}"
                if not cache.has("synthcity", cache_name):
                    study_base_name = hpo_mod.resolve_study_name(
                        f"hpo_{name}",
                        implementation_fingerprint,
                        gen_cfg.hpo,
                        output_dir,
                        seed,
                        hpo_context=hpo_context,
                        stage_a_root=stage_a_root,
                    )
                    objective = sc.build_synthcity_objective(
                        name,
                        train_loader,
                        gen_cfg.hpo,
                        seed,
                        workspace=str(output_dir / "synthcity_workspace"),
                        device=device,
                        tuning_loader=tuning_loader,
                        task_type=task_type,
                        classification_score=cfg.evaluation.synthcity.classification_score,
                        semantic_context=generation_semantic_context,
                        synthetic_size=n_samples,
                        stage_a_contract=stage_a_contract,
                        stage_a_source_df=fit_imputed_df,
                        stage_a_root=str(stage_a_root) if stage_a_root is not None else None,
                        study_name=hpo_mod.contextual_study_name(study_base_name, hpo_context),
                        group_context=hpo_group_context,
                        expected_emitted_keys=(
                            hpo_context["expected_emitted_keys"]
                            if hpo_context is not None
                            else None
                        ),
                        train_df=fit_imputed_df,
                        tuning_df=tuning_df,
                        target_column=dataset.target_column,
                        feature_types={
                            column: entry["kind"]
                            for column, entry in dataset.variable_schema.items()
                        },
                        release_generalization=dataset.release_generalization,
                    )
                    try:
                        params = hpo_mod.run_study(
                            study_base_name,
                            objective,
                            gen_cfg.hpo,
                            output_dir,
                            seed,
                            hpo_context=hpo_context,
                            checkpoint_workspace=output_dir / "synthcity_workspace",
                            checkpoint_plugin=name,
                            checkpoint_implementation_fingerprint=implementation_fingerprint,
                            stage_a_root=stage_a_root,
                        )
                    except hpo_mod.StageAExhaustionError as exc:
                        failed_model_outcomes.append(
                            _stage_a_failure_outcome(f"{name}_hpo", exc, study_name=study_base_name)
                        )
                        logger.warning(
                            "[generation] skipping HPO output=%s after Stage A exhaustion; "
                            "study=%s evidence=%s",
                            f"{name}_hpo",
                            exc.study_name,
                            exc.evidence_references,
                        )
                        continue
                    cache.set("synthcity", cache_name, params)
                params = dict(cache.get("synthcity", cache_name))

                override = gen_cfg.hpo.final_n_iter_override
                if override and sc.plugin_accepts(name, "n_iter"):
                    params["n_iter"] = override

                _cached_or_build(
                    f"{name}_hpo",
                    lambda name=name, params=params: sc.fit_generate(
                        name,
                        params,
                        train_loader,
                        n_samples,
                        seed,
                        workspace=str(output_dir / "synthcity_workspace"),
                        device=device,
                    ),
                    hpo_context=hpo_context,
                    resolved_parameters=params,
                )

    # ------------------------------------------------------------------
    # TabPFN models (no HPO; can fit on the original pre-imputation train
    # split and/or the imputed one -- see gen_cfg.tabpfn.data_variants)
    # ------------------------------------------------------------------
    if gen_cfg.tabpfn.enabled:
        tpfn.validate_tabpfn_target(dataset.target_column, dataset.target_is_categorical)
        for data_variant in gen_cfg.tabpfn.data_variants:
            if data_variant == "imputed":
                train_df_variant = fit_imputed_df
                suffix = "_imputed"
            else:
                train_df_variant = _role_frame(dataset, "train", imputed=False)
                suffix = ""
            if train_df_variant is None:
                raise RuntimeError(f"TabPFN {data_variant} generation requires a train role")

            if "standard" in gen_cfg.tabpfn.variants:
                _cached_or_build(
                    f"tabpfn_standard{suffix}",
                    lambda train_df_variant=train_df_variant: tpfn.generate_tabpfn_standard(
                        train_df_variant,
                        dataset.feature_columns,
                        dataset.categorical_columns,
                        dataset.target_column,
                        n_samples,
                        target_is_categorical=dataset.target_is_categorical,
                        variable_schema_fingerprint=dataset.variable_schema_fingerprint,
                        semantic_context=generation_semantic_context,
                    ),
                )
            if "custom" in gen_cfg.tabpfn.variants:
                _cached_or_build(
                    f"tabpfn_custom{suffix}",
                    lambda train_df_variant=train_df_variant: tpfn.generate_tabpfn_custom(
                        train_df_variant,
                        dataset.categorical_columns,
                        dataset.target_column,
                        n_samples,
                        target_is_categorical=dataset.target_is_categorical,
                        variable_schema_fingerprint=dataset.variable_schema_fingerprint,
                        semantic_context=generation_semantic_context,
                    ),
                )

    # ------------------------------------------------------------------
    # TabPFGen models (use the imputed train split)
    # ------------------------------------------------------------------
    if gen_cfg.tabpfgen.enabled:
        from synthdata.generation import tabpfgen_backend as tpfgen

        tpfgen.validate_tabpfgen_target(dataset.target_column, dataset.target_is_categorical)
        fit_imputed_df = _role_frame(dataset, "train", imputed=True)
        if fit_imputed_df is None:
            raise RuntimeError("TabPFGen generation requires an imputed train role")

        eval_fn = None
        if gen_cfg.hpo.enabled:
            tuning_imputed_df = _role_frame(dataset, "tuning", imputed=True)
            if tuning_imputed_df is None:
                raise RuntimeError("Canonical HPO generation requires an imputed tuning role")
            eval_fn = hpo_mod.build_synthetic_eval_fn(
                fit_imputed_df,
                tuning_imputed_df,
                dataset.target_column,
                dataset.protected_columns,
                gen_cfg.hpo.metric_config,
                seed,
                workspace=output_dir / "synthcity_workspace",
                task_type=task_type,
                semantic_context=generation_semantic_context,
                feature_types={
                    column: entry["kind"] for column, entry in dataset.variable_schema.items()
                },
                source_table={
                    column: entry["source_table"]
                    for column, entry in dataset.variable_schema.items()
                    if entry.get("source_table") is not None
                },
                quasi_identifier_columns=dataset.quasi_identifier_columns,
                sensitive_target_types={
                    column: dataset.variable_schema[column]["kind"]
                    for column in dataset.sensitive_columns
                },
                classification_score=cfg.evaluation.synthcity.classification_score,
                group_context=hpo_group_context,
                train_group_ids=dataset.role_groups.get("train"),
                holdout_group_ids=dataset.role_groups.get("tuning"),
                expected_emitted_keys=(
                    hpo_context["expected_emitted_keys"] if hpo_context is not None else None
                ),
                release_generalization=dataset.release_generalization,
                utility_policy=gen_cfg.hpo.utility_policy,
            )
            if eval_fn is None:
                raise RuntimeError("Enabled HPO requires a synthetic-data evaluator")
        evaluator = cast(Callable[[pd.DataFrame], float], eval_fn)

        if "standard" in gen_cfg.tabpfgen.variants:
            standard_params = dict(gen_cfg.tabpfgen.standard_params)
            _cached_or_build(
                "tabpfgen_standard",
                lambda standard_params=standard_params: tpfgen.generate_tabpfgen_standard(
                    fit_imputed_df,
                    dataset.feature_columns,
                    dataset.categorical_columns,
                    dataset.target_column,
                    n_samples,
                    tabpfgen_params=standard_params,
                    target_is_categorical=dataset.target_is_categorical,
                    variable_schema_fingerprint=dataset.variable_schema_fingerprint,
                    semantic_context=generation_semantic_context,
                ),
                resolved_parameters=standard_params,
            )

            if gen_cfg.hpo.enabled:
                cache = _require_best_params_cache()
                implementation_fingerprint = sc.generator_implementation_fingerprint(
                    "tabpfgen_standard"
                )
                cache_name = f"tabpfgen_standard-{implementation_fingerprint}"
                if not cache.has("tabpfgen", cache_name):
                    study_name = hpo_mod.resolve_study_name(
                        "hpo_tabpfgen_standard",
                        implementation_fingerprint,
                        gen_cfg.hpo,
                        output_dir,
                        seed,
                        hpo_context=hpo_context,
                        stage_a_root=stage_a_root,
                    )
                    objective = tpfgen.build_tabpfgen_standard_objective(
                        fit_imputed_df,
                        dataset.feature_columns,
                        dataset.categorical_columns,
                        dataset.target_column,
                        n_samples,
                        gen_cfg.hpo.sgld_step_cap,
                        evaluator,
                        seed=seed,
                        stage_a_contract=stage_a_contract,
                        stage_a_source_df=fit_imputed_df,
                        stage_a_root=str(stage_a_root) if stage_a_root is not None else None,
                        study_name=hpo_mod.contextual_study_name(study_name, hpo_context),
                        target_is_categorical=dataset.target_is_categorical,
                        variable_schema_fingerprint=dataset.variable_schema_fingerprint,
                        semantic_context=generation_semantic_context,
                    )
                    try:
                        params = hpo_mod.run_study(
                            study_name,
                            objective,
                            gen_cfg.hpo,
                            output_dir,
                            seed,
                            drop_keys=(),
                            hpo_context=hpo_context,
                            checkpoint_implementation_fingerprint=implementation_fingerprint,
                            stage_a_root=stage_a_root,
                        )
                    except hpo_mod.StageAExhaustionError as exc:
                        failed_model_outcomes.append(
                            _stage_a_failure_outcome(
                                "tabpfgen_standard_hpo", exc, study_name=study_name
                            )
                        )
                        logger.warning(
                            "[generation] skipping HPO output=tabpfgen_standard_hpo after "
                            "Stage A exhaustion; study=%s evidence=%s",
                            exc.study_name,
                            exc.evidence_references,
                        )
                        params = None
                    if params is not None:
                        cache.set("tabpfgen", cache_name, params)
                if cache.has("tabpfgen", cache_name):
                    params = cache.get("tabpfgen", cache_name)

                    _cached_or_build(
                        "tabpfgen_standard_hpo",
                        lambda params=params: tpfgen.generate_tabpfgen_standard(
                            fit_imputed_df,
                            dataset.feature_columns,
                            dataset.categorical_columns,
                            dataset.target_column,
                            n_samples,
                            tabpfgen_params=params,
                            relabel_with_classifier=True,
                            target_is_categorical=dataset.target_is_categorical,
                            variable_schema_fingerprint=dataset.variable_schema_fingerprint,
                            semantic_context=generation_semantic_context,
                        ),
                        hpo_context=hpo_context,
                        resolved_parameters=params,
                    )

        if "custom" in gen_cfg.tabpfgen.variants:
            custom_params = dict(gen_cfg.tabpfgen.custom_params)
            _cached_or_build(
                "tabpfgen_custom",
                lambda custom_params=custom_params: tpfgen.generate_tabpfgen_custom(
                    fit_imputed_df,
                    dataset.feature_columns,
                    dataset.categorical_columns,
                    dataset.target_column,
                    n_samples,
                    seed=seed,
                    sgld_params=custom_params,
                    target_is_categorical=dataset.target_is_categorical,
                    variable_schema_fingerprint=dataset.variable_schema_fingerprint,
                    semantic_context=generation_semantic_context,
                ),
                resolved_parameters=custom_params,
            )

            if gen_cfg.hpo.enabled:
                cache = _require_best_params_cache()
                implementation_fingerprint = sc.generator_implementation_fingerprint(
                    "tabpfgen_custom"
                )
                cache_name = f"tabpfgen_custom-{implementation_fingerprint}"
                if not cache.has("tabpfgen", cache_name):
                    study_name = hpo_mod.resolve_study_name(
                        "hpo_tabpfgen_custom",
                        implementation_fingerprint,
                        gen_cfg.hpo,
                        output_dir,
                        seed,
                        hpo_context=hpo_context,
                        stage_a_root=stage_a_root,
                    )
                    objective = tpfgen.build_tabpfgen_custom_objective(
                        fit_imputed_df,
                        dataset.feature_columns,
                        dataset.categorical_columns,
                        dataset.target_column,
                        n_samples,
                        gen_cfg.hpo.sgld_step_cap,
                        evaluator,
                        seed=seed,
                        stage_a_contract=stage_a_contract,
                        stage_a_source_df=fit_imputed_df,
                        stage_a_root=str(stage_a_root) if stage_a_root is not None else None,
                        study_name=hpo_mod.contextual_study_name(study_name, hpo_context),
                        target_is_categorical=dataset.target_is_categorical,
                        variable_schema_fingerprint=dataset.variable_schema_fingerprint,
                        semantic_context=generation_semantic_context,
                    )
                    try:
                        params = hpo_mod.run_study(
                            study_name,
                            objective,
                            gen_cfg.hpo,
                            output_dir,
                            seed,
                            drop_keys=(),
                            hpo_context=hpo_context,
                            checkpoint_implementation_fingerprint=implementation_fingerprint,
                            stage_a_root=stage_a_root,
                        )
                    except hpo_mod.StageAExhaustionError as exc:
                        failed_model_outcomes.append(
                            _stage_a_failure_outcome(
                                "tabpfgen_custom_hpo", exc, study_name=study_name
                            )
                        )
                        logger.warning(
                            "[generation] skipping HPO output=tabpfgen_custom_hpo after "
                            "Stage A exhaustion; study=%s evidence=%s",
                            exc.study_name,
                            exc.evidence_references,
                        )
                        params = None
                    if params is not None:
                        cache.set("tabpfgen", cache_name, params)
                if cache.has("tabpfgen", cache_name):
                    params = cache.get("tabpfgen", cache_name)

                    _cached_or_build(
                        "tabpfgen_custom_hpo",
                        lambda params=params: tpfgen.generate_tabpfgen_custom(
                            fit_imputed_df,
                            dataset.feature_columns,
                            dataset.categorical_columns,
                            dataset.target_column,
                            n_samples,
                            seed=seed,
                            sgld_params=params,
                            target_is_categorical=dataset.target_is_categorical,
                            variable_schema_fingerprint=dataset.variable_schema_fingerprint,
                            semantic_context=generation_semantic_context,
                        ),
                        hpo_context=hpo_context,
                        resolved_parameters=params,
                    )

    logger.info(
        "Generated/loaded %d synthetic datasets: %s",
        len(synthetic_datasets),
        sorted(synthetic_datasets),
    )
    if experiment is not None:
        produced_outputs = sorted(synthetic_datasets)
        failed_outputs = [outcome["model"] for outcome in failed_model_outcomes]
        experiment.record(
            "generation",
            artifacts={
                "synthetic_data_dir": str(output_dir),
                "models": produced_outputs,
            },
            n_models=len(synthetic_datasets),
            status="partial" if failed_model_outcomes else "complete",
            expected_outputs=expected_outputs,
            produced_outputs=produced_outputs,
            failed_outputs=failed_outputs,
            failed_models=failed_model_outcomes,
        )
    return synthetic_datasets
