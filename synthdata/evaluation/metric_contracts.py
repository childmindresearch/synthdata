"""Versioned metric contracts and fail-closed result validation.

Framework evaluators emit loosely structured rows whose configured preset name
is not always the emitted result key. This module is the semantic boundary for
those rows: it identifies the emitted metric, records its intended use, and
turns incomplete or unsafe observations into explicit status records.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any

CONTRACT_SCHEMA_VERSION = 1
CONTRACT_REGISTRY_VERSION = "metric-contracts-v1"

METRIC_USES = frozenset(
    {"audit", "hpo_screen", "hpo_objective", "gate", "policy_rank", "final_audit_score"}
)
CONTRACT_STATES = frozenset({"operational", "calibrating", "audit_only", "blocked"})
VALUE_ROLES = frozenset({"policy_scalar", "diagnostic"})
DIRECTIONS = frozenset({"maximize", "minimize"})
POLICY_TRANSFORMS = frozenset({"identity", "absolute", "one_minus_absolute", "effective_auc"})
FRAMEWORK_IDENTITIES = {
    "synthcity": "vanderschaarlab/synthcity-editable-fork",
    "syntheval": "schneiderkamplab/syntheval-editable-fork",
    "custom": "childmindresearch/synthdata-root",
}
PREPROCESSING_CONTRACTS = {
    "synthcity": "synthcity-tabular-fit-v2",
    "syntheval": "syntheval-fit-role-v2",
    "custom": "synthdata-log-disparity-raw-role-v1",
}
GROUP_SAFETY = frozenset({"group_safe", "row_only", "not_applicable", "unknown"})
GROUP_SAFE_SYNTHCITY_CONTRACT_PREFIXES = (
    "synthcity.performance.linear_model",
    "synthcity.performance.mlp",
    "synthcity.performance.xgb",
    "synthcity.detection.detection_xgb",
    "synthcity.detection.detection_mlp",
    "synthcity.detection.detection_linear",
)
SYNTHCITY_METRIC_DIRECTIONS = {
    "sanity.data_mismatch": "minimize",
    "sanity.common_rows_proportion": "minimize",
    "sanity.nearest_syn_neighbor_distance": "minimize",
    "sanity.close_values_probability": "maximize",
    "sanity.distant_values_probability": "minimize",
    "stats.jensenshannon_dist": "minimize",
    "stats.chi_squared_test": "maximize",
    "stats.inv_kl_divergence": "maximize",
    "stats.ks_test": "maximize",
    "stats.max_mean_discrepancy": "minimize",
    "stats.wasserstein_dist": "minimize",
    "stats.prdc": "maximize",
    "stats.alpha_precision": "maximize",
    "performance.linear_model": "maximize",
    "performance.mlp": "maximize",
    "performance.xgb": "maximize",
    "performance.feat_rank_distance": "maximize",
    "performance.linear_model_augmentation": "maximize",
    "performance.mlp_augmentation": "maximize",
    "performance.xgb_augmentation": "maximize",
    "detection.detection_xgb": "minimize",
    "detection.detection_mlp": "minimize",
    "detection.detection_linear": "minimize",
    "privacy.delta-presence": "minimize",
    "privacy.k-anonymization": "maximize",
    "privacy.k-map": "maximize",
    "privacy.distinct l-diversity": "maximize",
    "privacy.identifiability_score": "minimize",
    "privacy.DomiasMIA_prior": "minimize",
    "attack.data_leakage_mlp": "minimize",
    "attack.data_leakage_xgb": "minimize",
    "attack.data_leakage_linear": "minimize",
}
RESULT_STATUSES = frozenset(
    {
        "succeeded",
        "missing",
        "duplicate",
        "non_finite",
        "invalid_value",
        "out_of_range",
        "failed",
        "unknown_contract",
        "unexpected",
        "wrong_role",
        "wrong_target_view",
        "wrong_direction",
        "group_unsafe",
        "blocked",
    }
)

# Error text is diagnostic-only at the evaluator boundary.  These are the only
# failure details allowed to cross into durable metric-status artifacts.
METRIC_STATUS_REASON_CODES = {
    "failed": "metric_evaluation_failed",
    "missing": "metric_observation_missing",
    "duplicate": "metric_observation_duplicate",
    "non_finite": "metric_value_non_finite",
    "invalid_value": "metric_value_invalid",
    "out_of_range": "metric_value_out_of_range",
    "unknown_contract": "metric_contract_unknown",
    "unexpected": "metric_observation_unexpected",
    "wrong_role": "metric_evidence_role_invalid",
    "wrong_target_view": "metric_target_view_invalid",
    "wrong_direction": "metric_direction_invalid",
    "group_unsafe": "metric_group_unsafe",
    "blocked": "metric_evaluation_blocked",
}
_EXCEPTION_TYPE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)$")
_SAFE_METADATA_KEY = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")
_SENSITIVE_METADATA = re.compile(
    r"(?:^|[^a-z])(?:traceback|exception|secret|password|passwd|hmac|api[_ -]?key|private[_ -]?key|patient[_ -]?(?:id|identifier)|authorization|[a-z_][a-z0-9_]*(?:error|exception))(?:[^a-z]|$)",
    re.IGNORECASE,
)
_OMIT_METADATA = object()


def safe_metric_metadata(value: Any, *, label: str = "metadata", strict: bool = False) -> Any:
    """Return deterministic JSON-safe metric evidence, failing closed when strict."""

    def reject(path: str) -> Any:
        if strict:
            raise ValueError(f"{path} contains unsupported or unsafe metadata")
        return _OMIT_METADATA

    def visit(item: Any, path: str) -> Any:
        if item is None or isinstance(item, bool):
            return item
        if isinstance(item, Integral):
            return int(item)
        if isinstance(item, Real):
            number = float(item)
            return number if math.isfinite(number) else reject(path)
        if isinstance(item, str):
            if (
                len(item) > 256
                or any(ord(character) < 32 for character in item)
                or Path(item).is_absolute()
                or re.match(r"^[A-Za-z]:[\\/]", item)
                or _SENSITIVE_METADATA.search(item)
            ):
                return reject(path)
            return item
        if isinstance(item, Mapping):
            result: dict[str, Any] = {}
            for key, nested in sorted(item.items(), key=lambda pair: str(pair[0])):
                if not isinstance(key, str) or not _SAFE_METADATA_KEY.fullmatch(key):
                    rejected = reject(f"{path}.key")
                    if rejected is not _OMIT_METADATA:
                        return rejected
                    continue
                safe = visit(nested, f"{path}.{key}")
                if safe is not _OMIT_METADATA:
                    result[key] = safe
            return result
        if isinstance(item, (list, tuple)):
            result = []
            for index, nested in enumerate(item):
                safe = visit(nested, f"{path}[{index}]")
                if safe is not _OMIT_METADATA:
                    result.append(safe)
            return result
        return reject(path)

    sanitized = visit(value, label)
    return {} if sanitized is _OMIT_METADATA else sanitized


def safe_metric_status_error(status: str, diagnostic: object = None) -> str | None:
    """Convert transient validation diagnostics to a safe durable reason."""
    reason_code = METRIC_STATUS_REASON_CODES.get(status)
    if reason_code is None:
        return None
    exception_type = None
    if isinstance(diagnostic, str):
        prefix = f"reason_code={reason_code}; exception_type="
        candidate = (
            diagnostic[len(prefix) :]
            if diagnostic.startswith(prefix)
            else diagnostic.split(":", 1)[0].strip()
        )
        if _EXCEPTION_TYPE.fullmatch(candidate):
            exception_type = candidate
    return (
        f"reason_code={reason_code}; exception_type={exception_type}"
        if exception_type
        else f"reason_code={reason_code}"
    )


class MetricContractError(ValueError):
    """Base error for invalid or ambiguous metric-contract operations."""


class UnknownMetricContractError(MetricContractError):
    """Raised when an emitted key has no registered contract."""


class AmbiguousMetricContractError(MetricContractError):
    """Raised when an emitted key matches more than one contract."""


@dataclasses.dataclass(frozen=True)
class MetricAnchors:
    """Reference values used by a future versioned policy transform."""

    ideal: float | None = None
    chance: float | None = None
    bad: float | None = None

    def __post_init__(self) -> None:
        for name, value in dataclasses.asdict(self).items():
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"Metric anchor {name!r} must be finite or None")

    def to_dict(self) -> dict[str, float | None]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class MetricContract:
    """Semantic and eligibility metadata for one emitted metric identity.

    ``emitted_key_pattern`` is either an exact key or a pattern with one
    ``*`` wildcard. Wildcards are used only where a framework emits qualified
    target, class, subgroup, or submetric keys.
    """

    contract_id: str
    framework: str
    emitted_key_pattern: str
    semantic_family: str
    direction: str | None
    value_role: str
    lifecycle_state: str
    allowed_uses: frozenset[str] = dataclasses.field(default_factory=lambda: frozenset({"audit"}))
    execution_pass: str = "main"
    target_view: str = "native"
    population_unit: str = "row"
    group_safety: str = "unknown"
    required_roles: tuple[str, ...] = ()
    raw_range: tuple[float | None, float | None] | None = None
    anchors: MetricAnchors = dataclasses.field(default_factory=MetricAnchors)
    uncertainty_field: str | None = None
    sample_size_field: str | None = None
    preprocessing_contract: str | None = None
    classification_score_policy: str | None = None
    schema_prerequisites: tuple[str, ...] = ()
    qualifiers: tuple[str, ...] = ()
    status_reason: str = ""
    framework_identity: str = "root-unknown-v1"
    metric_version: str = "v1"
    policy_transform: str = "identity"
    preprocessing_fit_role: str | None = "train"
    uncertainty_semantics: str | None = None
    sample_size_unit: str | None = None
    normalization_method: str = "none"
    required_support: str | None = None
    release_transform_digest: str | None = None
    seed: int | None = None
    protocol_version: str = "evaluation-protocol-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_uses", frozenset(self.allowed_uses))
        object.__setattr__(self, "required_roles", tuple(self.required_roles))
        object.__setattr__(self, "schema_prerequisites", tuple(self.schema_prerequisites))
        object.__setattr__(self, "qualifiers", tuple(self.qualifiers))

        if not self.contract_id or not self.framework or not self.emitted_key_pattern:
            raise ValueError("Metric contracts require an id, framework, and emitted key pattern")
        if self.emitted_key_pattern.count("*") > 1:
            raise ValueError("Metric key patterns may contain at most one wildcard")
        if self.lifecycle_state not in CONTRACT_STATES:
            raise ValueError(f"Unknown metric contract lifecycle state: {self.lifecycle_state!r}")
        if not self.allowed_uses or not self.allowed_uses <= METRIC_USES:
            raise ValueError(
                f"Metric contract {self.contract_id!r} has invalid allowed uses: "
                f"{sorted(self.allowed_uses - METRIC_USES)}"
            )
        for name, value in (
            ("framework_identity", self.framework_identity),
            ("metric_version", self.metric_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Metric contract {self.contract_id!r} needs a non-empty {name}")
        if self.policy_transform not in POLICY_TRANSFORMS:
            raise ValueError(
                f"Metric contract {self.contract_id!r} has unknown policy transform "
                f"{self.policy_transform!r}"
            )
        if self.value_role == "diagnostic" and self.policy_transform != "identity":
            raise ValueError(
                f"Diagnostic {self.contract_id!r} must use the identity policy transform"
            )
        for name, value in (
            ("preprocessing_contract", self.preprocessing_contract),
            ("preprocessing_fit_role", self.preprocessing_fit_role),
            ("uncertainty_semantics", self.uncertainty_semantics),
            ("sample_size_unit", self.sample_size_unit),
            ("classification_score_policy", self.classification_score_policy),
            ("normalization_method", self.normalization_method),
            ("required_support", self.required_support),
            ("release_transform_digest", self.release_transform_digest),
            ("protocol_version", self.protocol_version),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"Metric contract {self.contract_id!r} has an invalid {name}")
        if "audit" not in self.allowed_uses:
            raise ValueError(f"Metric contract {self.contract_id!r} must remain audit-visible")
        if self.lifecycle_state != "operational" and not self.status_reason:
            raise ValueError(
                f"Metric contract {self.contract_id!r} needs a reason for state "
                f"{self.lifecycle_state!r}"
            )
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError(
                f"Metric contract {self.contract_id!r} seed must be an integer or None"
            )
        if self.lifecycle_state != "operational" and self.allowed_uses & {
            "hpo_objective",
            "gate",
            "policy_rank",
        }:
            raise ValueError(
                f"Non-operational contract {self.contract_id!r} cannot permit operational use"
            )
        if self.value_role not in VALUE_ROLES:
            raise ValueError(f"Unknown metric value role: {self.value_role!r}")
        if self.value_role == "policy_scalar" and self.direction not in DIRECTIONS:
            raise ValueError(
                f"Policy scalar {self.contract_id!r} needs direction maximize or minimize"
            )
        if self.value_role == "diagnostic" and self.direction is not None:
            raise ValueError(f"Diagnostic {self.contract_id!r} must not have a policy direction")
        if self.policy_transform == "one_minus_absolute" and self.direction != "maximize":
            raise ValueError(
                f"Agreement transform for {self.contract_id!r} requires maximize direction"
            )
        if self.group_safety not in GROUP_SAFETY:
            raise ValueError(f"Unknown group-safety value: {self.group_safety!r}")
        if self.population_unit not in {"row", "patient_group"}:
            raise ValueError(
                f"Unknown population unit for {self.contract_id!r}: {self.population_unit!r}"
            )
        if self.raw_range is not None:
            lower, upper = self.raw_range
            if lower is not None and not math.isfinite(float(lower)):
                raise ValueError(f"Lower raw bound for {self.contract_id!r} must be finite")
            if upper is not None and not math.isfinite(float(upper)):
                raise ValueError(f"Upper raw bound for {self.contract_id!r} must be finite")
            if lower is not None and upper is not None and lower >= upper:
                raise ValueError(f"Raw range for {self.contract_id!r} must be increasing")

    def matches(self, *, framework: str, emitted_key: str, execution_pass: str) -> bool:
        """Return whether this contract owns an emitted identity."""
        if self.framework != framework or self.execution_pass != execution_pass:
            return False
        if "*" in self.emitted_key_pattern:
            prefix, suffix = self.emitted_key_pattern.split("*", maxsplit=1)
            return (
                emitted_key.startswith(prefix)
                and emitted_key.endswith(suffix)
                and len(emitted_key) >= len(prefix) + len(suffix)
            )
        return emitted_key == self.emitted_key_pattern

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-stable representation for manifests and digests."""
        return {
            "contract_id": self.contract_id,
            "framework": self.framework,
            "emitted_key_pattern": self.emitted_key_pattern,
            "semantic_family": self.semantic_family,
            "direction": self.direction,
            "value_role": self.value_role,
            "lifecycle_state": self.lifecycle_state,
            "allowed_uses": sorted(self.allowed_uses),
            "execution_pass": self.execution_pass,
            "target_view": self.target_view,
            "population_unit": self.population_unit,
            "group_safety": self.group_safety,
            "required_roles": list(self.required_roles),
            "raw_range": list(self.raw_range) if self.raw_range is not None else None,
            "anchors": self.anchors.to_dict(),
            "uncertainty_field": self.uncertainty_field,
            "sample_size_field": self.sample_size_field,
            "preprocessing_contract": self.preprocessing_contract,
            "classification_score_policy": self.classification_score_policy,
            "schema_prerequisites": list(self.schema_prerequisites),
            "qualifiers": list(self.qualifiers),
            "status_reason": self.status_reason,
            "framework_identity": self.framework_identity,
            "metric_version": self.metric_version,
            "policy_transform": self.policy_transform,
            "preprocessing_fit_role": self.preprocessing_fit_role,
            "uncertainty_semantics": self.uncertainty_semantics,
            "sample_size_unit": self.sample_size_unit,
            "normalization_method": self.normalization_method,
            "required_support": self.required_support,
            "release_transform_digest": self.release_transform_digest,
            "seed": self.seed,
            "protocol_version": self.protocol_version,
        }


class MetricContractRegistry:
    """Immutable lookup table for emitted metric contracts."""

    def __init__(self, contracts: Iterable[MetricContract]):
        self._contracts = tuple(contracts)
        ids = [contract.contract_id for contract in self._contracts]
        duplicates = sorted({contract_id for contract_id in ids if ids.count(contract_id) > 1})
        if duplicates:
            raise MetricContractError(f"Duplicate metric contract id(s): {duplicates}")
        self._by_id = {contract.contract_id: contract for contract in self._contracts}

    def __iter__(self):
        return iter(self._contracts)

    def __len__(self) -> int:
        return len(self._contracts)

    def get(self, contract_id: str) -> MetricContract:
        try:
            return self._by_id[contract_id]
        except KeyError as exc:
            raise UnknownMetricContractError(f"Unknown metric contract id {contract_id!r}") from exc

    def matching(
        self, *, framework: str, emitted_key: str, execution_pass: str = "main"
    ) -> tuple[MetricContract, ...]:
        return tuple(
            contract
            for contract in self._contracts
            if contract.matches(
                framework=framework,
                emitted_key=emitted_key,
                execution_pass=execution_pass,
            )
        )

    def resolve(
        self, *, framework: str, emitted_key: str, execution_pass: str = "main"
    ) -> MetricContract:
        """Resolve the most specific contract or fail closed.

        Exact emitted identities intentionally take precedence over wildcard
        diagnostic families. Among wildcard matches, the pattern with the most
        literal characters owns the key. This lets versioned diagnostic families
        coexist with their broader historical family; equally specific matches
        remain configuration errors.
        """
        matches = self.matching(
            framework=framework,
            emitted_key=emitted_key,
            execution_pass=execution_pass,
        )
        if not matches:
            raise UnknownMetricContractError(
                f"No metric contract for framework={framework!r}, emitted_key={emitted_key!r}, "
                f"execution_pass={execution_pass!r}"
            )
        exact_matches = tuple(
            contract for contract in matches if contract.emitted_key_pattern == emitted_key
        )
        if len(exact_matches) == 1:
            return exact_matches[0]
        if len(exact_matches) > 1:
            raise AmbiguousMetricContractError(
                f"Multiple metric contracts match framework={framework!r}, "
                f"emitted_key={emitted_key!r}, execution_pass={execution_pass!r}: "
                f"{[contract.contract_id for contract in exact_matches]}"
            )
        if len(matches) > 1:
            specificity = max(
                len(contract.emitted_key_pattern.replace("*", "")) for contract in matches
            )
            most_specific = tuple(
                contract
                for contract in matches
                if len(contract.emitted_key_pattern.replace("*", "")) == specificity
            )
            if len(most_specific) == 1:
                return most_specific[0]
            raise AmbiguousMetricContractError(
                f"Multiple wildcard metric contracts match framework={framework!r}, "
                f"emitted_key={emitted_key!r}, execution_pass={execution_pass!r}: "
                f"{[contract.contract_id for contract in most_specific]}"
            )
        return matches[0]

    def resolve_many(
        self,
        *,
        framework: str,
        emitted_keys: Sequence[str],
        execution_pass: str = "main",
    ) -> tuple[MetricContract, ...]:
        """Resolve a selected emitted-key list and reject duplicate selections."""
        if len(emitted_keys) != len(set(emitted_keys)):
            raise MetricContractError("Selected emitted metric keys must be unique")
        return tuple(
            self.resolve(
                framework=framework,
                emitted_key=emitted_key,
                execution_pass=execution_pass,
            )
            for emitted_key in emitted_keys
        )

    def digest(self) -> str:
        payload = {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "registry_version": CONTRACT_REGISTRY_VERSION,
            "contracts": [
                contract.to_dict()
                for contract in sorted(self._contracts, key=lambda item: item.contract_id)
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "registry_version": CONTRACT_REGISTRY_VERSION,
            "digest": self.digest(),
            "contracts": [
                contract.to_dict()
                for contract in sorted(self._contracts, key=lambda item: item.contract_id)
            ],
        }


@dataclasses.dataclass(frozen=True)
class MetricEvaluationContext:
    """Run-specific population and configuration context for validation."""

    execution_pass: str = "main"
    target_view: str = "native"
    evaluation_role: str = "tuning"
    population_unit: str = "row"
    group_mode: str = "row"
    role_hashes: Mapping[str, str] = dataclasses.field(default_factory=dict)
    resolved_configuration: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "role_hashes", dict(self.role_hashes))
        configuration = dict(self.resolved_configuration)
        # Reserved role metadata carries the release transform trust anchor through
        # legacy validators which construct this context themselves.
        if "release_transform_digest" not in configuration:
            digest = self.role_hashes.get("__release_transform_digest__")
            if digest is not None:
                configuration["release_transform_digest"] = digest
        object.__setattr__(self, "resolved_configuration", configuration)
        if self.evaluation_role not in {"tuning", "final_holdout"}:
            raise ValueError(f"Unknown evaluation role: {self.evaluation_role!r}")
        if self.group_mode not in {"row", "patient_group"}:
            raise ValueError(f"Unknown evaluation group mode: {self.group_mode!r}")
        if self.population_unit not in {"row", "patient_group"}:
            raise ValueError(f"Unknown evaluation population unit: {self.population_unit!r}")


@dataclasses.dataclass(frozen=True)
class MetricObservation:
    """One raw emitted value before contract validation."""

    model_name: str
    framework: str
    emitted_key: str
    raw_value: Any
    execution_pass: str = "main"
    target_view: str = "native"
    direction: str | None = None
    uncertainty: float | None = None
    sample_size: int | None = None
    error: str | None = None
    role_hashes: Mapping[str, str] = dataclasses.field(default_factory=dict)
    source_metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    result_metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    fit_roles: tuple[str, ...] = ()
    support: Any = None
    bandwidth: Any = None
    provenance: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "role_hashes", dict(self.role_hashes))
        object.__setattr__(self, "source_metadata", dict(self.source_metadata))
        object.__setattr__(self, "result_metadata", dict(self.result_metadata))
        object.__setattr__(self, "fit_roles", tuple(self.fit_roles))
        object.__setattr__(self, "provenance", dict(self.provenance))


_SHA256_DIGEST = re.compile(r"^[0-9a-f]{64}$")
LEGACY_TSTR_PRODUCER = "task10_tstr"


def is_verified_authoritative_tstr(
    payload: Mapping[str, object],
    *,
    trusted_final_holdout_hash: str | None = None,
    trusted_role_hashes: Mapping[str, str] | None = None,
    trusted_release_transform_digest: str | None = None,
) -> bool:
    """Check complete producer-owned final-holdout TSTR provenance.

    Optional trust anchors are supplied by the current evaluation context. They
    prevent a well-formed producer envelope from becoming evidence for a
    different final-holdout population.
    """
    metadata = payload.get("result_metadata", payload.get("metadata", {}))
    if not isinstance(metadata, Mapping):
        return False
    artifact = payload.get("prediction_artifact")
    if not isinstance(artifact, Mapping):
        artifact = metadata.get("prediction_artifact")
    artifact_valid = isinstance(artifact, Mapping) and artifact.get(
        "artifact_digest"
    ) == _tstr_artifact_digest(artifact)
    role_hashes = metadata.get("role_hashes")
    artifact_roles = artifact.get("role_hashes") if isinstance(artifact, Mapping) else None
    if not isinstance(role_hashes, Mapping) or not isinstance(artifact_roles, Mapping):
        return False
    digests = [
        metadata.get("common_protocol_digest"),
        metadata.get("release_transform_digest"),
        artifact.get("population_digest") if isinstance(artifact, Mapping) else None,
        artifact.get("population_identity") if isinstance(artifact, Mapping) else None,
        artifact.get("prediction_identity") if isinstance(artifact, Mapping) else None,
        artifact.get("artifact_digest") if isinstance(artifact, Mapping) else None,
        *role_hashes.values(),
    ]
    if any(
        not isinstance(value, str) or _SHA256_DIGEST.fullmatch(value) is None for value in digests
    ):
        return False
    if (
        trusted_final_holdout_hash is not None
        and artifact_roles.get("final_holdout") != trusted_final_holdout_hash
    ):
        return False
    if trusted_role_hashes is not None:
        expected = trusted_role_hashes.get("final_holdout")
        if expected is not None and artifact_roles.get("final_holdout") != expected:
            return False
    if (
        trusted_release_transform_digest is not None
        and metadata.get("release_transform_digest") != trusted_release_transform_digest
    ):
        return False
    return (
        metadata.get("producer") in {"authoritative_tstr", LEGACY_TSTR_PRODUCER}
        and metadata.get("protocol_version") == "tstr-v1"
        and isinstance(metadata.get("seed"), int)
        and not isinstance(metadata.get("seed"), bool)
        and metadata.get("evaluation_role") == "final_holdout"
        and metadata.get("source_role") == "synthetic"
        and metadata.get("release_form") is True
        and isinstance(metadata.get("common_protocol_digest"), str)
        and isinstance(metadata.get("release_transform_digest"), str)
        and tuple(metadata.get("fit_roles", ())) == ("train", "tuning")
        and isinstance(artifact, Mapping)
        and artifact.get("producer") in {"authoritative_tstr", LEGACY_TSTR_PRODUCER}
        and artifact.get("protocol_version") == "tstr-v1"
        and isinstance(artifact.get("seed"), int)
        and not isinstance(artifact.get("seed"), bool)
        and artifact.get("verified") is True
        and artifact.get("source_role") == "final_holdout"
        and artifact.get("population_role") == "final_holdout"
        and artifact.get("prediction_source") == "tstr_model"
        and isinstance(artifact.get("population_digest"), str)
        and isinstance(artifact.get("population_identity"), str)
        and isinstance(artifact.get("prediction_identity"), str)
        and artifact.get("prediction_length") == artifact.get("population_length")
        and artifact.get("common_protocol_digest") == metadata.get("common_protocol_digest")
        and artifact.get("seed") == metadata.get("seed")
        and artifact_valid
        and artifact_roles == role_hashes
        and artifact.get("release_transform_digest") == metadata.get("release_transform_digest")
        and artifact.get("target_identity") == metadata.get("target_identity")
        and artifact.get("protected_identity") == metadata.get("protected_identity")
    )


def _tstr_artifact_digest(artifact: Mapping[str, object]) -> str:
    """Recompute immutable authoritative TSTR prediction-artifact identity."""
    import hashlib

    payload = {key: value for key, value in artifact.items() if key != "artifact_digest"}
    return hashlib.sha256(
        repr(sorted(payload.items(), key=lambda item: item[0])).encode("utf-8")
    ).hexdigest()


# Deprecated compatibility alias for historical callers.
is_verified_task10_tstr = is_verified_authoritative_tstr


@dataclasses.dataclass(frozen=True)
class MetricStatusRecord:
    """Durable validation state for one expected or unexpected observation."""

    model_name: str
    expected_key: str
    framework: str
    status: str
    contract_id: str | None = None
    is_expected: bool = True
    raw_value: float | None = None
    policy_value: float | None = None
    uncertainty: float | None = None
    sample_size: int | None = None
    observed_count: int = 0
    execution_pass: str = "main"
    target_view: str = "native"
    population_unit: str = "row"
    group_mode: str = "row"
    required_roles: tuple[str, ...] = ()
    role_hashes: Mapping[str, str] = dataclasses.field(default_factory=dict)
    allowed_uses: frozenset[str] = dataclasses.field(default_factory=frozenset)
    value_role: str = "diagnostic"
    lifecycle_state: str = "blocked"
    direction: str | None = None
    policy_transform: str = "identity"
    qualifiers: tuple[str, ...] = ()
    error: str | None = None
    source_metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    result_metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    fit_roles: tuple[str, ...] = ()
    support: Any = None
    bandwidth: Any = None
    provenance: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in RESULT_STATUSES:
            raise ValueError(f"Unknown metric result status: {self.status!r}")
        object.__setattr__(self, "required_roles", tuple(self.required_roles))
        object.__setattr__(self, "role_hashes", dict(self.role_hashes))
        object.__setattr__(self, "allowed_uses", frozenset(self.allowed_uses))
        object.__setattr__(self, "qualifiers", tuple(self.qualifiers))
        object.__setattr__(self, "source_metadata", dict(self.source_metadata))
        object.__setattr__(self, "result_metadata", dict(self.result_metadata))
        object.__setattr__(self, "fit_roles", tuple(self.fit_roles))
        object.__setattr__(self, "provenance", dict(self.provenance))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "expected_key": self.expected_key,
            "framework": self.framework,
            "status": self.status,
            "contract_id": self.contract_id,
            "is_expected": self.is_expected,
            "raw_value": self.raw_value,
            "policy_value": self.policy_value,
            "uncertainty": self.uncertainty,
            "sample_size": self.sample_size,
            "observed_count": self.observed_count,
            "execution_pass": self.execution_pass,
            "target_view": self.target_view,
            "population_unit": self.population_unit,
            "group_mode": self.group_mode,
            "required_roles": list(self.required_roles),
            "role_hashes": dict(self.role_hashes),
            "allowed_uses": sorted(self.allowed_uses),
            "value_role": self.value_role,
            "lifecycle_state": self.lifecycle_state,
            "direction": self.direction,
            "policy_transform": self.policy_transform,
            "qualifiers": list(self.qualifiers),
            "error": safe_metric_status_error(self.status, self.error),
            "source_metadata": safe_metric_metadata(self.source_metadata, label="source_metadata"),
            "result_metadata": safe_metric_metadata(self.result_metadata, label="result_metadata"),
            "fit_roles": list(self.fit_roles),
            "support": safe_metric_metadata(self.support, label="support"),
            "bandwidth": safe_metric_metadata(self.bandwidth, label="bandwidth"),
            "provenance": safe_metric_metadata(self.provenance, label="provenance"),
        }


@dataclasses.dataclass(frozen=True)
class MetricValidationResult:
    """Resolved records for one model and one declared evaluation use."""

    model_name: str
    requested_use: str
    contract_digest: str
    records: tuple[MetricStatusRecord, ...]
    evaluation_context: MetricEvaluationContext | None = None

    def __post_init__(self) -> None:
        if self.requested_use not in METRIC_USES:
            raise ValueError(f"Unknown metric use: {self.requested_use!r}")
        object.__setattr__(self, "records", tuple(self.records))

    @property
    def expected_records(self) -> tuple[MetricStatusRecord, ...]:
        return tuple(record for record in self.records if record.is_expected)

    @property
    def expected_keys(self) -> tuple[str, ...]:
        """Stable ordered identities required by the selected evaluation."""
        return tuple(record.expected_key for record in self.expected_records)

    @property
    def completed_keys(self) -> tuple[str, ...]:
        """Expected identities with a finite, contract-valid observation."""
        return tuple(
            record.expected_key for record in self.expected_records if record.status == "succeeded"
        )

    @property
    def failed_keys(self) -> tuple[str, ...]:
        """Expected identities that emitted an evaluator failure."""
        return tuple(
            record.expected_key for record in self.expected_records if record.status == "failed"
        )

    @property
    def indeterminate_keys(self) -> tuple[str, ...]:
        """Expected identities that cannot support the requested decision use."""
        return tuple(
            record.expected_key for record in self.expected_records if record.status != "succeeded"
        )

    @property
    def complete(self) -> bool:
        return bool(self.expected_records) and all(
            record.status == "succeeded" for record in self.expected_records
        )

    @property
    def audit_complete(self) -> bool:
        """Whether every expected identity has a terminal recorded outcome."""
        return bool(self.expected_records) and all(
            record.status in RESULT_STATUSES for record in self.expected_records
        )

    @property
    def succeeded(self) -> bool:
        """Whether every expected identity produced a valid audit value."""
        return self.complete

    @property
    def decision_eligible(self) -> bool:
        if not self.complete:
            return False
        if self.requested_use in {"hpo_objective", "gate", "policy_rank", "final_audit_score"}:
            policy_records = tuple(
                record
                for record in self.expected_records
                if record.value_role == "policy_scalar"
                and self.requested_use in record.allowed_uses
            )
        else:
            ranked_records = tuple(
                record
                for record in self.expected_records
                if record.value_role == "policy_scalar" and "policy_rank" in record.allowed_uses
            )
            policy_records = ranked_records or tuple(
                record for record in self.expected_records if record.value_role == "policy_scalar"
            )
        return bool(policy_records) and all(
            record.status == "succeeded" and record.lifecycle_state == "operational"
            for record in policy_records
        )

    @property
    def policy_rank_eligible(self) -> bool:
        """Whether complete audit evidence contains usable policy-rank values."""
        if not self.complete:
            return False
        policy_records = tuple(
            record
            for record in self.expected_records
            if record.value_role == "policy_scalar" and "policy_rank" in record.allowed_uses
        )
        return bool(policy_records) and all(
            record.status == "succeeded" and record.lifecycle_state == "operational"
            for record in policy_records
        )

    @property
    def decision_status(self) -> str:
        """Return the model-level decision state without hiding audit records."""
        return "eligible" if self.decision_eligible else "indeterminate"

    @property
    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.status] = counts.get(record.status, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        context = self.evaluation_context
        return {
            "model_name": self.model_name,
            "requested_use": self.requested_use,
            "contract_digest": self.contract_digest,
            "audit_complete": self.audit_complete,
            "complete": self.complete,
            "succeeded": self.succeeded,
            "decision_eligible": self.decision_eligible,
            "policy_rank_eligible": self.policy_rank_eligible,
            "decision_status": self.decision_status,
            "expected_keys": list(self.expected_keys),
            "completed_keys": list(self.completed_keys),
            "failed_keys": list(self.failed_keys),
            "indeterminate_keys": list(self.indeterminate_keys),
            "status_counts": self.status_counts,
            "evaluation_context": (
                {
                    "execution_pass": context.execution_pass,
                    "target_view": context.target_view,
                    "evaluation_role": context.evaluation_role,
                    "population_unit": context.population_unit,
                    "group_mode": context.group_mode,
                    "role_hashes": dict(context.role_hashes),
                    "resolved_configuration": dict(context.resolved_configuration),
                }
                if context is not None
                else None
            ),
            "records": [record.to_dict() for record in self.records],
        }


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _role_hash(role_hashes: Mapping[str, str], role: str) -> str | None:
    """Read a canonical role hash, with an explicit historical test alias."""
    if role in role_hashes:
        return role_hashes[role]
    if role in {"tuning", "final_holdout"}:
        return role_hashes.get("test")
    return None


def _effective_required_roles(
    contract: MetricContract, context: MetricEvaluationContext
) -> tuple[str, ...]:
    """Resolve contract roles against the named evidence boundary.

    Contracts describe the candidate pass using ``tuning``. The same metric
    semantics may be observed after selection, but that evidence must bind to
    the canonical ``final_holdout`` role rather than aliasing its hash under
    the tuning name.
    """
    if context.evaluation_role != "final_holdout":
        return contract.required_roles
    return tuple("final_holdout" if role == "tuning" else role for role in contract.required_roles)


def _status_record(
    *,
    model_name: str,
    expected_key: str,
    framework: str,
    status: str,
    contract: MetricContract | None,
    context: MetricEvaluationContext,
    observation: MetricObservation | None = None,
    is_expected: bool = True,
    error: str | None = None,
    raw_value: float | None = None,
    policy_value: float | None = None,
    observed_count: int = 0,
) -> MetricStatusRecord:
    required_roles = _effective_required_roles(contract, context) if contract else ()
    return MetricStatusRecord(
        model_name=model_name,
        expected_key=expected_key,
        framework=framework,
        status=status,
        contract_id=contract.contract_id if contract else None,
        is_expected=is_expected,
        raw_value=raw_value,
        policy_value=policy_value,
        uncertainty=observation.uncertainty if observation else None,
        sample_size=observation.sample_size if observation else None,
        observed_count=observed_count,
        execution_pass=context.execution_pass,
        target_view=observation.target_view if observation else context.target_view,
        population_unit=context.population_unit,
        group_mode=context.group_mode,
        required_roles=required_roles,
        role_hashes=observation.role_hashes if observation else context.role_hashes,
        allowed_uses=contract.allowed_uses if contract else frozenset(),
        value_role=contract.value_role if contract else "diagnostic",
        lifecycle_state=contract.lifecycle_state if contract else "blocked",
        direction=contract.direction if contract else None,
        policy_transform=contract.policy_transform if contract else "identity",
        qualifiers=contract.qualifiers if contract else (),
        error=safe_metric_status_error(status, error),
        source_metadata=observation.source_metadata if observation else {},
        result_metadata=observation.result_metadata if observation else {},
        fit_roles=observation.fit_roles if observation else (),
        support=observation.support if observation else None,
        bandwidth=observation.bandwidth if observation else None,
        provenance=observation.provenance if observation else {},
    )


def _validate_observation(
    *,
    observation: MetricObservation,
    contract: MetricContract,
    context: MetricEvaluationContext,
    requested_use: str,
) -> tuple[str, str | None, float | None, float | None]:
    def metadata_value(name: str) -> Any:
        return observation.provenance.get(
            name,
            observation.result_metadata.get(name, observation.source_metadata.get(name)),
        )

    group_safety = observation.source_metadata.get("group_safety")
    if observation.provenance.get("tstr_producer_available") is False:
        return "blocked", "TSTR producer result/provenance is unavailable", None, None
    if isinstance(group_safety, Mapping) and group_safety.get("status") == "group_unsafe":
        return (
            "group_unsafe",
            str(
                group_safety.get(
                    "reason", "Metric path is not group-safe for the requested evaluation"
                )
            ),
            17,
            17,
        )
    if observation.error:
        return "failed", observation.error, None, None
    if observation.raw_value is None:
        return "missing", "Observation has no raw value", None, None
    raw_value = _finite_number(observation.raw_value)
    if raw_value is None:
        if isinstance(observation.raw_value, Real):
            return "non_finite", "Raw value is NaN or infinite", None, None
        return "invalid_value", "Raw value is not a real number", None, None
    if contract.raw_range is not None:
        lower, upper = contract.raw_range
        if (lower is not None and raw_value < lower) or (upper is not None and raw_value > upper):
            return "out_of_range", "Raw value is outside the contract range", raw_value, None
    if context.population_unit == "patient_group" and contract.group_safety != "group_safe":
        return (
            "group_unsafe",
            "Contract is not group-safe for patient_group evaluation",
            17,
            None,
        )
    if contract.population_unit == "patient_group" and context.population_unit != "patient_group":
        return (
            "group_unsafe",
            "Patient-group contract cannot be used with row-level evaluation context",
            None,
            None,
        )
    expected_fit_roles = tuple(
        context.resolved_configuration.get(
            "fit_roles",
            ("train", "tuning") if context.evaluation_role == "final_holdout" else ("train",),
        )
    )
    observed_fit_roles = observation.fit_roles or observation.result_metadata.get(
        "fit_roles", observation.source_metadata.get("fit_roles")
    )
    if observed_fit_roles is not None and tuple(observed_fit_roles) != expected_fit_roles:
        return (
            "wrong_role",
            f"Observed preprocessing fit roles {observed_fit_roles!r} do not match "
            f"expected {expected_fit_roles!r}",
            None,
            None,
        )
    bandwidth_fit_roles = metadata_value("bandwidth_fit_roles")
    if bandwidth_fit_roles is not None and tuple(bandwidth_fit_roles) != expected_fit_roles:
        return (
            "wrong_role",
            f"Observed bandwidth fit roles {bandwidth_fit_roles!r} do not match "
            f"expected {expected_fit_roles!r}",
            None,
            None,
        )
    if contract.preprocessing_fit_role == "train" and "train" not in expected_fit_roles:
        return "wrong_role", "Contract requires train-fitted preprocessing", None, None
    required_support = contract.required_support
    if required_support is not None:
        support = observation.support
        if support is None:
            support = observation.result_metadata.get("support")
        if support is None:
            support = observation.source_metadata.get("support")
        if contract.emitted_key_pattern == "release_privacy.v1" and isinstance(
            observation.provenance.get("release_support"), Mapping
        ):
            support = observation.provenance["release_support"]
        if support is None:
            return "missing", f"Required support {required_support!r} was not recorded", None, None
        if isinstance(support, Mapping):
            support_contract = support.get("support_contract", support.get("contract"))
            if support_contract != required_support:
                return (
                    "blocked",
                    "Recorded support contract does not match metric contract",
                    None,
                    None,
                )
            support_state = support.get("state", support.get("status"))
            if support_state in {
                "missing",
                "insufficient",
                "invalid",
                "blocked",
                "unsupported",
                "indeterminate",
            }:
                return "blocked", f"Required support is not valid: {support_state}", None, None
            if len(support) <= 1:
                return "blocked", "Required support is empty or incomplete", None, None
            if required_support == "declared_support_v1":
                roles = support.get("roles")
                if not isinstance(roles, Mapping) or set(roles) != {"synthetic", "reference"}:
                    return (
                        "blocked",
                        "Release support must declare synthetic and reference roles",
                        None,
                        None,
                    )
                for role_name, role_support in roles.items():
                    if not isinstance(role_support, Mapping):
                        return (
                            "blocked",
                            f"Release support role {role_name!r} is incomplete",
                            None,
                            None,
                        )
                    population = role_support.get("population")
                    floor = role_support.get(
                        "population_floor", support.get("role_population_floor")
                    )
                    if (
                        isinstance(population, bool)
                        or not isinstance(population, int)
                        or population < 1
                        or isinstance(floor, bool)
                        or not isinstance(floor, int)
                        or floor < 1
                        or population < floor
                    ):
                        return (
                            "blocked",
                            f"Release support role {role_name!r} is below its population floor",
                            None,
                            None,
                        )
                    if (
                        not isinstance(role_support.get("role_hash"), str)
                        or not role_support["role_hash"]
                    ):
                        return (
                            "blocked",
                            f"Release support role {role_name!r} has no role hash",
                            None,
                            None,
                        )
                protected = support.get("protected_slices")
                if not isinstance(protected, Mapping) or protected.get("state") not in {
                    "valid",
                    "not_applicable",
                }:
                    return (
                        "blocked",
                        "Release support protected-slice state is missing or invalid",
                        None,
                        None,
                    )
            if required_support == "all_target_protected_cells":
                slices = support.get("slices")
                if not isinstance(slices, (list, tuple)) or not slices:
                    return (
                        "blocked",
                        "Equalized-odds support does not declare target/protected cells",
                        None,
                        None,
                    )
                target_classes = support.get("target_classes")
                protected_domains = support.get("protected_domains")
                if not isinstance(target_classes, (list, tuple)) or not target_classes:
                    return (
                        "blocked",
                        "Equalized-odds support does not declare target classes",
                        None,
                        None,
                    )
                if not isinstance(protected_domains, Mapping) or not protected_domains:
                    return (
                        "blocked",
                        "Equalized-odds support does not declare protected domains",
                        None,
                        None,
                    )
                expected_cells = {
                    (column, target)
                    for column, groups in protected_domains.items()
                    if isinstance(groups, (list, tuple, set))
                    for target in target_classes
                }
                if any(
                    not isinstance(groups, (list, tuple, set))
                    for groups in protected_domains.values()
                ):
                    return "blocked", "Equalized-odds protected domains are invalid", None, None
                observed_cells = []
                for cell in slices:
                    if not isinstance(cell, Mapping) or not {
                        "protected_column",
                        "target_class",
                        "state",
                    }.issubset(cell):
                        return "blocked", "Equalized-odds support has incomplete cells", None, None
                    if cell["state"] != "valid":
                        return (
                            "blocked",
                            "Equalized-odds support contains invalid cells",
                            None,
                            None,
                        )
                    observed_cells.append((cell["protected_column"], cell["target_class"]))
                if len(observed_cells) != len(set(observed_cells)):
                    return "blocked", "Equalized-odds support contains duplicate cells", None, None
                observed = set(observed_cells)
                if observed != expected_cells:
                    return (
                        "blocked",
                        "Equalized-odds support does not match declared Cartesian coverage",
                        None,
                        None,
                    )
            elif not any(
                key not in {"support_contract", "contract", "state", "status"} for key in support
            ):
                return "blocked", "Required support is incomplete", None, None
        elif isinstance(support, str) and support != required_support:
            return "blocked", "Recorded support does not match metric contract", None, None
        else:
            return "blocked", "Required support must be a complete mapping", None, None
    observed_release_digest = observation.provenance.get(
        "release_transform_digest",
        observation.result_metadata.get(
            "release_transform_digest",
            observation.source_metadata.get("release_transform_digest"),
        ),
    )
    if contract.protocol_version in {"release-evidence-v2", "task12-evaluation-v1"} and (
        not isinstance(observed_release_digest, str)
        or _SHA256_DIGEST.fullmatch(observed_release_digest) is None
    ):
        return (
            "wrong_role",
            "Release-evidence release-transform digest is missing or invalid",
            None,
            None,
        )
    trusted_release_digest = context.resolved_configuration.get("release_transform_digest")
    if trusted_release_digest is not None and observed_release_digest != trusted_release_digest:
        return (
            "wrong_role",
            "Release-transform provenance digest does not match context",
            None,
            None,
        )
    if contract.release_transform_digest is not None:
        observed_digest = observed_release_digest
        if observed_digest != contract.release_transform_digest:
            return (
                "wrong_role",
                "Release-transform provenance digest does not match contract",
                None,
                None,
            )
    for field, expected in (
        ("protocol_version", contract.protocol_version),
        ("seed", contract.seed),
    ):
        observed = metadata_value(field)
        requires_observed = expected is not None and (
            field == "seed" or expected in {"release-evidence-v2", "task12-evaluation-v1"}
        )
        if requires_observed and (observed is None or observed != expected):
            return "wrong_role", f"Observation {field} does not match contract", None, None
    for field, value in (("support", observation.support), ("bandwidth", observation.bandwidth)):
        if isinstance(value, Real) and (isinstance(value, bool) or not math.isfinite(float(value))):
            return "non_finite", f"Observation {field} is non-finite", None, None
    if requested_use not in contract.allowed_uses:
        return (
            "blocked",
            contract.status_reason or f"Use {requested_use!r} is not allowed",
            None,
            None,
        )
    if (
        contract.target_view != context.target_view
        or observation.target_view != contract.target_view
    ):
        return (
            "wrong_target_view",
            "Observation target view does not match its contract",
            None,
            None,
        )
    required_roles = _effective_required_roles(contract, context)
    role_mismatches = [
        role
        for role in required_roles
        if not _role_hash(context.role_hashes, role)
        or _role_hash(observation.role_hashes, role) != _role_hash(context.role_hashes, role)
    ]
    if role_mismatches:
        return (
            "wrong_role",
            f"Missing or mismatched required role hash(es): {role_mismatches}",
            None,
            None,
        )
    if (
        contract.value_role == "policy_scalar"
        and observation.direction != contract.direction
        and observation.direction is not None
    ):
        return "wrong_direction", "Observed direction does not match the contract", None, None
    if contract.value_role == "diagnostic":
        return "succeeded", None, raw_value, None
    if contract.policy_transform == "identity":
        transformed_value = raw_value
    elif contract.policy_transform == "absolute":
        transformed_value = abs(raw_value)
    elif contract.policy_transform == "one_minus_absolute":
        transformed_value = max(0.0, min(1.0, 1.0 - abs(raw_value)))
    elif contract.policy_transform == "effective_auc":
        transformed_value = max(raw_value, 1.0 - raw_value)
    else:
        return "invalid_value", "Policy transform is not implemented", None, None
    sign = 1.0 if contract.direction == "maximize" else -1.0
    return "succeeded", None, raw_value, sign * transformed_value


def resolve_metric_observations(
    *,
    registry: MetricContractRegistry,
    model_name: str,
    framework: str,
    expected_keys: Sequence[str],
    observations: Sequence[MetricObservation],
    context: MetricEvaluationContext,
    requested_use: str = "audit",
) -> MetricValidationResult:
    """Resolve expected observations and record every validation failure.

    Unknown expected keys, missing observations, duplicates, non-finite values,
    role/view mismatches, and disallowed uses are represented as records rather
    than omitted. Extra observations are also retained as ``unexpected`` or
    ``unknown_contract`` records so callers can persist the complete audit set.
    """
    if requested_use not in METRIC_USES:
        raise MetricContractError(f"Unknown metric use: {requested_use!r}")
    if len(expected_keys) != len(set(expected_keys)):
        raise MetricContractError("Expected emitted metric keys must be unique")

    model_observations = [
        observation
        for observation in observations
        if observation.model_name == model_name
        and observation.framework == framework
        and observation.execution_pass == context.execution_pass
    ]
    records: list[MetricStatusRecord] = []
    consumed: set[int] = set()

    for expected_key in expected_keys:
        try:
            contract = registry.resolve(
                framework=framework,
                emitted_key=expected_key,
                execution_pass=context.execution_pass,
            )
        except UnknownMetricContractError as exc:
            records.append(
                _status_record(
                    model_name=model_name,
                    expected_key=expected_key,
                    framework=framework,
                    status="unknown_contract",
                    contract=None,
                    context=context,
                    error=safe_metric_status_error("unknown_contract", type(exc).__name__),
                )
            )
            continue
        except AmbiguousMetricContractError as exc:
            records.append(
                _status_record(
                    model_name=model_name,
                    expected_key=expected_key,
                    framework=framework,
                    status="unknown_contract",
                    contract=None,
                    context=context,
                    error=safe_metric_status_error("unknown_contract", type(exc).__name__),
                )
            )
            continue

        candidates = [
            (index, observation)
            for index, observation in enumerate(model_observations)
            if observation.emitted_key == expected_key
        ]
        if context.population_unit == "patient_group" and contract.group_safety != "group_safe":
            records.append(
                _status_record(
                    model_name=model_name,
                    expected_key=expected_key,
                    framework=framework,
                    status="group_unsafe",
                    contract=contract,
                    context=context,
                    observation=candidates[0][1] if candidates else None,
                    observed_count=len(candidates),
                    error="Contract is not group-safe for patient_group evaluation",
                )
            )
            consumed.update(index for index, _ in candidates)
            continue
        if not candidates:
            records.append(
                _status_record(
                    model_name=model_name,
                    expected_key=expected_key,
                    framework=framework,
                    status="missing",
                    contract=contract,
                    context=context,
                    error="Expected emitted metric was not observed",
                )
            )
            continue
        if len(candidates) > 1:
            records.append(
                _status_record(
                    model_name=model_name,
                    expected_key=expected_key,
                    framework=framework,
                    status="duplicate",
                    contract=contract,
                    context=context,
                    observation=candidates[0][1],
                    observed_count=len(candidates),
                    error="More than one observation matched the expected emitted key",
                )
            )
            consumed.update(index for index, _ in candidates)
            continue

        index, observation = candidates[0]
        consumed.add(index)
        status, error, raw_value, policy_value = _validate_observation(
            observation=observation,
            contract=contract,
            context=context,
            requested_use=requested_use,
        )
        records.append(
            _status_record(
                model_name=model_name,
                expected_key=expected_key,
                framework=framework,
                status=status,
                contract=contract,
                context=context,
                observation=observation,
                error=error,
                raw_value=raw_value,
                policy_value=policy_value,
                observed_count=1,
            )
        )

    for index, observation in enumerate(model_observations):
        if index in consumed or observation.emitted_key in expected_keys:
            continue
        try:
            contract = registry.resolve(
                framework=framework,
                emitted_key=observation.emitted_key,
                execution_pass=context.execution_pass,
            )
            status = "unexpected"
            error = "Observed emitted key was not declared as expected"
        except (UnknownMetricContractError, AmbiguousMetricContractError) as exc:
            contract = None
            status = "unknown_contract"
            error = safe_metric_status_error("unknown_contract", type(exc).__name__)
        records.append(
            _status_record(
                model_name=model_name,
                expected_key=observation.emitted_key,
                framework=framework,
                status=status,
                contract=contract,
                context=context,
                observation=observation,
                is_expected=False,
                error=error,
                observed_count=1,
            )
        )

    return MetricValidationResult(
        model_name=model_name,
        requested_use=requested_use,
        contract_digest=registry.digest(),
        records=tuple(records),
        evaluation_context=context,
    )


def _audit_contract(
    *,
    contract_id: str,
    framework: str,
    emitted_key_pattern: str,
    semantic_family: str,
    direction: str | None,
    value_role: str = "policy_scalar",
    lifecycle_state: str = "audit_only",
    execution_pass: str = "main",
    target_view: str = "native",
    population_unit: str = "row",
    required_roles: tuple[str, ...] = ("train", "tuning"),
    group_safety: str = "row_only",
    qualifiers: tuple[str, ...] = (),
    raw_range: tuple[float | None, float | None] | None = None,
    anchors: MetricAnchors | None = None,
    allowed_uses: frozenset[str] | None = None,
    uncertainty_field: str | None = None,
    sample_size_field: str | None = None,
    classification_score_policy: str | None = None,
    framework_identity: str | None = None,
    metric_version: str | None = None,
    policy_transform: str = "identity",
    preprocessing_contract: str | None = None,
    preprocessing_fit_role: str | None = "train",
    uncertainty_semantics: str | None = None,
    sample_size_unit: str | None = None,
    status_reason: str | None = None,
    normalization_method: str = "none",
    required_support: str | None = None,
    release_transform_digest: str | None = None,
    seed: int | None = None,
    protocol_version: str = "evaluation-protocol-v1",
) -> MetricContract:
    reason = status_reason or (
        "Operational use is deferred until role-safe evaluation, framework semantics, "
        "and calibration are validated."
    )
    if lifecycle_state == "blocked":
        reason = "Known semantic or implementation defect requires a versioned repair."
    if lifecycle_state == "calibrating":
        reason = "Metric requires population-specific calibration before operational use."
    if framework == "synthcity" and any(
        contract_id == prefix or contract_id.startswith(f"{prefix}.")
        for prefix in GROUP_SAFE_SYNTHCITY_CONTRACT_PREFIXES
    ):
        group_safety = "group_safe"
    if uncertainty_field is None:
        uncertainty_field = (
            "stddev" if framework == "synthcity" else "err" if framework == "syntheval" else None
        )
    if sample_size_field is None and framework == "synthcity":
        sample_size_field = "rounds"
    framework_identity = framework_identity or FRAMEWORK_IDENTITIES.get(
        framework, f"{framework}-contract"
    )
    metric_version = metric_version or ("v2" if "_v2" in emitted_key_pattern else "v1")
    preprocessing_contract = preprocessing_contract or PREPROCESSING_CONTRACTS.get(
        framework, f"{framework}-preprocessing-v1"
    )
    if framework == "custom" and preprocessing_fit_role == "train":
        preprocessing_fit_role = "not_applicable"
    if uncertainty_semantics is None:
        uncertainty_semantics = {
            "synthcity": "native standard deviation across evaluation rounds",
            "syntheval": "native standard error or method-reported uncertainty",
            "custom": "method-reported uncertainty",
        }.get(framework)
    if sample_size_unit is None and sample_size_field is not None:
        sample_size_unit = {
            "synthcity": "evaluation rounds",
            "syntheval": "valid support units",
            "custom": "method-specific support units",
        }.get(framework)
    if raw_range is None:
        raw_range = (None, None)
    classification_score_policy = (
        classification_score_policy or "framework_native_or_not_applicable"
    )
    return MetricContract(
        contract_id=contract_id,
        framework=framework,
        emitted_key_pattern=emitted_key_pattern,
        semantic_family=semantic_family,
        direction=direction if value_role == "policy_scalar" else None,
        value_role=value_role,
        lifecycle_state=lifecycle_state,
        allowed_uses=allowed_uses or frozenset({"audit"}),
        execution_pass=execution_pass,
        target_view=target_view,
        population_unit=population_unit,
        group_safety=group_safety,
        required_roles=required_roles,
        raw_range=raw_range,
        anchors=anchors or MetricAnchors(),
        uncertainty_field=uncertainty_field,
        sample_size_field=sample_size_field,
        classification_score_policy=classification_score_policy,
        preprocessing_contract=preprocessing_contract,
        qualifiers=qualifiers,
        status_reason=reason,
        framework_identity=framework_identity,
        metric_version=metric_version,
        policy_transform=policy_transform,
        preprocessing_fit_role=preprocessing_fit_role,
        uncertainty_semantics=uncertainty_semantics,
        sample_size_unit=sample_size_unit,
        normalization_method=normalization_method,
        required_support=required_support,
        release_transform_digest=release_transform_digest,
        seed=seed,
        protocol_version=protocol_version,
    )


def _build_default_contracts() -> tuple[MetricContract, ...]:
    from synthdata.evaluation.catalog import (
        SYNTHCITY_CATEGORY_TO_TYPE,
        SYNTHCITY_METRIC_CONFIG,
    )

    contracts: list[MetricContract] = []
    blocked_synthcity = {"feat_rank_distance"}
    calibrating_synthcity = {
        "identifiability_score",
        "DomiasMIA_prior",
        "delta-presence",
        "k-anonymization",
        "k-map",
        "distinct l-diversity",
        "data_leakage_mlp",
        "data_leakage_xgb",
        "data_leakage_linear",
    }
    diagnostic_synthcity = {
        "prdc",
        "alpha_precision",
    }
    # Legacy framework metrics are never HPO objectives. Canonical objectives
    # below are the only identities granted ``hpo_objective``.
    hpo_objective_synthcity = set()
    for category, metric_names in SYNTHCITY_METRIC_CONFIG.items():
        for metric_name in metric_names:
            metric_key = f"{category}.{metric_name}"
            state = (
                "blocked"
                if metric_name in blocked_synthcity
                else "calibrating"
                if metric_name in calibrating_synthcity
                else "audit_only"
            )
            allowed_uses = None
            if metric_key in hpo_objective_synthcity:
                state = "operational"
                allowed_uses = frozenset({"audit", "hpo_objective", "policy_rank"})
            value_role = "diagnostic" if metric_name in diagnostic_synthcity else "policy_scalar"
            try:
                direction = SYNTHCITY_METRIC_DIRECTIONS[metric_key]
            except KeyError as exc:
                raise MetricContractError(
                    f"SynthCity metric {metric_key!r} has no native direction contract"
                ) from exc
            if metric_key == "stats.jensenshannon_dist":
                jensen_shannon_specs = (
                    (metric_key, "", "policy_scalar", ("legacy", "aggregate")),
                    (
                        f"{metric_key}.marginal",
                        ".marginal",
                        "policy_scalar",
                        ("legacy", "aggregate"),
                    ),
                    (
                        f"{metric_key}.variable_v2.*",
                        ".variable_v2.qualified",
                        "diagnostic",
                        ("variable", "per_variable"),
                    ),
                    (
                        f"{metric_key}.source_table_macro_v2",
                        ".source_table_macro_v2",
                        "policy_scalar",
                        ("aggregate", "source_table_macro"),
                    ),
                    (
                        f"{metric_key}.max_variable_v2",
                        ".max_variable_v2",
                        "policy_scalar",
                        ("aggregate", "max_variable"),
                    ),
                )
                for pattern, suffix, result_role, qualifiers in jensen_shannon_specs:
                    contracts.append(
                        _audit_contract(
                            contract_id=f"synthcity.{category}.{metric_name}{suffix}",
                            framework="synthcity",
                            emitted_key_pattern=pattern,
                            semantic_family=SYNTHCITY_CATEGORY_TO_TYPE[category],
                            direction=direction,
                            value_role=result_role,
                            lifecycle_state="audit_only",
                            raw_range=(0.0, 1.0),
                            qualifiers=qualifiers,
                            metric_version="v2",
                        )
                    )
                continue
            if metric_key in hpo_objective_synthcity:
                hpo_qualified_specs = {
                    "sanity.nearest_syn_neighbor_distance": (
                        ("mean", "policy_scalar", (), "operational"),
                    ),
                    "stats.wasserstein_dist": (("joint", "policy_scalar", (), "operational"),),
                    "stats.inv_kl_divergence": (("marginal", "policy_scalar", (), "operational"),),
                    "performance.xgb": (
                        ("gt", "diagnostic", ("candidate_independent",), "audit_only"),
                        ("syn_id", "policy_scalar", ("candidate_dependent",), "operational"),
                        ("syn_ood", "policy_scalar", ("candidate_dependent",), "operational"),
                        ("mean", "policy_scalar", ("legacy_aggregate",), "operational"),
                    ),
                }[metric_key]
                contracts.append(
                    _audit_contract(
                        contract_id=f"synthcity.{category}.{metric_name}",
                        framework="synthcity",
                        emitted_key_pattern=metric_key,
                        semantic_family=SYNTHCITY_CATEGORY_TO_TYPE[category],
                        direction=direction,
                        value_role=value_role,
                        lifecycle_state=state,
                        allowed_uses=allowed_uses,
                    )
                )
                for suffix, qualified_role, qualifiers, qualified_state in hpo_qualified_specs:
                    qualified_allowed_uses = (
                        frozenset({"audit"}) if qualified_role == "diagnostic" else allowed_uses
                    )
                    contracts.append(
                        _audit_contract(
                            contract_id=f"synthcity.{category}.{metric_name}.{suffix}",
                            framework="synthcity",
                            emitted_key_pattern=f"{metric_key}.{suffix}",
                            semantic_family=SYNTHCITY_CATEGORY_TO_TYPE[category],
                            direction=direction,
                            value_role=qualified_role,
                            lifecycle_state=qualified_state,
                            qualifiers=qualifiers,
                            allowed_uses=qualified_allowed_uses,
                        )
                    )
                continue
            if metric_name == "DomiasMIA_prior":
                domias_specs = (
                    (f"{category}.{metric_name}", "", "policy_scalar", ()),
                    (
                        f"{category}.{metric_name}.accuracy",
                        ".accuracy",
                        "diagnostic",
                        ("raw",),
                    ),
                    (
                        f"{category}.{metric_name}.aucroc",
                        ".aucroc",
                        "diagnostic",
                        ("raw", "auc"),
                    ),
                    (
                        f"{category}.{metric_name}.effective_auc_v2",
                        ".effective_auc_v2",
                        "policy_scalar",
                        ("policy", "inversion_aware"),
                    ),
                )
                for pattern, suffix, result_role, qualifiers in domias_specs:
                    contracts.append(
                        _audit_contract(
                            contract_id=f"synthcity.{category}.{metric_name}{suffix}",
                            framework="synthcity",
                            emitted_key_pattern=pattern,
                            semantic_family=SYNTHCITY_CATEGORY_TO_TYPE[category],
                            direction=direction,
                            value_role=result_role,
                            lifecycle_state=state,
                            raw_range=(0.0, 1.0),
                            anchors=MetricAnchors(ideal=0.5, chance=0.5, bad=1.0),
                            qualifiers=qualifiers,
                        )
                    )
                continue
            if metric_name == "identifiability_score":
                identifiability_specs = (
                    (f"{category}.{metric_name}", "", ("base",)),
                    (f"{category}.{metric_name}.score", ".score", ("legacy", "unweighted")),
                    (
                        f"{category}.{metric_name}.score_OC",
                        ".score_OC",
                        ("legacy", "oneclass", "unweighted"),
                    ),
                    (
                        f"{category}.{metric_name}.score_entropy_weighted",
                        ".score_entropy_weighted",
                        ("repaired", "entropy_weighted"),
                    ),
                    (
                        f"{category}.{metric_name}.score_OC_entropy_weighted",
                        ".score_OC_entropy_weighted",
                        ("repaired", "oneclass", "entropy_weighted"),
                    ),
                )
                for pattern, suffix, qualifiers in identifiability_specs:
                    contracts.append(
                        _audit_contract(
                            contract_id=f"synthcity.{category}.{metric_name}{suffix}",
                            framework="synthcity",
                            emitted_key_pattern=pattern,
                            semantic_family=SYNTHCITY_CATEGORY_TO_TYPE[category],
                            direction=direction,
                            lifecycle_state=state,
                            raw_range=(0.0, 1.0),
                            qualifiers=qualifiers,
                        )
                    )
                continue
            if category == "detection":
                detection_direction = "minimize"
                detection_specs = (
                    (f"{category}.{metric_name}", "", "policy_scalar", ("base",)),
                    (f"{category}.{metric_name}.mean", ".mean", "diagnostic", ("raw", "auc")),
                    (
                        f"{category}.{metric_name}.raw_auc",
                        ".raw_auc",
                        "diagnostic",
                        ("raw", "auc", "explicit"),
                    ),
                    (
                        f"{category}.{metric_name}.min",
                        ".min",
                        "diagnostic",
                        ("raw", "auc"),
                    ),
                    (
                        f"{category}.{metric_name}.max",
                        ".max",
                        "diagnostic",
                        ("raw", "auc"),
                    ),
                    (
                        f"{category}.{metric_name}.effective_auc_v2",
                        ".effective_auc_v2",
                        "policy_scalar",
                        ("policy", "inversion_aware"),
                    ),
                )
                for pattern, suffix, result_role, qualifiers in detection_specs:
                    contracts.append(
                        _audit_contract(
                            contract_id=f"synthcity.{category}.{metric_name}{suffix}",
                            framework="synthcity",
                            emitted_key_pattern=pattern,
                            semantic_family=SYNTHCITY_CATEGORY_TO_TYPE[category],
                            direction=detection_direction,
                            value_role=result_role,
                            lifecycle_state=state,
                            raw_range=(0.0, 1.0),
                            anchors=MetricAnchors(ideal=0.5, chance=0.5, bad=1.0),
                            qualifiers=qualifiers,
                        )
                    )
                continue
            if category == "attack":
                attack_prefix = f"{category}.{metric_name}"
                attack_specs = [
                    (attack_prefix, "", "policy_scalar", ()),
                    (
                        f"{attack_prefix}.baseline_adjusted_advantage_v2",
                        ".baseline_adjusted_advantage_v2",
                        "policy_scalar",
                        ("policy", "worst_target"),
                    ),
                    (
                        f"{attack_prefix}.baseline_adjusted_advantage_v2.*",
                        ".baseline_adjusted_advantage_v2.qualified",
                        "diagnostic",
                        ("target", "policy_component"),
                    ),
                    (
                        f"{attack_prefix}.raw_accuracy.*",
                        ".raw_accuracy",
                        "diagnostic",
                        ("target", "raw"),
                    ),
                    (
                        f"{attack_prefix}.majority_baseline.*",
                        ".majority_baseline",
                        "diagnostic",
                        ("target", "baseline"),
                    ),
                    (
                        f"{attack_prefix}.chance_baseline.*",
                        ".chance_baseline",
                        "diagnostic",
                        ("target", "baseline"),
                    ),
                    (
                        f"{attack_prefix}.balanced_accuracy.*",
                        ".balanced_accuracy",
                        "diagnostic",
                        ("target", "balanced"),
                    ),
                    (
                        f"{attack_prefix}.macro_f1.*",
                        ".macro_f1",
                        "diagnostic",
                        ("target", "macro_f1"),
                    ),
                    (
                        f"{attack_prefix}.classification_score.*",
                        ".classification_score",
                        "diagnostic",
                        ("target", "configured_score"),
                    ),
                    (
                        f"{attack_prefix}.classification_baseline.*",
                        ".classification_baseline",
                        "diagnostic",
                        ("target", "configured_baseline"),
                    ),
                    (
                        f"{attack_prefix}.normalized_mae_v2.*",
                        ".normalized_mae_v2",
                        "diagnostic",
                        ("target", "continuous"),
                    ),
                    (
                        f"{attack_prefix}.disclosure_risk_v2.*",
                        ".disclosure_risk_v2",
                        "diagnostic",
                        ("target", "continuous"),
                    ),
                    (
                        f"{attack_prefix}.baseline_mae.*",
                        ".baseline_mae",
                        "diagnostic",
                        ("target", "baseline", "continuous"),
                    ),
                    (
                        f"{attack_prefix}.n_eval.*",
                        ".n_eval",
                        "diagnostic",
                        ("target", "sample_size"),
                    ),
                    (
                        f"{attack_prefix}.uncertainty_v2.*",
                        ".uncertainty_v2",
                        "diagnostic",
                        ("target", "uncertainty"),
                    ),
                    (
                        f"{attack_prefix}.legacy_accuracy",
                        ".legacy_accuracy",
                        "diagnostic",
                        ("legacy",),
                    ),
                ]
                attack_specs.extend(
                    (
                        f"{attack_prefix}.{reduction}",
                        f".{reduction}",
                        "diagnostic",
                        ("legacy", "aggregate"),
                    )
                    for reduction in ("mean", "min", "max")
                )
                for pattern, suffix, result_role, qualifiers in attack_specs:
                    contracts.append(
                        _audit_contract(
                            contract_id=f"synthcity.{category}.{metric_name}{suffix}",
                            framework="synthcity",
                            emitted_key_pattern=pattern,
                            semantic_family=SYNTHCITY_CATEGORY_TO_TYPE[category],
                            direction=direction,
                            value_role=result_role,
                            lifecycle_state=state,
                            raw_range=(0.0, 1.0) if result_role == "policy_scalar" else None,
                            qualifiers=qualifiers,
                            allowed_uses=allowed_uses,
                            sample_size_field="mean" if suffix == ".n_eval" else None,
                            sample_size_unit=("evaluation rows" if suffix == ".n_eval" else None),
                        )
                    )
                continue
            if metric_name in {"k-anonymization", "distinct l-diversity"}:
                structural_specs = (
                    (f"{metric_key}.gt", ".gt", "diagnostic", ("reference",)),
                    (f"{metric_key}.syn", ".syn", "policy_scalar", ("synthetic",)),
                )
                contracts.append(
                    _audit_contract(
                        contract_id=f"synthcity.{category}.{metric_name}",
                        framework="synthcity",
                        emitted_key_pattern=metric_key,
                        semantic_family=SYNTHCITY_CATEGORY_TO_TYPE[category],
                        direction=direction,
                        lifecycle_state=state,
                        allowed_uses=allowed_uses,
                    )
                )
                for pattern, suffix, result_role, qualifiers in structural_specs:
                    contracts.append(
                        _audit_contract(
                            contract_id=f"synthcity.{category}.{metric_name}{suffix}",
                            framework="synthcity",
                            emitted_key_pattern=pattern,
                            semantic_family=SYNTHCITY_CATEGORY_TO_TYPE[category],
                            direction=direction,
                            value_role=result_role,
                            lifecycle_state=state,
                            qualifiers=qualifiers,
                            allowed_uses=allowed_uses,
                        )
                    )
                continue
            if metric_name == "feat_rank_distance":
                for pattern, suffix, feature_value_role in (
                    (f"{category}.{metric_name}", "", "policy_scalar"),
                    (f"{category}.{metric_name}.corr", ".corr", "policy_scalar"),
                    (f"{category}.{metric_name}.pvalue", ".pvalue", "diagnostic"),
                ):
                    contracts.append(
                        _audit_contract(
                            contract_id=f"synthcity.{category}.{metric_name}{suffix}",
                            framework="synthcity",
                            emitted_key_pattern=pattern,
                            semantic_family=SYNTHCITY_CATEGORY_TO_TYPE[category],
                            direction=direction,
                            value_role=feature_value_role,
                            lifecycle_state=state,
                            qualifiers=("diagnostic",)
                            if feature_value_role == "diagnostic"
                            else (),
                            allowed_uses=allowed_uses,
                        )
                    )
                continue
            if category == "performance":
                output_suffixes = (
                    ("gt", "diagnostic", ("candidate_independent",)),
                    (
                        "aug_ood",
                        "policy_scalar",
                        ("candidate_dependent", "augmentation"),
                    )
                    if metric_name.endswith("_augmentation")
                    else ("syn_id", "policy_scalar", ("candidate_dependent",)),
                    ("syn_ood", "policy_scalar", ("candidate_dependent",))
                    if not metric_name.endswith("_augmentation")
                    else None,
                )
                contracts.append(
                    _audit_contract(
                        contract_id=f"synthcity.{category}.{metric_name}",
                        framework="synthcity",
                        emitted_key_pattern=metric_key,
                        semantic_family=SYNTHCITY_CATEGORY_TO_TYPE[category],
                        direction=direction,
                        lifecycle_state=state,
                        allowed_uses=allowed_uses,
                    )
                )
                for output_spec in output_suffixes:
                    if output_spec is None:
                        continue
                    output_suffix, result_role, qualifiers = output_spec
                    contracts.append(
                        _audit_contract(
                            contract_id=(f"synthcity.{category}.{metric_name}.{output_suffix}"),
                            framework="synthcity",
                            emitted_key_pattern=f"{metric_key}.{output_suffix}",
                            semantic_family=SYNTHCITY_CATEGORY_TO_TYPE[category],
                            direction=direction,
                            value_role=result_role,
                            lifecycle_state=state,
                            qualifiers=qualifiers,
                            allowed_uses=allowed_uses,
                        )
                    )
                continue
            for pattern, suffix in (
                (f"{category}.{metric_name}", ""),
                (f"{category}.{metric_name}.*", ".qualified"),
            ):
                metric_qualifiers = (
                    ("structural_proxy",)
                    if metric_name
                    in {
                        "delta-presence",
                        "k-anonymization",
                        "k-map",
                        "distinct l-diversity",
                    }
                    else ("submetric",)
                    if value_role == "diagnostic"
                    else ()
                )
                contracts.append(
                    _audit_contract(
                        contract_id=f"synthcity.{category}.{metric_name}{suffix}",
                        framework="synthcity",
                        emitted_key_pattern=pattern,
                        semantic_family=SYNTHCITY_CATEGORY_TO_TYPE[category],
                        direction=direction,
                        value_role=value_role,
                        lifecycle_state=state,
                        qualifiers=metric_qualifiers,
                        allowed_uses=allowed_uses,
                    )
                )

    syntheval_utility = {
        "avg_dwm_diff": "minimize",
        "pca_eigval_diff": "minimize",
        "pca_eigvec_ang": "minimize",
        "avg_cio": "maximize",
        "corr_mat_diff": "minimize",
        "mutual_inf_diff": "minimize",
        "ks_tvd_stat": "minimize",
        "avg_h_dist": "minimize",
        "avg_pMSE": "minimize",
        "avg_qMSE": "minimize",
        "auroc": "minimize",
        "avg_F1_diff": "minimize",
        "avg_F1_diff_hout": "minimize",
        "b_mmd": "minimize",
        "u_mmd": "minimize",
    }
    for key, direction in syntheval_utility.items():
        contracts.append(
            _audit_contract(
                contract_id=f"syntheval.{key}",
                framework="syntheval",
                emitted_key_pattern=key,
                semantic_family="utility",
                direction=direction,
            )
        )

    syntheval_v2_utility = {
        "corr_mat_diff_v2": "minimize",
        "mutual_inf_diff_v2": "minimize",
        "ks_tvd_stat_v2": "minimize",
        "avg_h_dist_v2": "minimize",
        "avg_pMSE_v2": "minimize",
        "avg_qMSE_v2": "minimize",
        "auroc_v2": "minimize",
        "avg_macro_F1_diff_v2": "minimize",
        "avg_macro_F1_diff_v2_hout": "minimize",
        "avg_balanced_accuracy_diff_v2": "minimize",
        "avg_balanced_accuracy_diff_v2_hout": "minimize",
    }
    syntheval_v2_sample_size_fields = {
        "corr_mat_diff_v2": "metadata.valid_pairs",
        "mutual_inf_diff_v2": "metadata.valid_pairs",
        "ks_tvd_stat_v2": "metadata.valid_tests",
        "frac_ks_sigs_v2": "metadata.valid_tests",
        "avg_h_dist_v2": "metadata.valid_columns",
        "avg_qMSE_v2": "metadata.valid_columns",
        "avg_pMSE_v2": "metadata.oof_n",
    }
    signed_agreement_keys = {
        "auroc_v2",
        "avg_macro_F1_diff_v2",
        "avg_balanced_accuracy_diff_v2",
    }
    for key, direction in syntheval_v2_utility.items():
        base_key = key.removesuffix("_hout")
        contract_direction = "maximize" if base_key in signed_agreement_keys else direction
        contracts.append(
            _audit_contract(
                contract_id=f"syntheval.{key}",
                framework="syntheval",
                emitted_key_pattern=key,
                semantic_family="utility",
                direction=contract_direction,
                raw_range=(-1.0, 1.0)
                if key.startswith(("auroc", "avg_macro", "avg_balanced"))
                else None,
                qualifiers=("v2",),
                sample_size_field=syntheval_v2_sample_size_fields.get(key),
                policy_transform=(
                    "one_minus_absolute" if base_key in signed_agreement_keys else "identity"
                ),
            )
        )
    # OvR aggregates have their own identities and semantics. Keep them exact
    # so qualified per-target/class diagnostics remain diagnostics rather than
    # inheriting aggregate policy metadata from a broad wildcard.
    contracts.append(
        _audit_contract(
            contract_id="syntheval.auroc_macro_ovr_v3",
            framework="syntheval",
            emitted_key_pattern="auroc_macro_ovr_v3",
            semantic_family="utility",
            direction="maximize",
            raw_range=(-1.0, 1.0),
            qualifiers=("macro_ovr", "v3"),
            metric_version="macro_ovr_v3",
            policy_transform="one_minus_absolute",
        )
    )
    for key, framework in (
        ("statistical_parity_macro_ovr_v1", "syntheval"),
        ("equalized_odds_macro_ovr_v1", "custom"),
        ("equal_opportunity_macro_ovr_v1", "custom"),
    ):
        contracts.append(
            _audit_contract(
                contract_id=f"{framework}.{key}",
                framework=framework,
                emitted_key_pattern=key,
                semantic_family="fairness",
                direction="minimize",
                raw_range=(-1.0, 1.0),
                qualifiers=("macro_ovr", "v1"),
                metric_version="macro_ovr_v1",
            )
        )
    for key in ("frac_ks_sigs_v2",):
        contracts.append(
            _audit_contract(
                contract_id=f"syntheval.{key}",
                framework="syntheval",
                emitted_key_pattern=key,
                semantic_family="utility",
                direction=None,
                value_role="diagnostic",
                qualifiers=("v2", "significance_fraction"),
                sample_size_field=syntheval_v2_sample_size_fields[key],
            )
        )
    contracts.append(
        _audit_contract(
            contract_id="syntheval.frac_ks_sigs",
            framework="syntheval",
            emitted_key_pattern="frac_ks_sigs",
            semantic_family="utility",
            direction="minimize",
            value_role="diagnostic",
            qualifiers=("significance_fraction",),
        )
    )
    for key, direction in (("auroc_", "minimize"),):
        contracts.append(
            _audit_contract(
                contract_id=f"syntheval.{key.rstrip('_')}.diagnostic",
                framework="syntheval",
                emitted_key_pattern=key + "*",
                semantic_family="utility",
                direction=direction,
                value_role="diagnostic",
                qualifiers=("target", "class", "ovr"),
                metric_version="v2",
            )
        )
    contracts.append(
        _audit_contract(
            contract_id="syntheval.auroc_macro_ovr_v3.diagnostic",
            framework="syntheval",
            emitted_key_pattern="auroc_*_ovr_v3",
            semantic_family="utility",
            direction=None,
            value_role="diagnostic",
            raw_range=(-1.0, 1.0),
            qualifiers=("target", "class", "ovr", "v3"),
            metric_version="macro_ovr_v3",
        )
    )

    syntheval_privacy = {
        "nnaa": "minimize",
        "priv_loss_nnaa": "minimize",
        "avg_nndr": "minimize",
        "priv_loss_nndr": "minimize",
        "median_DCR": "maximize",
        "hit_rate": "minimize",
        "eps_identif_risk": "minimize",
        "priv_loss_eps": "minimize",
        "mia_recall": "minimize",
        "mia_precision": "minimize",
        "att_discl_risk": "minimize",
    }
    for key, direction in syntheval_privacy.items():
        lifecycle = "calibrating" if key != "hit_rate" else "audit_only"
        contracts.append(
            _audit_contract(
                contract_id=f"syntheval.{key}",
                framework="syntheval",
                emitted_key_pattern=key,
                semantic_family="privacy",
                direction=direction,
                lifecycle_state=lifecycle,
            )
        )

    fairness_specs = (
        ("statistical_parity", "syntheval", "sp_"),
        ("equalized_odds", "custom", "eqo_"),
        ("equal_opportunity", "custom", "eo_"),
    )
    for key, framework, diagnostic_prefix in fairness_specs:
        contracts.append(
            _audit_contract(
                contract_id=f"{framework}.{key}",
                framework=framework,
                emitted_key_pattern=key,
                semantic_family="fairness",
                direction="minimize",
            )
        )
        contracts.append(
            _audit_contract(
                contract_id=f"{framework}.{key}.diagnostic",
                framework=framework,
                emitted_key_pattern=f"{diagnostic_prefix}*",
                semantic_family="fairness",
                direction="minimize",
                value_role="diagnostic",
                qualifiers=("target", "protected_group", "cell", "ovr"),
            )
        )
    for contract_id, framework, pattern, version in (
        (
            "syntheval.statistical_parity_macro_ovr_v1.diagnostic",
            "syntheval",
            "sp_ovr_v1_*",
            "macro_ovr_v1",
        ),
        (
            "custom.equalized_odds_macro_ovr_v1.diagnostic",
            "custom",
            "eqo_ovr_v1_*",
            "macro_ovr_v1",
        ),
        (
            "custom.equal_opportunity_macro_ovr_v1.diagnostic",
            "custom",
            "eo_ovr_v1_*",
            "macro_ovr_v1",
        ),
    ):
        contracts.append(
            _audit_contract(
                contract_id=contract_id,
                framework=framework,
                emitted_key_pattern=pattern,
                semantic_family="fairness",
                direction=None,
                value_role="diagnostic",
                raw_range=(-1.0, 1.0),
                qualifiers=("target", "protected_group", "cell", "ovr", "v1"),
                metric_version=version,
            )
        )

    for key, direction, value_role in (
        ("log_disparity_mean_abs", "minimize", "policy_scalar"),
        ("log_disparity_median_abs", "minimize", "diagnostic"),
        ("log_disparity_share_significant", "minimize", "policy_scalar"),
    ):
        contracts.append(
            _audit_contract(
                contract_id=f"{framework}.{key}",
                framework="custom",
                emitted_key_pattern=key,
                semantic_family="fairness",
                direction=direction,
                value_role=value_role,
                required_roles=("tuning",),
                qualifiers=("subgroup",) if value_role == "diagnostic" else (),
            )
        )

    # Explicit durable records for legacy paths. They remain audit-visible,
    # but cannot become candidate, gate, or ranking evidence.
    legacy_specs = (
        ("synthcity.privacy.k-anonymization.v1", "privacy"),
        ("synthcity.privacy.k-map.v1", "privacy"),
        ("synthcity.privacy.distinct-l-diversity.v1", "privacy"),
        ("syntheval.median_DCR.legacy", "privacy"),
        ("syntheval.eps_identif_risk.legacy", "privacy"),
        ("syntheval.mia_recall.legacy", "privacy"),
        ("syntheval.att_discl_risk.legacy", "privacy"),
        ("synthcity.performance.hidden_split.legacy", "utility"),
        ("custom.fairness.synthetic_cv.legacy", "fairness"),
        ("syntheval.avg_macro_F1_diff_v2.legacy", "utility"),
        ("syntheval.avg_F1_diff.legacy", "utility"),
    )
    from synthdata.evaluation.catalog import LEGACY_AUDIT_MANIFEST

    if tuple(key for key, _ in legacy_specs) != LEGACY_AUDIT_MANIFEST:
        raise RuntimeError("Legacy metric registry order must match catalog manifest")
    for key, family in legacy_specs:
        contracts.append(
            _audit_contract(
                contract_id=key,
                framework=key.split(".", 1)[0],
                emitted_key_pattern=key,
                semantic_family=family,
                direction="minimize",
                lifecycle_state="blocked",
                allowed_uses=frozenset({"audit"}),
                required_roles=("train", "tuning"),
                status_reason="Legacy metric path is retained only as an explicit blocked audit record.",
                qualifiers=("legacy", "blocked"),
                protocol_version="release-evidence-v2",
            )
        )

    canonical_specs = (
        (
            "synthcity",
            "elastic_net_jsd.v1",
            "minimize",
            "hpo_objective",
            ("train", "tuning"),
            "row",
            "elastic-net-jsd-v1",
            MetricAnchors(ideal=0.0, bad=1.0),
            None,
        ),
        (
            "synthcity",
            "mixed_mmd.v1",
            "minimize",
            "hpo_objective",
            ("train", "tuning"),
            "row",
            "mixed-mmd-v1",
            MetricAnchors(ideal=0.0, bad=1.0),
            None,
        ),
        (
            "syntheval",
            "tstr_macro_f1.v1",
            "maximize",
            "hpo_objective",
            ("train", "tuning"),
            "row",
            "tstr-macro-f1-v1",
            MetricAnchors(ideal=1.0, chance=0.5, bad=0.0),
            None,
        ),
        (
            "custom",
            "release_privacy.v1",
            "minimize",
            "gate",
            ("train", "tuning"),
            "row",
            "release-evidence-v2",
            MetricAnchors(ideal=0.0, bad=1.0),
            None,
        ),
        (
            "custom",
            "representation_evidence.v1",
            "minimize",
            "final_audit_score",
            ("train", "tuning"),
            "row",
            "representation-evidence-v1",
            MetricAnchors(ideal=0.0, bad=1.0),
            None,
        ),
    )
    for (
        framework,
        key,
        direction,
        use,
        required_roles,
        population_unit,
        _transform_digest,
        anchors,
        seed,
    ) in canonical_specs:
        contracts.append(
            _audit_contract(
                contract_id=f"{framework}.{key}",
                framework=framework,
                emitted_key_pattern=key,
                semantic_family="privacy" if key.startswith("release") else "fairness",
                direction=direction,
                lifecycle_state="operational"
                if use in {"hpo_objective", "gate", "final_audit_score"}
                else "audit_only",
                allowed_uses=frozenset({"audit", use}),
                required_roles=required_roles,
                group_safety="group_safe",
                population_unit=population_unit,
                raw_range=(0.0, 1.0),
                anchors=anchors,
                normalization_method="versioned_release_transform",
                required_support="declared_support_v1",
                release_transform_digest=None,
                seed=seed,
                protocol_version="release-evidence-v2",
                preprocessing_fit_role="train",
            )
        )
    contracts.append(
        _audit_contract(
            contract_id="custom.equalized_odds.final.v1",
            framework="custom",
            emitted_key_pattern="equalized_odds.final.v1",
            semantic_family="fairness",
            direction="minimize",
            lifecycle_state="operational",
            execution_pass="final_audit",
            target_view="native",
            required_roles=("train", "final_holdout"),
            group_safety="group_safe",
            allowed_uses=frozenset({"audit", "final_audit_score"}),
            raw_range=(0.0, 1.0),
            normalization_method="identity",
            required_support="all_target_protected_cells",
            release_transform_digest=None,
            protocol_version="release-evidence-v2",
        )
    )

    binary_contracts = []
    for contract in contracts:
        if contract.emitted_key_pattern not in {
            "auroc",
            "auroc_v2",
            "auroc_macro_ovr_v3",
            "statistical_parity",
            "statistical_parity_macro_ovr_v1",
            "equalized_odds",
            "equalized_odds_macro_ovr_v1",
            "equal_opportunity",
            "equal_opportunity_macro_ovr_v1",
        } and not contract.emitted_key_pattern.endswith(
            (
                "auroc_*",
                "auroc_*_ovr_v3",
                "sp_*",
                "sp_ovr_v1_*",
                "eo_*",
                "eo_ovr_v1_*",
                "eqo_*",
                "eqo_ovr_v1_*",
            )
        ):
            continue
        binary_contracts.append(
            dataclasses.replace(
                contract,
                contract_id=f"{contract.contract_id}.binary_target",
                execution_pass="binary_target",
                target_view="binary_collapsed",
            )
        )
    contracts.extend(binary_contracts)
    return tuple(contracts)


DEFAULT_METRIC_CONTRACT_REGISTRY = MetricContractRegistry(_build_default_contracts())


def contract_for_emitted_key(
    *, framework: str, emitted_key: str, execution_pass: str = "main"
) -> MetricContract:
    """Resolve an emitted key using the repository's versioned default registry."""
    return DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
        framework=framework,
        emitted_key=emitted_key,
        execution_pass=execution_pass,
    )
