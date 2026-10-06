"""Release-form privacy evidence.

This module intentionally has no pipeline side effects.  It operates on role
frames supplied by callers and keeps protocol/provenance alongside results.
Patient identity is never used as a feature.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

PROTOCOL_VERSION = "release-privacy-v1"
RELEASE_EVIDENCE_PROTOCOL_VERSION = "release-evidence-v2"


def release_privacy_evidence(
    synthetic: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    quasi_identifiers: Sequence[str],
    sensitive_fields: Sequence[str],
    k_required: int = 5,
    l_required: int = 2,
    seed: int = 0,
    role_population_floor: int = 20,
    protected_slice_floor: int = 5,
    patient_id_column: str | None = None,
) -> dict[str, Any]:
    """Return one durable, release-form privacy observation.

    This is deliberately conservative: unverified or unsupported populations
    produce an indeterminate record rather than a legacy metric alias.
    ``reference`` is used only to bind the evidence to the requested release
    protocol; no patient identity is used as a feature.
    """
    support: dict[str, Any] = {
        "support_contract": "declared_support_v1",
        "schema_version": 1,
        "state": "indeterminate",
        "role_population_floor": role_population_floor,
        "protected_slice_floor": protected_slice_floor,
        "roles": {
            "synthetic": {"population": len(synthetic), "population_floor": role_population_floor},
            "reference": {"population": len(reference), "population_floor": role_population_floor},
        },
        # Release privacy has no protected attribute input. Keeping this field
        # explicit prevents downstream consumers from treating omitted support
        # as validated support.
        "protected_slices": {"state": "not_applicable", "floor": protected_slice_floor},
    }
    result: dict[str, Any] = {
        "metric": "release_privacy.v1",
        "protocol_version": RELEASE_EVIDENCE_PROTOCOL_VERSION,
        "producer_protocol_version": PROTOCOL_VERSION,
        "status": "indeterminate",
        "support": support,
        "invalid_reasons": [],
        "producer": "release_evidence_privacy",
        "seed": seed,
        "support_contract": "declared_support_v1",
        "fit_roles": [],
        "role_hashes": {},
        "population_identity": {},
        "release_support": support,
        "error_types": [],
    }
    identity_error = _identity_column_error(
        (synthetic, reference), patient_id_column=patient_id_column
    )
    if identity_error is not None:
        result["invalid_reasons"].append(identity_error)
        return result
    try:
        synthetic_provenance = _validate_frame_provenance(synthetic)
        reference_provenance = _validate_frame_provenance(reference)
    except ValueError as exc:
        result["invalid_reasons"].append("release_provenance_invalid")
        result["error_types"].append(type(exc).__name__)
        return result
    result["fit_roles"] = (
        ["train"]
        if reference_provenance.get("source_role") == "tuning"
        else [
            "train",
            "tuning",
        ]
    )
    result["role_hashes"] = {
        "synthetic": synthetic_provenance["role_hash"],
        reference_provenance["source_role"]: reference_provenance["role_hash"],
    }
    result["population_identity"] = {
        "synthetic": _digest(
            {"role_hash": synthetic_provenance["role_hash"], "rows": len(synthetic)}
        ),
        reference_provenance["source_role"]: _digest(
            {"role_hash": reference_provenance["role_hash"], "rows": len(reference)}
        ),
    }
    support["roles"]["synthetic"].update(
        {
            "role": synthetic_provenance["source_role"],
            "role_hash": synthetic_provenance["role_hash"],
        }
    )
    support["roles"]["reference"].update(
        {
            "role": reference_provenance["source_role"],
            "role_hash": reference_provenance["role_hash"],
        }
    )
    if synthetic_provenance.get("role") != "release":
        result["invalid_reasons"].append("synthetic population is not release-form")
    if synthetic_provenance.get("source_role") != "synthetic":
        result["invalid_reasons"].append("synthetic population has invalid source role")
    if reference_provenance.get("source_role") not in {"tuning", "final_holdout"}:
        result["invalid_reasons"].append("reference population has invalid role")
    if synthetic_provenance.get("common_protocol_digest") != reference_provenance.get(
        "common_protocol_digest"
    ):
        result["invalid_reasons"].append("release protocol digest mismatch")
    if result["invalid_reasons"]:
        return result
    if role_population_floor < 1 or protected_slice_floor < 1:
        result["invalid_reasons"].append("support floors must be positive integers")
        return result
    if len(synthetic) < role_population_floor or len(reference) < role_population_floor:
        result["invalid_reasons"].append("release populations are below the declared support floor")
        return result
    if not quasi_identifiers or not sensitive_fields:
        result["invalid_reasons"].append(
            "release privacy schema must declare QIs and sensitive fields"
        )
        return result
    support["state"] = "valid"
    result["provenance"] = {
        "synthetic": dict(synthetic_provenance),
        "reference": dict(reference_provenance),
    }
    result.update(
        {
            "protocol_version": RELEASE_EVIDENCE_PROTOCOL_VERSION,
            "producer_protocol_version": PROTOCOL_VERSION,
            "release_transform_digest": synthetic_provenance.get("release_transform_digest"),
            "common_protocol_digest": synthetic_provenance["common_protocol_digest"],
            "role_hashes": {
                "synthetic": synthetic_provenance["role_hash"],
                reference_provenance["source_role"]: reference_provenance["role_hash"],
            },
            "fit_roles": ["train"]
            if reference_provenance["source_role"] == "tuning"
            else ["train", "tuning"],
            "evaluation_role": reference_provenance["source_role"],
            "population_identity": {
                "synthetic": _digest(
                    {"role_hash": synthetic_provenance["role_hash"], "rows": len(synthetic)}
                ),
                reference_provenance["source_role"]: _digest(
                    {"role_hash": reference_provenance["role_hash"], "rows": len(reference)}
                ),
            },
        }
    )
    try:
        k_result = k_anonymity(synthetic, quasi_identifiers, required=k_required)
        l_result = l_diversity(synthetic, quasi_identifiers, sensitive_fields, required=l_required)
    except (KeyError, ValueError) as exc:
        result["invalid_reasons"].append("release_privacy_metric_invalid")
        result["error_types"].append(type(exc).__name__)
        return result
    result.update(
        {
            "status": "succeeded"
            if k_result["status"] == "succeeded" and l_result["status"] == "succeeded"
            else "indeterminate",
            "k": k_result,
            "l": l_result,
            "safety_score": min(k_result["safety_score"], l_result["safety_score"]),
            "value": max(0.0, 1.0 - min(k_result["safety_score"], l_result["safety_score"])),
            "provenance": result["provenance"],
        }
    )
    return result


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()


def _release_transform_digest(protocol_version: str, generalization: Mapping[str, Any]) -> str:
    """Return identity of release transformation, independent of population rows."""
    return _digest({"protocol_version": protocol_version, "generalization": generalization})


def _frame_role_hash(frame: pd.DataFrame, role: str) -> str:
    return _digest({"role": role, "rows": frame.to_dict("records")})


def _common_protocol_digest(protocol_version: str, frames: Sequence[Mapping[str, Any]]) -> str:
    """Digest immutable protocol and per-frame release identities."""
    return _digest(
        {
            "protocol_version": protocol_version,
            "frames": sorted(
                [
                    {
                        "source_role": item["source_role"],
                        "role": item["role"],
                        "digest": item["digest"],
                    }
                    for item in frames
                ],
                key=lambda item: (item["source_role"], item["role"]),
            ),
        }
    )


def _validate_frame_provenance(frame: pd.DataFrame) -> dict[str, Any]:
    provenance = frame.attrs.get("release_provenance", {})
    required = (
        "release_form",
        "role",
        "source_role",
        "protocol_version",
        "digest",
        "role_hash",
        "common_protocol_digest",
    )
    if not all(provenance.get(key) for key in required):
        raise ValueError("MIA requires verified release provenance on all populations")
    source_role = provenance["source_role"]
    expected_public = "release" if source_role == "synthetic" else source_role
    if provenance["role"] != expected_public:
        raise ValueError("release provenance public role does not match source role")
    role_hash = _frame_role_hash(frame, source_role)
    if role_hash != provenance["role_hash"]:
        raise ValueError("release provenance role_hash does not match frame contents")
    metadata = {
        "protocol_version": provenance["protocol_version"],
        "role": provenance.get("source_role", provenance["role"]),
        "row_count": len(frame),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "generalization": provenance.get("generalization", {}),
    }
    if _digest({**metadata, "role_hash": provenance["role_hash"]}) != provenance["digest"]:
        raise ValueError("release provenance digest does not match frame metadata")
    return provenance


_KNOWN_IDENTITY_COLUMNS = frozenset(
    {
        "patient_id",
        "patientid",
        "patient_identifier",
        "person_id",
        "subject_id",
    }
)


def _declared_identity_columns(frame: pd.DataFrame) -> set[str]:
    """Return identity columns declared by a frame's canonical metadata."""
    declared: set[str] = set()
    for metadata in (frame.attrs, frame.attrs.get("identity_metadata", {})):
        if not isinstance(metadata, Mapping):
            continue
        for key in ("identity_column", "patient_id_column"):
            value = metadata.get(key)
            if isinstance(value, str):
                declared.add(value)
        values = metadata.get("identity_columns")
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            declared.update(value for value in values if isinstance(value, str))
        if metadata.get("raw_identifier_persisted") is True:
            declared.update(_KNOWN_IDENTITY_COLUMNS & set(frame.columns))
    return declared


def _identity_column_error(
    frames: Sequence[pd.DataFrame], *, patient_id_column: str | None
) -> str | None:
    """Reject identity-bearing frames at canonical release/privacy boundaries."""
    for frame in frames:
        identity_columns = _declared_identity_columns(frame)
        if patient_id_column is not None:
            identity_columns.add(patient_id_column)
        identity_columns.update(_KNOWN_IDENTITY_COLUMNS & set(frame.columns))
        present = identity_columns & set(frame.columns)
        if present:
            columns = ", ".join(sorted(present))
            return f"patient ID cannot be present in canonical release frames: {columns}"
    return None


def _specs(generalization: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not generalization:
        return {}
    columns = generalization.get("columns") if isinstance(generalization, Mapping) else None
    return columns if isinstance(columns, Mapping) else generalization


def _interval(value: Any, intervals: Sequence[Mapping[str, Any]]) -> Any:
    if pd.isna(value):
        return value
    for item in intervals:
        lower, upper = item.get("lower"), item.get("upper")
        # Config intervals are contiguous [lower, upper), with open ends.
        if (lower is None or value >= lower) and (upper is None or value < upper):
            return item["label"]
    raise ValueError(f"value {value!r} falls outside release intervals")


def transform_release(
    frame: pd.DataFrame,
    generalization: Mapping[str, Any] | None = None,
    *,
    role: str,
    patient_id_column: str | None = None,
    protocol_version: str = PROTOCOL_VERSION,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply configured release transformations without suppressing rows.

    Returns transformed frame and deterministic provenance.  ``role`` is part
    of provenance but does not affect values, so real matching-role frames can
    be transformed with exactly the same rules.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if patient_id_column is not None and patient_id_column in frame.columns:
        raise ValueError("patient ID must be removed before release transformation")
    result = frame.copy(deep=True)
    specs = _specs(generalization)
    for column, spec in specs.items():
        if column not in result.columns:
            raise KeyError(f"release generalization column {column!r} is absent")
        if not isinstance(spec, Mapping) or not isinstance(spec.get("intervals"), Sequence):
            raise ValueError(f"release generalization for {column!r} requires intervals")
        intervals = list(spec["intervals"])
        result[column] = result[column].map(
            lambda value, bounds=intervals: _interval(value, bounds)
        )
    metadata = {
        "protocol_version": protocol_version,
        "role": role,
        "row_count": len(result),
        "columns": list(result.columns),
        "dtypes": {column: str(dtype) for column, dtype in result.dtypes.items()},
        "generalization": json.loads(json.dumps(specs, sort_keys=True, default=str)),
    }
    metadata["role_hash"] = _frame_role_hash(result, role)
    metadata["digest"] = _digest({**metadata, "role_hash": metadata["role_hash"]})
    result.attrs["release_provenance"] = {
        "release_form": True,
        "role": "release" if role == "synthetic" else role,
        "source_role": role,
        "protocol_version": protocol_version,
        "role_hash": metadata["role_hash"],
        "digest": metadata["digest"],
        "row_count": metadata["row_count"],
        "columns": metadata["columns"],
        "dtypes": metadata["dtypes"],
        "generalization": metadata["generalization"],
        "release_transform_digest": _release_transform_digest(
            protocol_version, metadata["generalization"]
        ),
        # Role is also carried as an explicit evaluation identity.  Consumers
        # must not infer final-audit provenance from a public role alone.
        "evaluation_role": role if role in {"tuning", "final_holdout"} else None,
    }
    return result, metadata


def transform_release_roles(
    synthetic: pd.DataFrame,
    real_roles: Mapping[str, pd.DataFrame],
    generalization: Mapping[str, Any] | None = None,
    *,
    patient_id_column: str | None = None,
    protocol_version: str = PROTOCOL_VERSION,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    """Transform synthetic and each matching real role under one protocol.

    ``real_roles`` must contain named role frames (normally ``train``,
    ``tuning`` and ``final_holdout``).  Metadata includes per-role hashes and a
    common protocol digest, allowing downstream callers to verify that all
    evidence used the same release rules.
    """
    identity_error = _identity_column_error(
        (synthetic, *real_roles.values()), patient_id_column=patient_id_column
    )
    if identity_error is not None:
        raise ValueError(identity_error)
    syn, syn_meta = transform_release(
        synthetic,
        generalization,
        role="synthetic",
        patient_id_column=patient_id_column,
        protocol_version=protocol_version,
    )
    released: dict[str, pd.DataFrame] = {}
    role_meta: dict[str, Any] = {}
    for role, frame in real_roles.items():
        released[role], role_meta[role] = transform_release(
            frame,
            generalization,
            role=role,
            patient_id_column=patient_id_column,
            protocol_version=protocol_version,
        )
    metadata = {
        "protocol_version": protocol_version,
        "synthetic": syn_meta,
        "roles": role_meta,
        "digest": _digest(
            {
                "protocol_version": protocol_version,
                "synthetic": syn_meta["digest"],
                "roles": {role: value["digest"] for role, value in role_meta.items()},
            }
        ),
    }
    common_digest = metadata["digest"]
    digest_inputs = [
        {**syn.attrs["release_provenance"]},
        *[frame.attrs["release_provenance"] for frame in released.values()],
    ]
    common_digest = _common_protocol_digest(protocol_version, digest_inputs)
    metadata["common_protocol_digest"] = common_digest
    if len(real_roles) == 1:
        evaluation_role = next(iter(real_roles))
        syn.attrs["release_provenance"]["evaluation_role"] = evaluation_role
    syn.attrs["release_provenance"]["common_protocol_digest"] = common_digest
    for frame in released.values():
        frame.attrs["release_provenance"]["common_protocol_digest"] = common_digest
        frame.attrs["release_provenance"]["evaluation_role"] = frame.attrs["release_provenance"][
            "source_role"
        ]
    return syn, released, metadata


# Descriptive alias for callers using task terminology.
release_form = transform_release


def equivalence_classes(
    frame: pd.DataFrame, quasi_identifiers: Sequence[str]
) -> dict[tuple[Any, ...], list[int]]:
    """Return actual release-form QI equivalence classes, preserving rows."""
    missing = set(quasi_identifiers) - set(frame.columns)
    if missing:
        raise KeyError(f"missing quasi-identifiers: {sorted(missing)}")
    groups: dict[tuple[Any, ...], list[int]] = {}
    for position, values in enumerate(
        frame[list(quasi_identifiers)].itertuples(index=False, name=None)
    ):
        key = tuple(None if pd.isna(value) else value for value in values)
        groups.setdefault(key, []).append(position)
    return groups


def k_anonymity(
    frame: pd.DataFrame, quasi_identifiers: Sequence[str], *, required: int = 5
) -> dict[str, Any]:
    """Measure k using every observed QI equivalence class."""
    if required != 5:
        raise ValueError("k-anonymity safety policy is fixed at 5")
    classes = equivalence_classes(frame, quasi_identifiers)
    sizes = [len(rows) for rows in classes.values()]
    observed = min(sizes) if sizes else 0
    return {
        "status": "succeeded" if sizes else "indeterminate",
        "k_observed": observed,
        "k_required": required,
        "safety_score": min(1.0, observed / 5),
        "n_classes": len(classes),
        "class_sizes": sizes,
        "rows_suppressed": 0,
    }


def l_diversity(
    frame: pd.DataFrame,
    quasi_identifiers: Sequence[str],
    sensitive_fields: Sequence[str],
    *,
    required: int = 2,
) -> dict[str, Any]:
    """Measure distinct l-diversity separately for every sensitive field."""
    if required != 2:
        raise ValueError("l-diversity safety policy is fixed at 2")
    missing = set(sensitive_fields) - set(frame.columns)
    if missing:
        raise KeyError(f"missing sensitive fields: {sorted(missing)}")
    classes = equivalence_classes(frame, quasi_identifiers)
    per_field = {
        field: [frame.iloc[rows][field].nunique(dropna=False) for rows in classes.values()]
        for field in sensitive_fields
    }
    observed_by_field = {field: min(values) if values else 0 for field, values in per_field.items()}
    observed = min((min(values) for values in per_field.values() if values), default=0)
    return {
        "status": "succeeded" if classes and sensitive_fields else "indeterminate",
        "l_observed": observed,
        "l_required": required,
        "safety_score": min(1.0, observed / 2),
        "l_observed_by_field": observed_by_field,
        "safety_score_by_field": {
            field: min(1.0, value / 2) for field, value in observed_by_field.items()
        },
        "per_field": per_field,
        "n_classes": len(classes),
        "rows_suppressed": 0,
    }


def _row_distances(left: pd.DataFrame, right: pd.DataFrame) -> np.ndarray:
    if left.empty or right.empty:
        return np.array([], dtype=float)
    columns = [c for c in left.columns if c in right.columns]
    if not columns:
        return np.array([], dtype=float)
    parts = []
    for column in columns:
        a, b = left[column], right[column]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            scale = float(b.max() - b.min()) if len(b) else 0.0
            scale = scale if math.isfinite(scale) and scale > 0 else 1.0
            parts.append(
                np.abs(a.to_numpy(dtype=float)[:, None] - b.to_numpy(dtype=float)[None, :]) / scale
            )
        else:
            parts.append(
                (
                    a.astype(object).to_numpy()[:, None] != b.astype(object).to_numpy()[None, :]
                ).astype(float)
            )
    distances = parts[0] if len(parts) == 1 else np.mean(np.stack(parts), axis=0)
    return np.min(distances, axis=1)


def closest_record_distance(
    synthetic: pd.DataFrame, train_tuning: pd.DataFrame, final_holdout: pd.DataFrame
) -> dict[str, Any]:
    """Compute formal median nearest-record ratio and tanh score."""
    numerator = _row_distances(synthetic, train_tuning)
    denominator = _row_distances(final_holdout, train_tuning)
    n, d = (
        (float(np.median(numerator)) if len(numerator) else math.nan),
        (float(np.median(denominator)) if len(denominator) else math.nan),
    )
    valid = math.isfinite(n) and math.isfinite(d) and d > 0
    return {
        "status": "succeeded" if valid else "indeterminate",
        "median_synthetic": n,
        "median_holdout": d,
        "ratio": n / d if valid else math.nan,
        "score": math.tanh(n / d) if valid else math.nan,
        "threshold_policy": "audit_only_no_pass_fail",
        "required_ratio": None,
    }


dcr = closest_record_distance


def epsilon_identifiability(
    member_risk: Sequence[float] | pd.DataFrame,
    nonmember_risk: Sequence[float] | pd.DataFrame,
    *,
    risk_function: Any | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Run exactly ten fixed balanced draws and report identifiability excess.

    For frames, ``risk_function(member_draw, nonmember_draw, seed)`` computes
    one risk pair.  For sequences, values are accepted as already-computed
    per-seed evidence (useful for adapters), but still must contain exactly
    ten entries.  This quantity is never differential-privacy epsilon.
    """
    if risk_function is not None:
        if seed != 0:
            raise ValueError("epsilon protocol uses fixed seeds 0 through 9")
        if not isinstance(member_risk, pd.DataFrame) or not isinstance(
            nonmember_risk, pd.DataFrame
        ):
            return {
                "status": "indeterminate",
                "reason": "draw runner requires member/non-member frames",
            }
        n = min(len(member_risk), len(nonmember_risk))
        if n < 1:
            return {"status": "indeterminate", "reason": "empty balanced populations"}
        members, nonmembers = [], []
        draw_size = max(1, n // 2)
        for draw_seed in range(10):
            rng = np.random.default_rng(draw_seed)
            member_indices = rng.integers(0, len(member_risk), size=draw_size)
            nonmember_indices = rng.integers(0, len(nonmember_risk), size=draw_size)
            result = risk_function(
                member_risk.iloc[member_indices], nonmember_risk.iloc[nonmember_indices], draw_seed
            )
            members.append(float(result[0]))
            nonmembers.append(float(result[1]))
    else:
        members, nonmembers = list(member_risk), list(nonmember_risk)
    members_array, nonmembers_array = (
        np.asarray(members, dtype=float),
        np.asarray(nonmembers, dtype=float),
    )
    if (
        members_array.shape != nonmembers_array.shape
        or members_array.size != 10
        or not np.isfinite(members_array).all()
        or not np.isfinite(nonmembers_array).all()
    ):
        return {
            "status": "indeterminate",
            "reason": "exactly ten finite balanced draws required",
            "seeds": list(range(10)),
        }
    mr, nr = float(members_array.mean()), float(nonmembers_array.mean())
    return {
        "status": "succeeded",
        "member_risk": mr,
        "nonmember_risk": nr,
        "member_risks": members_array.tolist(),
        "nonmember_risks": nonmembers_array.tolist(),
        "positive_excess": max(0.0, mr - nr),
        "repetitions": 10,
        "seeds": list(range(10)),
        "interpretation": "identifiability excess; not differential-privacy epsilon",
    }


def full_record_mia(
    synthetic: pd.DataFrame,
    members: pd.DataFrame,
    nonmembers: pd.DataFrame,
    *,
    patient_id_column: str | None = None,
    member_groups: Sequence[Any] | None = None,
    nonmember_groups: Sequence[Any] | None = None,
    protected_member: Sequence[Any] | None = None,
    protected_nonmember: Sequence[Any] | None = None,
    seeds: Sequence[int] = (0,),
    bootstrap_seed: int = 0,
    protected_slice_floor: int = 5,
    expected_common_protocol_digest: str | None = None,
) -> dict[str, Any]:
    """Run record-level membership inference using released records only.

    This lightweight, deterministic baseline trains one nearest-record score
    per seed.  It is intentionally not a patient-ID attack: identity columns
    are rejected/dropped, and callers must provide patient-disjoint nonmembers.
    ``protected`` is optional and must align with ``nonmembers`` for slice
    support reporting.
    """
    frames = (synthetic, members, nonmembers)
    if patient_id_column is not None and any(
        patient_id_column in frame.columns for frame in frames
    ):
        raise ValueError("patient ID cannot be an MIA feature")
    if not expected_common_protocol_digest:
        raise ValueError("MIA requires external expected_common_protocol_digest trust anchor")
    provenance = [_validate_frame_provenance(frame) for frame in frames]
    if (
        provenance[0]["role"] != "release"
        or provenance[1]["role"] != "train_tuning"
        or provenance[2]["role"] != "final_holdout"
    ):
        raise ValueError("MIA population roles must be release, train/tuning, and final_holdout")
    common_digest = _common_protocol_digest(provenance[0]["protocol_version"], provenance)
    if common_digest != expected_common_protocol_digest:
        raise ValueError("MIA common release digest does not match external trust anchor")
    if any(item["common_protocol_digest"] != common_digest for item in provenance):
        raise ValueError("MIA populations have mismatched common release digest")
    if member_groups is None or nonmember_groups is None:
        raise ValueError("explicit patient groups are required; row identity fallback is forbidden")
    if len(member_groups) != len(members) or len(nonmember_groups) != len(nonmembers):
        raise ValueError("patient groups must align with member/non-member rows")
    overlap = set(member_groups) & set(nonmember_groups)
    if overlap:
        raise ValueError("member and non-member patient groups must be disjoint")
    if (protected_member is None) != (protected_nonmember is None):
        raise ValueError("protected labels must be supplied for both populations")
    if (
        protected_member is not None
        and protected_nonmember is not None
        and (len(protected_member) != len(members) or len(protected_nonmember) != len(nonmembers))
    ):
        raise ValueError("protected labels must align with both populations")
    score_member = _row_distances(members, synthetic)
    score_nonmember = _row_distances(nonmembers, synthetic)
    if not len(score_member) or not len(score_nonmember):
        return {
            "status": "indeterminate",
            "reason": "empty member/non-member population",
            "seeds": list(seeds),
        }
    # Higher score means more likely member (closer to synthetic record).
    member_scores, nonmember_scores = -score_member, -score_nonmember

    def auc(member_values: np.ndarray, nonmember_values: np.ndarray) -> float:
        values = np.r_[member_values, nonmember_values]
        labels = np.r_[np.ones(len(member_values)), np.zeros(len(nonmember_values))]
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(order), dtype=float)
        sorted_values = values[order]
        ranks[order] = pd.Series(sorted_values).rank(method="average").to_numpy()
        return float(
            (ranks[labels == 1].sum() - len(member_values) * (len(member_values) + 1) / 2)
            / (len(member_values) * len(nonmember_values))
        )

    seed_results = []
    protected_seed_results: dict[str, list[dict[str, Any]]] = {}
    protected_values = (
        []
        if protected_member is None
        else list(
            pd.concat([pd.Series(protected_member), pd.Series(protected_nonmember)])
            .dropna()
            .unique()
        )
    )
    if (
        protected_member is not None
        and pd.concat([pd.Series(protected_member), pd.Series(protected_nonmember)]).isna().any()
    ):
        protected_values.append("<MISSING>")
    for seed_value in seeds:
        rng_seed = np.random.default_rng(seed_value)
        member_indices = rng_seed.integers(0, len(member_scores), size=len(member_scores))
        nonmember_indices = rng_seed.integers(0, len(nonmember_scores), size=len(nonmember_scores))
        seed_auc = auc(member_scores[member_indices], nonmember_scores[nonmember_indices])
        seed_results.append(
            {"seed": seed_value, "auc": seed_auc, "effective_auc_advantage": abs(seed_auc - 0.5)}
        )
        if protected_member is not None:
            sampled_member_protected = pd.Series(protected_member).to_numpy()[member_indices]
            sampled_nonmember_protected = pd.Series(protected_nonmember).to_numpy()[
                nonmember_indices
            ]
            for value in protected_values:
                m_mask = (
                    pd.isna(sampled_member_protected)
                    if value == "<MISSING>"
                    else sampled_member_protected == value
                )
                n_mask = (
                    pd.isna(sampled_nonmember_protected)
                    if value == "<MISSING>"
                    else sampled_nonmember_protected == value
                )
                entry = protected_seed_results.setdefault(str(value), [])
                if m_mask.sum() < protected_slice_floor or n_mask.sum() < protected_slice_floor:
                    entry.append(
                        {
                            "seed": seed_value,
                            "status": "indeterminate",
                            "member_support": int(m_mask.sum()),
                            "nonmember_support": int(n_mask.sum()),
                            "support_floor": protected_slice_floor,
                            "reason": "protected slice support below floor",
                        }
                    )
                else:
                    value_auc = auc(
                        member_scores[member_indices][m_mask],
                        nonmember_scores[nonmember_indices][n_mask],
                    )
                    complement_m = ~m_mask
                    complement_n = ~n_mask
                    if not complement_m.any() or not complement_n.any():
                        entry.append(
                            {
                                "seed": seed_value,
                                "status": "indeterminate",
                                "member_support": int(m_mask.sum()),
                                "nonmember_support": int(n_mask.sum()),
                                "support_floor": protected_slice_floor,
                                "reason": "missing OVR complement",
                            }
                        )
                    else:
                        all_scores = np.r_[
                            member_scores[member_indices], nonmember_scores[nonmember_indices]
                        ]
                        all_protected = np.r_[sampled_member_protected, sampled_nonmember_protected]
                        positive = (
                            all_protected == value
                            if value != "<MISSING>"
                            else pd.isna(all_protected)
                        )
                        ovr_auc = auc(all_scores[positive], all_scores[~positive])
                        advantage = max(value_auc, 1 - value_auc) - 0.5
                        ovr_advantage = max(ovr_auc, 1 - ovr_auc) - 0.5
                        entry.append(
                            {
                                "seed": seed_value,
                                "status": "succeeded",
                                "auc": value_auc,
                                "effective_auc_advantage": advantage,
                                "member_support": int(m_mask.sum()),
                                "nonmember_support": int(n_mask.sum()),
                                "support_floor": protected_slice_floor,
                                "ovr_auc": ovr_auc,
                                "ovr_effective_auc_advantage": ovr_advantage,
                                "ovr_complement_support": int((~positive).sum()),
                                "ovr_gap": advantage - ovr_advantage,
                            }
                        )
    global_auc = float(np.mean([item["auc"] for item in seed_results]))
    group_member = pd.Series(member_groups).reset_index(drop=True)
    group_nonmember = pd.Series(nonmember_groups).reset_index(drop=True)
    rng = np.random.default_rng(bootstrap_seed)
    all_groups = np.array(sorted(set(group_member) | set(group_nonmember), key=str), dtype=object)
    bootstrap = []
    for _ in range(1000):
        sampled = rng.choice(all_groups, size=len(all_groups), replace=True)
        member_indices = np.concatenate(
            [np.flatnonzero(group_member.to_numpy() == group) for group in sampled]
        )
        nonmember_indices = np.concatenate(
            [np.flatnonzero(group_nonmember.to_numpy() == group) for group in sampled]
        )
        if len(member_indices) and len(nonmember_indices):
            bootstrap.append(
                auc(member_scores[member_indices], nonmember_scores[nonmember_indices])
            )
    bootstrap_valid = len(bootstrap) == 1000
    interval = (
        [float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))]
        if bootstrap_valid
        else [math.nan, math.nan]
    )
    result: dict[str, Any] = {
        "status": "succeeded" if bootstrap_valid else "indeterminate",
        "global_auc": global_auc,
        "effective_auc_advantage": abs(global_auc - 0.5),
        "seeds": list(seeds),
        "seed_results": seed_results,
        "support": {
            "members": len(members),
            "nonmembers": len(nonmembers),
            "member_patients": len(set(member_groups)),
            "nonmember_patients": len(set(nonmember_groups)),
        },
        "patient_cluster_bootstrap": {
            "resamples": 1000,
            "seed": bootstrap_seed,
            "attempts": 1000,
            "valid_attempts": len(bootstrap),
            "status": "succeeded" if bootstrap_valid else "indeterminate",
            "percentile_95": interval,
        },
    }
    if protected_member is not None:
        member_protected = pd.Series(protected_member).reset_index(drop=True)
        nonmember_protected = pd.Series(protected_nonmember).reset_index(drop=True)
        slices = {}
        protected_values = list(
            pd.concat([member_protected, nonmember_protected]).dropna().unique()
        )
        if pd.concat([member_protected, nonmember_protected]).isna().any():
            protected_values.append("<MISSING>")
        for value in protected_values:
            member_value = (
                member_protected.isna() if value == "<MISSING>" else member_protected == value
            )
            nonmember_value = (
                nonmember_protected.isna() if value == "<MISSING>" else nonmember_protected == value
            )
            m_mask, n_mask = member_value.to_numpy(), nonmember_value.to_numpy()
            support = int(m_mask.sum() + n_mask.sum())
            key = str(value)
            if support < protected_slice_floor or not m_mask.any() or not n_mask.any():
                slices[key] = {
                    "status": "indeterminate",
                    "support": support,
                    "support_floor": protected_slice_floor,
                    "reason": "protected slice support below floor or missing class",
                }
                continue
            seeded = protected_seed_results.get(key, [])
            positive_member = member_scores[m_mask]
            positive_nonmember = nonmember_scores[n_mask]
            if not len(positive_member) or not len(positive_nonmember):
                slices[key] = {
                    "status": "indeterminate",
                    "support": support,
                    "support_floor": protected_slice_floor,
                    "reason": "protected slice has no one-vs-rest complement",
                }
                continue
            slice_auc = auc(positive_member, positive_nonmember)
            effective = max(slice_auc, 1.0 - slice_auc) - 0.5
            all_scores = np.r_[member_scores, nonmember_scores]
            all_protected = np.r_[member_protected.to_numpy(), nonmember_protected.to_numpy()]
            positive = all_protected == value if value != "<MISSING>" else pd.isna(all_protected)
            complement_auc = (
                auc(all_scores[positive], all_scores[~positive])
                if positive.any() and (~positive).any()
                else math.nan
            )
            complement_effective = (
                max(complement_auc, 1.0 - complement_auc) - 0.5
                if math.isfinite(complement_auc)
                else math.nan
            )
            slices[key] = {
                "status": "succeeded",
                "auc": slice_auc,
                "member_support": int(m_mask.sum()),
                "nonmember_support": int(n_mask.sum()),
                "support": support,
                "support_floor": protected_slice_floor,
                "seed_results": seeded,
                "ovr_auc": complement_auc,
                "ovr_effective_auc_advantage": complement_effective,
                "ovr_complement_support": int((~positive).sum()),
                "effective_auc_advantage": effective,
                "ovr_gap": effective - complement_effective,
            }
        result["protected_slices"] = slices
        gaps: list[float] = []
        for item in slices.values():
            gap = item.get("ovr_gap")
            if item.get("status") == "succeeded" and isinstance(gap, (int, float)):
                gaps.append(float(gap))
        result["protected_ovr_gap"] = max(gaps, default=math.nan)
    return result


def attribute_disclosure(
    synthetic: pd.DataFrame,
    final_holdout: pd.DataFrame,
    quasi_identifiers: Sequence[str],
    sensitive_fields: Sequence[str],
    *,
    sensitive_types: Mapping[str, str],
    random_state: int = 0,
) -> dict[str, Any]:
    """Fit synthetic-only XGBoost attackers and score final holdout only.

    No sklearn fallback is allowed: missing XGBoost is an explicit
    indeterminate result, preserving protocol provenance rather than changing
    attacker definition.
    """
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError:
        return {"status": "indeterminate", "reason": "xgboost is required; fallback rejected"}
    if not quasi_identifiers:
        raise ValueError("explicit non-empty quasi-identifiers are required")
    missing = (set(quasi_identifiers) | set(sensitive_fields)) - set(synthetic.columns) | (
        set(quasi_identifiers) | set(sensitive_fields)
    ) - set(final_holdout.columns)
    if missing:
        raise KeyError(f"missing disclosure columns: {sorted(missing)}")
    results = {}
    x_train = pd.get_dummies(synthetic[list(quasi_identifiers)], dummy_na=True)
    x_test = pd.get_dummies(final_holdout[list(quasi_identifiers)], dummy_na=True).reindex(
        columns=x_train.columns, fill_value=0
    )
    for field in sensitive_fields:
        kind = sensitive_types.get(field)
        if kind == "categorical":
            try:
                model = XGBClassifier(
                    n_estimators=50, max_depth=3, random_state=random_state, eval_metric="logloss"
                )
                actual = final_holdout[field].to_numpy()

                def stable(value: Any) -> str:
                    return "<MISSING>" if pd.isna(value) else f"{type(value).__name__}:{value}"

                labels = sorted(
                    set(stable(value) for value in synthetic[field])
                    | set(stable(value) for value in actual)
                )
                encoding = {value: index for index, value in enumerate(labels)}
                encoded_training = np.array([encoding[stable(value)] for value in synthetic[field]])
                encoded_actual = np.array([encoding[stable(value)] for value in actual])
                model.fit(x_train, encoded_training)
                encoded_prediction = np.asarray(model.predict(x_test), dtype=int)
                recalls = [
                    float(np.mean(encoded_prediction[encoded_actual == index] == index))
                    for index in range(len(labels))
                    if np.any(encoded_actual == index)
                ]
                balanced_accuracy = float(np.mean(recalls)) if recalls else math.nan
                final_labels = {stable(value) for value in actual}
                baseline = 1.0 / len(final_labels) if final_labels else math.nan
                results[field] = {
                    "status": "succeeded" if math.isfinite(balanced_accuracy) else "indeterminate",
                    "risk": max(0.0, balanced_accuracy - baseline)
                    if math.isfinite(balanced_accuracy)
                    else math.nan,
                    "balanced_accuracy": balanced_accuracy,
                    "baseline_balanced_accuracy": baseline,
                    "support": len(final_holdout),
                }
            except (ValueError, TypeError, RuntimeError) as exc:
                results[field] = {
                    "status": "indeterminate",
                    "risk": math.nan,
                    "reason": "attribute_disclosure_categorical_error",
                    "error_type": type(exc).__name__,
                    "support": len(final_holdout),
                }
        elif kind == "continuous":
            try:
                model = XGBRegressor(
                    n_estimators=50,
                    max_depth=3,
                    random_state=random_state,
                    objective="reg:squarederror",
                )
                model.fit(x_train, synthetic[field])
                prediction = model.predict(x_test)
                actual = final_holdout[field].to_numpy(dtype=float)
                prediction = np.asarray(prediction, dtype=float)
                error = float(np.mean(np.abs(prediction - actual)))
                baseline = float(np.mean(np.abs(actual - np.median(actual))))
                valid_baseline = math.isfinite(baseline) and baseline > 0 and math.isfinite(error)
                results[field] = {
                    "status": "succeeded" if valid_baseline else "indeterminate",
                    "risk": error / baseline if valid_baseline else math.nan,
                    "relative_mae": error / baseline if valid_baseline else math.nan,
                    "mae": error,
                    "baseline_mae": baseline,
                    "support": len(final_holdout),
                    "reason": None if valid_baseline else "zero or non-finite baseline/error",
                }
            except (ValueError, TypeError, RuntimeError, FloatingPointError) as exc:
                results[field] = {
                    "status": "indeterminate",
                    "risk": math.nan,
                    "reason": "attribute_disclosure_continuous_error",
                    "error_type": type(exc).__name__,
                    "support": len(final_holdout),
                }
        else:
            raise ValueError(f"sensitive_types[{field!r}] must be categorical or continuous")
    valid_results: dict[str, dict[str, Any]] = {}
    for name, value in results.items():
        risk = value.get("risk")
        if (
            value.get("status", "succeeded") == "succeeded"
            and isinstance(risk, (int, float))
            and math.isfinite(risk)
        ):
            valid_results[name] = value
    invalid_targets = [name for name, value in results.items() if name not in valid_results]
    worst = (
        max(valid_results, key=lambda name: float(valid_results[name]["risk"]))
        if valid_results
        else None
    )
    overall_valid = bool(results) and not invalid_targets
    overall_state = (
        "succeeded"
        if overall_valid
        else ("all_targets_invalid" if not valid_results else "mixed_invalid_targets")
    )
    return {
        "status": "succeeded" if overall_valid else "indeterminate",
        "overall_state": overall_state,
        "invalid_targets": invalid_targets,
        "overall_invalid_targets": invalid_targets,
        "reason": None
        if overall_valid
        else "no required target has finite valid risk"
        if not valid_results
        else "one or more required sensitive targets are invalid",
        "worst_sensitive_target": worst if worst is not None else None,
        "worst_sensitive_risk": valid_results[worst]["risk"] if worst is not None else math.nan,
        "attacker_training": "synthetic_only",
        "scoring_population": "final_holdout",
        "fallback": "rejected",
    }
