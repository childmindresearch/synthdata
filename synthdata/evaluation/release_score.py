"""Invariant, audit-only release score calculations.

This module deliberately contains no model comparison or ranking logic.  It
normalizes one model's already validated evidence against fixed anchors and
applies the release-score contract.  Results are plain dictionaries so they
can be embedded in JSON evidence without a custom encoder.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

DEFAULT_ANCHORS: dict[str, float] = {
    "mmd": 1.0,
    "epsilon_excess": 0.10,
    "mia_advantage": 0.10,
    "attribute_disclosure": 0.20,
    "equalized_odds_gap": 0.10,
    "worst_absolute_log_disparity": 0.69314718056,
}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _raw(evidence: Any, *keys: str) -> Any:
    if isinstance(evidence, Mapping):
        status = evidence.get("status")
        if status not in (None, "succeeded", "success", "ok", "valid"):
            return None
        for key in keys:
            if key in evidence:
                return evidence[key]
        return None
    return evidence


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def normalize_component(
    evidence: Any,
    *,
    kind: str = "direct",
    anchor: float | None = None,
) -> float | None:
    """Return fixed-contract score for one evidence item.

    ``direct`` values are already scores and are not normalized again.  Other
    kinds interpret evidence as a non-negative loss/risk and map zero to one
    using its configured fixed anchor.  Invalid, missing, or failed evidence
    returns ``None`` rather than being silently omitted.
    """
    aliases = {
        "direct": ("score", "safety_score", "value"),
        "risk": ("risk", "positive_excess", "effective_auc_advantage", "value", "score"),
        "gap": ("gap", "value", "score"),
        "distance": ("score", "value", "ratio"),
        "jsd": ("distance", "jsd", "value", "score"),
    }
    value = _raw(evidence, *aliases.get(kind, aliases["direct"]))
    if not _finite(value):
        return None
    if (
        kind == "direct"
        or kind == "distance"
        and isinstance(evidence, Mapping)
        and "score" in evidence
    ):
        return _clip(float(value))
    if kind == "jsd":
        return _clip(1.0 - float(value))
    if anchor is None or not _finite(anchor) or float(anchor) <= 0:
        return None
    return _clip(1.0 - float(value) / float(anchor))


def _component_record(evidence: Any, score: float | None) -> dict[str, Any]:
    return {
        "score": score,
        "status": "succeeded" if score is not None else "indeterminate",
        "evidence": evidence,
    }


def _get(mapping: Mapping[str, Any], name: str) -> Any:
    aliases = {
        "tstr": ("tstr", "S_TSTR", "tstr_macro_f1.v1"),
        "mmd": ("mmd", "S_MMD", "mixed_mmd.v1"),
        "jsd": ("jsd", "S_JSD", "elastic_net_jsd.v1"),
        "k": ("k", "S_k"),
        "l": ("l", "S_l"),
        "dcr": ("dcr", "S_DCR"),
        "epsilon": ("epsilon", "S_epsilon"),
        "mia": ("mia", "S_MIA"),
        "attribute": ("attribute", "S_attribute"),
        "representation": ("representation", "S_representation"),
        "eo": ("eo", "S_EO", "equalized_odds"),
        "worst_log_disparity": (
            "worst_log_disparity",
            "S_worst_log_disparity",
            "worst_absolute_log_disparity",
        ),
    }
    for key in aliases[name]:
        if key in mapping:
            return mapping[key]
    return None


def _get_component(mapping: Mapping[str, Any], name: str) -> tuple[Any, bool]:
    """Get component evidence and whether its alias is already normalized."""
    aliases = {
        "tstr": (("S_TSTR",), ("tstr", "tstr_macro_f1.v1")),
        "mmd": (("S_MMD",), ("mmd", "mixed_mmd.v1")),
        "jsd": (("S_JSD",), ("jsd", "elastic_net_jsd.v1")),
        "k": (("S_k",), ("k",)),
        "l": (("S_l",), ("l",)),
        "dcr": (("S_DCR",), ("dcr",)),
        "epsilon": (("S_epsilon",), ("epsilon",)),
        "mia": (("S_MIA",), ("mia",)),
        "attribute": (("S_attribute",), ("attribute",)),
        "representation": (("S_representation",), ("representation",)),
        "eo": (("S_EO",), ("eo", "equalized_odds")),
        "worst_log_disparity": (
            ("S_worst_log_disparity",),
            ("worst_log_disparity", "worst_absolute_log_disparity"),
        ),
    }
    direct_aliases, raw_aliases = aliases[name]
    for key in direct_aliases:
        if key in mapping:
            return mapping[key], True
    for key in raw_aliases:
        if key in mapping:
            return mapping[key], False
    return None, False


def _score(
    mapping: Mapping[str, Any],
    name: str,
    *,
    kind: str = "direct",
    anchor: float | None = None,
) -> float | None:
    evidence, already_normalized = _get_component(mapping, name)
    return normalize_component(
        evidence,
        kind="direct" if already_normalized else kind,
        anchor=anchor,
    )


def compute_release_score(
    *,
    utility: Mapping[str, Any],
    privacy: Mapping[str, Any],
    fairness: Mapping[str, Any],
    anchors: Mapping[str, float] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute invariant utility, privacy, fairness, and final audit score.

    Missing or invalid required evidence makes its dimension, and therefore
    ``R_final``, indeterminate.  Dimensions are never reweighted.
    """
    fixed = dict(DEFAULT_ANCHORS)
    fixed.update(anchors or {})
    u_scores = {
        "tstr": _score(utility, "tstr"),
        "mmd": _score(utility, "mmd", kind="risk", anchor=fixed["mmd"]),
        "jsd": _score(utility, "jsd", kind="jsd"),
    }
    p_scores = {
        "k": _score(privacy, "k"),
        "l": _score(privacy, "l"),
        "dcr": _score(privacy, "dcr", kind="distance"),
        "epsilon": _score(privacy, "epsilon", kind="risk", anchor=fixed["epsilon_excess"]),
        "mia": _score(privacy, "mia", kind="risk", anchor=fixed["mia_advantage"]),
        "attribute": _score(
            privacy, "attribute", kind="risk", anchor=fixed["attribute_disclosure"]
        ),
    }
    f_scores = {
        "representation": _score(fairness, "representation"),
        "eo": _score(fairness, "eo", kind="gap", anchor=fixed["equalized_odds_gap"]),
        "worst_log_disparity": _score(
            fairness,
            "worst_log_disparity",
            kind="gap",
            anchor=fixed["worst_absolute_log_disparity"],
        ),
    }
    utility_value = (
        sum(cast(float, v) for v in u_scores.values()) / 3
        if all(v is not None for v in u_scores.values())
        else None
    )
    identity_values = [p_scores[key] for key in ("k", "l", "dcr", "epsilon")]
    identity = (
        min(cast(float, v) for v in identity_values)
        if all(v is not None for v in identity_values)
        else None
    )
    privacy_value = (
        (identity * p_scores["mia"] * p_scores["attribute"]) ** (1 / 3)
        if identity is not None
        and p_scores["mia"] is not None
        and p_scores["attribute"] is not None
        else None
    )
    fairness_value = (
        0.40 * cast(float, f_scores["representation"])
        + 0.40 * cast(float, f_scores["eo"])
        + 0.20 * cast(float, f_scores["worst_log_disparity"])
        if all(v is not None for v in f_scores.values())
        else None
    )
    final = (
        0.45 * utility_value + 0.30 * cast(float, privacy_value) + 0.25 * fairness_value
        if utility_value is not None and privacy_value is not None and fairness_value is not None
        else None
    )
    reasons = [
        name
        for name, value in (
            ("utility", utility_value),
            ("privacy", privacy_value),
            ("fairness", fairness_value),
        )
        if value is None
    ]
    return {
        "status": "succeeded" if final is not None else "indeterminate",
        "score": final,
        "dimensions": {
            "utility": {
                "score": utility_value,
                "components": {
                    k: _component_record(_get(utility, k), v) for k, v in u_scores.items()
                },
            },
            "privacy": {
                "score": privacy_value,
                "identity": identity,
                "components": {
                    k: _component_record(_get(privacy, k), v) for k, v in p_scores.items()
                },
            },
            "fairness": {
                "score": fairness_value,
                "components": {
                    k: _component_record(_get(fairness, k), v) for k, v in f_scores.items()
                },
            },
        },
        "formula": "R_final=0.45*U+0.30*P+0.25*F; U=(S_TSTR+S_MMD+S_JSD)/3; I=min(S_k,S_l,S_DCR,S_epsilon); P=(I*S_MIA*S_attribute)^(1/3); F=0.40*S_representation+0.40*S_EO+0.20*S_worst_log_disparity",
        "weights": {"utility": 0.45, "privacy": 0.30, "fairness": 0.25},
        "anchors": fixed,
        "indeterminate_dimensions": reasons,
        "provenance": dict(provenance or {}),
        "audit_only": True,
    }


release_score = compute_release_score
final_release_score = compute_release_score

__all__ = [
    "DEFAULT_ANCHORS",
    "normalize_component",
    "compute_release_score",
    "release_score",
    "final_release_score",
]
