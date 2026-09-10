"""Tests for invariant final release-score evidence."""

import pytest

from synthdata.evaluation.release_score import compute_release_score, normalize_component

pytestmark = pytest.mark.unit


def _evidence():
    return {
        "utility": {"tstr": 0.8, "mmd": 0.2, "jsd": 0.1},
        "privacy": {
            "k": 0.9,
            "l": 0.8,
            "dcr": {"status": "succeeded", "score": 0.7, "ratio": 0.01},
            "epsilon": {"positive_excess": 0.02},
            "mia": {"effective_auc_advantage": 0.03},
            "attribute": {"risk": 0.04},
        },
        "fairness": {
            "representation": 0.7,
            "eo": {"gap": 0.02},
            "worst_log_disparity": {"value": 0.1},
        },
    }


def test_fixed_formula_and_decomposition():
    result = compute_release_score(**_evidence())
    utility = (0.8 + 0.8 + 0.9) / 3
    privacy = (min(0.9, 0.8, 0.7, 0.8) * 0.7 * 0.8) ** (1 / 3)
    fairness = 0.4 * 0.7 + 0.4 * 0.8 + 0.2 * (1 - 0.1 / 0.69314718056)
    assert result["dimensions"]["utility"]["score"] == pytest.approx(utility)
    assert result["dimensions"]["privacy"]["score"] == pytest.approx(privacy)
    assert result["dimensions"]["fairness"]["score"] == pytest.approx(fairness)
    assert result["score"] == pytest.approx(0.45 * utility + 0.30 * privacy + 0.25 * fairness)


def test_direct_scores_are_not_double_normalized():
    assert normalize_component({"score": 0.75}, kind="direct") == 0.75
    assert normalize_component({"score": 0.75}, kind="distance", anchor=0.1) == 0.75


def test_normalized_metric_aliases_are_consumed_directly():
    result = compute_release_score(
        utility={"S_TSTR": 0.81, "S_MMD": 0.72, "S_JSD": 0.63},
        privacy={
            "S_k": 0.91,
            "S_l": 0.82,
            "S_DCR": 0.73,
            "S_epsilon": 0.64,
            "S_MIA": 0.55,
            "S_attribute": 0.46,
        },
        fairness={
            "S_representation": 0.87,
            "S_EO": 0.78,
            "S_worst_log_disparity": 0.69,
        },
    )

    assert result["dimensions"]["utility"]["score"] == pytest.approx((0.81 + 0.72 + 0.63) / 3)
    assert result["dimensions"]["privacy"]["identity"] == pytest.approx(0.64)
    assert result["dimensions"]["privacy"]["score"] == pytest.approx(
        (0.64 * 0.55 * 0.46) ** (1 / 3)
    )
    assert result["dimensions"]["fairness"]["score"] == pytest.approx(
        0.4 * 0.87 + 0.4 * 0.78 + 0.2 * 0.69
    )


def test_normalized_aliases_inside_evidence_records_are_not_retransformed():
    result = compute_release_score(
        utility={"S_MMD": {"score": 0.75}, "S_TSTR": 0.8, "S_JSD": 0.7},
        privacy={
            "S_k": 0.9,
            "S_l": 0.9,
            "S_DCR": {"score": 0.8},
            "S_epsilon": {"score": 0.7},
            "S_MIA": {"score": 0.6},
            "S_attribute": {"score": 0.5},
        },
        fairness={
            "S_representation": 0.9,
            "S_EO": {"score": 0.8},
            "S_worst_log_disparity": {"score": 0.7},
        },
    )

    assert result["dimensions"]["utility"]["components"]["mmd"]["score"] == 0.75
    assert result["dimensions"]["privacy"]["components"]["epsilon"]["score"] == 0.7
    assert result["dimensions"]["fairness"]["components"]["eo"]["score"] == 0.8


def test_missing_component_propagates_without_reweighting():
    evidence = _evidence()
    del evidence["privacy"]["mia"]
    result = compute_release_score(**evidence)
    assert result["dimensions"]["privacy"]["score"] is None
    assert result["score"] is None
    assert result["status"] == "indeterminate"
    assert "privacy" in result["indeterminate_dimensions"]


def test_fixed_anchors_make_score_independent_of_cohort():
    first = compute_release_score(**_evidence())
    second = compute_release_score(**_evidence(), provenance={"candidate_count": 99})
    assert first["score"] == second["score"]
    assert first["audit_only"] is True
