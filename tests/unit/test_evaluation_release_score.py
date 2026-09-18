"""Focused tests for release-score normalization and indeterminate states."""

import math

import pytest

from synthdata.evaluation import release_evidence_eval
from synthdata.evaluation.release_score import compute_release_score, normalize_component

pytestmark = pytest.mark.unit


def test_normalize_component_rejects_non_finite_values():
    assert normalize_component({"score": math.nan}) is None
    assert normalize_component({"score": math.inf}) is None


def test_compute_release_score_preserves_indeterminate_state():
    score = compute_release_score(
        utility={"tstr": 0.8},
        privacy={"k": 0.8},
        fairness={"representation": 0.8},
    )

    assert score["status"] == "indeterminate"
    assert score["score"] is None
    assert set(score["indeterminate_dimensions"]) == {"utility", "privacy", "fairness"}


def _contains(value, sentinel):
    if isinstance(value, dict):
        return any(
            _contains(key, sentinel) or _contains(item, sentinel) for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return any(_contains(item, sentinel) for item in value)
    return value == sentinel


@pytest.mark.parametrize("status", ["blocked", "indeterminate"])
def test_release_evidence_blocks_and_sanitizes_nested_untrusted_reasons(status):
    sentinel = "SECRET /private/input.csv traceback TOKEN"
    valid_hash = "a" * 64
    record = release_evidence_eval._release_evidence_record(
        "model",
        "release_privacy.v1",
        0.75,
        role_hashes={"train": valid_hash, "tuning": sentinel, "sentinel": sentinel},
        evaluation_role="tuning",
        status=status,
        reason=sentinel,
        metadata={
            "producer": "release_evidence_privacy",
            "seed": 0,
            "common_protocol_digest": "b" * 64,
            "fit_roles": ["train"],
            "invalid_reasons": [{"nested": [sentinel]}],
            "release_evidence_reason": {"detail": sentinel},
            "error_code": "arbitrary caller error code",
            "error_type": "arbitrary caller error type",
            "status": "succeeded",
            "protocol_version": {"nested": sentinel},
            "release_transform_digest": {"nested": sentinel},
            "role_hashes": {"train": sentinel},
            "error": {"nested": [sentinel]},
            "traceback": sentinel,
            "path": sentinel,
            "nested": {"producer_text": sentinel, "path": "/private/input.csv"},
        },
    )

    assert record.raw_value is None
    assert record.role_hashes == {"train": valid_hash}
    assert sentinel not in record.role_hashes.values()
    assert record.error == "release_evidence_release_or_representation_error"
    assert record.source_metadata["status"] == status
    assert record.result_metadata["status"] == status
    for container in (
        record.error,
        record.source_metadata,
        record.result_metadata,
        record.provenance,
    ):
        assert not _contains(container, sentinel)
    assert "error_code" not in record.source_metadata
    assert "error_type" not in record.source_metadata


def test_non_success_retains_only_valid_structural_metadata():
    record = release_evidence_eval._release_evidence_record(
        "model",
        "release_privacy.v1",
        0.75,
        role_hashes={"train": "c" * 64},
        evaluation_role="tuning",
        status="blocked",
        metadata={
            "producer": "release_evidence_privacy",
            "protocol_version": "release-evidence-v2",
            "producer_protocol_version": "unknown",
            "seed": 7,
            "release_transform_digest": "d" * 64,
            "common_protocol_digest": "e" * 64,
            "role_hashes": {"train": "f" * 64, "unknown": "g" * 64},
            "fit_roles": ["train"],
            "execution_pass": "main",
            "support": {"support_contract": "declared_support_v1", "synthetic": 3},
            "error_code": "evaluator_exception",
            "error_type": "ValueError",
        },
    )

    assert record.raw_value is None
    assert record.source_metadata["protocol_version"] == "release-evidence-v2"
    assert record.source_metadata["release_transform_digest"] == "d" * 64
    assert record.source_metadata["role_hashes"] == {"train": "f" * 64}
    assert record.source_metadata["support"] == {
        "support_contract": "declared_support_v1",
        "synthetic": 3,
    }
    assert "producer_protocol_version" not in record.source_metadata


def test_successful_release_evidence_preserves_producer_metadata():
    metadata = {
        "producer": "release_evidence_privacy",
        "protocol_version": "release-evidence-v2",
        "seed": 7,
        "release_transform_digest": "a" * 64,
        "common_protocol_digest": "b" * 64,
        "role_hashes": {"train": "c" * 64},
        "fit_roles": ["train"],
        "support": {"support_contract": "declared_support_v1"},
        "producer_detail": {"valid": [1, "producer-value"]},
    }
    record = release_evidence_eval._release_evidence_record(
        "model",
        "release_privacy.v1",
        0.75,
        role_hashes={"train": "c" * 64},
        evaluation_role="tuning",
        metadata=metadata,
    )

    assert record.raw_value == 0.75
    assert record.role_hashes == metadata["role_hashes"]
    for container in (record.source_metadata, record.result_metadata, record.provenance):
        assert container["producer_detail"] == metadata["producer_detail"]
