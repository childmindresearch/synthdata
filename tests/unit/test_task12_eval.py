"""Unit tests for Task 12 custom evaluation ownership."""

import pytest

from synthdata.evaluation.task12_eval import TASK12_CUSTOM_KEYS, _task12_record

pytestmark = pytest.mark.unit


def _metadata(*, producer="task12-test", fit_roles=("train", "tuning")):
    return {
        "producer": producer,
        "protocol_version": "task12-evaluation-v1",
        "seed": 0,
        "release_transform_digest": "release-digest",
        "common_protocol_digest": "common-digest",
        "role_hashes": {"train": "train", "tuning": "tuning"},
        "fit_roles": list(fit_roles),
    }


def test_task12_record_preserves_producer_metadata_and_final_audit_pass():
    record = _task12_record(
        "model",
        "equalized_odds.final.v1",
        0.8,
        role_hashes={"train": "train", "tuning": "tuning"},
        evaluation_role="final_holdout",
        metadata=_metadata(),
    )

    assert record.execution_pass == "final_audit"
    assert record.result_metadata["producer"] == "task12-test"
    assert record.fit_roles == ("train", "tuning")


def test_task12_record_blocks_missing_metadata_and_wrong_fit_roles():
    record = _task12_record(
        "model",
        TASK12_CUSTOM_KEYS[0],
        1.0,
        role_hashes={},
        evaluation_role="tuning",
        metadata={"fit_roles": ["train", "tuning"]},
    )

    assert record.source_metadata["status"] == "blocked"
    assert record.raw_value is None
