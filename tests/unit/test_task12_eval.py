"""Unit tests for Task 12 custom evaluation ownership."""

import pandas as pd
import pytest

from synthdata.evaluation import task12_eval
from synthdata.evaluation.task12_eval import (
    TASK12_CUSTOM_KEYS,
    _task12_record,
    run_task12_custom_evaluation,
)

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


def test_task12_forwards_configured_identity_column_to_release(monkeypatch):
    frame = pd.DataFrame({"custom_patient": ["p1"], "value": [1]})
    calls = []

    class Dataset:
        role_metadata = {"identity": {"identity_column": "custom_patient"}}

        def role_frame(self, _role, *, imputed):
            assert imputed is False
            return frame.copy()

    def release(*args, **kwargs):
        calls.append(kwargs)
        return {
            "status": "indeterminate",
            "invalid_reasons": ["patient ID cannot be present in release frames"],
        }

    monkeypatch.setattr(task12_eval, "release_privacy_evidence", release)
    monkeypatch.setattr(
        task12_eval.custom_eval,
        "run_log_disparity_evaluation",
        lambda *args, **kwargs: {},
    )

    result = run_task12_custom_evaluation(
        {"model": frame.copy()},
        Dataset(),
        evaluation_role="tuning",
        generalization=None,
        quasi_identifiers=[],
        sensitive_fields=[],
        protected_columns=[],
        role_hashes={},
        release_form_inputs=(frame.copy(), {"tuning": frame.copy()}, {}),
    )

    assert calls[0]["patient_id_column"] == "custom_patient"
    release_record = result["model"][0]
    assert release_record.source_metadata["status"] == "blocked"
    assert release_record.raw_value is None
    assert release_record.error == "task12_release_or_representation_error"


def test_task12_does_not_persist_release_exception_body(monkeypatch):
    sentinel = "SENTINEL_RAW_EXCEPTION_BODY /private/patient/path"
    frame = pd.DataFrame({"value": [1]})

    class Dataset:
        role_metadata = {"identity": {}}

        def role_frame(self, _role, *, imputed):
            return frame.copy()

    def release(*args, **kwargs):
        raise ValueError(sentinel)

    monkeypatch.setattr(task12_eval, "release_privacy_evidence", release)
    result = run_task12_custom_evaluation(
        {"model": frame.copy()},
        Dataset(),
        evaluation_role="tuning",
        generalization=None,
        quasi_identifiers=[],
        sensitive_fields=[],
        protected_columns=[],
        role_hashes={},
        release_form_inputs=(frame.copy(), {"tuning": frame.copy()}, {}),
    )

    for record in result["model"][:2]:
        assert record.source_metadata["status"] == "blocked"
        assert record.raw_value is None
        assert record.error == "task12_release_or_representation_error"
        assert record.result_metadata["error_type"] == "ValueError"
        assert sentinel not in repr(record)


def test_task12_final_uses_released_reference_without_reading_raw_role(monkeypatch):
    released = pd.DataFrame({"value": [2]})
    synthetic = released.copy()
    synthetic.attrs["release_provenance"] = {}
    calls = []

    class Dataset:
        role_metadata = {"identity": {}}

        def role_frame(self, _role, *, imputed):
            raise AssertionError("raw final_holdout must not be requested")

    monkeypatch.setattr(
        task12_eval,
        "release_privacy_evidence",
        lambda *args, **kwargs: {"status": "succeeded", "invalid_reasons": []},
    )

    def evaluate(*args, **kwargs):
        calls.append(kwargs)
        return {"model": {"summary_stats": {"representation_safety": 0.2}}}

    monkeypatch.setattr(task12_eval.custom_eval, "run_log_disparity_evaluation", evaluate)

    result = run_task12_custom_evaluation(
        {"model": synthetic},
        Dataset(),
        evaluation_role="final_holdout",
        generalization=None,
        quasi_identifiers=[],
        sensitive_fields=[],
        protected_columns=[],
        role_hashes={},
        release_form_inputs=(synthetic, {"final_holdout": released}, {}),
    )

    assert calls[0]["reference_frame"] is released
    assert result["model"][1].source_metadata["status"] == "succeeded"


def test_task12_final_missing_released_reference_blocks_without_raw_fallback(monkeypatch):
    frame = pd.DataFrame({"value": [1]})

    class Dataset:
        role_metadata = {"identity": {}}

        def role_frame(self, _role, *, imputed):
            raise AssertionError("raw final_holdout must not be requested")

    result = run_task12_custom_evaluation(
        {"model": frame},
        Dataset(),
        evaluation_role="final_holdout",
        generalization=None,
        quasi_identifiers=[],
        sensitive_fields=[],
        protected_columns=[],
        role_hashes={},
        release_form_inputs=(frame, {}, {}),
    )

    assert [record.source_metadata["status"] for record in result["model"][:2]] == [
        "blocked",
        "blocked",
    ]
