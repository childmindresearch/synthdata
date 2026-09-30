"""Focused tests for release-score normalization and indeterminate states."""

import math

import pandas as pd
import pytest

from synthdata.evaluation import release_evidence_eval
from synthdata.evaluation.release import transform_release_roles
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


@pytest.mark.parametrize(
    ("synthetic_rows", "reference_rows", "floor", "succeeded"),
    [
        (19, 20, 20, False),
        (20, 19, 20, False),
        (20, 20, 20, True),
        (20, 20, 21, False),
        (21, 21, 21, True),
    ],
)
def test_release_producer_forwards_population_and_protected_floors(
    monkeypatch, synthetic_rows, reference_rows, floor, succeeded
):
    synthetic, roles, _ = transform_release_roles(
        pd.DataFrame(
            {"qi": [0] * synthetic_rows, "sensitive": [i % 2 for i in range(synthetic_rows)]}
        ),
        {
            "tuning": pd.DataFrame(
                {"qi": [0] * reference_rows, "sensitive": [i % 2 for i in range(reference_rows)]}
            )
        },
    )
    calls = []
    producer = release_evidence_eval.release_privacy_evidence

    def release(*args, **kwargs):
        calls.append(kwargs)
        return producer(*args, **kwargs)

    monkeypatch.setattr(release_evidence_eval, "release_privacy_evidence", release)
    monkeypatch.setattr(
        release_evidence_eval.custom_eval,
        "run_log_disparity_evaluation",
        lambda *args, **kwargs: {},
    )

    class Dataset:
        role_metadata = {}

        def role_frame(self, role, *, imputed):
            return roles[role]

    record = release_evidence_eval.run_release_evidence_evaluation(
        {"model": synthetic},
        Dataset(),
        evaluation_role="tuning",
        generalization=None,
        quasi_identifiers=["qi"],
        sensitive_fields=["sensitive"],
        protected_columns=[],
        role_hashes={},
        role_population_floor=floor,
        protected_slice_floor=3,
        release_form_inputs=(synthetic, roles, {}),
    )["model"][0]

    assert calls[0]["role_population_floor"] == floor
    assert calls[0]["protected_slice_floor"] == 3
    assert (record.source_metadata["status"] == "succeeded") is succeeded
    assert record.support["role_population_floor"] == floor
    assert record.support["protected_slice_floor"] == 3
    assert record.support["protected_slices"] == {"state": "not_applicable", "floor": 3}
    assert record.support["synthetic"] == synthetic_rows
    assert record.support["reference"] == reference_rows
    if succeeded:
        assert record.raw_value == 0.0
        assert (
            record.result_metadata["release_support"]["roles"]["reference"]["population_floor"]
            == floor
        )


def test_release_failure_retains_bounded_floor_context_without_exception_text(monkeypatch):
    sentinel = "SECRET /private/rows.csv traceback"
    frame = pd.DataFrame({"value": [1]})

    class Dataset:
        role_metadata = {}

        def role_frame(self, role, *, imputed):
            return frame

    def release(*args, **kwargs):
        raise ValueError(sentinel)

    monkeypatch.setattr(release_evidence_eval, "release_privacy_evidence", release)
    records = release_evidence_eval.run_release_evidence_evaluation(
        {"model": frame},
        Dataset(),
        evaluation_role="final_holdout",
        generalization=None,
        quasi_identifiers=[],
        sensitive_fields=[],
        protected_columns=[],
        role_hashes={},
        role_population_floor=24,
        protected_slice_floor=2,
        release_form_inputs=(frame, {"final_holdout": frame}, {}),
    )["model"]
    for record in records:
        assert record.raw_value is None
        assert record.support["role_population_floor"] == 24
        assert record.support["protected_slice_floor"] == 2
        assert sentinel not in repr(record)
    assert records[0].support["protected_slices"]["state"] == "not_applicable"


@pytest.mark.parametrize("value", [True, 1.5, -1, "SECRET", 1_000_000_001])
def test_blocked_floor_metadata_rejects_unbounded_or_non_integer_values(value):
    record = release_evidence_eval._release_evidence_record(
        "model",
        "release_privacy.v1",
        None,
        role_hashes={},
        evaluation_role="tuning",
        status="blocked",
        metadata={
            "support": {
                "role_population_floor": value,
                "protected_slice_floor": value,
                "protected_slices": {"state": "not_applicable", "floor": value, "path": "SECRET"},
            }
        },
    )
    assert "role_population_floor" not in record.support
    assert "protected_slice_floor" not in record.support
    assert "protected_slices" not in record.support


def test_final_release_fairness_preserves_effective_tstr_floor(monkeypatch):
    from synthdata.evaluation.tstr import run_tstr_evaluation

    frame = pd.DataFrame({"x": [0, 1, 0, 1], "y": [0, 1, 0, 1], "group": ["a", "a", "b", "b"]})
    synthetic, roles, _ = transform_release_roles(frame, {"final_holdout": frame})
    hashes = {"train": "a" * 64, "tuning": "b" * 64, "final_holdout": "c" * 64}

    class Model:
        def fit(self, x, y):
            return self

        def predict(self, x):
            return pd.Series([0, 1, 0, 1]).to_numpy()

        def predict_proba(self, x):
            return pd.DataFrame([[1.0, 0.0], [0.0, 1.0]] * 2).to_numpy()

    class Dataset:
        role_metadata = {}

        def role_frame(self, role, *, imputed):
            raise AssertionError("final release evidence must not read raw holdout")

    monkeypatch.setattr("synthdata.evaluation.tstr._xgb", lambda seed, classes: Model())
    monkeypatch.setattr(
        release_evidence_eval.custom_eval,
        "run_log_disparity_evaluation",
        lambda *args, **kwargs: {},
    )
    for floor in (1, 2):
        tstr = run_tstr_evaluation(
            synthetic,
            roles["final_holdout"],
            target_column="y",
            evaluation_role="final_holdout",
            protected_columns=["group"],
            role_hashes=hashes,
            protected_slice_floor=floor,
        )
        record = release_evidence_eval.run_release_evidence_evaluation(
            {"model": synthetic},
            Dataset(),
            evaluation_role="final_holdout",
            generalization=None,
            quasi_identifiers=[],
            sensitive_fields=[],
            protected_columns=["group"],
            role_hashes=hashes,
            tstr_results={"model": tstr},
            release_form_inputs=(synthetic, roles, {}),
            protected_slice_floor=floor,
        )["model"][-1]
        assert record.support["protected_slice_floor"] == floor
        if floor == 1:
            assert record.raw_value == 0.0
            assert record.support["state"] == "complete"
            assert record.result_metadata["support_policy"]["protected_slice_floor"] == floor
        else:
            assert record.raw_value is None
            assert record.source_metadata["status"] == "blocked"
            assert record.support["state"] == "indeterminate"
