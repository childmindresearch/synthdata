"""Unit tests for synthdata.evaluation.custom_eval."""

import hashlib
import json

import pandas as pd
import pytest

from synthdata.config import FrameworkSelectionConfig, LogDisparityConfig
from synthdata.evaluation.custom_eval import (
    build_log_disparity_summary_table,
    run_log_disparity_evaluation,
    validate_log_disparity_results,
)
from synthdata.evaluation.release import transform_release_roles
from synthdata.log_disparity import metric_log_disparity

pytestmark = pytest.mark.unit


@pytest.fixture
def fairness_dataset(make_dataset):
    df = pd.DataFrame(
        {
            "sex": ["M", "F", "M", "F", "M", "F", "M", "F"],
            "age": [20, 30, 40, 50, 25, 35, 45, 55],
            "target": [0, 1, 0, 1, 1, 0, 1, 0],
        }
    )
    dataset = make_dataset(df=df, target_column="target", sensitive_columns=["sex"])
    dataset.protected_columns = ["sex"]
    return dataset


def _mark_release(frame, *, role="release", evaluation_role="tuning"):
    frame = frame.copy()
    source_role = "synthetic" if role == "release" else role
    role_payload = {"role": source_role, "rows": frame.to_dict("records")}
    role_hash = hashlib.sha256(
        json.dumps(role_payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()
    metadata = {
        "protocol_version": "release-privacy-v1",
        "role": source_role,
        "row_count": len(frame),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "generalization": {},
    }
    content_digest = hashlib.sha256(
        json.dumps(
            {**metadata, "role_hash": role_hash}, sort_keys=True, default=str, separators=(",", ":")
        ).encode()
    ).hexdigest()
    frame.attrs["release_provenance"] = {
        "release_form": True,
        "digest": content_digest,
        "common_protocol_digest": "protocol-for-test",
        "role_hash": role_hash,
        "source_role": source_role,
        "protocol_version": "release-privacy-v1",
        "evaluation_role": evaluation_role,
        "role": role,
    }
    return frame


class TestRunLogDisparityEvaluation:
    def test_unexpected_evaluator_exception_becomes_sanitized_failure(
        self, make_canonical_dataset, monkeypatch
    ):
        dataset = make_canonical_dataset()
        real_data = _mark_release(dataset.role_frame("tuning"), role="tuning")
        dataset.roles["tuning"] = real_data
        synthetic = _mark_release(pd.DataFrame({"protected": ["A", "B"], "target": [0, 1]}))

        def explode(**kwargs):
            raise RuntimeError("secret raw evaluator details")

        monkeypatch.setattr(metric_log_disparity, "compute_log_disparity_report", explode)
        reports = run_log_disparity_evaluation(
            {"model": synthetic},
            dataset,
            LogDisparityConfig(protected_columns=["protected"]),
            FrameworkSelectionConfig(enabled=True),
        )

        report = reports["model"]
        assert report["state"] == "failed"
        assert report["error_type"] == "RuntimeError"
        assert "secret raw evaluator details" not in repr(report)

    def test_incomplete_report_is_indeterminate_with_missing_tables(self):
        reports = {"model": {"state": "succeeded", "summary_stats": {}}}

        validations = validate_log_disparity_results(
            reports, ["model"], role_hashes={"train": "hash"}
        )

        assert validations["model"].decision_eligible is False
        assert all(record.raw_value is None for record in validations["model"].records)
        assert all(record.status != "succeeded" for record in validations["model"].records)

    def test_custom_validation_rejects_role_hashes_not_matching_raw_map(self):
        with pytest.raises(ValueError, match="declared raw role map"):
            validate_log_disparity_results(
                {"model": {"state": "indeterminate", "reason": "missing"}},
                ["model"],
                role_hashes={"train": "imputed", "tuning": "raw-tuning"},
                expected_role_hashes={"train": "raw-train", "tuning": "raw-tuning"},
            )

    def test_custom_context_keeps_raw_role_hashes_when_release_binding_is_enriched(self):
        raw_role_hashes = {"train": "raw-train", "tuning": "raw-tuning"}
        validations = validate_log_disparity_results(
            {"model": {"state": "indeterminate", "reason": "missing"}},
            ["model"],
            role_hashes=dict(raw_role_hashes),
            expected_role_hashes=dict(raw_role_hashes),
        )

        release_role_hashes = dict(raw_role_hashes)
        release_role_hashes["__release_transform_digest__"] = "release-digest"

        assert validations["model"].evaluation_context.role_hashes == raw_role_hashes
        assert set(validations["model"].evaluation_context.role_hashes) == {"train", "tuning"}
        assert raw_role_hashes == {"train": "raw-train", "tuning": "raw-tuning"}

    @pytest.mark.parametrize("state", ["unknown", None])
    def test_complete_evidence_with_unknown_or_absent_state_is_not_success(self, state):
        report = {
            "summary_stats": {
                "mean_abs_log_disparity": 0.1,
                "median_abs_log_disparity": 0.1,
                "share_significant_bh": 0.0,
            }
        }
        if state is not None:
            report["state"] = state

        validations = validate_log_disparity_results(
            {"model": report}, ["model"], role_hashes={"train": "hash"}
        )

        assert validations["model"].decision_eligible is False
        assert all(record.raw_value is None for record in validations["model"].records)

    def test_valid_release_evaluation_returns_succeeded_report(self, make_canonical_dataset):
        fairness_dataset = make_canonical_dataset()
        real_data = _mark_release(fairness_dataset.role_frame("tuning"), role="tuning")
        fairness_dataset.roles["tuning"] = real_data
        synthetic = _mark_release(
            pd.DataFrame(
                {
                    "patient_id": [101, 102, 103, 104],
                    "feature": [1.0, 2.0, 3.0, 4.0],
                    "protected": ["A", "B", "A", "B"],
                    "target": [0, 1, 1, 0],
                }
            )
        )

        reports = run_log_disparity_evaluation(
            {"good_model": synthetic},
            fairness_dataset,
            LogDisparityConfig(protected_columns=["protected"]),
            FrameworkSelectionConfig(enabled=True),
        )

        report = reports["good_model"]
        assert report["state"] == "succeeded"
        assert report["leaf_results"] is not None

    def test_disabled_selection_returns_empty(self, fairness_dataset):
        reports = run_log_disparity_evaluation(
            {"model_a": fairness_dataset.train_df},
            fairness_dataset,
            LogDisparityConfig(protected_columns=["sex"]),
            FrameworkSelectionConfig(enabled=False),
        )
        assert reports == {}

    def test_no_protected_columns_warns_and_returns_empty(self, fairness_dataset):
        fairness_dataset.sensitive_columns = []
        fairness_dataset.protected_columns = []
        reports = run_log_disparity_evaluation(
            {"model_a": fairness_dataset.train_df},
            fairness_dataset,
            LogDisparityConfig(protected_columns=[]),
            FrameworkSelectionConfig(enabled=True),
        )
        assert reports["model_a"]["result_metadata"]["release_evidence_state"] == "indeterminate"

    def test_missing_legacy_tuning_role_is_indeterminate(self, fairness_dataset):
        good_synth = _mark_release(
            pd.DataFrame({"sex": ["M", "F", "M", "F"], "target": [0, 1, 1, 0]})
        )
        reports = run_log_disparity_evaluation(
            {"good_model": good_synth},
            fairness_dataset,
            LogDisparityConfig(protected_columns=["sex"]),
            FrameworkSelectionConfig(enabled=True),
        )
        assert reports["good_model"]["result_metadata"]["release_evidence_state"] == "indeterminate"

    def test_missing_provenance_is_indeterminate_for_canonical_dataset(
        self, make_canonical_dataset
    ):
        dataset = make_canonical_dataset()
        syn = pd.DataFrame({"protected": ["A"], "target": [0], "feature": [1.0]})
        reports = run_log_disparity_evaluation(
            {"model": syn},
            dataset,
            LogDisparityConfig(protected_columns=["protected"]),
            FrameworkSelectionConfig(enabled=True),
            evaluation_role="final_holdout",
            reference_frame=dataset.role_frame("final_holdout", imputed=False),
        )
        assert reports["model"]["result_metadata"]["release_evidence_state"] == "indeterminate"

    def test_final_holdout_requires_explicit_released_reference(self, make_canonical_dataset):
        dataset = make_canonical_dataset()

        with pytest.raises(ValueError, match="requires an explicit released reference frame"):
            run_log_disparity_evaluation(
                {},
                dataset,
                LogDisparityConfig(protected_columns=["protected"]),
                FrameworkSelectionConfig(enabled=True),
                evaluation_role="final_holdout",
            )

    def test_final_holdout_evaluator_uses_explicit_released_reference(
        self, make_canonical_dataset, monkeypatch
    ):
        dataset = make_canonical_dataset()
        raw_reference = dataset.role_frame("final_holdout", imputed=False)
        released_source = raw_reference.copy()
        released_source.loc[:, "feature"] = 999
        _released_synthetic, released_roles, _metadata = transform_release_roles(
            dataset.role_frame("train", imputed=True).copy(),
            {"final_holdout": released_source},
            None,
        )
        released_reference = released_roles["final_holdout"]
        synthetic = _released_synthetic
        received = {}

        def capture(**kwargs):
            received["real_data"] = kwargs["real_data"]
            return {
                "leaf_results": {},
                "hierarchy_results": {},
                "subgroup_table": {},
                "leaf_equity_table": {},
                "legend_table": {},
                "label_counts": {},
                "summary_stats": {},
            }

        monkeypatch.setattr(metric_log_disparity, "compute_log_disparity_report", capture)
        run_log_disparity_evaluation(
            {"model": synthetic},
            dataset,
            LogDisparityConfig(protected_columns=["protected"]),
            FrameworkSelectionConfig(enabled=True),
            evaluation_role="final_holdout",
            reference_frame=released_reference,
        )

        pd.testing.assert_frame_equal(received["real_data"], released_reference)

    def test_failing_model_recorded_not_raised(self, fairness_dataset):
        # Missing the "sex" protected column entirely -> KeyError inside
        # compute_log_disparity_report, which must be caught and persisted,
        # not raised or silently dropped.
        bad_synth = pd.DataFrame({"target": [0, 1, 0, 1]})
        good_synth = pd.DataFrame({"sex": ["M", "F", "M", "F"], "target": [0, 1, 1, 0]})
        reports = run_log_disparity_evaluation(
            {"good_model": good_synth, "bad_model": bad_synth},
            fairness_dataset,
            LogDisparityConfig(protected_columns=["sex"]),
            FrameworkSelectionConfig(enabled=True),
        )
        assert reports["good_model"]["result_metadata"]["release_evidence_state"] == "indeterminate"
        assert reports["bad_model"]["result_metadata"]["release_evidence_state"] == "indeterminate"

    def test_indeterminate_and_failed_reports_have_distinct_non_decision_states(self):
        reports = {
            "incomplete": {
                "state": "indeterminate",
                "reason": "missing tables",
                "missing_tables": ["leaf_results"],
                "result_metadata": {"release_evidence_state": "indeterminate"},
            },
            "failed": {"state": "failed", "error": "boom", "error_type": "KeyError"},
        }
        table = build_log_disparity_summary_table(reports)
        validations = validate_log_disparity_results(
            reports, ["incomplete", "failed"], role_hashes={"train": "hash"}
        )

        assert table.isna().all().all()
        assert validations["incomplete"].decision_eligible is False
        assert validations["failed"].decision_eligible is False
        assert all(record.raw_value is None for record in validations["failed"].records)
        assert all(record.raw_value is None for record in validations["incomplete"].records)


class TestBuildLogDisparitySummaryTable:
    def test_success_report_extracts_summary_stats(self):
        reports = {
            "model_a": {
                "state": "succeeded",
                "summary_stats": {
                    "mean_abs_log_disparity": 0.1,
                    "median_abs_log_disparity": 0.2,
                    "share_significant_bh": 0.3,
                },
            }
        }
        table = build_log_disparity_summary_table(reports)
        assert table.loc["model_a", "log_disparity_mean_abs"] == 0.1
        assert table.loc["model_a", "log_disparity_median_abs"] == 0.2
        assert table.loc["model_a", "log_disparity_share_significant"] == 0.3

    @pytest.mark.parametrize("state", ["unknown", None])
    def test_unknown_or_absent_state_does_not_extract_summary_stats(self, state):
        report = {"summary_stats": {"mean_abs_log_disparity": 0.1}}
        if state is not None:
            report["state"] = state

        table = build_log_disparity_summary_table({"model_a": report})

        assert table.loc["model_a"].isna().all()

    def test_failed_report_yields_all_nan_row_not_keyerror(self):
        reports = {"model_a": {"error": "boom", "error_type": "KeyError"}}
        table = build_log_disparity_summary_table(reports)
        assert table.loc["model_a"].isna().all()

    def test_mixed_success_and_failure(self):
        reports = {
            "good": {
                "state": "succeeded",
                "summary_stats": {
                    "mean_abs_log_disparity": 0.5,
                    "median_abs_log_disparity": 0.4,
                    "share_significant_bh": 0.1,
                },
            },
            "bad": {"error": "boom", "error_type": "ValueError"},
        }
        table = build_log_disparity_summary_table(reports)
        assert table.loc["good"].notna().all()
        assert table.loc["bad"].isna().all()
