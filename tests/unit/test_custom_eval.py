"""Unit tests for synthdata.evaluation.custom_eval."""

import hashlib
import json

import pandas as pd
import pytest

from synthdata.config import FrameworkSelectionConfig, LogDisparityConfig
from synthdata.evaluation.custom_eval import (
    build_log_disparity_summary_table,
    run_log_disparity_evaluation,
)

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
        )
        assert reports["model"]["result_metadata"]["release_evidence_state"] == "indeterminate"

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


class TestBuildLogDisparitySummaryTable:
    def test_success_report_extracts_summary_stats(self):
        reports = {
            "model_a": {
                "summary_stats": {
                    "mean_abs_log_disparity": 0.1,
                    "median_abs_log_disparity": 0.2,
                    "share_significant_bh": 0.3,
                }
            }
        }
        table = build_log_disparity_summary_table(reports)
        assert table.loc["model_a", "log_disparity_mean_abs"] == 0.1
        assert table.loc["model_a", "log_disparity_median_abs"] == 0.2
        assert table.loc["model_a", "log_disparity_share_significant"] == 0.3

    def test_failed_report_yields_all_nan_row_not_keyerror(self):
        reports = {"model_a": {"error": "boom", "error_type": "KeyError"}}
        table = build_log_disparity_summary_table(reports)
        assert table.loc["model_a"].isna().all()

    def test_mixed_success_and_failure(self):
        reports = {
            "good": {
                "summary_stats": {
                    "mean_abs_log_disparity": 0.5,
                    "median_abs_log_disparity": 0.4,
                    "share_significant_bh": 0.1,
                }
            },
            "bad": {"error": "boom", "error_type": "ValueError"},
        }
        table = build_log_disparity_summary_table(reports)
        assert table.loc["good"].notna().all()
        assert table.loc["bad"].isna().all()
