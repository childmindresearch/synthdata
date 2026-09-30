"""Unit tests for the pure-function parts of synthdata.evaluation.syntheval_eval:
the binary-target collapsing helpers used to let auroc_diff/statistical_parity/
equalized_odds/equal_opportunity run against a target with more than 2 classes,
and the syntheval benchmark result caching helpers.
"""

import json
import os
import queue
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest
from syntheval.execution import build_metric_execution

from synthdata.config import FrameworkSelectionConfig, SynthEvalExecutionConfig
from synthdata.data import dataframe_fingerprint, semantic_context_digest
from synthdata.evaluation.catalog import (
    FAIRNESS_METRICS_WITH_POSITIVE_CLASS,
    SYNTHEVAL_PRESET,
    syntheval_execution_keys_by_framework,
    syntheval_execution_manifest,
)
from synthdata.evaluation.metric_contracts import MetricValidationResult
from synthdata.evaluation.syntheval_eval import (
    BINARY_ONLY_METRICS,
    InsufficientSynthEvalCPUError,
    _atomic_parquet,
    _candidate_role_frames,
    _checkpoint_paths,
    _compute_cache_key,
    _evaluation_context_fingerprint,
    _evaluation_role_frames,
    _execution_payload_failed,
    _execution_payload_partial,
    _execution_payload_succeeded,
    _execution_sidecar_payload,
    _failed_execution_payload,
    _frame_fingerprint,
    _load_syntheval_cache,
    _load_syntheval_execution_sidecars,
    _native_plot_dir,
    _run_resumable_syntheval,
    _safe_failure_evidence,
    _safe_metric_status,
    _safe_value_digest,
    _sanitize_execution_payload,
    _save_syntheval_cache,
    _shutdown_nested_joblib_executor,
    _structured_observations,
    _synthetic_unseen_categorical_values,
    _valid_checkpoint,
    _validated_cached_syntheval_tables,
    build_binary_preset,
    build_binary_target_series,
    build_group_context,
    build_metric_execution_passes,
    build_preset,
    build_syntheval_tables_from_executions,
    extend_syntheval_expected_diagnostics,
    merge_binary_target_results,
    resolve_model_workers,
    run_binary_target_syntheval_evaluation,
    run_syntheval_evaluation,
    validate_syntheval_results,
)

pytestmark = pytest.mark.unit


def test_safe_failure_evidence_discards_untrusted_exception_and_exit_details():
    evidence = _safe_failure_evidence(
        exception_type="ValueError: /secret/checkpoint/status.json",
        reason="worker exploded; traceback=/secret/traceback",
        exit_code="1",
    )

    assert evidence["error_type"] == "WorkerExit"
    assert evidence["reason_code"] == "unknown_exception"
    assert evidence["failure_reason"] == "SynthEval worker failed with an unknown error."
    assert evidence["exit_code"] is None
    assert evidence["reason_detail_digest"] == _safe_value_digest(
        "worker exploded; traceback=/secret/traceback"
    )


def test_runtime_unknown_category_exception_is_classified_and_redacted():
    sentinel = "sentinel-unknown-category"
    evidence = _safe_failure_evidence(
        exception_type="ValueError",
        reason=f"Found unknown categories: {sentinel}",
        exit_code=1,
    )

    serialized = json.dumps(evidence)
    assert evidence["reason_code"] == "synthetic_unknown_category"
    assert evidence["reason_detail_digest"] == _safe_value_digest(
        f"Found unknown categories: {sentinel}"
    )
    assert sentinel not in serialized


def test_real_holdout_unknown_category_has_distinct_failure_classification():
    evidence = _safe_failure_evidence(
        exception_type="RealHoldoutUnknownCategoryError",
        reason=(
            "real_holdout_unknown_category: Real holdout contains categorical values "
            "absent from train"
        ),
        exit_code=1,
    )

    assert evidence["reason_code"] == "real_holdout_unknown_category"
    assert evidence["failure_reason"] == (
        "Real holdout contains categories absent from train; affected metric was blocked."
    )
    assert evidence["error_type"] == "RealHoldoutUnknownCategoryError"
    status = _safe_metric_status(
        {
            "method": "cls_acc",
            "state": "blocked",
            "exception_type": "RealHoldoutUnknownCategoryError",
            "exception_message": "Real holdout contains categorical values absent from train",
        }
    )
    assert status["reason_code"] == "real_holdout_unknown_category"
    assert status["failure_class"] == "real_holdout_unknown_category"


def test_expected_holdout_block_persists_partial_worker_checkpoint(
    monkeypatch, tmp_path, make_canonical_dataset
):
    import syntheval

    import synthdata.evaluation.syntheval_eval as syntheval_eval

    frame = make_canonical_dataset().role_frame("train", imputed=True).copy()
    expected_manifest = {
        "supported": ("metric_a",),
        "holdout_sensitive": ("cls_acc",),
    }

    class MetricStatus:
        def __init__(self, method, state, key):
            self.method = method
            self.state = state
            self.succeeded = state == "succeeded"
            self.exception_type = "RealHoldoutUnknownCategoryError" if state == "blocked" else None
            self.exception_message = (
                "Real holdout contains categories absent from train" if state == "blocked" else None
            )
            self.key = key

        def to_dict(self):
            succeeded = self.succeeded
            payload = {
                "method": self.method,
                "state": self.state,
                "expected_keys": [self.key],
                "observed_keys": [self.key] if succeeded else [],
                "completed_keys": [self.key] if succeeded else [],
                "failed_keys": [] if succeeded else [self.key],
                "missing_keys": [] if succeeded else [self.key],
                "duplicate_keys": [],
                "non_finite_keys": [],
                "unexpected_keys": [],
            }
            if not succeeded:
                payload.update(
                    {
                        "exception_type": self.exception_type,
                        "exception_message": self.exception_message,
                    }
                )
            return payload

    def metric(method, state, key, rows):
        return SimpleNamespace(
            method=method,
            status=MetricStatus(method, state, key),
            normalized_rows=rows,
            normalized_rows_v2=[],
        )

    execution = SimpleNamespace(
        schema_version="syntheval-execution-v1",
        pass_id="main",
        target_view="native",
        expected_manifest_digest="manifest-digest",
        execution_complete=True,
        succeeded=False,
        policy_eligible=False,
        preprocessing_fingerprint=None,
        preprocessing_metadata=None,
        normalized_table=pd.DataFrame(),
        metric_executions=[
            metric(
                "supported",
                "succeeded",
                "metric_a",
                [
                    {
                        "metric": "metric_a",
                        "dim": "u",
                        "val": 0.5,
                        "err": 0.0,
                        "n_val": 0.5,
                        "n_err": 0.0,
                    }
                ],
            ),
            metric("holdout_sensitive", "blocked", "cls_acc", []),
        ],
    )

    class FakeSynthEval:
        def __init__(self, *_args, **_kwargs):
            pass

        def evaluate(self, *_args, **_kwargs):
            return execution

    monkeypatch.setattr(syntheval, "AnalysisConfig", lambda **_kwargs: object())
    monkeypatch.setattr(syntheval, "SynthEval", FakeSynthEval)
    monkeypatch.setattr(syntheval_eval, "_shutdown_nested_joblib_executor", lambda: None)
    checkpoint_root = tmp_path / "checkpoints"

    syntheval_eval._model_worker(
        "partial_model",
        frame,
        frame,
        frame,
        [],
        "target",
        [],
        [],
        str(tmp_path / "preset.json"),
        str(checkpoint_root),
        "main",
        expected_manifest,
        "native",
        "manifest-digest",
        "context-fingerprint",
        "model-fingerprint",
        None,
        1,
    )

    model_dir, status_path, _result_path = _checkpoint_paths(
        checkpoint_root, "main", "partial_model"
    )
    status = json.loads(status_path.read_text())
    payload = json.loads((model_dir / "execution.json").read_text())
    assert status["state"] == "partial"
    assert status["execution_succeeded"] is False
    assert status["policy_eligible"] is False
    assert status["incomplete_reasons"] == ["real_holdout_unknown_category"]
    blocked_status = payload["metric_executions"][1]["status"]
    assert blocked_status["state"] == "blocked"
    assert blocked_status["reason_code"] == "real_holdout_unknown_category"
    assert payload["model_status"] == "partial"
    assert _execution_payload_partial(payload, expected_manifest=expected_manifest)
    assert (
        _valid_checkpoint(
            checkpoint_root,
            "main",
            "partial_model",
            "context-fingerprint",
            "model-fingerprint",
            False,
            expected_manifest_digest="manifest-digest",
            return_execution=True,
            expected_manifest=expected_manifest,
            expected_target_view="native",
        )
        is not None
    )


def test_safe_failure_evidence_preserves_safe_classification_only():
    evidence = _safe_failure_evidence(
        exception_type="ValueError", reason="unknown target in /secret/input.csv", exit_code=1
    )

    assert evidence["error_type"] == "ValueError"
    assert evidence["reason_code"] == "unknown_target"
    assert evidence["failure_reason"] == "Synthetic data failed target validation."
    assert evidence["exit_code"] == 1


def test_sanitize_execution_payload_recursively_removes_nested_worker_evidence():
    payload = {
        "model_name": "model_a",
        "metric_executions": [
            {
                "status": {
                    "state": "failed",
                    "exception": "sentinel exception body",
                    "exception_traceback": "sentinel traceback",
                    "failure_reason": "sentinel /secret/checkpoint.json",
                    "checkpoint_path": "/secret/checkpoint.json",
                    "nested": {
                        "traceback": "sentinel nested traceback",
                        "path": "/secret/nested.json",
                    },
                }
            }
        ],
    }

    sanitized = _sanitize_execution_payload(payload)
    serialized = json.dumps(sanitized)

    assert "sentinel" not in serialized
    assert "/secret" not in serialized
    status = sanitized["metric_executions"][0]["status"]
    assert status["reason_code"] == "unknown_exception"
    assert status["failure_reason"] == "SynthEval worker failed with an unknown error."


def _model_index(*names: str) -> pd.Index:
    return pd.Index(names)


class TestBuildBinaryTargetSeries:
    def test_maps_positive_and_negative_classes(self):
        series = pd.Series([0, 1, 2, 0, 2], name="CGAS_class")
        out = build_binary_target_series(series, positive_classes=[0, 1], negative_classes=[2])
        assert out.tolist() == [1, 1, 0, 1, 0]
        assert out.name == "CGAS_class"

    def test_output_is_int_dtype(self):
        # Must be int (not float): SynthEval's AnalysisConfig only treats
        # object/int-dtype columns as categorical -- a float output would
        # silently get classified as continuous ("num"), defeating the
        # entire point of this function.
        series = pd.Series([0, 1, 2], name="t")
        out = build_binary_target_series(series, positive_classes=[0, 1], negative_classes=[2])
        assert out.dtype == np.int64

    def test_missing_value_raises(self):
        series = pd.Series([0, np.nan, 2], name="t")
        with pytest.raises(ValueError, match="missing value"):
            build_binary_target_series(series, positive_classes=[0], negative_classes=[2])


class TestJoblibCleanup:
    def test_shutdown_nested_executor_terminates_reusable_pool(self, monkeypatch):
        shutdown_calls = []

        class Executor:
            def shutdown(self, *, wait, kill_workers):
                shutdown_calls.append((wait, kill_workers))

        monkeypatch.setattr("joblib.externals.loky.reusable_executor._executor", Executor())

        _shutdown_nested_joblib_executor()

        assert shutdown_calls == [(True, True)]

    def test_unmapped_value_raises(self):
        series = pd.Series([0, 1, 2], name="CGAS_class")
        with pytest.raises(ValueError, match="CGAS_class"):
            build_binary_target_series(series, positive_classes=[0], negative_classes=[2])

    def test_native_plot_dir_uses_bounded_model_id(self, tmp_path):
        model_name = "../team/model: candidate"

        plot_dir = _native_plot_dir(tmp_path, model_name)

        assert plot_dir.parent == tmp_path.resolve()
        assert plot_dir.resolve().parent == tmp_path.resolve()

    def test_unmapped_value_message_lists_offending_value(self):
        series = pd.Series([0, 1, 2], name="t")
        with pytest.raises(ValueError, match=r"\[1\]"):
            build_binary_target_series(series, positive_classes=[0], negative_classes=[2])


class TestBuildPreset:
    def _selection(self, **overrides) -> FrameworkSelectionConfig:
        return FrameworkSelectionConfig(**overrides)

    def test_default_positive_class_matches_preset_default(self):
        preset = build_preset(self._selection())
        for name in FAIRNESS_METRICS_WITH_POSITIVE_CLASS:
            assert preset[name]["positive_class"] == 1

    def test_positive_class_override_applies_to_fairness_metrics_only(self):
        preset = build_preset(self._selection(), positive_class=0)
        for name in FAIRNESS_METRICS_WITH_POSITIVE_CLASS:
            assert preset[name]["positive_class"] == 0
        # A non-fairness metric's params must be untouched.
        assert preset["dwm"] == {}

    def test_override_does_not_mutate_shared_preset_constant(self):
        # SYNTHEVAL_PRESET's nested dicts are shared module-level objects --
        # build_preset must shallow-copy before overriding, or this would
        # corrupt the global constant for every subsequent call in-process.
        build_preset(self._selection(), positive_class=0)
        for name in FAIRNESS_METRICS_WITH_POSITIVE_CLASS:
            assert SYNTHEVAL_PRESET[name]["positive_class"] == 1

    def test_disabled_selection_returns_empty(self):
        preset = build_preset(self._selection(enabled=False))
        assert preset == {}

    def test_multiclass_native_preset_includes_ovr_metrics(self):
        preset = build_preset(self._selection(), target_is_binary=False, target_is_multiclass=True)

        assert set(BINARY_ONLY_METRICS) <= set(preset)
        assert "dwm" in preset

    def test_non_multiclass_nonbinary_target_still_excludes_binary_metrics(self):
        preset = build_preset(self._selection(), target_is_binary=False)

        assert not set(BINARY_ONLY_METRICS) & set(preset)

    def test_binary_target_preset_retains_existing_metric_methods(self):
        preset = build_preset(self._selection(), target_is_binary=True)

        assert set(BINARY_ONLY_METRICS) <= set(preset)

    def test_auroc_macro_ovr_rows_require_matching_normalizer_version(self):
        key = "auroc_macro_ovr_v3"
        payload = {
            "schema_version": "syntheval-execution-v1",
            "execution_complete": True,
            "execution_succeeded": True,
            "metric_executions": [
                {
                    "method": "auroc_diff",
                    "status": {
                        "method": "auroc_diff",
                        "state": "succeeded",
                        "expected_keys": [key],
                        "observed_keys": [key],
                        "completed_keys": [key],
                        "failed_keys": [],
                        "missing_keys": [],
                        "duplicate_keys": [],
                        "non_finite_keys": [],
                        "unexpected_keys": [],
                    },
                    "normalized_rows": [],
                    "normalized_rows_v2": [
                        {
                            "metric": key,
                            "dim": "u",
                            "val": 0.2,
                            "n_val": 0.8,
                            "raw_value": 0.2,
                            "normalized_value": 0.8,
                            "metric_version": "macro_ovr_v3",
                        }
                    ],
                }
            ],
        }

        assert _execution_payload_succeeded(payload)
        payload["metric_executions"][0]["normalized_rows_v2"][0]["metric_version"] = "v2"
        assert not _execution_payload_succeeded(payload)

    def test_auroc_class_ovr_diagnostic_rows_require_macro_normalizer_version(self):
        key = "auroc_income_class_0_ovr_v3"
        row = {
            "metric": key,
            "dim": "u",
            "val": 0.2,
            "n_val": 0.8,
            "raw_value": 0.2,
            "normalized_value": 0.8,
            "metric_version": "macro_ovr_v3",
        }
        payload = {
            "schema_version": "syntheval-execution-v1",
            "execution_complete": True,
            "execution_succeeded": True,
            "metric_executions": [
                {
                    "method": "auroc_diff",
                    "status": {
                        "method": "auroc_diff",
                        "state": "succeeded",
                        "expected_keys": [key],
                        "observed_keys": [key],
                        "completed_keys": [key],
                        "failed_keys": [],
                        "missing_keys": [],
                        "duplicate_keys": [],
                        "non_finite_keys": [],
                        "unexpected_keys": [],
                    },
                    "normalized_rows": [row.copy()],
                    "normalized_rows_v2": [row.copy()],
                }
            ],
        }

        assert _execution_payload_succeeded(payload)
        payload["metric_executions"][0]["normalized_rows"][0]["metric_version"] = "v2"
        payload["metric_executions"][0]["normalized_rows_v2"][0]["metric_version"] = "v2"
        assert not _execution_payload_succeeded(payload)

    @pytest.mark.parametrize("payload_state", ["succeeded", "failed"])
    @pytest.mark.parametrize(
        ("field", "invalid_value"),
        [
            ("metric_version", "unexpected"),
            ("raw_value", float("nan")),
            ("normalized_value", float("inf")),
        ],
    )
    def test_legacy_holdout_rows_enforce_v2_contract_in_success_and_failure_payloads(
        self, payload_state, field, invalid_value
    ):
        key = "avg_macro_F1_diff_v2_hout"

        def make_payload():
            succeeded = payload_state == "succeeded"
            row = {
                "metric": key,
                "dim": "u",
                "val": 0.2,
                "n_val": 0.8,
                "raw_value": 0.2,
                "normalized_value": 0.8,
                "metric_version": "v2",
            }
            status = {
                "method": "tstr",
                "state": "succeeded" if succeeded else "failed",
                "expected_keys": [key],
                "observed_keys": [key],
                "completed_keys": [key] if succeeded else [],
                "failed_keys": [] if succeeded else [key],
                "missing_keys": [],
                "duplicate_keys": [],
                "non_finite_keys": [],
                "unexpected_keys": [],
            }
            payload = {
                "schema_version": "syntheval-execution-v1",
                "execution_complete": True,
                "execution_succeeded": succeeded,
                "metric_executions": [
                    {
                        "method": "tstr",
                        "status": status,
                        "normalized_rows": [],
                        "normalized_rows_v2": [row],
                    }
                ],
            }
            if not succeeded:
                payload.update(
                    policy_eligible=False,
                    failure_reason="Metric execution failed.",
                )
            return payload

        valid_payload = make_payload()
        if payload_state == "succeeded":
            assert _execution_payload_succeeded(valid_payload)
        else:
            assert _execution_payload_failed(valid_payload)

        invalid_payload = make_payload()
        invalid_payload["metric_executions"][0]["normalized_rows_v2"][0][field] = invalid_value
        if payload_state == "succeeded":
            assert not _execution_payload_succeeded(invalid_payload)
        else:
            assert not _execution_payload_failed(invalid_payload)


class TestEvaluationRoleContext:
    def test_candidate_role_resolution_never_reads_final_holdout(self, make_canonical_dataset):
        dataset = make_canonical_dataset()

        class HoldoutGuard:
            legacy_two_role = False

            def __init__(self):
                self.roles_read = []

            def role_frame(self, role, *, imputed):
                self.roles_read.append(role)
                if role == "final_holdout":
                    raise AssertionError("candidate evaluation read final_holdout")
                return dataset.role_frame(role, imputed=imputed)

            def __getattr__(self, name):
                return getattr(dataset, name)

        guarded = HoldoutGuard()
        fit_frame, tuning_frame = _candidate_role_frames(guarded)
        assert fit_frame is dataset.role_frame("train", imputed=True)
        assert tuning_frame is dataset.role_frame("tuning", imputed=True)

        guarded.roles_read.clear()
        _evaluation_context_fingerprint(guarded, {}, "main", False)

        assert "final_holdout" not in guarded.roles_read
        assert set(guarded.roles_read) == {"train", "tuning"}

    def test_explicit_final_evaluation_reads_final_holdout(self, make_canonical_dataset):
        dataset = make_canonical_dataset()

        class RoleReadTracker:
            legacy_two_role = False

            def __init__(self):
                self.roles_read = []

            def role_frame(self, role, *, imputed):
                self.roles_read.append(role)
                return dataset.role_frame(role, imputed=imputed)

            def __getattr__(self, name):
                return getattr(dataset, name)

        tracked = RoleReadTracker()
        fit_frame, final_holdout_frame = _evaluation_role_frames(tracked, "final_holdout")

        assert fit_frame is dataset.role_frame("train", imputed=True)
        assert final_holdout_frame is dataset.role_frame("final_holdout", imputed=True)
        assert tracked.roles_read == ["train", "final_holdout"]

    def test_resumable_candidate_fallback_never_reads_final_holdout(
        self, make_canonical_dataset, tmp_path, monkeypatch
    ):
        dataset = make_canonical_dataset()
        original_role_frame = dataset.role_frame
        roles_read = []

        def guarded_role_frame(role, *, imputed):
            roles_read.append(role)
            if role == "final_holdout":
                raise AssertionError("candidate fallback read final_holdout")
            return original_role_frame(role, imputed=imputed)

        def stop_after_frame_resolution(*_args, **_kwargs):
            raise RuntimeError("stop after frame resolution")

        monkeypatch.setattr(dataset, "role_frame", guarded_role_frame)
        monkeypatch.setattr(
            "synthdata.evaluation.syntheval_eval._evaluation_context_fingerprint",
            stop_after_frame_resolution,
        )

        with pytest.raises(RuntimeError, match="stop after frame resolution"):
            _run_resumable_syntheval(
                {},
                dataset,
                {},
                tmp_path / "preset.json",
                tmp_path / "run",
                "linear",
                None,
                "main",
            )

        assert "final_holdout" not in roles_read
        assert set(roles_read) == {"train", "tuning"}

    def test_binary_candidate_evaluation_never_reads_final_holdout(
        self, make_canonical_dataset, tmp_path, monkeypatch
    ):
        dataset = make_canonical_dataset()
        original_role_frame = dataset.role_frame
        roles_read = []

        def guarded_role_frame(role, *, imputed):
            roles_read.append(role)
            if role == "final_holdout":
                raise AssertionError("binary candidate evaluation read final_holdout")
            return original_role_frame(role, imputed=imputed)

        monkeypatch.setattr(dataset, "role_frame", guarded_role_frame)
        invalid_synthetic = original_role_frame("train", imputed=True).copy()
        invalid_synthetic["target"] = 99

        run_binary_target_syntheval_evaluation(
            {"invalid_model": invalid_synthetic},
            dataset,
            FrameworkSelectionConfig(),
            SimpleNamespace(column="target", positive_classes=[1], negative_classes=[0]),
            preset_dir=tmp_path,
            output_folder=tmp_path / "binary",
        )

        assert "final_holdout" not in roles_read
        assert set(roles_read) == {"train", "tuning"}

    def test_binary_mixed_inventory_survives_fresh_and_cached_paths(
        self, make_canonical_dataset, tmp_path, monkeypatch
    ):
        dataset = make_canonical_dataset()
        valid = dataset.role_frame("train", imputed=True).copy()
        valid.loc[valid.index[:4], "target"] = [0, 1, 2, 2]
        invalid = valid.drop(columns=["target"])
        selection = FrameworkSelectionConfig(metrics=["auroc_diff"])
        binary_cfg = SimpleNamespace(column="target", positive_classes=[1], negative_classes=[0, 2])
        worker_inputs = []

        valid_execution = {
            "model_name": "valid_model",
            "schema_version": "syntheval-execution-v1",
            "execution_complete": True,
            "execution_succeeded": True,
            "policy_eligible": True,
            "metric_executions": [
                {
                    "method": "auroc_diff",
                    "status": {
                        "method": "auroc_diff",
                        "state": "succeeded",
                        "expected_keys": ["auroc_v2"],
                        "observed_keys": ["auroc_v2"],
                        "completed_keys": ["auroc_v2"],
                        "failed_keys": [],
                        "missing_keys": [],
                        "duplicate_keys": [],
                        "non_finite_keys": [],
                        "unexpected_keys": [],
                    },
                    "normalized_rows_v2": [
                        {
                            "metric": "auroc_v2",
                            "dim": "u",
                            "val": 0.2,
                            "err": 0.0,
                            "n_val": 0.8,
                            "n_err": 0.0,
                            "raw_value": 0.2,
                            "normalized_value": 0.8,
                            "metric_version": "v2",
                        }
                    ],
                    "normalized_rows": [],
                }
            ],
        }

        class FakeProcess:
            def __init__(self, target, args, **_kwargs):
                self.target = target
                self.args = args
                self.exitcode = None
                self.pid = 1

            def start(self):
                worker_inputs.append({self.args[0]})
                self.target(*self.args)
                self.exitcode = 0

            def is_alive(self):
                return False

            def join(self):
                return None

        class FakeContext:
            Process = FakeProcess

        def fake_model_worker(
            model_name,
            frame,
            _real_frame,
            _holdout_frame,
            _cat_cols,
            _target_column,
            _sensitive_columns,
            _protected_columns,
            _preset_path,
            checkpoint_root,
            pass_name,
            expected_output_manifest,
            target_view,
            expected_manifest_digest,
            context_fingerprint,
            model_fingerprint,
            *_unused,
        ):
            checkpoint_root = Path(checkpoint_root)
            execution = {**valid_execution, "model_name": model_name}
            execution["model_fingerprint"] = model_fingerprint
            execution["pass_id"] = pass_name
            execution["target_view"] = target_view
            execution["expected_manifest_digest"] = expected_manifest_digest
            execution["context_fingerprint"] = context_fingerprint
            expected_keys = list(expected_output_manifest["auroc_diff"])
            execution["metric_executions"][0]["status"].update(
                {
                    "expected_keys": expected_keys,
                    "observed_keys": expected_keys,
                    "completed_keys": expected_keys,
                }
            )
            model_dir, status_path, result_path = _checkpoint_paths(
                checkpoint_root, pass_name, model_name
            )
            model_dir.mkdir(parents=True, exist_ok=True)
            status_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "state": "succeeded",
                        "model_name": model_name,
                        "context_fingerprint": context_fingerprint,
                        "model_fingerprint": model_fingerprint,
                        "expected_manifest_digest": expected_manifest_digest,
                        "plots_completed": False,
                    }
                )
            )
            result, ranks = build_syntheval_tables_from_executions(
                {model_name: execution}, [model_name], "linear"
            )
            result.to_parquet(result_path)
            (model_dir / "execution.json").write_text(json.dumps(execution))

        monkeypatch.setattr(
            "synthdata.evaluation.syntheval_eval.multiprocessing.get_context",
            lambda _method: FakeContext(),
        )
        monkeypatch.setattr("synthdata.evaluation.syntheval_eval._model_worker", fake_model_worker)

        first_results, first_ranks, first_executions = run_binary_target_syntheval_evaluation(
            {"valid_model": valid, "invalid_model": invalid},
            dataset,
            selection,
            binary_cfg,
            preset_dir=tmp_path,
            output_folder=tmp_path / "binary",
            return_execution=True,
        )

        assert worker_inputs == [{"valid_model"}]
        assert list(first_results.index) == ["valid_model", "invalid_model"]
        assert list(first_ranks.index) == ["valid_model", "invalid_model"]
        assert pd.notna(first_results.loc["valid_model", ("auroc_v2", "value")])
        assert pd.isna(first_results.loc["invalid_model", ("auroc_v2", "value")])
        assert pd.isna(first_ranks.loc["invalid_model", "auroc_v2"])
        assert first_executions["invalid_model"]["execution_succeeded"] is False
        assert first_executions["invalid_model"]["policy_eligible"] is False
        assert (
            first_executions["invalid_model"]["worker_exit"]["reason_code"]
            == "synthetic_binary_target_invalid"
        )
        assert first_executions["invalid_model"]["worker_exit"]["state"] == "failed"
        sidecar_path = (
            tmp_path
            / "binary"
            / "checkpoints-v1"
            / "binary_target"
            / _checkpoint_paths(tmp_path / "binary", "binary_target", "valid_model")[0].name
            / "execution.json"
        )
        assert json.loads(sidecar_path.read_text())["model_fingerprint"] == _frame_fingerprint(
            valid
        )
        assert (
            _load_syntheval_execution_sidecars(
                tmp_path / "binary",
                "binary_target",
                ["valid_model", "invalid_model"],
                first_executions["valid_model"]["expected_manifest_digest"],
                first_executions["valid_model"]["context_fingerprint"],
                expected_manifest={"auroc_diff": ("auroc_v2",)},
                expected_target_view="binary_collapsed",
                model_fingerprints={
                    "valid_model": _frame_fingerprint(valid),
                    "invalid_model": _frame_fingerprint(invalid),
                },
            )
            is not None
        )

        def fail_if_worker_runs(*args, **kwargs):
            raise AssertionError("cached binary aggregate must not reschedule workers")

        monkeypatch.setattr(
            "synthdata.evaluation.syntheval_eval._run_resumable_syntheval",
            fail_if_worker_runs,
        )
        second_results, second_ranks, second_executions = run_binary_target_syntheval_evaluation(
            {"valid_model": valid, "invalid_model": invalid},
            dataset,
            selection,
            binary_cfg,
            preset_dir=tmp_path,
            output_folder=tmp_path / "binary",
            return_execution=True,
        )

        pd.testing.assert_frame_equal(first_results, second_results, check_dtype=False)
        pd.testing.assert_frame_equal(first_ranks, second_ranks, check_dtype=False)
        assert set(second_executions) == {"valid_model", "invalid_model"}
        assert second_executions["invalid_model"]["policy_eligible"] is False

    def test_binary_preprocessing_failure_uses_complete_inventory_in_fresh_path(
        self, make_canonical_dataset, tmp_path, monkeypatch
    ):
        dataset = make_canonical_dataset()
        bad = dataset.role_frame("train", imputed=True).copy()
        bad["target"] = 99
        process_boundary_calls = []

        def fail_if_process_boundary_runs(_method):
            process_boundary_calls.append(_method)
            raise AssertionError("all-failed binary input must not reach process boundary")

        monkeypatch.setattr(
            "synthdata.evaluation.syntheval_eval.multiprocessing.get_context",
            fail_if_process_boundary_runs,
        )

        first_results, first_ranks, first_executions = run_binary_target_syntheval_evaluation(
            {"bad_model": bad},
            dataset,
            FrameworkSelectionConfig(),
            SimpleNamespace(column="target", positive_classes=[1], negative_classes=[0]),
            preset_dir=tmp_path,
            output_folder=tmp_path / "binary",
            return_execution=True,
        )

        second_results, second_ranks, second_executions = run_binary_target_syntheval_evaluation(
            {"bad_model": bad},
            dataset,
            FrameworkSelectionConfig(),
            SimpleNamespace(column="target", positive_classes=[1], negative_classes=[0]),
            preset_dir=tmp_path,
            output_folder=tmp_path / "binary",
            return_execution=True,
        )

        assert process_boundary_calls == []
        for results, ranks, executions in (
            (first_results, first_ranks, first_executions),
            (second_results, second_ranks, second_executions),
        ):
            assert list(results.index) == ["bad_model"]
            assert list(ranks.index) == ["bad_model"]
            assert executions["bad_model"]["execution_succeeded"] is False
            assert executions["bad_model"]["policy_eligible"] is False
            assert (
                executions["bad_model"]["worker_exit"]["reason_code"]
                == "synthetic_binary_target_invalid"
            )

        model_dir, _status_path, _result_path = _checkpoint_paths(
            tmp_path / "binary", "binary_target", "bad_model"
        )
        assert json.loads((model_dir / "execution.json").read_text())["model_fingerprint"] == (
            _frame_fingerprint(bad)
        )

    def test_binary_valid_only_reuses_complete_cache_without_worker(
        self, make_canonical_dataset, tmp_path, monkeypatch
    ):
        dataset = make_canonical_dataset()
        valid = dataset.role_frame("train", imputed=True).copy()
        valid.loc[valid.index[:4], "target"] = [0, 1, 2, 2]
        selection = FrameworkSelectionConfig(metrics=["auroc_diff"])
        binary_cfg = SimpleNamespace(column="target", positive_classes=[1], negative_classes=[0, 2])
        worker_calls = []
        process_boundary_calls = []

        class FakeProcess:
            def __init__(self, target, args, **_kwargs):
                self.target = target
                self.args = args
                self.exitcode = None
                self.pid = 1

            def start(self):
                worker_calls.append(self.args[0])
                self.target(*self.args)
                self.exitcode = 0

            def is_alive(self):
                return False

            def join(self):
                return None

        class FakeContext:
            Process = FakeProcess

        def fake_model_worker(
            model_name,
            _frame,
            _real_frame,
            _holdout_frame,
            _cat_cols,
            _target_column,
            _sensitive_columns,
            _protected_columns,
            _preset_path,
            checkpoint_root,
            pass_name,
            expected_output_manifest,
            target_view,
            expected_manifest_digest,
            context_fingerprint,
            model_fingerprint,
            *_unused,
        ):
            expected_keys = list(expected_output_manifest["auroc_diff"])
            execution = {
                "model_name": model_name,
                "schema_version": "syntheval-execution-v1",
                "pass_id": pass_name,
                "target_view": target_view,
                "expected_manifest_digest": expected_manifest_digest,
                "context_fingerprint": context_fingerprint,
                "model_fingerprint": model_fingerprint,
                "execution_complete": True,
                "execution_succeeded": True,
                "policy_eligible": True,
                "metric_executions": [
                    {
                        "method": "auroc_diff",
                        "status": {
                            "method": "auroc_diff",
                            "state": "succeeded",
                            "expected_keys": expected_keys,
                            "observed_keys": expected_keys,
                            "completed_keys": expected_keys,
                            "failed_keys": [],
                            "missing_keys": [],
                            "duplicate_keys": [],
                            "non_finite_keys": [],
                            "unexpected_keys": [],
                        },
                        "normalized_rows": [],
                        "normalized_rows_v2": [
                            {
                                "metric": key,
                                "dim": "u",
                                "val": 0.2,
                                "err": 0.0,
                                "n_val": 0.8,
                                "n_err": 0.0,
                                "raw_value": 0.2,
                                "normalized_value": 0.8,
                                "metric_version": "v2",
                            }
                            for key in expected_keys
                        ],
                    }
                ],
            }
            model_dir, status_path, result_path = _checkpoint_paths(
                Path(checkpoint_root), pass_name, model_name
            )
            model_dir.mkdir(parents=True, exist_ok=True)
            status_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "state": "succeeded",
                        "model_name": model_name,
                        "context_fingerprint": context_fingerprint,
                        "model_fingerprint": model_fingerprint,
                        "expected_manifest_digest": expected_manifest_digest,
                        "plots_completed": False,
                    }
                )
            )
            result, _ranks = build_syntheval_tables_from_executions(
                {model_name: execution}, [model_name], "linear"
            )
            result.to_parquet(result_path)
            (model_dir / "execution.json").write_text(json.dumps(execution))

        monkeypatch.setattr(
            "synthdata.evaluation.syntheval_eval.multiprocessing.get_context",
            lambda _method: process_boundary_calls.append(_method) or FakeContext(),
        )
        monkeypatch.setattr("synthdata.evaluation.syntheval_eval._model_worker", fake_model_worker)

        first = run_binary_target_syntheval_evaluation(
            {"valid_model": valid},
            dataset,
            selection,
            binary_cfg,
            preset_dir=tmp_path,
            output_folder=tmp_path / "binary",
            return_execution=True,
        )
        model_dir, _status_path, _result_path = _checkpoint_paths(
            tmp_path / "binary", "binary_target", "valid_model"
        )
        sidecar_path = model_dir / "execution.json"
        source_fingerprint = _frame_fingerprint(valid)
        assert worker_calls == ["valid_model"]
        assert pd.notna(first[0].loc["valid_model", ("auroc_v2", "value")])
        assert pd.notna(first[1].loc["valid_model", "auroc_v2"])
        assert json.loads(sidecar_path.read_text())["model_fingerprint"] == source_fingerprint

        second = run_binary_target_syntheval_evaluation(
            {"valid_model": valid},
            dataset,
            selection,
            binary_cfg,
            preset_dir=tmp_path,
            output_folder=tmp_path / "binary",
            return_execution=True,
        )
        pd.testing.assert_frame_equal(first[0], second[0], check_dtype=False)
        pd.testing.assert_frame_equal(first[1], second[1], check_dtype=False)
        assert second[2]["valid_model"]["model_fingerprint"] == source_fingerprint
        assert json.loads(sidecar_path.read_text())["model_fingerprint"] == source_fingerprint
        assert worker_calls == ["valid_model"]
        assert len(process_boundary_calls) == 1

    def test_binary_invalid_source_change_rejects_failed_cache(
        self, make_canonical_dataset, tmp_path, monkeypatch
    ):
        dataset = make_canonical_dataset()
        invalid = dataset.role_frame("train", imputed=True).copy()
        invalid["target"] = 99
        cfg = FrameworkSelectionConfig(metrics=["auroc_diff"])
        binary_cfg = SimpleNamespace(column="target", positive_classes=[1], negative_classes=[0])

        process_boundary_calls = []

        def fail_if_process_boundary_runs(_method):
            process_boundary_calls.append(_method)
            raise AssertionError("invalid binary model must never reach worker")

        monkeypatch.setattr(
            "synthdata.evaluation.syntheval_eval.multiprocessing.get_context",
            fail_if_process_boundary_runs,
        )
        first = run_binary_target_syntheval_evaluation(
            {"invalid": invalid},
            dataset,
            cfg,
            binary_cfg,
            preset_dir=tmp_path,
            output_folder=tmp_path / "binary",
            return_execution=True,
        )
        meta_path = tmp_path / "binary" / "binary_target_cache_meta.json"
        first_meta = json.loads(meta_path.read_text())
        first_model_dir, _status_path, _result_path = _checkpoint_paths(
            tmp_path / "binary", "binary_target", "invalid"
        )
        first_sidecar = json.loads((first_model_dir / "execution.json").read_text())
        artifact_path = tmp_path / "binary" / "binary_target_results.parquet"
        first_artifact_identity = (
            artifact_path.stat().st_ino,
            artifact_path.stat().st_mtime_ns,
        )
        invalid.loc[0, "feature"] += 1
        second = run_binary_target_syntheval_evaluation(
            {"invalid": invalid},
            dataset,
            cfg,
            binary_cfg,
            preset_dir=tmp_path,
            output_folder=tmp_path / "binary",
            return_execution=True,
        )
        second_meta = json.loads(meta_path.read_text())
        second_sidecar = json.loads((first_model_dir / "execution.json").read_text())
        second_artifact_identity = (
            artifact_path.stat().st_ino,
            artifact_path.stat().st_mtime_ns,
        )

        assert process_boundary_calls == []
        assert first_meta["cache_key"] != second_meta["cache_key"]
        assert first_sidecar["model_fingerprint"] != second_sidecar["model_fingerprint"]
        assert first_artifact_identity != second_artifact_identity
        assert second[2]["invalid"]["model_fingerprint"] == _frame_fingerprint(invalid)
        assert list(first[0].index) == ["invalid"]
        assert list(second[0].index) == ["invalid"]
        assert pd.isna(second[0].loc["invalid", ("auroc_v2", "value")])
        assert pd.isna(second[1].loc["invalid", "auroc_v2"])
        assert second[2]["invalid"]["policy_eligible"] is False

    def test_main_source_change_reprocesses_stale_cache(
        self, make_canonical_dataset, tmp_path, monkeypatch
    ):
        dataset = make_canonical_dataset()
        synthetic = dataset.role_frame("train", imputed=True).copy()
        worker_calls = []

        class FakeProcess:
            def __init__(self, target, args, **_kwargs):
                self.target = target
                self.args = args
                self.exitcode = None
                self.pid = 1

            def start(self):
                worker_calls.append(self.args[0])
                self.target(*self.args)
                self.exitcode = 0

            def is_alive(self):
                return False

            def join(self):
                return None

        class FakeContext:
            Process = FakeProcess

        def fake_model_worker(
            model_name,
            _frame,
            _real_frame,
            _holdout_frame,
            _cat_cols,
            _target_column,
            _sensitive_columns,
            _protected_columns,
            _preset_path,
            checkpoint_root,
            pass_name,
            expected_output_manifest,
            target_view,
            expected_manifest_digest,
            context_fingerprint,
            model_fingerprint,
            *_unused,
        ):
            execution = {
                "model_name": model_name,
                "schema_version": "syntheval-execution-v1",
                "pass_id": pass_name,
                "target_view": target_view,
                "expected_manifest_digest": expected_manifest_digest,
                "context_fingerprint": context_fingerprint,
                "model_fingerprint": model_fingerprint,
                "execution_complete": True,
                "execution_succeeded": True,
                "policy_eligible": True,
                "metric_executions": [
                    {
                        "method": "dwm",
                        "status": {
                            "method": "dwm",
                            "state": "succeeded",
                            "expected_keys": list(expected_output_manifest["dwm"]),
                            "observed_keys": list(expected_output_manifest["dwm"]),
                            "completed_keys": list(expected_output_manifest["dwm"]),
                            "failed_keys": [],
                            "missing_keys": [],
                            "duplicate_keys": [],
                            "non_finite_keys": [],
                            "unexpected_keys": [],
                        },
                        "normalized_rows": [
                            {
                                "metric": key,
                                "dim": "u",
                                "val": 0.5,
                                "err": 0.0,
                                "n_val": 0.5,
                                "n_err": 0.0,
                            }
                            for key in expected_output_manifest["dwm"]
                        ],
                        "normalized_rows_v2": [],
                    }
                ],
            }
            model_dir, status_path, result_path = _checkpoint_paths(
                Path(checkpoint_root), pass_name, model_name
            )
            model_dir.mkdir(parents=True, exist_ok=True)
            status_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "state": "succeeded",
                        "model_name": model_name,
                        "context_fingerprint": context_fingerprint,
                        "model_fingerprint": model_fingerprint,
                        "expected_manifest_digest": expected_manifest_digest,
                        "plots_completed": False,
                    }
                )
            )
            result, ranks = build_syntheval_tables_from_executions(
                {model_name: execution}, [model_name], "linear"
            )
            result.to_parquet(result_path)
            (model_dir / "execution.json").write_text(json.dumps(execution))

        monkeypatch.setattr(
            "synthdata.evaluation.syntheval_eval.multiprocessing.get_context",
            lambda _method: FakeContext(),
        )
        monkeypatch.setattr("synthdata.evaluation.syntheval_eval._model_worker", fake_model_worker)
        cfg = FrameworkSelectionConfig(metrics=["dwm"])

        first = run_syntheval_evaluation(
            {"model_a": synthetic},
            dataset,
            cfg,
            preset_dir=tmp_path,
            output_folder=tmp_path / "main",
            return_execution=True,
        )
        sidecar_path = _checkpoint_paths(tmp_path / "main", "main", "model_a")[0] / "execution.json"
        first_sidecar = json.loads(sidecar_path.read_text())
        status_path = _checkpoint_paths(tmp_path / "main", "main", "model_a")[1]
        current_status = json.loads(status_path.read_text())
        tampered_sidecar = {**first_sidecar, "model_fingerprint": "stale-or-tampered"}
        sidecar_path.write_text(json.dumps(tampered_sidecar))

        second = run_syntheval_evaluation(
            {"model_a": synthetic},
            dataset,
            cfg,
            preset_dir=tmp_path,
            output_folder=tmp_path / "main",
            return_execution=True,
        )
        second_sidecar = json.loads(sidecar_path.read_text())

        assert worker_calls == ["model_a", "model_a"]
        assert json.loads(status_path.read_text()) == current_status
        assert tampered_sidecar["model_fingerprint"] != current_status["model_fingerprint"]
        assert second_sidecar["model_fingerprint"] == _frame_fingerprint(synthetic)
        assert second[2]["model_a"]["model_fingerprint"] == _frame_fingerprint(synthetic)
        assert first[2]["model_a"]["model_fingerprint"] == second[2]["model_a"]["model_fingerprint"]

    def test_main_final_holdout_requires_both_released_inputs(
        self, make_canonical_dataset, tmp_path
    ):
        dataset = make_canonical_dataset()
        with pytest.raises(ValueError, match="requires both released_synthetic_datasets"):
            run_syntheval_evaluation(
                {"model_a": dataset.role_frame("train", imputed=True)},
                dataset,
                FrameworkSelectionConfig(),
                preset_dir=tmp_path,
                evaluation_role="final_holdout",
            )

    def test_binary_final_holdout_requires_both_released_inputs(
        self, make_canonical_dataset, tmp_path
    ):
        dataset = make_canonical_dataset()
        with pytest.raises(ValueError, match="requires both released_synthetic_datasets"):
            run_binary_target_syntheval_evaluation(
                {"model_a": dataset.role_frame("train", imputed=True)},
                dataset,
                FrameworkSelectionConfig(),
                SimpleNamespace(column="target", positive_classes=[1], negative_classes=[0]),
                preset_dir=tmp_path,
                evaluation_role="final_holdout",
            )

    @pytest.mark.parametrize(
        "released_argument", ["released_synthetic_datasets", "released_final_holdout_frame"]
    )
    def test_main_released_inputs_must_be_paired(
        self, released_argument, make_canonical_dataset, tmp_path
    ):
        dataset = make_canonical_dataset()
        released_synthetic = {"model_a": dataset.role_frame("train", imputed=True)}
        released_holdout = dataset.role_frame("final_holdout", imputed=True)
        if released_argument == "released_synthetic_datasets":
            released_synthetic = None
        else:
            released_holdout = None

        with pytest.raises(ValueError, match="must be provided together"):
            run_syntheval_evaluation(
                {"model_a": dataset.role_frame("train", imputed=True)},
                dataset,
                FrameworkSelectionConfig(),
                preset_dir=tmp_path,
                evaluation_role="final_holdout",
                released_synthetic_datasets=released_synthetic,
                released_final_holdout_frame=released_holdout,
            )

    def test_binary_released_inputs_must_be_paired(self, make_canonical_dataset, tmp_path):
        dataset = make_canonical_dataset()
        with pytest.raises(ValueError, match="must be provided together"):
            run_binary_target_syntheval_evaluation(
                {"model_a": dataset.role_frame("train", imputed=True)},
                dataset,
                FrameworkSelectionConfig(),
                SimpleNamespace(column="target", positive_classes=[1], negative_classes=[0]),
                preset_dir=tmp_path,
                evaluation_role="final_holdout",
                released_final_holdout_frame=dataset.role_frame("final_holdout", imputed=True),
            )

    def test_final_holdout_has_a_distinct_evidence_context(self, make_canonical_dataset):
        dataset = make_canonical_dataset()
        train_for_tuning, tuning = _evaluation_role_frames(dataset, "tuning")
        train_for_final, final_holdout = _evaluation_role_frames(dataset, "final_holdout")

        candidate_context = _evaluation_context_fingerprint(
            dataset,
            {},
            "main",
            False,
            fit_frame=train_for_tuning,
            tuning_frame=tuning,
            evaluation_role="tuning",
        )
        final_context = _evaluation_context_fingerprint(
            dataset,
            {},
            "main",
            False,
            fit_frame=train_for_final,
            tuning_frame=final_holdout,
            evaluation_role="final_holdout",
        )

        assert train_for_tuning is train_for_final
        assert tuning is not final_holdout
        assert candidate_context != final_context

    def test_candidate_context_ignores_holdout_assignment_changes(self, make_canonical_dataset):
        dataset = make_canonical_dataset()
        train_frame, tuning = _evaluation_role_frames(dataset, "tuning")
        candidate_before = _evaluation_context_fingerprint(
            dataset,
            {},
            "main",
            False,
            fit_frame=train_frame,
            tuning_frame=tuning,
            evaluation_role="tuning",
        )

        dataset.assignment = dataset.assignment.copy()
        final_rows = dataset.assignment["role"].eq("final_holdout")
        dataset.assignment.loc[final_rows, "population_group_hash"] = "holdout-only-change"
        dataset.assignment_fingerprint = dataframe_fingerprint(dataset.assignment)

        candidate_after = _evaluation_context_fingerprint(
            dataset,
            {},
            "main",
            False,
            fit_frame=train_frame,
            tuning_frame=tuning,
            evaluation_role="tuning",
        )
        assert candidate_after == candidate_before

    @pytest.mark.parametrize(
        "values",
        [["never-seen"], [["never-seen"]], [{"secret": "never-seen"}]],
    )
    def test_unseen_categorical_values_are_safe_for_unhashable_values(self, values):
        fit = pd.DataFrame({"category": ["known"]})
        synthetic = pd.DataFrame({"category": pd.Series(values, dtype=object)})

        violations = _synthetic_unseen_categorical_values(synthetic, fit, ["category"])

        assert violations["category"]["count"] == 1
        assert violations["category"]["distinct_count"] == 1
        assert len(violations["category"]["value_digests"][0]) == 64
        assert "never-seen" not in json.dumps(violations)

    def test_unseen_vocabulary_uses_fit_frame_only(self):
        train = pd.DataFrame({"category": ["train-only"]})
        tuning = pd.DataFrame({"category": ["tuning-only"]})
        final = pd.DataFrame({"category": ["final-only"]})
        candidate = pd.DataFrame({"category": ["tuning-only"]})
        final_candidate = pd.DataFrame({"category": ["final-only"]})

        assert _synthetic_unseen_categorical_values(candidate, train, ["category"])
        assert _synthetic_unseen_categorical_values(
            final_candidate, pd.concat([train, tuning]), ["category"]
        )
        assert not _synthetic_unseen_categorical_values(
            tuning, pd.concat([train, tuning]), ["category"]
        )
        assert _synthetic_unseen_categorical_values(final, pd.concat([train, tuning]), ["category"])

    def test_semantic_context_changes_evaluation_cache_identity(self, make_canonical_dataset):
        dataset = make_canonical_dataset()
        train_frame, tuning = _evaluation_role_frames(dataset, "tuning")
        semantic_context = {
            "schema_version": "semantic-context-v1",
            "target_column": "target",
            "task_type": "classification",
            "feature_columns": ["feature", "protected"],
            "protected_columns": ["protected"],
            "quasi_identifier_columns": ["feature"],
            "feature_types": {
                "feature": "continuous",
                "protected": "categorical",
                "target": "categorical",
            },
            "source_table": {"feature": "measurements"},
        }
        first = _evaluation_context_fingerprint(
            dataset,
            {},
            "main",
            False,
            fit_frame=train_frame,
            tuning_frame=tuning,
            semantic_context=semantic_context,
        )
        changed_context = {**semantic_context, "quasi_identifier_columns": ["protected"]}

        assert semantic_context_digest(semantic_context) != semantic_context_digest(changed_context)
        assert (
            _evaluation_context_fingerprint(
                dataset,
                {},
                "main",
                False,
                fit_frame=train_frame,
                tuning_frame=tuning,
                semantic_context=changed_context,
            )
            != first
        )

    def test_unseen_synthetic_category_persists_failed_checkpoint_without_worker(
        self, make_canonical_dataset, tmp_path, monkeypatch
    ):
        dataset = make_canonical_dataset()
        fit_frame, tuning_frame = _evaluation_role_frames(dataset, "tuning")
        column = dataset.all_categorical_columns[0]
        synthetic = dataset.role_frame("train", imputed=True).copy()
        raw_value = "category-never-seen"
        synthetic[column] = raw_value

        def fail_if_worker_starts(_method):
            raise AssertionError("preprocessing rejection must skip worker creation")

        monkeypatch.setattr(
            "synthdata.evaluation.syntheval_eval.multiprocessing.get_context", fail_if_worker_starts
        )
        cfg = SynthEvalExecutionConfig(model_workers=1, max_model_workers=1, cores_per_model=1)
        _results, _ranks, executions = _run_resumable_syntheval(
            {"bad_model": synthetic},
            dataset,
            {},
            tmp_path / "preset.json",
            tmp_path / "checkpoints",
            "linear",
            cfg,
            "main",
            expected_output_manifest={"metric_method": ("metric_a",)},
            fit_frame=fit_frame,
            tuning_frame=tuning_frame,
            semantic_context={
                "schema_version": "semantic-context-v1",
                "task_type": "classification",
            },
        )

        payload = executions["bad_model"]
        assert payload["execution_succeeded"] is False
        assert payload["policy_eligible"] is False
        assert payload["semantic_context"] is not None
        assert payload["semantic_context_digest"] == semantic_context_digest(
            payload["semantic_context"]
        )
        assert payload["worker_exit"]["exception_type"] == ("SyntheticPreprocessingValidationError")
        assert payload["preprocessing_metadata"]["reason_code"] == "synthetic_unknown_category"
        assert payload["preprocessing_metadata"]["legacy_reason_code"] == (
            "synthetic_unseen_categorical_values"
        )
        assert payload["preprocessing_metadata"]["columns"][column]["count"] == len(synthetic)
        assert payload["preprocessing_metadata"]["failure_class"] == "synthetic_unknown_category"
        assert raw_value not in json.dumps(payload)
        model_dir, status_path, _result_path = _checkpoint_paths(
            tmp_path / "checkpoints", "main", "bad_model"
        )
        assert status_path.exists()
        assert (model_dir / "execution.json").exists()

    def test_failed_execution_rows_are_null_and_excluded_from_ranks(self):
        def payload(model_name, succeeded):
            return {
                "model_name": model_name,
                "execution_succeeded": succeeded,
                "policy_eligible": succeeded,
                "metric_executions": [
                    {
                        "status": {"expected_keys": ["metric_a"]},
                        "normalized_rows": (
                            [
                                {
                                    "metric": "metric_a",
                                    "dim": "u",
                                    "val": 0.5,
                                    "err": 0.0,
                                    "n_val": 0.5,
                                    "n_err": 0.0,
                                }
                            ]
                            if succeeded
                            else []
                        ),
                    }
                ],
            }

        results, ranks = build_syntheval_tables_from_executions(
            {"good": payload("good", True), "bad": payload("bad", False)},
            ["good", "bad"],
            "linear",
        )

        assert pd.notna(results.loc["good", ("metric_a", "value")])
        assert pd.isna(results.loc["bad", ("metric_a", "value")])
        assert pd.notna(ranks.loc["good", "metric_a"])
        assert pd.isna(ranks.loc["bad", "metric_a"])

    def test_all_failed_execution_tables_are_schema_compatible_and_null(self):
        payload = {
            "execution_succeeded": False,
            "policy_eligible": False,
            "metric_executions": [
                {"status": {"expected_keys": ["metric_a"]}, "normalized_rows": []}
            ],
        }

        results, ranks = build_syntheval_tables_from_executions({"bad": payload}, ["bad"], "linear")

        assert results.loc["bad"].isna().all()
        assert ranks.loc["bad"].isna().all()

    def test_partial_blocked_metric_keeps_supported_values_but_is_unranked(self):
        expected = {
            "supported": ["metric_a"],
            "holdout_sensitive": ["cls_acc"],
        }
        semantic_context = {
            "schema_version": "semantic-context-v1",
            "sensitive_columns": ["protected"],
        }

        def row(metric, value):
            return {
                "metric": metric,
                "dim": "u",
                "val": value,
                "err": 0.0,
                "n_val": value,
                "n_err": 0.0,
            }

        def status(method, key, state):
            values = {
                "method": method,
                "state": state,
                "expected_keys": [key],
                "observed_keys": [key] if state == "succeeded" else [],
                "completed_keys": [key] if state == "succeeded" else [],
                "failed_keys": [] if state == "succeeded" else [key],
                "missing_keys": [key] if state == "blocked" else [],
                "duplicate_keys": [],
                "non_finite_keys": [],
                "unexpected_keys": [],
            }
            if state == "blocked":
                values.update(
                    {
                        "exception_type": "RealHoldoutUnknownCategoryError",
                        "reason_code": "real_holdout_unknown_category",
                        "failure_class": "real_holdout_unknown_category",
                        "failure_reason": (
                            "Real holdout contains categories absent from train; "
                            "affected metric was blocked."
                        ),
                    }
                )
            return values

        partial = {
            "schema_version": "syntheval-execution-v1",
            "model_status": "partial",
            "incomplete_reasons": ["real_holdout_unknown_category"],
            "execution_complete": True,
            "execution_succeeded": False,
            "policy_eligible": False,
            "semantic_context": semantic_context,
            "semantic_context_digest": semantic_context_digest(semantic_context),
            "metric_executions": [
                {
                    "method": "supported",
                    "status": status("supported", "metric_a", "succeeded"),
                    "normalized_rows": [row("metric_a", 0.25)],
                    "normalized_rows_v2": [],
                },
                {
                    "method": "holdout_sensitive",
                    "status": status("holdout_sensitive", "cls_acc", "blocked"),
                    "normalized_rows": [],
                    "normalized_rows_v2": [],
                },
            ],
        }
        complete = {
            **partial,
            "model_status": "complete",
            "incomplete_reasons": [],
            "execution_succeeded": True,
            "policy_eligible": True,
            "metric_executions": [
                {
                    "method": "supported",
                    "status": status("supported", "metric_a", "succeeded"),
                    "normalized_rows": [row("metric_a", 0.75)],
                    "normalized_rows_v2": [],
                },
                {
                    "method": "holdout_sensitive",
                    "status": status("holdout_sensitive", "cls_acc", "succeeded"),
                    "normalized_rows": [row("cls_acc", 0.5)],
                    "normalized_rows_v2": [],
                },
            ],
        }

        assert _execution_payload_partial(partial, expected_manifest=expected)
        for field, value in (
            ("exception_type", "ValueError"),
            ("exception_type", None),
            ("failure_reason", "Another failure was recorded."),
            ("failure_class", "synthetic_unknown_category"),
        ):
            forged = {
                **partial,
                "metric_executions": [dict(item) for item in partial["metric_executions"]],
            }
            blocked = forged["metric_executions"][1]
            blocked["status"] = {**blocked["status"]}
            if value is None:
                blocked["status"].pop(field)
            else:
                blocked["status"][field] = value
            assert not _execution_payload_partial(forged, expected_manifest=expected)

        mismatched_semantic_context = {
            **partial,
            "semantic_context": {"sensitive_columns": ["changed"]},
        }
        assert not _execution_payload_partial(
            mismatched_semantic_context,
            expected_manifest=expected,
        )
        results, ranks = build_syntheval_tables_from_executions(
            {"partial": partial, "complete": complete}, ["partial", "complete"], "linear"
        )

        assert results.loc["partial", ("metric_a", "value")] == pytest.approx(0.25)
        assert pd.isna(results.loc["partial", ("cls_acc", "value")])
        assert pd.isna(ranks.loc["partial"]).all()
        assert pd.notna(ranks.loc["complete", "metric_a"])
        assert pd.notna(results.loc["complete", ("cls_acc", "value")])

    def test_mixed_child_failure_preserves_method_evidence_and_supported_values(
        self, make_canonical_dataset, monkeypatch, tmp_path
    ):
        import syntheval

        import synthdata.evaluation.syntheval_eval as syntheval_eval

        dataset = make_canonical_dataset()
        frame = dataset.role_frame("train", imputed=True).copy()
        sentinel = "private-category-error-detail"
        diagnostic_messages = []

        def capture_diagnostic(message, *args):
            diagnostic_messages.append(message % args if args else message)

        expected_manifest = {
            "supported": ("avg_dwm_diff",),
            "pca": ("pca",),
            "cls_acc": ("cls_acc",),
            "mia": ("mia",),
            "att_discl": ("att_discl",),
        }

        class MetricStatus:
            def __init__(self, method, key, state, exception_type=None, message=None):
                self.method = method
                self.state = state
                self.succeeded = state == "succeeded"
                self.exception_type = exception_type
                self.exception_message = message
                self.key = key

            def to_dict(self):
                payload = {
                    "method": self.method,
                    "state": self.state,
                    "expected_keys": [self.key],
                    "observed_keys": [self.key] if self.succeeded else [],
                    "completed_keys": [self.key] if self.succeeded else [],
                    "failed_keys": [] if self.succeeded else [self.key],
                    "missing_keys": [] if self.succeeded else [self.key],
                    "duplicate_keys": [],
                    "non_finite_keys": [],
                    "unexpected_keys": [],
                }
                if not self.succeeded:
                    payload.update(
                        {
                            "exception_type": self.exception_type,
                            "exception_message": self.exception_message,
                        }
                    )
                return payload

        def metric(method, key, state, rows, exception_type=None, message=None):
            return SimpleNamespace(
                method=method,
                status=MetricStatus(method, key, state, exception_type, message),
                normalized_rows=rows,
                normalized_rows_v2=[],
            )

        class FakeSynthEval:
            def __init__(self, *_args, **_kwargs):
                pass

            def evaluate(self, *_args, **kwargs):
                rows = [
                    {
                        "metric": "avg_dwm_diff",
                        "dim": "u",
                        "val": 0.75,
                        "err": 0.0,
                        "n_val": 0.75,
                        "n_err": 0.0,
                    }
                ]
                executions = [
                    metric("supported", "avg_dwm_diff", "succeeded", rows),
                    metric(
                        "pca",
                        "pca",
                        "failed",
                        [],
                        "ValueError",
                        f"metric failed for {sentinel}",
                    ),
                    *[
                        metric(
                            method,
                            method,
                            "blocked",
                            [],
                            "RealHoldoutUnknownCategoryError",
                            "Real holdout contains categorical values absent from train",
                        )
                        for method in ("cls_acc", "mia", "att_discl")
                    ],
                ]
                return SimpleNamespace(
                    schema_version="syntheval-execution-v1",
                    pass_id=kwargs["pass_id"],
                    target_view=kwargs["target_view"],
                    expected_manifest_digest=kwargs["expected_manifest_digest"],
                    execution_complete=True,
                    succeeded=False,
                    policy_eligible=False,
                    preprocessing_fingerprint="fit-fingerprint",
                    preprocessing_metadata={"fit_role": "train"},
                    normalized_table=pd.DataFrame(rows),
                    metric_executions=executions,
                )

        class FakeProcess:
            def __init__(self, target, args, **_kwargs):
                self.target = target
                self.args = args
                self.exitcode = None
                self.pid = 1

            def start(self):
                try:
                    self.target(*self.args)
                except RuntimeError:
                    self.exitcode = 1
                else:
                    self.exitcode = 0

            def is_alive(self):
                return False

            def join(self):
                return None

        class FakeContext:
            Process = FakeProcess
            Queue = staticmethod(queue.Queue)

        monkeypatch.setattr(syntheval, "AnalysisConfig", lambda **_kwargs: object())
        monkeypatch.setattr(syntheval, "SynthEval", FakeSynthEval)
        monkeypatch.setattr(syntheval_eval, "_shutdown_nested_joblib_executor", lambda: None)
        monkeypatch.setattr(
            syntheval_eval,
            "logger",
            SimpleNamespace(
                debug=capture_diagnostic,
                info=capture_diagnostic,
                warning=capture_diagnostic,
                error=capture_diagnostic,
            ),
        )
        monkeypatch.setattr(
            syntheval_eval.multiprocessing, "get_context", lambda _method: FakeContext()
        )

        results, ranks, executions = _run_resumable_syntheval(
            {"mixed_model": frame},
            dataset,
            {},
            tmp_path / "preset.json",
            tmp_path / "checkpoints",
            "linear",
            SynthEvalExecutionConfig(model_workers=1, max_model_workers=1, cores_per_model=1),
            "main",
            expected_output_manifest=expected_manifest,
        )

        model_dir, status_path, _result_path = _checkpoint_paths(
            tmp_path / "checkpoints", "main", "mixed_model"
        )
        worker_status = json.loads(status_path.read_text())
        payload = executions["mixed_model"]
        states = {item["method"]: item["status"]["state"] for item in payload["metric_executions"]}
        assert worker_status["pass_id"] == "main"
        assert worker_status["failure_reason"]
        assert payload["pass_id"] == "main"
        assert payload["model_status"] == "incomplete"
        assert payload["execution_complete"] is True
        assert payload["execution_succeeded"] is False
        assert payload["failure_reason"]
        assert states == {
            "supported": "succeeded",
            "pca": "failed",
            "cls_acc": "blocked",
            "mia": "blocked",
            "att_discl": "blocked",
        }
        status_by_method = {item["method"]: item["status"] for item in payload["metric_executions"]}
        assert status_by_method["pca"]["reason_code"] == "unknown_exception"
        assert len(status_by_method["pca"]["reason_detail_digest"]) == 64
        for method in ("cls_acc", "mia", "att_discl"):
            assert status_by_method[method]["reason_code"] == "real_holdout_unknown_category"
            assert status_by_method[method]["failed_keys"] == [method]
        assert status_by_method["pca"]["failed_keys"] == ["pca"]
        assert _execution_payload_failed(
            payload,
            expected_manifest=expected_manifest,
            expected_pass_id="main",
            expected_target_view="native",
        )
        assert results.loc["mixed_model", ("avg_dwm_diff", "value")] == pytest.approx(0.75)
        assert pd.isna(ranks.loc["mixed_model"]).all()
        saved = (model_dir / "execution.json").read_text()
        assert sentinel not in saved
        assert "private-category-error-detail" not in str(payload)
        assert any("state=execution_written rows=1" in item for item in diagnostic_messages)
        assert any("state=failed" in item for item in diagnostic_messages)
        assert any(
            "state=failure_evidence_written child_sidecar_present=True" in item
            for item in diagnostic_messages
        )
        assert any(
            "aggregate eligibility eligible=0 ineligible=1 total=1" in item
            for item in diagnostic_messages
        )
        assert sentinel not in "\n".join(diagnostic_messages)

    @pytest.mark.parametrize("interactive", [True, False])
    def test_model_supervisor_logs_progress_for_success_and_failure(
        self, interactive, make_canonical_dataset, monkeypatch, tmp_path
    ):
        import synthdata.evaluation.syntheval_eval as syntheval_eval

        dataset = make_canonical_dataset()
        frame = dataset.role_frame("train", imputed=True).copy()
        progress_instances = []
        log_messages = []
        progress_lines = []

        def capture_log(message, *args):
            log_messages.append(message % args if args else message)

        class Progress:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.updates = 0
                self.closed = False
                progress_instances.append(self)

            def update(self, amount):
                self.updates += amount

            def close(self):
                self.closed = True

            def write(self, message):
                progress_lines.append(message)

        monkeypatch.setattr(syntheval_eval, "tqdm", Progress)
        monkeypatch.setattr(
            syntheval_eval,
            "logger",
            SimpleNamespace(
                info=capture_log,
                warning=capture_log,
                error=capture_log,
                debug=capture_log,
            ),
        )
        monkeypatch.setattr(
            syntheval_eval,
            "sys",
            SimpleNamespace(stderr=SimpleNamespace(isatty=lambda: interactive)),
        )

        def fake_worker(*args):
            model_name = args[0]
            progress_queue = args[-1]
            progress_queue.put(
                {
                    "model_name": model_name,
                    "method": "supported",
                    "event": "started",
                    "duration_seconds": 0.0,
                    "outcome": "running",
                }
            )
            if model_name == "bad_model":
                progress_queue.put(
                    {
                        "model_name": model_name,
                        "method": "supported",
                        "event": "failure",
                        "duration_seconds": 0.25,
                        "outcome": "failed",
                        "failure_class": "ValueError",
                    }
                )
                raise RuntimeError("worker failure")
            progress_queue.put(
                {
                    "model_name": model_name,
                    "method": "supported",
                    "event": "completed",
                    "duration_seconds": 0.5,
                    "outcome": "succeeded",
                }
            )
            checkpoint_root = Path(args[9])
            pass_name = args[10]
            expected_manifest = args[11]
            target_view = args[12]
            expected_manifest_digest = args[13]
            context_fingerprint = args[14]
            model_fingerprint = args[15]
            model_dir, status_path, result_path = _checkpoint_paths(
                checkpoint_root, pass_name, model_name
            )
            model_dir.mkdir(parents=True, exist_ok=True)
            metric_keys = list(expected_manifest["supported"])
            execution = {
                "model_name": model_name,
                "model_status": "complete",
                "schema_version": "syntheval-execution-v1",
                "pass_id": pass_name,
                "target_view": target_view,
                "expected_manifest_digest": expected_manifest_digest,
                "context_fingerprint": context_fingerprint,
                "model_fingerprint": model_fingerprint,
                "execution_complete": True,
                "execution_succeeded": True,
                "policy_eligible": True,
                "metric_executions": [
                    {
                        "method": "supported",
                        "status": {
                            "method": "supported",
                            "state": "succeeded",
                            "expected_keys": metric_keys,
                            "observed_keys": metric_keys,
                            "completed_keys": metric_keys,
                            "failed_keys": [],
                            "missing_keys": [],
                            "duplicate_keys": [],
                            "non_finite_keys": [],
                            "unexpected_keys": [],
                        },
                        "normalized_rows": [
                            {
                                "metric": key,
                                "dim": "u",
                                "val": 0.5,
                                "err": 0.0,
                                "n_val": 0.5,
                                "n_err": 0.0,
                            }
                            for key in metric_keys
                        ],
                        "normalized_rows_v2": [],
                    }
                ],
            }
            status_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "state": "succeeded",
                        "model_name": model_name,
                        "context_fingerprint": context_fingerprint,
                        "model_fingerprint": model_fingerprint,
                        "expected_manifest_digest": expected_manifest_digest,
                        "plots_completed": False,
                    }
                )
            )
            pd.DataFrame().to_parquet(result_path)
            (model_dir / "execution.json").write_text(json.dumps(execution))

        class FakeProcess:
            next_pid = 100

            def __init__(self, target, args, **_kwargs):
                self.target = target
                self.args = args
                self.exitcode = None
                self.pid = FakeProcess.next_pid
                FakeProcess.next_pid += 1

            def start(self):
                try:
                    self.target(*self.args)
                except RuntimeError:
                    self.exitcode = 1
                else:
                    self.exitcode = 0

            def is_alive(self):
                return False

            def join(self):
                return None

        class FakeContext:
            Process = FakeProcess

        monkeypatch.setattr(
            syntheval_eval.multiprocessing, "get_context", lambda _method: FakeContext()
        )
        monkeypatch.setattr(syntheval_eval, "_model_worker", fake_worker)

        results, ranks, executions = _run_resumable_syntheval(
            {"good_model": frame, "bad_model": frame},
            dataset,
            {},
            tmp_path / "preset.json",
            tmp_path / "checkpoints",
            "linear",
            SynthEvalExecutionConfig(model_workers=1, max_model_workers=1, cores_per_model=1),
            "main",
            expected_output_manifest={"supported": ("metric_a",)},
        )

        output = "\n".join(log_messages)
        assert "model=good_model status=started progress=1/2" in output
        assert "model=good_model status=completed progress=1/2" in output
        assert "model=bad_model status=started progress=2/2" in output
        assert "model=bad_model status=failed progress=2/2" in output
        assert progress_instances[0].kwargs["total"] == 2
        assert progress_instances[0].kwargs["disable"] is (not interactive)
        assert progress_instances[0].updates == 2
        assert progress_instances[0].closed
        assert any(
            "model=good_model method=supported event=completed" in line for line in progress_lines
        )
        assert any(
            "model=bad_model method=supported event=failure" in line
            and "failure_class=ValueError" in line
            for line in progress_lines
        )
        assert any(
            "checkpoint validation pass=main model=good_model result=miss" in message
            for message in log_messages
        )
        assert any(
            "aggregate eligibility eligible=1 ineligible=1 total=2" in message
            for message in log_messages
        )
        assert "worker failure" not in "\n".join(progress_lines)
        assert "worker failure" not in "\n".join(log_messages)
        assert pd.notna(results.loc["good_model", ("metric_a", "value")])
        assert pd.isna(ranks.loc["bad_model"]).all()
        assert executions["bad_model"]["execution_succeeded"] is False

    @pytest.mark.parametrize(
        "failure_point",
        ["preflight_validation", "worker_resolution", "process_start", "checkpoint_validation"],
    )
    def test_model_supervisor_cleans_owned_resources_on_errors(
        self, failure_point, make_canonical_dataset, monkeypatch, tmp_path
    ):
        import synthdata.evaluation.syntheval_eval as syntheval_eval

        dataset = make_canonical_dataset()
        frame = dataset.role_frame("train", imputed=True).copy()
        process_instances = []
        progress_instances = []
        progress_queues = []
        checkpoint_calls = 0

        class Progress:
            def __init__(self, **_kwargs):
                self.closed = False
                self.close_calls = 0
                progress_instances.append(self)

            def update(self, _amount):
                pass

            def close(self):
                self.closed = True
                self.close_calls += 1

            def write(self, _message):
                pass

        class TrackedQueue(queue.Queue):
            def __init__(self):
                super().__init__()
                self.closed = False
                self.joined = False

            def close(self):
                self.closed = True

            def join_thread(self):
                self.joined = True

        def make_queue():
            created_queue = TrackedQueue()
            progress_queues.append(created_queue)
            return created_queue

        class FakeProcess:
            def __init__(self, target, args, **_kwargs):
                self.target = target
                self.args = args
                self.exitcode = None
                self.pid = 123
                self.alive = False
                self.terminated = False
                self.joined = False
                process_instances.append(self)

            def start(self):
                if failure_point == "process_start":
                    self.alive = True
                    raise RuntimeError("process start failed")
                self.exitcode = 0

            def is_alive(self):
                return self.alive

            def terminate(self):
                self.terminated = True
                self.alive = False

            def join(self):
                self.joined = True

        class FakeContext:
            Process = FakeProcess
            Queue = staticmethod(make_queue)

        monkeypatch.setattr(syntheval_eval, "tqdm", Progress)
        monkeypatch.setattr(
            syntheval_eval,
            "logger",
            SimpleNamespace(info=lambda *_args: None, debug=lambda *_args: None),
        )
        monkeypatch.setattr(
            syntheval_eval.multiprocessing, "get_context", lambda _method: FakeContext()
        )

        def checkpoint_validation(*_args, **_kwargs):
            nonlocal checkpoint_calls
            checkpoint_calls += 1
            if failure_point == "preflight_validation" or (
                failure_point == "checkpoint_validation" and checkpoint_calls == 2
            ):
                raise RuntimeError("checkpoint validation failed")
            return None

        monkeypatch.setattr(syntheval_eval, "_valid_checkpoint", checkpoint_validation)
        if failure_point == "worker_resolution":

            def fail_worker_resolution(*_args, **_kwargs):
                raise RuntimeError("worker resolution failed")

            monkeypatch.setattr(syntheval_eval, "resolve_model_workers", fail_worker_resolution)

        with pytest.raises(RuntimeError, match="failed"):
            _run_resumable_syntheval(
                {"model": frame},
                dataset,
                {},
                tmp_path / "preset.json",
                tmp_path / "checkpoints",
                "linear",
                SynthEvalExecutionConfig(model_workers=1, max_model_workers=1, cores_per_model=1),
                "main",
                expected_output_manifest={"method": ("metric",)},
            )

        resources_created = failure_point in {"process_start", "checkpoint_validation"}
        assert len(process_instances) == (1 if resources_created else 0)
        assert len(progress_queues) == (1 if resources_created else 0)
        if resources_created:
            process = process_instances[0]
            assert process.joined
            assert process.terminated is (failure_point == "process_start")
            assert progress_queues[0].closed
            assert progress_queues[0].joined
        assert progress_instances[0].closed
        assert progress_instances[0].close_calls == 1

    def test_preprocessing_contract_changes_context_identity(self, make_canonical_dataset):
        dataset = make_canonical_dataset()
        fit_frame, tuning_frame = _evaluation_role_frames(dataset, "tuning")
        first = _evaluation_context_fingerprint(
            dataset, {}, "main", False, fit_frame=fit_frame, tuning_frame=tuning_frame
        )
        altered = _evaluation_context_fingerprint(
            dataset,
            {"preprocessing_contract": "changed"},
            "main",
            False,
            fit_frame=fit_frame,
            tuning_frame=tuning_frame,
        )
        assert altered != first

    def test_main_resumable_evaluation_receives_semantic_context(
        self, make_canonical_dataset, monkeypatch, tmp_path
    ):
        dataset = make_canonical_dataset()
        semantic_context = {
            "schema_version": "semantic-context-v1",
            "task_type": "classification",
        }
        captured = {}

        monkeypatch.setattr(
            "synthdata.evaluation.syntheval_eval._load_syntheval_cache",
            lambda *args, **kwargs: None,
        )

        def fake_run_resumable(*args, **kwargs):
            captured["semantic_context"] = kwargs["semantic_context"]
            return pd.DataFrame(), pd.DataFrame(), {}

        monkeypatch.setattr(
            "synthdata.evaluation.syntheval_eval._run_resumable_syntheval",
            fake_run_resumable,
        )

        run_syntheval_evaluation(
            {"model_a": dataset.role_frame("train", imputed=True)},
            dataset,
            FrameworkSelectionConfig(),
            preset_dir=tmp_path,
            output_folder=tmp_path / "syntheval",
            semantic_context=semantic_context,
        )

        assert captured["semantic_context"] == semantic_context

    def test_final_holdout_is_rejected_for_legacy_dataset(self, make_dataset):
        dataset = make_dataset()
        dataset.train_imputed_df = dataset.train_df.copy()
        dataset.test_imputed_df = dataset.test_df.copy()

        with pytest.raises(RuntimeError, match="final-holdout evaluation"):
            _evaluation_role_frames(dataset, "final_holdout")


class TestBuildBinaryPreset:
    def _selection(self, **overrides) -> FrameworkSelectionConfig:
        return FrameworkSelectionConfig(**overrides)

    def test_default_selection_includes_only_binary_only_metrics(self):
        preset = build_binary_preset(self._selection())
        assert set(preset) == set(BINARY_ONLY_METRICS)

    def test_disabled_selection_returns_empty(self):
        preset = build_binary_preset(self._selection(enabled=False))
        assert preset == {}

    def test_explicit_metrics_filters_to_binary_only_subset(self):
        preset = build_binary_preset(self._selection(metrics=["auroc_diff", "cls_acc"]))
        # cls_acc is a valid metric name but not a BINARY_ONLY_METRICS one --
        # it must never show up in a binary-target-only preset.
        assert set(preset) == {"auroc_diff"}

    def test_category_selection_excludes_fairness_excludes_fairness_metrics(self):
        preset = build_binary_preset(self._selection(categories=["utility"]))
        assert set(preset) == {"auroc_diff"}


class TestSynthEvalExecutionManifest:
    def test_repaired_metrics_require_static_v2_rows(self):
        manifest = syntheval_execution_manifest(
            {"dwm": {}, "corr_diff": {}, "ks_test": {}},
            include_holdout_outputs=True,
        )

        assert manifest["dwm"] == ("avg_dwm_diff",)
        assert manifest["corr_diff"] == ("corr_mat_diff_v2",)
        assert manifest["ks_test"] == ("ks_tvd_stat_v2", "frac_ks_sigs_v2")

        assert syntheval_execution_keys_by_framework(manifest) == {
            "syntheval": [
                "avg_dwm_diff",
                "corr_mat_diff_v2",
                "ks_tvd_stat_v2",
                "frac_ks_sigs_v2",
            ],
            "custom": [],
        }

    def test_classification_manifest_tracks_resolved_v2_score_policy(self):
        macro = syntheval_execution_manifest(
            {"cls_acc": {"F1_type": "micro"}},
            include_holdout_outputs=True,
        )
        balanced = syntheval_execution_manifest(
            {"cls_acc": {"F1_type": "balanced_accuracy"}},
            include_holdout_outputs=False,
        )

        assert macro["cls_acc"] == (
            "avg_macro_F1_diff_v2",
            "avg_macro_F1_diff_v2_hout",
        )
        assert balanced["cls_acc"] == ("avg_balanced_accuracy_diff_v2",)

    def test_full_output_manifest_requires_declared_target_and_group_rows(self):
        manifest = syntheval_execution_manifest(
            {
                "statistical_parity": {"full_output": True},
                "equalized_odds": {"full_output": True},
                "equal_opportunity": {"full_output": True},
            },
            include_holdout_outputs=False,
            target_columns=["Target Label"],
            protected_columns=["Sex"],
        )

        assert manifest == {
            "statistical_parity": ("statistical_parity", "sp_target_label_Sex"),
            "equalized_odds": ("equalized_odds", "eqo_target_label_Sex"),
            "equal_opportunity": ("equal_opportunity", "eo_target_label_Sex"),
        }


class TestMergeBinaryTargetResults:
    @staticmethod
    def _comb_df(metric_values: dict, rank: float | list[float], index=("m1",)) -> pd.DataFrame:
        """Build a DataFrame matching SynthEval.benchmark()'s real comb_df structure:
        a (metric, 'value'/'error') MultiIndex for metrics, plus a scalar 'rank'
        column that pandas pads to ('rank', '') once the MultiIndex is set.
        """
        df = pd.DataFrame(index=pd.Index(index))
        for metric, (value, error) in metric_values.items():
            df[(metric, "value")] = value
            df[(metric, "error")] = error
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        df["rank"] = rank
        return df

    def test_both_none_returns_none(self):
        results, ranks = merge_binary_target_results(None, None, None, None)
        assert results is None
        assert ranks is None

    def test_binary_none_returns_main_unchanged(self):
        main_results = self._comb_df({"dwm": ([1.0], [0.1])}, rank=[0.9])
        main_ranks = pd.DataFrame({"dwm": [1.0], "rank": [0.9]}, index=_model_index("m1"))
        results, ranks = merge_binary_target_results(main_results, main_ranks, None, None)
        assert results is main_results
        assert ranks is main_ranks

    def test_main_none_returns_binary_unchanged(self):
        binary_results = self._comb_df({"auroc_diff": ([0.5], [0.05])}, rank=[0.3])
        binary_ranks = pd.DataFrame({"auroc_diff": [0.5], "rank": [0.3]}, index=_model_index("m1"))
        results, ranks = merge_binary_target_results(None, None, binary_results, binary_ranks)
        assert results is binary_results
        assert ranks is binary_ranks

    def test_merges_new_metric_columns_without_touching_existing(self):
        main_results = self._comb_df({"dwm": ([1.0], [0.1])}, rank=[0.9])
        main_ranks = pd.DataFrame({"dwm": [1.0], "rank": [0.9]}, index=_model_index("m1"))

        # rank=[0.3] here is the mini-benchmark's OWN aggregate rank (computed
        # from only auroc_diff) -- it must NOT overwrite the main pass's rank.
        binary_results = self._comb_df({"auroc_diff": ([0.5], [0.05])}, rank=[0.3])
        binary_ranks = pd.DataFrame({"auroc_diff": [0.7], "rank": [0.3]}, index=_model_index("m1"))

        results, ranks = merge_binary_target_results(
            main_results, main_ranks, binary_results, binary_ranks
        )
        assert results is not None
        assert ranks is not None

        assert results[("dwm", "value")].tolist() == [1.0]
        assert results[("auroc_diff", "value")].tolist() == [0.5]
        assert results[("auroc_diff", "error")].tolist() == [0.05]
        # The main pass's own aggregate rank must be untouched by the binary pass's.
        assert results["rank"].tolist() == [0.9]
        assert ranks["auroc_diff"].tolist() == [0.7]
        assert ranks["rank"].tolist() == [0.9]

    def test_native_metric_wins_when_binary_pass_collides(self):
        main_results = self._comb_df({"auroc": ([0.1], [0.01])}, rank=[0.9])
        main_ranks = pd.DataFrame({"auroc": [0.9], "rank": [0.9]}, index=_model_index("m1"))
        binary_results = self._comb_df({"auroc": ([0.6], [0.06])}, rank=[0.2])
        binary_ranks = pd.DataFrame({"auroc": [0.4], "rank": [0.2]}, index=_model_index("m1"))

        results, ranks = merge_binary_target_results(
            main_results, main_ranks, binary_results, binary_ranks
        )
        assert results is not None
        assert ranks is not None

        assert results[("auroc", "value")].tolist() == [0.1]
        assert results[("auroc", "error")].tolist() == [0.01]
        assert ranks["auroc"].tolist() == [0.9]


class TestSynthEvalMetricValidation:
    def test_native_cls_acc_cannot_satisfy_canonical_tstr(self):
        results = pd.DataFrame(index=_model_index("model_a"))
        results[("avg_macro_F1_diff_v2", "value")] = [0.2]
        results.columns = pd.MultiIndex.from_tuples(results.columns)
        ranks = pd.DataFrame({"avg_macro_F1_diff_v2": [0.8]}, index=_model_index("model_a"))

        validations = validate_syntheval_results(
            results,
            ranks,
            {"syntheval": ["tstr_macro_f1.v1"], "custom": []},
            role_hashes={"train": "train-hash", "tuning": "tuning-hash"},
            model_names=["model_a"],
            requested_use="hpo_objective",
        )

        validation = validations[("syntheval", "main")]["model_a"]
        assert validation.expected_records[0].status == "blocked"
        assert validation.expected_records[0].lifecycle_state == "operational"

    def test_canonical_tstr_requires_task10_producer_metadata(self):
        executions = {
            "model_a": {
                "pass_id": "main",
                "target_view": "native",
                "metric_executions": [
                    {
                        "method": "tstr",
                        "status": {"state": "succeeded", "failed_keys": []},
                        "normalized_rows_v2": [
                            {
                                "metric": "macro_f1",
                                "val": 0.8,
                                "n_val": 0.8,
                                "metadata": {"producer": "unrelated"},
                            }
                        ],
                    }
                ],
            }
        }
        observations = _structured_observations(
            executions, role_hashes={"train": "train-hash", "tuning": "tuning-hash"}
        )["model_a"]
        assert all(item.emitted_key != "tstr_macro_f1.v1" for item in observations)

    def test_pre_labeled_canonical_tstr_row_is_not_trusted(self):
        observations = _structured_observations(
            {
                "model_a": {
                    "pass_id": "main",
                    "target_view": "native",
                    "metric_executions": [
                        {
                            "method": "tstr",
                            "status": {"state": "succeeded", "failed_keys": []},
                            "normalized_rows_v2": [
                                {
                                    "metric": "tstr_macro_f1.v1",
                                    "val": 0.8,
                                    "n_val": 0.8,
                                    "result_metadata": {"producer": "task10_tstr"},
                                }
                            ],
                        }
                    ],
                }
            },
            role_hashes={"train": "train-hash", "tuning": "tuning-hash"},
        )["model_a"]
        assert all(item.emitted_key != "tstr_macro_f1.v1" for item in observations)

    def test_failed_execution_payload_retain_expected_keys_as_failure_evidence(self):
        payload = _failed_execution_payload(
            model_name="model_a",
            pass_name="main",
            target_view="native",
            expected_manifest_digest="manifest-hash",
            expected_output_manifest={"metric_method": ("metric_a",)},
            context_fingerprint="context-hash",
            role_context={},
            group_context=None,
            failure_status={"exit_code": 1, "failure_reason": "worker failed"},
        )
        assert _execution_payload_failed(
            payload,
            expected_manifest={"metric_method": ("metric_a",)},
            expected_pass_id="main",
            expected_target_view="native",
        )
        assert not _execution_payload_succeeded(payload)
        status = payload["metric_executions"][0]["status"]
        assert status["failed_keys"] == ["metric_a"]
        assert status["missing_keys"] == ["metric_a"]

    def test_mixed_framework_structured_observations_use_framework_role_hashes(self):
        executions = {
            "model_a": {
                "pass_id": "main",
                "target_view": "native",
                "metric_executions": [
                    {
                        "method": "fairness",
                        "status": {"state": "succeeded", "failed_keys": []},
                        "normalized_rows_v2": [
                            {"metric": "avg_dwm_diff", "val": 0.1, "n_val": 0.1},
                            {"metric": "equalized_odds", "val": 0.2, "n_val": 0.2},
                        ],
                    }
                ],
            }
        }
        validations = validate_syntheval_results(
            None,
            None,
            {"syntheval": ["avg_dwm_diff"], "custom": ["equalized_odds"]},
            role_hashes={"train": "train-imputed", "tuning": "tuning-imputed"},
            role_hashes_by_framework={
                "syntheval": {"train": "train-imputed", "tuning": "tuning-imputed"},
                "custom": {"train": "train-raw", "tuning": "tuning-raw"},
            },
            model_names=["model_a"],
            requested_use="audit",
            structured_executions=executions,
        )

        assert validations[("syntheval", "main")]["model_a"].evaluation_context.role_hashes == {
            "train": "train-imputed",
            "tuning": "tuning-imputed",
        }
        assert validations[("custom", "main")]["model_a"].evaluation_context.role_hashes == {
            "train": "train-raw",
            "tuning": "tuning-raw",
        }

    def test_known_qualified_diagnostics_remain_successful_at_root_boundary(self):
        execution = build_metric_execution(
            "equalized_odds",
            [
                {"metric": "equalized_odds", "val": 0.1},
                {"metric": "eqo_target_group", "val": 0.2},
            ],
            expected_keys=["equalized_odds"],
        )

        assert execution.status.succeeded is True

    def test_structured_tables_prefer_v2_values_over_legacy_rows(self):
        executions = {
            "model_a": {
                "metric_executions": [
                    {
                        "method": "corr_diff",
                        "status": {"expected_keys": ["corr_mat_diff_v2"]},
                        "normalized_rows": [
                            {
                                "metric": "corr_mat_diff",
                                "dim": "u",
                                "val": 0.9,
                                "err": 0.1,
                                "n_val": 0.1,
                                "n_err": 0.01,
                            }
                        ],
                        "normalized_rows_v2": [
                            {
                                "metric": "corr_mat_diff_v2",
                                "dim": "u",
                                "val": 0.2,
                                "err": 0.02,
                                "n_val": 0.8,
                                "n_err": 0.08,
                                "raw_value": 0.2,
                                "normalized_value": 0.8,
                            }
                        ],
                    }
                ]
            }
        }

        results, ranks = build_syntheval_tables_from_executions(
            executions,
            ["model_a"],
            "summation",
        )

        assert results.loc["model_a", ("corr_mat_diff_v2", "value")] == 0.2
        assert ranks.loc["model_a", "corr_mat_diff_v2"] == 0.8

    def test_selected_normalized_metric_missing_is_indeterminate(self):
        results = pd.DataFrame(index=_model_index("model_a"))
        results[("avg_dwm_diff", "value")] = [0.2]
        results.columns = pd.MultiIndex.from_tuples(results.columns)
        ranks = pd.DataFrame({"avg_dwm_diff": [0.8]}, index=_model_index("model_a"))
        expected = {
            "syntheval": ["avg_dwm_diff", "pca_eigval_diff"],
            "custom": [],
        }

        validations = validate_syntheval_results(
            results,
            ranks,
            expected,
            role_hashes={"train": "train-hash", "test": "test-hash"},
            model_names=["model_a"],
            requested_use="audit",
        )

        validation = validations[("syntheval", "main")]["model_a"]
        assert validation.expected_keys == ("avg_dwm_diff", "pca_eigval_diff")
        assert validation.completed_keys == ("avg_dwm_diff",)
        assert validation.indeterminate_keys == ("pca_eigval_diff",)
        assert validation.expected_records[1].status == "missing"
        assert validation.decision_status == "indeterminate"

    def test_observed_qualified_diagnostics_are_contract_checked(self):
        results = pd.DataFrame(index=_model_index("model_a"))
        results[("statistical_parity", "value")] = [0.1]
        results[("sp_target_sex", "value")] = [0.2]
        results.columns = pd.MultiIndex.from_tuples(results.columns)
        ranks = pd.DataFrame(
            {"statistical_parity": [0.9], "sp_target_sex": [0.8]}, index=_model_index("model_a")
        )
        expected = extend_syntheval_expected_diagnostics(
            {"syntheval": ["statistical_parity"], "custom": []},
            results,
        )

        validations = validate_syntheval_results(
            results,
            ranks,
            expected,
            role_hashes={"train": "train-hash", "test": "test-hash"},
            model_names=["model_a"],
            requested_use="audit",
        )

        validation = validations[("syntheval", "main")]["model_a"]
        assert "sp_target_sex" not in validation.expected_keys
        diagnostic_record = next(
            record for record in validation.records if record.expected_key == "sp_target_sex"
        )
        assert diagnostic_record.is_expected is False
        assert diagnostic_record.value_role == "diagnostic"
        assert diagnostic_record.status == "unexpected"

    def test_missing_declared_qualified_diagnostic_is_indeterminate(self):
        results = pd.DataFrame(index=_model_index("model_a"))
        results[("statistical_parity", "value")] = [0.1]
        results.columns = pd.MultiIndex.from_tuples(results.columns)
        ranks = pd.DataFrame({"statistical_parity": [0.9]}, index=_model_index("model_a"))
        expected = syntheval_execution_keys_by_framework(
            syntheval_execution_manifest(
                {"statistical_parity": {"full_output": True}},
                include_holdout_outputs=False,
                target_columns=["target"],
                protected_columns=["sex"],
            )
        )

        validations = validate_syntheval_results(
            results,
            ranks,
            expected,
            role_hashes={"train": "train-hash", "tuning": "tuning-hash"},
            model_names=["model_a"],
            requested_use="audit",
        )

        validation = validations[("syntheval", "main")]["model_a"]
        assert "sp_target_sex" in validation.expected_keys
        assert (
            next(
                record
                for record in validation.expected_records
                if record.expected_key == "sp_target_sex"
            ).status
            == "missing"
        )
        assert validation.complete is False

    def test_missing_qualified_diagnostics_do_not_change_static_expectations(self):
        expected = extend_syntheval_expected_diagnostics(
            {"syntheval": ["statistical_parity"], "custom": []},
            pd.DataFrame(),
            structured_executions={},
        )

        assert expected == {"syntheval": ["statistical_parity"], "custom": []}

    def test_structured_execution_rows_select_v2_and_preserve_failures(self):
        executions = {
            "model_a": {
                "model_name": "model_a",
                "pass_id": "main",
                "target_view": "native",
                "metric_executions": [
                    {
                        "method": "corr_diff",
                        "status": {
                            "state": "succeeded",
                            "expected_keys": ["corr_mat_diff_v2"],
                            "failed_keys": [],
                        },
                        "normalized_rows": [{"metric": "corr_mat_diff", "val": 0.9, "n_val": 0.1}],
                        "normalized_rows_v2": [
                            {
                                "metric": "corr_mat_diff_v2",
                                "val": 0.2,
                                "n_val": 0.8,
                                "raw_value": 0.2,
                                "normalized_value": 0.8,
                                "metric_version": "v2",
                            }
                        ],
                    },
                    {
                        "method": "ks_test",
                        "status": {
                            "state": "failed",
                            "expected_keys": ["ks_tvd_stat_v2"],
                            "failed_keys": ["ks_tvd_stat_v2"],
                            "exception_message": "invalid support",
                        },
                        "normalized_rows": [],
                        "normalized_rows_v2": [],
                    },
                ],
            }
        }
        expected = {"syntheval": ["corr_mat_diff_v2", "ks_tvd_stat_v2"], "custom": []}

        validations = validate_syntheval_results(
            pd.DataFrame(),
            None,
            expected,
            role_hashes={"train": "train-hash", "test": "test-hash"},
            model_names=["model_a"],
            requested_use="audit",
            structured_executions=executions,
        )

        validation = validations[("syntheval", "main")]["model_a"]
        assert validation.expected_records[0].raw_value == 0.2
        assert validation.expected_records[0].source_metadata["normalized_value"] == 0.8
        assert validation.expected_records[0].result_metadata == {}
        assert validation.expected_records[0].status == "succeeded"
        assert validation.expected_records[1].status == "failed"

    @pytest.mark.parametrize(
        ("emitted_key", "metadata", "expected_sample_size"),
        [
            ("corr_mat_diff_v2", {"valid_pairs": 7, "total_pairs": 8}, 7),
            ("mutual_inf_diff_v2", {"valid_pairs": 5, "total_pairs": 6}, 5),
            ("ks_tvd_stat_v2", {"valid_tests": 4, "support_size": 99}, 4),
            ("frac_ks_sigs_v2", {"valid_tests": 4, "support_size": 99}, 4),
            ("avg_h_dist_v2", {"valid_columns": 3, "support_size": 99}, 3),
            ("avg_qMSE_v2", {"valid_columns": 2, "support_size": 99}, 2),
            ("avg_pMSE_v2", {"oof_n": 12, "support_size": 99}, 12),
        ],
    )
    def test_structured_observations_extract_declared_support_not_normalized_score(
        self, emitted_key, metadata, expected_sample_size
    ):
        observations = _structured_observations(
            {
                "model_a": {
                    "pass_id": "main",
                    "target_view": "native",
                    "metric_executions": [
                        {
                            "method": "metric_method",
                            "status": {
                                "state": "succeeded",
                                "expected_keys": [emitted_key],
                                "failed_keys": [],
                            },
                            "normalized_rows_v2": [
                                {
                                    "metric": emitted_key,
                                    "val": 0.2,
                                    "err": 0.01,
                                    "n_val": 0.8,
                                    "n_err": 0.02,
                                    "raw_value": 0.2,
                                    "normalized_value": 0.8,
                                    "metric_version": "v2",
                                    "metadata": metadata,
                                }
                            ],
                        }
                    ],
                }
            },
            role_hashes={"train": "train-hash", "tuning": "tuning-hash"},
        )

        observation = observations["model_a"][0]
        assert observation.sample_size == expected_sample_size
        assert observation.uncertainty == pytest.approx(0.01)
        assert observation.source_metadata["normalized_value"] == 0.8
        assert observation.result_metadata == metadata
        assert observation.source_metadata["result_metadata"] == metadata

    def test_structured_observations_leave_unsupported_sample_size_missing(self):
        observations = _structured_observations(
            {
                "model_a": {
                    "pass_id": "main",
                    "target_view": "native",
                    "metric_executions": [
                        {
                            "method": "metric_method",
                            "status": {
                                "state": "succeeded",
                                "expected_keys": ["auroc_v2"],
                                "failed_keys": [],
                            },
                            "normalized_rows_v2": [
                                {
                                    "metric": "auroc_v2",
                                    "val": 0.2,
                                    "err": 0.01,
                                    "n_val": 0.8,
                                    "metadata": {"support_size": 99},
                                }
                            ],
                        }
                    ],
                }
            },
            role_hashes={},
        )

        assert observations["model_a"][0].sample_size is None

    def test_structured_observations_drop_non_finite_uncertainty(self):
        observations = _structured_observations(
            {
                "model_a": {
                    "pass_id": "main",
                    "target_view": "native",
                    "metric_executions": [
                        {
                            "method": "metric_method",
                            "status": {"state": "succeeded", "failed_keys": []},
                            "normalized_rows": [
                                {
                                    "metric": "mia_recall",
                                    "val": 0.2,
                                    "err": float("nan"),
                                    "n_val": 0.8,
                                }
                            ],
                        }
                    ],
                }
            },
            role_hashes={},
        )

        assert observations["model_a"][0].uncertainty is None

    def test_main_pass_wins_for_repeated_emitted_key(self):
        class Validation:
            def __init__(self, expected_keys, completed_keys=None):
                self.expected_keys = expected_keys
                self.completed_keys = expected_keys if completed_keys is None else completed_keys

        execution_passes = build_metric_execution_passes(
            cast(
                dict[tuple[str, str], dict[str, MetricValidationResult]],
                {
                    ("syntheval", "main"): {"model_a": Validation(("auroc", "avg_dwm_diff"))},
                    ("syntheval", "binary_target"): {"model_a": Validation(("auroc",))},
                },
            )
        )

        assert execution_passes[("syntheval", "auroc", "model_a")] == "main"
        assert execution_passes[("syntheval", "avg_dwm_diff", "model_a")] == "main"

    def test_execution_pass_ownership_is_model_specific(self):
        class Validation:
            def __init__(self, expected_keys, completed_keys=()):
                self.expected_keys = expected_keys
                self.completed_keys = completed_keys

        execution_passes = build_metric_execution_passes(
            cast(
                dict[tuple[str, str], dict[str, MetricValidationResult]],
                {
                    ("syntheval", "main"): {
                        "model_a": Validation(("auroc",), ("auroc",)),
                        "model_b": Validation(("auroc",)),
                    },
                    ("syntheval", "binary_target"): {
                        "model_a": Validation(("auroc",), ("auroc",)),
                        "model_b": Validation(("auroc",), ("auroc",)),
                    },
                },
            )
        )

        assert execution_passes[("syntheval", "auroc", "model_a")] == "main"
        assert execution_passes[("syntheval", "auroc", "model_b")] == "binary_target"

    def test_binary_pass_owns_repeated_key_when_main_is_missing(self):
        main_results = pd.DataFrame(index=_model_index("model_a"))
        main_validations = validate_syntheval_results(
            main_results,
            None,
            {"syntheval": ["auroc"]},
            role_hashes={"train": "train-hash", "test": "test-hash"},
            model_names=["model_a"],
            requested_use="audit",
        )

        binary_results = pd.DataFrame(index=_model_index("model_a"))
        binary_results[("auroc", "value")] = [0.6]
        binary_results.columns = pd.MultiIndex.from_tuples(binary_results.columns)
        binary_ranks = pd.DataFrame({"auroc": [0.4]}, index=_model_index("model_a"))
        binary_validations = validate_syntheval_results(
            binary_results,
            binary_ranks,
            {"syntheval": ["auroc"]},
            role_hashes={"train": "train-hash", "test": "test-hash"},
            model_names=["model_a"],
            execution_pass="binary_target",
            target_view="binary_collapsed",
            requested_use="audit",
        )

        assert main_validations[("syntheval", "main")]["model_a"].indeterminate_keys == ("auroc",)
        assert binary_validations[("syntheval", "binary_target")]["model_a"].completed_keys == (
            "auroc",
        )
        execution_passes = build_metric_execution_passes({**main_validations, **binary_validations})

        assert execution_passes[("syntheval", "auroc", "model_a")] == "binary_target"


class TestGroupContext:
    @staticmethod
    def _dataset(make_dataset):
        frame = pd.DataFrame(
            {
                "patient_id": [1, 1, 2, 2, 3, 3],
                "feature": [0, 1, 2, 3, 4, 5],
                "target": [0, 1, 0, 1, 0, 1],
            }
        )
        dataset = make_dataset(
            df=frame,
            feature_columns=["patient_id", "feature"],
        )
        dataset.train_imputed_df = frame.iloc[:4].copy()
        dataset.test_imputed_df = frame.iloc[4:].copy()
        return dataset

    def test_patient_group_context_records_role_fingerprints(self, make_dataset):
        dataset = self._dataset(make_dataset)
        synthetic = pd.DataFrame(
            {
                "patient_id": [10, 10, 11, 11],
                "feature": [0, 1, 2, 3],
                "target": [0, 1, 0, 1],
            }
        )

        context = build_group_context(
            dataset,
            {"model_a": synthetic},
            group_mode="patient_group",
            group_column="patient_id",
        )

        assert context["population_unit"] == "patient_group"
        assert context["roles"]["train"]["groups"] == 2
        assert context["roles"]["holdout"]["groups"] == 1
        assert len(context["roles"]["train"]["fingerprint"]) == 64
        assert context["models"]["model_a"]["groups"] == 2

    def test_candidate_group_context_excludes_final_holdout_metadata(self, make_canonical_dataset):
        dataset = make_canonical_dataset()
        synthetic = dataset.role_frame("train", imputed=True).copy()
        original_holdout_groups = dataset.role_groups["final_holdout"].copy()
        dataset.role_groups["final_holdout"] = dataset.role_groups["train"].copy()

        candidate_context = build_group_context(
            dataset,
            {"model_a": synthetic},
            group_mode="patient_group",
            group_column="patient_id",
            include_final_holdout=False,
        )

        assert set(candidate_context["roles"]) == {"train", "tuning"}
        dataset.role_groups["final_holdout"] = original_holdout_groups
        final_context = build_group_context(
            dataset,
            {"model_a": synthetic},
            group_mode="patient_group",
            group_column="patient_id",
        )
        assert set(final_context["roles"]) == {"train", "tuning", "final_holdout"}

    def test_patient_group_context_rejects_missing_synthetic_identifier(self, make_dataset):
        dataset = self._dataset(make_dataset)
        synthetic = dataset.train_imputed_df.drop(columns=["patient_id"])

        with pytest.raises(ValueError, match="missing from synthetic model 'model_a' frame"):
            build_group_context(
                dataset,
                {"model_a": synthetic},
                group_mode="patient_group",
                group_column="patient_id",
            )

    @pytest.mark.parametrize("role, attribute", [("train", "train"), ("holdout", "test")])
    def test_patient_group_context_rejects_missing_real_identifier(
        self, make_dataset, role, attribute
    ):
        dataset = self._dataset(make_dataset)
        frame = getattr(dataset, f"{attribute}_imputed_df").drop(columns=["patient_id"])
        setattr(dataset, f"{attribute}_imputed_df", frame)
        synthetic = dataset.train_imputed_df.copy()
        synthetic["patient_id"] = 10

        with pytest.raises(ValueError, match="missing from .* frame"):
            build_group_context(
                dataset,
                {"model_a": synthetic},
                group_mode="patient_group",
                group_column="patient_id",
            )

    def test_patient_group_context_rejects_null_identifier(self, make_dataset):
        dataset = self._dataset(make_dataset)
        dataset.train_imputed_df = dataset.train_imputed_df.copy()
        dataset.train_imputed_df.loc[dataset.train_imputed_df.index[0], "patient_id"] = None
        synthetic = dataset.train_imputed_df.copy()

        with pytest.raises(ValueError, match="contains missing values in train frame"):
            build_group_context(
                dataset,
                {"model_a": synthetic},
                group_mode="patient_group",
                group_column="patient_id",
            )

    def test_patient_group_context_rejects_train_holdout_overlap(self, make_dataset):
        dataset = self._dataset(make_dataset)
        dataset.test_imputed_df = dataset.test_imputed_df.copy()
        dataset.test_imputed_df.loc[:, "patient_id"] = [2, 3]
        synthetic = dataset.train_imputed_df.copy()

        with pytest.raises(ValueError, match="overlap between train and holdout"):
            build_group_context(
                dataset,
                {"model_a": synthetic},
                group_mode="patient_group",
                group_column="patient_id",
            )

    def test_patient_group_validation_marks_row_only_metric_unsafe(self):
        results = pd.DataFrame(index=_model_index("model_a"))
        results[("avg_dwm_diff", "value")] = [0.2]
        results.columns = pd.MultiIndex.from_tuples(results.columns)
        ranks = pd.DataFrame({"avg_dwm_diff": [0.8]}, index=_model_index("model_a"))

        validations = validate_syntheval_results(
            results,
            ranks,
            {"syntheval": ["avg_dwm_diff"], "custom": []},
            role_hashes={"train": "train-hash", "test": "test-hash"},
            model_names=["model_a"],
            requested_use="audit",
            population_unit="patient_group",
            group_mode="patient_group",
            resolved_configuration={"group_column": "patient_id"},
        )

        validation = validations[("syntheval", "main")]["model_a"]
        assert validation.expected_records[0].status == "group_unsafe"
        assert validation.decision_eligible is False
        context = validation.evaluation_context
        assert context is not None
        assert context.group_mode == "patient_group"
        assert context.resolved_configuration["group_column"] == "patient_id"

    @pytest.mark.parametrize(
        ("framework", "emitted_key"),
        [
            ("syntheval", "avg_dwm_diff"),
            ("syntheval", "mia_recall"),
            ("syntheval", "sp_target_sex"),
            ("custom", "eo_target_sex"),
        ],
    )
    def test_patient_group_validation_blocks_unsupported_metric_families(
        self, framework, emitted_key
    ):
        results = pd.DataFrame({(emitted_key, "value"): [0.2]}, index=_model_index("model_a"))
        results.columns = pd.MultiIndex.from_tuples(results.columns)
        ranks = pd.DataFrame({emitted_key: [0.2]}, index=_model_index("model_a"))

        validations = validate_syntheval_results(
            results,
            ranks,
            {
                "syntheval": [emitted_key] if framework == "syntheval" else [],
                "custom": [emitted_key] if framework == "custom" else [],
            },
            role_hashes={"train": "train-hash", "tuning": "tuning-hash"},
            model_names=["model_a"],
            requested_use="audit",
            population_unit="patient_group",
            group_mode="patient_group",
        )

        validation = validations[(framework, "main")]["model_a"]
        assert validation.expected_records[0].status == "group_unsafe"
        assert validation.decision_eligible is False

    def test_execution_sidecar_contains_group_context(self):
        execution = SimpleNamespace(
            schema_version="syntheval-execution-v1",
            pass_id="main",
            target_view="native",
            expected_manifest_digest="manifest-hash",
            execution_complete=True,
            policy_eligible=False,
            preprocessing_fingerprint="preprocessing-hash",
            preprocessing_metadata={"fit_role": "train"},
            metric_executions=(),
        )
        group_context = {"group_mode": "patient_group", "group_column": "patient_id"}
        semantic_context = {
            "schema_version": "semantic-context-v1",
            "target_column": "target",
            "task_type": "classification",
            "feature_columns": ["feature"],
            "protected_columns": [],
            "quasi_identifier_columns": [],
            "feature_types": {"feature": "continuous", "target": "categorical"},
            "source_table": {},
        }

        payload = _execution_sidecar_payload(
            execution,
            "model_a",
            group_context,
            semantic_context=semantic_context,
        )

        assert payload["model_name"] == "model_a"
        assert payload["group_context"] == group_context
        assert payload["preprocessing_fingerprint"] == "preprocessing-hash"
        assert payload["preprocessing_metadata"] == {"fit_role": "train"}
        assert payload["semantic_context"] == semantic_context
        assert payload["semantic_context_digest"] == semantic_context_digest(semantic_context)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _make_results(index=("m1", "m2")) -> pd.DataFrame:
    """Minimal benchmark_results DataFrame with a MultiIndex column level."""
    df = pd.DataFrame(index=pd.Index(index))
    df[("dwm", "value")] = [0.8, 0.7]
    df[("dwm", "error")] = [0.01, 0.02]
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df["rank"] = [0.9, 0.8]
    return df


def _make_ranks(index=("m1", "m2")) -> pd.DataFrame:
    return pd.DataFrame({"dwm": [0.8, 0.7], "rank": [0.9, 0.8]}, index=pd.Index(index))


class TestComputeCacheKey:
    def test_same_inputs_produce_same_key(self):
        preset = {"dwm": {}, "cls_acc": {}}
        k1 = _compute_cache_key(preset, ["m1", "m2"], "linear")
        k2 = _compute_cache_key(preset, ["m1", "m2"], "linear")
        assert k1 == k2

    def test_different_model_order_same_key(self):
        # Model names are sorted before hashing -- order must not matter.
        preset = {"dwm": {}}
        k1 = _compute_cache_key(preset, ["m1", "m2"], "linear")
        k2 = _compute_cache_key(preset, ["m2", "m1"], "linear")
        assert k1 == k2

    def test_different_models_different_key(self):
        preset = {"dwm": {}}
        k1 = _compute_cache_key(preset, ["m1"], "linear")
        k2 = _compute_cache_key(preset, ["m1", "m2"], "linear")
        assert k1 != k2

    def test_binary_cache_identity_uses_complete_model_inventory(self):
        preset = {"auroc_diff": {}}
        executable_models = ["valid_model"]
        all_models = ["valid_model", "invalid_model"]

        executable_key = _compute_cache_key(preset, executable_models, "linear")
        filtered_key = _compute_cache_key(preset, all_models, "linear")

        # Invalid preprocessing evidence remains part of binary aggregate-cache
        # identity, even though it is excluded from worker scheduling.
        assert executable_key != filtered_key

    def test_failed_binary_sibling_remains_failed_inventory_row(self):
        successful = {
            "model_name": "valid_model",
            "metric_executions": [
                {
                    "status": {"expected_keys": ["metric_a"]},
                    "normalized_rows": [
                        {
                            "metric": "metric_a",
                            "dim": "u",
                            "val": 0.5,
                            "err": None,
                            "n_val": 0.5,
                            "n_err": None,
                        }
                    ],
                }
            ],
        }
        failed = {
            "model_name": "invalid_model",
            "metric_executions": [
                {
                    "status": {
                        "expected_keys": ["metric_a"],
                        "failed_keys": ["metric_a"],
                    },
                    "normalized_rows": [],
                }
            ],
        }

        results, _ranks = build_syntheval_tables_from_executions(
            {"valid_model": successful, "invalid_model": failed},
            ["valid_model", "invalid_model"],
            "linear",
        )

        assert list(results.index) == ["valid_model", "invalid_model"]
        assert pd.isna(results.loc["invalid_model", ("metric_a", "value")])

    def test_different_preset_different_key(self):
        k1 = _compute_cache_key({"dwm": {}}, ["m1"], "linear")
        k2 = _compute_cache_key({"cls_acc": {}}, ["m1"], "linear")
        assert k1 != k2

    def test_different_ranking_strategy_different_key(self):
        preset = {"dwm": {}}
        k1 = _compute_cache_key(preset, ["m1"], "linear")
        k2 = _compute_cache_key(preset, ["m1"], "summation")
        assert k1 != k2

    def test_different_evaluation_fingerprint_different_key(self):
        k1 = _compute_cache_key({"dwm": {}}, ["m1"], "linear", "real-data-a")
        k2 = _compute_cache_key({"dwm": {}}, ["m1"], "linear", "real-data-b")
        assert k1 != k2

    def test_binary_failure_payload_changes_cache_identity(self):
        base = {"context": "same", "models": {"invalid": "frame"}}
        changed = {
            **base,
            "preprocessing_failures": {"invalid": "different-failure"},
        }
        k1 = _compute_cache_key(
            {"auroc_diff": {}}, ["valid", "invalid"], "linear", json.dumps(base, sort_keys=True)
        )
        k2 = _compute_cache_key(
            {"auroc_diff": {}},
            ["valid", "invalid"],
            "linear",
            json.dumps(changed, sort_keys=True),
        )
        assert k1 != k2

    def test_returns_hex_string(self):
        key = _compute_cache_key({"dwm": {}}, ["m1"], "linear")
        assert isinstance(key, str)
        int(key, 16)  # raises ValueError if not valid hex


class TestSaveLoadSynthevalCache:
    @pytest.mark.filterwarnings("error::FutureWarning")
    def test_binary_failed_model_survives_aggregate_cache_reload(self, tmp_path):
        manifest = {"metric_a": ["metric_a"]}
        failed = _failed_execution_payload(
            model_name="invalid_model",
            pass_name="binary_target",
            target_view="binary_collapsed",
            expected_manifest_digest="manifest-digest",
            expected_output_manifest=manifest,
            context_fingerprint="context",
            role_context={},
            group_context=None,
            failure_status={
                "exception_type": "SyntheticPreprocessingValidationError",
                "reason_code": "synthetic_binary_target_invalid",
            },
        )
        successful = {
            "model_name": "valid_model",
            "schema_version": "syntheval-execution-v1",
            "pass_id": "binary_target",
            "target_view": "binary_collapsed",
            "expected_manifest_digest": "manifest-digest",
            "context_fingerprint": "context",
            "execution_complete": True,
            "execution_succeeded": True,
            "policy_eligible": True,
            "metric_executions": [
                {
                    "method": "metric_a",
                    "status": {
                        "method": "metric_a",
                        "state": "succeeded",
                        "expected_keys": ["metric_a"],
                        "observed_keys": ["metric_a"],
                        "completed_keys": ["metric_a"],
                        "failed_keys": [],
                        "missing_keys": [],
                        "duplicate_keys": [],
                        "non_finite_keys": [],
                        "unexpected_keys": [],
                    },
                    "normalized_rows": [
                        {
                            "metric": "metric_a",
                            "dim": "u",
                            "val": 0.5,
                            "err": None,
                            "n_val": 0.5,
                            "n_err": None,
                        }
                    ],
                }
            ],
        }
        executions = {"valid_model": successful, "invalid_model": failed}
        results, ranks = build_syntheval_tables_from_executions(
            executions, ["valid_model", "invalid_model"], "linear"
        )
        key = _compute_cache_key({"auroc_diff": {}}, list(executions), "linear", "failure-bound")
        _save_syntheval_cache(results, ranks, tmp_path, "binary_target", key)
        for model_name, execution in executions.items():
            model_dir, _status_path, _result_path = _checkpoint_paths(
                tmp_path, "binary_target", model_name
            )
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "execution.json").write_text(json.dumps(execution))

        loaded = _load_syntheval_cache(tmp_path, "binary_target", key)
        assert loaded is not None
        loaded_executions = _load_syntheval_execution_sidecars(
            tmp_path,
            "binary_target",
            ["valid_model", "invalid_model"],
            "manifest-digest",
            "context",
            expected_manifest=manifest,
            expected_target_view="binary_collapsed",
        )
        assert loaded_executions is not None
        assert loaded_executions["valid_model"]["execution_succeeded"] is not False
        assert loaded_executions["invalid_model"]["execution_succeeded"] is False
        assert loaded_executions["invalid_model"]["policy_eligible"] is False
        validated = _validated_cached_syntheval_tables(
            loaded[0],
            loaded[1],
            executions,
            ["valid_model", "invalid_model"],
            "linear",
            "binary-target",
        )
        assert validated is not None
        assert list(validated[0].index) == ["valid_model", "invalid_model"]
        assert pd.isna(validated[0].loc["invalid_model", ("metric_a", "value")])
        assert pd.isna(validated[1].loc["invalid_model", "metric_a"])

        def validate_cached_tables(results_to_check):
            return _validated_cached_syntheval_tables(
                results_to_check,
                loaded[1],
                executions,
                ["valid_model", "invalid_model"],
                "linear",
                "binary-target",
            )

        paired_none_nan = next(
            (
                (row, column)
                for row in range(results.shape[0])
                for column in range(results.shape[1])
                if (
                    results.iat[row, column] is None
                    and isinstance(loaded[0].iat[row, column], (float, np.floating))
                    and np.isnan(loaded[0].iat[row, column])
                )
                or (
                    loaded[0].iat[row, column] is None
                    and isinstance(results.iat[row, column], (float, np.floating))
                    and np.isnan(results.iat[row, column])
                )
            ),
            None,
        )
        assert paired_none_nan is not None

        one_sided_missing = loaded[0].copy()
        one_sided_missing.iat[paired_none_nan[0], paired_none_nan[1]] = 0.25
        assert validate_cached_tables(one_sided_missing) is None

        other_missing_type = loaded[0].astype(object)
        other_missing_type.iat[paired_none_nan[0], paired_none_nan[1]] = pd.NA
        assert validate_cached_tables(other_missing_type) is None

        non_null_mismatch = loaded[0].copy()
        non_null_mismatch.loc["valid_model", ("metric_a", "value")] = 0.25
        assert validate_cached_tables(non_null_mismatch) is None

        changed_row_label = loaded[0].copy()
        changed_row_label.index = ["renamed_model", "invalid_model"]
        assert validate_cached_tables(changed_row_label) is None
        assert validate_cached_tables(loaded[0].iloc[::-1]) is None

        changed_column_labels = list(loaded[0].columns)
        changed_column_labels[0] = ("renamed_metric", changed_column_labels[0][1])
        changed_columns = loaded[0].copy()
        changed_columns.columns = pd.MultiIndex.from_tuples(
            changed_column_labels, names=loaded[0].columns.names
        )
        assert validate_cached_tables(changed_columns) is None
        assert validate_cached_tables(loaded[0].iloc[:, ::-1]) is None

        tampered_key = _compute_cache_key(
            {"auroc_diff": {}}, ["valid_model"], "linear", "failure-bound"
        )
        assert _load_syntheval_cache(tmp_path, "binary_target", tampered_key) is None

    def test_roundtrip_results_and_ranks(self, tmp_path):
        results = _make_results()
        ranks = _make_ranks()
        key = _compute_cache_key({"dwm": {}}, ["m1", "m2"], "linear")
        _save_syntheval_cache(results, ranks, tmp_path, "main", key)

        loaded = _load_syntheval_cache(tmp_path, "main", key)
        assert loaded is not None
        loaded_results, loaded_ranks = loaded
        pd.testing.assert_frame_equal(loaded_results, results)
        pd.testing.assert_frame_equal(loaded_ranks, ranks)

    def test_cache_miss_when_no_files(self, tmp_path):
        key = _compute_cache_key({"dwm": {}}, ["m1"], "linear")
        assert _load_syntheval_cache(tmp_path, "main", key) is None

    def test_cache_miss_when_key_changed(self, tmp_path):
        results, ranks = _make_results(), _make_ranks()
        old_key = _compute_cache_key({"dwm": {}}, ["m1", "m2"], "linear")
        new_key = _compute_cache_key({"dwm": {}}, ["m1", "m2", "m3"], "linear")
        _save_syntheval_cache(results, ranks, tmp_path, "main", old_key)
        assert _load_syntheval_cache(tmp_path, "main", new_key) is None

    def test_cache_miss_when_meta_corrupted(self, tmp_path):
        results, ranks = _make_results(), _make_ranks()
        key = _compute_cache_key({"dwm": {}}, ["m1", "m2"], "linear")
        _save_syntheval_cache(results, ranks, tmp_path, "main", key)
        (tmp_path / "main_cache_meta.json").write_text("not json")
        assert _load_syntheval_cache(tmp_path, "main", key) is None

    def test_cache_miss_when_results_parquet_missing(self, tmp_path):
        results, ranks = _make_results(), _make_ranks()
        key = _compute_cache_key({"dwm": {}}, ["m1", "m2"], "linear")
        _save_syntheval_cache(results, ranks, tmp_path, "main", key)
        (tmp_path / "main_results.parquet").unlink()
        assert _load_syntheval_cache(tmp_path, "main", key) is None

    def test_separate_prefixes_do_not_collide(self, tmp_path):
        results_a = _make_results(index=["a1", "a2"])
        ranks_a = _make_ranks(index=["a1", "a2"])
        results_b = _make_results(index=["b1", "b2"])
        ranks_b = _make_ranks(index=["b1", "b2"])
        key = _compute_cache_key({"dwm": {}}, ["m1", "m2"], "linear")

        _save_syntheval_cache(results_a, ranks_a, tmp_path, "main", key)
        _save_syntheval_cache(results_b, ranks_b, tmp_path, "binary_target", key)

        loaded_a = _load_syntheval_cache(tmp_path, "main", key)
        loaded_b = _load_syntheval_cache(tmp_path, "binary_target", key)
        assert loaded_a is not None and loaded_b is not None
        assert list(loaded_a[0].index) == ["a1", "a2"]
        assert list(loaded_b[0].index) == ["b1", "b2"]

    def test_meta_json_contains_cache_key(self, tmp_path):
        results, ranks = _make_results(), _make_ranks()
        key = _compute_cache_key({"dwm": {}}, ["m1", "m2"], "linear")
        _save_syntheval_cache(results, ranks, tmp_path, "main", key)
        meta = json.loads((tmp_path / "main_cache_meta.json").read_text())
        assert meta["cache_key"] == key


class TestResolveModelWorkers:
    def test_explicit_limit_obeys_model_and_max_bounds(self):
        cfg = SynthEvalExecutionConfig(model_workers=10, max_model_workers=4)
        assert resolve_model_workers(cfg, n_models=3, n_columns=10) == 3

    def test_auto_uses_memory_and_cpu_bounds(self, monkeypatch):
        cfg = SynthEvalExecutionConfig(
            model_workers="auto",
            max_model_workers=8,
            cores_per_model=4,
            memory_reserve_gib=16,
        )
        monkeypatch.setattr("synthdata.evaluation.syntheval_eval.os.cpu_count", lambda: 24)
        monkeypatch.setattr(
            "synthdata.evaluation.syntheval_eval._available_memory_gib", lambda: 118.0
        )
        assert resolve_model_workers(cfg, n_models=18, n_columns=1038) == 6

    def test_auto_rejects_memory_budget_that_cannot_fit_one_model(self, monkeypatch):
        cfg = SynthEvalExecutionConfig(
            model_workers="auto",
            max_model_workers=6,
            cores_per_model=4,
            memory_reserve_gib=16,
            memory_per_model_gib=14,
        )
        monkeypatch.setattr("synthdata.evaluation.syntheval_eval.os.cpu_count", lambda: 24)
        monkeypatch.setattr(
            "synthdata.evaluation.syntheval_eval._available_memory_gib", lambda: 29.9
        )

        with pytest.raises(ValueError, match="cannot fit one model") as exc_info:
            resolve_model_workers(cfg, n_models=18, n_columns=664)

        assert "MemAvailable=29.90 GiB" in str(exc_info.value)
        assert "reserve=16.00 GiB" in str(exc_info.value)
        assert "per-model estimate=14.00 GiB" in str(exc_info.value)

    @pytest.mark.parametrize("cpu_count", [1, 2, 3])
    def test_auto_rejects_cpu_budget_below_cores_per_model(self, monkeypatch, cpu_count):
        cfg = SynthEvalExecutionConfig(
            model_workers="auto",
            max_model_workers=6,
            cores_per_model=4,
            memory_reserve_gib=16,
            memory_per_model_gib=14,
        )
        monkeypatch.setattr("synthdata.evaluation.syntheval_eval.os.cpu_count", lambda: cpu_count)
        monkeypatch.setattr(
            "synthdata.evaluation.syntheval_eval._available_memory_gib", lambda: 200.0
        )

        with pytest.raises(ValueError, match="cannot satisfy the per-model CPU budget") as exc_info:
            resolve_model_workers(cfg, n_models=18, n_columns=664)

        assert f"available CPUs={cpu_count}" in str(exc_info.value)
        assert "cores per model=4" in str(exc_info.value)

    def test_cpu_preflight_fails_before_worker_context_creation(
        self, make_canonical_dataset, monkeypatch, tmp_path
    ):
        import synthdata.evaluation.syntheval_eval as syntheval_eval

        dataset = make_canonical_dataset()
        frame = dataset.role_frame("train", imputed=True).copy()
        monkeypatch.setattr(syntheval_eval.os, "cpu_count", lambda: 3)
        monkeypatch.setattr(syntheval_eval, "_available_memory_gib", lambda: 200.0)

        def fail_worker_context(*_args, **_kwargs):
            raise AssertionError("worker context must not be created before CPU preflight")

        monkeypatch.setattr(syntheval_eval.multiprocessing, "get_context", fail_worker_context)

        with pytest.raises(InsufficientSynthEvalCPUError):
            _run_resumable_syntheval(
                {"model": frame},
                dataset,
                {},
                tmp_path / "preset.json",
                tmp_path / "checkpoints",
                "linear",
                SynthEvalExecutionConfig(
                    model_workers="auto",
                    max_model_workers=6,
                    cores_per_model=4,
                    memory_reserve_gib=16,
                    memory_per_model_gib=14,
                ),
                "main",
                expected_output_manifest={"method": ("metric",)},
            )

    def test_auto_does_not_read_total_memory_after_available_memory(self, monkeypatch):
        cfg = SynthEvalExecutionConfig(
            model_workers="auto",
            max_model_workers=8,
            cores_per_model=4,
            memory_reserve_gib=16,
        )
        monkeypatch.setattr("synthdata.evaluation.syntheval_eval.os.cpu_count", lambda: 24)
        monkeypatch.setattr(
            "synthdata.evaluation.syntheval_eval._available_memory_gib", lambda: 118.0
        )

        def fail_read(_path: Path) -> bool:
            raise AssertionError("worker resolution must not read MemTotal")

        monkeypatch.setattr("synthdata.evaluation.syntheval_eval.Path.read_text", fail_read)

        assert resolve_model_workers(cfg, n_models=18, n_columns=1038) == 6


class TestCheckpointPaths:
    @staticmethod
    def _execution_payload(state="succeeded"):
        failed = [] if state == "succeeded" else ["metric_a"]
        rows = (
            [
                {
                    "metric": "metric_a",
                    "dim": "u",
                    "val": 0.5,
                    "err": None,
                    "n_val": 0.5,
                    "n_err": None,
                }
            ]
            if not failed
            else []
        )
        return {
            "schema_version": "syntheval-execution-v1",
            "pass_id": "main",
            "target_view": "native",
            "execution_complete": True,
            "execution_succeeded": state == "succeeded",
            "metric_executions": [
                {
                    "method": "metric_method",
                    "status": {
                        "method": "metric_method",
                        "state": state,
                        "expected_keys": ["metric_a"],
                        "observed_keys": [] if failed else ["metric_a"],
                        "completed_keys": [] if failed else ["metric_a"],
                        "failed_keys": failed,
                        "missing_keys": [],
                        "duplicate_keys": [],
                        "non_finite_keys": [],
                        "unexpected_keys": [],
                    },
                    "normalized_rows": rows,
                    "normalized_rows_v2": [],
                }
            ],
        }

    def test_execution_payload_requires_all_metrics_to_succeed(self):
        assert _execution_payload_succeeded(self._execution_payload()) is True
        assert _execution_payload_succeeded(self._execution_payload("failed")) is False

    def test_execution_payload_requires_structured_rows(self):
        payload = self._execution_payload()
        payload["metric_executions"][0]["normalized_rows"] = []

        assert _execution_payload_succeeded(payload) is False

    def test_execution_payload_rejects_non_finite_structured_values(self):
        payload = self._execution_payload()
        payload["metric_executions"][0]["normalized_rows"][0]["n_val"] = float("nan")

        assert _execution_payload_succeeded(payload) is False

    def test_execution_payload_binds_expected_manifest(self):
        payload = self._execution_payload()

        assert (
            _execution_payload_succeeded(
                payload,
                expected_manifest={"metric_method": ("metric_a",)},
                expected_pass_id="main",
                expected_target_view="native",
            )
            is True
        )
        assert (
            _execution_payload_succeeded(
                payload,
                expected_manifest={"metric_method": ("other_metric",)},
                expected_pass_id="main",
                expected_target_view="native",
            )
            is False
        )

    def test_resumable_run_returns_failed_model_evidence_with_cached_models(
        self, tmp_path, make_canonical_dataset, monkeypatch
    ):
        dataset = make_canonical_dataset()
        fit_frame, tuning_frame = _evaluation_role_frames(dataset, "tuning")
        manifest = {"metric_method": ("metric_a",)}
        context_fingerprint = _evaluation_context_fingerprint(
            dataset,
            {},
            "main",
            False,
            expected_output_manifest=manifest,
            fit_frame=fit_frame,
            tuning_frame=tuning_frame,
        )
        expected_manifest_digest = "manifest-hash"
        checkpoint_root = tmp_path / "checkpoints"
        cached_dir, cached_status_path, cached_result_path = _checkpoint_paths(
            checkpoint_root, "main", "model_cached"
        )
        cached_dir.mkdir(parents=True)
        cached_frame = pd.DataFrame({"metric": ["metric_a"], "val": [0.5]})
        _atomic_parquet(cached_result_path, cached_frame)
        cached_status_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "state": "succeeded",
                    "model_name": "model_cached",
                    "context_fingerprint": context_fingerprint,
                    "model_fingerprint": _frame_fingerprint(pd.DataFrame({"metric": ["cached"]})),
                    "expected_manifest_digest": expected_manifest_digest,
                    "plots_completed": False,
                }
            )
        )
        cached_execution = self._execution_payload()
        cached_execution.update(
            {
                "model_name": "model_cached",
                "context_fingerprint": context_fingerprint,
                "expected_manifest_digest": expected_manifest_digest,
                "model_fingerprint": _frame_fingerprint(pd.DataFrame({"metric": ["cached"]})),
            }
        )
        (cached_dir / "execution.json").write_text(json.dumps(cached_execution))

        class FailedProcess:
            pid = 123
            exitcode = 1

            def start(self):
                return None

            def is_alive(self):
                return False

            def join(self):
                return None

        class FailedContext:
            def Process(self, **_kwargs):
                return FailedProcess()

        monkeypatch.setattr(
            "synthdata.evaluation.syntheval_eval.multiprocessing.get_context",
            lambda _method: FailedContext(),
        )
        cfg = SynthEvalExecutionConfig(model_workers=1, max_model_workers=1, cores_per_model=1)
        synthetic_datasets = {
            "model_cached": pd.DataFrame({"metric": ["cached"]}),
            "model_failed": pd.DataFrame({"metric": ["failed"]}),
        }

        results, _ranks, executions = _run_resumable_syntheval(
            synthetic_datasets,
            dataset,
            {},
            tmp_path / "preset.json",
            checkpoint_root,
            "linear",
            cfg,
            "main",
            expected_output_manifest=manifest,
            expected_manifest_digest=expected_manifest_digest,
            fit_frame=fit_frame,
            tuning_frame=tuning_frame,
        )

        assert set(executions) == {"model_cached", "model_failed"}
        assert executions["model_failed"]["execution_succeeded"] is False
        assert executions["model_failed"]["metric_executions"][0]["status"]["failed_keys"] == [
            "metric_a"
        ]
        assert results.loc["model_cached", ("metric_a", "value")] == 0.5
        assert pd.isna(results.loc["model_failed", ("metric_a", "value")])

    def test_resumable_run_preserves_partial_child_execution_evidence(
        self, tmp_path, make_canonical_dataset, monkeypatch
    ):
        dataset = make_canonical_dataset()
        fit_frame, tuning_frame = _evaluation_role_frames(dataset, "tuning")
        manifest = {
            "statistics": ("avg_dwm_diff",),
            "ks_test": ("ks_tvd_stat_v2",),
        }
        context_fingerprint = _evaluation_context_fingerprint(
            dataset,
            {},
            "main",
            False,
            expected_output_manifest=manifest,
            fit_frame=fit_frame,
            tuning_frame=tuning_frame,
        )
        expected_manifest_digest = "manifest-hash"
        checkpoint_root = tmp_path / "checkpoints"
        model_dir, _status_path, _result_path = _checkpoint_paths(
            checkpoint_root, "main", "model_partial"
        )
        model_dir.mkdir(parents=True)

        successful = build_metric_execution(
            "statistics",
            [
                {
                    "metric": "avg_dwm_diff",
                    "dim": "u",
                    "val": 0.2,
                    "err": 0.01,
                    "n_val": 0.8,
                    "n_err": 0.02,
                }
            ],
            expected_keys=manifest["statistics"],
        )
        failed = build_metric_execution(
            "ks_test",
            None,
            expected_keys=manifest["ks_test"],
            error=RuntimeError("invalid support"),
        )
        partial_execution = SimpleNamespace(
            schema_version="syntheval-execution-v1",
            pass_id="main",
            target_view="native",
            expected_manifest_digest=expected_manifest_digest,
            execution_complete=True,
            succeeded=False,
            policy_eligible=False,
            preprocessing_fingerprint="preprocessing-hash",
            preprocessing_metadata={},
            metric_executions=(successful, failed),
        )
        (model_dir / "execution.json").write_text(
            json.dumps(
                _execution_sidecar_payload(
                    partial_execution,
                    "model_partial",
                    context_fingerprint=context_fingerprint,
                    role_context={},
                    model_fingerprint=_frame_fingerprint(pd.DataFrame({"metric": ["partial"]})),
                )
            )
        )

        class FailedProcess:
            pid = 123
            exitcode = 1

            def start(self):
                return None

            def is_alive(self):
                return False

            def join(self):
                return None

        class FailedContext:
            def Process(self, **_kwargs):
                return FailedProcess()

        monkeypatch.setattr(
            "synthdata.evaluation.syntheval_eval.multiprocessing.get_context",
            lambda _method: FailedContext(),
        )
        cfg = SynthEvalExecutionConfig(model_workers=1, max_model_workers=1, cores_per_model=1)

        results, ranks, executions = _run_resumable_syntheval(
            {"model_partial": pd.DataFrame({"metric": ["partial"]})},
            dataset,
            {},
            tmp_path / "preset.json",
            checkpoint_root,
            "summation",
            cfg,
            "main",
            expected_output_manifest=manifest,
            expected_manifest_digest=expected_manifest_digest,
            fit_frame=fit_frame,
            tuning_frame=tuning_frame,
        )

        payload = executions["model_partial"]
        assert _execution_payload_failed(
            payload,
            expected_manifest=manifest,
            expected_pass_id="main",
            expected_target_view="native",
        )
        assert not _execution_payload_succeeded(payload)
        assert payload["worker_exit"]["exit_code"] == 1
        methods = {item["method"]: item for item in payload["metric_executions"]}
        assert methods["statistics"]["status"]["state"] == "failed"
        assert methods["statistics"]["normalized_rows"] == []
        assert methods["statistics"]["normalized_rows_v2"] == []
        assert methods["ks_test"]["status"]["failed_keys"] == ["ks_tvd_stat_v2"]
        assert pd.isna(results.loc["model_partial", ("avg_dwm_diff", "value")])
        assert pd.isna(results.loc["model_partial", ("ks_tvd_stat_v2", "value")])

        observations = _structured_observations(executions, role_hashes={})
        observed = {item.emitted_key: item for item in observations["model_partial"]}
        assert observed["avg_dwm_diff"].raw_value is None
        assert observed["ks_tvd_stat_v2"].error == "SynthEval worker failed with an unknown error."

        validations = validate_syntheval_results(
            results,
            ranks,
            {"syntheval": ["avg_dwm_diff", "ks_tvd_stat_v2"], "custom": []},
            role_hashes={},
            model_names=["model_partial"],
            requested_use="policy_rank",
            structured_executions=executions,
        )
        validation = validations[("syntheval", "main")]["model_partial"]
        assert validation.complete is False
        assert validation.decision_eligible is False

    def test_worker_failure_replaces_stale_child_execution_evidence(
        self, tmp_path, make_canonical_dataset, monkeypatch
    ):
        dataset = make_canonical_dataset()
        fit_frame, tuning_frame = _evaluation_role_frames(dataset, "tuning")
        manifest = {"statistics": ("avg_dwm_diff",)}
        model_name = "model_stale"
        model_frame = pd.DataFrame({"metric": ["current"]})
        current_fingerprint = _frame_fingerprint(model_frame)
        context_fingerprint = _evaluation_context_fingerprint(
            dataset,
            {},
            "main",
            False,
            expected_output_manifest=manifest,
            fit_frame=fit_frame,
            tuning_frame=tuning_frame,
        )
        checkpoint_root = tmp_path / "checkpoints"
        model_dir, _status_path, _result_path = _checkpoint_paths(
            checkpoint_root, "main", model_name
        )
        model_dir.mkdir(parents=True)
        (model_dir / "execution.json").write_text(
            json.dumps(
                {
                    "model_name": model_name,
                    "model_fingerprint": "stale-fingerprint",
                    "schema_version": "syntheval-execution-v1",
                    "pass_id": "main",
                    "target_view": "native",
                    "expected_manifest_digest": "manifest-hash",
                    "context_fingerprint": context_fingerprint,
                    "execution_complete": True,
                    "execution_succeeded": False,
                    "policy_eligible": False,
                    "metric_executions": [
                        {
                            "method": "statistics",
                            "status": {
                                "state": "failed",
                                "failed_keys": [],
                                "expected_keys": ["avg_dwm_diff"],
                                "observed_keys": ["avg_dwm_diff"],
                            },
                            "normalized_rows": [
                                {
                                    "metric": "avg_dwm_diff",
                                    "dim": "u",
                                    "val": 0.99,
                                    "n_val": 0.99,
                                    "raw_value": 0.99,
                                    "normalized_value": 0.99,
                                }
                            ],
                            "normalized_rows_v2": [
                                {
                                    "metric": "avg_dwm_diff",
                                    "dim": "u",
                                    "val": 0.99,
                                    "n_val": 0.99,
                                    "raw_value": 0.99,
                                    "normalized_value": 0.99,
                                    "metric_version": "v2",
                                }
                            ],
                        }
                    ],
                }
            )
        )

        class FailedProcess:
            pid = 123
            exitcode = 1

            def start(self):
                return None

            def is_alive(self):
                return False

            def join(self):
                return None

        class FailedContext:
            def Process(self, **_kwargs):
                return FailedProcess()

        monkeypatch.setattr(
            "synthdata.evaluation.syntheval_eval.multiprocessing.get_context",
            lambda _method: FailedContext(),
        )
        cfg = SynthEvalExecutionConfig(model_workers=1, max_model_workers=1, cores_per_model=1)
        results, ranks, executions = _run_resumable_syntheval(
            {model_name: model_frame},
            dataset,
            {},
            tmp_path / "preset.json",
            checkpoint_root,
            "summation",
            cfg,
            "main",
            expected_output_manifest=manifest,
            expected_manifest_digest="manifest-hash",
            fit_frame=fit_frame,
            tuning_frame=tuning_frame,
        )

        payload = executions[model_name]
        assert payload["model_fingerprint"] == current_fingerprint
        assert payload["execution_succeeded"] is False
        assert payload["policy_eligible"] is False
        assert payload["metric_executions"][0]["normalized_rows"] == []
        assert payload["metric_executions"][0]["normalized_rows_v2"] == []
        status = json.loads(_checkpoint_paths(checkpoint_root, "main", model_name)[1].read_text())
        assert status["model_fingerprint"] == current_fingerprint
        assert status["state"] == "failed"
        assert pd.isna(results.loc[model_name, ("avg_dwm_diff", "value")])
        assert pd.isna(ranks.loc[model_name, "avg_dwm_diff"])

    def test_failed_execution_sidecar_invalidates_checkpoint(self, tmp_path):
        checkpoint_root = tmp_path / "evaluation" / "syntheval_benchmark"
        model_dir, status_path, result_path = _checkpoint_paths(checkpoint_root, "main", "model_a")
        model_dir.mkdir(parents=True)
        context_fingerprint = "context-hash"
        model_fingerprint = "model-hash"
        expected_manifest_digest = "manifest-hash"
        status_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "state": "succeeded",
                    "model_name": "model_a",
                    "context_fingerprint": context_fingerprint,
                    "model_fingerprint": model_fingerprint,
                    "expected_manifest_digest": expected_manifest_digest,
                    "plots_completed": False,
                }
            )
        )
        result_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_parquet(result_path, pd.DataFrame({"metric": ["metric_a"], "val": [0.5]}))
        execution = self._execution_payload("failed")
        execution.update(
            {
                "model_name": "model_a",
                "context_fingerprint": context_fingerprint,
                "expected_manifest_digest": expected_manifest_digest,
            }
        )
        (model_dir / "execution.json").write_text(json.dumps(execution))

        from synthdata.evaluation.syntheval_eval import _valid_checkpoint

        assert (
            _valid_checkpoint(
                checkpoint_root,
                "main",
                "model_a",
                context_fingerprint,
                model_fingerprint,
                False,
                expected_manifest_digest=expected_manifest_digest,
            )
            is None
        )

    def test_sidecar_loader_rejects_failed_execution(self, tmp_path):
        model_dir, _status_path, _result_path = _checkpoint_paths(tmp_path, "main", "model_a")
        model_dir.mkdir(parents=True)
        payload = self._execution_payload("failed")
        payload.update(
            {
                "model_name": "model_a",
                "schema_version": "syntheval-execution-v1",
                "expected_manifest_digest": "manifest-hash",
                "context_fingerprint": "context-hash",
            }
        )
        (model_dir / "execution.json").write_text(json.dumps(payload))

        from synthdata.evaluation.syntheval_eval import _load_syntheval_execution_sidecars

        assert (
            _load_syntheval_execution_sidecars(
                tmp_path,
                "main",
                ["model_a"],
                "manifest-hash",
                "context-hash",
            )
            is None
        )

    def test_sidecar_loader_rejects_source_fingerprint_mismatch(self, tmp_path):
        model_dir, _status_path, _result_path = _checkpoint_paths(tmp_path, "main", "model_a")
        model_dir.mkdir(parents=True)
        payload = self._execution_payload()
        payload.update(
            {
                "model_name": "model_a",
                "expected_manifest_digest": "manifest-hash",
                "context_fingerprint": "context-hash",
                "model_fingerprint": "stored-source-hash",
            }
        )
        (model_dir / "execution.json").write_text(json.dumps(payload))

        assert (
            _load_syntheval_execution_sidecars(
                tmp_path,
                "main",
                ["model_a"],
                "manifest-hash",
                "context-hash",
                model_fingerprints={"model_a": "current-source-hash"},
            )
            is None
        )

    def test_absolute_checkpoint_path_survives_plot_directory_change(self, tmp_path):
        checkpoint_root = tmp_path / "evaluation" / "syntheval_benchmark"
        model_dir, _, result_path = _checkpoint_paths(checkpoint_root.resolve(), "main", "model_a")
        model_dir.mkdir(parents=True)
        plot_dir = tmp_path / "plots" / "model_a"
        plot_dir.mkdir(parents=True)
        previous_dir = Path.cwd()
        try:
            os.chdir(plot_dir)
            _atomic_parquet(result_path, pd.DataFrame({"metric": ["dwm"], "val": [0.5]}))
        finally:
            os.chdir(previous_dir)
        assert result_path.exists()
