"""Unit tests for synthcity metric selection and native category names."""

import math

import numpy as np
import pandas as pd
import pytest

from synthdata.config import FrameworkSelectionConfig
from synthdata.evaluation.catalog import (
    SYNTHCITY_METRIC_CONFIG,
    emitted_keys_for_synthcity_metrics,
)
from synthdata.evaluation.metric_contracts import MetricEvaluationContext
from synthdata.evaluation.synthcity_eval import (
    _schema_mismatch_score,
    resolve_metric_config,
    run_synthcity_evaluation,
    run_synthcity_metrics,
    validate_synthcity_report,
    validate_synthcity_results,
)
from tests.unit.synthcity_emitted_key_fixtures import (
    SELECTED_SYNTHCITY_ATTACK_TARGET_TYPES,
    SELECTED_SYNTHCITY_EMITTED_KEY_FIXTURES,
    SELECTED_SYNTHCITY_VARIABLE_COLUMNS,
)

pytestmark = pytest.mark.unit


class TestResolveMetricConfig:
    def test_default_selection_uses_native_attack_category(self):
        result = resolve_metric_config(FrameworkSelectionConfig())

        assert result["attack"] == SYNTHCITY_METRIC_CONFIG["attack"]
        assert "attacks" not in result
        assert "detection_gmm" not in result["detection"]

    def test_gmm_detection_is_not_selectable(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            resolve_metric_config(FrameworkSelectionConfig(metrics=["detection_gmm"]))

    def test_run_metrics_preserves_semantic_context(self, tmp_path):
        frame = pd.DataFrame({"value": [0.0, 1.0, 2.0, 3.0], "target": [0, 1, 0, 1]})
        semantic_context = {
            "schema_version": "semantic-context-v1",
            "task_type": "classification",
        }

        result = run_synthcity_metrics(
            frame,
            frame,
            frame,
            n_samples=len(frame),
            target_column="target",
            sensitive_features=[],
            metrics={"sanity": ["common_rows_proportion"]},
            workspace=tmp_path,
            semantic_context=semantic_context,
        )

        assert result.attrs["semantic_context"] == semantic_context
        validation = validate_synthcity_report(
            "model_a",
            result,
            expected_base_keys=["sanity.common_rows_proportion"],
        )
        source_metadata = validation.expected_records[0].source_metadata
        assert source_metadata["semantic_context"] == semantic_context
        assert source_metadata["semantic_context_digest"]

    @pytest.mark.parametrize(
        ("detector_name", "model_attribute"),
        [
            ("detection_xgb", "XGBClassifier"),
            ("detection_mlp", "MLP"),
            ("detection_linear", "LogisticRegression"),
        ],
    )
    @pytest.mark.parametrize(
        ("mode", "real_marker", "synthetic_marker", "expected_raw_auc", "expected_effective_auc"),
        [
            ("chance", 0.0, 0.0, 0.5, 0.5),
            ("separable", 0.0, 100.0, 1.0, 1.0),
            ("inverted", 0.0, 100.0, 0.0, 1.0),
        ],
    )
    def test_detector_contract_matrix(
        self,
        monkeypatch,
        tmp_path,
        detector_name,
        model_attribute,
        mode,
        real_marker,
        synthetic_marker,
        expected_raw_auc,
        expected_effective_auc,
    ):
        from synthcity.metrics import eval_detection

        class ControlledDetector:
            def __init__(self, **kwargs):
                del kwargs

            def fit(self, data, labels):
                del data, labels
                return self

            def predict_proba(self, data):
                if mode == "chance":
                    positive_probability = np.full(len(data), 0.5)
                else:
                    positive_probability = (data[:, 0] >= 50.0).astype(float)
                    if mode == "inverted":
                        positive_probability = 1.0 - positive_probability
                return np.column_stack((1.0 - positive_probability, positive_probability))

        monkeypatch.setattr(eval_detection, model_attribute, ControlledDetector)
        real = pd.DataFrame(
            {
                "marker": np.arange(20, dtype=float) + real_marker,
                "target": [0, 1] * 10,
            }
        )
        synthetic = pd.DataFrame(
            {
                "marker": np.arange(20, dtype=float) + synthetic_marker,
                "target": [0, 1] * 10,
            }
        )

        result = run_synthcity_metrics(
            synthetic,
            real,
            real,
            n_samples=20,
            target_column="target",
            sensitive_features=[],
            metrics={"detection": [detector_name]},
            workspace=tmp_path,
        )

        result_prefix = f"detection.{detector_name}"
        assert result.loc[f"{result_prefix}.mean", "mean"] == pytest.approx(expected_raw_auc)
        assert result.loc[f"{result_prefix}.raw_auc", "mean"] == pytest.approx(expected_raw_auc)
        assert result.loc[f"{result_prefix}.effective_auc_v2", "mean"] == pytest.approx(
            expected_effective_auc
        )
        metadata = result.attrs["metric_metadata"][result_prefix]
        assert metadata["raw_auc_key"] == "raw_auc"
        assert metadata["effective_auc_key"] == "effective_auc_v2"
        assert metadata["default_key"] == "effective_auc_v2"

        validation = validate_synthcity_results(
            {"model_a": result},
            {"detection": [detector_name]},
            role_hashes={"train": "train-hash", "tuning": "tuning-hash"},
        )["model_a"]
        assert validation.complete
        raw_record = next(
            record
            for record in validation.records
            if record.expected_key == f"{result_prefix}.raw_auc"
        )
        effective_record = next(
            record
            for record in validation.records
            if record.expected_key == f"{result_prefix}.effective_auc_v2"
        )
        assert raw_record.value_role == "diagnostic"
        assert raw_record.raw_value == pytest.approx(expected_raw_auc)
        assert effective_record.value_role == "policy_scalar"
        assert effective_record.policy_value == pytest.approx(-expected_effective_auc)

    def test_final_evaluation_catalog_retains_domias(self):
        assert "DomiasMIA_prior" in SYNTHCITY_METRIC_CONFIG["privacy"]

    def test_privacy_category_includes_attack_metrics(self):
        result = resolve_metric_config(FrameworkSelectionConfig(categories=["privacy"]))

        assert result["attack"] == SYNTHCITY_METRIC_CONFIG["attack"]

    def test_explicit_attack_metric_uses_native_category(self):
        result = resolve_metric_config(FrameworkSelectionConfig(metrics=["data_leakage_linear"]))

        assert result == {"attack": ["data_leakage_linear"]}

    def test_schema_mismatch_is_computed_before_dtype_alignment(self):
        reference = pd.DataFrame({"age": pd.Series([1, 2], dtype="int64")})
        synthetic = pd.DataFrame({"age": pd.Series([1.0, 2.0], dtype="float64")})

        assert _schema_mismatch_score(reference, synthetic) == pytest.approx(0.5)

    def test_run_metrics_rejects_unseen_numeric_categorical_code(self, tmp_path):
        reference = pd.DataFrame(
            {
                "status": list(range(1, 16)),
                "target": [0, 1] * 7 + [0],
            }
        )
        synthetic = pd.DataFrame(
            {
                "status": list(range(15)),
                "target": [0, 1] * 7 + [0],
            }
        )

        with pytest.raises(ValueError, match="previously unseen labels"):
            run_synthcity_metrics(
                synthetic,
                reference,
                reference,
                n_samples=len(synthetic),
                target_column="target",
                sensitive_features=[],
                metrics={"stats": ["jensenshannon_dist"]},
                workspace=tmp_path,
                feature_types={"status": "categorical"},
            )

    def test_run_metrics_preserves_pre_alignment_schema_result(self, monkeypatch, tmp_path):
        from synthcity.metrics import Metrics

        received_dtypes = []
        received_feature_types = []
        received_source_tables = []
        received_classification_scores = []

        def fake_evaluate(*args, **kwargs):
            received_dtypes.append(str(args[1].dataframe()["age"].dtype))
            received_feature_types.append(args[0].feature_types)
            received_source_tables.append(kwargs["source_table"])
            received_classification_scores.append(kwargs["classification_score"])
            return pd.DataFrame(
                {
                    "min": [0.0],
                    "max": [0.0],
                    "mean": [0.0],
                    "stddev": [0.0],
                    "median": [0.0],
                    "iqr": [0.0],
                    "rounds": [1],
                    "errors": [0],
                    "direction": ["minimize"],
                },
                index=pd.Index(["sanity.data_mismatch.score"]),
            )

        monkeypatch.setattr(Metrics, "evaluate", staticmethod(fake_evaluate))
        reference = pd.DataFrame({"age": pd.Series([1, 2], dtype="int64")})
        synthetic = pd.DataFrame({"age": pd.Series([1.0, 2.0], dtype="float64")})

        results = run_synthcity_metrics(
            synthetic,
            reference,
            reference,
            n_samples=2,
            target_column="age",
            sensitive_features=[],
            metrics={"sanity": ["data_mismatch"]},
            workspace=tmp_path,
            feature_types={"age": "continuous"},
            source_table={"age": "measurements"},
            classification_score="macro_f1",
        )

        assert received_dtypes == ["int64"]
        assert received_feature_types == [{"age": "continuous"}]
        assert received_source_tables == [{"age": "measurements"}]
        assert received_classification_scores == ["macro_f1"]
        assert results.loc["sanity.data_mismatch.score", "mean"] == pytest.approx(0.5)
        assert results.loc["sanity.data_mismatch.score", "schema_source"] == "pre_dtype_alignment"

    def test_run_metrics_namespaces_cache_by_semantic_context(self, monkeypatch, tmp_path):
        from synthcity.metrics import Metrics

        workspaces = []

        def fake_evaluate(*args, **kwargs):
            workspaces.append(kwargs["workspace"])
            return pd.DataFrame(
                {"mean": [0.0], "direction": ["minimize"]},
                index=pd.Index(["sanity.data_mismatch.score"]),
            )

        monkeypatch.setattr(Metrics, "evaluate", staticmethod(fake_evaluate))
        reference = pd.DataFrame({"age": [1, 2]})
        synthetic = pd.DataFrame({"age": [1, 2]})

        for feature_type in ("continuous", "categorical"):
            run_synthcity_metrics(
                synthetic,
                reference,
                reference,
                n_samples=2,
                target_column="age",
                sensitive_features=[],
                metrics={"sanity": ["data_mismatch"]},
                workspace=tmp_path,
                feature_types={"age": feature_type},
            )

        assert workspaces[0] != workspaces[1]

    def test_run_metrics_retains_jensen_shannon_aggregation_metadata(self, tmp_path):
        reference = pd.DataFrame(
            {
                "category": ["a", "b", "a", "b"],
                "measurement": [0.0, 1.0, 2.0, 1.0],
            }
        )
        synthetic = pd.DataFrame(
            {
                "category": ["a", "a", "a", "b"],
                "measurement": [0.0, 2.0, 2.0, 1.0],
            }
        )

        result = run_synthcity_metrics(
            synthetic,
            reference,
            reference,
            n_samples=4,
            target_column="category",
            sensitive_features=[],
            metrics={"stats": ["jensenshannon_dist"]},
            workspace=tmp_path,
            feature_types={"category": "categorical", "measurement": "continuous"},
            source_table={"category": "demographics", "measurement": "labs"},
        )

        assert "stats.jensenshannon_dist.source_table_macro_v2" in result.index
        assert "stats.jensenshannon_dist.max_variable_v2" in result.index
        assert result.attrs["result_metadata"] == result.attrs["metric_metadata"]
        metadata = result.attrs["metric_metadata"]["stats.jensenshannon_dist"]
        assert metadata["aggregation_contract"]["schema_version"] == ("source-table-aggregation-v1")
        validations = validate_synthcity_results(
            {"model_a": result},
            {"stats": ["jensenshannon_dist"]},
            variable_columns=["category", "measurement"],
            context=MetricEvaluationContext(
                role_hashes={"train": "train-hash", "tuning": "tuning-hash"}
            ),
        )
        validation = validations["model_a"]
        assert validation.complete
        aggregate_record = next(
            record
            for record in validation.records
            if record.expected_key == "stats.jensenshannon_dist.source_table_macro_v2"
        )
        assert (
            aggregate_record.result_metadata
            == result.attrs["result_metadata"]["stats.jensenshannon_dist"]
        )
        assert aggregate_record.source_metadata["metric_metadata"]["stats.jensenshannon_dist"][
            "aggregation_contract"
        ]["source_table_macro_v2"]["source_tables"]["demographics"]["variables"] == ["category"]

    def test_run_metrics_retains_structural_proxy_calibration_metadata(self, tmp_path):
        reference = pd.DataFrame(
            {
                "quasi_id": np.arange(20, dtype=float),
                "secret": [0, 1] * 10,
            }
        )

        result = run_synthcity_metrics(
            reference.copy(),
            reference,
            reference,
            n_samples=len(reference),
            target_column="secret",
            sensitive_features=["secret"],
            metrics={"privacy": ["k-anonymization"]},
            workspace=tmp_path,
            structural_n_clusters=[2, 5],
            structural_min_rows_per_cluster=2,
        )

        metadata = result.attrs["metric_metadata"]["privacy.k-anonymization"]
        assert result.attrs["result_metadata"] == result.attrs["metric_metadata"]
        assert metadata["result_version"] == "structural-proxy-v2"
        assert metadata["proxy_label"] == "kmeans_partition_screen_not_formal_guarantee"
        assert metadata["calibration_only"] is True
        assert metadata["calibration_required"] is True

    def test_run_metrics_retains_domias_protocol_metadata(self, monkeypatch, tmp_path):
        from synthcity.metrics.eval_privacy import DOMIAS_RESULT_VERSION, DomiasMIAPrior

        score_values = np.array(
            [
                0.05,
                0.10,
                0.15,
                0.20,
                0.25,
                0.30,
                0.60,
                0.65,
                0.70,
                0.75,
                0.35,
                0.40,
                0.45,
                0.50,
                0.55,
            ]
        )

        def controlled_density_scores(
            self, synth_set, synth_val_set, reference_set, X_test, device
        ):
            del self
            del synth_set, synth_val_set, reference_set, device
            return score_values, np.ones(len(X_test))

        monkeypatch.setattr(DomiasMIAPrior, "evaluate_p_R", controlled_density_scores)
        reference = pd.DataFrame({"value": np.arange(10, dtype=float), "target": [0, 1] * 5})
        train = reference.copy()

        result = run_synthcity_metrics(
            reference.copy(),
            reference,
            train,
            n_samples=len(reference),
            target_column="target",
            sensitive_features=[],
            metrics={"privacy": ["DomiasMIA_prior"]},
            workspace=tmp_path,
        )

        result_prefix = "privacy.DomiasMIA_prior"
        assert result.loc[f"{result_prefix}.aucroc", "mean"] == pytest.approx(0.4)
        assert result.loc[f"{result_prefix}.effective_auc_v2", "mean"] == pytest.approx(0.6)
        metadata = result.attrs["metric_metadata"][result_prefix]
        assert result.attrs["result_metadata"] == result.attrs["metric_metadata"]
        assert metadata["result_version"] == DOMIAS_RESULT_VERSION
        assert metadata["calibration_only"] is True
        assert metadata["calibration_required"] is True
        assert metadata["random_state"] == 42
        assert metadata["population_roles"]["evidence"]["rows"] == len(reference)
        protocol = metadata["domias_protocol"]
        assert protocol["reference_size_requested"] == len(reference) // 2
        assert protocol["auc_chance"] == pytest.approx(0.5)
        assert protocol["accuracy_prevalence_dependent"] is True
        assert protocol["group_disjoint"] is False
        assert protocol["random_state"] == 42

    def test_run_metrics_passes_aligned_group_ids_to_all_loaders(self, monkeypatch, tmp_path):
        from synthcity.metrics import Metrics

        received_groups = {}

        def fake_evaluate(*args, **kwargs):
            received_groups.update(
                {
                    "reference": args[0].group_ids.tolist(),
                    "synthetic": args[1].group_ids.tolist(),
                    "train": args[2].group_ids.tolist(),
                    "reference_synthetic": args[3].group_ids.tolist(),
                    "augmented": args[4].group_ids.tolist(),
                }
            )
            assert kwargs["X_gt_group_ids"] == ["r1", "r1", "r2", "r2"]
            assert kwargs["X_train_group_ids"] == ["t1", "t1", "t2", "t2"]
            assert kwargs["group_mode"] == "patient_group"
            return pd.DataFrame()

        monkeypatch.setattr(Metrics, "evaluate", staticmethod(fake_evaluate))
        reference = pd.DataFrame({"value": [1, 2, 3, 4], "target": [0, 0, 1, 1]})
        train = pd.DataFrame({"value": [5, 6, 7, 8], "target": [0, 0, 1, 1]})
        synthetic = pd.DataFrame({"value": [9, 10], "target": [0, 1]})

        run_synthcity_metrics(
            synthetic,
            reference,
            train,
            n_samples=2,
            target_column="target",
            sensitive_features=[],
            metrics={},
            workspace=tmp_path,
            real_reference_group_ids=["r1", "r1", "r2", "r2"],
            real_train_group_ids=["t1", "t1", "t2", "t2"],
            group_mode="patient_group",
        )

        assert received_groups["reference"] == ["r1", "r1", "r2", "r2"]
        assert received_groups["train"] == ["t1", "t1", "t2", "t2"]
        assert len(received_groups["synthetic"]) == 2
        assert len(received_groups["reference_synthetic"]) == 2
        assert received_groups["augmented"][:4] == ["t1", "t1", "t2", "t2"]

    def test_grouped_non_tabular_evaluation_reaches_root_status_bridge(self, tmp_path):
        selection = FrameworkSelectionConfig(metrics=["common_rows_proportion"])
        frame = pd.DataFrame({"value": [0, 1, 2, 3], "target": [0, 1, 0, 1]})

        results = run_synthcity_evaluation(
            {"model_a": frame.iloc[:2].copy()},
            frame,
            frame,
            target_column="target",
            sensitive_features=[],
            selection_cfg=selection,
            n_samples=2,
            workspace=tmp_path,
            real_reference_group_ids=["p1", "p1", "p2", "p2"],
            real_train_group_ids=["t1", "t1", "t2", "t2"],
            task_type="time_series",
        )

        report = results["model_a"]
        assert report.attrs["group_safety"]["status"] == "group_unsafe"
        validation = validate_synthcity_results(
            results,
            resolve_metric_config(selection),
            context=MetricEvaluationContext(
                role_hashes={"train": "train-hash", "tuning": "tuning-hash"},
                population_unit="patient_group",
                group_mode="patient_group",
            ),
        )["model_a"]

        assert validation.expected_keys == ("sanity.common_rows_proportion.score",)
        assert validation.expected_records[0].status == "group_unsafe"
        assert validation.expected_records[0].source_metadata["report_state"] == "group_unsafe"


class TestSynthcityContractBridge:
    @staticmethod
    def _context():
        return MetricEvaluationContext(role_hashes={"train": "train-hash", "test": "test-hash"})

    def test_successful_native_row_is_contract_validated(self):
        report = pd.DataFrame(
            {
                "mean": [0.25],
                "stddev": [0.01],
                "rounds": [5],
                "errors": [0],
                "direction": ["maximize"],
            },
            index=pd.Index(["stats.ks_test.marginal"]),
        )

        validation = validate_synthcity_report(
            "model_a",
            report,
            context=self._context(),
        )

        record = validation.expected_records[0]
        assert record.status == "succeeded"
        assert record.raw_value == pytest.approx(0.25)
        assert record.uncertainty == pytest.approx(0.01)
        assert record.sample_size == 5
        assert record.source_metadata["uncertainty_field"] == "stddev"
        assert record.source_metadata["sample_size_field"] == "rounds"
        assert validation.complete is True

    def test_target_attack_support_uses_n_eval_as_sample_size(self):
        report = pd.DataFrame(
            {
                "mean": [17.0],
                "direction": ["minimize"],
                "stddev": [None],
                "rounds": [None],
            },
            index=pd.Index(["attack.data_leakage_xgb.n_eval.secret"]),
        )

        validation = validate_synthcity_report(
            "model_a",
            report,
            expected_base_keys=["attack.data_leakage_xgb.n_eval.secret"],
            context=self._context(),
        )

        record = validation.expected_records[0]
        assert record.status == "succeeded"
        assert record.sample_size == 17
        assert record.source_metadata["sample_size_field"] == "mean"

    def test_attack_uncertainty_and_protocol_metadata_are_preserved(self):
        report = pd.DataFrame(
            {
                "mean": [0.125],
                "direction": ["minimize"],
                "stddev": [None],
                "rounds": [None],
            },
            index=pd.Index(["attack.data_leakage_xgb.uncertainty_v2.secret"]),
        )
        protocol = {
            "schema_version": "attribute-inference-v4",
            "protocol_digest": "protocol-a",
            "worst_target_selection": {
                "selected_target": "secret",
                "target_risks": {"secret": 0.5},
            },
        }
        report.attrs["metric_metadata"] = {
            "attack.data_leakage_xgb": protocol,
        }

        validation = validate_synthcity_report(
            "model_a",
            report,
            expected_base_keys=["attack.data_leakage_xgb.uncertainty_v2.secret"],
            context=self._context(),
        )

        record = validation.expected_records[0]
        assert record.status == "succeeded"
        assert record.raw_value == pytest.approx(0.125)
        assert record.source_metadata["metric_metadata"] == {
            "attack.data_leakage_xgb": protocol,
        }

    def test_identifiability_variant_metadata_is_preserved(self):
        report = pd.DataFrame(
            {
                "mean": [0.25],
                "direction": ["minimize"],
                "stddev": [None],
                "rounds": [None],
            },
            index=pd.Index(["privacy.identifiability_score.score_entropy_weighted"]),
        )
        protocol = {
            "result_version": "identifiability-v2",
            "calibration_only": True,
            "variants": {
                "score_entropy_weighted": {
                    "embedding": "raw",
                    "weighting": "entropy",
                }
            },
        }
        report.attrs["metric_metadata"] = {
            "privacy.identifiability_score": protocol,
        }

        validation = validate_synthcity_report(
            "model_a",
            report,
            expected_base_keys=["privacy.identifiability_score.score_entropy_weighted"],
            context=self._context(),
        )

        record = validation.expected_records[0]
        assert record.status == "succeeded"
        assert record.source_metadata["metric_metadata"] == {
            "privacy.identifiability_score": protocol,
        }

    @pytest.mark.parametrize(
        "emitted_key",
        [
            "privacy.identifiability_score.score_OC",
            "attack.data_leakage_xgb.mean",
            "stats.ks_test.marginal",
            "performance.linear_model_augmentation.aug_ood",
        ],
    )
    def test_patient_group_validation_blocks_unsupported_paths(self, emitted_key):
        report = pd.DataFrame(
            {"mean": [0.25], "direction": ["minimize"]},
            index=pd.Index([emitted_key]),
        )
        context = MetricEvaluationContext(
            role_hashes={"train": "train-hash", "tuning": "tuning-hash"},
            population_unit="patient_group",
            group_mode="patient_group",
        )

        validation = validate_synthcity_report(
            "model_a",
            report,
            context=context,
            requested_use="audit",
        )

        assert validation.expected_records[0].status == "group_unsafe"
        assert validation.decision_eligible is False

    def test_result_metadata_is_retained_in_status_record(self):
        report = pd.DataFrame(
            {
                "mean": [0.25],
                "errors": [0],
                "direction": ["maximize"],
            },
            index=pd.Index(["stats.ks_test.marginal"]),
        )
        report.attrs["metric_metadata"] = {
            "privacy.k-anonymization": {
                "result_version": "structural-proxy-v2",
                "proxy_label": "kmeans_partition_screen_not_formal_guarantee",
            }
        }

        validation = validate_synthcity_report(
            "model_a",
            report,
            context=self._context(),
        )

        metadata = validation.expected_records[0].source_metadata["metric_metadata"]
        assert metadata["privacy.k-anonymization"]["result_version"] == "structural-proxy-v2"

    def test_duplicate_native_rows_are_explicitly_invalid(self):
        report = pd.DataFrame(
            {
                "mean": [0.25, 0.3],
                "errors": [0, 0],
                "direction": ["maximize", "maximize"],
            },
            index=pd.Index(["stats.ks_test.marginal", "stats.ks_test.marginal"]),
        )

        validation = validate_synthcity_report(
            "model_a",
            report,
            context=self._context(),
        )

        assert validation.expected_records[0].status == "duplicate"
        assert validation.expected_records[0].observed_count == 2

    def test_materialized_metric_failure_is_retained(self):
        report = pd.DataFrame(
            {
                "mean": [math.nan],
                "errors": [1],
                "error_types": ["ValueError"],
                "error_messages": ["invalid input shape"],
                "direction": ["minimize"],
            },
            index=pd.Index(["stats.ks_test"]),
        )

        validation = validate_synthcity_report(
            "model_a",
            report,
            context=self._context(),
        )

        record = validation.expected_records[0]
        assert record.status == "failed"
        assert "ValueError" in (record.error or "")
        assert validation.complete is False

    def test_benchmark_metric_failure_reaches_root_status_bridge(self, monkeypatch, tmp_path):
        import synthcity.benchmark as benchmark_module
        from synthcity.benchmark import Benchmarks
        from synthcity.plugins import Plugins
        from synthcity.plugins.core.dataloader import GenericDataLoader

        frame = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0], "target": [0, 1, 0, 1]})
        loader = GenericDataLoader(frame, target_column="target")

        class WorkingGenerator:
            def fit(self, _fit_loader):
                return self

            def generate(self, **_kwargs):
                return loader

        monkeypatch.setattr(
            Plugins,
            "get",
            lambda self, name, **kwargs: WorkingGenerator(),
        )

        def failing_metrics_evaluate(*_args, **_kwargs):
            raise ValueError("metric evaluation exploded")

        class FailingMetrics:
            evaluate = staticmethod(failing_metrics_evaluate)

        monkeypatch.setattr(benchmark_module, "Metrics", FailingMetrics)

        result = Benchmarks.evaluate(
            [("failed_metrics", "fake", {})],
            loader,
            X_test=loader,
            metrics={"stats": ["ks_test"]},
            repeats=1,
            workspace=tmp_path / "workspace",
            synthetic_cache=False,
            synthetic_reuse_if_exists=False,
            use_metric_cache=False,
            fit_on_X=True,
        )

        validation = validate_synthcity_report(
            "model_a",
            result["failed_metrics"],
            expected_base_keys=["stats.ks_test"],
            context=self._context(),
        )

        record = validation.expected_records[0]
        assert record.status == "failed"
        assert "ValueError" in (record.error or "")
        assert "metric evaluation exploded" in (record.error or "")
        assert validation.complete is False

    def test_non_finite_score_is_retained_as_failed_observation(self):
        from synthcity.metrics.scores import ScoreEvaluator

        scores = ScoreEvaluator()
        scores.add(
            "stats.ks_test.marginal",
            math.nan,
            failed=0,
            duration=0.25,
            direction="minimize",
        )

        validation = validate_synthcity_report(
            "model_a",
            scores.to_dataframe(),
            expected_keys=["stats.ks_test.marginal"],
            context=self._context(),
        )

        record = validation.expected_records[0]
        assert record.status == "failed"
        assert record.raw_value is None
        assert record.policy_value is None
        assert "NonFiniteMetricResult" in (record.error or "")
        assert validation.complete is False

    def test_feature_rank_failure_has_no_policy_value(self):
        report = pd.DataFrame(
            {
                "mean": [math.nan],
                "errors": [1],
                "error_types": ["RuntimeError"],
                "error_messages": ["SHAP output did not preserve feature axis"],
                "direction": ["maximize"],
            },
            index=pd.Index(["performance.feat_rank_distance.corr"]),
        )

        validation = validate_synthcity_report(
            "model_a",
            report,
            context=self._context(),
        )

        record = validation.expected_records[0]
        assert record.status == "failed"
        assert record.raw_value is None
        assert record.policy_value is None
        assert validation.complete is False

    def test_feature_rank_contract_reaches_root_report(self, monkeypatch, tmp_path):
        from synthcity.metrics import eval_performance

        class RecordingClassifier:
            def __init__(self, **kwargs):
                del kwargs

            def fit(self, data, labels):
                del data, labels
                return self

        class ControlledExplainer:
            def __init__(self, model):
                del model

            def shap_values(self, data):
                return np.tile(np.asarray([[1.0, 2.0, 3.0]]), (len(data), 1))

        monkeypatch.setattr(eval_performance, "XGBClassifier", RecordingClassifier)
        monkeypatch.setattr(
            eval_performance.shap,
            "TreeExplainer",
            ControlledExplainer,
        )
        frame = pd.DataFrame(
            {
                "first": np.arange(12, dtype=float),
                "second": np.arange(12, dtype=float) + 1.0,
                "third": np.arange(12, dtype=float) + 2.0,
                "target": [0, 1] * 6,
            }
        )

        result = run_synthcity_metrics(
            frame.copy(),
            frame.copy(),
            frame.copy(),
            n_samples=len(frame),
            target_column="target",
            sensitive_features=[],
            metrics={"performance": ["feat_rank_distance"]},
            workspace=tmp_path,
        )

        result_prefix = "performance.feat_rank_distance"
        assert result.loc[f"{result_prefix}.corr", "mean"] == pytest.approx(1.0)
        assert np.isfinite(result.loc[f"{result_prefix}.pvalue", "mean"])
        metadata = result.attrs["metric_metadata"][result_prefix]
        assert metadata["schema_version"] == "rank-v3"
        assert metadata["default_key"] == "corr"
        assert metadata["pvalue_role"] == "audit_only"

        validation = validate_synthcity_results(
            {"model_a": result},
            {"performance": ["feat_rank_distance"]},
            context=self._context(),
        )["model_a"]
        assert validation.complete
        corr_record = next(
            record
            for record in validation.records
            if record.expected_key == f"{result_prefix}.corr"
        )
        assert corr_record.result_metadata == metadata
        assert corr_record.value_role == "policy_scalar"

    def test_group_unsafe_report_is_retained_as_expected_status(self):
        report = pd.DataFrame(index=pd.Index(["sanity.common_rows_proportion"]))
        report.attrs["group_safety"] = {
            "schema_version": "group-safety-v1",
            "status": "group_unsafe",
            "reason": "time-series metric internals are not group-safe",
        }
        context = MetricEvaluationContext(
            role_hashes={"train": "train-hash", "tuning": "tuning-hash"},
            population_unit="patient_group",
            group_mode="patient_group",
        )

        validation = validate_synthcity_report(
            "model_a",
            report,
            expected_base_keys=["sanity.common_rows_proportion"],
            context=context,
        )

        record = validation.expected_records[0]
        assert record.expected_key == "sanity.common_rows_proportion.score"
        assert record.status == "group_unsafe"
        assert "not group-safe" in (record.error or "")
        assert record.source_metadata["group_safety"]["schema_version"] == "group-safety-v1"

    def test_model_level_failure_expands_selected_base_keys(self):
        report = pd.DataFrame({"error": ["framework crashed"], "error_type": ["RuntimeError"]})

        validation = validate_synthcity_report(
            "model_a",
            report,
            expected_base_keys=["stats.ks_test"],
            context=self._context(),
        )

        record = validation.expected_records[0]
        assert record.expected_key == "stats.ks_test.marginal"
        assert record.status == "failed"
        assert "RuntimeError" in (record.error or "")

    def test_declared_selection_retains_omitted_metric_as_missing(self):
        report = pd.DataFrame(
            {
                "mean": [0.25],
                "errors": [0],
                "direction": ["maximize"],
            },
            index=pd.Index(["stats.ks_test.marginal"]),
        )

        validation = validate_synthcity_report(
            "model_a",
            report,
            expected_base_keys=["stats.ks_test", "stats.wasserstein_dist"],
            context=self._context(),
        )

        assert [record.expected_key for record in validation.expected_records] == [
            "stats.ks_test.marginal",
            "stats.wasserstein_dist.joint",
        ]
        assert [record.status for record in validation.expected_records] == [
            "succeeded",
            "missing",
        ]
        assert validation.complete is False

    def test_declared_metric_requires_every_fixed_submetric(self):
        report = pd.DataFrame(
            {
                "mean": [0.25],
                "errors": [0],
                "direction": ["maximize"],
            },
            index=pd.Index(["stats.prdc.precision"]),
        )

        validation = validate_synthcity_report(
            "model_a",
            report,
            expected_base_keys=["stats.prdc"],
            context=self._context(),
        )

        assert [record.expected_key for record in validation.expected_records] == [
            "stats.prdc.precision",
            "stats.prdc.recall",
            "stats.prdc.density",
            "stats.prdc.coverage",
        ]
        assert [record.status for record in validation.expected_records] == [
            "succeeded",
            "missing",
            "missing",
            "missing",
        ]
        assert validation.complete is False

    def test_contextual_manifest_requires_declared_qualified_rows(self):
        report = pd.DataFrame(
            {
                "mean": [0.25],
                "errors": [0],
                "direction": ["minimize"],
            },
            index=pd.Index(["stats.jensenshannon_dist.marginal"]),
        )

        validation = validate_synthcity_results(
            {"model_a": report},
            {"stats": ["jensenshannon_dist"]},
            role_hashes={"train": "train-hash", "tuning": "tuning-hash"},
            variable_columns=["age", "target"],
        )["model_a"]

        assert validation.expected_keys == (
            "stats.jensenshannon_dist.marginal",
            "stats.jensenshannon_dist.source_table_macro_v2",
            "stats.jensenshannon_dist.max_variable_v2",
            "stats.jensenshannon_dist.variable_v2.age",
            "stats.jensenshannon_dist.variable_v2.target",
        )
        assert validation.completed_keys == ("stats.jensenshannon_dist.marginal",)
        assert validation.indeterminate_keys == (
            "stats.jensenshannon_dist.source_table_macro_v2",
            "stats.jensenshannon_dist.max_variable_v2",
            "stats.jensenshannon_dist.variable_v2.age",
            "stats.jensenshannon_dist.variable_v2.target",
        )
        assert validation.complete is False

    def test_explicit_manifest_keeps_unexpected_rows_audit_only(self):
        report = pd.DataFrame(
            {
                "mean": [0.25, 0.5],
                "errors": [0, 0],
                "direction": ["minimize", "minimize"],
            },
            index=pd.Index(
                [
                    "stats.jensenshannon_dist.marginal",
                    "stats.jensenshannon_dist.variable_v2.unexpected",
                ]
            ),
        )

        validation = validate_synthcity_report(
            "model_a",
            report,
            expected_keys=emitted_keys_for_synthcity_metrics(
                {"stats": ["jensenshannon_dist"]},
                variable_columns=["age"],
            ),
            context=self._context(),
        )

        assert validation.expected_keys == (
            "stats.jensenshannon_dist.marginal",
            "stats.jensenshannon_dist.source_table_macro_v2",
            "stats.jensenshannon_dist.max_variable_v2",
            "stats.jensenshannon_dist.variable_v2.age",
        )
        unexpected = next(
            record
            for record in validation.records
            if record.expected_key == "stats.jensenshannon_dist.variable_v2.unexpected"
        )
        assert unexpected.is_expected is False
        assert unexpected.status == "unexpected"
        assert validation.complete is False

    def test_contextual_attack_manifest_requires_each_declared_target_output(self):
        report = pd.DataFrame(
            {
                "mean": [0.25, 0.1, 0.1],
                "errors": [0, 0, 0],
                "direction": ["minimize", "minimize", "minimize"],
            },
            index=pd.Index(
                [
                    "attack.data_leakage_xgb.baseline_adjusted_advantage_v2",
                    "attack.data_leakage_xgb.legacy_accuracy",
                    "attack.data_leakage_xgb.mean",
                ]
            ),
        )

        validation = validate_synthcity_results(
            {"model_a": report},
            {"attack": ["data_leakage_xgb"]},
            role_hashes={"train": "train-hash", "tuning": "tuning-hash"},
            attack_target_types={"sex": "categorical"},
        )["model_a"]

        assert "attack.data_leakage_xgb.raw_accuracy.sex" in validation.expected_keys
        missing = next(
            record
            for record in validation.expected_records
            if record.expected_key == "attack.data_leakage_xgb.raw_accuracy.sex"
        )
        assert missing.status == "missing"
        assert validation.complete is False

    def test_complete_typed_attack_manifest_survives_root_validation(self):
        expected_keys = tuple(
            fixture_key
            for category, _metric_name, fixture_keys in SELECTED_SYNTHCITY_EMITTED_KEY_FIXTURES
            if category == "attack"
            for fixture_key in fixture_keys
        )
        report = pd.DataFrame(
            {
                "mean": [
                    17.0 if key.endswith((".n_eval.secret", ".n_eval.income")) else 0.25
                    for key in expected_keys
                ],
                "errors": [0] * len(expected_keys),
                "direction": ["minimize"] * len(expected_keys),
            },
            index=pd.Index(expected_keys),
        )

        validation = validate_synthcity_results(
            {"model_a": report},
            {"attack": SYNTHCITY_METRIC_CONFIG["attack"]},
            context=self._context(),
            attack_target_types=SELECTED_SYNTHCITY_ATTACK_TARGET_TYPES,
        )["model_a"]

        assert validation.expected_keys == expected_keys
        assert validation.completed_keys == expected_keys
        assert validation.status_counts == {"succeeded": len(expected_keys)}
        records = {record.expected_key: record for record in validation.expected_records}
        assert records["attack.data_leakage_xgb.raw_accuracy.secret"].qualifiers == (
            "target",
            "raw",
        )
        assert records["attack.data_leakage_xgb.normalized_mae_v2.income"].qualifiers == (
            "target",
            "continuous",
        )
        assert records["attack.data_leakage_xgb.n_eval.secret"].sample_size == 17
        assert records["attack.data_leakage_xgb.n_eval.income"].sample_size == 17
        assert all(
            f"attack.{family}.raw_accuracy.secret" in validation.expected_keys
            for family in SYNTHCITY_METRIC_CONFIG["attack"]
        )
        assert all(
            f"attack.{family}.normalized_mae_v2.income" in validation.expected_keys
            for family in SYNTHCITY_METRIC_CONFIG["attack"]
        )
        assert not any(".raw_accuracy.income" in key for key in validation.expected_keys)
        assert not any(".normalized_mae_v2.secret" in key for key in validation.expected_keys)

    @pytest.mark.parametrize(
        ("category", "metric_name", "expected_keys"),
        SELECTED_SYNTHCITY_EMITTED_KEY_FIXTURES,
        ids=[
            f"{category}.{metric_name}"
            for category, metric_name, _expected_keys in SELECTED_SYNTHCITY_EMITTED_KEY_FIXTURES
        ],
    )
    def test_production_selection_uses_literal_manifest(self, category, metric_name, expected_keys):
        validation = validate_synthcity_results(
            {"model_a": pd.DataFrame()},
            {category: [metric_name]},
            context=self._context(),
            variable_columns=list(SELECTED_SYNTHCITY_VARIABLE_COLUMNS),
            attack_target_types=SELECTED_SYNTHCITY_ATTACK_TARGET_TYPES,
        )["model_a"]

        assert validation.expected_keys == expected_keys
        assert validation.indeterminate_keys == expected_keys
        assert validation.status_counts == {"failed": len(expected_keys)}

    def test_declared_unknown_metric_cannot_derive_expectations_from_observed_rows(self):
        report = pd.DataFrame(
            {
                "mean": [0.25],
                "errors": [0],
                "direction": ["maximize"],
            },
            index=pd.Index(["stats.future_metric.observed"]),
        )

        with pytest.raises(ValueError, match="no static emitted-key contract"):
            validate_synthcity_report(
                "model_a",
                report,
                expected_base_keys=["stats.future_metric"],
                context=self._context(),
            )
