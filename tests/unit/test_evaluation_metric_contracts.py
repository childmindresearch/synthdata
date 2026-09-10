"""Tests for the versioned metric contract registry and resolver."""

import math

import pytest
from synthcity.metrics.eval_attacks import DataLeakageLinear, DataLeakageMLP, DataLeakageXGB
from synthcity.metrics.eval_detection import (
    SyntheticDetectionLinear,
    SyntheticDetectionMLP,
    SyntheticDetectionXGB,
)
from synthcity.metrics.eval_performance import (
    PerformanceEvaluatorLinear,
    PerformanceEvaluatorMLP,
    PerformanceEvaluatorXGB,
)
from synthcity.metrics.eval_privacy import (
    DeltaPresence,
    DomiasMIAPrior,
    IdentifiabilityScore,
    kAnonymization,
    kMap,
    lDiversityDistinct,
)
from synthcity.metrics.eval_sanity import (
    CloseValuesProbability,
    CommonRowsProportion,
    DataMismatchScore,
    DistantValuesProbability,
    NearestSyntheticNeighborDistance,
)
from synthcity.metrics.eval_statistical import (
    ChiSquaredTest,
    InverseKLDivergence,
    JensenShannonDistance,
    KolmogorovSmirnovTest,
    MaximumMeanDiscrepancy,
    WassersteinDistance,
)

from synthdata.evaluation.catalog import SYNTHCITY_METRIC_CONFIG
from synthdata.evaluation.metric_contracts import (
    DEFAULT_METRIC_CONTRACT_REGISTRY,
    SYNTHCITY_METRIC_DIRECTIONS,
    AmbiguousMetricContractError,
    MetricContract,
    MetricContractRegistry,
    MetricEvaluationContext,
    MetricObservation,
    MetricValidationResult,
    UnknownMetricContractError,
    resolve_metric_observations,
)

pytestmark = pytest.mark.unit


def _contract(
    key: str = "test.metric",
    *,
    framework: str = "test",
    direction: str = "maximize",
    lifecycle_state: str = "operational",
    allowed_uses=frozenset({"audit", "hpo_objective"}),
    target_view: str = "native",
    group_safety: str = "group_safe",
    required_roles=("train",),
) -> MetricContract:
    return MetricContract(
        contract_id=f"{framework}.{key}.{target_view}",
        framework=framework,
        emitted_key_pattern=key,
        semantic_family="utility",
        direction=direction,
        value_role="policy_scalar",
        lifecycle_state=lifecycle_state,
        allowed_uses=allowed_uses,
        target_view=target_view,
        group_safety=group_safety,
        required_roles=required_roles,
        status_reason="test contract is intentionally constrained"
        if lifecycle_state != "operational"
        else "",
    )


def _context(**overrides) -> MetricEvaluationContext:
    values = {
        "role_hashes": {"train": "train-hash"},
        "target_view": "native",
        "population_unit": "row",
    }
    values.update(overrides)
    return MetricEvaluationContext(**values)


class TestDefaultRegistry:
    def test_native_direction_table_covers_selected_synthcity_metrics(self):
        selected_keys = {
            f"{category}.{metric_name}"
            for category, metric_names in SYNTHCITY_METRIC_CONFIG.items()
            for metric_name in metric_names
        }

        assert set(SYNTHCITY_METRIC_DIRECTIONS) == selected_keys

    @pytest.mark.parametrize(
        ("emitted_key", "native_direction"),
        [
            ("sanity.data_mismatch.score", DataMismatchScore.direction()),
            ("sanity.common_rows_proportion.score", CommonRowsProportion.direction()),
            (
                "sanity.nearest_syn_neighbor_distance.mean",
                NearestSyntheticNeighborDistance.direction(),
            ),
            ("sanity.close_values_probability.score", CloseValuesProbability.direction()),
            ("sanity.distant_values_probability.score", DistantValuesProbability.direction()),
            ("stats.jensenshannon_dist.marginal", JensenShannonDistance.direction()),
            ("stats.chi_squared_test.marginal", ChiSquaredTest.direction()),
            ("stats.inv_kl_divergence.marginal", InverseKLDivergence.direction()),
            ("stats.ks_test.marginal", KolmogorovSmirnovTest.direction()),
            ("stats.max_mean_discrepancy.joint", MaximumMeanDiscrepancy.direction()),
            ("stats.wasserstein_dist.joint", WassersteinDistance.direction()),
            ("performance.linear_model.syn_id", PerformanceEvaluatorLinear.direction()),
            ("performance.mlp.syn_ood", PerformanceEvaluatorMLP.direction()),
            ("performance.xgb.syn_id", PerformanceEvaluatorXGB.direction()),
            (
                "performance.linear_model_augmentation.aug_ood",
                PerformanceEvaluatorLinear.direction(),
            ),
            ("detection.detection_xgb.effective_auc_v2", SyntheticDetectionXGB.direction()),
            ("detection.detection_mlp.effective_auc_v2", SyntheticDetectionMLP.direction()),
            (
                "detection.detection_linear.effective_auc_v2",
                SyntheticDetectionLinear.direction(),
            ),
            ("privacy.delta-presence.score", DeltaPresence.direction()),
            ("privacy.k-anonymization.syn", kAnonymization.direction()),
            ("privacy.k-map.score", kMap.direction()),
            ("privacy.distinct l-diversity.syn", lDiversityDistinct.direction()),
            ("privacy.identifiability_score.score_OC", IdentifiabilityScore.direction()),
            ("privacy.DomiasMIA_prior.effective_auc_v2", DomiasMIAPrior.direction()),
            (
                "attack.data_leakage_linear.baseline_adjusted_advantage_v2",
                DataLeakageLinear.direction(),
            ),
            (
                "attack.data_leakage_mlp.baseline_adjusted_advantage_v2",
                DataLeakageMLP.direction(),
            ),
            (
                "attack.data_leakage_xgb.baseline_adjusted_advantage_v2",
                DataLeakageXGB.direction(),
            ),
        ],
    )
    def test_synthcity_policy_directions_match_native_evaluators(
        self, emitted_key, native_direction
    ):
        contract = DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
            framework="synthcity", emitted_key=emitted_key
        )

        assert contract.direction == native_direction

    @pytest.mark.parametrize(
        ("emitted_key", "value_role"),
        [
            ("performance.linear_model.gt", "diagnostic"),
            ("performance.mlp.gt", "diagnostic"),
            ("performance.xgb.gt", "diagnostic"),
            ("performance.linear_model_augmentation.gt", "diagnostic"),
            ("performance.mlp_augmentation.gt", "diagnostic"),
            ("performance.xgb_augmentation.gt", "diagnostic"),
            ("performance.linear_model.syn_id", "policy_scalar"),
            ("performance.mlp.syn_ood", "policy_scalar"),
            ("performance.xgb.syn_id", "policy_scalar"),
            ("performance.linear_model_augmentation.aug_ood", "policy_scalar"),
            ("privacy.k-anonymization.gt", "diagnostic"),
            ("privacy.k-anonymization.syn", "policy_scalar"),
            ("privacy.distinct l-diversity.gt", "diagnostic"),
            ("privacy.distinct l-diversity.syn", "policy_scalar"),
        ],
    )
    def test_native_baseline_rows_have_explicit_roles(self, emitted_key, value_role):
        contract = DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
            framework="synthcity", emitted_key=emitted_key
        )

        assert contract.value_role == value_role

    def test_covers_current_synthcity_selection_keys(self):
        for category, metric_names in SYNTHCITY_METRIC_CONFIG.items():
            for metric_name in metric_names:
                contract = DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
                    framework="synthcity",
                    emitted_key=f"{category}.{metric_name}",
                )
                assert contract.semantic_family in {"utility", "privacy"}
                assert "audit" in contract.allowed_uses

    @pytest.mark.parametrize(
        ("framework", "key", "family"),
        [
            ("syntheval", "mia_recall", "privacy"),
            ("syntheval", "mia_precision", "privacy"),
            ("syntheval", "att_discl_risk", "privacy"),
            ("syntheval", "eps_identif_risk", "privacy"),
            ("syntheval", "priv_loss_eps", "privacy"),
            ("syntheval", "avg_nndr", "privacy"),
            ("syntheval", "priv_loss_nndr", "privacy"),
            ("syntheval", "nnaa", "privacy"),
            ("syntheval", "priv_loss_nnaa", "privacy"),
            ("syntheval", "median_DCR", "privacy"),
            ("syntheval", "hit_rate", "privacy"),
            ("custom", "eo_target_sex", "fairness"),
        ],
    )
    def test_emitted_privacy_and_diagnostic_keys_are_explicitly_classified(
        self, framework, key, family
    ):
        contract = DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
            framework=framework,
            emitted_key=key,
        )
        assert contract.semantic_family == family
        assert "audit" in contract.allowed_uses

    def test_subgroup_rows_are_diagnostics_not_policy_scalars(self):
        contract = DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
            framework="syntheval", emitted_key="sp_target_sex"
        )
        assert contract.value_role == "diagnostic"
        assert contract.direction is None
        assert contract.qualifiers == ("target", "protected_group", "cell", "ovr")

    def test_unknown_emitted_key_fails_closed(self):
        with pytest.raises(UnknownMetricContractError):
            DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
                framework="syntheval", emitted_key="privacy_not_in_registry"
            )

    def test_feature_rank_submetrics_have_distinct_roles(self):
        correlation = DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
            framework="synthcity", emitted_key="performance.feat_rank_distance.corr"
        )
        pvalue = DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
            framework="synthcity", emitted_key="performance.feat_rank_distance.pvalue"
        )

        assert correlation.value_role == "policy_scalar"
        assert correlation.direction == "maximize"
        assert pvalue.value_role == "diagnostic"
        assert pvalue.direction is None

    @pytest.mark.parametrize(
        ("framework", "emitted_key"),
        [
            ("synthcity", "performance.linear_model.gt"),
            ("synthcity", "performance.mlp.syn_id"),
            ("synthcity", "performance.xgb.syn_ood"),
            ("synthcity", "detection.detection_xgb.mean"),
            ("synthcity", "detection.detection_mlp.effective_auc_v2"),
        ],
    )
    def test_only_verified_synthcity_tabular_paths_are_group_safe(self, framework, emitted_key):
        contract = DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
            framework=framework,
            emitted_key=emitted_key,
        )

        assert contract.group_safety == "group_safe"

    @pytest.mark.parametrize(
        ("framework", "emitted_key"),
        [
            ("synthcity", "performance.linear_model_augmentation.aug_ood"),
            ("synthcity", "performance.feat_rank_distance.corr"),
            ("synthcity", "privacy.identifiability_score.score_OC"),
            ("synthcity", "attack.data_leakage_xgb.mean"),
            ("synthcity", "stats.ks_test.marginal"),
            ("syntheval", "avg_dwm_diff"),
            ("syntheval", "mia_recall"),
            ("custom", "equalized_odds"),
        ],
    )
    def test_unverified_paths_are_not_group_safe(self, framework, emitted_key):
        contract = DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
            framework=framework,
            emitted_key=emitted_key,
        )

        assert contract.group_safety != "group_safe"

    def test_domias_raw_and_effective_auc_contracts_are_distinct(self):
        raw_auc = DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
            framework="synthcity",
            emitted_key="privacy.DomiasMIA_prior.aucroc",
        )
        effective_auc = DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
            framework="synthcity",
            emitted_key="privacy.DomiasMIA_prior.effective_auc_v2",
        )

        assert raw_auc.value_role == "diagnostic"
        assert raw_auc.direction is None
        assert effective_auc.value_role == "policy_scalar"
        assert effective_auc.direction == "minimize"
        assert effective_auc.anchors.chance == pytest.approx(0.5)

    def test_identifiability_variants_have_explicit_calibration_contracts(self):
        for key, qualifiers in {
            "privacy.identifiability_score.score": ("legacy", "unweighted"),
            "privacy.identifiability_score.score_OC": (
                "legacy",
                "oneclass",
                "unweighted",
            ),
            "privacy.identifiability_score.score_entropy_weighted": (
                "repaired",
                "entropy_weighted",
            ),
            "privacy.identifiability_score.score_OC_entropy_weighted": (
                "repaired",
                "oneclass",
                "entropy_weighted",
            ),
        }.items():
            contract = DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
                framework="synthcity", emitted_key=key
            )
            assert contract.lifecycle_state == "calibrating"
            assert contract.value_role == "policy_scalar"
            assert contract.direction == "minimize"
            assert contract.qualifiers == qualifiers

    def test_detection_raw_and_effective_auc_contracts_are_distinct(self):
        raw_auc = DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
            framework="synthcity",
            emitted_key="detection.detection_xgb.mean",
        )
        effective_auc = DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
            framework="synthcity",
            emitted_key="detection.detection_xgb.effective_auc_v2",
        )

        assert raw_auc.value_role == "diagnostic"
        assert raw_auc.direction is None
        assert effective_auc.value_role == "policy_scalar"
        assert effective_auc.direction == "minimize"
        assert effective_auc.anchors.chance == pytest.approx(0.5)

    def test_attribute_attack_per_target_outputs_are_diagnostics(self):
        target_accuracy = DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
            framework="synthcity",
            emitted_key="attack.data_leakage_linear.raw_accuracy.sex",
        )
        aggregate_advantage = DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
            framework="synthcity",
            emitted_key="attack.data_leakage_linear.baseline_adjusted_advantage_v2",
        )

        assert target_accuracy.value_role == "diagnostic"
        assert target_accuracy.direction is None
        assert aggregate_advantage.value_role == "policy_scalar"
        assert aggregate_advantage.direction == "minimize"

    def test_attribute_attack_uncertainty_is_a_target_diagnostic(self):
        uncertainty = DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
            framework="synthcity",
            emitted_key="attack.data_leakage_linear.uncertainty_v2.secret",
        )

        assert uncertainty.value_role == "diagnostic"
        assert uncertainty.direction is None
        assert uncertainty.qualifiers == ("target", "uncertainty")

    def test_structural_privacy_contract_is_calibration_only(self):
        contract = DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
            framework="synthcity",
            emitted_key="privacy.k-map.score",
        )

        assert contract.lifecycle_state == "calibrating"
        assert contract.qualifiers == ("structural_proxy",)

    def test_binary_target_contract_has_distinct_execution_and_target_view(self):
        contract = DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
            framework="syntheval",
            emitted_key="auroc",
            execution_pass="binary_target",
        )
        assert contract.execution_pass == "binary_target"
        assert contract.target_view == "binary_collapsed"

    def test_digest_is_stable_and_manifest_is_versioned(self):
        assert (
            DEFAULT_METRIC_CONTRACT_REGISTRY.digest() == DEFAULT_METRIC_CONTRACT_REGISTRY.digest()
        )
        manifest = DEFAULT_METRIC_CONTRACT_REGISTRY.manifest()
        assert manifest["schema_version"] == 1
        assert manifest["registry_version"] == "metric-contracts-v1"
        assert manifest["digest"] == DEFAULT_METRIC_CONTRACT_REGISTRY.digest()

    def test_default_contracts_have_complete_semantic_metadata(self):
        for contract in DEFAULT_METRIC_CONTRACT_REGISTRY:
            assert contract.framework_identity
            assert contract.metric_version
            assert contract.raw_range is not None
            assert contract.preprocessing_contract
            assert contract.classification_score_policy
            if contract.uncertainty_field is not None:
                assert contract.uncertainty_semantics
            if contract.sample_size_field is not None:
                assert contract.sample_size_unit

    def test_signed_syntheval_v2_value_uses_agreement_transform(self):
        contract = DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
            framework="syntheval", emitted_key="auroc_v2"
        )
        context = MetricEvaluationContext(
            role_hashes={"train": "train-hash", "tuning": "tuning-hash"},
        )
        result = resolve_metric_observations(
            registry=DEFAULT_METRIC_CONTRACT_REGISTRY,
            model_name="model_a",
            framework="syntheval",
            expected_keys=["auroc_v2"],
            observations=[
                MetricObservation(
                    model_name="model_a",
                    framework="syntheval",
                    emitted_key="auroc_v2",
                    raw_value=-0.25,
                    role_hashes=dict(context.role_hashes),
                )
            ],
            context=context,
            requested_use="audit",
        )

        assert contract.policy_transform == "one_minus_absolute"
        assert result.expected_records[0].policy_value == pytest.approx(0.75)
        assert result.expected_records[0].policy_transform == "one_minus_absolute"


class TestMetricContractRegistry:
    def test_duplicate_selected_keys_are_rejected(self):
        registry = MetricContractRegistry([_contract()])
        with pytest.raises(ValueError, match="unique"):
            registry.resolve_many(framework="test", emitted_keys=["test.metric", "test.metric"])

    def test_exact_contract_takes_precedence_over_wildcard(self):
        registry = MetricContractRegistry(
            [
                _contract(key="test.*"),
                _contract(key="test.metric", target_view="other"),
            ]
        )
        assert registry.resolve(framework="test", emitted_key="test.metric").target_view == "other"

    def test_duplicate_wildcard_matches_are_ambiguous(self):
        registry = MetricContractRegistry(
            [
                _contract(key="test.*"),
                _contract(key="test.m*", target_view="other"),
            ]
        )
        with pytest.raises(AmbiguousMetricContractError):
            registry.resolve(framework="test", emitted_key="test.metric")

    def test_default_auroc_v2_contract_is_unambiguous(self):
        contract = DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
            framework="syntheval", emitted_key="auroc_v2"
        )
        assert contract.contract_id == "syntheval.auroc_v2"

    @pytest.mark.parametrize(
        ("emitted_key", "sample_size_field"),
        [
            ("corr_mat_diff_v2", "metadata.valid_pairs"),
            ("mutual_inf_diff_v2", "metadata.valid_pairs"),
            ("ks_tvd_stat_v2", "metadata.valid_tests"),
            ("frac_ks_sigs_v2", "metadata.valid_tests"),
            ("avg_h_dist_v2", "metadata.valid_columns"),
            ("avg_qMSE_v2", "metadata.valid_columns"),
            ("avg_pMSE_v2", "metadata.oof_n"),
        ],
    )
    def test_v2_support_fields_are_explicit_and_normalized_scores_are_not_sample_sizes(
        self, emitted_key, sample_size_field
    ):
        contract = DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
            framework="syntheval", emitted_key=emitted_key
        )

        assert contract.sample_size_field == sample_size_field
        assert contract.sample_size_field != "n_val"

    def test_metrics_without_explicit_support_do_not_claim_a_sample_size(self):
        contract = DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
            framework="syntheval", emitted_key="auroc_v2"
        )

        assert contract.sample_size_field is None

    def test_synthcity_contract_uses_native_uncertainty_and_support_fields(self):
        contract = DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
            framework="synthcity", emitted_key="stats.ks_test.marginal"
        )

        assert contract.uncertainty_field == "stddev"
        assert contract.sample_size_field == "rounds"


class TestResolveMetricObservations:
    def test_success_orients_policy_scalar(self):
        registry = MetricContractRegistry([_contract(direction="minimize")])
        result = resolve_metric_observations(
            registry=registry,
            model_name="model_a",
            framework="test",
            expected_keys=["test.metric"],
            observations=[
                MetricObservation(
                    model_name="model_a",
                    framework="test",
                    emitted_key="test.metric",
                    raw_value=0.25,
                    direction="minimize",
                    role_hashes={"train": "train-hash"},
                )
            ],
            context=_context(),
        )
        assert isinstance(result, MetricValidationResult)
        record = result.expected_records[0]
        assert record.status == "succeeded"
        assert record.raw_value == pytest.approx(0.25)
        assert record.policy_value == pytest.approx(-0.25)
        assert result.complete is True
        assert result.audit_complete is True
        assert result.decision_eligible is True
        assert result.policy_rank_eligible is False

    @pytest.mark.parametrize(
        ("observations", "status"),
        [
            ([], "missing"),
            (
                [
                    MetricObservation("model_a", "test", "test.metric", 0.1),
                    MetricObservation("model_a", "test", "test.metric", 0.2),
                ],
                "duplicate",
            ),
            ([MetricObservation("model_a", "test", "test.metric", math.nan)], "non_finite"),
            ([MetricObservation("model_a", "test", "test.metric", math.inf)], "non_finite"),
            ([MetricObservation("model_a", "test", "test.metric", 0.1, error="boom")], "failed"),
        ],
    )
    def test_invalid_observations_are_explicit_statuses(self, observations, status):
        registry = MetricContractRegistry([_contract()])
        result = resolve_metric_observations(
            registry=registry,
            model_name="model_a",
            framework="test",
            expected_keys=["test.metric"],
            observations=observations,
            context=_context(),
        )
        assert result.expected_records[0].status == status
        assert result.complete is False
        assert result.decision_eligible is False

    def test_disallowed_use_is_blocked_without_disappearing(self):
        registry = MetricContractRegistry(
            [
                _contract(
                    lifecycle_state="audit_only",
                    allowed_uses=frozenset({"audit"}),
                )
            ]
        )
        result = resolve_metric_observations(
            registry=registry,
            model_name="model_a",
            framework="test",
            expected_keys=["test.metric"],
            observations=[
                MetricObservation(
                    "model_a", "test", "test.metric", 0.1, role_hashes={"train": "hash"}
                )
            ],
            context=_context(),
            requested_use="hpo_objective",
        )
        assert result.expected_records[0].status == "blocked"
        assert result.decision_eligible is False

    def test_incomplete_diagnostic_evidence_blocks_policy_decision(self):
        registry = MetricContractRegistry(
            [
                _contract(
                    key="test.policy",
                    allowed_uses=frozenset({"audit", "policy_rank"}),
                ),
                MetricContract(
                    contract_id="test.diagnostic.native",
                    framework="test",
                    emitted_key_pattern="test.diagnostic",
                    semantic_family="utility",
                    direction=None,
                    value_role="diagnostic",
                    lifecycle_state="audit_only",
                    allowed_uses=frozenset({"audit"}),
                    group_safety="group_safe",
                    required_roles=("train",),
                    status_reason="Diagnostic evidence is audit-only",
                ),
            ]
        )
        result = resolve_metric_observations(
            registry=registry,
            model_name="model_a",
            framework="test",
            expected_keys=["test.policy", "test.diagnostic"],
            observations=[
                MetricObservation(
                    "model_a",
                    "test",
                    "test.policy",
                    0.1,
                    direction="maximize",
                    role_hashes={"train": "train-hash"},
                ),
                MetricObservation(
                    "model_a",
                    "test",
                    "test.diagnostic",
                    0.2,
                    role_hashes={"train": "train-hash"},
                ),
            ],
            context=_context(),
            requested_use="policy_rank",
        )

        assert result.complete is False
        assert result.expected_records[0].status == "succeeded"
        assert result.expected_records[1].status == "blocked"
        assert result.decision_eligible is False

    def test_failed_required_diagnostic_blocks_policy_decision(self):
        registry = MetricContractRegistry(
            [
                _contract(
                    key="test.policy",
                    allowed_uses=frozenset({"audit", "policy_rank"}),
                ),
                MetricContract(
                    contract_id="test.diagnostic.failed",
                    framework="test",
                    emitted_key_pattern="test.diagnostic",
                    semantic_family="utility",
                    direction=None,
                    value_role="diagnostic",
                    lifecycle_state="operational",
                    allowed_uses=frozenset({"audit", "policy_rank"}),
                    group_safety="group_safe",
                    required_roles=("train",),
                ),
            ]
        )
        result = resolve_metric_observations(
            registry=registry,
            model_name="model_a",
            framework="test",
            expected_keys=["test.policy", "test.diagnostic"],
            observations=[
                MetricObservation(
                    "model_a",
                    "test",
                    "test.policy",
                    0.1,
                    direction="maximize",
                    role_hashes={"train": "train-hash"},
                ),
                MetricObservation(
                    "model_a",
                    "test",
                    "test.diagnostic",
                    None,
                    error="diagnostic failed",
                    role_hashes={"train": "train-hash"},
                ),
            ],
            context=_context(),
            requested_use="policy_rank",
        )

        assert result.complete is False
        assert result.expected_records[0].status == "succeeded"
        assert result.expected_records[1].status == "failed"
        assert result.expected_records[1].error == "diagnostic failed"
        assert result.indeterminate_keys == ("test.diagnostic",)
        assert result.decision_eligible is False

    def test_validation_serializes_evaluation_context(self):
        context = MetricEvaluationContext(
            role_hashes={"train": "train-hash"},
            resolved_configuration={"task_type": "regression"},
        )
        result = resolve_metric_observations(
            registry=DEFAULT_METRIC_CONTRACT_REGISTRY,
            model_name="model_a",
            framework="synthcity",
            expected_keys=["stats.ks_test"],
            observations=[
                MetricObservation(
                    model_name="model_a",
                    framework="synthcity",
                    emitted_key="stats.ks_test",
                    raw_value=0.5,
                    direction="maximize",
                    role_hashes={"train": "train-hash"},
                )
            ],
            context=context,
        )

        assert result.to_dict()["evaluation_context"] == {
            "execution_pass": "main",
            "target_view": "native",
            "evaluation_role": "tuning",
            "population_unit": "row",
            "group_mode": "row",
            "role_hashes": {"train": "train-hash"},
            "resolved_configuration": {"task_type": "regression"},
        }

    def test_final_holdout_context_requires_final_holdout_hash(self):
        contract = _contract(required_roles=("train", "tuning"))
        observation = MetricObservation(
            "model_a",
            "test",
            "test.metric",
            0.1,
            role_hashes={"train": "train-hash", "final_holdout": "holdout-hash"},
        )
        context = _context(
            evaluation_role="final_holdout",
            role_hashes={"train": "train-hash", "final_holdout": "holdout-hash"},
        )

        result = resolve_metric_observations(
            registry=MetricContractRegistry([contract]),
            model_name="model_a",
            framework="test",
            expected_keys=["test.metric"],
            observations=[observation],
            context=context,
            requested_use="audit",
        )

        assert result.expected_records[0].status == "succeeded"
        assert result.expected_records[0].required_roles == ("train", "final_holdout")
        assert result.to_dict()["evaluation_context"]["evaluation_role"] == "final_holdout"

    @pytest.mark.parametrize(
        ("context_overrides", "status"),
        [
            ({"role_hashes": {}}, "wrong_role"),
            ({"target_view": "binary_collapsed"}, "wrong_target_view"),
            ({"population_unit": "patient_group"}, "group_unsafe"),
        ],
    )
    def test_context_mismatches_are_not_row_level_fallbacks(self, context_overrides, status):
        registry = MetricContractRegistry([_contract(group_safety="row_only")])
        result = resolve_metric_observations(
            registry=registry,
            model_name="model_a",
            framework="test",
            expected_keys=["test.metric"],
            observations=[
                MetricObservation(
                    "model_a",
                    "test",
                    "test.metric",
                    0.1,
                    role_hashes={"train": "train-hash"},
                )
            ],
            context=_context(**context_overrides),
        )
        assert result.expected_records[0].status == status

    def test_group_safety_diagnosis_precedes_disallowed_policy_use(self):
        registry = MetricContractRegistry(
            [_contract(group_safety="row_only", allowed_uses=frozenset({"audit"}))]
        )
        result = resolve_metric_observations(
            registry=registry,
            model_name="model_a",
            framework="test",
            expected_keys=["test.metric"],
            observations=[
                MetricObservation(
                    "model_a",
                    "test",
                    "test.metric",
                    0.1,
                    role_hashes={"train": "train-hash"},
                )
            ],
            context=_context(population_unit="patient_group"),
            requested_use="policy_rank",
        )

        assert result.expected_records[0].status == "group_unsafe"

    def test_unknown_expected_key_and_unexpected_observation_are_retained(self):
        registry = MetricContractRegistry([_contract()])
        result = resolve_metric_observations(
            registry=registry,
            model_name="model_a",
            framework="test",
            expected_keys=["not_registered"],
            observations=[MetricObservation("model_a", "test", "test.metric", 0.1)],
            context=_context(),
        )
        assert [record.status for record in result.records] == ["unknown_contract", "unexpected"]
        assert result.complete is False
