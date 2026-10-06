"""Unit tests for synthdata.evaluation.catalog.resolve_selection (partial-
selection precedence resolver used by every evaluation framework) and the
SynthEval result-column classification helpers.
"""

import pytest
from syntheval.execution import build_metric_execution

from synthdata.evaluation.catalog import (
    CANONICAL_EXPECTED_MANIFEST,
    CANONICAL_HPO_ALLOWLIST,
    LEGACY_AUDIT_MANIFEST,
    LOG_DISPARITY_METRICS,
    SYNTHCITY_METRIC_CONFIG,
    SYNTHEVAL_CUSTOM_FAIRNESS_KEYS,
    SYNTHEVAL_EMITTED_KEY_DIRECTION,
    SYNTHEVAL_EMITTED_KEY_TYPE,
    classify_syntheval_metric,
    emitted_keys_for_synthcity_metrics,
    is_custom_syntheval_metric,
    is_redundant_synthcity_submetric,
    resolve_selection,
    syntheval_execution_manifest,
    syntheval_framework_for_emitted_key,
)
from synthdata.evaluation.metric_contracts import (
    DEFAULT_METRIC_CONTRACT_REGISTRY,
    UnknownMetricContractError,
)
from tests.unit.synthcity_emitted_key_fixtures import (
    SELECTED_SYNTHCITY_ATTACK_TARGET_TYPES,
    SELECTED_SYNTHCITY_EMITTED_KEY_FIXTURES,
    SELECTED_SYNTHCITY_VARIABLE_COLUMNS,
)

pytestmark = pytest.mark.unit

ALL_METRICS = ["m1", "m2", "m3", "m4"]
TYPE_MAP = {"m1": "utility", "m2": "utility", "m3": "privacy", "m4": "fairness"}


class TestResolveSelection:
    def test_disabled_returns_empty(self):
        assert resolve_selection(False, None, None, ALL_METRICS, TYPE_MAP) == []

    def test_disabled_wins_over_explicit_metrics(self):
        assert resolve_selection(False, None, ["m1"], ALL_METRICS, TYPE_MAP) == []

    def test_explicit_metrics_take_precedence_over_categories(self):
        result = resolve_selection(True, ["privacy"], ["m1"], ALL_METRICS, TYPE_MAP)
        assert result == ["m1"]

    def test_explicit_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            resolve_selection(True, None, ["not_a_metric"], ALL_METRICS, TYPE_MAP)

    def test_categories_filter_by_type(self):
        result = resolve_selection(True, ["utility"], None, ALL_METRICS, TYPE_MAP)
        assert result == ["m1", "m2"]

    def test_multiple_categories(self):
        result = resolve_selection(True, ["privacy", "fairness"], None, ALL_METRICS, TYPE_MAP)
        assert result == ["m3", "m4"]

    def test_neither_given_returns_all(self):
        result = resolve_selection(True, None, None, ALL_METRICS, TYPE_MAP)
        assert result == ALL_METRICS

    def test_empty_metrics_list_falls_through_to_categories(self):
        # An empty (falsy) explicit list should not be treated as "given".
        result = resolve_selection(True, ["utility"], [], ALL_METRICS, TYPE_MAP)
        assert result == ["m1", "m2"]


class TestClassifySynthevalMetric:
    def test_known_metric_key_matches_dict(self):
        assert classify_syntheval_metric("statistical_parity") == "fairness"
        assert classify_syntheval_metric("avg_dwm_diff") == "utility"
        assert classify_syntheval_metric("nnaa") == "privacy"

    def test_auroc_diffs_actual_result_column_name_is_utility(self):
        # auroc_diff's own result column is literally "auroc", not "auroc_diff".
        assert classify_syntheval_metric("auroc") == "utility"

    def test_auroc_per_target_submetric_is_utility(self):
        assert classify_syntheval_metric("auroc_CGAS_class") == "utility"

    @pytest.mark.parametrize("prefix", ["sp_", "eo_", "eqo_"])
    def test_fairness_submetrics_are_fairness(self, prefix):
        assert classify_syntheval_metric(f"{prefix}CGAS_class_Sex") == "fairness"

    def test_unknown_metric_fails_closed(self):
        with pytest.raises(UnknownMetricContractError, match="some_unrecognised_metric"):
            classify_syntheval_metric("some_unrecognised_metric")

    @pytest.mark.parametrize(
        ("emitted_key", "category", "direction"),
        [
            ("auroc_macro_ovr_v3", "utility", "maximize"),
            ("statistical_parity_macro_ovr_v1", "fairness", "minimize"),
            ("equalized_odds_macro_ovr_v1", "fairness", "minimize"),
            ("equal_opportunity_macro_ovr_v1", "fairness", "minimize"),
        ],
    )
    def test_multiclass_ovr_identity_has_category_and_contract_direction(
        self, emitted_key, category, direction
    ):
        assert classify_syntheval_metric(emitted_key) == category
        assert SYNTHEVAL_EMITTED_KEY_TYPE[emitted_key] == category
        contract = DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
            framework=syntheval_framework_for_emitted_key(emitted_key),
            emitted_key=emitted_key,
        )
        assert SYNTHEVAL_EMITTED_KEY_DIRECTION[emitted_key] == direction
        assert SYNTHEVAL_EMITTED_KEY_DIRECTION[emitted_key] == contract.direction

    def test_multiclass_custom_fairness_metrics_keep_custom_framework_owner(self):
        for emitted_key in (
            "equalized_odds_macro_ovr_v1",
            "equal_opportunity_macro_ovr_v1",
        ):
            assert emitted_key in SYNTHEVAL_CUSTOM_FAIRNESS_KEYS
            assert syntheval_framework_for_emitted_key(emitted_key) == "custom"


class TestIsCustomSynthevalMetric:
    def test_primary_custom_fairness_keys(self):
        assert is_custom_syntheval_metric("equalized_odds") is True
        assert is_custom_syntheval_metric("equal_opportunity") is True

    def test_non_custom_fairness_key(self):
        assert is_custom_syntheval_metric("statistical_parity") is False

    @pytest.mark.parametrize("prefix", ["eo_", "eqo_"])
    def test_custom_submetric_prefixes(self, prefix):
        assert is_custom_syntheval_metric(f"{prefix}CGAS_class_Sex") is True

    def test_non_custom_submetric_prefix(self):
        assert is_custom_syntheval_metric("sp_CGAS_class_Sex") is False

    def test_unrelated_metric(self):
        assert is_custom_syntheval_metric("dwm") is False


class TestIsRedundantSynthcitySubmetric:
    @pytest.mark.parametrize(
        "metric_key",
        [
            "stats.alpha_precision.delta_precision_alpha_naive",
            "stats.alpha_precision.delta_coverage_beta_naive",
            "stats.alpha_precision.authenticity_naive",
        ],
    )
    def test_naive_alpha_precision_submetrics_flagged_redundant(self, metric_key):
        assert is_redundant_synthcity_submetric(metric_key) is True

    @pytest.mark.parametrize(
        "metric_key",
        [
            "stats.alpha_precision.delta_precision_alpha_OC",
            "stats.alpha_precision.delta_coverage_beta_OC",
            "stats.alpha_precision.authenticity_OC",
            "stats.ks_test.marginal",
            "privacy.identifiability_score.score_OC",
        ],
    )
    def test_non_naive_submetrics_not_flagged(self, metric_key):
        assert is_redundant_synthcity_submetric(metric_key) is False


class TestContextualEmittedKeys:
    def test_canonical_manifests_are_exact_and_disjoint_by_owner(self):
        assert CANONICAL_EXPECTED_MANIFEST == (
            "elastic_net_jsd.v1",
            "mixed_mmd.v1",
            "release_privacy.v1",
            "tstr_macro_f1.v1",
            "equalized_odds.final.v1",
            "representation_evidence.v1",
        )
        assert {"elastic_net_jsd.v1", "mixed_mmd.v1", "tstr_macro_f1.v1"} == CANONICAL_HPO_ALLOWLIST
        assert (
            tuple(
                contract.contract_id
                for contract in DEFAULT_METRIC_CONTRACT_REGISTRY
                if contract.lifecycle_state == "blocked" and "legacy" in contract.qualifiers
            )
            == LEGACY_AUDIT_MANIFEST
        )

    def test_manifest_is_not_expanded_by_observed_keys(self):
        manifest = syntheval_execution_manifest({"dwm": {}}, include_holdout_outputs=False)
        assert manifest == {"dwm": ("avg_dwm_diff",)}

    def test_synthcity_manifest_expands_declared_variables_and_attack_targets(self):
        keys = emitted_keys_for_synthcity_metrics(
            {
                "stats": ["jensenshannon_dist"],
                "attack": ["data_leakage_xgb"],
            },
            variable_columns=["age", "target"],
            attack_target_types={"sex": "categorical", "income": "continuous"},
        )

        assert "stats.jensenshannon_dist.marginal" in keys
        assert "stats.jensenshannon_dist.variable_v2.age" in keys
        assert "stats.jensenshannon_dist.variable_v2.target" in keys
        assert "stats.jensenshannon_dist.source_table_macro_v2" in keys
        assert "stats.jensenshannon_dist.max_variable_v2" in keys
        assert "attack.data_leakage_xgb.raw_accuracy.sex" in keys
        assert "attack.data_leakage_xgb.baseline_adjusted_advantage_v2.sex" in keys
        assert "attack.data_leakage_xgb.disclosure_risk_v2.income" in keys
        assert "attack.data_leakage_xgb.n_eval.income" in keys

    def test_syntheval_manifest_expands_only_explicit_full_output(self):
        manifest = syntheval_execution_manifest(
            {
                "statistical_parity": {"full_output": True},
                "dwm": {},
            },
            include_holdout_outputs=False,
            target_columns=["Target Label"],
            protected_columns=["Sex"],
        )

        assert manifest["statistical_parity"] == (
            "statistical_parity",
            "sp_target_label_Sex",
        )
        assert manifest["dwm"] == ("avg_dwm_diff",)

    def test_syntheval_manifest_uses_ovr_identities_only_for_multiclass_native_pass(self):
        preset = {
            "auroc_diff": {},
            "statistical_parity": {"full_output": True},
            "equalized_odds": {"full_output": True},
            "equal_opportunity": {"full_output": True},
        }

        multiclass_manifest = syntheval_execution_manifest(
            preset,
            include_holdout_outputs=False,
            target_columns=["Target Label"],
            protected_columns=["Sex"],
            target_is_binary=False,
        )
        binary_manifest = syntheval_execution_manifest(
            preset,
            include_holdout_outputs=False,
            target_columns=["Target Label"],
            protected_columns=["Sex"],
            target_is_binary=True,
        )

        assert multiclass_manifest["auroc_diff"] == ("auroc_macro_ovr_v3",)
        assert multiclass_manifest["statistical_parity"] == ("statistical_parity_macro_ovr_v1",)
        assert multiclass_manifest["equalized_odds"] == ("equalized_odds_macro_ovr_v1",)
        assert multiclass_manifest["equal_opportunity"] == ("equal_opportunity_macro_ovr_v1",)
        assert binary_manifest["auroc_diff"] == ("auroc_v2",)
        assert binary_manifest["statistical_parity"] == (
            "statistical_parity",
            "sp_target_label_Sex",
        )
        assert binary_manifest["equalized_odds"] == (
            "equalized_odds",
            "eqo_target_label_Sex",
        )
        assert binary_manifest["equal_opportunity"] == (
            "equal_opportunity",
            "eo_target_label_Sex",
        )

    @pytest.mark.parametrize(
        ("preset_name", "aggregate_key", "diagnostic_key"),
        [
            (
                "statistical_parity",
                "statistical_parity_macro_ovr_v1",
                "sp_ovr_v1_target_label_a_Sex",
            ),
            (
                "equalized_odds",
                "equalized_odds_macro_ovr_v1",
                "eqo_ovr_v1_target_label_a_Sex",
            ),
            (
                "equal_opportunity",
                "equal_opportunity_macro_ovr_v1",
                "eo_ovr_v1_target_label_a_Sex",
            ),
        ],
    )
    def test_multiclass_fairness_status_accepts_classwise_diagnostics_but_requires_aggregate(
        self, preset_name, aggregate_key, diagnostic_key
    ):
        normalized_rows = [
            {
                "metric": aggregate_key,
                "dim": "f",
                "val": 0.2,
                "err": 0.01,
                "n_val": 0.8,
                "n_err": 0.01,
            },
            {
                "metric": diagnostic_key,
                "dim": "f",
                "val": 0.3,
                "err": 0.02,
                "n_val": 0.7,
                "n_err": 0.02,
            },
        ]

        expected_keys = syntheval_execution_manifest(
            {preset_name: {"full_output": True}},
            include_holdout_outputs=False,
            target_columns=["Target Label"],
            protected_columns=["Sex"],
            target_is_binary=False,
        )[preset_name]
        successful = build_metric_execution(
            preset_name,
            normalized_rows,
            expected_keys=expected_keys,
        )
        missing_aggregate = build_metric_execution(
            preset_name,
            normalized_rows[1:],
            expected_keys=expected_keys,
        )

        assert successful.status.succeeded is True
        assert successful.status.expected_keys == (aggregate_key,)
        assert successful.status.observed_keys == (aggregate_key, diagnostic_key)
        assert successful.status.unexpected_keys == ()
        assert missing_aggregate.status.succeeded is False
        assert missing_aggregate.status.missing_keys == (aggregate_key,)


class TestSynthcityEmittedKeyFixtures:
    def test_selected_fixture_covers_every_configured_metric(self):
        configured = {
            (category, metric_name)
            for category, metric_names in SYNTHCITY_METRIC_CONFIG.items()
            for metric_name in metric_names
        }
        fixture_keys = {
            (category, metric_name)
            for category, metric_name, _expected_keys in SELECTED_SYNTHCITY_EMITTED_KEY_FIXTURES
        }

        assert fixture_keys == configured

    @pytest.mark.parametrize(
        ("category", "metric_name", "expected_keys"),
        SELECTED_SYNTHCITY_EMITTED_KEY_FIXTURES,
        ids=[
            f"{category}.{metric_name}"
            for category, metric_name, _expected_keys in SELECTED_SYNTHCITY_EMITTED_KEY_FIXTURES
        ],
    )
    def test_selected_metric_manifest_matches_literal_fixture(
        self, category, metric_name, expected_keys
    ):
        assert emitted_keys_for_synthcity_metrics(
            {category: [metric_name]},
            variable_columns=SELECTED_SYNTHCITY_VARIABLE_COLUMNS,
            attack_target_types=SELECTED_SYNTHCITY_ATTACK_TARGET_TYPES,
        ) == list(expected_keys)


class TestLogDisparityMetricsExcludesMedian:
    def test_median_abs_not_a_ranked_metric(self):
        # log_disparity_median_abs is computed from the exact same per-subgroup
        # value array as log_disparity_mean_abs (see metric_log_disparity.py) --
        # deliberately excluded here to avoid double-counting one signal.
        assert "log_disparity_median_abs" not in LOG_DISPARITY_METRICS

    def test_mean_abs_and_share_significant_still_ranked(self):
        assert "log_disparity_mean_abs" in LOG_DISPARITY_METRICS
        assert "log_disparity_share_significant" in LOG_DISPARITY_METRICS
