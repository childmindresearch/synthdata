"""Unit tests for synthdata.config: dataclass composition, validation, YAML loading."""

from pathlib import Path

import pytest
import yaml

from synthdata.config import (
    BinaryTargetConfig,
    Config,
    DataConfig,
    DataSplitConfig,
    GenerationConfig,
    HPOConfig,
    ImputationConfig,
    PrivacyGateConfig,
    RefiDiffConfig,
    StageAScreenConfig,
    SynthEvalExecutionConfig,
    _from_dict,
    _validate,
    load_config,
)

pytestmark = pytest.mark.unit


class TestFromDict:
    def test_none_returns_defaults(self):
        cfg = _from_dict(Config, None)
        assert cfg == Config()

    def test_empty_dict_returns_defaults(self):
        cfg = _from_dict(Config, {})
        assert cfg == Config()

    def test_default_hpo_objective_uses_approved_utility_metrics(self):
        cfg = _from_dict(Config, {})

        assert cfg.generation.hpo.metric_config == {
            "task12": [
                "tstr_macro_f1.v1",
                "mixed_mmd.v1",
                "elastic_net_jsd.v1",
            ]
        }
        metrics = [
            metric for values in cfg.generation.hpo.metric_config.values() for metric in values
        ]
        assert len(metrics) == len(set(metrics)) == 3
        assert cfg.generation.hpo.utility_policy == {
            "metrics": [
                "tstr_macro_f1.v1",
                "mixed_mmd.v1",
                "elastic_net_jsd.v1",
            ],
            "weights": [1 / 3, 1 / 3, 1 / 3],
        }

    def test_flat_fields_applied(self):
        cfg = _from_dict(Config, {"name": "mydata", "seed": 7})
        assert cfg.name == "mydata"
        assert cfg.seed == 7
        # Untouched fields keep their defaults.
        assert cfg.device == "auto"

    def test_nested_dict_builds_nested_dataclass(self):
        cfg = _from_dict(Config, {"data": {"source": "csv", "path": "x.csv"}})
        assert isinstance(cfg.data, DataConfig)
        assert cfg.data.source == "csv"
        assert cfg.data.path == "x.csv"
        # Sibling nested defaults are untouched.
        assert cfg.data.target_column == "target"

    def test_doubly_nested_dict(self):
        cfg = _from_dict(
            Config,
            {"generation": {"hpo": {"n_trials": 3}, "n_samples": 50}},
        )
        assert isinstance(cfg.generation, GenerationConfig)
        assert isinstance(cfg.generation.hpo, HPOConfig)
        assert cfg.generation.hpo.n_trials == 3
        assert cfg.generation.n_samples == 50
        # HPOConfig's other defaults are preserved.
        assert cfg.generation.hpo.n_iter_cap == 300

    def test_stage_a_hpo_config_builds_nested_dataclass(self):
        cfg = _from_dict(
            Config,
            {
                "generation": {
                    "hpo": {
                        "stage_a": {
                            "minimum_class_count": 2,
                            "dependency_rules": [
                                {"child": "derived", "parents": ["group", "target"]}
                            ],
                        }
                    }
                }
            },
        )

        assert isinstance(cfg.generation.hpo.stage_a, StageAScreenConfig)
        assert cfg.generation.hpo.stage_a.minimum_class_count == 2
        assert cfg.generation.hpo.stage_a.dependency_rules[0]["child"] == "derived"

    def test_unknown_top_level_key_raises(self):
        with pytest.raises(ValueError, match="Unknown config key"):
            _from_dict(Config, {"not_a_real_field": 1})

    def test_unknown_nested_key_raises(self):
        with pytest.raises(ValueError, match="Unknown config key"):
            _from_dict(Config, {"data": {"not_a_real_field": 1}})

    def test_refidiff_nested_dict_builds_nested_dataclass(self):
        cfg = _from_dict(
            Config,
            {"imputation": {"method": "refidiff", "refidiff": {"hidden_dim": 64}}},
        )
        assert isinstance(cfg.imputation, ImputationConfig)
        assert isinstance(cfg.imputation.refidiff, RefiDiffConfig)
        assert cfg.imputation.method == "refidiff"
        assert cfg.imputation.refidiff.hidden_dim == 64
        # Sibling RefiDiffConfig defaults are preserved.
        assert cfg.imputation.refidiff.denoiser == "auto"

    @pytest.mark.parametrize(
        "config_name",
        ["config_loris_refidiff_reference.yaml", "config_loris_refidiff_hpo.yaml"],
    )
    def test_refidiff_benchmark_profiles_load(self, config_name):
        root = Path(__file__).parents[2]
        cfg = load_config(root / "configs" / config_name)
        assert cfg.imputation.method == "refidiff"
        assert cfg.imputation.refidiff.denoiser == "mamba"
        assert cfg.imputation.refidiff.catboost_warmup_iterations == 1000
        assert cfg.imputation.benchmark.enabled

    def test_shipped_hepatitis_hpo_profile_uses_native_metrics(self):
        root = Path(__file__).parents[2]
        cfg = load_config(root / "configs" / "config_hepatitis.yaml")

        assert cfg.generation.hpo.metric_config == {
            "stats": ["wasserstein_dist", "inv_kl_divergence"],
            "sanity": ["nearest_syn_neighbor_distance"],
            "performance": ["xgb"],
        }

    def test_shipped_loris_hpo_profile_uses_exact_canonical_metrics(self):
        root = Path(__file__).parents[2]
        cfg = load_config(root / "configs" / "config_loris.yaml")
        metric_config = cfg.generation.hpo.metric_config
        metrics = [metric for values in metric_config.values() for metric in values]

        assert metric_config == {
            "task12": [
                "tstr_macro_f1.v1",
                "mixed_mmd.v1",
                "elastic_net_jsd.v1",
            ]
        }
        assert len(metrics) == len(set(metrics)) == 3
        assert all(metric.endswith(".v1") for metric in metrics)
        assert cfg.generation.hpo.utility_policy == {
            "metrics": metrics,
            "weights": [1 / 3, 1 / 3, 1 / 3],
        }

    def test_evaluation_binary_target_nested_dict_builds_nested_dataclass(self):
        cfg = _from_dict(
            Config,
            {
                "evaluation": {
                    "binary_target": {
                        "enabled": True,
                        "positive_classes": [0, 1],
                        "negative_classes": [2],
                    }
                }
            },
        )
        assert isinstance(cfg.evaluation.binary_target, BinaryTargetConfig)
        assert cfg.evaluation.binary_target.enabled is True
        assert cfg.evaluation.binary_target.positive_classes == [0, 1]
        assert cfg.evaluation.binary_target.negative_classes == [2]

    def test_structural_privacy_settings_load(self):
        cfg = _from_dict(
            Config,
            {
                "evaluation": {
                    "synthcity": {
                        "structural_n_clusters": [2, 7],
                        "structural_min_rows_per_cluster": 12,
                    }
                }
            },
        )

        assert cfg.evaluation.synthcity.structural_n_clusters == [2, 7]
        assert cfg.evaluation.synthcity.structural_min_rows_per_cluster == 12

    def test_evaluation_privacy_gate_nested_dict_builds_nested_dataclass(self):
        cfg = _from_dict(
            Config,
            {
                "evaluation": {
                    "privacy_gate": {
                        "enabled": False,
                        "thresholds": {"mia_recall": {"bound": "max", "value": 0.7}},
                    }
                }
            },
        )
        assert isinstance(cfg.evaluation.privacy_gate, PrivacyGateConfig)
        assert cfg.evaluation.privacy_gate.enabled is False
        assert cfg.evaluation.privacy_gate.thresholds == {
            "mia_recall": {"bound": "max", "value": 0.7}
        }

    def test_syntheval_execution_nested_dict_builds_nested_dataclass(self):
        cfg = _from_dict(
            Config,
            {"evaluation": {"syntheval_execution": {"model_workers": 3}}},
        )
        assert isinstance(cfg.evaluation.syntheval_execution, SynthEvalExecutionConfig)
        assert cfg.evaluation.syntheval_execution.model_workers == 3


class TestValidate:
    def _base_valid(self, **overrides) -> Config:
        cfg = Config(data=DataConfig(source="csv", path="x.csv", target_column="target"))
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg

    def test_valid_config_passes(self):
        _validate(self._base_valid())  # should not raise

    @pytest.mark.parametrize(
        "field, value",
        [
            ("structural_n_clusters", []),
            ("structural_n_clusters", [2, 2]),
            ("structural_n_clusters", [1, 5]),
            ("structural_min_rows_per_cluster", 0),
        ],
    )
    def test_invalid_structural_privacy_settings_raise(self, field, value):
        cfg = self._base_valid()
        setattr(cfg.evaluation.synthcity, field, value)

        with pytest.raises(ValueError, match="structural"):
            _validate(cfg)

    def test_bad_data_source_raises(self):
        cfg = self._base_valid()
        cfg.data.source = "json"
        with pytest.raises(ValueError, match="data.source"):
            _validate(cfg)

    def test_uci_requires_uci_id(self):
        cfg = Config(data=DataConfig(source="uci", uci_id=None, target_column="target"))
        with pytest.raises(ValueError, match="data.uci_id"):
            _validate(cfg)

    def test_uci_with_id_is_valid(self):
        cfg = Config(data=DataConfig(source="uci", uci_id=42, target_column="target"))
        _validate(cfg)  # should not raise

    def test_csv_requires_path(self):
        cfg = Config(data=DataConfig(source="csv", path=None, target_column="target"))
        with pytest.raises(ValueError, match="data.path"):
            _validate(cfg)

    def test_parquet_requires_path(self):
        cfg = Config(data=DataConfig(source="parquet", path=None, target_column="target"))
        with pytest.raises(ValueError, match="data.path"):
            _validate(cfg)

    def test_parquet_with_path_is_valid(self):
        cfg = Config(data=DataConfig(source="parquet", path="x.parquet", target_column="target"))
        _validate(cfg)  # should not raise

    def test_empty_target_column_raises(self):
        cfg = self._base_valid()
        cfg.data.target_column = ""
        with pytest.raises(ValueError, match="target_column"):
            _validate(cfg)

    @pytest.mark.parametrize("device", ["gpu", "tpu", ""])
    def test_bad_device_raises(self, device):
        cfg = self._base_valid()
        cfg.device = device
        with pytest.raises(ValueError, match="device"):
            _validate(cfg)

    @pytest.mark.parametrize("device", ["auto", "cpu", "cuda", "mps"])
    def test_valid_devices_pass(self, device):
        cfg = self._base_valid()
        cfg.device = device
        _validate(cfg)  # should not raise

    def test_bad_ranking_strategy_raises(self):
        cfg = self._base_valid()
        cfg.evaluation.ranking_strategy = "bogus"
        with pytest.raises(ValueError, match="ranking_strategy"):
            _validate(cfg)

    def test_group_mode_defaults_to_rows(self):
        cfg = self._base_valid()
        _validate(cfg)
        assert cfg.evaluation.group_mode == "row"
        assert cfg.evaluation.group_column is None

    def test_bad_group_mode_raises(self):
        cfg = self._base_valid()
        cfg.evaluation.group_mode = "encounter"
        with pytest.raises(ValueError, match="evaluation.group_mode"):
            _validate(cfg)

    def test_patient_group_requires_group_column(self):
        cfg = self._base_valid()
        cfg.evaluation.group_mode = "patient_group"
        with pytest.raises(ValueError, match="evaluation.group_column"):
            _validate(cfg)

    def test_patient_group_with_identifier_passes(self):
        cfg = self._base_valid()
        cfg.evaluation.group_mode = "patient_group"
        cfg.evaluation.group_column = "patient_id"
        _validate(cfg)

    @pytest.mark.parametrize("group_column", ["", "  ", 7])
    def test_invalid_group_column_raises(self, group_column):
        cfg = self._base_valid()
        cfg.evaluation.group_column = group_column
        with pytest.raises(ValueError, match="evaluation.group_column"):
            _validate(cfg)

    def test_group_column_cannot_be_dropped(self):
        cfg = self._base_valid()
        cfg.evaluation.group_mode = "patient_group"
        cfg.evaluation.group_column = "patient_id"
        cfg.data.drop_columns = ["patient_id"]
        with pytest.raises(ValueError, match="data.drop_columns"):
            _validate(cfg)

    @pytest.mark.parametrize(
        "field, value, message",
        [
            ("target_column", "patient_id", "target/identity"),
            ("protected_columns", ["patient_id"], "declared/identity"),
            ("drop_columns", ["patient_id"], "drop/identity"),
            ("drop_columns", ["target"], "target/drop"),
        ],
    )
    def test_split_column_declaration_conflicts_raise(self, field, value, message):
        cfg = self._base_valid()
        cfg.data.split = DataSplitConfig(
            mode="patient_group",
            patient_id_column="patient_id",
        )
        setattr(cfg.data, field, value)

        with pytest.raises(ValueError, match=message):
            _validate(cfg)

    def test_encounter_label_cannot_be_target_or_dropped(self):
        cfg = self._base_valid()
        cfg.data.split = DataSplitConfig(
            mode="patient_group",
            patient_id_column="patient_id",
            encounter_label_column="encounter",
        )
        cfg.data.drop_columns = ["encounter"]

        with pytest.raises(ValueError, match="encounter/drop"):
            _validate(cfg)

    def test_mapping_patient_key_is_checked_as_identity_column(self):
        cfg = self._base_valid()
        cfg.data.split = DataSplitConfig(
            mode="patient_group",
            identity_mapping_path="identity.csv",
            mapping_row_key_column="row_id",
            mapping_patient_key_column="patient_id",
        )
        cfg.data.quasi_identifier_columns = ["patient_id"]

        with pytest.raises(ValueError, match="declared/identity"):
            _validate(cfg)

    @pytest.mark.parametrize("workers", [0, -1, "many", 1.5])
    def test_invalid_syntheval_model_workers_raise(self, workers):
        cfg = self._base_valid()
        cfg.evaluation.syntheval_execution.model_workers = workers
        with pytest.raises(ValueError, match="syntheval_execution.model_workers"):
            _validate(cfg)

    def test_valid_explicit_syntheval_model_workers_pass(self):
        cfg = self._base_valid()
        cfg.evaluation.syntheval_execution.model_workers = 3
        _validate(cfg)

    @pytest.mark.parametrize("field", ["max_model_workers", "cores_per_model"])
    def test_non_positive_syntheval_integer_bounds_raise(self, field):
        cfg = self._base_valid()
        setattr(cfg.evaluation.syntheval_execution, field, 0)
        with pytest.raises(ValueError, match=field):
            _validate(cfg)

    @pytest.mark.parametrize("field", ["memory_reserve_gib", "memory_per_model_gib"])
    def test_non_positive_syntheval_memory_bounds_raise(self, field):
        cfg = self._base_valid()
        setattr(cfg.evaluation.syntheval_execution, field, 0)
        with pytest.raises(ValueError, match=field):
            _validate(cfg)

    def test_bad_tabpfn_data_variant_raises(self):
        cfg = self._base_valid()
        cfg.generation.tabpfn.data_variants = ["raw", "bogus"]
        with pytest.raises(ValueError, match="data_variants"):
            _validate(cfg)

    def test_valid_tabpfn_data_variants_pass(self):
        cfg = self._base_valid()
        cfg.generation.tabpfn.data_variants = ["raw", "imputed"]
        _validate(cfg)  # should not raise

    def test_bad_imputation_method_raises(self):
        cfg = self._base_valid()
        cfg.imputation.method = "bogus"
        with pytest.raises(ValueError, match="imputation.method"):
            _validate(cfg)

    @pytest.mark.parametrize("method", ["tabimpute", "refidiff"])
    def test_valid_imputation_methods_pass(self, method):
        cfg = self._base_valid()
        cfg.imputation.method = method
        _validate(cfg)  # should not raise

    def test_hyperimpute_is_default_and_continuous_plugin_is_validated(self):
        cfg = self._base_valid()
        assert cfg.imputation.method == "hyperimpute"
        assert cfg.imputation.continuous_plugin == "median"
        cfg.imputation.continuous_plugin = "mean"
        _validate(cfg)
        cfg.imputation.continuous_plugin = "bogus"
        with pytest.raises(ValueError, match="continuous_plugin"):
            _validate(cfg)

    def test_bad_refidiff_denoiser_raises(self):
        cfg = self._base_valid()
        cfg.imputation.refidiff.denoiser = "bogus"
        with pytest.raises(ValueError, match="imputation.refidiff.denoiser"):
            _validate(cfg)

    @pytest.mark.parametrize("denoiser", ["auto", "mamba", "mlp"])
    def test_valid_refidiff_denoisers_pass(self, denoiser):
        cfg = self._base_valid()
        cfg.imputation.refidiff.denoiser = denoiser
        _validate(cfg)  # should not raise

    def test_bad_refidiff_categorical_decode_policy_raises(self):
        cfg = self._base_valid()
        cfg.imputation.refidiff.categorical_decode_policy = "unknown"
        with pytest.raises(ValueError, match="categorical_decode_policy"):
            _validate(cfg)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("hidden_dim", 0),
            ("epochs", 0),
            ("early_stopping_patience", 0),
            ("batch_size", 0),
            ("num_trials", 0),
            ("checkpoint_every", 0),
            ("catboost_warmup_iterations", 0),
        ],
    )
    def test_non_positive_refidiff_integer_raises(self, field, value):
        cfg = self._base_valid()
        setattr(cfg.imputation.refidiff, field, value)
        with pytest.raises(ValueError, match=field):
            _validate(cfg)

    @pytest.mark.parametrize("num_steps", [0, 1])
    def test_refidiff_requires_at_least_two_sampling_steps(self, num_steps):
        cfg = self._base_valid()
        cfg.imputation.refidiff.num_steps = num_steps
        with pytest.raises(ValueError, match="num_steps"):
            _validate(cfg)

    def test_invalid_refidiff_benchmark_mask_fraction_raises(self):
        cfg = self._base_valid()
        cfg.imputation.benchmark.mask_fraction = 1.0
        with pytest.raises(ValueError, match="benchmark.mask_fraction"):
            _validate(cfg)

    def test_invalid_refidiff_benchmark_mechanism_raises(self):
        cfg = self._base_valid()
        cfg.imputation.benchmark.mechanisms = ["unknown"]
        with pytest.raises(ValueError, match="benchmark.mechanisms"):
            _validate(cfg)

    def test_ordinal_column_not_in_ordinal_columns_raises(self):
        cfg = self._base_valid()
        cfg.data.ordinal_columns = []
        cfg.data.ordinal_column_categories = {"activity": ["Light", "Heavy"]}
        with pytest.raises(ValueError, match="ordinal_column_categories"):
            _validate(cfg)

    def test_ordinal_column_in_ordinal_columns_passes(self):
        cfg = self._base_valid()
        cfg.data.ordinal_columns = ["activity"]
        cfg.data.ordinal_column_categories = {"activity": ["Light", "Heavy"]}
        _validate(cfg)  # should not raise

    def test_nominal_and_ordinal_columns_overlap_raises(self):
        cfg = self._base_valid()
        cfg.data.nominal_columns = ["activity"]
        cfg.data.ordinal_columns = ["activity"]
        with pytest.raises(ValueError, match="nominal_columns"):
            _validate(cfg)

    def test_nominal_and_ordinal_columns_disjoint_passes(self):
        cfg = self._base_valid()
        cfg.data.nominal_columns = ["other_cat"]
        cfg.data.ordinal_columns = ["activity"]
        _validate(cfg)  # should not raise

    def test_ordinal_column_categories_not_a_list_raises(self):
        cfg = self._base_valid()
        cfg.data.ordinal_column_categories = {"activity": "Light"}
        with pytest.raises(ValueError, match="ordinal_column_categories"):
            _validate(cfg)

    def test_ordinal_column_categories_with_duplicates_raises(self):
        cfg = self._base_valid()
        cfg.data.ordinal_column_categories = {"activity": ["Light", "Light"]}
        with pytest.raises(ValueError, match="ordinal_column_categories"):
            _validate(cfg)

    def test_schema_takes_precedence_over_legacy_column_typing(self):
        cfg = self._base_valid()
        cfg.data.variable_schema_path = "schema.csv"
        cfg.data.nominal_columns = ["category"]
        _validate(cfg)

    def test_auto_nominal_typing_is_rejected(self):
        cfg = self._base_valid()
        cfg.data.nominal_columns = "auto"
        with pytest.raises(ValueError, match="no longer supported"):
            _validate(cfg)

    def test_binary_target_disabled_by_default_passes(self):
        cfg = self._base_valid()
        _validate(cfg)  # should not raise

    def test_binary_target_enabled_without_classes_raises(self):
        cfg = self._base_valid()
        cfg.evaluation.binary_target.enabled = True
        with pytest.raises(ValueError, match="binary_target"):
            _validate(cfg)

    def test_binary_target_enabled_with_only_positive_raises(self):
        cfg = self._base_valid()
        cfg.evaluation.binary_target.enabled = True
        cfg.evaluation.binary_target.positive_classes = [0, 1]
        with pytest.raises(ValueError, match="binary_target"):
            _validate(cfg)

    def test_binary_target_overlapping_classes_raises(self):
        cfg = self._base_valid()
        cfg.evaluation.binary_target.enabled = True
        cfg.evaluation.binary_target.positive_classes = [0, 1]
        cfg.evaluation.binary_target.negative_classes = [1, 2]
        with pytest.raises(ValueError, match="binary_target"):
            _validate(cfg)

    def test_binary_target_valid_config_passes(self):
        cfg = self._base_valid()
        cfg.evaluation.binary_target.enabled = True
        cfg.evaluation.binary_target.positive_classes = [0, 1]
        cfg.evaluation.binary_target.negative_classes = [2]
        _validate(cfg)  # should not raise

    def test_rank_weights_default_passes(self):
        cfg = self._base_valid()
        _validate(cfg)  # should not raise

    def test_rank_weights_missing_key_raises(self):
        cfg = self._base_valid()
        cfg.evaluation.rank_weights = {"utility": 1.0, "privacy": 1.0}
        with pytest.raises(ValueError, match="rank_weights"):
            _validate(cfg)

    def test_rank_weights_extra_key_raises(self):
        cfg = self._base_valid()
        cfg.evaluation.rank_weights = {
            "utility": 1.0,
            "privacy": 1.0,
            "fairness": 1.0,
            "bogus": 1.0,
        }
        with pytest.raises(ValueError, match="rank_weights"):
            _validate(cfg)

    def test_rank_weights_negative_value_raises(self):
        cfg = self._base_valid()
        cfg.evaluation.rank_weights = {"utility": 1.0, "privacy": -0.5, "fairness": 1.0}
        with pytest.raises(ValueError, match="rank_weights"):
            _validate(cfg)

    def test_rank_weights_zero_is_allowed(self):
        cfg = self._base_valid()
        cfg.evaluation.rank_weights = {"utility": 1.0, "privacy": 0.0, "fairness": 1.0}
        _validate(cfg)  # should not raise

    def test_privacy_gate_default_thresholds_pass(self):
        cfg = self._base_valid()
        _validate(cfg)  # should not raise

    def test_privacy_gate_missing_bound_or_value_raises(self):
        cfg = self._base_valid()
        cfg.evaluation.privacy_gate.thresholds = {"mia_recall": {"value": 0.6}}
        with pytest.raises(ValueError, match="privacy_gate.thresholds"):
            _validate(cfg)

    def test_privacy_gate_bad_bound_raises(self):
        cfg = self._base_valid()
        cfg.evaluation.privacy_gate.thresholds = {"mia_recall": {"bound": "sideways", "value": 0.6}}
        with pytest.raises(ValueError, match="privacy_gate.thresholds"):
            _validate(cfg)

    def test_privacy_gate_non_numeric_value_raises(self):
        cfg = self._base_valid()
        cfg.evaluation.privacy_gate.thresholds = {"mia_recall": {"bound": "max", "value": "high"}}
        with pytest.raises(ValueError, match="privacy_gate.thresholds"):
            _validate(cfg)

    def test_enabled_privacy_gate_requires_explicit_contract_ids(self):
        cfg = self._base_valid()
        cfg.evaluation.privacy_gate.enabled = True
        with pytest.raises(ValueError, match="explicit.*contract_id"):
            _validate(cfg)


class TestLoadConfig:
    @staticmethod
    def _canonical_fixture() -> Config:
        """Build canonical config with all fixed policy metadata populated."""
        data = DataConfig(
            source="csv",
            path="x.csv",
            target_column="target",
            canonical=True,
            patient_id_column="patient_id",
            split=DataSplitConfig(mode="patient_group"),
        )
        cfg = Config(data=data)
        cfg.imputation.method = "hyperimpute"
        cfg.generation.hpo.utility_policy_provenance = "Task 13 consumes fixed utility_policy."
        cfg.evaluation.release_generalization.columns = {
            "Age": {
                "intervals": [
                    {"label": "<18", "lower": None, "upper": 18},
                    {"label": ">=18", "lower": 18, "upper": None},
                ]
            }
        }
        return cfg

    def test_loris_canonical_policy_loads(self):
        cfg = load_config(Path(__file__).parents[2] / "configs" / "config_loris.yaml")
        assert cfg.data.patient_id_column == "patient_id"
        assert cfg.data.quasi_identifier_columns == [
            "Age",
            "Sex",
            "region",
            "PreInt_Demos_Fam__Child_Ethnicity",
        ]
        assert cfg.evaluation.privacy_policy.k_required == 5
        assert cfg.evaluation.privacy_policy.mia_epsilon_repetitions == 10
        assert cfg.data.split.patient_id_column is None
        assert cfg.generation.hpo.utility_policy_provenance.startswith("Task 13")

    def test_loris_hpo_policy_provenance_describes_fixed_fail_closed_policy(self):
        cfg = load_config(Path(__file__).parents[2] / "configs" / "config_loris.yaml")
        provenance = cfg.generation.hpo.utility_policy_provenance
        assert "consumes fixed generation.hpo.utility_policy" in provenance
        assert "mismatches fail closed" in provenance
        assert "not wired" not in provenance

    def test_canonical_requires_direct_patient_id(self):
        cfg = Config(data=DataConfig(source="csv", path="x.csv", canonical=True))
        with pytest.raises(ValueError, match="patient_id_column"):
            _validate(cfg)

    @pytest.mark.parametrize("overlap", ["sensitive", "target", "patient"])
    def test_canonical_forbids_qi_role_overlap(self, overlap):
        cfg = self._canonical_fixture()
        data = cfg.data
        data.quasi_identifier_columns = {
            "sensitive": ["secret"],
            "target": ["target"],
            "patient": ["patient_id"],
        }[overlap]
        if overlap == "sensitive":
            data.sensitive_columns = ["secret"]
        with pytest.raises(ValueError, match="(quasi_identifier_columns|target_column)"):
            _validate(cfg)

    def test_canonical_qi_protected_overlap_is_allowed(self):
        cfg = self._canonical_fixture()
        cfg.data.quasi_identifier_columns = ["age"]
        cfg.data.protected_columns = ["age"]
        _validate(cfg)

    def test_canonical_rejects_malformed_release_intervals(self):
        cfg = self._canonical_fixture()
        cfg.evaluation.release_generalization.columns = {
            "Age": {"intervals": [{"label": "bad", "lower": 2, "upper": 1}]}
        }
        with pytest.raises(ValueError, match="lower >= upper"):
            _validate(cfg)

    @pytest.mark.parametrize(
        "intervals",
        [
            [
                {"label": "<18", "lower": None, "upper": None},
                {"label": ">60", "lower": 61, "upper": None},
            ],
            [
                {"label": "<18", "lower": None, "upper": 18},
                {"label": "middle", "lower": 18, "upper": None},
                {"label": ">60", "lower": 61, "upper": None},
            ],
        ],
    )
    def test_canonical_rejects_misordered_open_ended_release_intervals(self, intervals):
        cfg = self._canonical_fixture()
        cfg.evaluation.release_generalization.columns = {"Age": {"intervals": intervals}}

        with pytest.raises(ValueError, match="unbounded|contiguous"):
            _validate(cfg)

    def test_canonical_invalid_support_setting_is_rejected(self):
        cfg = self._canonical_fixture()
        cfg.evaluation.privacy_policy.k_required = 0
        with pytest.raises(ValueError, match="k_required"):
            _validate(cfg)

    def test_canonical_missing_anchor_is_rejected(self):
        cfg = self._canonical_fixture()
        cfg.evaluation.scoring_policy.bh_alpha = None
        with pytest.raises(ValueError, match="bh_alpha"):
            _validate(cfg)

    def test_legacy_profile_cannot_declare_canonical_patient_id(self):
        cfg = Config(data=DataConfig(source="csv", path="x.csv", patient_id_column="subject_id"))
        with pytest.raises(ValueError, match="canonical"):
            _validate(cfg)

    @pytest.mark.parametrize(
        "mutator, message",
        [
            (lambda c: setattr(c.data.split, "identity_mapping_path", "map.csv"), "mapping"),
            (lambda c: setattr(c.data.split, "one_row_per_patient", True), "one_row_per_patient"),
            (lambda c: setattr(c.data.split, "mode", "row"), "identity settings"),
            (lambda c: setattr(c.evaluation, "group_mode", "patient_group"), "group_mode"),
            (lambda c: setattr(c.evaluation, "group_column", "patient_id"), "group_column"),
        ],
    )
    def test_canonical_forbids_legacy_identity_settings(self, mutator, message):
        cfg = load_config(Path(__file__).parents[2] / "configs" / "config_loris.yaml")
        mutator(cfg)
        with pytest.raises(ValueError, match=message):
            _validate(cfg)

    def test_canonical_nested_identity_is_rejected_at_load(self, tmp_path):
        source = Path(__file__).parents[2] / "configs" / "config_loris.yaml"
        raw = yaml.safe_load(source.read_text())
        raw["data"]["split"]["patient_id_column"] = "patient_id"
        path = tmp_path / "nested.yaml"
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError, match="nested split"):
            load_config(path)

    def test_canonical_nested_identity_is_rejected_directly(self):
        cfg = self._canonical_fixture()
        cfg.data.split.patient_id_column = "patient_id"
        with pytest.raises(ValueError, match="nested split identity"):
            _validate(cfg)

    @pytest.mark.parametrize(
        "path_parts",
        [
            ("evaluation", "privacy_policy"),
            ("evaluation", "scoring_policy"),
            ("evaluation", "release_generalization"),
            ("generation", "hpo", "utility_policy"),
            ("generation", "hpo", "utility_policy_provenance"),
        ],
    )
    def test_canonical_omitted_required_policy_fields_fail_closed(self, tmp_path, path_parts):
        source = Path(__file__).parents[2] / "configs" / "config_loris.yaml"
        raw = yaml.safe_load(source.read_text())
        value = raw
        for key in path_parts[:-1]:
            value = value[key]
        del value[path_parts[-1]]
        path = tmp_path / ("missing-" + "-".join(path_parts) + ".yaml")
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError, match=path_parts[-1]):
            load_config(path)

    @pytest.mark.parametrize(
        "block, field",
        [
            ("privacy_policy", "k_required"),
            ("privacy_policy", "l_required"),
            ("privacy_policy", "role_population_floor"),
            ("privacy_policy", "protected_slice_floor"),
            ("privacy_policy", "mia_epsilon_repetitions"),
            ("privacy_policy", "epsilon_excess_anchor"),
            ("privacy_policy", "mia_advantage_anchor"),
            ("privacy_policy", "attribute_disclosure_anchor"),
            ("scoring_policy", "equalized_odds_gap_anchor"),
            ("scoring_policy", "worst_absolute_log_disparity_anchor"),
            ("scoring_policy", "bh_alpha"),
            ("scoring_policy", "practical_log_disparity_floor"),
            ("scoring_policy", "valid_comparison_fraction"),
        ],
    )
    def test_canonical_omitted_policy_value_fails_closed(self, tmp_path, block, field):
        source = Path(__file__).parents[2] / "configs" / "config_loris.yaml"
        raw = yaml.safe_load(source.read_text())
        del raw["evaluation"][block][field]
        path = tmp_path / f"missing-{block}-{field}.yaml"
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError, match=field):
            load_config(path)

    @pytest.mark.parametrize(
        "config_name",
        [
            "config_hepatitis.yaml",
            "config_loris_refidiff_reference.yaml",
            "config_loris_refidiff_hpo.yaml",
        ],
    )
    def test_legacy_profiles_are_explicitly_noncanonical(self, config_name):
        root = Path(__file__).parents[2]
        cfg = load_config(root / "configs" / config_name)
        assert cfg.data.canonical is False

    @pytest.mark.parametrize(
        "config_name",
        [
            "config_hepatitis.yaml",
            "config_loris_refidiff_reference.yaml",
            "config_loris_refidiff_hpo.yaml",
        ],
    )
    def test_legacy_profiles_marked_canonical_fail_actionably(self, tmp_path, config_name):
        root = Path(__file__).parents[2]
        raw = yaml.safe_load((root / "configs" / config_name).read_text())
        raw["data"]["canonical"] = True
        path = tmp_path / config_name
        path.write_text(yaml.safe_dump(raw))

        with pytest.raises(ValueError, match="Canonical evaluation (requires|rejects)"):
            load_config(path)

    def test_canonical_omitted_policy_block_fails_closed(self, tmp_path):
        source = Path(__file__).parents[2] / "configs" / "config_loris.yaml"
        raw = yaml.safe_load(source.read_text())
        del raw["evaluation"]["privacy_policy"]
        path = tmp_path / "missing-policy.yaml"
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError, match="privacy_policy"):
            load_config(path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "does_not_exist.yaml")

    def test_valid_yaml_loaded_and_validated(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(
            "name: mydata\ndata:\n  source: csv\n  path: raw.csv\n  target_column: outcome\n"
        )
        cfg = load_config(yaml_path)
        assert cfg.name == "mydata"
        assert cfg.data.source == "csv"
        assert cfg.data.target_column == "outcome"
        assert cfg.config_path == yaml_path.resolve()

    def test_empty_yaml_raises_because_defaults_need_uci_id(self, tmp_path):
        # Config()'s default data.source is "uci" with no uci_id -- an empty
        # YAML file is therefore invalid on its own (must specify a source).
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text("")
        with pytest.raises(ValueError, match="data.uci_id"):
            load_config(yaml_path)

    def test_invalid_config_raises_on_load(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text("data:\n  source: not_a_real_source\n")
        with pytest.raises(ValueError, match="data.source"):
            load_config(yaml_path)

    def test_malformed_yaml_raises(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text("data: [unclosed\n")
        with pytest.raises(yaml.YAMLError):
            load_config(yaml_path)
