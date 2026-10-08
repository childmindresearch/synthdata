"""Unit tests for synthdata.config: dataclass composition, validation, YAML loading."""

from pathlib import Path

import pytest
import yaml

from synthdata.config import (
    BinaryTargetConfig,
    Config,
    DataConfig,
    GenerationConfig,
    HPOConfig,
    ImputationConfig,
    RefiDiffConfig,
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

    def test_default_hpo_privacy_objective_excludes_domias(self):
        cfg = _from_dict(Config, {})

        assert cfg.generation.hpo.metric_config["privacy"] == ["identifiability_score"]

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
        assert cfg.generation.hpo.pruner == "median"

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
        "config_name", ["config_hepatitis.yaml", "config_loris.yaml", "config_sim.yaml"]
    )
    def test_shipped_hpo_profiles_exclude_domias(self, config_name):
        root = Path(__file__).parents[2]
        cfg = load_config(root / "configs" / config_name)

        assert cfg.generation.hpo.metric_config["privacy"] == ["identifiability_score"]

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

    def test_removed_privacy_gate_section_raises(self):
        with pytest.raises(ValueError, match="privacy_gate was removed"):
            _from_dict(Config, {"evaluation": {"privacy_gate": {"enabled": True}}})

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

    def test_bad_data_source_raises(self):
        cfg = self._base_valid()
        cfg.data.source = "json"
        with pytest.raises(ValueError, match="data.source"):
            _validate(cfg)

    def test_sensitive_columns_without_protected_columns_fail_as_the_old_meaning(self):
        cfg = self._base_valid()
        cfg.data.sensitive_columns = ["Sex", "Age"]
        with pytest.raises(ValueError, match="protected_columns"):
            _validate(cfg)
        cfg.data.protected_columns = []
        _validate(cfg)  # declared: sensitive_columns now means the secrets

    def test_column_roles_may_overlap_except_quasi_identifier_and_sensitive(self):
        cfg = self._base_valid()
        cfg.data.quasi_identifier_columns = ["Age", "Sex"]
        cfg.data.sensitive_columns = ["Diagnosis"]
        cfg.data.protected_columns = ["Sex", "Diagnosis"]
        _validate(cfg)  # should not raise
        cfg.data.sensitive_columns = ["Diagnosis", "Age"]
        with pytest.raises(ValueError, match=r"\['Age'\] cannot be both"):
            _validate(cfg)

    @pytest.mark.parametrize(
        "key", ["quasi_identifier_columns", "sensitive_columns", "protected_columns"]
    )
    def test_column_roles_exclude_target_and_patient_id(self, key):
        cfg = self._base_valid()
        cfg.data.protected_columns = []
        cfg.data.patient_id_column = "pid"
        for reserved in ("target", "pid"):
            setattr(cfg.data, key, [reserved])
            with pytest.raises(ValueError, match=f"data.{key} must not contain"):
                _validate(cfg)

    def test_column_role_lists_reject_duplicates(self):
        cfg = self._base_valid()
        cfg.data.quasi_identifier_columns = ["Age", "Age"]
        with pytest.raises(ValueError, match="more than once"):
            _validate(cfg)

    def test_patient_id_column_cannot_be_the_target(self):
        cfg = self._base_valid()
        cfg.data.patient_id_column = cfg.data.target_column
        with pytest.raises(ValueError, match="patient_id_column"):
            _validate(cfg)

    def test_hpo_needs_a_tuning_split(self):
        cfg = self._base_valid()
        cfg.generation.hpo.enabled = True
        cfg.data.train_fraction, cfg.data.tuning_fraction = 0.8, 0.0
        with pytest.raises(ValueError, match="tuning_fraction"):
            _validate(cfg)
        cfg.generation.hpo.enabled = False
        _validate(cfg)  # no tuning split needed without HPO

    def test_hpo_objective_and_screens_are_validated(self):
        cfg = self._base_valid()
        assert cfg.generation.hpo.objective == "tstr_macro_f1"
        cfg.generation.hpo.objective = "accuracy"
        with pytest.raises(ValueError, match="hpo.objective"):
            _validate(cfg)
        cfg.generation.hpo.objective = "tstr_macro_auprc"
        cfg.generation.hpo.tstr_seeds = 0
        with pytest.raises(ValueError, match="tstr_seeds"):
            _validate(cfg)
        cfg.generation.hpo.tstr_seeds = 2
        cfg.generation.hpo.constraints.max_out_of_range = 1.5
        with pytest.raises(ValueError, match="max_out_of_range"):
            _validate(cfg)
        cfg.generation.hpo.constraints.max_out_of_range = None
        _validate(cfg)

    def test_split_fractions_must_sum_to_one(self):
        cfg = self._base_valid()
        cfg.data.holdout_fraction = 0.3
        with pytest.raises(ValueError, match="must be 1"):
            _validate(cfg)

    def test_split_fractions_must_be_whole_folds(self):
        cfg = self._base_valid()
        cfg.data.train_fraction, cfg.data.holdout_fraction = 0.6123, 0.1877
        with pytest.raises(ValueError, match="multiple of 1/k"):
            _validate(cfg)

    def test_stratify_bins_align_with_columns(self):
        cfg = self._base_valid()
        cfg.data.stratify_columns = ["target", "age"]
        cfg.data.stratify_bins = [None]
        with pytest.raises(ValueError, match="one entry"):
            _validate(cfg)
        cfg.data.stratify_bins = [None, [60, 30]]
        with pytest.raises(ValueError, match="increasing"):
            _validate(cfg)

    def test_removed_split_keys_point_to_their_replacement(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("data:\n  source: csv\n  path: x.csv\n  train_size: 0.7\n")
        with pytest.raises(ValueError, match="train_fraction"):
            load_config(path)

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

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("unit", "encounter"),
            ("anonymeter.singling_out_mode", "bivariate"),
            ("anonymeter.n_attacks", 0),
            ("anonymeter.singling_out_max_attempts", True),
            ("anonymeter.confidence_level", 1.0),
        ],
    )
    def test_invalid_privacy_attack_settings_raise(self, key, value):
        cfg = self._base_valid()
        *parents, leaf = key.split(".")
        target = cfg.evaluation.privacy_attacks
        for parent in parents:
            target = getattr(target, parent)
        setattr(target, leaf, value)
        with pytest.raises(ValueError, match=leaf):
            _validate(cfg)

    def test_binary_target_disabled_by_default_passes(self):
        cfg = self._base_valid()
        _validate(cfg)  # should not raise

    def test_binary_target_enabled_without_classes_raises(self):
        cfg = self._base_valid()
        cfg.evaluation.class_averaging = "binary"
        cfg.evaluation.binary_target.enabled = True
        with pytest.raises(ValueError, match="binary_target"):
            _validate(cfg)

    def test_binary_target_enabled_with_only_positive_raises(self):
        cfg = self._base_valid()
        cfg.evaluation.class_averaging = "binary"
        cfg.evaluation.binary_target.enabled = True
        cfg.evaluation.binary_target.positive_classes = [0, 1]
        with pytest.raises(ValueError, match="binary_target"):
            _validate(cfg)

    def test_binary_target_overlapping_classes_raises(self):
        cfg = self._base_valid()
        cfg.evaluation.class_averaging = "binary"
        cfg.evaluation.binary_target.enabled = True
        cfg.evaluation.binary_target.positive_classes = [0, 1]
        cfg.evaluation.binary_target.negative_classes = [1, 2]
        with pytest.raises(ValueError, match="binary_target"):
            _validate(cfg)

    def test_binary_target_valid_config_passes(self):
        cfg = self._base_valid()
        cfg.evaluation.class_averaging = "binary"
        cfg.evaluation.binary_target.enabled = True
        cfg.evaluation.binary_target.positive_classes = [0, 1]
        cfg.evaluation.binary_target.negative_classes = [2]
        _validate(cfg)  # should not raise

    def test_class_averaging_defaults_to_ovr_macro(self):
        cfg = self._base_valid()
        assert cfg.evaluation.class_averaging == "ovr_macro"
        _validate(cfg)

    def test_unknown_class_averaging_raises(self):
        cfg = self._base_valid()
        cfg.evaluation.class_averaging = "micro"
        with pytest.raises(ValueError, match="class_averaging"):
            _validate(cfg)

    def test_binary_averaging_without_the_collapse_raises(self):
        cfg = self._base_valid()
        cfg.evaluation.class_averaging = "binary"
        with pytest.raises(ValueError, match="class_averaging"):
            _validate(cfg)

    def test_collapse_with_ovr_macro_averaging_raises(self):
        cfg = self._base_valid()
        cfg.evaluation.binary_target.enabled = True
        cfg.evaluation.binary_target.positive_classes = [0, 1]
        cfg.evaluation.binary_target.negative_classes = [2]
        with pytest.raises(ValueError, match="class_averaging"):
            _validate(cfg)

    def test_zero_tstr_seeds_raise(self):
        cfg = self._base_valid()
        cfg.evaluation.tstr_seeds = 0
        with pytest.raises(ValueError, match="tstr_seeds"):
            _validate(cfg)

    @pytest.mark.parametrize("value", [0, -1, 1.5])
    def test_non_positive_replicates_raise(self, value):
        cfg = self._base_valid()
        cfg.generation.n_replicates = value
        with pytest.raises(ValueError, match="n_replicates"):
            _validate(cfg)

    def test_unknown_baseline_raises(self):
        cfg = self._base_valid()
        cfg.evaluation.baselines = ["train_copy", "holdout"]
        with pytest.raises(ValueError, match="evaluation.baselines"):
            _validate(cfg)

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


class TestLoadConfig:
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
