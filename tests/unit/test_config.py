"""Unit tests for synthdata.config: dataclass composition, validation, YAML loading."""

import dataclasses
import shutil
from pathlib import Path
from typing import cast
from uuid import uuid4

import pandas as pd
import pytest
import yaml

from synthdata.config import (
    Config,
    DataConfig,
    DataSplitConfig,
    EvaluationConfig,
    GenerationConfig,
    HPOConfig,
    ImputationConfig,
    PrivacyGateConfig,
    PrivacyPolicyConfig,
    RefiDiffConfig,
    ScoringPolicyConfig,
    StageAScreenConfig,
    SynthEvalExecutionConfig,
    _from_dict,
    _protected_attribute_bin_intervals,
    _validate,
    load_config,
)
from synthdata.data import _configured_stratification_frame
from synthdata.evaluation.release import transform_release_roles
from synthdata.evaluation.syntheval_eval import resolve_model_workers

pytestmark = pytest.mark.unit

REMOVED_POLICY_SETTINGS = [
    (EvaluationConfig, "", "rank_weights", {"utility": 1.0, "privacy": 1.0, "fairness": 1.0}),
    (PrivacyPolicyConfig, "privacy_policy", "k_required", 5),
    (PrivacyPolicyConfig, "privacy_policy", "l_required", 2),
    (PrivacyPolicyConfig, "privacy_policy", "mia_epsilon_repetitions", 10),
    (PrivacyPolicyConfig, "privacy_policy", "epsilon_excess_anchor", 0.10),
    (PrivacyPolicyConfig, "privacy_policy", "mia_advantage_anchor", 0.10),
    (PrivacyPolicyConfig, "privacy_policy", "attribute_disclosure_anchor", 0.20),
    (ScoringPolicyConfig, "scoring_policy", "bh_alpha", 0.05),
    (ScoringPolicyConfig, "scoring_policy", "practical_log_disparity_floor", 0.22314355131),
    (ScoringPolicyConfig, "scoring_policy", "valid_comparison_fraction", 0.80),
]


@pytest.mark.parametrize("cls,block,field,value", REMOVED_POLICY_SETTINGS)
@pytest.mark.parametrize(
    "route", ["mapping", "yaml", "constructor", "assignment", "injected", "replacement"]
)
def test_removed_policy_settings_fail_with_migration_path(
    tmp_path, cls, block, field, value, route
):
    path = ".".join(part for part in ("evaluation", block, field) if part)
    settings = {field: value}
    raw = {"data": {"source": "csv", "path": "x.csv"}, "evaluation": {}}
    raw["evaluation"] = {block: settings} if block else settings
    with pytest.raises(ValueError, match=path.replace(".", r"\.") + ".*remove"):
        if route == "mapping":
            _from_dict(Config, raw)
        elif route == "yaml":
            yaml_path = tmp_path / "removed.yaml"
            yaml_path.write_text(yaml.safe_dump(raw))
            load_config(yaml_path)
        elif route == "constructor":
            cls(**settings)
        else:
            cfg = Config(data=DataConfig(source="csv", path="x.csv"))
            owner = getattr(cfg.evaluation, block) if block else cfg.evaluation
            if route == "assignment":
                setattr(owner, field, value)
            elif route == "replacement" and block:
                setattr(cfg.evaluation, block, settings)
                _validate(cfg)
            else:
                owner.__dict__[field] = value
                _validate(cfg)


@pytest.mark.parametrize("cls,block,field,value", REMOVED_POLICY_SETTINGS)
def test_removed_policy_settings_are_absent_from_schema(cls, block, field, value):
    assert field not in {entry.name for entry in dataclasses.fields(cls)}
    assert not hasattr(cls(), field)


@pytest.mark.parametrize("field", ["role_population_floor", "protected_slice_floor"])
@pytest.mark.parametrize("value", [True, False, None, 0, -1, 1.5, float("nan"), float("inf"), "2"])
def test_retained_support_floors_reject_invalid_values(field, value):
    cfg = Config(data=DataConfig(source="csv", path="x.csv"))
    setattr(cfg.evaluation.privacy_policy, field, value)
    with pytest.raises(ValueError, match=f"evaluation.privacy_policy.{field}"):
        _validate(cfg)


@pytest.mark.parametrize(
    "field", ["equalized_odds_gap_anchor", "worst_absolute_log_disparity_anchor"]
)
@pytest.mark.parametrize(
    "value", [True, False, None, 0, -1, float("nan"), float("inf"), -float("inf"), "2"]
)
def test_retained_fairness_anchors_reject_invalid_values(field, value):
    cfg = Config(data=DataConfig(source="csv", path="x.csv"))
    setattr(cfg.evaluation.scoring_policy, field, value)
    with pytest.raises(ValueError, match=f"evaluation.scoring_policy.{field}"):
        _validate(cfg)


def test_retained_policy_nondefaults_load_and_validate(tmp_path):
    raw = TestLoadConfig._canonical_yaml_data()
    raw["evaluation"]["privacy_policy"] = {"role_population_floor": 30, "protected_slice_floor": 3}
    raw["evaluation"]["scoring_policy"] = {
        "equalized_odds_gap_anchor": 0.25,
        "worst_absolute_log_disparity_anchor": 2,
    }
    path = tmp_path / "retained.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = load_config(path)
    assert dataclasses.asdict(cfg.evaluation.privacy_policy) == raw["evaluation"]["privacy_policy"]
    assert dataclasses.asdict(cfg.evaluation.scoring_policy) == raw["evaluation"]["scoring_policy"]


@pytest.fixture
def tmp_path():
    """Create an owned per-test scratch directory under the repository tmp/ root."""
    scratch_root = Path(__file__).parents[2] / "tmp"
    scratch_root.mkdir(exist_ok=True)
    scratch_path = scratch_root / f"test-config-{uuid4().hex}"
    scratch_path.mkdir()
    try:
        yield scratch_path
    finally:
        resolved_root = scratch_root.resolve()
        resolved_path = scratch_path.resolve()
        if resolved_path.parent != resolved_root or not resolved_path.name.startswith(
            "test-config-"
        ):
            raise RuntimeError(f"Refusing to remove unowned config test scratch: {resolved_path}")
        shutil.rmtree(resolved_path)


class TestFromDict:
    def test_none_returns_defaults(self):
        cfg = _from_dict(Config, None)
        assert cfg == Config()

    def test_empty_dict_returns_defaults(self):
        cfg = _from_dict(Config, {})
        assert cfg == Config()

    def test_default_hpo_objective_is_tstr_only(self):
        cfg = _from_dict(Config, {})

        assert cfg.generation.hpo.metric_config == {"canonical_objectives": ["tstr_macro_f1.v1"]}
        metrics = [
            metric for values in cfg.generation.hpo.metric_config.values() for metric in values
        ]
        assert metrics == ["tstr_macro_f1.v1"]
        assert cfg.generation.hpo.utility_policy == {
            "metrics": ["tstr_macro_f1.v1"],
            "weights": [1.0],
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

    def test_synthcity_params_parse_with_empty_default(self):
        cfg = _from_dict(Config, {})
        assert cfg.generation.synthcity.params == {}

        cfg = _from_dict(
            Config,
            {"generation": {"synthcity": {"params": {"ctgan": {"n_iter": 50}}}}},
        )
        assert cfg.generation.synthcity.params == {"ctgan": {"n_iter": 50}}

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

    def test_refidiff_settings_build_from_test_owned_data(self):
        cfg = _from_dict(
            Config,
            {
                "imputation": {
                    "method": "refidiff",
                    "refidiff": {
                        "denoiser": "mamba",
                        "catboost_warmup_iterations": 1000,
                    },
                    "benchmark": {"enabled": True},
                }
            },
        )
        assert cfg.imputation.method == "refidiff"
        assert cfg.imputation.refidiff.denoiser == "mamba"
        assert cfg.imputation.refidiff.catboost_warmup_iterations == 1000
        assert cfg.imputation.benchmark.enabled

    def test_hpo_metric_policy_uses_test_owned_settings(self):
        cfg = _from_dict(
            Config,
            {
                "generation": {
                    "hpo": {
                        "metric_config": {"canonical_objectives": ["tstr_macro_f1.v1"]},
                    }
                }
            },
        )
        metric_config = cfg.generation.hpo.metric_config
        metrics = [metric for values in metric_config.values() for metric in values]

        assert metric_config == {"canonical_objectives": ["tstr_macro_f1.v1"]}
        assert metrics == ["tstr_macro_f1.v1"]
        assert all(metric.endswith(".v1") for metric in metrics)
        assert cfg.generation.hpo.utility_policy == {
            "metrics": metrics,
            "weights": [1.0],
        }

    def test_hpo_scoring_policy_is_derived_from_canonical_objectives(self):
        cfg = _from_dict(
            Config,
            {
                "generation": {
                    "hpo": {"metric_config": {"canonical_objectives": ["tstr_macro_f1.v1"]}}
                }
            },
        )

        assert cfg.generation.hpo.utility_policy == {
            "metrics": ["tstr_macro_f1.v1"],
            "weights": [1.0],
        }

    def test_noncanonical_hpo_metric_categories_are_rejected(self):
        cfg = _from_dict(
            Config,
            {
                "data": {"source": "csv", "path": "x.csv"},
                "generation": {"hpo": {"metric_config": {"stats": ["wasserstein_dist"]}}},
            },
        )
        with pytest.raises(ValueError, match="only canonical_objectives"):
            _validate(cfg)

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

    def test_stratification_declarations_allow_null_bins(self):
        cfg = self._base_valid()
        cfg.data.stratification_variables = ["target", "sex"]
        cfg.data.stratification_bins = [None, ["female", "male"]]

        _validate(cfg)

    def test_protected_attribute_bins_use_aligned_age_labels(self):
        cfg = self._base_valid()
        cfg.data.protected_columns = ["sex", "Age", "ethnicity"]
        cfg.data.protected_attribute_bins = [
            None,
            ["<18", "18-30", "30-45", "45-60", "60+"],
            None,
        ]

        _validate(cfg)

    def test_greater_than_age_label_is_alias_for_inclusive_upper_open_bin(self):
        cfg = self._base_valid()
        cfg.data.protected_columns = ["Age"]
        cfg.data.protected_attribute_bins = [["<18", "18-30", "30-45", "45-60", ">60"]]
        cfg.data.stratification_variables = ["Age"]
        cfg.data.stratification_bins = [["<18", "18-30", "30-45", "45-60", ">60"]]

        _validate(cfg)

        parsed = _protected_attribute_bin_intervals(
            cfg.data.protected_columns, cfg.data.protected_attribute_bins
        )
        assert parsed["Age"]["intervals"][-1] == {
            "label": ">60",
            "lower": 60,
            "upper": None,
        }

    def test_misaligned_protected_attribute_bins_raise(self):
        cfg = self._base_valid()
        cfg.data.protected_columns = ["sex", "Age"]
        cfg.data.protected_attribute_bins = [None]

        with pytest.raises(ValueError, match="aligned with data.protected_columns"):
            _validate(cfg)

    def test_shared_protected_stratification_labels_must_match(self):
        cfg = self._base_valid()
        cfg.data.protected_columns = ["Age"]
        cfg.data.protected_attribute_bins = [["<18", "18-30", "30+"]]
        cfg.data.stratification_variables = ["Age"]
        cfg.data.stratification_bins = [None]

        with pytest.raises(ValueError, match="must match data.protected_attribute_bins"):
            _validate(cfg)

    def test_misaligned_stratification_bins_raise(self):
        cfg = self._base_valid()
        cfg.data.stratification_variables = ["target", "age"]
        cfg.data.stratification_bins = [None]

        with pytest.raises(ValueError, match="aligned"):
            _validate(cfg)

    @pytest.mark.parametrize(
        "labels",
        [
            ["<18", "18-18", "18+"],
            ["<18", "19-30", "30+"],
            ["<18", "30-45", "18-30", "45+"],
            ["<18", "18-30", "18-30", "30+"],
            ["under 18", "18+"],
        ],
    )
    def test_invalid_protected_attribute_intervals_raise(self, labels):
        cfg = self._base_valid()
        cfg.data.protected_columns = ["Age"]
        cfg.data.protected_attribute_bins = [labels]

        with pytest.raises(ValueError, match="data.protected_attribute_bins"):
            _validate(cfg)

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

    @pytest.mark.parametrize("fraction", [-0.01, 1.5, True, "0.05", None])
    def test_invalid_holdout_unknown_row_fraction_raises(self, fraction):
        cfg = self._base_valid()
        cfg.evaluation.syntheval_execution.max_holdout_unknown_row_fraction = fraction
        with pytest.raises(ValueError, match="max_holdout_unknown_row_fraction"):
            _validate(cfg)

    @pytest.mark.parametrize("fraction", [0, 0.05, 1])
    def test_valid_holdout_unknown_row_fraction_passes(self, fraction):
        cfg = self._base_valid()
        cfg.evaluation.syntheval_execution.max_holdout_unknown_row_fraction = fraction
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
        cfg.data.nominal_columns = cast(list[str] | None, "auto")
        with pytest.raises(ValueError, match="no longer supported"):
            _validate(cfg)

    @pytest.mark.parametrize("removed_key", ["binary_target", "release_generalization"])
    def test_removed_evaluation_config_is_rejected(self, tmp_path, removed_key):
        path = tmp_path / "removed-evaluation-config.yaml"
        path.write_text(
            f"data:\n  source: csv\n  path: data.csv\nevaluation:\n  {removed_key}: {{}}\n"
        )

        with pytest.raises(ValueError, match=f"evaluation\\.{removed_key}"):
            load_config(path)

    def test_release_generalization_is_not_a_public_config_field(self):
        evaluation_fields = {field.name for field in dataclasses.fields(EvaluationConfig)}
        assert "release_generalization" not in evaluation_fields

        with pytest.raises(TypeError, match="unexpected keyword argument 'release_generalization'"):
            EvaluationConfig(release_generalization={"columns": {}})
        with pytest.raises(ValueError, match="Unknown config key 'release_generalization'"):
            _from_dict(Config, {"evaluation": {"release_generalization": {"columns": {}}}})

    def test_binary_target_is_not_a_supported_python_config_field(self):
        evaluation_fields = {field.name for field in dataclasses.fields(EvaluationConfig)}
        assert "binary_target" not in evaluation_fields

        with pytest.raises(TypeError, match="unexpected keyword argument 'binary_target'"):
            EvaluationConfig(binary_target={"enabled": True})
        with pytest.raises(ValueError, match="Unknown config key 'binary_target'"):
            _from_dict(Config, {"evaluation": {"binary_target": {"enabled": True}}})
        evaluation = Config().evaluation
        with pytest.raises(AttributeError, match="binary_target"):
            _ = evaluation.binary_target
        with pytest.raises(AttributeError, match="binary_target"):
            evaluation.binary_target = {"enabled": True}

    @pytest.mark.parametrize(
        "weights",
        [
            {"utility": 1.0, "privacy": 1.0, "fairness": 1.0},
            {"utility": 1.0, "privacy": 1.0},
            {"utility": 1.0, "privacy": 1.0, "fairness": 1.0, "bogus": 1.0},
            {"utility": 1.0, "privacy": -0.5, "fairness": 1.0},
            {"utility": 1.0, "privacy": 0.0, "fairness": 1.0},
            {},
            None,
        ],
    )
    def test_removed_rank_weights_are_rejected(self, weights):
        with pytest.raises(ValueError, match=r"evaluation\.rank_weights.*remove"):
            _from_dict(Config, {"evaluation": {"rank_weights": weights}})

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
        """Build canonical config with all policy metadata populated."""
        data = DataConfig(
            source="csv",
            path="x.csv",
            target_column="target",
            canonical=True,
            patient_id_column="patient_id",
            protected_columns=["Age"],
            split=DataSplitConfig(mode="patient_group"),
        )
        cfg = Config(data=data)
        cfg.imputation.method = "hyperimpute"
        cfg.data.protected_attribute_bins = [["<18", "18+"]]
        cfg.generation.hpo.metric_config = {"canonical_objectives": ["tstr_macro_f1.v1"]}
        return cfg

    @classmethod
    def _canonical_yaml_data(cls) -> dict:
        """Return YAML-compatible data derived only from the canonical test fixture."""
        raw = dataclasses.asdict(cls._canonical_fixture())
        raw.pop("config_path")
        raw["data"]["split"].pop("patient_id_column")
        return raw

    def test_canonical_policy_settings_are_populated(self):
        cfg = self._canonical_fixture()
        assert cfg.data.patient_id_column == "patient_id"
        assert cfg.evaluation.privacy_policy.role_population_floor == 20
        assert cfg.evaluation.privacy_policy.protected_slice_floor == 1
        assert cfg.data.split is not None
        assert cfg.data.split.patient_id_column is None

    def test_protected_evaluation_and_generation_settings_align(self):
        shared_data = {
            "version": "2.0-protected-smoke",
            "quasi_identifier_columns": ["age", "sex", "region"],
            "protected_columns": ["risk_group", "orientation"],
        }
        evaluation = _from_dict(
            Config,
            {
                "data": shared_data,
                "generation": {
                    "n_samples": 100,
                    "synthcity": {"names": ["ctgan"], "params": {"ctgan": {"n_iter": 40}}},
                },
            },
        )
        generation = _from_dict(
            Config,
            {
                "data": shared_data,
                "generation": {
                    "n_samples": 100,
                    "force_retrain": True,
                    "synthcity": {"names": ["ctgan"], "params": {"ctgan": {"n_iter": 40}}},
                },
            },
        )

        assert evaluation.experiment.id is None
        assert evaluation.data.version == generation.data.version == "2.0-protected-smoke"
        assert evaluation.data.quasi_identifier_columns == ["age", "sex", "region"]
        assert evaluation.data.protected_columns == ["risk_group", "orientation"]
        assert evaluation.data.protected_columns == generation.data.protected_columns
        assert evaluation.data.quasi_identifier_columns == generation.data.quasi_identifier_columns
        assert (
            evaluation.generation.synthcity.names
            == generation.generation.synthcity.names
            == ["ctgan"]
        )
        assert (
            evaluation.generation.synthcity.params["ctgan"]["n_iter"]
            == generation.generation.synthcity.params["ctgan"]["n_iter"]
            == 40
        )
        assert evaluation.generation.n_samples == generation.generation.n_samples == 100
        assert evaluation.generation.force_retrain is False
        assert generation.generation.force_retrain is True

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
        cfg.data.protected_attribute_bins = [["<18", "18+"]]
        _validate(cfg)

    def test_canonical_protected_sensitive_overlap_is_allowed(self):
        cfg = self._canonical_fixture()
        cfg.data.protected_columns = ["age"]
        cfg.data.protected_attribute_bins = [["<18", "18+"]]
        cfg.data.sensitive_columns = ["age"]
        _validate(cfg)

    def test_canonical_rejects_malformed_release_intervals(self):
        cfg = self._canonical_fixture()
        cfg.data.protected_attribute_bins = [["<18", "30-20", "20+"]]
        with pytest.raises(ValueError, match="lower >= upper"):
            _validate(cfg)

    def test_protected_attribute_bins_accept_lower_inclusive_upper_exclusive_intervals(self):
        cfg = self._canonical_fixture()
        cfg.data.protected_attribute_bins = [["<18", "18-30", "30+"]]

        _validate(cfg)

    @pytest.mark.parametrize(
        "labels",
        [
            ["<18", "19+"],
            ["<18", "18-30", "29-60", "60+"],
            ["18-30", "30+"],
        ],
    )
    def test_canonical_rejects_misordered_open_ended_release_intervals(self, labels):
        cfg = self._canonical_fixture()
        cfg.data.protected_attribute_bins = [labels]

        with pytest.raises(ValueError, match="open|contiguous|ordered"):
            _validate(cfg)

    def test_canonical_invalid_support_setting_is_rejected(self):
        cfg = self._canonical_fixture()
        cfg.evaluation.privacy_policy.role_population_floor = 0
        with pytest.raises(ValueError, match="role_population_floor"):
            _validate(cfg)

    def test_canonical_missing_anchor_is_rejected(self):
        cfg = self._canonical_fixture()
        cfg.evaluation.scoring_policy.equalized_odds_gap_anchor = cast(float, None)
        with pytest.raises(ValueError, match="equalized_odds_gap_anchor"):
            _validate(cfg)

    def test_legacy_profile_cannot_declare_canonical_patient_id(self):
        cfg = Config(data=DataConfig(source="csv", path="x.csv", patient_id_column="subject_id"))
        with pytest.raises(ValueError, match="canonical"):
            _validate(cfg)

    def test_legacy_settings_are_noncanonical_by_default(self):
        cfg = _from_dict(Config, {"data": {"source": "csv", "path": "x.csv"}})

        assert cfg.data.canonical is False

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
        cfg = self._canonical_fixture()
        mutator(cfg)
        with pytest.raises(ValueError, match=message):
            _validate(cfg)

    def test_canonical_nested_identity_is_rejected_at_load(self, tmp_path):
        raw = self._canonical_yaml_data()
        raw["data"]["split"]["patient_id_column"] = "patient_id"
        path = tmp_path / "nested.yaml"
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError, match="nested split"):
            load_config(path)

    def test_canonical_nested_identity_is_rejected_directly(self):
        cfg = self._canonical_fixture()
        assert cfg.data.split is not None
        cfg.data.split.patient_id_column = "patient_id"
        with pytest.raises(ValueError, match="nested split identity"):
            _validate(cfg)

    @pytest.mark.parametrize(
        "path_parts",
        [
            ("evaluation", "privacy_policy"),
            ("evaluation", "scoring_policy"),
            ("data", "protected_attribute_bins"),
            ("generation", "hpo", "metric_config", "canonical_objectives"),
        ],
    )
    def test_canonical_omitted_required_policy_fields_fail_closed(self, tmp_path, path_parts):
        raw = self._canonical_yaml_data()
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
            ("privacy_policy", "role_population_floor"),
            ("privacy_policy", "protected_slice_floor"),
            ("scoring_policy", "equalized_odds_gap_anchor"),
            ("scoring_policy", "worst_absolute_log_disparity_anchor"),
        ],
    )
    def test_canonical_omitted_policy_value_fails_closed(self, tmp_path, block, field):
        raw = self._canonical_yaml_data()
        del raw["evaluation"][block][field]
        path = tmp_path / f"missing-{block}-{field}.yaml"
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError, match=field):
            load_config(path)

    def test_legacy_settings_marked_canonical_fail_actionably(self, tmp_path):
        raw = self._canonical_yaml_data()
        raw["imputation"]["method"] = "refidiff"
        raw["data"]["canonical"] = True
        path = tmp_path / "legacy-settings.yaml"
        path.write_text(yaml.safe_dump(raw))

        with pytest.raises(ValueError, match="Canonical profiles require imputation.method"):
            load_config(path)

    def test_canonical_omitted_policy_block_fails_closed(self, tmp_path):
        raw = self._canonical_yaml_data()
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

    def test_loris_syntheval_execution_uses_cpu_and_live_memory_bounds(self, monkeypatch):
        config_path = Path(__file__).parents[2] / "configs" / "config_loris.yaml"
        execution = load_config(config_path).evaluation.syntheval_execution

        assert execution.model_workers == "auto"
        assert execution.max_model_workers == 6
        assert execution.cores_per_model == 4
        assert execution.memory_reserve_gib == 16
        assert execution.memory_per_model_gib == 14

        monkeypatch.setattr("synthdata.evaluation.syntheval_eval.os.cpu_count", lambda: 24)
        monkeypatch.setattr(
            "synthdata.evaluation.syntheval_eval._available_memory_gib", lambda: 100.0
        )
        assert resolve_model_workers(execution, n_models=20, n_columns=664) == 6

        # Live available memory below the six-worker budget lowers concurrency.
        monkeypatch.setattr(
            "synthdata.evaluation.syntheval_eval._available_memory_gib", lambda: 72.0
        )
        assert resolve_model_workers(execution, n_models=20, n_columns=664) == 4

        # CPU availability independently limits workers even with ample memory.
        monkeypatch.setattr("synthdata.evaluation.syntheval_eval.os.cpu_count", lambda: 16)
        monkeypatch.setattr(
            "synthdata.evaluation.syntheval_eval._available_memory_gib", lambda: 200.0
        )
        assert resolve_model_workers(execution, n_models=20, n_columns=664) == 4

    def test_shipped_profiles_declare_stratification_and_tstr_only_hpo(self):
        root = Path(__file__).parents[2]
        loris = load_config(root / "configs" / "config_loris.yaml")
        hepatitis = load_config(root / "configs" / "config_hepatitis.yaml")

        assert loris.data.stratification_variables == ["CGAS_class", "Sex", "Age"]
        assert loris.data.stratification_bins == [
            None,
            None,
            ["<18", "18-30", "30-45", "45-60", ">60"],
        ]
        assert loris.data.protected_attribute_bins == [
            None,
            ["<18", "18-30", "30-45", "45-60", ">60"],
            None,
        ]
        runtime_bins = _protected_attribute_bin_intervals(
            loris.data.protected_columns,
            loris.data.protected_attribute_bins,
        )
        age_intervals = runtime_bins["Age"]["intervals"]
        assert [(interval["lower"], interval["upper"]) for interval in age_intervals] == [
            (None, 18),
            (18, 30),
            (30, 45),
            (45, 60),
            (60, None),
        ]
        assert [interval["label"] for interval in age_intervals] == [
            "<18",
            "18-30",
            "30-45",
            "45-60",
            ">60",
        ]

        age_values = [17, 18, 29, 30, 44, 45, 59, 60]
        age_frame = pd.DataFrame({"Age": age_values})
        stratified, _policy = _configured_stratification_frame(
            age_frame,
            ["Age"],
            [loris.data.stratification_bins[2]],
            runtime_bins,
        )
        assert stratified is not None
        released_synthetic, released_roles, _metadata = transform_release_roles(
            age_frame,
            {"train": age_frame},
            runtime_bins,
        )
        assert released_synthetic["Age"].tolist() == stratified["Age"].tolist()
        assert released_roles["train"]["Age"].tolist() == stratified["Age"].tolist()

        assert hepatitis.data.stratification_variables == ["target"]
        assert hepatitis.data.stratification_bins == [None]
        for cfg in (loris, hepatitis):
            assert cfg.generation.hpo.metric_config == {
                "canonical_objectives": ["tstr_macro_f1.v1"]
            }
            assert cfg.generation.hpo.utility_policy == {
                "metrics": ["tstr_macro_f1.v1"],
                "weights": [1.0],
            }

    def test_null_drop_columns_loads_as_empty_list(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(
            "name: mydata\ndata:\n  source: csv\n  path: raw.csv\n  drop_columns: null\n"
        )

        cfg = load_config(yaml_path)

        assert cfg.data.drop_columns == []

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
