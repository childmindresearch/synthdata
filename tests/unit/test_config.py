"""Unit tests for synthdata.config: dataclass composition, validation, YAML loading."""

import pytest
import yaml

from synthdata.config import (
    Config,
    DataConfig,
    GenerationConfig,
    HPOConfig,
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

    def test_unknown_top_level_key_raises(self):
        with pytest.raises(ValueError, match="Unknown config key"):
            _from_dict(Config, {"not_a_real_field": 1})

    def test_unknown_nested_key_raises(self):
        with pytest.raises(ValueError, match="Unknown config key"):
            _from_dict(Config, {"data": {"not_a_real_field": 1}})


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
        cfg.data.source = "parquet"
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

    def test_bad_tabpfn_data_variant_raises(self):
        cfg = self._base_valid()
        cfg.generation.tabpfn.data_variants = ["raw", "bogus"]
        with pytest.raises(ValueError, match="data_variants"):
            _validate(cfg)

    def test_valid_tabpfn_data_variants_pass(self):
        cfg = self._base_valid()
        cfg.generation.tabpfn.data_variants = ["raw", "imputed"]
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
