"""Tests for direct config auditing without running pipeline stages."""

import pytest
import yaml

from synthdata.config_audit import audit_config
from synthdata.evaluation.syntheval_eval import InsufficientSynthEvalMemoryError

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("machine_resources")]


def _write_valid_audit_files(tmp_path):
    source_path = tmp_path / "input.csv"
    source_path.write_text(
        "age,outcome\n" + "".join(f"{age},{age % 2}\n" for age in range(30)),
        encoding="utf-8",
    )
    schema_path = tmp_path / "schema.csv"
    schema_path.write_text(
        "column,kind\nage,continuous\noutcome,categorical\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "custom-config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "custom-audit-data",
                "data": {
                    "source": "csv",
                    "path": str(source_path),
                    "data_dir": str(tmp_path / "dataset-cache"),
                    "target_column": "outcome",
                    "variable_schema_path": str(schema_path),
                    "split": {"mode": "row"},
                },
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_audit_config_loads_supplied_config_and_schema(tmp_path):
    config_path = _write_valid_audit_files(tmp_path)

    result = audit_config(config_path)

    assert result.config_path == config_path.resolve()
    assert result.dataset_name == "custom-audit-data"
    assert result.row_count == 30
    assert result.feature_columns == ("age",)
    assert result.target_column == "outcome"
    assert result.schema_columns == ("age", "outcome")
    # Default budget on the faked 16-CPU, 200 GiB machine: 16 // 4 cores.
    assert result.syntheval_model_workers == 4


def test_audit_config_rejects_resource_budget_that_cannot_fit_one_model(
    tmp_path, machine_resources
):
    config_path = _write_valid_audit_files(tmp_path)
    machine_resources(available_gib=8.0)

    with pytest.raises(InsufficientSynthEvalMemoryError, match="cannot fit one model"):
        audit_config(config_path)


def test_audit_config_skips_resource_check_when_syntheval_is_disabled(tmp_path, machine_resources):
    config_path = _write_valid_audit_files(tmp_path)
    raw = yaml.safe_load(config_path.read_text())
    raw["evaluation"] = {"syntheval": {"enabled": False}}
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    machine_resources(available_gib=8.0)

    assert audit_config(config_path).syntheval_model_workers is None


def test_audit_config_applies_canonical_policy_checks_to_supplied_config(tmp_path):
    config_path = tmp_path / "incomplete-canonical.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "source": "csv",
                    "path": str(tmp_path / "unused.csv"),
                    "canonical": True,
                    "patient_id_column": "patient_id",
                    "split": {"mode": "patient_group"},
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Canonical policy is incomplete"):
        audit_config(config_path)
