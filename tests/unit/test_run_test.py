"""CLI behavior tests for synthdata-test."""

import pytest
import yaml

from scripts import run_test

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
    return config_path, schema_path


def _run_cli(monkeypatch, config_path):
    monkeypatch.setattr("sys.argv", ["synthdata-test", "--config", str(config_path)])
    return run_test.main()


def test_cli_returns_zero_and_reports_valid_config(tmp_path, monkeypatch, capsys):
    config_path, _ = _write_valid_audit_files(tmp_path)

    status = _run_cli(monkeypatch, config_path)

    captured = capsys.readouterr()
    assert status == 0
    assert "audit passed" in captured.out
    assert "dataset=custom-audit-data" in captured.out
    assert "syntheval_workers=4" in captured.out
    assert captured.err == ""


def test_cli_reports_config_validation_failure_and_nonzero_exit(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "invalid-config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "device": "invalid-device",
                "data": {"source": "csv", "path": str(tmp_path / "unused.csv")},
            }
        ),
        encoding="utf-8",
    )

    status = _run_cli(monkeypatch, config_path)

    captured = capsys.readouterr()
    assert status == 1
    assert "audit failed" in captured.err
    assert "device must be one of" in captured.err
    assert captured.out == ""


def test_cli_reports_semantic_policy_failure_and_nonzero_exit(tmp_path, monkeypatch, capsys):
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

    status = _run_cli(monkeypatch, config_path)

    captured = capsys.readouterr()
    assert status == 1
    assert "Canonical policy is incomplete" in captured.err
    assert "evaluation.privacy_policy" in captured.err


def test_cli_reports_data_schema_failure_and_nonzero_exit(tmp_path, monkeypatch, capsys):
    config_path, schema_path = _write_valid_audit_files(tmp_path)
    schema_path.write_text(
        "column,kind\nage,continuous\noutcome,categorical\nstale,continuous\n",
        encoding="utf-8",
    )

    status = _run_cli(monkeypatch, config_path)

    captured = capsys.readouterr()
    assert status == 1
    assert "Variable schema" in captured.err
    assert "stale/non-modeling declaration(s): ['stale']" in captured.err
    assert captured.out == ""
