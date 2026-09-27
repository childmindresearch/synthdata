"""Behavioral tests for imputation validation-report CLI artifacts."""

import sys
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import run_imputation as imputation_cli
from synthdata.config import Config
from synthdata.experiment import imputation_output_dir

pytestmark = pytest.mark.unit
_UNSET = object()
VALIDATION_REPORT_COLUMNS = [
    "column",
    "datatype",
    "categorical",
    "n_missing",
    "obs_cardinality",
    "imp_cardinality",
    "obs_mean",
    "obs_std",
    "imp_mean",
    "imp_std",
    "obs_mode",
    "imp_mode",
    "n_imputed",
    "n_valid",
    "all_valid",
]


def _run_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cfg: Config,
    validation_df: pd.DataFrame | object = _UNSET,
) -> list[tuple[str, object]]:
    """Run CLI with all data/model work replaced by local test doubles."""
    monkeypatch.chdir(tmp_path)
    logged: list[tuple[str, object]] = []
    dataset = SimpleNamespace(data_dir=tmp_path / "cache", version=cfg.data.version)
    monkeypatch.setattr(imputation_cli, "load_config", lambda _: cfg)
    monkeypatch.setattr(imputation_cli, "load_dataset", lambda _: dataset)
    monkeypatch.setattr(imputation_cli, "run_imputation", lambda _cfg, value: value)
    if validation_df is not _UNSET:
        monkeypatch.setattr(
            imputation_cli,
            "build_validation_report",
            lambda _cfg, _dataset: validation_df,
        )
    monkeypatch.setattr(imputation_cli, "set_global_seed", lambda _seed: None)
    monkeypatch.setattr(
        imputation_cli.logger,
        "info",
        lambda message, *args: logged.append((message, args)),
    )
    monkeypatch.setattr(sys, "argv", ["synthdata-impute", "--config", "config.yaml"])

    imputation_cli.main()
    return logged


@pytest.mark.parametrize("version, scope", [("v7", "data_v_v7"), (None, "data_v_unversioned")])
def test_validation_report_uses_version_scoped_output(
    make_config, monkeypatch, tmp_path, version, scope
):
    cfg = make_config(data_version=version)
    validation_df = pd.DataFrame(
        {
            "column": ["age"],
            "datatype": ["continuous"],
            "categorical": [False],
            "n_missing": [2],
            "obs_cardinality": [3],
            "imp_cardinality": [4],
            "obs_mean": [10.5],
            "obs_std": [1.5],
            "imp_mean": [10.0],
            "imp_std": [1.2],
            "obs_mode": [float("nan")],
            "imp_mode": [float("nan")],
            "n_imputed": [2],
            "n_valid": [2],
            "all_valid": [True],
        }
    )

    _run_cli(monkeypatch, tmp_path, cfg, validation_df)

    report_path = (
        tmp_path / "output" / cfg.name / "imputation" / scope / "imputation_validation_report.csv"
    )
    assert report_path.is_file()


def test_validation_report_csv_preserves_columns_and_rows(make_config, monkeypatch, tmp_path):
    cfg = make_config(data_version="v1")
    validation_df = pd.DataFrame(
        {
            "column": ["age", "status"],
            "datatype": ["continuous", "categorical"],
            "categorical": [False, True],
            "n_missing": [2, 1],
            "obs_cardinality": [3, 2],
            "imp_cardinality": [4, 2],
            "obs_mean": [10.5, None],
            "obs_std": [1.5, None],
            "imp_mean": [10.0, None],
            "imp_std": [1.2, None],
            "obs_mode": [None, "active"],
            "imp_mode": [None, "active"],
            "n_imputed": [2, 1],
            "n_valid": [2, 1],
            "all_valid": [True, False],
        }
    )

    _run_cli(monkeypatch, tmp_path, cfg, validation_df)

    report_path = imputation_output_dir(cfg) / "imputation_validation_report.csv"
    persisted = pd.read_csv(report_path)
    csv_normalized = pd.read_csv(StringIO(validation_df.to_csv(index=False)))
    pd.testing.assert_frame_equal(persisted, csv_normalized, check_dtype=False)


def test_empty_validation_report_writes_expected_header(make_config, monkeypatch, tmp_path):
    cfg = make_config(data_version="v1")

    _run_cli(monkeypatch, tmp_path, cfg, pd.DataFrame())

    report_path = imputation_output_dir(cfg) / "imputation_validation_report.csv"
    assert report_path.read_text() == ",".join(VALIDATION_REPORT_COLUMNS) + "\n"


def test_logs_report_location_without_validation_rows(make_config, monkeypatch, tmp_path):
    cfg = make_config(data_version="v3")
    validation_df = pd.DataFrame(columns=VALIDATION_REPORT_COLUMNS)
    validation_df.loc[0, "column"] = "secret_row_value"
    validation_df.loc[0, "n_missing"] = 1

    logged = _run_cli(monkeypatch, tmp_path, cfg, validation_df)

    messages = [message % args if args else message for message, args in logged]
    report_path = str(imputation_output_dir(cfg) / "imputation_validation_report.csv")
    assert any(report_path in message for message in messages)
    assert all("secret_row_value" not in message for message in messages)


def test_disabled_imputation_does_not_create_validation_report(make_config, monkeypatch, tmp_path):
    cfg = make_config(data_version="v1")
    cfg.imputation.enabled = False
    build_report_called = False

    def fail_if_report_built(_cfg, _dataset):
        nonlocal build_report_called
        build_report_called = True
        raise AssertionError("disabled imputation must not build validation report")

    monkeypatch.setattr(imputation_cli, "build_validation_report", fail_if_report_built)
    _run_cli(monkeypatch, tmp_path, cfg)

    assert not build_report_called
    assert not (tmp_path / "output").exists()
