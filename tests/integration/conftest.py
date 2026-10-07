"""Harness for end-to-end pipeline tests on the committed clinic fixture.

Each pipeline run renders ``fixtures/clinic_config.yaml`` into its own scratch
root and runs the real CLI entry points (``python -m scripts.run_*``) as
subprocesses, exactly as a user would. Subprocesses keep each stage's global
state (seeds, logging, torch threads) isolated and exercise the real exit
codes. Session-scoped fixtures share one run across many assertions, so the
expensive training happens once per pytest session.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from synthdata.config import Config, load_config

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG_TEMPLATE = FIXTURES / "clinic_config.yaml"
FIXTURE_CSV = FIXTURES / "clinic.csv"
STAGE_TIMEOUT_SECONDS = 1800


def _deep_merge(base: dict, overrides: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def render_config(root: Path, overrides: dict | None = None, data_path: Path | None = None) -> Path:
    """Write a runnable config for ``root`` and return its path."""
    text = CONFIG_TEMPLATE.read_text()
    text = text.replace("{root}", str(root)).replace("{fixtures}", str(FIXTURES))
    raw = yaml.safe_load(text)
    if data_path is not None:
        raw["data"]["path"] = str(data_path)
    if overrides:
        raw = _deep_merge(raw, overrides)
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return config_path


@dataclasses.dataclass
class PipelineRun:
    """Paths and logs for one isolated pipeline run."""

    root: Path
    config_path: Path
    #: Empty working directory the CLIs run from; must stay empty.
    cwd: Path
    logs: dict[str, str] = dataclasses.field(default_factory=dict)

    @property
    def cfg(self) -> Config:
        return load_config(self.config_path)

    def run(self, stage: str, *args: str) -> subprocess.CompletedProcess:
        """Run ``scripts.run_<stage>`` and fail the test with its log on error."""
        env = {**os.environ, "MPLBACKEND": "Agg"}
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                f"scripts.run_{stage}",
                "--config",
                str(self.config_path),
                *args,
            ],
            cwd=self.cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=STAGE_TIMEOUT_SECONDS,
            check=False,
        )
        self.logs[stage] = result.stdout + result.stderr
        if result.returncode != 0:
            tail = "\n".join(self.logs[stage].splitlines()[-60:])
            pytest.fail(f"synthdata {stage} exited {result.returncode}:\n{tail}", pytrace=False)
        return result

    # -- artifact locations -------------------------------------------------
    @property
    def data_dir(self) -> Path:
        cfg = self.cfg
        return Path(cfg.data.data_dir) / f"data_v_{cfg.data.version}"

    @property
    def experiments_dir(self) -> Path:
        return Path(self.cfg.generation.output_dir).parent / "experiments" / self.data_dir.name

    @property
    def experiment_id(self) -> str:
        return json.loads((self.experiments_dir / "latest.json").read_text())["experiment_id"]

    def _scoped(self, output_dir: str) -> Path:
        return Path(output_dir) / self.data_dir.name / f"exp_v_{self.experiment_id}"

    @property
    def generation_dir(self) -> Path:
        return self._scoped(self.cfg.generation.output_dir)

    @property
    def evaluation_dir(self) -> Path:
        return self._scoped(self.cfg.evaluation.output_dir)

    @property
    def plots_dir(self) -> Path:
        return self._scoped(self.cfg.plots.output_dir)

    @property
    def experiment_manifest(self) -> dict:
        return json.loads(
            (self.experiments_dir / f"exp_v_{self.experiment_id}" / "manifest.json").read_text()
        )

    def synthetic(self) -> dict[str, pd.DataFrame]:
        return {path.stem: pd.read_csv(path) for path in sorted(self.generation_dir.glob("*.csv"))}

    def combined(self) -> pd.DataFrame:
        return pd.read_csv(
            self.evaluation_dir / "combined_evaluation.csv", header=[0, 1, 2], index_col=0
        )


def new_run(
    tmp_path_factory: pytest.TempPathFactory,
    name: str,
    *,
    overrides: dict | None = None,
    data_path: Path | None = None,
) -> PipelineRun:
    root = tmp_path_factory.mktemp(name)
    cwd = root / "cwd"
    cwd.mkdir()
    return PipelineRun(root, render_config(root, overrides, data_path), cwd)


def run_full_pipeline(run: PipelineRun) -> PipelineRun:
    run.run("imputation", "--plot")
    run.run("generation", "--plot")
    run.run("evaluation", "--plot")
    run.run("plots")
    return run


@pytest.fixture(scope="session")
def pipeline_run(tmp_path_factory) -> PipelineRun:
    """One complete impute → generate → evaluate → plot run on the fixture."""
    pytest.importorskip("catboost", reason="RefiDiff imputation needs catboost")
    return run_full_pipeline(new_run(tmp_path_factory, "baseline"))
