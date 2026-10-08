"""Harness for end-to-end pipeline tests on the committed clinic fixture.

Each pipeline run renders ``fixtures/clinic_config.yaml`` into its own scratch
root and runs the real CLI entry points (``python -m scripts.run_*``) as
subprocesses, exactly as a user would. Subprocesses keep each stage's global
state (seeds, logging, torch threads) isolated and exercise the real exit
codes. Session-scoped fixtures share each run across many assertions, and the
independent runs execute concurrently (see ``RUNS``), so the expensive
training happens once per pytest session and in parallel.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest
import yaml

from synthdata.config import Config, load_config
from synthdata.data import load_dataset

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG_TEMPLATE = FIXTURES / "clinic_config.yaml"
FIXTURE_CSV = FIXTURES / "clinic.csv"
STAGE_TIMEOUT_SECONDS = 1800
#: The runs execute side by side (see ``RUNS``), so each stage process gets one
#: math thread. Left at the default, every torch/BLAS/OpenMP pool claims every
#: core and the oversubscribed runner crawls. A value already set wins.
THREAD_LIMITS = {
    var: os.environ.get(var, "1")
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT")
}


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
    text = text.replace("{root}", root.as_posix()).replace("{fixtures}", FIXTURES.as_posix())
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
        env = {**os.environ, **THREAD_LIMITS, "MPLBACKEND": "Agg"}
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


#: Feature shifts applied to held-out rows only by the leakage canaries.
HOLDOUT_SHIFTS = {"BMI": 15.0, "LAB_A": 3.0, "LAB_B": 40.0}

#: BMI and LAB_A (about 10% missing) treated as mostly missing, to exercise the
#: CART fill. Median/mode imputation here, so both imputers run in the suite,
#: and ARF with its search, so the leakage canary covers a second generator.
INDICATOR_ONLY = {
    "imputation": {"method": "simple", "missing_indicators": {"indicator_only_fraction": 0.08}},
    "generation": {"synthcity": {"names": ["arf"]}},
}

IMPUTATION = ("imputation",)
IMPUTE_AND_GENERATE = ("imputation", "generation")
FULL = ("imputation", "generation", "evaluation")


@dataclasses.dataclass(frozen=True)
class RunSpec:
    """One independent pipeline run the suite needs."""

    stages: tuple = FULL
    overrides: dict | None = None
    #: Shift held-out feature values before running (a leakage canary).
    perturb_holdout: bool = False
    #: Pass ``--plot`` to each stage and finish with ``synthdata-plot``.
    plots: bool = False


#: Every pipeline run in the non-slow suite. They don't depend on each other,
#: so they start together on a thread pool (each run is a chain of subprocess
#: stages) and the suite takes about as long as its slowest run, not their
#: sum. Each test asks for its run by fixture name (``<name>_run``).
RUNS: dict[str, RunSpec] = {
    "baseline": RunSpec(plots=True),
    "rerun": RunSpec(),
    "holdout_canary": RunSpec(IMPUTE_AND_GENERATE, perturb_holdout=True),
    "indicator_only": RunSpec(IMPUTE_AND_GENERATE, INDICATOR_ONLY),
    "cart_canary": RunSpec(IMPUTE_AND_GENERATE, INDICATOR_ONLY, perturb_holdout=True),
    # Imputed splits only, for in-process evaluation of planted datasets.
    "imputed": RunSpec(IMPUTATION),
}


def _perturbed_source(tmp_path_factory, name: str, overrides: dict | None) -> Path:
    """A copy of the fixture where only the held-out rows' features changed.

    Feature values (never the target) are shifted, so the stratified patient
    split is unchanged and only held-out information differs.
    """
    probe = new_run(tmp_path_factory, f"{name}_split", overrides=overrides)
    test_rows = load_dataset(probe.cfg).test_df.index
    source = pd.read_csv(FIXTURE_CSV)
    for column, shift in HOLDOUT_SHIFTS.items():
        source.loc[test_rows, column] = source.loc[test_rows, column] + shift
    data_path = probe.root / "clinic_perturbed.csv"
    source.to_csv(data_path, index=False)
    return data_path


def _start(tmp_path_factory, name: str, spec: RunSpec) -> PipelineRun:
    data_path = (
        _perturbed_source(tmp_path_factory, name, spec.overrides) if spec.perturb_holdout else None
    )
    return new_run(tmp_path_factory, name, overrides=spec.overrides, data_path=data_path)


def _execute(run: PipelineRun, spec: RunSpec) -> PipelineRun:
    flags = ("--plot",) if spec.plots else ()
    for stage in spec.stages:
        run.run(stage, *flags)
    if spec.plots:
        run.run("plots")
    return run


@pytest.fixture(scope="session")
def pipeline_runs(request, tmp_path_factory):
    """Start every run some selected test needs, in parallel; return a getter.

    Only runs reachable from the collected tests' fixtures are started, so
    ``-k`` selections stay cheap.
    """
    pytest.importorskip("catboost", reason="RefiDiff imputation needs catboost")
    # In-process evaluations (known answers) share the cores with the runs too.
    import torch
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(1)
    limits = threadpool_limits(1)
    needed = {
        name
        for item in request.session.items
        for name in RUNS
        if f"{name}_run" in getattr(item, "fixturenames", ())
    }
    if any("pipeline_run" in getattr(item, "fixturenames", ()) for item in request.session.items):
        needed.add("baseline")
    pool = ThreadPoolExecutor(max_workers=max(1, len(needed)))
    futures = {
        name: pool.submit(_execute, _start(tmp_path_factory, name, RUNS[name]), RUNS[name])
        for name in sorted(needed)
    }

    def get(name: str) -> PipelineRun:
        if name not in futures:
            futures[name] = pool.submit(
                _execute, _start(tmp_path_factory, name, RUNS[name]), RUNS[name]
            )
        return futures[name].result()

    yield get
    pool.shutdown(wait=True, cancel_futures=True)
    limits.restore_original_limits()


@pytest.fixture(scope="session")
def pipeline_run(pipeline_runs) -> PipelineRun:
    """One complete impute → generate → evaluate → plot run on the fixture."""
    return pipeline_runs("baseline")


@pytest.fixture(scope="session")
def rerun_run(pipeline_runs) -> PipelineRun:
    """An independent second run of the identical config in a fresh root."""
    return pipeline_runs("rerun")


@pytest.fixture(scope="session")
def holdout_canary_run(pipeline_runs) -> PipelineRun:
    return pipeline_runs("holdout_canary")


@pytest.fixture(scope="session")
def indicator_only_run(pipeline_runs) -> PipelineRun:
    return pipeline_runs("indicator_only")


@pytest.fixture(scope="session")
def cart_canary_run(pipeline_runs) -> PipelineRun:
    return pipeline_runs("cart_canary")


@pytest.fixture(scope="session")
def imputed_run(pipeline_runs) -> PipelineRun:
    return pipeline_runs("imputed")
