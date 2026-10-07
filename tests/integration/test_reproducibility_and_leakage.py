"""Same-seed reproducibility, cache reuse, and a holdout-leakage canary.

These compare two real pipeline runs against each other, so they need a
second run on top of the shared ``pipeline_run`` baseline.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from synthdata.data import load_dataset, load_imputed_splits

from .conftest import FIXTURE_CSV, new_run, run_full_pipeline

pytestmark = pytest.mark.integration


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def rerun(pipeline_run, tmp_path_factory):
    """An independent second run of the identical config in a fresh root."""
    return run_full_pipeline(new_run(tmp_path_factory, "rerun"))


def _splits(run):
    return load_imputed_splits(load_dataset(run.cfg))


def test_same_seed_gives_the_same_split_and_imputation(pipeline_run, rerun):
    first, second = _splits(pipeline_run), _splits(rerun)
    assert first.train_df.index.equals(second.train_df.index)
    assert first.test_df.index.equals(second.test_df.index)
    pd.testing.assert_frame_equal(first.full_imputed_df, second.full_imputed_df)


def test_same_seed_gives_identical_synthetic_data(pipeline_run, rerun):
    first, second = pipeline_run.synthetic(), rerun.synthetic()
    assert sorted(first) == sorted(second)
    different = [
        name
        for name in first
        if _digest(pipeline_run.generation_dir / f"{name}.csv")
        != _digest(rerun.generation_dir / f"{name}.csv")
    ]
    assert not different, different


@pytest.mark.xfail(
    strict=True,
    reason="SynthEval MIA, NNAA, attribute disclosure and the fairness metrics draw "
    "unseeded random samples and classifiers inside their worker processes, so identical "
    "inputs score differently (overall rank moved by ~0.37 between two identical runs).",
)
def test_same_inputs_give_the_same_metrics(pipeline_run, rerun):
    first, second = pipeline_run.combined(), rerun.combined()
    second = second.loc[first.index, first.columns]
    mismatched = []
    for column in first.columns:
        a = pd.to_numeric(first[column], errors="coerce")
        b = pd.to_numeric(second[column], errors="coerce")
        if a.isna().all():  # text column (e.g. privacy-gate reasons)
            if not first[column].equals(second[column]):
                mismatched.append(column)
        elif not np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True):
            mismatched.append(column)
    assert not mismatched, mismatched


def test_rerunning_generation_reuses_cached_models(rerun):
    before = {path.name: _digest(path) for path in rerun.generation_dir.glob("*.csv")}
    result = rerun.run("generation", "--experiment-id", rerun.experiment_id)
    after = {path.name: _digest(path) for path in rerun.generation_dir.glob("*.csv")}
    assert after == before
    log = result.stdout + result.stderr
    assert "generating synthetic data" not in log
    for name in before:
        assert f"[{name.removesuffix('.csv')}] using cached synthetic data" in log, name


@pytest.fixture(scope="module")
def perturbed_holdout_run(pipeline_run, tmp_path_factory):
    """Imputation and generation on a copy of the source where only test rows changed.

    Feature values (never the target) of test rows are shifted, so the
    stratified row split is unchanged and only held-out information differs.
    """
    test_rows = load_dataset(pipeline_run.cfg).test_df.index
    source = pd.read_csv(FIXTURE_CSV)
    for column, shift in {"BMI": 15.0, "LAB_A": 3.0, "LAB_B": 40.0}.items():
        source.loc[test_rows, column] = source.loc[test_rows, column] + shift
    root = tmp_path_factory.mktemp("holdout_canary")
    data_path = root / "clinic_perturbed.csv"
    source.to_csv(data_path, index=False)
    run = new_run(tmp_path_factory, "holdout_canary_run", data_path=data_path)
    run.run("imputation")
    run.run("generation")
    return run


def test_test_rows_do_not_influence_train_imputation(pipeline_run, perturbed_holdout_run):
    baseline, canary = _splits(pipeline_run), _splits(perturbed_holdout_run)
    assert baseline.test_df.index.equals(canary.test_df.index), "split must not change"
    train_rows = baseline.train_df.index
    pd.testing.assert_frame_equal(
        baseline.full_imputed_df.loc[train_rows],
        canary.full_imputed_df.loc[train_rows],
    )


def test_test_rows_do_not_influence_tuning_or_generation(pipeline_run, perturbed_holdout_run):
    baseline, canary = _splits(pipeline_run), _splits(perturbed_holdout_run)
    pd.testing.assert_frame_equal(baseline.tuning_imputed_df, canary.tuning_imputed_df)
    # Hyperparameter search and every final model only see train and tuning,
    # so the chosen parameters and all synthetic data must be unchanged.
    best = "hpo_best_params.json"
    assert (pipeline_run.generation_dir / best).read_text() == (
        perturbed_holdout_run.generation_dir / best
    ).read_text()
    names = sorted(path.name for path in pipeline_run.generation_dir.glob("*.csv"))
    assert names == sorted(path.name for path in perturbed_holdout_run.generation_dir.glob("*.csv"))
    different = [
        name
        for name in names
        if _digest(pipeline_run.generation_dir / name)
        != _digest(perturbed_holdout_run.generation_dir / name)
    ]
    assert not different, different
