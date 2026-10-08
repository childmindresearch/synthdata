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

pytestmark = pytest.mark.integration


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _splits(run):
    return load_imputed_splits(load_dataset(run.cfg))


def test_same_seed_gives_the_same_split_and_imputation(pipeline_run, rerun_run):
    first, second = _splits(pipeline_run), _splits(rerun_run)
    assert first.train_df.index.equals(second.train_df.index)
    assert first.test_df.index.equals(second.test_df.index)
    pd.testing.assert_frame_equal(first.full_imputed_df, second.full_imputed_df)


def test_same_seed_gives_identical_synthetic_data(pipeline_run, rerun_run):
    first, second = pipeline_run.synthetic(), rerun_run.synthetic()
    assert sorted(first) == sorted(second)
    different = [
        name
        for name in first
        if _digest(pipeline_run.generation_dir / f"{name}.csv")
        != _digest(rerun_run.generation_dir / f"{name}.csv")
    ]
    assert not different, different


def test_same_inputs_give_the_same_metrics(pipeline_run, rerun_run):
    # Compared per model, for every model whose synthetic data came out
    # identical in both runs; whether generation itself is reproducible is
    # test_same_seed_gives_identical_synthetic_data's job. Rank columns are
    # left out: they scale each metric across all models, so they move when
    # any other model's data does.
    same_data = [
        path.stem
        for path in sorted(pipeline_run.generation_dir.glob("*.csv"))
        if _digest(path) == _digest(rerun_run.generation_dir / path.name)
    ]
    assert same_data, "no model generated identical data in both runs"
    first, second = pipeline_run.combined(), rerun_run.combined()
    columns = [c for c in first.columns if c[2] != "rank" and c[0] != "__all__"]
    first, second = first.loc[same_data, columns], second.loc[same_data, columns]
    mismatched = []
    for column in columns:
        a = pd.to_numeric(first[column], errors="coerce")
        b = pd.to_numeric(second[column], errors="coerce")
        if not np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True):
            mismatched.append(column)
    assert not mismatched, mismatched


def test_rerunning_generation_reuses_cached_models(rerun_run):
    before = {path.name: _digest(path) for path in rerun_run.generation_dir.glob("*.csv")}
    result = rerun_run.run("generation", "--experiment-id", rerun_run.experiment_id)
    after = {path.name: _digest(path) for path in rerun_run.generation_dir.glob("*.csv")}
    assert after == before
    log = result.stdout + result.stderr
    assert "generating synthetic data" not in log
    for name in before:
        assert f"[{name.removesuffix('.csv')}] using cached synthetic data" in log, name


def test_test_rows_do_not_influence_train_imputation(pipeline_run, holdout_canary_run):
    baseline, canary = _splits(pipeline_run), _splits(holdout_canary_run)
    assert baseline.test_df.index.equals(canary.test_df.index), "split must not change"
    train_rows = baseline.train_df.index
    pd.testing.assert_frame_equal(
        baseline.full_imputed_df.loc[train_rows],
        canary.full_imputed_df.loc[train_rows],
    )


def test_test_rows_do_not_influence_tuning_or_generation(pipeline_run, holdout_canary_run):
    baseline, canary = _splits(pipeline_run), _splits(holdout_canary_run)
    pd.testing.assert_frame_equal(baseline.tuning_imputed_df, canary.tuning_imputed_df)
    # Hyperparameter search and every final model only see train and tuning,
    # so the chosen parameters and all synthetic data must be unchanged.
    best = "hpo_best_params.json"
    assert (pipeline_run.generation_dir / best).read_text() == (
        holdout_canary_run.generation_dir / best
    ).read_text()
    names = sorted(path.name for path in pipeline_run.generation_dir.glob("*.csv"))
    assert names == sorted(path.name for path in holdout_canary_run.generation_dir.glob("*.csv"))
    different = [
        name
        for name in names
        if _digest(pipeline_run.generation_dir / name)
        != _digest(holdout_canary_run.generation_dir / name)
    ]
    assert not different, different
