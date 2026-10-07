"""Known-answer checks: metrics must rank planted "generators" correctly.

Two synthetic datasets with known properties go through the real evaluation
path (``synthdata.evaluation.run_evaluation``):

* ``copy`` -- rows copied verbatim from the real train split. Perfect fidelity
  and utility, and the worst possible privacy (every row is a real record).
* ``shuffle`` -- the same rows with every column permuted independently.
  Marginals are identical to ``copy``, but all joint structure, including
  the feature/target relationship, is destroyed and no real row survives.

A sign flip, a swapped train/test role or a misaligned column shows up as one
of these orderings failing, without needing exact reference values.
"""

from __future__ import annotations

import copy as copy_module

import numpy as np
import pandas as pd
import pytest

from synthdata.data import load_dataset, load_imputed_splits
from synthdata.evaluation import run_evaluation

pytestmark = pytest.mark.integration

N_ROWS = 120


@pytest.fixture(scope="module")
def known_answers(pipeline_run, tmp_path_factory) -> pd.DataFrame:
    cfg = copy_module.deepcopy(pipeline_run.cfg)
    cfg.evaluation.output_dir = str(tmp_path_factory.mktemp("known_answers"))
    cfg.evaluation.generate_report = False
    cfg.evaluation.save_per_model_syntheval_plots = False
    dataset = load_imputed_splits(load_dataset(cfg))

    real = dataset.train_imputed_df.reset_index(drop=True)
    copied = real.sample(n=N_ROWS, random_state=0).reset_index(drop=True)
    rng = np.random.default_rng(0)
    shuffled = pd.DataFrame(
        {column: rng.permutation(copied[column].to_numpy()) for column in copied.columns}
    )
    combined, _ = run_evaluation(
        cfg,
        dataset,
        {"copy": copied, "copy_twin": copied.copy(), "shuffle": shuffled},
    )
    return combined


@pytest.fixture(scope="module")
def train_size(pipeline_run) -> int:
    return len(load_dataset(pipeline_run.cfg).train_df)


def _metric(combined: pd.DataFrame, metric: str) -> pd.Series:
    matches = [c for c in combined.columns if c[2] == metric]
    assert len(matches) == 1, (metric, matches)
    return combined[matches[0]].astype(float)


def test_copied_rows_are_at_zero_distance_from_real_rows(known_answers):
    dcr = _metric(known_answers, "median_DCR")
    assert dcr["copy"] == pytest.approx(0.0, abs=1e-9)
    assert dcr["shuffle"] > 0


@pytest.mark.xfail(
    strict=True,
    reason="SynthCity evaluation passes the test split as X_gt, so its copy and "
    "nearest-neighbour checks compare synthetic rows against data the generator never "
    "saw; a generator that memorises train scores 0 copied rows.",
)
def test_synthcity_detects_rows_copied_from_train(known_answers):
    common = _metric(known_answers, "sanity.common_rows_proportion.score")
    assert common["copy"] >= 0.95
    assert common["shuffle"] <= 0.1


def test_hit_rate_counts_reproduced_real_records(known_answers, train_size):
    # Every copied row matches a real train row, so the share of real records
    # hit equals the share of train that was copied.
    hit_rate = _metric(known_answers, "hit_rate")
    assert hit_rate["copy"] == pytest.approx(N_ROWS / train_size, abs=1e-9)
    assert hit_rate["shuffle"] < hit_rate["copy"]


def test_syntheval_marginals_cannot_tell_shuffle_from_copy(known_answers):
    # Shuffling preserves every marginal exactly, so marginal fidelity must be
    # identical for both; it must not be what separates them.
    for metric in ("ks_tvd_stat", "avg_h_dist"):
        values = _metric(known_answers, metric)
        assert values["shuffle"] == pytest.approx(values["copy"], abs=1e-12), metric


@pytest.mark.xfail(
    strict=True,
    reason="SynthCity evaluation bootstrap-resamples each synthetic dataset with "
    "replacement (PregeneratedSyntheticModel.sample), so two datasets with identical "
    "marginals get different marginal scores.",
)
def test_synthcity_marginals_cannot_tell_shuffle_from_copy(known_answers):
    ks = _metric(known_answers, "stats.ks_test.marginal")
    assert ks["shuffle"] == pytest.approx(ks["copy"], abs=1e-12)


def test_joint_structure_metrics_prefer_the_copy(known_answers):
    corr = _metric(known_answers, "corr_mat_diff")
    assert corr["copy"] < corr["shuffle"]
    detection = _metric(known_answers, "detection.detection_xgb.mean")
    assert detection["copy"] < detection["shuffle"]


def test_train_on_synthetic_utility_tracks_the_target_signal(known_answers):
    tstr = _metric(known_answers, "performance.xgb.syn_id")
    assert tstr["copy"] - tstr["shuffle"] >= 0.15
    assert abs(tstr["shuffle"] - 0.5) <= 0.2


def test_ranking_puts_copy_above_shuffle_on_utility(known_answers):
    utility = known_answers[("__all__", "utility", "rank")].astype(float)
    assert utility["copy"] > utility["shuffle"]


@pytest.mark.xfail(
    strict=True,
    reason="SynthEval MIA, NNAA, attribute disclosure and the fairness metrics draw "
    "unseeded random samples and classifiers inside their worker processes, so identical "
    "inputs score differently (overall rank moved by ~0.37 between two identical runs).",
)
def test_identical_inputs_get_identical_scores(known_answers):
    numeric = known_answers.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.loc[:, numeric.notna().any()]
    first, twin = numeric.loc["copy"], numeric.loc["copy_twin"]
    differs = [c for c in numeric.columns if not np.isclose(first[c], twin[c], equal_nan=True)]
    assert not differs, differs
