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
def known_answer_run(pipeline_run, tmp_path_factory) -> tuple[pd.DataFrame, dict]:
    cfg = copy_module.deepcopy(pipeline_run.cfg)
    cfg.evaluation.output_dir = str(tmp_path_factory.mktemp("known_answers"))
    cfg.evaluation.generate_report = False
    cfg.evaluation.save_per_model_syntheval_plots = False
    # The planted copy and shuffle are this module's references already.
    cfg.evaluation.baselines = []
    dataset = load_imputed_splits(load_dataset(cfg))

    real = dataset.train_imputed_df.reset_index(drop=True)
    copied = real.sample(n=N_ROWS, random_state=0).reset_index(drop=True)
    rng = np.random.default_rng(0)
    shuffled = pd.DataFrame(
        {column: rng.permutation(copied[column].to_numpy()) for column in copied.columns}
    )
    # A second seed replicate of the shuffle (a fresh permutation) gives the
    # ranking summary one model with a real spread to put an interval on.
    reshuffled = pd.DataFrame(
        {column: rng.permutation(copied[column].to_numpy()) for column in copied.columns}
    )
    return run_evaluation(
        cfg,
        dataset,
        {
            "copy": copied,
            "copy_twin": copied.copy(),
            "shuffle": shuffled,
            "shuffle__rep1": reshuffled,
        },
    )


@pytest.fixture(scope="module")
def known_answers(known_answer_run) -> pd.DataFrame:
    return known_answer_run[0]


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


def test_synthcity_detects_rows_copied_from_train(known_answers, train_size):
    # synthcity scores equal-size samples: it draws N_ROWS of the train rows and
    # reports the share found among the N_ROWS copies. At most
    # train_size - N_ROWS of those draws can fall outside the copied rows.
    common = _metric(known_answers, "sanity.common_rows_proportion.score")
    assert common["copy"] >= 1 - (train_size - N_ROWS) / N_ROWS - 1e-9
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


def test_identical_inputs_get_identical_scores(known_answers):
    numeric = known_answers.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.loc[:, numeric.notna().any()]
    first, twin = numeric.loc["copy"], numeric.loc["copy_twin"]
    differs = [c for c in numeric.columns if not np.isclose(first[c], twin[c], equal_nan=True)]
    assert not differs, differs


def test_ranking_summary_pools_seed_replicates(known_answer_run):
    combined, extras = known_answer_run
    summary = extras["ranking_summary"]
    assert sorted(summary.index) == ["copy", "copy_twin", "shuffle"]
    shuffle = summary.loc["shuffle"]
    assert shuffle["n_replicates"] == 2
    utility = combined[("__all__", "utility", "rank")].astype(float)
    assert shuffle["utility_mean"] == pytest.approx(utility[["shuffle", "shuffle__rep1"]].mean())
    assert shuffle["utility_ci_low"] <= shuffle["utility_mean"] <= shuffle["utility_ci_high"]
    # The copy's utility sits above everything seed noise allows the shuffle.
    assert summary.loc["copy", "utility_mean"] > shuffle["utility_ci_high"]
