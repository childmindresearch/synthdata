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
import dataclasses

import numpy as np
import pandas as pd
import pytest

from synthdata.data import load_dataset, load_imputed_splits
from synthdata.evaluation import run_evaluation

pytestmark = pytest.mark.integration

N_ROWS = 120


@pytest.fixture(scope="module")
def known_answer_run(imputed_run, tmp_path_factory) -> tuple[pd.DataFrame, dict]:
    cfg = copy_module.deepcopy(imputed_run.cfg)
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
def train_size(imputed_run) -> int:
    return len(load_dataset(imputed_run.cfg).train_df)


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
    # Two seeds give a wide interval (t = 12.7 at one degree of freedom), so
    # only check that the copy beats every shuffle replicate.
    assert (utility[["shuffle", "shuffle__rep1"]] < summary.loc["copy", "utility_mean"]).all()


def test_class_metrics_track_the_target_signal(known_answers):
    # Holdout TSTR (XGBoost fit on each dataset, scored on the real test split).
    # Prior matching keeps the shuffle's macro-F1 well above zero, so only
    # its order is checked; balanced accuracy has a fixed chance level.
    f1 = _metric(known_answers, "tstr_macro_f1")
    assert f1["copy"] > f1["shuffle"]
    balanced = _metric(known_answers, "tstr_balanced_accuracy")
    assert balanced["copy"] > balanced["shuffle"]
    assert abs(balanced["shuffle"] - 0.5) <= 0.2


def test_holdout_distances_flag_the_copy_as_closer_to_train(known_answers):
    # The share is taken against a train sample as large as the test split
    # (0.5 = no memorization). Copies of sampled rows sit at distance 0, so
    # the copy lands well above 0.5; a shuffle has no such pull.
    closer = _metric(known_answers, "dcr_closer_to_train_share")
    assert closer["copy"] - closer["shuffle"] >= 0.15
    assert abs(closer["shuffle"] - 0.5) <= 0.15
    auc = _metric(known_answers, "distance_mia_auc")
    assert auc["copy"] > auc["shuffle"]
    ratio = _metric(known_answers, "dcr_holdout_ratio")
    assert ratio["copy"] == pytest.approx(0.0, abs=1e-9)


@pytest.fixture(scope="module")
def roleless_run(imputed_run, tmp_path_factory) -> pd.DataFrame:
    """The copy scored with no sensitive or protected columns declared."""
    cfg = copy_module.deepcopy(imputed_run.cfg)
    cfg.evaluation.output_dir = str(tmp_path_factory.mktemp("no_roles"))
    cfg.evaluation.generate_report = False
    cfg.evaluation.baselines = []
    cfg.evaluation.synthcity.enabled = False
    cfg.evaluation.log_disparity.protected_columns = []
    cfg.evaluation.log_disparity.protected_map = []
    cfg.evaluation.log_disparity.protected_bins = []
    cfg.data.sensitive_columns = []
    cfg.data.protected_columns = []
    dataset = dataclasses.replace(
        load_imputed_splits(load_dataset(imputed_run.cfg)),
        sensitive_columns=[],
        protected_columns=[],
    )
    real = dataset.train_imputed_df.reset_index(drop=True)
    return run_evaluation(cfg, dataset, {"copy": real.sample(n=N_ROWS, random_state=0)})[0]


def test_role_metrics_are_skipped_when_their_role_is_empty(roleless_run):
    metrics = set(roleless_run.columns.get_level_values("metric"))
    # SynthEval's attribute disclosure needs sensitive columns and its
    # fairness metrics need protected ones; both are left out, not NaN.
    assert "att_discl_risk" not in metrics
    assert not {"statistical_parity", "equalized_odds", "equal_opportunity"} & metrics
    # Anonymeter falls back to singling out, which needs no roles.
    assert np.isfinite(_metric(roleless_run, "anonymeter_singling_out_risk")["copy"])
    assert "mia_recall" in metrics
