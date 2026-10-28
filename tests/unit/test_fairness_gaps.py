"""Holdout fairness gaps on planted data: kept, erased, inverted and no-signal gaps."""

import numpy as np
import pandas as pd
import pytest

from synthdata.evaluation import fairness_gaps as fg
from synthdata.evaluation.combine import build_combined_table
from synthdata.evaluation.privacy_attacks import ANONYMETER_METRICS

pytestmark = pytest.mark.unit

TARGET = "y"


def _draw(n, seed, group_effect=1.5, signal=2.0):
    """x carries the label signal; group g (30% of rows) shifts the label rate."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    g = (rng.random(n) < 0.3).astype(int)
    logit = signal * x + group_effect * g - 0.5
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"x": x, "noise": rng.normal(size=n), "g": g, TARGET: y})


@pytest.fixture(scope="module")
def real():
    train, test = _draw(4000, 0), _draw(3000, 1)
    masks = {("g", "1"): test["g"].to_numpy() == 1}
    real_pred = fg.real_holdout_predictions(train, test, TARGET, ["g"], [0, 1], seed=0)
    return train, test, masks, real_pred


def _score(real, synthetic):
    train, test, masks, real_pred = real
    return fg.fairness_gap_scores(
        synthetic, train, test, TARGET, ["g"], [0, 1], masks, seed=0, real_predictions=real_pred
    )


def test_a_dataset_that_keeps_the_real_gap_scores_near_zero(real):
    result = _score(real, _draw(4000, 2))
    assert result.signal
    assert result.balanced_accuracy_ci[0] > result.chance
    for metric in fg.FAIRNESS_GAP_METRICS:
        assert result.errors[metric] < 0.06, metric


def test_erasing_or_inverting_the_gap_scores_high(real):
    kept = _score(real, _draw(4000, 2)).errors
    erased = _score(real, _draw(4000, 3, group_effect=0.0)).errors
    inverted = _score(real, _draw(4000, 4, group_effect=-1.5)).errors
    for metric in fg.FAIRNESS_GAP_METRICS:
        assert erased[metric] > kept[metric] + 0.08, metric
        assert inverted[metric] > erased[metric] + 0.08, metric


def test_a_dataset_with_no_class_signal_gets_nan_and_a_flag(real):
    result = _score(real, _draw(4000, 5, group_effect=0.0, signal=0.0))
    assert not result.signal
    assert all(np.isnan(v) for v in result.errors.values())
    # The gaps behind it stay in the detail table.
    assert not result.gaps.empty


def test_a_constant_prediction_has_no_signal(real):
    train, test, masks, real_pred = real
    one_class = _draw(4000, 6).assign(**{TARGET: 1})
    result = fg.fairness_gap_scores(
        one_class, train, test, TARGET, ["g"], [0, 1], masks, seed=0, real_predictions=real_pred
    )
    assert not result.signal
    assert result.balanced_accuracy == pytest.approx(0.5)


def test_small_groups_are_not_scored():
    y = np.r_[np.zeros(50), np.ones(50)].astype(int)
    small = np.zeros(100, dtype=bool)
    small[:10] = True
    assert fg.group_gaps(y, y, {("g", "a"): small}, 2).empty


def test_gaps_are_group_minus_rest_per_class():
    # Group selection rate 0.8 vs rest 0.2 for class 1, so the class-1 gap is +0.6.
    y_true = np.r_[np.ones(50), np.zeros(50), np.ones(50), np.zeros(50)].astype(int)
    group = np.r_[np.ones(100), np.zeros(100)].astype(bool)
    y_pred = np.r_[np.ones(80), np.zeros(20), np.ones(20), np.zeros(80)].astype(int)
    gaps = fg.group_gaps(y_true, y_pred, {("g", "a"): group}, 2).set_index("class")
    assert gaps.loc[1, "dp"] == pytest.approx(0.6)
    assert gaps.loc[0, "dp"] == pytest.approx(-0.6)
    assert gaps.loc[1, "tpr"] == pytest.approx(1.0 - 0.4)
    assert gaps.loc[1, "fpr"] == pytest.approx(0.6 - 0.0)


def test_a_gated_model_is_left_out_of_the_fairness_rank_not_scored_worst():
    table = pd.DataFrame(
        {
            fg.FAIRNESS_SIGNAL_COLUMN: [1, 1, 0],
            **{m: [0.01, 0.30, np.nan] for m in fg.FAIRNESS_GAP_METRICS},
        },
        index=["good", "bad", "nosignal"],
    )
    # Every model ties on privacy (scaled 0.5), so only fairness separates them.
    privacy = {"scores": {m: {ANONYMETER_METRICS[0]: 0.1} for m in table.index}}
    combined = build_combined_table(
        {}, None, None, {}, list(table.index), privacy_result=privacy, fairness_gap_table=table
    )
    fairness = combined[("__all__", "fairness", "rank")]
    assert fairness["good"] == pytest.approx(1.0)
    assert fairness["bad"] == pytest.approx(0.0)
    assert np.isnan(fairness["nosignal"])
    overall = combined[("__all__", "overall", "rank")]
    # The gated model is scored on privacy alone, not given the worst fairness.
    assert overall["nosignal"] == pytest.approx(0.5)
    assert overall["good"] > overall["nosignal"] > overall["bad"]
    directions = combined.attrs["metric_directions"]
    assert directions.loc[("custom", "fairness", fg.FAIRNESS_SIGNAL_COLUMN), "direction"] == (
        "unranked"
    )
