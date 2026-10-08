"""Unit tests for per-class quota sampling (conditional and rejection)."""

import numpy as np
import pandas as pd
import pytest

from synthdata.generation.class_quota import MAX_ROUNDS, class_quotas, sample_to_quota

pytestmark = pytest.mark.unit

PRIOR = pd.Series({"a": 0.7, "b": 0.2, "c": 0.1})


def _skewed_sampler(shares, calls=None):
    """Unconditional sampler whose class shares are ``shares``; fresh rows every call."""

    def sample(count, seed, labels):
        assert labels is None
        if calls is not None:
            calls.append(count)
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            {
                "x": rng.normal(size=count),
                "y": rng.choice(list(shares), count, p=list(shares.values())),
            }
        )

    return sample


@pytest.mark.parametrize(("n", "expected"), [(200, [140, 40, 20]), (10, [7, 2, 1]), (7, [5, 1, 1])])
def test_quotas_use_largest_remainder_rounding(n, expected):
    assert class_quotas(PRIOR, n).tolist() == expected


def test_rejection_tops_up_a_rare_class_without_repeating_rows():
    calls = []
    sample = _skewed_sampler({"a": 0.8, "b": 0.18, "c": 0.02}, calls)
    out, report = sample_to_quota(sample, "y", PRIOR, 200, seed=0)

    assert out["y"].value_counts().to_dict() == {"a": 140, "b": 40, "c": 20}
    assert out["x"].is_unique
    assert report.method == "rejection"
    assert report.shortfall == {"a": 0, "b": 0, "c": 0}
    assert report.rounds > 1 and calls[0] == 200
    assert report.raw_shares["c"] < 0.05


def test_a_class_the_model_never_makes_stays_short():
    out, report = sample_to_quota(_skewed_sampler({"a": 0.8, "b": 0.2}), "y", PRIOR, 100, seed=0)

    assert report.shortfall == {"a": 0, "b": 0, "c": 10}
    assert report.rounds == MAX_ROUNDS
    assert len(out) == 90 and out["x"].is_unique


def test_a_backend_that_ignores_the_seed_cannot_fill_quotas_with_repeats():
    fixed = pd.DataFrame({"x": range(100), "y": ["a"] * 95 + ["b"] * 5})
    out, report = sample_to_quota(lambda count, seed, labels: fixed, "y", PRIOR, 100, seed=0)

    assert report.filled == {"a": 70, "b": 5, "c": 0}
    assert report.rounds == 2
    assert not out.duplicated().any()


def test_first_batch_is_reused_and_output_is_seeded():
    sample = _skewed_sampler({"a": 0.5, "b": 0.3, "c": 0.2})
    first = sample(300, 3, None)  # enough rows of every class
    out, report = sample_to_quota(sample, "y", PRIOR, 100, seed=3, first_batch=first)
    again, _ = sample_to_quota(sample, "y", PRIOR, 100, seed=3, first_batch=first)

    assert report.rounds == 1
    assert set(out["x"]) <= set(first["x"])
    pd.testing.assert_frame_equal(out, again)


def test_conditional_sampling_asks_for_the_missing_labels():
    requested = []

    def sample(count, seed, labels):
        requested.append(list(labels))
        # Drops the last row, as synthcity's constraint filtering can.
        labels = labels[:-1]
        return pd.DataFrame(
            {"x": np.random.default_rng(seed).normal(size=len(labels)), "y": labels}
        )

    out, report = sample_to_quota(sample, "y", PRIOR, 10, seed=0, conditional=True)

    assert requested[0] == ["a"] * 7 + ["b"] * 2 + ["c"]
    assert requested[1] == ["c"]
    assert report.method == "conditional" and report.raw_shares == {}
    assert out["y"].value_counts().to_dict() == {"a": 7, "b": 2}
    assert report.shortfall["c"] == 1
