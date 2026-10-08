"""Reference checks for synthcity statistical metrics patched in the fork.

Each test pins a metric to its textbook definition, computed independently
with scipy or by hand, so a regression to the upstream behaviour fails here.
See docs/verification.md for the full table.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chisquare
from synthcity.metrics._utils import get_frequency
from synthcity.metrics.eval_statistical import (
    AlphaPrecision,
    ChiSquaredTest,
    InverseKLDivergence,
)
from synthcity.plugins.core.dataloader import GenericDataLoader

pytestmark = pytest.mark.unit


@pytest.fixture
def no_cache(tmp_path) -> dict:
    """synthcity caches metric results on disk by data hash; keep reruns honest."""
    return {"use_cache": False, "workspace": tmp_path}


def _binary(n_ones: int, n: int, name: str = "x") -> pd.DataFrame:
    return pd.DataFrame({name: [1] * n_ones + [0] * (n - n_ones)})


class TestGetFrequency:
    def test_pairs_categories_by_key_not_by_frequency_rank(self):
        real = _binary(70, 100)
        synthetic = _binary(30, 100)

        gt, syn = get_frequency(real, synthetic)["x"]

        # Same category order on both sides: 1 is 70% real and 30% synthetic.
        assert sorted(zip(gt, syn, strict=True)) == [
            pytest.approx((0.3, 0.7)),
            pytest.approx((0.7, 0.3)),
        ]

    def test_flipped_split_is_not_scored_as_identical(self, no_cache):
        real = GenericDataLoader(_binary(70, 100))
        flipped = GenericDataLoader(_binary(30, 100))

        score = InverseKLDivergence(**no_cache).evaluate(real, flipped)["marginal"]

        # 1 / (1 + KL(0.7,0.3 || 0.3,0.7)) = 1 / (1 + 0.4 ln(7/3)).
        assert score == pytest.approx(1 / (1 + 0.4 * np.log(7 / 3)), rel=1e-6)


class TestChiSquaredTest:
    def test_matches_scipy_on_counts(self, no_cache):
        real = _binary(70, 200)
        synthetic = _binary(110, 200)

        pvalue = ChiSquaredTest(**no_cache).evaluate(
            GenericDataLoader(real), GenericDataLoader(synthetic)
        )["marginal"]

        # Observed real counts against counts expected from synthetic shares.
        expected = chisquare([70, 130], [110, 90]).pvalue
        assert pvalue == pytest.approx(expected, rel=1e-6)
        assert pvalue < 1e-6

    def test_same_distribution_is_not_rejected(self, no_cache):
        real = _binary(70, 200)

        pvalue = ChiSquaredTest(**no_cache).evaluate(
            GenericDataLoader(real), GenericDataLoader(real.copy())
        )["marginal"]

        assert pvalue == pytest.approx(1.0)


class TestAuthenticity:
    @staticmethod
    def _authenticity(real: np.ndarray, synthetic: np.ndarray) -> float:
        results = AlphaPrecision().metrics(real, synthetic)
        return results[5]

    def test_mode_collapse_onto_one_real_record_is_unauthentic(self):
        real = np.random.default_rng(0).normal(size=(200, 3))
        collapsed = np.repeat(real[:1], len(real), axis=0)

        assert self._authenticity(real, collapsed) == 0.0

    def test_matches_per_synthetic_row_definition(self):
        rng = np.random.default_rng(1)
        real = rng.normal(size=(150, 3))
        synthetic = rng.normal(size=(150, 3))

        # Alaa et al. (2022): synthetic row j is authentic when its distance to
        # the nearest real record i* exceeds i*'s distance to its own nearest
        # real neighbour.
        real_d = np.linalg.norm(real[:, None] - real[None], axis=-1)
        np.fill_diagonal(real_d, np.inf)
        real_radius = real_d.min(axis=1)
        syn_d = np.linalg.norm(synthetic[:, None] - real[None], axis=-1)
        nearest = syn_d.argmin(axis=1)
        expected = np.mean(syn_d.min(axis=1) > real_radius[nearest])

        assert self._authenticity(real, synthetic) == pytest.approx(expected)
