"""Reference checks for SynthEval metrics patched in the fork.

Each test pins a metric to its published definition, computed independently
with numpy or by hand, so a regression to the upstream behaviour fails here.
See docs/verification.md for the full table.
"""

import numpy as np
import pandas as pd
import pytest
from syntheval import AnalysisConfig
from syntheval.metrics.fairness.metric_statistical_parity import StatisticalParity
from syntheval.metrics.privacy.metric_epsilon_identifiability import EpsilonIdentifiability
from syntheval.metrics.privacy.metric_nn_adversarial_accuracy import (
    NearestNeighbourAdversarialAccuracy,
)
from syntheval.metrics.utility.metric_hellinger_distance import (
    HellingerDistance,
    _scott_ref_rule,
)

pytestmark = pytest.mark.unit


def _hellinger(p: np.ndarray, q: np.ndarray) -> float:
    p, q = p / p.sum(), q / q.sum()
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


class TestHellingerDistance:
    def test_bins_follow_scotts_rule_on_the_pooled_sample(self):
        rng = np.random.default_rng(0)
        real, synthetic = rng.uniform(size=300), rng.uniform(size=200)

        edges = _scott_ref_rule(real, synthetic)

        expected = np.histogram_bin_edges(np.concatenate([real, synthetic]), bins="scott")
        np.testing.assert_allclose(edges, expected)
        assert len(edges) > 2

    def test_shifted_scaled_column_has_a_positive_distance(self):
        rng = np.random.default_rng(1)
        real = pd.DataFrame({"x": rng.uniform(0.0, 0.6, size=400)})
        synthetic = pd.DataFrame({"x": rng.uniform(0.4, 1.0, size=400)})

        result = HellingerDistance(
            real, synthetic, cat_cols=[], num_cols=["x"], do_preprocessing=False
        ).evaluate()

        edges = np.histogram_bin_edges(np.concatenate([real.x, synthetic.x]), bins="scott")
        expected = _hellinger(
            np.histogram(real.x, bins=edges)[0].astype(float),
            np.histogram(synthetic.x, bins=edges)[0].astype(float),
        )
        assert result["avg"] == pytest.approx(expected)
        assert result["avg"] > 0.5

    def test_categories_are_counted_over_a_shared_level_set(self):
        real = pd.DataFrame({"c": [0] * 50 + [1] * 50})
        synthetic = pd.DataFrame({"c": [1] * 50 + [2] * 50})

        result = HellingerDistance(
            real, synthetic, cat_cols=["c"], num_cols=[], do_preprocessing=False
        ).evaluate()

        assert result["avg"] == pytest.approx(
            _hellinger(np.array([50.0, 50.0, 0.0]), np.array([0.0, 50.0, 50.0]))
        )


class TestEpsilonIdentifiability:
    @staticmethod
    def _run(real, synthetic, holdout, cat_cols, num_cols):
        return EpsilonIdentifiability(
            real,
            synthetic,
            hout_data=holdout,
            cat_cols=cat_cols,
            num_cols=num_cols,
            nn_dist="gower",
            do_preprocessing=False,
        ).evaluate()

    def test_constant_column_does_not_change_the_risk(self):
        rng = np.random.default_rng(2)
        frame = lambda n: pd.DataFrame(  # noqa: E731
            {"a": rng.integers(0, 3, n), "b": rng.integers(0, 4, n), "x": rng.uniform(size=n)}
        )
        real, synthetic, holdout = frame(80), frame(80), frame(40)
        without = self._run(real, synthetic, holdout, ["a", "b"], ["x"])

        def with_constant(df: pd.DataFrame) -> pd.DataFrame:
            return df.assign(k=0)

        with_k = self._run(
            with_constant(real),
            with_constant(synthetic),
            with_constant(holdout),
            ["a", "b", "k"],
            ["x"],
        )

        assert with_k["eps_risk"] == pytest.approx(without["eps_risk"])
        assert with_k["priv_loss"] == pytest.approx(without["priv_loss"])

    def test_holdout_risk_is_a_share_of_holdout_rows(self):
        rng = np.random.default_rng(3)
        real = pd.DataFrame({"x": rng.uniform(size=120)})
        holdout = pd.DataFrame({"x": rng.uniform(size=30)})
        # Synthetic rows a hair from every holdout row make every holdout row
        # identifiable: holdout risk 1. (An exact copy would be mistaken for a
        # self-comparison by SynthEval's nearest-neighbour helper.)
        result = self._run(real, holdout + 1e-9, holdout, [], ["x"])

        assert result["priv_loss"] == pytest.approx(result["eps_risk"] - 1.0)


class TestStatisticalParity:
    def test_opposite_gaps_on_two_attributes_do_not_cancel(self, monkeypatch):
        rng = np.random.default_rng(4)
        data = pd.DataFrame(
            {
                "a": rng.integers(0, 2, 60),
                "b": rng.integers(0, 2, 60),
                "x": rng.uniform(size=60),
                "label": np.tile([0, 1], 30),
            }
        )
        gaps = {"a": 0.1, "b": -0.1}
        monkeypatch.setattr(
            StatisticalParity,
            "statistical_parity",
            staticmethod(lambda X, S, preds, positive_pred=1: gaps[S]),
        )
        config = AnalysisConfig(dataset=data, target_vars="label", sensitive_vars=["a", "b"])

        result = StatisticalParity(
            data, data.copy(), analysis_target=config, do_preprocessing=False
        ).evaluate(folds=2)

        assert result["statistical_parity"] == pytest.approx(0.1)


class TestNearestNeighbourAdversarialAccuracy:
    @pytest.mark.parametrize(("accuracy", "expected"), [(0.0, 0.0), (0.5, 1.0), (0.75, 0.5)])
    def test_normalized_score_peaks_at_one_half(self, accuracy, expected):
        metric = NearestNeighbourAdversarialAccuracy.__new__(NearestNeighbourAdversarialAccuracy)
        metric.results = {"avg": accuracy, "err": 0.01}
        metric.hout_data = None

        (row,) = metric.normalize_output()

        assert row["n_val"] == pytest.approx(expected)
        assert row["n_err"] == pytest.approx(0.02)


class TestAnalysisConfigTargetTypes:
    """Integer and categorical targets must be typed categorical on every platform.

    The fork used ``dtype == "int"``, which is int32 on Windows with NumPy < 2, so
    an int64 target was typed numeric there and every classification and fairness
    metric failed with "no categorical target variables".
    """

    @pytest.mark.parametrize("dtype", ["int64", "int32", "category", "object", "bool"])
    def test_discrete_targets_are_categorical(self, dtype):
        frame = pd.DataFrame({"x": [0.1, 0.2, 0.3, 0.4], "y": [0, 1, 1, 0]})
        frame["y"] = frame["y"].astype(dtype)

        config = AnalysisConfig(dataset=frame, target_vars="y")

        assert config.target_types == {"y": 2}

    def test_float_targets_stay_numeric(self):
        frame = pd.DataFrame({"x": [0.1, 0.2, 0.3], "y": [0.5, 1.5, 2.5]})

        assert AnalysisConfig(dataset=frame, target_vars="y").target_types == {"y": "num"}
