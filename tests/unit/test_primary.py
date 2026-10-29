"""Unit tests for synthdata.evaluation.primary: baseline-anchored primary
scores, the Pareto front and the ranking bootstrap."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score

from synthdata.evaluation import primary
from synthdata.evaluation.combine import build_combined_table, summarize_replicates
from synthdata.evaluation.fairness_gaps import gap_errors, group_gaps

pytestmark = pytest.mark.unit

AUPRC = ("custom", "utility", "tstr_macro_auprc")
MIA = ("custom", "privacy", "distance_mia_auc")
DP = ("custom", "fairness", "demographic_parity_gap_error")


def _raw(rows: dict) -> pd.DataFrame:
    table = pd.DataFrame.from_dict(rows, orient="index", columns=[AUPRC, MIA, DP])
    table.columns = pd.MultiIndex.from_tuples(table.columns)
    return table


def _spec(raw):
    return primary.resolve_primary([c[2] for c in raw.columns], raw, pd.DataFrame())


def _baselines(**extra):
    return _raw(
        {
            "baseline_train_copy": [0.40, 1.00, 0.01],
            "baseline_marginals": [0.30, 0.50, 0.09],
            **extra,
        }
    )


class TestAnchoring:
    def test_baselines_sit_at_zero_and_one(self):
        raw = _baselines(model=[0.35, 0.60, 0.05])
        out = primary.point_scores(raw, _spec(raw), {"utility": 1, "privacy": 1, "fairness": 1})
        anchored = out["anchored"]
        assert anchored.loc["baseline_train_copy", AUPRC] == pytest.approx(1.0)
        assert anchored.loc["baseline_marginals", AUPRC] == pytest.approx(0.0)
        # Privacy runs the other way: the copy is the worst.
        assert anchored.loc["baseline_train_copy", MIA] == pytest.approx(0.0)
        assert anchored.loc["baseline_marginals", MIA] == pytest.approx(1.0)
        assert anchored.loc["model", AUPRC] == pytest.approx(0.5)
        assert anchored.loc["model", MIA] == pytest.approx(0.8)
        assert anchored.loc["model", DP] == pytest.approx(0.5)
        overall = out["replicates"].loc["model", "overall"]
        assert overall == pytest.approx((0.5 * 0.8 * 0.5) ** (1 / 3))

    def test_scores_do_not_depend_on_the_other_models(self):
        weights = {"utility": 1, "privacy": 1, "fairness": 1}
        alone = _baselines(model=[0.35, 0.60, 0.05])
        crowd = _baselines(model=[0.35, 0.60, 0.05], other=[0.39, 0.95, 0.02])
        a = primary.point_scores(alone, _spec(alone), weights)["replicates"].loc["model"]
        b = primary.point_scores(crowd, _spec(crowd), weights)["replicates"].loc["model"]
        pd.testing.assert_series_equal(a, b)

    def test_beyond_reference_privacy_earns_no_credit(self):
        raw = _baselines(model=[0.35, 0.40, 0.05])  # AUC below 0.5 clips to 0.5
        out = primary.point_scores(raw, _spec(raw), {"utility": 1, "privacy": 1, "fairness": 1})
        assert out["anchored"].loc["model", MIA] == pytest.approx(1.0)

    def test_out_of_order_baselines_drop_the_metric(self):
        raw = _raw(
            {
                "baseline_train_copy": [0.30, 1.00, 0.01],
                "baseline_marginals": [0.40, 0.50, 0.09],  # marginals beat the copy on AUPRC
                "model": [0.35, 0.60, 0.05],
            }
        )
        spec = _spec(raw)
        out = primary.point_scores(raw, spec, {"utility": 1, "privacy": 1, "fairness": 1})
        assert "tstr_macro_auprc" in spec.dropped
        assert out["anchored"][AUPRC].isna().all()
        assert out["replicates"]["utility"].isna().all()
        # The overall score uses the remaining types.
        assert out["replicates"].loc["model", "overall"] == pytest.approx((0.8 * 0.5) ** 0.5)

    def test_unknown_and_ambiguous_names(self):
        raw = _baselines()
        spec = primary.resolve_primary(["not_a_metric"], raw, pd.DataFrame())
        assert spec.columns == [] and "not_a_metric" in spec.dropped
        twice = raw.copy()
        twice[("syntheval", "utility", "tstr_macro_auprc")] = 0.0
        with pytest.raises(ValueError, match="framework/type/metric"):
            primary.resolve_primary(["tstr_macro_auprc"], twice, pd.DataFrame())
        spec = primary.resolve_primary(["custom/utility/tstr_macro_auprc"], twice, pd.DataFrame())
        assert spec.columns == [AUPRC]


def test_pareto_front():
    u = pd.Series({"a": 0.9, "b": 0.5, "c": 0.4, "d": 0.2})
    p = pd.Series({"a": 0.1, "b": 0.6, "c": 0.5, "d": 0.9})
    assert primary.pareto_front(u, p).to_dict() == {"a": True, "b": True, "c": False, "d": True}


class TestWeightedMetrics:
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 3, n)
    proba = rng.dirichlet(np.ones(3), n)
    proba[np.arange(n), y] += 0.3
    counts = rng.integers(0, 3, n).astype(float)
    rows = np.repeat(np.arange(n), counts.astype(int))

    def test_average_precision_matches_sklearn_on_repeated_rows(self):
        label = self.y == 1
        score = np.round(self.proba[:, 1], 2)  # ties
        got = primary.weighted_average_precision(score, label, self.counts[None])[0]
        assert got == pytest.approx(average_precision_score(label[self.rows], score[self.rows]))

    def test_auc_matches_sklearn_on_repeated_rows(self):
        label = (self.y == 0).astype(int)
        score = np.round(self.proba[:, 0], 2)
        got = primary.weighted_auc(score, label, self.counts[None])[0]
        assert got == pytest.approx(roc_auc_score(label[self.rows], score[self.rows]))

    @pytest.mark.parametrize("ordinal", [False, True])
    def test_kappa_matches_sklearn(self, ordinal):
        pred = self.proba.argmax(axis=1)
        got = primary.weighted_kappa(self.y, pred, 3, ordinal, self.counts[None])[0]
        want = cohen_kappa_score(
            self.y[self.rows], pred[self.rows], weights="quadratic" if ordinal else None
        )
        assert got == pytest.approx(want)

    def test_dp_gap_error_matches_fairness_gaps(self):
        group = self.rng.random(self.n) < 0.4
        masks = {("g", "a"): group, ("g", "b"): ~group}
        syn = self.proba.argmax(axis=1)
        real = np.where(self.rng.random(self.n) < 0.7, self.y, syn)
        want, _ = gap_errors(group_gaps(self.y, syn, masks, 3), group_gaps(self.y, real, masks, 3))
        got = primary.weighted_dp_gap_error(syn, real, masks, 3, np.ones((1, self.n)), 30)[0]
        assert got == pytest.approx(want["demographic_parity_gap_error"])


def _rows_for(names, quality, n=400, seed=0):
    """HoldoutRows where each name's classifier has the given signal strength."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    rows = primary.HoldoutRows(holdout_keys=np.arange(n) // 2, y_true=y)
    for i, name in enumerate(names):
        noise = np.random.default_rng(seed + i + 1).normal(size=n)
        score = quality[name] * y + noise
        p1 = 1 / (1 + np.exp(-score))
        rows.proba[name] = np.c_[1 - p1, p1]
    return rows


class TestBootstrap:
    def _setup(self, quality):
        names = [
            "baseline_train_copy",
            "baseline_marginals",
            *(f"{m}__rep{r}" if r else m for m in ("good", "close", "poor") for r in range(2)),
        ]
        rows = _rows_for(names, {n: quality[n.split("__")[0]] for n in names})
        auprc = {
            n: primary.weighted_macro_auprc(rows.y_true, rows.proba[n], np.ones((1, 400)))[0]
            for n in names
        }
        raw = pd.DataFrame({AUPRC: auprc})
        raw.columns = pd.MultiIndex.from_tuples(raw.columns)
        return raw, rows

    def test_ties_and_paired_differences(self):
        quality = {
            "baseline_train_copy": 3.0,
            "baseline_marginals": 0.0,
            "good": 2.0,
            "close": 1.95,
            "poor": 0.5,
        }
        raw, rows = self._setup(quality)
        spec = _spec_utility(raw)
        weights = {"utility": 1, "privacy": 1, "fairness": 1}
        point = primary.point_scores(raw, spec, weights)
        combined = pd.concat(
            {("__all__", d, "rank"): point["replicates"][d] for d in ("overall", "utility")}, axis=1
        )
        draws = primary.bootstrap_draws(raw, spec, weights, rows, 300, seed=1)
        assert draws["overall"].shape == (300, 5)
        summary = summarize_replicates(combined, draws=draws)
        assert bool(summary.loc["good", "tied_with_best"]) or bool(
            summary.loc["close", "tied_with_best"]
        )
        assert not bool(summary.loc["poor", "tied_with_best"])
        assert summary.loc["poor", "diff_ci_low"] > 0
        assert summary.loc["poor", "overall_ci_low"] <= summary.loc["poor", "overall_mean"]

    def test_same_seed_same_draws(self):
        quality = {
            "baseline_train_copy": 3.0,
            "baseline_marginals": 0.0,
            "good": 2.0,
            "close": 1.0,
            "poor": 0.5,
        }
        raw, rows = self._setup(quality)
        spec = _spec_utility(raw)
        a = primary.bootstrap_draws(raw, spec, {}, rows, 50, seed=3)["overall"]
        b = primary.bootstrap_draws(raw, spec, {}, rows, 50, seed=3)["overall"]
        pd.testing.assert_frame_equal(a, b)


def _spec_utility(raw):
    return primary.resolve_primary(["tstr_macro_auprc"], raw, pd.DataFrame())


def test_build_combined_table_ranks_on_primary_and_keeps_the_composite():
    from synthdata.evaluation.tstr import TSTRScores

    def scores(v):
        return TSTRScores(macro_f1=v, macro_auprc=v, balanced_accuracy=v, per_class_f1={}, kappa=v)

    values = {"baseline_train_copy": 0.5, "baseline_marginals": 0.3, "a": 0.45, "b": 0.35}
    combined = build_combined_table(
        {},
        None,
        None,
        {},
        list(values),
        tstr_result={"scores": {m: scores(v) for m, v in values.items()}},
        primary_metrics=["tstr_macro_auprc"],
    )
    assert combined.loc["a", ("__all__", "utility", "rank")] == pytest.approx(0.75)
    assert combined.loc["b", ("__all__", "utility", "rank")] == pytest.approx(0.25)
    assert ("legacy", "overall", "rank") in combined.columns
    assert ("custom", "utility", "tstr_kappa") in combined.columns
    assert list(primary.primary_table(combined)["anchored"].round(2))
    summary = summarize_replicates(combined)
    assert "legacy_overall_mean" in summary.columns
    assert summary.index[summary["eligible"].astype(bool)][0] == "a"
