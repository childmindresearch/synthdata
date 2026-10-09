"""Known-answer checks: every metric family on planted data with a theoretical answer.

The reference tests (test_*_reference_metrics.py) pin individual fixes. These
pin what each metric must say about synthetic data whose properties are known
by construction, run through the pipeline's own wrappers:

* ``same``: an independent draw from the real distribution. Fidelity is
  near perfect and every privacy attack sits at chance.
* ``copy``: the training rows, shuffled. Fidelity is perfect and privacy is
  the worst possible.
* ``shift``: ``x1`` moved by one standard deviation. Only ``x1``'s marginal
  changes; the KS statistic of N(0,1) vs N(1,1) is 2*Phi(1/2) - 1 = 0.383.
* ``indep``: ``x2`` permuted. Every marginal is kept; the 0.6 correlation
  with ``x1`` (and ``x2``'s link to the secret ``s``) is gone.
* ``label_noise``: the target permuted. Classifiers trained on it are at chance.
* ``collapse``: one training row repeated.

A wrong sign, a smoothing bug or a degenerate encoding shows up as a metric
outside its expected range. docs/verification.md lists what each check found.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp, norm

from synthdata.config import AnonymeterConfig
from synthdata.evaluation.catalog import SYNTHEVAL_PRESET
from synthdata.evaluation.privacy_attacks import anonymeter_risks, holdout_distance_scores
from synthdata.evaluation.synthcity_eval import run_synthcity_metrics
from synthdata.evaluation.syntheval_eval import _evaluate_seeded
from synthdata.evaluation.tstr import tstr_scores

pytestmark = pytest.mark.unit

N = 600
CATEGORICAL = ["c1", "s", "target"]


def _draw(n: int, seed: int, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], n)
    c1 = rng.integers(0, 3, n)
    target = rng.random(n) < 1 / (1 + np.exp(-(1.5 * z[:, 0] + (c1 == 2) - 0.5)))
    secret = rng.random(n) < 1 / (1 + np.exp(-2 * z[:, 1]))
    return pd.DataFrame(
        {
            "x1": z[:, 0] + shift,
            "x2": z[:, 1],
            "c1": c1,
            "s": secret.astype(int),
            "target": target.astype(int),
        }
    )


TRAIN, HOLDOUT = _draw(N, 1), _draw(N, 2)


def _scenarios() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(9)
    fresh = _draw(N, 3)
    return {
        "same": fresh,
        "copy": TRAIN.sample(frac=1, random_state=0).reset_index(drop=True),
        "shift": _draw(N, 3, shift=1.0),
        "indep": fresh.assign(x2=rng.permutation(fresh["x2"].to_numpy())),
        "label_noise": fresh.assign(target=rng.permutation(fresh["target"].to_numpy())),
        "collapse": pd.concat([TRAIN.iloc[[0]]] * N, ignore_index=True),
    }


SCENARIOS = _scenarios()


@pytest.fixture(scope="module")
def synthcity(tmp_path_factory):
    metrics = {
        "sanity": ["common_rows_proportion"],
        "stats": ["jensenshannon_dist", "chi_squared_test", "ks_test", "wasserstein_dist"],
        "detection": ["detection_xgb", "detection_linear"],
        "privacy": ["identifiability_score"],
    }
    workspace = tmp_path_factory.mktemp("synthcity")
    return {
        name: run_synthcity_metrics(
            frame,
            HOLDOUT,
            TRAIN,
            "target",
            ["s"],
            metrics,
            workspace=str(workspace),
            discrete_columns=CATEGORICAL,
        )["mean"]
        for name, frame in SCENARIOS.items()
        if name != "label_noise"
    }


@pytest.fixture(scope="module")
def syntheval():
    from syntheval import AnalysisConfig, SynthEval

    preset = {
        name: SYNTHEVAL_PRESET[name]
        for name in ("corr_diff", "mi_diff", "ks_test", "h_dist", "p_mse", "q_mse", "auroc_diff")
        + ("nnaa", "dcr", "eps_risk")
    }
    evaluator = SynthEval(
        TRAIN,
        holdout_dataframe=HOLDOUT,
        cat_cols=CATEGORICAL,
        verbose=False,
        enable_plots=False,
        console="off",
        show_warnings=False,
    )
    config = AnalysisConfig(
        dataset=TRAIN, target_vars="target", confounder_vars=None, sensitive_vars=["s"]
    )
    results = {}
    for name in ("same", "copy", "shift", "indep", "label_noise"):
        frame, failed = _evaluate_seeded(
            evaluator, SCENARIOS[name], {"sensitive": config, "protected": config}, preset, 0, name
        )
        assert not failed, failed
        results[name] = frame.set_index("metric")["val"]
    return results


# --- synthcity ---------------------------------------------------------------


def test_synthcity_copy_scores_perfect_fidelity(synthcity):
    copy = synthcity["copy"]
    assert copy["sanity.common_rows_proportion.score"] == pytest.approx(1.0)
    assert copy["stats.jensenshannon_dist.marginal"] == pytest.approx(0.0, abs=1e-9)
    assert copy["stats.chi_squared_test.marginal"] == pytest.approx(1.0)
    assert copy["stats.ks_test.marginal"] == pytest.approx(1.0)
    assert copy["stats.wasserstein_dist.joint"] == pytest.approx(0.0, abs=1e-6)
    assert synthcity["same"]["sanity.common_rows_proportion.score"] == 0.0


def test_synthcity_ks_matches_scipy_on_a_known_shift(synthcity):
    # 1 - KS statistic, averaged over columns (categorical ones label-encoded).
    expected = np.mean([1 - ks_2samp(TRAIN[c], SCENARIOS["shift"][c]).statistic for c in TRAIN])
    assert synthcity["shift"]["stats.ks_test.marginal"] == pytest.approx(expected, abs=1e-9)
    x1 = ks_2samp(TRAIN["x1"], SCENARIOS["shift"]["x1"]).statistic
    assert x1 == pytest.approx(2 * norm.cdf(0.5) - 1, abs=0.08)


def test_synthcity_jensenshannon_uses_its_full_range(synthcity):
    # Natural-log JS distance tops out at sqrt(ln 2) = 0.83 for disjoint
    # histograms; a collapsed generator is nearly disjoint on the continuous
    # columns. Smoothing the normalized histograms had capped this at 0.10.
    assert synthcity["collapse"]["stats.jensenshannon_dist.marginal"] > 0.4
    assert synthcity["same"]["stats.jensenshannon_dist.marginal"] < 0.05


def test_synthcity_detection_flags_copies_and_not_fresh_draws(synthcity):
    for model in ("detection_xgb", "detection_linear"):
        key = f"detection.{model}.mean"
        assert 0.5 <= synthcity["same"][key] < 0.6, model
        assert synthcity["collapse"][key] > 0.9, model
    # An AUC below 0.5 is as detectable as above: a copy must not look perfect.
    assert synthcity["copy"]["detection.detection_xgb.mean"] > 0.8


def test_synthcity_identifiability_is_one_for_a_copy_and_chance_for_a_draw(synthcity):
    assert synthcity["copy"]["privacy.identifiability_score.score"] == pytest.approx(1.0)
    assert synthcity["same"]["privacy.identifiability_score.score"] == pytest.approx(0.5, abs=0.1)


# --- SynthEval ---------------------------------------------------------------


def test_syntheval_copy_scores_zero_distance_everywhere(syntheval):
    copy = syntheval["copy"]
    for metric in ("corr_mat_diff", "mutual_inf_diff", "ks_tvd_stat", "avg_h_dist", "avg_qMSE"):
        assert copy[metric] == pytest.approx(0.0, abs=1e-9), metric
    assert copy["avg_pMSE"] < 0.005
    assert copy["auroc"] == pytest.approx(0.0, abs=1e-9)
    assert copy["nnaa"] == 0.0
    assert copy["median_DCR"] == 0.0
    assert copy["eps_identif_risk"] == 1.0


def test_syntheval_dependence_metrics_see_a_broken_correlation(syntheval):
    # Binned NMI of a bivariate normal with rho = 0.6 is about 0.09; both
    # off-diagonal entries go, so the Frobenius norm grows by about 0.12.
    # NMI on raw continuous values could not see it at all.
    same, indep = syntheval["same"], syntheval["indep"]
    assert indep["mutual_inf_diff"] - same["mutual_inf_diff"] > 0.06
    assert indep["corr_mat_diff"] - same["corr_mat_diff"] > 0.5
    # Marginals are untouched, so the marginal metrics do not move.
    for metric in ("ks_tvd_stat", "avg_h_dist", "avg_qMSE"):
        assert indep[metric] == pytest.approx(same[metric], abs=1e-9), metric


def test_syntheval_marginal_metrics_see_a_known_shift(syntheval):
    same, shift = syntheval["same"], syntheval["shift"]
    # Hellinger distance of N(0,1) and N(1,1) is sqrt(1 - exp(-1/8)) = 0.34,
    # averaged over 5 columns; KS of x1 is 0.38.
    assert shift["avg_h_dist"] - same["avg_h_dist"] == pytest.approx(0.34 / 5, abs=0.03)
    assert shift["ks_tvd_stat"] - same["ks_tvd_stat"] == pytest.approx(0.38 / 5, abs=0.03)
    assert shift["avg_qMSE"] > same["avg_qMSE"]


def test_syntheval_classifier_metrics_see_label_noise(syntheval):
    assert syntheval["same"]["auroc"] == pytest.approx(0.0, abs=0.03)
    assert syntheval["label_noise"]["auroc"] < -0.1


def test_syntheval_privacy_metrics_sit_at_chance_for_a_draw(syntheval):
    same = syntheval["same"]
    assert same["nnaa"] == pytest.approx(0.5, abs=0.06)
    assert same["eps_identif_risk"] == pytest.approx(0.5, abs=0.1)


# --- custom ------------------------------------------------------------------


def test_tstr_matches_real_data_and_falls_to_chance_on_label_noise():
    def scores(frame):
        return tstr_scores(frame, HOLDOUT, "target", ["c1", "s"], [0, 1], [0, 1, 2])

    real, copy = scores(SCENARIOS["same"]), scores(SCENARIOS["copy"])
    assert copy.macro_f1 == pytest.approx(real.macro_f1, abs=0.05)
    assert scores(SCENARIOS["label_noise"]).balanced_accuracy == pytest.approx(0.5, abs=0.07)
    # One class only: its F1 is 2p / (1 + p), the other class's is 0.
    collapsed = scores(SCENARIOS["collapse"])
    assert collapsed.balanced_accuracy == pytest.approx(0.5)


@pytest.mark.parametrize(("name", "expected"), [("same", 0.5), ("copy", 1.0)])
def test_holdout_distances_are_at_chance_or_maximal(name, expected):
    scores = holdout_distance_scores(
        TRAIN,
        HOLDOUT,
        SCENARIOS[name],
        CATEGORICAL,
        pd.Series(TRAIN.index),
        pd.Series(HOLDOUT.index),
        0,
    )
    assert scores["dcr_closer_to_train_share"] == pytest.approx(expected, abs=0.06)
    assert scores["distance_mia_auc"] == pytest.approx(expected, abs=0.06)


def test_anonymeter_attacks_succeed_on_a_copy_and_fail_on_a_draw():
    def risks(name):
        found = anonymeter_risks(
            TRAIN, SCENARIOS[name], HOLDOUT, CATEGORICAL, ["x2", "c1"], ["s"], AnonymeterConfig(), 0
        )
        return {risk.attack: risk.risk for risk in found}

    copy, same = risks("copy"), risks("same")
    # Linkability joins the quasi-identifiers to the rest of the record; with
    # only the one binary secret as the second half, a full copy scored 0.01.
    assert copy["linkability"] > 0.8
    assert copy["inference"] > 0.8
    assert copy["singling_out"] > 0.1
    for attack, risk in same.items():
        assert risk < 0.1, attack
