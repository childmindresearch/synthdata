"""Unit tests for objective selection, noise estimates and search control in HPO."""

import datetime

import numpy as np
import optuna
import pandas as pd
import pytest

from synthdata.config import HPOConfig, HPOConstraintsConfig, _validate_hpo
from synthdata.evaluation.tstr import chance_scores, tstr_scores
from synthdata.generation import hpo as hpo_module
from synthdata.generation.hpo import (
    REFIT_SEED_ATTR,
    SCREENS,
    STUDY_OBJECTIVE_ATTR,
    _Patience,
    _refit_top_k,
    build_hpo_eval_fn,
    choose_objective,
    fidelity_scores,
    protected_groups,
    protected_scores,
    run_study,
    trial_seed,
    warm_start_params,
)

pytestmark = pytest.mark.unit

FEASIBLE = [-1.0] * len(SCREENS)


def _clinic(n: int, seed: int, signal: float = 2.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.where(x * signal + rng.normal(size=n) > 1.2, "high", "low")
    return pd.DataFrame(
        {
            "x": x,
            "age": rng.integers(18, 90, size=n),
            "sex": rng.choice(["f", "m"], size=n),
            "y": y,
        }
    )


def _hpo(**overrides) -> HPOConfig:
    defaults = dict(
        n_trials=6,
        timeout_seconds=None,
        n_startup_trials=3,
        patience=None,
        refit_seeds=0,
        n_bootstrap=50,
        log_fidelity=False,
    )
    return HPOConfig(**(defaults | overrides))


# --- TSTR noise and chance --------------------------------------------------


def test_chance_scores_match_an_uninformed_classifier():
    score = pd.DataFrame({"y": ["a"] * 80 + ["b"] * 20, "x": 0.0})
    chance = chance_scores(score, "y", ["a", "b"])
    # Always "a": F1 = 2 * 0.8 / 1.8 for "a", 0 for "b".
    assert chance["macro_f1"] == pytest.approx((2 * 0.8 / 1.8) / 2)
    assert chance["macro_auprc"] == pytest.approx(0.5)
    assert chance["macro_auroc"] == 0.5
    assert chance["balanced_accuracy"] == 0.5


def test_bootstrap_se_auroc_and_group_auprc_are_reported():
    fit, score = _clinic(400, 0), _clinic(300, 1)
    groups = {"sex": score["sex"].to_numpy()}
    scores = tstr_scores(
        fit, score, "y", ["sex"], ["high", "low"], [0], n_bootstrap=50, groups=groups
    )
    assert scores.macro_auroc > 0.8
    assert set(scores.boot_se) == {
        "macro_f1",
        "macro_auprc",
        "macro_auroc",
        "balanced_accuracy",
        "kappa",
    }
    assert all(0 < se < 0.1 for se in scores.boot_se.values())
    assert set(scores.group_auprc["sex"]) == {"f", "m"}
    # Same seed, same rows: the bootstrap is reproducible.
    again = tstr_scores(
        fit, score, "y", ["sex"], ["high", "low"], [0], n_bootstrap=50, groups=groups
    )
    assert again.boot_se == scores.boot_se


def test_eval_fn_reports_headroom_per_candidate_objective():
    train, tuning = _clinic(400, 0), _clinic(200, 1)
    eval_fn = build_hpo_eval_fn(
        train, tuning, "y", ["sex"], ["sex"], True, [], _hpo(), seed=0, match_prior=False
    )
    assert set(eval_fn.headroom) == {
        "tstr_macro_auprc",
        "tstr_macro_auroc",
        "tstr_macro_f1",
        "tstr_kappa",
    }
    auroc = eval_fn.headroom["tstr_macro_auroc"]
    assert auroc["chance"] == 0.5 and auroc["z"] > 2
    score = eval_fn(_clinic(400, 2))
    for key in ("tstr_macro_auroc", "tstr_boot_se", "trtr_macro_auroc", "dcr_closer_to_train"):
        assert key in score.attrs
    assert len(score.constraints) == len(SCREENS)


# --- objective: auto ----------------------------------------------------------


def _trial(values: dict, se: float, feasible: bool = True):
    return optuna.trial.create_trial(
        value=0.0,
        user_attrs=values
        | {
            "constraints": FEASIBLE if feasible else [1.0] * len(SCREENS),
            "tstr_boot_se": {k.removeprefix("tstr_"): se for k in values},
        },
    )


def test_auto_drops_candidates_without_headroom_then_takes_the_most_reliable():
    headroom = {
        "tstr_macro_f1": {"z": 0.4},
        "tstr_macro_auprc": {"z": 6.0},
        "tstr_macro_auroc": {"z": 6.0},
    }
    trials = [
        # macro-F1 varies most but has no headroom; AUPRC varies well beyond
        # its noise (SE 0.005); AUROC only by about its noise.
        _trial({"tstr_macro_f1": f1, "tstr_macro_auprc": a, "tstr_macro_auroc": r}, se=0.005)
        for f1, a, r in [(0.1, 0.30, 0.600), (0.5, 0.36, 0.605), (0.9, 0.42, 0.598)]
    ]
    choice, table = choose_objective(
        ["tstr_macro_f1", "tstr_macro_auroc", "tstr_macro_auprc"], headroom, trials, 2.0
    )
    assert choice == "tstr_macro_auprc"
    assert not table["tstr_macro_f1"]["passes_headroom"]
    assert table["tstr_macro_auprc"]["reliability"] > 0.9
    assert table["tstr_macro_auroc"]["reliability"] < 0.5


def test_auto_ties_go_to_the_first_candidate_and_no_headroom_falls_back():
    trials = [_trial({"tstr_macro_auprc": 0.3, "tstr_macro_auroc": 0.6}, se=0.01)] * 3
    headroom = {"tstr_macro_auprc": {"z": 3.0}, "tstr_macro_auroc": {"z": 5.0}}
    choice, _ = choose_objective(["tstr_macro_auprc", "tstr_macro_auroc"], headroom, trials, 2.0)
    assert choice == "tstr_macro_auprc"
    low = {"tstr_macro_auprc": {"z": 0.5}, "tstr_macro_auroc": {"z": 1.5}}
    choice, _ = choose_objective(["tstr_macro_auprc", "tstr_macro_auroc"], low, trials, 2.0)
    assert choice == "tstr_macro_auroc"


def test_auto_runs_random_start_up_trials_then_continues_on_the_chosen_objective(tmp_path):
    def objective(trial):
        x = trial.suggest_float("x", 0, 1)
        trial.set_user_attr("constraints", FEASIBLE)
        trial.set_user_attr("tstr_macro_auprc", x)
        trial.set_user_attr("tstr_macro_auroc", 0.6)
        trial.set_user_attr("tstr_boot_se", {"macro_auprc": 0.01, "macro_auroc": 0.01})
        study_objective = trial.study.user_attrs.get(STUDY_OBJECTIVE_ATTR)
        return trial.user_attrs.get(study_objective, -1.0)

    cfg = _hpo(objective_candidates=["tstr_macro_auroc", "tstr_macro_auprc"])
    headroom = {"tstr_macro_auroc": {"z": 5.0}, "tstr_macro_auprc": {"z": 5.0}}
    best = run_study("hpo_toy", objective, cfg, tmp_path, seed=0, headroom=headroom)

    storage = f"sqlite:///{tmp_path / 'optuna_studies.db'}"
    study = optuna.load_study(study_name="hpo_toy", storage=storage)
    probe = optuna.load_study(study_name="hpo_toy_probe", storage=storage)
    assert study.user_attrs[STUDY_OBJECTIVE_ATTR] == "tstr_macro_auprc"
    assert len(probe.trials) == 3 and len(study.trials) == 6
    # Probe trials were copied valued by the chosen objective.
    assert [t.value for t in study.trials[:3]] == [
        t.user_attrs["tstr_macro_auprc"] for t in probe.trials
    ]
    assert best["x"] == max(t.params["x"] for t in study.trials)


# --- search control -----------------------------------------------------------


def _toy_study(values, feasible=None):
    study = optuna.create_study(direction="maximize")
    study.set_user_attr(STUDY_OBJECTIVE_ATTR, "tstr_macro_auprc")
    for i, v in enumerate(values):
        ok = feasible is None or feasible[i]
        study.add_trial(
            optuna.trial.create_trial(
                value=v,
                user_attrs={
                    "constraints": FEASIBLE if ok else [1.0] * len(SCREENS),
                    "tstr_boot_se": {"macro_auprc": 0.01},
                },
            )
        )
    return study


def _stopped(study, callback) -> bool:
    stopped = []
    study.stop = lambda: stopped.append(True)
    callback(study, study.trials[-1])
    return bool(stopped)


def test_patience_stops_after_trials_without_a_gain_above_min_delta():
    callback = _Patience("toy", patience=3, min_delta=None, min_trials=1)
    # Gains of 0.005 are under the median SE (0.01), so they don't reset patience.
    assert _stopped(_toy_study([0.30, 0.305, 0.308, 0.309]), callback)
    assert not _stopped(_toy_study([0.30, 0.305, 0.40, 0.402]), callback)


def test_patience_does_not_count_the_start_up_trials():
    # Start-up trials without a gain set the bar but don't use up patience,
    # so TPE gets its own trials before the study can stop.
    flat = [0.30] * 10
    assert not _stopped(_toy_study(flat + [0.30]), _Patience("toy", 3, 0.0, min_trials=10))
    assert _stopped(_toy_study(flat + [0.30] * 3), _Patience("toy", 3, 0.0, min_trials=10))


def test_patience_waits_for_start_up_trials_and_a_feasible_trial():
    assert not _stopped(_toy_study([0.3] * 4), _Patience("toy", 3, 0.0, min_trials=5))
    infeasible = _toy_study([0.3] * 6, feasible=[False] * 6)
    assert not _stopped(infeasible, _Patience("toy", 3, 0.0, min_trials=3))


def test_patience_ends_the_study_early(tmp_path):
    def objective(trial):
        trial.suggest_float("x", 0, 1)
        trial.set_user_attr("constraints", FEASIBLE)
        return 1.0

    cfg = _hpo(objective="tstr_macro_auprc", n_trials=30, patience=4, n_startup_trials=2)
    run_study("hpo_flat", objective, cfg, tmp_path, seed=0)
    study = optuna.load_study(
        study_name="hpo_flat", storage=f"sqlite:///{tmp_path / 'optuna_studies.db'}"
    )
    # Trial 0 sets the best; patience counts trials 2-5, after the 2 start-up trials.
    assert len(study.trials) == 6


def test_refits_stop_at_the_deadline(monkeypatch):
    study = optuna.create_study(direction="maximize")
    choice = optuna.distributions.CategoricalDistribution(["a", "b"])
    start = datetime.datetime(2026, 1, 1)
    for name, value in (("a", 0.9), ("b", 0.8)):
        study.add_trial(
            optuna.trial.FrozenTrial(
                number=0,
                trial_id=0,
                state=optuna.trial.TrialState.COMPLETE,
                value=value,
                values=None,
                datetime_start=start,
                datetime_complete=start + datetime.timedelta(hours=1),
                params={"name": name},
                distributions={"name": choice},
                user_attrs={"constraints": FEASIBLE},
                system_attrs={},
                intermediate_values={},
            )
        )
    clock = [0.0]
    monkeypatch.setattr(hpo_module.time, "time", lambda: clock[0])

    def objective(trial):
        clock[0] += 3600  # each refit takes as long as the trials did
        return 0.5

    cfg = _hpo(refit_top_k=2, refit_seeds=2)
    # Three hours: trial a's two refits fit, trial b's two more don't.
    best = _refit_top_k(study, objective, cfg, 0, "x", deadline=3 * 3600)
    assert clock[0] == 2 * 3600 and best.params == {"name": "a"}
    assert set(study.user_attrs["refit"]) == {"0"}


def test_defaults_are_the_first_trial_and_failures_keep_their_reason(tmp_path):
    def objective(trial):
        x = trial.suggest_float("x", 0, 1)
        if x > 0.9:
            raise ValueError("diverged")
        trial.set_user_attr("constraints", FEASIBLE)
        return x

    cfg = _hpo(objective="tstr_macro_auprc", n_trials=8)
    run_study("hpo_d", objective, cfg, tmp_path, seed=3, default_params={"x": 0.95})
    study = optuna.load_study(
        study_name="hpo_d", storage=f"sqlite:///{tmp_path / 'optuna_studies.db'}"
    )
    first = study.trials[0]
    assert first.params == {"x": 0.95}
    assert first.state == optuna.trial.TrialState.FAIL
    assert first.user_attrs["fail_reason"] == "ValueError: diverged"


def test_refit_picks_the_best_mean_over_new_seeds():
    # "lucky" scored best once but drops on new seeds; "steady" stays high.
    scores = {"lucky": [0.9, 0.5, 0.5], "steady": [0.8, 0.8, 0.8], "weak": [0.1, 0.1, 0.1]}
    choice = optuna.distributions.CategoricalDistribution(list(scores))
    study = optuna.create_study(direction="maximize")
    for name in scores:
        study.add_trial(
            optuna.trial.create_trial(
                value=scores[name][0],
                params={"name": name},
                distributions={"name": choice},
                user_attrs={"constraints": FEASIBLE},
            )
        )

    def objective(trial):
        name = trial.suggest_categorical("name", list(scores))
        return scores[name][trial_seed(trial, 0) // 1000]

    cfg = _hpo(refit_top_k=2, refit_seeds=2)
    best = _refit_top_k(study, objective, cfg, 0, "tstr_macro_auprc")
    assert best.params == {"name": "steady"}
    assert study.user_attrs["refit"] == {"0": [0.9, 0.5, 0.5], "1": [0.8, 0.8, 0.8]}
    # A resumed run reuses the stored refits instead of fitting again.
    assert _refit_top_k(study, None, cfg, 0, "tstr_macro_auprc").params == {"name": "steady"}


def test_trial_seed_defaults_to_the_run_seed():
    fixed = optuna.trial.FixedTrial({})
    assert trial_seed(fixed, 7) == 7
    fixed.set_user_attr(REFIT_SEED_ATTR, 1007)
    assert trial_seed(fixed, 7) == 1007


def test_warm_start_takes_the_top_feasible_trials_of_the_same_study(tmp_path):
    storage = f"sqlite:///{tmp_path / 'earlier.db'}"
    study = optuna.create_study(study_name="hpo_tvae", storage=storage, direction="maximize")
    for x, auprc, ok in [(1, 0.30, True), (2, 0.40, True), (3, 0.50, False), (4, 0.35, True)]:
        study.add_trial(
            optuna.trial.create_trial(
                value=0.0,
                params={"x": x},
                distributions={"x": optuna.distributions.IntDistribution(0, 10)},
                user_attrs={
                    "tstr_macro_auprc": auprc,
                    "constraints": FEASIBLE if ok else [1.0] * len(SCREENS),
                },
            )
        )
    cfg = _hpo(warm_start={"storage": str(tmp_path / "earlier.db"), "top_k": 2})
    assert warm_start_params(cfg, "hpo_tvae") == [{"x": 2}, {"x": 4}]
    assert warm_start_params(cfg, "hpo_ctgan") == []


# --- protected groups and fidelity -------------------------------------------


def test_protected_screens_see_missing_and_skewed_groups():
    train = _clinic(1000, 0)
    groupers = protected_groups(train, ["sex", "age"])
    fresh = protected_scores(_clinic(1000, 1), train, "y", groupers)
    assert fresh["protected_coverage"] == 1.0
    assert fresh["protected_share_gap"] < 0.05

    only_women = _clinic(1000, 2).assign(sex="f")
    skewed = protected_scores(only_women, train, "y", groupers)
    assert skewed["protected_coverage"] == 0.5
    assert skewed["protected_share_gap"] > 0.4

    # Age forms quartile groups: everyone 18 leaves three of four groups empty.
    young = protected_scores(_clinic(1000, 3).assign(age=18), train, "y", groupers)
    assert young["protected_coverage"] == 0.25


def test_protected_screens_use_the_user_bins():
    train = _clinic(1000, 0)
    groupers = protected_groups(train, ["age"], protected_bins={"age": [0, 40, 200]})
    labels = groupers["age"](pd.DataFrame({"age": [18, 40, 41, 250, np.nan]}))
    assert labels[0] == labels[1] and labels[3] == "outside" and labels[4] == "nan"
    # Everyone 18 leaves one of the two user groups empty (four with quartiles).
    young = protected_scores(_clinic(1000, 3).assign(age=18), train, "y", groupers)
    assert young["protected_coverage"] == 0.5


def test_protected_target_gap_sees_a_broken_group_label_relation():
    train = _clinic(1000, 0)
    groupers = protected_groups(train, ["sex"])
    flipped = _clinic(1000, 1)
    flipped.loc[flipped["sex"] == "f", "y"] = "high"
    assert protected_scores(flipped, train, "y", groupers)["protected_target_gap"] > 0.5


def test_fidelity_scores_separate_real_like_from_broken_rows():
    real = _clinic(600, 0)
    close = fidelity_scores(_clinic(600, 1), real, ["sex", "y"], ["sex"], seed=0)
    broken = real.copy()
    broken["x"] = np.random.default_rng(0).permutation(broken["x"].to_numpy()) * 3 + 2
    broken["sex"] = "m"
    far = fidelity_scores(broken, real, ["sex", "y"], ["sex"], seed=0)
    assert close["fid_marginal"] < far["fid_marginal"]
    assert close["fid_association"] < far["fid_association"]
    assert close["fid_detection_auc"] < 0.65 < far["fid_detection_auc"]


# --- config -------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"objective_candidates": ["tstr_macro_f1", "nope"]}, "objective_candidates"),
        ({"patience": 0}, "patience"),
        ({"min_delta": -1}, "min_delta"),
        ({"refit_seeds": -1}, "refit_seeds"),
        ({"warm_start": {"path": "x"}}, "warm_start"),
        ({"search_params": {"tvae": "lr"}}, "search_params"),
        ({"objective": "auto", "n_bootstrap": 0}, "n_bootstrap"),
        ({"constraints": HPOConstraintsConfig(enforce="no")}, "enforce"),
    ],
)
def test_new_hpo_options_are_validated(overrides, message):
    with pytest.raises(ValueError, match=message):
        _validate_hpo(HPOConfig(**overrides))


def test_new_defaults():
    cfg = HPOConfig()
    assert cfg.objective == "auto" and cfg.pruner is None and cfg.tstr_seeds == 1
    assert HPOConstraintsConfig().min_category_coverage == 0.8
    assert HPOConstraintsConfig().enforce is True
    _validate_hpo(cfg)
