"""Unit tests for the HPO headroom gate (``hpo.skip_without_headroom``)."""

import numpy as np
import optuna
import pandas as pd
import pytest

from synthdata import dashboard
from synthdata.config import HPOConfig
from synthdata.evaluation.tstr import paired_gap, target_classes, tstr_scores
from synthdata.generation.hpo import (
    HEADROOM_SKIP_ATTR,
    SCREENS,
    defaults_headroom_skip,
    headroom_skip,
    objective_headroom_skip,
    run_study,
)

pytestmark = pytest.mark.unit

FEASIBLE = [-1.0] * len(SCREENS)
HEADROOM = {"tstr_macro_auprc": {"trtr": 0.6, "chance": 0.3, "se": 0.02, "z": 15.0}}


def _hpo(**overrides) -> HPOConfig:
    defaults = dict(
        objective="tstr_macro_auprc",
        n_trials=6,
        timeout_seconds=None,
        n_startup_trials=3,
        patience=None,
        refit_seeds=0,
    )
    return HPOConfig(**(defaults | overrides))


def _frame(n: int, seed: int, signal: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.where(x * signal + rng.normal(size=n) > 0.5, "high", "low")
    return pd.DataFrame({"x": x, "z": rng.normal(size=n), "y": y})


def test_paired_gap_is_zero_for_the_same_fit_and_tighter_than_unpaired():
    real, tuning = _frame(400, 0, 2.0), _frame(300, 1, 2.0)
    classes = target_classes(real["y"])

    def scores(df):
        return tstr_scores(df, tuning, "y", [], classes, [0], n_bootstrap=100)

    trtr, again, weaker = scores(real), scores(real), scores(_frame(400, 2, 0.7))
    assert paired_gap(trtr, again, "macro_auprc") == (0.0, 0.0)
    gap, se = paired_gap(trtr, weaker, "macro_auprc")
    unpaired = np.hypot(trtr.boot_se["macro_auprc"], weaker.boot_se["macro_auprc"])
    assert gap > 0 and 0 < se < unpaired


def _defaults_trial(z: float, feasible: bool = True):
    return optuna.trial.create_trial(
        value=0.5,
        user_attrs={
            "constraints": FEASIBLE if feasible else [1.0] * len(SCREENS),
            "ceiling_gap": {"tstr_macro_auprc": {"gap": 0.01 * z, "se": 0.01, "z": z}},
        },
    )


def test_defaults_gate_skips_only_without_room_below_the_ceiling():
    cfg = _hpo()
    skip = defaults_headroom_skip(cfg, HEADROOM, _defaults_trial(z=0.5))
    assert skip is not None and skip["reason"] == "defaults reach the ceiling"
    # Defaults above the ceiling (negative gap) leave no room either.
    assert defaults_headroom_skip(cfg, HEADROOM, _defaults_trial(z=-1.0)) is not None
    assert defaults_headroom_skip(cfg, HEADROOM, _defaults_trial(z=3.0)) is None
    # Defaults failing an enforced screen: the search may find a feasible config.
    assert defaults_headroom_skip(cfg, HEADROOM, _defaults_trial(z=0.5, feasible=False)) is None
    assert (
        defaults_headroom_skip(_hpo(skip_without_headroom=False), HEADROOM, _defaults_trial(0.5))
        is None
    )
    # No bootstrap (NaN z) is treated as headroom.
    assert defaults_headroom_skip(cfg, HEADROOM, _defaults_trial(z=float("nan"))) is None


def test_auto_skips_only_when_no_candidate_with_headroom_is_left():
    cfg = _hpo(objective="auto", objective_candidates=["tstr_macro_auprc", "tstr_macro_auroc"])
    headroom = HEADROOM | {"tstr_macro_auroc": {"trtr": 0.7, "chance": 0.5, "se": 0.02, "z": 10}}
    trial = _defaults_trial(z=0.5)
    # AUROC has headroom over chance but its gap is not logged: no decision.
    assert defaults_headroom_skip(cfg, headroom, trial) is None
    trial.user_attrs["ceiling_gap"]["tstr_macro_auroc"] = {"gap": 0.05, "se": 0.01, "z": 5.0}
    assert defaults_headroom_skip(cfg, headroom, trial) is None
    trial.user_attrs["ceiling_gap"]["tstr_macro_auroc"] = {"gap": 0.0, "se": 0.01, "z": 0.0}
    assert defaults_headroom_skip(cfg, headroom, trial) is not None


def test_objective_gate_skips_when_real_data_cannot_beat_chance():
    flat = {"tstr_macro_auprc": {"trtr": 0.31, "chance": 0.3, "se": 0.02, "z": 0.5}}
    assert objective_headroom_skip(_hpo(), flat)["reason"] == "objective has no headroom"
    assert objective_headroom_skip(_hpo(), HEADROOM) is None
    assert objective_headroom_skip(_hpo(objective="synthcity_composite"), flat) is None


def _objective(gap_z):
    def objective(trial):
        x = trial.suggest_float("x", 0, 1)
        trial.set_user_attr("constraints", FEASIBLE)
        trial.set_user_attr(
            "ceiling_gap", {"tstr_macro_auprc": {"gap": 0.01 * gap_z, "se": 0.01, "z": gap_z}}
        )
        return x

    return objective


@pytest.mark.parametrize("gap_z, n_trials", [(0.5, 1), (5.0, 6)])
def test_run_study_stops_after_the_defaults_when_they_reach_the_ceiling(tmp_path, gap_z, n_trials):
    cfg = _hpo()
    best = run_study(
        "hpo_g",
        _objective(gap_z),
        cfg,
        tmp_path,
        seed=0,
        default_params={"x": 0.2},
        headroom=HEADROOM,
    )
    study = optuna.load_study(
        study_name="hpo_g", storage=f"sqlite:///{tmp_path / 'optuna_studies.db'}"
    )
    assert len(study.trials) == n_trials
    skip = headroom_skip(cfg, tmp_path, "hpo_g")
    assert (skip is not None) == (n_trials == 1)
    if skip:
        assert best == {} and study.user_attrs[HEADROOM_SKIP_ATTR] == skip
        # A resumed run keeps the decision without new trials.
        assert run_study("hpo_g", _objective(gap_z), cfg, tmp_path, seed=0, headroom=HEADROOM) == {}
        assert len(study.trials) == 1


def test_run_study_auto_gates_on_the_probe_defaults(tmp_path):
    cfg = _hpo(objective="auto", objective_candidates=["tstr_macro_auprc"])
    assert (
        run_study(
            "hpo_a",
            _objective(0.0),
            cfg,
            tmp_path,
            seed=0,
            default_params={"x": 0.2},
            headroom=HEADROOM,
        )
        == {}
    )
    storage = f"sqlite:///{tmp_path / 'optuna_studies.db'}"
    assert len(optuna.load_study(study_name="hpo_a_probe", storage=storage).trials) == 1
    assert headroom_skip(cfg, tmp_path, "hpo_a")["reason"] == "defaults reach the ceiling"


def test_run_study_skips_without_trials_when_the_objective_has_no_headroom(tmp_path):
    flat = {"tstr_macro_auprc": {"trtr": 0.31, "chance": 0.3, "se": 0.02, "z": 0.5}}
    assert run_study("hpo_f", _objective(5.0), _hpo(), tmp_path, seed=0, headroom=flat) == {}
    study = optuna.load_study(
        study_name="hpo_f", storage=f"sqlite:///{tmp_path / 'optuna_studies.db'}"
    )
    assert study.trials == [] and study.user_attrs[HEADROOM_SKIP_ATTR]
    assert headroom_skip(_hpo(), tmp_path, "hpo_missing") is None


def test_a_skipped_search_is_logged_and_read_by_the_dashboard(tmp_path, caplog):
    caplog.set_level("INFO", logger="synthdata")
    run_study(
        "hpo_ctgan",
        _objective(0.0),
        _hpo(),
        tmp_path,
        seed=0,
        default_params={"x": 0.2},
        headroom=HEADROOM,
    )
    line = next(r.getMessage() for r in caplog.records if "skipping the search" in r.getMessage())
    assert line.startswith("[hpo_ctgan] skipping the search: defaults reach the ceiling;")
    assert "auprc: +0.0000 below the real-data ceiling (0.0 paired SEs)" in line
    stop = next(pattern.match(line) for pattern, _ in dashboard._HPO_STOPS if pattern.match(line))
    assert stop["reason"] == "defaults reach the ceiling"
    assert dashboard._NONE_PASSED.match("[ctgan_hpo] no headroom: the search was skipped (x)")
