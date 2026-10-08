"""Unit tests for the explicit scope of resumable HPO artifacts."""

from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest

from synthdata.config import HPOConfig, HPOConstraintsConfig
from synthdata.generation.hpo import (
    SCREENS,
    TrialScore,
    build_hpo_eval_fn,
    default_best_params_path,
    default_storage_url,
    exact_match_rate,
    run_study,
    score_candidate,
    screen_violations,
)

pytestmark = pytest.mark.unit


def test_default_hpo_artifacts_live_inside_the_experiment_directory(tmp_path):
    experiment_dir = tmp_path / "output" / "dataset" / "synthetic_data" / "v2" / "exp-1"

    storage_url = default_storage_url(experiment_dir)

    assert storage_url == f"sqlite:///{experiment_dir / 'optuna_studies.db'}"
    assert default_best_params_path(experiment_dir) == experiment_dir / "hpo_best_params.json"
    assert Path(experiment_dir).exists()


def test_completed_hpo_keeps_best_and_latest_generator_checkpoints(tmp_path):
    workspace = tmp_path / "synthcity_workspace"
    output_dir = tmp_path / "output"
    values = [3.0, 1.0, 2.0]

    def objective(trial):
        for suffix in ("base_cache", "augmentation_cache"):
            path = workspace / (f"data_trial_{trial.number}_tvae_{suffix}_3.11.15_generator_0.bkp")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"checkpoint")
        return values[trial.number]

    run_study(
        "hpo_tvae",
        objective,
        HPOConfig(n_trials=3, timeout_seconds=None, objective="synthcity_composite"),
        output_dir,
        seed=0,
        drop_keys=(),
        checkpoint_workspace=workspace,
        checkpoint_plugin="tvae",
    )

    assert (workspace / "data_trial_1_tvae_base_cache_3.11.15_generator_0.bkp").exists()
    assert (workspace / "data_trial_1_tvae_augmentation_cache_3.11.15_generator_0.bkp").exists()
    assert (workspace / "data_trial_2_tvae_base_cache_3.11.15_generator_0.bkp").exists()
    assert (workspace / "data_trial_2_tvae_augmentation_cache_3.11.15_generator_0.bkp").exists()
    assert not (workspace / "data_trial_0_tvae_base_cache_3.11.15_generator_0.bkp").exists()
    assert not (workspace / "data_trial_0_tvae_augmentation_cache_3.11.15_generator_0.bkp").exists()


def test_partial_hpo_keeps_best_and_latest_generator_checkpoints(tmp_path):
    workspace = tmp_path / "synthcity_workspace"

    def objective(trial):
        path = workspace / f"data_trial_{trial.number}_tvae_cache_3.11.15_generator_0.bkp"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")
        if trial.number == 0:
            return 1.0
        raise optuna.TrialPruned()

    run_study(
        "hpo_tvae",
        objective,
        HPOConfig(n_trials=3, timeout_seconds=None, objective="synthcity_composite"),
        tmp_path / "output",
        seed=0,
        drop_keys=(),
        checkpoint_workspace=workspace,
        checkpoint_plugin="tvae",
    )

    assert (workspace / "data_trial_0_tvae_cache_3.11.15_generator_0.bkp").exists()
    assert (workspace / "data_trial_2_tvae_cache_3.11.15_generator_0.bkp").exists()
    assert not (workspace / "data_trial_1_tvae_cache_3.11.15_generator_0.bkp").exists()


def _real(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame(
        {
            "x": x,
            "color": rng.choice(["red", "blue", "green"], size=n),
            "target": np.where(x > 0.8, 2, np.where(x > -0.8, 0, 1)),
        }
    )


def test_screens_pass_for_fresh_real_rows():
    train, tuning, fresh = _real(seed=0), _real(seed=1), _real(seed=2)
    violations = screen_violations(
        fresh, train, tuning, "target", ["color"], HPOConstraintsConfig()
    )
    assert all(v <= 0 for v in violations.values()), violations


def test_screens_flag_copies_collapse_and_out_of_range():
    train, tuning = _real(seed=0), _real(seed=1)
    copies = train.sample(100, random_state=0)
    assert exact_match_rate(copies, train) == 1.0
    assert (
        screen_violations(copies, train, tuning, "target", ["color"], HPOConstraintsConfig())[
            "copies"
        ]
        > 0
    )

    collapsed = train.assign(color="red", target=0)
    violations = screen_violations(
        collapsed, train, tuning, "target", ["color"], HPOConstraintsConfig()
    )
    assert violations["missing_classes"] == 2
    assert violations["category_coverage"] > 0

    wild = train.assign(x=train["x"] * 100)
    assert (
        screen_violations(wild, train, tuning, "target", ["color"], HPOConstraintsConfig())[
            "out_of_range"
        ]
        > 0
    )


def test_screens_can_be_turned_off():
    train = _real()
    off = HPOConstraintsConfig(copy_margin=None, min_category_coverage=None, max_out_of_range=None)
    violations = screen_violations(train.assign(target=0), train, train, "target", ["color"], off)
    assert all(v == -1.0 for v in violations.values())


def test_exact_match_ignores_int_float_spelling():
    a = pd.DataFrame({"x": [1, 2], "c": ["a", "b"]})
    b = pd.DataFrame({"x": [1.0, 3.0], "c": ["a", "b"]})
    assert exact_match_rate(a, b) == 0.5


def test_tstr_objective_scores_tuning_and_logs_the_ceiling():
    train, tuning = _real(seed=0), _real(seed=1)
    eval_fn = build_hpo_eval_fn(
        train, tuning, "target", ["color"], ["color"], True, [], HPOConfig(tstr_seeds=2), seed=0
    )
    good = eval_fn(_real(seed=2))
    noise = eval_fn(
        _real(seed=3).assign(target=lambda d: np.random.default_rng(0).permutation(d.target))
    )
    assert good.value == good.attrs["tstr_macro_f1"] > noise.value
    assert 0 < good.attrs["tstr_macro_auprc"] <= 1
    assert good.attrs["trtr_macro_f1"] > 0.8
    assert len(good.constraints) == len(SCREENS)


def test_auprc_objective_is_a_config_switch():
    train, tuning = _real(seed=0), _real(seed=1)
    cfg = HPOConfig(objective="tstr_macro_auprc", tstr_seeds=1)
    score = build_hpo_eval_fn(train, tuning, "target", ["color"], ["color"], True, [], cfg, 0)(
        _real(seed=2)
    )
    assert score.value == score.attrs["tstr_macro_auprc"]


def test_tstr_objective_needs_a_categorical_target():
    train = _real()
    with pytest.raises(ValueError, match="categorical target"):
        build_hpo_eval_fn(train, train, "target", [], [], False, [], HPOConfig(), 0)


def test_best_params_come_from_feasible_trials_only(tmp_path):
    # Trial 0 scores highest but fails a screen; trial 2 is the best feasible one.
    outcomes = [(0.9, [1.0, -1.0, -1.0, -1.0]), (0.5, [-1.0] * 4), (0.7, [-1.0] * 4)]

    def objective(trial):
        trial.suggest_float("p", 0, 1)
        value, constraints = outcomes[trial.number]
        return score_candidate(trial, lambda _: TrialScore(value, constraints), None)

    best = run_study(
        "hpo_x", objective, HPOConfig(n_trials=3, timeout_seconds=None), tmp_path, 0, drop_keys=()
    )
    study = optuna.load_study(study_name="hpo_x", storage=default_storage_url(tmp_path))
    assert best == study.trials[2].params
    assert study.trials[0].user_attrs["constraints"][0] == 1.0


def test_crashed_trials_are_failed_and_infeasible_studies_fall_back(tmp_path):
    def objective(trial):
        trial.suggest_float("p", 0, 1)
        if trial.number == 0:
            raise RuntimeError("generator crashed")
        return score_candidate(trial, lambda _: TrialScore(0.5, [1.0, -1.0, -1.0, -1.0]), None)

    best = run_study(
        "hpo_y", objective, HPOConfig(n_trials=2, timeout_seconds=None), tmp_path, 0, drop_keys=()
    )
    study = optuna.load_study(study_name="hpo_y", storage=default_storage_url(tmp_path))
    assert study.trials[0].state == optuna.trial.TrialState.FAIL
    assert best == {}
