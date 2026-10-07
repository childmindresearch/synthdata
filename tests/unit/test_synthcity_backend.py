"""Unit tests for synthdata.generation.synthcity_backend's HPO search space."""

import optuna
import pytest

from synthdata.generation.synthcity_backend import (
    HPO_EXCLUDED_CHOICES,
    _TrialWithoutChoices,
    get_plugin_class,
)

pytestmark = pytest.mark.unit


def _sampled(name: str, n_trials: int) -> list[dict]:
    plugin_cls = get_plugin_class(name)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    sampled = []

    def objective(trial):
        sampled.append(
            plugin_cls.sample_hyperparameters_optuna(
                _TrialWithoutChoices(trial, HPO_EXCLUDED_CHOICES[name])
            )
        )
        return 0.0

    study.optimize(objective, n_trials=n_trials)
    return sampled


def test_bayesian_network_search_never_tries_pc():
    pytest.importorskip("synthcity.plugins")
    methods = {
        params["struct_learning_search_method"] for params in _sampled("bayesian_network", 30)
    }
    assert methods == {"hillclimb", "tree_search"}


def test_other_trial_attributes_pass_through():
    study = optuna.create_study()
    trial = study.ask()
    proxy = _TrialWithoutChoices(trial, {"x": frozenset({"b"})})
    assert proxy.number == trial.number
    assert proxy.suggest_categorical("x", ["a", "b"]) == "a"
    assert proxy.suggest_int("n", 1, 1) == 1
