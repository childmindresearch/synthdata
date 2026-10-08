"""Unit tests for synthdata.generation.synthcity_backend's HPO search space."""

import optuna
import pytest

from synthdata.generation.synthcity_backend import (
    HPO_EXCLUDED_CHOICES,
    _SearchSpaceTrial,
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
                _SearchSpaceTrial(trial, HPO_EXCLUDED_CHOICES[name])
            )
        )
        return 0.0

    study.optimize(objective, n_trials=n_trials)
    return sampled


def test_bayesian_network_search_only_tries_tree_search():
    pytest.importorskip("synthcity.plugins")
    methods = {
        params["struct_learning_search_method"] for params in _sampled("bayesian_network", 30)
    }
    assert methods == {"tree_search"}


def test_other_trial_attributes_pass_through():
    study = optuna.create_study()
    trial = study.ask()
    proxy = _SearchSpaceTrial(trial, {"x": frozenset({"b"})})
    assert proxy.number == trial.number
    assert proxy.suggest_categorical("x", ["a", "b"]) == "a"
    assert proxy.suggest_int("n", 1, 1) == 1


def test_epoch_range_replaces_the_n_iter_range():
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    values = []
    for _ in range(20):
        proxy = _SearchSpaceTrial(study.ask(), epochs=(5, 15, 5))
        values.append(proxy.suggest_int("n_iter", 100, 1000, 100))
        assert proxy.suggest_int("other", 100, 100, 100) == 100
    assert set(values) <= {5, 10, 15} and len(set(values)) > 1


def _objective_params(name: str, hpo_cfg) -> dict:
    """Params one trial of ``name``'s objective passes to fit_generate."""
    from unittest import mock

    from synthdata.generation import synthcity_backend as sc

    captured = {}

    def fake_fit(name, params, *args, **kwargs):
        captured.update(params)
        return "synthetic"

    objective = sc.build_synthcity_objective(
        name, None, hpo_cfg, 0, lambda df: None, 10, device="cpu"
    )
    trial = optuna.create_study().ask()
    with (
        mock.patch.object(sc, "fit_generate", fake_fit),
        mock.patch.object(sc, "score_candidate", lambda *a: 0.0),
    ):
        objective(trial)
    return captured | {"_trial_params": trial.params}


@pytest.mark.parametrize(
    ("name", "low", "high"), [("ctgan", 100, 1000), ("tvae", 100, 500), ("adsgan", 100, 1000)]
)
def test_epochs_are_searched_not_capped(name, low, high):
    pytest.importorskip("synthcity.plugins")
    from synthdata.config import HPOConfig

    params = _objective_params(name, HPOConfig(pruner=None))
    assert low <= params["n_iter"] <= high
    # The searched value is the stored trial param, so the final fit reuses it.
    assert params["_trial_params"]["n_iter"] == params["n_iter"]
    assert "patience_metric" not in params


def test_config_epoch_range_and_pruning_metric():
    pytest.importorskip("synthcity.plugins")
    from synthdata.config import HPOConfig

    params = _objective_params("ctgan", HPOConfig(epoch_ranges={"ctgan": [2, 4, 1]}))
    assert 2 <= params["n_iter"] <= 4
    assert type(params["patience_metric"]).__name__ == "PruningPatienceMetric"


def test_pruning_metric_reports_and_prunes():
    pytest.importorskip("synthcity.plugins")
    from unittest import mock

    from synthcity.metrics.weighted_metrics import WeightedMetrics

    from synthdata.generation.synthcity_backend import _pruning_patience_metric_class

    trial = mock.Mock()
    trial.should_prune.side_effect = [False, True]
    metric = _pruning_patience_metric_class()(trial)
    with mock.patch.object(WeightedMetrics, "evaluate", return_value=0.7):
        assert metric.evaluate(None, None) == 0.7
        with pytest.raises(optuna.TrialPruned):
            metric.evaluate(None, None)
    # detection is "lower is better"; Optuna gets it negated (higher is better).
    trial.report.assert_has_calls([mock.call(-0.7, 1), mock.call(-0.7, 2)])


def test_plugin_accepts_follows_kwargs_to_the_base_plugin():
    from synthdata.generation.synthcity_backend import plugin_accepts

    assert plugin_accepts("marginal_distributions", "workspace")
    assert not plugin_accepts("marginal_distributions", "n_iter")


def test_fit_generate_seeds_training_not_only_sampling(mocker):
    from synthcity.plugins import Plugins

    from synthdata.generation import synthcity_backend as sc

    get = mocker.patch.object(Plugins, "get")
    sc.fit_generate("ctgan", {}, mocker.Mock(), 5, random_state=11)
    _, kwargs = get.call_args
    assert kwargs["random_state"] == 11


def test_fit_generate_conditions_ddpm_on_a_categorical_target(mocker):
    from synthcity.plugins import Plugins

    from synthdata.generation import synthcity_backend as sc

    get = mocker.patch.object(Plugins, "get")
    sc.fit_generate("ddpm", {}, mocker.Mock(), 5, classification=True)
    assert get.call_args.kwargs["is_classification"] is True
    sc.fit_generate("ddpm", {}, mocker.Mock(), 5)
    assert "is_classification" not in get.call_args.kwargs
    # Plugins without the argument never receive it.
    sc.fit_generate("ctgan", {}, mocker.Mock(), 5, classification=True)
    assert "is_classification" not in get.call_args.kwargs
