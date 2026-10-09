"""Unit tests for synthdata.generation.synthcity_backend's HPO search space."""

import optuna
import pytest

from synthdata.generation.synthcity_backend import (
    HPO_EXCLUDED_CHOICES,
    HPO_FIXED_PARAMS,
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
                _SearchSpaceTrial(
                    trial, HPO_EXCLUDED_CHOICES.get(name), fixed=HPO_FIXED_PARAMS.get(name)
                )
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


def test_arf_search_holds_delta_at_zero_and_fits():
    """synthcity searches ARF's delta over 0-50; arfpy rejects anything above 0.5."""
    pytest.importorskip("arfpy")
    import numpy as np
    import pandas as pd

    from synthdata.generation.synthcity_backend import fit_generate, make_loader

    sampled = _sampled("arf", 20)
    assert {params["delta"] for params in sampled} == {0}
    assert len({params["num_trees"] for params in sampled}) > 1

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=60), "c": rng.integers(0, 3, 60)})
    df["y"] = (df["x"] > 0).astype(int)
    params = sampled[0] | {"num_trees": 10, "max_iters": 1, "verbose": False}
    loader = make_loader(df, "y", [])
    syn = fit_generate("arf", params, loader, 30, random_state=1, discrete_columns=["c", "y"])
    assert syn.shape == (30, 3)
    # A declared-continuous column with few distinct values stays continuous.
    syn = fit_generate("arf", params, loader, 30, random_state=1, discrete_columns=["y"])
    assert syn.shape == (30, 3)


def test_fixed_params_skip_the_search():
    study = optuna.create_study()
    proxy = _SearchSpaceTrial(study.ask(), fixed={"d": 0, "k": "a"})
    assert proxy.suggest_int("d", 0, 50, 2) == 0
    assert proxy.suggest_categorical("k", ["a", "b"]) == "a"
    assert proxy.params == {}


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
        name, None, hpo_cfg, 0, lambda df: None, 10, device="cpu", discrete_columns=[]
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
    sc.fit_generate("ctgan", {}, mocker.Mock(), 5, random_state=11, discrete_columns=[])
    _, kwargs = get.call_args
    assert kwargs["random_state"] == 11


def test_fit_generate_conditions_ddpm_on_a_categorical_target(mocker):
    from synthcity.plugins import Plugins

    from synthdata.generation import synthcity_backend as sc

    get = mocker.patch.object(Plugins, "get")
    sc.fit_generate("ddpm", {}, mocker.Mock(), 5, classification=True, discrete_columns=[])
    assert get.call_args.kwargs["is_classification"] is True
    sc.fit_generate("ddpm", {}, mocker.Mock(), 5, discrete_columns=[])
    assert "is_classification" not in get.call_args.kwargs
    # Plugins without the argument never receive it.
    sc.fit_generate("ctgan", {}, mocker.Mock(), 5, classification=True, discrete_columns=[])
    assert "is_classification" not in get.call_args.kwargs


def test_fit_generate_uses_the_declared_column_types(mocker):
    """synthcity sees the schema's types, not its own distinct-value guess."""
    import pandas as pd
    from synthcity.plugins import Plugins
    from synthcity.utils.dataframe import discrete_columns

    from synthdata.generation import synthcity_backend as sc

    df = pd.DataFrame({"few": [0.5, 1.5] * 10, "code": range(20), "y": [0, 1] * 10})
    seen = {}

    def fit(loader):
        seen["discrete"] = discrete_columns(df)

    model = mocker.Mock()
    model.fit.side_effect = fit
    mocker.patch.object(Plugins, "get", return_value=model)
    sc.fit_generate("ctgan", {}, mocker.Mock(), 5, discrete_columns=["code", "y"])
    assert seen["discrete"] == ["code", "y"]
    # Outside the call synthcity falls back to its own guess.
    assert discrete_columns(df) == ["few", "y"]


def test_ddpm_searches_the_reference_mlp_size():
    pytest.importorskip("synthcity.plugins")
    from synthcity.plugins import Plugins

    from synthdata.generation.synthcity_backend import get_plugin_class

    names = {d.name for d in get_plugin_class("ddpm").hyperparameter_space()}
    assert {"n_layers_hidden", "n_units_hidden"} <= names
    plugin = Plugins().get("ddpm", n_layers_hidden=4, n_units_hidden=512, n_iter=1)
    assert plugin.model.model_params == {
        "n_layers_hidden": 4,
        "n_units_hidden": 512,
        "dropout": 0.0,
    }


def test_ddpm_slice_normalizer_keeps_small_slices_finite():
    """TabDDPM's per-feature log-normalizer must not cancel to -inf with many features."""
    import torch
    from synthcity.plugins.core.models.tabular_ddpm.utils import sliced_logsumexp

    x = torch.zeros(1, 2000)
    x[:, -2:] = -30.0  # the last feature's two classes are far below the running total
    slices = torch.arange(0, 2001, 2)
    out = sliced_logsumexp(x, slices)
    expected = torch.cat(
        [
            torch.logsumexp(x[:, a:b], 1, keepdim=True).expand(-1, b - a)
            for a, b in zip(slices[:-1], slices[1:], strict=True)
        ],
        dim=1,
    )
    assert torch.isfinite(out).all()
    assert torch.allclose(out, expected, atol=1e-5)
