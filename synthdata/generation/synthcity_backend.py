"""synthcity plugin fit/generate/hyperparameter-search glue."""

import inspect
from collections.abc import Callable

import optuna
import pandas as pd
import torch

from synthdata.config import HPOConfig
from synthdata.utils import get_logger

logger = get_logger(__name__)

#: Search-space choices left out of HPO because they make runs irreproducible.
#: pgmpy's PC ("pc") and hill-climbing ("hillclimb") structure searches return
#: different DAGs for the same data on repeated runs, even with NumPy and
#: ``random`` seeded and PYTHONHASHSEED and n_jobs fixed (on the integration
#: fixture: PC gave 3 different DAGs in 6 runs, hill climbing 3 in 5). A
#: bayesian_network study then scored those trials differently each run and
#: could pick a different winner. tree_search (Chow-Liu), synthcity's default,
#: gave the same DAG every time.
HPO_EXCLUDED_CHOICES: dict[str, dict[str, frozenset]] = {
    "bayesian_network": {"struct_learning_search_method": frozenset({"pc", "hillclimb"})},
}


class _TrialWithoutChoices:
    """Optuna trial proxy that drops excluded categorical choices.

    synthcity's ``sample_hyperparameters_optuna`` reads the plugin's own search
    space and calls ``trial.suggest_categorical`` per parameter; filtering here
    keeps the rest of that space as synthcity defines it.
    """

    def __init__(self, trial: optuna.Trial, excluded: dict[str, frozenset]):
        self._trial = trial
        self._excluded = excluded

    def suggest_categorical(self, name, choices):
        drop = self._excluded.get(name, frozenset())
        return self._trial.suggest_categorical(name, [c for c in choices if c not in drop])

    def __getattr__(self, attr):
        return getattr(self._trial, attr)


def make_loader(
    df: pd.DataFrame,
    target_column: str,
    sensitive_features: list,
    random_state: int = 0,
    fairness_column: str | None = None,
):
    from synthcity.plugins.core.dataloader import GenericDataLoader

    kwargs = dict(
        target_column=target_column,
        sensitive_features=sensitive_features,
        random_state=random_state,
    )
    if fairness_column:
        kwargs["fairness_column"] = fairness_column
    return GenericDataLoader(df, **kwargs)


def get_plugin_class(name: str):
    from synthcity.plugins import Plugins

    return Plugins().get_type(name)


def plugin_accepts(name: str, param_name: str) -> bool:
    from synthcity.plugins.core.plugin import Plugin

    params = inspect.signature(get_plugin_class(name).__init__).parameters
    if param_name in params:
        return True
    # Plugins such as marginal_distributions take only **kwargs and forward
    # them to the base Plugin, which accepts workspace, device and random_state.
    forwards = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    return forwards and param_name in inspect.signature(Plugin.__init__).parameters


def fit_generate(
    name: str,
    params: dict,
    train_loader,
    n_samples: int,
    random_state: int = 42,
    workspace: str | None = None,
    device: str | None = None,
) -> pd.DataFrame:
    from pathlib import Path

    from synthcity.plugins import Plugins

    plugin_kwargs = dict(params)
    if workspace is not None and plugin_accepts(name, "workspace"):
        plugin_kwargs["workspace"] = Path(workspace)
    # Seed training as well as sampling; plugins otherwise train with their
    # default random_state (0) whatever seed the run uses.
    if "random_state" not in plugin_kwargs and plugin_accepts(name, "random_state"):
        plugin_kwargs["random_state"] = random_state
    if device is not None and "device" not in plugin_kwargs and plugin_accepts(name, "device"):
        plugin_kwargs["device"] = torch.device(device)

    model = Plugins().get(name, **plugin_kwargs)
    model.fit(train_loader)
    return model.generate(count=n_samples, random_state=random_state).dataframe()


def build_synthcity_objective(
    name: str,
    train_loader,
    hpo_cfg: HPOConfig,
    seed: int,
    eval_fn: Callable[[pd.DataFrame], float],
    n_samples: int,
    workspace: str | None = None,
    device: str = "cpu",
):
    """Build an Optuna objective for a synthcity plugin's native hyperparameter space.

    Each trial samples from the plugin's own ``sample_hyperparameters_optuna``,
    caps ``n_iter`` for speed (only if the plugin exposes it), forces CPU for
    MPS (which lacks the float64 support synthcity's metrics need internally)
    but otherwise uses ``device``, fits the plugin on ``train_loader`` and
    scores ``n_samples`` generated rows with ``eval_fn`` (lower is better).
    ``eval_fn`` compares against the tuning split, so the search never sees
    the test split.
    """
    plugin_cls = get_plugin_class(name)
    accepts_device = plugin_accepts(name, "device")
    accepts_iter = plugin_accepts(name, "n_iter")
    iter_cap = hpo_cfg.model_iter_caps.get(name, hpo_cfg.n_iter_cap)
    trial_device = "cpu" if device == "mps" else device

    def objective(trial: optuna.Trial) -> float:
        excluded = HPO_EXCLUDED_CHOICES.get(name)
        params = plugin_cls.sample_hyperparameters_optuna(
            _TrialWithoutChoices(trial, excluded) if excluded else trial
        )
        if accepts_iter:
            params["n_iter"] = min(params.get("n_iter", iter_cap), iter_cap)
        params["random_state"] = seed
        if accepts_device:
            params["device"] = torch.device(trial_device)

        try:
            synthetic = fit_generate(
                name, params, train_loader, n_samples, seed, workspace=workspace
            )
            return eval_fn(synthetic)
        except (ValueError, RuntimeError) as exc:
            logger.warning("[%s] trial %d failed: %s", name, trial.number, exc)
            raise optuna.TrialPruned() from exc

    return objective
