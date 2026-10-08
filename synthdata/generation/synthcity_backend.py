"""synthcity plugin fit/generate/hyperparameter-search glue."""

import inspect
from collections.abc import Callable
from pathlib import Path

import optuna
import pandas as pd
import torch

from synthdata.config import HPOConfig
from synthdata.generation.hpo import TrialScore, epoch_range, score_candidate
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


#: Hyperparameters held at a fixed value instead of searched. synthcity's ARF
#: space searches ``delta`` over the integers 0-50, but arfpy asserts
#: ``0 <= delta <= 0.5`` (stop once the discriminator's OOB accuracy is below
#: 0.5 + delta), so every draw above 0 crashes the trial; and the plugin types
#: it ``int``, so a float range would be truncated to 0 anyway. 0 is the
#: default of arfpy and of the reference R package (Watson et al. 2023).
HPO_FIXED_PARAMS: dict[str, dict[str, object]] = {"arf": {"delta": 0}}

#: Searched epochs for plugins that take ``n_iter`` but leave it out of their
#: own search space: ADS-GAN gets CTGAN's range (the same conditional GAN
#: family); its default of 10000 epochs with early stopping is far too long
#: for a search.
EPOCH_RANGE_DEFAULTS: dict[str, tuple[int, int, int]] = {"adsgan": (100, 1000, 100)}


def _pruning_patience_metric_class():
    from synthcity.metrics.weighted_metrics import WeightedMetrics

    class PruningPatienceMetric(WeightedMetrics):
        """synthcity's default GAN early-stopping metric, reported to Optuna.

        CTGAN and ADS-GAN evaluate their ``patience_metric`` (detection by an
        MLP on a held-back slice of the training rows, lower is better) every
        ``n_iter_print`` epochs to stop early. This subclass keeps that metric
        and that early stopping, reports the negated score as the trial's
        intermediate value and raises ``TrialPruned`` when Optuna's pruner
        says so. It is the only per-epoch hook synthcity's GANs offer; the
        VAEs and PATE-GAN have none and are not pruned.
        """

        def __init__(self, trial: optuna.Trial, workspace=None):
            super().__init__(
                metrics=[("detection", "detection_mlp")],
                weights=[1],
                **({"workspace": Path(workspace)} if workspace is not None else {}),
            )
            self._trial = trial
            self._step = 0

        def evaluate(self, X_gt, X_syn):
            score = super().evaluate(X_gt, X_syn)
            self._step += 1
            self._trial.report(-score if self.direction() == "minimize" else score, self._step)
            if self._trial.should_prune():
                raise optuna.TrialPruned(f"pruned at check {self._step}")
            return score

    return PruningPatienceMetric


class _SearchSpaceTrial:
    """Optuna trial proxy that adjusts synthcity's own search space.

    synthcity's ``sample_hyperparameters_optuna`` reads the plugin's own search
    space and calls ``trial.suggest_*`` per parameter. This proxy drops
    excluded categorical choices, returns ``fixed`` values without searching
    them, replaces the ``n_iter`` range with ``epochs`` when given, and keeps
    the rest of the space as synthcity defines it.
    """

    def __init__(
        self,
        trial: optuna.Trial,
        excluded: dict[str, frozenset] | None = None,
        epochs: tuple[int, int, int] | None = None,
        fixed: dict[str, object] | None = None,
    ):
        self._trial = trial
        self._excluded = excluded or {}
        self._epochs = epochs
        self._fixed = fixed or {}

    def suggest_categorical(self, name, choices):
        if name in self._fixed:
            return self._fixed[name]
        drop = self._excluded.get(name, frozenset())
        return self._trial.suggest_categorical(name, [c for c in choices if c not in drop])

    def suggest_int(self, name, low, high, step=1, log=False):
        if name in self._fixed:
            return self._fixed[name]
        if name == "n_iter" and self._epochs is not None:
            low, high, step = self._epochs
        return self._trial.suggest_int(name, low, high, step=step, log=log)

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
    eval_fn: Callable[[pd.DataFrame], TrialScore],
    n_samples: int,
    workspace: str | None = None,
    device: str = "cpu",
):
    """Build an Optuna objective for a synthcity plugin's native hyperparameter space.

    Each trial samples from the plugin's own ``sample_hyperparameters_optuna``
    (training length included, see ``hpo.epoch_ranges``), forces CPU for
    MPS (which lacks the float64 support synthcity's metrics need internally)
    but otherwise uses ``device``, fits the plugin on ``train_loader`` and
    scores ``n_samples`` generated rows with ``eval_fn``.
    ``eval_fn`` compares against the tuning split, so the search never sees
    the test split.
    """
    plugin_cls = get_plugin_class(name)
    accepts_device = plugin_accepts(name, "device")
    accepts_iter = plugin_accepts(name, "n_iter")
    accepts_patience_metric = plugin_accepts(name, "patience_metric")
    native_iter = any(d.name == "n_iter" for d in plugin_cls.hyperparameter_space())
    default_range = None if native_iter else EPOCH_RANGE_DEFAULTS.get(name)
    epochs = epoch_range(hpo_cfg, name, default_range) if accepts_iter else None
    trial_device = "cpu" if device == "mps" else device

    def objective(trial: optuna.Trial) -> float:
        # Training length is searched, not capped: the plugin's own n_iter
        # range unless hpo.epoch_ranges (or EPOCH_RANGE_DEFAULTS) sets one.
        proxy = _SearchSpaceTrial(
            trial, HPO_EXCLUDED_CHOICES.get(name), epochs, HPO_FIXED_PARAMS.get(name)
        )
        params = plugin_cls.sample_hyperparameters_optuna(proxy)
        if epochs is not None and "n_iter" not in params:
            params["n_iter"] = proxy.suggest_int("n_iter", *epochs)
        params["random_state"] = seed
        if accepts_device:
            params["device"] = torch.device(trial_device)
        if accepts_patience_metric and hpo_cfg.pruner:
            params["patience_metric"] = _pruning_patience_metric_class()(trial, workspace)

        # A crash propagates and run_study marks the trial failed.
        synthetic = fit_generate(name, params, train_loader, n_samples, seed, workspace=workspace)
        return score_candidate(trial, eval_fn, synthetic)

    return objective
