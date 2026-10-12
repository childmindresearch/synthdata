"""synthcity plugin fit/generate/hyperparameter-search glue."""

import inspect
from collections.abc import Callable
from pathlib import Path

import optuna
import pandas as pd
import torch

from synthdata import gpu_slots
from synthdata.config import HPOConfig
from synthdata.generation import class_quota
from synthdata.generation.hpo import TrialScore, epoch_range, score_candidate, trial_seed
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
#: ARF also holds ``early_stop`` on, the default of arfpy and of the R
#: package: with it off every adversarial round refits the forest and
#: re-samples from it, and one 40-tree, 5-round trial ran over 11 h on the
#: 682-column sim data. PATE-GAN holds ``generator_n_iter`` (GAN epochs per
#: teacher round, searched 1-100) at the plugin default of 10: each outer
#: iteration trains the GAN for that many epochs, so one draw of 91 made a
#: 10-iteration trial take 7.5 h; ``hpo.epoch_ranges.pategan`` sets the rounds.
HPO_FIXED_PARAMS: dict[str, dict[str, object]] = {
    "arf": {"delta": 0, "early_stop": True},
    "pategan": {"generator_n_iter": 10},
}

#: Narrowed ``[low, high, step]`` integer ranges. ARF's trees stop at 50 (the
#: plugin searches 10-100; arfpy's default is 30, the R package's 10).
HPO_PARAM_RANGES: dict[str, dict[str, tuple[int, int, int]]] = {
    "arf": {"num_trees": (10, 50, 10)},
}

#: Searched epochs for plugins that take ``n_iter`` but leave it out of their
#: own search space: ADS-GAN gets CTGAN's range (the same conditional GAN
#: family); its default of 10000 epochs with early stopping is far too long
#: for a search.
EPOCH_RANGE_DEFAULTS: dict[str, tuple[int, int, int]] = {"adsgan": (100, 1000, 100)}


def untuned_params(name: str, hpo_cfg) -> dict:
    """Params of the untuned (default) model: the library's, with training
    length capped at the top of the searched range.

    Library defaults can be far longer than anything the search tries (ADS-GAN
    10000 epochs, about 50 h on the 682-column sim data; CTGAN 2000). Capping
    them at ``hpo.epoch_ranges`` (or ``EPOCH_RANGE_DEFAULTS``) keeps the
    default model comparable with the tuned one and its fit time bounded. A
    library default already below the cap is kept.
    """
    if not plugin_accepts(name, "n_iter"):
        return {}
    from synthdata.generation.hpo import epoch_range

    bounds = epoch_range(hpo_cfg, name, EPOCH_RANGE_DEFAULTS.get(name))
    default = inspect.signature(get_plugin_class(name).__init__).parameters.get("n_iter")
    if bounds is None or default is None or default.default is inspect.Parameter.empty:
        return {}
    return {"n_iter": bounds[1]} if bounds[1] < default.default else {}


def library_defaults(name: str, hpo_cfg) -> dict:
    """The plugin's constructor defaults, training length capped as for the untuned model."""
    params = {
        key: p.default
        for key, p in inspect.signature(get_plugin_class(name).__init__).parameters.items()
        if p.default is not inspect.Parameter.empty
    }
    params.update(untuned_params(name, hpo_cfg))
    return params


def default_trial_params(name: str, hpo_cfg) -> dict:
    """The library defaults that fall inside the searched space, as trial params.

    Enqueued as the first trial (``hpo.enqueue_defaults``), so the search
    starts from the untuned model; a default outside the space (an excluded
    choice, a value off the integer grid) is left for the sampler to draw.
    """
    defaults = library_defaults(name, hpo_cfg)
    plugin_cls = get_plugin_class(name)
    native_iter = any(d.name == "n_iter" for d in plugin_cls.hyperparameter_space())
    epochs = (
        epoch_range(hpo_cfg, name, None if native_iter else EPOCH_RANGE_DEFAULTS.get(name))
        if plugin_accepts(name, "n_iter")
        else None
    )
    fixed = HPO_FIXED_PARAMS.get(name, {})
    ranges = HPO_PARAM_RANGES.get(name, {})
    excluded = HPO_EXCLUDED_CHOICES.get(name, {})
    searched = hpo_cfg.search_params.get(name)
    trial_params = {}
    spaces = list(plugin_cls.hyperparameter_space())
    names = [d.name for d in spaces] + (["n_iter"] if epochs and not native_iter else [])
    for dist_name in names:
        if dist_name in fixed or dist_name not in defaults:
            continue
        if searched is not None and dist_name not in searched:
            continue
        value = defaults[dist_name]
        dist = next((d for d in spaces if d.name == dist_name), None)
        if dist_name == "n_iter" and epochs is not None:
            low, high, step = epochs
            if isinstance(value, int) and low <= value <= high and (value - low) % step == 0:
                trial_params[dist_name] = value
        elif hasattr(dist, "choices"):
            if value in dist.choices and value not in excluded.get(dist_name, frozenset()):
                trial_params[dist_name] = value
        elif dist is not None:
            low, high = ranges.get(dist_name, (dist.low, dist.high))[:2]
            step = ranges.get(dist_name, (None, None, getattr(dist, "step", None)))[2]
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            if low <= value <= high and (
                not step or isinstance(value, float) or (value - low) % step == 0
            ):
                trial_params[dist_name] = value
    return trial_params


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
    them, replaces integer ranges with ``ranges`` and the ``n_iter`` range with
    ``epochs`` when given, and keeps
    the rest of the space as synthcity defines it. With ``searched`` (a set of
    names), any other parameter takes its ``defaults`` value instead of being
    searched (``hpo.search_params``).
    """

    def __init__(
        self,
        trial: optuna.Trial,
        excluded: dict[str, frozenset] | None = None,
        epochs: tuple[int, int, int] | None = None,
        fixed: dict[str, object] | None = None,
        ranges: dict[str, tuple[int, int, int]] | None = None,
        searched: set | None = None,
        defaults: dict | None = None,
    ):
        self._trial = trial
        self._searched = searched
        self._defaults = defaults or {}
        self._ranges = ranges or {}
        self._excluded = excluded or {}
        self._epochs = epochs
        self._fixed = fixed or {}

    def _held(self, name) -> bool:
        return name in self._fixed or (
            self._searched is not None and name not in self._searched and name in self._defaults
        )

    def _held_value(self, name):
        return self._fixed[name] if name in self._fixed else self._defaults[name]

    def suggest_categorical(self, name, choices):
        if self._held(name):
            return self._held_value(name)
        drop = self._excluded.get(name, frozenset())
        return self._trial.suggest_categorical(name, [c for c in choices if c not in drop])

    def suggest_int(self, name, low, high, step=1, log=False):
        if self._held(name):
            return self._held_value(name)
        if name == "n_iter" and self._epochs is not None:
            low, high, step = self._epochs
        elif name in self._ranges:
            low, high, step = self._ranges[name]
        return self._trial.suggest_int(name, low, high, step=step, log=log)

    def suggest_float(self, name, low, high, *, step=None, log=False):
        if self._held(name):
            return self._held_value(name)
        return self._trial.suggest_float(name, low, high, step=step, log=log)

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
    classification: bool = False,
    class_prior: pd.Series | None = None,
    *,
    discrete_columns: list,
    kind: str = "fit",
) -> pd.DataFrame:
    """Fit plugin ``name`` on ``train_loader`` and generate ``n_samples`` rows.

    With models training in parallel, the fit holds a GPU slot (see
    :mod:`synthdata.gpu_slots`) and runs on the GPU the slot is on; ``kind``
    ("untuned", "trial", "tuned") only labels it in the slot records.

    ``discrete_columns`` (the schema's categorical columns, target included when
    categorical) are the only columns synthcity treats as discrete; without it
    synthcity would guess from each column's number of distinct values.

    With ``class_prior`` (class shares of the fit data), rows are drawn per
    class to those shares (see :mod:`synthdata.generation.class_quota`):
    ddpm on a categorical target is given the labels, other plugins are
    rejection-sampled. The :class:`ClassSamplingReport` is attached as
    ``df.attrs["class_sampling"]``.
    """
    from synthcity.plugins import Plugins
    from synthcity.utils.dataframe import declared_discrete_columns

    plugin_kwargs = dict(params)
    # TabDDPM conditions on the class label for classification; synthcity's
    # ddpm only does so when told, and otherwise models the label as a column.
    if classification and plugin_accepts(name, "is_classification"):
        plugin_kwargs.setdefault("is_classification", True)
    if workspace is not None and plugin_accepts(name, "workspace"):
        plugin_kwargs["workspace"] = Path(workspace)
    # Seed training as well as sampling; plugins otherwise train with their
    # default random_state (0) whatever seed the run uses.
    if "random_state" not in plugin_kwargs and plugin_accepts(name, "random_state"):
        plugin_kwargs["random_state"] = random_state
    uses_gpu = device is not None and plugin_accepts(name, "device")
    if "device" in plugin_kwargs:  # an HPO trial passes its device in params
        device, uses_gpu = str(plugin_kwargs["device"]), True
    with gpu_slots.gpu_slot(
        name, kind, _epochs(name, plugin_kwargs), device if uses_gpu else None
    ) as slot_device:
        if uses_gpu and slot_device is not None:
            plugin_kwargs["device"] = torch.device(slot_device)
        with declared_discrete_columns(discrete_columns):
            return _fit_generate(
                Plugins().get(name, **plugin_kwargs),
                train_loader,
                n_samples,
                random_state,
                class_prior,
                conditional=bool(plugin_kwargs.get("is_classification")),
            )


def _epochs(name: str, plugin_kwargs: dict) -> float:
    """Training epochs (iterations) of this fit: its ``n_iter`` or the plugin's default."""
    if "n_iter" in plugin_kwargs:
        return float(plugin_kwargs["n_iter"])
    if not plugin_accepts(name, "n_iter"):
        return 0.0
    default = inspect.signature(get_plugin_class(name).__init__).parameters["n_iter"].default
    return 0.0 if default is inspect.Parameter.empty else float(default)


def _fit_generate(model, train_loader, n_samples, random_state, class_prior, conditional):
    model.fit(train_loader)
    if class_prior is None:
        return model.generate(count=n_samples, random_state=random_state).dataframe()

    target = train_loader.target_column
    encoder = (model._data_encoders or {}).get(target)

    def sample(count, seed, labels):
        kwargs = {}
        if labels is not None:
            # The plugin conditions on its label-encoded target (Plugin.fit
            # encodes it); generate() decodes the output again.
            kwargs["cond"] = encoder.transform(labels) if encoder is not None else labels
        return model.generate(count=count, random_state=seed, **kwargs).dataframe()

    df, report = class_quota.sample_to_quota(
        sample,
        target,
        class_prior,
        n_samples,
        random_state,
        conditional=conditional,
    )
    df.attrs["class_sampling"] = report
    return df


def build_synthcity_objective(
    name: str,
    train_loader,
    hpo_cfg: HPOConfig,
    seed: int,
    eval_fn: Callable[[pd.DataFrame], TrialScore],
    n_samples: int,
    workspace: str | None = None,
    device: str = "cpu",
    classification: bool = False,
    class_prior: pd.Series | None = None,
    *,
    discrete_columns: list,
):
    """Build an Optuna objective for a synthcity plugin's native hyperparameter space.

    Each trial samples from the plugin's own ``sample_hyperparameters_optuna``
    (training length included, see ``hpo.epoch_ranges``), forces CPU for
    MPS (which lacks the float64 support synthcity's metrics need internally)
    but otherwise uses ``device``, fits the plugin on ``train_loader`` and
    scores ``n_samples`` generated rows with ``eval_fn``, drawn to the
    ``class_prior`` class shares when given.
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
    searched = hpo_cfg.search_params.get(name)
    defaults = library_defaults(name, hpo_cfg) if searched is not None else {}

    def objective(trial: optuna.Trial) -> float:
        # Training length is searched, not capped: the plugin's own n_iter
        # range unless hpo.epoch_ranges (or EPOCH_RANGE_DEFAULTS) sets one.
        proxy = _SearchSpaceTrial(
            trial,
            HPO_EXCLUDED_CHOICES.get(name),
            epochs,
            HPO_FIXED_PARAMS.get(name),
            HPO_PARAM_RANGES.get(name),
            searched=set(searched) if searched is not None else None,
            defaults=defaults,
        )
        params = plugin_cls.sample_hyperparameters_optuna(proxy)
        if epochs is not None and "n_iter" not in params:
            params["n_iter"] = proxy.suggest_int("n_iter", *epochs)
        fit_seed = trial_seed(trial, seed)
        params["random_state"] = fit_seed
        if accepts_device:
            params["device"] = torch.device(trial_device)
        if accepts_patience_metric and hpo_cfg.pruner:
            params["patience_metric"] = _pruning_patience_metric_class()(trial, workspace)

        # A crash propagates and run_study marks the trial failed.
        synthetic = fit_generate(
            name,
            params,
            train_loader,
            n_samples,
            fit_seed,
            workspace=workspace,
            classification=classification,
            class_prior=class_prior,
            discrete_columns=discrete_columns,
            kind="trial",
        )
        return score_candidate(trial, eval_fn, synthetic)

    return objective
