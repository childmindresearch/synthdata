"""synthcity plugin fit/generate/hyperparameter-search glue."""

import inspect

import optuna
import pandas as pd
import torch

from synthdata.config import HPOConfig
from synthdata.generation.hpo import hpo_score
from synthdata.utils import get_logger

logger = get_logger(__name__)


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
    cls = get_plugin_class(name)
    sig = inspect.signature(cls.__init__)
    return param_name in sig.parameters


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
    workspace: str | None = None,
    device: str = "cpu",
):
    """Build an Optuna objective for a synthcity plugin's native hyperparameter space.

    Mirrors the hepatitis notebook's HPO cell: samples from the plugin's own
    ``sample_hyperparameters_optuna``, caps ``n_iter`` for speed (only if the
    plugin exposes it), forces CPU for MPS (which lacks the float64 support
    synthcity's metrics need internally) but otherwise uses ``device``, and
    scores each trial via a single-model, single-repeat ``Benchmarks.evaluate``
    call.
    """
    from pathlib import Path

    from synthcity.benchmark import Benchmarks

    plugin_cls = get_plugin_class(name)
    accepts_device = plugin_accepts(name, "device")
    accepts_iter = plugin_accepts(name, "n_iter")
    iter_cap = hpo_cfg.model_iter_caps.get(name, hpo_cfg.n_iter_cap)
    workspace_path = Path(workspace) if workspace else Path("workspace")
    trial_device = "cpu" if device == "mps" else device

    def objective(trial: optuna.Trial) -> float:
        params = plugin_cls.sample_hyperparameters_optuna(trial)
        if accepts_iter:
            params["n_iter"] = min(params.get("n_iter", iter_cap), iter_cap)
        params["random_state"] = seed
        if accepts_device:
            params["device"] = torch.device(trial_device)

        trial_id = f"trial_{trial.number}"
        try:
            report = Benchmarks.evaluate(
                [(trial_id, name, params)],
                train_loader,
                repeats=1,
                metrics=hpo_cfg.metric_config,
                task_type="classification",
                workspace=workspace_path,
            )
        except Exception as exc:  # noqa: BLE001 - HPO trials should soft-fail
            logger.warning("[%s] trial %d failed: %s", name, trial.number, exc)
            raise optuna.TrialPruned()
        return hpo_score(report[trial_id])

    return objective
