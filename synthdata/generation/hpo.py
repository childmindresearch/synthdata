"""Generic Optuna study management shared by all generation backends.

Provides:
- ``hpo_score``: the direction-aware composite objective used throughout the
  hepatitis notebooks (orient every metric so higher = better, then average).
- ``build_synthetic_eval_fn``: scores an arbitrary candidate synthetic DataFrame
  via synthcity's ``Metrics.evaluate`` (used as the HPO objective for generators,
  like TabPFGen, that don't go through synthcity's ``Benchmarks``).
- ``create_study``/``run_study``: Optuna study creation with SQLite-backed
  persistence (resumable across runs, inspectable with optuna-dashboard).
- ``BestParamsCache``: JSON-backed cache of best hyperparameters per model,
  keyed by generator family (``synthcity`` / ``tabpfgen``), mirroring
  ``output/hepatitis/hpo_best_params.json`` from the notebooks.
"""

import re
from collections.abc import Callable
from pathlib import Path

import optuna
import pandas as pd

from synthdata.config import HPOConfig
from synthdata.utils import ensure_dir, get_logger, load_json, save_json

logger = get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def hpo_score(report_df: pd.DataFrame) -> float:
    """Direction-aware composite score: orient metrics so higher=better, negate mean.

    ``report_df`` must have ``mean`` and ``direction`` columns (as returned by
    synthcity's ``Metrics.evaluate``/``Benchmarks.evaluate``). The result is
    suitable as an Optuna objective under ``direction="minimize"``.
    """
    sign = report_df["direction"].map({"maximize": 1.0, "minimize": -1.0})
    return -(report_df["mean"] * sign).mean()


def build_synthetic_eval_fn(
    train_reference_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    target_column: str,
    sensitive_features: list,
    metric_config: dict,
    seed: int,
    workspace: str | Path | None = None,
) -> Callable[[pd.DataFrame], float]:
    """Build a ``syn_df -> score`` function via synthcity's Metrics.evaluate.

    Mirrors the notebooks' ``_eval_syn_df`` helper: builds a second independent
    synthetic draw (bootstrap resample) for DomiasMIA's reference set, and an
    augmented train+synthetic set for augmentation metrics.
    """
    from synthcity.metrics import Metrics
    from synthcity.plugins.core.dataloader import GenericDataLoader

    workspace_path = Path(workspace) if workspace else Path("workspace")

    def _loader(df: pd.DataFrame) -> GenericDataLoader:
        return GenericDataLoader(
            df, target_column=target_column, sensitive_features=sensitive_features
        )

    def eval_fn(syn_df: pd.DataFrame) -> float:
        ref_df = syn_df.sample(n=len(syn_df), replace=True, random_state=seed + 1).reset_index(
            drop=True
        )
        x_aug = pd.concat([train_reference_df, syn_df], ignore_index=True)
        report = Metrics.evaluate(
            _loader(holdout_df),
            _loader(syn_df),
            _loader(train_reference_df),
            _loader(ref_df),
            _loader(x_aug),
            metrics=metric_config,
            task_type="classification",
            random_state=seed,
            workspace=workspace_path,
        )
        return hpo_score(report)

    return eval_fn


def default_storage_url(output_dir: str | Path) -> str:
    db_path = Path(output_dir) / "optuna_studies.db"
    ensure_dir(db_path.parent)
    return f"sqlite:///{db_path}"


def default_best_params_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "hpo_best_params.json"


def cleanup_hpo_generator_checkpoints(
    workspace: str | Path,
    study: optuna.Study,
    plugin_name: str,
    target_trials: int,
) -> int:
    """Compact completed synthcity HPO generator caches.

    Synthcity stores a fully serialized generator for every benchmark trial.
    Keep the best trial and the highest-numbered trial with a saved generator
    as conservative recovery artifacts; remove the other generator caches only
    after the study reaches its configured completed-trial target. Metric and
    synthetic-data caches are intentionally left untouched.
    """
    if not plugin_name:
        raise ValueError("plugin_name must not be empty")
    if target_trials < 1:
        raise ValueError(f"target_trials must be positive, got {target_trials}")

    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < target_trials:
        logger.info(
            "[%s] HPO incomplete (%d/%d completed); retaining generator checkpoints in %s",
            study.study_name,
            len(completed),
            target_trials,
            workspace,
        )
        return 0

    workspace_path = Path(workspace)
    if not workspace_path.is_dir():
        logger.info(
            "[%s] no synthcity checkpoint workspace found at %s",
            study.study_name,
            workspace_path,
        )
        return 0

    trial_numbers = {trial.number for trial in study.trials}
    pattern = re.compile(rf"_trial_(?P<trial>\d+)_{re.escape(plugin_name)}_.*_generator_\d+\.bkp$")
    checkpoints_by_trial: dict[int, list[Path]] = {}
    for path in workspace_path.iterdir():
        if not path.is_file():
            continue
        match = pattern.search(path.name)
        if match is None:
            continue
        trial_number = int(match.group("trial"))
        if trial_number not in trial_numbers:
            continue
        checkpoints_by_trial.setdefault(trial_number, []).append(path)

    if not checkpoints_by_trial:
        logger.info(
            "[%s] no generator checkpoints found for plugin=%s in %s",
            study.study_name,
            plugin_name,
            workspace_path,
        )
        return 0

    keep_trial_numbers = {study.best_trial.number, max(checkpoints_by_trial)}
    to_delete = [
        path
        for trial_number, paths in checkpoints_by_trial.items()
        if trial_number not in keep_trial_numbers
        for path in paths
    ]

    failures: list[tuple[Path, OSError]] = []
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError as exc:
            failures.append((path, exc))

    if failures:
        details = "; ".join(f"{path}: {exc}" for path, exc in failures)
        logger.error(
            "[%s] failed to delete %d HPO generator checkpoint(s): %s",
            study.study_name,
            len(failures),
            details,
        )
        raise RuntimeError(
            f"Failed to delete {len(failures)} HPO generator checkpoint(s)"
        ) from failures[0][1]

    kept_count = sum(
        len(paths)
        for trial_number, paths in checkpoints_by_trial.items()
        if trial_number in keep_trial_numbers
    )
    logger.info(
        "[%s] HPO checkpoint cleanup complete for plugin=%s: kept %d checkpoint(s) "
        "for trial(s) %s; deleted %d from %s",
        study.study_name,
        plugin_name,
        kept_count,
        sorted(keep_trial_numbers),
        len(to_delete),
        workspace_path,
    )
    return len(to_delete)


def create_study(
    study_name: str, hpo_cfg: HPOConfig, output_dir: str | Path, seed: int
) -> optuna.Study:
    storage = hpo_cfg.storage or default_storage_url(output_dir)
    return optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=storage,
        load_if_exists=True,
    )


def run_study(
    study_name: str,
    objective_fn: Callable[[optuna.Trial], float],
    hpo_cfg: HPOConfig,
    output_dir: str | Path,
    seed: int,
    drop_keys: tuple = ("n_iter",),
    checkpoint_workspace: str | Path | None = None,
    checkpoint_plugin: str | None = None,
) -> dict:
    """Run (or resume, via SQLite storage) an Optuna study; return best params.

    ``drop_keys`` are removed from the returned best-params dict: e.g. ``n_iter``
    is capped during search for speed, so the searched value is unreliable and
    generation should fall back to the plugin's own default instead.

    When both checkpoint arguments are provided, completed synthcity HPO
    studies retain only their best and latest generator caches. Incomplete
    studies retain every cache so an interrupted run remains recoverable.
    """
    if (checkpoint_workspace is None) != (checkpoint_plugin is None):
        raise ValueError("checkpoint_workspace and checkpoint_plugin must be provided together")

    study = create_study(study_name, hpo_cfg, output_dir, seed)
    n_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_remaining = max(hpo_cfg.n_trials - n_done, 0)
    if n_remaining > 0:
        logger.info(
            "[%s] starting hyperparameter optimization: %d trial(s) remaining "
            "(%d already completed, target=%d)",
            study_name,
            n_remaining,
            n_done,
            hpo_cfg.n_trials,
        )
        study.optimize(
            objective_fn,
            n_trials=n_remaining,
            timeout=hpo_cfg.timeout_seconds,
            show_progress_bar=False,
        )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        logger.warning("[%s] all trials pruned/failed; falling back to defaults", study_name)
        return {}

    best = {k: v for k, v in study.best_params.items() if k not in drop_keys}
    logger.info(
        "[%s] best score=%.4f (n_trials=%d) params=%s",
        study_name,
        study.best_value,
        len(completed),
        best,
    )
    if checkpoint_workspace is not None and checkpoint_plugin is not None:
        cleanup_hpo_generator_checkpoints(
            checkpoint_workspace,
            study,
            checkpoint_plugin,
            len(completed),
        )
    return best


class BestParamsCache:
    """JSON-backed cache of ``{family: {model_name: params}}`` best hyperparameters."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data = load_json(self.path, default={})

    def get(self, family: str, model_name: str) -> dict:
        return self._data.get(family, {}).get(model_name, {})

    def has(self, family: str, model_name: str) -> bool:
        return model_name in self._data.get(family, {})

    def set(self, family: str, model_name: str, params: dict) -> None:
        self._data.setdefault(family, {})[model_name] = params
        save_json(self.path, self._data)
