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
    db_path = Path(output_dir).parent / "optuna_studies.db"
    ensure_dir(db_path.parent)
    return f"sqlite:///{db_path}"


def default_best_params_path(output_dir: str | Path) -> Path:
    return Path(output_dir).parent / "hpo_best_params.json"


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
) -> dict:
    """Run (or resume, via SQLite storage) an Optuna study; return best params.

    ``drop_keys`` are removed from the returned best-params dict: e.g. ``n_iter``
    is capped during search for speed, so the searched value is unreliable and
    generation should fall back to the plugin's own default instead.
    """
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
