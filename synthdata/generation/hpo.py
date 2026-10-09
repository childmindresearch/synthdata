"""Generic Optuna study management shared by all generation backends.

Provides:
- ``build_hpo_eval_fn``: scores a candidate synthetic DataFrame on the tuning
  split. The default objective is TSTR macro-F1 of a fixed XGBoost
  (:mod:`synthdata.evaluation.tstr`); privacy is reported by evaluation, not
  optimized. It also computes the sanity screens that Optuna treats as
  constraints (``screen_violations``).
- ``score_candidate``: records a candidate's screens and logged scores on its
  Optuna trial and returns the objective value.
- ``hpo_score``/``build_synthetic_eval_fn``: the older direction-aware
  synthcity composite (``hpo.objective: synthcity_composite``).
- ``create_study``/``run_study``: Optuna study creation with SQLite-backed
  persistence (resumable across runs, inspectable with optuna-dashboard).
- ``BestParamsCache``: JSON-backed cache of best hyperparameters per model,
  keyed by generator family (``synthcity`` / ``tabpfgen``), mirroring
  ``output/hepatitis/hpo_best_params.json`` from the notebooks.
"""

import dataclasses
import re
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from synthdata.config import HPOConfig, HPOConstraintsConfig
from synthdata.evaluation.tstr import tstr_scores
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
    *,
    discrete_columns: list,
) -> Callable[[pd.DataFrame], float]:
    """Build a ``syn_df -> score`` function via synthcity's Metrics.evaluate.

    ``discrete_columns`` are the columns synthcity's metrics treat as categorical.

    Mirrors the notebooks' ``_eval_syn_df`` helper: builds a second independent
    synthetic draw (bootstrap resample) for DomiasMIA's reference set, and an
    augmented train+synthetic set for augmentation metrics.
    """
    from synthcity.metrics import Metrics
    from synthcity.plugins.core.dataloader import GenericDataLoader
    from synthcity.utils.dataframe import declared_discrete_columns

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
        with declared_discrete_columns(discrete_columns):
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
                # The cache keys on data and metric name, not code; see synthcity_eval.
                use_cache=False,
            )
        return hpo_score(report)

    return eval_fn


#: Constraint order stored on every trial; a value above 0 marks it infeasible.
SCREENS = ("copies", "missing_classes", "category_coverage", "out_of_range", "class_share_gap")

#: Categories rarer than this in train are not required in the synthetic data.
MIN_CATEGORY_FREQUENCY = 0.01


@dataclasses.dataclass
class TrialScore:
    """One candidate's objective value, constraint violations and logged scores."""

    value: float
    constraints: list = dataclasses.field(default_factory=lambda: [0.0] * len(SCREENS))
    attrs: dict = dataclasses.field(default_factory=dict)


def objective_direction(objective: str) -> str:
    return "minimize" if objective == "synthcity_composite" else "maximize"


def _comparable(series: pd.Series) -> pd.Series:
    """Numbers as floats, everything else as strings, so 1, 1.0 and "1" agree."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == series.notna().sum():
        return numeric.astype(float)
    return series.astype(str)


def _row_hashes(df: pd.DataFrame, columns: list) -> pd.Series:
    normalized = pd.DataFrame({c: _comparable(df[c]) for c in columns})
    return pd.util.hash_pandas_object(normalized, index=False)


def exact_match_rate(rows: pd.DataFrame, reference: pd.DataFrame) -> float:
    """Share of ``rows`` that equal some ``reference`` row on every shared column."""
    columns = [c for c in reference.columns if c in rows.columns]
    if rows.empty:
        return 0.0
    return float(_row_hashes(rows, columns).isin(set(_row_hashes(reference, columns))).mean())


def out_of_range_share(rows: pd.DataFrame, reference: pd.DataFrame, columns: list) -> float:
    """Share of numeric values in ``rows`` outside ``reference``'s [min, max] per column."""
    outside, total = 0, 0
    for column in columns:
        values = pd.to_numeric(rows[column], errors="coerce").dropna()
        low, high = reference[column].min(), reference[column].max()
        outside += int(((values < low) | (values > high)).sum())
        total += len(values)
    return outside / total if total else 0.0


def screen_violations(
    synthetic_df: pd.DataFrame,
    search_train_df: pd.DataFrame,
    tuning_df: pd.DataFrame,
    target_column: str,
    categorical_columns: list,
    constraints: HPOConstraintsConfig,
) -> dict:
    """Lenient sanity screens; each value is a violation size (<= 0 is feasible).

    - ``copies``: exact-match rate of synthetic rows against search-train,
      minus that of the real tuning rows, minus ``copy_margin``. Calibrating
      on held-out real rows keeps legitimately repeated clinical rows from
      failing the screen.
    - ``missing_classes``: number of real target classes the synthetic data lacks.
    - ``category_coverage``: ``min_category_coverage`` minus the lowest
      coverage of train categories (with at least 1% frequency) over the
      categorical columns.
    - ``out_of_range``: share of numeric values outside the search-train
      range, minus that share for the real tuning rows (fresh real rows fall
      outside too, about 2/n per column), minus ``max_out_of_range``.
    - ``class_share_gap`` is set by :func:`build_hpo_eval_fn`, not here.

    A screen turned off in the config reports -1.
    """
    violations = dict.fromkeys(SCREENS, -1.0)

    if constraints.copy_margin is not None:
        baseline = exact_match_rate(tuning_df, search_train_df)
        violations["copies"] = (
            exact_match_rate(synthetic_df, search_train_df) - baseline - constraints.copy_margin
        )

    if constraints.min_category_coverage is not None:
        real_classes = set(_comparable(search_train_df[target_column]).dropna())
        if target_column in synthetic_df:
            synthetic_classes = set(_comparable(synthetic_df[target_column]).dropna())
        else:
            synthetic_classes = set()
        violations["missing_classes"] = float(len(real_classes - synthetic_classes)) or -1.0
        coverages = []
        for column in categorical_columns:
            if column not in synthetic_df or column not in search_train_df:
                continue
            shares = _comparable(search_train_df[column]).value_counts(normalize=True)
            required = set(shares[shares >= MIN_CATEGORY_FREQUENCY].index)
            if required:
                present = set(_comparable(synthetic_df[column]).dropna())
                coverages.append(len(required & present) / len(required))
        if coverages:
            violations["category_coverage"] = constraints.min_category_coverage - min(coverages)

    if constraints.max_out_of_range is not None:
        numeric = [
            c
            for c in search_train_df.columns
            if c != target_column
            and c not in categorical_columns
            and c in synthetic_df
            and pd.api.types.is_numeric_dtype(search_train_df[c])
        ]
        if numeric:
            violations["out_of_range"] = (
                out_of_range_share(synthetic_df, search_train_df, numeric)
                - out_of_range_share(tuning_df, search_train_df, numeric)
                - constraints.max_out_of_range
            )

    return violations


def class_share_gap(labels: pd.Series, prior: pd.Series) -> float:
    """Total variation distance between the class shares of ``labels`` and ``prior``."""
    shares = labels.value_counts(normalize=True)
    classes = prior.index.union(shares.index)
    return float(
        (shares.reindex(classes, fill_value=0) - prior.reindex(classes, fill_value=0)).abs().sum()
        / 2
    )


def build_hpo_eval_fn(
    search_train_df: pd.DataFrame,
    tuning_df: pd.DataFrame,
    target_column: str,
    nominal_columns: list,
    categorical_columns: list,
    target_is_categorical: bool,
    sensitive_features: list,
    hpo_cfg: HPOConfig,
    seed: int,
    match_prior: bool = True,
    workspace: str | Path | None = None,
) -> Callable[[pd.DataFrame], TrialScore]:
    """Build a ``synthetic_df -> TrialScore`` function scored on the tuning split.

    Candidates are fitted on ``search_train_df`` and never see the tuning or
    test rows. With ``match_prior``, candidates are drawn to the search-train
    class shares (see :mod:`synthdata.generation.class_quota`), and the
    ``class_share_gap`` screen fails one whose shares still miss them by more
    than ``max_class_share_gap`` because a class fell short. The TSTR scores of the
    same classifier trained on the real search-train rows (TRTR) are the
    ceiling, logged on every trial.
    """
    tstr = hpo_cfg.objective.startswith("tstr_")
    if tstr and not target_is_categorical:
        raise ValueError(
            f"hpo.objective {hpo_cfg.objective!r} needs a categorical target; use "
            "synthcity_composite for a numeric target"
        )
    prior = search_train_df[target_column].value_counts(normalize=True)
    classes = sorted(prior.index, key=str)
    seeds = [seed + i for i in range(hpo_cfg.tstr_seeds)]

    def _tstr(fit_df):
        return tstr_scores(
            fit_df, tuning_df, target_column, nominal_columns, classes, seeds, weighted=True
        )

    trtr = None
    if target_is_categorical:
        trtr = _tstr(search_train_df)
        logger.info(
            "HPO ceiling (fixed XGBoost trained on real search-train rows, scored on tuning): "
            "macro-F1=%.4f macro-AUPRC=%.4f balanced-accuracy=%.4f; class-weighted fit: "
            "macro-F1=%.4f balanced-accuracy=%.4f",
            trtr.macro_f1,
            trtr.macro_auprc,
            trtr.balanced_accuracy,
            trtr.weighted_macro_f1,
            trtr.weighted_balanced_accuracy,
        )
    composite_fn = (
        None
        if tstr
        else build_synthetic_eval_fn(
            search_train_df,
            tuning_df,
            target_column,
            sensitive_features,
            hpo_cfg.metric_config,
            seed,
            workspace=workspace,
            discrete_columns=list(categorical_columns)
            + ([target_column] if target_is_categorical else []),
        )
    )

    def eval_fn(synthetic_df: pd.DataFrame) -> TrialScore:
        violations = screen_violations(
            synthetic_df,
            search_train_df,
            tuning_df,
            target_column,
            categorical_columns,
            hpo_cfg.constraints,
        )
        if not target_is_categorical:
            violations["missing_classes"] = -1.0
        elif match_prior and hpo_cfg.constraints.max_class_share_gap is not None:
            violations["class_share_gap"] = (
                class_share_gap(synthetic_df[target_column], prior)
                - hpo_cfg.constraints.max_class_share_gap
            )
        attrs = {"screens": violations}
        if target_is_categorical:
            scores = _tstr(synthetic_df)
            attrs |= {
                "tstr_macro_f1": scores.macro_f1,
                "tstr_macro_auprc": scores.macro_auprc,
                "tstr_balanced_accuracy": scores.balanced_accuracy,
                "tstr_weighted_macro_f1": scores.weighted_macro_f1,
                "tstr_weighted_balanced_accuracy": scores.weighted_balanced_accuracy,
                "tstr_per_class_f1": scores.per_class_f1,
                "tstr_seed_sd": scores.seed_sd,
                "trtr_macro_f1": trtr.macro_f1,
                "trtr_macro_auprc": trtr.macro_auprc,
                "trtr_balanced_accuracy": trtr.balanced_accuracy,
                "trtr_weighted_macro_f1": trtr.weighted_macro_f1,
                "trtr_weighted_balanced_accuracy": trtr.weighted_balanced_accuracy,
            }
        value = attrs[hpo_cfg.objective] if tstr else composite_fn(synthetic_df)
        if not np.isfinite(value):
            raise ValueError(f"HPO objective {hpo_cfg.objective} is not finite: {value}")
        return TrialScore(value, [violations[name] for name in SCREENS], attrs)

    return eval_fn


def score_candidate(
    trial: optuna.Trial, eval_fn: Callable[[pd.DataFrame], TrialScore], synthetic_df
) -> float:
    """Score one candidate, store its screens and logged scores on ``trial``."""
    score = eval_fn(synthetic_df)
    trial.set_user_attr("constraints", list(score.constraints))
    for key, value in score.attrs.items():
        trial.set_user_attr(key, value)
    failed = [name for name, v in zip(SCREENS, score.constraints, strict=True) if v > 0]
    if failed:
        logger.info("trial %d is infeasible: failed screens %s", trial.number, failed)
    return score.value


def _trial_constraints(trial: optuna.trial.FrozenTrial) -> list:
    # Trials stopped before scoring have no screens; they are never "best".
    # Trials stored before a screen was added get 0 (feasible) for it.
    constraints = list(trial.user_attrs.get("constraints", []))
    return constraints + [0.0] * (len(SCREENS) - len(constraints))


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
    best_trial_number: int | None = None,
) -> int:
    """Compact completed synthcity HPO generator caches.

    Synthcity stores a fully serialized generator for every benchmark trial.
    Keep the best trial and the highest-numbered trial with a saved generator
    as conservative recovery artifacts once no trial is running; remove the
    other generator caches even when a time-limited study finished below its
    configured target. Metric and synthetic-data caches are intentionally left
    untouched.
    """
    if not plugin_name:
        raise ValueError("plugin_name must not be empty")

    running = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.RUNNING]
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if running:
        logger.info(
            "[%s] HPO still has %d running trial(s); retaining generator checkpoints in %s",
            study.study_name,
            len(running),
            workspace,
        )
        return 0
    if not completed:
        logger.info(
            "[%s] HPO has no completed trials; retaining generator checkpoints in %s",
            study.study_name,
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

    if best_trial_number is None:
        best_trial_number = study.best_trial.number
    keep_trial_numbers = {best_trial_number, max(checkpoints_by_trial)}
    to_delete = [
        path
        for trial_number, paths in checkpoints_by_trial.items()
        if trial_number not in keep_trial_numbers
        for path in paths
    ]

    failures: list[tuple[Path, OSError]] = []
    for path in to_delete:
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


def model_budget(hpo_cfg: HPOConfig, model: str) -> tuple[int, int | None]:
    """``(n_trials, timeout_seconds)`` for ``model``'s study, per-model values first."""
    return (
        hpo_cfg.n_trials_per_model.get(model, hpo_cfg.n_trials),
        hpo_cfg.timeout_seconds_per_model.get(model, hpo_cfg.timeout_seconds),
    )


def epoch_range(
    hpo_cfg: HPOConfig, model: str, default: tuple | None
) -> tuple[int, int, int] | None:
    """Searched training length ``(low, high, step)``: the config's, else ``default``."""
    bounds = hpo_cfg.epoch_ranges.get(model, default)
    return tuple(bounds) if bounds is not None else None


def make_pruner(hpo_cfg: HPOConfig) -> optuna.pruners.BasePruner:
    if hpo_cfg.pruner == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=hpo_cfg.pruner_startup_trials,
            n_warmup_steps=hpo_cfg.pruner_warmup_steps,
        )
    return optuna.pruners.NopPruner()


def create_study(
    study_name: str, hpo_cfg: HPOConfig, output_dir: str | Path, seed: int
) -> optuna.Study:
    storage = hpo_cfg.storage or default_storage_url(output_dir)
    with warnings.catch_warnings():
        # constraints_func is marked experimental but stable since Optuna 3.0.
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = optuna.samplers.TPESampler(seed=seed, constraints_func=_trial_constraints)
    return optuna.create_study(
        study_name=study_name,
        direction=objective_direction(hpo_cfg.objective),
        # Constrained TPE (Watanabe & Hutter, 2023): infeasible trials keep
        # their true value and teach the sampler where screens fail.
        sampler=sampler,
        # Pruning stops weak trials during training, from the intermediate
        # scores an objective reports (see synthcity_backend.PruningPatienceMetric).
        pruner=make_pruner(hpo_cfg),
        storage=storage,
        load_if_exists=True,
    )


def run_study(
    study_name: str,
    objective_fn: Callable[[optuna.Trial], float],
    hpo_cfg: HPOConfig,
    output_dir: str | Path,
    seed: int,
    drop_keys: tuple = (),
    checkpoint_workspace: str | Path | None = None,
    checkpoint_plugin: str | None = None,
    model: str | None = None,
) -> dict:
    """Run (or resume, via SQLite storage) an Optuna study; return best params.

    The trial count and timeout come from ``model_budget`` for ``model``
    (default: ``study_name`` without its ``hpo_`` prefix); 0 trials skips the
    search and returns ``{}``. Pruned trials count towards the trial budget.
    ``drop_keys`` are removed from the returned best-params dict.

    When both checkpoint arguments are provided, synthcity HPO studies with no
    running trial retain only their best and latest generator caches. Studies
    with a running trial or no completed trial retain every cache so an active
    or unsuccessful run remains recoverable.
    """
    if (checkpoint_workspace is None) != (checkpoint_plugin is None):
        raise ValueError("checkpoint_workspace and checkpoint_plugin must be provided together")

    n_trials, timeout = model_budget(hpo_cfg, model or study_name.removeprefix("hpo_"))
    if n_trials == 0:
        logger.info("[%s] hyperparameter search skipped (0 trials); using defaults", study_name)
        return {}
    study = create_study(study_name, hpo_cfg, output_dir, seed)
    finished = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    n_done = len([t for t in study.trials if t.state in finished])
    n_remaining = max(n_trials - n_done, 0)
    if n_remaining > 0:
        logger.info(
            "[%s] starting hyperparameter optimization: %d trial(s) remaining "
            "(%d already finished, target=%d, timeout=%ss)",
            study_name,
            n_remaining,
            n_done,
            n_trials,
            timeout,
        )
        study.optimize(
            objective_fn,
            n_trials=n_remaining,
            timeout=timeout,
            # A crashed fit or a non-finite score marks the trial failed.
            catch=(ValueError, RuntimeError),
            show_progress_bar=False,
        )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        logger.warning("[%s] all trials pruned/failed; falling back to defaults", study_name)
        return {}
    feasible = [t for t in completed if all(v <= 0 for v in _trial_constraints(t))]
    if not feasible:
        logger.warning(
            "[%s] no trial passed the HPO screens (%d completed); falling back to defaults",
            study_name,
            len(completed),
        )
        return {}

    pick = max if study.direction == optuna.study.StudyDirection.MAXIMIZE else min
    best_trial = pick(feasible, key=lambda t: t.value)
    best = {k: v for k, v in best_trial.params.items() if k not in drop_keys}
    logger.info(
        "[%s] best %s=%.4f (trial %d; %d feasible of %d completed) params=%s",
        study_name,
        hpo_cfg.objective,
        best_trial.value,
        best_trial.number,
        len(feasible),
        len(completed),
        best,
    )
    if checkpoint_workspace is not None and checkpoint_plugin is not None:
        cleanup_hpo_generator_checkpoints(
            checkpoint_workspace,
            study,
            checkpoint_plugin,
            best_trial_number=best_trial.number,
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
