#!/usr/bin/env python
"""CLI: tune the TSTR XGBoost on real training rows, for ``evaluation.xgboost_params``.

The same class-balanced XGBoost predicts the target in holdout TSTR, the HPO
objective, the real-data ceiling and the fairness gaps. This script searches
its parameters on real data only, with Optuna, so the classifier gets out
what signal the data has before it is used to judge generators:

* rows: train minus the tuning split (all of train when there is none). The
  tuning split and the holdout are never read, so the HPO objective and the
  final scores stay out of sample;
* score: macro AUPRC (the sim HPO objective), mean over ``--folds``
  patient-grouped, stratified folds (``StratifiedGroupKFold``);
* the defaults and the config's current ``evaluation.xgboost_params`` are
  scored on the same folds, with the paired fold difference to the best
  trial, so you can see whether tuning is worth it.

The best of many trials scored on the same folds is optimistic; the TRTR
ceiling on the holdout in the next run is the out-of-sample check. Writes
``trials.csv`` and ``summary.json`` under ``<output root>/xgboost_tuning/data_v_<version>/`` (scores and parameters only, no data rows)
and prints the line to paste into the config.

Imputation must have run (``synthdata-impute --config <path>``).

Usage:
    python -m scripts.tune_tstr_xgboost --config configs/config_sim.yaml [--trials 60] [--folds 5] [--device cuda]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from synthdata.config import load_config
from synthdata.data import load_dataset, load_imputed_splits
from synthdata.evaluation.tstr import XGB_PARAMS, _macro_auprc, target_classes, tstr_predictor
from synthdata.utils import ensure_dir, get_logger, set_global_seed

logger = get_logger("tune_tstr_xgboost")


def suggest_params(trial) -> dict:
    """The searched XGBClassifier arguments (the rest keep :data:`XGB_PARAMS`)."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 100.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
    }


def grouped_folds(frame: pd.DataFrame, target: str, groups, n_folds: int, seed: int) -> list:
    """(fit, score) row labels of patient-grouped, stratified folds over ``frame``."""
    groups = np.arange(len(frame)) if groups is None else np.asarray(groups)
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [
        (frame.index[fit], frame.index[score])
        for fit, score in splitter.split(frame, frame[target].astype(str), groups)
    ]


def fold_scores(
    frame: pd.DataFrame,
    folds: list,
    target: str,
    nominal_columns: list,
    classes: list,
    params: dict | None,
    seed: int,
) -> np.ndarray:
    """Macro AUPRC of the class-balanced TSTR classifier on each fold."""
    scores = []
    for fit, score in folds:
        y_true, predict = tstr_predictor(
            frame.loc[fit], frame.loc[score], target, nominal_columns, classes, params
        )
        scores.append(_macro_auprc(y_true, predict(seed)))
    return np.asarray(scores)


def _summary(scores: np.ndarray, best: np.ndarray) -> dict:
    diff = best - scores
    return {
        "macro_auprc": float(scores.mean()),
        "fold_se": float(scores.std(ddof=1) / np.sqrt(len(scores))),
        "best_minus_this": float(diff.mean()),
        "best_minus_this_se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
    }


def tune(
    frame: pd.DataFrame,
    groups,
    target: str,
    nominal_columns: list,
    classes: list,
    n_trials: int,
    n_folds: int,
    seed: int,
    configured: dict | None = None,
    device: str | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Search the parameters; return the summary and one row per trial."""
    import optuna

    folds = grouped_folds(frame, target, groups, n_folds, seed)
    extra = {"device": device} if device else {}

    def score(params):
        return fold_scores(frame, folds, target, nominal_columns, classes, params, seed)

    def objective(trial):
        values = score({**suggest_params(trial), **extra})
        trial.set_user_attr("fold_scores", values.tolist())
        return float(values.mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True)
    )
    # Trial 0 is the defaults, so the search starts from them.
    defaults = {k: XGB_PARAMS[k] for k in ("n_estimators", "learning_rate", "max_depth")}
    study.enqueue_trial(
        {
            **defaults,
            "min_child_weight": 1.0,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_lambda": 1.0,
            "gamma": 0.0,
        }
    )
    study.optimize(objective, n_trials=n_trials)

    best = np.asarray(study.best_trial.user_attrs["fold_scores"])
    summary = {
        "rows": len(frame),
        "folds": n_folds,
        "trials": n_trials,
        "best_params": study.best_params,
        "best": _summary(best, best),
        "defaults": _summary(score(extra or None), best),
    }
    if configured:
        summary["configured"] = {
            "params": configured,
            **_summary(score({**configured, **extra}), best),
        }
    trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    return summary, trials


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default=None, help="XGBoost device, e.g. cuda")
    parser.add_argument("--out", default=None, help="default: <output root>/xgboost_tuning")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_global_seed(cfg.seed)
    dataset = load_imputed_splits(load_dataset(cfg))
    has_tuning = len(dataset.tuning_index) > 0
    frame = dataset.search_train_imputed_df if has_tuning else dataset.train_imputed_df
    if frame is None:
        raise SystemExit(f"No imputed data: run synthdata-impute --config {args.config} first")
    target = cfg.data.target_column
    groups = None if dataset.patient_ids is None else dataset.patient_ids.loc[frame.index]
    classes = target_classes(frame[target], target in dataset.ordinal_columns)
    logger.info(
        "Tuning on %d %s rows, %d folds, %d trials",
        len(frame),
        "train-minus-tuning" if has_tuning else "train",
        args.folds,
        args.trials,
    )
    summary, trials = tune(
        frame,
        groups,
        target,
        list(dataset.nominal_columns),
        classes,
        args.trials,
        args.folds,
        cfg.seed,
        configured=cfg.evaluation.xgboost_params,
        device=args.device,
    )

    out = ensure_dir(
        Path(args.out)
        if args.out
        else Path(cfg.evaluation.output_dir).parent
        / "xgboost_tuning"
        / f"data_v_{cfg.data.version}"
    )
    trials.to_csv(out / "trials.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    params = {
        k: round(v, 4) if isinstance(v, float) else v for k, v in summary["best_params"].items()
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "best_params"}, indent=2))
    print("\nevaluation:\n  xgboost_params: " + json.dumps(params).replace('"', ""))
    print(f"\nWrote {out / 'trials.csv'} and {out / 'summary.json'}")


if __name__ == "__main__":
    main()
