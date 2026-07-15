#!/usr/bin/env python
"""CLI: evaluate synthetic data with synthcity + SynthEval + custom (log
disparity / fork-only fairness) metrics, combined into a single ranked table.

Evaluates the most recent experiment started by `synthdata-generate` unless
``--experiment-id`` is given to target a specific past one (see
:mod:`synthdata.experiment`).

Usage:
    synthdata-evaluate --config configs/config.yaml [--plot] [--experiment-id ID]

Requires generated synthetic data (run `synthdata-generate` first).
"""

import argparse
from pathlib import Path

import pandas as pd

from synthdata.config import load_config
from synthdata.data import load_dataset, load_imputed_splits
from synthdata.evaluation import run_evaluation
from synthdata.experiment import load_experiment
from synthdata.utils import get_logger, set_global_seed

logger = get_logger("run_evaluation")


def _load_synthetic_datasets(cfg) -> dict:
    output_dir = Path(cfg.generation.output_dir)
    datasets = {}
    for path in sorted(output_dir.glob("*.csv")):
        datasets[path.stem] = pd.read_csv(path)
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic data with synthcity + SynthEval + custom fairness metrics."
    )
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument(
        "--plot", action="store_true", help="Save rank trade-off + log-disparity + SynthEval plots."
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Evaluate a specific past experiment instead of the most recent one "
        "(overrides experiment.id).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.experiment_id:
        cfg.experiment.id = args.experiment_id
    set_global_seed(cfg.seed)

    dataset = load_dataset(cfg)
    dataset = load_imputed_splits(dataset)
    if dataset.train_imputed_df is None:
        raise SystemExit(
            "No imputed data found. Run `synthdata-impute --config <path>` first."
        )

    experiment = load_experiment(cfg)
    cfg.generation.output_dir = str(experiment.generation_dir)
    cfg.evaluation.output_dir = str(experiment.evaluation_dir)
    cfg.plots.output_dir = str(experiment.plots_dir)

    synthetic_datasets = _load_synthetic_datasets(cfg)
    if not synthetic_datasets:
        raise SystemExit(
            f"No synthetic datasets found in {cfg.generation.output_dir}. "
            "Run `synthdata-generate --config <path>` first."
        )

    combined, extras = run_evaluation(cfg, dataset, synthetic_datasets)
    logger.info(
        "Combined evaluation table (top 10 by overall rank):\n%s",
        combined.head(10).to_string(),
    )

    if args.plot:
        from synthdata.plotting.evaluation_plots import (
            save_log_disparity_plots,
            save_per_model_syntheval_plots,
            save_rank_tradeoff_plots,
        )

        save_rank_tradeoff_plots(cfg, combined, cfg.plots.output_dir)
        save_log_disparity_plots(extras["log_disparity_reports"], cfg.plots.output_dir)

        if (
            cfg.evaluation.save_per_model_syntheval_plots
            and extras["syntheval_benchmark_results"] is not None
        ):
            preset_path = Path(cfg.evaluation.output_dir) / "syntheval_preset.json"
            save_per_model_syntheval_plots(
                dataset, extras["selected_datasets"], preset_path, cfg.plots.output_dir
            )

    experiment.record(
        "evaluation",
        artifacts={
            "evaluation_dir": str(experiment.evaluation_dir),
            "combined_table": str(Path(cfg.evaluation.output_dir) / "combined_evaluation.csv"),
        },
        n_models=len(synthetic_datasets),
    )

    logger.info("Done. Combined table saved under %s", cfg.evaluation.output_dir)


if __name__ == "__main__":
    main()
