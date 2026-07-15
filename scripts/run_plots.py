#!/usr/bin/env python
"""CLI: (re)generate every figure from already-computed artifacts on disk.

Useful for re-plotting without re-running expensive imputation/generation
stages. Controlled by ``plots.sections`` in the config (``data``, ``imputation``,
``generation``, ``hpo``, ``evaluation``).

The ``data``/``imputation`` sections describe the dataset itself and are saved
directly under ``plots.output_dir`` (shared across experiments). The
``generation``/``hpo``/``evaluation`` sections are experiment-specific and are
nested under ``plots.output_dir/<experiment_id>/``, resolved the same way as
`synthdata-evaluate` (most recent experiment, or ``--experiment-id``).

Note: the "evaluation" section recomputes evaluation metrics from the cached
synthetic-data CSVs (fast -- no model retraining) since the figures need the
full in-memory results (including per-model log-disparity Plotly reports),
which aren't losslessly cached to disk.

Usage:
    synthdata-plot --config configs/config.yaml [--experiment-id ID]
"""

import argparse
from pathlib import Path

import pandas as pd

from synthdata.config import load_config
from synthdata.data import load_dataset, load_imputed_splits
from synthdata.experiment import load_experiment
from synthdata.utils import get_logger, set_global_seed

logger = get_logger("run_plots")

_EXPERIMENT_SECTIONS = {"generation", "hpo", "evaluation"}


def _load_synthetic_datasets(cfg) -> dict:
    output_dir = Path(cfg.generation.output_dir)
    if not output_dir.exists():
        return {}
    return {path.stem: pd.read_csv(path) for path in sorted(output_dir.glob("*.csv"))}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate all figures for the sections listed in plots.sections."
    )
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Plot a specific past experiment's generation/hpo/evaluation figures "
        "instead of the most recent one (overrides experiment.id). Ignored if "
        "plots.sections has no experiment-specific sections.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.experiment_id:
        cfg.experiment.id = args.experiment_id
    set_global_seed(cfg.seed)
    sections = set(cfg.plots.sections)
    logger.info("Plotting sections: %s", sorted(sections))

    dataset = load_dataset(cfg)
    dataset = load_imputed_splits(dataset)

    if "data" in sections:
        from synthdata.plotting.data_plots import save_data_plots

        save_data_plots(dataset, cfg.plots.output_dir, cfg.plots.dpi, cfg.plots.formats)

    if "imputation" in sections and dataset.full_imputed_df is not None:
        from synthdata.imputation import build_validation_report
        from synthdata.plotting.imputation_plots import save_imputation_plots

        validation_df = build_validation_report(cfg, dataset)
        save_imputation_plots(cfg, dataset, validation_df, cfg.plots.output_dir)

    experiment = None
    if sections & _EXPERIMENT_SECTIONS:
        experiment = load_experiment(cfg)
        cfg.generation.output_dir = str(experiment.generation_dir)
        cfg.evaluation.output_dir = str(experiment.evaluation_dir)
        cfg.plots.output_dir = str(experiment.plots_dir)

    synthetic_datasets = _load_synthetic_datasets(cfg)

    if "generation" in sections and synthetic_datasets and dataset.train_imputed_df is not None:
        from synthdata.plotting.generation_plots import save_generation_plots

        save_generation_plots(cfg, dataset, synthetic_datasets, cfg.plots.output_dir)

    if "hpo" in sections:
        from synthdata.plotting.generation_plots import save_hpo_plots

        save_hpo_plots(cfg, cfg.plots.output_dir)

    if "evaluation" in sections and synthetic_datasets and dataset.train_imputed_df is not None:
        from synthdata.evaluation import run_evaluation
        from synthdata.plotting.evaluation_plots import (
            save_log_disparity_plots,
            save_per_model_syntheval_plots,
            save_rank_tradeoff_plots,
        )

        combined, extras = run_evaluation(cfg, dataset, synthetic_datasets)
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

    if experiment is not None:
        experiment.record("plots", artifacts={"plots_dir": str(experiment.plots_dir)}, sections=sorted(sections))

    logger.info("Done. Figures saved under %s", cfg.plots.output_dir)


if __name__ == "__main__":
    main()
