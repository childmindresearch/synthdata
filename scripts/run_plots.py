#!/usr/bin/env python
"""CLI: (re)generate every figure from already-computed artifacts on disk.

Useful for re-plotting without re-running expensive imputation/generation
stages. Controlled by ``plots.sections`` in the config (``data``, ``imputation``,
``generation``, ``hpo``, ``evaluation``).

The ``data``/``imputation`` sections describe the dataset itself and are saved
under ``plots.output_dir/<dataset-version>/dataset/`` (shared across
experiments for that version only). The ``generation``/``hpo``/``evaluation``
sections are experiment-specific and are nested under
``plots.output_dir/<dataset-version>/<experiment_id>/``, resolved the same way as
`synthdata-evaluate` (most recent experiment, or ``--experiment-id``).

The "evaluation" section is artifact-only: rank plots are redrawn from the
combined evaluation CSV and log-disparity reports are rebuilt from the
evaluation artifact bundle. Native SynthEval diagnostics are created during
evaluation and verified here; this command never reruns evaluation metrics.

Usage:
    synthdata-plot --config configs/config.yaml [--experiment-id ID] [--dataset-version v2]
"""

import argparse
from pathlib import Path

import pandas as pd

from synthdata.config import load_config
from synthdata.data import load_dataset, load_imputed_splits
from synthdata.experiment import dataset_plots_dir, load_experiment
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
    parser.add_argument(
        "--dataset-version",
        default=None,
        help="Override data.version and select that version's artifact lineage.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.experiment_id:
        cfg.experiment.id = args.experiment_id
    if args.dataset_version:
        cfg.data.version = args.dataset_version
    set_global_seed(cfg.seed)
    sections = set(cfg.plots.sections)
    logger.info("Plotting sections: %s", sorted(sections))

    dataset = load_dataset(cfg)
    dataset = load_imputed_splits(dataset)

    if "data" in sections:
        from synthdata.plotting.data_plots import save_data_plots

        save_data_plots(dataset, dataset_plots_dir(cfg), cfg.plots.dpi, cfg.plots.formats)

    if "imputation" in sections and dataset.full_imputed_df is not None:
        from synthdata.imputation import build_validation_report
        from synthdata.plotting.imputation_plots import save_imputation_plots

        validation_df = build_validation_report(cfg, dataset)
        save_imputation_plots(cfg, dataset, validation_df, dataset_plots_dir(cfg))

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

    if "evaluation" in sections:
        from synthdata.evaluation.artifacts import (
            load_log_disparity_reports,
            verify_native_syntheval_artifacts,
        )
        from synthdata.evaluation.combine import load_combined_table
        from synthdata.plotting.evaluation_plots import (
            save_log_disparity_plots,
            save_rank_tradeoff_plots,
        )

        combined_path = Path(cfg.evaluation.output_dir) / "combined_evaluation.csv"
        if not combined_path.exists():
            raise FileNotFoundError(
                f"Evaluation table not found at {combined_path}. "
                "Run `synthdata-evaluate --config <path>` first."
            )
        combined = load_combined_table(str(combined_path))
        log_disparity_reports = load_log_disparity_reports(cfg.evaluation.output_dir)
        save_rank_tradeoff_plots(cfg, combined, cfg.plots.output_dir)
        save_log_disparity_plots(log_disparity_reports, cfg.plots.output_dir)
        verify_native_syntheval_artifacts(cfg.evaluation.output_dir)

        if cfg.evaluation.generate_report:
            from synthdata.evaluation.report import save_evaluation_report

            save_evaluation_report(
                cfg,
                dataset,
                combined,
                {
                    "selected_datasets": synthetic_datasets,
                    "log_disparity_reports": log_disparity_reports,
                },
                experiment,
            )

    if experiment is not None:
        experiment.record(
            "plots", artifacts={"plots_dir": str(experiment.plots_dir)}, sections=sorted(sections)
        )

    logger.info("Done. Figures saved under %s", cfg.plots.output_dir)


if __name__ == "__main__":
    main()
