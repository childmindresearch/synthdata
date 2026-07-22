#!/usr/bin/env python
"""CLI: generate synthetic data (synthcity + TabPFN + TabPFGen), with optional
Optuna hyperparameter optimization.

Every run is tracked as a new, timestamped "experiment" (see
:mod:`synthdata.experiment`); pass ``--tag`` to label it or ``--experiment-id``
to resume/extend a specific past one. `synthdata-evaluate`/`synthdata-plot`
automatically pick up the most recent experiment unless told otherwise.

Usage:
    synthdata-generate --config configs/config.yaml [--plot] [--tag baseline]

Requires imputed data (run `synthdata-impute` first).
"""

import argparse

from synthdata.config import load_config
from synthdata.data import load_dataset, load_imputed_splits
from synthdata.experiment import start_experiment
from synthdata.generation import run_generation
from synthdata.generation.pipeline import needs_imputed_data
from synthdata.utils import get_logger, set_global_seed

logger = get_logger("run_generation")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic data via synthcity + TabPFN + TabPFGen, with optional Optuna HPO."
    )
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save real-vs-synthetic + HPO plots inline as models complete.",
    )
    parser.add_argument(
        "--tag", default=None, help="Freeform label for this experiment (overrides experiment.tag)."
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Resume/extend a specific past experiment instead of starting a new one "
        "(overrides experiment.id).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.tag:
        cfg.experiment.tag = args.tag
    if args.experiment_id:
        cfg.experiment.id = args.experiment_id
    set_global_seed(cfg.seed)

    dataset = load_dataset(cfg)
    dataset = load_imputed_splits(dataset)
    if dataset.train_imputed_df is None and needs_imputed_data(cfg.generation):
        raise SystemExit("No imputed data found. Run `synthdata-impute --config <path>` first.")

    experiment = start_experiment(cfg)
    # Nest this run's artifacts under the experiment id (also relocates HPO
    # storage/best-params-cache defaults, which derive from output_dir).
    cfg.generation.output_dir = str(experiment.generation_dir)
    cfg.plots.output_dir = str(experiment.plots_dir)

    plot_callback = None
    if args.plot:
        import matplotlib.pyplot as plt

        from synthdata.plotting import save_matplotlib_figure
        from synthdata.plotting.generation_plots import plot_real_vs_synthetic

        def plot_callback(name, df, extra):  # noqa: ARG001 - extra unused, kept for interface symmetry
            real_df = (
                dataset.train_imputed_df
                if dataset.train_imputed_df is not None
                else dataset.train_df
            )
            fig = plot_real_vs_synthetic(
                real_df,
                df,
                dataset.feature_columns + [dataset.target_column],
                dataset.all_categorical_columns,
            )
            save_matplotlib_figure(
                fig,
                f"{cfg.plots.output_dir}/generation/{name}",
                cfg.plots.dpi,
                cfg.plots.formats,
            )
            plt.close(fig)

    synthetic_datasets = run_generation(
        cfg, dataset, plot_callback=plot_callback, experiment=experiment
    )

    if args.plot:
        from synthdata.plotting.generation_plots import save_hpo_plots

        save_hpo_plots(cfg, cfg.plots.output_dir)

    experiment.record(
        "generation",
        artifacts={
            "synthetic_data_dir": str(experiment.generation_dir),
            "models": sorted(synthetic_datasets),
        },
        n_models=len(synthetic_datasets),
    )

    logger.info(
        "Generated/loaded %d synthetic datasets under %s",
        len(synthetic_datasets),
        cfg.generation.output_dir,
    )
    logger.info(
        "Experiment id: %s (pass --experiment-id %s to `synthdata-evaluate`/`synthdata-plot` "
        "to target it explicitly, otherwise it is used automatically as the latest experiment).",
        experiment.id,
        experiment.id,
    )


if __name__ == "__main__":
    main()
