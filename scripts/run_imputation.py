#!/usr/bin/env python
"""CLI: load a dataset and run TabImpute-based imputation.

Usage:
    synthdata-impute --config configs/config.yaml [--plot] [--dataset-version v2]
"""

import argparse

from synthdata.config import load_config
from synthdata.data import load_dataset
from synthdata.imputation import build_validation_report, run_imputation
from synthdata.utils import get_logger, set_global_seed

logger = get_logger("run_imputation")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a dataset and run TabImpute-based missing-data imputation."
    )
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument(
        "--plot", action="store_true", help="Also save data + imputation QA plots."
    )
    parser.add_argument(
        "--dataset-version",
        default=None,
        help="Override data.version from the config (e.g. 'v2', '2024-06-01'). "
        "Cached raw/imputed/split CSVs are nested under data_dir/<version>/, "
        "and every downstream experiment manifest records which version was used.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset_version:
        cfg.data.version = args.dataset_version
    set_global_seed(cfg.seed)

    dataset = load_dataset(cfg)
    dataset = run_imputation(cfg, dataset)

    validation_df = None
    if cfg.imputation.enabled:
        validation_df = build_validation_report(cfg, dataset)
        if len(validation_df):
            logger.info("Imputation validation report:\n%s", validation_df.to_string(index=False))

    if args.plot:
        from synthdata.plotting.data_plots import save_data_plots
        from synthdata.plotting.imputation_plots import save_imputation_plots

        save_data_plots(dataset, cfg.plots.output_dir, cfg.plots.dpi, cfg.plots.formats)
        if cfg.imputation.enabled and validation_df is not None:
            save_imputation_plots(cfg, dataset, validation_df, cfg.plots.output_dir)

    logger.info(
        "Done. Imputed data cached under %s (dataset version=%s)",
        dataset.data_dir,
        dataset.version or "unversioned",
    )


if __name__ == "__main__":
    main()
