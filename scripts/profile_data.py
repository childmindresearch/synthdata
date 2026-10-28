#!/usr/bin/env python
"""CLI: profile the data before modelling: missingness patterns and dependencies.

Loads and splits the data only (no imputation) and writes, to the imputation
output folder:

* ``missing_blocks.csv`` (synthdata.missing_blocks): columns missing together
  and what kind of gap each pattern looks like (skip logic found by a
  decision tree).
* ``dependencies.csv`` (synthdata.dependencies): functional and conditional
  dependencies (skip logic found by one-condition rules), hierarchies,
  identifying sets and numeric redundancy, plus the full tables in
  ``dependencies_*.parquet``. Needs Desbordante (for identifying sets):
  ``uv sync --project envs/desbordante`` (Linux; WSL on Windows). Without it
  the command stops with an error unless ``--blocks-only`` is given.
* ``skip_logic_comparison.csv``: the two skip-logic views side by side.

All advisory. Rows with a ``check_rule`` can be kept in a rules file for
``evaluation.dependency_rules``.

Usage:
    synthdata-profile --config configs/config_sim.yaml [--blocks-only]
"""

import argparse

from synthdata.config import load_config
from synthdata.data import load_dataset
from synthdata.dependencies import add_report_arguments, report_options
from synthdata.generation.sidecar import resolve_env_dir
from synthdata.missing_blocks import save_blocks_report
from synthdata.utils import get_logger, set_global_seed

logger = get_logger("profile_data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    add_report_arguments(parser)
    parser.add_argument(
        "--blocks-only",
        action="store_true",
        help="Write only missing_blocks.csv (no Desbordante needed).",
    )
    args = parser.parse_args()
    if not args.blocks_only and not (resolve_env_dir(args.env_dir) / ".venv").exists():
        parser.error(
            f"Desbordante is not set up: run `uv sync --project {args.env_dir}` (Linux; WSL "
            "on Windows), or pass --blocks-only for the missing-blocks report alone."
        )

    cfg = load_config(args.config)
    set_global_seed(cfg.seed)
    dataset = load_dataset(cfg)
    blocks = save_blocks_report(cfg, dataset)
    if args.blocks_only:
        return
    from synthdata.dependencies import (
        COMPARISON_FILENAME,
        compare_skip_logic,
        save_dependencies_report,
    )
    from synthdata.experiment import imputation_output_dir

    dependencies = save_dependencies_report(
        cfg,
        dataset,
        **report_options(args),
    )
    comparison = compare_skip_logic(blocks, dependencies)
    path = imputation_output_dir(cfg) / COMPARISON_FILENAME
    comparison.to_csv(path, index=False)
    logger.info(
        "Skip logic found by: %s. See %s.", comparison["found_by"].value_counts().to_dict(), path
    )


if __name__ == "__main__":
    main()
