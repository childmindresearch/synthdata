#!/usr/bin/env python
"""CLI: profile the data before modelling: missingness patterns and dependencies.

Loads and splits the data only (no imputation) and writes, to the imputation
output folder:

* ``missing_blocks.csv`` (synthdata.missing_blocks): columns missing together
  and what kind of gap each pattern looks like (skip logic found by a
  decision tree).
* ``dependencies.csv`` (synthdata.dependencies): functional and conditional
  dependencies (skip logic found by conditional FDs), hierarchies,
  identifying sets and near-redundant numeric columns. Needs Desbordante:
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
from synthdata.generation.sidecar import resolve_env_dir
from synthdata.missing_blocks import save_blocks_report
from synthdata.utils import get_logger, set_global_seed

logger = get_logger("profile_data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--env-dir", default="envs/desbordante", help="Desbordante environment.")
    parser.add_argument("--max-lhs", type=int, default=2, help="Most columns on the left side.")
    parser.add_argument(
        "--max-error", type=float, default=0.01, help="Largest share of rows breaking a rule."
    )
    parser.add_argument(
        "--cfd-min-support",
        type=float,
        default=0.05,
        help="Smallest share of rows a conditional rule must cover (0 = no conditional search).",
    )
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
        env_dir=args.env_dir,
        max_lhs=args.max_lhs,
        max_error=args.max_error,
        cfd_min_support=args.cfd_min_support,
    )
    comparison = compare_skip_logic(blocks, dependencies)
    path = imputation_output_dir(cfg) / COMPARISON_FILENAME
    comparison.to_csv(path, index=False)
    logger.info(
        "Skip logic found by: %s. See %s.", comparison["found_by"].value_counts().to_dict(), path
    )


if __name__ == "__main__":
    main()
