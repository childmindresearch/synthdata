#!/usr/bin/env python
"""CLI: profile the data before modelling: missingness patterns and dependencies.

Loads and splits the data only (no imputation) and writes, to the imputation
output folder:

* ``missing_blocks.csv`` (always; synthdata.missing_blocks): columns missing
  together and what kind of gap each pattern looks like.
* ``dependencies.csv`` (when ``envs/desbordante`` is set up with
  ``uv sync --project envs/desbordante``; Linux or macOS; synthdata.dependencies):
  functional and conditional dependencies, hierarchies, identifying sets and
  near-redundant numeric columns.

Both are advisory. Rows with a ``check_rule`` can be kept in a rules file for
``evaluation.dependency_rules``.

Usage:
    synthdata-profile --config configs/config_sim.yaml
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
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.seed)
    dataset = load_dataset(cfg)
    save_blocks_report(cfg, dataset)
    if not (resolve_env_dir(args.env_dir) / ".venv").exists():
        logger.warning(
            "Skipping dependencies.csv: set up Desbordante first with "
            "`uv sync --project %s` (Linux or macOS; WSL on Windows).",
            args.env_dir,
        )
        return
    from synthdata.dependencies import save_dependencies_report

    save_dependencies_report(
        cfg,
        dataset,
        env_dir=args.env_dir,
        max_lhs=args.max_lhs,
        max_error=args.max_error,
        cfd_min_support=args.cfd_min_support,
    )


if __name__ == "__main__":
    main()
