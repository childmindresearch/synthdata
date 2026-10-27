#!/usr/bin/env python
"""CLI: report redundant columns, hierarchies and identifying column sets.

Loads and splits the data only, runs Desbordante in its own environment (set
it up once with ``uv sync --project envs/desbordante``; Linux or macOS, WSL on
Windows) and writes ``dependencies.csv`` to the imputation output folder (see
synthdata.dependencies). Advisory: nothing is changed.

Usage:
    synthdata-dependencies --config configs/config_sim.yaml [--max-lhs 2] [--max-error 0.01]
"""

import argparse

import pandas as pd

from synthdata.config import load_config
from synthdata.data import load_dataset
from synthdata.dependencies import save_dependencies_report
from synthdata.utils import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--env-dir", default="envs/desbordante", help="Desbordante environment.")
    parser.add_argument("--max-lhs", type=int, default=2, help="Most columns on the left side.")
    parser.add_argument(
        "--max-error", type=float, default=0.01, help="Largest share of rows breaking a rule."
    )
    parser.add_argument(
        "--min-rows-per-value",
        type=float,
        default=5,
        help="Fewest train rows per left-side value for a rule to count.",
    )
    parser.add_argument(
        "--min-r2", type=float, default=0.95, help="Smallest R² for numeric redundancy."
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
    table = save_dependencies_report(
        cfg,
        load_dataset(cfg),
        env_dir=args.env_dir,
        max_lhs=args.max_lhs,
        max_error=args.max_error,
        min_rows_per_value=args.min_rows_per_value,
        min_r2=args.min_r2,
        cfd_min_support=args.cfd_min_support,
    )
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
