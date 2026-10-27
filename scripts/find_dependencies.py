#!/usr/bin/env python
"""CLI: report redundant columns, hierarchies and identifying column sets.

Loads and splits the data only, scores every column pair exactly, runs
Desbordante (identifying sets) in its own environment (set
it up once with ``uv sync --project envs/desbordante``; Linux or macOS, WSL on
Windows) and writes ``dependencies.csv`` to the imputation output folder (see
synthdata.dependencies), with the full tables in ``dependencies_*.parquet``.
Advisory: nothing is changed.

Usage:
    synthdata-dependencies --config configs/config_sim.yaml [--max-lhs 1] [--max-error 0.01]
"""

import argparse

import pandas as pd

from synthdata.config import load_config
from synthdata.data import load_dataset
from synthdata.dependencies import add_report_arguments, report_options, save_dependencies_report
from synthdata.utils import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    add_report_arguments(parser)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.seed)
    table = save_dependencies_report(
        cfg,
        load_dataset(cfg),
        **report_options(args),
    )
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
