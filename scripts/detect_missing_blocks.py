#!/usr/bin/env python
"""CLI: find columns that go missing together and label each missingness pattern.

Loads and splits the data only (no imputation), then writes
``missing_blocks.csv`` (see synthdata.missing_blocks) to the imputation output
folder, prints the patterns, and prints a ``missing_indicators.blocks`` snippet
for the blocks found. Labels are candidates to check against the codebook.

Usage:
    synthdata-missing-blocks --config configs/config_sim.yaml [--min-jaccard 0.9]
"""

import argparse
import json

import pandas as pd

from synthdata.config import load_config
from synthdata.data import load_dataset
from synthdata.missing_blocks import save_blocks_report
from synthdata.utils import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument(
        "--min-jaccard",
        type=float,
        default=None,
        help="Override imputation.missing_indicators.block_min_jaccard for this report.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.min_jaccard is not None:
        cfg.imputation.missing_indicators.block_min_jaccard = args.min_jaccard
    set_global_seed(cfg.seed)
    table = save_blocks_report(cfg, load_dataset(cfg))

    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(table.drop(columns=["rule"]).to_string(index=False))
    blocks = table[table["kind"] == "block"]
    if len(blocks):
        print(
            "\nTo give these blocks one missing flag each, add under "
            "imputation.missing_indicators (indicator-only columns are never in a block):"
        )
        print("  blocks:")
        for row in blocks.itertuples():
            print(f"    {row.pattern}: {json.dumps(row.columns.split('; '))}")


if __name__ == "__main__":
    main()
