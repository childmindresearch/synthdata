#!/usr/bin/env python
"""CLI: run an append-only train-only masked-cell RefiDiff benchmark/HPO study."""

import argparse

from synthdata.config import load_config
from synthdata.data import load_dataset
from synthdata.imputation import run_refidiff_benchmark
from synthdata.utils import get_logger, set_global_seed

logger = get_logger("run_imputation_benchmark")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a resumable, masked-cell RefiDiff benchmark on the training split."
    )
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument(
        "--study-id",
        default=None,
        help="Explicit existing study id to resume, or a new unique id to start.",
    )
    parser.add_argument(
        "--dataset-version", default=None, help="Temporary override for data.version."
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.dataset_version:
        cfg.data.version = args.dataset_version
    set_global_seed(cfg.seed)
    dataset = load_dataset(cfg)
    study_dir = run_refidiff_benchmark(cfg, dataset, args.study_id)
    logger.info("RefiDiff benchmark complete: %s", study_dir)


if __name__ == "__main__":
    main()
