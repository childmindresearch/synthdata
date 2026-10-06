#!/usr/bin/env python
"""CLI for auditing a SynthData config and its configured dataset/schema."""

import argparse
import sys
from pathlib import Path

import yaml

from synthdata.config_audit import audit_config


def main() -> int:
    """Run the config audit and return a process exit code."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate a SynthData config, semantic policies, and configured "
            "dataset/schema without running pipeline stages."
        )
    )
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    args = parser.parse_args()

    try:
        result = audit_config(args.config)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        RuntimeError,
        AttributeError,
        yaml.YAMLError,
    ) as exc:
        print(
            f"synthdata-test: audit failed for config {Path(args.config).expanduser()}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    workers = result.syntheval_model_workers
    print(
        "synthdata-test: audit passed: "
        f"config={result.config_path} dataset={result.dataset_name} "
        f"version={result.dataset_version or 'unversioned'} rows={result.row_count} "
        f"features={len(result.feature_columns)} schema_columns={len(result.schema_columns)} "
        f"syntheval_workers={'disabled' if workers is None else workers}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
