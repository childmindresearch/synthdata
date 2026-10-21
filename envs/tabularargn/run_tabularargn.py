"""Fit MOSTLY AI's TabularARGN on one table and sample from it.

Runs inside envs/tabularargn (its own environment, see pyproject.toml); called
by synthdata.generation.tabularargn_backend, which writes the inputs and
reads the output. It imports nothing from synthdata.

    python run_tabularargn.py --train train.parquet --spec spec.json --out synthetic.parquet

spec.json keys: ``encoding_types`` (column -> mostlyai encoding type),
``params`` (TabularARGN() arguments), ``sample_params`` (TabularARGN.sample()
arguments), ``seed``, ``device``, and either ``n_samples`` or
``target_column`` plus ``labels`` (one target value per row to draw: the
target column is fixed to these as seed data, so each class gets exactly its
quota).
"""

import argparse
import json

import pandas as pd
from mostlyai.engine import TabularARGN


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.spec, encoding="utf-8") as fh:
        spec = json.load(fh)
    train = pd.read_parquet(args.train)

    with TabularARGN(
        tgt_encoding_types=spec["encoding_types"],
        random_state=spec["seed"],
        device=spec["device"],
        verbose=1,
        **spec["params"],
    ) as model:
        model.fit(train)
        if spec.get("labels") is not None:
            seed_data = pd.DataFrame({spec["target_column"]: spec["labels"]})
            synthetic = model.sample(seed_data=seed_data, **spec["sample_params"])
        else:
            synthetic = model.sample(n_samples=spec["n_samples"], **spec["sample_params"])
    synthetic.to_parquet(args.out, index=False)


if __name__ == "__main__":
    main()
