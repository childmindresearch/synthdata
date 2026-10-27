"""Find unique column combinations with Desbordante's HyUCC.

Run by synthdata/dependencies.py in this environment; imports nothing from
synthdata. Reads ``--train`` (parquet, every column as text, blanks as the
string "<NA>" so they compare equal) and ``--spec`` (JSON: ``ucc_columns``),
and writes ``--out`` (parquet, one row per minimal unique column combination:
``kind`` "ucc", ``lhs`` the column names as JSON, ``rhs`` "[]").
"""

import argparse
import json
import time

import desbordante
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    data = pd.read_parquet(args.train)
    with open(args.spec, encoding="utf-8") as handle:
        spec = json.load(handle)
    rows = []
    columns = spec["ucc_columns"]
    if columns:
        start = time.time()
        algorithm = desbordante.ucc.algorithms.HyUCC()
        algorithm.load_data(table=data[columns])
        algorithm.execute()
        for ucc in algorithm.get_uccs():
            rows.append(
                {
                    "kind": "ucc",
                    "lhs": json.dumps([columns[i] for i in ucc.indices]),
                    "rhs": json.dumps([]),
                }
            )
        print(f"HyUCC: {len(rows)} unique column combinations on {len(columns)} columns in {time.time() - start:.1f}s")
    pd.DataFrame(rows, columns=["kind", "lhs", "rhs"]).to_parquet(args.out, index=False)


if __name__ == "__main__":
    main()
