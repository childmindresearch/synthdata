"""Find approximate functional dependencies and unique column combinations with Desbordante.

Run by synthdata/dependencies.py in this environment; imports nothing from
synthdata. Reads ``--train`` (parquet, every column as text, blanks as the
string "<NA>" so they compare equal) and ``--spec`` (JSON: ``error``,
``max_lhs``, ``seed``, ``fd_columns``, ``cfd_columns``, ``cfd_minsup``,
``cfd_minconf``, ``ucc_columns``), and writes ``--out`` (parquet, one row per
finding: ``kind`` "afd", "cfd" or "ucc", ``lhs`` and ``rhs`` as JSON: column
names, or for "cfd" ``[column, value]`` pairs where a null value is "any").
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

    fd_columns = spec["fd_columns"]
    if len(fd_columns) >= 2:
        start = time.time()
        algorithm = desbordante.afd.algorithms.Pyro()
        algorithm.load_data(table=data[fd_columns])
        algorithm.execute(error=spec["error"], max_lhs=spec["max_lhs"], seed=spec["seed"])
        for fd in algorithm.get_fds():
            rows.append(
                {
                    "kind": "afd",
                    "lhs": json.dumps([fd_columns[i] for i in fd.lhs_indices]),
                    "rhs": json.dumps([fd_columns[fd.rhs_index]]),
                }
            )
        print(f"Pyro: {len(rows)} approximate FDs on {len(fd_columns)} columns in {time.time() - start:.1f}s")

    cfd_columns = spec.get("cfd_columns") or []
    if len(cfd_columns) >= 2:
        start = time.time()
        algorithm = desbordante.cfd.algorithms.FDFirst()
        algorithm.load_data(table=data[cfd_columns])
        algorithm.execute(
            cfd_minconf=spec["cfd_minconf"],
            cfd_minsup=spec["cfd_minsup"],
            cfd_max_lhs=spec["max_lhs"],
        )
        found = algorithm.get_cfds()
        for cfd in found:
            rows.append(
                {
                    "kind": "cfd",
                    "lhs": json.dumps([[cfd_columns[i.attribute], i.value] for i in cfd.lhs]),
                    "rhs": json.dumps([cfd_columns[cfd.rhs.attribute], cfd.rhs.value]),
                }
            )
        print(
            f"FDFirst: {len(found)} conditional FDs on {len(cfd_columns)} columns "
            f"in {time.time() - start:.1f}s"
        )

    ucc_columns = spec["ucc_columns"]
    if ucc_columns:
        start = time.time()
        algorithm = desbordante.ucc.algorithms.HyUCC()
        algorithm.load_data(table=data[ucc_columns])
        algorithm.execute()
        found = algorithm.get_uccs()
        for ucc in found:
            rows.append(
                {
                    "kind": "ucc",
                    "lhs": json.dumps([ucc_columns[i] for i in ucc.indices]),
                    "rhs": json.dumps([]),
                }
            )
        print(f"HyUCC: {len(found)} unique column combinations on {len(ucc_columns)} columns in {time.time() - start:.1f}s")

    pd.DataFrame(rows, columns=["kind", "lhs", "rhs"]).to_parquet(args.out, index=False)


if __name__ == "__main__":
    main()
