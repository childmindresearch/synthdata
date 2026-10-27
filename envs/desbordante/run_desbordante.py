"""Find approximate functional dependencies and unique column combinations with Desbordante.

Run by synthdata/dependencies.py in this environment; imports nothing from
synthdata. Reads ``--train`` (parquet, every column as text, blanks as the
string "<NA>" so they compare equal) and ``--spec`` (JSON: ``error``,
``max_lhs``, ``chunk_size``, ``fd_columns``, ``cfd_columns``, ``cfd_minsup``,
``cfd_minconf``, ``ucc_columns``), and writes ``--out`` (parquet, one row per
finding: ``kind`` "afd", "cfd" or "ucc", ``lhs`` and ``rhs`` as JSON: column
names, or for "cfd" ``[column, value]`` pairs where a null value is "any").
"""

import argparse
import itertools
import json
import time

import desbordante
import pandas as pd


def fd_subsets(columns: list, max_lhs: int, chunk_size: int) -> list:
    """Column sets to search; together they cover every FD with up to ``max_lhs`` columns.

    Tane's memory grows with the number of columns, so with one left-side
    column the search runs on every pair of ``chunk_size`` chunks: each FD
    X -> Y lies within one such pair. Wider left sides search all columns at once.
    """
    if max_lhs > 1 or len(columns) <= 2 * chunk_size:
        return [columns]
    chunks = [columns[i : i + chunk_size] for i in range(0, len(columns), chunk_size)]
    return [
        chunks[a] + (chunks[b] if b != a else [])
        for a, b in itertools.combinations_with_replacement(range(len(chunks)), 2)
    ]


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
        found = set()
        for subset in fd_subsets(fd_columns, spec["max_lhs"], spec["chunk_size"]):
            algorithm = desbordante.afd.algorithms.Tane()
            algorithm.load_data(table=data[subset])
            algorithm.execute(error=spec["error"], max_lhs=spec["max_lhs"])
            for fd in algorithm.get_fds():
                found.add((tuple(subset[i] for i in fd.lhs_indices), subset[fd.rhs_index]))
        for lhs, rhs in sorted(found):
            rows.append({"kind": "afd", "lhs": json.dumps(list(lhs)), "rhs": json.dumps([rhs])})
        print(f"Tane: {len(found)} approximate FDs on {len(fd_columns)} columns in {time.time() - start:.1f}s")

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
