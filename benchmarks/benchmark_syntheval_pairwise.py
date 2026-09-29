#!/usr/bin/env python
"""Opt-in seeded serial/parallel benchmark for SynthEval pairwise metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--columns", type=int, default=100)
    parser.add_argument("--ks-large-rows", type=int, default=2126)
    parser.add_argument("--ks-large-columns", type=int, default=665)
    parser.add_argument("--n-perms", type=int, default=25)
    parser.add_argument("--num-quants", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    if min(args.rows, args.ks_large_rows) < 2:
        parser.error("row counts must be at least 2")
    if min(args.columns, args.ks_large_columns) < 2:
        parser.error("column counts must be at least 2")
    if min(args.n_perms, args.num_quants) < 1:
        parser.error("permutations and quantiles must be positive")
    if args.workers < 2:
        parser.error("workers must be at least 2 to benchmark parallel execution")
    return args


def _canonical(value):
    import numpy as np
    import pandas as pd

    if isinstance(value, pd.DataFrame):
        return {
            "index": _canonical(value.index.tolist()),
            "columns": _canonical(value.columns.tolist()),
            "values": _canonical(value.to_numpy().tolist()),
        }
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, np.ndarray):
        return _canonical(value.tolist())
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return "NaN" if np.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def _digest(value) -> str:
    encoded = json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _make_frames(rows, columns, categorical_count, seed):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    values = rng.normal(size=(rows, columns))
    labels = [f"feature_{index:04d}" for index in range(columns)]
    real = pd.DataFrame(values, columns=pd.Index(labels))
    cat_count = min(categorical_count, columns - 1)
    cat_cols = labels[:cat_count]
    for column in cat_cols:
        real[column] = [f"level_{int(value)}" for value in rng.integers(0, 5, rows)]
    synt = real.copy()
    numeric_columns = labels[cat_count:]
    if numeric_columns:
        synt[numeric_columns[0]] = synt[numeric_columns[0]] + 0.15
    if cat_cols:
        synt[cat_cols[0]] = synt[cat_cols[0]].iloc[::-1].to_numpy()
    num_cols = [column for column in labels if column not in cat_cols]
    return real, synt, num_cols, cat_cols


def _run_pair(name, run, worker_count_for_budget, worker_cap, input_shape):
    records = {}
    for mode, budget in (("serial", 1), ("bounded_parallel", worker_cap)):
        os.environ["LOKY_MAX_CPU_COUNT"] = str(budget)
        selected_workers = worker_count_for_budget(budget)
        start = time.perf_counter()
        output = run()
        elapsed = time.perf_counter() - start
        records[mode] = {
            "workers": selected_workers if mode == "bounded_parallel" else 1,
            "seconds": round(elapsed, 6),
            "sha256": _digest(output),
        }
    if records["serial"]["sha256"] != records["bounded_parallel"]["sha256"]:
        raise RuntimeError(f"{name} serial/parallel outputs differ")
    return {
        "input_shape": input_shape,
        "outputs_equal": True,
        "output_sha256": records["serial"]["sha256"],
        "timings": records,
    }


def _benchmark(args):
    import joblib
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    from syntheval.metrics.utility.metric_kolmogorov_smirnov import (
        KolmogorovSmirnovTest,
        _ks_v2_worker_count,
    )
    from syntheval.metrics.utility.metric_mixed_correlation import (
        MixedCorrelation,
    )
    from syntheval.metrics.utility.metric_mixed_correlation import (
        _v2_worker_count as corr_worker_count,
    )
    from syntheval.metrics.utility.metric_mutual_information import (
        MutualInformation,
        _mi_v2_worker_count,
    )

    def metric_run(metric_type, real, synt, num_cols, cat_cols):
        metric = metric_type(
            real,
            synt,
            cat_cols=cat_cols,
            num_cols=num_cols,
            do_preprocessing=False,
            verbose=False,
            plot_figures=False,
        )
        if metric_type is MutualInformation:
            return metric.evaluate(num_quants=args.num_quants)
        if metric_type is KolmogorovSmirnovTest:
            return metric.evaluate(n_perms=args.n_perms, random_state=args.seed)
        return metric.evaluate()

    real, synt, num_cols, cat_cols = _make_frames(
        args.rows, args.columns, max(1, args.columns // 10), args.seed
    )
    sample_count = len(real) + len(synt)
    category_count = len(cat_cols)
    results = {}

    results["corr_diff"] = _run_pair(
        "corr_diff",
        lambda: metric_run(MixedCorrelation, real, synt, num_cols, cat_cols),
        lambda budget: (
            corr_worker_count(len(real.columns) * (len(real.columns) - 1) // 2)
            if budget == args.workers
            else 1
        ),
        args.workers,
        {"rows_per_frame": args.rows, "columns": args.columns},
    )
    results["mi_diff"] = _run_pair(
        "mi_diff",
        lambda: metric_run(MutualInformation, real, synt, num_cols, cat_cols),
        lambda budget: (
            _mi_v2_worker_count(len(real.columns) * (len(real.columns) - 1) // 2)
            if budget == args.workers
            else 1
        ),
        args.workers,
        {"rows_per_frame": args.rows, "columns": args.columns},
    )

    ks_small_real, ks_small_synt, ks_small_num, ks_small_cat = _make_frames(
        args.rows, args.columns, max(1, args.columns // 10), args.seed + 1
    )
    results["ks_test_small_serial"] = _run_pair(
        "ks_test_small_serial",
        lambda: metric_run(
            KolmogorovSmirnovTest,
            ks_small_real,
            ks_small_synt,
            ks_small_num,
            ks_small_cat,
        ),
        lambda budget: (
            _ks_v2_worker_count(
                args.columns,
                sample_count,
                category_count,
                args.n_perms,
            )
            if budget == args.workers
            else 1
        ),
        args.workers,
        {
            "rows_per_frame": args.rows,
            "columns": args.columns,
            "n_perms": args.n_perms,
        },
    )

    large_real, large_synt, large_num, large_cat = _make_frames(
        args.ks_large_rows,
        args.ks_large_columns,
        min(8, max(1, args.ks_large_columns // 10)),
        args.seed + 2,
    )
    large_samples = len(large_real) + len(large_synt)
    results["ks_test_large_parallel_candidate"] = _run_pair(
        "ks_test_large_parallel_candidate",
        lambda: metric_run(
            KolmogorovSmirnovTest,
            large_real,
            large_synt,
            large_num,
            large_cat,
        ),
        lambda budget: (
            _ks_v2_worker_count(
                args.ks_large_columns,
                large_samples,
                len(large_cat),
                args.n_perms,
            )
            if budget == args.workers
            else 1
        ),
        args.workers,
        {
            "rows_per_frame": args.ks_large_rows,
            "columns": args.ks_large_columns,
            "n_perms": args.n_perms,
        },
    )

    return {
        "seed": args.seed,
        "configured_worker_cap": args.workers,
        "parameters": {
            "rows": args.rows,
            "columns": args.columns,
            "ks_large_rows": args.ks_large_rows,
            "ks_large_columns": args.ks_large_columns,
            "n_perms": args.n_perms,
            "num_quants": args.num_quants,
            "workers": args.workers,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "joblib": joblib.__version__,
            "logical_cpus": os.cpu_count(),
        },
        "data": "seeded synthetic frames generated in memory; no real data",
        "metrics": results,
    }


def main() -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"Run from repository root: {ROOT}")
    args = _parse_args()
    scratch_root = ROOT / "tmp"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="benchmark-syntheval-pairwise-", dir=scratch_root
    ) as scratch:
        os.environ["TMPDIR"] = scratch
        os.environ["JOBLIB_TEMP_FOLDER"] = scratch
        report = _benchmark(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
