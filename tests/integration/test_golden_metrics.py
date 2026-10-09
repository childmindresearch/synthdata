"""Golden run: today's metrics on the fixture against a stored reference.

The other integration tests check properties (ranges, orderings, no leakage),
which survive a change that quietly moves every number. This test compares
every raw metric of the baseline run (see ``RUNS`` in ``conftest.py``) to
``fixtures/golden/baseline_run_<platform>_<device>.csv`` within a tolerance, so a dependency bump or a
refactor that changes what a metric returns shows up as a list of drifted
metrics.

Opt-in (marker ``regression``): run it before a release and after updating
dependencies or the submodules, not in the fast CI. Numbers differ a little
between CPUs and library builds, hence the tolerances. Torch-trained
generators and metrics (CTGAN, synthcity's MLP and OneClass embedding) differ
far more between CPU and CUDA or between operating systems, so each platform
and device keeps its own reference; on one without a reference the test is
skipped until one is recorded. The canonical reference is Linux with an NVIDIA
GPU (``baseline_run_linux_cuda.csv``), where the pipeline is meant to run;
``baseline_run_linux_cpu.csv`` is kept for machines without a GPU.

    uv run --with catboost==1.2.10 pytest tests/integration -m regression

When a change is intended, review the drift list, then rewrite the reference
with ``SYNTHDATA_UPDATE_GOLDEN=1`` and commit it with the change.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.regression]


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


REFERENCE = (
    Path(__file__).parent / "fixtures" / "golden" / f"baseline_run_{sys.platform}_{_device()}.csv"
)

#: (absolute, relative) tolerance: |now - ref| <= atol + rtol * |ref|.
BASELINE_TOLERANCE = (0.02, 0.05)
GENERATOR_TOLERANCE = (0.10, 0.25)


def _raw_metrics(combined: pd.DataFrame) -> pd.DataFrame:
    raw = combined[[c for c in combined.columns if c[0] != "__all__" and c[2] != "rank"]]
    raw.columns = ["/".join(c) for c in raw.columns]
    return raw.sort_index().sort_index(axis=1)


def test_metrics_match_the_golden_run(pipeline_run):
    now = _raw_metrics(pipeline_run.combined())
    if os.environ.get("SYNTHDATA_UPDATE_GOLDEN"):
        REFERENCE.parent.mkdir(parents=True, exist_ok=True)
        now.to_csv(REFERENCE, float_format="%.6g")
        pytest.skip(f"reference rewritten: {REFERENCE}")
    if not REFERENCE.exists():
        pytest.skip(
            f"no golden reference for this platform yet ({REFERENCE.name}); record one with "
            "SYNTHDATA_UPDATE_GOLDEN=1 on a known-good commit"
        )
    reference = pd.read_csv(REFERENCE, index_col=0)

    assert list(now.index) == list(reference.index), "models differ from the golden run"
    added = sorted(set(now.columns) - set(reference.columns))
    removed = sorted(set(reference.columns) - set(now.columns))
    assert not added and not removed, f"metrics added {added}, removed {removed}"

    drift = []
    for model in reference.index:
        atol, rtol = BASELINE_TOLERANCE if model.startswith("baseline_") else GENERATOR_TOLERANCE
        for metric in reference.columns:
            ref = float(reference.at[model, metric])
            value = float(now.at[model, metric])
            if np.isnan(ref) and np.isnan(value):
                continue
            if np.isnan(ref) != np.isnan(value) or abs(value - ref) > atol + rtol * abs(ref):
                drift.append(f"{model} {metric}: {ref:.4g} -> {value:.4g}")
    assert not drift, f"{len(drift)} metric(s) drifted from the golden run:\n" + "\n".join(drift)
