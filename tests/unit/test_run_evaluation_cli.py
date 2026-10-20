"""The evaluate CLI's loading of generated sets."""

import types

import pandas as pd
import pytest

from scripts.run_evaluation import _load_synthetic_datasets

pytestmark = pytest.mark.unit


def test_empty_synthetic_sets_are_skipped(tmp_path):
    reference = pd.DataFrame({"x": [1.0, 2.0], "y": ["a", "b"]})
    reference.to_csv(tmp_path / "ctgan.csv", index=False)
    reference.head(0).to_csv(tmp_path / "tvae.csv", index=False)  # header only
    (tmp_path / "arf.csv").write_text("")  # no header either
    cfg = types.SimpleNamespace(generation=types.SimpleNamespace(output_dir=str(tmp_path)))

    datasets = _load_synthetic_datasets(cfg, reference)

    assert list(datasets) == ["ctgan"]
