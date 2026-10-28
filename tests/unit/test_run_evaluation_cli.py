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
    cfg = types.SimpleNamespace(
        generation=types.SimpleNamespace(
            output_dir=str(tmp_path), hpo=types.SimpleNamespace(best_params_path=None)
        )
    )

    datasets = _load_synthetic_datasets(cfg, reference)

    assert list(datasets) == ["ctgan"]


def test_tuned_sets_of_a_search_with_no_passing_trial_are_skipped(tmp_path):
    reference = pd.DataFrame({"x": [1.0, 2.0], "y": ["a", "b"]})
    for name in ("ctgan__rep1", "ctgan_hpo__rep1", "tvae__rep1", "tvae_hpo__rep1"):
        reference.to_csv(tmp_path / f"{name}.csv", index=False)
    # ctgan's search found no passing trial (an empty entry); tvae's did.
    (tmp_path / "hpo_best_params.json").write_text(
        '{"synthcity": {"ctgan": {}, "tvae": {"lr": 1}}}'
    )
    cfg = types.SimpleNamespace(
        generation=types.SimpleNamespace(
            output_dir=str(tmp_path), hpo=types.SimpleNamespace(best_params_path=None)
        )
    )

    datasets = _load_synthetic_datasets(cfg, reference)

    assert sorted(datasets) == ["ctgan__rep1", "tvae__rep1", "tvae_hpo__rep1"]
