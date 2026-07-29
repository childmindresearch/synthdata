"""Unit tests for the explicit scope of resumable HPO artifacts."""

from pathlib import Path

import pytest

from synthdata.generation.hpo import default_best_params_path, default_storage_url

pytestmark = pytest.mark.unit


def test_default_hpo_artifacts_live_inside_the_experiment_directory(tmp_path):
    experiment_dir = tmp_path / "output" / "dataset" / "synthetic_data" / "v2" / "exp-1"

    storage_url = default_storage_url(experiment_dir)

    assert storage_url == f"sqlite:///{experiment_dir / 'optuna_studies.db'}"
    assert default_best_params_path(experiment_dir) == experiment_dir / "hpo_best_params.json"
    assert Path(experiment_dir).exists()
