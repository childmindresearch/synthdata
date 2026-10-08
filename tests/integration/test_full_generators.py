"""Full-generator end-to-end run, for a GPU workstation (``-m slow``).

Enables every SynthCity plugin plus TabPFN and TabPFGen on the clinic fixture
with HPO on. Needs the ``tabpfn`` extra and TabPFN model access (a TabPFN
token in the repository's ``.env``). Not run in CI:

    uv run pytest tests/integration -m slow
"""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import new_run, run_full_pipeline

pytestmark = [pytest.mark.integration, pytest.mark.slow]

SYNTHCITY = ["ctgan", "tvae", "adsgan", "bayesian_network", "pategan", "rtvae", "ddpm"]
TABPFN = [
    "tabpfn_standard",
    "tabpfn_custom",
    "tabpfn_standard_imputed",
    "tabpfn_custom_imputed",
]
TABPFGEN = ["tabpfgen_standard", "tabpfgen_custom"]
EXPECTED = sorted(
    [*SYNTHCITY, *(f"{name}_hpo" for name in SYNTHCITY), *TABPFN, *TABPFGEN]
    + [f"{name}_hpo" for name in TABPFGEN]
)

OVERRIDES = {
    "device": "auto",
    "imputation": {"device": "auto", "refidiff": {"denoiser": "auto"}},
    "generation": {
        "synthcity": {"names": SYNTHCITY},
        "tabpfn": {
            "enabled": True,
            "variants": ["standard", "custom"],
            "data_variants": ["raw", "imputed"],
        },
        "tabpfgen": {
            "enabled": True,
            "variants": ["standard", "custom"],
            "standard_params": {},
            "custom_params": {"n_sgld_steps": 200, "sgld_noise_scale": 0.1},
        },
        "hpo": {
            "n_trials": 3,
            "epoch_ranges": {name: [10, 50, 10] for name in ("ctgan", "tvae", "adsgan", "rtvae")}
            | {"pategan": [1, 5, 1], "tabpfgen_standard": [100, 200, 100]}
            | {"tabpfgen_custom": [100, 200, 100]},
            "final_n_iter_override": 50,
        },
    },
    "evaluation": {
        "syntheval_execution": {"model_workers": "auto", "max_model_workers": 4},
        "save_per_model_syntheval_plots": True,
    },
}


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    pytest.importorskip("catboost", reason="RefiDiff imputation needs catboost")
    pytest.importorskip("tabpfn", reason="needs the tabpfn extra: uv sync --extra tabpfn")
    pytest.importorskip("tabpfgen", reason="needs the tabpfn extra: uv sync --extra tabpfn")
    return run_full_pipeline(new_run(tmp_path_factory, "full", overrides=OVERRIDES))


def test_every_generator_produced_valid_output(full_run):
    synthetic = full_run.synthetic()
    assert sorted(synthetic) == EXPECTED
    n_samples = full_run.cfg.generation.n_samples
    for name, frame in synthetic.items():
        assert len(frame) == n_samples, name
        assert not frame.isna().any().any(), name
        assert frame["target"].nunique() == 2, name


def test_every_generator_was_evaluated(full_run):
    combined = full_run.combined()
    assert sorted(m for m in combined.index if not m.startswith("baseline_")) == EXPECTED
    for dim in ("utility", "privacy", "fairness", "overall"):
        assert np.isfinite(combined[("__all__", dim, "rank")].astype(float)).all(), dim
