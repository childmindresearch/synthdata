"""TabularARGN and CDTD, the generators with their own environments (``-m slow``).

They are on by default but need their environments, so CI leaves them out:

    git submodule update --init
    uv sync --project envs/tabularargn
    uv sync --project envs/cdtd
    uv run pytest tests/integration/test_sidecar_generators.py -m slow

A missing environment skips the tests, or fails them when
``SYNTHDATA_REQUIRE_SIDECARS=1`` (the scheduled run, scripts/scheduled_tests.sh),
so a broken install cannot pass as skipped.
"""

from __future__ import annotations

import os
import subprocess
import textwrap

import numpy as np
import pandas as pd
import pytest

from synthdata.generation import cdtd_backend, sidecar

from .conftest import new_run, run_full_pipeline

pytestmark = [pytest.mark.integration, pytest.mark.slow]

ENV = sidecar.REPO_ROOT / "envs" / "cdtd"
ARGN_ENV = sidecar.REPO_ROOT / "envs" / "tabularargn"
SOURCE = sidecar.REPO_ROOT / "submodules" / "cdtd_simple"
STEPS = {"num_steps_train": 200, "num_steps_warmup": 10, "batch_size": 128}
SAMPLE_STEPS = 10

# cdtd_simple's main.ipynb, step for step (preprocessing, CDTD(), fit, sample,
# inverse transforms), with the steps above and the notebook's seed 42. Like
# our runner, it also seeds CDTD()'s weight initialisation, which the notebook
# leaves unseeded (so two notebook runs differ).
REFERENCE = """
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer, StandardScaler

sys.path.insert(0, sys.argv[1])
from cdtd import CDTD
from cdtd.utils import set_seeds

torch.set_float32_matmul_precision("high")
df = pd.read_parquet(sys.argv[2])
cat_features = ["site", "flag", "target"]
cont_features = [c for c in df.columns if c not in cat_features]
X_cat = df[cat_features].to_numpy().astype("str")
X_cont = df[cont_features].to_numpy().astype("float")
cat_int_enc = OrdinalEncoder()
X_cat_enc = cat_int_enc.fit_transform(X_cat)
cont_enc = QuantileTransformer(
    output_distribution="normal",
    n_quantiles=max(min(X_cont.shape[0] // 30, 1000), 10),
    subsample=int(1e9),
    random_state=42,
)
X_cont_enc = cont_enc.fit_transform(X_cont)
cont_std = StandardScaler()
X_cont_enc_std = cont_std.fit_transform(X_cont_enc)
X_cat_train = torch.tensor(X_cat_enc).long()
X_cont_train = torch.tensor(X_cont_enc_std).float()
set_seeds(42)
cdtd = CDTD(X_cat_train, X_cont_train)
cdtd.fit(X_cat_train, X_cont_train, num_steps_train=200, num_steps_warmup=10, batch_size=128)
X_cat_gen, X_cont_gen = cdtd.sample(num_samples=int(sys.argv[3]), num_steps=10)
X_cat_gen = cat_int_enc.inverse_transform(X_cat_gen)
X_cont_gen = cont_enc.inverse_transform(cont_std.inverse_transform(X_cont_gen))
out = pd.DataFrame(X_cat_gen, columns=cat_features)
for i, c in enumerate(cont_features):
    out[c] = X_cont_gen[:, i]
out.to_parquet(sys.argv[4], index=False)
"""


def _require(ready: bool, setup: str) -> None:
    if ready:
        return
    message = f"needs its environment: {setup}"
    if os.environ.get("SYNTHDATA_REQUIRE_SIDECARS") == "1":
        pytest.fail(message, pytrace=False)
    pytest.skip(message)


@pytest.fixture(scope="module", autouse=True)
def _needs_envs():
    _require(
        (ENV / ".venv").exists() and (SOURCE / "cdtd" / "__init__.py").exists(),
        "git submodule update --init; uv sync --project envs/cdtd",
    )
    _require((ARGN_ENV / ".venv").exists(), "uv sync --project envs/tabularargn")


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "a": x,
            "b": 2 * x + rng.normal(scale=0.5, size=n),
            "age": rng.integers(20, 80, n).astype(float),
            "site": rng.choice(["x", "y", "z"], n),
            "flag": rng.integers(0, 2, n).astype(str),
        }
    )
    df["target"] = ((df["a"] > 0).astype(int) + (df["site"] == "z")).astype(str)
    return df


def test_wrapper_reproduces_the_authors_recipe(tmp_path):
    """Our runner adds nothing to the model: same seed, same rows as main.ipynb."""
    toy = _toy()
    ours = cdtd_backend.run_cdtd(
        toy,
        ["site", "flag", "target"],
        seed=42,
        n_samples=150,
        env_dir=ENV,
        source_dir=SOURCE,
        fit_params=STEPS,
        sample_params={"num_steps": SAMPLE_STEPS},
    )
    toy.to_parquet(tmp_path / "toy.parquet", index=False)
    (tmp_path / "reference.py").write_text(textwrap.dedent(REFERENCE))
    subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(ENV),
            "--no-sync",
            "python",
            str(tmp_path / "reference.py"),
            str(SOURCE),
            str(tmp_path / "toy.parquet"),
            "150",
            str(tmp_path / "ref.parquet"),
        ],
        check=True,
    )
    reference = pd.read_parquet(tmp_path / "ref.parquet")[list(toy.columns)]
    for col in ["site", "flag", "target"]:
        assert ours[col].tolist() == reference[col].tolist(), col
    np.testing.assert_allclose(
        ours[["a", "b", "age"]].to_numpy(), reference[["a", "b", "age"]].to_numpy(), rtol=1e-6
    )


#: Both generators in one run, as the default config has them; short training.
OVERRIDES = {
    "generation": {
        "synthcity": {"enabled": False},
        "hpo": {"enabled": False},
        "tabularargn": {"enabled": True, "params": {"max_epochs": 20}},
        "cdtd": {
            "enabled": True,
            "fit_params": STEPS,
            "sample_params": {"num_steps": SAMPLE_STEPS},
        },
    },
}


@pytest.fixture(scope="module")
def sidecar_run(tmp_path_factory):
    return run_full_pipeline(new_run(tmp_path_factory, "sidecars", overrides=OVERRIDES))


def test_tabularargn_meets_the_class_quotas_exactly(sidecar_run):
    # The quotas are passed to TabularARGN as fixed target values.
    frame = sidecar_run.synthetic()["tabularargn__rep1"]
    assert len(frame) == sidecar_run.cfg.generation.n_samples
    assert not frame.isna().any().any()
    report = pd.read_csv(sidecar_run.generation_dir / "diagnostics" / "class_sampling.csv")
    rows = report[report["model"] == "tabularargn__rep1"]
    assert len(rows) == 2 and (rows["filled"] == rows["quota"]).all(), rows


def test_cdtd_fills_the_quotas_from_its_pool(sidecar_run):
    # CDTD cannot condition on the label: classes are drawn from a larger pool,
    # so a class it makes too rarely may stay short, never over its quota.
    frame = sidecar_run.synthetic()["cdtd__rep1"]
    assert 0 < len(frame) <= sidecar_run.cfg.generation.n_samples
    assert not frame.isna().any().any()
    report = pd.read_csv(sidecar_run.generation_dir / "diagnostics" / "class_sampling.csv")
    rows = report[report["model"] == "cdtd__rep1"]
    assert len(rows) == 2 and (rows["filled"] <= rows["quota"]).all(), rows


def test_both_are_evaluated_and_ranked(sidecar_run):
    combined = sidecar_run.combined()
    for name in ("tabularargn__rep1", "cdtd__rep1"):
        assert name in combined.index
        assert np.isfinite(combined.loc[name, ("__all__", "overall", "rank")])
