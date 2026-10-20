"""Unit tests for synthdata.generation.tabularargn_backend (no mostlyai-engine needed)."""

import stat
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from synthdata.generation import tabularargn_backend as argn
from synthdata.generation.class_quota import class_quotas

pytestmark = pytest.mark.unit


def _train():
    return pd.DataFrame(
        {
            "age": [30.5, 41.0, 52.25, 63.0, 25.0, 38.0],
            "visits": np.array([1, 2, 3, 4, 5, 6], dtype="int64"),
            "smoker": [0, 1, 0, 1, 1, 0],
            "site": ["a", "b", "a", "c", "b", "a"],
            "target": [0, 1, 2, 0, 1, 0],
        }
    )


def test_encoding_types_mark_codes_and_target_categorical():
    enc = argn.encoding_types(list(_train()), ["smoker", "site"], "target", True)
    assert enc == {
        "age": "TABULAR_NUMERIC_AUTO",
        "visits": "TABULAR_NUMERIC_AUTO",
        "smoker": "TABULAR_CATEGORICAL",
        "site": "TABULAR_CATEGORICAL",
        "target": "TABULAR_CATEGORICAL",
    }
    assert argn.encoding_types(["y"], [], "y", False) == {"y": "TABULAR_NUMERIC_AUTO"}


def test_engine_frame_round_trip_restores_values_and_dtypes():
    train = _train()
    enc = argn.encoding_types(list(train), ["smoker", "site"], "target", True)
    engine_df, maps = argn.to_engine_frame(train, enc)
    assert engine_df["smoker"].tolist() == ["0", "1", "0", "1", "1", "0"]
    # The engine returns categories as strings and numbers as floats.
    returned = engine_df.copy()
    returned["visits"] = returned["visits"].astype(float) + 0.4
    back = argn.from_engine_frame(returned[list(reversed(train.columns))], train, maps)
    assert list(back.columns) == list(train.columns)
    assert back["visits"].tolist() == [1, 2, 3, 4, 5, 6]
    for col in ["smoker", "site", "target", "visits", "age"]:
        assert back[col].dtype == train[col].dtype, col
    pd.testing.assert_frame_equal(back, train)


def test_from_engine_frame_rejects_unknown_categories():
    train = _train()
    enc = argn.encoding_types(list(train), ["site"], "target", True)
    engine_df, maps = argn.to_engine_frame(train, enc)
    engine_df.loc[0, "site"] = "_RARE_"
    with pytest.raises(ValueError, match="not in train"):
        argn.from_engine_frame(engine_df, train, maps)


def test_generate_fixes_target_to_class_quotas(mocker):
    train = _train()
    calls = []

    def fake_run(engine_df, encodings, **kw):
        calls.append(kw)
        labels = kw["labels"]
        out = engine_df.sample(n=len(labels), replace=True, random_state=0).reset_index(drop=True)
        out["target"] = [str(v) for v in labels]
        return out

    mocker.patch.object(argn, "run_tabularargn", side_effect=fake_run)
    prior = train["target"].value_counts(normalize=True)
    df = argn.generate_tabularargn(
        train,
        ["smoker", "site"],
        "target",
        True,
        12,
        7,
        "cpu",
        "envs/tabularargn",
        class_prior=prior,
    )
    assert len(calls) == 1 and calls[0]["seed"] == 7
    counts = df["target"].value_counts().sort_index()
    pd.testing.assert_series_equal(
        counts, class_quotas(prior, 12).sort_index(), check_names=False, check_dtype=False
    )
    assert not any(df.attrs["class_sampling"].shortfall.values())


# The fake `uv` is a shebang script, which Windows cannot execute.
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shebang stand-in for uv")
def test_run_tabularargn_runs_the_env_script(tmp_path, monkeypatch):
    """The subprocess contract: train parquet + spec in, synthetic parquet out."""
    env = tmp_path / "env"
    env.mkdir()
    (env / argn.RUNNER).write_text(
        textwrap.dedent(
            """
            import argparse, json
            import pandas as pd
            p = argparse.ArgumentParser()
            for a in ("--train", "--spec", "--out"):
                p.add_argument(a)
            a = p.parse_args()
            spec = json.load(open(a.spec))
            print("fitting")
            df = pd.read_parquet(a.train).head(spec["n_samples"])
            df.to_parquet(a.out, index=False)
            """
        )
    )
    # Stands in for `uv run --project <env> --no-sync python <script> ...`.
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        f"#!{sys.executable}\nimport os, sys\n"
        f"os.execv({sys.executable!r}, [{sys.executable!r}] + sys.argv[6:])\n"
    )
    fake_uv.chmod(fake_uv.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(argn.shutil, "which", lambda name: str(fake_uv))
    out = argn.run_tabularargn(_train(), {}, seed=1, device="cpu", env_dir=env, n_samples=3)
    assert len(out) == 3

    (env / argn.RUNNER).write_text("import sys\nprint('boom')\nsys.exit(3)\n")
    with pytest.raises(RuntimeError, match="exit 3"):
        argn.run_tabularargn(_train(), {}, seed=1, device="cpu", env_dir=env, n_samples=3)
