"""Unit tests for synthdata.generation.sidecar."""

import stat
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from synthdata.generation import sidecar

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


def test_category_round_trip_restores_values_and_dtypes():
    train = _train()
    engine_df, maps = sidecar.encode_categories(train, ["smoker", "site", "target"])
    assert engine_df["smoker"].tolist() == ["0", "1", "0", "1", "1", "0"]
    # The engine returns categories as strings and numbers as floats.
    returned = engine_df.copy()
    returned["visits"] = returned["visits"].astype(float) + 0.4
    back = sidecar.decode_categories(returned[list(reversed(train.columns))], train, maps, "m")
    assert list(back.columns) == list(train.columns)
    assert back["visits"].tolist() == [1, 2, 3, 4, 5, 6]
    for col in ["smoker", "site", "target", "visits", "age"]:
        assert back[col].dtype == train[col].dtype, col
    pd.testing.assert_frame_equal(back, train)


def test_decode_rejects_unknown_categories():
    train = _train()
    engine_df, maps = sidecar.encode_categories(train, ["site", "target"])
    engine_df.loc[0, "site"] = "_RARE_"
    with pytest.raises(ValueError, match="not in train"):
        sidecar.decode_categories(engine_df, train, maps, "m")


# The fake `uv` is a shebang script, which Windows cannot execute.
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shebang stand-in for uv")
def test_run_in_env_runs_the_env_script(tmp_path, monkeypatch):
    """The subprocess contract: train parquet + spec in, synthetic parquet out."""
    env = tmp_path / "env"
    env.mkdir()
    (env / "run.py").write_text(
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
    monkeypatch.setattr(sidecar.shutil, "which", lambda name: str(fake_uv))
    out = sidecar.run_in_env(env, "run.py", _train(), {"n_samples": 3}, "m")
    assert len(out) == 3

    (env / "run.py").write_text("import sys\nprint('boom')\nsys.exit(3)\n")
    with pytest.raises(RuntimeError, match="exit 3"):
        sidecar.run_in_env(env, "run.py", _train(), {"n_samples": 3}, "m")
