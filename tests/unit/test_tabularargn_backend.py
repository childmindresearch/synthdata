"""Unit tests for synthdata.generation.tabularargn_backend (no mostlyai-engine needed)."""

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
