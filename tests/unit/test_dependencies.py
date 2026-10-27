"""Unit tests for the dependencies report (redundant columns, hierarchies, identifying sets)."""

import shutil

import numpy as np
import pandas as pd
import pytest

from synthdata.dependencies import (
    DERIVE,
    MERGE_HIERARCHY,
    PRIVACY_REVIEW,
    REVIEW,
    fd_candidate_columns,
    g3_error,
    numeric_redundancy,
    score_fds,
    ucc_rows,
)
from synthdata.generation.sidecar import resolve_env_dir

pytestmark = pytest.mark.unit


def _programs(n=1200, seed=0, broken=0):
    rng = np.random.default_rng(seed)
    major = rng.choice(["A", "B", "C"], n)
    sub = np.array([f"{m}{rng.integers(0, 3)}" for m in major])
    frame = pd.DataFrame(
        {
            "major": major,
            "sub": sub,
            "ward": rng.choice(["w1", "w2"], n),
            "visit_id": np.arange(n).astype(str),
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
        }
    )
    frame["total"] = frame["a"] + frame["b"]
    frame.loc[: broken - 1, "major"] = "C" if broken else frame["major"]
    return frame


def test_g3_counts_rows_to_remove_and_treats_blanks_as_values():
    frame = pd.DataFrame({"x": ["a", "a", "a", "b"], "y": [1, 1, 2, 3]})
    assert g3_error(frame, ["x"], "y") == pytest.approx(0.25)
    blanks = pd.DataFrame({"x": [np.nan, np.nan, "b"], "y": [1, 1, 2]})
    assert g3_error(blanks, ["x"], "y") == 0


def test_hierarchy_is_found_and_spurious_rules_are_dropped():
    train, holdout = _programs(seed=0), _programs(400, seed=1)
    categorical = ["major", "sub", "ward", "visit_id"]
    candidates = [(["sub"], "major"), (["visit_id"], "ward"), (["major"], "ward")]
    table = score_fds(train, holdout, candidates, categorical).set_index("lhs")
    assert table.loc["sub", "rhs"] == "major"
    assert table.loc["sub", "action"] == MERGE_HIERARCHY
    # visit_id "determines" ward only because it is unique; major does not determine ward.
    assert list(table.index) == ["sub"]


def test_near_hierarchy_with_a_few_bad_rows_is_flagged_for_review():
    train, holdout = _programs(seed=0, broken=5), _programs(400, seed=1)
    table = score_fds(train, holdout, [(["sub"], "major")], ["major", "sub"])
    assert table.loc[0, "action"] == REVIEW
    assert 0 < table.loc[0, "train_error"] <= 0.01


def test_multi_column_exact_rule_is_derive():
    train = _programs(seed=0)
    train["pair"] = train["major"] + "-" + train["ward"]
    holdout = _programs(400, seed=1)
    holdout["pair"] = holdout["major"] + "-" + holdout["ward"]
    table = score_fds(train, holdout, [(["major", "ward"], "pair")], ["major", "ward", "pair"])
    assert table.loc[0, "action"] == DERIVE


def test_numeric_sum_is_redundant_and_noise_is_not():
    table = numeric_redundancy(_programs(), ["a", "b", "total"]).set_index("rhs")
    assert set(table.index) == {"a", "b", "total"}  # each is the sum or difference of the others
    assert (table["action"] == DERIVE).all()
    rng = np.random.default_rng(0)
    noise = pd.DataFrame(rng.normal(size=(300, 3)), columns=["p", "q", "r"])
    assert numeric_redundancy(noise, ["p", "q", "r"]).empty


def test_candidate_columns_and_ucc_rows():
    frame = _programs()
    assert fd_candidate_columns(frame, ["major", "sub", "ward", "visit_id"]) == [
        "major",
        "sub",
        "ward",
        "visit_id",
    ]
    rows = ucc_rows(frame, [["visit_id"], ["a", "b", "major", "ward"]])
    assert rows["lhs"].tolist() == ["visit_id"]
    assert rows.loc[0, "action"] == PRIVACY_REVIEW


@pytest.mark.skipif(
    shutil.which("uv") is None or not (resolve_env_dir("envs/desbordante") / ".venv").exists(),
    reason="needs `uv sync --project envs/desbordante` (Linux or macOS)",
)
def test_desbordante_finds_the_hierarchy_in_its_own_environment():
    from synthdata.dependencies import run_desbordante

    train = _programs()
    fds, uccs = run_desbordante(
        train, ["major", "sub", "ward"], ["visit_id", "ward"], "envs/desbordante", 2, 0.01, 0
    )
    assert (["sub"], "major") in fds
    assert ["visit_id"] in uccs
