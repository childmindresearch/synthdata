"""Unit tests for the dependencies report (redundant columns, hierarchies, identifying sets)."""

import shutil

import numpy as np
import pandas as pd
import pytest

from synthdata.dependencies import (
    BLANK,
    DERIVE,
    MERGE_HIERARCHY,
    PRIVACY_REVIEW,
    REVIEW,
    SKIP_LOGIC,
    fd_candidate_columns,
    g3_error,
    load_rules,
    make_rule,
    numeric_redundancy,
    rule_check_rows,
    rule_error,
    rule_violations,
    score_cfds,
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


def _skip(frame, seed=2):
    """Add a gate question and an item that is blank whenever the answer is "no"."""
    rng = np.random.default_rng(seed)
    out = frame.copy()
    out["use"] = rng.choice(["no", "yes"], len(out))
    out["freq"] = np.where(out["use"] == "no", np.nan, rng.integers(1, 4, len(out)).astype(float))
    return out


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
    train = _skip(train)
    fds, cfds, uccs = run_desbordante(
        train,
        ["major", "sub", "ward", "use", "freq"],
        ["visit_id", "ward"],
        "envs/desbordante",
        2,
        0.01,
        0,
        cfd_minsup=50,
    )
    assert (["sub"], "major") in fds
    assert ["visit_id"] in uccs
    table = score_cfds(train, train, cfds, fds)
    assert "use=no" in set(table["lhs"])
    assert table.set_index("lhs").loc["use=no", "action"] == SKIP_LOGIC
    # One-column left sides search chunk pairs; "sub" and "major" sit in different chunks.
    chunked, _, _ = run_desbordante(
        train,
        ["sub", "ward", "use", "freq", "major"],
        [],
        "envs/desbordante",
        1,
        0.01,
        0,
        chunk_size=2,
    )
    assert (["sub"], "major") in chunked


def test_wide_left_sides_are_refused_on_many_columns():
    from synthdata.dependencies import MAX_WIDE_LHS_COLUMNS, run_desbordante

    columns = [f"c{i}" for i in range(MAX_WIDE_LHS_COLUMNS + 1)]
    with pytest.raises(ValueError, match="max_lhs 2"):
        run_desbordante(pd.DataFrame(columns=columns), columns, [], "envs/desbordante", 2, 0.01, 0)


def test_conditional_rules_find_skip_logic_and_drop_restated_fds():
    train, holdout = _skip(_programs()), _skip(_programs(400, seed=1), seed=3)
    candidates = [
        ([("use", "no")], ("freq", BLANK)),  # skip logic
        ([("sub", "C1")], ("major", "C")),  # restates sub -> major
        ([("use", "yes")], ("ward", "w1")),  # not a rule
        ([("sub", None)], ("major", None)),  # plain FD, reported elsewhere
    ]
    table = score_cfds(train, holdout, candidates, fds=[(["sub"], "major")])
    assert table["lhs"].tolist() == ["use=no"]
    assert table.loc[0, "rhs"] == f"freq={BLANK}"
    assert table.loc[0, "action"] == SKIP_LOGIC
    assert table.loc[0, "support"] == (train["use"] == "no").sum()


def test_rule_error_matches_two_and_two_point_zero_and_skips_absent_columns():
    frame = pd.DataFrame({"x": [1, 1, 2], "y": [2.0, 2.0, 3.0]})
    rule = make_rule([["x", "1"]], ["y", "2"])
    assert rule_error(frame, rule) == (2, 0.0)
    assert np.isnan(rule_error(frame.drop(columns="y"), rule)[1])


def test_rules_file_and_evaluation_check(tmp_path):
    from synthdata.missing_blocks import block_rules

    real = _skip(_programs(400, seed=1))
    rules = pd.DataFrame(
        {
            "kind": ["fd", "cfd", "block", "numeric"],
            "lhs": ["sub", "use=no", "a", "a"],
            "rhs": ["major", "freq", "", "total"],
            "check_rule": [
                make_rule([["sub", None]], ["major", None]),
                make_rule([["use", "no"]], ["freq", BLANK]),
                block_rules(["a", "b"]),
                "",
            ],
            "accept": [True, True, False, True],
        }
    )
    path = tmp_path / "rules.csv"
    rules.to_csv(path, index=False)
    loaded = load_rules(path)
    assert loaded["kind"].tolist() == ["fd", "cfd"]

    good = real.copy()
    bad = real.copy()
    bad["major"] = bad["major"].sample(frac=1, random_state=0).to_numpy()  # breaks sub -> major
    violations = rule_violations(loaded, {"holdout": real, "good": good, "bad": bad})
    assert violations["holdout"].max() == 0 and violations["good"].max() == 0
    assert violations.loc[0, "bad"] > 0.5
    rows = {r["model"]: r for r in rule_check_rows(violations, "holdout", ["good", "bad"])}
    assert rows["good"]["status"] == "pass" and rows["bad"]["status"] == "warn"


def test_skip_logic_comparison_shows_which_method_found_what():
    from synthdata.dependencies import compare_skip_logic

    blocks = pd.DataFrame(
        {
            "pattern": ["block_1", "lab", "freq"],
            "columns": ["form_x; form_y", "lab", "freq"],
            "label": ["form_not_given_candidate", "item_missing", "skip_logic_candidate"],
            "rule": ["", "", "|--- use <= 0.5"],
            "rule_auc": [0.5, 0.5, 1.0],
        }
    )
    dependencies = pd.DataFrame(
        {
            "lhs": ["use=no", "ward=w9", "sub"],
            "rhs": [f"freq={BLANK}", f"form_y={BLANK}", "major"],
            "action": [SKIP_LOGIC, SKIP_LOGIC, MERGE_HIERARCHY],
        }
    )
    table = compare_skip_logic(blocks, dependencies).set_index("columns")
    assert table.loc["freq", "found_by"] == "both"
    assert table.loc["freq", "cfd_rules"] == f"use=no -> freq={BLANK}"
    assert table.loc["form_y", "found_by"] == "cfd only"
    assert table.loc["form_y", "blocks_label"] == "form_not_given_candidate"
    assert len(table) == 2
