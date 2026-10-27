"""Unit tests for the dependencies report (redundant columns, hierarchies, identifying sets)."""

import shutil

import numpy as np
import pandas as pd
import pytest

from synthdata.dependencies import (
    BLANK,
    CO_MISSING,
    DERIVE,
    ID_LIKE,
    MERGE_HIERARCHY,
    NO_ACTION,
    PRIVACY_REVIEW,
    REVIEW,
    SKIP_LOGIC,
    _as_text,
    condition_rows,
    fd_rows,
    g3_error,
    load_rules,
    make_rule,
    numeric_redundancy,
    rule_check_rows,
    rule_error,
    rule_violations,
    ucc_rows,
)
from synthdata.dependency_search import single_column_search, two_column_search
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


def _search(train, holdout, **options):
    return single_column_search(_as_text(train), _as_text(holdout), **options)


CATEGORICAL = ["major", "sub", "ward", "visit_id", "use"]


def test_every_pair_is_scored_and_chance_rules_are_labelled_not_dropped():
    train, holdout = _programs(seed=0), _programs(400, seed=1)
    pairs, _ = _search(train, holdout)
    n = train.shape[1]
    assert len(pairs) == n * (n - 1)  # every ordered pair, continuous columns included
    table = fd_rows(pairs, CATEGORICAL).set_index(["lhs", "rhs"])
    assert table.loc[("sub", "major"), "action"] == MERGE_HIERARCHY
    assert table.loc[("sub", "major"), "score"] > 0.9
    # visit_id is unique, so it "fixes" ward by chance: listed, with a low score.
    assert table.loc[("visit_id", "ward"), "action"] == ID_LIKE
    assert table.loc[("visit_id", "ward"), "score"] < 0.1
    assert ("major", "ward") not in table.index


def test_near_hierarchy_with_a_few_bad_rows_is_flagged_for_review():
    train, holdout = _programs(seed=0, broken=5), _programs(400, seed=1)
    pairs, _ = _search(train, holdout)
    row = fd_rows(pairs, CATEGORICAL).set_index(["lhs", "rhs"]).loc[("sub", "major")]
    assert row["action"] == REVIEW
    assert 0 < row["train_error"] <= 0.01


def test_two_column_exact_rule_is_derive():
    train = _programs(seed=0)
    train["pair"] = train["major"] + "-" + train["ward"]
    holdout = _programs(400, seed=1)
    holdout["pair"] = holdout["major"] + "-" + holdout["ward"]
    columns = ["major", "ward", "pair", "use"]
    train, holdout = _skip(train)[columns], _skip(holdout)[columns]
    pairs, _ = _search(train, holdout)
    two = two_column_search(_as_text(train), _as_text(holdout), pairs, 0.01)
    table = fd_rows(two, columns).set_index(["lhs", "rhs"])
    assert table.loc[("major; ward", "pair"), "action"] == DERIVE
    # pair alone fixes major and ward, so pairs with it add nothing and are skipped.
    assert not any(lhs.startswith("pair") or "; pair" in lhs for lhs, _ in table.index)


def test_numeric_sum_is_redundant_and_noise_is_not():
    table = numeric_redundancy(_programs(), ["a", "b", "total"]).set_index("rhs")
    assert set(table.index) == {"a", "b", "total"}  # each is the sum or difference of the others
    assert (table["action"] == DERIVE).all()
    rng = np.random.default_rng(0)
    noise = pd.DataFrame(rng.normal(size=(300, 3)), columns=["p", "q", "r"])
    table = numeric_redundancy(noise, ["p", "q", "r"])
    assert len(table) == 3 and (table["action"] == NO_ACTION).all()  # reported, not dropped


def test_ucc_rows_keep_every_size():
    rows = ucc_rows(_programs(), [["visit_id"], ["a", "b", "major", "ward"]])
    assert rows["lhs"].tolist() == ["visit_id", "a; b; major; ward"]
    assert (rows["action"] == PRIVACY_REVIEW).all()


@pytest.mark.skipif(
    shutil.which("uv") is None or not (resolve_env_dir("envs/desbordante") / ".venv").exists(),
    reason="needs `uv sync --project envs/desbordante` (Linux or macOS)",
)
def test_hyucc_finds_the_identifier_in_its_own_environment():
    from synthdata.dependencies import run_hyucc

    uccs = run_hyucc(_programs(), ["visit_id", "ward", "major"], "envs/desbordante")
    assert uccs == [["visit_id"]]


def test_more_than_two_determining_columns_are_refused():
    from synthdata.dependencies import dependencies_table

    frame = _programs(50)
    with pytest.raises(ValueError, match="max_lhs must be 1 or 2"):
        dependencies_table(frame, frame, CATEGORICAL, [], max_lhs=3)


def test_one_condition_rules_find_skip_logic_and_ignore_rare_values():
    train, holdout = _skip(_programs()), _skip(_programs(400, seed=1), seed=3)
    train.loc[:1, "ward"] = "w9"  # two rows: a clean "rule" that chance explains
    pairs, rules = _search(train, holdout)
    table = condition_rows(rules, pairs).set_index(["lhs", "rhs"])
    row = table.loc[("use=no", f"freq={BLANK}")]
    assert row["action"] == SKIP_LOGIC
    assert row["support"] == (train["use"] == "no").sum()
    assert row["holdout_error"] == 0
    # freq blank -> use=no restates the listed FD freq -> use, so it is not repeated.
    assert (f"freq={BLANK}", "use=no") not in table.index
    assert not any(lhs.startswith("ward=w9") for lhs, _ in table.index)
    # sub=C1 -> major=C restates the listed FD sub -> major.
    assert not any(lhs.startswith("sub=") and rhs.startswith("major=") for lhs, rhs in table.index)
    assert CO_MISSING not in set(table["action"])


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
