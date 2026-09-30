from hashlib import sha256

import numpy as np
import pandas as pd
import pytest

from synthdata.evaluation.release import transform_release_roles
from synthdata.evaluation.tstr import compute_equalized_odds, run_tstr_evaluation


def _frame(rows, role, source, release=False):
    frame = pd.DataFrame(rows)
    frame.attrs["release_provenance"] = {
        "release_form": release,
        "role": role,
        "source_role": source,
        "protocol_version": "test",
        "digest": "test-digest",
        "role_hash": "test-role-hash",
        "common_protocol_digest": "test-common",
    }
    return frame


def test_rejects_non_release_synthetic_input():
    synthetic = _frame([{"x": 0, "y": 0}], "train", "synthetic")
    real = _frame([{"x": 0, "y": 0}], "tuning", "tuning")
    with pytest.raises(ValueError, match="release-form"):
        run_tstr_evaluation(synthetic, real, target_column="y")


def test_missing_class_preserves_real_support():
    synthetic, roles, _ = transform_release_roles(
        pd.DataFrame([{"x": 0, "y": 0}, {"x": 1, "y": 0}]),
        {"tuning": pd.DataFrame([{"x": 0, "y": 0}, {"x": 1, "y": 1}])},
    )
    real = roles["tuning"]
    result = run_tstr_evaluation(synthetic, real, target_column="y")
    assert result.report["state"] == "indeterminate"
    assert result.report["class_supports"] == {"0": 1, "1": 1}
    assert result.report["result_metadata"]["support_policy"]["protected_slice_floor"] == 1


def test_unexpected_classes_are_stably_ordered():
    synthetic, roles, _ = transform_release_roles(
        pd.DataFrame(
            [
                {"x": 0, "y": "known"},
                {"x": 1, "y": "zebra"},
                {"x": 2, "y": "ant"},
                {"x": 3, "y": "middle"},
            ]
        ),
        {"tuning": pd.DataFrame([{"x": 0, "y": "known"}])},
    )
    real = roles["tuning"]

    reports = [run_tstr_evaluation(synthetic, real, target_column="y").report for _ in range(5)]

    assert all(
        report["unexpected_synthetic_classes"] == ["ant", "middle", "zebra"] for report in reports
    )


def test_equalized_odds_known_ovr_fixture():
    y_true = [0, 0, 1, 1, 0, 0, 1, 1]
    y_pred = [0, 1, 1, 1, 0, 0, 0, 1]
    protected = pd.DataFrame({"group": ["a", "a", "a", "a", "b", "b", "b", "b"]})
    result = compute_equalized_odds(
        y_true,
        y_pred,
        protected,
        target_classes=[0, 1],
        prediction_artifact=_artifact(y_pred, y_true, protected),
    )
    assert result["state"] == "complete"
    assert result["macro_valid_slice_score"] == pytest.approx(0.5)
    assert result["worst_valid_gap"] == pytest.approx(0.5)


def test_equalized_odds_nested_multiclass_multigroup_aggregation():
    y_true = [0, 1, 2, 0, 0, 1, 2, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 2, 1, 0, 0, 1, 1, 0, 0, 1, 1]
    protected = pd.DataFrame(
        {"group": ["a", "a", "a", "a", "b", "b", "b", "b", "c", "c", "c", "c"]}
    )

    result = compute_equalized_odds(
        y_true,
        y_pred,
        protected,
        target_classes=[0, 1, 2],
        prediction_artifact=_artifact(y_pred, y_true, protected),
    )

    assert result["state"] == "complete"
    assert result["macro_valid_slice_score"] == pytest.approx(4 / 9)
    assert result["worst_valid_gap"] == pytest.approx(0.5)
    assert result["aggregation"] == (
        "macro valid protected slices within target class, then macro valid target classes"
    )
    assert [item["state"] for item in result["slices"]] == ["valid", "valid", "valid"]
    assert [item["valid_group_count"] for item in result["slices"]] == [3, 3, 2]
    assert result["slices"][2]["valid_group_count"] == 2
    assert [item["group"] for item in result["slices"][0]["group_rates"]] == ["a", "b", "c"]
    assert [item["tpr"] for item in result["slices"][0]["group_rates"]] == pytest.approx(
        [0.5, 1.0, 0.5]
    )


def test_equalized_odds_invalid_slice_is_not_zero():
    y_true = [0, 1]
    protected = pd.DataFrame({"group": ["a", "a"]})
    result = compute_equalized_odds(
        y_true,
        [0, 1],
        protected,
        min_support=2,
        prediction_artifact=_artifact([0, 1], y_true, protected),
    )
    assert result["state"] == "indeterminate"
    assert result["macro_valid_slice_score"] is None
    assert result["slices"][0]["state"] == "invalid"
    assert result["slices"][0]["invalid_groups"][0]["group"] == "a"
    assert result["slices"][0]["invalid_groups"][0]["positive_support"] == 1


def _artifact(predictions, y_true, protected):
    columns = list(protected.columns)
    values = protected.astype(object).where(protected.notna(), "<missing>")
    payload = {
        "target": list(y_true),
        "protected_columns": columns,
        "protected": values.to_dict("records"),
    }
    return {
        "source_role": "final_holdout",
        "prediction_source": "tstr_model",
        "verified": True,
        "prediction_length": len(predictions),
        "prediction_identity": sha256(repr(list(predictions)).encode()).hexdigest(),
        "population_role": "final_holdout",
        "population_length": len(y_true),
        "target_column": "y",
        "protected_columns": columns,
        "population_identity": sha256(repr(payload).encode()).hexdigest(),
    }


def test_equalized_odds_rejects_tuning_and_unbound_predictions():
    with pytest.raises(ValueError, match="final_holdout"):
        compute_equalized_odds(
            [0, 1],
            [0, 1],
            pd.DataFrame({"group": ["a", "b"]}),
            prediction_artifact={"source_role": "tuning", "verified": True},
        )
    protected = pd.DataFrame({"group": ["a", "b"]})
    with pytest.raises(ValueError, match="bound"):
        compute_equalized_odds(
            [0, 1], [1, 1], protected, prediction_artifact=_artifact([0, 1], [0, 1], protected)
        )


def test_equalized_odds_rejects_same_length_unbound_population():
    y_true = [0, 1]
    protected = pd.DataFrame({"group": ["a", "b"]})
    with pytest.raises(ValueError, match="population"):
        compute_equalized_odds(
            [1, 0], [0, 1], protected, prediction_artifact=_artifact([0, 1], y_true, protected)
        )


def test_equalized_odds_rejects_common_digest_mismatch():
    synthetic, roles, _ = transform_release_roles(
        pd.DataFrame([{"x": 0, "y": 0}, {"x": 1, "y": 1}]),
        {"final_holdout": pd.DataFrame([{"x": 0, "y": 0}, {"x": 1, "y": 1}])},
    )
    synthetic.attrs["release_provenance"]["common_protocol_digest"] = "synthetic"
    real = roles["final_holdout"]
    real.attrs["release_provenance"]["common_protocol_digest"] = "real"
    with pytest.raises(ValueError, match="share release provenance"):
        run_tstr_evaluation(synthetic, real, target_column="y", evaluation_role="final_holdout")


@pytest.mark.parametrize(
    ("evaluation_role", "source_role", "expected_role"),
    [
        ("tuning", "final_holdout", "tuning"),
        ("final_holdout", "tuning", "final_holdout"),
    ],
)
def test_run_rejects_real_frame_from_other_evaluation_role(
    evaluation_role, source_role, expected_role
):
    synthetic, roles, _ = transform_release_roles(
        pd.DataFrame([{"x": 0, "y": 0}, {"x": 1, "y": 1}]),
        {
            "tuning": pd.DataFrame([{"x": 0, "y": 0}, {"x": 1, "y": 1}]),
            "final_holdout": pd.DataFrame([{"x": 0, "y": 0}, {"x": 1, "y": 1}]),
        },
    )

    with pytest.raises(ValueError, match=f"{expected_role} provenance"):
        run_tstr_evaluation(
            synthetic,
            roles[source_role],
            target_column="y",
            evaluation_role=evaluation_role,
        )


def test_run_is_deterministic_and_uses_synthetic_schema(monkeypatch):
    synthetic, roles, _ = transform_release_roles(
        pd.DataFrame([{"x": 0, "category": "a", "y": "no"}, {"x": 1, "category": "b", "y": "yes"}]),
        {
            "tuning": pd.DataFrame(
                [{"x": 0, "extra": 9, "y": "no"}, {"x": 1, "extra": 8, "y": "yes"}]
            )
        },
    )
    real = roles["tuning"]
    synthetic.attrs["release_provenance"]["common_protocol_digest"] = "same"
    real.attrs["release_provenance"]["common_protocol_digest"] = "same"
    seen = []

    class Model:
        def fit(self, x, y):
            seen.append((len(x), tuple(x.columns), tuple(y)))
            return self

        def predict_proba(self, x):
            return pd.DataFrame([[1.0, 0.0], [0.0, 1.0]]).to_numpy()

        def predict(self, x):
            return pd.Series([0, 1]).to_numpy()

    monkeypatch.setattr("synthdata.evaluation.tstr._xgb", lambda seed, classes: Model())
    first = run_tstr_evaluation(synthetic, real, target_column="y")
    second = run_tstr_evaluation(synthetic, real, target_column="y")
    assert first.report == second.report
    assert seen == [(2, ("x", "category_a", "category_b", "category_nan"), (0, 1))] * 2


def test_run_sanitizes_forbidden_release_category_feature_names(monkeypatch):
    synthetic, roles, _ = transform_release_roles(
        pd.DataFrame(
            [
                {"category": "<18", "group": "a", "y": 0},
                {"category": "[$40,000", "group": "b", "y": 1},
                {"category": "age[1]", "group": "a", "y": 0},
                {"category": "plain", "group": "b", "y": 1},
            ]
        ),
        {
            "tuning": pd.DataFrame(
                [
                    {"category": "<18", "group": "a", "y": 0},
                    {"category": "[$40,000", "group": "b", "y": 1},
                    {"category": "age[1]", "group": "a", "y": 0},
                    {"category": "plain", "group": "b", "y": 1},
                ]
            )
        },
    )
    real = roles["tuning"]
    seen: list[tuple[str, ...]] = []

    class Model:
        def fit(self, x, y):
            seen.append(tuple(x.columns))
            return self

        def predict_proba(self, x):
            seen.append(tuple(x.columns))
            return pd.DataFrame([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]).to_numpy()

        def predict(self, x):
            return pd.Series([0, 1, 0, 1]).to_numpy()

    monkeypatch.setattr("synthdata.evaluation.tstr._xgb", lambda seed, classes: Model())
    result = run_tstr_evaluation(synthetic, real, target_column="y", protected_columns=["group"])

    assert result.report["state"] == "complete"

    assert seen[0] == seen[1]
    assert all(not any(char in name for char in "[]<") for name in seen[0])
    assert "category_plain" in seen[0]
    assert "group_a" not in seen[0]


def test_final_holdout_eo_uses_raw_non_numeric_labels():
    synthetic, roles, _ = transform_release_roles(
        pd.DataFrame([{"x": 0, "y": "cat"}, {"x": 1, "y": "dog"}]),
        {
            "final_holdout": pd.DataFrame(
                [{"x": 0, "group": "a", "y": "cat"}, {"x": 1, "group": "b", "y": "dog"}]
            )
        },
    )
    real = roles["final_holdout"]
    result = run_tstr_evaluation(
        synthetic,
        real,
        target_column="y",
        evaluation_role="final_holdout",
        protected_columns=["group"],
    )
    assert result.report["state"] == "complete"
    assert result.report["result_metadata"]["support_policy"]["protected_slice_floor"] == 1
    assert result.report["equalized_odds"]["support"]["protected_slice_floor"] == 1


def test_tuning_does_not_compute_equalized_odds():
    synthetic, roles, _ = transform_release_roles(
        pd.DataFrame([{"x": 0, "group": "a", "y": 0}, {"x": 1, "group": "b", "y": 1}]),
        {"tuning": pd.DataFrame([{"x": 0, "group": "a", "y": 0}, {"x": 1, "group": "b", "y": 1}])},
    )
    real = roles["tuning"]
    result = run_tstr_evaluation(
        synthetic, real, target_column="y", protected_columns=["group"], protected_slice_floor=3
    )
    assert "equalized_odds" not in result.report
    assert result.report["result_metadata"]["support_policy"]["protected_slice_floor"] == 3


@pytest.mark.parametrize(
    ("targets", "groups", "floor", "state", "valid_groups"),
    [
        ([0, 1, 0, 1], ["a", "a", "b", "b"], 1, "complete", 2),
        ([0, 1, 0, 1], ["a", "a", "b", "b"], 2, "indeterminate", 0),
        ([0, 0, 1, 1] * 2, ["a"] * 4 + ["b"] * 4, 2, "complete", 2),
        ([0, 1, 0, 0], ["a", "a", "b", "b"], 1, "indeterminate", 1),
        ([0, 1], ["a", "a"], 1, "indeterminate", 1),
    ],
)
def test_final_tstr_support_floor_uses_actual_labels(
    monkeypatch, targets, groups, floor, state, valid_groups
):
    frame = pd.DataFrame({"x": range(len(targets)), "y": targets, "group": groups})
    synthetic, roles, _ = transform_release_roles(frame, {"final_holdout": frame})

    class Model:
        def fit(self, x, y):
            return self

        def predict(self, x):
            # No predicted positives for class 1: support must still use actual labels.
            return np.zeros(len(x), dtype=int)

        def predict_proba(self, x):
            return np.tile([0.75, 0.25], (len(x), 1))

    monkeypatch.setattr("synthdata.evaluation.tstr._xgb", lambda seed, classes: Model())
    result = run_tstr_evaluation(
        synthetic,
        roles["final_holdout"],
        target_column="y",
        evaluation_role="final_holdout",
        protected_columns=["group"],
        protected_slice_floor=floor,
    )
    fairness = result.report["equalized_odds"]
    artifact = dict(result.report["prediction_artifact"])
    artifact["prediction_length"] += 1
    with pytest.raises(ValueError, match="immutable content binding"):
        compute_equalized_odds(
            roles["final_holdout"]["y"],
            result.predictions,
            roles["final_holdout"][["group"]],
            prediction_artifact=artifact,
        )
    assert fairness["state"] == state
    assert fairness["support"]["protected_slice_floor"] == floor
    assert result.report["result_metadata"]["support_policy"]["protected_slice_floor"] == floor
    assert all(item["valid_group_count"] == valid_groups for item in fairness["slices"])
    if state == "complete":
        assert fairness["macro_valid_slice_score"] == 0.0
        assert all(
            rate["positive_support"] >= floor and rate["negative_support"] >= floor
            for item in fairness["slices"]
            for rate in item["group_rates"]
        )
    else:
        assert fairness["macro_valid_slice_score"] is None
        assert all(item["reason"] == "fewer_than_two_valid_groups" for item in fairness["slices"])


@pytest.mark.parametrize("floor", [True, False, 0, -1, 1.5, "2", None])
def test_tstr_rejects_non_positive_integer_support_floor(floor):
    with pytest.raises(ValueError, match="protected_slice_floor must be a positive integer"):
        run_tstr_evaluation(
            pd.DataFrame(), pd.DataFrame(), target_column="y", protected_slice_floor=floor
        )


def test_indeterminate_tstr_records_floor_without_eo():
    synthetic, roles, _ = transform_release_roles(
        pd.DataFrame({"x": [0], "y": [0]}),
        {"final_holdout": pd.DataFrame({"x": [0, 1], "y": [0, 1]})},
    )
    result = run_tstr_evaluation(
        synthetic,
        roles["final_holdout"],
        target_column="y",
        evaluation_role="final_holdout",
        protected_slice_floor=4,
    )
    assert result.report["state"] == "indeterminate"
    assert result.report["result_metadata"]["support_policy"]["protected_slice_floor"] == 4
    assert "equalized_odds" not in result.report
