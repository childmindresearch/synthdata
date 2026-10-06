import math
import sys
import types

import pandas as pd
import pytest

from synthdata.evaluation import release as release_module
from synthdata.evaluation.release import (
    attribute_disclosure,
    closest_record_distance,
    epsilon_identifiability,
    equivalence_classes,
    full_record_mia,
    k_anonymity,
    l_diversity,
    release_privacy_evidence,
    transform_release,
    transform_release_roles,
)


def _generalization():
    return {
        "columns": {
            "age": {
                "intervals": [
                    {"lower": None, "upper": 18, "label": "<18"},
                    {"lower": 18, "upper": 65, "label": "18-64"},
                    {"lower": 65, "upper": None, "label": "65+"},
                ]
            }
        }
    }


def test_transform_boundaries_and_stable_digest():
    frame = pd.DataFrame({"age": [17, 18, 64, 65], "secret": ["a", "b", "a", "b"]})
    first, metadata = transform_release(frame, _generalization(), role="train")
    second, metadata2 = transform_release(frame, _generalization(), role="train")
    assert first.age.tolist() == ["<18", "18-64", "18-64", "65+"]
    assert metadata["digest"] == metadata2["digest"]
    assert metadata["role_hash"]


def test_equivalence_k_and_l_fail_without_deletion():
    frame = pd.DataFrame({"qi": ["a", "a", "b"], "s1": [1, 1, 2], "s2": ["x", "x", "y"]})
    assert equivalence_classes(frame, ["qi"])[("a",)] == [0, 1]
    k = k_anonymity(frame, ["qi"])
    diversity = l_diversity(frame, ["qi"], ["s1", "s2"])
    assert k["k_observed"] == 1 and k["rows_suppressed"] == 0
    assert diversity["l_observed"] == 1 and diversity["rows_suppressed"] == 0
    assert diversity["l_observed_by_field"] == {"s1": 1, "s2": 1}
    assert diversity["safety_score_by_field"] == {"s1": 0.5, "s2": 0.5}


def test_structural_scores_reject_non_policy_thresholds():
    frame = pd.DataFrame({"qi": [1, 1], "s": ["a", "b"]})
    with pytest.raises(ValueError):
        k_anonymity(frame, ["qi"], required=4)
    with pytest.raises(ValueError):
        l_diversity(frame, ["qi"], ["s"], required=3)


def test_dcr_invalid_baseline_is_indeterminate():
    frame = pd.DataFrame({"x": [1, 2]})
    result = closest_record_distance(frame, frame, frame)
    assert result["status"] == "indeterminate"
    assert math.isnan(result["score"])
    assert result["threshold_policy"] == "audit_only_no_pass_fail"
    assert result["required_ratio"] is None


def test_epsilon_excess_clamps_negative_and_requires_ten_draws():
    result = epsilon_identifiability([0.1] * 10, [0.2] * 10)
    assert result["positive_excess"] == 0
    assert "not differential-privacy epsilon" in result["interpretation"]
    assert epsilon_identifiability([0.1], [0.2])["status"] == "indeterminate"


def test_role_transform_attaches_common_provenance():
    frame = pd.DataFrame({"age": [18], "value": [1]})
    synthetic, roles, metadata = transform_release_roles(
        frame, {"train": frame.copy()}, _generalization()
    )
    assert synthetic.attrs["release_provenance"]["role"] == "release"
    assert roles["train"].attrs["release_provenance"]["role"] == "train"
    assert (
        synthetic.attrs["release_provenance"]["common_protocol_digest"]
        == metadata["common_protocol_digest"]
    )
    assert (
        roles["train"].attrs["release_provenance"]["common_protocol_digest"]
        == metadata["common_protocol_digest"]
    )


def test_canonical_release_rejects_patient_id_when_argument_is_omitted():
    frame = pd.DataFrame({"patient_id": ["p1"], "age": [42]})

    with pytest.raises(ValueError, match="patient ID cannot be present"):
        transform_release_roles(frame, {"tuning": frame.copy()})


def test_canonical_release_rejects_explicit_patient_id_column():
    frame = pd.DataFrame({"pid": ["p1"], "age": [42]})

    with pytest.raises(ValueError, match="patient ID cannot be present"):
        transform_release_roles(frame, {"tuning": frame.copy()}, patient_id_column="pid")


def test_canonical_release_accepts_clean_model_frames():
    frame = pd.DataFrame({"age": [42], "value": [1]})

    synthetic, roles, _ = transform_release_roles(frame, {"tuning": frame.copy()})

    assert list(synthetic.columns) == ["age", "value"]
    assert list(roles["tuning"].columns) == ["age", "value"]


def test_release_privacy_emits_complete_supported_release_evidence_envelope():
    frame = pd.DataFrame(
        {
            "qi": ["a"] * 20,
            "secret": ["x", "y"] * 10,
        }
    )
    synthetic, roles, _ = transform_release_roles(frame, {"tuning": frame.copy()})
    result = release_privacy_evidence(
        synthetic,
        roles["tuning"],
        quasi_identifiers=["qi"],
        sensitive_fields=["secret"],
        role_population_floor=20,
    )

    assert result["status"] == "succeeded"
    assert result["protocol_version"] == "release-evidence-v2"
    assert result["fit_roles"] == ["train"]
    assert (
        result["release_transform_digest"]
        == synthetic.attrs["release_provenance"]["release_transform_digest"]
    )
    assert result["support"]["state"] == "valid"
    assert set(result["role_hashes"]) == {"synthetic", "tuning"}
    assert set(result["population_identity"]) == {"synthetic", "tuning"}


def test_release_privacy_rejects_incomplete_support():
    frame = pd.DataFrame({"qi": ["a"] * 2, "secret": ["x", "y"]})
    synthetic, roles, _ = transform_release_roles(frame, {"tuning": frame.copy()})
    result = release_privacy_evidence(
        synthetic,
        roles["tuning"],
        quasi_identifiers=["qi"],
        sensitive_fields=["secret"],
        role_population_floor=20,
    )

    assert result["status"] == "indeterminate"
    assert "support floor" in result["invalid_reasons"][0]


def test_release_privacy_does_not_persist_exception_body(monkeypatch):
    sentinel = "SENTINEL_RAW_EXCEPTION_BODY /private/patient/path"
    frame = pd.DataFrame({"qi": ["a"] * 20, "secret": ["x", "y"] * 10})
    synthetic, roles, _ = transform_release_roles(frame, {"tuning": frame.copy()})

    def fail(_frame):
        raise ValueError(sentinel)

    monkeypatch.setattr(release_module, "_validate_frame_provenance", fail)
    result = release_privacy_evidence(
        synthetic,
        roles["tuning"],
        quasi_identifiers=["qi"],
        sensitive_fields=["secret"],
    )

    assert result["status"] == "indeterminate"
    assert result["invalid_reasons"] == ["release_provenance_invalid"]
    assert result["error_types"] == ["ValueError"]
    assert sentinel not in repr(result)


def test_attribute_disclosure_does_not_persist_exception_body(monkeypatch):
    sentinel = "SENTINEL_RAW_EXCEPTION_BODY /private/patient/path"

    class Classifier:
        def __init__(self, **kwargs):
            pass

        def fit(self, x, y):
            raise ValueError(sentinel)

    monkeypatch.setitem(
        sys.modules,
        "xgboost",
        types.SimpleNamespace(XGBClassifier=Classifier, XGBRegressor=Classifier),
    )
    frame = pd.DataFrame({"qi": [1, 2], "secret": ["a", "b"]})
    result = attribute_disclosure(
        frame,
        frame,
        ["qi"],
        ["secret"],
        sensitive_types={"secret": "categorical"},
    )

    assert result["status"] == "indeterminate"
    assert result["invalid_targets"] == ["secret"]
    assert result["reason"] == "no required target has finite valid risk"
    assert result["overall_state"] == "all_targets_invalid"
    assert sentinel not in repr(result)


def test_epsilon_frame_runner_draws_half_size_independently():
    members = pd.DataFrame({"x": range(8)})
    nonmembers = pd.DataFrame({"x": range(20, 28)})
    calls = []

    def risk(member_draw, nonmember_draw, seed):
        calls.append(
            (
                len(member_draw),
                len(nonmember_draw),
                seed,
                tuple(member_draw.index),
                tuple(nonmember_draw.index),
            )
        )
        return float(member_draw.x.mean()), float(nonmember_draw.x.mean())

    result = epsilon_identifiability(members, nonmembers, risk_function=risk)
    assert result["status"] == "succeeded"
    assert len(calls) == 10 and {call[0] for call in calls} == {4}
    assert [call[2] for call in calls] == list(range(10))
    assert calls[0][3:] != calls[1][3:]


def test_mia_requires_disjoint_groups_and_reports_bootstrap_and_unsupported_slice():
    synthetic = pd.DataFrame({"x": [0.0, 1.0]})
    members = pd.DataFrame({"x": [0.0, 0.1]})
    nonmembers = pd.DataFrame({"x": [2.0, 2.1]})
    with pytest.raises(ValueError, match="patient ID"):
        full_record_mia(
            synthetic.assign(pid=[1, 2]),
            members,
            nonmembers,
            patient_id_column="pid",
            member_groups=["a", "b"],
            nonmember_groups=["c", "d"],
        )
    synthetic, roles, metadata = transform_release_roles(
        synthetic, {"train_tuning": members, "final_holdout": nonmembers}
    )
    members, nonmembers = roles["train_tuning"], roles["final_holdout"]
    common = metadata["common_protocol_digest"]
    result = full_record_mia(
        synthetic,
        members,
        nonmembers,
        member_groups=["a", "b"],
        nonmember_groups=["c", "d"],
        protected_member=["small", "small"],
        protected_nonmember=["small", "large"],
        protected_slice_floor=5,
        expected_common_protocol_digest=common,
    )
    assert result["patient_cluster_bootstrap"]["resamples"] == 1000
    assert result["protected_slices"]["small"]["status"] == "indeterminate"


def test_mia_protected_membership_and_external_anchor():
    synthetic = pd.DataFrame({"x": [0.0] * 12})
    members = pd.DataFrame({"x": [0.0] * 12})
    nonmembers = pd.DataFrame({"x": [1.0] * 12})
    synthetic, roles, metadata = transform_release_roles(
        synthetic, {"train_tuning": members, "final_holdout": nonmembers}
    )
    groups = [f"m{i}" for i in range(12)]
    result = full_record_mia(
        synthetic,
        roles["train_tuning"],
        roles["final_holdout"],
        member_groups=groups,
        nonmember_groups=[f"n{i}" for i in range(12)],
        protected_member=["A"] * 6 + ["B"] * 6,
        protected_nonmember=["A"] * 6 + ["B"] * 6,
        protected_slice_floor=1,
        seeds=(0, 1),
        expected_common_protocol_digest=metadata["common_protocol_digest"],
    )
    slice_result = result["protected_slices"]["A"]
    assert len(slice_result["seed_results"]) == 2
    assert {"ovr_auc", "ovr_effective_auc_advantage", "ovr_gap"} <= slice_result.keys()
    assert slice_result["member_support"] == slice_result["nonmember_support"] == 6


def test_mia_rejects_missing_or_wrong_external_anchor():
    frame = pd.DataFrame({"x": [0.0, 1.0]})
    synthetic, roles, metadata = transform_release_roles(
        frame, {"train_tuning": frame.copy(), "final_holdout": frame.copy()}
    )
    with pytest.raises(ValueError, match="trust anchor"):
        full_record_mia(
            synthetic,
            roles["train_tuning"],
            roles["final_holdout"],
            member_groups=["a", "b"],
            nonmember_groups=["c", "d"],
        )
    with pytest.raises(ValueError, match="trust anchor"):
        full_record_mia(
            synthetic,
            roles["train_tuning"],
            roles["final_holdout"],
            expected_common_protocol_digest="bad",
            member_groups=["a", "b"],
            nonmember_groups=["c", "d"],
        )


def test_disclosure_missing_xgboost_is_indeterminate(monkeypatch):
    monkeypatch.setitem(sys.modules, "xgboost", None)
    frame = pd.DataFrame({"qi": [1, 2], "secret": ["a", "b"]})
    result = attribute_disclosure(
        frame, frame, ["qi"], ["secret"], sensitive_types={"secret": "categorical"}
    )
    assert result["status"] == "indeterminate"
    assert "fallback" in result["reason"]


def test_disclosure_string_missing_and_continuous_baseline(monkeypatch):
    class Classifier:
        def __init__(self, **kwargs):
            pass

        def fit(self, x, y):
            self.classes_ = sorted(set(y))
            return self

        def predict(self, x):
            return [0] * len(x)

    class Regressor:
        def __init__(self, **kwargs):
            pass

        def fit(self, x, y):
            return self

        def predict(self, x):
            return [1.0] * len(x)

    monkeypatch.setitem(
        sys.modules,
        "xgboost",
        types.SimpleNamespace(XGBClassifier=Classifier, XGBRegressor=Regressor),
    )
    synthetic = pd.DataFrame(
        {"qi": [0, 1, 2, 3], "cat": ["a", "b", None, "a"], "value": [0.0, 1.0, 2.0, 3.0]}
    )
    holdout = pd.DataFrame(
        {"qi": [0, 1, 2, 3], "cat": ["a", None, "b", "a"], "value": [0.0, 1.0, 2.0, 3.0]}
    )
    result = attribute_disclosure(
        synthetic,
        holdout,
        ["qi"],
        ["cat", "value"],
        sensitive_types={"cat": "categorical", "value": "continuous"},
    )
    assert result["status"] == "succeeded"
    assert result["worst_sensitive_target"] in {"cat", "value"}
    assert math.isfinite(result["worst_sensitive_risk"])


def test_mia_rejects_final_holdout_reference_role():
    frame = pd.DataFrame({"x": [0.0, 1.0]})
    synthetic, roles, metadata = transform_release_roles(
        frame, {"train_tuning": frame.copy(), "final_holdout_reference": frame.copy()}
    )
    with pytest.raises(ValueError, match="roles"):
        full_record_mia(
            synthetic,
            roles["train_tuning"],
            roles["final_holdout_reference"],
            member_groups=["a", "b"],
            nonmember_groups=["c", "d"],
            expected_common_protocol_digest=metadata["common_protocol_digest"],
        )


def test_disclosure_mixed_and_all_invalid_states(monkeypatch):
    class Regressor:
        def __init__(self, **kwargs):
            pass

        def fit(self, x, y):
            return self

        def predict(self, x):
            return [1.0] * len(x)

    class Classifier:
        def __init__(self, **kwargs):
            pass

        def fit(self, x, y):
            return self

        def predict(self, x):
            return [0] * len(x)

    monkeypatch.setitem(
        sys.modules,
        "xgboost",
        types.SimpleNamespace(XGBClassifier=Classifier, XGBRegressor=Regressor),
    )
    synthetic = pd.DataFrame(
        {"qi": [0, 1, 2], "valid": [0.0, 1.0, 2.0], "invalid": [1.0, 1.0, 1.0]}
    )
    holdout = pd.DataFrame({"qi": [0, 1, 2], "valid": [0.0, 1.0, 2.0], "invalid": [1.0, 1.0, 1.0]})
    mixed = attribute_disclosure(
        synthetic,
        holdout,
        ["qi"],
        ["valid", "invalid"],
        sensitive_types={"valid": "continuous", "invalid": "continuous"},
    )
    assert mixed["status"] == "indeterminate"
    assert mixed["overall_state"] == "mixed_invalid_targets"
    assert mixed["worst_sensitive_target"] == "valid"
    assert math.isfinite(mixed["worst_sensitive_risk"])
    all_invalid = attribute_disclosure(
        synthetic.assign(invalid2=1.0),
        holdout.assign(invalid2=1.0),
        ["qi"],
        ["invalid", "invalid2"],
        sensitive_types={"invalid": "continuous", "invalid2": "continuous"},
    )
    assert all_invalid["status"] == "indeterminate"
    assert all_invalid["overall_state"] == "all_targets_invalid"
    assert all_invalid["worst_sensitive_target"] is None
    assert math.isnan(all_invalid["worst_sensitive_risk"])
