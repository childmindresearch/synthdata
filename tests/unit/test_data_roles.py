"""Tests for canonical role allocation, identity isolation, and provenance."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from synthdata.data import (
    role_context_fingerprint,
    semantic_context_digest,
    semantic_context_fingerprint,
    semantic_context_payload,
    validate_semantic_context,
)
from synthdata.data_roles import (
    ROLE_NAMES,
    TOKENIZATION_VERSION,
    DataSplitConfig,
    _opaque_group_token,
    allocate_roles,
    resolve_population_identity,
    resolve_support_constraints,
)

pytestmark = pytest.mark.unit


def test_group_roles_are_deterministic_and_exclude_direct_identity(make_canonical_dataset):
    first = make_canonical_dataset("column")
    second = make_canonical_dataset("column")

    assert set(first.roles) == {"train", "tuning", "final_holdout"}
    assert first.assignment_fingerprint == second.assignment_fingerprint
    assert first.assignment_policy_fingerprint == second.assignment_policy_fingerprint
    pd.testing.assert_frame_equal(first.assignment, second.assignment)
    for role in first.roles:
        assert "patient_id" not in first.roles[role].columns
        assert len(first.roles[role]) == len(first.role_groups[role])
        assert not any(str(value) in {"1", "2", "3"} for value in first.role_groups[role])
    assert sum(len(frame) for frame in first.roles.values()) == len(first.full_df)
    for left_role, left_groups in first.role_groups.items():
        for right_role, right_groups in first.role_groups.items():
            if left_role >= right_role:
                continue
            assert set(left_groups).isdisjoint(set(right_groups))


def test_direct_identity_is_retained_only_in_aligned_sidecar(make_canonical_dataset):
    dataset = make_canonical_dataset("column")

    assert dataset.identity_sidecar is not None
    assert dataset.identity_sidecar.name == "patient_id"
    assert dataset.identity_sidecar.tolist() == list(np.repeat(np.arange(1, 13), 2))
    assert len(dataset.identity_sidecar) == len(dataset.full_df)
    assert "patient_id" not in dataset.full_df.columns
    assert "patient_id" not in dataset.role_metadata


def test_mapping_sidecar_identity_is_removed_from_model_frames(make_canonical_dataset):
    dataset = make_canonical_dataset("mapping")

    assert dataset.role_metadata["identity"]["identity_source"] == "mapping_file"
    assert "row_id" not in dataset.full_df.columns
    assert all("row_id" not in frame.columns for frame in dataset.roles.values())
    assert all(
        len(groups) == len(dataset.roles[role]) for role, groups in dataset.role_groups.items()
    )


def test_one_row_per_patient_cannot_create_row_index_identity():
    frame = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})
    split = DataSplitConfig(mode="patient_group", one_row_per_patient=True)

    with pytest.raises(ValueError, match="not a leakage-safe identity source"):
        resolve_population_identity(frame, split)


def test_candidate_role_context_ignores_final_holdout_changes(make_canonical_dataset):
    dataset = make_canonical_dataset("column")
    candidate_before = role_context_fingerprint(dataset, ("train", "tuning"))
    full_before = role_context_fingerprint(dataset)

    dataset.imputed_roles["final_holdout"] = dataset.imputed_roles["final_holdout"].copy()
    dataset.imputed_roles["final_holdout"].loc[:, "feature"] += 1000
    dataset.assignment = dataset.assignment.copy()
    final_rows = dataset.assignment["role"].eq("final_holdout")
    dataset.assignment.loc[final_rows, "population_group_hash"] = "holdout-only-change"
    dataset.role_metadata["identity"]["identity_fingerprint"] = "holdout-only-identity-change"

    assert role_context_fingerprint(dataset, ("train", "tuning")) == candidate_before
    assert role_context_fingerprint(dataset) != full_before


def test_semantic_context_binds_declarations_and_score_policy(make_canonical_dataset):
    dataset = make_canonical_dataset("column")
    dataset.quasi_identifier_columns = ["feature"]
    dataset.variable_schema["feature"]["source_table"] = "measurements"

    payload = semantic_context_payload(dataset, classification_score="balanced_accuracy")
    baseline = semantic_context_fingerprint(dataset, classification_score="balanced_accuracy")

    assert payload["schema_version"] == "semantic-context-v1"
    assert payload["source_table"] == {"feature": "measurements"}
    assert payload["quasi_identifier_columns"] == ["feature"]
    assert payload["sensitive_columns"] == ["protected"]
    assert "identity_fingerprint" in payload
    assert semantic_context_digest(payload) == baseline

    dataset.quasi_identifier_columns = ["protected"]
    assert (
        semantic_context_fingerprint(dataset, classification_score="balanced_accuracy") != baseline
    )

    dataset.quasi_identifier_columns = ["feature"]
    assert semantic_context_fingerprint(dataset, classification_score="macro_f1") != baseline


def test_semantic_context_defaults_isolate_final_holdout_and_explicit_full_roles(
    make_canonical_dataset,
):
    dataset = make_canonical_dataset("column")
    baseline_payload = semantic_context_payload(dataset)
    baseline_digest = semantic_context_digest(baseline_payload)
    baseline_fingerprint = semantic_context_fingerprint(dataset)

    dataset.imputed_roles["final_holdout"] = dataset.imputed_roles["final_holdout"].copy()
    dataset.imputed_roles["final_holdout"].loc[:, "feature"] += 1000
    dataset.assignment = dataset.assignment.copy()
    holdout = dataset.assignment["role"].eq("final_holdout")
    dataset.assignment.loc[holdout, "population_group_hash"] = "holdout-only-token"
    dataset.role_metadata["identity"]["identity_fingerprint"] = "holdout-only-identity"

    assert semantic_context_payload(dataset) == baseline_payload
    assert semantic_context_digest(semantic_context_payload(dataset)) == baseline_digest
    assert semantic_context_fingerprint(dataset) == baseline_fingerprint

    full_payload = semantic_context_payload(dataset, roles=ROLE_NAMES)
    assert semantic_context_digest(full_payload) != baseline_digest
    assert semantic_context_fingerprint(dataset, roles=ROLE_NAMES) != baseline_fingerprint
    raw_ids = dataset.identity_sidecar.tolist()
    assert json.dumps(raw_ids) not in json.dumps(baseline_payload)
    assert json.dumps(raw_ids) not in json.dumps(full_payload)


def test_semantic_context_exposes_independent_dimension_fingerprints(make_canonical_dataset):
    dataset = make_canonical_dataset("column")
    payload = semantic_context_payload(dataset)
    expected = {
        "quasi_identifier_fingerprint",
        "sensitive_fingerprint",
        "protected_fingerprint",
        "schema_fingerprint",
        "release_generalization_fingerprint",
        "role_fingerprint",
        "identity_fingerprint",
    }
    assert expected <= payload.keys()

    baseline = semantic_context_digest(payload)
    mutations = {
        "quasi_identifier_columns": ["feature"],
        "sensitive_columns": ["feature"],
        "protected_columns": ["feature"],
        "release_generalization": {"feature": {"kind": "bin"}},
        "variable_schema": {**dataset.variable_schema, "feature": {"kind": "categorical"}},
    }
    for attribute, value in mutations.items():
        original = getattr(dataset, attribute)
        setattr(dataset, attribute, value)
        assert semantic_context_digest(semantic_context_payload(dataset)) != baseline
        setattr(dataset, attribute, original)


def test_semantic_dimension_fingerprints_change_with_each_declaration(make_canonical_dataset):
    dataset = make_canonical_dataset("column")
    baseline = semantic_context_payload(dataset)
    mutations = {
        "quasi_identifier_columns": ["feature"],
        "sensitive_columns": ["feature"],
        "protected_columns": ["feature"],
        "variable_schema": {**dataset.variable_schema, "feature": {"kind": "categorical"}},
        "release_generalization": {"feature": {"kind": "bin"}},
    }
    fields = {
        "quasi_identifier_columns": "quasi_identifier_fingerprint",
        "sensitive_columns": "sensitive_fingerprint",
        "protected_columns": "protected_fingerprint",
        "variable_schema": "schema_fingerprint",
        "release_generalization": "release_generalization_fingerprint",
    }
    for attribute, value in mutations.items():
        setattr(dataset, attribute, value)
        assert semantic_context_payload(dataset)[fields[attribute]] != baseline[fields[attribute]]
        setattr(dataset, attribute, getattr(dataset, attribute, None))
        # Restore from freshly constructed baseline to avoid aliasing mutable declarations.
        dataset = make_canonical_dataset("column")


def test_semantic_dimension_fingerprint_tampering_is_rejected(make_canonical_dataset):
    dataset = make_canonical_dataset("column")
    payload = semantic_context_payload(dataset)
    payload["protected_fingerprint"] = "tampered"
    with pytest.raises(ValueError, match="protected_fingerprint"):
        validate_semantic_context(
            payload,
            target_column=dataset.target_column,
            feature_columns=dataset.feature_columns,
            categorical_columns=dataset.categorical_columns,
            target_is_categorical=dataset.target_is_categorical,
            variable_schema_fingerprint=dataset.variable_schema_fingerprint,
            frame_columns=dataset.full_df.columns,
        )


def test_identity_dimension_fingerprint_must_match(make_canonical_dataset):
    dataset = make_canonical_dataset("column")
    payload = semantic_context_payload(dataset)
    payload["identity_dimension"]["fingerprint"] = "tampered"
    with pytest.raises(ValueError, match="identity_dimension fingerprint"):
        validate_semantic_context(
            payload,
            target_column=dataset.target_column,
            feature_columns=dataset.feature_columns,
            categorical_columns=dataset.categorical_columns,
            target_is_categorical=dataset.target_is_categorical,
            variable_schema_fingerprint=dataset.variable_schema_fingerprint,
            frame_columns=dataset.full_df.columns,
        )


def test_patient_tokens_are_keyed_domain_separated_and_opaque():
    secret = "test-secret"
    first = _opaque_group_token("patient-1", "scope-a", token_secret=secret)
    assert first == _opaque_group_token("patient-1", "scope-a", token_secret=secret)
    assert first != _opaque_group_token("patient-2", "scope-a", token_secret=secret)
    assert first != _opaque_group_token("patient-1", "scope-b", token_secret=secret)
    assert first != hashlib.sha256(b"patient-1").hexdigest()
    assert TOKENIZATION_VERSION not in first


def test_semantic_sensitive_target_types_use_sensitive_not_protected(make_canonical_dataset):
    dataset = make_canonical_dataset("column")
    dataset.sensitive_columns = ["feature"]
    dataset.protected_columns = ["protected"]

    payload = semantic_context_payload(dataset)

    assert payload["sensitive_target_types"] == {"feature": "continuous"}


def test_external_generator_semantic_context_is_validated(make_canonical_dataset):
    dataset = make_canonical_dataset("column")
    dataset.quasi_identifier_columns = ["feature"]
    payload = semantic_context_payload(dataset, classification_score="macro_f1")

    resolved = validate_semantic_context(
        payload,
        target_column=dataset.target_column,
        feature_columns=dataset.feature_columns,
        categorical_columns=dataset.categorical_columns,
        target_is_categorical=dataset.target_is_categorical,
        variable_schema_fingerprint=dataset.variable_schema_fingerprint,
        frame_columns=dataset.full_df.columns,
    )

    assert resolved == payload


def test_external_generator_semantic_context_rejects_target_type_mismatch(
    make_canonical_dataset,
):
    dataset = make_canonical_dataset("column")
    payload = semantic_context_payload(dataset, classification_score="balanced_accuracy")
    payload["target_column"] = "feature"

    with pytest.raises(ValueError, match="target_column"):
        validate_semantic_context(
            payload,
            target_column=dataset.target_column,
            feature_columns=dataset.feature_columns,
            categorical_columns=dataset.categorical_columns,
            target_is_categorical=dataset.target_is_categorical,
            variable_schema_fingerprint=dataset.variable_schema_fingerprint,
            frame_columns=dataset.full_df.columns,
        )


def test_patient_group_identity_requires_one_explicit_resolution_method():
    frame = pd.DataFrame({"feature": [1], "target": [0]})
    split = DataSplitConfig(mode="patient_group")

    with pytest.raises(ValueError, match="requires patient_id_column"):
        resolve_population_identity(frame, split)


@pytest.mark.parametrize("values", [[None, "ok"], ["", "ok"], ["   ", "ok"], [np.nan, "ok"]])
def test_identity_rejects_missing_or_blank_values(values):
    frame = pd.DataFrame({"patient_id": values, "feature": [1, 2], "target": [0, 1]})
    split = DataSplitConfig(mode="patient_group", patient_id_column="patient_id")

    with pytest.raises(ValueError, match="missing|empty|whitespace"):
        resolve_population_identity(frame, split)


@pytest.mark.parametrize("bad_value", [["patient-1"], {"patient": 1}])
def test_identity_rejects_non_scalar_patient_ids(bad_value):
    patient_ids = pd.Series([bad_value, "patient-2"], dtype=object)
    frame = pd.DataFrame({"patient_id": patient_ids, "feature": [1, 2], "target": [0, 1]})
    split = DataSplitConfig(mode="patient_group", patient_id_column="patient_id")

    with pytest.raises(ValueError, match="must contain scalar values"):
        resolve_population_identity(frame, split)


def test_row_identity_resolution_does_not_require_patient_hmac_secret(monkeypatch):
    frame = pd.DataFrame({"feature": [1, 2], "target": [0, 1]})
    monkeypatch.delenv("SYNTHDATA_PATIENT_ID_HMAC_KEY", raising=False)

    identity = resolve_population_identity(frame, DataSplitConfig(mode="row"))

    assert identity.groups is None
    pd.testing.assert_frame_equal(identity.model_frame, frame)


def test_repeated_direct_patient_ids_share_opaque_role_assignment():
    raw_patient_ids = np.repeat(["patient-1", "patient-2", "patient-3", "patient-4"], 2)
    frame = pd.DataFrame(
        {
            "patient_id": raw_patient_ids,
            "feature": range(len(raw_patient_ids)),
            "target": [0, 1] * 4,
        }
    )
    split = DataSplitConfig(
        mode="patient_group",
        patient_id_column="patient_id",
        train_fraction=0.5,
        tuning_fraction=0.25,
        final_holdout_fraction=0.25,
        candidate_count=64,
    )

    identity = resolve_population_identity(frame, split)
    assignment = allocate_roles(
        identity.model_frame,
        "target",
        split,
        groups=identity.groups,
        seed=13,
    )

    assert identity.groups is not None
    assert identity.groups.groupby(raw_patient_ids).nunique().eq(1).all()
    assert all(
        assignment.assignment.loc[
            assignment.assignment["row_key"].isin(np.flatnonzero(raw_patient_ids == patient)),
            "role",
        ].nunique()
        == 1
        for patient in np.unique(raw_patient_ids)
    )
    pd.testing.assert_series_equal(
        assignment.assignment["population_group_hash"],
        identity.groups,
        check_names=False,
    )
    assert all(token not in raw_patient_ids for token in identity.groups)


def test_role_allocation_preserves_resolved_population_group_tokens():
    raw_patient_ids = np.repeat(["patient-1", "patient-2", "patient-3", "patient-4"], 2)
    frame = pd.DataFrame(
        {
            "patient_id": raw_patient_ids,
            "feature": range(len(raw_patient_ids)),
            "target": [0, 1] * 4,
        }
    )
    split = DataSplitConfig(
        mode="patient_group",
        patient_id_column="patient_id",
        train_fraction=0.5,
        tuning_fraction=0.25,
        final_holdout_fraction=0.25,
        candidate_count=64,
    )

    identity = resolve_population_identity(frame, split)
    assert identity.groups is not None
    result = allocate_roles(
        identity.model_frame,
        "target",
        split,
        groups=identity.groups,
        seed=13,
    )

    expected_by_role = {
        role: identity.groups.loc[result.assignment["role"].eq(role)].reset_index(drop=True)
        for role in ROLE_NAMES
    }
    pd.testing.assert_series_equal(
        result.assignment["population_group_hash"],
        identity.groups,
        check_names=False,
    )
    for role in ROLE_NAMES:
        pd.testing.assert_series_equal(
            result.groups[role],
            expected_by_role[role],
            check_names=False,
        )


@pytest.mark.parametrize("bad_value", [None, np.nan, "", "   "])
def test_mapping_rejects_missing_or_blank_patient_keys(tmp_path, bad_value):
    frame = pd.DataFrame({"row_id": [10, 11], "feature": [1, 2], "target": [0, 1]})
    mapping_path = tmp_path / "identity.csv"
    pd.DataFrame({"row_id": [10, 11], "patient_id": [bad_value, "patient-2"]}).to_csv(
        mapping_path, index=False
    )
    split = DataSplitConfig(
        mode="patient_group",
        identity_mapping_path=str(mapping_path),
        mapping_row_key_column="row_id",
        mapping_patient_key_column="patient_id",
    )

    with pytest.raises(ValueError, match="missing|empty|whitespace"):
        resolve_population_identity(frame, split)


def test_support_overrides_reject_unexpected_protected_columns():
    frame = pd.DataFrame({"protected": ["A", "B"], "target": [0, 1]})
    split = DataSplitConfig(protected_group_count_overrides={"typo": {"A": 1}})

    with pytest.raises(ValueError, match="unexpected protected column"):
        resolve_support_constraints(frame, "target", ["protected"], split)


def test_cell_overrides_reject_unexpected_protected_groups():
    frame = pd.DataFrame({"protected": ["A", "B"], "target": [0, 1]})
    split = DataSplitConfig(
        target_by_protected_group_count_overrides={"protected": {"typo": {0: 1}}}
    )

    with pytest.raises(ValueError, match="unknown group value"):
        resolve_support_constraints(frame, "target", ["protected"], split)


def test_mapping_rejects_conflicting_patient_key_in_source(tmp_path):
    frame = pd.DataFrame({"row_id": [10, 11], "patient_id": [1, 99], "target": [0, 1]})
    mapping_path = tmp_path / "identity.csv"
    pd.DataFrame({"row_id": [10, 11], "patient_id": [1, 2]}).to_csv(
        mapping_path,
        index=False,
    )
    split = DataSplitConfig(
        mode="patient_group",
        identity_mapping_path=str(mapping_path),
        mapping_row_key_column="row_id",
        mapping_patient_key_column="patient_id",
    )

    with pytest.raises(ValueError, match="conflicts with the identity mapping"):
        resolve_population_identity(frame, split)


def test_row_roles_are_target_stratified_and_keep_requested_counts():
    frame = pd.DataFrame({"feature": range(20), "target": [0] * 14 + [1] * 6})
    split = DataSplitConfig(
        train_fraction=0.5,
        tuning_fraction=0.25,
        final_holdout_fraction=0.25,
        candidate_count=1,
        target_balance_tolerance=0.20,
    )

    assignment = allocate_roles(frame, "target", split, seed=11)

    assert [len(assignment.frames[role]) for role in ("train", "tuning", "final_holdout")] == [
        10,
        5,
        5,
    ]
    full_rate = frame["target"].mean()
    for role in ("train", "tuning", "final_holdout"):
        assert abs(assignment.frames[role]["target"].mean() - full_rate) <= 0.20


def test_encounter_label_balance_is_recorded_and_deterministic():
    frame = pd.DataFrame(
        {
            "patient": [patient for patient in range(1, 9) for _ in (0, 1)],
            "encounter": ["early", "late"] * 8,
            "target": [0, 1] * 8,
        }
    )
    split = DataSplitConfig(
        mode="patient_group",
        train_fraction=0.5,
        tuning_fraction=0.25,
        final_holdout_fraction=0.25,
        patient_id_column="patient",
        encounter_label_column="encounter",
        candidate_count=16,
    )
    identity = resolve_population_identity(frame, split)
    first = allocate_roles(
        identity.model_frame,
        "target",
        split,
        groups=identity.groups,
        seed=9,
    )
    second = allocate_roles(
        identity.model_frame,
        "target",
        split,
        groups=identity.groups,
        seed=9,
    )

    assert first.assignment_fingerprint == second.assignment_fingerprint
    policy = first.metadata["encounter_policy"]
    assert policy["label_column"] == "encounter"
    assert policy["objective"] == "encounter_counts_and_target_distribution"
    assert (
        first.metadata["preflight"]["selected_candidate_encounter_balance_error"]
        <= split.encounter_balance_tolerance
    )
