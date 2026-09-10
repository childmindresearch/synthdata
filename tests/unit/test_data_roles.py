"""Tests for canonical role allocation, identity isolation, and provenance."""

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
    DataSplitConfig,
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
    assert sum(len(frame) for frame in first.roles.values()) == len(first.full_df)
    for left_role, left_groups in first.role_groups.items():
        for right_role, right_groups in first.role_groups.items():
            if left_role >= right_role:
                continue
            assert set(left_groups).isdisjoint(set(right_groups))


def test_mapping_sidecar_identity_is_removed_from_model_frames(make_canonical_dataset):
    dataset = make_canonical_dataset("mapping")

    assert dataset.role_metadata["identity"]["identity_source"] == "mapping_file"
    assert "row_id" not in dataset.full_df.columns
    assert all("row_id" not in frame.columns for frame in dataset.roles.values())
    assert all(
        len(groups) == len(dataset.roles[role]) for role, groups in dataset.role_groups.items()
    )


def test_one_row_per_patient_creates_explicit_disjoint_groups(make_canonical_dataset):
    dataset = make_canonical_dataset("one_row_per_patient")

    assert dataset.role_metadata["identity"]["identity_source"] == "one_row_per_patient"
    assert sum(groups.nunique() for groups in dataset.role_groups.values()) == len(dataset.full_df)
    assert all(
        len(groups) == len(dataset.roles[role]) for role, groups in dataset.role_groups.items()
    )


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
    assert semantic_context_digest(payload) == baseline

    dataset.quasi_identifier_columns = ["protected"]
    assert (
        semantic_context_fingerprint(dataset, classification_score="balanced_accuracy") != baseline
    )

    dataset.quasi_identifier_columns = ["feature"]
    assert semantic_context_fingerprint(dataset, classification_score="macro_f1") != baseline


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
