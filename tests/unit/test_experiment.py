"""Unit tests for synthdata.experiment: append-only manifest + resumability."""

import json

import pytest

from synthdata.config import Config, ExperimentConfig, GenerationConfig, SynthcityModelsConfig
from synthdata.data import role_context_fingerprint, role_context_payload
from synthdata.experiment import (
    _timestamp_id,
    dataset_plots_dir,
    dataset_version_scope,
    load_experiment,
    start_experiment,
)

pytestmark = pytest.mark.unit


class TestTimestampId:
    def test_no_tag_is_bare_timestamp(self):
        ts_id = _timestamp_id()
        assert "_" not in ts_id
        assert ts_id.endswith("Z")

    def test_with_tag_appends_tag(self):
        ts_id = _timestamp_id("baseline")
        assert ts_id.endswith("_baseline")
        timestamp_part = ts_id.rsplit("_", 1)[0]
        assert timestamp_part.endswith("Z")


def test_explicit_config_requires_fresh_auto_experiment_id():
    config = Config(
        experiment=ExperimentConfig(),
        generation=GenerationConfig(
            force_retrain=True,
            n_samples=100,
            synthcity=SynthcityModelsConfig(params={"ctgan": {"n_iter": 40}}),
        ),
    )

    assert config.experiment.id is None
    assert config.generation.force_retrain is True
    assert config.generation.n_samples == 100
    assert config.generation.synthcity.params["ctgan"]["n_iter"] == 40


class TestStartExperiment:
    def test_auto_generated_id_does_not_reuse_existing_artifacts(self, make_config, mocker):
        cfg = make_config(tag="smoke")
        mocker.patch("synthdata.experiment._timestamp_id", return_value="20260917T160200Z_smoke")
        first = start_experiment(cfg)
        second = start_experiment(cfg)

        assert first.id != second.id
        assert first.manifest_path.parent != second.manifest_path.parent

    def test_creates_directories_and_manifest(self, make_config):
        cfg = make_config()
        experiment = start_experiment(cfg)

        assert experiment.generation_dir.exists()
        assert experiment.evaluation_dir.exists()
        assert experiment.plots_dir.exists()
        assert experiment.generation_dir.name == f"exp_v_{experiment.id}"
        # No manifest file yet -- only created on first record().
        assert not experiment.manifest_path.exists()

    def test_writes_latest_pointer(self, make_config):
        cfg = make_config()
        experiment = start_experiment(cfg)

        latest_path = experiment.manifest_path.parent.parent / "latest.json"
        assert latest_path.exists()
        assert json.loads(latest_path.read_text())["experiment_id"] == experiment.id

    def test_writes_config_snapshot_once(self, make_config):
        cfg = make_config()
        experiment = start_experiment(cfg)
        snapshot_path = experiment.manifest_path.parent / "config_snapshot.json"
        assert snapshot_path.exists()
        first_snapshot = snapshot_path.read_text()

        # Starting again with the same id must not clobber the snapshot.
        cfg.experiment.id = experiment.id
        start_experiment(cfg)
        assert snapshot_path.read_text() == first_snapshot

    def test_explicit_id_is_used_verbatim(self, make_config):
        cfg = make_config(experiment_id="my-fixed-id")
        experiment = start_experiment(cfg)
        assert experiment.id == "my-fixed-id"

    def test_auto_generated_id_includes_tag(self, make_config):
        cfg = make_config(tag="baseline")
        experiment = start_experiment(cfg)
        assert experiment.id.endswith("_baseline")
        assert experiment.tag == "baseline"

    def test_dataset_version_isolates_all_artifact_directories(self, make_config):
        v1 = start_experiment(make_config(experiment_id="baseline", data_version="v1"))
        v2 = start_experiment(make_config(experiment_id="baseline", data_version="v2"))

        assert v1.generation_dir != v2.generation_dir
        assert v1.evaluation_dir != v2.evaluation_dir
        assert v1.plots_dir != v2.plots_dir
        assert v1.manifest_path != v2.manifest_path
        assert "data_v_v1" in v1.generation_dir.parts
        assert "data_v_v2" in v2.generation_dir.parts
        assert "exp_v_baseline" in v1.generation_dir.parts

    def test_dataset_level_plots_are_version_scoped(self, make_config):
        v1_dir = dataset_plots_dir(make_config(data_version="v1"))
        v2_dir = dataset_plots_dir(make_config(data_version="v2"))

        assert v1_dir != v2_dir
        assert v1_dir.parts[-2:] == ("data_v_v1", "dataset")
        assert v2_dir.parts[-2:] == ("data_v_v2", "dataset")

    def test_unversioned_artifacts_have_dedicated_scope(self, make_config):
        cfg = make_config()

        assert dataset_version_scope(cfg) == "data_v_unversioned"
        assert dataset_plots_dir(cfg).parts[-2:] == ("data_v_unversioned", "dataset")

    @pytest.mark.parametrize("version", ["", ".", "..", "v1/v2"])
    def test_rejects_unsafe_dataset_version_for_artifact_path(self, make_config, version):
        with pytest.raises(ValueError, match="path-safe"):
            start_experiment(make_config(data_version=version))


class TestRecordAppendOnly:
    def test_first_record_creates_manifest_with_one_entry(self, make_config):
        cfg = make_config()
        experiment = start_experiment(cfg)
        experiment.record("generation", artifacts={"model": "ctgan"})

        manifest = json.loads(experiment.manifest_path.read_text())
        assert manifest["experiment_id"] == experiment.id
        assert len(manifest["runs"]) == 1
        assert manifest["runs"][0]["stage"] == "generation"
        assert manifest["runs"][0]["artifacts"] == {"model": "ctgan"}

    def test_second_record_appends_not_replaces(self, make_config):
        cfg = make_config()
        experiment = start_experiment(cfg)
        experiment.record("generation", artifacts={"model": "ctgan"})
        experiment.record("evaluation", artifacts={"table": "combined.csv"})

        manifest = json.loads(experiment.manifest_path.read_text())
        assert len(manifest["runs"]) == 2
        assert manifest["runs"][0]["stage"] == "generation"
        assert manifest["runs"][1]["stage"] == "evaluation"

    def test_extra_kwargs_are_recorded(self, make_config):
        cfg = make_config()
        experiment = start_experiment(cfg)
        experiment.record("generation_plot_failed", model="ctgan", error_type="ValueError")

        manifest = json.loads(experiment.manifest_path.read_text())
        entry = manifest["runs"][0]
        assert entry["model"] == "ctgan"
        assert entry["error_type"] == "ValueError"

    def test_record_across_reloaded_experiment_object_still_appends(self, make_config):
        # Simulates separate CLI invocations (generate, then evaluate) each
        # building their own Experiment object for the same id.
        cfg = make_config()
        experiment1 = start_experiment(cfg)
        experiment1.record("generation", artifacts={"model": "ctgan"})

        cfg.experiment.id = experiment1.id
        experiment2 = load_experiment(cfg)
        experiment2.record("evaluation", artifacts={"table": "combined.csv"})

        manifest = json.loads(experiment1.manifest_path.read_text())
        assert [r["stage"] for r in manifest["runs"]] == ["generation", "evaluation"]


class TestLoadExperimentResumability:
    def _handoff_experiment(self, make_config, make_canonical_dataset):
        recorded = make_canonical_dataset()
        recorded.imputed_roles.pop("final_holdout")
        recorded.full_imputed_df = recorded.imputed_roles.get("train")
        experiment = start_experiment(make_config(), dataset=recorded)
        experiment.record("generation")
        return experiment, make_canonical_dataset()

    def test_final_holdout_imputation_handoff_is_explicit_and_manifest_preserved(
        self, make_config, make_canonical_dataset
    ):
        experiment, current = self._handoff_experiment(make_config, make_canonical_dataset)
        before = experiment.manifest_path.read_bytes()

        loaded = load_experiment(
            make_config(experiment_id=experiment.id),
            dataset=current,
            allow_final_holdout_handoff=True,
        )

        assert loaded.id == experiment.id
        assert experiment.manifest_path.read_bytes() == before

    def test_generation_context_uses_raw_equivalent_final_holdout_baseline(
        self, make_config, make_canonical_dataset
    ):
        dataset = make_canonical_dataset()
        experiment = start_experiment(make_config(), dataset=dataset)
        experiment.record("generation")
        manifest = json.loads(experiment.manifest_path.read_text())
        final_context = manifest["role_context"]["roles"]["final_holdout"]

        assert final_context["imputed_fingerprint"] == final_context["raw_fingerprint"]

    def test_candidate_resume_allows_changed_final_holdout_imputation(
        self, make_config, make_canonical_dataset
    ):
        cfg = make_config(experiment_id="protected-ctgan-n40-20260918-candidate")
        recorded = make_canonical_dataset()
        experiment = start_experiment(cfg, dataset=recorded)
        experiment.record("generation")
        before = experiment.manifest_path.read_bytes()

        current = make_canonical_dataset()
        current.imputed_roles["final_holdout"].iloc[0, 0] = -1
        resumed = start_experiment(cfg, dataset=current)

        assert resumed.id == experiment.id
        assert experiment.manifest_path.read_bytes() == before

        current.imputed_roles["train"].iloc[0, 0] = -1
        with pytest.raises(ValueError, match="role context"):
            start_experiment(cfg, dataset=current)

    def test_final_holdout_handoff_rejects_raw_change(self, make_config, make_canonical_dataset):
        experiment, current = self._handoff_experiment(make_config, make_canonical_dataset)
        current.roles["final_holdout"].iloc[0, 0] = -1

        with pytest.raises(ValueError, match="role context"):
            load_experiment(
                make_config(experiment_id=experiment.id),
                dataset=current,
                allow_final_holdout_handoff=True,
            )

    def test_final_holdout_handoff_rejects_rows_change(self, make_config, make_canonical_dataset):
        experiment, current = self._handoff_experiment(make_config, make_canonical_dataset)
        current.roles["final_holdout"] = current.roles["final_holdout"].iloc[:-1].copy()
        current.imputed_roles["final_holdout"] = (
            current.imputed_roles["final_holdout"].iloc[:-1].copy()
        )

        with pytest.raises(ValueError, match="role context"):
            load_experiment(
                make_config(experiment_id=experiment.id),
                dataset=current,
                allow_final_holdout_handoff=True,
            )

    def test_final_holdout_handoff_rejects_assignment_change(
        self, make_config, make_canonical_dataset
    ):
        experiment, current = self._handoff_experiment(make_config, make_canonical_dataset)
        current.assignment.loc[current.assignment.index[0], "role"] = "tuning"

        with pytest.raises(ValueError, match="role context"):
            load_experiment(
                make_config(experiment_id=experiment.id),
                dataset=current,
                allow_final_holdout_handoff=True,
            )

    @pytest.mark.parametrize("role", ["train", "tuning"])
    def test_final_holdout_handoff_rejects_candidate_imputed_change(
        self, make_config, make_canonical_dataset, role
    ):
        experiment, current = self._handoff_experiment(make_config, make_canonical_dataset)
        current.imputed_roles[role].iloc[0, 0] = -999

        with pytest.raises(ValueError, match="role context"):
            load_experiment(
                make_config(experiment_id=experiment.id),
                dataset=current,
                allow_final_holdout_handoff=True,
            )

    def test_final_holdout_handoff_rejects_arbitrary_context_change(
        self, make_config, make_canonical_dataset
    ):
        experiment, current = self._handoff_experiment(make_config, make_canonical_dataset)
        current.semantic_fingerprint = "changed"

        with pytest.raises(ValueError, match="role context"):
            load_experiment(
                make_config(experiment_id=experiment.id),
                dataset=current,
                allow_final_holdout_handoff=True,
            )

    def test_final_holdout_handoff_is_rejected_by_default(
        self, make_config, make_canonical_dataset
    ):
        experiment, current = self._handoff_experiment(make_config, make_canonical_dataset)

        with pytest.raises(ValueError, match="role context"):
            load_experiment(make_config(experiment_id=experiment.id), dataset=current)

    def test_resumes_via_explicit_id(self, make_config):
        cfg = make_config()
        started = start_experiment(cfg)
        started.record("generation")

        cfg.experiment.id = started.id
        loaded = load_experiment(cfg)
        assert loaded.id == started.id
        assert loaded.generation_dir == started.generation_dir

    def test_resumes_via_latest_pointer(self, make_config):
        cfg = make_config()
        started = start_experiment(cfg)

        # A fresh Config with no explicit experiment.id (mirrors a later CLI
        # invocation of `synthdata-evaluate` without --experiment-id).
        cfg.experiment.id = None
        loaded = load_experiment(cfg)
        assert loaded.id == started.id

    def test_no_experiment_and_no_pointer_raises(self, make_config):
        cfg = make_config()
        with pytest.raises(FileNotFoundError, match="No experiment found"):
            load_experiment(cfg)

    def test_generation_context_rejects_assignment_identity_mismatch(
        self, make_config, make_canonical_dataset
    ):
        dataset = make_canonical_dataset()
        experiment = start_experiment(make_config(), dataset=dataset)
        context = role_context_payload(dataset, ("train", "tuning"))
        context["roles"]["train"]["raw_fingerprint"] = "stale-assignment-identity"

        with pytest.raises(RuntimeError, match="differs from experiment dataset lineage"):
            experiment.validate_generation_context(
                context,
                role_context_fingerprint(dataset, ("train", "tuning")),
            )

    def test_generation_context_accepts_current_snapshot_without_rewriting_manifest(
        self, make_config, make_canonical_dataset
    ):
        dataset = make_canonical_dataset()
        experiment = start_experiment(make_config(), dataset=dataset)
        before = (
            experiment.manifest_path.read_bytes() if experiment.manifest_path.exists() else None
        )
        context = role_context_payload(dataset, ("train", "tuning"))
        full_roles = ("train", "tuning", "final_holdout")
        experiment.validate_generation_context(
            context,
            role_context_fingerprint(dataset, ("train", "tuning")),
            full_context=role_context_payload(dataset, full_roles),
            full_fingerprint=role_context_fingerprint(dataset, full_roles),
        )
        after = experiment.manifest_path.read_bytes() if experiment.manifest_path.exists() else None
        assert after == before

    def test_latest_pointer_tracks_most_recent_start(self, make_config):
        cfg = make_config(experiment_id="exp-1")
        start_experiment(cfg)

        cfg.experiment.id = "exp-2"
        start_experiment(cfg)

        cfg.experiment.id = None
        loaded = load_experiment(cfg)
        assert loaded.id == "exp-2"

    def test_latest_pointer_is_scoped_to_dataset_version(self, make_config):
        v1_cfg = make_config(experiment_id="v1-exp", data_version="v1")
        v2_cfg = make_config(experiment_id="v2-exp", data_version="v2")
        start_experiment(v1_cfg)
        start_experiment(v2_cfg)

        v1_cfg.experiment.id = None
        v2_cfg.experiment.id = None
        assert load_experiment(v1_cfg).id == "v1-exp"
        assert load_experiment(v2_cfg).id == "v2-exp"
