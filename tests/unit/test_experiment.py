"""Unit tests for synthdata.experiment: append-only manifest + resumability."""

import json

import pytest

from synthdata.experiment import (
    _timestamp_id,
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


class TestStartExperiment:
    def test_creates_directories_and_manifest(self, make_config):
        cfg = make_config()
        experiment = start_experiment(cfg)

        assert experiment.generation_dir.exists()
        assert experiment.evaluation_dir.exists()
        assert experiment.plots_dir.exists()
        assert experiment.generation_dir.name == experiment.id
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

    def test_latest_pointer_tracks_most_recent_start(self, make_config):
        cfg = make_config(experiment_id="exp-1")
        start_experiment(cfg)

        cfg.experiment.id = "exp-2"
        start_experiment(cfg)

        cfg.experiment.id = None
        loaded = load_experiment(cfg)
        assert loaded.id == "exp-2"
