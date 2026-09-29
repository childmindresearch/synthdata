"""CLI exit-status tests for generation completeness."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_generation as generation_cli

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("failed_outputs", "expected_exit"),
    [((), None), (("model_b",), 1)],
)
def test_generation_cli_exit_status_follows_persisted_output_coverage(
    make_config, monkeypatch, tmp_path, failed_outputs, expected_exit
):
    cfg = make_config()
    dataset = SimpleNamespace(role_frame=lambda *_args, **_kwargs: object())
    manifest_path = tmp_path / "manifest.json"
    generation_dir = tmp_path / "generation"
    generation_dir.mkdir()
    experiment = SimpleNamespace(
        id="generation-cli-test",
        manifest_path=manifest_path,
        generation_dir=generation_dir,
        plots_dir=tmp_path / "plots",
    )
    expected_outputs = ["model_a", "model_b"]
    produced_outputs = [name for name in expected_outputs if name not in failed_outputs]

    def record_generation():
        manifest_path.write_text(
            json.dumps(
                {
                    "runs": [
                        {
                            "stage": "generation",
                            "status": "partial" if failed_outputs else "complete",
                            "expected_outputs": expected_outputs,
                            "produced_outputs": produced_outputs,
                            "failed_outputs": list(failed_outputs),
                            "artifacts": {"models": produced_outputs},
                        }
                    ]
                }
            )
        )
        return {name: object() for name in produced_outputs}

    def load_inventory(path, _generation_dir):
        persisted = json.loads(Path(path).read_text())["runs"][-1]
        assert persisted["status"] == ("partial" if failed_outputs else "complete")
        assert persisted["produced_outputs"] == produced_outputs
        assert persisted["failed_outputs"] == list(failed_outputs)
        return SimpleNamespace(
            expected_outputs=tuple(expected_outputs),
            produced_outputs=tuple(produced_outputs),
            failed_outputs=tuple(failed_outputs),
        )

    monkeypatch.setattr(generation_cli, "load_config", lambda _path: cfg)
    monkeypatch.setattr(generation_cli, "set_global_seed", lambda _seed: None)
    monkeypatch.setattr(generation_cli, "load_dataset", lambda _cfg: dataset)
    monkeypatch.setattr(generation_cli, "_cache_key_record", lambda *_args: {"cache_key": "key"})
    monkeypatch.setattr(generation_cli, "load_imputed_splits", lambda value, **_kwargs: value)
    monkeypatch.setattr(
        generation_cli, "validate_imputation_cache_lineage", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(generation_cli, "needs_imputed_data", lambda _config: False)
    monkeypatch.setattr(generation_cli, "start_experiment", lambda *_args, **_kwargs: experiment)
    monkeypatch.setattr(
        generation_cli, "run_generation", lambda *_args, **_kwargs: record_generation()
    )
    monkeypatch.setattr(generation_cli.artifacts, "load_generation_inventory", load_inventory)
    monkeypatch.setattr("sys.argv", ["run_generation", "--config", "config.yaml"])

    if expected_exit is None:
        generation_cli.main()
    else:
        with pytest.raises(SystemExit) as exc_info:
            generation_cli.main()
        assert exc_info.value.code == expected_exit


def test_generation_cli_persists_partial_artifacts_when_generation_raises(
    make_config, monkeypatch, tmp_path
):
    cfg = make_config()
    dataset = SimpleNamespace(role_frame=lambda *_args, **_kwargs: object())
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"runs": []}))
    generation_dir = tmp_path / "generation"
    generation_dir.mkdir()
    experiment = SimpleNamespace(
        id="generation-cli-failure",
        manifest_path=manifest_path,
        generation_dir=generation_dir,
        plots_dir=tmp_path / "plots",
    )

    def record(stage, artifacts=None, **extra):
        manifest = json.loads(manifest_path.read_text())
        manifest["runs"].append({"stage": stage, "artifacts": artifacts or {}, **extra})
        manifest_path.write_text(json.dumps(manifest))

    experiment.record = record

    def fail_after_first_output(*_args, **_kwargs):
        (generation_dir / "model_a.csv").write_text("feature,target\n1,0\n")
        (generation_dir / "model_a.cache.json").write_text("{}")
        raise RuntimeError("model_b generation failed")

    monkeypatch.setattr(generation_cli, "load_config", lambda _path: cfg)
    monkeypatch.setattr(generation_cli, "set_global_seed", lambda _seed: None)
    monkeypatch.setattr(generation_cli, "load_dataset", lambda _cfg: dataset)
    monkeypatch.setattr(generation_cli, "_cache_key_record", lambda *_args: {"cache_key": "key"})
    monkeypatch.setattr(generation_cli, "load_imputed_splits", lambda value, **_kwargs: value)
    monkeypatch.setattr(
        generation_cli, "validate_imputation_cache_lineage", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(generation_cli, "needs_imputed_data", lambda _config: False)
    monkeypatch.setattr(generation_cli, "start_experiment", lambda *_args, **_kwargs: experiment)
    monkeypatch.setattr(
        generation_cli, "_expected_generation_outputs", lambda _config: ["model_a", "model_b"]
    )
    monkeypatch.setattr(generation_cli, "run_generation", fail_after_first_output)
    monkeypatch.setattr("sys.argv", ["run_generation", "--config", "config.yaml"])

    with pytest.raises(RuntimeError, match="model_b generation failed"):
        generation_cli.main()

    persisted = json.loads(manifest_path.read_text())["runs"][-1]
    assert persisted["stage"] == "generation_attempt"
    assert persisted["status"] == "failed"
    assert persisted["coverage_state"] == "incomplete"
    assert persisted["expected_outputs"] == ["model_a", "model_b"]
    assert persisted["observed_outputs"] == ["model_a"]
    assert persisted["missing_outputs"] == ["model_b"]
    assert persisted["artifacts"]["observed_artifacts"] == {
        "model_a": ["model_a.csv", "model_a.cache.json"]
    }
    assert persisted["exception_type"] == "RuntimeError"
    assert persisted["error_message"] == "model_b generation failed"
