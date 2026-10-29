"""Unit tests for the generator-environment check synthdata-run does first."""

import json
import subprocess

import pytest

from synthdata.config import Config
from synthdata.generation import env_preflight as ep

pytestmark = pytest.mark.unit


def _env(tmp_path, sources=()):
    env_dir = tmp_path / "envs" / "cdtd"
    env_dir.mkdir(parents=True)
    (env_dir / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    return ep.SidecarEnv("cdtd", str(env_dir), [str(s) for s in sources])


def _fake_run(calls, *, sync_code=0, probe_code=0, probe_out=None):
    probe_out = probe_out or json.dumps({"torch": "2.9.1+cu130", "cuda": True, "devices": 1})

    def run(cmd, **kwargs):
        calls.append(cmd)
        assert "VIRTUAL_ENV" not in kwargs["env"]
        if cmd[1] == "sync":
            return subprocess.CompletedProcess(cmd, sync_code, "", "lock file out of date")
        return subprocess.CompletedProcess(
            cmd, probe_code, probe_out + "\n", "No module named torch"
        )

    return run


def test_only_enabled_sidecar_generators_are_checked():
    cfg = Config()  # both on by default
    envs = ep.enabled_sidecar_envs(cfg.generation)
    assert [e.model for e in envs] == ["tabularargn", "cdtd"]
    assert envs[0].sources == []
    assert envs[1].sources == ["submodules/cdtd_simple"]
    cfg.generation.cdtd.enabled = False
    assert [e.model for e in ep.enabled_sidecar_envs(cfg.generation)] == ["tabularargn"]
    cfg.generation.tabularargn.enabled = False
    assert ep.enabled_sidecar_envs(cfg.generation) == []


def test_ready_environment_is_synced_frozen_then_probed(tmp_path, monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", "/parent/venv")
    source = tmp_path / "cdtd_simple"
    source.mkdir()
    (source / "cdtd.py").write_text("")
    env = _env(tmp_path, [source])
    calls = []
    monkeypatch.setattr(ep.subprocess, "run", _fake_run(calls))

    status = ep.check_env(env, uv="uv")

    assert calls[0] == ["uv", "sync", "--frozen", "--project", env.env_dir]
    assert calls[1][:5] == ["uv", "run", "--project", env.env_dir, "--no-sync"]
    assert status.endswith("torch 2.9.1+cu130, CUDA available (1 GPU)")


def test_sync_off_skips_the_sync_but_still_imports_torch(tmp_path, monkeypatch):
    env = _env(tmp_path)
    calls = []
    out = json.dumps({"torch": "2.9.1", "cuda": False, "devices": 0})
    monkeypatch.setattr(ep.subprocess, "run", _fake_run(calls, probe_out=out))

    status = ep.check_env(env, sync=False, uv="uv")

    assert [c[1] for c in calls] == ["run"]
    assert "no CUDA (runs on CPU)" in status


def test_an_empty_submodule_names_the_command_that_fetches_it(tmp_path, monkeypatch):
    source = tmp_path / "cdtd_simple"
    source.mkdir()  # an uninitialised submodule is an empty folder
    env = _env(tmp_path, [source])
    monkeypatch.setattr(ep.subprocess, "run", lambda *a, **k: pytest.fail("no uv call"))

    with pytest.raises(ep.EnvPreflightError, match="git submodule update --init"):
        ep.check_env(env, uv="uv")


def test_a_missing_project_and_missing_uv_fail_clearly(tmp_path, monkeypatch):
    with pytest.raises(ep.EnvPreflightError, match="no uv project"):
        ep.check_env(ep.SidecarEnv("tabularargn", str(tmp_path / "nowhere")))
    env = _env(tmp_path)
    monkeypatch.setattr(ep.shutil, "which", lambda name: None)
    with pytest.raises(ep.EnvPreflightError, match="not on PATH"):
        ep.check_env(env)


def test_a_failed_sync_stops_with_its_output(tmp_path, monkeypatch):
    env = _env(tmp_path)
    calls = []
    monkeypatch.setattr(ep.subprocess, "run", _fake_run(calls, sync_code=2))

    with pytest.raises(
        ep.EnvPreflightError, match="(?s)failed \\(exit 2\\).*lock file out of date"
    ):
        ep.check_env(env, uv="uv")
    assert len(calls) == 1


def test_torch_that_does_not_import_stops_the_run(tmp_path, monkeypatch):
    env = _env(tmp_path)
    monkeypatch.setattr(ep.subprocess, "run", _fake_run([], probe_code=1))

    with pytest.raises(
        ep.EnvPreflightError, match="(?s)cannot import torch.*No module named torch"
    ):
        ep.check_env(env, sync=False, uv="uv")


def test_run_preflight_logs_each_environment(monkeypatch):
    cfg = Config()
    cfg.generation.cdtd.enabled = False
    cfg.compute.sync_envs = False
    seen = []
    monkeypatch.setattr(
        ep, "check_env", lambda env, sync: seen.append((env.model, sync)) or f"{env.model}: ok"
    )
    lines = []

    assert ep.run_preflight(cfg, log=lines.append) == ["tabularargn: ok"]
    assert seen == [("tabularargn", False)]
    assert lines == ["synthdata-run: tabularargn: ok"]
