"""Check the separate generator environments before a run starts.

TabularARGN and CDTD run in their own uv projects under ``envs/`` (see
:mod:`synthdata.generation.sidecar`), and the runner calls them with
``uv run --no-sync`` so nothing is installed mid-run. A missing or stale
environment used to fail only when generation reached that model, hours into
a run. :func:`run_preflight` runs before any stage instead: for each enabled
sidecar generator it checks the source checkouts it needs (git submodules),
runs ``uv sync --frozen --project <env>`` (a quick no-op when the environment
already matches its lock file; ``compute.sync_envs: false`` skips it), and
imports torch in the environment's own Python to report whether it sees CUDA.
"""

import json
import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field

from synthdata.generation.sidecar import resolve_env_dir

#: Run in the environment's Python: torch's version and what CUDA it sees.
TORCH_PROBE = (
    "import json, torch; "
    "print(json.dumps({'torch': torch.__version__, 'cuda': torch.cuda.is_available(), "
    "'devices': torch.cuda.device_count() if torch.cuda.is_available() else 0}))"
)


class EnvPreflightError(RuntimeError):
    """A generator environment is missing, out of date or cannot import torch."""


@dataclass
class SidecarEnv:
    """One enabled generator that runs in its own environment."""

    model: str
    env_dir: str
    #: Checkouts the runner reads (git submodules), relative to the repo root.
    sources: list = field(default_factory=list)


def enabled_sidecar_envs(gen_cfg) -> list[SidecarEnv]:
    """The enabled generators that run in an ``envs/`` project, in run order."""
    envs = []
    if gen_cfg.tabularargn.enabled:
        envs.append(SidecarEnv("tabularargn", gen_cfg.tabularargn.env_dir))
    if gen_cfg.cdtd.enabled:
        envs.append(SidecarEnv("cdtd", gen_cfg.cdtd.env_dir, [gen_cfg.cdtd.source_dir]))
    return envs


def _child_env() -> dict:
    # The runner's own virtualenv must not leak into the environment's uv calls.
    return {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}


def _tail(text: str, lines: int = 15) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


def check_env(env: SidecarEnv, *, sync: bool = True, uv: str | None = None) -> str:
    """Check one environment; return a one-line status or raise :class:`EnvPreflightError`."""
    env_dir = resolve_env_dir(env.env_dir)
    if not (env_dir / "pyproject.toml").exists():
        raise EnvPreflightError(
            f"{env.model}: no uv project at {env_dir} (generation.{env.model}.env_dir)"
        )
    for source in env.sources:
        path = resolve_env_dir(source)
        if not path.is_dir() or not any(path.iterdir()):
            raise EnvPreflightError(
                f"{env.model}: {path} is missing or empty; fetch it with "
                f"`git submodule update --init {source}`"
            )
    uv = uv or shutil.which("uv")
    if uv is None:
        raise EnvPreflightError(f"{env.model} runs through uv, which is not on PATH")
    if sync:
        done = subprocess.run(
            [uv, "sync", "--frozen", "--project", str(env_dir)],
            capture_output=True,
            text=True,
            env=_child_env(),
        )
        if done.returncode != 0:
            raise EnvPreflightError(
                f"{env.model}: `uv sync --frozen --project {env.env_dir}` failed "
                f"(exit {done.returncode}):\n{_tail(done.stderr or done.stdout)}"
            )
    probe = subprocess.run(
        [uv, "run", "--project", str(env_dir), "--no-sync", "python", "-c", TORCH_PROBE],
        capture_output=True,
        text=True,
        env=_child_env(),
    )
    if probe.returncode != 0:
        hint = (
            ""
            if sync
            else f"; run `uv sync --project {env.env_dir}` or set compute.sync_envs: true"
        )
        raise EnvPreflightError(
            f"{env.model}: cannot import torch in {env.env_dir}{hint}:\n"
            f"{_tail(probe.stderr or probe.stdout)}"
        )
    try:
        info = json.loads(probe.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise EnvPreflightError(
            f"{env.model}: unexpected torch check output: {_tail(probe.stdout)}"
        ) from exc
    cuda = f"CUDA available ({info['devices']} GPU)" if info["cuda"] else "no CUDA (runs on CPU)"
    return f"{env.model}: {env.env_dir} ready, torch {info['torch']}, {cuda}"


def run_preflight(cfg, log: Callable[[str], None] = print) -> list[str]:
    """Check every enabled sidecar environment of ``cfg``; raise on the first failure."""
    statuses = []
    for env in enabled_sidecar_envs(cfg.generation):
        status = check_env(env, sync=cfg.compute.sync_envs)
        log(f"synthdata-run: {status}")
        statuses.append(status)
    return statuses
