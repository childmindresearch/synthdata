#!/usr/bin/env python
"""CLI: run every pipeline stage in order, with a log file, optionally detached.

Runs ``synthdata-impute``, ``synthdata-generate`` and ``synthdata-evaluate``
one after another, each in its own process, and stops at the first stage
that fails (with that stage's exit code). Each stage draws its own figures
as it finishes (``--plot``), so ``synthdata-plot`` only redraws them; add
``plot`` to ``--stages`` to run it too. Everything the stages print is shown
and also written to

    <output root>/logs/<UTC time>_run.log

where ``<output root>`` is the parent of ``generation.output_dir`` (for
example ``output/sim/logs/``). Next to it, ``<UTC time>_resources.csv``
records the run's RAM, CPU and GPU use once a minute, and
``<UTC time>_resources_by_task.csv`` splits it by model (see
``synthdata.resources``); the log gets each stage's peak use and its
heaviest models. ``<UTC time>_dashboard.html`` is a live status page,
rewritten every 30 seconds, that reloads itself in a browser (see
``synthdata.dashboard`` and ``synthdata-dashboard``). With ``--detach`` the run moves to the
background in its own session, so closing the terminal or losing an SSH
connection does not stop it; the command prints the log path and the
process id and returns at once. Rerunning with the same ``--experiment-id``
resumes: finished models, HPO trials and evaluation checkpoints are reused.

Only one run at a time may use an output root: a second ``synthdata-run``
on the same ``<output root>`` stops at once, since both would write the same
experiment, HPO studies and GPU slot table.

Usage:
    synthdata-run --config configs/config_sim.yaml [--detach] [--experiment-id ID]
                  [--stages impute generate evaluate [plot]] [--no-plot] [--strict-checks]
                  [--monitor-interval SECONDS] [--dashboard-interval SECONDS]
"""

import argparse
import contextlib
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import BaseFileLock, FileLock, Timeout

from synthdata.config import load_config
from synthdata.dashboard import DashboardWriter, dashboard_path
from synthdata.resources import ResourceMonitor

#: Stage name -> module run with ``python -m``, in pipeline order.
STAGES = {
    "impute": "scripts.run_imputation",
    "generate": "scripts.run_generation",
    "evaluate": "scripts.run_evaluation",
    "plot": "scripts.run_plots",
}

#: Stages run when ``--stages`` is not given: the others draw their own
#: figures, so ``plot`` would only redraw them.
DEFAULT_STAGES = ["impute", "generate", "evaluate"]


class RunLocked(RuntimeError):
    """Another synthdata-run is using the same output root."""


def run_lock(logs: Path) -> BaseFileLock:
    """The lock one run holds on its output root (``<output root>/logs/run.lock``)."""
    return FileLock(str(logs / "run.lock"))


def acquire_run_lock(logs: Path) -> BaseFileLock:
    """Take the output root's run lock, or raise :class:`RunLocked` naming the holder.

    The operating system frees the lock when the holding process dies, so a
    crashed or killed run never blocks the next one.
    """
    lock = run_lock(logs)
    try:
        lock.acquire(timeout=0)
    except Timeout:
        holder = logs / "run.owner"
        owner = holder.read_text(encoding="utf-8").strip() if holder.exists() else "unknown"
        raise RunLocked(
            f"another synthdata-run is using {logs.parent} ({owner}). Stop it first, or "
            "give this run a different output root."
        ) from None
    return lock


def log_dir(cfg) -> Path:
    """``<output root>/logs``: next to the stage folders, outside every experiment."""
    return Path(cfg.generation.output_dir).parent / "logs"


def stage_command(stage: str, args) -> list[str]:
    """The ``python -m`` command line for one stage, with the run's options."""
    command = [sys.executable, "-m", STAGES[stage], "--config", str(args.config)]
    if stage != "plot" and not args.no_plot:
        command.append("--plot")
    if args.dataset_version:
        command += ["--dataset-version", args.dataset_version]
    if args.experiment_id and stage != "impute":
        command += ["--experiment-id", args.experiment_id]
    if args.tag and stage == "generate":
        command += ["--tag", args.tag]
    if args.strict_checks and stage == "evaluate":
        command.append("--strict-checks")
    return command


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def resources_path(log_path: Path) -> Path:
    """``<UTC time>_resources.csv`` next to ``<UTC time>_run.log``."""
    return log_path.with_name(log_path.name.removesuffix("_run.log") + "_resources.csv")


def run_stages(args, log_path: Path, echo: bool = True) -> int:
    """Run the stages in order, copying their output to ``log_path``; 0 on success."""
    try:
        lock = acquire_run_lock(log_path.parent)
    except RunLocked as exc:
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"[{_stamp()}] synthdata-run: {exc}\n")
        if echo:
            print(f"synthdata-run: {exc}", file=sys.stderr)
        return 1
    try:
        (log_path.parent / "run.owner").write_text(
            f"pid {os.getpid()}, host {socket.gethostname()}, log {log_path.name}",
            encoding="utf-8",
        )
        return _run_with_monitors(args, log_path, echo)
    finally:
        lock.release()


def _run_with_monitors(args, log_path: Path, echo: bool) -> int:
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if not (echo and sys.stdout.isatty()):
        # Nobody watches the bars redraw: in the log each update is a line of
        # its own. An explicit TQDM_DISABLE=0 keeps them.
        env.setdefault("TQDM_DISABLE", "1")
    monitor = None
    dashboard = None
    if args.monitor_interval > 0:
        monitor = ResourceMonitor(resources_path(log_path), args.monitor_interval).start()
    try:
        if args.dashboard_interval > 0:
            cfg = None
            with contextlib.suppress(Exception):  # the dashboard never stops a run
                cfg = load_config(args.config)
            dashboard = DashboardWriter(log_path, cfg, args.dashboard_interval)
        return _run_stages(args, log_path, echo, env, monitor, dashboard)
    finally:
        if monitor is not None:
            monitor.stop()
        if dashboard is not None:
            dashboard.stop()


def _run_stages(args, log_path: Path, echo: bool, env: dict, monitor, dashboard=None) -> int:
    with open(log_path, "a", encoding="utf-8") as log:

        def write(line: str) -> None:
            log.write(line)
            log.flush()
            if echo:
                sys.stdout.write(line)
                sys.stdout.flush()

        write(
            f"[{_stamp()}] synthdata-run: stages {' '.join(args.stages)}, pid {os.getpid()}, "
            f"host {socket.gethostname()}\n"
        )
        if monitor is not None:
            write(f"[{_stamp()}] resource use: {monitor.path}\n")
        if dashboard is not None:
            dashboard.start()
            write(f"[{_stamp()}] dashboard: {dashboard.path}\n")
        for stage in args.stages:
            command = stage_command(stage, args)
            write(f"[{_stamp()}] === {stage}: {' '.join(command[1:])}\n")
            if monitor is not None:
                monitor.set_stage(stage)
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert process.stdout is not None  # stdout=PIPE above
            for line in process.stdout:
                write(line)
            code = process.wait()
            if monitor is not None:
                write(f"[{_stamp()}] {monitor.peak_line(stage)}\n")
                if heaviest := monitor.heaviest_line(stage):
                    write(f"[{_stamp()}] {heaviest}\n")
            if code != 0:
                write(f"[{_stamp()}] === {stage} failed (exit code {code}); stopping\n")
                return code
            write(f"[{_stamp()}] === {stage} finished\n")
        write(f"[{_stamp()}] synthdata-run: all stages finished\n")
    return 0


def detach(argv: list[str], log_path: Path) -> int:
    """Start this command again in the background, in its own session; return its pid."""
    command = [sys.executable, "-m", "scripts.run_pipeline", *argv, "--log-file", str(log_path)]
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        )
    else:
        # A new session has no controlling terminal, so a closed terminal or a
        # dropped SSH connection sends it no hangup signal.
        kwargs["start_new_session"] = True
    with open(log_path, "a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            # A crash of the runner itself still lands in the log.
            stderr=log,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            **kwargs,
        )
    return process.pid


def forwarded_argv(args) -> list[str]:
    """The options of ``args`` as a command line, without ``--detach``."""
    argv = ["--config", str(args.config), "--stages", *args.stages]
    for flag, value in (
        ("--experiment-id", args.experiment_id),
        ("--tag", args.tag),
        ("--dataset-version", args.dataset_version),
    ):
        if value:
            argv += [flag, value]
    if args.no_plot:
        argv.append("--no-plot")
    if args.strict_checks:
        argv.append("--strict-checks")
    argv += ["--monitor-interval", f"{args.monitor_interval:g}"]
    argv += ["--dashboard-interval", f"{args.dashboard_interval:g}"]
    return argv


def parse_args(argv: list[str]):
    parser = argparse.ArgumentParser(
        description="Run impute, generate, evaluate and plot in order, logging to a file."
    )
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Run in the background, detached from this terminal; print the log path and return.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=list(STAGES),
        default=list(DEFAULT_STAGES),
        help="Stages to run, in pipeline order (default: impute generate evaluate; add plot "
        "to redraw every figure at the end).",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Experiment to create or resume (overrides experiment.id).",
    )
    parser.add_argument("--tag", default=None, help="Experiment tag (overrides experiment.tag).")
    parser.add_argument(
        "--dataset-version", default=None, help="Override data.version for every stage."
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip the figures drawn during each stage."
    )
    parser.add_argument(
        "--strict-checks",
        action="store_true",
        help="Make the evaluation stage fail when an output check warns.",
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="Seconds between rows of the resource-use CSV next to the log; 0 turns it off.",
    )
    parser.add_argument(
        "--dashboard-interval",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="Seconds between rewrites of the HTML status page next to the log; 0 turns it off.",
    )
    # Set by --detach for the background process; not meant to be typed.
    parser.add_argument("--log-file", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.config = Path(args.config).expanduser().resolve()
    # Keep pipeline order whatever order the stages were typed in.
    args.stages = [stage for stage in STAGES if stage in args.stages]
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.log_file:  # the detached background process
        sys.exit(run_stages(args, Path(args.log_file), echo=False))

    cfg = load_config(args.config)  # fail fast on a bad config, before detaching
    logs = log_dir(cfg)
    logs.mkdir(parents=True, exist_ok=True)
    try:  # fail fast, before detaching, if another run holds this output root
        acquire_run_lock(logs).release()
    except RunLocked as exc:
        sys.exit(f"synthdata-run: {exc}")
    log_path = logs / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_run.log"

    if args.detach:
        pid = detach(forwarded_argv(args), log_path)
        follow = f'Get-Content -Wait "{log_path}"' if os.name == "nt" else f"tail -f {log_path}"
        # The whole process group: the runner, the running stage and its workers.
        stop = f"taskkill /PID {pid} /T /F" if os.name == "nt" else f"kill -- -{pid}"
        print(
            f"Running in the background (pid {pid}).\n"
            f"  Log:    {log_path}\n"
            f"  Follow: {follow}\n"
            f"  Status: {dashboard_path(log_path)} (open in a browser)\n"
            f"  Stop:   {stop}"
        )
        return

    print(f"Logging to {log_path}")
    if args.dashboard_interval > 0:
        print(f"Status page: {dashboard_path(log_path)}")
    sys.exit(run_stages(args, log_path))


if __name__ == "__main__":
    main()
