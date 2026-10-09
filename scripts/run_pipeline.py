#!/usr/bin/env python
"""CLI: run every pipeline stage in order, with a log file, optionally detached.

Runs ``synthdata-impute``, ``synthdata-generate``, ``synthdata-evaluate`` and
``synthdata-plot`` one after another, each in its own process, and stops at
the first stage that fails (with that stage's exit code). Everything the
stages print is shown and also written to

    <output root>/logs/<UTC time>_run.log

where ``<output root>`` is the parent of ``generation.output_dir`` (for
example ``output/sim/logs/``). With ``--detach`` the run moves to the
background in its own session, so closing the terminal or losing an SSH
connection does not stop it; the command prints the log path and the
process id and returns at once. Rerunning with the same ``--experiment-id``
resumes: finished models, HPO trials and evaluation checkpoints are reused.

Usage:
    synthdata-run --config configs/config_sim.yaml [--detach] [--experiment-id ID]
                  [--stages impute generate evaluate plot] [--no-plot] [--strict-checks]
"""

import argparse
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from synthdata.config import load_config

#: Stage name -> module run with ``python -m``, in pipeline order.
STAGES = {
    "impute": "scripts.run_imputation",
    "generate": "scripts.run_generation",
    "evaluate": "scripts.run_evaluation",
    "plot": "scripts.run_plots",
}


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


def run_stages(args, log_path: Path, echo: bool = True) -> int:
    """Run the stages in order, copying their output to ``log_path``; 0 on success."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with open(log_path, "a", encoding="utf-8") as log:

        def write(line: str) -> None:
            log.write(line)
            log.flush()
            if echo:
                sys.stdout.write(line)
                sys.stdout.flush()

        write(f"[{_stamp()}] synthdata-run: stages {' '.join(args.stages)}, pid {os.getpid()}\n")
        for stage in args.stages:
            command = stage_command(stage, args)
            write(f"[{_stamp()}] === {stage}: {' '.join(command[1:])}\n")
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
            for line in process.stdout:
                write(line)
            code = process.wait()
            if code != 0:
                write(f"[{_stamp()}] === {stage} failed (exit code {code}); stopping\n")
                return code
            write(f"[{_stamp()}] === {stage} finished\n")
        write(f"[{_stamp()}] synthdata-run: all stages finished\n")
    return 0


def detach(argv: list[str], log_path: Path) -> int:
    """Start this command again in the background, in its own session; return its pid."""
    command = [sys.executable, "-m", "scripts.run_pipeline", *argv, "--log-file", str(log_path)]
    kwargs = {}
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
        default=list(STAGES),
        help="Stages to run, in pipeline order (default: all four).",
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
            f"  Stop:   {stop}"
        )
        return

    print(f"Logging to {log_path}")
    sys.exit(run_stages(args, log_path))


if __name__ == "__main__":
    main()
