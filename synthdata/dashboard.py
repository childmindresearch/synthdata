"""A live, single-page status dashboard for a ``synthdata-run`` pipeline run.

``synthdata-run`` rewrites ``<UTC time>_dashboard.html`` next to its log
every 30 seconds (``--dashboard-interval``), and the page reloads itself, so
opening the file in a browser shows the run as it goes. ``synthdata-dashboard``
builds the same page for any run, finished or not, and can serve it on
localhost for viewing through an SSH tunnel.

Everything on the page is read from files the run already writes; nothing
here talks to the running processes:

* ``<UTC time>_run.log``: stages, per-model workers (``[step] started`` /
  ``finished`` lines from :mod:`synthdata.compute`), HPO budgets, the
  experiment id, errors and warnings, and the log itself (progress-bar
  lines dropped).
* ``<UTC time>_resources.csv`` and ``_resources_by_task.csv``
  (:mod:`synthdata.resources`): RAM, CPU and GPU use over time and by model.
* ``optuna_studies.db`` in the experiment's generation folder: HPO trials,
  read directly with ``sqlite3`` in read-only mode.
* ``ranking_summary.csv``, ``checks.csv`` and ``report.md`` in the
  experiment's evaluation folder, once evaluation has written them.

This module only uses the standard library and psutil, and building the page
never raises: a file that is missing, half-written or locked leaves its
section empty, and the page says so.
"""

import contextlib
import csv
import json
import os
import re
import socket
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psutil

#: Seconds of silence in the log after which a running run is flagged as quiet.
QUIET_AFTER_SECONDS = 20 * 60
#: Most log lines embedded in the page: the first ``LOG_HEAD_LINES`` and the
#: rest from the end, so a very long run keeps the page loadable.
MAX_LOG_LINES = 20000
LOG_HEAD_LINES = 2000
#: Points per resource chart (rows are bucketed, keeping each bucket's maximum).
MAX_POINTS = 600

_LOCAL_TZ = datetime.now().astimezone().tzinfo

_RUNNER = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) UTC\] (.*)$")
_TASK_PREFIX = re.compile(r"^\[(?P<task>[^\[\]]+?: [^\[\]]+)\] ")
_LOGLINE = re.compile(
    r"^\[(?P<hms>\d\d:\d\d:\d\d)\] (?P<logger>[\w.]+) "
    r"(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL): (?P<msg>.*)$"
)
_PY_WARNING = re.compile(r"\b\w*Warning: ")
_TQDM = re.compile(r"\d+%\||it/s\]|s/it\]")
_REPLICATE = re.compile(r"__rep\d+$")

_RUN_START = re.compile(
    r"^synthdata-run: stages (?P<stages>.+?), pid (?P<pid>\d+)(?:, host (?P<host>\S+))?$"
)
_STAGE_START = re.compile(r"^=== (?P<stage>\w+): ")
_STAGE_DONE = re.compile(r"^=== (?P<stage>\w+) finished$")
_STAGE_FAILED = re.compile(r"^=== (?P<stage>\w+) failed \(exit code (?P<code>-?\d+)\); stopping$")
_PEAK = re.compile(r"^peak use during (?P<stage>\w+): (?P<text>.*)$")

_POOL = re.compile(
    r"^\[(?P<label>[^\]]+)\] (?P<n>\d+) models, (?P<w>\d+) at a time with (?P<c>\d+) CPU threads each"
)
_TASK_START = re.compile(
    r"^\[(?P<label>[^\]]+)\] started (?P<model>\S+) \(pid (?P<pid>\d+); (?P<i>\d+) of (?P<n>\d+)\)"
)
_TASK_END = re.compile(
    r"^\[(?P<label>[^\]]+)\] (?P<model>\S+) (?P<outcome>finished|failed) after .+?; "
    r"(?P<k>\d+) of (?P<n>\d+) done"
)
_TASK_FAIL = re.compile(r"^\[(?P<label>[^\]]+)\] (?P<model>\S+) failed: (?P<why>.*)")
_EXPERIMENT = re.compile(r"^Experiment '(?P<id>[^']+)' \(tag=")
_HPO_START = re.compile(
    r"^\[(?P<study>[^\]]+)\] starting hyperparameter optimization: \d+ trial\(s\) remaining "
    r"\(\d+ already finished, target=(?P<target>\d+), timeout=(?P<timeout>\w+)s\)"
)

STAGE_ORDER = ["impute", "generate", "evaluate", "plot"]


# ---------------------------------------------------------------- helpers


def _iso(moment: datetime | None) -> str | None:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ") if moment else None


def _parse_utc(text: str) -> datetime | None:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        with contextlib.suppress(ValueError):
            return datetime.strptime(text, fmt).replace(tzinfo=UTC)
    return None


def _local_naive_to_utc(text: str | None) -> datetime | None:
    """Optuna stores ``datetime.now()`` without a zone: local time of the run's machine."""
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).replace(tzinfo=_LOCAL_TZ).astimezone(UTC)
    except ValueError:
        return None


def _float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # drop NaN


class _Clock:
    """Turn the ``HH:MM:SS`` (local time) of logging lines into UTC datetimes.

    Logging lines carry no date. Lines arrive roughly in order, so each one is
    placed at the first moment with that clock time not more than a few
    minutes before the latest moment seen (worker lines interleave slightly).
    """

    def __init__(self):
        self.latest: datetime | None = None

    def runner(self, moment: datetime) -> datetime:
        self.latest = max(self.latest, moment) if self.latest else moment
        return moment

    def local(self, hms: str) -> datetime | None:
        if self.latest is None:
            return None
        h, m, s = (int(part) for part in hms.split(":"))
        ref = self.latest.astimezone(_LOCAL_TZ)
        candidate = ref.replace(hour=h, minute=m, second=s, microsecond=0)
        if candidate < ref - timedelta(minutes=10):
            candidate += timedelta(days=1)
        elif candidate > ref + timedelta(hours=12):
            candidate -= timedelta(days=1)
        moment = candidate.astimezone(UTC)
        self.latest = max(self.latest, moment)
        return moment


def companion_path(log_path: Path, suffix: str) -> Path:
    """``<UTC time>_<suffix>`` next to ``<UTC time>_run.log``."""
    return log_path.with_name(log_path.name.removesuffix("_run.log") + f"_{suffix}")


def dashboard_path(log_path: Path) -> Path:
    return companion_path(Path(log_path), "dashboard.html")


def latest_log(logs_dir: Path) -> Path | None:
    """The newest ``*_run.log`` in ``logs_dir`` (names sort by start time)."""
    logs = sorted(Path(logs_dir).glob("*_run.log"))
    return logs[-1] if logs else None


# ---------------------------------------------------------------- log


def parse_log(lines: list[str]) -> dict:
    """Everything the dashboard reads from the run log."""
    clock = _Clock()
    run: dict = {"started": None, "ended": None, "pid": None, "host": None, "stages": []}
    stages: dict[str, dict] = {}
    pools: dict[str, dict] = {}
    tasks: dict[tuple[str, str], dict] = {}
    hpo_budget: dict[str, dict] = {}
    problems: list[dict] = []
    experiment = None
    outcome = None
    current_problem: dict | None = None

    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        runner = _RUNNER.match(line)
        if runner:
            current_problem = None
            moment = clock.runner(_parse_utc(runner.group(1)))
            msg = runner.group(2)
            if m := _RUN_START.match(msg):
                run.update(
                    started=moment,
                    pid=int(m["pid"]),
                    host=m["host"],
                    stages=m["stages"].split(),
                )
            elif m := _STAGE_DONE.match(msg):
                stages.setdefault(m["stage"], {}).update(status="done", end=moment)
            elif m := _STAGE_FAILED.match(msg):
                stages.setdefault(m["stage"], {}).update(
                    status="failed", end=moment, code=int(m["code"])
                )
                problems.append(
                    {
                        "level": "error",
                        "time": moment,
                        "source": m["stage"],
                        "text": msg,
                        "context": [],
                    }
                )
                outcome, run["ended"] = "failed", moment
            elif m := _STAGE_START.match(msg):
                stages[m["stage"]] = {"status": "running", "start": moment}
            elif m := _PEAK.match(msg):
                stages.setdefault(m["stage"], {})["peak"] = m["text"]
            elif msg.startswith("synthdata-run: all stages finished"):
                outcome, run["ended"] = "finished", moment
            continue

        task = None
        body = line
        if prefix := _TASK_PREFIX.match(line):
            task, body = prefix["task"], line[prefix.end() :]
        logline = _LOGLINE.match(body)
        if not logline:
            if (
                current_problem is not None
                and len(current_problem["context"]) < 25
                and not _TQDM.search(body)
                and (task is None or task == current_problem["task"])
            ):
                current_problem["context"].append(body)
                continue
            if body.startswith("Traceback (most recent call last)") or _PY_WARNING.search(body):
                level = "error" if body.startswith("Traceback") else "warning"
                current_problem = {
                    "level": level,
                    "time": clock.latest,
                    "source": task or "",
                    "task": task,
                    "text": body,
                    "context": [],
                }
                problems.append(current_problem)
            elif current_problem is not None and current_problem["task"] != task:
                current_problem = None
            continue

        current_problem = None
        moment = clock.local(logline["hms"])
        msg, level = logline["msg"], logline["level"]
        if m := _EXPERIMENT.match(msg):
            experiment = m["id"]
        elif m := _POOL.match(msg):
            pools[m["label"]] = {"total": int(m["n"]), "workers": int(m["w"]), "cores": int(m["c"])}
        elif m := _TASK_START.match(msg):
            pools.setdefault(m["label"], {"total": int(m["n"])})
            tasks[(m["label"], m["model"])] = {
                "step": m["label"],
                "model": m["model"],
                "status": "running",
                "start": moment,
                "end": None,
                "pid": int(m["pid"]),
                "order": int(m["i"]),
            }
        elif m := _TASK_END.match(msg):
            entry = tasks.setdefault(
                (m["label"], m["model"]),
                {"step": m["label"], "model": m["model"], "start": None, "pid": None, "order": 0},
            )
            entry.update(status="failed" if m["outcome"] == "failed" else "done", end=moment)
        elif m := _HPO_START.match(msg):
            timeout = m["timeout"]
            hpo_budget[m["study"]] = {
                "target": int(m["target"]),
                "timeout": int(timeout) if timeout.isdigit() else None,
                "session_start": moment,
            }
        if level in ("ERROR", "CRITICAL", "WARNING"):
            if level != "WARNING" and (m := _TASK_FAIL.match(msg)):
                entry = tasks.setdefault(
                    (m["label"], m["model"]),
                    {
                        "step": m["label"],
                        "model": m["model"],
                        "start": None,
                        "end": moment,
                        "pid": None,
                        "order": 0,
                    },
                )
                entry["status"] = "failed"
                entry["error"] = m["why"]
            current_problem = {
                "level": "warning" if level == "WARNING" else "error",
                "time": moment,
                "source": task or logline["logger"],
                "task": task,
                "text": msg,
                "context": [],
            }
            problems.append(current_problem)

    return {
        "run": run,
        "outcome": outcome,
        "stages": stages,
        "pools": pools,
        "tasks": list(tasks.values()),
        "hpo_budget": hpo_budget,
        "problems": problems,
        "experiment": experiment,
        "latest": clock.latest,
    }


def _group_problems(problems: list[dict], limit: int = 40) -> list[dict]:
    """Errors first, newest first; repeated messages folded into one entry with a count."""
    groups: dict[tuple, dict] = {}
    for problem in problems:
        key = (
            problem["level"],
            problem["source"],
            re.sub(r"\d+(\.\d+)?", "#", problem["text"])[:200],
        )
        group = groups.get(key)
        if group is None:
            groups[key] = {**problem, "count": 1, "first": problem["time"]}
        else:
            group["count"] += 1
            group["time"] = problem["time"] or group["time"]
            if problem["context"]:
                group["context"] = problem["context"]
    ordered = sorted(
        groups.values(),
        key=lambda g: (g["level"] != "error", -(g["time"].timestamp() if g["time"] else 0)),
    )
    return [
        {
            "level": g["level"],
            "source": g["source"],
            "text": g["text"],
            "context": g["context"],
            "count": g["count"],
            "first": _iso(g["first"]),
            "last": _iso(g["time"]),
        }
        for g in ordered[:limit]
    ]


# ---------------------------------------------------------------- resources


def _read_csv(path: Path) -> list[dict]:
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error, UnicodeDecodeError):
        return []


def read_resources(log_path: Path) -> dict:
    """Resource series (bucketed to ``MAX_POINTS``), stage peaks and use by model."""
    rows = _read_csv(companion_path(log_path, "resources.csv"))
    columns = [
        "pipeline_ram_gib",
        "system_ram_used_gib",
        "system_ram_total_gib",
        "pipeline_cpu_cores",
        "gpu_util_pct",
        "gpu_mem_used_mib",
        "pipeline_vram_mib",
        "gpu_mem_total_mib",
    ]
    rows = [row for row in rows if _parse_utc(row.get("time_utc") or "")]
    series: dict[str, list] = {"t": [], "stage": [], **{c: [] for c in columns}}
    step = max(1, -(-len(rows) // MAX_POINTS))
    for start in range(0, len(rows), step):
        bucket = rows[start : start + step]
        series["t"].append(bucket[0]["time_utc"])
        series["stage"].append(bucket[-1].get("stage", ""))
        for column in columns:
            values = [v for row in bucket if (v := _float(row.get(column))) is not None]
            series[column].append(max(values) if values else None)

    peaks: dict[str, dict] = {}
    for row in rows:
        stage = row.get("stage") or "start"
        for column in (
            "pipeline_ram_gib",
            "pipeline_cpu_cores",
            "pipeline_vram_mib",
            "gpu_mem_used_mib",
        ):
            value = _float(row.get(column))
            if value is not None:
                peaks.setdefault(stage, {})
                peaks[stage][column] = max(peaks[stage].get(column, value), value)

    task_rows = _read_csv(companion_path(log_path, "resources_by_task.csv"))
    latest_time = task_rows[-1]["time_utc"] if task_rows else None
    by_task: dict[str, dict] = {}
    for row in task_rows:
        entry = by_task.setdefault(
            row["task"], {"task": row["task"], "ram_peak": 0.0, "cpu_peak": 0.0, "vram_peak": 0.0}
        )
        ram, cpu, vram = (_float(row.get(k)) for k in ("ram_gib", "cpu_cores", "vram_mib"))
        entry["ram_peak"] = max(entry["ram_peak"], ram or 0)
        entry["cpu_peak"] = max(entry["cpu_peak"], cpu or 0)
        entry["vram_peak"] = max(entry["vram_peak"], vram or 0)
        if row["time_utc"] == latest_time:
            entry.update(ram_now=ram, cpu_now=cpu, vram_now=vram, active=True)
    tasks = sorted(by_task.values(), key=lambda e: (not e.get("active"), -e["ram_peak"]))
    return {"series": series, "peaks": peaks, "tasks": tasks, "n_rows": len(rows)}


# ---------------------------------------------------------------- HPO


def read_hpo(db_path: Path | None, budgets: dict, now: datetime) -> tuple[list[dict], str | None]:
    """One entry per Optuna study: trial counts, best so far, curve and ETA."""
    if db_path is None or not db_path.exists():
        return [], None
    try:
        con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error as exc:
        return [], f"could not open {db_path.name}: {exc}"
    try:
        studies = dict(con.execute("SELECT study_id, study_name FROM studies").fetchall())
        directions = dict(
            con.execute("SELECT study_id, direction FROM study_directions WHERE objective = 0")
        )
        trials = con.execute(
            "SELECT t.trial_id, t.study_id, t.number, t.state, t.datetime_start, "
            "t.datetime_complete, v.value FROM trials t LEFT JOIN trial_values v "
            "ON v.trial_id = t.trial_id AND v.objective = 0 ORDER BY t.study_id, t.number"
        ).fetchall()
        constraints = dict(
            con.execute(
                "SELECT trial_id, value_json FROM trial_user_attributes WHERE key = 'constraints'"
            )
        )
        fail_reasons = dict(
            con.execute(
                "SELECT trial_id, value_json FROM trial_system_attributes WHERE key = 'fail_reason'"
            )
        )
    except sqlite3.Error as exc:
        return [], f"could not read {db_path.name}: {exc}"
    finally:
        con.close()

    out = []
    with_trials = {study_id for _, study_id, *_ in trials}
    by_name = {name: study_id for study_id, name in studies.items()}
    for study_id, name in studies.items():
        if name.endswith("_probe"):
            # objective: auto's start-up trials; once copied into the main
            # study they show there, until then they stand in for it.
            main = by_name.get(name.removesuffix("_probe"))
            if main in with_trials:
                continue
            name = name.removesuffix("_probe")
        elif name + "_probe" in by_name and study_id not in with_trials:
            continue
        maximize = str(directions.get(study_id, "MAXIMIZE")).upper().endswith("MAXIMIZE")
        rows = [t for t in trials if t[1] == study_id]
        counts = {"COMPLETE": 0, "PRUNED": 0, "FAIL": 0, "RUNNING": 0}
        best = None
        curve: list[list] = []
        durations = []
        last_error = None
        first_start = None
        for trial_id, _, number, state, start, end, value in rows:
            state = str(state).split(".")[-1]
            counts[state] = counts.get(state, 0) + 1
            started = _local_naive_to_utc(start)
            if started and (first_start is None or started < first_start):
                first_start = started
            finished = _local_naive_to_utc(end)
            seconds = (finished - started).total_seconds() if started and finished else 0
            # Trials copied from the objective probe (objective: auto) are
            # stamped at copy time: zero length, so they say nothing of speed.
            if seconds > 0.5 and state in ("COMPLETE", "PRUNED"):
                durations.append(seconds)
            if state == "FAIL" and trial_id in fail_reasons:
                with contextlib.suppress(ValueError, TypeError):
                    last_error = json.loads(fail_reasons[trial_id])
            if state != "COMPLETE" or value is None:
                continue
            feasible = True
            if trial_id in constraints:
                with contextlib.suppress(ValueError, TypeError):
                    feasible = all(c <= 0 for c in json.loads(constraints[trial_id]))
            if feasible and (best is None or (value > best if maximize else value < best)):
                best = value
            curve.append([number, value, best, feasible])
        model = name.removeprefix("hpo_")
        budget = budgets.get(name, {})
        target = budget.get("target")
        done = counts["COMPLETE"] + counts["PRUNED"]
        eta = None
        recent = sorted(durations[-10:])
        if target and done < target and recent:
            eta = (target - done) * recent[len(recent) // 2]
            if budget.get("timeout") and budget.get("session_start"):
                left = budget["timeout"] - (now - budget["session_start"]).total_seconds()
                eta = max(0.0, min(eta, left))
        out.append(
            {
                "study": name,
                "model": model,
                "maximize": maximize,
                "complete": counts["COMPLETE"],
                "pruned": counts["PRUNED"],
                "failed": counts["FAIL"],
                "running": counts["RUNNING"],
                "target": target,
                "timeout": budget.get("timeout"),
                "best": best,
                "curve": curve[-300:],
                "median_trial_s": recent[len(recent) // 2] if recent else None,
                "eta_s": eta,
                "first_start": _iso(first_start),
                "last_error": last_error,
            }
        )
    out.sort(key=lambda s: (-int(s["running"]), s["model"]))
    return out, None


# ---------------------------------------------------------------- results


def read_results(eval_dir: Path | None, logs_dir: Path, best_params: Path | None = None) -> dict:
    if eval_dir is None or not eval_dir.is_dir():
        return {}
    out: dict = {}
    # Searches with no trial passing the screens store {} (see
    # synthdata.generation.hpo.untuned_fallbacks): "<model>_hpo" then repeats "<model>".
    untuned: set[str] = set()
    if best_params is not None:
        with contextlib.suppress(OSError, ValueError, AttributeError):
            cache = json.loads(best_params.read_text(encoding="utf-8"))
            untuned = {f"{m}_hpo" for m, p in cache.get("synthcity", {}).items() if not p}
    ranking = _read_csv(eval_dir / "ranking_summary.csv")
    if ranking:
        out["ranking"] = [
            {
                "rank": row.get("rank", ""),
                "model": row.get("model", ""),
                "overall": _float(row.get("overall_mean")),
                "low": _float(row.get("overall_ci_low")),
                "high": _float(row.get("overall_ci_high")),
                # True, False, or None when one side has a single replicate.
                "tied": {"true": True, "false": False}.get(
                    str(row.get("tied_with_best", "")).lower()
                ),
                "baseline": str(row.get("baseline", "")).lower() == "true",
                "untuned": row.get("model", "") in untuned,
                "n": row.get("n_replicates", ""),
                **{
                    dim: _float(row.get(f"{dim}_mean"))
                    for dim in ("utility", "privacy", "fairness")
                },
            }
            for row in ranking
        ]
    checks = _read_csv(eval_dir / "checks.csv")
    if checks:
        warned = [row for row in checks if row.get("status") == "warn"]
        out["checks"] = {
            "total": len(checks),
            "warn": len(warned),
            "warned": [
                {k: row.get(k, "") for k in ("check", "model", "value", "limit", "detail")}
                for row in warned[:20]
            ],
        }
    report = eval_dir / "report.md"
    if report.exists():
        out["report"] = str(report.resolve())
        with contextlib.suppress(ValueError):
            out["report_rel"] = Path(os.path.relpath(report, logs_dir)).as_posix()
    return out


# ---------------------------------------------------------------- assemble


def experiment_dirs(cfg, experiment_id: str | None) -> tuple[Path | None, Path | None]:
    """``(generation_dir, evaluation_dir)`` of an experiment, without creating them."""
    if cfg is None or not experiment_id:
        return None, None
    from synthdata.experiment import dataset_version_scope

    scope = dataset_version_scope(cfg)
    leaf = f"exp_v_{experiment_id}"
    return (
        Path(cfg.generation.output_dir) / scope / leaf,
        Path(cfg.evaluation.output_dir) / scope / leaf,
    )


def _latest_pointer(cfg) -> str | None:
    from synthdata.experiment import _read_latest_pointer

    with contextlib.suppress(OSError, ValueError):
        return _read_latest_pointer(cfg)
    return None


def _hpo_db(cfg, generation_dir: Path | None) -> Path | None:
    storage = getattr(getattr(cfg, "generation", None), "hpo", None)
    storage = getattr(storage, "storage", None)
    if storage:
        return (
            Path(storage.removeprefix("sqlite:///")) if storage.startswith("sqlite:///") else None
        )
    return generation_dir / "optuna_studies.db" if generation_dir else None


def _best_params(cfg, generation_dir: Path | None) -> Path | None:
    path = getattr(getattr(getattr(cfg, "generation", None), "hpo", None), "best_params_path", None)
    if path:
        return Path(path)
    return generation_dir / "hpo_best_params.json" if generation_dir else None


def collect(log_path: Path, cfg=None, now: datetime | None = None) -> dict:
    """The dashboard's data for the run logged at ``log_path``."""
    log_path = Path(log_path)
    now = now or datetime.now(UTC)
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        log_mtime = datetime.fromtimestamp(log_path.stat().st_mtime, UTC)
    except OSError:
        lines, log_mtime = [], None
    parsed = parse_log(lines)
    run = parsed["run"]
    this_host = socket.gethostname()
    same_host = run["host"] in (None, this_host)

    state = parsed["outcome"] or "running"
    note = None
    if state == "running":
        alive = None
        if same_host and run["pid"]:
            alive = psutil.pid_exists(run["pid"])
        if alive is False:
            state = "stopped"
            note = "The runner process is gone but never logged an end: it was killed or crashed."
        elif log_mtime and (now - log_mtime).total_seconds() > QUIET_AFTER_SECONDS:
            note = f"No log output for {_duration((now - log_mtime).total_seconds())}."
    if not lines:
        state, note = "waiting", "The log is empty or missing."

    end = run["ended"] or (log_mtime if state == "stopped" else now)
    started = run["started"]

    stages = []
    for name in run["stages"] or STAGE_ORDER:
        info = parsed["stages"].get(name, {})
        status = info.get("status", "pending")
        if status == "running" and state in ("stopped", "failed"):
            status = "stopped"
        s_start, s_end = info.get("start"), info.get("end")
        stages.append(
            {
                "name": name,
                "status": status,
                "start": _iso(s_start),
                "end": _iso(s_end),
                "duration_s": ((s_end or end) - s_start).total_seconds() if s_start else None,
                "peak": info.get("peak"),
            }
        )

    experiment = parsed["experiment"]
    if experiment is None and cfg is not None:
        experiment = getattr(cfg.experiment, "id", None) or _latest_pointer(cfg)
    generation_dir, evaluation_dir = experiment_dirs(cfg, experiment)
    hpo, hpo_error = read_hpo(_hpo_db(cfg, generation_dir), parsed["hpo_budget"], now)

    tasks = []
    for task in parsed["tasks"]:
        status = task["status"]
        if status == "running" and state in ("stopped", "failed", "finished"):
            status = "stopped"
        t_end = task.get("end")
        tasks.append(
            {
                "step": task["step"],
                "model": task["model"],
                "status": status,
                "start": _iso(task.get("start")),
                "end": _iso(t_end),
                "duration_s": ((t_end or end) - task["start"]).total_seconds()
                if task.get("start")
                else None,
                "order": task.get("order", 0),
                "error": task.get("error"),
            }
        )
    steps = []
    for label, pool in parsed["pools"].items():
        mine = [t for t in tasks if t["step"] == label]
        steps.append(
            {
                "label": label,
                "total": pool.get("total", len(mine)),
                "workers": pool.get("workers"),
                "cores": pool.get("cores"),
                "done": sum(t["status"] == "done" for t in mine),
                "failed": sum(t["status"] == "failed" for t in mine),
                "running": sum(t["status"] == "running" for t in mine),
            }
        )

    resources = read_resources(log_path)
    machine = {}
    if same_host:
        affinity = getattr(os, "sched_getaffinity", None)  # Linux only
        if affinity is not None:
            with contextlib.suppress(OSError):
                machine["cpu_cores"] = len(affinity(0))
        machine.setdefault("cpu_cores", psutil.cpu_count())

    problems = list(parsed["problems"])
    series = resources["series"]
    if state == "running" and series["t"]:
        used, total = series["system_ram_used_gib"][-1], series["system_ram_total_gib"][-1]
        if used and total and used / total > 0.9:
            problems.append(
                {
                    "level": "warning",
                    "time": _parse_utc(series["t"][-1]),
                    "source": "resources",
                    "text": f"Machine RAM is {used / total:.0%} full ({used:.1f} of {total:.1f} GiB); "
                    "a worker may be killed for running out of memory.",
                    "context": [],
                }
            )

    log_lines, n_progress = log_excerpt(lines)
    return {
        "generated": _iso(now),
        "log": str(log_path.resolve()),
        "log_name": log_path.name,
        "config_name": getattr(cfg, "name", None),
        "host": run["host"] or (this_host if same_host else None),
        "run": {
            "state": state,
            "note": note,
            "started": _iso(started),
            "ended": _iso(run["ended"]),
            "elapsed_s": (end - started).total_seconds() if started else None,
            "pid": run["pid"],
            "last_output": _iso(log_mtime),
        },
        "experiment": experiment,
        "stages": stages,
        "steps": steps,
        "tasks": tasks,
        "hpo": hpo,
        "hpo_error": hpo_error,
        "resources": resources,
        "machine": machine,
        "problems": _group_problems(problems),
        "n_errors": sum(p["level"] == "error" for p in problems),
        "n_warnings": sum(p["level"] == "warning" for p in problems),
        "results": read_results(evaluation_dir, log_path.parent, _best_params(cfg, generation_dir)),
        "log_lines": log_lines,
        "log_progress_dropped": n_progress,
        "log_tasks": sorted({m["task"] for line in log_lines if (m := _TASK_PREFIX.match(line))}),
    }


def log_excerpt(lines: list[str]) -> tuple[list[str], int]:
    """The log without progress-bar lines, capped at ``MAX_LOG_LINES``; and how many bars were dropped.

    A progress bar written to a file is a line per update, which buries the
    messages around it; ``synthdata-run`` turns bars off when nobody watches
    its output, so this mostly matters for runs started in a terminal.
    """
    kept = [line.rstrip("\n") for line in lines if not _TQDM.search(line)]
    dropped = len(lines) - len(kept)
    if len(kept) > MAX_LOG_LINES:
        tail = MAX_LOG_LINES - LOG_HEAD_LINES
        skipped = len(kept) - MAX_LOG_LINES
        kept = [
            *kept[:LOG_HEAD_LINES],
            f"... {skipped} lines left out here; the full log is the file named above ...",
            *kept[-tail:],
        ]
    return kept, dropped


def base_model(name: str) -> str:
    """``ctgan`` for a replicate's task name such as ``ctgan__rep2``."""
    return _REPLICATE.sub("", name)


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = int(max(seconds, 0))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours >= 48:
        return f"{hours // 24} d {hours % 24} h"
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {secs:02d} s"
    return f"{secs} s"


def build(log_path: Path, cfg=None, refresh_seconds: int = 30) -> str:
    """The dashboard page for the run logged at ``log_path``, as HTML."""
    from synthdata.dashboard_page import render

    return render(collect(log_path, cfg), refresh_seconds)


def write_dashboard(log_path: Path, cfg=None, refresh_seconds: int = 30) -> Path:
    """Write ``<UTC time>_dashboard.html`` next to the log, atomically; return its path."""
    path = dashboard_path(Path(log_path))
    page = build(log_path, cfg, refresh_seconds)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(page, encoding="utf-8")
    os.replace(tmp, path)
    return path


class DashboardWriter:
    """Rewrite a run's dashboard every ``interval`` seconds from a background thread."""

    def __init__(self, log_path: Path, cfg=None, interval: float = 30.0):
        self.log_path = Path(log_path)
        self.cfg = cfg
        self.interval = interval
        self.path = dashboard_path(self.log_path)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def write(self) -> None:
        # The dashboard must never stop the run.
        with contextlib.suppress(Exception):
            write_dashboard(self.log_path, self.cfg, refresh_seconds=round(self.interval))

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.write()

    def start(self) -> "DashboardWriter":
        self.write()
        self._thread = threading.Thread(target=self._loop, name="dashboard", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop the thread and write the final page."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
        self.write()
