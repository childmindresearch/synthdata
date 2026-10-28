"""Unit tests for the run dashboard (synthdata.dashboard, synthdata-dashboard)."""

import csv
import os
from datetime import UTC, datetime, timedelta

import optuna
import pytest

from scripts import run_dashboard
from synthdata import dashboard
from synthdata.config import Config

pytestmark = pytest.mark.unit

T0 = datetime(2026, 10, 10, 22, 0, 0, tzinfo=UTC)


def _runner(t: datetime, msg: str) -> str:
    return f"[{t:%Y-%m-%d %H:%M:%S} UTC] {msg}\n"


def _logline(t: datetime, logger: str, level: str, msg: str, task: str | None = None) -> str:
    local = t.astimezone(dashboard._LOCAL_TZ)
    prefix = f"[{task}] " if task else ""
    return f"{prefix}[{local:%H:%M:%S}] {logger} {level}: {msg}\n"


def _minutes(m: float) -> datetime:
    return T0 + timedelta(minutes=m)


def _write_run(tmp_path, finished: bool = False, host: str | None = None):
    """A run that imputed, and is generating three models (one failed, one running)."""
    cfg = Config()
    cfg.name = "sim"
    cfg.data.version = "1.2"
    cfg.generation.output_dir = str(tmp_path / "out" / "synthetic_data")
    cfg.evaluation.output_dir = str(tmp_path / "out" / "evaluation")
    logs = tmp_path / "out" / "logs"
    logs.mkdir(parents=True)
    log = logs / "20261010T220000Z_run.log"
    host = host or dashboard.socket.gethostname()
    lines = [
        _runner(
            T0, f"synthdata-run: stages impute generate evaluate, pid {os.getpid()}, host {host}"
        ),
        _runner(T0, "=== impute: -m scripts.run_imputation --config c.yaml"),
        _runner(_minutes(5), "peak use during impute: RAM 2.0 GiB, CPU 4.0 cores"),
        _runner(_minutes(5), "=== impute finished"),
        _runner(_minutes(5), "=== generate: -m scripts.run_generation --config c.yaml"),
        _logline(
            _minutes(5.1),
            "synthdata.experiment",
            "INFO",
            "Experiment '1.2' (tag=-, dataset=sim@1.2)",
        ),
        _logline(
            _minutes(6),
            "synthdata.compute",
            "INFO",
            "[generation] 4 models, 2 at a time with 4 CPU threads each",
        ),
        _logline(
            _minutes(6), "synthdata.compute", "INFO", "[generation] started ctgan (pid 11; 1 of 4)"
        ),
        _logline(
            _minutes(6), "synthdata.compute", "INFO", "[generation] started tvae (pid 12; 2 of 4)"
        ),
        _logline(
            _minutes(6.5),
            "synthdata.generation.hpo",
            "INFO",
            "[hpo_ctgan] starting hyperparameter optimization of tstr_macro_auprc: 10 trial(s) remaining "
            "(0 already finished, target=10, timeout=7200s)",
            task="generation: ctgan",
        ),
        "[generation: ctgan] Epochs:  40%|####      | 40/100 [00:30<00:45, 1.3it/s]\n",
        # Printed by synthcity itself, outside the logging format.
        "[generation: ctgan] [synthcity] 'stats.prdc' failed after 0.1s: empty array\n",
        "[generation: arf] [synthcity] 'stats.prdc' failed after 0.2s: empty array\n",
        _logline(
            _minutes(7),
            "py.warnings",
            "WARNING",
            "UserWarning: column age has 3 values",
            task="generation: tvae",
        ),
        _logline(
            _minutes(8),
            "py.warnings",
            "WARNING",
            "UserWarning: column age has 4 values",
            task="generation: tvae",
        ),
        _logline(
            _minutes(20),
            "synthdata.compute",
            "INFO",
            "[generation] tvae failed after 14 min 00 s; 1 of 4 done, running: ctgan",
        ),
        _logline(
            _minutes(20), "synthdata.compute", "INFO", "[generation] started arf (pid 13; 3 of 4)"
        ),
        _logline(
            _minutes(21),
            "synthdata.compute",
            "ERROR",
            "[generation] tvae failed: RuntimeError: CUDA out of memory",
        ),
        "Traceback (most recent call last):\n",
        '  File "x.py", line 1, in <module>\n',
        "RuntimeError: CUDA out of memory\n",
        _logline(
            _minutes(30),
            "synthdata.compute",
            "INFO",
            "[generation] arf finished after 10 min 00 s; 2 of 4 done, running: ctgan",
        ),
    ]
    if finished:
        lines += [
            _runner(_minutes(40), "=== generate finished"),
            _runner(_minutes(40), "=== evaluate: -m scripts.run_evaluation"),
            _runner(_minutes(50), "=== evaluate finished"),
            _runner(_minutes(50), "synthdata-run: all stages finished"),
        ]
    log.write_text("".join(lines), encoding="utf-8")

    with open(dashboard.companion_path(log, "resources.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time_utc",
                "stage",
                "pipeline_ram_gib",
                "pipeline_cpu_cores",
                "system_ram_used_gib",
                "system_ram_total_gib",
                "gpu_util_pct",
                "gpu_mem_used_mib",
                "pipeline_vram_mib",
                "gpu_mem_total_mib",
            ]
        )
        for i in range(31):
            stage = "impute" if i < 5 else "generate"
            writer.writerow(
                [
                    f"{_minutes(i):%Y-%m-%dT%H:%M:%SZ}",
                    stage,
                    2 + i * 0.1,
                    4,
                    10 + i * 0.1,
                    64,
                    50,
                    4000,
                    3000,
                    24576,
                ]
            )
    with open(dashboard.companion_path(log, "resources_by_task.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_utc", "stage", "task", "ram_gib", "cpu_cores", "vram_mib"])
        writer.writerow(
            [f"{_minutes(29):%Y-%m-%dT%H:%M:%SZ}", "generate", "generation: tvae", 9.0, 4, 2000]
        )
        writer.writerow(
            [f"{_minutes(30):%Y-%m-%dT%H:%M:%SZ}", "generate", "generation: ctgan", 3.0, 4, 3000]
        )
        writer.writerow([f"{_minutes(30):%Y-%m-%dT%H:%M:%SZ}", "generate", "main", 1.0, 0.1, ""])

    gen_dir, eval_dir = dashboard.experiment_dirs(cfg, "1.2")
    gen_dir.mkdir(parents=True)
    storage = f"sqlite:///{(gen_dir / 'optuna_studies.db').as_posix()}"
    study = optuna.create_study(study_name="hpo_ctgan", direction="maximize", storage=storage)
    study.set_user_attr("objective", "tstr_macro_auprc")
    for value in (0.5, 0.7, 0.6):
        study.add_trial(
            optuna.trial.create_trial(
                params={}, distributions={}, value=value, user_attrs={"constraints": [0.0]}
            )
        )
    # Infeasible: a better value that fails a screen must not count as best.
    study.add_trial(
        optuna.trial.create_trial(
            params={},
            distributions={},
            value=0.9,
            user_attrs={
                "constraints": [1.0],
                "screens": {"copies": -0.1, "protected_share_gap": 0.04},
                "screen_details": {
                    "protected_share_gap": {"value": 0.15, "limit": 0.11, "bound": "max"}
                },
            },
        )
    )
    study.add_trial(
        optuna.trial.create_trial(params={}, distributions={}, state=optuna.trial.TrialState.FAIL)
    )
    if finished:
        eval_dir.mkdir(parents=True)
        (eval_dir / "ranking_summary.csv").write_text(
            "model,rank,n_replicates,baseline,overall_mean,overall_ci_low,overall_ci_high,eligible,tied_with_best\n"
            "ctgan,1,3,False,0.81,0.78,0.84,True,True\n"
            "ctgan_hpo,3,3,False,0.81,0.78,0.84,True,True\n"
            "baseline_train_copy,2,1,True,0.79,,,False,False\n"
        )
        (eval_dir / "checks.csv").write_text(
            "check,model,status,value,limit,detail\nrows,ctgan,pass,1,1,\nclass_share,ctgan,warn,0.4,0.3,off\n"
        )
        (eval_dir / "report.md").write_text("# Report\n")
        (gen_dir / "hpo_best_params.json").write_text('{"synthcity": {"ctgan": {}}}')
    return cfg, log


def test_collect_reads_stages_models_hpo_resources_and_problems(tmp_path):
    cfg, log = _write_run(tmp_path)
    d = dashboard.collect(log, cfg, now=_minutes(31))

    assert d["run"]["state"] == "running"
    assert d["run"]["elapsed_s"] == 31 * 60
    assert d["experiment"] == "1.2"
    assert [(s["name"], s["status"]) for s in d["stages"]] == [
        ("impute", "done"),
        ("generate", "running"),
        ("evaluate", "pending"),
    ]
    assert d["stages"][0]["duration_s"] == 300 and "RAM 2.0 GiB" in d["stages"][0]["peak"]

    # ctgan's worker is searching: its untuned fits are done, and the search
    # runs on the tuned model's row.
    status = {t["model"]: t["status"] for t in d["tasks"]}
    assert status == {"ctgan": "done", "ctgan_hpo": "running", "tvae": "failed", "arf": "done"}
    tvae = next(t for t in d["tasks"] if t["model"] == "tvae")
    assert "CUDA out of memory" in tvae["error"]
    assert d["steps"] == [
        {
            "label": "generation",
            "total": 4,
            "workers": 2,
            "cores": 4,
            "done": 2,
            "failed": 1,
            "running": 1,
            "with_metric_failures": 2,
        }
    ]
    assert {t["model"]: t["metric_failures"] for t in d["tasks"]} == {
        "ctgan": 1,
        "ctgan_hpo": 0,
        "tvae": 0,
        "arf": 1,
    }

    (study,) = d["hpo"]
    assert (study["model"], study["complete"], study["failed"], study["target"]) == (
        "ctgan",
        4,
        1,
        10,
    )
    assert study["best"] == 0.7  # the 0.9 trial failed a screen
    assert study["status"] == "running"  # its generation worker is running
    assert study["timeout"] == 7200
    assert study["objective"] == "tstr_macro_auprc"
    assert study["infeasible"] == 1 and study["screen_fails"] == {"protected_share_gap": 1}
    assert study["curve"][3][4] == [
        {
            "name": "protected_share_gap",
            "violation": 0.04,
            "value": 0.15,
            "limit": 0.11,
            "bound": "max",
        }
    ]

    series = d["resources"]["series"]
    assert len(series["t"]) == 31 and series["pipeline_ram_gib"][-1] == 5.0
    assert series["gpu_mem_total_mib"][-1] == 24576
    assert d["resources"]["peaks"]["impute"]["pipeline_ram_gib"] == pytest.approx(2.4)
    active = [t["task"] for t in d["resources"]["tasks"] if t.get("active")]
    assert active == ["generation: ctgan", "main"]

    # The two warnings differ only in a number: one entry with a count; the
    # error keeps its traceback.
    # The two printed metric failures fold into one error naming both models.
    assert d["n_errors"] == 3 and d["n_warnings"] == 2
    error, metric, warning = d["problems"]
    # A worker failure names its model in the context, so the same failure
    # on several models folds into one entry.
    assert error["text"] == "[generation] failed: RuntimeError: CUDA out of memory"
    assert error["context"][0] == "1 model(s): tvae" and "Traceback" in error["context"][1]
    assert metric["text"] == "metric 'stats.prdc' failed: empty array"
    assert metric["count"] == 2 and metric["source"] == "generation"
    assert metric["context"] == ["2 model(s): ctgan, arf"]
    assert warning["count"] == 2 and warning["source"] == "generation: tvae"
    assert d["results"] == {}
    assert "generation: ctgan" in d["log_tasks"]
    # The whole log is kept, without the progress bar.
    assert d["log_progress_dropped"] == 1
    assert d["log_lines"][0].startswith("[2026-10-10 22:00:00 UTC] synthdata-run")
    assert not any("40%|" in line for line in d["log_lines"])


def test_finished_run_shows_results_and_page_renders(tmp_path):
    cfg, log = _write_run(tmp_path, finished=True)
    d = dashboard.collect(log, cfg, now=_minutes(60))
    assert d["run"]["state"] == "finished" and d["run"]["elapsed_s"] == 50 * 60
    assert [s["status"] for s in d["stages"]] == ["done", "done", "done"]
    # A worker never reported as finished is not left "running" after the run ended.
    assert {t["model"]: t["status"] for t in d["tasks"]}["ctgan_hpo"] == "stopped"
    assert d["results"]["ranking"][0]["model"] == "ctgan"
    # Its search found no passing trial: the tuned row repeats the untuned model.
    assert [x["untuned"] for x in d["results"]["ranking"]] == [False, True, False]
    assert d["results"]["checks"]["warn"] == 1
    assert d["results"]["report_rel"] == "../evaluation/data_v_1.2/exp_v_1.2/report.md"

    path = dashboard.write_dashboard(log, cfg)
    assert path == log.with_name("20261010T220000Z_dashboard.html")
    page = path.read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>") and "Finished" in page
    assert "ctgan" in page and "report.md" in page and "CUDA out of memory" in page
    # The ranking draws the interval and says tied in words, not with a status badge.
    assert 'class="cibar"' in page and '<span class="pill good">best</span>' in page
    # One timeline row per model, a segment per step, ticks and the stage bands.
    assert page.count('class="seg ') == 4 and 'class="band"' in page and 'class="tick"' in page


def test_replicates_fold_into_one_timeline_row_and_one_resource_row(tmp_path):
    cfg, log = _write_run(tmp_path, finished=True)
    with open(log, "a", encoding="utf-8") as f:
        for i, (start, end) in enumerate([(41, 44), (41, 45), (46, 48)]):
            f.write(
                _logline(
                    _minutes(start),
                    "synthdata.compute",
                    "INFO",
                    f"[syntheval utility] started arf__rep{i} (pid 2{i}; {i + 1} of 3)",
                )
            )
            f.write(
                _logline(
                    _minutes(end),
                    "synthdata.compute",
                    "INFO",
                    f"[syntheval utility] arf__rep{i} finished after 1 min; {i + 1} of 3 done",
                )
            )
    with open(dashboard.companion_path(log, "resources_by_task.csv"), "a", newline="") as f:
        writer = csv.writer(f)
        for i in range(3):
            writer.writerow(
                [
                    f"{_minutes(30):%Y-%m-%dT%H:%M:%SZ}",
                    "evaluate",
                    f"syntheval utility: arf__rep{i}",
                    1.0 + i,
                    1,
                    0,
                ]
            )
    page = dashboard.build(log, cfg)
    arf_row = page.split('<span class="model">arf</span>')[1].split('class="g-row')[0]
    # generation, then the two overlapping replicates merged, then the third.
    assert arf_row.count('class="seg ') == 3
    assert "2 replicates" in arf_row
    assert page.count('<span class="model">arf__rep') == 0
    # SynthEval's passes are one step in the resource table too.
    assert "arf ×3" in page and "syntheval ×3" in page


def test_long_logs_keep_their_start_and_end():
    lines = [f"line {i}\n" for i in range(dashboard.MAX_LOG_LINES + 500)]
    bar = "Epochs:  10%|#   | 1/10 [00:01<00:09, 1.0it/s]\n"
    kept, dropped = dashboard.log_excerpt([*lines, bar])
    assert dropped == 1 and len(kept) == dashboard.MAX_LOG_LINES + 1
    assert kept[0] == "line 0" and kept[-1] == f"line {dashboard.MAX_LOG_LINES + 499}"
    assert "500 lines left out" in kept[dashboard.LOG_HEAD_LINES]


def test_runner_gone_without_an_end_is_stopped_and_log_text_is_escaped(tmp_path):
    cfg, log = _write_run(tmp_path)
    with open(log, "a", encoding="utf-8") as f:
        f.write(_logline(_minutes(30), "x", "ERROR", "<script>alert(1)</script>"))
    # A pid that is not running on this host.
    text = log.read_text().replace(f"pid {os.getpid()}", "pid 999999999")
    log.write_text(text)
    d = dashboard.collect(log, cfg, now=_minutes(31))
    assert d["run"]["state"] == "stopped"
    assert all(t["status"] != "running" for t in d["tasks"])
    page = dashboard.build(log, cfg)
    assert "<script>alert(1)</script>" not in page.split('<script type="application/json"')[0]
    assert "<\\/script>" in page  # the JSON copy of the log tail cannot close the script


def test_other_host_skips_the_process_check(tmp_path):
    cfg, log = _write_run(tmp_path, host="some-other-node")
    text = log.read_text().replace(f"pid {os.getpid()}", "pid 999999999")
    log.write_text(text)
    d = dashboard.collect(log, cfg, now=_minutes(31))
    assert d["run"]["state"] == "running" and d["machine"] == {}


def test_missing_files_give_an_empty_but_valid_page(tmp_path):
    log = tmp_path / "20261010T000000Z_run.log"
    d = dashboard.collect(log, None)
    assert d["run"]["state"] == "waiting" and d["hpo"] == [] and d["results"] == {}
    assert "<!doctype html>" in dashboard.build(log, None)


def test_log_lines_without_a_date_roll_over_midnight():
    clock = dashboard._Clock()
    late = datetime(2026, 10, 10, 23, 59, 50, tzinfo=dashboard._LOCAL_TZ).astimezone(UTC)
    clock.runner(late)
    moment = clock.local("00:00:10")
    assert moment - late == timedelta(seconds=20)
    # A worker line a little behind the newest one stays on the same day.
    assert late - clock.local("23:59:40") == timedelta(seconds=10)


def test_cli_writes_the_newest_runs_page(tmp_path, monkeypatch):
    cfg, log = _write_run(tmp_path, finished=True)
    monkeypatch.setattr(run_dashboard, "load_config", lambda _: cfg)
    run_dashboard.main(["--config", "c.yaml"])
    assert dashboard.dashboard_path(log).exists()


def test_tuned_model_row_sits_under_its_untuned_one(tmp_path):
    cfg, log = _write_run(tmp_path, finished=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(
            _logline(
                _minutes(41),
                "synthdata.compute",
                "INFO",
                "[synthcity] started ctgan_hpo (pid 31; 1 of 1)",
            )
        )
    page = dashboard.build(log, cfg)
    order = [m.split("</span>")[0] for m in page.split('<span class="model">')[1:]]
    assert order.index("ctgan_hpo") == order.index("ctgan") + 1


def _study(**kw) -> dict:
    return {"complete": 5, "pruned": 0, "target": 10, "running": 0, "infeasible": 0, **kw}


@pytest.mark.parametrize(
    ("study", "generating", "worker", "expected"),
    [
        (_study(), True, "running", "running"),
        (_study(complete=10), True, "running", "done"),  # final fits after the search
        (_study(complete=0), True, None, "queued"),
        (_study(), True, None, "paused"),
        # A later stage evaluating the same model is not HPO.
        (_study(), False, None, "stopped"),
        (_study(complete=10), False, None, "done"),
        (_study(complete=0), False, None, "not run"),
        (_study(infeasible=5), False, None, "no pass"),
        (_study(target=None), False, None, "done"),
    ],
)
def test_hpo_status(study, generating, worker, expected):
    assert dashboard.hpo_status(study, generating, worker) == expected


def test_main_process_fit_is_a_model_row_and_explains_a_quiet_log(tmp_path):
    log = tmp_path / "20261010T220000Z_run.log"
    log.write_text(
        _runner(T0, "synthdata-run: stages generate, pid 1, host other-host")
        + _runner(T0, "=== generate: -m scripts.run_generation")
        + _logline(
            _minutes(1),
            "synthdata.compute",
            "INFO",
            "[generation] 1 models, 1 at a time with 4 CPU threads each",
        )
        + _logline(
            _minutes(1), "synthdata.compute", "INFO", "[generation] started ctgan (pid 11; 1 of 1)"
        )
        + _logline(
            _minutes(5),
            "synthdata.compute",
            "INFO",
            "[generation] ctgan finished after 4 min; 1 of 1 done",
        )
        + _logline(
            _minutes(6),
            "synthdata.generation.pipeline",
            "INFO",
            "[tabpfn_standard] generating synthetic data (n_samples=100, seed=42)",
        ),
        encoding="utf-8",
    )
    os.utime(log, (_minutes(6).timestamp(), _minutes(6).timestamp()))
    d = dashboard.collect(log, None, now=_minutes(120))
    # A fit in the runner's own process is a generation row like the workers'.
    main = next(t for t in d["tasks"] if t["model"] == "tabpfn_standard")
    assert (main["step"], main["status"]) == (dashboard.MAIN_STEP, "running")
    assert [s["label"] for s in d["steps"]] == ["generation"]
    assert "tabpfn_standard is fitting in the runner's own process" in d["run"]["note"]

    with open(log, "a", encoding="utf-8") as f:
        f.write(
            _logline(
                _minutes(130),
                "synthdata.generation.pipeline",
                "INFO",
                "Generated/loaded 2 synthetic datasets: ['ctgan', 'tabpfn_standard']",
            )
        )
    parsed = dashboard.parse_log(log.read_text(encoding="utf-8").splitlines())
    (main,) = [t for t in parsed["tasks"] if t.get("main")]
    assert main["status"] == "done" and main["end"] is not None


def test_a_python_warning_is_its_own_problem_not_context():
    lines = [
        _logline(
            T0, "synthdata.data", "WARNING", "1 feature column(s) look like direct identifiers"
        ),
        "/x/sklearn/_split.py:1036: UserWarning: The least populated class in y has only 1 members\n",
        "  warnings.warn(\n",
        "[synthcity: a] [synthcity] 'stats.prdc' failed after 0.1s: empty array\n",
    ]
    problems = dashboard.parse_log(lines)["problems"]
    assert [p["text"][:20] for p in problems] == [
        "1 feature column(s) ",
        "/x/sklearn/_split.py",
        "metric 'stats.prdc' ",
    ]
    assert problems[0]["context"] == []
    assert problems[1]["context"] == ["  warnings.warn("]


def test_generation_worker_splits_into_untuned_fits_hpo_and_tuned_fits():
    task = "generation: ctgan"
    lines = [
        _runner(T0, "=== generate: -m scripts.run_generation"),
        _logline(T0, "synthdata.compute", "INFO", "[generation] started ctgan (pid 11; 1 of 1)"),
        _logline(
            _minutes(1),
            "synthdata.generation.pipeline",
            "INFO",
            "[ctgan] generating synthetic data (n_samples=100, seed=42)",
            task=task,
        ),
        _logline(
            _minutes(5),
            "synthdata.generation.hpo",
            "INFO",
            "[hpo_ctgan_probe] trial 0 tstr_macro_auprc=0.3 after 2 min (1 of 2 done)",
            task=task,
        ),
        _logline(
            _minutes(9),
            "synthdata.generation.hpo",
            "INFO",
            "[hpo_ctgan] trial 1 tstr_macro_auprc=0.4 after 2 min (2 of 2 done)",
            task=task,
        ),
        _logline(
            _minutes(10),
            "synthdata.generation.pipeline",
            "INFO",
            "[ctgan_hpo] generating synthetic data (n_samples=100, seed=42)",
            task=task,
        ),
        _logline(
            _minutes(12),
            "synthdata.compute",
            "INFO",
            "[generation] ctgan finished after 12 min; 1 of 1 done",
        ),
    ]
    (worker,) = dashboard.parse_log(lines)["tasks"]
    assert [(p["kind"], p["start"], p["end"]) for p in worker["phases"]] == [
        ("fit", _minutes(1).replace(microsecond=0), _minutes(5).replace(microsecond=0)),
        ("HPO", _minutes(5).replace(microsecond=0), _minutes(10).replace(microsecond=0)),
        ("tuned fit", _minutes(10).replace(microsecond=0), _minutes(12).replace(microsecond=0)),
    ]


def test_syntheval_passes_show_as_one_step_with_a_phase_each():
    lines = [_runner(T0, "=== evaluate: -m scripts.run_evaluation")]
    for i, cls in enumerate(("main", "ovr_0", "ovr_1")):
        pool = f"syntheval {cls}"
        lines += [
            _logline(
                _minutes(10 * i),
                "synthdata.compute",
                "INFO",
                f"[{pool}] 1 models, 1 at a time with 4 CPU threads each",
            ),
            _logline(
                _minutes(10 * i + 1),
                "synthdata.compute",
                "INFO",
                f"[{pool}] started ctgan (pid {100 + i}; 1 of 1)",
            ),
            _logline(
                _minutes(10 * i + 5),
                "synthdata.compute",
                "INFO",
                f"[{pool}] ctgan finished after 4 min; 1 of 1 done",
            ),
        ]
    parsed = dashboard.parse_log(lines)
    assert list(parsed["pools"]) == [dashboard.SYNTHEVAL_STEP]
    (task,) = parsed["tasks"]
    assert (task["status"], task["start"], task["end"]) == (
        "done",
        _minutes(1).replace(microsecond=0),
        _minutes(25).replace(microsecond=0),
    )
    assert [p["kind"] for p in task["phases"]] == ["main", "one-vs-rest", "one-vs-rest"]


def test_a_stopped_search_gives_its_reason():
    task = "generation: ctgan"
    lines = [
        _logline(
            _minutes(1),
            "synthdata.generation.hpo",
            "INFO",
            "[hpo_ctgan] stopping early: 10 trials in a row without a gain above 0.0050 "
            "(best 0.4000; patience 10)",
            task=task,
        ),
        _logline(
            _minutes(2),
            "synthdata.generation.hpo",
            "INFO",
            "[hpo_tvae] search ended at its time budget: 7 of 40 trials done",
            task="generation: tvae",
        ),
    ]
    budgets = dashboard.parse_log(lines)["hpo_budget"]
    assert budgets["hpo_ctgan"]["stop"] == "no gain in 10 trials"
    assert budgets["hpo_tvae"]["stop"] == "time budget"


def test_generation_segment_names_its_phases_and_trial_dots_name_the_objective():
    from synthdata import dashboard_page as page

    def at(minutes):
        return f"{T0 + timedelta(minutes=minutes):%Y-%m-%dT%H:%M:%SZ}"

    tasks = [
        {
            "phases": [
                {"kind": "fit", "start": at(0), "end": at(60)},
                {"kind": "HPO", "start": at(60), "end": at(120)},
                {"kind": "tuned fit", "start": at(120), "end": at(150)},
                {"kind": "tuned fit", "start": at(150), "end": at(180)},
            ]
        }
    ]
    spans = page._phase_spans(tasks, (T0 + timedelta(hours=4)).timestamp())
    assert [page._phase_text(*span) for span in spans] == [
        "untuned fits (1) 1 h 00 min",
        "HPO search 1 h 00 min",
        "tuned fits (2) 1 h 00 min",
    ]
    curve = [[0, 0.3, None, False, []], [1, 0.4, 0.4, True, []]]
    svg = page._sparkline(curve, True, "tstr_macro_f1")
    assert "trial 0: tstr_macro_f1 0.3" in svg
    assert "<b>tstr_macro_f1</b>" in svg and "higher is better" in svg


def test_a_worker_reports_its_own_use_and_a_finished_run_has_no_active_task(tmp_path):
    cfg, log = _write_run(tmp_path, finished=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(
            _logline(
                _minutes(41),
                "synthdata.compute",
                "INFO",
                "[synthcity] started tvae__rep1 (pid 7; 1 of 1)",
            )
            + _logline(
                _minutes(41),
                "synthdata.compute",
                "INFO",
                "[synthcity] tvae__rep1 finished after 20 s (peak RAM 1.50 GiB, CPU 2.25 cores "
                "on average, peak GPU memory 0.50 GiB); 1 of 1 done",
            )
        )
    d = dashboard.collect(log, cfg, now=_minutes(60))
    # Finished between two samples, but listed with what it measured itself.
    (entry,) = [e for e in d["resources"]["tasks"] if e["task"] == "synthcity: tvae__rep1"]
    assert (entry["ram_peak"], entry["cpu_peak"], entry["vram_peak"]) == (1.5, 2.25, 512)
    assert not any(e.get("active") for e in d["resources"]["tasks"])


def test_a_search_with_no_passing_trial_ends_the_tuned_row_as_none_passed(tmp_path):
    lines = [
        _runner(T0, "=== generate: -m scripts.run_generation"),
        _logline(
            _minutes(1), "synthdata.compute", "INFO", "[generation] started bn (pid 7; 1 of 1)"
        ),
        "[generation: bn] "
        + _logline(
            _minutes(2),
            "synthdata.generation.pipeline",
            "INFO",
            "[bn__rep1] generating synthetic data (n_samples=10, seed=42)",
        ),
        "[generation: bn] "
        + _logline(_minutes(3), "synthdata.generation.hpo", "INFO", "[hpo_bn] trial 0 done"),
        "[generation: bn] "
        + _logline(
            _minutes(4),
            "synthdata.generation.pipeline",
            "INFO",
            "[bn_hpo] none passed: no HPO trial passed the screens, so bn_hpo would repeat bn",
        ),
        _logline(
            _minutes(5),
            "synthdata.compute",
            "INFO",
            "[generation] bn finished after 4 min; 1 of 1 done",
        ),
    ]
    parsed = dashboard.parse_log(lines)
    assert parsed["none_passed"] == {"bn"}
    untuned, tuned = dashboard.split_tuned(parsed["tasks"], parsed["none_passed"])
    assert (untuned["model"], untuned["status"], untuned["end"]) == (
        "bn",
        "done",
        _minutes(3).replace(microsecond=0),
    )
    assert (tuned["model"], tuned["status"], tuned["start"]) == (
        "bn_hpo",
        "none passed",
        _minutes(3).replace(microsecond=0),
    )
