"""Unit tests for synthdata-run (scripts/run_pipeline.py)."""

import sys
from pathlib import Path

import pytest

from scripts import run_pipeline
from synthdata.config import Config

pytestmark = pytest.mark.unit


def _args(*extra):
    return run_pipeline.parse_args(["--config", "cfg.yaml", *extra])


def test_stages_keep_pipeline_order_and_options_reach_the_right_stage():
    args = _args(
        "--stages", "plot", "generate", "--experiment-id", "e1", "--tag", "t", "--strict-checks"
    )
    assert args.stages == ["generate", "plot"]
    generate = run_pipeline.stage_command("generate", args)
    assert generate[1:3] == ["-m", "scripts.run_generation"]
    assert generate[3:5] == ["--config", str(Path("cfg.yaml").resolve())]
    assert "--plot" in generate and "--tag" in generate and "e1" in generate
    plot = run_pipeline.stage_command("plot", args)
    assert "--plot" not in plot and "--tag" not in plot and "e1" in plot


def test_impute_takes_no_experiment_id_and_no_plot_drops_inline_figures():
    args = _args("--experiment-id", "e1", "--no-plot")
    impute = run_pipeline.stage_command("impute", args)
    assert "--experiment-id" not in impute and "--plot" not in impute
    assert "--strict-checks" in run_pipeline.stage_command("evaluate", _args("--strict-checks"))


def test_forwarded_argv_round_trips_without_detach():
    args = _args("--detach", "--stages", "impute", "evaluate", "--experiment-id", "e1", "--no-plot")
    again = run_pipeline.parse_args(run_pipeline.forwarded_argv(args))
    assert not again.detach
    for field in (
        "config",
        "stages",
        "experiment_id",
        "no_plot",
        "strict_checks",
        "tag",
        "monitor_interval",
        "dashboard_interval",
    ):
        assert getattr(again, field) == getattr(args, field), field


def test_logs_live_next_to_the_stage_folders():
    cfg = Config()
    cfg.generation.output_dir = "output/sim/synthetic_data"
    assert run_pipeline.log_dir(cfg) == Path("output/sim/logs")


def test_run_stops_at_the_first_failing_stage_and_logs_everything(tmp_path, monkeypatch):
    scripts = {
        "impute": "print('imputed')",
        "generate": "import sys; print('boom'); sys.exit(3)",
        "evaluate": "print('never')",
    }
    monkeypatch.setattr(
        run_pipeline, "stage_command", lambda stage, args: [sys.executable, "-c", scripts[stage]]
    )
    log = tmp_path / "run.log"
    code = run_pipeline.run_stages(_args("--stages", *scripts), log, echo=False)
    text = log.read_text()
    assert code == 3
    assert "imputed" in text and "boom" in text and "generate failed (exit code 3)" in text
    assert "never" not in text


def test_progress_bars_are_off_when_nobody_watches(tmp_path, monkeypatch):
    stage = "import os; print('bars', os.environ.get('TQDM_DISABLE'))"
    monkeypatch.setattr(run_pipeline, "stage_command", lambda s, a: [sys.executable, "-c", stage])
    monkeypatch.delenv("TQDM_DISABLE", raising=False)
    log = tmp_path / "run.log"
    args = _args("--stages", "impute", "--monitor-interval", "0", "--dashboard-interval", "0")
    assert run_pipeline.run_stages(args, log, echo=False) == 0
    assert "bars 1" in log.read_text()
    # An explicit setting wins.
    monkeypatch.setenv("TQDM_DISABLE", "0")
    assert run_pipeline.run_stages(args, log, echo=False) == 0
    assert "bars 0" in log.read_text()


def test_run_records_resource_use_per_stage(tmp_path, monkeypatch):
    resources = tmp_path / "20261010T000000Z_resources.csv"
    # The stage runs until the monitor has written two impute rows, however slow sampling is.
    stage = (
        "import pathlib, time\n"
        f"path = pathlib.Path({str(resources)!r})\n"
        "deadline = time.time() + 60\n"
        "while time.time() < deadline:\n"
        "    rows = path.read_text().splitlines() if path.exists() else []\n"
        "    if sum(',impute,' in row for row in rows) >= 2:\n"
        "        break\n"
        "    time.sleep(0.1)\n"
    )
    monkeypatch.setattr(
        run_pipeline, "stage_command", lambda stage_name, args: [sys.executable, "-c", stage]
    )
    log = tmp_path / "20261010T000000Z_run.log"
    args = _args("--stages", "impute", "--monitor-interval", "0.3")
    assert run_pipeline.run_stages(args, log, echo=False) == 0
    rows = resources.read_text().splitlines()
    assert rows[0].startswith("time_utc,stage,pipeline_ram_gib")
    assert sum(",impute," in row for row in rows[1:]) >= 2
    assert "peak use during impute: RAM" in log.read_text()
    by_task = tmp_path / "20261010T000000Z_resources_by_task.csv"
    assert by_task.read_text().startswith("time_utc,stage,task,ram_gib")


def test_monitor_interval_zero_writes_no_resource_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_pipeline, "stage_command", lambda stage, args: [sys.executable, "-c", "pass"]
    )
    log = tmp_path / "x_run.log"
    run_pipeline.run_stages(_args("--stages", "impute", "--monitor-interval", "0"), log, echo=False)
    assert not (tmp_path / "x_resources.csv").exists()
    assert "peak use" not in log.read_text()


def test_plot_stage_is_opt_in_because_each_stage_draws_its_own_figures():
    assert _args().stages == ["impute", "generate", "evaluate"]
    assert "--plot" in run_pipeline.stage_command("evaluate", _args())
    assert _args("--stages", "impute", "generate", "evaluate", "plot").stages[-1] == "plot"


def test_a_second_run_on_the_same_output_root_stops_at_once(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_pipeline, "stage_command", lambda stage, args: [sys.executable, "-c", "print('ran')"]
    )
    held = run_pipeline.acquire_run_lock(tmp_path)  # a run already in progress
    (tmp_path / "run.owner").write_text("pid 1, host augur, log first_run.log")
    try:
        log = tmp_path / "second_run.log"
        code = run_pipeline.run_stages(_args("--stages", "impute"), log, echo=False)
        text = log.read_text()
        assert code == 1
        assert "another synthdata-run" in text and "first_run.log" in text
        assert "ran" not in text
    finally:
        held.release()
    # Once the first run ends, the lock is free again.
    assert run_pipeline.run_stages(_args("--stages", "impute"), log, echo=False) == 0


def test_a_config_with_no_generator_stops_before_any_stage(tmp_path, monkeypatch):
    config = tmp_path / "cfg.yaml"
    config.write_text(
        "data: {source: uci, uci_id: 46}\n"
        "generation:\n"
        "  synthcity: {enabled: false}\n"
        "  tabpfn: {enabled: false}\n"
        "  tabpfgen: {enabled: false}\n"
    )
    monkeypatch.chdir(tmp_path)  # the logs folder lands under the default output root
    monkeypatch.setattr(
        run_pipeline, "run_stages", lambda *a, **k: pytest.fail("no stage should start")
    )
    with pytest.raises(SystemExit, match="No generator is enabled"):
        run_pipeline.main(["--config", str(config)])
    # Stages without generation still run on such a config.
    monkeypatch.setattr(run_pipeline, "run_stages", lambda *a, **k: 0)
    with pytest.raises(SystemExit) as exit_info:
        run_pipeline.main(["--config", str(config), "--stages", "impute"])
    assert exit_info.value.code == 0
