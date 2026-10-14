"""GPU slots: priority, least busy GPU, dead processes, fit records."""

import csv
import json
import os
import threading
import time
from pathlib import Path

import pytest

from synthdata import gpu_slots


@pytest.fixture
def slots(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_slots, "POLL_SECONDS", 0.01)
    monkeypatch.setattr(gpu_slots, "_PLANS", {})
    monkeypatch.delenv(gpu_slots.SLOTS_ENV_VAR, raising=False)
    monkeypatch.delenv(gpu_slots.RECORDS_ENV_VAR, raising=False)
    (tmp_path / "local").mkdir()
    monkeypatch.setattr(gpu_slots.tempfile, "tempdir", str(tmp_path / "local"))

    def make(n_gpus=1, fits_per_gpu=1):
        """Set up slots; returns the (local) folder of the slot table."""
        gpu_slots.setup(tmp_path / "gpu_slots", n_gpus, fits_per_gpu)
        return Path(os.environ[gpu_slots.SLOTS_ENV_VAR])

    yield make
    gpu_slots.teardown()


def _records():
    return Path(os.environ[gpu_slots.RECORDS_ENV_VAR])


def _table(folder):
    return json.loads((folder / "table.json").read_text())


def _waiter(pid, priority, since):
    return {"pid": pid, "model": "m", "kind": "trial", "priority": priority, "since": since}


@pytest.mark.parametrize(
    "value, expected", [("auto", gpu_slots.UNLIMITED), (3, 3), (0, None), (None, None)]
)
def test_resolve_fits_per_gpu(value, expected):
    assert gpu_slots.resolve_fits_per_gpu(value) == expected


def test_without_slot_table_device_is_unchanged():
    with gpu_slots.gpu_slot("tvae", "fit", 10, "cuda:1") as device:
        assert device == "cuda:1"
    with gpu_slots.gpu_slot("tvae", "fit", 10, "cpu") as device:
        assert device == "cpu"


def test_work_plan_uses_prior_then_measured_speed():
    work = gpu_slots.WorkPlan("ddpm")
    work.add(fits=2, epochs_per_fit=100)
    assert work.seconds_left() == pytest.approx(200 * gpu_slots.PRIOR_SECONDS_PER_EPOCH["ddpm"])
    work.done(epochs=100, seconds=50)
    assert work.seconds_per_epoch() == pytest.approx(0.5)
    assert work.seconds_left() == pytest.approx(50)


def test_least_busy_gpu_is_taken(slots):
    folder = slots(n_gpus=2, fits_per_gpu=2)
    with (
        gpu_slots.gpu_slot("ctgan", "trial", 5, "cuda") as first,
        gpu_slots.gpu_slot("tvae", "trial", 5, "cuda") as second,
    ):
        assert {first, second} == {"cuda:0", "cuda:1"}
        assert [len(h) for h in _table(folder)["gpus"].values()] == [1, 1]
    assert [len(h) for h in _table(folder)["gpus"].values()] == [0, 0]


def test_highest_priority_waiter_goes_first():
    table = {"slots": 1, "gpus": {"0": []}, "waiting": {}}
    pid = os.getpid()
    table["waiting"]["small"] = _waiter(pid, priority=10, since=1.0)
    table["waiting"]["big"] = _waiter(pid, priority=1000, since=2.0)
    assert gpu_slots._try_take(table, "small", 10) is None
    assert gpu_slots._try_take(table, "big", 1000) == "0"
    # The GPU is now full, so the small model waits for the release.
    assert gpu_slots._try_take(table, "small", 10) is None
    table["gpus"]["0"] = []
    assert gpu_slots._try_take(table, "small", 10) == "0"


def test_equal_priority_is_first_come_first_served():
    table = {"slots": 1, "gpus": {"0": []}, "waiting": {}}
    table["waiting"]["late"] = _waiter(os.getpid(), priority=5, since=2.0)
    table["waiting"]["early"] = _waiter(os.getpid(), priority=5, since=1.0)
    assert gpu_slots._try_take(table, "late", 5) is None
    assert gpu_slots._try_take(table, "early", 5) == "0"


def test_dead_holders_and_waiters_are_freed(slots, monkeypatch):
    folder = slots(n_gpus=1, fits_per_gpu=1)
    dead = 999_999_999
    table = _table(folder)
    table["gpus"]["0"] = [_waiter(dead, 1, 1.0)]
    table["waiting"]["gone"] = _waiter(dead, 1e9, 0.5)
    (folder / "table.json").write_text(json.dumps(table))
    monkeypatch.setattr(gpu_slots, "_alive", lambda pid: pid != dead)
    with gpu_slots.gpu_slot("ctgan", "untuned", 5, "cuda") as device:
        assert device == "cuda:0"


def test_waiting_process_gets_slot_after_release_and_fits_are_recorded(slots):
    folder = slots(n_gpus=1, fits_per_gpu=1)
    got = []

    def second():
        with gpu_slots.gpu_slot("tvae", "trial", 7, "cuda") as device:
            got.append(device)

    with gpu_slots.gpu_slot("ctgan", "untuned", 5, "cuda"):
        thread = threading.Thread(target=second)
        thread.start()
        time.sleep(0.2)
        assert got == []  # the only slot is held
    thread.join(timeout=10)
    assert got == ["cuda:0"]
    with open(_records() / "fits.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["model"], r["kind"], r["gpu"]) for r in rows] == [
        ("ctgan", "untuned", "0"),
        ("tvae", "trial", "0"),
    ]
    assert float(rows[1]["epochs"]) == 7
    assert _table(folder)["waiting"] == {}


def test_failed_fit_still_releases_its_slot(slots):
    folder = slots(n_gpus=1, fits_per_gpu=1)
    with pytest.raises(RuntimeError), gpu_slots.gpu_slot("ctgan", "trial", 5, "cuda"):
        raise RuntimeError("boom")
    assert _table(folder)["gpus"]["0"] == []


def test_cpu_models_never_wait_for_a_slot(slots):
    slots(n_gpus=1, fits_per_gpu=1)
    with (
        gpu_slots.gpu_slot("ctgan", "trial", 5, "cuda"),
        gpu_slots.gpu_slot("arf", "untuned", 0, "cuda") as device,
    ):
        assert device == "cuda"
    with open(_records() / "fits.csv", newline="") as f:
        assert [r["model"] for r in csv.DictReader(f)] == ["ctgan"]


def test_without_a_limit_fits_never_wait_and_spread_over_gpus(slots):
    folder = slots(n_gpus=2, fits_per_gpu=gpu_slots.UNLIMITED)
    with (
        gpu_slots.gpu_slot("ctgan", "trial", 5, "cuda") as a,
        gpu_slots.gpu_slot("tvae", "trial", 5, "cuda") as b,
        gpu_slots.gpu_slot("ddpm", "trial", 5, "cuda") as c,
    ):
        assert [a, b, c] == ["cuda:0", "cuda:1", "cuda:0"]
        assert [len(h) for h in _table(folder)["gpus"].values()] == [2, 1]


def test_table_write_is_atomic(slots, monkeypatch):
    """A writer dying mid-write never leaves an empty table behind."""
    folder = slots()
    before = _table(folder)

    def die(self, *args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(type(folder), "write_text", die)
    with pytest.raises(KeyboardInterrupt):
        gpu_slots._update(folder, lambda table: None)
    monkeypatch.undo()
    assert _table(folder) == before


def test_torn_read_is_retried(slots, monkeypatch):
    folder = slots()
    monkeypatch.setattr(gpu_slots.time, "sleep", lambda s: None)
    real = type(folder).read_text
    reads = []

    def flaky(self, *args, **kwargs):
        reads.append(self.name)
        return "" if len(reads) < 3 else real(self, *args, **kwargs)

    monkeypatch.setattr(type(folder), "read_text", flaky)
    assert gpu_slots._update(folder, lambda table: table["slots"]) == 1
    assert len(reads) == 3


def test_unreadable_table_still_raises(slots, monkeypatch):
    folder = slots()
    monkeypatch.setattr(gpu_slots.time, "sleep", lambda s: None)
    (folder / "table.json").write_text("")
    with pytest.raises(json.JSONDecodeError):
        gpu_slots._update(folder, lambda table: None)


def test_table_is_local_and_removed_at_teardown(slots, tmp_path):
    """The table stays off the output folder (often a network drive)."""
    folder = slots()
    assert folder.parent == tmp_path / "local"
    assert not (tmp_path / "gpu_slots" / "table.json").exists()
    gpu_slots.teardown()
    assert not folder.exists()


def test_lost_waiting_entry_is_put_back():
    """A waiter whose entry vanished from the table keeps waiting instead of failing."""
    table = {"slots": 1, "gpus": {"0": [_waiter(os.getpid(), 1, 0.0)]}, "waiting": {}}
    entry = _waiter(os.getpid(), 5, 1.0)
    assert gpu_slots._try_take(table, "me", 5, entry) is None
    assert table["waiting"]["me"]["since"] == 1.0
    table["gpus"]["0"] = []
    assert gpu_slots._try_take(table, "me", 5, entry) == "0"
