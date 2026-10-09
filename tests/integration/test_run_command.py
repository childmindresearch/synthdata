"""synthdata-run --detach: the run continues in the background and logs to a file."""

import contextlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.integration.conftest import THREAD_LIMITS, new_run

pytestmark = pytest.mark.integration


@pytest.mark.skipif(os.name == "nt", reason="checks the POSIX session")
def test_detached_run_finishes_in_its_own_session_and_logs(tmp_path_factory):
    run = new_run(tmp_path_factory, "detached", overrides={"imputation": {"method": "simple"}})
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_pipeline",
            "--config",
            str(run.config_path),
            "--detach",
            "--stages",
            "impute",
            "--no-plot",
        ],
        cwd=run.cwd,
        env={**os.environ, **THREAD_LIMITS, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    pid = int(re.search(r"pid (\d+)", result.stdout).group(1))
    log = Path(re.search(r"Log:\s+(\S+)", result.stdout).group(1))
    assert log.parent == Path(run.cfg.generation.output_dir).parent / "logs"
    with contextlib.suppress(ProcessLookupError):  # unless it already finished
        assert os.getsid(pid) != os.getsid(0)  # not tied to this terminal
    deadline = time.time() + 600
    while "all stages finished" not in log.read_text() and time.time() < deadline:
        assert "failed (exit code" not in log.read_text(), log.read_text()[-3000:]
        time.sleep(1)
    text = log.read_text()
    assert "all stages finished" in text, text[-3000:]
    assert "=== impute finished" in text
    assert run.data_dir.exists()
