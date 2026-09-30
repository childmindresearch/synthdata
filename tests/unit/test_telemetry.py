"""Privacy startup regressions without optional models or network access."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("inherited_value", "dotenv_value"),
    [(None, None), (None, "0"), ("0", "0"), ("false", "false"), ("true", "0"), ("1", "0")],
)
def test_pipeline_import_forces_telemetry_off_before_dependencies_and_in_workers(
    tmp_path: Path, inherited_value: str | None, dotenv_value: str | None
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "SYNTHDATA_TEST_ENVIRONMENT=loaded\n"
        + (f"TABPFN_DISABLE_TELEMETRY={dotenv_value}\n" if dotenv_value is not None else ""),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.pop("TABPFN_DISABLE_TELEMETRY", None)
    environment.pop("SYNTHDATA_TEST_ENVIRONMENT", None)
    if inherited_value is not None:
        environment["TABPFN_DISABLE_TELEMETRY"] = inherited_value
    for variable in (
        "HOME",
        "TMPDIR",
        "MPLCONFIGDIR",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "HF_HOME",
        "TABPFN_STATE_DIR",
        "TABPFN_MODEL_CACHE_DIR",
    ):
        directory = tmp_path / variable.lower()
        directory.mkdir()
        environment[variable] = str(directory)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    script = """
import os
import subprocess
import sys
import dotenv

network_attempts = []
def check_access(event, arguments):
    if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo", "urllib.Request"}:
        network_attempts.append(event)
        raise AssertionError("Startup verification must not access the network")
    if event == "import" and arguments[0].split(".")[0] in {"yaml", "matplotlib"}:
        assert os.environ.get("TABPFN_DISABLE_TELEMETRY") == "1", (
            "TabPFN telemetry must be off before dependency imports"
        )
    if event == "import" and arguments[0].split(".")[0] in {
        "tabpfn", "tabpfn_extensions", "tabpfgen", "tabimpute"
    }:
        raise AssertionError("Privacy startup must not require optional model packages")

sys.addaudithook(check_access)
load_dotenv = dotenv.load_dotenv
def isolated_load_dotenv(*args, **kwargs):
    assert os.environ.get("TABPFN_DISABLE_TELEMETRY") == "1", (
        "TabPFN telemetry must be off before dotenv and dependency initialization"
    )
    return load_dotenv(sys.argv[1], *args, **kwargs)

dotenv.load_dotenv = isolated_load_dotenv
from synthdata.config import Config

assert os.environ["TABPFN_DISABLE_TELEMETRY"] == "1"
assert os.environ["SYNTHDATA_TEST_ENVIRONMENT"] == "loaded"
worker = subprocess.run(
    [sys.executable, "-c", 'import os; print(os.environ["TABPFN_DISABLE_TELEMETRY"])'],
    check=True, capture_output=True, text=True, timeout=30,
)
assert worker.stdout.strip() == "1"
assert network_attempts == []
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(dotenv_path)],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
