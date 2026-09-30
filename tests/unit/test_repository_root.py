"""Source-checkout detection for repository-local test scratch."""

import ast
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

pytestmark = pytest.mark.unit

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ROOT_HELPERS = [
    ("tests/conftest.py", "_selected_repository_root"),
    ("submodules/synthcity/tests/conftest.py", "_repository_root"),
    ("submodules/synthcity/tests/nb_eval.py", "_repository_root"),
    ("submodules/synthcity/tests/utils/test_optimizer.py", "_repository_root"),
    ("submodules/synthcity/tests/metrics/test_performance.py", "_repository_root"),
    ("submodules/synthcity/tests/benchmarks/test_benchmarks.py", "_repository_root"),
    ("submodules/syntheval/tests/conftest.py", "_selected_repository_root"),
]


def _root_selector(source: str, name: str, location: Path) -> Callable[[], Path]:
    """Execute the actual helper without importing optional models or notebooks."""
    source_path = REPOSITORY_ROOT / source
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {"Path": Path, "__file__": str(location)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), namespace)
    return cast(Callable[[], Path], namespace[name])


def _isolate_markers(monkeypatch: pytest.MonkeyPatch, fixture_root: Path) -> None:
    """Prevent the enclosing real checkout from satisfying a fixture's markers."""
    is_file = Path.is_file
    monkeypatch.setattr(
        Path, "is_file", lambda path: path.is_relative_to(fixture_root) and is_file(path)
    )


def _checkout_location(root: Path, source: str) -> Path:
    location = root / source
    location.parent.mkdir(parents=True, exist_ok=True)
    location.write_text("", encoding="utf-8")
    if source.startswith("submodules/"):
        fork_root = root / Path(source).parts[0] / Path(source).parts[1]
        (fork_root / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
        (fork_root / ".git").write_text("gitdir: local-test-marker\n", encoding="utf-8")
    return location


@pytest.mark.parametrize(("source", "name"), ROOT_HELPERS)
def test_source_checkout_is_selected_from_package_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, name: str
) -> None:
    checkout = tmp_path / "checkout"
    package = checkout / "synthdata"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (checkout / "pyproject.toml").write_text('[project]\nname = "synthdata"\n', encoding="utf-8")
    location = _checkout_location(checkout, source)
    select_root = _root_selector(source, name, location)

    with monkeypatch.context() as isolated:
        _isolate_markers(isolated, tmp_path)
        assert select_root() == checkout


@pytest.mark.parametrize(("source", "name"), ROOT_HELPERS[1:])
def test_standalone_fork_checkout_is_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, name: str
) -> None:
    checkout = tmp_path / "standalone"
    relative_source = str(Path(*Path(source).parts[2:]))
    location = _checkout_location(checkout, relative_source)
    (checkout / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
    (checkout / ".git").mkdir()
    select_root = _root_selector(source, name, location)

    with monkeypatch.context() as isolated:
        _isolate_markers(isolated, tmp_path)
        assert select_root() == checkout


@pytest.mark.parametrize(("source", "name"), ROOT_HELPERS)
def test_pyproject_alone_does_not_identify_synthdata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, name: str
) -> None:
    (tmp_path / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
    location = _checkout_location(tmp_path, source)
    select_root = _root_selector(source, name, location)

    with monkeypatch.context() as isolated:
        _isolate_markers(isolated, tmp_path)
        if source == "tests/conftest.py":
            with pytest.raises(RuntimeError, match="markers are absent"):
                select_root()
        else:
            assert select_root() == tmp_path / Path(source).parts[0] / Path(source).parts[1]
