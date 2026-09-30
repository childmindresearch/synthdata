"""Shared pytest fixtures for the synthdata test suite.

Fixtures here are deliberately in-memory / tmp_path-rooted so unit tests never
touch the real ``data/``/``output/`` directories or require network access.
"""

import os
import stat
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from synthdata.config import (
    Config,
    DataConfig,
    DataSplitConfig,
    EvaluationConfig,
    ExperimentConfig,
    GenerationConfig,
    PlotsConfig,
)
from synthdata.data import Dataset
from synthdata.data_roles import allocate_roles, resolve_population_identity
from synthdata.utils import ensure_dir


def _selected_repository_root() -> Path:
    """Find the nearest SynthData source checkout containing this conftest."""
    conftest_path = Path(__file__).resolve()
    for candidate in conftest_path.parents:
        resolved_candidate = candidate.resolve()
        if (resolved_candidate / "synthdata" / "__init__.py").is_file() and (
            resolved_candidate / "pyproject.toml"
        ).is_file():
            return resolved_candidate
    raise RuntimeError(
        "Unable to locate SynthData repository root from tests/conftest.py: "
        "required synthdata/__init__.py and pyproject.toml markers are absent"
    )


def _reject_symlink_components(path: Path) -> None:
    """Reject scratch path components that could redirect pytest outside checkout."""
    current = Path(path.anchor) if path.is_absolute() else Path()
    for component in path.parts[1:] if path.is_absolute() else path.parts:
        current /= component
        if current.is_symlink():
            raise RuntimeError(f"Refusing symlink in pytest scratch path: {current}")


def _secure_directory_fd(name: str, parent_fd: int, mode: int, label: str) -> int:
    """Create/open one directory below ``parent_fd`` without following links."""
    try:
        os.mkdir(name, mode=mode, dir_fd=parent_fd)
    except FileExistsError:
        pass
    except OSError as exc:
        raise RuntimeError(f"Unable to create {label}: {name}") from exc

    no_follow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if not no_follow or not directory:
        raise RuntimeError("Pinned platform lacks no-follow directory support")
    directory_fd: int | None = None
    try:
        directory_fd = os.open(
            name,
            os.O_RDONLY | directory | no_follow,
            dir_fd=parent_fd,
        )
        directory_stat = os.fstat(directory_fd)
        if not stat.S_ISDIR(directory_stat.st_mode):
            raise RuntimeError(f"Refusing {label} that is not a directory: {name}")
        if directory_stat.st_uid != os.geteuid():
            raise RuntimeError(f"Refusing {label} not owned by current user: {name}")
        if directory_stat.st_mode & 0o022:
            raise RuntimeError(f"Refusing group/world-writable {label}: {name}")
        os.fchmod(directory_fd, mode)
        return directory_fd
    except RuntimeError:
        if directory_fd is not None:
            os.close(directory_fd)
        raise
    except OSError as exc:
        if directory_fd is not None:
            os.close(directory_fd)
        raise RuntimeError(f"Unable to secure {label}: {name}") from exc


def _validate_final_basetemp_boundary(
    basetemp: Path,
    repository_root: Path,
    expected_stats: tuple[os.stat_result, os.stat_result, os.stat_result],
) -> None:
    """Reopen every basetemp component before pytest consumes its pathname.

    Pytest accepts only a pathname, so descriptors cannot pin this path through
    its internal setup. This is the final security boundary: no-follow opens,
    ownership/type/mode checks, containment checks, and identity comparisons
    reject replacement before handing the pathname to pytest.
    """
    if not basetemp.is_absolute() or not basetemp.is_relative_to(repository_root):
        raise RuntimeError(f"Refusing pytest basetemp outside repository: {basetemp}")
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if not no_follow or not directory:
        raise RuntimeError("Pinned platform lacks no-follow directory support")

    descriptors: list[int] = []
    labels = ("repository root", "pytest scratch root", "pytest basetemp")
    try:
        try:
            descriptors.append(os.open(str(repository_root), os.O_RDONLY | directory | no_follow))
            descriptors.append(
                os.open("tmp", os.O_RDONLY | directory | no_follow, dir_fd=descriptors[0])
            )
            descriptors.append(
                os.open(basetemp.name, os.O_RDONLY | directory | no_follow, dir_fd=descriptors[1])
            )
        except OSError as exc:
            raise RuntimeError(f"Unable to reopen pytest basetemp securely: {basetemp}") from exc

        for index, descriptor in enumerate(descriptors):
            current = os.fstat(descriptor)
            expected = expected_stats[index]
            if (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino):
                raise RuntimeError(f"Refusing replaced {labels[index]}: {basetemp}")
            if not stat.S_ISDIR(current.st_mode):
                raise RuntimeError(f"Refusing {labels[index]} that is not a directory: {basetemp}")
            if current.st_uid != os.geteuid():
                raise RuntimeError(
                    f"Refusing {labels[index]} not owned by current user: {basetemp}"
                )
            if index > 0 and current.st_mode & 0o022:
                raise RuntimeError(f"Refusing group/world-writable {labels[index]}: {basetemp}")
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Set one unique, repository-local basetemp before pytest creates fixtures."""
    repository_root = _selected_repository_root()
    scratch_root = repository_root / "tmp"
    if not scratch_root.is_absolute() or not scratch_root.is_relative_to(repository_root):
        raise RuntimeError(f"Refusing pytest scratch root outside repository: {scratch_root}")
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if not no_follow or not directory:
        raise RuntimeError("Pinned platform lacks no-follow directory support")
    try:
        repository_fd = os.open(str(repository_root), os.O_RDONLY | directory | no_follow)
    except OSError as exc:
        raise RuntimeError(f"Unable to open repository root securely: {repository_root}") from exc
    try:
        scratch_fd = _secure_directory_fd("tmp", repository_fd, 0o700, "pytest scratch root")
        try:
            repository_stat = os.fstat(repository_fd)
            scratch_stat = os.fstat(scratch_fd)
            if scratch_stat.st_uid != os.geteuid() or scratch_stat.st_mode & 0o022:
                raise RuntimeError(f"Refusing unsafe pytest scratch root: {scratch_root}")
            run_id = f"{os.getpid()}-{time.time_ns()}-{uuid4().hex}"
            basetemp = scratch_root / run_id
            basetemp_fd = _secure_directory_fd(run_id, scratch_fd, 0o700, "pytest basetemp")
            basetemp_stat = os.fstat(basetemp_fd)
            os.close(basetemp_fd)
            # Pytest only accepts a pathname; perform final no-follow validation
            # immediately before assigning it, then pytest may consume it.
            _validate_final_basetemp_boundary(
                basetemp,
                repository_root,
                (repository_stat, scratch_stat, basetemp_stat),
            )
            config.option.basetemp = str(basetemp)
        finally:
            os.close(scratch_fd)
    finally:
        os.close(repository_fd)


@pytest.fixture(autouse=True)
def patient_id_hmac_secret(monkeypatch, tmp_path: Path):
    """Provide only test-local key material for canonical identity resolution."""
    scratch_root = (_selected_repository_root() / "tmp").resolve()
    _reject_symlink_components(tmp_path)
    if not tmp_path.is_absolute() or not tmp_path.resolve().is_relative_to(scratch_root):
        raise AssertionError(f"pytest tmp_path escaped repository scratch: {tmp_path}")
    monkeypatch.setenv("SYNTHDATA_PATIENT_ID_HMAC_KEY", "unit-test-only-patient-id-secret")


@pytest.fixture
def sample_mixed_df() -> pd.DataFrame:
    """A small (30-row) DataFrame mixing numeric, string-categorical, a {1,2}
    binary quirk column, and injected missingness -- for synthdata.data's
    pure column-typing/transform functions.
    """
    rng = np.random.default_rng(0)
    n = 30
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 90, size=n).astype(float),
            "score": rng.normal(50, 10, size=n),
            "smoker": rng.choice(["Light", "Heavy"], size=n),
            "binary_flag": rng.choice([1, 2], size=n),
            "group": rng.integers(0, 3, size=n),
            "target": rng.integers(0, 2, size=n),
        }
    )
    df.loc[df.index[:5], "age"] = np.nan
    df.loc[df.index[5:8], "score"] = np.nan
    return df


@pytest.fixture
def make_config(tmp_path):
    """Factory building a minimal :class:`~synthdata.config.Config` rooted at
    ``tmp_path`` (construction alone performs no I/O).
    """

    def _make_config(
        name: str = "testds",
        tag: str | None = None,
        experiment_id: str | None = None,
        data_version: str | None = None,
    ) -> Config:
        output_root = tmp_path / "output" / name
        return Config(
            name=name,
            data=DataConfig(
                source="csv",
                path=str(tmp_path / "raw.csv"),
                target_column="target",
                version=data_version,
            ),
            generation=GenerationConfig(output_dir=str(output_root / "synthetic_data")),
            evaluation=EvaluationConfig(output_dir=str(output_root / "evaluation")),
            plots=PlotsConfig(output_dir=str(output_root / "plots")),
            experiment=ExperimentConfig(tag=tag, id=experiment_id),
        )

    return _make_config


@pytest.fixture
def make_dataset(tmp_path, sample_mixed_df):
    """Factory building a :class:`~synthdata.data.Dataset` from small in-memory
    DataFrames, without going through :func:`synthdata.data.load_dataset`'s
    real I/O (UCI fetch/CSV read).
    """

    def _make_dataset(
        df: pd.DataFrame | None = None,
        target_column: str = "target",
        feature_columns: list | None = None,
        nominal_columns: list | None = None,
        ordinal_columns: list | None = None,
        sensitive_columns: list | None = None,
        name: str = "testds",
    ) -> Dataset:
        df = sample_mixed_df.copy() if df is None else df
        feature_columns = feature_columns or [c for c in df.columns if c != target_column]
        nominal_columns = nominal_columns if nominal_columns is not None else []
        ordinal_columns = ordinal_columns if ordinal_columns is not None else []
        sensitive_columns = sensitive_columns if sensitive_columns is not None else []
        data_dir = ensure_dir(tmp_path / "data" / name)
        train_df, test_df = train_test_split(df, train_size=0.7, random_state=0)
        return Dataset(
            name=name,
            target_column=target_column,
            feature_columns=feature_columns,
            nominal_columns=nominal_columns,
            ordinal_columns=ordinal_columns,
            sensitive_columns=sensitive_columns,
            data_dir=data_dir,
            full_df=df,
            train_df=train_df,
            test_df=test_df,
        )

    return _make_dataset


@pytest.fixture
def make_canonical_dataset(tmp_path):
    """Factory for a small canonical dataset with explicit population roles."""

    def _make(identity_mode: str = "column") -> Dataset:
        if identity_mode == "one_row_per_patient":
            raise ValueError("fixture no longer supports row-index identity")
        else:
            patient_ids = np.repeat(np.arange(1, 13), 2)
            frame = pd.DataFrame(
                {
                    "feature": np.arange(24, dtype=float),
                    "protected": np.repeat(["A"] * 6 + ["B"] * 6, 2),
                    "target": np.tile([0, 1], 12),
                }
            )
            if identity_mode == "column":
                frame.insert(0, "patient_id", patient_ids)
                split = DataSplitConfig(
                    mode="patient_group",
                    train_fraction=0.5,
                    tuning_fraction=0.25,
                    final_holdout_fraction=0.25,
                    candidate_count=64,
                    patient_id_column="patient_id",
                )
            elif identity_mode == "mapping":
                frame.insert(0, "row_id", np.arange(24))
                mapping_path = tmp_path / "patient_identity.csv"
                pd.DataFrame({"row_id": np.arange(24), "patient_id": patient_ids}).to_csv(
                    mapping_path, index=False
                )
                split = DataSplitConfig(
                    mode="patient_group",
                    train_fraction=0.5,
                    tuning_fraction=0.25,
                    final_holdout_fraction=0.25,
                    candidate_count=64,
                    identity_mapping_path=str(mapping_path),
                    mapping_row_key_column="row_id",
                    mapping_patient_key_column="patient_id",
                )
            else:
                raise ValueError(f"Unknown fixture identity mode: {identity_mode!r}")

        identity = resolve_population_identity(frame, split)
        model_frame = identity.model_frame.reset_index(drop=True)
        groups = identity.groups.reset_index(drop=True) if identity.groups is not None else None
        assignment = allocate_roles(
            model_frame,
            "target",
            split,
            protected_columns=["protected"],
            groups=groups,
            seed=17,
        )
        data_dir = ensure_dir(tmp_path / "data" / f"canonical_{identity_mode}")
        dataset = Dataset(
            name="canonical_fixture",
            target_column="target",
            feature_columns=["feature", "protected"],
            nominal_columns=["protected"],
            ordinal_columns=[],
            sensitive_columns=["protected"],
            data_dir=data_dir,
            full_df=model_frame,
            train_df=None,
            test_df=None,
            roles=assignment.frames,
            role_groups=assignment.groups,
            assignment=assignment.assignment,
            role_metadata={"identity": identity.metadata, "split": assignment.metadata},
            version="fixture-1",
            protected_columns=["protected"],
            variable_schema={
                "feature": {"kind": "continuous", "ordinal_order": None},
                "protected": {"kind": "categorical", "ordinal_order": None},
                "target": {"kind": "categorical", "ordinal_order": None},
            },
            assignment_fingerprint=assignment.assignment_fingerprint,
            identity_sidecar=identity.identity_sidecar,
        )
        dataset.set_imputed_roles(
            {role: role_frame.copy() for role, role_frame in assignment.frames.items()}
        )
        return dataset

    return _make


@pytest.fixture
def real_synth_pair():
    """Small paired real/synthetic DataFrames with two protected columns and a
    binary target, with deliberately shifted proportions in ``sex`` -- for
    log_disparity/evaluation-combine tests.
    """
    rng = np.random.default_rng(42)
    n = 40
    real = pd.DataFrame(
        {
            "sex": rng.choice(["M", "F"], size=n, p=[0.5, 0.5]),
            "age_group": rng.choice(["young", "old"], size=n, p=[0.6, 0.4]),
            "outcome": rng.integers(0, 2, size=n),
        }
    )
    synth = pd.DataFrame(
        {
            "sex": rng.choice(["M", "F"], size=n, p=[0.2, 0.8]),
            "age_group": rng.choice(["young", "old"], size=n, p=[0.6, 0.4]),
            "outcome": rng.integers(0, 2, size=n),
        }
    )
    return real, synth
