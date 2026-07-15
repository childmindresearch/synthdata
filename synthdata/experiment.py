"""Experiment tracking: every pipeline run is versioned, tagged, and logged.

`synthdata-generate` starts a new :class:`Experiment` each time it runs (a
timestamped id, optionally suffixed with a user-supplied ``--tag``, or an
explicit ``--experiment-id`` to resume/extend a previous one), and nests its
synthetic-data output under ``generation.output_dir/<experiment_id>/``. That
experiment id is recorded as the "latest" experiment for this dataset, so
`synthdata-evaluate` and `synthdata-plot` automatically pick it up (nesting
their own artifacts the same way) without the user needing to pass it again --
unless they explicitly want to target a different, earlier experiment via
``--experiment-id``.

A JSON manifest at ``<generation.output_dir>/../experiments/<experiment_id>/manifest.json``
records what each stage produced (dataset version, git commit, artifact paths),
so any artifact can be traced back to exactly the run that produced it.
"""

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from synthdata.config import Config
from synthdata.utils import ensure_dir, get_logger, git_commit

logger = get_logger(__name__)

_LATEST_FILENAME = "latest.json"


def _timestamp_id(tag: str | None = None) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{tag}" if tag else ts


def _experiments_root(cfg: Config) -> Path:
    return Path(cfg.generation.output_dir).parent / "experiments"


@dataclasses.dataclass
class Experiment:
    """A single versioned pipeline run.

    Use :meth:`record` after each stage (generation/evaluation/plots) to
    append an entry to this experiment's ``manifest.json``.
    """

    id: str
    tag: str | None
    dataset_name: str
    dataset_version: str | None
    generation_dir: Path
    evaluation_dir: Path
    plots_dir: Path
    manifest_path: Path
    created_at: str
    git_commit: str | None

    def record(self, stage: str, artifacts: dict[str, Any] | None = None, **extra: Any) -> None:
        """Append a stage entry to this experiment's manifest.json."""
        entry = {
            "stage": stage,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "artifacts": artifacts or {},
            **extra,
        }
        manifest = self._load_manifest()
        manifest.setdefault("runs", []).append(entry)
        self._save_manifest(manifest)
        logger.info("[experiment %s] recorded stage=%s", self.id, stage)

    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return {
            "experiment_id": self.id,
            "tag": self.tag,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "created_at": self.created_at,
            "git_commit": self.git_commit,
            "runs": [],
        }

    def _save_manifest(self, manifest: dict) -> None:
        ensure_dir(self.manifest_path.parent)
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)


def _build_experiment(experiment_id: str, cfg: Config) -> Experiment:
    generation_dir = ensure_dir(Path(cfg.generation.output_dir) / experiment_id)
    evaluation_dir = ensure_dir(Path(cfg.evaluation.output_dir) / experiment_id)
    plots_dir = ensure_dir(Path(cfg.plots.output_dir) / experiment_id)

    experiment_root = ensure_dir(_experiments_root(cfg) / experiment_id)
    manifest_path = experiment_root / "manifest.json"

    experiment = Experiment(
        id=experiment_id,
        tag=cfg.experiment.tag,
        dataset_name=cfg.name,
        dataset_version=cfg.data.version,
        generation_dir=generation_dir,
        evaluation_dir=evaluation_dir,
        plots_dir=plots_dir,
        manifest_path=manifest_path,
        created_at=datetime.now(timezone.utc).isoformat(),
        git_commit=git_commit(),
    )

    config_snapshot_path = experiment_root / "config_snapshot.json"
    if not config_snapshot_path.exists():
        with open(config_snapshot_path, "w") as f:
            json.dump(dataclasses.asdict(cfg), f, indent=2, default=str)

    return experiment


def _write_latest_pointer(cfg: Config, experiment_id: str) -> None:
    path = ensure_dir(_experiments_root(cfg)) / _LATEST_FILENAME
    with open(path, "w") as f:
        json.dump({"experiment_id": experiment_id}, f, indent=2)


def _read_latest_pointer(cfg: Config) -> str | None:
    path = _experiments_root(cfg) / _LATEST_FILENAME
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f).get("experiment_id")


def start_experiment(cfg: Config) -> Experiment:
    """Start (or explicitly resume) an experiment; used by `synthdata-generate`.

    If ``cfg.experiment.id`` is not set, a new timestamped id is generated
    (suffixed with ``cfg.experiment.tag`` if given) and recorded as the
    "latest" experiment for this dataset's output directory.
    """
    experiment_id = cfg.experiment.id or _timestamp_id(cfg.experiment.tag)
    experiment = _build_experiment(experiment_id, cfg)
    _write_latest_pointer(cfg, experiment_id)
    logger.info(
        "Experiment '%s' (tag=%s, dataset=%s@%s)",
        experiment.id,
        experiment.tag or "-",
        experiment.dataset_name,
        experiment.dataset_version or "unversioned",
    )
    return experiment


def load_experiment(cfg: Config) -> Experiment:
    """Load a previously-started experiment; used by `synthdata-evaluate`/`synthdata-plot`.

    Resolution order: ``cfg.experiment.id`` if explicitly set, else the
    "latest" experiment started by `synthdata-generate` for this dataset's
    output directory. Raises if neither is available.
    """
    experiment_id = cfg.experiment.id or _read_latest_pointer(cfg)
    if experiment_id is None:
        raise FileNotFoundError(
            "No experiment found to load. Run `synthdata-generate` first, or pass "
            "--experiment-id to target a specific past experiment."
        )
    experiment = _build_experiment(experiment_id, cfg)
    logger.info(
        "Loaded experiment '%s' (tag=%s, dataset=%s@%s)",
        experiment.id,
        experiment.tag or "-",
        experiment.dataset_name,
        experiment.dataset_version or "unversioned",
    )
    return experiment
