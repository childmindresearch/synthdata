"""Experiment tracking: every pipeline run is versioned, tagged, and logged.

`synthdata-generate` starts a new :class:`Experiment` each time it runs (a
timestamped id, optionally suffixed with a user-supplied ``--tag``, or an
explicit ``--experiment-id`` to resume/extend a previous one), and nests its
synthetic-data output under
``generation.output_dir/data_v_<dataset-version>/exp_v_<experiment_id>/``. That experiment
id is recorded as the "latest" experiment for this dataset *version*, so
`synthdata-evaluate` and `synthdata-plot` automatically pick it up (nesting
their own artifacts the same way) without the user needing to pass it again --
unless they explicitly want to target a different, earlier experiment via
``--experiment-id``.

A JSON manifest at
``<generation.output_dir>/../experiments/data_v_<dataset-version>/exp_v_<experiment_id>/manifest.json``
records what each stage produced (dataset version, git commit, artifact paths),
so any artifact can be traced back to exactly the run that produced it.
"""

import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synthdata.config import Config
from synthdata.data import role_context_fingerprint, role_context_payload
from synthdata.data_roles import ROLE_NAMES
from synthdata.utils import ensure_dir, get_logger, git_commit

logger = get_logger(__name__)

_LATEST_FILENAME = "latest.json"
_UNVERSIONED_ARTIFACT_SCOPE = "data_v_unversioned"


def _data_version_scope_label(version: str | None) -> str:
    """Return the human-readable directory label for a dataset version."""
    return f"data_v_{version or 'unversioned'}"


def _timestamp_id(tag: str | None = None) -> str:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{tag}" if tag else ts


def dataset_version_scope(cfg: Config) -> str:
    """Return a path-safe, stable scope for artifacts derived from this dataset.

    The configured version is deliberately a directory component: changing it
    must create a new artifact lineage rather than reuse cached outputs from a
    prior dataset revision. ``None`` remains supported but is explicitly
    segregated under ``unversioned`` so it cannot collide with versioned runs.
    """
    version = cfg.data.version
    if version is None:
        return _UNVERSIONED_ARTIFACT_SCOPE
    if (
        not isinstance(version, str)
        or not version
        or Path(version).name != version
        or version in {".", ".."}
    ):
        raise ValueError(
            "data.version must be a non-empty, path-safe label without path separators; "
            f"got {version!r}."
        )
    return _data_version_scope_label(version)


def dataset_plots_dir(cfg: Config) -> Path:
    """Return the version-scoped destination for dataset-level QA figures."""
    return ensure_dir(Path(cfg.plots.output_dir) / dataset_version_scope(cfg) / "dataset")


def imputation_output_dir(cfg: Config) -> Path:
    """Return the version-scoped destination for imputation artifacts."""
    return ensure_dir(Path("output") / cfg.name / "imputation" / dataset_version_scope(cfg))


def _experiments_root(cfg: Config) -> Path:
    return Path(cfg.generation.output_dir).parent / "experiments" / dataset_version_scope(cfg)


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
    role_context_fingerprint: str | None = None
    role_context: dict[str, Any] | None = None

    def validate_generation_context(
        self,
        context: dict[str, Any],
        fingerprint: str,
        *,
        full_context: dict[str, Any] | None = None,
        full_fingerprint: str | None = None,
    ) -> None:
        """Reject generation cache context from a different dataset snapshot."""
        if self.role_context_fingerprint is None or self.role_context is None:
            return
        if not fingerprint or not isinstance(context, dict):
            raise RuntimeError("Generation requires a complete validated role context")
        if full_context is not None and (
            self.role_context != full_context or self.role_context_fingerprint != full_fingerprint
        ):
            raise RuntimeError(
                "Generation dataset snapshot differs from experiment manifest lineage; "
                "refusing model execution."
            )
        shared_fields = (
            "dataset_name",
            "dataset_version",
            "assignment_policy_fingerprint",
            "semantic_fingerprint",
            "variable_schema_fingerprint",
            "compatibility_mode",
        )
        mismatches = {
            field: (self.role_context.get(field), context.get(field))
            for field in shared_fields
            if self.role_context.get(field) != context.get(field)
        }
        recorded_roles = self.role_context.get("roles", {})
        current_roles = context.get("roles", {})
        for role, current in current_roles.items():
            recorded = recorded_roles.get(role)
            if recorded != current:
                mismatches[f"roles.{role}"] = (recorded, current)
        if mismatches:
            raise RuntimeError(
                "Generation context differs from experiment dataset lineage; "
                f"refusing model execution (mismatches={mismatches})."
            )

    def record(self, stage: str, artifacts: dict[str, Any] | None = None, **extra: Any) -> None:
        """Append a stage entry to this experiment's manifest.json."""
        entry = {
            "stage": stage,
            "timestamp": datetime.now(UTC).isoformat(),
            "git_commit": git_commit(),
            "artifacts": artifacts or {},
            **extra,
        }
        if self.role_context_fingerprint is not None:
            entry["role_context_fingerprint"] = self.role_context_fingerprint
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
            **(
                {
                    "role_context_fingerprint": self.role_context_fingerprint,
                    "role_context": self.role_context,
                }
                if self.role_context_fingerprint is not None
                else {}
            ),
            "runs": [],
        }

    def _save_manifest(self, manifest: dict) -> None:
        ensure_dir(self.manifest_path.parent)
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)


def _build_experiment(
    experiment_id: str,
    cfg: Config,
    dataset=None,
    *,
    allow_final_holdout_handoff: bool = False,
    candidate_phase: bool = False,
) -> Experiment:
    scope = dataset_version_scope(cfg)
    experiment_scope = f"exp_v_{experiment_id}"
    experiment_root = _experiments_root(cfg) / experiment_scope
    manifest_path = experiment_root / "manifest.json"
    role_context = None
    role_context_digest = None
    if dataset is not None:
        role_names = ROLE_NAMES if dataset.has_canonical_roles else ("train", "final_holdout")
        role_context = role_context_payload(dataset, role_names, candidate_phase=candidate_phase)
        role_context_digest = role_context_fingerprint(
            dataset, role_names, candidate_phase=candidate_phase
        )
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        actual_identity = (manifest.get("dataset_name"), manifest.get("dataset_version"))
        expected_identity = (cfg.name, cfg.data.version)
        if actual_identity != expected_identity:
            raise ValueError(
                f"Refusing to resume experiment '{experiment_id}' at {manifest_path}: "
                f"manifest belongs to dataset {actual_identity[0]!r}@{actual_identity[1]!r}, "
                f"not {expected_identity[0]!r}@{expected_identity[1]!r}."
            )
        if role_context_digest is not None:
            recorded_context = manifest.get("role_context_fingerprint")
            recorded_payload = manifest.get("role_context")
            context_matches = recorded_context == role_context_digest
            if (
                not candidate_phase
                and not allow_final_holdout_handoff
                and isinstance(recorded_payload, dict)
                and dataset is not None
                and dataset.has_canonical_roles
                and dataset.role_frame("final_holdout", imputed=True) is not None
                and _is_candidate_final_holdout_baseline(recorded_payload)
            ):
                context_matches = False
            if (
                allow_final_holdout_handoff
                and isinstance(recorded_payload, dict)
                and role_context is not None
            ):
                context_matches = _final_holdout_handoff_matches(recorded_payload, role_context)
            if not context_matches:
                raise ValueError(
                    f"Refusing to resume experiment '{experiment_id}' at {manifest_path}: "
                    "resolved dataset role context does not match the recorded context "
                    f"(recorded={recorded_context!r}, current={role_context_digest!r})."
                )

    generation_dir = ensure_dir(Path(cfg.generation.output_dir) / scope / experiment_scope)
    evaluation_dir = ensure_dir(Path(cfg.evaluation.output_dir) / scope / experiment_scope)
    plots_dir = ensure_dir(Path(cfg.plots.output_dir) / scope / experiment_scope)
    ensure_dir(experiment_root)

    experiment = Experiment(
        id=experiment_id,
        tag=cfg.experiment.tag,
        dataset_name=cfg.name,
        dataset_version=cfg.data.version,
        generation_dir=generation_dir,
        evaluation_dir=evaluation_dir,
        plots_dir=plots_dir,
        manifest_path=manifest_path,
        created_at=datetime.now(UTC).isoformat(),
        git_commit=git_commit(),
        role_context_fingerprint=role_context_digest,
        role_context=role_context,
    )

    config_snapshot_path = experiment_root / "config_snapshot.json"
    if not config_snapshot_path.exists():
        with open(config_snapshot_path, "w") as f:
            json.dump(dataclasses.asdict(cfg), f, indent=2, default=str)

    return experiment


def _final_holdout_handoff_matches(recorded: dict[str, Any], current: dict[str, Any]) -> bool:
    """Return whether current context is the validated final-phase handoff.

    Generation records candidate role imputations before final-phase imputation
    exists. Evaluation may add only that derived final-holdout fingerprint;
    every other context field must remain identical.
    """
    if recorded == current:
        return True
    if set(recorded) != set(current):
        return False
    for field, recorded_value in recorded.items():
        current_value = current[field]
        if field != "roles":
            if recorded_value != current_value:
                return False
            continue
        if not isinstance(recorded_value, dict) or not isinstance(current_value, dict):
            return False
        if set(recorded_value) != set(current_value):
            return False
        for role, recorded_role in recorded_value.items():
            current_role = current_value[role]
            if (
                role != "final_holdout"
                or not isinstance(recorded_role, dict)
                or not isinstance(current_role, dict)
            ):
                if recorded_role != current_role:
                    return False
                continue
            if set(recorded_role) != set(current_role):
                return False
            for role_field, recorded_field in recorded_role.items():
                current_field = current_role[role_field]
                if role_field == "imputed_fingerprint":
                    raw_fingerprint = recorded_role.get("raw_fingerprint")
                    baseline = {None, raw_fingerprint}
                    if (
                        recorded_field not in baseline
                        or not isinstance(current_field, str)
                        or not current_field
                    ):
                        return False
                elif recorded_field != current_field:
                    return False
    return True


def _is_candidate_final_holdout_baseline(context: dict[str, Any]) -> bool:
    """Return whether context records the candidate final-holdout baseline."""
    role = context.get("roles", {}).get("final_holdout", {})
    return isinstance(role, dict) and role.get("imputed_fingerprint") in {
        None,
        role.get("raw_fingerprint"),
    }


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


def start_experiment(cfg: Config, dataset=None) -> Experiment:
    """Start (or explicitly resume) an experiment; used by `synthdata-generate`.

    If ``cfg.experiment.id`` is not set, a new timestamped id is generated
    (suffixed with ``cfg.experiment.tag`` if given) and recorded as the
    "latest" experiment for this dataset's output directory.
    """
    if cfg.experiment.id:
        experiment_id = cfg.experiment.id
    else:
        base_id = _timestamp_id(cfg.experiment.tag)
        experiment_id = base_id
        suffix = 1
        while (_experiments_root(cfg) / f"exp_v_{experiment_id}").exists():
            experiment_id = f"{base_id}_{suffix}"
            suffix += 1
    experiment = _build_experiment(experiment_id, cfg, dataset=dataset, candidate_phase=True)
    _write_latest_pointer(cfg, experiment_id)
    logger.info(
        "Experiment '%s' (tag=%s, dataset=%s@%s)",
        experiment.id,
        experiment.tag or "-",
        experiment.dataset_name,
        experiment.dataset_version or "unversioned",
    )
    return experiment


def load_experiment(
    cfg: Config, dataset=None, *, allow_final_holdout_handoff: bool = False
) -> Experiment:
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
    experiment = _build_experiment(
        experiment_id,
        cfg,
        dataset=dataset,
        allow_final_holdout_handoff=allow_final_holdout_handoff,
    )
    logger.info(
        "Loaded experiment '%s' (tag=%s, dataset=%s@%s)",
        experiment.id,
        experiment.tag or "-",
        experiment.dataset_name,
        experiment.dataset_version or "unversioned",
    )
    return experiment
