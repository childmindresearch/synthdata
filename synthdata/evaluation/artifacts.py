"""Durable artifacts needed to render evaluation plots without recomputing metrics.

The evaluation stage persists the full log-disparity report tables because
``combined_evaluation.csv`` deliberately contains only their summary metrics.
This lets ``synthdata-plot`` reconstruct Plotly reports from recorded results
rather than rerunning any evaluator.
"""

import hashlib
import json
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from synthdata.utils import ensure_dir, get_logger

logger = get_logger(__name__)

_ARTIFACT_SCHEMA_VERSION = 1
_BUNDLE_NAME = "evaluation_artifacts-v1"
_LOG_REPORT_TABLES = (
    "leaf_results",
    "hierarchy_results",
    "subgroup_table",
    "leaf_equity_table",
    "legend_table",
    "label_counts",
)


def artifact_bundle_dir(evaluation_dir: str | Path) -> Path:
    """Return the versioned artifact bundle directory for an evaluation."""
    return Path(evaluation_dir) / _BUNDLE_NAME


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_parquet(temporary)
    os.replace(temporary, path)


def _model_artifact_id(model_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_name).strip("-") or "model"
    return f"{slug}-{hashlib.sha256(model_name.encode()).hexdigest()[:12]}"


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def persist_evaluation_artifacts(
    evaluation_dir: str | Path,
    combined: pd.DataFrame,
    log_disparity_reports: dict[str, dict],
    *,
    native_syntheval_plot_dir: str | Path | None,
) -> Path:
    """Persist plot-ready evaluation outputs and return the bundle manifest path.

    Failed log-disparity model reports are recorded with their diagnostic
    context rather than omitted.  Every file is written atomically so a plot
    command never mistakes a partial report for a successful evaluation.
    """
    evaluation_dir = Path(evaluation_dir)
    bundle_dir = ensure_dir(artifact_bundle_dir(evaluation_dir))
    combined_path = evaluation_dir / "combined_evaluation.csv"
    if not combined_path.exists():
        raise FileNotFoundError(
            f"Cannot persist evaluation artifacts: combined table is missing at {combined_path}"
        )

    log_manifest: dict[str, dict[str, Any]] = {}
    log_root = ensure_dir(bundle_dir / "log_disparity")
    for model_name, report in sorted(log_disparity_reports.items()):
        model_dir = ensure_dir(log_root / _model_artifact_id(model_name))
        metadata_path = model_dir / "metadata.json"
        if "error" in report:
            metadata = {
                "state": "failed",
                "model_name": model_name,
                "error_type": report.get("error_type"),
                "error": report["error"],
            }
            _atomic_json(metadata_path, metadata)
            log_manifest[model_name] = {
                "path": str(model_dir.relative_to(bundle_dir)),
                "state": "failed",
                "metadata_sha256": _file_digest(metadata_path),
            }
            continue

        missing = [name for name in _LOG_REPORT_TABLES if name not in report]
        if missing:
            raise ValueError(
                f"Cannot persist log-disparity report for model {model_name!r}: "
                f"missing required table(s) {missing}."
            )
        metadata = {
            "state": "succeeded",
            "model_name": model_name,
            "summary_stats": report["summary_stats"],
            "protected_group_cols": report["protected_group_cols"],
            "protected_order_map": report["protected_order_map"],
            "target_order": report["target_order"],
        }
        _atomic_json(metadata_path, metadata)
        table_manifest = {}
        for name in _LOG_REPORT_TABLES:
            path = model_dir / f"{name}.parquet"
            _atomic_parquet(path, report[name])
            table_manifest[name] = {
                "filename": path.name,
                "sha256": _file_digest(path),
            }
        log_manifest[model_name] = {
            "path": str(model_dir.relative_to(bundle_dir)),
            "state": "succeeded",
            "metadata_sha256": _file_digest(metadata_path),
            "tables": table_manifest,
        }

    native_files = []
    if native_syntheval_plot_dir is not None:
        native_root = Path(native_syntheval_plot_dir)
        if native_root.exists():
            native_files = [
                {
                    "path": str(path.relative_to(native_root)),
                    "sha256": _file_digest(path),
                }
                for path in sorted(native_root.rglob("*"))
                if path.is_file()
            ]

    manifest = {
        "schema_version": _ARTIFACT_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "combined_evaluation": {
            "path": str(combined_path.relative_to(evaluation_dir)),
            "sha256": _file_digest(combined_path),
            "models": list(combined.index),
        },
        "log_disparity": log_manifest,
        "native_syntheval_plots": {
            "root": str(native_syntheval_plot_dir) if native_syntheval_plot_dir else None,
            "files": native_files,
        },
    }
    manifest_path = bundle_dir / "manifest.json"
    _atomic_json(manifest_path, manifest)
    logger.info(
        "[evaluation artifacts] persisted %d log-disparity report(s) and %d native SynthEval file(s) under %s",
        len(log_manifest),
        len(native_files),
        bundle_dir,
    )
    return manifest_path


def _load_manifest(evaluation_dir: str | Path) -> tuple[Path, dict]:
    bundle_dir = artifact_bundle_dir(evaluation_dir)
    path = bundle_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation artifact manifest is missing at {path}. Run `synthdata-evaluate` "
            "once with this version to persist plot-ready evaluation artifacts."
        )
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Evaluation artifact manifest is unreadable at {path}: {exc}") from exc
    if manifest.get("schema_version") != _ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported evaluation artifact schema at {path}: "
            f"{manifest.get('schema_version')!r}; expected {_ARTIFACT_SCHEMA_VERSION}."
        )
    return bundle_dir, manifest


def load_log_disparity_reports(evaluation_dir: str | Path) -> dict[str, dict]:
    """Load persisted log-disparity reports for offline Plotly rendering."""
    bundle_dir, manifest = _load_manifest(evaluation_dir)
    reports = {}
    for model_name, entry in manifest.get("log_disparity", {}).items():
        model_dir = bundle_dir / entry["path"]
        metadata_path = model_dir / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Log-disparity metadata for model {model_name!r} is unreadable at "
                f"{metadata_path}: {exc}"
            ) from exc
        if metadata.get("state") == "failed":
            reports[model_name] = {
                "error": metadata.get("error", "unknown persisted failure"),
                "error_type": metadata.get("error_type", "UnknownError"),
            }
            continue
        if metadata.get("state") != "succeeded":
            raise ValueError(
                f"Log-disparity artifact for model {model_name!r} has invalid state "
                f"{metadata.get('state')!r} at {metadata_path}."
            )
        report = {
            "summary_stats": metadata["summary_stats"],
            "protected_group_cols": metadata["protected_group_cols"],
            "protected_order_map": metadata["protected_order_map"],
            "target_order": metadata["target_order"],
        }
        for table_name, table_entry in entry.get("tables", {}).items():
            path = model_dir / table_entry["filename"]
            if not path.exists():
                raise FileNotFoundError(
                    f"Log-disparity table {table_name!r} for model {model_name!r} is missing at {path}."
                )
            if _file_digest(path) != table_entry["sha256"]:
                raise ValueError(
                    f"Log-disparity table {table_name!r} for model {model_name!r} failed integrity "
                    f"verification at {path}."
                )
            report[table_name] = pd.read_parquet(path)
        missing = [name for name in _LOG_REPORT_TABLES if name not in report]
        if missing:
            raise ValueError(
                f"Log-disparity artifact for model {model_name!r} is incomplete; missing {missing}."
            )
        reports[model_name] = report
    return reports


def verify_native_syntheval_artifacts(evaluation_dir: str | Path) -> None:
    """Fail loudly if a recorded native SynthEval plot was deleted or changed."""
    _bundle_dir, manifest = _load_manifest(evaluation_dir)
    native = manifest.get("native_syntheval_plots", {})
    root_value = native.get("root")
    files = native.get("files", [])
    if root_value is None:
        logger.info(
            "[evaluation artifacts] native SynthEval plots were disabled for this evaluation"
        )
        return
    if not files:
        raise FileNotFoundError(
            f"No native SynthEval plot files were recorded under {root_value}. "
            "Re-run `synthdata-evaluate` to regenerate its native diagnostics."
        )
    root = Path(root_value)
    missing_or_changed = []
    for entry in files:
        path = root / entry["path"]
        if not path.exists() or _file_digest(path) != entry["sha256"]:
            missing_or_changed.append(str(path))
    if missing_or_changed:
        raise FileNotFoundError(
            "Native SynthEval plot artifact(s) are missing or changed: "
            + ", ".join(missing_or_changed[:10])
            + (" ..." if len(missing_or_changed) > 10 else "")
            + ". Re-run `synthdata-evaluate` to regenerate them."
        )
    logger.info("[evaluation artifacts] verified %d native SynthEval plot file(s)", len(files))
