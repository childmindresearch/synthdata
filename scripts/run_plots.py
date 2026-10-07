#!/usr/bin/env python
"""CLI: (re)generate every figure from already-computed artifacts on disk.

Useful for re-plotting without re-running expensive imputation/generation
stages. Controlled by ``plots.sections`` in the config (``data``, ``imputation``,
``generation``, ``hpo``, ``evaluation``).

The ``data``/``imputation`` sections describe the dataset itself and are saved
under ``plots.output_dir/<dataset-version>/dataset/`` (shared across
experiments for that version only). The ``generation``/``hpo``/``evaluation``
sections are experiment-specific and are nested under
``plots.output_dir/<dataset-version>/<experiment_id>/``, resolved the same way as
`synthdata-evaluate` (most recent experiment, or ``--experiment-id``).

The "evaluation" section is artifact-only: rank plots are redrawn from the
combined evaluation CSV and log-disparity reports are rebuilt from the
evaluation artifact bundle. Native SynthEval diagnostics are created during
evaluation and verified here; this command never reruns evaluation metrics.

Usage:
    synthdata-plot --config path/to/your-config.yaml [--experiment-id ID] [--dataset-version v2]
"""

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from synthdata.config import load_config
from synthdata.data import load_dataset, load_imputed_splits
from synthdata.experiment import dataset_plots_dir, load_experiment
from synthdata.imputation.pipeline import _cache_key_record
from synthdata.utils import get_logger, set_global_seed

logger = get_logger("run_plots")

_EXPERIMENT_SECTIONS = {"generation", "hpo", "evaluation"}


def _mapping_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _matches_final_holdout_handoff(
    candidate_context: Mapping[str, Any], evaluation_context: Mapping[str, Any]
) -> bool:
    """Check evaluation context differs from candidate cache only at final-holdout imputation."""
    if set(candidate_context) != set(evaluation_context):
        return False
    for field, candidate_value in candidate_context.items():
        evaluation_value = evaluation_context[field]
        if field != "roles":
            if candidate_value != evaluation_value:
                return False
            continue
        if not isinstance(candidate_value, Mapping) or not isinstance(evaluation_value, Mapping):
            return False
        if set(candidate_value) != set(evaluation_value):
            return False
        for role, candidate_role in candidate_value.items():
            evaluation_role = evaluation_value[role]
            if not isinstance(candidate_role, Mapping) or not isinstance(evaluation_role, Mapping):
                return False
            if set(candidate_role) != set(evaluation_role):
                return False
            if role != "final_holdout":
                if dict(candidate_role) != dict(evaluation_role):
                    return False
                continue
            for role_field, candidate_value in candidate_role.items():
                evaluation_value = evaluation_role[role_field]
                if role_field == "imputed_fingerprint":
                    raw_fingerprint = candidate_role.get("raw_fingerprint")
                    if candidate_value != raw_fingerprint:
                        return False
                    if not isinstance(evaluation_value, str) or not evaluation_value:
                        return False
                elif candidate_value != evaluation_value:
                    return False
    return True


def _expected_plot_context(dataset, evaluation_dir: str | Path) -> dict[str, Any]:
    """Use validated evaluation identity, allowing only its recorded holdout handoff."""
    from synthdata.data import role_context_payload
    from synthdata.evaluation.artifacts import artifact_bundle_dir, expected_evaluation_context

    expected = expected_evaluation_context(dataset)
    if getattr(dataset, "legacy_two_role", True):
        return expected

    manifest_path = artifact_bundle_dir(evaluation_dir) / "manifest.json"
    if not manifest_path.is_file():
        return expected
    manifest = json.loads(manifest_path.read_text())
    recorded_fingerprints = manifest.get("role_context_fingerprint")
    if (
        not isinstance(recorded_fingerprints, Mapping)
        or dict(recorded_fingerprints) == expected["role_context_fingerprints"]
        or "final_holdout_evidence" not in manifest
        or set(recorded_fingerprints) != set(expected["role_context_fingerprints"])
    ):
        return expected

    recorded_contexts = manifest.get("role_context")
    if not isinstance(recorded_contexts, Mapping) or set(recorded_contexts) != set(
        expected["role_context_fingerprints"]
    ):
        return expected
    recorded_candidate = recorded_contexts.get("candidate")
    recorded_full = recorded_contexts.get("full")
    current_candidate = role_context_payload(dataset, ("train", "tuning"))
    current_full = role_context_payload(
        dataset, ("train", "tuning", "final_holdout"), candidate_phase=True
    )
    from synthdata.data import dataframe_fingerprint

    raw_final_holdout = dataset.role_frame("final_holdout", imputed=False)
    imputed_final_holdout = dataset.role_frame("final_holdout", imputed=True)
    if raw_final_holdout is None or (
        imputed_final_holdout is not None
        and dataframe_fingerprint(imputed_final_holdout) != dataframe_fingerprint(raw_final_holdout)
    ):
        return expected
    if (
        not isinstance(recorded_candidate, Mapping)
        or not isinstance(recorded_full, Mapping)
        or dict(recorded_candidate) != current_candidate
        or recorded_fingerprints.get("candidate")
        != expected["role_context_fingerprints"].get("candidate")
        or _mapping_digest(recorded_candidate) != recorded_fingerprints.get("candidate")
        or _mapping_digest(recorded_full) != recorded_fingerprints.get("full")
        or not _matches_final_holdout_handoff(current_full, recorded_full)
    ):
        return expected

    handoff_expected = dict(expected)
    handoff_expected["role_context_fingerprints"] = dict(recorded_fingerprints)
    return handoff_expected


def _load_synthetic_datasets(cfg) -> dict:
    output_dir = Path(cfg.generation.output_dir)
    if not output_dir.exists():
        return {}
    return {path.stem: pd.read_csv(path) for path in sorted(output_dir.glob("*.csv"))}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate all figures for the sections listed in plots.sections."
    )
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Plot a specific past experiment's generation/hpo/evaluation figures "
        "instead of the most recent one (overrides experiment.id). Ignored if "
        "plots.sections has no experiment-specific sections.",
    )
    parser.add_argument(
        "--dataset-version",
        default=None,
        help="Override data.version and select that version's artifact lineage.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.experiment_id:
        cfg.experiment.id = args.experiment_id
    if args.dataset_version:
        cfg.data.version = args.dataset_version
    set_global_seed(cfg.seed)
    sections = set(cfg.plots.sections)
    logger.info("Plotting sections: %s", sorted(sections))

    dataset = load_dataset(cfg)
    dataset = load_imputed_splits(
        dataset,
        expected_cache_key=_cache_key_record(cfg, dataset)["cache_key"],
    )

    if "data" in sections:
        from synthdata.plotting.data_plots import save_data_plots

        save_data_plots(dataset, dataset_plots_dir(cfg), cfg.plots.dpi, cfg.plots.formats)

    if "imputation" in sections and dataset.full_imputed_df is not None:
        from synthdata.imputation import build_validation_report
        from synthdata.plotting.imputation_plots import save_imputation_plots

        validation_df = build_validation_report(cfg, dataset)
        save_imputation_plots(cfg, dataset, validation_df, dataset_plots_dir(cfg))

    experiment = None
    if sections & _EXPERIMENT_SECTIONS:
        experiment = load_experiment(cfg, dataset=dataset, allow_final_holdout_handoff=True)
        cfg.generation.output_dir = str(experiment.generation_dir)
        cfg.evaluation.output_dir = str(experiment.evaluation_dir)
        cfg.plots.output_dir = str(experiment.plots_dir)

    synthetic_datasets = _load_synthetic_datasets(cfg)

    if (
        "generation" in sections
        and synthetic_datasets
        and dataset.role_frame("train", imputed=True) is not None
    ):
        from synthdata.plotting.generation_plots import save_generation_plots

        save_generation_plots(cfg, dataset, synthetic_datasets, cfg.plots.output_dir)

    if "hpo" in sections:
        from synthdata.plotting.generation_plots import save_hpo_plots

        save_hpo_plots(cfg, cfg.plots.output_dir)

    if "evaluation" in sections:
        from synthdata.evaluation.artifacts import (
            artifact_bundle_dir,
            load_generation_inventory,
            load_log_disparity_reports,
            validate_evaluation_bundle,
            verify_native_syntheval_artifacts,
        )
        from synthdata.evaluation.combine import load_combined_table
        from synthdata.plotting.evaluation_plots import (
            save_log_disparity_plots,
            save_rank_tradeoff_plots,
        )

        combined_path = Path(cfg.evaluation.output_dir) / "combined_evaluation.csv"
        if not combined_path.exists():
            raise FileNotFoundError(
                f"Evaluation table not found at {combined_path}. "
                "Run `synthdata-evaluate --config <path>` first."
            )
        expected_context = _expected_plot_context(dataset, cfg.evaluation.output_dir)
        evaluation_manifest = validate_evaluation_bundle(
            cfg.evaluation.output_dir,
            expected_config_path=cfg.config_path,
            expected_role_context_fingerprints=expected_context["role_context_fingerprints"],
            expected_role_hashes=expected_context["role_hashes"],
            expected_role_hashes_by_framework=expected_context["role_hashes_by_framework"],
            expected_population_unit=(
                "patient_group" if cfg.evaluation.group_mode == "patient_group" else "row"
            ),
            expected_group_mode=cfg.evaluation.group_mode,
            allow_legacy=dataset.legacy_two_role,
        )
        combined = load_combined_table(str(combined_path))
        log_disparity_reports = load_log_disparity_reports(cfg.evaluation.output_dir)
        if experiment is None:
            raise RuntimeError("Evaluation plotting requires a loaded experiment")
        generation_inventory = load_generation_inventory(
            experiment.manifest_path, experiment.generation_dir
        )
        save_rank_tradeoff_plots(
            cfg,
            combined,
            cfg.plots.output_dir,
            produced_outputs=generation_inventory.produced_outputs,
            missing_stage_a_outputs=generation_inventory.failed_outputs,
        )
        save_log_disparity_plots(log_disparity_reports, cfg.plots.output_dir)
        verify_native_syntheval_artifacts(cfg.evaluation.output_dir)

        if cfg.evaluation.generate_report:
            from synthdata.evaluation.report import save_evaluation_report

            save_evaluation_report(
                cfg,
                dataset,
                combined,
                {
                    "selected_datasets": synthetic_datasets,
                    "log_disparity_reports": log_disparity_reports,
                    "artifact_manifest": str(
                        artifact_bundle_dir(cfg.evaluation.output_dir) / "manifest.json"
                    ),
                    "evaluation_coverage": evaluation_manifest.get("evaluation_attempt", {}).get(
                        "evaluation_coverage"
                    ),
                },
                experiment,
            )

    if experiment is not None:
        experiment.record(
            "plots", artifacts={"plots_dir": str(experiment.plots_dir)}, sections=sorted(sections)
        )

    logger.info("Done. Figures saved under %s", cfg.plots.output_dir)


if __name__ == "__main__":
    main()
