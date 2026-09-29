#!/usr/bin/env python
"""CLI: evaluate synthetic data with synthcity + SynthEval + custom (log
disparity / fork-only fairness) metrics, combined into a single ranked table.

Evaluates the most recent experiment started by `synthdata-generate` unless
``--experiment-id`` is given to target a specific past one (see
:mod:`synthdata.experiment`).

Usage:
    synthdata-evaluate --config configs/config.yaml [--plot] [--experiment-id ID] [--dataset-version v2]

Requires generated synthetic data (run `synthdata-generate` first).
"""

import argparse
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

from synthdata.config import load_config
from synthdata.data import (
    load_dataset,
    load_imputed_splits,
    validate_imputation_cache_lineage,
)
from synthdata.evaluation import artifacts, run_evaluation
from synthdata.evaluation.combine import simple_rank_summary
from synthdata.evaluation.syntheval_eval import InsufficientSynthEvalResourcesError
from synthdata.experiment import load_experiment
from synthdata.imputation.pipeline import _cache_key_record, run_imputation
from synthdata.utils import get_logger, set_global_seed

logger = get_logger("run_evaluation")


@contextmanager
def _evaluation_debug_logging() -> Iterator[None]:
    """Enable evaluation DEBUG output for one CLI invocation, then restore levels."""
    evaluation_loggers = []
    for name, candidate in logging.Logger.manager.loggerDict.items():
        if not isinstance(candidate, logging.Logger):
            continue
        if name != "synthdata.evaluation" and not name.startswith("synthdata.evaluation."):
            continue
        evaluation_loggers.append(candidate)

    target_loggers = [*evaluation_loggers, logger]
    logger_levels = [(target, target.level) for target in target_loggers]
    handlers = {handler for target in target_loggers for handler in target.handlers}
    handler_levels = [(handler, handler.level) for handler in handlers]
    try:
        for target in target_loggers:
            target.setLevel(logging.DEBUG)
        for handler in handlers:
            handler.setLevel(logging.DEBUG)
        yield
    finally:
        for target, level in logger_levels:
            target.setLevel(level)
        for handler, level in handler_levels:
            handler.setLevel(level)


def _role_shape(dataset, role: str) -> tuple[int, int] | None:
    frame = dataset.role_frame(role, imputed=True)
    return None if frame is None else frame.shape


def _validation_counts(value: object) -> tuple[int, int, int]:
    """Return total, complete, and decision-eligible validation counts."""
    if not isinstance(value, dict):
        return 0, 0, 0
    if isinstance(value.get("complete"), bool):
        return (
            1,
            int(value["complete"]),
            int(value.get("decision_eligible") is True),
        )
    counts = [0, 0, 0]
    for child in value.values():
        for index, count in enumerate(_validation_counts(child)):
            counts[index] += count
    return counts[0], counts[1], counts[2]


def _execution_counts(value: object) -> tuple[int, int, int]:
    """Summarize safe SynthEval execution states without logging payload details."""
    if not isinstance(value, dict):
        return 0, 0, 0
    if "execution_succeeded" in value or "model_status" in value:
        return (
            1,
            int(value.get("execution_succeeded") is True),
            int(value.get("execution_succeeded") is False),
        )
    counts = [0, 0, 0]
    for child in value.values():
        for index, count in enumerate(_execution_counts(child)):
            counts[index] += count
    return counts[0], counts[1], counts[2]


def _incomplete_metric_coverage(extras: dict) -> list[dict]:
    """Collect persisted metric validations that did not cover all expected metrics."""
    incomplete = []

    def collect_validations(section: str, value: object, path: tuple[str, ...] = ()) -> None:
        if not isinstance(value, dict):
            return
        if isinstance(value.get("complete"), bool):
            if value["complete"] is False:
                incomplete.append(
                    {
                        "section": section,
                        "group": ":".join(path[:-1]),
                        "model": path[-1] if path else None,
                        "failed_keys": value.get("failed_keys", []),
                        "indeterminate_keys": value.get("indeterminate_keys", []),
                    }
                )
            return
        for key, child in value.items():
            collect_validations(section, child, (*path, str(key)))

    for section in (
        "synthcity_validation",
        "syntheval_validation",
        "custom_validation",
        "release_evidence_validation",
    ):
        collect_validations(section, extras.get(section, {}))

    final_holdout = extras.get("final_holdout_evidence")
    if isinstance(final_holdout, dict) and (
        final_holdout.get("metric_completeness_state") == "incomplete"
        or final_holdout.get("score_completeness_state") == "indeterminate"
        or final_holdout.get("state") == "failed"
    ):
        incomplete.append(
            {
                "section": "final_holdout_evidence",
                "group": "final_holdout",
                "model": final_holdout.get("selected_model"),
                "failed_keys": final_holdout.get("failed_metric_keys", []),
                "indeterminate_keys": final_holdout.get("indeterminate_metric_keys", []),
                "score_completeness_state": final_holdout.get("score_completeness_state"),
            }
        )
    return incomplete


def _load_synthetic_datasets(cfg, dataset=None, *, generation_inventory=None) -> dict:
    """Load generated inputs after strict canonical cache preflight."""
    if dataset is None:
        raise ValueError("Canonical evaluation requires candidate dataset context for preflight")
    configured_models = list(cfg.evaluation.models) if cfg.evaluation.models else None
    return artifacts.load_validated_generated_datasets(
        cfg.generation.output_dir,
        dataset,
        model_names=configured_models,
        classification_score=cfg.evaluation.synthcity.classification_score,
        generation_inventory=generation_inventory,
    )


def main() -> None:
    with _evaluation_debug_logging():
        _run_evaluation_cli()


def _run_evaluation_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic data with synthcity + SynthEval + custom fairness metrics."
    )
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save rank trade-off + log-disparity plots after evaluation.",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Evaluate a specific past experiment instead of the most recent one "
        "(overrides experiment.id).",
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
    logger.debug(
        "event=evaluation.cli.config_loaded experiment_id=%s dataset_version=%s",
        cfg.experiment.id or "latest",
        cfg.data.version,
    )

    logger.debug("event=evaluation.cache.validation_start phase=candidate roles=train,tuning")
    candidate_dataset = load_dataset(cfg)
    candidate_dataset = load_imputed_splits(
        candidate_dataset,
        expected_cache_key=_cache_key_record(cfg, candidate_dataset)["cache_key"],
    )
    validate_imputation_cache_lineage(
        candidate_dataset,
        _cache_key_record(cfg, candidate_dataset),
        required=True,
    )
    logger.debug(
        "event=evaluation.cache.validation_complete phase=candidate dataset=%s "
        "dataset_version=%s train_shape=%s tuning_shape=%s lineage=valid",
        candidate_dataset.name,
        candidate_dataset.version,
        _role_shape(candidate_dataset, "train"),
        _role_shape(candidate_dataset, "tuning"),
    )
    if any(
        candidate_dataset.role_frame(role, imputed=True) is None for role in ("train", "tuning")
    ):
        raise SystemExit(
            "No validated candidate imputation found. Run `synthdata-impute --config <path>` first."
        )

    logger.debug("event=evaluation.final_holdout.setup_start phase=final roles=train,final_holdout")
    final_dataset = load_dataset(cfg)
    final_dataset = run_imputation(cfg, final_dataset, phase="final")
    final_dataset = load_imputed_splits(
        final_dataset,
        expected_cache_key=_cache_key_record(cfg, final_dataset, phase="final")["cache_key"],
        phase="final",
    )
    final_holdout = final_dataset.role_frame("final_holdout", imputed=True)
    if final_holdout is None:
        raise SystemExit(
            "No validated final-phase imputation found; refusing to evaluate raw final_holdout."
        )
    logger.debug(
        "event=evaluation.cache.validation_complete phase=final dataset=%s "
        "dataset_version=%s final_holdout_shape=%s imputation=validated",
        final_dataset.name,
        final_dataset.version,
        final_holdout.shape,
    )
    train = candidate_dataset.role_frame("train", imputed=True)
    tuning = candidate_dataset.role_frame("tuning", imputed=True)
    if train is None or tuning is None:
        raise SystemExit("Validated candidate imputation is incomplete; refusing evaluation.")
    candidate_dataset.set_imputed_roles(
        {
            "train": train,
            "tuning": tuning,
            "final_holdout": final_holdout,
        }
    )
    dataset = candidate_dataset

    experiment = load_experiment(cfg, dataset=dataset, allow_final_holdout_handoff=True)
    cfg.generation.output_dir = str(experiment.generation_dir)
    cfg.evaluation.output_dir = str(experiment.evaluation_dir)
    cfg.plots.output_dir = str(experiment.plots_dir)

    logger.debug(
        "event=evaluation.generation_cache.preflight_start experiment_id=%s dataset=%s "
        "dataset_version=%s",
        getattr(experiment, "id", "unknown"),
        dataset.name,
        dataset.version,
    )
    generation_inventory = artifacts.load_generation_inventory(
        experiment.manifest_path,
        experiment.generation_dir,
    )
    synthetic_datasets = _load_synthetic_datasets(
        cfg,
        dataset,
        generation_inventory=generation_inventory,
    )
    if not synthetic_datasets:
        raise SystemExit(
            f"No synthetic datasets found in {cfg.generation.output_dir}. "
            "Run `synthdata-generate --config <path>` first."
        )

    logger.debug(
        "event=evaluation.generation_cache.preflight_complete model_count=%d models=%s",
        len(synthetic_datasets),
        sorted(synthetic_datasets),
    )
    logger.debug(
        "event=evaluation.stage.schedule framework=synthcity enabled=%s role=tuning model_count=%d",
        bool(cfg.evaluation.synthcity.metrics),
        len(synthetic_datasets),
    )
    logger.debug(
        "event=evaluation.stage.schedule framework=syntheval enabled=%s role=tuning "
        "model_count=%d checkpoint_policy=validated_successful_only",
        cfg.evaluation.syntheval.enabled,
        len(synthetic_datasets),
    )
    evaluation_started = perf_counter()
    logger.debug(
        "event=evaluation.pipeline.start experiment_id=%s dataset=%s dataset_version=%s "
        "models=%s roles=train,tuning,final_holdout",
        getattr(experiment, "id", "unknown"),
        dataset.name,
        dataset.version,
        sorted(synthetic_datasets),
    )
    try:
        combined, extras = run_evaluation(cfg, dataset, synthetic_datasets, experiment=experiment)
    except InsufficientSynthEvalResourcesError as exc:
        logger.error(
            "event=evaluation.pipeline.failed experiment_id=%s dataset=%s "
            "dataset_version=%s error_type=%s error=%s",
            getattr(experiment, "id", "unknown"),
            dataset.name,
            dataset.version,
            type(exc).__name__,
            exc,
        )
        experiment.record(
            "evaluation",
            artifacts={"evaluation_dir": str(cfg.evaluation.output_dir)},
            status="incomplete",
            failure_stage="syntheval_resource_resolution",
            failure_type=type(exc).__name__,
            failure_reason=str(exc),
        )
        raise SystemExit(1) from exc
    logger.debug(
        "event=evaluation.pipeline.complete elapsed_seconds=%.3f model_count=%d",
        perf_counter() - evaluation_started,
        len(combined),
    )
    for framework in ("synthcity", "syntheval", "custom", "release_evidence"):
        total, complete, eligible = _validation_counts(extras.get(f"{framework}_validation", {}))
        if total:
            logger.debug(
                "event=evaluation.metric_execution.summary framework=%s model_count=%d "
                "complete_count=%d decision_eligible_count=%d",
                framework,
                total,
                complete,
                eligible,
            )
    execution_total, execution_succeeded, execution_failed = _execution_counts(
        extras.get("syntheval_execution", {})
    )
    if execution_total:
        logger.debug(
            "event=evaluation.checkpoint.execution_summary framework=syntheval "
            "execution_count=%d succeeded_count=%d failed_count=%d",
            execution_total,
            execution_succeeded,
            execution_failed,
        )
    incomplete_metrics = _incomplete_metric_coverage(extras)
    partial_generation_coverage = (
        isinstance(extras.get("evaluation_coverage"), dict)
        and extras["evaluation_coverage"].get("status") == "partial"
    )
    incomplete = partial_generation_coverage or bool(incomplete_metrics)
    artifact_status = "incomplete" if incomplete else "complete"
    logger.debug(
        "event=evaluation.artifacts.completeness manifest_returned=%s status=%s "
        "incomplete_validation_count=%d generation_coverage=%s",
        bool(extras.get("artifact_manifest")),
        artifact_status,
        len(incomplete_metrics),
        "partial" if partial_generation_coverage else "complete",
    )
    rank_column = ("__all__", "overall", "rank")
    eligible_rank_count = (
        int(combined[rank_column].notna().sum()) if rank_column in combined.columns else 0
    )
    logger.debug(
        "event=evaluation.ranking.eligibility model_count=%d ranked_model_count=%d "
        "incomplete=%s reason_code=%s",
        len(combined),
        eligible_rank_count,
        incomplete,
        "incomplete_evaluation_coverage" if incomplete else "eligible_coverage",
    )
    logger.info(
        "Combined evaluation summary (ranked, higher=better):\n%s",
        simple_rank_summary(combined).to_string(),
    )

    final_holdout_evidence = extras.get("final_holdout_evidence")
    if final_holdout_evidence is not None:
        logger.debug(
            "event=evaluation.final_holdout.result state=%s selected_model=%s "
            "metric_completeness=%s score_completeness=%s",
            final_holdout_evidence.get("state"),
            final_holdout_evidence.get("selected_model"),
            final_holdout_evidence.get("metric_completeness_state"),
            final_holdout_evidence.get("score_completeness_state"),
        )
        evidence_path = Path(extras["artifact_manifest"]).parent / "final_holdout_evidence.json"
        final_refit = final_holdout_evidence.get("final_refit", {})
        experiment.record(
            "final_holdout_evidence",
            artifacts={
                "evidence": str(evidence_path),
                **(
                    {
                        "final_refit_data": final_refit["path"],
                        "final_refit_metadata": final_refit["metadata_path"],
                    }
                    if final_refit.get("path") and final_refit.get("metadata_path")
                    else {}
                ),
            },
            state=final_holdout_evidence.get("state"),
            selected_model=final_holdout_evidence.get("selected_model"),
            fit_roles=final_holdout_evidence.get("fit_roles"),
        )

    experiment.record(
        "evaluation",
        artifacts={
            "evaluation_dir": str(experiment.evaluation_dir),
            "combined_table": str(Path(cfg.evaluation.output_dir) / "combined_evaluation.csv"),
            "artifact_manifest": extras["artifact_manifest"],
            **({"report": extras["report_path"]} if "report_path" in extras else {}),
        },
        n_models=len(synthetic_datasets),
        status="partial" if incomplete else "complete",
        metric_coverage={
            "status": "incomplete" if incomplete_metrics else "complete",
            "incomplete_validations": incomplete_metrics,
        },
        evaluation_coverage=extras.get("evaluation_coverage"),
    )
    logger.debug(
        "event=evaluation.artifacts.persisted manifest_returned=%s evaluation_status=%s "
        "model_count=%d",
        bool(extras.get("artifact_manifest")),
        "partial" if incomplete else "complete",
        len(synthetic_datasets),
    )

    if args.plot:
        from synthdata.plotting.evaluation_plots import (
            save_log_disparity_plots,
            save_rank_tradeoff_plots,
        )

        save_rank_tradeoff_plots(cfg, combined, cfg.plots.output_dir)
        save_log_disparity_plots(extras["log_disparity_reports"], cfg.plots.output_dir)
        # Native per-model SynthEval plots (SE_*.png) are produced as a side effect of
        # run_evaluation()'s syntheval benchmark pass above (via enable_syntheval_plots),
        # so no separate/redundant recomputation pass is needed here.

        if cfg.evaluation.generate_report:
            # The report was already written once inside run_evaluation(), but its
            # "Plots" section links to files that only exist *after* the plotting
            # calls above -- re-save it now (cheap: just re-renders markdown from
            # already-computed results, no metric recomputation) so those links
            # are accurate.
            from synthdata.evaluation.report import save_evaluation_report

            report_path = save_evaluation_report(cfg, dataset, combined, extras, experiment)
            extras["report_path"] = str(report_path)

    if "report_path" in extras:
        logger.info("Evaluation report written to %s", extras["report_path"])

    logger.info("Done. Combined table saved under %s", cfg.evaluation.output_dir)
    if incomplete:
        logger.error(
            "Evaluation artifacts were persisted with incomplete coverage: "
            "partial_generation_coverage=%s incomplete_metric_validations=%s",
            partial_generation_coverage,
            incomplete_metrics,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
