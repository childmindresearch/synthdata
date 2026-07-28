"""Orchestrates the evaluation stage: synthcity + SynthEval + custom (log
disparity / fork-only fairness) evaluators, combined into one ranked table.
"""

from pathlib import Path

import pandas as pd

from synthdata.config import Config
from synthdata.data import Dataset
from synthdata.evaluation import (
    combine,
    custom_eval,
    privacy_gate,
    report,
    synthcity_eval,
    syntheval_eval,
)
from synthdata.utils import ensure_dir, get_logger

logger = get_logger(__name__)


def select_models(
    cfg: Config, synthetic_datasets: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """Restrict to cfg.evaluation.models if set, else evaluate everything generated."""
    if not cfg.evaluation.models:
        return synthetic_datasets
    missing = [m for m in cfg.evaluation.models if m not in synthetic_datasets]
    if missing:
        logger.warning(
            "Requested evaluation models not found among generated datasets: %s", missing
        )
    return {k: v for k, v in synthetic_datasets.items() if k in cfg.evaluation.models}


def run_evaluation(
    cfg: Config,
    dataset: Dataset,
    synthetic_datasets: dict[str, pd.DataFrame],
    enable_syntheval_plots: bool = False,
    experiment=None,
) -> tuple[pd.DataFrame, dict]:
    """Run the full evaluation stage.

    Returns ``(combined_table, extras)`` where ``combined_table`` is the single
    ranked, multi-index DataFrame (see :mod:`synthdata.evaluation.combine`) and
    ``extras`` holds the raw per-framework results (useful for plotting).

    If ``enable_syntheval_plots`` is True (and ``cfg.evaluation.save_per_model_syntheval_plots``
    is enabled), SynthEval's native per-metric plots are produced as a side effect of the
    single syntheval benchmark pass below -- callers must NOT also call the old
    ``save_per_model_syntheval_plots`` helper afterwards, as that would redundantly
    recompute every metric (including the expensive ``corr_diff``/``mi_diff`` ones) a
    second time just to get plots that this call already produced.

    ``experiment`` (optional :class:`synthdata.experiment.Experiment`) is only used to
    include the experiment id in the generated Markdown report (see
    ``cfg.evaluation.generate_report``); it has no effect on any other stage output.
    """
    eval_cfg = cfg.evaluation
    output_dir = ensure_dir(eval_cfg.output_dir)

    selected_datasets = select_models(cfg, synthetic_datasets)
    model_names = sorted(selected_datasets)
    logger.info("Evaluating %d models: %s", len(model_names), model_names)

    synthcity_results = synthcity_eval.run_synthcity_evaluation(
        selected_datasets,
        dataset.train_imputed_df,
        dataset.test_imputed_df,
        dataset.target_column,
        dataset.sensitive_columns,
        eval_cfg.synthcity,
        n_samples=cfg.generation.n_samples,
        seed=cfg.seed,
        workspace=output_dir / "synthcity_workspace",
    )

    want_syntheval_plots = enable_syntheval_plots and eval_cfg.save_per_model_syntheval_plots
    plots_output_dir = (
        Path(cfg.plots.output_dir) / "evaluation" / "syntheval_plots"
        if want_syntheval_plots
        else None
    )
    benchmark_results, benchmark_ranks = syntheval_eval.run_syntheval_evaluation(
        selected_datasets,
        dataset,
        eval_cfg.syntheval,
        preset_dir=output_dir,
        ranking_strategy=eval_cfg.ranking_strategy,
        output_folder=output_dir / "syntheval_benchmark",
        plots_output_dir=plots_output_dir,
        positive_class=eval_cfg.positive_class,
    )

    if eval_cfg.binary_target.enabled:
        binary_results, binary_ranks = syntheval_eval.run_binary_target_syntheval_evaluation(
            selected_datasets,
            dataset,
            eval_cfg.syntheval,
            eval_cfg.binary_target,
            preset_dir=output_dir,
            ranking_strategy=eval_cfg.ranking_strategy,
            output_folder=output_dir / "syntheval_benchmark",
        )
        benchmark_results, benchmark_ranks = syntheval_eval.merge_binary_target_results(
            benchmark_results, benchmark_ranks, binary_results, binary_ranks
        )

    log_disparity_reports = custom_eval.run_log_disparity_evaluation(
        selected_datasets, dataset, eval_cfg.log_disparity, eval_cfg.custom
    )

    combined = combine.build_combined_table(
        synthcity_results,
        benchmark_results,
        benchmark_ranks,
        log_disparity_reports,
        model_names,
        rank_weights=eval_cfg.rank_weights,
    )

    gate_result = privacy_gate.evaluate_privacy_gate(combined, eval_cfg.privacy_gate)
    combined = privacy_gate.merge_privacy_gate_results(combined, gate_result)

    combined.to_csv(output_dir / "combined_evaluation.csv")

    extras = {
        "selected_datasets": selected_datasets,
        "synthcity_results": synthcity_results,
        "syntheval_benchmark_results": benchmark_results,
        "syntheval_benchmark_ranks": benchmark_ranks,
        "log_disparity_reports": log_disparity_reports,
        "privacy_gate_result": gate_result,
    }

    if eval_cfg.generate_report:
        report_path = report.save_evaluation_report(cfg, dataset, combined, extras, experiment)
        extras["report_path"] = str(report_path)

    return combined, extras
