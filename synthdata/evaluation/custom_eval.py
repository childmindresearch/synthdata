"""Custom fairness evaluation: log disparity (Bhanot et al. 2021) summary metrics.

The equalized_odds/equal_opportunity metrics (custom additions to this repo's
SynthEval fork) are *computed* via :mod:`synthdata.evaluation.syntheval_eval`
but re-tagged to framework="custom" downstream in
:mod:`synthdata.evaluation.combine`; this module only covers log disparity,
which has no SynthEval equivalent.
"""

from collections.abc import Mapping

import pandas as pd

from synthdata.data import Dataset
from synthdata.evaluation.catalog import LOG_DISPARITY_METRICS, resolve_selection
from synthdata.evaluation.metric_contracts import (
    DEFAULT_METRIC_CONTRACT_REGISTRY,
    MetricEvaluationContext,
    MetricObservation,
    MetricValidationResult,
    resolve_metric_observations,
)
from synthdata.utils import get_logger

logger = get_logger(__name__)

_LOG_DISPARITY_NAME = "log_disparity"


def _log_disparity_result_metadata(
    report: Mapping[str, object] | None,
    *,
    role_hashes: Mapping[str, str],
    evaluation_role: str,
    state: str,
) -> dict:
    supplied = report.get("result_metadata") if report is not None else None
    if supplied is None and report is not None:
        supplied = report.get("metadata")
    if supplied is not None and not isinstance(supplied, Mapping):
        raise ValueError("Custom log-disparity result_metadata must be an object")
    metadata = {
        "schema_version": "log-disparity-result-v1",
        "metric": _LOG_DISPARITY_NAME,
        "evaluation_role": evaluation_role,
        "protected_columns": list(report.get("protected_group_cols", []))
        if report is not None
        else [],
        "target_order": list(report.get("target_order", [])) if report is not None else [],
        "population_role_hashes": dict(role_hashes),
        "report_state": state,
    }
    if supplied is not None:
        metadata.update(dict(supplied))
    return metadata


def run_log_disparity_evaluation(
    synthetic_datasets: dict[str, pd.DataFrame],
    dataset: Dataset,
    log_disparity_cfg,
    selection_cfg,
    *,
    evaluation_role: str = "tuning",
) -> dict[str, dict]:
    """Compute a log-disparity fairness report for every synthetic dataset.

    Returns ``{model_name: report}`` (the full dict from
    ``compute_log_disparity_report``, including the Plotly ``report_figure``),
    or ``{}`` if log_disparity is not in the configured selection.
    """
    all_names = [_LOG_DISPARITY_NAME]
    selected = resolve_selection(
        selection_cfg.enabled,
        selection_cfg.categories,
        selection_cfg.metrics,
        all_names,
        {_LOG_DISPARITY_NAME: "fairness"},
    )
    if _LOG_DISPARITY_NAME not in selected:
        return {}

    from synthdata.log_disparity.metric_log_disparity import compute_log_disparity_report

    protected_cols = log_disparity_cfg.protected_columns or list(dataset.sensitive_columns)
    if not protected_cols:
        logger.warning("[custom] log_disparity requires protected columns; skipping")
        return {}
    real_data = dataset.role_frame(evaluation_role, imputed=False)
    if real_data is None and dataset.legacy_two_role and evaluation_role == "tuning":
        real_data = dataset.role_frame("final_holdout", imputed=False)
    if real_data is None:
        raise RuntimeError(f"[custom] log_disparity requires a populated {evaluation_role!r} role")

    reports = {}
    for name, syn_df in synthetic_datasets.items():
        try:
            reports[name] = compute_log_disparity_report(
                real_data=real_data,
                synth_data=syn_df,
                target_col=dataset.target_column,
                protected_cols=protected_cols,
                model_name=name,
                target_map=log_disparity_cfg.target_map,
                protected_map=log_disparity_cfg.protected_map,
                protected_bins=log_disparity_cfg.protected_bins,
            )
        except (KeyError, ValueError) as exc:
            logger.warning("[custom] log_disparity failed for %s: %s", name, exc)
            reports[name] = {"error": str(exc), "error_type": type(exc).__name__}
    return reports


def build_log_disparity_summary_table(reports: dict[str, dict]) -> pd.DataFrame:
    """Models x {log_disparity_mean_abs, log_disparity_median_abs, log_disparity_share_significant}.

    Models whose report failed (see ``run_log_disparity_evaluation``'s
    ``{"error": ...}`` entries) get all-NaN rows here rather than being
    silently dropped, so a failure is still visible in the summary table.
    """
    rows = {}
    for name, report in reports.items():
        if "error" in report:
            rows[name] = {
                "log_disparity_mean_abs": None,
                "log_disparity_median_abs": None,
                "log_disparity_share_significant": None,
            }
            continue
        stats = report["summary_stats"]
        rows[name] = {
            "log_disparity_mean_abs": stats.get("mean_abs_log_disparity"),
            "log_disparity_median_abs": stats.get("median_abs_log_disparity"),
            "log_disparity_share_significant": stats.get("share_significant_bh"),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def validate_log_disparity_results(
    reports: dict[str, dict],
    model_names: list[str],
    *,
    role_hashes: dict[str, str],
    requested_use: str = "policy_rank",
    population_unit: str = "row",
    group_mode: str = "row",
    resolved_configuration: dict | None = None,
    evaluation_role: str = "tuning",
) -> dict[str, MetricValidationResult]:
    """Validate log-disparity summaries without dropping failed reports."""
    expected_keys = [
        "log_disparity_mean_abs",
        "log_disparity_median_abs",
        "log_disparity_share_significant",
    ]
    validations = {}
    context = MetricEvaluationContext(
        role_hashes=role_hashes,
        evaluation_role=evaluation_role,
        population_unit=population_unit,
        group_mode=group_mode,
        resolved_configuration=resolved_configuration or {},
    )
    for model_name in model_names:
        report = reports.get(model_name)
        observations = []
        if report is not None and "error" not in report:
            result_metadata = _log_disparity_result_metadata(
                report,
                role_hashes=role_hashes,
                evaluation_role=evaluation_role,
                state="succeeded",
            )
            stats = report.get("summary_stats", {})
            values = {
                "log_disparity_mean_abs": stats.get("mean_abs_log_disparity"),
                "log_disparity_median_abs": stats.get("median_abs_log_disparity"),
                "log_disparity_share_significant": stats.get("share_significant_bh"),
            }
            observations = [
                MetricObservation(
                    model_name=model_name,
                    framework="custom",
                    emitted_key=key,
                    raw_value=value,
                    role_hashes=role_hashes,
                    source_metadata={
                        "report_state": "succeeded",
                        "result_metadata": result_metadata,
                    },
                    result_metadata=result_metadata,
                )
                for key, value in values.items()
            ]
        elif report is not None:
            result_metadata = _log_disparity_result_metadata(
                report,
                role_hashes=role_hashes,
                evaluation_role=evaluation_role,
                state="failed",
            )
            observations = [
                MetricObservation(
                    model_name=model_name,
                    framework="custom",
                    emitted_key=key,
                    raw_value=None,
                    error=f"{report.get('error_type', 'UnknownError')}: {report.get('error', 'log-disparity failed')}",
                    role_hashes=role_hashes,
                    source_metadata={
                        "report_state": "failed",
                        "result_metadata": result_metadata,
                    },
                    result_metadata=result_metadata,
                )
                for key in expected_keys
            ]
        validations[model_name] = resolve_metric_observations(
            registry=DEFAULT_METRIC_CONTRACT_REGISTRY,
            model_name=model_name,
            framework="custom",
            expected_keys=expected_keys,
            observations=observations,
            context=context,
            requested_use=requested_use,
        )
    return validations


#: True => lower is "better" (orient as -value for ranking); mirrors LOG_DISPARITY_METRICS.
LOG_DISPARITY_MINIMIZE = dict(LOG_DISPARITY_METRICS)
