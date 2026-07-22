"""Custom fairness evaluation: log disparity (Bhanot et al. 2021) summary metrics.

The equalized_odds/equal_opportunity metrics (custom additions to this repo's
SynthEval fork) are *computed* via :mod:`synthdata.evaluation.syntheval_eval`
but re-tagged to framework="custom" downstream in
:mod:`synthdata.evaluation.combine`; this module only covers log disparity,
which has no SynthEval equivalent.
"""

import pandas as pd

from synthdata.data import Dataset
from synthdata.evaluation.catalog import LOG_DISPARITY_METRICS, resolve_selection
from synthdata.utils import get_logger

logger = get_logger(__name__)

_LOG_DISPARITY_NAME = "log_disparity"


def run_log_disparity_evaluation(
    synthetic_datasets: dict[str, pd.DataFrame],
    dataset: Dataset,
    log_disparity_cfg,
    selection_cfg,
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

    reports = {}
    for name, syn_df in synthetic_datasets.items():
        try:
            reports[name] = compute_log_disparity_report(
                real_data=dataset.train_df,
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


#: True => lower is "better" (orient as -value for ranking); mirrors LOG_DISPARITY_METRICS.
LOG_DISPARITY_MINIMIZE = dict(LOG_DISPARITY_METRICS)
