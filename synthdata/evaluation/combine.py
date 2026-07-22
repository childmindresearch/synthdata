"""Combines synthcity + SynthEval + custom (log disparity / fork-only fairness)
results into a single ranked table with 3-level MultiIndex columns:
``(framework, type, metric)`` where ``framework in {synthcity, syntheval, custom}``
and ``type in {utility, privacy, fairness}``.

Ranking (see module docstring of :func:`build_combined_table` for details) is
appended as extra columns in the same table, both per ``(framework, type)`` group
and rolled up across frameworks per ``type``, plus one overall rank.
"""

import pandas as pd

from synthdata.evaluation.catalog import (
    LOG_DISPARITY_METRICS,
    SYNTHCITY_CATEGORY_TO_TYPE,
    SYNTHEVAL_CUSTOM_FAIRNESS_KEYS,
    SYNTHEVAL_METRIC_TYPE,
)
from synthdata.evaluation.custom_eval import build_log_disparity_summary_table
from synthdata.evaluation.syntheval_eval import extract_oriented_values, extract_raw_values
from synthdata.utils import get_logger

logger = get_logger(__name__)

_ALL = "__all__"
_RANK = "rank"


def _synthcity_frames(
    synthcity_results: dict[str, pd.DataFrame], model_names: list
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Build (raw, oriented) models x metric-key tables from synthcity results.

    Entries with an "error" column (a model whose synthcity evaluation failed,
    see ``synthcity_eval.run_synthcity_evaluation``) are excluded from the
    metric table -- they carry no "mean"/"direction" data to combine -- but
    are logged so the exclusion is visible, not silent.
    """
    if not synthcity_results:
        empty = pd.DataFrame(index=model_names)
        return empty, empty

    failed = {name for name, res in synthcity_results.items() if "error" in res.columns}
    if failed:
        logger.warning(
            "[synthcity] excluding failed models from combined table: %s", sorted(failed)
        )
    ok_results = {name: res for name, res in synthcity_results.items() if name not in failed}
    if not ok_results:
        empty = pd.DataFrame(index=model_names)
        return empty, empty

    raw = pd.DataFrame({name: res["mean"] for name, res in ok_results.items()}).T

    directions = {}
    for res in ok_results.values():
        for metric_key, direction in res["direction"].items():
            directions.setdefault(metric_key, direction)
    sign = pd.Series(directions).map({"maximize": 1.0, "minimize": -1.0})

    common = raw.columns.intersection(sign.index)
    oriented = raw[common].multiply(sign[common], axis=1)

    raw = raw.reindex(model_names)
    oriented = oriented.reindex(model_names)

    columns = [
        ("synthcity", SYNTHCITY_CATEGORY_TO_TYPE.get(col.split(".")[0], "utility"), col)
        for col in raw.columns
    ]
    raw.columns = pd.MultiIndex.from_tuples(columns)
    oriented.columns = pd.MultiIndex.from_tuples(columns)
    return raw, oriented


def _syntheval_frames(
    benchmark_results: pd.DataFrame | None,
    benchmark_ranks: pd.DataFrame | None,
    model_names: list,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Build (raw, oriented) models x metric-name tables from SynthEval results.

    Metrics in SYNTHEVAL_CUSTOM_FAIRNESS_KEYS (fork-only additions) are tagged
    framework="custom" instead of "syntheval".
    """
    if benchmark_results is None:
        empty = pd.DataFrame(index=model_names)
        return empty, empty

    raw = extract_raw_values(benchmark_results).reindex(model_names)
    oriented = extract_oriented_values(benchmark_ranks).reindex(model_names)

    def _framework(metric: str) -> str:
        return "custom" if metric in SYNTHEVAL_CUSTOM_FAIRNESS_KEYS else "syntheval"

    columns = [
        (_framework(col), SYNTHEVAL_METRIC_TYPE.get(col, "utility"), col) for col in raw.columns
    ]
    raw.columns = pd.MultiIndex.from_tuples(columns)
    oriented.columns = pd.MultiIndex.from_tuples(columns)
    return raw, oriented


def _log_disparity_frames(
    reports: dict[str, dict], model_names: list
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    if not reports:
        empty = pd.DataFrame(index=model_names)
        return empty, empty

    raw = build_log_disparity_summary_table(reports).reindex(model_names)
    sign = pd.Series(
        {m: (-1.0 if minimize else 1.0) for m, minimize in LOG_DISPARITY_METRICS.items()}
    )
    common = raw.columns.intersection(sign.index)
    oriented = raw[common].multiply(sign[common], axis=1)

    columns = [("custom", "fairness", col) for col in raw.columns]
    raw.columns = pd.MultiIndex.from_tuples(columns)
    oriented.columns = pd.MultiIndex.from_tuples(
        [("custom", "fairness", col) for col in oriented.columns]
    )
    return raw, oriented


def _minmax_scale(col: pd.Series) -> pd.Series:
    """Per-column min-max scaling; NaN-safe (ties -> 0.5, NaNs preserved)."""
    valid = col.dropna()
    if valid.empty:
        return col
    lo, hi = valid.min(), valid.max()
    if hi == lo:
        return col.where(col.isna(), 0.5)
    return (col - lo) / (hi - lo)


def build_combined_table(
    synthcity_results: dict[str, pd.DataFrame],
    syntheval_benchmark_results: pd.DataFrame | None,
    syntheval_benchmark_ranks: pd.DataFrame | None,
    log_disparity_reports: dict[str, dict],
    model_names: list,
) -> pd.DataFrame:
    """Build the single combined, ranked, multi-index evaluation table.

    Ranking scheme (see repo docs for rationale):
      1. Every metric is oriented so "higher = better", then min-max scaled
         across models (independently per metric).
      2. A sub-rank is computed per ``(framework, type)`` group by summing its
         scaled metrics -- column ``(framework, type, "rank")``.
      3. A rolled-up rank per ``type`` (utility/privacy/fairness) sums the
         scaled metrics of that type across *all* frameworks -- column
         ``("__all__", type, "rank")``.
      4. One overall rank sums every scaled metric -- column
         ``("__all__", "overall", "rank")``.
    Models are sorted descending by the overall rank.
    """
    sc_raw, sc_oriented = _synthcity_frames(synthcity_results, model_names)
    se_raw, se_oriented = _syntheval_frames(
        syntheval_benchmark_results, syntheval_benchmark_ranks, model_names
    )
    ld_raw, ld_oriented = _log_disparity_frames(log_disparity_reports, model_names)

    raw_parts = [df for df in (sc_raw, se_raw, ld_raw) if not df.empty]
    oriented_parts = [df for df in (sc_oriented, se_oriented, ld_oriented) if not df.empty]

    if not raw_parts:
        raise ValueError("No evaluation results to combine: check evaluation config selection")

    raw_df = pd.concat(raw_parts, axis=1)
    oriented_df = pd.concat(oriented_parts, axis=1)

    scaled_df = oriented_df.apply(_minmax_scale, axis=0)

    combined = raw_df.copy()

    # Per (framework, type) sub-rank.
    groups = sorted(
        set(
            zip(
                scaled_df.columns.get_level_values(0),
                scaled_df.columns.get_level_values(1),
                strict=True,
            )
        )
    )
    for framework, type_ in groups:
        cols = [c for c in scaled_df.columns if c[0] == framework and c[1] == type_]
        combined[(framework, type_, _RANK)] = scaled_df[cols].sum(axis=1, skipna=True)

    # Rolled-up rank per type, across frameworks.
    for type_ in ("utility", "privacy", "fairness"):
        cols = [c for c in scaled_df.columns if c[1] == type_]
        if cols:
            combined[(_ALL, type_, _RANK)] = scaled_df[cols].sum(axis=1, skipna=True)

    # Overall rank.
    combined[(_ALL, "overall", _RANK)] = scaled_df.sum(axis=1, skipna=True)

    combined.columns = pd.MultiIndex.from_tuples(
        combined.columns, names=["framework", "type", "metric"]
    )
    combined = combined.sort_values((_ALL, "overall", _RANK), ascending=False)
    combined.index.name = "model"
    return combined


def load_combined_table(path: "str") -> pd.DataFrame:
    """Load a ``combined_evaluation.csv`` written by :func:`build_combined_table`,
    reconstructing its 3-level ``(framework, type, metric)`` column MultiIndex.
    """
    return pd.read_csv(path, header=[0, 1, 2], index_col=0)


def simple_rank_summary(combined: pd.DataFrame) -> pd.DataFrame:
    """Flatten ``combined`` down to a plain model x {overall,utility,privacy,fairness}
    rank table (one row per model, sorted best-to-worst) for readable printing.
    """
    columns = {}
    if (_ALL, "overall", _RANK) in combined.columns:
        columns["overall"] = combined[(_ALL, "overall", _RANK)]
    for type_ in ("utility", "privacy", "fairness"):
        key = (_ALL, type_, _RANK)
        if key in combined.columns:
            columns[type_] = combined[key]

    summary = pd.DataFrame(columns)
    summary.index.name = "model"
    if "overall" in summary.columns:
        summary = summary.sort_values("overall", ascending=False)
    return summary.round(3)
