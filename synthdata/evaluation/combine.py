"""Combines synthcity + SynthEval + custom (log disparity / fork-only fairness)
results into a single ranked table with 3-level MultiIndex columns:
``(framework, type, metric)`` where ``framework in {synthcity, syntheval, custom}``
and ``type in {utility, privacy, fairness}``.

Ranking (see module docstring of :func:`build_combined_table` for details) is
appended as extra columns in the same table, both per ``(framework, type)`` group
and rolled up across frameworks per ``type``, plus one overall rank.
"""

import numpy as np
import pandas as pd
from scipy import stats

from synthdata.evaluation.baselines import is_baseline
from synthdata.evaluation.catalog import (
    LOG_DISPARITY_METRICS,
    SYNTHCITY_CATEGORY_TO_TYPE,
    SYNTHCITY_UNRANKED_SUBMETRICS,
    SYNTHEVAL_UNRANKED_METRICS,
    TSTR_METRICS,
    is_custom_syntheval_metric,
    is_redundant_synthcity_submetric,
    is_syntheval_fairness_gap,
)
from synthdata.evaluation.custom_eval import build_log_disparity_summary_table
from synthdata.evaluation.fairness_gaps import FAIRNESS_GAP_METRICS, FAIRNESS_SIGNAL_COLUMN
from synthdata.evaluation.privacy_attacks import (
    ANONYMETER_METRICS,
    HOLDOUT_DISTANCE_METRICS,
    SUBGROUP_PRIVACY_METRICS,
)
from synthdata.evaluation.syntheval_eval import (
    extract_metric_types,
    extract_oriented_values,
    extract_raw_values,
)
from synthdata.utils import get_logger, split_replicate_name

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

    redundant_cols = [c for c in raw.columns if is_redundant_synthcity_submetric(c)]
    if redundant_cols:
        logger.info(
            "[synthcity] excluding known-redundant duplicate sub-metric(s) from combined "
            "table: %s (see catalog.SYNTHCITY_REDUNDANT_SUBMETRIC_SUFFIXES)",
            redundant_cols,
        )
        raw = raw.drop(columns=redundant_cols)

    directions = {}
    for res in ok_results.values():
        for metric_key, direction in res["direction"].items():
            directions.setdefault(metric_key, direction)
    sign = pd.Series(directions).map({"maximize": 1.0, "minimize": -1.0})

    # Unranked sub-metrics stay in the raw table but carry no score.
    common = [c for c in raw.columns if c in sign.index and c not in SYNTHCITY_UNRANKED_SUBMETRICS]
    oriented = raw[common].multiply(sign[common], axis=1)

    raw = raw.reindex(model_names)
    oriented = oriented.reindex(model_names)

    def _columns(frame: pd.DataFrame) -> pd.MultiIndex:
        return pd.MultiIndex.from_tuples(
            [
                ("synthcity", SYNTHCITY_CATEGORY_TO_TYPE.get(col.split(".")[0], "utility"), col)
                for col in frame.columns
            ]
        )

    raw.columns = _columns(raw)
    oriented.columns = _columns(oriented)
    return raw, oriented


def _syntheval_frames(
    benchmark_results: pd.DataFrame | None,
    benchmark_ranks: pd.DataFrame | None,
    model_names: list,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Build (raw, oriented) models x metric-name tables from SynthEval results.

    Each column's type is the one SynthEval itself tagged the result with.
    Metrics matching is_custom_syntheval_metric() (fork-only additions, plus
    their full_output=True per-(target_var, protected_attribute) sub-columns)
    are tagged framework="custom" instead of "syntheval".
    """
    if benchmark_results is None:
        empty = pd.DataFrame(index=model_names)
        return empty, empty

    # Results and ranks come in pairs: both None or both set.
    assert benchmark_ranks is not None
    raw = extract_raw_values(benchmark_results).reindex(model_names)
    types = extract_metric_types(benchmark_results)
    oriented = extract_oriented_values(benchmark_ranks).reindex(model_names)

    def _framework(metric: str) -> str:
        return "custom" if is_custom_syntheval_metric(metric) else "syntheval"

    columns = [(_framework(col), types[col], col) for col in raw.columns]
    raw.columns = pd.MultiIndex.from_tuples(columns)
    oriented.columns = pd.MultiIndex.from_tuples(columns)
    # Their classifier is fitted and scored inside the synthetic data, so a
    # dataset with no class signal gets a perfect 0; the holdout fairness
    # gaps (_fairness_gap_frames) are ranked instead.
    oriented = oriented[
        [
            c
            for c in oriented.columns
            if not is_syntheval_fairness_gap(c[2]) and c[2] not in SYNTHEVAL_UNRANKED_METRICS
        ]
    ]
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


def _tstr_frames(
    tstr_result: dict | None, model_names: list
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Holdout TSTR macro scores as custom utility columns (all higher-is-better).

    TSTS scores (train and test on synthetic rows) are reported next to them
    but unranked: a high TSTS is not better in itself, only its gap to TSTR
    says something (docs/verification.md).
    """
    scores = (tstr_result or {}).get("scores") or {}
    if not scores:
        empty = pd.DataFrame(index=model_names)
        return empty, empty
    tsts = (tstr_result or {}).get("tsts") or {}

    def _scores(prefix: str, results: dict) -> dict:
        return {
            f"{prefix}_macro_f1": {m: s.macro_f1 for m, s in results.items()},
            f"{prefix}_balanced_accuracy": {m: s.balanced_accuracy for m, s in results.items()},
            f"{prefix}_macro_auprc": {m: s.macro_auprc for m, s in results.items()},
        }

    oriented = pd.DataFrame(_scores("tstr", scores)).reindex(
        index=model_names, columns=list(TSTR_METRICS)
    )
    raw = oriented.copy()
    if tsts:
        raw = pd.concat([raw, pd.DataFrame(_scores("tsts", tsts)).reindex(model_names)], axis=1)
    raw.columns = pd.MultiIndex.from_tuples([("custom", "utility", c) for c in raw.columns])
    oriented.columns = pd.MultiIndex.from_tuples(
        [("custom", "utility", c) for c in oriented.columns]
    )
    return raw, oriented


def _privacy_attack_frames(
    privacy_result: dict | None, model_names: list
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Anonymeter risks and holdout-referenced distances as custom privacy columns.

    Values on the safe side of the no-memorization reference (share or AUC
    below 0.5, ratio above 1) earn no extra credit: being further from the
    training rows than unseen real rows are is not more private, only less
    useful. So the oriented share/AUC are clipped at 0.5 and the ratios at 1.

    The subgroup privacy gap is added to the raw table only, as custom
    fairness columns: reported, not ranked (see SUBGROUP_PRIVACY_METRICS).
    """
    scores = (privacy_result or {}).get("scores") or {}
    if not scores:
        empty = pd.DataFrame(index=model_names)
        return empty, empty
    metrics = [*ANONYMETER_METRICS, *HOLDOUT_DISTANCE_METRICS]
    table = pd.DataFrame.from_dict(scores, orient="index").reindex(index=model_names)
    raw = table[[m for m in metrics if m in table.columns]].astype(float)
    fairness = table[[m for m in SUBGROUP_PRIVACY_METRICS if m in table.columns]].astype(float)
    oriented = pd.DataFrame(index=raw.index)
    for metric in raw.columns:
        if metric in ANONYMETER_METRICS:
            oriented[metric] = -raw[metric]
        elif HOLDOUT_DISTANCE_METRICS[metric]:
            oriented[metric] = -raw[metric].clip(lower=0.5)
        else:
            oriented[metric] = raw[metric].clip(upper=1.0)
    for frame in (raw, oriented):
        frame.columns = pd.MultiIndex.from_tuples([("custom", "privacy", c) for c in frame.columns])
    if not len(fairness.columns):
        return raw, oriented
    fairness.columns = pd.MultiIndex.from_tuples(
        [("custom", "fairness", c) for c in fairness.columns]
    )
    return pd.concat([raw, fairness], axis=1), oriented


def _fairness_gap_frames(
    table: pd.DataFrame | None, model_names: list
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Holdout fairness gap errors as custom fairness columns (lower is better).

    ``fairness_signal`` is reported, not ranked.
    """
    if table is None or table.empty:
        empty = pd.DataFrame(index=model_names)
        return empty, empty
    table = table.reindex(model_names)
    metrics = [m for m in FAIRNESS_GAP_METRICS if m in table.columns]
    raw = table[[*metrics, FAIRNESS_SIGNAL_COLUMN]].astype(float)
    oriented = -raw[metrics]
    raw.columns = pd.MultiIndex.from_tuples([("custom", "fairness", c) for c in raw.columns])
    oriented.columns = pd.MultiIndex.from_tuples(
        [("custom", "fairness", c) for c in oriented.columns]
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


def _missing_as_worst(col: pd.Series) -> pd.Series:
    """A scaled metric some rows have and others lack scores 0 (worst) for those.

    A metric is missing for a model when it failed or produced no result on
    that model's data (e.g. synthcity's data_leakage raising on a category the
    synthetic data lacks). Averaging over the metrics a model happens to have
    would let such a failure lift it above models that were fully scored.
    A metric missing for every model stays NaN and drops out.
    """
    if col.notna().any():
        return col.fillna(0.0)
    return col


#: Types rolled up in the combined table. Order matters for iteration below
#: but not for correctness (the weighted geometric mean is order-independent).
_TYPES = ("utility", "privacy", "fairness")

#: Equal weighting used when the caller doesn't pass ``rank_weights`` (e.g.
#: existing direct callers/tests predating evaluation.rank_weights) --
#: mirrors the pre-existing implicit behavior before per-type weights existed.
DEFAULT_RANK_WEIGHTS = {"utility": 1.0, "privacy": 1.0, "fairness": 1.0}

#: Type scores are floored here before the geometric mean, so a row that is
#: worst on every metric of one type (scaled score 0) is still ordered by its
#: other types instead of collapsing to an overall score of 0.
GEOMETRIC_FLOOR = 0.01


#: File (in the evaluation folder) with each metric's direction and ideal value.
METRIC_DIRECTIONS_FILENAME = "metric_directions.csv"

#: Values that mean "no privacy loss" for the custom privacy metrics: an
#: attack no better than its baseline, synthetic rows no closer to the
#: training rows than to unseen real rows.
_KNOWN_IDEALS = {
    **{name: 0.0 for name in ANONYMETER_METRICS},
    "dcr_closer_to_train_share": 0.5,
    "distance_mia_auc": 0.5,
    "dcr_holdout_ratio": 1.0,
    "nndr_holdout_ratio": 1.0,
    "subgroup_distance_mia_auc_gap": 0.0,
    "subgroup_distance_mia_auc_max": 0.5,
    **{name: 0.0 for name in LOG_DISPARITY_METRICS},
    **{name: 0.0 for name in FAIRNESS_GAP_METRICS},
}


def metric_directions(raw: pd.DataFrame, oriented: pd.DataFrame) -> pd.DataFrame:
    """Which way each raw metric is better, read from how it was oriented for ranking.

    ``direction`` is "higher" or "lower" (the oriented score rises or falls
    with the raw value, by Spearman correlation across models), "closer to 0"
    (a signed gap scored as 1 - |value| or -|value|) or "unranked" (reported,
    not ranked). ``ideal`` is the value that means perfect or no risk where
    one exists, else NaN. Index: the raw table's ``(framework, type, metric)``.
    """
    rows = []
    for column in raw.columns:
        ideal = _KNOWN_IDEALS.get(column[2], np.nan)
        if column not in oriented.columns:
            rows.append({"direction": "unranked", "ideal": ideal})
            continue
        pair = pd.concat([raw[column], oriented[column]], axis=1).dropna().astype(float)
        r, o = pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy()
        signed = len(r) and (r < 0).any() and (r > 0).any()
        if signed and (np.allclose(o, 1 - np.abs(r)) or np.allclose(o, -np.abs(r))):
            rows.append({"direction": "closer to 0", "ideal": 0.0})
            continue
        rho = stats.spearmanr(r, o).statistic if len(r) > 2 and np.ptp(r) > 0 else np.nan
        if np.isnan(rho):
            # Too few distinct values to tell: identity or negation decides.
            rho = 1.0 if np.allclose(o, r) else -1.0 if np.allclose(o, -r) else np.nan
        direction = "higher" if rho > 0 else "lower" if rho < 0 else "unknown"
        rows.append({"direction": direction, "ideal": ideal})
    table = pd.DataFrame(rows, index=raw.columns)
    table.index = pd.MultiIndex.from_tuples(table.index, names=["framework", "type", "metric"])
    return table


def load_metric_directions(evaluation_dir) -> "pd.DataFrame | None":
    """Read :data:`METRIC_DIRECTIONS_FILENAME` from ``evaluation_dir``, or None."""
    from pathlib import Path

    path = Path(evaluation_dir) / METRIC_DIRECTIONS_FILENAME
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=[0, 1, 2])


def build_combined_table(
    synthcity_results: dict[str, pd.DataFrame],
    syntheval_benchmark_results: pd.DataFrame | None,
    syntheval_benchmark_ranks: pd.DataFrame | None,
    log_disparity_reports: dict[str, dict],
    model_names: list,
    rank_weights: dict | None = None,
    tstr_result: dict | None = None,
    privacy_result: dict | None = None,
    fairness_gap_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the single combined, ranked, multi-index evaluation table.

    ``tstr_result`` (from :func:`synthdata.evaluation.custom_eval.run_tstr_evaluation`)
    adds the holdout TSTR macro scores as a ``("custom", "utility")`` group, and
    ``privacy_result`` (from :func:`synthdata.evaluation.custom_eval.run_privacy_evaluation`)
    the Anonymeter risks and holdout-referenced distances as ``("custom", "privacy")``.
    ``fairness_gap_table`` (from
    :func:`synthdata.evaluation.fairness_gaps.build_fairness_gap_table`) adds
    the holdout fairness gap errors to ``("custom", "fairness")``; they and
    log disparity are the ranked fairness metrics (SynthEval's subgroup gaps
    are reported only).

    Ranking scheme -- a hierarchical *mean-of-means*, not a flat sum, so a
    framework/type with many metric columns (e.g. synthcity's ~7-column
    "performance" category) can't silently outweigh one with few (e.g.
    syntheval's single-column ``cls_acc``) purely by virtue of column count:

      1. Every metric is oriented so "higher = better", then min-max scaled
         across models (independently per metric). A metric a model lacks
         (failed or empty) while others have it scores 0, the worst value.
      2. A sub-rank is computed per ``(framework, type)`` group as the MEAN
         of that group's scaled metrics -- column ``(framework, type, "rank")``.
      3. A rolled-up rank per ``type`` (utility/privacy/fairness) is the MEAN
         of the *group ranks* from step 2 across frameworks (not a flat mean/
         sum of every individual metric of that type) -- column
         ``("__all__", type, "rank")``.
      4. One overall rank is the WEIGHTED GEOMETRIC MEAN of the 3 type-level
         rollups from step 3, using ``rank_weights`` (default: equal weight 1.0
         each, see ``DEFAULT_RANK_WEIGHTS``) -- column
         ``("__all__", "overall", "rank")``, in [GEOMETRIC_FLOOR, 1]. Unlike a
         sum, a near-zero score on one type (e.g. privacy for a model that
         copies its training rows) cannot be bought back by the others.
    Models are sorted descending by the overall rank.
    """
    rank_weights = rank_weights or DEFAULT_RANK_WEIGHTS

    sc_raw, sc_oriented = _synthcity_frames(synthcity_results, model_names)
    se_raw, se_oriented = _syntheval_frames(
        syntheval_benchmark_results, syntheval_benchmark_ranks, model_names
    )
    ld_raw, ld_oriented = _log_disparity_frames(log_disparity_reports, model_names)
    ts_raw, ts_oriented = _tstr_frames(tstr_result, model_names)
    pa_raw, pa_oriented = _privacy_attack_frames(privacy_result, model_names)
    fg_raw, fg_oriented = _fairness_gap_frames(fairness_gap_table, model_names)

    raw_parts = [df for df in (sc_raw, se_raw, ld_raw, ts_raw, pa_raw, fg_raw) if not df.empty]
    oriented_parts = [
        df
        for df in (sc_oriented, se_oriented, ld_oriented, ts_oriented, pa_oriented, fg_oriented)
        if not df.empty
    ]

    if not raw_parts:
        raise ValueError("No evaluation results to combine: check evaluation config selection")

    raw_df = pd.concat(raw_parts, axis=1)
    oriented_df = pd.concat(oriented_parts, axis=1)

    scaled_df = oriented_df.apply(_minmax_scale, axis=0).apply(_missing_as_worst, axis=0)

    combined = raw_df.copy()

    # Step 2: per (framework, type) sub-rank -- MEAN of that group's scaled metrics.
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
        combined[(framework, type_, _RANK)] = scaled_df[cols].mean(axis=1, skipna=True)

    # Step 3: rolled-up rank per type -- MEAN of the group ranks just computed
    # for that type (across frameworks), NOT a flat mean/sum of every
    # individual metric of that type.
    for type_ in _TYPES:
        group_rank_cols = [(fw, t, _RANK) for fw, t in groups if t == type_]
        if group_rank_cols:
            combined[(_ALL, type_, _RANK)] = combined[group_rank_cols].mean(axis=1, skipna=True)

    # Step 4: overall rank -- WEIGHTED GEOMETRIC MEAN of the type-level rollups.
    log_total = pd.Series(0.0, index=combined.index)
    weight_total = 0.0
    for type_ in _TYPES:
        key = (_ALL, type_, _RANK)
        weight = rank_weights.get(type_, 1.0)
        if key in combined.columns and weight > 0:
            floored = combined[key].fillna(0.0).clip(lower=GEOMETRIC_FLOOR)
            log_total = log_total + weight * np.log(floored)
            weight_total += weight
    overall = np.exp(log_total / weight_total) if weight_total else log_total
    combined[(_ALL, "overall", _RANK)] = overall

    combined.columns = pd.MultiIndex.from_tuples(
        combined.columns, names=["framework", "type", "metric"]
    )
    combined = combined.sort_values((_ALL, "overall", _RANK), ascending=False)
    combined.index.name = "model"
    combined.attrs["metric_directions"] = metric_directions(raw_df, oriented_df)
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


#: Two-sided confidence level of the intervals in :func:`summarize_replicates`,
#: and 1 - the significance level of its "tied with best" test.
CONFIDENCE = 0.95

_SUMMARY_DIMS = ("overall", "utility", "privacy", "fairness")


def mean_and_interval(values: pd.Series, confidence: float) -> tuple[float, float, float]:
    """Mean and Student-t confidence interval; the interval is NaN below 2 values."""
    values = values.dropna()
    mean = values.mean() if len(values) else np.nan
    if len(values) < 2:
        return mean, np.nan, np.nan
    sem = values.std(ddof=1) / np.sqrt(len(values))
    half = stats.t.ppf((1 + confidence) / 2, len(values) - 1) * sem
    return mean, mean - half, mean + half


def _differs(best: pd.Series, other: pd.Series, alpha: float) -> "bool | float":
    """Whether ``best`` scores higher than ``other`` beyond seed noise (Welch's t-test).

    NaN when either side has fewer than two replicates, so nothing can be said.
    Replicates of two models share no randomness, so the samples are unpaired.
    """
    best, other = best.dropna(), other.dropna()
    if len(best) < 2 or len(other) < 2:
        return np.nan
    if best.std(ddof=1) == 0 and other.std(ddof=1) == 0:
        return bool(best.mean() > other.mean())
    result = stats.ttest_ind(best, other, equal_var=False, alternative="greater")
    return bool(result.pvalue < alpha)


def summarize_replicates(combined: pd.DataFrame, confidence: float = CONFIDENCE) -> pd.DataFrame:
    """Collapse seed replicates (``<model>__rep<r>``) into one row per model.

    For each rank score (overall, utility, privacy, fairness) the table gives
    the mean over replicates and a ``confidence`` Student-t interval
    (``<dim>_mean``, ``<dim>_ci_low``, ``<dim>_ci_high``). The interval is
    NaN for a single replicate: one seed carries no uncertainty estimate.

    ``eligible`` marks the models a recommendation may pick: every model
    that is not a baseline. Among
    them, ``tied_with_best`` is True for the model with the highest mean
    overall score and for every model whose overall score a one-sided Welch
    t-test cannot place below it at ``1 - confidence``; it is NaN when either
    side has a single replicate. ``rank`` orders all rows by mean overall score.
    """
    alpha = 1 - confidence
    models = pd.Series(
        [split_replicate_name(name)[0] for name in combined.index], index=combined.index
    )
    dims = [d for d in _SUMMARY_DIMS if (_ALL, d, _RANK) in combined.columns]
    scores = {d: combined[(_ALL, d, _RANK)].astype(float) for d in dims}

    rows = {}
    overall_by_model = {}
    for model, names in models.groupby(models, sort=False).groups.items():
        row: dict = {"n_replicates": len(names), "baseline": is_baseline(str(model))}
        for dim in dims:
            mean, low, high = mean_and_interval(scores[dim][names], confidence)
            row[f"{dim}_mean"] = mean
            row[f"{dim}_ci_low"] = low
            row[f"{dim}_ci_high"] = high
        row["eligible"] = not row["baseline"]
        rows[model] = row
        if "overall" in scores:
            overall_by_model[model] = scores["overall"][names]

    summary = pd.DataFrame.from_dict(rows, orient="index")
    summary.index.name = "model"
    if "overall_mean" not in summary.columns:
        return summary

    summary = summary.sort_values("overall_mean", ascending=False)
    summary.insert(0, "rank", range(1, len(summary) + 1))
    tied = pd.Series(np.nan, index=summary.index, dtype=object)
    eligible = summary.index[summary["eligible"].astype(bool)]
    if len(eligible):
        best = eligible[0]
        tied[best] = True
        for model in eligible[1:]:
            differs = _differs(overall_by_model[best], overall_by_model[model], alpha)
            tied[model] = np.nan if pd.isna(differs) else not differs
    summary["tied_with_best"] = tied
    return summary
