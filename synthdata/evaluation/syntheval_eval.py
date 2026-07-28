"""SynthEval-based evaluation: runs SynthEval's benchmark() across all cached
synthetic datasets using a custom preset (built from
:mod:`synthdata.evaluation.catalog`, filtered by the configured selection).
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from synthdata.data import Dataset
from synthdata.evaluation.catalog import (
    FAIRNESS_METRICS_WITH_POSITIVE_CLASS,
    SYNTHEVAL_METRIC_TYPE,
    SYNTHEVAL_PRESET,
    resolve_selection,
)
from synthdata.utils import ensure_dir, get_logger, save_json

logger = get_logger(__name__)

_RANK_COLUMNS = {"rank", "u_rank", "p_rank", "f_rank"}

# ---------------------------------------------------------------------------
# Benchmark result caching
# ---------------------------------------------------------------------------
# SynthEval's benchmark() is expensive (tens of minutes for large datasets).
# After a successful run we persist the results and ranks as Parquet files
# alongside a small metadata sidecar that captures the cache key (SHA-256 of
# the preset JSON + sorted model names).  On the next run we load from cache
# if the key matches, skipping the full benchmark pass.
#
# Parquet is used (not CSV) because benchmark_results has a MultiIndex column
# level that CSV cannot round-trip without bespoke reconstruction logic.
# ---------------------------------------------------------------------------


def _compute_cache_key(preset: dict, model_names: list[str], ranking_strategy: str) -> str:
    """Stable SHA-256 hex digest of the preset dict + sorted model names + ranking strategy.

    ``ranking_strategy`` is included because it affects the ranks DataFrame
    returned by ``se.benchmark()`` (not just the metric values).
    """
    payload = json.dumps(
        {"preset": preset, "models": sorted(model_names), "ranking_strategy": ranking_strategy},
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _save_syntheval_cache(
    results: pd.DataFrame,
    ranks: pd.DataFrame,
    cache_dir: Path,
    prefix: str,
    cache_key: str,
) -> None:
    """Persist benchmark results + ranks to Parquet and write the cache-key sidecar.

    Files written:
    - ``<cache_dir>/<prefix>_results.parquet``  -- MultiIndex-column results
    - ``<cache_dir>/<prefix>_ranks.parquet``    -- ranks DataFrame
    - ``<cache_dir>/<prefix>_cache_meta.json``  -- {"cache_key": <sha256>}
    """
    ensure_dir(cache_dir)
    results.to_parquet(cache_dir / f"{prefix}_results.parquet")
    ranks.to_parquet(cache_dir / f"{prefix}_ranks.parquet")
    meta_path = cache_dir / f"{prefix}_cache_meta.json"
    save_json(meta_path, {"cache_key": cache_key})
    logger.info(
        "[syntheval] cached %s benchmark results to %s (key=%s…)",
        prefix,
        cache_dir,
        cache_key[:12],
    )


def _load_syntheval_cache(
    cache_dir: Path,
    prefix: str,
    cache_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Return (results, ranks) from cache if the key matches, else None.

    Returns None (cache miss) if any of the three expected files are absent or
    if the stored cache key doesn't match the current one.
    """
    results_path = cache_dir / f"{prefix}_results.parquet"
    ranks_path = cache_dir / f"{prefix}_ranks.parquet"
    meta_path = cache_dir / f"{prefix}_cache_meta.json"

    if not (results_path.exists() and ranks_path.exists() and meta_path.exists()):
        return None

    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("[syntheval] %s cache meta unreadable (%s); treating as miss", prefix, exc)
        return None

    if meta.get("cache_key") != cache_key:
        logger.info(
            "[syntheval] %s cache key mismatch (stored=%s…, current=%s…); recomputing",
            prefix,
            str(meta.get("cache_key", ""))[:12],
            cache_key[:12],
        )
        return None

    try:
        results = pd.read_parquet(results_path)
        ranks = pd.read_parquet(ranks_path)
    except Exception as exc:  # noqa: BLE001 -- any parquet read error → cache miss
        logger.warning("[syntheval] %s cache files unreadable (%s); recomputing", prefix, exc)
        return None

    logger.info(
        "[syntheval] loaded %s benchmark results from cache (key=%s…, %d models, %d metrics)",
        prefix,
        cache_key[:12],
        len(results),
        len([c for c in results.columns.get_level_values(0).unique() if c not in _RANK_COLUMNS]),
    )
    return results, ranks


#: Metrics that syntheval refuses to run unless the target has EXACTLY 2
#: classes (see e.g. metric_auroc_difference.py / metric_statistical_parity.py
#: / metric_equalized_odds.py / metric_equal_opportunity.py's own
#: `target_types.items() if value == 2` filtering) -- these are the only ones
#: a binary-target evaluation pass (see run_binary_target_syntheval_evaluation)
#: can newly enable; every other metric already runs fine against a 3+ class
#: target and must NOT be re-run against the collapsed binary one (which
#: would silently double-count/duplicate their results).
BINARY_ONLY_METRICS = frozenset(
    {"auroc_diff", "statistical_parity", "equalized_odds", "equal_opportunity"}
)


def build_preset(selection_cfg, positive_class=1) -> dict:
    """Filter the full SynthEval preset down to the configured selection.

    ``positive_class`` overrides the "positive_class" preset param (default
    ``1`` in SYNTHEVAL_PRESET) for the 3 fairness metrics in
    FAIRNESS_METRICS_WITH_POSITIVE_CLASS -- wired from
    ``cfg.evaluation.positive_class``, so it only makes sense when the real
    target column is already exactly 2 classes (e.g. hepatitis). It does NOT
    apply to the separate binary_target pass (see build_binary_preset), whose
    collapsed target is always 1=positive/0=negative by construction.
    """
    all_names = list(SYNTHEVAL_PRESET.keys())
    selected = resolve_selection(
        selection_cfg.enabled,
        selection_cfg.categories,
        selection_cfg.metrics,
        all_names,
        SYNTHEVAL_METRIC_TYPE,
    )
    preset = {k: v for k, v in SYNTHEVAL_PRESET.items() if k in selected}
    for name in FAIRNESS_METRICS_WITH_POSITIVE_CLASS & preset.keys():
        # Shallow-copy before overriding -- SYNTHEVAL_PRESET's nested dicts are
        # shared module-level objects reused on every call; mutating in place
        # would corrupt the global constant for subsequent calls in-process.
        preset[name] = {**preset[name], "positive_class": positive_class}
    return preset


def run_syntheval_evaluation(
    synthetic_datasets: dict[str, pd.DataFrame],
    dataset: Dataset,
    selection_cfg,
    preset_dir: str | Path,
    ranking_strategy: str = "linear",
    output_folder: str | Path | None = None,
    plots_output_dir: str | Path | None = None,
    positive_class=1,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Run SynthEval's benchmark() across all datasets. Returns (benchmark_results, benchmark_ranks).

    Both are None if the selection resolves to zero metrics.

    If ``plots_output_dir`` is given, SynthEval's native per-metric plots (``SE_*.png``)
    are produced as a side effect of this same benchmark pass (one subfolder per model
    under ``plots_output_dir``), instead of requiring a separate, fully redundant
    benchmark/evaluate pass just to regenerate them.

    ``positive_class`` (from ``cfg.evaluation.positive_class``) is forwarded to
    :func:`build_preset` for the 3 fairness metrics -- see its docstring.
    """
    preset = build_preset(selection_cfg, positive_class)
    if not preset:
        logger.info("[syntheval] no metrics selected; skipping")
        return None, None

    from syntheval import AnalysisConfig, SynthEval

    preset_dir = ensure_dir(preset_dir)
    preset_path = preset_dir / "syntheval_preset.json"
    save_json(preset_path, preset)

    analysis_config = AnalysisConfig(
        dataset=dataset.train_imputed_df,
        target_vars=dataset.target_column,
        confounder_vars=None,
        sensitive_vars=dataset.sensitive_columns,
    )

    se = SynthEval(
        dataset.train_imputed_df,
        holdout_dataframe=dataset.test_imputed_df,
        cat_cols=dataset.all_categorical_columns,
        verbose=False,
        enable_plots=plots_output_dir is not None,
        console="off",
        show_warnings=False,
    )

    cache_dir = Path(output_folder) if output_folder else None
    cache_key = _compute_cache_key(preset, list(synthetic_datasets.keys()), ranking_strategy)

    if cache_dir is not None:
        cached = _load_syntheval_cache(cache_dir, "main", cache_key)
        if cached is not None:
            return cached

    logger.info(
        "[syntheval] benchmarking %d datasets across %d metrics%s",
        len(synthetic_datasets),
        len(preset),
        " (with native plots)" if plots_output_dir else "",
    )
    benchmark_results, benchmark_ranks = se.benchmark(
        synthetic_datasets,
        analysis_target=analysis_config,
        presets_file=str(preset_path),
        rank_strategy=ranking_strategy,
        output_folder=str(output_folder) if output_folder else None,
        plot_output_dir=str(plots_output_dir) if plots_output_dir else None,
    )

    if cache_dir is not None:
        _save_syntheval_cache(benchmark_results, benchmark_ranks, cache_dir, "main", cache_key)

    return benchmark_results, benchmark_ranks


def build_binary_target_series(
    series: pd.Series, positive_classes: list, negative_classes: list
) -> pd.Series:
    """Collapse a categorical Series to binary (1=positive_classes, 0=negative_classes).

    Fails loudly (rather than silently coercing to NaN) if any observed
    non-null value isn't covered by either list -- an uncovered value would
    otherwise silently vanish as an unexplained NaN in a supposedly
    fully-observed target column. Also fails loudly if the input has any
    missing values at all: the result is returned as ``int64`` (not float),
    which can't represent NaN -- and that's not merely a dtype nicety: only
    ``object``/``int`` dtypes get treated as *categorical* by SynthEval's
    ``AnalysisConfig`` (anything else, e.g. float, is classified as "num"
    i.e. continuous -- see ``syntheval.utils.configuration.AnalysisConfig``),
    so a float output here would silently make auroc_diff/statistical_parity/
    equalized_odds/equal_opportunity go right back to refusing to run, in a
    much harder-to-diagnose way than an explicit error at build time.
    """
    positive_set, negative_set = set(positive_classes), set(negative_classes)
    observed = set(series.dropna().unique().tolist())
    unmapped = observed - positive_set - negative_set
    if unmapped:
        raise ValueError(
            f"Observed value(s) {sorted(unmapped, key=str)} in column {series.name!r} are not "
            "covered by evaluation.binary_target.positive_classes/negative_classes "
            f"(positive={sorted(positive_classes, key=str)}, negative={sorted(negative_classes, key=str)})"
            " -- every observed value must be listed in one of the two."
        )
    if series.isna().any():
        raise ValueError(
            f"Column {series.name!r} has missing value(s) -- evaluation.binary_target requires a "
            "fully-observed target column (the main evaluation pipeline already drops rows with a "
            "missing target before this point; if this column has genuine missingness, it isn't a "
            "valid evaluation.binary_target.column)."
        )
    return pd.Series(
        np.where(series.isin(positive_set), 1, 0),
        index=series.index,
        name=series.name,
        dtype="int64",
    )


def build_binary_preset(selection_cfg) -> dict:
    """Filter SYNTHEVAL_PRESET down to just the exactly-2-classes-only metrics
    (BINARY_ONLY_METRICS), further filtered by the same selection_cfg used for
    the main preset -- e.g. disabling the 'fairness' category or explicitly
    excluding 'auroc_diff' via evaluation.syntheval.metrics also excludes it
    from this binary-target pass.

    Deliberately does NOT take a ``positive_class`` override (unlike
    build_preset): build_binary_target_series always normalizes the collapsed
    target to 1=positive_classes/0=negative_classes, so "positive_class" must
    stay SYNTHEVAL_PRESET's default of 1 here regardless of
    ``cfg.evaluation.positive_class`` -- threading it through would
    double-remap an already-fixed convention.
    """
    all_names = list(SYNTHEVAL_PRESET.keys())
    selected = resolve_selection(
        selection_cfg.enabled,
        selection_cfg.categories,
        selection_cfg.metrics,
        all_names,
        SYNTHEVAL_METRIC_TYPE,
    )
    return {k: v for k, v in SYNTHEVAL_PRESET.items() if k in selected and k in BINARY_ONLY_METRICS}


def run_binary_target_syntheval_evaluation(
    synthetic_datasets: dict[str, pd.DataFrame],
    dataset: Dataset,
    selection_cfg,
    binary_target_cfg,
    preset_dir: str | Path,
    ranking_strategy: str = "linear",
    output_folder: str | Path | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Run a second, separate SynthEval benchmark() pass against a binary-
    collapsed copy of the target column, for the metrics that require exactly
    2 target classes (BINARY_ONLY_METRICS) and would otherwise be unable to
    run at all against a 3+ class target.

    This never touches ``dataset.train_imputed_df``/``test_imputed_df``/the
    caller's ``synthetic_datasets`` -- only disposable copies, with the target
    column's values (not its name -- see build_binary_target_series) replaced
    in place, so it's still correctly excluded from the model's own feature
    set exactly like the original target (no leakage from the original,
    finer-grained labels lingering as a feature).

    Returns (benchmark_results, benchmark_ranks), both None if no
    BINARY_ONLY_METRICS are selected.
    """
    preset = build_binary_preset(selection_cfg)
    if not preset:
        logger.info("[syntheval] binary-target pass: no eligible metrics selected; skipping")
        return None, None

    column = binary_target_cfg.column or dataset.target_column
    positive_classes = binary_target_cfg.positive_classes
    negative_classes = binary_target_cfg.negative_classes

    def _binarize(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[column] = build_binary_target_series(out[column], positive_classes, negative_classes)
        return out

    train_df = _binarize(dataset.train_imputed_df)
    hout_df = _binarize(dataset.test_imputed_df) if dataset.test_imputed_df is not None else None
    binary_synthetic_datasets = {name: _binarize(df) for name, df in synthetic_datasets.items()}

    from syntheval import AnalysisConfig, SynthEval

    preset_dir = ensure_dir(preset_dir)
    preset_path = preset_dir / "syntheval_binary_target_preset.json"
    save_json(preset_path, preset)

    analysis_config = AnalysisConfig(
        dataset=train_df,
        target_vars=column,
        confounder_vars=None,
        sensitive_vars=dataset.sensitive_columns,
    )

    se = SynthEval(
        train_df,
        holdout_dataframe=hout_df,
        cat_cols=dataset.all_categorical_columns,
        verbose=False,
        enable_plots=False,
        console="off",
        show_warnings=False,
    )

    cache_dir = Path(output_folder) if output_folder else None
    cache_key = _compute_cache_key(preset, list(binary_synthetic_datasets.keys()), ranking_strategy)

    if cache_dir is not None:
        cached = _load_syntheval_cache(cache_dir, "binary_target", cache_key)
        if cached is not None:
            return cached

    logger.info(
        "[syntheval] binary-target pass: benchmarking %d datasets across %d metric(s) "
        "(column %r collapsed to binary: positive=%s, negative=%s)",
        len(binary_synthetic_datasets),
        len(preset),
        column,
        positive_classes,
        negative_classes,
    )
    benchmark_results, benchmark_ranks = se.benchmark(
        binary_synthetic_datasets,
        analysis_target=analysis_config,
        presets_file=str(preset_path),
        rank_strategy=ranking_strategy,
        output_folder=str(output_folder) if output_folder else None,
    )

    if cache_dir is not None:
        _save_syntheval_cache(
            benchmark_results, benchmark_ranks, cache_dir, "binary_target", cache_key
        )

    return benchmark_results, benchmark_ranks


def merge_binary_target_results(
    benchmark_results: pd.DataFrame | None,
    benchmark_ranks: pd.DataFrame | None,
    binary_results: pd.DataFrame | None,
    binary_ranks: pd.DataFrame | None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Merge a binary-target-only pass's per-metric columns into the main
    pass's results.

    Only metric-value columns are merged, never the aggregate rank/u_rank/
    p_rank/f_rank columns from either pass -- those are always recomputed
    from scratch downstream in combine.build_combined_table purely from the
    per-metric oriented values (see extract_oriented_values), so a mini
    (BINARY_ONLY_METRICS-sized) benchmark's own internal aggregate ranks
    would be meaningless to propagate.
    """
    if binary_results is None:
        return benchmark_results, benchmark_ranks
    if benchmark_results is None:
        return binary_results, binary_ranks

    new_metrics = [
        m for m in binary_results.columns.get_level_values(0).unique() if m not in _RANK_COLUMNS
    ]
    for metric in new_metrics:
        benchmark_results[(metric, "value")] = binary_results[(metric, "value")]
        benchmark_results[(metric, "error")] = binary_results[(metric, "error")]
        benchmark_ranks[metric] = binary_ranks[metric]
    return benchmark_results, benchmark_ranks


def extract_raw_values(benchmark_results: pd.DataFrame) -> pd.DataFrame:
    """Models x metrics table of raw metric values from SynthEval's benchmark_results."""
    return benchmark_results.xs("value", axis=1, level=1)


def extract_oriented_values(benchmark_ranks: pd.DataFrame) -> pd.DataFrame:
    """Models x metrics table of SynthEval's pre-oriented (higher=better) n_val scores."""
    cols = [c for c in benchmark_ranks.columns if c not in _RANK_COLUMNS]
    return benchmark_ranks[cols]
