"""SynthEval-based evaluation: runs SynthEval's benchmark() across all cached
synthetic datasets using a custom preset (built from
:mod:`synthdata.evaluation.catalog`, filtered by the configured selection).
"""

import dataclasses
import hashlib
import json
import multiprocessing
import os
import socket
import time
import traceback
import uuid
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
_CHECKPOINT_SCHEMA_VERSION = 1


def _atomic_json(path: Path, payload: dict) -> None:
    """Atomically replace a JSON sidecar in its destination directory."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    """Atomically replace a Parquet result checkpoint."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_parquet(temporary)
    os.replace(temporary, path)


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    """Content hash including column order and dtypes for cache validity."""
    digest = hashlib.sha256()
    digest.update(
        repr([(str(column), str(dtype)) for column, dtype in frame.dtypes.items()]).encode()
    )
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def _checkpoint_model_id(model_name: str) -> str:
    """Stable path-safe model id while retaining the original name in metadata."""
    return hashlib.sha256(model_name.encode()).hexdigest()[:16]


def _available_memory_gib() -> float | None:
    """Return Linux MemAvailable in GiB, or None where it cannot be observed."""
    try:
        lines = Path("/proc/meminfo").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    return None


def resolve_model_workers(execution_cfg, *, n_models: int, n_columns: int) -> int:
    """Resolve a memory- and CPU-bounded number of concurrent model processes.

    The Linux ``MemAvailable`` value is already the kernel's estimate of
    immediately allocatable memory. It is therefore the only host-memory
    value used for auto-sizing: ``MemTotal`` can describe a smaller container
    or runner limit than the available-memory probe supplied by callers/tests,
    making worker selection depend on an unrelated second system read.
    """
    if not n_models:
        return 0
    requested = execution_cfg.model_workers
    if requested != "auto":
        return min(requested, execution_cfg.max_model_workers, n_models)

    cpu_count = os.cpu_count() or 1
    cpu_bound = max(1, cpu_count // execution_cfg.cores_per_model)
    per_model_gib = execution_cfg.memory_per_model_gib or max(6.0, 0.0135 * n_columns)
    available_gib = _available_memory_gib()
    if available_gib is None:
        memory_bound = 1
    else:
        budget_gib = max(0.0, available_gib - execution_cfg.memory_reserve_gib)
        memory_bound = max(1, int(budget_gib // per_model_gib))
    return max(1, min(n_models, execution_cfg.max_model_workers, cpu_bound, memory_bound))


def _checkpoint_paths(
    checkpoint_root: Path, pass_name: str, model_name: str
) -> tuple[Path, Path, Path]:
    model_dir = (
        checkpoint_root
        / f"checkpoints-v{_CHECKPOINT_SCHEMA_VERSION}"
        / pass_name
        / _checkpoint_model_id(model_name)
    )
    return model_dir, model_dir / "status.json", model_dir / "result.parquet"


def _valid_checkpoint(
    checkpoint_root: Path,
    pass_name: str,
    model_name: str,
    context_fingerprint: str,
    model_fingerprint: str,
    require_plots: bool,
) -> pd.DataFrame | None:
    model_dir, status_path, result_path = _checkpoint_paths(checkpoint_root, pass_name, model_name)
    if not (status_path.exists() and result_path.exists()):
        return None
    try:
        status = json.loads(status_path.read_text())
        if (
            status.get("state") != "succeeded"
            or status.get("schema_version") != _CHECKPOINT_SCHEMA_VERSION
            or status.get("model_name") != model_name
            or status.get("context_fingerprint") != context_fingerprint
            or status.get("model_fingerprint") != model_fingerprint
            or (require_plots and not status.get("plots_completed"))
        ):
            return None
        return pd.read_parquet(result_path)
    except (OSError, ValueError, json.JSONDecodeError):
        logger.warning(
            "[syntheval] invalid checkpoint for %s at %s; recomputing", model_name, model_dir
        )
        return None


def _shutdown_nested_joblib_executor() -> None:
    """Stop SynthEval's reusable loky pool before a disposable worker exits.

    SynthEval's metric-level joblib calls intentionally keep the reusable pool
    alive. In a one-model subprocess that leaves interpreter shutdown waiting
    for loky's 300-second idle timeout, so terminate the pool explicitly here.
    """
    from joblib.externals.loky import reusable_executor

    executor = getattr(reusable_executor, "_executor", None)
    if executor is not None:
        executor.shutdown(wait=True, kill_workers=True)


def _model_worker(
    model_name: str,
    synthetic_frame: pd.DataFrame,
    real_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame | None,
    cat_cols: list,
    target_column: str,
    sensitive_columns: list,
    preset_path: str,
    checkpoint_root: str,
    pass_name: str,
    context_fingerprint: str,
    model_fingerprint: str,
    plots_output_dir: str | None,
    cores_per_model: int,
) -> None:
    """Run exactly one model in a disposable child process and checkpoint it."""
    os.environ["LOKY_MAX_CPU_COUNT"] = str(cores_per_model)
    os.environ["OMP_NUM_THREADS"] = str(cores_per_model)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cores_per_model)
    # Workers change into their model-specific native-plot directory before
    # checkpointing. Keep checkpoint paths absolute so a successful evaluation
    # cannot fail merely because its current working directory changed.
    checkpoint_root_path = Path(checkpoint_root).resolve()
    model_dir, status_path, result_path = _checkpoint_paths(
        checkpoint_root_path, pass_name, model_name
    )
    ensure_dir(model_dir)
    start = time.time()
    _atomic_json(
        status_path,
        {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "state": "running",
            "model_name": model_name,
            "context_fingerprint": context_fingerprint,
            "model_fingerprint": model_fingerprint,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at": start,
            "shape": list(synthetic_frame.shape),
            "plots_completed": False,
        },
    )
    original_dir = Path.cwd()
    try:
        from syntheval import AnalysisConfig, SynthEval

        analysis_config = AnalysisConfig(
            dataset=real_frame,
            target_vars=target_column,
            confounder_vars=None,
            sensitive_vars=sensitive_columns,
        )
        plot_dir = None
        if plots_output_dir is not None:
            # Resolve before changing the worker's directory. SynthEval writes
            # native diagnostics relative to CWD; retaining an absolute path
            # lets us accurately inventory the files after evaluation.
            plot_dir = ensure_dir(Path(plots_output_dir).resolve() / model_name)
            os.chdir(plot_dir)
        se = SynthEval(
            real_frame,
            holdout_dataframe=holdout_frame,
            cat_cols=cat_cols,
            verbose=False,
            enable_plots=plot_dir is not None,
            console="off",
            show_warnings=False,
        )
        result = se.evaluate(
            synthetic_frame,
            analysis_target=analysis_config,
            presets_file=preset_path,
            _dataset_name=model_name,
        )
        if result is None:
            raise RuntimeError("SynthEval returned no normalized metric results")
        _atomic_parquet(result_path, result)
        plot_files = (
            sorted(
                str(path.relative_to(plot_dir)) for path in plot_dir.rglob("*") if path.is_file()
            )
            if plot_dir
            else []
        )
        _atomic_json(
            status_path,
            {
                "schema_version": _CHECKPOINT_SCHEMA_VERSION,
                "state": "succeeded",
                "model_name": model_name,
                "context_fingerprint": context_fingerprint,
                "model_fingerprint": model_fingerprint,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "started_at": start,
                "completed_at": time.time(),
                "elapsed_seconds": time.time() - start,
                "shape": list(synthetic_frame.shape),
                "plots_completed": plot_dir is not None,
                "plot_files": plot_files,
            },
        )
    except (OSError, ValueError, RuntimeError) as exc:
        _atomic_json(
            status_path,
            {
                "schema_version": _CHECKPOINT_SCHEMA_VERSION,
                "state": "failed",
                "model_name": model_name,
                "context_fingerprint": context_fingerprint,
                "model_fingerprint": model_fingerprint,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "started_at": start,
                "failed_at": time.time(),
                "elapsed_seconds": time.time() - start,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
                "shape": list(synthetic_frame.shape),
                "plots_completed": False,
            },
        )
        raise
    finally:
        try:
            _shutdown_nested_joblib_executor()
        finally:
            os.chdir(original_dir)


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


def _compute_cache_key(
    preset: dict,
    model_names: list[str],
    ranking_strategy: str,
    evaluation_fingerprint: str | None = None,
) -> str:
    """Stable SHA-256 digest of the benchmark configuration and input fingerprint.

    ``ranking_strategy`` is included because it affects the ranks DataFrame
    returned by ``se.benchmark()`` (not just the metric values).
    """
    payload = json.dumps(
        {
            "preset": preset,
            "models": sorted(model_names),
            "ranking_strategy": ranking_strategy,
            "evaluation_fingerprint": evaluation_fingerprint,
        },
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
    _atomic_parquet(cache_dir / f"{prefix}_results.parquet", results)
    _atomic_parquet(cache_dir / f"{prefix}_ranks.parquet", ranks)
    meta_path = cache_dir / f"{prefix}_cache_meta.json"
    _atomic_json(meta_path, {"cache_key": cache_key})
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


def _evaluation_context_fingerprint(
    dataset: Dataset,
    preset: dict,
    pass_name: str,
    plots_enabled: bool,
) -> str:
    """Fingerprint inputs shared by every model in one evaluation pass."""
    payload = {
        "schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "pass_name": pass_name,
        "preset": preset,
        "dataset_name": dataset.name,
        "dataset_version": dataset.version,
        "target_column": dataset.target_column,
        "sensitive_columns": dataset.sensitive_columns,
        "categorical_columns": dataset.all_categorical_columns,
        "train": _frame_fingerprint(dataset.train_imputed_df),
        "holdout": _frame_fingerprint(dataset.test_imputed_df),
        "plots_enabled": plots_enabled,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _run_resumable_syntheval(
    synthetic_datasets: dict[str, pd.DataFrame],
    dataset: Dataset,
    preset: dict,
    preset_path: Path,
    output_folder: str | Path,
    ranking_strategy: str,
    execution_cfg,
    pass_name: str,
    plots_output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate models in disposable bounded processes and resume checkpoints.

    Each child handles exactly one model. Its process exit releases all native
    and allocator high-water memory before the parent admits another model.
    """
    from syntheval.syntheval import aggregate_benchmark_results

    checkpoint_root = ensure_dir(output_folder).resolve()
    plots_enabled = plots_output_dir is not None
    context_fingerprint = _evaluation_context_fingerprint(dataset, preset, pass_name, plots_enabled)
    results: dict[str, pd.DataFrame] = {}
    pending: list[tuple[str, pd.DataFrame, str]] = []
    for model_name, frame in synthetic_datasets.items():
        model_fingerprint = _frame_fingerprint(frame)
        cached = _valid_checkpoint(
            checkpoint_root,
            pass_name,
            model_name,
            context_fingerprint,
            model_fingerprint,
            plots_enabled,
        )
        if cached is None:
            pending.append((model_name, frame, model_fingerprint))
        else:
            logger.info("[syntheval] %s checkpoint hit for model %s", pass_name, model_name)
            results[model_name] = cached

    if pending:
        workers = resolve_model_workers(
            execution_cfg,
            n_models=len(pending),
            n_columns=dataset.train_imputed_df.shape[1],
        )
        logger.info(
            "[syntheval] %s scheduling %d missing model(s) with %d disposable worker(s) "
            "(train=%s, holdout=%s, features=%d, plots=%s)",
            pass_name,
            len(pending),
            workers,
            dataset.train_imputed_df.shape,
            dataset.test_imputed_df.shape if dataset.test_imputed_df is not None else None,
            dataset.train_imputed_df.shape[1],
            plots_enabled,
        )
        context = multiprocessing.get_context("spawn")
        active: dict[str, multiprocessing.Process] = {}
        pending_iter = iter(pending)

        def start_next() -> bool:
            try:
                model_name, frame, model_fingerprint = next(pending_iter)
            except StopIteration:
                return False
            process = context.Process(
                target=_model_worker,
                args=(
                    model_name,
                    frame,
                    dataset.train_imputed_df,
                    dataset.test_imputed_df,
                    dataset.all_categorical_columns,
                    dataset.target_column,
                    dataset.sensitive_columns,
                    str(preset_path.resolve()),
                    str(checkpoint_root),
                    pass_name,
                    context_fingerprint,
                    model_fingerprint,
                    str(plots_output_dir) if plots_output_dir else None,
                    execution_cfg.cores_per_model,
                ),
                name=f"syntheval-{pass_name}-{model_name}",
            )
            process.start()
            active[model_name] = process
            logger.info(
                "[syntheval] %s started model=%s pid=%s", pass_name, model_name, process.pid
            )
            return True

        for _ in range(workers):
            if not start_next():
                break

        failures = []
        while active:
            completed = []
            for model_name, process in active.items():
                if process.is_alive():
                    continue
                process.join()
                completed.append((model_name, process.exitcode))
            if not completed:
                time.sleep(0.1)
                continue
            for model_name, exitcode in completed:
                del active[model_name]
                model_fingerprint = _frame_fingerprint(synthetic_datasets[model_name])
                cached = _valid_checkpoint(
                    checkpoint_root,
                    pass_name,
                    model_name,
                    context_fingerprint,
                    model_fingerprint,
                    plots_enabled,
                )
                if exitcode == 0 and cached is not None:
                    results[model_name] = cached
                    logger.info("[syntheval] %s completed model=%s", pass_name, model_name)
                else:
                    _, status_path, _ = _checkpoint_paths(checkpoint_root, pass_name, model_name)
                    _atomic_json(
                        status_path,
                        {
                            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
                            "state": "failed",
                            "model_name": model_name,
                            "context_fingerprint": context_fingerprint,
                            "model_fingerprint": model_fingerprint,
                            "exit_code": exitcode,
                            "failed_at": time.time(),
                            "failure_reason": "worker exited without a valid succeeded checkpoint",
                        },
                    )
                    failures.append(f"{model_name} (exit={exitcode}, status={status_path})")
                    logger.error(
                        "[syntheval] %s model=%s exited %s without a valid checkpoint",
                        pass_name,
                        model_name,
                        exitcode,
                    )
                start_next()
        if failures:
            raise RuntimeError(
                f"SynthEval {pass_name} failed for {len(failures)} model(s): {', '.join(failures)}. "
                f"Completed model checkpoints remain resumable under {checkpoint_root}."
            )

    ordered_results = {name: results[name] for name in synthetic_datasets}
    return aggregate_benchmark_results(ordered_results, ranking_strategy)


def run_syntheval_evaluation(
    synthetic_datasets: dict[str, pd.DataFrame],
    dataset: Dataset,
    selection_cfg,
    preset_dir: str | Path,
    ranking_strategy: str = "linear",
    output_folder: str | Path | None = None,
    plots_output_dir: str | Path | None = None,
    positive_class=1,
    execution_cfg=None,
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

    preset_dir = ensure_dir(preset_dir)
    preset_path = preset_dir / "syntheval_preset.json"
    save_json(preset_path, preset)

    cache_dir = Path(output_folder) if output_folder else preset_dir / "syntheval_benchmark"
    context_fingerprint = _evaluation_context_fingerprint(
        dataset, preset, "main", plots_output_dir is not None
    )
    model_fingerprints = {
        name: _frame_fingerprint(frame) for name, frame in synthetic_datasets.items()
    }
    cache_key = _compute_cache_key(
        preset,
        list(synthetic_datasets.keys()),
        ranking_strategy,
        hashlib.sha256(
            json.dumps(
                {"context": context_fingerprint, "models": model_fingerprints}, sort_keys=True
            ).encode()
        ).hexdigest(),
    )

    cached = _load_syntheval_cache(cache_dir, "main", cache_key)
    if cached is not None:
        return cached

    if execution_cfg is None:
        from synthdata.config import SynthEvalExecutionConfig

        execution_cfg = SynthEvalExecutionConfig()
    benchmark_results, benchmark_ranks = _run_resumable_syntheval(
        synthetic_datasets,
        dataset,
        preset,
        preset_path,
        cache_dir,
        ranking_strategy,
        execution_cfg,
        "main",
        plots_output_dir,
    )

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
    execution_cfg=None,
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

    preset_dir = ensure_dir(preset_dir)
    preset_path = preset_dir / "syntheval_binary_target_preset.json"
    save_json(preset_path, preset)
    binary_dataset = dataclasses.replace(
        dataset,
        target_column=column,
        train_imputed_df=train_df,
        test_imputed_df=hout_df,
    )
    cache_dir = Path(output_folder) if output_folder else preset_dir / "syntheval_benchmark"
    context_fingerprint = _evaluation_context_fingerprint(
        binary_dataset, preset, "binary_target", False
    )
    model_fingerprints = {
        name: _frame_fingerprint(frame) for name, frame in binary_synthetic_datasets.items()
    }
    cache_key = _compute_cache_key(
        preset,
        list(binary_synthetic_datasets.keys()),
        ranking_strategy,
        hashlib.sha256(
            json.dumps(
                {"context": context_fingerprint, "models": model_fingerprints}, sort_keys=True
            ).encode()
        ).hexdigest(),
    )
    cached = _load_syntheval_cache(cache_dir, "binary_target", cache_key)
    if cached is not None:
        return cached

    if execution_cfg is None:
        from synthdata.config import SynthEvalExecutionConfig

        execution_cfg = SynthEvalExecutionConfig()
    logger.info(
        "[syntheval] binary-target pass: scheduling %d datasets across %d metric(s) "
        "(column %r collapsed to binary: positive=%s, negative=%s)",
        len(binary_synthetic_datasets),
        len(preset),
        column,
        positive_classes,
        negative_classes,
    )
    benchmark_results, benchmark_ranks = _run_resumable_syntheval(
        binary_synthetic_datasets,
        binary_dataset,
        preset,
        preset_path,
        cache_dir,
        ranking_strategy,
        execution_cfg,
        "binary_target",
    )
    _save_syntheval_cache(benchmark_results, benchmark_ranks, cache_dir, "binary_target", cache_key)

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
