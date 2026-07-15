"""SynthEval-based evaluation: runs SynthEval's benchmark() across all cached
synthetic datasets using a custom preset (built from
:mod:`synthdata.evaluation.catalog`, filtered by the configured selection).
"""

from pathlib import Path

import pandas as pd

from synthdata.data import Dataset
from synthdata.evaluation.catalog import (
    SYNTHEVAL_METRIC_TYPE,
    SYNTHEVAL_PRESET,
    resolve_selection,
)
from synthdata.utils import ensure_dir, get_logger, save_json

logger = get_logger(__name__)

_RANK_COLUMNS = {"rank", "u_rank", "p_rank", "f_rank"}


def build_preset(selection_cfg) -> dict:
    """Filter the full SynthEval preset down to the configured selection."""
    all_names = list(SYNTHEVAL_PRESET.keys())
    selected = resolve_selection(
        selection_cfg.enabled,
        selection_cfg.categories,
        selection_cfg.metrics,
        all_names,
        SYNTHEVAL_METRIC_TYPE,
    )
    return {k: v for k, v in SYNTHEVAL_PRESET.items() if k in selected}


def run_syntheval_evaluation(
    synthetic_datasets: dict[str, pd.DataFrame],
    dataset: Dataset,
    selection_cfg,
    preset_dir: str | Path,
    ranking_strategy: str = "linear",
    output_folder: str | Path | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Run SynthEval's benchmark() across all datasets. Returns (benchmark_results, benchmark_ranks).

    Both are None if the selection resolves to zero metrics.
    """
    preset = build_preset(selection_cfg)
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
        enable_plots=False,
        console="off",
        show_warnings=False,
    )

    logger.info(
        "[syntheval] benchmarking %d datasets across %d metrics",
        len(synthetic_datasets),
        len(preset),
    )
    benchmark_results, benchmark_ranks = se.benchmark(
        synthetic_datasets,
        analysis_target=analysis_config,
        presets_file=str(preset_path),
        rank_strategy=ranking_strategy,
        output_folder=str(output_folder) if output_folder else None,
    )
    return benchmark_results, benchmark_ranks


def extract_raw_values(benchmark_results: pd.DataFrame) -> pd.DataFrame:
    """Models x metrics table of raw metric values from SynthEval's benchmark_results."""
    return benchmark_results.xs("value", axis=1, level=1)


def extract_oriented_values(benchmark_ranks: pd.DataFrame) -> pd.DataFrame:
    """Models x metrics table of SynthEval's pre-oriented (higher=better) n_val scores."""
    cols = [c for c in benchmark_ranks.columns if c not in _RANK_COLUMNS]
    return benchmark_ranks[cols]
