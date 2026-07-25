"""Figures produced during synthetic data generation: real-vs-synthetic per-column
comparisons and Optuna HPO diagnostic plots (optimization history, parameter
importances, slice plots).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from synthdata.config import Config
from synthdata.data import Dataset
from synthdata.generation.hpo import default_storage_url
from synthdata.plotting import save_matplotlib_figure, save_plotly_figure
from synthdata.utils import get_logger

logger = get_logger(__name__)


def _category_value_counts(series: pd.Series) -> pd.Series:
    """Frequency table for one categorical column's observed values.

    Most nominal/ordinal columns are integer-coded (round+cast to int to collapse
    float noise like 1.0000001 before counting), but some nominal columns are
    genuinely text-valued (e.g. ``Pegboard__peg_dom_hand``: 'Left'/'Right'/
    'Ambidexterous' -- unlike most categorical columns, no upstream step maps
    it to numeric codes). Rounding/casting those raises
    ``TypeError: Expected numeric dtype, got object instead.`` from
    ``Series.round()``, so only do it for numeric-dtype columns.
    """
    values = series.dropna()
    if pd.api.types.is_numeric_dtype(values):
        values = values.round().astype(int)
    return values.value_counts(normalize=True)


def plot_real_vs_synthetic(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    feature_columns: list,
    categorical_columns: list,
    ncols: int = 4,
):
    """Per-column comparison: bar charts for categorical columns, histograms otherwise."""
    import matplotlib.pyplot as plt

    n = len(feature_columns)
    ncols = min(ncols, n) or 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, col in zip(axes_flat, feature_columns, strict=False):
        if col not in real_df.columns or col not in synth_df.columns:
            ax.axis("off")
            continue
        if col in categorical_columns:
            real_freq = _category_value_counts(real_df[col])
            synth_freq = _category_value_counts(synth_df[col])
            all_categories = set(real_freq.index) | set(synth_freq.index)
            try:
                categories = sorted(all_categories)
            except TypeError:
                # Mixed types (e.g. real is text-valued but synthesis produced a
                # stray numeric code) -- fall back to a stable, comparable order.
                categories = sorted(all_categories, key=str)
            positions = np.arange(len(categories))
            width = 0.4
            ax.bar(
                positions - width / 2,
                [real_freq.get(c, 0) for c in categories],
                width,
                label="real",
            )
            ax.bar(
                positions + width / 2,
                [synth_freq.get(c, 0) for c in categories],
                width,
                label="synthetic",
            )
            ax.set_xticks(positions)
            ax.set_xticklabels(categories)
            ax.set_ylabel("frequency")
        else:
            ax.hist(real_df[col].dropna(), bins=20, density=True, alpha=0.5, label="real")
            ax.hist(synth_df[col].dropna(), bins=20, density=True, alpha=0.5, label="synthetic")
            ax.set_ylabel("density")
        ax.set_title(col, fontsize=9)
        ax.legend(fontsize=7)

    for ax in axes_flat[len(feature_columns) :]:
        ax.axis("off")

    fig.tight_layout()
    return fig


def save_generation_plots(
    cfg: Config,
    dataset: Dataset,
    synthetic_datasets: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> None:
    """Save one real-vs-synthetic figure per generated model.

    All comparisons use the imputed train split as the "real" reference (even
    for TabPFN variants fit on the pre-imputation split), since it is fully
    observed and comparable across every column.
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir) / "generation"
    real_df = dataset.train_imputed_df
    all_columns = dataset.feature_columns + [dataset.target_column]
    categorical = dataset.all_categorical_columns

    for name, synth_df in synthetic_datasets.items():
        try:
            fig = plot_real_vs_synthetic(real_df, synth_df, all_columns, categorical)
            save_matplotlib_figure(fig, output_dir / name, cfg.plots.dpi, cfg.plots.formats)
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            logger.warning("real-vs-synthetic plot failed for %s: %s", name, exc)
        finally:
            plt.close("all")


# ---------------------------------------------------------------------------
# Optuna HPO plots
# ---------------------------------------------------------------------------


def _load_study(study_name: str, storage: str):
    import optuna

    try:
        return optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        return None


def save_hpo_plots(cfg: Config, output_dir: str | Path) -> None:
    """Save Optuna optimization-history/param-importance/slice plots per HPO'd model."""
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_slice,
    )

    gen_cfg = cfg.generation
    if not gen_cfg.hpo.enabled:
        logger.info("HPO disabled; skipping HPO plots")
        return

    storage = gen_cfg.hpo.storage or default_storage_url(gen_cfg.output_dir)
    output_dir = Path(output_dir) / "hpo"

    study_names = []
    if gen_cfg.synthcity.enabled:
        study_names += [f"hpo_{name}" for name in gen_cfg.synthcity.names]
    if gen_cfg.tabpfgen.enabled:
        if "standard" in gen_cfg.tabpfgen.variants:
            study_names.append("hpo_tabpfgen_standard")
        if "custom" in gen_cfg.tabpfgen.variants:
            study_names.append("hpo_tabpfgen_custom")

    for study_name in study_names:
        study = _load_study(study_name, storage)
        if study is None or not study.trials:
            continue
        try:
            save_plotly_figure(
                plot_optimization_history(study), output_dir / f"{study_name}_history", ("html",)
            )
            save_plotly_figure(
                plot_param_importances(study),
                output_dir / f"{study_name}_param_importances",
                ("html",),
            )
            save_plotly_figure(plot_slice(study), output_dir / f"{study_name}_slice", ("html",))
        except (ValueError, RuntimeError, OSError) as exc:
            logger.warning("HPO plots failed for %s: %s", study_name, exc)
