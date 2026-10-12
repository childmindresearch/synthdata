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
from synthdata.plotting import (
    add_histogram_with_kde,
    ordered_categories,
    save_matplotlib_figure,
    save_plotly_figure,
    set_category_ticks,
    wrapped_title,
)
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
    category_orders: dict | None = None,
):
    """Per-column comparison with categorical bars and continuous KDEs."""
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
            categories = ordered_categories(
                all_categories, category_orders.get(col) if category_orders else None
            )
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
            set_category_ticks(ax, positions, categories)
            ax.set_ylabel("frequency")
        else:
            add_histogram_with_kde(ax, real_df[col], bins=20, label="real", alpha=0.4, color="C0")
            add_histogram_with_kde(
                ax, synth_df[col], bins=20, label="synthetic", alpha=0.4, color="C1"
            )
            ax.set_ylabel("density")
        ax.set_title(wrapped_title(col), fontsize=9)
        ax.legend(fontsize=7)

    for ax in axes_flat[len(feature_columns) :]:
        ax.axis("off")

    fig.tight_layout()
    return fig


#: Columns in each model's summary figure (the ones that differ most).
TOP_COLUMNS = 12


def column_differences(
    real_df: pd.DataFrame, synth_df: pd.DataFrame, columns: list, categorical_columns: list
) -> pd.Series:
    """How far each column's synthetic distribution is from the real one, in [0, 1].

    Categorical (or text) columns: total variation distance between the
    category shares.
    Numeric columns outside ``categorical_columns``: the two-sample Kolmogorov-Smirnov statistic
    (``scipy.stats.ks_2samp``). 0 is identical, 1 disjoint; a column missing
    from the synthetic data scores 1.
    """
    from scipy.stats import ks_2samp

    out = {}
    for col in columns:
        if col not in real_df.columns:
            continue
        real, synth = real_df[col].dropna(), synth_df[col].dropna() if col in synth_df else None
        if synth is None or synth.empty or real.empty:
            out[col] = 1.0
        elif col in categorical_columns or not pd.api.types.is_numeric_dtype(real):
            a, b = _category_value_counts(real), _category_value_counts(synth)
            keys = a.index.union(b.index)
            out[col] = float(
                (a.reindex(keys, fill_value=0) - b.reindex(keys, fill_value=0)).abs().sum() / 2
            )
        else:
            out[col] = float(
                ks_2samp(
                    pd.to_numeric(real, errors="coerce").dropna(),
                    pd.to_numeric(synth, errors="coerce").dropna(),
                ).statistic
            )
    return pd.Series(out, name="difference", dtype=float)


def save_model_comparison(
    cfg: Config, dataset: Dataset, name: str, synth_df: pd.DataFrame, output_dir: str | Path
) -> None:
    """One model's real-vs-synthetic figures and its column differences.

    Writes ``<name>.png`` (every column), ``<name>_top_columns.png`` (the
    TOP_COLUMNS columns that differ most) and the model's column of
    ``column_differences.csv``, all under ``output_dir/generation``. The real
    reference is the imputed train split (fully observed, so every column is
    comparable), falling back to the raw train split.
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir) / "generation"
    real_model_df = (
        dataset.train_imputed_df if dataset.train_imputed_df is not None else dataset.train_df
    )
    real_df = dataset.decode_ordinal_frame(real_model_df)
    synth_df = dataset.decode_ordinal_frame(synth_df)
    columns = dataset.feature_columns + [dataset.target_column]
    categorical = dataset.all_categorical_columns
    differences = column_differences(real_df, synth_df, columns, categorical)
    _record_differences(output_dir / "column_differences.csv", name, differences)
    top = differences.sort_values(ascending=False, kind="stable").index[:TOP_COLUMNS].tolist()
    for suffix, cols in (("", columns), ("_top_columns", top)):
        fig = plot_real_vs_synthetic(
            real_df, synth_df, cols, categorical, category_orders=dataset.ordinal_category_orders
        )
        try:
            save_matplotlib_figure(
                fig, output_dir / f"{name}{suffix}", cfg.plots.dpi, cfg.plots.formats
            )
        finally:
            plt.close(fig)


def _record_differences(path: Path, name: str, differences: pd.Series) -> None:
    """Put one model's column differences into the columns x models table."""
    from filelock import FileLock

    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(f"{path}.lock"):
        table = pd.read_csv(path, index_col=0) if path.exists() else pd.DataFrame()
        table = table.reindex(table.index.union(differences.index, sort=False))
        table[name] = differences
        table.index.name = "column"
        table.to_csv(path)


def save_generation_plots(
    cfg: Config,
    dataset: Dataset,
    synthetic_datasets: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> None:
    """Save each model's real-vs-synthetic figures (see :func:`save_model_comparison`)."""
    import matplotlib.pyplot as plt

    if dataset.train_imputed_df is None:
        raise RuntimeError("save_generation_plots() requires imputed training data")
    for name, synth_df in synthetic_datasets.items():
        try:
            save_model_comparison(cfg, dataset, name, synth_df, output_dir)
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
