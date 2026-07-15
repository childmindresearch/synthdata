"""Figures describing the raw/loaded dataset: column distributions and missingness."""

from pathlib import Path

import numpy as np
import pandas as pd

from synthdata.data import Dataset
from synthdata.plotting import save_matplotlib_figure


def plot_column_distributions(
    df: pd.DataFrame,
    columns: list,
    categorical_columns: list,
    ncols: int = 5,
):
    """Grid of bar charts (categorical) / histograms (continuous) for each column."""
    import matplotlib.pyplot as plt

    n = len(columns)
    ncols = min(ncols, n) or 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3 * nrows))
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, col in zip(axes_flat, columns):
        series = df[col].dropna()
        if col in categorical_columns:
            counts = series.value_counts().sort_index()
            ax.bar(counts.index.astype(str), counts.values, zorder=3)
        else:
            ax.hist(series, bins=20, zorder=3)
        ax.set_title(col, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4, zorder=0)

    for ax in axes_flat[len(columns):]:
        ax.axis("off")

    fig.suptitle("Column distributions", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_missingness(df: pd.DataFrame, feature_columns: list):
    """Bar chart of missing-value counts per feature column."""
    import matplotlib.pyplot as plt

    missing = df[feature_columns].isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(missing)), 4))
    if len(missing):
        ax.bar(missing.index.astype(str), missing.values, zorder=3)
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel("# missing")
    ax.set_title("Missing values per column")
    ax.grid(True, linestyle="--", alpha=0.4, axis="y", zorder=0)
    fig.tight_layout()
    return fig


def save_data_plots(dataset: Dataset, output_dir: str | Path, dpi: int = 150, formats=("png",)) -> None:
    output_dir = Path(output_dir)
    fig1 = plot_column_distributions(
        dataset.full_df, dataset.feature_columns, dataset.categorical_columns
    )
    save_matplotlib_figure(fig1, output_dir / "data" / "column_distributions", dpi, formats)

    fig2 = plot_missingness(dataset.full_df, dataset.feature_columns)
    save_matplotlib_figure(fig2, output_dir / "data" / "missingness", dpi, formats)

    import matplotlib.pyplot as plt

    plt.close("all")
