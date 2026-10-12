"""Figures describing the raw/loaded dataset: column distributions and missingness."""

from pathlib import Path

import numpy as np
import pandas as pd

from synthdata.data import Dataset
from synthdata.plotting import (
    add_histogram_with_kde,
    column_bar_chart,
    ordered_categories,
    save_matplotlib_figure,
    set_category_ticks,
    wrapped_title,
)


def plot_column_distributions(
    df: pd.DataFrame,
    columns: list,
    categorical_columns: list,
    ncols: int = 5,
    category_orders: dict | None = None,
):
    """Grid of categorical bars and continuous density histograms with KDEs."""
    import matplotlib.pyplot as plt

    n = len(columns)
    ncols = min(ncols, n) or 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3 * nrows))
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, col in zip(axes_flat, columns, strict=False):
        series = df[col].dropna()
        if col in categorical_columns:
            counts = series.value_counts()
            categories = ordered_categories(
                set(counts.index), category_orders.get(col) if category_orders else None
            )
            counts = counts.reindex(categories, fill_value=0)
            positions = np.arange(len(categories))
            ax.bar(positions, counts.values, zorder=3)
            set_category_ticks(ax, positions, categories)
        else:
            add_histogram_with_kde(ax, series, bins=20, label="data")
            ax.set_ylabel("density")
        ax.set_title(wrapped_title(col), fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4, zorder=0)

    for ax in axes_flat[len(columns) :]:
        ax.axis("off")

    fig.tight_layout()
    return fig


def plot_missingness(df: pd.DataFrame, feature_columns: list):
    """Bar chart of missing-value fractions per feature column."""
    import matplotlib.pyplot as plt

    missing = df[feature_columns].isna().mean()
    missing = missing[missing > 0].sort_values(ascending=False, kind="stable")

    if not len(missing):
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig
    fig, _ = column_bar_chart(
        missing.index.astype(str),
        missing.values,
        xlabel="Fraction missing",
        title="Missing values per column",
    )
    return fig


def save_data_plots(
    dataset: Dataset, output_dir: str | Path, dpi: int = 150, formats=("png",)
) -> None:
    output_dir = Path(output_dir)
    full_df = dataset.decode_ordinal_frame(dataset.full_df)
    fig1 = plot_column_distributions(
        full_df,
        dataset.feature_columns,
        dataset.categorical_columns,
        category_orders=dataset.ordinal_category_orders,
    )
    save_matplotlib_figure(fig1, output_dir / "data" / "column_distributions", dpi, formats)

    fig2 = plot_missingness(dataset.full_df, dataset.feature_columns)
    save_matplotlib_figure(fig2, output_dir / "data" / "missingness", dpi, formats)

    import matplotlib.pyplot as plt

    plt.close("all")
