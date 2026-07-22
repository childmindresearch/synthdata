"""Figures validating imputation quality: observed-vs-imputed distributions."""

from pathlib import Path

import numpy as np
import pandas as pd

from synthdata.config import Config
from synthdata.data import Dataset
from synthdata.plotting import save_matplotlib_figure


def plot_observed_vs_imputed(
    full_df: pd.DataFrame,
    full_imputed_df: pd.DataFrame,
    columns_with_missing: list,
    categorical_columns: list,
    ncols: int = 4,
):
    """For each column with missing values: observed vs. imputed distribution."""
    import matplotlib.pyplot as plt

    n = len(columns_with_missing)
    if n == 0:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "No missing values to validate", ha="center", va="center")
        ax.axis("off")
        return fig

    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3 * nrows))
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, col in zip(axes_flat, columns_with_missing, strict=False):
        missing_mask = full_df[col].isna()
        observed = full_df.loc[~missing_mask, col]
        imputed = full_imputed_df.loc[missing_mask, col]

        if col in categorical_columns:
            obs_counts = observed.value_counts(normalize=True).sort_index()
            imp_counts = imputed.value_counts(normalize=True).sort_index()
            categories = sorted(set(obs_counts.index) | set(imp_counts.index))
            positions = np.arange(len(categories))
            width = 0.4
            ax.bar(
                positions - width / 2,
                [obs_counts.get(c, 0) for c in categories],
                width,
                label="observed",
            )
            ax.bar(
                positions + width / 2,
                [imp_counts.get(c, 0) for c in categories],
                width,
                label="imputed",
            )
            ax.set_xticks(positions)
            ax.set_xticklabels(categories)
        else:
            ax.hist(observed, bins=15, density=True, alpha=0.6, label="observed")
            ax.hist(imputed, bins=15, density=True, alpha=0.6, label="imputed")
        ax.set_title(col, fontsize=9)
        ax.legend(fontsize=7)

    for ax in axes_flat[len(columns_with_missing) :]:
        ax.axis("off")

    fig.suptitle("Observed vs. imputed distributions", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_validation_summary(validation_df: pd.DataFrame):
    """Bar chart of per-column imputation validation pass rates."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(validation_df)), 4))
    if len(validation_df):
        pass_rate = validation_df["n_valid"] / validation_df["n_imputed"].replace(0, np.nan)
        colors = ["#2a9d8f" if ok else "#e76f51" for ok in validation_df["all_valid"]]
        ax.bar(validation_df["column"], pass_rate.fillna(1.0), color=colors, zorder=3)
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of imputed values within plausible range")
    ax.set_title("Imputation validation summary")
    ax.grid(True, linestyle="--", alpha=0.4, axis="y", zorder=0)
    fig.tight_layout()
    return fig


def save_imputation_plots(
    cfg: Config, dataset: Dataset, validation_df: pd.DataFrame, output_dir: str | Path
) -> None:
    output_dir = Path(output_dir)
    columns_with_missing = [c for c in dataset.feature_columns if dataset.full_df[c].isna().any()]
    fig1 = plot_observed_vs_imputed(
        dataset.full_df, dataset.full_imputed_df, columns_with_missing, dataset.categorical_columns
    )
    save_matplotlib_figure(
        fig1, output_dir / "imputation" / "observed_vs_imputed", cfg.plots.dpi, cfg.plots.formats
    )

    fig2 = plot_validation_summary(validation_df)
    save_matplotlib_figure(
        fig2, output_dir / "imputation" / "validation_summary", cfg.plots.dpi, cfg.plots.formats
    )

    import matplotlib.pyplot as plt

    plt.close("all")
