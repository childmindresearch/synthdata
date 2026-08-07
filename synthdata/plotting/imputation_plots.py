"""Figures validating imputation quality: observed-vs-imputed distributions."""

from pathlib import Path

import numpy as np
import pandas as pd

from synthdata.config import Config
from synthdata.data import Dataset
from synthdata.plotting import add_histogram_with_kde, ordered_categories, save_matplotlib_figure


def plot_observed_vs_imputed(
    full_df: pd.DataFrame,
    full_imputed_df: pd.DataFrame,
    columns_with_missing: list,
    categorical_columns: list,
    ncols: int = 4,
    category_orders: dict | None = None,
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
        n_imputed = int(missing_mask.sum())

        if col in categorical_columns:
            obs_counts = observed.value_counts(normalize=True)
            imp_counts = imputed.value_counts(normalize=True)
            all_categories = set(obs_counts.index) | set(imp_counts.index)
            categories = ordered_categories(
                all_categories, category_orders.get(col) if category_orders else None
            )
            obs_frequencies = obs_counts.to_dict()
            imp_frequencies = imp_counts.to_dict()
            positions = np.arange(len(categories))
            width = 0.4
            ax.bar(
                positions - width / 2,
                [obs_frequencies.get(c, 0) for c in categories],
                width,
                label="observed",
            )
            ax.bar(
                positions + width / 2,
                [imp_frequencies.get(c, 0) for c in categories],
                width,
                label="imputed",
            )
            ax.set_xticks(positions)
            ax.set_xticklabels(categories)
        else:
            add_histogram_with_kde(ax, observed, bins=15, label="observed", alpha=0.45, color="C0")
            add_histogram_with_kde(ax, imputed, bins=15, label="imputed", alpha=0.45, color="C1")
        ax.set_title(f"{col} (n_imputed = {n_imputed})", fontsize=9)
        ax.legend(fontsize=7)

    for ax in axes_flat[len(columns_with_missing) :]:
        ax.axis("off")

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
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of imputed values\nwithin plausible range")
    ax.set_title("Imputation validation summary")
    ax.grid(True, linestyle="--", alpha=0.4, axis="y", zorder=0)
    return fig


def save_imputation_plots(
    cfg: Config, dataset: Dataset, validation_df: pd.DataFrame, output_dir: str | Path
) -> None:
    output_dir = Path(output_dir)
    full_imputed_model_df = dataset.full_imputed_df
    if full_imputed_model_df is None:
        raise RuntimeError("save_imputation_plots() requires imputed data")
    full_df = dataset.decode_ordinal_frame(dataset.full_df)
    full_imputed_df = dataset.full_imputed_decoded_df
    if full_imputed_df is None:
        full_imputed_df = dataset.decode_ordinal_frame(full_imputed_model_df)
    columns_with_missing = [c for c in dataset.feature_columns if full_df[c].isna().any()]
    fig1 = plot_observed_vs_imputed(
        full_df,
        full_imputed_df,
        columns_with_missing,
        dataset.categorical_columns,
        category_orders=dataset.ordinal_category_orders,
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
