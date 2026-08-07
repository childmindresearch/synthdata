"""Unit tests for imputation plots."""

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from synthdata.plotting.imputation_plots import plot_observed_vs_imputed


def test_observed_vs_imputed_handles_mixed_category_types():
    full_df = pd.DataFrame({"category": ["alpha", None, "beta"]})
    full_imputed_df = pd.DataFrame({"category": ["alpha", 1, "beta"]})

    fig = plot_observed_vs_imputed(
        full_df,
        full_imputed_df,
        columns_with_missing=["category"],
        categorical_columns=["category"],
    )

    labels = {label.get_text() for label in fig.axes[0].get_xticklabels()}
    assert "1" in labels

    plt.close(fig)


def test_observed_vs_imputed_preserves_configured_ordinal_order():
    full_df = pd.DataFrame({"activity": ["Low", None, "High"]})
    full_imputed_df = pd.DataFrame({"activity": ["Low", "Medium", "High"]})

    fig = plot_observed_vs_imputed(
        full_df,
        full_imputed_df,
        columns_with_missing=["activity"],
        categorical_columns=["activity"],
        category_orders={"activity": ["Low", "Medium", "High"]},
    )

    labels = [label.get_text() for label in fig.axes[0].get_xticklabels()]
    assert labels == ["Low", "Medium", "High"]

    plt.close(fig)
