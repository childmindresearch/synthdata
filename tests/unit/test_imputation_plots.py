"""Unit tests for imputation plots."""

from types import SimpleNamespace

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from synthdata.plotting import imputation_plots
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


def test_canonical_save_plots_excludes_final_holdout_missingness(
    make_canonical_dataset, monkeypatch, tmp_path
):
    dataset = make_canonical_dataset()
    holdout = dataset.roles["final_holdout"].copy()
    holdout.loc[holdout.index[0], "feature"] = None
    dataset.roles["final_holdout"] = holdout
    dataset.set_imputed_roles(
        {role: role_frame.copy() for role, role_frame in dataset.roles.items()}
    )

    captured = {}

    def capture_plot(raw, imputed, columns, categorical, **kwargs):
        captured["raw"] = raw
        captured["imputed"] = imputed
        captured["columns"] = columns
        return plt.figure()

    monkeypatch.setattr(imputation_plots, "plot_observed_vs_imputed", capture_plot)
    monkeypatch.setattr(imputation_plots, "save_matplotlib_figure", lambda *args: None)
    cfg = SimpleNamespace(plots=SimpleNamespace(dpi=100, formats=["png"]))

    imputation_plots.save_imputation_plots(cfg, dataset, pd.DataFrame(), tmp_path)

    assert captured["columns"] == []
    assert len(captured["raw"]) == len(dataset.roles["train"]) + len(dataset.roles["tuning"])
    assert len(captured["imputed"]) == len(captured["raw"])


def test_legacy_save_plots_uses_full_imputed_frame(make_dataset, monkeypatch, tmp_path):
    dataset = make_dataset()
    dataset.full_imputed_df = dataset.full_df.copy()

    captured = {}

    def capture_plot(raw, imputed, columns, categorical, **kwargs):
        captured["raw"] = raw
        captured["imputed"] = imputed
        return plt.figure()

    monkeypatch.setattr(imputation_plots, "plot_observed_vs_imputed", capture_plot)
    monkeypatch.setattr(imputation_plots, "save_matplotlib_figure", lambda *args: None)
    cfg = SimpleNamespace(plots=SimpleNamespace(dpi=100, formats=["png"]))

    imputation_plots.save_imputation_plots(cfg, dataset, pd.DataFrame(), tmp_path)

    assert len(captured["raw"]) == len(dataset.full_df)
    assert len(captured["imputed"]) == len(dataset.full_df)
