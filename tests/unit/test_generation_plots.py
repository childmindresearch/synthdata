"""Unit tests for real-vs-synthetic plots."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from synthdata.plotting import wrapped_title
from synthdata.plotting.generation_plots import column_differences, plot_real_vs_synthetic

pytestmark = pytest.mark.unit

LONG_NAMES = [
    f"Questionnaire section {i} - an item described by a long sentence of text" for i in range(4)
]


def test_long_column_names_do_not_squeeze_the_axes():
    # tight_layout keeps titles inside their grid cell, so an unwrapped title
    # wider than the axes used to shrink each plot to a sliver.
    rng = np.random.default_rng(0)
    real = pd.DataFrame({c: rng.integers(0, 7, 200).astype(float) for c in LONG_NAMES})
    synth = real.sample(frac=1, random_state=1).reset_index(drop=True)

    fig = plot_real_vs_synthetic(real, synth, LONG_NAMES, LONG_NAMES)

    cell = 1 / 4
    for ax in fig.axes:
        assert ax.get_position().width > 0.6 * cell
    plt.close(fig)


def test_wrapped_title_keeps_short_names_and_caps_long_ones():
    assert wrapped_title("age") == "age"
    lines = wrapped_title("word " * 40).split("\n")
    assert len(lines) == 3 and lines[-1].endswith("...")


def test_long_category_labels_are_rotated_and_many_are_thinned():
    rng = np.random.default_rng(0)
    answers = [f"Very long free text answer number {i} about something" for i in range(5)]
    real = pd.DataFrame(
        {
            "a": rng.integers(0, 7, 200).astype(float),
            "b": rng.integers(0, 7, 200).astype(float),
            "answer": rng.choice(answers, 200),
            "site": rng.integers(0, 80, 200).astype(float),
        }
    )
    fig = plot_real_vs_synthetic(real, real, list(real.columns), list(real.columns))

    for ax in fig.axes:
        assert ax.get_position().width > 0.6 / 4
    answer_ax, site_ax = fig.axes[2], fig.axes[3]
    assert all(label.get_rotation() == 45 for label in answer_ax.get_xticklabels())
    assert len(site_ax.get_xticks()) <= 20
    plt.close(fig)


def test_column_bar_charts_have_no_empty_band_with_many_columns():
    from synthdata.plotting.data_plots import plot_missingness
    from synthdata.plotting.imputation_plots import plot_validation_summary

    names = [f"simulated_feature_{i:03d}_with_a_long_descriptive_name" for i in range(120)]
    df = pd.DataFrame({name: [1.0, None, 2.0] for name in names})
    validation = pd.DataFrame({"column": names, "n_imputed": 3, "n_valid": 3, "all_valid": True})
    # Every column is equally missing: ties keep the schema order.
    for fig in (plot_missingness(df, names), plot_validation_summary(validation)):
        ax = fig.axes[0]
        low, high = sorted(ax.get_ylim())
        assert (low, high) == (-0.5, len(names) - 0.5)
        assert ax.get_yticklabels()[0].get_text().startswith("simulated_feature_000")
        plt.close(fig)


def test_column_differences_are_zero_for_a_copy_and_one_for_disjoint_data():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"cat": rng.integers(0, 3, 300), "num": rng.normal(size=300)})
    same = column_differences(real, real, ["cat", "num"], ["cat"])
    assert same.tolist() == [0.0, 0.0]
    shifted = pd.DataFrame({"cat": real["cat"] + 10, "num": real["num"] + 100})
    apart = column_differences(real, shifted, ["cat", "num", "gone"], ["cat"])
    assert apart.tolist() == [1.0, 1.0]  # "gone" is in neither frame: skipped
    missing = column_differences(real, real[["num"]], ["cat"], ["cat"])
    assert missing["cat"] == 1.0


def test_model_comparison_writes_both_figures_and_one_table_column_per_model(
    make_config, make_dataset
):
    from pathlib import Path

    from synthdata.plotting.generation_plots import TOP_COLUMNS, save_model_comparison

    cfg = make_config()
    dataset = make_dataset()
    real = dataset.train_df
    for name, df in (("copy", real), ("shuffled", real.sample(frac=1, random_state=0))):
        save_model_comparison(cfg, dataset, name, df, cfg.plots.output_dir)
    generation = Path(cfg.plots.output_dir) / "generation"
    for name in ("copy", "shuffled"):
        assert (generation / f"{name}.png").exists()
        assert (generation / f"{name}_top_columns.png").exists()
    table = pd.read_csv(generation / "column_differences.csv", index_col=0)
    assert list(table.columns) == ["copy", "shuffled"]
    assert len(table) == len(dataset.feature_columns) + 1 > 0
    assert (table["copy"] == 0).all()
    assert TOP_COLUMNS == 12


def test_numeric_labels_sort_by_value_even_when_stored_as_text():
    from synthdata.plotting import ordered_categories

    labels = {"1.0", "10.0", "2.0", "11.0", "Unknown"}
    assert ordered_categories(labels) == ["1.0", "2.0", "10.0", "11.0", "Unknown"]
    assert ordered_categories({1.0, 10.0, "2"}) == [1.0, "2", 10.0]
    # A configured order still wins.
    assert ordered_categories({"1", "10", "2"}, ["10", "2", "1"]) == ["10", "2", "1"]


def test_comparison_uses_observed_real_and_released_synthetic(make_config, make_dataset):
    from pathlib import Path

    from synthdata.plotting.generation_plots import released_views, save_model_comparison

    rng = np.random.default_rng(0)
    n = 200
    raw = pd.DataFrame(
        {"a": rng.normal(size=n), "b": rng.normal(size=n), "target": rng.integers(0, 2, n)}
    )
    raw.loc[::2, "b"] = np.nan
    raw["b__missing"] = raw["b"].isna().astype(int)
    cfg = make_config()
    dataset = make_dataset(df=raw, feature_columns=["a", "b", "b__missing"])
    dataset.missing_indicator_columns = {"b__missing": "b"}
    # Imputation and generation filled b with a spike at 100.
    dataset.train_imputed_df = dataset.train_df.fillna({"b": 100.0})
    synthetic = dataset.train_imputed_df.reset_index(drop=True)

    real, released, columns = released_views(cfg, dataset, synthetic)
    assert columns == ["a", "b", "target"]
    assert (released["b"] == 100).sum() == 0  # the imputed spike is blanked again
    assert released["b"].isna().mean() == pytest.approx(real["b"].isna().mean())

    save_model_comparison(cfg, dataset, "model", synthetic, cfg.plots.output_dir)
    generation = Path(cfg.plots.output_dir) / "generation"
    assert (generation / "model_filled.png").exists()
    table = pd.read_csv(generation / "column_differences.csv", index_col=0)
    assert table.index.tolist() == ["a", "b", "target"]
    assert (table["model"] == 0).all()
