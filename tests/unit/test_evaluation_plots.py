"""Unit tests for synthdata.plotting.evaluation_plots."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from synthdata.plotting.evaluation_plots import _base_model, _is_hpo, plot_rank_tradeoff

pytestmark = pytest.mark.unit

UTILITY = ("__all__", "utility", "rank")
PRIVACY = ("__all__", "privacy", "rank")


def test_replicates_share_their_generator_and_hpo_flag():
    assert _base_model("ctgan_hpo__rep2") == "ctgan"
    assert _base_model("ctgan__rep1") == "ctgan"
    assert _is_hpo("ctgan_hpo__rep2")
    assert not _is_hpo("ctgan__rep1")


def test_more_generators_than_palette_colors_still_plot():
    models = [f"model{i}" for i in range(25)] + [f"model{i}_hpo__rep1" for i in range(25)]
    combined = pd.DataFrame(
        {UTILITY: range(len(models)), PRIVACY: range(len(models))}, index=models
    )

    figure = plot_rank_tradeoff(combined, UTILITY, PRIVACY, "Utility", "Privacy", "t")

    assert len(figure.axes[0].collections) == len(models)
    plt.close(figure)


def test_metric_overview_folds_seeds_and_marks_directions():
    from synthdata.plotting.evaluation_plots import metric_overview_table, plot_metric_overview

    index = ["ctgan", "ctgan__rep1", "baseline_train_copy", "baseline_marginals"]
    combined = pd.DataFrame(
        {
            ("custom", "utility", "tstr_macro_f1"): [0.4, 0.6, 0.9, 0.3],
            ("custom", "privacy", "distance_mia_auc"): [0.55, 0.6, 0.95, 0.5],
            ("__all__", "utility", "rank"): [0.5, 0.6, 1.0, 0.0],
        },
        index=index,
    )
    table = metric_overview_table(combined, ("custom", "utility", "tstr_macro_f1"))
    assert list(table.index) == ["ctgan", "baseline_train_copy", "baseline_marginals"]
    assert table.loc["ctgan", "mean"] == pytest.approx(0.5)
    assert table.loc["ctgan", "ci_low"] < 0.5 < table.loc["ctgan", "ci_high"]
    directions = pd.DataFrame(
        {"direction": ["higher", "lower"], "ideal": [float("nan"), 0.5]},
        index=pd.MultiIndex.from_tuples(
            [("custom", "utility", "tstr_macro_f1"), ("custom", "privacy", "distance_mia_auc")]
        ),
    )
    fig = plot_metric_overview(combined, directions, "privacy")
    try:
        (ax,) = [a for a in fig.axes if a.get_visible()]
        assert "lower is better" in ax.get_title()
        assert any(line.get_xdata()[0] == 0.5 for line in ax.get_lines())  # the ideal line
    finally:
        plt.close(fig)
    assert plot_metric_overview(combined, directions, "fairness") is None
