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
