"""Unit tests for real-vs-synthetic plots."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from synthdata.plotting import wrapped_title
from synthdata.plotting.generation_plots import plot_real_vs_synthetic

pytestmark = pytest.mark.unit

LONG_NAMES = [
    "Life Events and History - Serious accident or physical impairment",
    "Life Events and History - Death of a close family member or friend",
    "Life Events and History - Conflict-laden or severed relationship",
    "Life Events and History - Immigration, including refugee status",
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
