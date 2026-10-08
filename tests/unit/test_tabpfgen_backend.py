"""Unit tests for synthdata.generation.tabpfgen_backend's custom labelling."""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def test_sgld_labels_come_from_the_nearest_train_row():
    pytest.importorskip("tabpfgen")
    from synthdata.generation.tabpfgen_backend import TabPFGenSGLDLabels

    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    y = np.repeat([0, 1, 2], 20)
    x = centers[y] + rng.normal(scale=0.5, size=(60, 3))

    np.random.seed(0)
    x_syn, y_syn = TabPFGenSGLDLabels(n_sgld_steps=0, device="cpu").generate_classification(
        x, y, n_samples=30
    )
    # With no SGLD steps each row stays next to the train row it started from.
    brute = y[np.argmin(((x_syn[:, None, :] - x[None, :, :]) ** 2).sum(-1), axis=1)]
    np.testing.assert_array_equal(y_syn, brute)
    assert sorted(set(y_syn)) == [0, 1, 2]
