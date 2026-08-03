import numpy as np
import pandas as pd
import pytest
from syntheval.utils.plot_metrics import plot_significantly_dissimilar_variables

pytestmark = pytest.mark.unit


def test_continuous_histogram_bounds_near_tied_float_bins(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    real = pd.DataFrame({"BILIRUBIN": np.concatenate(([0.0], 0.5 + np.arange(98) * 1e-12, [1.0]))})
    fake = pd.DataFrame(
        {"BILIRUBIN": np.concatenate(([0.25], 0.5 + (np.arange(98) + 0.5) * 1e-12, [0.75]))}
    )

    plot_significantly_dissimilar_variables(real, fake, ["BILIRUBIN"], cat_cols=[])

    assert list(tmp_path.glob("SE_sig_hists_*.png"))
