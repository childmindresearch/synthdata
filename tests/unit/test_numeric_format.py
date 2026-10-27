"""numeric_format follows rdt's FloatFormatter rounding and min/max rules.

Every expected value was computed with rdt 2.0.0
(``rdt.transformers.utils.learn_rounding_digits``, and
``FloatFormatter(learn_rounding_scheme=True, enforce_min_max_values=True)`` for
the clip-then-round order).
"""

import numpy as np
import pandas as pd
import pytest

from synthdata.generation import numeric_format as nf

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("values", "digits"),
    [
        ([15, None, 25], 0),
        ([15.0, np.inf, -np.inf, 25.0], 0),
        ([1.1, 2.2, None], 1),
        ([1.12345, 2.5, 3.0], 5),
        ([0.123456789012345, 1.0], None),  # 15 places: more than 15 - 1 integer digit
        ([123.456, 7.1], 3),
        ([12345678.123456, 1.5], 6),
        ([0.1 + 0.2, 1.0], None),
        ([1 / 3, 2 / 3], None),
        ([None, None], None),
        ([np.inf, -np.inf], None),
    ],
)
def test_learn_rounding_digits_matches_rdt(values, digits):
    assert nf.learn_rounding_digits(pd.Series(values, dtype=float)) == digits


def test_max_decimals_is_rdt_value():
    assert nf.MAX_DECIMALS == 15


def test_apply_format_clips_then_rounds_and_restores_integer_dtype():
    train = pd.DataFrame(
        {
            "count": pd.Series([0, 3, 103], dtype="int64"),
            "count_float": [0.0, 3.0, 103.0],
            "lab": [1.25, 2.5, 3.75],
            "irrational": [1 / 3, 2 / 3, 1.0],
        }
    )
    fmt = nf.learn_format(train, list(train.columns))
    synthetic = pd.DataFrame(
        {
            "count": [2.6, -1.0, 150.2],
            "count_float": [2.4, 0.51, np.nan],
            "lab": [2.234, 9.0, 0.5],
            "irrational": [0.123456, 5.0, 0.5],
            "category": ["a", "b", "c"],
        }
    )
    out = nf.apply_format(synthetic, fmt)

    assert out["count"].tolist() == [3, 0, 103]
    assert out["count"].dtype == "int64"
    assert out["count_float"].iloc[:2].tolist() == [2.0, 1.0]
    assert np.isnan(out["count_float"].iloc[2])  # missing stays missing, dtype stays float
    assert out["lab"].tolist() == [2.23, 3.75, 1.25]  # clipped to [1.25, 3.75], 2 places
    assert out["irrational"].tolist() == [1 / 3, 1.0, 0.5]  # clipped only
    assert out["category"].tolist() == ["a", "b", "c"]  # not in the format: untouched


def test_format_is_idempotent_on_train():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"x": rng.integers(0, 50, 200), "y": rng.normal(size=200).round(3)})
    fmt = nf.learn_format(train, ["x", "y"])
    pd.testing.assert_frame_equal(nf.apply_format(train, fmt), train)
