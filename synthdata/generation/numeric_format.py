"""Give synthetic numeric columns the decimals and range of the train data.

Some generators write values a real record never has: ARF draws a continuous
column from a truncated normal inside a forest leaf, so counts that are whole
numbers in train (0-103) come out with decimals in about 95% of rows. This step does
what SDV does for every numeric column by default (``enforce_rounding=True``
and ``enforce_min_max_values=True`` on its synthesizers, implemented by
rdt's ``FloatFormatter(learn_rounding_scheme=True,
enforce_min_max_values=True)``): learn from train the fewest decimal places
that represent every value exactly and the min and max, then clip each
synthetic value to that range and round it to those places. rdt is not a
dependency (it would pull in SDV's Faker stack for these few lines), so the
rule is reimplemented here; ``tests/unit/test_numeric_format.py`` holds cases
checked against rdt 2.0.0.

Only non-categorical columns are touched; categorical columns already hold
train categories. A column whose train values need more than
:data:`MAX_DECIMALS` places is clipped but not rounded, as in rdt.
"""

import sys

import numpy as np
import pandas as pd

#: rdt's ``MAX_DECIMALS``: the float precision a decimal place count can represent.
MAX_DECIMALS = sys.float_info.dig


def learn_rounding_digits(values: pd.Series) -> int | None:
    """Fewest decimal places that represent every finite value exactly.

    rdt 2.0's ``learn_rounding_digits``: a float holds :data:`MAX_DECIMALS`
    significant digits, so the places tried stop at that minus the integer
    digits of the largest value. None when there are no finite values or no
    such count represents them all.
    """
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return None
    highest_int = int(np.max(np.abs(data)))
    most_digits = len(str(highest_int)) if highest_int != 0 else 0
    max_decimals = max(0, MAX_DECIMALS - most_digits)
    if (data == data.round(max_decimals)).all():
        for digits in range(max_decimals + 1):
            if (data == data.round(digits)).all():
                return digits
    return None


def learn_format(train: pd.DataFrame, columns: list) -> dict:
    """``{column: (digits, min, max, dtype)}`` for the numeric ``columns`` of ``train``."""
    fmt = {}
    for col in columns:
        values = pd.to_numeric(train[col], errors="coerce")
        finite = values[np.isfinite(values)]
        if finite.empty:
            continue
        fmt[col] = (learn_rounding_digits(finite), finite.min(), finite.max(), train[col].dtype)
    return fmt


def apply_format(synthetic: pd.DataFrame, fmt: dict) -> pd.DataFrame:
    """Clip each formatted column to its train range, then round it (rdt's order).

    Missing values stay missing; a column with none goes back to an integer
    train dtype.
    """
    out = synthetic.copy()
    for col, (digits, low, high, dtype) in fmt.items():
        if col not in out:
            continue
        values = pd.to_numeric(out[col], errors="coerce").clip(low, high)
        if digits is not None:
            values = values.round(digits)
        if pd.api.types.is_integer_dtype(dtype) and not values.isna().any():
            values = values.astype(dtype)
        out[col] = values
    return out
