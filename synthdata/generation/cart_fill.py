"""Fill values of indicator-only columns by CART leaf sampling (synthpop's ``syn.cart``).

Columns missing in most rows are generated as their ``<column>__missing``
indicator only (see ``imputation.missing_indicators.indicator_only_fraction``).
This is the second step of synthpop's two-step synthesis of variables with
missing values (Nowok, Raab & Dibben, *J Stat Softw* 2016): for synthetic
rows whose indicator says "recorded", draw a value conditional on the rest
of the synthetic row; rows marked "missing" stay blank.

Port of ``synthpop::syn.cart`` (CRAN, ``R/functions.syn.r``):

- Continuous: a regression tree (``minbucket = 5``, ``cp = 1e-8``, here
  ``min_samples_leaf=5`` and no pruning) is fitted on the real train rows
  where the value was observed; each synthetic row draws a random observed
  value (a donor) from the leaf it falls into. With ``smoothing="density"``
  each draw gets Gaussian noise, values are kept within the observed range
  and rounded to the observed decimals; columns where one value has more
  than 70% of draws keep that value unsmoothed, as in ``syn.smooth``.
  Deviation: the bandwidth uses Silverman's rule instead of R's
  Sheather-Jones (``width = "SJ"``), which SciPy does not provide.
- Categorical: a classification tree; each synthetic row draws a category
  from its leaf's class shares.

Donors are real values, which is why smoothing is on for continuous columns
(synthpop's documented disclosure control for CART).
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

#: synthpop's ``minbucket`` default.
MIN_LEAF = 5


def _decimals(values: pd.Series) -> int:
    text = values.dropna().astype(float).map(lambda v: f"{v:.10f}".rstrip("0"))
    return int(text.map(lambda t: len(t.split(".")[1]) if "." in t else 0).max())


def _smooth(draws: np.ndarray, observed: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """synthpop's ``syn.smooth(smoothing = "density")`` with a Silverman bandwidth."""
    values, counts = np.unique(draws, return_counts=True)
    smooth = np.ones(len(draws), dtype=bool)
    if counts.max() / counts.sum() > 0.7:
        smooth = draws != values[counts.argmax()]
    if smooth.sum() < 2:
        return draws
    x = draws[smooth]
    spread = min(x.std(ddof=1), (np.percentile(x, 75) - np.percentile(x, 25)) / 1.34) or x.std(
        ddof=1
    )
    bandwidth = 0.9 * spread * len(x) ** -0.2
    out = draws.astype(float).copy()
    noisy = rng.normal(x, bandwidth)
    noisy = np.clip(noisy, observed.min(), observed.max() + bandwidth)
    out[smooth] = np.round(noisy, _decimals(observed))
    return out


def encode_predictors(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Numeric predictor matrix; non-numeric columns use ``reference``'s category codes."""
    out = pd.DataFrame(index=frame.index)
    for column in reference.columns:
        if pd.api.types.is_numeric_dtype(reference[column]):
            out[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
        else:
            categories = sorted(reference[column].dropna().astype(str).unique())
            codes = pd.Categorical(frame[column].astype(str), categories=categories).codes
            out[column] = codes.astype(float)
    return out.fillna(-1)


def fit_cart_fills(
    real_predictors: pd.DataFrame, values: pd.DataFrame, categorical: set, seed: int
) -> dict:
    """Fit one tree per column on the real rows where that column was observed.

    ``real_predictors`` is the imputed real train data (complete); ``values``
    holds the raw values of the indicator-only columns on the same rows.
    """
    models = {}
    for column in values.columns:
        observed = values[column].notna()
        x, y = real_predictors.loc[observed], values.loc[observed, column]
        if y.empty:
            continue
        if column in categorical or not pd.api.types.is_numeric_dtype(y):
            tree = DecisionTreeClassifier(min_samples_leaf=MIN_LEAF, random_state=seed).fit(x, y)
            models[column] = ("categorical", tree, None, y)
        else:
            tree = DecisionTreeRegressor(min_samples_leaf=MIN_LEAF, random_state=seed).fit(x, y)
            models[column] = ("continuous", tree, pd.Series(tree.apply(x), index=y.index), y)
    return models


def fill(
    synthetic: pd.DataFrame,
    models: dict,
    indicators: dict,
    predictors: pd.DataFrame,
    seed: int,
    smoothing: bool = True,
) -> pd.DataFrame:
    """Return values for synthetic rows marked "recorded", NaN elsewhere.

    ``predictors`` is :func:`encode_predictors` applied to ``synthetic``.
    """
    rng = np.random.default_rng(seed)
    by_column = {column: name for name, column in indicators.items()}
    out = pd.DataFrame(index=synthetic.index)
    x = predictors
    for column, (kind, tree, leaves, observed) in models.items():
        recorded = pd.to_numeric(synthetic[by_column[column]], errors="coerce").round() != 1
        filled = pd.Series(np.nan, index=synthetic.index, dtype=object)
        if recorded.any():
            rows = x.loc[recorded]
            if kind == "categorical":
                probabilities = tree.predict_proba(rows)
                draws = [tree.classes_[rng.choice(len(p), p=p)] for p in probabilities]
            else:
                draws = np.empty(len(rows))
                synthetic_leaves = tree.apply(rows)
                for leaf in np.unique(synthetic_leaves):
                    donors = observed[leaves == leaf].to_numpy()
                    hit = synthetic_leaves == leaf
                    draws[hit] = rng.choice(donors, size=hit.sum(), replace=True)
                if smoothing:
                    draws = _smooth(draws, observed, rng)
            filled.loc[recorded] = draws
        out[column] = filled.astype(float) if kind == "continuous" else filled
    return out


def recorded_rows_report(filled: pd.DataFrame, observed: pd.DataFrame, categorical: set) -> list:
    """Real observed vs synthetic recorded values per column (KS or total variation)."""
    rows = []
    for column in filled.columns:
        real, synth = observed[column].dropna(), filled[column].dropna()
        if real.empty or synth.empty:
            continue
        if column in categorical or not pd.api.types.is_numeric_dtype(real):
            shares = pd.concat(
                [real.value_counts(normalize=True), synth.value_counts(normalize=True)], axis=1
            ).fillna(0)
            metric, value = (
                "total_variation",
                0.5 * float(shares.diff(axis=1).iloc[:, 1].abs().sum()),
            )
        else:
            metric, value = "ks_statistic", float(ks_2samp(real, synth.astype(float)).statistic)
        rows.append(
            {
                "column": column,
                "real_recorded_share": len(real) / len(observed),
                "synthetic_recorded_share": len(synth) / len(filled),
                "metric": metric,
                "value": value,
            }
        )
    return rows
