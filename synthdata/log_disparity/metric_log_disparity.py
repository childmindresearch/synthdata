"""Log Disparity Fairness Metric for Synthetic Data Evaluation.

This module implements the log disparity fairness metric for evaluating synthetic
healthcare datasets, based on the methodology from:

    Bhanot, K., et al. (2021). "The Problem of Fairness in Synthetic Healthcare Data."
    Entropy, 23(9), 1165. https://doi.org/10.3390/e23091165

Original R implementation:
    https://github.rpi.edu/RensselaerIDEA/SyntheticDataFairness

The log disparity metric quantifies representation bias across protected subgroups
by comparing background rates (real data) with observed rates (synthetic data) using
log odds ratios. Statistical significance is assessed via two-sided proportion tests
with Benjamini-Hochberg correction for multiple comparisons.

Typical usage:
    >>> import pandas as pd
    >>> from metric_log_disparity import compute_log_disparity_report
    >>>
    >>> real_data = pd.read_csv("real_data.csv")
    >>> synth_data = pd.read_csv("synthetic_data.csv")
    >>>
    >>> report = compute_log_disparity_report(
    ...     real_data=real_data,
    ...     synth_data=synth_data,
    ...     target_col="outcome",
    ...     protected_cols=["sex", "age"],
    ...     model_name="ctgan",
    ...     protected_bins=[None, [18, 30, 45, 60, 75]]
    ... )
    >>>
    >>> # Display results
    >>> print(report["summary_stats"])
    >>> report["report_figure"].show()

Author: Adapted from original R implementation by Karan Bhanot
Date: 2024-03
"""

import math
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency

from synthdata.utils import get_logger

logger = get_logger(__name__)

# ============================================================================
# Constants and Configuration
# ============================================================================

# Statistical significance threshold (alpha level)
SIG_THRESHOLD: float = 0.05

# Tolerance thresholds for equity classification
# Based on log(0.9) and log(0.8) from original R implementation
TOLERANCE_1: float = -math.log(0.9)  # ~0.10536 (moderate threshold)
TOLERANCE_2: float = -math.log(0.8)  # ~0.22314 (extreme threshold)

# Sentinel values for special cases
SENTINEL_NO_INFO: int = -9999999  # Both rates are zero
SENTINEL_NO_BASE: int = -8888888  # Background rate is zero
SENTINEL_INSUFFICIENT: int = -7777777  # Insufficient sample size

# Equity classification colors
EQUITY_COLORS: dict[str, str] = {
    "Highly Underrepresented": "#d58570",
    "Underrepresented": "#eabcad",
    "Equitable": "#d4e6e8",
    "Equitable(p)": "#d4e6e8",  # Non-significant difference
    "Overrepresented": "#a5b0cb",
    "Highly Overrepresented": "#00205b",
    "Absent": "#ab2328",
    "No Info": "#000000",
    "No Base Data": "#54585a",
    "Insufficient Data": "#000000",
}


# ============================================================================
# Core Metric Computation Functions
# ============================================================================


def rate_calculation(subgroup_n: int, total_n: int) -> float:
    """Compute subgroup rate with zero-safe denominator handling.

    Args:
        subgroup_n: Number of observations in the subgroup.
        total_n: Total number of observations.

    Returns:
        Proportion of subgroup within total, or 0.0 if total is zero.

    Example:
        >>> rate_calculation(25, 100)
        0.25
        >>> rate_calculation(0, 0)
        0.0
    """
    if total_n == 0:
        return 0.0
    return subgroup_n / total_n


def log_disparate_impact(alpha_rate: float, beta_rate: float, for_plot: bool = False) -> float:
    """Compute log disparate impact from background and observed rates.

    The metric is defined as:
        log(β/(1-β)) - log(α/(1-α))

    where:
        α (alpha) = background rate from real data
        β (beta) = observed rate from synthetic data

    Args:
        alpha_rate: Background rate from real training data.
        beta_rate: Observed rate from synthetic data.
        for_plot: If True, skip sentinel value checks for visualization.

    Returns:
        Log disparity value or sentinel value for special cases:
            - SENTINEL_NO_INFO if both rates are zero
            - -inf if synthetic rate is zero (absent subgroup)
            - SENTINEL_NO_BASE if real rate is zero

    References:
        Bhanot et al. (2021), Equation 3.

    Example:
        >>> log_disparate_impact(0.5, 0.6)
        0.405...
        >>> log_disparate_impact(0.0, 0.5)
        -8888888
    """
    if not for_plot:
        # Handle special cases with sentinels
        if beta_rate == 0 and alpha_rate == 0:
            return float(SENTINEL_NO_INFO)
        if beta_rate == 0:
            return -math.inf
        if alpha_rate == 0:
            return float(SENTINEL_NO_BASE)

    # Handle boundary cases to avoid divide-by-zero
    if beta_rate == 1 and alpha_rate == 1:
        return 0.0
    if beta_rate == 1:
        return math.inf
    if alpha_rate == 1:
        return -math.inf

    # Compute log odds ratio
    left_odds = beta_rate / (1 - beta_rate)
    right_odds = alpha_rate / (1 - alpha_rate)
    return math.log(left_odds) - math.log(right_odds)


def compare_population_proportion(
    background_pos: int,
    background_total: int,
    observed_pos: int,
    observed_total: int,
) -> float:
    """Two-sided population proportion test with chi-squared statistic.

    Compares whether the proportion in the synthetic data significantly differs
    from the proportion in the real data. Uses chi-squared test without continuity
    correction when sample size requirements are met (all counts >= 5).

    Args:
        background_pos: Number of positive cases in real data.
        background_total: Total number of cases in real data.
        observed_pos: Number of positive cases in synthetic data.
        observed_total: Total number of cases in synthetic data.

    Returns:
        P-value from two-sided proportion test, or sentinel value:
            - SENTINEL_NO_INFO if both counts are zero
            - -inf if synthetic count is zero
            - SENTINEL_NO_BASE if real count is zero
            - SENTINEL_INSUFFICIENT if sample size requirements not met

    Example:
        >>> compare_population_proportion(50, 100, 45, 100)
        0.49...
    """
    # Handle special cases
    if observed_pos == 0 and background_pos == 0:
        return float(SENTINEL_NO_INFO)
    if observed_pos == 0:
        return -math.inf
    if background_pos == 0:
        return float(SENTINEL_NO_BASE)

    observed_neg = observed_total - observed_pos
    background_neg = background_total - background_pos

    # Check sample size requirements for chi-squared test
    if observed_pos >= 5 and observed_neg >= 5 and background_pos >= 5 and background_neg >= 5:
        # Construct 2x2 contingency table
        table = np.array(
            [[observed_pos, observed_neg], [background_pos, background_neg]],
            dtype=float,
        )

        try:
            # Perform chi-squared test without continuity correction
            _, p_value, _, _ = chi2_contingency(table, correction=False)
            return float(p_value)
        except ValueError:
            logger.warning(
                "chi2_contingency degenerate table observed=(%d,%d) background=(%d,%d); "
                "returning insufficient-data sentinel",
                observed_pos,
                observed_neg,
                background_pos,
                background_neg,
            )
            return float(SENTINEL_INSUFFICIENT)

    return float(SENTINEL_INSUFFICIENT)


def benjamini_hochberg_correction(p_values: pd.Series) -> pd.Series:
    """Apply Benjamini-Hochberg FDR correction to p-values.

    Adjusts p-values for multiple comparisons using the Benjamini-Hochberg
    procedure to control the false discovery rate. Sentinel values are
    preserved as NaN in the output.

    Args:
        p_values: Series of p-values, may contain sentinel values.

    Returns:
        Series of BH-adjusted p-values, with NaN for sentinels.

    References:
        Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery
        rate: A practical and powerful approach to multiple testing.
        Journal of the Royal Statistical Society, 57(1), 289-300.

    Example:
        >>> p_vals = pd.Series([0.01, 0.04, 0.03, 0.10])
        >>> benjamini_hochberg_correction(p_vals)
        0    0.04...
        1    0.066...
        2    0.06...
        3    0.10...
        dtype: float64
    """
    # Convert to numeric, marking sentinels as NaN
    p = pd.to_numeric(p_values, errors="coerce")

    # Filter to valid p-values in [0, 1]
    valid_mask = p.between(0, 1, inclusive="both")
    adjusted = pd.Series(np.nan, index=p.index, dtype=float)

    valid = p[valid_mask]
    if valid.empty:
        return adjusted

    # Sort p-values and compute BH adjustment
    order = np.argsort(valid.values)
    ordered = valid.values[order]
    n = len(ordered)

    # Apply BH procedure: p_adj[i] = min(p[i] * n / rank[i], previous_adj)
    bh = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ordered[i] * n / rank
        prev = min(prev, val)
        bh[i] = min(prev, 1.0)

    # Restore original order
    restored = np.empty(n, dtype=float)
    restored[order] = bh
    adjusted.loc[valid.index] = restored

    return adjusted


def classify_equity_outcome(
    disparity_value: float,
    adjusted_p_value: float,
    sig_threshold: float = SIG_THRESHOLD,
    tol_moderate: float = TOLERANCE_1,
    tol_extreme: float = TOLERANCE_2,
) -> str:
    """Assign equity classification label based on log disparity and significance.

    Classification rules (from R implementation):
        1. Handle sentinel values first (Absent, No Info, No Base Data, Insufficient Data)
        2. If BH-adjusted p > sig_threshold: "Equitable(p)" (not significant)
        3. If disparity < -tol_extreme: "Highly Underrepresented"
        4. If disparity < -tol_moderate: "Underrepresented"
        5. If -tol_moderate <= disparity <= tol_moderate: "Equitable"
        6. If disparity > tol_moderate: "Overrepresented"
        7. If disparity > tol_extreme: "Highly Overrepresented"

    Args:
        disparity_value: Log disparity value.
        adjusted_p_value: BH-adjusted p-value.
        sig_threshold: Significance threshold for p-values.
        tol_moderate: Moderate deviation threshold.
        tol_extreme: Extreme deviation threshold.

    Returns:
        Equity classification label.

    Example:
        >>> classify_equity_outcome(0.05, 0.80)
        'Equitable(p)'
        >>> classify_equity_outcome(-0.25, 0.01)
        'Highly Underrepresented'
    """
    # Handle sentinel values first (these override p-value checks)
    if disparity_value == -math.inf:
        return "Absent"
    if disparity_value == SENTINEL_NO_INFO:
        return "No Info"
    if disparity_value == SENTINEL_NO_BASE:
        return "No Base Data"
    if disparity_value == SENTINEL_INSUFFICIENT:
        return "Insufficient Data"

    # Check significance
    if pd.notna(adjusted_p_value) and adjusted_p_value > sig_threshold:
        return "Equitable(p)"

    # Classify based on thresholds
    if disparity_value < -tol_moderate:
        if disparity_value < -tol_extreme:
            return "Highly Underrepresented"
        return "Underrepresented"

    if disparity_value > tol_moderate:
        if disparity_value > tol_extreme:
            return "Highly Overrepresented"
        return "Overrepresented"

    return "Equitable"


def assign_equity_color(
    disparity_value: float,
    adjusted_p_value: float,
    sig_threshold: float = SIG_THRESHOLD,
    tol_moderate: float = TOLERANCE_1,
    tol_extreme: float = TOLERANCE_2,
) -> str:
    """Map equity outcome to display color.

    Args:
        disparity_value: Log disparity value.
        adjusted_p_value: BH-adjusted p-value.
        sig_threshold: Significance threshold.
        tol_moderate: Moderate deviation threshold.
        tol_extreme: Extreme deviation threshold.

    Returns:
        Hex color code for visualization.
    """
    label = classify_equity_outcome(
        disparity_value, adjusted_p_value, sig_threshold, tol_moderate, tol_extreme
    )
    return EQUITY_COLORS.get(label, "#000000")


# ============================================================================
# Data Preparation Functions
# ============================================================================


def _is_integer_like_edges(edges: list[float]) -> bool:
    """Check if all bin edges represent integer values."""
    return all(float(edge).is_integer() for edge in edges)


def _max_decimal_places(edges: list[float]) -> int:
    """Return maximum decimal precision across numeric edge literals."""
    max_places = 0
    for edge in edges:
        dec = Decimal(str(edge))
        places = max(0, -dec.as_tuple().exponent)
        max_places = max(max_places, places)
    return max_places


def generate_bin_labels(bin_edges: list[float]) -> list[str]:
    """Generate human-readable bin labels for numeric edges.

    For integer bins, uses consecutive closed ranges (e.g., [7, 18, 30] -> ["7-18", "19-30"]).
    For float bins, infers precision and adds one decimal place (e.g., [0, 0.5, 1] -> ["0.00-0.50", "0.51-1.00"]).

    Args:
        bin_edges: List of bin edge values in ascending order.

    Returns:
        List of bin label strings.

    Raises:
        ValueError: If bin_edges has fewer than 2 elements.

    Example:
        >>> generate_bin_labels([18, 30, 45, 60])
        ['18-30', '31-45', '46-60']
        >>> generate_bin_labels([0.0, 0.5, 1.0])
        ['0.00-0.50', '0.51-1.00']
    """
    if not isinstance(bin_edges, list) or len(bin_edges) < 2:
        raise ValueError("bin_edges must be a list of at least 2 values")

    labels = []

    if _is_integer_like_edges(bin_edges):
        # Integer binning: consecutive ranges
        for i in range(len(bin_edges) - 1):
            left = int(round(float(bin_edges[i])))
            right = int(round(float(bin_edges[i + 1])))
            left_display = left if i == 0 else left + 1
            labels.append(f"{left_display}-{right}")
    else:
        # Float binning: add precision
        precision = _max_decimal_places(bin_edges) + 1
        step = 10 ** (-precision)
        for i in range(len(bin_edges) - 1):
            left = float(bin_edges[i])
            right = float(bin_edges[i + 1])
            left_display = left if i == 0 else left + step
            labels.append(f"{left_display:.{precision}f}-{right:.{precision}f}")

    return labels


def prepare_data_for_analysis(
    data: pd.DataFrame,
    target_col: str,
    protected_cols: list[str],
    target_map: dict[Any, str] | None = None,
    protected_map: list[dict[Any, str] | None] | None = None,
    target_bins: list[float] | None = None,
    protected_bins: list[list[float] | None] | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]], list[str] | None]:
    """Prepare target and protected columns with mapping and optional binning.

    This function:
        1. Validates column existence
        2. Applies categorical mappings if provided
        3. Bins numerical columns if bin edges provided
        4. Creates derived group columns with standardized naming

    Args:
        data: Input DataFrame.
        target_col: Name of target/outcome column.
        protected_cols: List of protected attribute column names.
        target_map: Optional mapping dict for target values {value: label}.
        protected_map: Optional list of mapping dicts, aligned with protected_cols.
        target_bins: Optional list of bin edges for target column.
        protected_bins: Optional list of bin edge lists, aligned with protected_cols.

    Returns:
        Tuple of:
            - prepared_df: DataFrame with derived columns
            - protected_group_cols: List of derived group column names
            - protected_order_map: Dict mapping group columns to ordered categories
            - target_order: Optional list of ordered target categories

    Raises:
        KeyError: If specified columns don't exist.
        ValueError: If both map and bins provided for same column, or if list lengths don't match.

    Example:
        >>> df = pd.DataFrame({"age": [25, 35, 50], "outcome": [0, 1, 0]})
        >>> prepared, groups, orders, target_ord = prepare_data_for_analysis(
        ...     df, "outcome", ["age"],
        ...     protected_bins=[None, [0, 30, 40, 100]]
        ... )
    """
    # Validate inputs
    if target_col not in data.columns:
        raise KeyError(f"target_col '{target_col}' not found in data columns")

    if not isinstance(protected_cols, list) or len(protected_cols) == 0:
        raise ValueError("protected_cols must be a non-empty list")

    for col in protected_cols:
        if col not in data.columns:
            raise KeyError(f"protected column '{col}' not found in data columns")

    # Validate optional argument lengths
    if protected_map is None:
        protected_map = [None] * len(protected_cols)
    elif len(protected_map) != len(protected_cols):
        raise ValueError(
            f"protected_map length ({len(protected_map)}) must match protected_cols length ({len(protected_cols)})"
        )

    if protected_bins is None:
        protected_bins = [None] * len(protected_cols)
    elif len(protected_bins) != len(protected_cols):
        raise ValueError(
            f"protected_bins length ({len(protected_bins)}) must match protected_cols length ({len(protected_cols)})"
        )

    if target_map is not None and target_bins is not None:
        raise ValueError("target_map and target_bins are mutually exclusive")

    # Prepare output dataframe
    out = data.copy()

    # Process target column
    target_order = None
    if target_bins is not None:
        target_labels = generate_bin_labels(target_bins)
        numeric_target = pd.to_numeric(out[target_col], errors="coerce")
        target_binned = pd.cut(
            numeric_target,
            bins=target_bins,
            labels=target_labels,
            include_lowest=True,
            right=True,
        )
        out["TARGET_LABEL"] = target_binned.astype("object").fillna("OutOfRange").astype(str)
        target_order = target_labels + ["OutOfRange"]
    elif target_map is not None:
        out["TARGET_LABEL"] = (
            out[target_col].map(target_map).fillna(out[target_col].astype(str)).astype(str)
        )
        target_order = list(dict.fromkeys(target_map.values()))
    else:
        out["TARGET_LABEL"] = out[target_col].astype(str)

    # Process protected columns
    derived_cols = []
    order_maps = {}

    for idx, col in enumerate(protected_cols):
        map_dict = protected_map[idx]
        bins = protected_bins[idx]
        derived_col = f"{col}__GROUP"

        values = out[col].copy()

        # Apply mapping if provided
        if map_dict is not None:
            values = values.map(map_dict).fillna(values.astype(str))
            order_maps[derived_col] = list(dict.fromkeys(map_dict.values()))

        # Apply binning if provided
        if bins is not None:
            labels = generate_bin_labels(bins)
            numeric_values = pd.to_numeric(out[col], errors="coerce")
            binned = pd.cut(
                numeric_values,
                bins=bins,
                labels=labels,
                include_lowest=True,
                right=True,
            )
            values = binned.astype("object").fillna("OutOfRange")
            order_maps[derived_col] = labels + ["OutOfRange"]

        out[derived_col] = values.astype(str)
        derived_cols.append(derived_col)

    return out, derived_cols, order_maps, target_order


# ============================================================================
# Main Analysis Function
# ============================================================================


def compute_log_disparity_report(
    real_data: pd.DataFrame,
    synth_data: pd.DataFrame,
    target_col: str,
    protected_cols: list[str],
    model_name: str = "synthetic_model",
    target_map: dict[Any, str] | None = None,
    protected_map: list[dict[Any, str] | None] | None = None,
    target_bins: list[float] | None = None,
    protected_bins: list[list[float] | None] | None = None,
) -> dict[str, Any]:
    """Compute comprehensive log disparity fairness report for synthetic dataset.

    This function performs a complete fairness analysis comparing a synthetic dataset
    to real data across protected subgroups. It computes:
        - Log disparity metrics for all subgroup combinations
        - Statistical significance tests with BH correction
        - Equity classifications and visualizations
        - Summary statistics and tables

    Args:
        real_data: Real/reference DataFrame (training data).
        synth_data: Synthetic DataFrame to evaluate.
        target_col: Target/outcome column name.
        protected_cols: Ordered list of protected attribute columns. This order
            determines the hierarchy in sunburst visualization.
        model_name: Model identifier for labeling outputs.
        target_map: Optional dict mapping target values to labels {value: label}.
        protected_map: Optional list of dicts mapping protected values to labels,
            aligned with protected_cols.
        target_bins: Optional list of bin edges for binning target column.
        protected_bins: Optional list of bin edge lists for binning protected
            columns, aligned with protected_cols.

    Returns:
        Dictionary containing:
            - 'summary_stats': Dict with aggregate metrics (mean/median disparity, etc.)
            - 'leaf_results': DataFrame with all subgroup combinations and metrics
            - 'hierarchy_results': DataFrame with hierarchical aggregations
            - 'subgroup_table': Formatted DataFrame for display
            - 'legend_table': Color legend DataFrame
            - 'label_counts': Count of each equity classification
            - 'report_figure': Plotly figure with sunburst and tables

    Raises:
        KeyError: If specified columns don't exist in data.
        ValueError: If input validation fails.

    Example:
        >>> import pandas as pd
        >>> real = pd.DataFrame({
        ...     "age": [25, 35, 50, 45],
        ...     "sex": [0, 1, 0, 1],
        ...     "outcome": [0, 1, 1, 0]
        ... })
        >>> synth = pd.DataFrame({
        ...     "age": [28, 38, 48, 42],
        ...     "sex": [0, 1, 1, 0],
        ...     "outcome": [1, 1, 0, 0]
        ... })
        >>> report = compute_log_disparity_report(
        ...     real, synth, "outcome", ["sex", "age"],
        ...     model_name="test_model",
        ...     protected_map=[{0: "male", 1: "female"}, None],
        ...     protected_bins=[None, [0, 30, 40, 100]]
        ... )
        >>> print(report["summary_stats"])

    References:
        Bhanot, K., Qi, M., Erickson, J. S., Guyon, I., & Bennett, K. P. (2021).
        The Problem of Fairness in Synthetic Healthcare Data. Entropy, 23(9), 1165.
        https://doi.org/10.3390/e23091165
    """
    # Prepare data
    real_prepared, protected_group_cols, protected_order_map, target_order = (
        prepare_data_for_analysis(
            data=real_data,
            target_col=target_col,
            protected_cols=protected_cols,
            target_map=target_map,
            protected_map=protected_map,
            target_bins=target_bins,
            protected_bins=protected_bins,
        )
    )

    synth_prepared, _, _, _ = prepare_data_for_analysis(
        data=synth_data,
        target_col=target_col,
        protected_cols=protected_cols,
        target_map=target_map,
        protected_map=protected_map,
        target_bins=target_bins,
        protected_bins=protected_bins,
    )

    # Aggregate counts
    group_cols = protected_group_cols + ["TARGET_LABEL"]
    baseline_counts = (
        real_prepared.groupby(group_cols, dropna=False).size().reset_index(name="background_n")
    )
    user_counts = synth_prepared.groupby(group_cols, dropna=False).size().reset_index(name="user_n")

    total_background = int(baseline_counts["background_n"].sum())
    total_user = int(user_counts["user_n"].sum())

    # Merge counts
    leaf = baseline_counts.merge(
        user_counts,
        on=group_cols,
        how="outer",
    )
    leaf["background_n"] = leaf["background_n"].fillna(0).astype(int)
    leaf["user_n"] = leaf["user_n"].fillna(0).astype(int)
    leaf["Model"] = model_name

    # Compute metrics
    leaf = _attach_disparity_metrics(leaf, total_background, total_user)

    # Build hierarchy for sunburst
    hierarchy = _build_hierarchy_frame(
        baseline_counts, user_counts, protected_group_cols, model_name
    )

    # Create display tables
    protected_display_names = {
        p_col: _prettify_name(p_col.replace("__GROUP", "")) for p_col in protected_group_cols
    }

    subgroup_table = _build_subgroup_equity_table(
        hierarchy, protected_group_cols, protected_display_names
    )

    leaf_equity_table = _build_leaf_equity_table(hierarchy, protected_group_cols)

    legend_table = _build_legend_table()

    # Create visualization
    report_figure = _build_model_report_figure(
        model_name,
        hierarchy,
        subgroup_table,
        leaf_equity_table,
        legend_table,
        protected_group_cols,
        protected_order_map,
        target_order,
    )

    # Compute summary statistics
    valid_mask = ~leaf["EquityValue"].isin(
        [SENTINEL_NO_INFO, SENTINEL_NO_BASE, SENTINEL_INSUFFICIENT]
    ) & ~np.isinf(leaf["EquityValue"])
    valid_values = leaf.loc[valid_mask, "EquityValue"]
    sig_share = (
        leaf["BH_p"].between(0, SIG_THRESHOLD, inclusive="both").mean() if len(leaf) else np.nan
    )

    summary_stats = {
        "model": model_name,
        "n_subgroups": len(leaf),
        "mean_abs_log_disparity": float(valid_values.abs().mean()) if len(valid_values) else np.nan,
        "median_abs_log_disparity": float(valid_values.abs().median())
        if len(valid_values)
        else np.nan,
        "share_significant_bh": float(sig_share) if pd.notna(sig_share) else np.nan,
    }

    label_counts = (
        leaf.groupby(["Model", "EquityLabel"], dropna=False).size().reset_index(name="count")
    )

    return {
        "summary_stats": summary_stats,
        "leaf_results": leaf,
        "hierarchy_results": hierarchy,
        "subgroup_table": subgroup_table,
        "leaf_equity_table": leaf_equity_table,
        "legend_table": legend_table,
        "label_counts": label_counts,
        "report_figure": report_figure,
    }


# ============================================================================
# Helper Functions (Internal)
# ============================================================================


def _attach_disparity_metrics(
    df: pd.DataFrame, total_background: int, total_user: int
) -> pd.DataFrame:
    """Attach rates, p-values, BH correction, equity values, labels, and colors."""
    out = df.copy()
    out["total_background"] = total_background
    out["total_user"] = total_user

    # Compute rates
    out["Background_Rate"] = out.apply(
        lambda r: rate_calculation(int(r["background_n"]), total_background),
        axis=1,
    )
    out["Observed_Rate"] = out.apply(
        lambda r: rate_calculation(int(r["user_n"]), total_user),
        axis=1,
    )

    # Compute p-values
    out["pValue"] = out.apply(
        lambda r: compare_population_proportion(
            int(r["background_n"]),
            total_background,
            int(r["user_n"]),
            total_user,
        ),
        axis=1,
    )

    # Apply BH correction
    out["BH_p"] = benjamini_hochberg_correction(out["pValue"])

    # Compute equity metrics
    out["EquityValue"] = out.apply(
        lambda r: log_disparate_impact(
            float(r["Background_Rate"]), float(r["Observed_Rate"]), for_plot=False
        ),
        axis=1,
    )

    out["EquityLabel"] = out.apply(
        lambda r: classify_equity_outcome(float(r["EquityValue"]), r["BH_p"]),
        axis=1,
    )

    out["EquityColor"] = out.apply(
        lambda r: assign_equity_color(float(r["EquityValue"]), r["BH_p"]),
        axis=1,
    )

    return out


def _build_hierarchy_frame(
    baseline_counts: pd.DataFrame,
    user_counts: pd.DataFrame,
    protected_group_cols: list[str],
    model_name: str,
) -> pd.DataFrame:
    """Build hierarchical aggregations for sunburst visualization."""
    total_background = int(baseline_counts["background_n"].sum())
    total_user = int(user_counts["user_n"].sum())

    # Define hierarchy levels
    levels = [("target", ["TARGET_LABEL"])]
    for i in range(len(protected_group_cols)):
        chain_cols = ["TARGET_LABEL"] + protected_group_cols[: i + 1]
        levels.append((f"target_chain_{i + 1}", chain_cols))
    for col in protected_group_cols:
        levels.append((f"protected::{col}", [col]))

    # Aggregate at each level
    level_frames = []
    for level_name, cols in levels:
        bg = baseline_counts.groupby(cols, dropna=False)["background_n"].sum().reset_index()
        us = user_counts.groupby(cols, dropna=False)["user_n"].sum().reset_index()

        merged = bg.merge(us, on=cols, how="outer")
        merged["background_n"] = merged["background_n"].fillna(0).astype(int)
        merged["user_n"] = merged["user_n"].fillna(0).astype(int)

        merged["Model"] = model_name
        merged["level"] = level_name

        for col in ["TARGET_LABEL"] + protected_group_cols:
            if col not in merged.columns:
                merged[col] = np.nan

        level_frames.append(
            merged[
                ["Model", "level", "TARGET_LABEL"]
                + protected_group_cols
                + ["background_n", "user_n"]
            ]
        )

    hierarchy = pd.concat(level_frames, ignore_index=True)
    hierarchy = _attach_disparity_metrics(hierarchy, total_background, total_user)

    return hierarchy


def _build_subgroup_equity_table(
    hierarchy_df: pd.DataFrame,
    protected_group_cols: list[str],
    display_names: dict[str, str],
) -> pd.DataFrame:
    """Create formatted equity table for protected subgroups (excludes leaf intersectional rows)."""
    table_parts = []

    # Add protected subgroup rows
    for p_col in protected_group_cols:
        part = hierarchy_df[hierarchy_df["level"] == f"protected::{p_col}"].copy()
        if part.empty:
            continue
        part["Characteristic"] = display_names.get(p_col, p_col)
        part["Protected Subgroup"] = part[p_col].astype(str)
        table_parts.append(part)

    # Add target rows
    target_part = hierarchy_df[hierarchy_df["level"] == "target"].copy()
    target_part["Characteristic"] = "Target"
    target_part["Protected Subgroup"] = target_part["TARGET_LABEL"].astype(str)
    table_parts.append(target_part)

    table = pd.concat(table_parts, ignore_index=True)

    # Order rows
    char_order = {display_names.get(col, col): i for i, col in enumerate(protected_group_cols)}
    char_order["Target"] = len(protected_group_cols)
    table["_char_order"] = table["Characteristic"].map(char_order).fillna(999)
    table["_sub_order"] = (
        table.groupby("Characteristic")["Protected Subgroup"].rank(method="dense").astype(int)
    )

    # Format values
    table["Equity Value"] = table.apply(
        lambda r: _format_equity_with_star(r["EquityValue"], r["BH_p"]),
        axis=1,
    )
    table["BH-adjusted p-value"] = table["BH_p"].map(_format_p_value)

    return table.sort_values(["_char_order", "_sub_order"])[
        [
            "Characteristic",
            "Protected Subgroup",
            "Equity Value",
            "BH-adjusted p-value",
            "EquityLabel",
            "EquityColor",
        ]
    ].reset_index(drop=True)


def _build_leaf_equity_table(
    hierarchy_df: pd.DataFrame,
    protected_group_cols: list[str],
) -> pd.DataFrame:
    """Create formatted equity table for leaf (intersectional) subgroup combinations."""
    _empty = pd.DataFrame(
        columns=[
            "Characteristic",
            "Protected Subgroup",
            "Equity Value",
            "BH-adjusted p-value",
            "EquityLabel",
            "EquityColor",
        ]
    )

    if len(protected_group_cols) == 0:
        return _empty

    deepest_level = f"target_chain_{len(protected_group_cols)}"
    part = hierarchy_df[hierarchy_df["level"] == deepest_level].copy()

    if part.empty:
        return _empty

    subgroup_labels = []
    for _, row in part.iterrows():
        label_parts = [str(row["TARGET_LABEL"])]
        for p_col in protected_group_cols:
            if pd.notna(row[p_col]):
                label_parts.append(str(row[p_col]))
        subgroup_labels.append(" × ".join(label_parts))

    part["Characteristic"] = "Intersectional"
    part["Protected Subgroup"] = subgroup_labels
    part["_sub_order"] = (
        part.groupby("Characteristic")["Protected Subgroup"].rank(method="dense").astype(int)
    )
    part["Equity Value"] = part.apply(
        lambda r: _format_equity_with_star(r["EquityValue"], r["BH_p"]),
        axis=1,
    )
    part["BH-adjusted p-value"] = part["BH_p"].map(_format_p_value)

    return part.sort_values("_sub_order")[
        [
            "Characteristic",
            "Protected Subgroup",
            "Equity Value",
            "BH-adjusted p-value",
            "EquityLabel",
            "EquityColor",
        ]
    ].reset_index(drop=True)


def _build_legend_table() -> pd.DataFrame:
    """Build color legend table for equity classifications."""
    return pd.DataFrame(
        [
            {
                "Description": "Missing / Absent",
                "Metric Value Rule": "-Inf",
                "Color": "#ab2328",
            },
            {
                "Description": "Highly under-represented",
                "Metric Value Rule": "< log(0.8)",
                "Color": "#d58570",
            },
            {
                "Description": "Under-represented",
                "Metric Value Rule": "Between log(0.8) and log(0.9)",
                "Color": "#eabcad",
            },
            {
                "Description": "Adequately represented",
                "Metric Value Rule": "Between log(0.9) and -log(0.9) or BH p > 0.05",
                "Color": "#d4e6e8",
            },
            {
                "Description": "Over-represented",
                "Metric Value Rule": "Between -log(0.9) and -log(0.8)",
                "Color": "#a5b0cb",
            },
            {
                "Description": "Highly over-represented",
                "Metric Value Rule": "> -log(0.8)",
                "Color": "#00205b",
            },
            {
                "Description": "No Base Data",
                "Metric Value Rule": "-8888888",
                "Color": "#54585a",
            },
            {
                "Description": "No Info / Insufficient Data",
                "Metric Value Rule": "-9999999 or -7777777",
                "Color": "#000000",
            },
        ]
    )


def _prettify_name(name: str) -> str:
    """Convert snake_case/CAPS to Title Case for display."""
    return str(name).replace("_", " ").title()


def _format_equity_value(value: float) -> str:
    """Format equity value with sentinel handling."""
    if pd.isna(value):
        return "NA"
    if np.isneginf(value):
        return "-Inf"
    if np.isposinf(value):
        return "+Inf"
    if value in (SENTINEL_NO_INFO, SENTINEL_NO_BASE, SENTINEL_INSUFFICIENT):
        return str(int(value))
    return f"{float(value):.3f}"


def _format_equity_with_star(value: float, bh_p: float) -> str:
    """Append star marker when difference is not statistically significant."""
    formatted = _format_equity_value(value)
    if pd.notna(bh_p) and float(bh_p) > SIG_THRESHOLD:
        return f"{formatted}*"
    return formatted


def _format_p_value(value: float) -> str:
    """Format p-value for display."""
    if pd.isna(value):
        return "NA"
    return f"{float(value):.3f}"


def _equity_tooltip_value(value: float) -> str:
    """Format equity value for tooltip display."""
    if value == SENTINEL_NO_INFO:
        return "No Info"
    if value == SENTINEL_NO_BASE:
        return "No Base Data"
    if value == SENTINEL_INSUFFICIENT:
        return "Insufficient Data"
    if np.isneginf(value):
        return "-Inf (Absent)"
    if np.isposinf(value):
        return "+Inf"
    return f"{float(value):+.4f}"


def _get_equity_text_color(equity_label: str) -> str:
    """Return black or white text based on equity classification.

    Rules:
        - Black text for moderate categories: Underrepresented, Overrepresented,
          Equitable, Equitable(p)
        - White text for all others: Highly Underrepresented, Highly Overrepresented,
          Absent, No Info, No Base Data, Insufficient Data

    Args:
        equity_label: Equity classification label.

    Returns:
        Hex color code for text (#111111 for black, #ffffff for white).
    """
    # Moderate categories get black text
    black_text_labels = {
        "Underrepresented",
        "Overrepresented",
        "Equitable",
        "Equitable(p)",
    }

    return "#111111" if equity_label in black_text_labels else "#ffffff"


def _get_equity_text_colors(equity_labels: list[str]) -> list[str]:
    """Vectorized wrapper for equity text color calculation."""
    return [_get_equity_text_color(label) for label in equity_labels]


def _get_legend_text_color(description: str) -> str:
    """Return black or white text for legend descriptions.

    Rules:
        - Black text for: "Under-represented", "Over-represented",
          "Adequately represented"
        - White text for all others

    Args:
        description: Legend description text.

    Returns:
        Hex color code for text (#111111 for black, #ffffff for white).
    """
    black_text_descriptions = {
        "Under-represented",
        "Over-represented",
        "Adequately represented",
    }

    return "#111111" if description in black_text_descriptions else "#ffffff"


def _get_legend_text_colors(descriptions: list[str]) -> list[str]:
    """Vectorized wrapper for legend text color calculation."""
    return [_get_legend_text_color(desc) for desc in descriptions]


# ============================================================================
# Visualization Functions
# ============================================================================


def _build_sunburst_trace(
    hierarchy_df: pd.DataFrame,
    protected_group_cols: list[str],
    protected_order_map: dict[str, list[str]],
    target_order: list[str] | None = None,
) -> go.Sunburst:
    """Build sunburst trace with hierarchical subgroup visualization."""
    ids, labels, parents, values, colors, customdata = [], [], [], [], [], []

    # Add target level
    target_rows = hierarchy_df[hierarchy_df["level"] == "target"].copy()
    if target_order is not None:
        target_rows["TARGET_LABEL"] = pd.Categorical(
            target_rows["TARGET_LABEL"], categories=target_order, ordered=True
        )
    target_rows = target_rows.sort_values(["TARGET_LABEL"])

    for _, row in target_rows.iterrows():
        t = str(row["TARGET_LABEL"])
        node_id = f"t::{t}"

        ids.append(node_id)
        labels.append(t)
        parents.append("")
        values.append(int(row["user_n"]))
        colors.append(str(row["EquityColor"]))

        p_txt = "NA" if pd.isna(row["BH_p"]) else f"{float(row['BH_p']):.4f}"
        customdata.append(
            f"Target: {t}<br>"
            f"Equity label: {row['EquityLabel']}<br>"
            f"Log disparity: {_equity_tooltip_value(row['EquityValue'])}<br>"
            f"Real rate: {float(row['Background_Rate']):.4f}<br>"
            f"Synthetic rate: {float(row['Observed_Rate']):.4f}<br>"
            f"Real n: {int(row['background_n'])}<br>"
            f"Synthetic n: {int(row['user_n'])}<br>"
            f"BH-adjusted p: {p_txt}"
        )

    # Add protected attribute levels
    for depth, p_col in enumerate(protected_group_cols, start=1):
        level_name = f"target_chain_{depth}"
        sort_cols = ["TARGET_LABEL"] + protected_group_cols[:depth]
        level_rows = hierarchy_df[hierarchy_df["level"] == level_name].copy()

        if target_order is not None:
            level_rows["TARGET_LABEL"] = pd.Categorical(
                level_rows["TARGET_LABEL"], categories=target_order, ordered=True
            )

        for c in protected_group_cols[:depth]:
            if c in protected_order_map:
                level_rows[c] = pd.Categorical(
                    level_rows[c], categories=protected_order_map[c], ordered=True
                )

        level_rows = level_rows.sort_values(sort_cols)

        for _, row in level_rows.iterrows():
            path_parts = [f"t::{str(row['TARGET_LABEL'])}"]
            for c in protected_group_cols[:depth]:
                path_parts.append(f"{c}::{str(row[c])}")
            node_id = "|".join(path_parts)
            parent_id = "|".join(path_parts[:-1])
            label = str(row[p_col])

            ids.append(node_id)
            labels.append(label)
            parents.append(parent_id)
            values.append(int(row["user_n"]))
            colors.append(str(row["EquityColor"]))

            p_txt = "NA" if pd.isna(row["BH_p"]) else f"{float(row['BH_p']):.4f}"
            customdata.append(
                f"Target: {row['TARGET_LABEL']}<br>"
                + "".join(
                    [
                        f"{_prettify_name(c.replace('__GROUP', ''))}: {row[c]}<br>"
                        for c in protected_group_cols[:depth]
                    ]
                )
                + f"Equity label: {row['EquityLabel']}<br>"
                + f"Log disparity: {_equity_tooltip_value(row['EquityValue'])}<br>"
                + f"Real rate: {float(row['Background_Rate']):.4f}<br>"
                + f"Synthetic rate: {float(row['Observed_Rate']):.4f}<br>"
                + f"Real n: {int(row['background_n'])}<br>"
                + f"Synthetic n: {int(row['user_n'])}<br>"
                + f"BH-adjusted p: {p_txt}"
            )

    return go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(colors=colors, line=dict(color="#ffffff", width=1)),
        leaf=dict(opacity=1.0),
        customdata=customdata,
        hovertemplate="%{customdata}<extra></extra>",
        insidetextorientation="radial",
    )


def _build_model_report_figure(
    model_name: str,
    hierarchy_df: pd.DataFrame,
    subgroup_table: pd.DataFrame,
    leaf_table: pd.DataFrame,
    legend_table: pd.DataFrame,
    protected_group_cols: list[str],
    protected_order_map: dict[str, list[str]],
    target_order: list[str] | None = None,
) -> go.Figure:
    """Create comprehensive report figure with sunburst and tables."""
    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"type": "domain", "rowspan": 3}, {"type": "table"}],
            [None, {"type": "table"}],
            [None, {"type": "table"}],
        ],
        column_widths=[0.60, 0.40],
        row_heights=[0.30, 0.40, 0.30],
        horizontal_spacing=0.08,
        vertical_spacing=0.03,
        subplot_titles=(
            "Sunburst",
            "Protected Subgroup Equity",
            "Leaf Subgroup Equity",
            "Color Legend",
        ),
    )

    # Add sunburst
    fig.add_trace(
        _build_sunburst_trace(
            hierarchy_df, protected_group_cols, protected_order_map, target_order
        ),
        row=1,
        col=1,
    )

    # Add subgroup table (protected attributes + target, no intersectional)
    row_colors = subgroup_table["EquityColor"].tolist()
    row_equity_labels = subgroup_table["EquityLabel"].tolist()
    row_text_colors = _get_equity_text_colors(row_equity_labels)
    white_col = ["#ffffff"] * len(subgroup_table)

    fig.add_trace(
        go.Table(
            header=dict(
                values=[
                    "Characteristic",
                    "Protected Subgroup",
                    "Equity Value",
                    "BH-adjusted p-value",
                ],
                fill_color="#f2f5f9",
                align="left",
                font=dict(color="#1f2937", size=12),
            ),
            cells=dict(
                values=[
                    subgroup_table["Characteristic"],
                    subgroup_table["Protected Subgroup"],
                    subgroup_table["Equity Value"],
                    subgroup_table["BH-adjusted p-value"],
                ],
                fill_color=[white_col, white_col, row_colors, white_col],
                font=dict(color=["#111111", "#111111", row_text_colors, "#111111"], size=11),
                align="left",
                height=24,
            ),
        ),
        row=1,
        col=2,
    )

    # Add leaf table (intersectional combinations)
    leaf_colors = leaf_table["EquityColor"].tolist()
    leaf_equity_labels = leaf_table["EquityLabel"].tolist()
    leaf_text_colors = _get_equity_text_colors(leaf_equity_labels)
    leaf_white = ["#ffffff"] * len(leaf_table)

    fig.add_trace(
        go.Table(
            header=dict(
                values=[
                    "Subgroup",
                    "Equity Value",
                    "BH-adjusted p-value",
                ],
                fill_color="#f2f5f9",
                align="left",
                font=dict(color="#1f2937", size=12),
            ),
            cells=dict(
                values=[
                    leaf_table["Protected Subgroup"],
                    leaf_table["Equity Value"],
                    leaf_table["BH-adjusted p-value"],
                ],
                fill_color=[leaf_white, leaf_colors, leaf_white],
                font=dict(color=["#111111", leaf_text_colors, "#111111"], size=11),
                align="left",
                height=24,
            ),
        ),
        row=2,
        col=2,
    )

    # Add legend table
    legend_white = ["#ffffff"] * len(legend_table)
    legend_color_cells = legend_table["Color"].tolist()
    legend_descriptions = legend_table["Description"].tolist()
    legend_color_text = _get_legend_text_colors(legend_descriptions)

    fig.add_trace(
        go.Table(
            header=dict(
                values=["Description", "Metric Value Rule"],
                fill_color="#f2f5f9",
                align="left",
                font=dict(color="#1f2937", size=12),
            ),
            cells=dict(
                values=[
                    legend_table["Description"],
                    legend_table["Metric Value Rule"],
                ],
                fill_color=[legend_color_cells, legend_white],
                font=dict(color=[legend_color_text, "#111111"], size=11),
                align="left",
                height=23,
            ),
        ),
        row=3,
        col=2,
    )

    fig.update_layout(
        title=dict(
            text=f"Log Disparity Fairness Report: {model_name}",
            x=0.5,
            xanchor="center",
        ),
        margin=dict(t=70, l=20, r=20, b=20),
        width=1600,
        height=1000,
    )

    return fig


# ============================================================================
# Multi-Model Analysis Functions
# ============================================================================


def summarize_multi_model_reports(
    model_reports: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate multiple model reports into cross-model summary tables.

    Args:
        model_reports: Dict mapping model names to report dicts from
            compute_log_disparity_report().

    Returns:
        Tuple of:
            - model_summary_df: Summary statistics per model
            - label_counts: Long-format label counts
            - label_counts_pivot: Wide-format pivot table of label counts

    Example:
        >>> reports = {
        ...     "model1": compute_log_disparity_report(...),
        ...     "model2": compute_log_disparity_report(...),
        ... }
        >>> summary, counts, pivot = summarize_multi_model_reports(reports)
    """
    # Aggregate summary statistics
    summaries = [model_reports[m]["summary_stats"] for m in sorted(model_reports)]
    model_summary_df = pd.DataFrame(summaries).sort_values("mean_abs_log_disparity")

    # Aggregate leaf results
    leaf_frames = [model_reports[m]["leaf_results"] for m in sorted(model_reports)]
    all_leaf = pd.concat(leaf_frames, ignore_index=True)

    # Count equity labels
    label_counts = (
        all_leaf.groupby(["Model", "EquityLabel"], dropna=False).size().reset_index(name="count")
    )

    label_counts_pivot = (
        label_counts.pivot(index="Model", columns="EquityLabel", values="count")
        .fillna(0)
        .astype(int)
    )

    return model_summary_df, label_counts, label_counts_pivot


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example demonstrating usage with hepatitis dataset
    print("Log Disparity Fairness Metric")
    print("=" * 60)
    print("\nThis module implements log disparity fairness analysis.")
    print("Import and use compute_log_disparity_report() to evaluate")
    print("synthetic datasets against real data.")
    print("\nFor detailed usage, see docstrings or refer to:")
    print("  - notebooks/test_hepatitis_data.ipynb")
    print("  - https://doi.org/10.3390/e23091165")
