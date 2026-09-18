"""Evaluation figures: interactive rank trade-offs and log-disparity reports."""

from colorsys import hls_to_rgb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from synthdata.config import Config
from synthdata.evaluation.artifacts import _model_artifact_id
from synthdata.plotting import save_matplotlib_figure, save_plotly_figure
from synthdata.utils import get_logger

logger = get_logger(__name__)


def _base_model(name: str) -> str:
    return name[: -len("_hpo")] if name.endswith("_hpo") else name


def _finite_decision_eligible_rows(
    combined: pd.DataFrame, rank_keys: tuple[tuple, ...]
) -> pd.DataFrame:
    """Return complete, eligible rows with finite values for required ranks."""
    if combined.empty:
        return combined.iloc[0:0]
    if any(key not in combined.columns for key in rank_keys):
        return combined.iloc[0:0]

    eligible = pd.Series(True, index=combined.index)
    for column in combined.columns:
        metric = column[-1] if isinstance(column, tuple) else column
        if str(metric).endswith(("_decision_eligible", "decision_eligible")):
            eligible &= combined[column].eq(True)
        elif str(metric).endswith(("_audit_status", "audit_status")):
            eligible &= combined[column].eq("succeeded")

    for key in rank_keys:
        values = pd.to_numeric(combined[key], errors="coerce")
        eligible &= values.map(np.isfinite)
    return combined.loc[eligible]


def _plot_status(combined: pd.DataFrame, rank_keys: tuple[tuple, ...]) -> tuple[str, str]:
    """Return explicit status for rank evidence used by a plot."""
    if combined.empty:
        return "withheld", "no evaluation records"
    status_columns = [
        column
        for column in combined.columns
        if isinstance(column, tuple)
        and str(column[-1]).endswith(
            ("_decision_eligible", "decision_eligible", "_audit_status", "audit_status")
        )
    ]
    if not status_columns:
        return "withheld", "evaluation status columns are absent"
    if any(key not in combined.columns for key in rank_keys):
        return "withheld", "required rank columns are absent"
    if _finite_decision_eligible_rows(combined, rank_keys).empty:
        return "indeterminate", "no eligible finite evaluation rows"
    return "succeeded", ""


def _mark_matplotlib_status(figure, status: str, reason: str) -> None:
    """Attach machine-readable status and visible explanation to a figure."""
    figure._synthdata_status = {"status": status, "reason": reason}
    if status != "succeeded":
        figure.axes[0].text(
            0.5,
            0.5,
            f"Plot {status}: {reason}",
            transform=figure.axes[0].transAxes,
            ha="center",
            va="center",
            color="darkorange",
        )


def plot_rank_tradeoff(
    combined: pd.DataFrame,
    x_key: tuple,
    y_key: tuple,
    x_label: str,
    y_label: str,
    title: str,
):
    """Build a two-dimensional rank trade-off scatter plot.

    HPO-tuned variants are diamond markers; regular variants are circles.
    All variants of the same base model share a color.
    """
    from matplotlib.lines import Line2D

    filtered = _finite_decision_eligible_rows(
        combined, (x_key, y_key, ("__all__", "overall", "rank"))
    )
    models = list(filtered.index)
    base_models = sorted({_base_model(model) for model in models})
    palette = {
        base: (*hls_to_rgb(index / max(len(base_models), 1), 0.48, 0.58), 1.0)
        for index, base in enumerate(base_models)
    }

    fig, ax = plt.subplots(figsize=(11, 7))
    for model in models:
        is_hpo = model.endswith("_hpo")
        base = _base_model(model)
        x_value = filtered.loc[model, x_key]
        y_value = filtered.loc[model, y_key]
        ax.scatter(
            x_value,
            y_value,
            s=130,
            marker="D" if is_hpo else "o",
            color=palette[base],
            alpha=0.85,
            edgecolors="black" if is_hpo else palette[base],
            linewidths=1.2 if is_hpo else 0.0,
            zorder=3,
        )
        ax.annotate(
            str(model),
            (x_value, y_value),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=palette[base],
            markeredgecolor=palette[base],
            markersize=9,
            label=base,
        )
        for base in base_models
    ]
    variant_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color="grey",
            markersize=9,
            markeredgecolor="none",
            label="Regular model",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            color="grey",
            markersize=9,
            markeredgecolor="black",
            label="HPO-tuned model",
        ),
    ]
    ax.legend(handles=color_handles + variant_handles, loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    status, reason = _plot_status(combined, (x_key, y_key, ("__all__", "overall", "rank")))
    _mark_matplotlib_status(fig, status, reason)
    return fig


def plot_rank_tradeoff_3d(combined: pd.DataFrame):
    """Build an interactive utility/privacy/fairness rank scatter plot.

    HPO-tuned variants are diamonds; regular variants are circles. All
    variants use the same marker size, and each base model's variants share a
    color.
    """
    import plotly.graph_objects as go

    rank_keys = {
        "Utility": ("__all__", "utility", "rank"),
        "Privacy": ("__all__", "privacy", "rank"),
        "Fairness": ("__all__", "fairness", "rank"),
    }
    filtered = _finite_decision_eligible_rows(
        combined, (*tuple(rank_keys.values()), ("__all__", "overall", "rank"))
    )
    status, reason = _plot_status(
        combined, (*tuple(rank_keys.values()), ("__all__", "overall", "rank"))
    )
    models = list(filtered.index)
    base_models = sorted({_base_model(model) for model in models})
    palette = {
        base: f"hsl({index * 360 / max(len(base_models), 1):.0f}, 58%, 48%)"
        for index, base in enumerate(base_models)
    }
    fig = go.Figure()
    for model in models:
        is_hpo = model.endswith("_hpo")
        ranks = [filtered.loc[model, key] for key in rank_keys.values()]
        base = _base_model(model)
        fig.add_trace(
            go.Scatter3d(
                x=[ranks[0]],
                y=[ranks[1]],
                z=[ranks[2]],
                mode="markers+text",
                name=str(model),
                showlegend=False,
                text=[str(model)],
                textposition="top center",
                marker={
                    "size": 8,
                    "symbol": "diamond" if is_hpo else "circle",
                    "color": palette[base],
                    "line": {"color": "black" if is_hpo else palette[base], "width": 1},
                },
                hovertemplate=(
                    f"<b>{model}</b><br>Utility rank: %{{x:.3f}}<br>"
                    "Privacy rank: %{y:.3f}<br>Fairness rank: %{z:.3f}<extra></extra>"
                ),
            )
        )

    if models:
        for base in base_models:
            fig.add_trace(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode="markers",
                    name=base,
                    marker={"size": 8, "color": palette[base]},
                    hoverinfo="skip",
                )
            )
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                name="Regular model",
                marker={"size": 8, "color": "grey", "symbol": "circle"},
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                name="HPO-tuned model",
                marker={"size": 8, "color": "grey", "symbol": "diamond"},
                hoverinfo="skip",
            )
        )
    fig.update_layout(
        title="Utility, Privacy, and Fairness Rank Trade-off",
        scene={
            "xaxis_title": "Utility rank",
            "yaxis_title": "Privacy rank",
            "zaxis_title": "Fairness rank",
        },
        legend_title_text="Base model / variant",
        margin={"l": 0, "r": 0, "b": 0, "t": 50},
        meta={"status": status, "reason": reason},
    )
    return fig


def save_rank_tradeoff_plots(cfg: Config, combined: pd.DataFrame, output_dir: str | Path) -> None:
    """Save interactive 3D and static pairwise rank trade-off plots."""
    output_dir = Path(output_dir) / "evaluation"
    fig = plot_rank_tradeoff_3d(combined)
    save_plotly_figure(fig, output_dir / "rank_tradeoff_3d", ("html",))

    pairs = [
        (
            ("__all__", "utility", "rank"),
            ("__all__", "privacy", "rank"),
            "Utility rank",
            "Privacy rank",
            "Utility vs Privacy Trade-off",
            "utility_vs_privacy",
        ),
        (
            ("__all__", "utility", "rank"),
            ("__all__", "fairness", "rank"),
            "Utility rank",
            "Fairness rank",
            "Utility vs Fairness Trade-off",
            "utility_vs_fairness",
        ),
        (
            ("__all__", "privacy", "rank"),
            ("__all__", "fairness", "rank"),
            "Privacy rank",
            "Fairness rank",
            "Privacy vs Fairness Trade-off",
            "privacy_vs_fairness",
        ),
    ]
    for x_key, y_key, x_label, y_label, title, filename in pairs:
        missing_keys = [key for key in (x_key, y_key) if key not in combined.columns]
        if missing_keys:
            logger.warning("Skipping %s; missing rank columns: %s", filename, missing_keys)
            continue
        figure = plot_rank_tradeoff(combined, x_key, y_key, x_label, y_label, title)
        try:
            save_matplotlib_figure(figure, output_dir / filename, cfg.plots.dpi, cfg.plots.formats)
        finally:
            plt.close(figure)


def save_log_disparity_plots(
    log_disparity_reports: dict[str, dict], output_dir: str | Path
) -> None:
    output_dir = Path(output_dir) / "evaluation" / "log_disparity"
    for name, report in log_disparity_reports.items():
        if report.get("state") != "succeeded":
            logger.warning("[plot] skipping non-succeeded log-disparity report for model %s", name)
            continue
        fig = report.get("report_figure")
        if fig is None and "error" not in report:
            from synthdata.log_disparity.metric_log_disparity import (
                build_log_disparity_report_figure,
            )

            fig = build_log_disparity_report_figure(report)
        if fig is None:
            if "error" in report:
                logger.warning(
                    "[plot] skipping persisted failed log-disparity report for model %s: %s",
                    name,
                    report["error"],
                )
            continue
        save_plotly_figure(fig, output_dir / _model_artifact_id(name), ("html",))
