"""Figures produced during evaluation: utility/privacy/fairness rank trade-off
scatter plots, per-model SynthEval diagnostic plots, and log-disparity sunburst
reports.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from synthdata.config import Config
from synthdata.plotting import save_matplotlib_figure, save_plotly_figure
from synthdata.utils import get_logger

logger = get_logger(__name__)


def _base_model(name: str) -> str:
    return name[: -len("_hpo")] if name.endswith("_hpo") else name


def plot_rank_tradeoff(
    combined: pd.DataFrame,
    x_key: tuple,
    y_key: tuple,
    x_label: str,
    y_label: str,
    title: str,
):
    """Generic scatter of two rank columns from the combined evaluation table.

    HPO-tuned models (name ending in ``_hpo``) are drawn as larger star markers;
    all variants of the same base model share a color.
    """
    from matplotlib.lines import Line2D

    models = list(combined.index)
    base_models = sorted({_base_model(m) for m in models})
    palette = dict(zip(base_models, plt.cm.tab20.colors[: len(base_models)], strict=True))

    fig, ax = plt.subplots(figsize=(11, 7))
    for model in models:
        is_hpo = model.endswith("_hpo")
        base = _base_model(model)
        color = palette.get(base, "grey")
        x = combined.loc[model, x_key]
        y = combined.loc[model, y_key]
        ax.scatter(
            x,
            y,
            s=350 if is_hpo else 160,
            marker="*" if is_hpo else "o",
            color=color,
            alpha=0.85,
            edgecolors="black" if is_hpo else color,
            linewidths=1.2 if is_hpo else 0.0,
            zorder=3,
        )
        ax.annotate(str(model), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    color_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=palette[b], markersize=9, label=b)
        for b in base_models
    ]
    type_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="grey",
            markersize=9,
            markeredgecolor="none",
            label="Regular",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="grey",
            markersize=12,
            markeredgecolor="black",
            linewidth=0,
            label="HPO",
        ),
    ]
    ax.legend(
        handles=color_handles + [Line2D([], [], linestyle="none")] + type_handles,
        loc="best",
        fontsize=8,
        ncol=2,
    )
    fig.tight_layout()
    return fig


def save_rank_tradeoff_plots(cfg: Config, combined: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir) / "evaluation"
    pairs = [
        (
            ("__all__", "utility", "rank"),
            ("__all__", "privacy", "rank"),
            "Utility vs Privacy Trade-off",
            "utility_vs_privacy",
        ),
        (
            ("__all__", "utility", "rank"),
            ("__all__", "fairness", "rank"),
            "Utility vs Fairness Trade-off",
            "utility_vs_fairness",
        ),
        (
            ("__all__", "privacy", "rank"),
            ("__all__", "fairness", "rank"),
            "Privacy vs Fairness Trade-off",
            "privacy_vs_fairness",
        ),
    ]
    for x_key, y_key, title, fname in pairs:
        if x_key not in combined.columns or y_key not in combined.columns:
            continue
        fig = plot_rank_tradeoff(
            combined, x_key, y_key, x_key[1].title() + " rank", y_key[1].title() + " rank", title
        )
        save_matplotlib_figure(fig, output_dir / fname, cfg.plots.dpi, cfg.plots.formats)
        plt.close(fig)


def save_log_disparity_plots(
    log_disparity_reports: dict[str, dict], output_dir: str | Path
) -> None:
    output_dir = Path(output_dir) / "evaluation" / "log_disparity"
    for name, report in log_disparity_reports.items():
        fig = report.get("report_figure")
        if fig is None:
            continue
        save_plotly_figure(fig, output_dir / name, ("html",))
