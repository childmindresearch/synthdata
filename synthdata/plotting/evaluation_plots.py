"""Evaluation figures: interactive rank trade-offs and log-disparity reports."""

import json
from colorsys import hls_to_rgb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from synthdata.config import Config
from synthdata.evaluation.artifacts import _model_artifact_id
from synthdata.plotting import save_plotly_figure
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


def _mark_matplotlib_status(
    figure, status: str, reason: str, metadata: dict[str, object] | None = None
) -> None:
    """Attach machine-readable status and visible explanation to a figure."""
    figure._synthdata_status = (
        metadata if metadata is not None else {"status": status, "reason": reason}
    )
    if status not in {"succeeded", "partial"}:
        figure.axes[0].text(
            0.5,
            0.5,
            f"Plot {status}: {reason}",
            transform=figure.axes[0].transAxes,
            ha="center",
            va="center",
            color="darkorange",
        )


def _filter_produced_outputs(
    combined: pd.DataFrame, produced_outputs: tuple[str, ...] | None
) -> pd.DataFrame:
    """Keep saved evaluation rows belonging to manifest-declared outputs."""
    if produced_outputs is None:
        return combined
    return combined.loc[combined.index.isin(produced_outputs)]


def _rank_plot_status(
    combined: pd.DataFrame,
    rank_keys: tuple[tuple, ...],
    missing_stage_a_outputs: tuple[str, ...],
) -> tuple[str, str, dict[str, object]]:
    """Combine rank evidence status with validated generation coverage."""
    status, reason = _plot_status(combined, rank_keys)
    if not missing_stage_a_outputs:
        return status, reason, {"status": status, "reason": reason}

    missing_text = ", ".join(missing_stage_a_outputs)
    coverage_reason = f"Partial cohort; missing Stage A outputs: {missing_text}"
    plot_status = "partial" if status == "succeeded" else status
    plot_reason = coverage_reason if status == "succeeded" else f"{reason}; {coverage_reason}"
    return (
        plot_status,
        plot_reason,
        {
            "status": plot_status,
            "reason": plot_reason,
            "generation_status": "partial",
            "missing_stage_a_outputs": list(missing_stage_a_outputs),
        },
    )


def _partial_plot_title(title: str, missing_stage_a_outputs: tuple[str, ...]) -> str:
    if not missing_stage_a_outputs:
        return title
    missing_text = ", ".join(missing_stage_a_outputs)
    return f"{title}\nPARTIAL COHORT — missing Stage A outputs: {missing_text}"


def _static_plot_metadata(metadata: dict[str, object], fmt: str) -> dict[str, str]:
    """Map status JSON to metadata field supported by Matplotlib's static backends."""
    metadata_fields = {
        "png": "Description",
        "pdf": "Subject",
        "svg": "Description",
        "ps": "Creator",
        "eps": "Creator",
    }
    metadata_field = metadata_fields.get(fmt.lower())
    if metadata_field is None:
        supported = ", ".join(sorted(metadata_fields))
        raise ValueError(
            f"Static rank plot format {fmt!r} cannot persist status metadata; "
            f"supported formats: {supported}"
        )
    return {metadata_field: json.dumps(metadata, sort_keys=True)}


def plot_rank_tradeoff(
    combined: pd.DataFrame,
    x_key: tuple,
    y_key: tuple,
    x_label: str,
    y_label: str,
    title: str,
    *,
    produced_outputs: tuple[str, ...] | None = None,
    missing_stage_a_outputs: tuple[str, ...] = (),
):
    """Build a two-dimensional rank trade-off scatter plot.

    HPO-tuned variants are diamond markers; regular variants are circles.
    All variants of the same base model share a color.
    """
    from matplotlib.lines import Line2D

    cohort = _filter_produced_outputs(combined, produced_outputs)
    rank_keys = (x_key, y_key, ("__all__", "overall", "rank"))
    filtered = _finite_decision_eligible_rows(cohort, rank_keys)
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
    ax.set_title(
        _partial_plot_title(title, missing_stage_a_outputs), fontsize=14, fontweight="bold"
    )
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
    status, reason, metadata = _rank_plot_status(cohort, rank_keys, missing_stage_a_outputs)
    _mark_matplotlib_status(fig, status, reason, metadata)
    return fig


def plot_rank_tradeoff_3d(
    combined: pd.DataFrame,
    *,
    produced_outputs: tuple[str, ...] | None = None,
    missing_stage_a_outputs: tuple[str, ...] = (),
):
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
    cohort = _filter_produced_outputs(combined, produced_outputs)
    required_rank_keys = (*tuple(rank_keys.values()), ("__all__", "overall", "rank"))
    filtered = _finite_decision_eligible_rows(cohort, required_rank_keys)
    status, reason, metadata = _rank_plot_status(
        cohort, required_rank_keys, missing_stage_a_outputs
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
        title=_partial_plot_title(
            "Utility, Privacy, and Fairness Rank Trade-off", missing_stage_a_outputs
        ),
        scene={
            "xaxis_title": "Utility rank",
            "yaxis_title": "Privacy rank",
            "zaxis_title": "Fairness rank",
        },
        legend_title_text="Base model / variant",
        margin={"l": 0, "r": 0, "b": 0, "t": 50},
        meta=metadata,
    )
    return fig


def save_rank_tradeoff_plots(
    cfg: Config,
    combined: pd.DataFrame,
    output_dir: str | Path,
    *,
    produced_outputs: tuple[str, ...] | None = None,
    missing_stage_a_outputs: tuple[str, ...] = (),
) -> None:
    """Save interactive 3D and static pairwise rank trade-off plots."""
    output_dir = Path(output_dir) / "evaluation"
    static_formats = tuple(cfg.plots.formats)
    # Validate all metadata carriers before writing any files, so unsupported
    # configured formats cannot leave a partial set of rank plot artifacts.
    for fmt in static_formats:
        _static_plot_metadata({}, fmt)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_rank_tradeoff_3d(
        combined,
        produced_outputs=produced_outputs,
        missing_stage_a_outputs=missing_stage_a_outputs,
    )
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
        figure = plot_rank_tradeoff(
            combined,
            x_key,
            y_key,
            x_label,
            y_label,
            title,
            produced_outputs=produced_outputs,
            missing_stage_a_outputs=missing_stage_a_outputs,
        )
        try:
            metadata = figure._synthdata_status
            for fmt in static_formats:
                path = (output_dir / filename).with_suffix(f".{fmt}")
                figure.savefig(
                    path,
                    dpi=cfg.plots.dpi,
                    bbox_inches="tight",
                    metadata=_static_plot_metadata(metadata, fmt),
                )
                logger.info("Saved figure: %s", path)
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
