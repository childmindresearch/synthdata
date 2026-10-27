"""Shared plotting helpers: figure saving for both matplotlib and Plotly figures."""

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from synthdata.utils import ensure_dir, get_logger

logger = get_logger(__name__)


def _label_sort_key(value) -> tuple:
    """Sort numbers (and labels that read as numbers, e.g. "10.0") by value, then text."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float("nan")
    return (0, number, "") if number == number else (1, 0.0, str(value))


def ordered_categories(values: set, configured_order: list | None = None) -> list:
    """Return category labels in configured order, then the rest sorted.

    Labels outside the configured order sort by numeric value when they read
    as numbers, so codes stored as text come out 1, 2, ..., 10 rather than
    1, 10, 11, 2.
    """
    present = set(values)
    ordered = [value for value in configured_order or [] if value in present]
    extras = present - set(ordered)
    return ordered + sorted(extras, key=_label_sort_key)


def _shorten(text: str, width: int) -> str:
    """Cut ``text`` to ``width`` characters, ending in "..." when cut.

    Unlike ``textwrap.shorten`` this also works on names without spaces.
    """
    return text if len(text) <= width else text[: width - 3] + "..."


def wrapped_title(text: str, width: int = 32, max_lines: int = 3) -> str:
    """Wrap a long column name so it fits over one subplot.

    ``tight_layout`` keeps every title inside its grid cell, so a single-line
    title wider than the axes squeezes the axes to a sliver. Wrapping (and
    shortening past ``max_lines``) keeps the plot area at full width.
    """
    return "\n".join(textwrap.wrap(str(text), width, max_lines=max_lines, placeholder="..."))


def set_category_ticks(ax, positions, categories, max_labels: int = 20) -> None:
    """Label categorical bars so the labels stay inside the subplot.

    Long or many labels are shortened and rotated, and past ``max_labels`` only
    every k-th bar is labelled. Unrotated labels wider than the axes would make
    ``tight_layout`` shrink every plot in that grid column.
    """
    labels = [str(c) for c in categories]
    step = -(-len(labels) // max_labels) if labels else 1
    ax.set_xticks(list(positions)[::step])
    shown = labels[::step]
    if len(shown) > 6 or sum(len(label) for label in shown) > 24:
        shown = [_shorten(label, 18) for label in shown]
        ax.set_xticklabels(shown, rotation=45, ha="right", fontsize=7)
    else:
        ax.set_xticklabels(shown)


def column_bar_chart(labels, values, *, xlabel: str, title: str, colors=None, xmax=None):
    """Horizontal bar chart with one bar per column, first label on top.

    Horizontal bars keep long column names readable however many there are, and
    the axis is clipped to the bars: the default 5% margin turns into a wide
    empty band once a chart has hundreds of columns.
    """
    import matplotlib.pyplot as plt

    labels = [_shorten(str(label), 48) for label in labels]
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.22 * len(labels) + 1.2)))
    positions = np.arange(len(labels))
    ax.barh(positions, values, color=colors, zorder=3)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8 if len(labels) > 30 else 10)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    if xmax is not None:
        ax.set_xlim(0, xmax)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4, axis="x", zorder=0)
    fig.tight_layout()
    return fig, ax


def add_histogram_with_kde(
    ax,
    values: pd.Series,
    *,
    bins: int,
    label: str | None = None,
    alpha: float = 0.5,
    color: str | None = None,
) -> None:
    """Draw a density histogram and KDE when a continuous sample supports it.

    KDE cannot be estimated for empty, constant, or singleton samples. The
    histogram remains useful in those cases, so it is retained and the omitted
    KDE is logged rather than replaced with a misleading artificial curve.
    """
    numeric_values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    numeric_values = numeric_values[np.isfinite(numeric_values)]
    histogram_label = f"{label} histogram" if label else None
    ax.hist(
        numeric_values,
        bins=bins,
        density=True,
        alpha=alpha,
        color=color,
        label=histogram_label,
    )

    if len(numeric_values) < 2 or np.unique(numeric_values).size < 2:
        logger.info("Skipping KDE for %s: fewer than two distinct finite values", label or "series")
        return

    try:
        density = gaussian_kde(numeric_values)
    except (np.linalg.LinAlgError, ValueError) as exc:
        logger.warning("Skipping KDE for %s: %s", label or "series", exc)
        return

    x_values = np.linspace(numeric_values.min(), numeric_values.max(), 200)
    ax.plot(
        x_values,
        density(x_values),
        color=color,
        linewidth=1.8,
        label=f"{label} KDE" if label else "KDE",
    )


def save_matplotlib_figure(fig, path: str | Path, dpi: int = 150, formats=("png",)) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        logger.info("Saved figure: %s", out)


def save_plotly_figure(fig, path: str | Path, formats=("html",)) -> None:
    """Save a Plotly figure. Falls back to HTML if 'png'/'pdf' requested but
    kaleido (the static-image exporter) isn't installed."""
    path = Path(path)
    ensure_dir(path.parent)
    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        try:
            if fmt == "html":
                fig.write_html(str(out))
            else:
                fig.write_image(str(out))
            logger.info("Saved figure: %s", out)
        except (ValueError, ImportError) as exc:
            logger.warning(
                "Could not save %s (%s); falling back to HTML. Install 'kaleido' for "
                "static image export of Plotly figures.",
                out,
                exc,
            )
            fallback = path.with_suffix(".html")
            fig.write_html(str(fallback))
            logger.info("Saved figure: %s", fallback)
