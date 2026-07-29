"""Shared plotting helpers: figure saving for both matplotlib and Plotly figures."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from synthdata.utils import ensure_dir, get_logger

logger = get_logger(__name__)


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
